"""
Image Generation Module.

Supports both Imagen 4.0 and Gemini image generation models.
"""
import logging
import base64
import time
from pathlib import Path
from typing import Optional, Tuple
import io

from google import genai
from google.genai import types
from httpx import TimeoutException
from PIL import Image

from config import (
    IMAGE_GEN_PER_PROMPT
)
import config
from styles import get_style_preset, build_image_prompt_with_style, generate_seed_from_run
from gen_qa import verify_image_vision, verify_visual_consistency, select_best_image_fallback

logger = logging.getLogger(__name__)

# Image generation timeout and retry settings
IMAGE_TIMEOUT_SECONDS = 120  # 120 seconds per attempt
IMAGE_MAX_RETRIES = 10       # Max retry attempts as per user request

# Chat-based image editing settings
MAX_CHAT_EDIT_TURNS = 3      # Maximum edit iterations per image


def apply_programmatic_rotation(image_bytes: bytes, qa_fail_reason: str) -> Optional[bytes]:
    """
    Apply programmatic rotation to fix orientation issues detected by QA.
    
    This is more reliable than asking the LLM to regenerate because:
    1. It physically rotates the pixels
    2. It's deterministic and fast
    3. The LLM often regenerates with the same orientation bias
    
    Args:
        image_bytes: The image data to rotate
        qa_fail_reason: The QA failure message to parse rotation direction
        
    Returns:
        Rotated image bytes, or None if rotation failed
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        original_size = img.size
        
        # Determine rotation direction from QA feedback
        reason_lower = qa_fail_reason.lower()
        
        if "counter-clockwise" in reason_lower or "counter clockwise" in reason_lower:
            # Image is rotated counter-clockwise, so we need to rotate clockwise to fix
            logger.info("[ROTATION FIX] Rotating 90 degrees CLOCKWISE (to fix counter-clockwise rotation)")
            rotated_img = img.rotate(-90, expand=True)
        elif "clockwise" in reason_lower:
            # Image is rotated clockwise, so we need to rotate counter-clockwise to fix
            logger.info("[ROTATION FIX] Rotating 90 degrees COUNTER-CLOCKWISE (to fix clockwise rotation)")
            rotated_img = img.rotate(90, expand=True)
        elif "sideways" in reason_lower or "rotated 90" in reason_lower:
            # Generic "sideways" - try clockwise first (most common case)
            logger.info("[ROTATION FIX] Rotating 90 degrees CLOCKWISE (generic sideways fix)")
            rotated_img = img.rotate(-90, expand=True)
        else:
            logger.warning("[ROTATION FIX] Could not determine rotation direction from QA feedback")
            return None
        
        # Check if rotation resulted in proper vertical orientation (taller than wide)
        new_size = rotated_img.size
        if new_size[1] < new_size[0]:
            # Still landscape, try the other direction
            logger.info("[ROTATION FIX] Result still landscape, trying opposite rotation...")
            rotated_img = img.rotate(90, expand=True)
        
        # Convert back to bytes
        png_buffer = io.BytesIO()
        rotated_img.save(png_buffer, format="PNG")
        rotated_bytes = png_buffer.getvalue()
        
        logger.info(f"[ROTATION FIX] Rotation complete: {original_size} -> {rotated_img.size}")
        return rotated_bytes
        
    except Exception as e:
        logger.error(f"[ROTATION FIX] Failed to apply rotation: {e}")
        return None


def generate_image_with_chat_edit(
    prompt: str,
    avoid: str = "",
    output_path: Optional[Path] = None,
    style_context: Optional[str] = None,
    style_id: str = config.DEFAULT_STYLE,
    visual_context: str = "",
    reference_image_bytes: Optional[bytes] = None,
    shot_type: Optional[str] = None,
    reference_images_list: Optional[list] = None  # List of (name, bytes) tuples
) -> Tuple[bytes, str, str]:
    """
    Generate an image using Gemini 3 Pro Image Preview with multi-turn chat editing.
    
    This approach uses a chat session to:
    1. Generate the initial image
    2. If QA fails, send edit instructions to fix specific issues
    3. Maintain visual consistency through the chat context
    
    Args:
        prompt: The image generation prompt (in English)
        avoid: Elements to avoid in the image
        output_path: Optional path to save the image locally
        style_context: Optional global style context to include
        style_id: Style preset ID for consistency
        visual_context: Visual context for QA verification
        reference_image_bytes: Optional reference image for consistency QA
        
    Returns:
        Tuple of (image_bytes, mime_type, final_prompt_used)
        
    Raises:
        Exception: If image generation fails after all attempts
    """
    logger.info(f"[CHAT EDIT MODE] Generating image with style: {style_id}")
    
    # Get style preset
    style = get_style_preset(style_id)
    
    # Gemini 3 Pro Image requires 'global' location
    location = "global"
    
    # Initialize the chat client
    client = genai.Client(
        vertexai=True,
        project=config.PROJECT_ID,
        location=location,
        http_options=types.HttpOptions(timeout=300000)
    )
    
    # Create chat session for multi-turn image editing
    chat = client.chats.create(
        model=config.IMAGE_MODEL,
        config=types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE'],
            temperature=config.IMAGE_TEMPERATURE,
            image_config=types.ImageConfig(aspect_ratio="9:16"),
        )
    )
    session_is_fresh = True  # Track when session needs reference re-injection
    
    # Build the full initial prompt with style injection
    # CRITICAL: Orientation rules go FIRST and are VERY explicit
    full_prompt = f"""⚠️ MANDATORY ORIENTATION - READ FIRST ⚠️
This image MUST be generated in UPRIGHT VERTICAL orientation:
- The FLOOR/GROUND must be at the BOTTOM of the frame
- The SKY/CEILING must be at the TOP of the frame  
- Characters must STAND UPRIGHT (heads UP, feet DOWN)
- The horizon line (if visible) must be HORIZONTAL
- NO sideways, NO tilted, NO rotated compositions
- Think of how a person naturally holds their phone VERTICALLY to take a portrait photo

---

{style['style_positive']}

{style['framing_style']}. Generate a high-quality image for a vertical video scene.

{f"🎬 DIRECTORIAL INTENT / CAMERA ANGLE (CRITICAL): {shot_type}" if shot_type else ""}

VISUAL DESCRIPTION:
{prompt}

{f"MANDATORY CHARACTER & STYLE RULES:{chr(10)}{style_context}" if style_context else ""}

COLOR GRADING: {style['lut']}
PALETTE: {style['palette']}
{style['rendering_notes']}

AVOID (do not include):
{style['style_negative']}
{avoid}
{config.CONTENT_AVOID_RULES}

TECHNICAL REQUIREMENTS:
- Vertical 9:16 aspect ratio (taller than wide)
- High quality, detailed imagery
- No text, logos, or watermarks
- REMINDER: Floor at BOTTOM, ceiling/sky at TOP, characters UPRIGHT
"""
    
    # Collect images for fallback selection
    collected_images: list[Tuple[bytes, str, str]] = []
    last_error = None
    current_image_bytes = None
    mime_type = "image/png"
    
    # TURN 1: Generate initial image with retry logic for 429 errors
    max_initial_retries = 5
    initial_backoff = 10  # Start with 10 seconds
    image_bytes = None
    
    for retry_attempt in range(max_initial_retries):
        try:
            # If reference images available, inject them into every fresh session
            # (covers initial session AND sessions recreated after rate-limit errors)
            if reference_images_list and session_is_fresh:
                session_is_fresh = False  # Mark as consumed for this session
                ref_parts = []
                ref_names = []
                for ref_name, ref_bytes in reference_images_list:
                    ref_parts.append(types.Part.from_bytes(data=ref_bytes, mime_type="image/png"))
                    ref_names.append(ref_name)

                ref_context_msg = f"""These are REFERENCE IMAGES for visual consistency.
You MUST use these as guides to maintain the EXACT same appearance for: {', '.join(ref_names)}.
Preserve their face features, hair, clothing, colors, and proportions in the scene you generate next.

Reference images:"""
                ref_parts.insert(0, types.Part.from_text(text=ref_context_msg))

                try:
                    logger.info(f"[CHAT TURN 0] Injecting {len(reference_images_list)} reference images for consistency...")
                    chat.send_message(ref_parts)
                    logger.info(f"[CHAT TURN 0] Reference context injected successfully.")
                except Exception as ref_err:
                    logger.warning(f"[CHAT TURN 0] Failed to inject references: {ref_err}. Continuing without.")
            
            logger.info(f"[CHAT TURN 1] Sending initial generation request (attempt {retry_attempt + 1}/{max_initial_retries})...")
            response = chat.send_message(full_prompt)
            
            # Extract image from response
            if not response.candidates or not getattr(response.candidates[0], 'content', None) or not getattr(response.candidates[0].content, 'parts', None):
                raise ValueError("API returned empty parts, likely safety filter")
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_data = part.inline_data.data
                    mime_type = part.inline_data.mime_type or "image/png"
                    if isinstance(image_data, str):
                        image_bytes = base64.b64decode(image_data)
                    else:
                        image_bytes = image_data
                    break
            
            if image_bytes is not None:
                break  # Success! Exit retry loop
            else:
                raise ValueError("No image data in Gemini response")
                
        except Exception as gen_error:
            error_str = str(gen_error)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait_time = initial_backoff * (2 ** retry_attempt)  # Exponential backoff
                logger.warning(f"[CHAT TURN 1] Rate limited (429). Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                
                # Recreate chat session after rate limit (session might be invalid)
                if retry_attempt >= 2:
                    logger.info("[CHAT TURN 1] Recreating chat session after multiple rate limits...")
                    chat = client.chats.create(
                        model=config.IMAGE_MODEL,
                        config=types.GenerateContentConfig(
                            response_modalities=['TEXT', 'IMAGE'],
                            temperature=config.IMAGE_TEMPERATURE,
                            image_config=types.ImageConfig(aspect_ratio="9:16"),
                        )
                    )
                    session_is_fresh = True  # New session needs reference re-injection
            else:
                last_error = gen_error
                logger.error(f"[CHAT TURN 1] Non-retryable error: {gen_error}")
                break
    
    # Check if we got an image
    if image_bytes is None:
        if last_error:
            raise last_error
        raise Exception("Failed to generate initial image after all retries")
    
    current_image_bytes = image_bytes
    logger.info(f"[CHAT TURN 1] Initial image generated: {len(image_bytes)} bytes")
    
    # Helper function to check if QA failure is rotation-related
    def is_rotation_issue(reason: str) -> bool:
        reason_lower = reason.lower()
        rotation_keywords = ["rotated", "sideways", "tilted", "horizontal orientation", 
                           "90 degrees", "wrong orientation", "landscape orientation"]
        return any(keyword in reason_lower for keyword in rotation_keywords)
    
    try:
        # Visual QA Step
        if config.USE_VISUAL_QA:
            is_valid, qa_fail_reason, qa_recommended_fix = verify_image_vision(
                image_bytes, 
                prompt,
                visual_context=visual_context,
                style_name=style.get('name', style_id),
                style_description=style.get('rendering_notes', '')
            )
            
            if not is_valid:
                logger.warning(f"[CHAT TURN 1] Image failed Visual QA: {qa_fail_reason}")
                collected_images.append((image_bytes, mime_type, prompt))
                
                # STRATEGY: Different handling based on problem type
                if is_rotation_issue(qa_fail_reason):
                    # ==============================================
                    # ROTATION ISSUE → REGENERATE FROM SCRATCH
                    # ==============================================
                    logger.info("[ROTATION DETECTED] Will regenerate image completely with enhanced orientation prompt...")
                    
                    # Create enhanced prompt with even stronger orientation instructions
                    orientation_enhanced_prompt = f"""🚨 CRITICAL ORIENTATION OVERRIDE 🚨
PREVIOUS ATTEMPT WAS ROTATED/SIDEWAYS - THIS MUST BE FIXED!

The image MUST be rendered as if viewing a VERTICAL PHONE SCREEN:
- BOTTOM of the frame = floor/ground/feet
- TOP of the frame = sky/ceiling/head
- LEFT and RIGHT = sides of the scene
- The characters stand UPRIGHT like normal people standing

Imagine you are taking a PORTRAIT PHOTO with your phone HELD VERTICALLY.
The scene should look NATURAL, not like looking at a photo turned on its side.

---

{full_prompt}
"""
                    
                    # Regenerate with up to 3 attempts
                    max_rotation_retries = 3
                    for rotation_retry in range(max_rotation_retries):
                        logger.info(f"[ROTATION RETRY {rotation_retry + 1}/{max_rotation_retries}] Regenerating with enhanced orientation prompt...")
                        
                        # Create a fresh chat session for the new attempt
                        fresh_chat = client.chats.create(
                            model=config.IMAGE_MODEL,
                            config=types.GenerateContentConfig(
                                response_modalities=['TEXT', 'IMAGE'],
                                temperature=config.IMAGE_TEMPERATURE,
                                image_config=types.ImageConfig(aspect_ratio="9:16"),
                            )
                        )
                        
                        try:
                            retry_response = fresh_chat.send_message(orientation_enhanced_prompt)
                            
                            # Extract new image
                            new_image_bytes = None
                            if not retry_response.candidates or not getattr(retry_response.candidates[0], 'content', None) or not getattr(retry_response.candidates[0].content, 'parts', None):
                                raise ValueError("API returned empty parts in retry")
                            for part in retry_response.candidates[0].content.parts:
                                if hasattr(part, 'inline_data') and part.inline_data:
                                    new_data = part.inline_data.data
                                    if isinstance(new_data, str):
                                        new_image_bytes = base64.b64decode(new_data)
                                    else:
                                        new_image_bytes = new_data
                                    break
                            
                            if new_image_bytes is None:
                                logger.warning(f"[ROTATION RETRY {rotation_retry + 1}] No image in response")
                                continue
                            
                            logger.info(f"[ROTATION RETRY {rotation_retry + 1}] New image generated: {len(new_image_bytes)} bytes")
                            
                            # Re-verify the new image
                            is_valid_new, qa_reason_new, qa_fix_new = verify_image_vision(
                                new_image_bytes, 
                                prompt,
                                visual_context=visual_context,
                                style_name=style.get('name', style_id),
                                style_description=style.get('rendering_notes', '')
                            )
                            
                            collected_images.append((new_image_bytes, mime_type, prompt))
                            
                            if is_valid_new:
                                logger.info(f"[ROTATION RETRY {rotation_retry + 1}] SUCCESS! QA PASSED.")
                                current_image_bytes = new_image_bytes
                                is_valid = True
                                chat = fresh_chat  # Use this chat for consistency edits
                                break
                            elif is_rotation_issue(qa_reason_new):
                                logger.warning(f"[ROTATION RETRY {rotation_retry + 1}] Still rotated: {qa_reason_new}")
                                # Continue to next rotation retry
                            else:
                                # No longer a rotation issue, but has other problems
                                # Use this image as base for chat edits
                                logger.info(f"[ROTATION RETRY {rotation_retry + 1}] Orientation fixed but has other issue: {qa_reason_new}")
                                current_image_bytes = new_image_bytes
                                qa_fail_reason = qa_reason_new
                                qa_recommended_fix = qa_fix_new
                                chat = fresh_chat
                                break
                                
                        except Exception as retry_error:
                            error_str = str(retry_error)
                            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                                wait_time = 15 * (rotation_retry + 1)
                                logger.warning(f"[ROTATION RETRY {rotation_retry + 1}] Rate limited. Waiting {wait_time}s...")
                                time.sleep(wait_time)
                            else:
                                logger.error(f"[ROTATION RETRY {rotation_retry + 1}] Error: {retry_error}")
                    
                    # After rotation retries, check if we have a valid image
                    if is_valid:
                        logger.info("[ROTATION HANDLING] Successfully obtained correctly oriented image!")
                    else:
                        logger.warning("[ROTATION HANDLING] Could not fix rotation after all retries. Will use best available.")
                
                else:
                    # ==============================================
                    # OTHER QA ISSUE → USE CHAT EDIT
                    # ==============================================
                    logger.info(f"[QA ISSUE] Non-rotation problem detected. Using chat edit to fix...")
                
                # TURN 2+: Edit the image based on QA feedback (only if still not valid and NOT a persistent rotation issue)
                if not is_valid and not is_rotation_issue(qa_fail_reason):
                    for edit_turn in range(2, MAX_CHAT_EDIT_TURNS + 2):
                        logger.info(f"[CHAT TURN {edit_turn}] Sending edit request based on QA feedback...")
                        
                        edit_instruction = f"""Edit this image to fix the following issue:

PROBLEM DETECTED: {qa_fail_reason}
RECOMMENDED FIX: {qa_recommended_fix}

IMPORTANT RULES:
- Keep ALL other visual elements exactly the same
- Maintain the same composition, lighting, and style
- Only fix the specific issue mentioned above
- Do NOT change the character's basic design or colors
- Preserve the vertical 9:16 aspect ratio
"""
                        
                        try:
                            edit_response = chat.send_message(
                                edit_instruction,
                                config=types.GenerateContentConfig(
                                    response_modalities=['TEXT', 'IMAGE'],
                                    image_config=types.ImageConfig(aspect_ratio="9:16"),
                                )
                            )
                            
                            # Extract edited image
                            edited_bytes = None
                            if not edit_response.candidates or not getattr(edit_response.candidates[0], 'content', None) or not getattr(edit_response.candidates[0].content, 'parts', None):
                                raise ValueError("API returned empty parts in edit")
                            for part in edit_response.candidates[0].content.parts:
                                if hasattr(part, 'inline_data') and part.inline_data:
                                    image_data = part.inline_data.data
                                    mime_type = part.inline_data.mime_type or "image/png"
                                    if isinstance(image_data, str):
                                        edited_bytes = base64.b64decode(image_data)
                                    else:
                                        edited_bytes = image_data
                                    break
                            
                            if edited_bytes is None:
                                logger.warning(f"[CHAT TURN {edit_turn}] No image in edit response, keeping previous")
                                continue
                            
                            current_image_bytes = edited_bytes
                            logger.info(f"[CHAT TURN {edit_turn}] Edited image received: {len(edited_bytes)} bytes")
                            
                            # Check if this is a rotation issue that we should fix programmatically
                            if "rotated 90" in qa_fail_reason.lower() or "sideways" in qa_fail_reason.lower():
                                logger.info(f"[CHAT TURN {edit_turn}] Detected rotation issue - attempting programmatic fix...")
                                rotated_bytes = apply_programmatic_rotation(edited_bytes, qa_fail_reason)
                                if rotated_bytes:
                                    current_image_bytes = rotated_bytes
                                    edited_bytes = rotated_bytes
                                    logger.info(f"[CHAT TURN {edit_turn}] Applied programmatic rotation fix")
                            
                            # Re-verify with QA
                            is_valid, qa_fail_reason, qa_recommended_fix = verify_image_vision(
                                edited_bytes, 
                                prompt,
                                visual_context=visual_context,
                                style_name=style.get('name', style_id),
                                style_description=style.get('rendering_notes', '')
                            )
                            
                            if is_valid:
                                logger.info(f"[CHAT TURN {edit_turn}] Edit successful! QA PASSED.")
                                current_image_bytes = edited_bytes
                                break
                            else:
                                logger.warning(f"[CHAT TURN {edit_turn}] Edit still fails QA: {qa_fail_reason}")
                                collected_images.append((edited_bytes, mime_type, prompt))
                                
                        except Exception as edit_error:
                            logger.warning(f"[CHAT TURN {edit_turn}] Edit failed: {edit_error}")
                            time.sleep(5)
                            continue
        
        # Consistency QA Step (if reference provided)
        if config.USE_CONSISTENCY_QA and reference_image_bytes and current_image_bytes:
            is_valid_cons, cons_fail_reason, cons_recommended_fix = verify_visual_consistency(
                current_image_bytes, 
                reference_image_bytes, 
                prompt,
                style_name=style.get('name', style_id)
            )
            
            if not is_valid_cons:
                logger.warning(f"Image failed Consistency QA: {cons_fail_reason}")
                collected_images.append((current_image_bytes, mime_type, prompt))
                
                # Try one more edit for consistency
                logger.info("[CHAT CONSISTENCY EDIT] Attempting to fix consistency issue...")
                
                consistency_instruction = f"""Edit this image to match the reference character exactly:

CONSISTENCY ISSUE: {cons_fail_reason}
REQUIRED FIX: {cons_recommended_fix}

CRITICAL: The character must look IDENTICAL to the reference in terms of:
- Face shape and features
- Eye color and style
- Body proportions
- Clothing and accessories
- Material textures

Do NOT change the scene composition or action, only fix the character appearance.
"""
                
                try:
                    cons_response = chat.send_message(
                        consistency_instruction,
                        config=types.GenerateContentConfig(
                            response_modalities=['TEXT', 'IMAGE'],
                            image_config=types.ImageConfig(aspect_ratio="9:16"),
                        )
                    )
                    
                    if not cons_response.candidates or not getattr(cons_response.candidates[0], 'content', None) or not getattr(cons_response.candidates[0].content, 'parts', None):
                        raise ValueError("API returned empty parts in consistency")
                    for part in cons_response.candidates[0].content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            image_data = part.inline_data.data
                            if isinstance(image_data, str):
                                current_image_bytes = base64.b64decode(image_data)
                            else:
                                current_image_bytes = image_data
                            logger.info("[CHAT CONSISTENCY EDIT] Consistency edit applied")
                            break
                            
                except Exception as cons_error:
                    logger.warning(f"[CHAT CONSISTENCY EDIT] Failed: {cons_error}")
        
        # Save locally if path provided
        if output_path and current_image_bytes:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if mime_type != "image/png":
                img = Image.open(io.BytesIO(current_image_bytes))
                png_buffer = io.BytesIO()
                img.save(png_buffer, format="PNG")
                current_image_bytes = png_buffer.getvalue()
                mime_type = "image/png"
            output_path.write_bytes(current_image_bytes)
            logger.info(f"[CHAT EDIT MODE] Image saved to {output_path}")
        
        return current_image_bytes, mime_type, prompt
        
    except Exception as e:
        logger.error(f"[CHAT EDIT MODE] Generation failed: {e}")
        last_error = e
        
        # Fallback: select best from collected images
        if collected_images:
            logger.info(f"[CHAT EDIT MODE] Using fallback selection from {len(collected_images)} collected images")
            best_idx = select_best_image_fallback(
                [img[0] for img in collected_images],
                prompt=prompt,
                visual_context=visual_context,
                style_name=style_id
            )
            return collected_images[best_idx]
        
        if last_error:
            raise last_error
        raise Exception("Failed to generate image in chat edit mode")


def paraphrase_prompt(original_prompt: str, attempt: int, style_context: Optional[str] = None) -> str:
    """
    Paraphrase an image prompt to retry generation with different wording.
    Uses Gemini 3 Flash to create a variation while keeping the same meaning.
"""
    if attempt <= 1:
        return original_prompt
    
    try:
        logger.info(f"Paraphrasing prompt for retry attempt {attempt}...")
        
        client = genai.Client(
            vertexai=True,
            project=config.PROJECT_ID,
            location=config.TEXT_LOCATION,
            http_options=types.HttpOptions(timeout=300000)  # 300s timeout
        )
        
        paraphrase_request = f"""Rephrase this image generation prompt while keeping the EXACT same visual meaning.
        
        PARAPHRASE INSTRUCTIONS:
        1. Keep the subject, style, and composition of the original prompt.
        2. MANDATORY: Explicitly describe the scene as 'upright 9:16 vertical composition'.
        3. MANDATORY: Ensure 'Natural Orientation Sanity' by describing the sky/ceiling at the top and ground/floor at the bottom.
        4. Use different descriptive words but keep the visual essence identical.
        5. DO NOT use words like "horizontal", "sideways", or "tilted".
        
        ORIGINAL PROMPT:
        {original_prompt}
        
        {f"CHARACTER/STYLE CONTEXT (DO NOT CHANGE THESE DETAILS): {style_context}" if style_context else ""}
        
        NEW PARAPHRASED PROMPT (just the prompt text):"""

        response = client.models.generate_content(
            model=config.TEXT_MODEL,
            contents=paraphrase_request,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=2000,
            )
        )
        
        paraphrased = response.text.strip()
        logger.info(f"Paraphrased prompt (first 100 chars): {paraphrased[:100]}...")
        return paraphrased
        
    except Exception as e:
        logger.warning(f"Could not paraphrase prompt: {e}, using original")
        return original_prompt

def refine_prompt_vision(original_prompt: str, fail_reason: str, recommended_fix: str, style_context: Optional[str] = None) -> str:
    """Uses Gemini 3 Flash to fix a prompt based on visual QA feedback."""
    try:
        logger.info(f"Refining prompt based on Vision QA: {fail_reason}...")
        client = genai.Client(
            vertexai=True,
            project=config.PROJECT_ID,
            location=config.TEXT_LOCATION,
            http_options=types.HttpOptions(timeout=300000)  # 300s timeout
        )
        
        refine_request = f"""Refine this image generation prompt to FIX a technical error detected by a vision model.
        
        ORIGINAL PROMPT: {original_prompt}
        DETECTED ERROR: {fail_reason}
        RECOMMENDED FIX: {recommended_fix}
        
        {f"STYLE/CHARACTER CONTEXT (MUST PRESERVE): {style_context}" if style_context else ""}
        
        INSTRUCTIONS:
        1. Keep the spirit and subject of the original prompt.
        2. Incorporate the recommended fix into the visual description.
        3. MANDATORY: Focus on UPRIGHT VERTICAL 9:16 composition. 
        4. MANDATORY: Ensure NATURAL ORIENTATION SANITY (Heads at top, Ground at bottom).
        5. MANDATORY: Ensure it is ONE SINGLE CONTINUOUS IMAGE. NO panels, NO tiles, NO grids.
        6. MANDATORY: Ensure NO ROTATION. The subject must be upright.
        7. Add explicit negative constraints if needed (e.g., 'Do NOT use manga panels', 'Single unified frame only', 'No tilted horizon').
        
        NEW REFINED PROMPT (just the prompt text):"""
        
        response = client.models.generate_content(
            model=config.TEXT_MODEL,
            contents=refine_request
        )
        return response.text.strip()
    except Exception as e:
        logger.warning(f"Could not refine prompt with vision data: {e}")
        return original_prompt


def generate_image(
    prompt: str,
    avoid: str = "",
    output_path: Optional[Path] = None,
    style_context: Optional[str] = None,
    style_id: str = config.DEFAULT_STYLE,
    seed: Optional[int] = None,
    visual_context: str = "",
    reference_image_bytes: Optional[bytes] = None,
    shot_type: Optional[str] = None
) -> Tuple[bytes, str, str]:
    """
    Generate an image using Imagen 4.0 or Gemini image generation.
    
    Args:
        prompt: The image generation prompt (in English)
        avoid: Elements to avoid in the image
        output_path: Optional path to save the image locally
        style_context: Optional global style context to include
        style_id: Style preset ID for consistency
        seed: Optional seed for deterministic generation
        reference_image_bytes: Optional reference image for consistency QA
        
    Returns:
        Tuple of (image_bytes, mime_type, final_prompt_used)
        
    Raises:
        Exception: If image generation fails
    """
    logger.info(f"Generating image with style: {style_id}, prompt: {prompt[:100]}...")
    
    # Get style preset
    style = get_style_preset(style_id)
    
    # Determine location based on model
    # gemini-3-pro-image-preview requires 'global' location
    if "gemini-3" in config.IMAGE_MODEL.lower():
        location = "global"
    else:
        location = config.LOCATION
    
    logger.info(f"Using location: {location} for model: {config.IMAGE_MODEL}")
    
    # Nested Retry Strategy:
    # 1. Outer Loop: Prompt Variants - Created via LLM refinement based on QA
    # 2. Inner Loop: Generation Attempts - To catch 'biological/random' successes
    current_prompt = prompt
    last_error = None
    qa_fail_reason = ""
    qa_recommended_fix = ""
    
    # Best of N Fallback Strategy:
    # Collect all images that were generated but failed strict QA
    collected_images: list[Tuple[bytes, str, str]] = [] # (bytes, mime, prompt)
    
    for variant_idx in range(1, config.IMAGE_PROMPT_VARIANTS + 1):
        # Refine prompt if we've moved to a new variant due to QA failure
        if variant_idx > 1:
            logger.info(f"--- PROMPT VARIANT {variant_idx}/{config.IMAGE_PROMPT_VARIANTS} ---")
            if qa_fail_reason:
                logger.info(f"Refining prompt based on previous variant failure: {qa_fail_reason}")
                current_prompt = refine_prompt_vision(prompt, qa_fail_reason, qa_recommended_fix, style_context=style_context)
            else:
                current_prompt = paraphrase_prompt(prompt, variant_idx, style_context=style_context)
        
        for gen_idx in range(1, config.IMAGE_GEN_PER_PROMPT + 1):
            total_attempt = (variant_idx - 1) * config.IMAGE_GEN_PER_PROMPT + gen_idx
            try:
                logger.info(f"Attempting Generation {gen_idx}/{config.IMAGE_GEN_PER_PROMPT} for Prompt Variant {variant_idx}")
                logger.debug(f"Total pipeline attempt for this scene: {total_attempt}")
                start_time = time.time()
                
                # Initialize the client with timeout
                client = genai.Client(
                    vertexai=True,
                    project=config.PROJECT_ID,
                    location=location,
                    http_options=types.HttpOptions(timeout=300000) # 300s timeout
                )
        
                # Build the full prompt with style injection
                full_prompt = f"""{style['style_positive']}
    
{style['framing_style']}. Generate a high-quality image for a vertical video scene.

{f"🎬 DIRECTORIAL INTENT / CAMERA ANGLE (CRITICAL): {shot_type}" if shot_type else ""}
    
VISUAL DESCRIPTION:
{current_prompt}
    
{f"MANDATORY CHARACTER & STYLE RULES:{chr(10)}{style_context}" if style_context else ""}

COLOR GRADING: {style['lut']}
PALETTE: {style['palette']}
{style['rendering_notes']}
    
AVOID (do not include):
{style['style_negative']}
{avoid}
{config.CONTENT_AVOID_RULES}
    
**CRITICAL - VERTICAL 9:16 COMPOSITION (MANDATORY):**
- PORTRAIT ORIENTATION: The image MUST be taller than wide (like a phone screen held upright)
- SINGLE MAIN SUBJECT: Focus on ONE primary subject/character - avoid showing multiple people side-by-side
- If showing 2+ people: stack them vertically (one in front, one behind) - NEVER arrange horizontally
- WIDE/LONG SHOT SCALE: Prefer WIDE SHOTS or FULL BODY SHOTS to ensure all elements fit in one cohesive scene.
- Subject should be positioned at a distance unless a specific 'close-up' is requested.
- Camera position: eye level or looking UP at subject, NEVER from the side looking across
- For landscapes: show a TALL vertical slice (tree, waterfall, tower) not a wide panorama
- NEVER create images that look rotated or sideways - check your composition!
- FORCE UPRIGHT: All subjects and structures MUST be vertically aligned with the frame edges.
- NO LANDSCAPE BIAS: Avoid wide vistas; focus on tall vertical segments.
    
**CRITICAL - SINGLE CONTINUOUS IMAGE (NO TILING/STACKING):**
- Generate ONE SINGLE CONTINUOUS image that fills the entire 9:16 canvas
- Do NOT divide the image into panels, frames, sections, or tiles
- Do NOT create stacked horizontal bands or split-screen layouts
- Do NOT create triptych, diptych, or comic-panel arrangements
- The ENTIRE vertical space must be ONE COHESIVE UNIFIED SCENE
- NO collage effects, NO multiple photos combined, NO filmstrip layouts
- NEGATIVE CONSTRAINTS: split frame, grid, collage, multiple views, stacked boxes, border between frames, tilted horizon, sideways orientation, landscape bias, rotated composition, tilted ground, horizontal bias, side-down composition, 90 degree rotation.
    
TECHNICAL REQUIREMENTS:
- Vertical composition optimized for 9:16 aspect ratio (portrait phone screen)
- Balanced negative space to allow the subject to 'breathe' in the frame
- High quality, detailed imagery
- No text, logos, or watermarks
- Adults only (20+ years old)
"""
        
                # Detect if using Imagen or Gemini model
                is_imagen = "imagen" in config.IMAGE_MODEL.lower()
        
                if is_imagen:
                    response = client.models.generate_images(
                        model=config.IMAGE_MODEL,
                        prompt=full_prompt,
                        config=types.GenerateImagesConfig(
                            number_of_images=1,
                            aspect_ratio="9:16",
                            safety_filter_level="BLOCK_MEDIUM_AND_ABOVE",
                        ),
                    )
                    
                    if response.generated_images:
                        image = response.generated_images[0].image
                        image_bytes = image.image_bytes
                        mime_type = "image/png"
                    else:
                        raise ValueError("No image generated in Imagen response")
                
                else:
                    config_gen = types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                        temperature=config.IMAGE_TEMPERATURE,
                        image_config=types.ImageConfig(aspect_ratio="9:16"),
                    )
    
                    response = client.models.generate_content(
                        model=config.IMAGE_MODEL,
                        contents=full_prompt,
                        config=config_gen,
                    )
                    
                    image_bytes = None
                    mime_type = "image/png"
                    
                    if not response.candidates or not getattr(response.candidates[0], 'content', None) or not getattr(response.candidates[0].content, 'parts', None):
                        raise ValueError("API returned empty parts in single_shot")
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            image_data = part.inline_data.data
                            mime_type = part.inline_data.mime_type or "image/png"
                            if isinstance(image_data, str):
                                image_bytes = base64.b64decode(image_data)
                            else:
                                image_bytes = image_data
                            break
                    
                    if image_bytes is None:
                        raise ValueError("No image data in Gemini response")
                
                # Save locally if path provided
                if output_path:
                    output_path = Path(output_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    if mime_type != "image/png":
                        img = Image.open(io.BytesIO(image_bytes))
                        png_buffer = io.BytesIO()
                        img.save(png_buffer, format="PNG")
                        image_bytes = png_buffer.getvalue()
                        mime_type = "image/png"
                    output_path.write_bytes(image_bytes)
                
                # Visual QA Step
                if config.USE_VISUAL_QA:
                    is_valid, qa_fail_reason, qa_recommended_fix = verify_image_vision(
                        image_bytes, 
                        current_prompt,
                        visual_context=visual_context,
                        style_name=style.get('name', style_id),
                        style_description=style.get('rendering_notes', '')
                    )
                    
                    if not is_valid:
                        logger.warning(f"Image failed Visual QA. Attempt {gen_idx}/10 for this prompt. Reason: {qa_fail_reason}")
                        # Store for potential fallback if we never find a perfect image
                        collected_images.append((image_bytes, mime_type, current_prompt))
                        time.sleep(2)
                        continue # Inner loop: retry with same prompt
                    else:
                        # Reset failure info on success
                        qa_fail_reason = ""
                        qa_recommended_fix = ""
                
                # Visual Consistency QA Step
                if config.USE_CONSISTENCY_QA and reference_image_bytes:
                    is_valid_cons, cons_fail_reason, cons_recommended_fix = verify_visual_consistency(
                        image_bytes, 
                        reference_image_bytes, 
                        current_prompt,
                        style_name=style.get('name', style_id)
                    )
                    
                    if not is_valid_cons:
                        logger.warning(f"Image failed Consistency QA. Reason: {cons_fail_reason}")
                        qa_fail_reason = cons_fail_reason
                        qa_recommended_fix = cons_recommended_fix
                        continue # Inner loop: retry with same prompt
                
                elapsed = time.time() - start_time
                logger.info(f"Image generated successfully in {elapsed:.1f}s (Variant {variant_idx}, Gen {gen_idx})")
                return image_bytes, mime_type, current_prompt
                
            except (TimeoutException, TimeoutError) as e:
                logger.warning(f"Timeout on Variant {variant_idx}, Gen {gen_idx}: {e}")
                last_error = e
                time.sleep(5)
                continue # Inner loop: retry with same prompt
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error on Variant {variant_idx}, Gen {gen_idx}: {error_msg}")
                last_error = e
                
                if "SAFETY" in error_msg.upper() or "BLOCKED" in error_msg.upper():
                    logger.warning("Content blocked - moving to PROMPT refinement immediately")
                    qa_fail_reason = "SAFETY_BLOCK"
                    qa_recommended_fix = "Use a safer, more metaphorical allegory to avoid content filters."
                    break # Break inner loop to trigger prompt refinement in outer loop
                
                time.sleep(5)
                continue # Inner loop
        
        # If we finished the inner loop without a success, the outer loop will refine the prompt
        logger.warning(f"Failed 10 attempts for Prompt Variant {variant_idx}. Moving to next refinement.")
    
    # All retries exhausted
    logger.warning(f"--- FAILED ALL {config.IMAGE_PROMPT_VARIANTS * config.IMAGE_GEN_PER_PROMPT} ATTEMPTS ---")
    
    if collected_images:
        logger.info(f"Triggering 'Best of N' Fallback Selection from {len(collected_images)} collected images...")
        best_idx = select_best_image_fallback(
            [img[0] for img in collected_images],
            prompt=prompt,
            visual_context=visual_context,
            style_name=style_id
        )
        
        logger.info(f"Fallback selection complete. Using image {best_idx} from attempts.")
        return collected_images[best_idx]
    
    if last_error:
        raise last_error
    raise Exception(f"Failed to generate image after {config.IMAGE_PROMPT_VARIANTS * config.IMAGE_GEN_PER_PROMPT} attempts and 0 images collected.")


def generate_scene_image(
    scene_idx: int,
    image_prompt: str,
    image_avoid: str,
    global_style: Optional[dict] = None,
    output_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    style_id: str = config.DEFAULT_STYLE,
    visual_context: str = "",
    reference_image_bytes: Optional[bytes] = None,
    shot_type: Optional[str] = None,
    reference_images_list: Optional[list] = None  # List of (name, bytes) tuples
) -> Tuple[bytes, Path]:
    """
    Generate an image for a specific scene.
    
    Args:
        scene_idx: Scene index number
        image_prompt: The image prompt from the manifest
        image_avoid: Elements to avoid
        global_style: Global style settings dict
        output_dir: Base directory to save the image
        run_id: Unique run identifier (creates subfolder to avoid overwrites)
        style_id: Style preset ID (e.g., 'ultra_real', 'ghibli_dark')
        reference_image_bytes: Optional reference image for consistency QA
        
    Returns:
        Tuple of (image_bytes, local_path)
    """
    logger.info(f"Generating image for scene {scene_idx}")
    
    # Build style context from global style
    style_context = None
    if global_style:
        style_context = f"""
Color grading: {global_style.get('lut', 'cinematic')}
Palette: {global_style.get('palette', 'warm natural tones')}
Visual rules: {global_style.get('rules', 'consistent lighting')}
"""
    
    # Determine output path with run_id subfolder
    local_path = None
    if output_dir:
        if run_id:
            # Structure: out/scenes/{run_id}/01/image.png
            local_path = output_dir / run_id / f"{scene_idx:02d}" / "image.png"
        else:
            # Fallback: out/scenes/01/image.png
            local_path = output_dir / f"{scene_idx:02d}" / "image.png"
    
    # Generate the image with the specified style_id
    # Use chat edit mode for better consistency if enabled
    if config.USE_CHAT_EDIT_MODE:
        logger.info(f"[Scene {scene_idx}] Using CHAT EDIT MODE for image generation")
        image_bytes, _, final_prompt = generate_image_with_chat_edit(
            prompt=image_prompt,
            avoid=image_avoid,
            output_path=local_path,
            style_context=style_context,
            style_id=style_id,
            visual_context=visual_context,
            reference_image_bytes=reference_image_bytes,
            shot_type=shot_type,
            reference_images_list=reference_images_list
        )
    else:
        # Fallback to standard generation
        image_bytes, _, final_prompt = generate_image(
            prompt=image_prompt,
            avoid=image_avoid,
            output_path=local_path,
            style_context=style_context,
            style_id=style_id,
            visual_context=visual_context,
            reference_image_bytes=reference_image_bytes,
            shot_type=shot_type
        )
    
    return image_bytes, local_path, final_prompt


if __name__ == "__main__":
    # Test image generation
    logging.basicConfig(level=logging.INFO)
    
    test_prompt = """
    A mystical forest clearing at golden hour. Ancient oak trees with 
    glowing amber leaves surround a small pond reflecting the warm sunset. 
    Fireflies beginning to emerge, creating sparkles of light. 
    Photorealistic style with cinematic lighting. Vertical composition.
    """
    
    test_avoid = "people, text, logos, modern objects, buildings"
    
    print("Testing image generation...")
    try:
        image_bytes, mime = generate_image(
            prompt=test_prompt,
            avoid=test_avoid,
            output_path=Path("test_image.png")
        )
        print(f"Success! Generated {len(image_bytes)} bytes")
    except Exception as e:
        print(f"Failed: {e}")
