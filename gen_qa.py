"""
Visual Quality Assurance module using Gemini 3 Flash Vision.
Verifies image composition, orientation, and style before video generation.
"""
import logging
import base64
from typing import Tuple, Optional
from google import genai
from google.genai import types

from config import PROJECT_ID, TEXT_MODEL, TEXT_LOCATION

logger = logging.getLogger(__name__)

# Using Gemini 3 Flash Preview as requested by the user for Vision QA
VISION_QA_MODEL = "gemini-3.1-pro-preview"

def verify_image_vision(image_bytes: bytes, prompt: str, visual_context: str = "", style_name: str = "custom", style_description: str = "") -> Tuple[bool, str, str]:
    """
    Verifies an image using Gemini 3 Flash Vision.
    
    Returns:
        (is_valid, fail_reason, recommended_fix)
    """
    logger.info(f"Performing Visual QA (Style: {style_name})...")
    
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=TEXT_LOCATION,
        http_options=types.HttpOptions(timeout=300000)
    )
    
    qa_prompt = f"""Identify if this image COMPLIES with all technical and artistic standards for a vertical video scene.
    
    SCENE DESCRIPTION: {prompt}
    SCENE NARRATIVE CONTEXT (STATE): {visual_context}
    REQUIRED STYLE: {style_name}
    STYLE DETAILS: {style_description}
    
    **CRITICAL STANDARDS (MUST COMPLY):**
    
    1. **VERTICAL 9:16 COMPOSITION (MANDATORY):**
       - PORTRAIT ORIENTATION: The image MUST be taller than wide (like a phone screen held upright).
       - GRAVITY AND ORIENTATION (CRITICAL): The subjects, horizons, and environments MUST be upright relative to the vertical frame. FAIL IMMEDIATELY if the image looks like a horizontal 16:9 scene that has been rotated 90 degrees (sideways characters, sideways horizon, sideways buildings).
       - FORCE UPRIGHT: All subjects and structures MUST be vertically aligned with the frame edges.
       - NO ROTATION: The image must NOT be rotated 90 degrees, look sideways, or be UPSIDE DOWN (INVERTED).
       - NATURAL ORIENTATION SANITY: The 'sky', 'ceiling', or 'tops of heads' MUST be near the top edge. The 'ground', 'floor', or 'feet' MUST be near the bottom edge. Any visual that puts ground at the left, right, or top is a CRITICAL FAILURE.
       - SINGLE MAIN SUBJECT: Focus on ONE primary subject - avoid showing multiple people side-by-side horizontally.
       - VERTICAL STACKING: If showing 2+ people, stack them vertically (one in front/behind) - NEVER arrange horizontally.
       - CAMERA POSITION: Eye level or looking UP at subject, NEVER from the side looking across.
    
    2. **SINGLE CONTINUOUS IMAGE (NO TILING/STACKING/PANELS):**
       - Generate ONE SINGLE CONTINUOUS image that fills the entire 9:16 canvas.
       - NO COMIC PANELS: Do NOT divide the image into panels, frames, sections, or tiles.
       - NO SPLIT-SCREEN: Do NOT create stacked horizontal bands or split-screen layouts.
       - NO COLLAGE: No multiple photos combined, no filmstrip, no grid arrangements.
       - UNIFIED SCENE: The entire vertical space must be ONE cohesive unified scene.
       - NO 'RECUAUDROS': The image must NOT look like a set of smaller boxes or frames.
    
    3. **LOGICAL CONCORDANCE & PHYSICAL ANCHORING (STRICT):**
       - PHYSICAL PLAUSIBILITY: Objects must follow natural physics relative to their nature. 
       - PHYSICAL ANCHORING (CRITICAL): Verify that characters and key objects are PHYSICALLY CONNECTED to their environment as described. If the prompt says they are "standing on a deck", they must NOT be floating or in a separate vessel unless explicitly stated.
       - SHIP/VEHICLE LOGIC: Ships, boats, and cars MUST be on their respective surfaces (water/road). A ship floating in the air is a LOGICAL FAILURE.
       - NARRATIVE ALIGNMENT: The image MUST represent the scene goal: "{prompt}". 
       - STATE CONSISTENCY (CRITICAL): The image MUST match the character and environmental states described in: "{visual_context}".
         * If context says "sleeping" -> character MUST have eyes closed.
         * FAIL if the character state contradicts the narrative context.
       - ENVIRONMENT CONSISTENCY: If the scene mentions water/ocean/sea, the background MUST show water. Characters should NOT appear on land if the story says they are at sea.

    4. **KEY OBJECT & SUBJECT FIDELITY (CRITICAL - DYNAMIC):**
       - Read the scene description and identify KEY OBJECTS (ships, vehicles, buildings, etc.)
       - Verify EACH key object matches its description in the prompt (Subject Bible details):
         * COLOR ACCURACY: If prompt specifies "black hull" and "orange funnels", the image MUST show exactly those colors.
         * PART ACCURACY: If prompt specifies "four funnels", fail if there are 2 or 5.
         * MATERIALITY: If prompt says "knitted" or "felt", the texture must be clearly visible.
       - UNIFIED SCENE: If the prompt describes characters ON a ship, they MUST be on THAT ship. Fail if they appear on a separate smaller boat or disconnected platform.

    5. **SCALE PLAUSIBILITY (CRITICAL - DYNAMIC):**
       - If the scene description implies a LARGE object (ship, building, mountain, dragon), verify it appears LARGE relative to characters
       - No "toy effect": Large objects should dominate the frame or appear in the distance - never smaller than the characters unless explicitly described as a miniature/toy
       - If both large objects AND characters appear, the large object MUST be significantly bigger

    6. **NEGATIVE CONSTRAINTS (FAIL IF PRESENT):**
       - Rotated 90 degrees, sideways, horizontal gravity, letterbox, split frame, grid, collage, multiple views, stacked boxes, border between frames, tilted horizon, sideways orientation, rotated composition, tilted ground, horizontal bias, side-down composition, 180 degree rotation, upside down, inverted orientation, sky-at-bottom, ground-at-top, head-at-bottom, comic panels, floating ships, objects in void, wrong object type vs prompt description, toy-sized large objects.
    
    7. **STYLE INTEGRITY:**
       - Verify it follows the requested style ({style_name}).
       - Check for {style_description}.
       - Material/Texture must match the intended style (e.g. if amigurumi, must see yarn fibers; if cinematic, must see realistic skin/lighting).
    
    8. **ANTHROPOMORPHIC FACE INTEGRITY (CRITICAL FOR OBJECT CHARACTERS):**
       - If the scene features an ANTHROPOMORPHIC object (refrigerator, phone, car, clock, etc. with a face):
         * Each character should have EXACTLY ONE FACE (two eyes + one mouth in the same location).
         * FAIL if a character appears to have MULTIPLE FACES (e.g., a face on the door AND another face visible inside when the door is open).
         * FAIL if facial features are duplicated or scattered across multiple surfaces of the same object.
         * The face should remain in a CONSISTENT LOCATION as defined (e.g., "face on the upper door" should stay on the upper door).
       - EXCEPTION: Only pass if the prompt EXPLICITLY describes multiple faces (e.g., "a two-headed monster").
    
    9. **TECHNICAL REQUIREMENTS:**
       - TEXT AND LOGOS: The image must NOT contain any text, watermarks, or logos UNLESS they are explicitly mentioned in the SCENE DESCRIPTION above.
         * If the description asks for specific text (e.g. "a sign that says 'X'", "embroidered words 'Y'"), then that text is ALLOWED and should be checked for accuracy.
         * If the description DOES NOT mention specific text, any text or logos found are a CRITICAL FAILURE.
       - ADULTS ONLY: All people shown must look 20+ years old.
       - BALANCED SPACE: Ensure a cohesive composition where the subject isn't cramped at the edges.
    
    10. **PRAGMATIC TOLERANCE:**
       - Do NOT fail for minor aesthetic variations (slight lighting changes, background cloud positions).
       - ONLY fail for CRITICAL technical errors (rotation, tiling, wrong style) or NARRATIVE-BREAKING logical/visual bugs (wrong ship type, wrong scale, wrong environment).
    
    RESPONSE FORMAT (JSON):
    {{
        "is_valid": boolean,
        "fail_reason": "Brief technical description of why it failed (if applicable)",
        "recommended_fix": "Clear instruction for the image generator to fix the error (e.g. 'Force single panel mode', 'Rotate clockwise', 'Remove horizontal bias')"
    }}
    """
    # Retry logic for QA to handle 429 limits gracefully
    max_retries = 3
    result = None
    for attempt in range(max_retries):
        try:
            # Create image part
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
            
            response = client.models.generate_content(
                model=TEXT_MODEL,
                contents=[qa_prompt, image_part],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            
            result = response.parsed
            
            # Fallback: if parsed is empty but text contains JSON, parse manually
            if not result and response.text:
                import json
                try:
                    result = json.loads(response.text)
                    logger.info("Successfully parsed JSON from response.text fallback")
                except json.JSONDecodeError:
                    pass
            
            if not result:
                 logger.warning("Visual QA returned empty response. Failing by default for safety.")
                 return False, "Empty QA response", "Retry generation"
            
            break # Success, break out of retry loop
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg and attempt < max_retries - 1:
                import time
                wait_time = 30 * (attempt + 1)
                logger.warning(f"QA hit API limit (429). Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
            logger.error(f"Visual QA Technical Error: {e}")
            # FAIL on technical error to avoid proceeding with unverified images
            return False, f"Vision QA Technical Error: {str(e)}", "Retry image generation (API timeout/error)"

    # Handle edge case where API returns a list instead of dict
    if isinstance(result, list):
        result = result[0] if result else {}

    is_valid = result.get("is_valid", True)
    fail_reason = result.get("fail_reason", "")
    recommended_fix = result.get("recommended_fix", "")
    
    if not is_valid:
        logger.warning(f"VISUAL QA FAILED: {fail_reason}")
        logger.info(f"Recommended fix: {recommended_fix}")
    else:
        logger.info("VISUAL QA PASSED.")
        
    return is_valid, fail_reason, recommended_fix

def verify_visual_consistency(current_image: bytes, reference_image: bytes, prompt: str, style_name: str = "custom") -> Tuple[bool, str, str]:
    """
    Compares current scene image with a reference image to ensure character and style consistency.
    
    Returns:
        (is_valid, fail_reason, recommended_fix)
    """
    logger.info(f"Performing Visual Consistency QA (Style: {style_name})...")
    
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=TEXT_LOCATION,
        http_options=types.HttpOptions(timeout=300000)
    )
    
    consistency_prompt = f"""Compare the CURRENT IMAGE with the REFERENCE IMAGE to ensure character and style consistency.
    
    SCENE GOAL: {prompt}
    REQUIRED STYLE: {style_name}
    
    **CONSISTENCY RULES:**
    1. **CHARACTER IDENTITY (STRICT):** The generated character MUST BE IDENTICAL to the reference character in structural shape, features, and vibe.
       - **FACES AND EYES (CRITICAL):** Eye color, age, face shape MUST match exactly.
       - **HAIR/FUR:** Color, length, and texture must match exactly.
       - **CLOTHING (CRITICAL):** The specific outfit, clothing items, hats, and colors MUST match the reference exactly. If the character wears a distinct shirt/uniform in the reference, missing or completely changing that outfit is a CRITICAL FAILURE.
    2. **MATERIAL INTEGRITY:** The texture and material quality (e.g., yarn for amigurumi, skin for realism, fur for kittens) must match the reference exactly.
    3. **ARTISTIC STYLE:** The lighting, color palette (LUT), and framing style must look like they belong to the SAME movie/project.
    4. **EVOLUTION EXCEPTION:** If the prompt explicitly says the character changed (e.g. "is now rich" or "turned into gold"), ignore the clothing mismatch but keep the facial features.
    5. **ENVIRONMENT CONTINUITY (CRITICAL):** If the SCENE GOAL does not explicitly mention a location change, the background/environment/setting MUST be visually consistent with the reference image. Same type of trees, same ground texture, same lighting atmosphere, same architectural elements. FAIL if the environment changes drastically (e.g., forest → desert, indoor → outdoor) without the scene explicitly describing a location transition.
    
    6. **PRAGMATIC TOLERANCE:**
       - Do NOT fail for minor background detail changes or slight angle shifts.
       - ONLY fail if:
         a) Character features (especially EYES) are inconsistent.
         b) The core style ({style_name}) is lost.
         c) There is a LOGICAL DISCONNECT (e.g. ship in the air, characters floating, missing ground).
         d) The environment/location changed drastically without narrative justification.
    
    RESPONSE FORMAT (JSON):
    {{
        "is_valid": boolean,
        "fail_reason": "Description of the inconsistency (e.g. 'The farmers hair turned from brown to red')",
        "recommended_fix": "Instruction to fix the prompt (e.g. 'Force brown hair to match reference', 'Add {style_name} texture to the background')"
    }}
    """

    max_retries = 3
    result = None
    for attempt in range(max_retries):
        try:
            # Create image parts
            current_part = types.Part.from_bytes(data=current_image, mime_type="image/png")
            reference_part = types.Part.from_bytes(data=reference_image, mime_type="image/png")
            
            response = client.models.generate_content(
                model=TEXT_MODEL,
                contents=[
                    "REFERENCE IMAGE (Target Look):", reference_part,
                    "CURRENT IMAGE (To Verify):", current_part,
                    consistency_prompt
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            
            result = response.parsed
            
            # Fallback: if parsed is empty but text contains JSON, parse manually
            if not result and response.text:
                import json
                try:
                    result = json.loads(response.text)
                    logger.info("Successfully parsed JSON from response.text fallback (Consistency)")
                except json.JSONDecodeError:
                    pass
            
            if not result:
                 logger.warning("Consistency QA returned empty response. Failing by default for safety.")
                 return False, "Empty Consistency QA response", "Retry generation to ensure character consistency"
            
            break # Success, exit retry loop
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg and attempt < max_retries - 1:
                import time
                wait_time = 30 * (attempt + 1)
                logger.warning(f"Consistency QA hit API limit (429). Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
            logger.error(f"Consistency QA Technical Error: {e}")
            # FAIL on technical error to avoid proceeding with potentially inconsistent images
            return False, f"Consistency QA Technical Error: {str(e)}", "Retry image generation (API timeout/error)"

    # Handle edge case where API returns a list instead of dict
    if isinstance(result, list):
        result = result[0] if result else {}

    is_valid = result.get("is_valid", True)
    fail_reason = result.get("fail_reason", "")
    recommended_fix = result.get("recommended_fix", "")
    
    if not is_valid:
        logger.warning(f"CONSISTENCY FAILED: {fail_reason}")
    else:
        logger.info("VISUAL CONSISTENCY PASSED.")
        
    return is_valid, fail_reason, recommended_fix

def select_best_image_fallback(image_list: list[bytes], prompt: str, visual_context: str = "", style_name: str = "custom") -> int:
    """
    Evaluates a list of images and selects the best one based on professionalism and narrative alignment.
    Used as a fallback when strict QA fails after multiple attempts.
    
    Returns:
        Index of the best image in the list.
    """
    if not image_list:
        return -1
    if len(image_list) == 1:
        return 0
        
    logger.info(f"Performing 'Best of N' Fallback Selection for {len(image_list)} images...")
    
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=TEXT_LOCATION,
        http_options=types.HttpOptions(timeout=300000)
    )
    
    selection_prompt = f"""Evaluate these {len(image_list)} images and select the ONE that is the most professional, 
    visually pleasing, and best represents the requested scene.
    
    SCENE DESCRIPTION: {prompt}
    SCENE NARRATIVE CONTEXT: {visual_context}
    REQUIRED STYLE: {style_name}
    
    CRITERIA FOR SELECTION:
    1. **Style Consistency**: Does it look like {style_name}? (e.g., if amigurumi, are yarn textures visible?)
    2. **Composition**: Does it follow the 9:16 vertical orientation correctly? (no rotation, no tiling)
    3. **Professionalism**: Which one has the fewest 'AI artifacts' or logical errors (floating objects, etc.)?
    4. **Narrative Accuracy**: Which one best depicts the specific action and characters described?
    
    Even if none are perfect, you MUST pick the one that is 'least broken' and most usable for a final video.
    
    RESPONSE FORMAT (JSON):
    {{
        "best_index": int (0 to {len(image_list)-1}),
        "reason": "Brief explanation of why this image was chosen over the others"
    }}
    """

    try:
        # Create image parts for all collected images
        image_parts = []
        for i, img_bytes in enumerate(image_list):
            image_parts.append(f"IMAGE {i}:")
            image_parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
            
        response = client.models.generate_content(
            model=VISION_QA_MODEL,
            contents=[selection_prompt] + image_parts,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        
        result = response.parsed
        
        # Fallback manual parse
        if not result and response.text:
            import json
            try:
                result = json.loads(response.text)
            except json.JSONDecodeError:
                pass
                
        best_index = result.get("best_index", 0) if result else 0
        reason = result.get("reason", "Unknown") if result else "Defaulting to first image"
        
        logger.info(f"Selected image index {best_index} as the best fallback. Reason: {reason}")
        
        # Ensure index is within bounds
        if best_index < 0 or best_index >= len(image_list):
            return 0
            
        return best_index
        
    except Exception as e:
        logger.error(f"Fallback Selection Error: {e}")
        return 0 # Default to first on error

if __name__ == "__main__":
    # Test with a dummy check if needed
    pass
