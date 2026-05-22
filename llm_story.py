"""
LLM Story Generation Module.

Uses Gemini to generate a complete story with scenes and prompts.
"""
import json
import logging
from typing import Optional

from google import genai
from google.genai import types

from config import (
    PROJECT_ID,
    LOCATION,
    TEXT_LOCATION,
    TEXT_MODEL,
    NUM_SCENES,
    SECONDS_PER_SCENE,
    CONTENT_AVOID_RULES,
    STORY_TEMPERATURE,
    DEFAULT_STYLE,
)
from schemas import Manifest, MANIFEST_JSON_SCHEMA
from styles import get_style_preset, STYLE_PRESETS

logger = logging.getLogger(__name__)


def create_story_prompt(
    theme: Optional[str] = None,
    num_scenes: Optional[int] = None,  # None = let LLM decide
    seconds_per_scene: int = SECONDS_PER_SCENE,
    style_id: str = DEFAULT_STYLE,
    max_duration: Optional[int] = None
) -> str:
    """
    Create the prompt for story generation with professional Director + Prompt Engineer approach.
    Optimized for Veo 3.1 Lite vertical video generation.
    
    If num_scenes is None, the LLM will determine the optimal number based on the story.
    """
    
    # Get style preset for injection
    style = get_style_preset(style_id)
    
    theme_instruction = ""
    if theme:
        theme_instruction = f"\nTHEME/TOPIC: {theme}\n"
    
    # Get optional prompt instructions for this style
    prompt_instructions = style.get('prompt_instructions', '')
    prompt_instructions_block = f"""
STYLE-SPECIFIC PROMPT INSTRUCTIONS:
{prompt_instructions}
""" if prompt_instructions else ""

    # Style lock block - prepended to ALL image and video prompts
    style_lock_block = f"""
=============================================================================
MANDATORY STYLE LOCK (CRITICAL - APPLY TO ALL SCENES)
=============================================================================
STYLE PRESET: {style['name']}

STYLE POSITIVE (include in EVERY image_prompt_en and video_prompt_en):
{style['style_positive']}

STYLE NEGATIVE (include in EVERY image_avoid):
{style['style_negative']}

COLOR GRADING: {style['lut']}
PALETTE: {style['palette']}
RENDERING: {style['rendering_notes']}
FRAMING: {style['framing_style']}
{prompt_instructions_block}
CRITICAL RULES:
- Every image_prompt_en MUST start with the STYLE POSITIVE block
- Every image_avoid MUST include the STYLE NEGATIVE block
- NEVER mix styles between scenes

⚠️ ORIENTATION RULES (EXTREMELY IMPORTANT - READ CAREFULLY) ⚠️:
- Every image_prompt_en MUST end with: 'Vertical 9:16, floor/ground at BOTTOM edge, sky/ceiling at TOP edge, characters standing UPRIGHT.'
- The scene must look like a NORMAL photo taken with phone held VERTICALLY
- Example CORRECT: "A clock on the wall. The floor is at the bottom of the image, the ceiling at the top."
- Example WRONG: "A clock rotated sideways" or any mention of horizontal/tilted orientation
- FORBIDDEN WORDS in image_prompt_en: "horizontal", "sideways", "tilted", "rotated", "inverted", "landscape orientation"
- Characters have HEAD at TOP and FEET at BOTTOM (like a person standing naturally)

- SINGLE CONTINUOUS FRAME: Absolutely no panels, tiling, or split screens.
"""

    
    # Scene count instruction - allow dynamic duration per scene
    duration_instruction = """- Duration per scene: YOU CHOOSE (4, 6, or 8 seconds per scene)
  * Use 8s for epic reveals, emotional moments, complex action sequences.
  * Use 6s for establishing shots, standard scenes, character actions.
  * Use 4s ONLY as a mathematical fallback to make the total duration fit perfectly. VARY the duration across scenes for better pacing."""
    
    if max_duration is not None and num_scenes is not None:
        # Both constraints: max_duration + forced scene count (from retry loop)
        scene_instruction = f"""- TARGET DURATION: EXACTLY {max_duration} seconds total.
  * ⚠️ HARD LIMIT: You MUST generate EXACTLY {num_scenes} scenes. NO MORE, NO LESS.
  * Distribute durations (4s, 6s, or 8s per scene) so the sum equals or is slightly under {max_duration}s.
  * ⚠️ CRITICAL: Do NOT exceed {num_scenes} scenes. Do NOT exceed {max_duration}s total. Keep dialogue SHORT.
{duration_instruction}"""
    elif max_duration is not None:
        scene_instruction = f"""- TARGET DURATION: EXACTLY {max_duration} seconds total.
  * You MUST mathematically compute the exact number of scenes required using ONLY 4s, 6s, and 8s durations to equal {max_duration}s.
  * For example, if target is 20s: you MUST generate exactly 3 scenes (8s + 8s + 4s = 20s) or (8s + 6s + 6s = 20s). Do not generate 16 scenes!
  * ⚠️ CRITICAL RULES ⚠️: Do NOT exceed the target duration. Keep the story concise to fit the EXACT number of scenes you computed.
{duration_instruction}"""
    elif num_scenes is not None:
        scene_instruction = f"""- Total scenes: EXACTLY {num_scenes} scenes (fixed)
{duration_instruction}"""
    else:
        scene_instruction = f"""- Total scenes & duration: YOU DECIDE. 
  * If the user prompt explicitly requests a specific total duration (e.g. 60 seconds), YOU MUST mathematically combine 4s, 6s, and 8s scenes to reach exactly or very close to that total duration.
  * If no duration is specified, target 2-4 minutes with 15-30 scenes.
{duration_instruction}"""
    
    return f"""You are a professional Script Director + Prompt Engineer + Virality Expert specialized in vertical 9:16 clips for AI video generators (Veo 3.1).

ROLE: Create VIRAL, cinematic, engaging stories optimized for Veo 3.1 Lite video generation.
Your PRIMARY GOAL is to generate content that hooks viewers in the FIRST 2-3 SECONDS and keeps them engaged until the end.

VIRALITY PRINCIPLES:
- HOOK FIRST: The opening must be irresistible - a shocking fact, unexpected visual, urgent question, or promise of revelation
- CURIOSITY GAP: Create tension that makes viewers NEED to see what happens next
- EMOTIONAL TRIGGERS: Awe, surprise, intrigue, wonder - evoke strong emotions immediately
- PATTERN INTERRUPT: Start with something unexpected that breaks the scroll pattern
- PAYOFF PROMISE: Signal early that staying will be rewarding

=============================================================================
LONG STORY MODE (CRITICAL)
=============================================================================
If the THEME provided contains a long, detailed narrative (not just a short prompt):
- You MUST preserve 100% of the narrative beats and chronological order.
- Do NOT summarize or skip parts.
- Expand EACH paragraph or key sentence of the provided story into its own detailed scene.
- Ensure the visual descriptions (image_prompt_en) reflect the specific details and atmosphere described in the source text.
- Maintain absolute consistency with the provided narrative.

{theme_instruction}

{style_lock_block}

=============================================================================
PRODUCTION REQUIREMENTS
=============================================================================
{scene_instruction}
- Aspect ratio: 9:16 (vertical format)
- Target: Family-friendly (G-rated)
- Narration: Latin American Spanish (neutral accent, no Spain Spanish)

=============================================================================
MANDATORY WORKFLOW: Characters → Story → Chained Scenes → Prompts
=============================================================================

**MANDATORY CHARACTER EVOLUTION & STATE MANAGEMENT:**
Characters are DYNAMIC. You must track their physical state, appearance, and situation through the story:
1. **Physical Transformations:** If a character's body changes (e.g., loss of a tail to gain legs, turning into an animal, growing wings), YOU MUST explicitly describe this new state in the `image_prompt_en` of EVERY scene where it applies.
2. **Aging & Time Progression:** If the story mentions time jumps (passing years, decades, or centuries), update the character description: add "grey hair", "wrinkles", "older face", "hunched posture", or even "skeletal/spirit form" if appropriate.
3. **Situational Appearance:** If a character gets wet, changes clothes, grows a beard, or gets injured, reflect this in the prompt.
4. **Override Mechanism:** In scenes with these changes, ignore the static "Character Bible" description and write: "CHARACTER UPDATE [Name]: [Full new physical description]". This ensures the AI doesn't revert to the original look (e.g., doesn't put the tail back on).
5. **Prompt Verbosity (MANDATORY):** In EVERY SINGLE scene, you MUST provide the FULL physical description of the characters (face features, EYE COLOR, eye shape, hair, skin, clothes).
6. **NO LLM LAZINESS:** Do NOT just say "Goldilocks is here". You MUST say "Goldilocks (blonde curly yarn hair, monolid blue felt eyes, yellow dress) is here". If you omit the eyes, the AI will default to black beads, and the project will FAIL.

7. **CONTINUITY & STATE TRACKING (CRITICAL):**
Use the `visual_context` field to store the CURRENT state of the world and characters. 
- If a character fell asleep in scene 3, scene 4-10 must have `visual_context: "[Character Name] is still deeply asleep"`.
- If it started raining in scene 5, subsequent scenes must have `visual_context: "Ground is wet, atmosphere is rainy"`.
- This field ensures that when we generate scene 10, we don't accidentally show the character awake or the ground dry.

STEP 1: CHARACTER BIBLE (CRITICAL FOR CONTINUITY)
- HAIR: Color, length, style, texture (must match ethnicity)
- BODY: Build, height (relative), posture
- CLOTHING: Detailed outfit description (colors, materials, style) - must be culturally appropriate
- ACCESSORIES: Jewelry, glasses, hats, bags, etc.
- DISTINGUISHING FEATURES: Unique marks, scars, tattoos, expressions
- ANIMATION STYLE: How they move, their typical expressions

**ANTHROPOMORPHIC FACE RULES (CRITICAL FOR NON-HUMAN CHARACTERS):**
- Each character (object, appliance, animal, etc.) has EXACTLY ONE FACE unless explicitly stated otherwise.
- The FACE LOCATION must be specified (e.g., "eyes and mouth on the upper door", "face on the front screen").
- NO additional faces, eyes, or mouths should appear on other parts of the character.
- When the character OPENS (door, lid, etc.), the interior should NOT have separate faces - the face stays on the EXTERIOR.
- Example: "Frigi the refrigerator: Friendly face with two blue eyes and smiling mouth ONLY on the upper freezer door. When Frigi opens, the interior shows shelves with food but NO additional faces on the interior."
- Example WRONG: A refrigerator with a face on the door AND another face visible inside when opened.

**IMPORTANT: Characters must have ethnically accurate features matching the story's cultural setting.**
For Korean stories: Korean facial features, Korean traditional clothing (hanbok, diving suits, etc.)
DO NOT give Korean characters Western or non-Asian features.

Example character definition (Korean):
"Young haenyeo woman: KOREAN ethnicity - oval face with soft features, warm brown monolid eyes with kind expression, straight black hair in a practical bun with loose strands, light tan skin with warm undertones, small nose with soft bridge, full natural lips. Wearing traditional black neoprene diving suit with orange trim, white cotton headscarf tied at the back, carrying a round orange buoy net (tewak). Moves with confident, practiced grace."

STEP 3: SUBJECT BIBLE (Objects, Vehicles, Landmarks) - CRITICAL FOR SCALABILITY
If the story centers on iconic non-living subjects (e.g., Titanic, a specific iPhone model, the Eiffel Tower, a branded product):
- Define the SUBJECT BIBLE with EXPLICIT COLORS and PARTS.
- **MANDATORY COLORS**: Specify the primary and secondary colors (e.g., "Titanic: Black hull, white upper decks, four yellow/orange funnels with black tops").
- **MATERIALITY**: Define the physical material (e.g., "brushed titanium", "felt and yarn", "oxidized copper").
- **PURPOSE**: This ensures consistency across all niches (News, Documentary, Viral, Brand).

STEP 2: FOR EACH SCENE, deliver exactly these 7 outputs:

    **LONG TAKES (extend_scene):**
    - Set `extend_scene` to true ONLY IF this specific scene needs to be an uninterrupted continuous shot that lasts longer than normal (e.g. following a character down a hallway, a long slow pan across a battlefield).
    - If true, you MUST provide an `extension_prompt_en` describing what happens in the *second half* of that continuous shot.
    - If false, omit `extension_prompt_en`.
    
    OUTPUT FORMAT PER SCENE:
1. **image_prompt_en** (English, 500-1500 chars):
   - **STRUCTURE (MANDATORY)**: Use XML-style tags within the string to separate reasoning from description:
     * `<thinking>`: Briefly plan the 3D layout, object placement, and spatial logic (1-2 sentences).
     * `<subject>`: Detailed description of the main subject(s) using details from the Bibles.
     * `<composition>`: Describe the vertical arrangement, lighting, and environment.
   - **ENVIRONMENT PERSISTENCE (CRITICAL):** In EVERY scene's `<composition>` tag, you MUST explicitly describe the current environment/location/background (e.g., "dark enchanted forest with tall oak trees and moss-covered ground", "rocky mountain path with grey boulders"). If the location has NOT changed from the previous scene, you MUST use the SAME environment description word-for-word. Do NOT leave the background vague or unspecified. The image generator has NO memory of previous scenes — if you don't describe the environment, it will invent a random one.
   - **MATERIAL & SUBJECT INTEGRITY (MANDATORY):** You MUST explicitly describe the physical material of EVERY main subject (e.g., "carved wood grain", "hyper-detailed aged leather"). 
   - **SUBJECT FIDELITY**: You MUST repeat the specific visual details defined in the SUBJECT BIBLE (colors, parts, iconic features). For example, do not just say "The ship"; say "The Titanic ship with its black hull, white upper decks, and four yellow-orange funnels".
   - MUST be highly descriptive, lighting, mood, colors, camera angle (9:16 optimized). Include character details from the Character Bible if applicable.
   - For non-human physical objects (like a refrigerator, clock, or chest), you **MUST** ensure they have EXACTLY ONE face located dynamically (e.g., "eyes and smiling mouth on the upper door"). DO NOT place faces on secondary parts (like the hinges, knobs, or interior shelves). If it opens, the inside MUST remain a normal object interior without weird faces.
   - Specify "cinematic lighting", "8k resolution".
   - **CAMERA PERSPECTIVE**: Use specific POV terms (e.g., "Low angle (Worm's eye view) looking UP", "High angle looking DOWN", "Bird's eye view from ABOVE").
   - If the character is moving towards something, the target MUST be "AHEAD" of them unless reached.
   - Example: "The finish line is CLEARLY AHEAD (upper frame). The hare is in LOWER frame, seen from behind (back view), facing UPWARD toward the finish line. He has NOT crossed it."
   - This prevents the generator from accidentally creating "after-the-fact" compositions.
   
2. **video_prompt_en** (English, 100-3000 chars):
   - **MANDATORY**: Only describe the motion and camera. **DO NOT describe the scene setting or character appearance here.** Veo 3.1 Lite uses the image as the semantic base.
   - **GOOD:** "The camera slowly pans right following the subject. Natural wind moves through the trees."
   - **BAD:** "A blonde girl in a red dress walks through a dark forest as the camera slowly pans right..." (Redundant description).
   - Environmental motion effects (wind, fog, dust, light shifts, falling petals, flowing water, shadows) are ENCOURAGED — they bring life to the scene.
   - **FORBIDDEN IN VIDEO PROMPTS:** Do NOT add off-story visual effects (random sparks, lens flares, unexplained white smoke explosions) unless they are explicitly part of the story.
   
3. **image_avoid** (English): Negative prompt (e.g., "text, watermarks, blurry, low resolution, extra fingers").
   
4. **visual_context** (English): 
   - INTERNAL state tracking. MUST always include ALL of the following:
     a) **CURRENT LOCATION** (e.g., "LOCATION: Dark enchanted forest with tall oak trees"). If the location has NOT changed from the previous scene, REPEAT the same location description.
     b) **CHARACTER STATES** (e.g., "Barnaby is holding the star", "The clock is broken").
     c) **ENVIRONMENTAL CONDITIONS** (e.g., "It is nighttime", "It is raining", "Ground is wet").
   - This prevents continuity errors across scenes. The image generator relies on this field to maintain environment consistency.

5. **dialogue** (V3): MANDATORY per-scene array. Each element: `{{"speaker": "narrator"|"child", "text": "...", "emotion": "...", "intensity": 0.0-1.0}}`.
   
6. **tts_emotion**, **tts_pace**, **tts_intensity**: Choose these purely based on the visual/scene mood.

7. **motion_type** (CRITICAL FOR CONTINUITY):
   - Choose exactly between EXACTLY TWO options: `"image_to_video"` OR `"interpolated"`.
   - USE `"interpolated"` ONLY IF the scene continues naturally from the PREVIOUS scene in the EXACT same location, with the same general camera angle, and continuous character action (e.g. Scene 1: Character walking down street -> Scene 2: Character keeps walking down same street). This tells the AI to fuse the end of Scene 1 into the start of Scene 2.
   - USE `"image_to_video"` for hard cuts, changes in physical location, large time jumps, or completely different camera angles (e.g. Scene 1: Outside house -> Scene 2: Inside house).
   - **MANDATORY EXCEPTION:** Scene 1 (idx=1) MUST ALWAYS use `"image_to_video"` because there is no previous scene to interpolate from.

8. **style_id** (Optional): A specific style override for *this scene only*, or "ultra_real", "ghibli_dark", etc. Leave empty string `""` to default to global style.
   
   **CRITICAL: VERTICAL ORIENTATION (9:16) - PREVENTS ROTATED CONTENT**
   
   THE PROBLEM: If you use horizontal spatial language like "behind", "beside", 
   "side by side", the image generator creates horizontal content and ROTATES it 
   to fit 9:16. Result: subjects appear LYING DOWN instead of UPRIGHT.
   
   THE SOLUTION: Use ONLY vertical/depth spatial language in ALL image prompts.
   
   **SPATIAL LANGUAGE CONVERSION TABLE (MANDATORY):**
   
   | ❌ BANNED (causes rotation)     | ✅ USE INSTEAD                          |
   |---------------------------------|-----------------------------------------|
   | "behind him/her"                | "ABOVE him/her" or "in UPPER frame"     |
   | "beside", "next to"             | "ABOVE/BELOW" or "in front of/back of"  |
   | "left to right"                 | "top to bottom"                         |
   | "side by side"                  | "one in front of the other"             |
   | "eye level"                     | "low angle looking UP"                  |
   | "horizon", "horizontal"         | "vertical lines", "rising upward"       |
   | "wide shot", "panoramic"        | "wide VERTICAL shot"                    |
   | "landscape view"                | "environmental vertical composition"    |
   | "stretching across"             | "rising upward", "extending upward"     |
   | "horizon line"                  | "NO HORIZON - vertical alignment ONLY"  |
   
   **AGGRESSIVE VERTICALITY RULES:**
   - NEVER describe a wide horizon or sweeping landscape.
   - Describe elements as "stacked vertically" or "rising from bottom to top".
   - Subjects MUST be UPRIGHT (vertically aligned) - use words like "tall", "vertical", "upright".
   **MANDATORY SPATIAL STRUCTURE FOR MULTI-SUBJECT SCENES:**
   When 2+ subjects appear, describe positions as:
   - "[Subject A] in FOREGROUND / LOWER portion of frame"
   - "[Subject B] in BACKGROUND / UPPER portion / looming from TOP"
   - "Depth composition: [foreground element], [far background element]"
   
   **MANDATORY ENDING FOR ALL image_prompt_en (choose one):**
   - "Vertical 9:16, environmental wide shot, subject at a distance."
   - "Vertical 9:16, depth composition, subjects at different depths."
   - "Vertical 9:16, low angle looking UP at subject, full body visible."
   
   **EXAMPLE CORRECTIONS:**
   
   ❌ WRONG: "Close-up of Mateo's face. Behind him, the polar bear emerges."
   ✅ CORRECT: "Close-up of Mateo's face in LOWER half of frame, head UPRIGHT.
   ABOVE him in UPPER frame, polar bear's head LOOMS DOWN from top.
   Depth composition: human FOREGROUND, bear BACKGROUND. Vertical 9:16, both subjects UPRIGHT."
   
   ❌ WRONG: "Two warriors standing side by side, facing each other."
   ✅ CORRECT: "Warrior A in FOREGROUND (lower frame), back partially visible.
   Warrior B in BACKGROUND (upper frame), facing camera.
   Depth composition. Vertical 9:16, subjects at different depths."
   
   ❌ WRONG: "Warrior standing in the valley, mountains across the horizon."
   ✅ CORRECT: "Warrior standing UPRIGHT, feet at BOTTOM, head near TOP.
   Mountain peak RISES ABOVE her toward sky. Valley floor at BOTTOM.
   Low angle looking UP. Vertical 9:16, subject UPRIGHT."
   
   **SELF-CHECK BEFORE FINALIZING image_prompt_en:**
   - NO horizontal words: "beside", "next to", "behind" (unless referring to depth), "left", "right".
   - NO "stacking" or "stacked" words (interferes with anti-tiling rules).
   - SCAN for RED FLAG words. If found, REWRITE:
   □ "behind" → "background" or "in upper frame"
   □ "beside" / "next to" → "in front of" or "above/below"  
   □ "side by side" → "depth composition" or "arranged vertically"
   □ "eye level" → "low angle" 
   □ "wide" / "panoramic" → "wide vertical"
   □ "fills frame" → "visible in center" or remove

    **CINEMATOGRAPHY, ANGLES & LIGHTING (MANDATORY):**
    Use professional director terminology to frame the shot (adapted for 9:16 vertical):
    - **SHOT SIZES**: "Extreme Long Shot" (ELS), "Long Shot" (LS/Full Body), "Medium Shot" (MS/Waist up), "Close-Up" (CU/Face/Detail). For complex scenes/groups, ALWAYS use Long Shots.
    - **CAMERA ANGLES**: "Low Angle" (Worm's-eye, looking up, makes subject heroic), "High Angle" (looking down, makes subject vulnerable), "Bird's-eye view", "Eye-Level", "Over-the-Shoulder" (OTS).
    - **LENSES/DEPTH**: "Wide-angle lens" (exaggerates depth/perspective), "Telephoto lens" (compresses background space), "Macro lens" (for extreme details), "Shallow depth of field".
    - **LIGHTING**: "Volumetric lighting" (light shafts), "Chiaroscuro" (high contrast), "Golden hour", "Rim lighting".
    - Scale: Full body visible, environment taking up 60% of vertical space unless ECU/CU.
    - Static image description ONLY (no motion, no timeline, no seconds).
    - **Character details must be EXPLICIT**: describe exactly what they wear, their face features, hair, accessories


2. **visual_context** (English):
   - Description of the character states and world status.
   - Be specific about physical/emotional states (e.g. "Aurora is SLEEPING", "The engine is ON FIRE", "The sky is now PITCH BLACK").
   - This is used by the Quality Assurance models to verify the scene.

3. **image_avoid** (English):
   - Content restrictions from: {CONTENT_AVOID_RULES}

3. **video_prompt_en** (English, 400-500 characters):
   - **IMPORTANT: IMAGE COMPOSITION RULES (Wide Shot, NO Stacking, etc.) DO NOT APPLY TO VIDEO.**
   - Video prompts focus ONLY on MOTION and direction. You can describe close-up details or dynamic actions here.
   - **WRITE 400-500 CHARACTERS** - stay within this range
   - For I2V: describe ONLY motion, NOT the scene (image already has that)
   - NO AUDIO - video is completely silent
   - START with CAMERA movement
   - MATCH timeline to YOUR CHOSEN duration:
     * 6s = 3 beats (0s-2s, 2s-4s, 4s-6s)
     * 8s = 3-4 beats (0s-2s, 2s-5s, 5s-8s)
   - Use 2-3 ACTION VERBS in CAPS
   - Add rich atmospheric details (natural wind, light shifts, shadows, reflections, environmental movement) to reach 400+ chars
   - Environmental effects (wind, fog, dust, light shifts, shadows, reflections) are ENCOURAGED — they add depth and motion to the scene.
   - **FORBIDDEN in video prompt:** Do NOT write "NO body deformations", "NO additional limbs", "Maintain exactly N legs" — the video generation system already handles structural integrity. These phrases suppress animation and make the character static.
   - **FORBIDDEN:** Do NOT add off-prompt visual effects (sparks, lens flares, white smoke explosions) unless they are part of the story.
   
   **DISTINCT CHARACTER ACTIONS (MANDATORY - CRITICAL):**
   - Every character in EVERY scene must perform a DIFFERENT, UNIQUE action.
   - NEVER repeat the same action for a character across scenes.
   - Secondary/background characters MUST also be doing something specific.
   - Track what each character did in previous scenes and ensure variety.
   
   **EXAMPLE 500-CHAR VIDEO PROMPT:**
   "CAMERA: Slow cinematic push in toward the subject's face. 
   MOTION: The wooden character SLOWLY TILTS his head toward the lens. 
   Deep shadows STRETCH and CRAWL across the coarse wood grain as the 
   light source shifts slightly. In the background, two figures 
   EXCHANGE a quiet glance and one REACHES for a hanging lantern.
   Dramatic flickering of firelight reflects in the glassy doll eyes. 
   Hand-crafted stop-motion look with a frame rate jitter that 
   emphasizes the tactile clay and wood textures."


4. **dialogue** (Spanish) - MANDATORY V3: PER-SCENE DUAL-NARRATOR:

   V3 DUAL-NARRATOR SYSTEM:
   - NARRADORA: Hidden adult female narrator. She establishes the scene, guides the narrative, describes the action. Professional, warm, cinematic tone.
   - COMPAÑERO: A second hidden voice — curious, reactive, emotional. Whispers questions, reacts to danger, gasps at surprises, comments on the action. Short, natural reactions. Do NOT define an age or physical appearance for this voice — it is purely a disembodied emotional presence that accompanies the narrator.

   FOR EACH SCENE, you MUST generate a `dialogue` array with interleaved turns between narrator and companion (2-5 lines per scene).

   **DIALOGUE FORMAT:**
   Each line has: `{{"speaker": "narrator" OR "child", "text": "Spanish text", "emotion": "...", "intensity": 0.0-1.0}}`
   (Note: use `"child"` as the speaker identifier for the companion voice — it is just a technical label.)

   **NARRATOR LINES (Norah):**
   - Descriptive, flowing Spanish prose that describes the scene visually and contextually.
   - Sets mood, explains cause-and-effect, moves story forward.
   - ~40-100 words per line (natural speech pace).
   - **REACTION RULE (CRITICAL):** When Norah speaks AFTER Daniela, she MUST briefly acknowledge what Daniela just said — a short phrase (3-10 words) that directly responds to Daniela's emotion, question, or comment — BEFORE continuing with the story narration. This creates natural chemistry and makes both voices feel connected.
     * If Daniela whispered in fear → Norah might say "Sí, tienes razón en tener miedo..." then continue.
     * If Daniela asked a question → Norah answers it briefly, then continues the story.
     * If Daniela gasped in shock → Norah mirrors or validates: "Y eso era solo el comienzo..." then continues.
     * If Daniela giggled → Norah can smile in her tone: "Sí, pero la risa pronto se convirtió en silencio..." then continues.
   - The reaction must feel natural and warm — NOT robotic or formulaic. Vary the phrasing every time.
   - Example (WITH reaction): "[gently] Exacto, Daniela... y fue justo en ese momento cuando el bosque comenzó a susurrar su secreto."
   - Example (no reaction needed if Norah opens the scene): "En el bosque oscuro, el viento murmuraba entre los árboles. La luna reflejaba un brillo plateado en el sendero."

   **COMPANION LINES (Daniela):**
   - Short, reactive, emotional Spanish. 1-3 sentences max.
   - Reacts to what the narrator said or what's happening visually on screen.
   - MUST use Gemini TTS Audio Tags to express emotion.

   **AUDIO TAGS — CRITICAL RULES:**
   ⚠️  TAGS MUST ALWAYS BE IN ENGLISH, even though the rest of the text is in Spanish.
   ⚠️  There is no fixed list — use any descriptive English word or phrase that captures the emotion.
   ⚠️  Place the tag at the START of the text, before the Spanish words.
   ⚠️  NEVER write tags in Spanish. [susurra], [emocionado], [asustado] are WRONG.

   CORRECT format: [english emotion description] Spanish text here.
   WRONG format:   [español] Texto en español aquí.

   - Examples for companion (Daniela):
     * "[whispering] ¿Escuchas eso? Me da mucho miedo..."
     * "[excited] ¡Oye, mira la luz! ¿Es un hada?"
     * "[frightened] ¡No, espera! ¿Y si nos ve?"
     * "[gasping in shock] ¡Es enorme!"
     * "[giggling] ¡Ja! ¡Se cayó!"
     * "[sobbing quietly] Quiero irme a casa..."
     * "[in awe, barely breathing] Es lo más bonito que he visto..."
     * "[nervously] N-no sé si deberíamos entrar..."

   - Examples for narrator (Norah) — OPENING a scene (no child reaction yet):
     * "[softly, with wonder] En el corazón del bosque, algo brillaba entre las sombras."
     * "[building tension] Cada paso los acercaba más al peligro."
     * "[triumphantly] ¡Por fin, el tesoro estaba ante ellos!"
     * "En el bosque oscuro, el viento murmuraba. La luna iluminaba el sendero." (no tag = neutral delivery)

   - Examples for narrator (Norah) — AFTER the child spoke (MUST acknowledge first):
     * "[gently] Sí, exacto... y fue justo entonces cuando la puerta comenzó a abrirse sola."  ← reacts to child's fear, then continues
     * "[warmly amused] Buena pregunta... nadie lo sabía con certeza. Pero lo que sí sabían era que esa noche todo cambiaría."  ← answers child's question, then continues
     * "[tenderly] Tienes razón en sentir eso... porque ese lugar guardaba un secreto que llevaba siglos esperando ser descubierto."  ← validates child's awe, then continues
     * "[with a knowing tone] Sí, parece gracioso... pero la risa pronto se convirtió en un silencio muy profundo."  ← echoes child's giggle, then pivots back to story

   - Tag style guide:
     * Simple: [whispering], [excited], [crying], [laughing], [nervous], [afraid]
     * Descriptive: [whispering with fear], [excited but cautious], [crying softly]
     * Action: [gasping], [sighing deeply], [gulping nervously], [shouting]
     * Mood: [in awe], [sarcastically], [triumphantly], [sadly], [with wonder]

   **ALTERNATION RULE:**
   - Typically: Narrator opens scene → Child reacts → Narrator briefly reacts to child + continues → Child reacts/asks
   - OR: Narrator → Child → Narrator briefly reacts + continues (3-turn scene for pacing variety)
   - NEVER have narrator speak twice in a row without child reaction (except very short scenes)
   - NEVER have narrator ignore what the child just said — always acknowledge it before continuing

   **MATHEMATICAL AUDIO-VIDEO SYNCHRONIZATION (CRITICAL - V3 DUAL VOICE):**
   - V3 uses TWO voices with pauses between speaker turns (~0.5s per turn change).
   - Effective speech rate with dual voices: ~10 chars/sec (accounts for turn-taking pauses).
   - The sum of ALL dialogue text across ALL scenes MUST be ≤ (target_duration × 10) characters.
   - Example: 60s target → MAX 600 total characters across all scenes combined.
   - Example: 75s target → MAX 750 total characters across all scenes combined.
   - Each scene's dialogue should be proportional to its video duration (8s scene → ~80 chars total for that scene, split between narrator and child).
   - ⚠️ ALWAYS aim for 10-15% UNDER the target duration, NEVER over. If target is 75s, aim for ~65s of dialogue.
   - If the target seems tight, use FEWER scenes with shorter dialogue rather than cramming text.

   **NARRATIVE & EMOTIONAL RULES:**
   - No announcements, no "subscribe" hooks, no breaking 4th wall.
   - Vocabulario simple, accents neutral (Latin American Spanish preferred).
   - The child's voice adds emotional dimension—use fear/wonder/excitement to highlight key moments.
   - Narrator voice remains grounded, cinematic, poetic but not pretentious.

5. **tts_emotion** - Choose based on scene mood (fallback for dialogue lines that don't specify):
   - "neutral" = Calm, informative
   - "suspense" = Building tension
   - "fear" = Ominous, frightening
   - "wonder" = Magical, awe-inspiring
   - "dramatic" = Intense, climactic
   - "calm" = Peaceful, reflective
   - "excited" = Energetic, enthusiastic

6. **tts_pace** - Choose based on scene rhythm (fallback):
   - "slow" = Atmospheric scenes
   - "normal" = Standard pacing
   - "fast" = Action, urgency

7. **tts_intensity** (0.0 to 1.0) - Default for scene:
   - 0.0-0.3 = Subtle, understated
   - 0.4-0.6 = Moderate emotional expression
   - 0.7-1.0 = High emotional impact

8. **shot_type** (OPTIONAL but STRONGLY RECOMMENDED):
   A single string that captures the directorial choice of shot size + camera angle for this scene. 
   This is used as a frame-anchor when generating the image AND the video so the AI models stay consistent.
   Format: `"[ANGLE] [SIZE]"` — choose from:
   - **ANGLES**: "EYE-LEVEL", "LOW ANGLE", "HIGH ANGLE", "BIRD_EYE", "WORM_EYE", "OTS" (over-the-shoulder)
   - **SIZE**: "ELS" (extreme long shot), "LS" (long shot), "MS" (medium shot), "CU" (close-up), "ECU" (extreme close-up)
   - Examples: `"LOW ANGLE LS"`, `"BIRD_EYE ELS"`, `"EYE-LEVEL CU"`, `"HIGH ANGLE MS"`
   - If the scene doesn't call for a specific angle, omit this field (it's optional).

=============================================================================
REFERENCE SHEETS (MANDATORY FOR ALL STORIES)
=============================================================================
You MUST generate a `reference_sheets` array in the manifest. These describe key visual elements
for consistency across all scenes. YOU decide how many are needed (3-14 total).

For EACH reference sheet:
- **name**: Short identifier (e.g., "Santa Rosa", "Dragon", "Castle")
- **type**: One of: "character", "environment", "object", "animal"
- **description**: 200-500 char visual description for image generation
  * Characters: face shape, eye color/shape, skin tone, hair style/color, clothing, accessories, body type, age
  * Environments: architecture, materials, lighting, atmosphere, key features
  * Objects: shape, material, color, texture, size, distinguishing marks
  * Animals: species, colors, markings, size, distinctive features

RULES:
- At least 1 reference per unique character in the story
- 1-3 references for key environments if they differ significantly
- References for important recurring objects or animals
- Total: 3-14 reference sheets depending on story complexity

=============================================================================
VIDEO PROMPT FORMAT (I2V - MOVEMENT PRIORITY)
=============================================================================

**CRITICAL FOR VEO:** The IMAGE already shows the scene. Your prompt
should describe ONLY what MOVES and HOW. Remove any static descriptions.

Write a RICH, DETAILED prompt (MINIMUM 400 characters required):

**STRUCTURE (ADAPT TIMELINE TO YOUR CHOSEN DURATION):**

CAMERA: [BOLD camera verb - dolly in / crane up / track left / pan right / zoom in]

For 6-second scenes (3 beats):
0s-2s: [SUBJECT] [VERB IN CAPS] [motion].
2s-4s: [Next action with CAPS VERB].
4s-6s: [Final action - end on clear visual].

For 8-second scenes (3-4 beats):
0s-2s: [SUBJECT] [VERB IN CAPS] [motion].
2s-5s: [Development with CAPS VERB].
5s-8s: [Final action/climax - end on clear visual].

**NOTE:** Transitions between scenes are AUTOMATIC (seamless interpolation).
Do NOT include "TRANSITION: Hard cut to black" or similar - just end on a clear visual.

**MANDATORY ACTION VERBS (use 2-3 per prompt):**
RISES, FALLS, DRIFTS, SWIRLS, EMERGES, APPROACHES, RETREATS,
TURNS, REACHES, BILLOWS, FLICKERS, PULSES, CRASHES, FLOATS,
GLIDES, SWEEPS, RACES, CRAWLS, TREMBLES, SHATTERS, UNFURLS,
SPIRALS, SURGES, BLOOMS, DISSOLVES, IGNITES

**CINEMATIC CAMERA MOVEMENT (MANDATORY - pick ONE primary motion):**

*— DEPTH & APPROACH —*
- **DOLLY/PUSH**: "Slow dolly in" (moves toward subject physically), "Dolly out" (moves away), "Fast push in" (dramatic approach). Best for: drama, romance, horror.
- **VERTIGO EFFECT (ZOLLY)**: "Dolly zoom — camera pulls back while lens zooms in simultaneously, background stretches unnervingly." Best for: psychological horror, disorientation, surreal/dreamcore, thriller.
- **THROUGH SHOT**: "Camera moves through [doorway/arch/window/foliage] into the scene, transitioning from darkness into light." Best for: narrative transitions, mystery, adventure, fantasy.

*— FOCUS & REVEAL —*
- **RACK FOCUS**: "Rack focus from [foreground] to [subject]" — shifts depth-of-field mid-clip for dramatic reveals. Best for: drama, mystery, romance.
- **REVEAL FROM BLUR**: "Camera starts completely out of focus, slowly pulls to sharp clarity revealing the subject." Best for: dream sequences, memory flashbacks, mystery, fantasy.
- **REVEAL FROM BEHIND**: "Camera slides laterally from behind an obstacle, slowly revealing the scene." Best for: suspense, horror reveals, epic landscapes, adventure.

*— VERTICAL MOVEMENT —*
- **CRANE/PEDESTAL**: "Crane up revealing [subject/landscape]" (camera flies vertically through space), "Pedestal down to ground level". Best for: epic, fantasy, adventure.
- **TILT**: "Tilt up from feet to face" (vertical rotation on axis). Best for: character introduction, drama, action.
- **TOP DOWN / GOD'S EYE VIEW**: "Camera pointing straight down, slow clockwise drift, revealing subject from directly above." Best for: epic scale, dance/choreography, battle scenes, surreal.

*— ORBIT & ARC —*
- **ORBIT**: "360 orbit around subject" (full circle), "True orbit 180" (half circle revealing new angle). Best for: character triumph, epic reveals, action climax.
- **SLOW CINEMATIC ARC**: "Camera moves in a wide, elegant curve revealing the subject's side profile and background environment." Best for: romance, drama, epic character moments.
- **BARREL ROLL / VORTEX**: "Camera spins 360 degrees clockwise while advancing forward, creating a vortex inception effect." Best for: action, sci-fi, surreal/dreamcore, psychedelic sequences.

*— TRACKING —*
- **TRACKING/FOLLOW**: "Tracking shot behind [subject]" (camera follows them from behind), "Side tracking parallel to subject". Best for: action, adventure, drama.
- **LEADING SHOT (BACKWARD TRACKING)**: "Camera retreats backward facing the subject, matching their walking speed, subject fills lower frame." Best for: drama, romance, character-driven moments, documentary.
- **POV WALK (FIRST PERSON)**: "First-person camera advances forward with natural head-bob motion, immersive POV perspective." Best for: horror, action, adventure, immersive documentary.

*— AERIAL / DRONE —*
- **DYNAMIC SHOTS**: "Handheld camera motion" (shaky, realism), "Smooth Steadicam glide" (fluid, elegant). Best for: documentary, drama, horror.
- **FPV DRONE DIVE**: "Aggressive FPV drone dive — rapidly descends down a vertical structure or canyon, turbulent motion." Best for: action, extreme sports, sci-fi chase sequences.
- **EPIC DRONE REVEAL**: "Camera rises high then tilts down to reveal vast landscape below, establishing epic scale." Best for: epic fantasy, adventure opening shots, nature documentary.
- **DRONE FLY OVER**: "High-altitude flight moving forward over the landscape at sustained speed, horizon in view." Best for: establishing shots, journey montages, epic scale.

*— ZOOM —*
- **SMOOTH OPTICAL ZOOM IN/OUT**: "Lens magnifies subject, camera stays stationary — background compresses." Best for: sports, nature, documentary, intimacy without physical movement.
- **SNAP ZOOM / CRASH ZOOM**: "Rapid smash zoom directly into subject's eyes/face — single violent push." Best for: comedy, action punctuation, horror jump-scare setup.
- **COSMIC HYPER ZOOM**: "Extreme zoom transition — pulls from macro surface detail out to cosmic wide view (or reverse), traversing multiple scales." Best for: sci-fi, nature documentary, dreamcore, surreal.

*— LENS EFFECTS —*
- **FISHEYE / PEEPHOLE**: "Extreme wide-angle fisheye distortion — circular barrel perspective warps the entire frame." Best for: surreal, horror POV, music video, experimental.
- **DUTCH ANGLE**: "Camera rolled on Z-axis — canted frame creates psychological unease and visual tension." Best for: horror, thriller, villain POV, psychological drama.

*— SPEED & TIMING —*
- **WHIP PAN**: "Whip pan to the right" — ultra-fast horizontal blur transition. Best for: POV shifts, comedy, action reveals.
- **SPEED/TIMING**: "Slow-motion tracking" (emotional weight), "Timelapse of passing shadows" (time passing), "Speed ramp from fast to slow mo" (dramatic highlight).

❌ NO conflicting movements: pick ONE primary motion (e.g. dolly in OR pan, never both simultaneously).
❌ For vertical 9:16: minimize heavy LATERAL panning — prioritize depth (in/out) and vertical (up/down/tilt) movements.
✅ Best for vertical 9:16: dolly in/out, crane up/down, tilt, rack focus, pedestal.

**CRITICAL I2V KINEMATICS RULE (PREVENT 3D MORPHING):**
❌ NEVER use verbs that force the subject to rotate their body toward the camera from a static starting position (e.g., "turns around", "spins", "looks back"). 
Since the AI only has one static 2D image, forcing a character to rotate their body will cause the AI to "melt" or "morph" their back into a face, creating horrific anatomical distortions.
✅ INSTEAD, use subtle linear body actions ("leans forward", "reaches hand", "breathes heavily", "kneels without turning") and rely entirely on CAMERA movement (dolly, orbit, crane) to create dynamic energy.

**WRONG (too static, too descriptive, no motion verbs):**
❌ "A ghostly figure in white robes stands in a misty forest. The atmosphere is eerie 
and the trees are tall with fog surrounding everything. Cherry blossoms fall gently."

**RIGHT - 6 seconds (350+ chars):**
✅ "CAMERA: Slow dolly in through atmospheric haze, depth of field shifts from foreground mist to subject. 0s-2s: Dense fog ROLLS across ancient stone floor, mysterious figure MATERIALIZES from shadows, spectral robes RIPPLE with supernatural energy. 2s-4s: Hair DRIFTS upward defying gravity, pale hands RISE slowly from sides, cold breath VAPORS visible in moonlight. 4s-6s: Head TURNS with eerie slowness, eyes PIERCE through darkness toward camera."

**RIGHT - 8 seconds (400+ chars):**
✅ "CAMERA: Crane up with slow dolly forward, revealing scale of scene, shallow depth creating dreamy bokeh. 0s-2s: Ground-level mist SWIRLS and CHURNS, spectral figure EMERGES from ancient doorway, pale silk robes BILLOW dramatically in supernatural wind, dust motes FLOAT upward in moonbeam shafts. 2s-5s: Figure GLIDES forward without footsteps, robes TRAIL behind leaving misty wisps, cherry petals SPIRAL around her ethereal form, hair FLOWS as if underwater. 5s-8s: Eyes slowly PULSE with pale blue luminescence, translucent hand REACHES toward camera with sorrowful gesture, tears of light STREAM down porcelain cheeks."


**KEY RULES:**
- START with camera movement
- MATCH timeline beats to YOUR CHOSEN duration (6s or 8s)
- Every sentence needs an ACTION VERB
- No "is", "has", "there is" (passive = static)
- Keep total prompt under 500 characters
- Image already shows: style, colors, composition → DON'T repeat these
- NO conflicting camera movements (pick ONE: dolly OR pan, not both)
- For vertical 9:16: avoid lateral movements, use dolly/crane/push/tilt

=============================================================================
GLOBAL STYLE (apply to all scenes)
=============================================================================
Provide:
- lut: Color grading (e.g., "cinematic warm", "noir contrast", "vibrant saturated")
- palette: Color palette description
- rules: Visual consistency rules INCLUDING the full character_bible definitions

=============================================================================
NARRATOR CONFIGURATION (YOU DECIDE BASED ON THEME)
=============================================================================

**content_type** - Choose the category that BEST fits your story:
- "horror" - Dark, scary, supernatural, terrifying
- "mystery" - Suspenseful, intriguing, detective-like
- "documentary" - Educational, informative, factual
- "action" - Exciting, fast-paced, thrilling
- "comedy" - Light, funny, entertaining
- "drama" - Emotional, touching, serious
- "fantasy" - Magical, otherworldly, epic
- "fable" - Traditional storytelling, moral tales, children's stories (RECOMMENDED: Use with "storyteller" voice/Leda)
- "educational" - Teaching, explaining, instructional
- "general" - Mixed or neutral content

**narrator_archetype** - Describe the narrator's persona IN ENGLISH. Be creative!
- Fable example: "A wise grandmotherly voice, weaving a classic tale with warmth and authority"
- Horror example: "A whispered voice from the shadows, telling tales that should remain untold"
- Documentary example: "An enthusiastic expert sharing fascinating discoveries"
- Fantasy example: "An ancient bard recounting legendary tales of heroes and magic"
- Comedy example: "A witty storyteller with perfect comedic timing and infectious energy"
- Mystery example: "A captivating detective unveiling secrets and intrigue"

**narrator_voice** - Choose the voice TONE (code maps to Gemini TTS voice):
- "deep" - Low, serious, commanding (best for horror, drama)
- "warm" - Friendly, inviting, comfortable (best for documentary, educational)
- "mysterious" - Ethereal, enigmatic, otherworldly (best for mystery, fantasy)
- "energetic" - Upbeat, excited, dynamic (best for action, comedy)
- "calm" - Soothing, peaceful, measured (best for calm stories)
- "storyteller" - Wise, rhythmic, maternal (best for fables, children's stories)

=============================================================================
OUTPUT FORMAT
=============================================================================
Return ONLY valid JSON matching the schema. No additional text.
Each scene MUST have: image_prompt_en, image_avoid, video_prompt_en, visual_context, dialogue array.
If extend_scene is True, you MUST include extension_prompt_en.
The dialogue array in each scene is MANDATORY - do not leave it empty! (2-5 lines per scene, alternating narrator/companion).
Include the narrator configuration: content_type, narrator_archetype (adult female voice), child_archetype (companion voice - emotional, reactive).

**CRITICAL REMINDER ON CONTINUITY:**
Character states (sleeping, awake, injured, transformed, wet, etc.) MUST be consistent across the `image_prompt_en` and correctly documented in `visual_context`. If a character is asleep in the previous scene, they stay asleep unless the story explicitly says they woke up."""



def generate_story(
    theme: Optional[str] = None,
    num_scenes: Optional[int] = None,  # None = let LLM decide (4-8 scenes)
    seconds_per_scene: int = SECONDS_PER_SCENE,
    style_id: str = DEFAULT_STYLE,
    max_duration: Optional[int] = None
) -> Manifest:
    """
    Generate a complete story with scenes using Gemini.
    
    Args:
        theme: Optional theme or topic for the story
        num_scenes: Number of scenes to generate (None = LLM decides 4-8 scenes)
        seconds_per_scene: Duration of each scene in seconds
        style_id: Style preset ID for visual consistency
        max_duration: Optional target total duration in seconds.
        
    Returns:
        Manifest object with validated story data
        
    Raises:
        ValueError: If LLM output is invalid
        Exception: If API call fails
    """
    logger.info(f"Generating story with max_duration: {max_duration}s, style: {style_id}, theme: {theme or 'random'}")
    
    # Get style preset for injection into prompt
    style = get_style_preset(style_id)
    
    # Initialize the Gemini client with global location for Gemini 3.1 Pro
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=TEXT_LOCATION
    )
    
    # Create the prompt with style injection
    prompt = create_story_prompt(theme, num_scenes, seconds_per_scene, style_id, max_duration)
    
    # Configure generation with JSON schema - lower temperature for consistency
    config = types.GenerateContentConfig(
        temperature=STORY_TEMPERATURE,  # Use config value (0.6) for more consistency
        top_p=0.95,
        max_output_tokens=65536,  # Increased for large stories (15+ scenes)
        response_mime_type="application/json",
        response_schema=MANIFEST_JSON_SCHEMA,
    )

    
    logger.info("Calling Gemini API for story generation...")
    
    # Generate content with retry for rate limits
    max_retries = 5
    response = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=TEXT_MODEL,
                contents=prompt,
                config=config,
            )
            break
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < max_retries:
                    wait = min(60, 5 * (2 ** (attempt - 1)))  # 5, 10, 20, 40, 60
                    logger.warning(f"Rate limited (attempt {attempt}/{max_retries}). Retrying in {wait}s...")
                    import time as _time
                    _time.sleep(wait)
                else:
                    logger.error(f"Rate limit persists after {max_retries} attempts.")
                    raise
            else:
                raise
    
    if response is None:
        raise RuntimeError("Story generation failed: no response received")
    
    # Parse the response
    try:
        response_text = response.text
        logger.debug(f"Raw LLM response: {response_text[:500]}...")
        
        # Parse JSON
        story_data = json.loads(response_text)
        
        # Post-process: Convert 'seconds' from string to int (Vertex AI requires STRING type for enums)
        if 'scenes' in story_data:
            for scene in story_data['scenes']:
                if 'seconds' in scene and isinstance(scene['seconds'], str):
                    scene['seconds'] = int(scene['seconds'])
        
        # Validate with Pydantic
        manifest = Manifest(**story_data)
        
        # Set calculated fields
        manifest.total_duration_seconds = manifest.calculate_total_duration()
        
        logger.info(f"Successfully generated story: '{manifest.title}' with {len(manifest.scenes)} scenes")
        
        return manifest
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.error(f"Response was: {response_text[:1000]}")
        raise ValueError(f"Invalid JSON from LLM: {e}")
    except Exception as e:
        logger.error(f"Failed to validate manifest: {e}")
        raise ValueError(f"Invalid manifest structure: {e}")


def regenerate_scene(
    manifest: Manifest,
    scene_idx: int,
    additional_instructions: Optional[str] = None
) -> Manifest:
    """
    Regenerate a specific scene (e.g., after a failure).
    
    Args:
        manifest: The current manifest
        scene_idx: Index of the scene to regenerate
        additional_instructions: Extra instructions (e.g., "avoid person generation")
        
    Returns:
        Updated manifest with regenerated scene
    """
    logger.info(f"Regenerating scene {scene_idx}")
    
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=TEXT_LOCATION
    )
    
    # Get the current scene and context
    current_scene = manifest.get_scene_by_idx(scene_idx)
    prev_scene = manifest.get_scene_by_idx(scene_idx - 1) if scene_idx > 1 else None
    next_scene = manifest.get_scene_by_idx(scene_idx + 1) if scene_idx < len(manifest.scenes) else None
    
    context_prompt = f"""Regenerate scene {scene_idx} of the story "{manifest.title}".

GLOBAL STYLE:
- LUT: {manifest.global_style.lut}
- Palette: {manifest.global_style.palette}
- Rules: {manifest.global_style.rules}

{"PREVIOUS SCENE CONTEXT: " + prev_scene.visual_context + " | " + prev_scene.image_prompt_en[:200] if prev_scene else "This is the first scene."}

{"NEXT SCENE CONTEXT: " + next_scene.visual_context + " | " + next_scene.image_prompt_en[:200] if next_scene else "This is the last scene."}

CURRENT SCENE (to regenerate):
{json.dumps(current_scene.model_dump(exclude={'status', 'error_message', 'image_gcs_path', 'video_gcs_path'}), indent=2)}

{f"ADDITIONAL INSTRUCTIONS: {additional_instructions}" if additional_instructions else ""}

IMPORTANT CHANGES NEEDED:
- Maintain story continuity
- Keep the same emotional tone
- Fix any issues that caused previous failure
- If person generation failed, try without people or use adults in silhouette/distant shots

Return ONLY the regenerated scene as JSON (single scene object, not full manifest)."""

    config = types.GenerateContentConfig(
        temperature=0.8,
        max_output_tokens=2048,
        response_mime_type="application/json",
    )
    
    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=context_prompt,
        config=config,
    )
    
    try:
        scene_data = json.loads(response.text)
        scene_data['idx'] = scene_idx
        scene_data['seconds'] = SECONDS_PER_SCENE
        scene_data['status'] = 'pending'
        scene_data['error_message'] = None
        
        # Update the scene in the manifest
        from schemas import Scene
        new_scene = Scene(**scene_data)
        
        for i, scene in enumerate(manifest.scenes):
            if scene.idx == scene_idx:
                manifest.scenes[i] = new_scene
                break
        
        logger.info(f"Successfully regenerated scene {scene_idx}")
        return manifest
        
    except Exception as e:
        logger.error(f"Failed to regenerate scene {scene_idx}: {e}")
        raise ValueError(f"Scene regeneration failed: {e}")


if __name__ == "__main__":
    # Test story generation
    logging.basicConfig(level=logging.INFO)
    
    print("Testing story generation...")
    manifest = generate_story(
        theme="A magical journey through an enchanted forest",
        num_scenes=3  # Small test
    )
    
    print(f"\nGenerated story: {manifest.title}")
    print(f"Total duration: {manifest.total_duration_seconds}s")
    print(f"\nGlobal style:")
    print(f"  LUT: {manifest.global_style.lut}")
    print(f"  Palette: {manifest.global_style.palette}")
    
    for scene in manifest.scenes:
        print(f"\nScene {scene.idx}:")
        print(f"  Image prompt: {scene.image_prompt_en[:100]}...")
        print(f"  Video prompt: {scene.video_prompt_en[:100]}...")
