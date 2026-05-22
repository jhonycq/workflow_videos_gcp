"""
Configuration settings for the Video Generation Pipeline V3 - ElevenLabs Dual-Narrator.

IMPORTANT:
- Update PROJECT_ID and BUCKET_NAME with your GCP values
- Update ELEVENLABS_API_KEY with your ElevenLabs key
- Update NARRATOR_VOICE_ID and CHILD_VOICE_ID with your chosen voice IDs
"""
import os
from pathlib import Path

# =============================================================================
# GCP Settings (USER MUST CONFIGURE)
# =============================================================================
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "celtic-hub-492704-g2")  # Set your project ID -- proeyect id de matiassolito

# Set GOOGLE_CLOUD_PROJECT so all Google libraries can auto-detect the project
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", PROJECT_ID)
LOCATION = os.getenv("GCP_LOCATION", "us-central1")  # Vertex AI location
BUCKET_NAME = os.getenv("GCS_BUCKET", "mvp-new-scenes")  # Set your bucket name

# =============================================================================
# Google Cloud Text-to-Speech (Gemini 3.1 Flash Voices)
# =============================================================================
# Using the google-genai SDK for TTS
TTS_MODEL = "gemini-3.1-flash-tts-preview" 

# Voice IDs (Gemini Voices: Aoede, Charon, Fenrir, Kore, Puck)
NARRATOR_VOICE_ID = os.getenv("NARRATOR_VOICE_ID", "Aoede")  # Narrator voice
CHILD_VOICE_ID = os.getenv("CHILD_VOICE_ID", "Puck")         # Companion voice (if needed)

# TTS audio format settings
TTS_SAMPLE_RATE = 24000
TTS_CHANNELS = 1
TTS_SAMPLE_WIDTH = 2  # 16-bit

# TTS API settings
TTS_TIMEOUT_SECONDS = 120
TTS_MAX_RETRIES = 5

# =============================================================================
# Model IDs
# =============================================================================
# Image generation model (Gemini 3 Pro Image - global availability)
IMAGE_MODEL = "gemini-3-pro-image-preview"

# Text generation model for story/prompts (Gemini 3.1 Pro - available in global)
TEXT_MODEL = "gemini-3.1-pro-preview"
TEXT_LOCATION = "global"  # Gemini 3 requires global location

# Veo 3.1 Lite for video generation on Vertex AI
VIDEO_MODEL = "veo-3.1-lite-generate-001"

# Video generation mode: True = text-to-video (no image), False = image-to-video
# Text-to-video gives Veo more creative freedom for camera angles and scene variations
USE_TEXT_TO_VIDEO = False  # Set to False for image-to-video mode

# =============================================================================
# Style Settings
# =============================================================================
DEFAULT_STYLE = os.getenv("DEFAULT_STYLE", "auto")  # 'auto' uses LLM to suggest best style based on theme

# =============================================================================
# Temperature Settings (lower = more consistent, higher = more creative)
# =============================================================================
STORY_TEMPERATURE = 0.6  # For LLM story generation
IMAGE_TEMPERATURE = 0.3  # For image generation

# =============================================================================
# Deterministic Generation
# =============================================================================
USE_DETERMINISTIC_SEED = True  # Enable seed-based generation for consistency

# =============================================================================
# Video Generation Settings
# =============================================================================
NUM_SCENES = 15  # Default number of scenes (15 × 8s = 120s total)
SECONDS_PER_SCENE = 8  # Always 8 seconds for this MVP (adjusted by audio duration)
ASPECT_RATIO = "9:16"  # Vertical video format
RESOLUTION = 720  # 720p resolution

# Total video duration
TOTAL_DURATION = NUM_SCENES * SECONDS_PER_SCENE  # 120 seconds default

# =============================================================================
# Visual QA Settings (Vision-based verification)
# =============================================================================
USE_VISUAL_QA = True  # Enable Gemini 3 Flash to verify images before video generation
USE_CONSISTENCY_QA = True  # Compare scenes to ensure character/style consistency
USE_CHAT_EDIT_MODE = True  # Use multi-turn chat to edit images instead of regenerating when QA fails

# =============================================================================
# Budget & Cost Control
# =============================================================================
# Stop generation if we exceed this many seconds (safety limit)
MAX_SECONDS_BUDGET = 600  # 10 minutes max

# Approximate cost tracking (USD) - Updated based on Vertex AI pricing
# Veo 3.1 Lite: ~$0.15/second
# Image generation: ~$0.01-0.02 per image
# ElevenLabs: ~$0.03 per 1000 characters
COST_PER_SECOND_720P_VEO = 0.15  # Veo 3.1 Lite rate
COST_PER_IMAGE_NANO = 0.02  # Image generation estimate

# Budget warning threshold (percentage of max)
BUDGET_WARNING_THRESHOLD = 0.9  # Warn at 90%

# =============================================================================
# Retry Settings
# =============================================================================
MAX_RETRIES = 10  # Maximum retry attempts per operation
RETRY_BACKOFF = 2.0  # Exponential backoff multiplier
RETRY_INITIAL_WAIT = 1.0  # Initial wait time in seconds

# Hierarchical Image Retries (User Strategy)
IMAGE_PROMPT_VARIANTS = 3  # Different prompt refinements
IMAGE_GEN_PER_PROMPT = 5   # Generation attempts per prompt variant

# =============================================================================
# Rate Limiting & Parallelization
# =============================================================================
# Official Vertex AI Rate Limits (from documentation):
# - Veo 3.1 Lite: 10 requests/min
# - Veo 2.0: 20 requests/min
# - Gemini Image (gemini-3-pro): 5-20 requests/min
# - Imagen 3.0 Fast: 20 requests/min

# Parallelization settings (safe limits to avoid 429 errors)
PARALLEL_IMAGE_WORKERS = 1    # Sequential generation to avoid 429 with QA + retries
PARALLEL_VIDEO_WORKERS = 1    # Sequential: one scene at a time to avoid 429 and ensure order
IMAGE_RATE_LIMIT_RPM = 1      # Conservative limit for gemini-3-pro-image (set to 2 to prevent 429)
VIDEO_RATE_LIMIT_RPM = 4      # Safest limit to avoid Vertex AI congestion

# Legacy settings (for backward compatibility)
MAX_REQUESTS_PER_MINUTE = VIDEO_RATE_LIMIT_RPM
REQUEST_DELAY_SECONDS = 60 / VIDEO_RATE_LIMIT_RPM  # Auto-calculated delay

# =============================================================================
# Local Paths
# =============================================================================
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "out"
SCENES_DIR = OUTPUT_DIR / "scenes"
FINAL_VIDEO_PATH = OUTPUT_DIR / "final.mp4"

# =============================================================================
# Background Music Settings (Lyria 2)
# =============================================================================
MUSIC_VOLUME = 0.25  # Volume for background music (0.0-1.0), 25% default
MUSIC_CROSSFADE_DURATION = 1.0  # Crossfade between music segments (seconds)
MUSIC_FADE_IN_DURATION = 2.0  # Fade in at video start (seconds)
MUSIC_FADE_OUT_DURATION = 2.0  # Fade out at video end for seamless loop (seconds)

# =============================================================================
# Watermark Settings
# =============================================================================
WATERMARK_ENABLED = True  # Set to False to disable watermark
WATERMARK_IMAGE = os.getenv("WATERMARK_IMAGE", "")  # Path to watermark PNG
WATERMARK_OPACITY = 0.15  # 15% transparency
WATERMARK_TELEPORT_INTERVAL = 3  # Seconds between random position jumps
WATERMARK_SCALE = 0.08  # 8% of video width

# =============================================================================
# GCS Paths (templates)
# =============================================================================
def get_gcs_base_path(run_id: str) -> str:
    """Get base GCS path for a run."""
    return f"mvp/{run_id}"

def get_gcs_scene_path(run_id: str, scene_idx: int) -> str:
    """Get GCS path for a scene's assets."""
    return f"mvp/{run_id}/scenes/{scene_idx:02d}"

def get_gcs_manifest_path(run_id: str) -> str:
    """Get GCS path for the manifest file."""
    return f"mvp/{run_id}/manifest.json"

def get_gcs_final_video_path(run_id: str) -> str:
    """Get GCS path for the final concatenated video."""
    return f"mvp/{run_id}/final.mp4"

# =============================================================================
# Content Safety Rules
# =============================================================================
CONTENT_AVOID_RULES = """
- No gore, violence, blood, or graphic content
- No sexual or suggestive content
- No hate speech, discrimination, or offensive material
- No drugs, alcohol abuse, or substance-related content
- No self-harm or suicide references
- No copyrighted characters, logos, or trademarked content
- No legible text or brand names in images
- People/faces/hands ARE ALLOWED - use adults for realistic human representation
- Avoid children/minors to prevent policy restrictions
- STRICTLY PROHIBITED: comic panels, storyboards, split screens, borders, dividers, collage, grid layout, frames within frame, multiple images in one
- CRITICAL ORIENTATION RULES: No rotated 90 degrees, no sideways generation, no sideways subjects, no horizontal layout rotated vertically to fit into 9:16.
"""

# =============================================================================
# Prompt Templates
# =============================================================================
STORY_SYSTEM_PROMPT = """You are a creative storyteller and video director.
Generate engaging, family-friendly stories suitable for vertical video format (9:16).
All visual prompts must be in English. Voiceover text must be in Spanish.
Follow all content safety guidelines strictly.
V3: Generate a single-narrator story. The voiceover should be delivered by a single person, do not use dialogue or multiple characters for the narration."""

VIDEO_PROMPT_TEMPLATE = """
Scene {idx} of {total}: {description}

VISUAL DIRECTION:
{visual_direction}

CAMERA:
{camera_movement}

CONTINUITY:
{continuity_notes}

AUDIO:
VO (Spanish): "{voiceover_spanish}"
SFX: {sfx}
Ambience: {ambience}

TRANSITION: {transition}

AVOID: {avoid_list}
"""
