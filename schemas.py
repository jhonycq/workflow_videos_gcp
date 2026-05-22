"""
Pydantic schemas for manifest validation - V3 Dual-Narrator Edition.

Defines the structure for story generation output and scene metadata.
V3: Uses per-scene interleaved dialogue instead of a global narration string.
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
import re


class GlobalStyle(BaseModel):
    """Global visual style settings for the entire video."""

    lut: str = Field(
        ...,
        description="Color grading/LUT style (e.g., 'cinematic warm', 'noir', 'vibrant')"
    )
    palette: str = Field(
        ...,
        description="Color palette description (e.g., 'warm earth tones with golden highlights')"
    )
    rules: str = Field(
        ...,
        description="Visual consistency rules to maintain across all scenes"
    )


class DialogueLine(BaseModel):
    """A single line of dialogue in a scene (V3: dual-narrator)."""

    speaker: Literal["narrator", "child"] = Field(
        ...,
        description="Who speaks this line: 'narrator' (adult female) or 'child' (companion)"
    )
    text: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Spanish dialogue text. May include Gemini expression tags like [whispers], [gasps], [excited]"
    )
    emotion: str = Field(
        default="neutral",
        description="Emotion for this specific line: neutral, suspense, fear, wonder, dramatic, calm, excited"
    )
    intensity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Emotional intensity 0.0 (subtle) to 1.0 (intense). Controls Gemini TTS emotion level."
    )


class Scene(BaseModel):
    """A single scene in the video story (V3: with per-scene dialogue)."""

    idx: int = Field(
        ...,
        ge=1,
        description="Scene index (1-based)"
    )
    seconds: Optional[int] = Field(
        default=8,
        description="Duration in seconds (determined by audio transcription length). Adjusted by audio-first pipeline."
    )
    image_prompt_en: str = Field(
        ...,
        min_length=100,
        max_length=2000,
        description="Detailed visual prompt in English (500-1500 chars)"
    )
    image_avoid: str = Field(
        ...,
        description="Elements to avoid in the image generation"
    )
    video_prompt_en: str = Field(
        ...,
        min_length=100,
        max_length=3000,
        description="Motion-focused video prompt (CAMERA + ACTION verbs only, no scene description)"
    )
    visual_context: str = Field(
        default="",
        description="INTERNAL: Detailed scene context and character states (e.g., 'Aurora is sleeping', 'Patient is pale'). Used for continuity and QA."
    )
    extend_scene: bool = Field(
        default=False,
        description="Whether this scene should be extended by an additional 7 seconds (for long continuous takes)."
    )
    extension_prompt_en: Optional[str] = Field(
        default=None,
        description="If extend_scene is true, provide the motion prompt for the extension segment."
    )
    motion_type: str = Field(
        default="image_to_video",
        description="Type of video generation: 'image_to_video' (hard cut) or 'interpolated' (seamless transition)."
    )

    # V3: Per-scene interleaved dialogue (narrator + child alternating)
    dialogue: List[DialogueLine] = Field(
        default_factory=list,
        description="Interleaved dialogue between narrator and child for this scene. Replaces global_narration."
    )

    # OPTIONAL/DEPRECATED in V3: kept for fallback compatibility
    narration_text: str = Field(
        default="",
        description="DEPRECATED (V3): Use dialogue instead. Left empty for backwards compatibility."
    )

    # TTS style parameters - LLM determines based on scene context (used as defaults for dialogue lines that don't specify)
    tts_emotion: str = Field(
        default="neutral",
        description="Scene-level default emotion (for dialogue lines that don't specify): neutral, suspense, fear, wonder, dramatic, calm, excited"
    )
    tts_pace: str = Field(
        default="normal",
        description="Scene-level default pace: slow, normal, fast (legacy, used in max_duration fallback)"
    )
    tts_intensity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Scene-level default intensity 0.0-1.0"
    )

    # Style preset for this scene
    style_id: str = Field(
        default="",
        description="Style preset ID (e.g., 'ghibli_dark', 'ultra_real'). If empty, uses global style."
    )

    # Optional fields for tracking
    shot_type: Optional[str] = Field(
        default=None,
        description="Cinematographic shot type (e.g. 'LOW ANGLE LS', 'BIRD_EYE ELS', 'MEDIUM CU'). Propagated to generators."
    )
    image_gcs_path: Optional[str] = Field(
        default=None,
        description="GCS path where the generated image is stored"
    )
    video_gcs_path: Optional[str] = Field(
        default=None,
        description="GCS path where the generated video is stored"
    )
    status: str = Field(
        default="pending",
        description="Generation status: pending, image_done, video_done, failed"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if generation failed"
    )

    @field_validator('video_prompt_en')
    @classmethod
    def validate_prompt_length(cls, v: str) -> str:
        """Validate prompt length for Veo 3.1"""
        char_count = len(v)
        if char_count < 100:
            raise ValueError(
                f"video_prompt_en too short ({char_count} chars), minimum 100"
            )
        if char_count > 3000:
            raise ValueError(
                f"video_prompt_en too long ({char_count} chars), maximum 3000"
            )
        return v


class ReferenceSheet(BaseModel):
    """Visual reference for character/environment/object consistency."""
    name: str = Field(
        ...,
        description="Short identifier (e.g., 'Santa Rosa', 'Dragon', 'Castle')"
    )
    type: str = Field(
        ...,
        description="Type: character, environment, object, or animal"
    )
    description: str = Field(
        ...,
        min_length=50,
        description="200-500 char detailed visual description for image generation"
    )


class Manifest(BaseModel):
    """Complete manifest for the V3 video generation pipeline with dual-narrator."""

    title: str = Field(
        ...,
        min_length=5,
        max_length=200,
        description="Title of the video/story"
    )
    global_style: GlobalStyle = Field(
        ...,
        description="Global visual style settings"
    )
    scenes: List[Scene] = Field(
        ...,
        min_length=1,
        description="List of scenes in order, each with interleaved narrator/child dialogue"
    )
    reference_sheets: Optional[List[ReferenceSheet]] = Field(
        default=None,
        description="Visual reference sheets for character/environment/object consistency (3-14 items)"
    )

    # Metadata fields
    run_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for this generation run"
    )
    total_duration_seconds: Optional[int] = Field(
        default=0,
        description="Total video duration in seconds"
    )
    status: str = Field(
        default="pending",
        description="Overall status: pending, in_progress, completed, failed"
    )
    style_id: Optional[str] = Field(
        default=None,
        description="Style preset ID used for image/video generation. Set by pipeline, persisted for --resume."
    )

    # Narrator configuration (V3: dual-narrator with fixed voice IDs in config)
    content_type: str = Field(
        default="general",
        description="Content type decided by LLM: horror, mystery, documentary, action, comedy, drama, fantasy, educational, fable, general"
    )
    narrator_archetype: str = Field(
        default="",
        description="Adult narrator persona description in English (e.g., 'A warm, experienced storyteller')"
    )
    narrator_voice: str = Field(
        default="",
        description="DEPRECATED (V3): Voice is fixed via config NARRATOR_VOICE_ID. Kept for backwards compat."
    )
    child_archetype: str = Field(
        default="",
        description="Child companion persona description in English (e.g., 'A curious, 8-year-old with wonder and occasional nervousness')"
    )


    @field_validator('scenes')
    @classmethod
    def validate_scene_indices(cls, v: List[Scene]) -> List[Scene]:
        """Ensure scene indices are sequential starting from 1."""
        for i, scene in enumerate(v, start=1):
            if scene.idx != i:
                raise ValueError(
                    f"Scene indices must be sequential. Expected {i}, got {scene.idx}"
                )
        return v

    def calculate_total_duration(self) -> int:
        """Calculate total video duration from all scenes."""
        return sum(scene.seconds or 8 for scene in self.scenes)

    def get_scene_by_idx(self, idx: int) -> Optional[Scene]:
        """Get a scene by its index."""
        for scene in self.scenes:
            if scene.idx == idx:
                return scene
        return None

    def update_scene_status(
        self,
        idx: int,
        status: str,
        error_message: Optional[str] = None
    ) -> None:
        """Update the status of a specific scene."""
        for scene in self.scenes:
            if scene.idx == idx:
                scene.status = status
                if error_message:
                    scene.error_message = error_message
                break

    def get_pending_scenes(self) -> List[Scene]:
        """Get all scenes that haven't been processed yet."""
        return [s for s in self.scenes if s.status == "pending"]

    def get_failed_scenes(self) -> List[Scene]:
        """Get all scenes that failed generation."""
        return [s for s in self.scenes if s.status == "failed"]

    def all_complete(self) -> bool:
        """Check if all scenes have been successfully generated."""
        return all(s.status == "video_done" for s in self.scenes)


class GenerationStats(BaseModel):
    """Statistics for tracking generation progress and costs."""

    run_id: str
    total_scenes: int = 0
    completed_scenes: int = 0
    failed_scenes: int = 0
    total_seconds_generated: int = 0
    estimated_cost_usd: float = 0.0
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    def update_progress(
        self,
        completed: int = 0,
        failed: int = 0,
        seconds: int = 0,
        cost: float = 0.0
    ) -> None:
        """Update generation statistics."""
        self.completed_scenes += completed
        self.failed_scenes += failed
        self.total_seconds_generated += seconds
        self.estimated_cost_usd += cost

    def get_progress_percentage(self) -> float:
        """Get completion percentage."""
        if self.total_scenes == 0:
            return 0.0
        return (self.completed_scenes / self.total_scenes) * 100


# JSON Schema for LLM output (for structured output) - V3 Edition
MANIFEST_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "global_style": {
            "type": "object",
            "properties": {
                "lut": {"type": "string"},
                "palette": {"type": "string"},
                "rules": {"type": "string"}
            },
            "required": ["lut", "palette", "rules"]
        },
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idx": {"type": "integer"},
                    "seconds": {"type": "integer", "description": "Hint (4/6/8). Overridden by audio-first pipeline."},
                    "image_prompt_en": {"type": "string"},
                    "image_avoid": {"type": "string"},
                    "video_prompt_en": {"type": "string"},
                    "visual_context": {"type": "string", "description": "Character states for continuity."},
                    "extend_scene": {"type": "boolean", "description": "Extend by 7s for long takes."},
                    "extension_prompt_en": {"type": "string", "description": "Motion for extension if extend_scene=true."},
                    "motion_type": {"type": "string", "enum": ["image_to_video", "interpolated"]},
                    "dialogue": {
                        "type": "array",
                        "description": "Interleaved narrator/child dialogue (V3).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "speaker": {"type": "string", "enum": ["narrator", "child"]},
                                "text": {"type": "string", "description": "Spanish dialogue, may include [emotion] tags"},
                                "emotion": {"type": "string", "enum": ["neutral", "suspense", "fear", "wonder", "dramatic", "calm", "excited"]},
                                "intensity": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                            },
                            "required": ["speaker", "text", "emotion", "intensity"]
                        }
                    },
                    "shot_type": {"type": "string", "description": "Cinematographic shot type (optional)"},
                    "tts_emotion": {"type": "string", "enum": ["neutral", "suspense", "fear", "wonder", "dramatic", "calm", "excited"]},
                    "tts_pace": {"type": "string", "enum": ["slow", "normal", "fast"]},
                    "tts_intensity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "style_id": {"type": "string"}
                },
                "required": ["idx", "image_prompt_en", "image_avoid", "video_prompt_en", "visual_context", "dialogue", "tts_emotion", "tts_pace", "tts_intensity"]
            }
        },
        "content_type": {"type": "string", "enum": ["horror", "mystery", "documentary", "action", "comedy", "drama", "fantasy", "educational", "fable", "general"]},
        "narrator_archetype": {"type": "string", "description": "Adult narrator persona"},
        "child_archetype": {"type": "string", "description": "Child companion persona"},
        "reference_sheets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": ["character", "environment", "object", "animal"]},
                    "description": {"type": "string"}
                },
                "required": ["name", "type", "description"]
            }
        }
    },
    "required": ["title", "global_style", "scenes", "content_type", "narrator_archetype", "child_archetype", "reference_sheets"]
}
