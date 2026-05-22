"""
TTS Generation Module - Google Cloud TTS (Gemini 3.1 Flash Voices).
Generates single-narrator audio using the google-genai SDK.
"""
import logging
import time
import wave
import subprocess
import re
from pathlib import Path
from typing import Optional, Tuple

from google import genai
from google.genai import types

from config import (
    NARRATOR_VOICE_ID,
    PROJECT_ID,
    LOCATION,
    TTS_MODEL
)

logger = logging.getLogger(__name__)

def get_audio_duration(file_path: Path) -> float:
    """Get the duration of an audio file using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of",
            "default=noprint_wrappers=1:nokey=1", str(file_path)
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"Could not get duration via ffprobe: {e}")
        return 0.0

def generate_dialogue_audio(
    scenes: list,
    output_path: Path,
    content_type: str = "story"
) -> Tuple[Optional[str], float]:
    """
    Generates continuous narration audio for all scenes using Gemini 3.1 Flash TTS.
    Returns (output_path_str, duration_seconds)
    """
    logger.info(f"[Gemini Flash TTS] Generating audio for {len(scenes)} scenes...")
    logger.info(f"   Voice: {NARRATOR_VOICE_ID} | Model: {TTS_MODEL}")

    # Initialize client explicitly with Vertex AI configuration
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    all_audio_content = []
    
    for scene_idx, scene in enumerate(scenes, start=1):
        dialogue = getattr(scene, 'dialogue', [])
        if not dialogue:
            continue

        # Combine all dialogue lines in the scene into one string
        scene_text = " ".join([getattr(line, 'text', '') for line in dialogue if getattr(line, 'text', '')])
        if not scene_text:
            continue
            
        logger.info(f"  > Synthesizing Scene {scene_idx} ({len(scene_text)} chars)...")
        
        # Remove any ElevenLabs brackets like [excited] just in case
        clean_text = re.sub(r'\[.*?\]', '', scene_text).strip()
        
        try:
            response = client.models.generate_content(
                model=TTS_MODEL,
                contents=clean_text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=NARRATOR_VOICE_ID,
                            )
                        )
                    ),
                )
            )
            
            # Extract raw PCM bytes from the inline_data
            if response.candidates and response.candidates[0].content.parts:
                audio_data = response.candidates[0].content.parts[0].inline_data.data
                all_audio_content.append(audio_data)
            else:
                logger.error(f"[ERROR] No audio data returned for scene {scene_idx}")
                
            time.sleep(1) # respect rate limits for Preview models
        except Exception as e:
            logger.error(f"[ERROR] Error generating audio for scene {scene_idx}: {e}")
            return None, 0.0

    if not all_audio_content:
        logger.error("[ERROR] No audio generated!")
        return None, 0.0

    # Extract PCM data and concatenate
    final_pcm = b""
    for idx, pcm_bytes in enumerate(all_audio_content):
        final_pcm += pcm_bytes
        # Add 0.5s silence between scenes to pace the video
        if idx < len(all_audio_content) - 1:
            # 24000 samples/sec * 2 bytes/sample * 1 channel = 48000 bytes/sec
            # 0.5 sec = 24000 bytes of zeros
            final_pcm += b'\x00' * 24000

    # Write final WAV
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), 'wb') as final_wav:
            final_wav.setnchannels(1)
            final_wav.setsampwidth(2) # 16-bit
            final_wav.setframerate(24000) # rate=24000
            final_wav.writeframes(final_pcm)
            
        # Get duration using ffprobe
        duration = get_audio_duration(output_path)
        logger.info(f"[SUCCESS] Audio saved: {output_path} ({duration:.2f}s)")
        return str(output_path), duration
    except Exception as e:
        logger.error(f"[ERROR] Failed to save final audio: {e}")
        return None, 0.0
