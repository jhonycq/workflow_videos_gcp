"""
Fix Audio Sync for Run 20260124_152119
"""
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import List, Optional

# Import existing modules
from config import OUTPUT_DIR
from gen_tts import generate_single_audio, get_audio_duration

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("fix_audio")

RUN_ID = "20260124_152119"
RUN_DIR = OUTPUT_DIR / RUN_ID
MANIFEST_PATH = RUN_DIR / "manifest.json"

def clean_temp_audio(temp_dir: Path):
    """Remove existing temp audio files to ensure fresh generation."""
    if temp_dir.exists():
        for f in temp_dir.glob("*.wav"):
            f.unlink()
        logger.info(f"Cleaned temp directory: {temp_dir}")

def regenerate_aligned_audio():
    if not MANIFEST_PATH.exists():
        logger.error(f"Manifest not found: {MANIFEST_PATH}")
        return

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    scenes = manifest.get("scenes", [])
    if not scenes:
        logger.error("No scenes found in manifest")
        return

    logger.info(f"Loaded manifest with {len(scenes)} scenes")
    
    # Setup temp dir
    temp_dir = RUN_DIR / "temp_audio_fixed"
    temp_dir.mkdir(parents=True, exist_ok=True)
    clean_temp_audio(temp_dir)

    aligned_segments = []
    
    # Global settings
    narrator_voice = manifest.get("narrator_voice", "storyteller")
    content_type = manifest.get("content_type", "fable")
    
    for scene in scenes:
        idx = scene["idx"]
        duration = float(scene["seconds"])
        text = scene["narration_text"]
        
        # Override strict styles for this specific fable
        emotion = "wonder"  # Enforce consistent tone
        pace = "normal"
        intensity = 0.8
        
        logger.info(f"Processing Scene {idx}: Duration {duration}s")
        
        output_path = temp_dir / f"scene_{idx:02d}_fixed.wav"
        
        # Generate with strict limits
        try:
            # We use a slightly modified call here - we need to ensure it fits!
            # The existing generate_single_audio has logic for this, let's use it but monitor closely
            success_path = generate_single_audio(
                narration_texts=[text],
                tts_emotions=[emotion],
                tts_paces=[pace],
                tts_intensities=[intensity],
                total_duration=duration,
                output_path=output_path,
                content_type=content_type,
                narrator_archetype=manifest.get("narrator_archetype", ""),
                narrator_voice=narrator_voice
            )
            
            if success_path and Path(success_path).exists():
                # DOUBLE CHECK DURATION
                actual_dur = get_audio_duration(success_path)
                logger.info(f"Scene {idx} Generated: {actual_dur:.2f}s / Target: {duration:.2f}s")
                
                # Rigid enforcement: If it's still too long, force speed up with atempo
                if actual_dur > duration + 0.1: # 100ms tolerance
                    logger.warning(f"Scene {idx} audio too long. Forcing compression.")
                    tempo = actual_dur / duration
                    # Setup temp fix path
                    fix_path = temp_dir / f"scene_{idx:02d}_forced.wav"
                    
                    # ffmpeg atempo filter
                    # Note: atempo is limited to 2.0, chained for higher if needed (unlikely here)
                    subprocess.run([
                        "ffmpeg", "-y", "-i", str(success_path),
                        "-filter:a", f"atempo={tempo}",
                        "-c:a", "pcm_s16le",
                        str(fix_path)
                    ], check=True, capture_output=True)
                    
                    # Verify fix
                    fixed_dur = get_audio_duration(str(fix_path))
                    logger.info(f"Scene {idx} Compressed: {fixed_dur:.2f}s")
                    
                    # Replace original
                    fix_path.replace(output_path)
                
                # If too short, pad with silence (ffprobe checked above)
                elif actual_dur < duration - 0.1:
                     logger.warning(f"Scene {idx} audio too short. Padding.")
                     pad_path = temp_dir / f"scene_{idx:02d}_padded.wav"
                     subprocess.run([
                        "ffmpeg", "-y", "-i", str(success_path),
                        "-af", f"apad=whole_dur={duration}",
                        "-c:a", "pcm_s16le",
                        str(pad_path)
                     ], check=True, capture_output=True)
                     pad_path.replace(output_path)

                aligned_segments.append(output_path)
            else:
                 logger.error(f"Failed to generate Scene {idx}")
                 # Fallback silence
                 subprocess.run([
                    "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                    "-t", str(duration), "-c:a", "pcm_s16le", str(output_path)
                 ], check=True, capture_output=True)
                 aligned_segments.append(output_path)

        except Exception as e:
            logger.error(f"Error on Scene {idx}: {e}")
            # Fallback silence
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                "-t", str(duration), "-c:a", "pcm_s16le", str(output_path)
            ], check=True, capture_output=True)
            aligned_segments.append(output_path)

    # Concatenate all audio
    concat_list = temp_dir / "concat_list.txt"
    with open(concat_list, "w") as f:
        for p in aligned_segments:
            f.write(f"file '{p.resolve()}'\n")
            
    final_audio = RUN_DIR / "narration_fixed.wav"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-c:a", "pcm_s16le", str(final_audio)
    ], check=True)
    
    logger.info(f"Final Adjusted Audio created: {final_audio}")
    final_dur = get_audio_duration(str(final_audio))
    logger.info(f"Final Audio Duration: {final_dur:.2f}s")
    
    # Mux with video
    # Assuming video_silent.mp4 exists (it should from the previous run)
    video_silent = RUN_DIR / "video_silent.mp4"
    if not video_silent.exists():
        logger.error("video_silent.mp4 not found! Cannot mux.")
        return

    final_video_fixed = RUN_DIR / "final_fixed.mp4"
    
    logger.info("Muxing final video...")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(video_silent),
        "-i", str(final_audio),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-longest", # Match longest stream (video should be target)
        str(final_video_fixed)
    ], check=True)
    
    logger.info(f"SUCCESS! Fixed video saved to: {final_video_fixed}")

if __name__ == "__main__":
    regenerate_aligned_audio()
