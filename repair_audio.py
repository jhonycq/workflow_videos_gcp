"""
Repair Script: Regenerates and re-assembles narration for an existing run.
Usage: python repair_audio.py --run-id <run_id>
"""
import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import subprocess

from config import OUTPUT_DIR, BUCKET_NAME
from gen_tts import generate_aligned_narration
from assemble import get_video_duration

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger("repair_audio")

def repair_run(run_id: str):
    run_dir = OUTPUT_DIR / run_id
    manifest_path = run_dir / "manifest.json"
    
    if not manifest_path.exists():
        logger.error(f"Manifest not found for run {run_id} at {manifest_path}")
        return
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest_data = json.load(f)
    
    # We only want to process scenes that actually have video generated
    # (assuming video generation was semi-successful)
    scenes = manifest_data.get("scenes", [])
    successful_scenes = [
        SimpleNamespace(**s) for s in scenes 
        if s.get("status") == "video_done" or s.get("video_gcs_path")
    ]
    
    if not successful_scenes:
        logger.error("No successful scenes found in manifest.")
        return

    logger.info(f"Repairing audio for run {run_id} ({len(successful_scenes)} scenes)")
    
    # Destination paths
    audio_path = run_dir / "narration_repaired.wav"
    final_path = run_dir / "final_with_narration_repaired.mp4"
    video_silent_path = run_dir / "video_silent.mp4"
    
    if not video_silent_path.exists():
        logger.warning(f"{video_silent_path} not found. Trying to find any silent video in run dir...")
        # Fallback to concatenate scene videos if video_silent doesn't exist? 
        # For now, assume user wants to repair an existing final_with_narration
        logger.error("video_silent.mp4 missing. Please run assemble first or ensure files exist.")
        return

    # Generate new aligned narration
    generate_aligned_narration(
        scenes=successful_scenes,
        output_path=audio_path,
        content_type=manifest_data.get("content_type", "general"),
        narrator_archetype=manifest_data.get("narrator_archetype", ""),
        narrator_voice=manifest_data.get("narrator_voice", "")
    )
    
    if not audio_path.exists():
        logger.error("Failed to generate repaired audio.")
        return
        
    logger.info(f"New narration generated: {audio_path}")
    
    # Merge with existing silent video
    video_duration = get_video_duration(video_silent_path)
    
    logger.info(f"Merging with video ({video_duration}s)...")
    
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(video_silent_path),
        "-i", str(audio_path),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-t", str(video_duration),
        str(final_path)
    ], check=True)
    
    logger.info(f"REPAIR COMPLETE: {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repair audio for a video run")
    parser.add_argument("--run-id", type=str, required=True, help="Run ID to repair")
    args = parser.parse_args()
    
    repair_run(args.run_id)
