"""
Repair Video Script - Re-sincroniza audio y video usando minterpolate.

Este script toma una carpeta de salida existente y:
1. Genera audios TTS por escena (y los guarda en scene_audios/)
2. Ajusta la duración de cada video de escena para coincidir con su audio
3. Usa minterpolate para generar frames intermedios y mantener fluidez
4. Re-ensambla el video final con audio sincronizado

Uso:
    python repair_video.py 20260111_170736

Sin costo adicional porque usa:
- Videos de escena existentes (scene_XX_silent.mp4)
- Regenera solo audio TTS (costo mínimo)
- Procesa localmente con FFmpeg
"""
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "out"


def get_audio_duration(audio_path: Path) -> float:
    """Get duration of audio file in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"Could not get audio duration for {audio_path}: {e}")
        return 0.0


def get_video_duration(video_path: Path) -> float:
    """Get duration of video file in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"Could not get video duration for {video_path}: {e}")
        return 0.0


def generate_scene_audios(manifest: dict, run_dir: Path) -> list[Path]:
    """
    Generate individual audio files for each scene and save them.
    Returns list of audio file paths.
    """
    from gen_tts import generate_single_audio
    
    audio_dir = run_dir / "scene_audios"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    scenes = manifest.get("scenes", [])
    content_type = manifest.get("content_type", "general")
    narrator_archetype = manifest.get("narrator_archetype", "")
    narrator_voice = manifest.get("narrator_voice", "")
    
    audio_paths = []
    
    for scene in scenes:
        idx = scene["idx"]
        audio_path = audio_dir / f"scene_{idx:02d}.wav"
        
        # Skip if already exists
        if audio_path.exists():
            logger.info(f"Scene {idx} audio already exists: {audio_path}")
            audio_paths.append(audio_path)
            continue
        
        logger.info(f"Generating audio for scene {idx}...")
        
        # Generate audio for this scene
        result = generate_single_audio(
            narration_texts=[scene["narration_text"]],
            tts_emotions=[scene.get("tts_emotion", "neutral")],
            tts_paces=[scene.get("tts_pace", "normal")],
            tts_intensities=[scene.get("tts_intensity", 0.5)],
            total_duration=float(scene["seconds"]),
            output_path=audio_path,
            content_type=content_type,
            narrator_archetype=narrator_archetype,
            narrator_voice=narrator_voice
        )
        
        if result and audio_path.exists():
            audio_paths.append(audio_path)
            logger.info(f"Scene {idx} audio saved: {audio_path}")
        else:
            logger.error(f"Failed to generate audio for scene {idx}")
            # Create silence as fallback
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=24000:cl=mono",
                "-t", str(float(scene["seconds"])), "-c:a", "pcm_s16le", str(audio_path)
            ], capture_output=True, check=True)
            audio_paths.append(audio_path)
    
    return audio_paths


def stretch_video_with_minterpolate(
    input_video: Path,
    output_video: Path,
    target_duration: float,
    fps: int = 24
) -> bool:
    """
    Stretch a video to target duration using minterpolate for smooth frame interpolation.
    
    This generates intermediate frames to maintain fluidity when stretching.
    """
    current_duration = get_video_duration(input_video)
    
    if current_duration <= 0:
        logger.error(f"Could not read duration of {input_video}")
        return False
    
    # Calculate the speed factor (PTS multiplier)
    # If target > current, we need to slow down (multiply PTS by > 1)
    speed_factor = target_duration / current_duration
    
    logger.info(f"Stretching {input_video.name}: {current_duration:.2f}s -> {target_duration:.2f}s (factor: {speed_factor:.2f}x)")
    
    if abs(speed_factor - 1.0) < 0.05:
        # Less than 5% difference, just copy
        logger.info(f"Duration difference < 5%, copying as-is")
        subprocess.run([
            "ffmpeg", "-y", "-i", str(input_video),
            "-t", str(target_duration),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            str(output_video)
        ], capture_output=True, check=True)
        return True
    
    try:
        # Use setpts to change speed, then minterpolate to generate missing frames
        # setpts=PTS*factor slows down (factor > 1) or speeds up (factor < 1)
        # minterpolate fills in the missing frames
        
        filter_complex = f"setpts={speed_factor}*PTS,minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_video),
            "-filter:v", filter_complex,
            "-t", str(target_duration),  # Ensure exact target duration
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-an",  # No audio
            str(output_video)
        ]
        
        logger.info(f"Running FFmpeg with minterpolate...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg minterpolate failed: {result.stderr}")
            # Fallback to simple setpts without minterpolate
            logger.info("Falling back to simple stretch without interpolation...")
            cmd_fallback = [
                "ffmpeg", "-y",
                "-i", str(input_video),
                "-filter:v", f"setpts={speed_factor}*PTS",
                "-t", str(target_duration),
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-an",
                str(output_video)
            ]
            subprocess.run(cmd_fallback, capture_output=True, check=True)
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Video stretching failed: {e}")
        return False


def repair_video(run_id: str, skip_audio_gen: bool = False) -> Path:
    """
    Main repair function that re-synchronizes video with audio.
    
    Args:
        run_id: The folder name in out/ (e.g., "20260111_170736")
        skip_audio_gen: If True, skip audio generation and use existing scene_audios/
    
    Returns:
        Path to the repaired video
    """
    run_dir = OUTPUT_DIR / run_id
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    # Load manifest
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    scenes = manifest.get("scenes", [])
    logger.info(f"Found {len(scenes)} scenes in manifest")
    
    # Step 1: Generate or load scene audios
    audio_dir = run_dir / "scene_audios"
    
    if skip_audio_gen and audio_dir.exists():
        logger.info("Using existing scene audios...")
        audio_paths = sorted(audio_dir.glob("scene_*.wav"))
    else:
        logger.info("Generating scene audios (this may take a few minutes)...")
        audio_paths = generate_scene_audios(manifest, run_dir)
    
    if len(audio_paths) != len(scenes):
        logger.warning(f"Audio count mismatch: {len(audio_paths)} audios vs {len(scenes)} scenes")
    
    # Step 2: Stretch each video to match its audio duration
    stretched_dir = run_dir / "stretched_videos"
    stretched_dir.mkdir(parents=True, exist_ok=True)
    
    stretched_videos = []
    
    for i, scene in enumerate(scenes):
        idx = scene["idx"]
        
        # Find the source video
        video_path = run_dir / f"scene_{idx:02d}_silent.mp4"
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            continue
        
        # Get the target duration from audio
        if i < len(audio_paths):
            target_duration = get_audio_duration(audio_paths[i])
        else:
            target_duration = float(scene["seconds"])
        
        logger.info(f"Scene {idx}: Target duration = {target_duration:.2f}s")
        
        # Stretch the video
        stretched_path = stretched_dir / f"scene_{idx:02d}_stretched.mp4"
        
        success = stretch_video_with_minterpolate(
            input_video=video_path,
            output_video=stretched_path,
            target_duration=target_duration
        )
        
        if success and stretched_path.exists():
            stretched_videos.append(stretched_path)
        else:
            logger.error(f"Failed to stretch scene {idx}")
            stretched_videos.append(video_path)  # Use original as fallback
    
    # Step 3: Concatenate all stretched videos
    logger.info("Concatenating stretched videos...")
    
    concat_file = run_dir / "concat_stretched.txt"
    with open(concat_file, "w") as f:
        for video in stretched_videos:
            f.write(f"file '{video.absolute()}'\n")
    
    video_only_path = run_dir / "video_stretched_silent.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(video_only_path)
    ], capture_output=True, check=True)
    
    # Step 4: Concatenate all audios
    logger.info("Concatenating scene audios...")
    
    audio_concat_file = run_dir / "concat_audio.txt"
    with open(audio_concat_file, "w") as f:
        for audio in audio_paths:
            f.write(f"file '{audio.absolute()}'\n")
    
    narration_path = run_dir / "narration_repaired.wav"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(audio_concat_file),
        "-c:a", "pcm_s16le",
        str(narration_path)
    ], capture_output=True, check=True)
    
    # Step 5: Merge video + audio
    logger.info("Merging video and audio...")
    
    final_path = run_dir / "final_repaired.mp4"
    
    video_duration = get_video_duration(video_only_path)
    audio_duration = get_audio_duration(narration_path)
    
    logger.info(f"Final video duration: {video_duration:.2f}s")
    logger.info(f"Final audio duration: {audio_duration:.2f}s")
    
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(video_only_path),
        "-i", str(narration_path),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        str(final_path)
    ], capture_output=True, check=True)
    
    logger.info(f"✅ Repaired video saved: {final_path}")
    
    return final_path


def main():
    parser = argparse.ArgumentParser(
        description="Repair video sync by stretching videos to match audio duration"
    )
    parser.add_argument(
        "run_id",
        help="Run folder name (e.g., 20260111_170736)"
    )
    parser.add_argument(
        "--skip-audio-gen",
        action="store_true",
        help="Skip audio generation if scene_audios/ already exists"
    )
    
    args = parser.parse_args()
    
    try:
        final_path = repair_video(args.run_id, args.skip_audio_gen)
        print(f"\n✅ Success! Repaired video: {final_path}")
    except Exception as e:
        logger.error(f"Repair failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
