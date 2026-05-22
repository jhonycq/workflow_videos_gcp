"""
Add Background Music to Videos

Mixes background music with the video's existing audio (narration).
Automatically adjusts music volume to not overpower the voice.

Usage:
    python add_background_music.py <run_id> --genre fairy_tale
    python add_background_music.py 20260118_160120 --music /path/to/music.mp3
"""

import argparse
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent / "out"

# Audio mixing settings
MUSIC_VOLUME = 0.15  # 15% volume for background music (won't overpower narration)
NARRATION_VOLUME = 1.0  # Keep narration at full volume

# Fade settings
FADE_IN_DURATION = 2.0  # Seconds
FADE_OUT_DURATION = 3.0  # Seconds


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip())


def loop_music_to_duration(music_path: Path, duration: float, output_path: Path) -> Path:
    """
    Loop music file to match video duration with fade in/out.
    """
    logger.info(f"Preparing music track ({duration:.1f}s)...")
    
    # Build filter for looping, trimming, and fading
    filter_complex = (
        f"aloop=loop=-1:size=2s,"  # Loop indefinitely
        f"atrim=0:{duration},"  # Trim to video duration
        f"afade=t=in:st=0:d={FADE_IN_DURATION},"  # Fade in
        f"afade=t=out:st={duration - FADE_OUT_DURATION}:d={FADE_OUT_DURATION},"  # Fade out
        f"volume={MUSIC_VOLUME}"  # Lower volume
    )
    
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",  # Loop input
        "-i", str(music_path),
        "-t", str(duration),  # Duration
        "-af", f"afade=t=in:st=0:d={FADE_IN_DURATION},afade=t=out:st={duration - FADE_OUT_DURATION}:d={FADE_OUT_DURATION},volume={MUSIC_VOLUME}",
        "-acodec", "aac", "-b:a", "128k",
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Music preparation failed: {result.stderr}")
        raise RuntimeError("Music preparation failed")
    
    logger.info(f"Music prepared: {output_path}")
    return output_path


def mix_audio_with_video(
    video_path: Path,
    music_path: Path,
    output_path: Path
) -> Path:
    """
    Mix background music with video's existing audio.
    
    The video's audio (narration) is kept at full volume.
    The background music is lowered and faded.
    """
    logger.info("Mixing audio tracks...")
    
    # Get video duration
    duration = get_video_duration(video_path)
    
    # Prepare looped/faded music track
    temp_music = output_path.parent / "temp_music.m4a"
    loop_music_to_duration(music_path, duration, temp_music)
    
    # Mix: video audio + background music
    # Using amix filter to combine both audio streams
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),      # Input 0: video with narration
        "-i", str(temp_music),       # Input 1: prepared music
        "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=2[aout]",
        "-map", "0:v",               # Keep video from input 0
        "-map", "[aout]",            # Use mixed audio
        "-c:v", "copy",              # Copy video (no re-encode)
        "-c:a", "aac", "-b:a", "192k",
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Cleanup temp file
    temp_music.unlink(missing_ok=True)
    
    if result.returncode != 0:
        logger.error(f"Audio mixing failed: {result.stderr}")
        raise RuntimeError("Audio mixing failed")
    
    logger.info(f"Video with background music: {output_path}")
    return output_path


def add_background_music(
    run_id: str,
    genre: str = None,
    music_file: Path = None,
    use_lyria: bool = True
) -> Path:
    """
    Add background music to a video run.
    
    Args:
        run_id: The video run ID
        genre: Music genre to use (if no music_file provided)
        music_file: Specific music file to use
        use_lyria: Whether to try Lyria generation
        
    Returns:
        Path to the final video with background music
    """
    run_dir = BASE_DIR / run_id
    
    # Find input video (prefer subtitled version)
    input_video = run_dir / "final_with_subtitles.mp4"
    if not input_video.exists():
        input_video = run_dir / "final_with_narration.mp4"
    if not input_video.exists():
        input_video = run_dir / "final.mp4"
    
    if not input_video.exists():
        raise FileNotFoundError(f"No video found in {run_dir}")
    
    logger.info(f"Input video: {input_video}")
    
    # Get music file
    if music_file and music_file.exists():
        music_path = music_file
        logger.info(f"Using provided music: {music_path}")
    else:
        # Try to get music from gen_music module
        try:
            from gen_music import generate_background_music
            
            music_path = generate_background_music(
                genre=genre or "fairy_tale",
                duration=int(get_video_duration(input_video)),
                use_lyria=use_lyria
            )
            
            if not music_path:
                raise RuntimeError("No music available")
                
        except ImportError:
            logger.error("gen_music module not available")
            raise RuntimeError("No music source available")
    
    # Output path
    output_video = run_dir / "final_with_music.mp4"
    
    # Mix audio
    mix_audio_with_video(input_video, music_path, output_video)
    
    logger.info(f"✅ Done! Video with music: {output_video}")
    return output_video


def main():
    parser = argparse.ArgumentParser(description="Add background music to video")
    parser.add_argument("run_id", help="Video run ID")
    parser.add_argument("--genre", "-g", default="fairy_tale",
                        help="Music genre (calm, adventure, mysterious, happy, sad, dramatic, fairy_tale, horror, romantic, action)")
    parser.add_argument("--music", "-m", type=Path, help="Specific music file to use")
    parser.add_argument("--volume", "-v", type=float, default=0.15, help="Music volume (0.0-1.0)")
    parser.add_argument("--no-lyria", action="store_true", help="Skip Lyria, use local library only")
    
    args = parser.parse_args()
    
    global MUSIC_VOLUME
    MUSIC_VOLUME = args.volume
    
    try:
        output = add_background_music(
            run_id=args.run_id,
            genre=args.genre,
            music_file=args.music,
            use_lyria=not args.no_lyria
        )
        print(f"\n🎬 Video with background music: {output}")
    except Exception as e:
        logger.error(f"Failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
