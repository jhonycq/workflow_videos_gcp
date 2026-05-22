"""
Video Assembly Module using FFmpeg.

Downloads scene videos from GCS and concatenates them into a final video.
"""
import logging
import subprocess
from pathlib import Path
from typing import List, Optional

from config import OUTPUT_DIR, SCENES_DIR, FINAL_VIDEO_PATH, BUCKET_NAME
from storage import download_file, list_scene_videos, upload_file

logger = logging.getLogger(__name__)


def ensure_output_dirs() -> None:
    """Create output directories if they don't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCENES_DIR.mkdir(parents=True, exist_ok=True)


def download_all_scene_videos(
    run_id: str,
    bucket_name: str = BUCKET_NAME,
    output_dir: Optional[Path] = None
) -> List[Path]:
    """
    Download all scene videos from GCS to local filesystem.
    
    Args:
        run_id: The run identifier
        bucket_name: GCS bucket name
        output_dir: Local directory for downloaded videos
        
    Returns:
        List of local paths to downloaded videos (sorted by scene number)
    """
    import re
    output_dir = output_dir or SCENES_DIR
    
    # Create run-specific directory to avoid overwrites
    run_output_dir = output_dir / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading scene videos for run {run_id}")
    
    # List all videos in GCS
    video_uris = list_scene_videos(run_id, bucket_name)
    
    if not video_uris:
        raise ValueError(f"No videos found for run {run_id}")
    
    # Deduplicate by scene number - use a dictionary to keep only the most specific/recent one
    # If we have both gs://.../04/video.mp4 and gs://.../04/video.mp4/123/sample_0.mp4
    # The one with more slashes or the newer one is usually preferred.
    # We'll use a dict: scene_num -> uri
    deduplicated_uris = {}
    for uri in video_uris:
        match = re.search(r'/scenes/(\d+)/', uri)
        if match:
            scene_num = match.group(1)
            # If we already have this scene, prefer the 'deeper' one (more nested)
            # as Veo often generates sample_0.mp4 in a subdirectory
            if scene_num in deduplicated_uris:
                current_depth = deduplicated_uris[scene_num].count('/')
                new_depth = uri.count('/')
                if new_depth >= current_depth:
                    deduplicated_uris[scene_num] = uri
            else:
                deduplicated_uris[scene_num] = uri
    
    logger.info(f"Found {len(video_uris)} blobs, deduplicated to {len(deduplicated_uris)} scene videos")
    
    local_paths = []
    # Process sorted by scene number
    for scene_num in sorted(deduplicated_uris.keys()):
        uri = deduplicated_uris[scene_num]
        
        # Save in run-specific folder: out/scenes/{run_id}/{scene_num}_video.mp4
        local_path = run_output_dir / f"{scene_num}_video.mp4"
        
        download_file(uri, local_path, bucket_name)
        local_paths.append(local_path)
    
    logger.info(f"Downloaded {len(local_paths)} videos to {run_output_dir}")
    return local_paths


def create_concat_file(
    video_paths: List[Path],
    output_path: Optional[Path] = None
) -> Path:
    """
    Create a concat file for FFmpeg.
    
    Args:
        video_paths: List of video file paths
        output_path: Path for the concat file
        
    Returns:
        Path to the concat file
    """
    output_path = output_path or (OUTPUT_DIR / "concat_list.txt")
    
    logger.info(f"Creating concat file with {len(video_paths)} videos")
    
    with open(output_path, 'w') as f:
        for video_path in video_paths:
            # Use absolute paths and escape for FFmpeg
            abs_path = video_path.resolve()
            f.write(f"file '{abs_path}'\n")
    
    logger.info(f"Created concat file: {output_path}")
    return output_path


def concatenate_videos(
    video_paths: List[Path],
    output_path: Optional[Path] = None,
    use_reencode: bool = False
) -> Path:
    """
    Concatenate multiple videos into a single video using FFmpeg.
    
    Args:
        video_paths: List of video file paths (in order)
        output_path: Output path for final video
        use_reencode: If True, re-encode instead of stream copy (slower but safer)
        
    Returns:
        Path to the final concatenated video
    """
    output_path = output_path or FINAL_VIDEO_PATH
    ensure_output_dirs()
    
    logger.info(f"Concatenating {len(video_paths)} videos to {output_path}")
    
    # Create concat file
    concat_file = create_concat_file(video_paths)
    
    # Build FFmpeg command
    if use_reencode:
        # Re-encode for compatibility (slower)
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(output_path)
        ]
    else:
        # Stream copy (fast, but requires same codec/params)
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(output_path)
        ]
    
    logger.info(f"Running FFmpeg: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"FFmpeg completed successfully")
        logger.debug(f"FFmpeg stdout: {result.stdout}")
        
        # Verify output exists
        if not output_path.exists():
            raise RuntimeError("FFmpeg completed but output file not found")
        
        file_size = output_path.stat().st_size
        logger.info(f"Final video created: {output_path} ({file_size / 1024 / 1024:.2f} MB)")
        
        return output_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed with code {e.returncode}")
        logger.error(f"FFmpeg stderr: {e.stderr}")
        
        # If stream copy failed, try with re-encode
        if not use_reencode:
            logger.warning("Stream copy failed, retrying with re-encode...")
            return concatenate_videos(video_paths, output_path, use_reencode=True)
        
        raise RuntimeError(f"FFmpeg concatenation failed: {e.stderr}")


def assemble_final_video(
    run_id: str,
    bucket_name: str = BUCKET_NAME,
    upload_to_gcs: bool = True
) -> Path:
    """
    Full assembly pipeline: download videos, concatenate, optionally upload.
    
    Args:
        run_id: The run identifier
        bucket_name: GCS bucket name
        upload_to_gcs: Whether to upload final video to GCS
        
    Returns:
        Local path to the final video
    """
    logger.info(f"Starting final video assembly for run {run_id}")
    
    # Download all scene videos
    video_paths = download_all_scene_videos(run_id, bucket_name)
    
    # Set output path in run-specific folder
    run_output_dir = OUTPUT_DIR / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    final_path = run_output_dir / "final.mp4"
    
    # Concatenate
    final_path = concatenate_videos(video_paths, final_path)
    
    # Upload to GCS if requested
    if upload_to_gcs:
        gcs_path = f"mvp/{run_id}/final.mp4"
        upload_file(final_path, gcs_path, bucket_name)
        logger.info(f"Uploaded final video to gs://{bucket_name}/{gcs_path}")
    
    logger.info(f"Assembly complete: {final_path}")
    return final_path


def get_video_duration(video_path: Path) -> float:
    """
    Get the duration of a video file using FFprobe.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Duration in seconds
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"Could not get duration for {video_path}: {e}")
        return 0.0


def verify_final_video(video_path: Path, expected_scenes: int) -> dict:
    """
    Verify the final video was assembled correctly.
    
    Args:
        video_path: Path to the final video
        expected_scenes: Number of scenes expected
        
    Returns:
        Dictionary with verification results
    """
    from config import SECONDS_PER_SCENE
    
    results = {
        "exists": video_path.exists(),
        "file_size_mb": 0,
        "duration_seconds": 0,
        "expected_duration": expected_scenes * SECONDS_PER_SCENE,
        "is_valid": False
    }
    
    if not results["exists"]:
        return results
    
    results["file_size_mb"] = video_path.stat().st_size / 1024 / 1024
    results["duration_seconds"] = get_video_duration(video_path)
    
    # Allow 10% tolerance for duration
    duration_diff = abs(results["duration_seconds"] - results["expected_duration"])
    results["is_valid"] = duration_diff <= (results["expected_duration"] * 0.1)
    
    return results


def assemble_with_tts(
    run_id: str,
    narration_texts: list[str],
    scene_durations: list[float],
    bucket_name: str = BUCKET_NAME,
    upload_to_gcs: bool = True,
) -> Path:
    """
    Assemble final video with TTS narration.
    
    Steps:
    1. Download scene videos from GCS
    2. Strip audio from each video
    3. Generate TTS audio for each scene
    4. Concatenate all silent videos with fades
    5. Concatenate all TTS audio
    6. Merge video + TTS audio
    
    Args:
        run_id: The run identifier
        narration_texts: List of narration text for each scene
        scene_durations: List of scene durations in seconds
        bucket_name: GCS bucket name
        upload_to_gcs: Whether to upload final video to GCS
        
    Returns:
        Path to the final video with TTS
    """
    from gen_tts import generate_scene_tts
    
    logger.info(f"Starting TTS assembly for run {run_id}")
    
    # Step 1: Download scene videos
    video_paths = download_all_scene_videos(run_id, bucket_name)
    run_output_dir = OUTPUT_DIR / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    total_duration = sum(scene_durations)
    
    # Step 2: Strip audio from videos and add fades
    silent_videos = []
    for i, video_path in enumerate(video_paths):
        scene_idx = i + 1
        silent_path = run_output_dir / f"{scene_idx:02d}_silent.mp4"
        
        # FFmpeg to strip audio (straight cut)
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-an",  # No audio
            "-c:v", "copy",
            str(silent_path)
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            silent_videos.append(silent_path)
            logger.info(f"Stripped audio from scene {scene_idx} (Straight Cut)")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to strip audio from scene {scene_idx}: {e}")
            # Use original as fallback
            silent_videos.append(video_path)
    
    # Step 3: Generate TTS for each scene
    tts_dir = run_output_dir / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)
    
    tts_paths = []
    for i, (text, duration) in enumerate(zip(narration_texts, scene_durations)):
        scene_idx = i + 1
        tts_path = generate_scene_tts(
            narration_text=text,
            scene_idx=scene_idx,
            target_duration=duration,
            output_dir=tts_dir
        )
        if tts_path:
            tts_paths.append(Path(tts_path))
        else:
            logger.warning(f"TTS failed for scene {scene_idx}, using silence")
            # Create silent audio as fallback
            silence_path = tts_dir / f"{scene_idx:02d}_silence.wav"
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=24000:cl=mono",
                "-t", str(duration), str(silence_path)
            ], capture_output=True)
            tts_paths.append(silence_path)
    
    # Step 4: Concatenate silent videos
    concat_video_path = run_output_dir / "video_silent.mp4"
    concat_file = create_concat_file(silent_videos, run_output_dir / "concat_video.txt")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-c", "copy", str(concat_video_path)
    ], check=True)
    logger.info(f"Concatenated silent videos: {concat_video_path}")
    
    # Step 5: Concatenate TTS audio
    concat_audio_path = run_output_dir / "narration.wav"
    audio_concat_file = create_concat_file(tts_paths, run_output_dir / "concat_audio.txt")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(audio_concat_file),
        "-c:a", "pcm_s16le", str(concat_audio_path)
    ], check=True)
    logger.info(f"Concatenated TTS audio: {concat_audio_path}")
    
    # Step 6: Merge video + audio
    final_path = run_output_dir / "final_with_narration.mp4"
    
    # Get actual video duration
    video_duration = get_video_duration(concat_video_path)
    audio_duration = get_video_duration(concat_audio_path)
    
    logger.info(f"Video duration: {video_duration:.2f}s, Audio duration: {audio_duration:.2f}s")
    
    # Always trim/adjust audio to EXACTLY match video duration
    # This ensures narration never extends past video
    if audio_duration > video_duration:
        # Audio is longer - need to either speed up or trim
        if audio_duration - video_duration < 1.0:
            # Less than 1s difference - just trim
            logger.info(f"Trimming audio to match video ({audio_duration:.2f}s -> {video_duration:.2f}s)")
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(concat_video_path),
                "-i", str(concat_audio_path),
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v:0", "-map", "1:a:0",
                "-t", str(video_duration),  # Trim to exact video duration
                str(final_path)
            ], check=True)
        else:
            # More than 1s difference - speed up audio
            tempo = audio_duration / video_duration
            tempo = min(2.0, tempo)  # Max 2x speed
            logger.info(f"Speeding up audio: tempo={tempo:.2f}")
            subprocess.run([
                "ffmpeg", "-y", 
                "-i", str(concat_video_path), 
                "-i", str(concat_audio_path),
                "-filter:a", f"atempo={tempo}",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v:0", "-map", "1:a:0",
                "-t", str(video_duration),  # Safety trim
                str(final_path)
            ], check=True)
    elif video_duration > audio_duration + 0.5:
        # Video is longer - slow down audio or pad with silence
        tempo = audio_duration / video_duration
        tempo = max(0.5, tempo)  # Min 0.5x speed
        logger.info(f"Slowing down audio: tempo={tempo:.2f}")
        subprocess.run([
            "ffmpeg", "-y", 
            "-i", str(concat_video_path), 
            "-i", str(concat_audio_path),
            "-filter:a", f"atempo={tempo}",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest",
            str(final_path)
        ], check=True)
    else:
        # Durations are close enough - just merge
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(concat_video_path),
            "-i", str(concat_audio_path),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", "1:a:0",
            "-t", str(video_duration),  # Always trim to video duration
            str(final_path)
        ], check=True)
    
    logger.info(f"Final video with TTS: {final_path}")
    
    # Upload to GCS if requested
    if upload_to_gcs:
        gcs_path = f"mvp/{run_id}/final_with_narration.mp4"
        upload_file(final_path, gcs_path, bucket_name)
        logger.info(f"Uploaded to gs://{bucket_name}/{gcs_path}")
    
    return final_path


def assemble_with_single_tts(
    run_id: str,
    narration_texts: list[str],
    scene_durations: list[float],
    tts_emotions: list[str],
    tts_paces: list[str],
    tts_intensities: list[float],
    content_type: str = "general",            # ← Narrator config from LLM
    narrator_archetype: str = "",              # ← Narrator config from LLM
    narrator_voice: str = "",                  # ← Narrator config from LLM
    bucket_name: str = BUCKET_NAME,
    upload_to_gcs: bool = True,
    reuse_existing_audio: bool = False,
    pre_stitched_path: Optional[Path] = None, # ← Smart-stitched silent video (skips download+concat)
) -> Path:
    """
    Assemble final video with a SINGLE continuous TTS narration.
    
    This creates one audio file for the entire video, providing more natural
    flow between scenes instead of per-scene audio cuts.
    
    Args:
        run_id: The run identifier
        narration_texts: List of narration text for each scene
        scene_durations: List of scene durations in seconds
        tts_emotions: List of emotions for each scene
        tts_paces: List of paces for each scene
        tts_intensities: List of intensities for each scene
        content_type: Content category (horror, documentary, etc.) from LLM
        narrator_archetype: Narrator persona description from LLM
        narrator_voice: Voice tone (deep, warm, mysterious) from LLM
        bucket_name: GCS bucket name
        upload_to_gcs: Whether to upload final video to GCS
        reuse_existing_audio: If True, do not call TTS engine, just merge with existing narration.wav
        
    Returns:
        Path to the final video with TTS
    """
    
    logger.info(f"Starting AUDIO-FIRST TTS assembly for run {run_id}")
    if reuse_existing_audio:
        logger.info("AUDIO REUSE ENABLED: Skipping TTS generation.")
    if pre_stitched_path:
        logger.info(f"SMART STITCH: Using pre-stitched silent video: {pre_stitched_path}")

    logger.info(f"Content type: {content_type}, Voice: {narrator_voice}")

    # Calculate total duration
    total_duration = sum(scene_durations)
    logger.info(f"Total target video duration: {total_duration}s from {len(scene_durations)} scenes")

    # Setup directories
    run_output_dir = OUTPUT_DIR / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Step 4 (concat): If a pre-stitched video is available, use it directly
    # and skip downloading + per-scene processing entirely.
    concat_video_path = run_output_dir / "video_silent.mp4"

    if pre_stitched_path and pre_stitched_path.exists():
        import shutil as _shutil
        _shutil.copy2(pre_stitched_path, concat_video_path)
        logger.info(f"Copied smart-stitched video as video_silent.mp4 ({pre_stitched_path.stat().st_size // 1024}KB)")
    else:
        # Step 1: Download scene videos
        video_paths = download_all_scene_videos(run_id, bucket_name)
        if not video_paths:
            raise RuntimeError(f"No videos found for run {run_id}")

        # Step 2: Strip audio from scene videos
        silent_videos = []
        for scene_idx, video_path in enumerate(video_paths, start=1):
            silent_path = run_output_dir / f"scene_{scene_idx:02d}_silent.mp4"
            scene_dur = scene_durations[scene_idx - 1] if scene_idx <= len(scene_durations) else 8
            cmd = [
                "ffmpeg", "-y", "-i", str(video_path),
                "-an",
                "-c:v", "copy",
                str(silent_path)
            ]
            try:
                subprocess.run(cmd, capture_output=True, check=True)
                silent_videos.append(silent_path)
                logger.info(f"Processed scene {scene_idx} ({scene_dur}s - Straight Cut)")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to process scene {scene_idx}: {e}")
                silent_videos.append(video_path)

        # Step 3-4: Concatenate silent videos cleanly (Hard Cuts)
        if len(silent_videos) == 1:
            subprocess.run([
                "ffmpeg", "-y", "-i", str(silent_videos[0]),
                "-c:v", "copy", str(concat_video_path)
            ], check=True)
            logger.info(f"Only 1 video, copied to: {concat_video_path}")
        else:
            concat_file_path = run_output_dir / "concat_list.txt"
            with open(concat_file_path, "w") as f:
                for p in silent_videos:
                    f.write(f"file '{p.absolute()}'\n")
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_file_path),
                "-c", "copy",
                str(concat_video_path)
            ]
            subprocess.run(cmd, check=True)
            logger.info(f"Concatenated {len(silent_videos)} videos with clean hard cuts: {concat_video_path}")

    # Step 3: Verify CONTINUOUS TTS audio (already generated in Phase 1)
    audio_path = run_output_dir / "narration.wav"
    if not audio_path.exists() and len(narration_texts) > 0:
        logger.error(f"Missing audio file: {audio_path}. Audio generation in Phase 1B must have failed.")
        raise RuntimeError(f"Missing audio file: {audio_path}")
    if not reuse_existing_audio:
        logger.info(f"Generated single continuous narration: {audio_path}")

    # Step 4B: Freeze last frame if smart stitch shortened the video below audio length
    if pre_stitched_path and pre_stitched_path.exists() and audio_path.exists():
        _vid_dur = get_video_duration(concat_video_path)
        _aud_dur = get_video_duration(audio_path)
        if _aud_dur > _vid_dur + 0.1:
            _extra = _aud_dur - _vid_dur
            logger.info(
                f"[SmartStitch] Audio {_aud_dur:.2f}s > Video {_vid_dur:.2f}s — "
                f"freezing last frame for {_extra:.2f}s to fill the gap."
            )
            padded_path = run_output_dir / "video_silent_padded.mp4"
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(concat_video_path),
                "-vf", f"tpad=stop_mode=clone:stop_duration={_extra:.3f}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an",
                str(padded_path)
            ], check=True)
            concat_video_path = padded_path
            logger.info(f"[SmartStitch] Padded video: {padded_path}")

    # Step 5: Merge video + single continuous audio with automatic sync
    final_path = run_output_dir / "final_with_narration.mp4"
    
    video_duration = get_video_duration(concat_video_path)
    audio_duration = get_video_duration(audio_path)
    logger.info(f"Video: {video_duration:.2f}s, Audio: {audio_duration:.2f}s")
    
    duration_diff = audio_duration - video_duration
    
    if duration_diff > 0.5:
        # Audio is longer than video → speed up audio with atempo to fit exactly
        tempo = round(audio_duration / video_duration, 6)
        tempo = min(tempo, 1.15)  # Cap at 1.15x to keep it imperceptible
        logger.info(f"[AUDIO SYNC] Audio is {duration_diff:.2f}s longer than video. Applying atempo={tempo:.4f}x to sync.")
        if tempo > 1.10:
            logger.warning(f"⚠️ Audio speedup is {((tempo - 1) * 100):.1f}% — may be slightly noticeable.")
        
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(concat_video_path),
            "-i", str(audio_path),
            "-filter:a", f"atempo={tempo}",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", "1:a:0",
            "-t", str(video_duration),  # Trim to exact video duration
            str(final_path)
        ], check=True)
        logger.info(f"[AUDIO SYNC] Audio synced to video duration ({video_duration:.2f}s)")
    elif duration_diff < -0.5:
        # Video is longer than audio → pad audio with silence at the end
        logger.info(f"[AUDIO SYNC] Video is {abs(duration_diff):.2f}s longer than audio. Padding audio with silence.")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(concat_video_path),
            "-i", str(audio_path),
            "-filter:a", f"apad=pad_dur={abs(duration_diff):.3f}",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", "1:a:0",
            "-t", str(video_duration),
            str(final_path)
        ], check=True)
        logger.info(f"[AUDIO SYNC] Audio padded to match video duration ({video_duration:.2f}s)")
    else:
        # Durations are close enough — merge directly
        logger.info(f"[AUDIO SYNC] Durations match (diff={abs(duration_diff):.2f}s). Direct merge.")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(concat_video_path),
            "-i", str(audio_path),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", "1:a:0",
            "-t", str(video_duration),
            str(final_path)
        ], check=True)
    
    logger.info(f"Final video with synced TTS: {final_path}")
    
    # Upload to GCS if requested
    if upload_to_gcs:
        gcs_path = f"mvp/{run_id}/final_with_narration.mp4"
        upload_file(final_path, gcs_path, bucket_name)
        logger.info(f"Uploaded to gs://{bucket_name}/{gcs_path}")
    
    return final_path


if __name__ == "__main__":
    # Test assembly module
    logging.basicConfig(level=logging.INFO)
    
    print("Assembly module loaded successfully.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Scenes directory: {SCENES_DIR}")
    print(f"Final video path: {FINAL_VIDEO_PATH}")
    print("\nTo assemble, call: assemble_final_video(run_id)")
    print("To assemble with per-scene TTS: assemble_with_tts(run_id, narration_texts, scene_durations)")
    print("To assemble with single TTS: assemble_with_single_tts(run_id, narration_texts, scene_durations, tts_emotions, tts_paces, tts_intensities)")
