"""
Random Teleporting Watermark Module.

Overlays a semi-transparent watermark image that randomly teleports
to different positions at configurable intervals, making it resistant
to automated watermark removal tools.

Uses FFmpeg overlay with sendcmd to change position at intervals.
"""
import logging
import random
import subprocess
import json
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Watermark defaults
DEFAULT_OPACITY = 0.15          # 15% opacity
DEFAULT_TELEPORT_INTERVAL = 3   # seconds between position changes
DEFAULT_WATERMARK_SCALE = 0.08  # 8% of video width
MARGIN_PERCENT = 0.05           # 5% margin from edges


def _get_video_info(video_path: str) -> dict:
    """Get video width, height, and duration via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)

    video_stream = next(
        (s for s in info["streams"] if s["codec_type"] == "video"), None
    )
    if not video_stream:
        raise ValueError(f"No video stream found in {video_path}")

    width = int(video_stream["width"])
    height = int(video_stream["height"])
    duration = float(info["format"]["duration"])

    return {"width": width, "height": height, "duration": duration}


def _generate_random_positions(
    video_width: int,
    video_height: int,
    wm_width: int,
    wm_height: int,
    duration: float,
    interval: float,
    margin_pct: float = MARGIN_PERCENT,
) -> list[dict]:
    """Generate random (x, y) positions for each interval."""
    margin_x = int(video_width * margin_pct)
    margin_y = int(video_height * margin_pct)

    max_x = video_width - wm_width - margin_x
    max_y = video_height - wm_height - margin_y
    min_x = margin_x
    min_y = margin_y

    if max_x < min_x:
        max_x = min_x
    if max_y < min_y:
        max_y = min_y

    positions = []
    t = 0.0
    while t < duration:
        x = random.randint(min_x, max_x)
        y = random.randint(min_y, max_y)
        positions.append({"t": t, "x": x, "y": y})
        t += interval

    return positions


def add_watermark(
    video_path: str,
    watermark_path: str,
    output_path: Optional[str] = None,
    opacity: float = DEFAULT_OPACITY,
    teleport_interval: float = DEFAULT_TELEPORT_INTERVAL,
    scale_pct: float = DEFAULT_WATERMARK_SCALE,
) -> str:
    """
    Add a randomly teleporting watermark to a video.

    Args:
        video_path: Path to input video
        watermark_path: Path to watermark image (PNG with transparency recommended)
        output_path: Output path (defaults to video_watermarked.mp4)
        opacity: Watermark opacity (0.0 = invisible, 1.0 = fully opaque)
        teleport_interval: Seconds between random position changes
        scale_pct: Watermark size as fraction of video width

    Returns:
        Path to watermarked video
    """
    video_path = str(video_path)
    watermark_path = str(watermark_path)

    if not Path(watermark_path).exists():
        raise FileNotFoundError(f"Watermark image not found: {watermark_path}")
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_path is None:
        p = Path(video_path)
        output_path = str(p.parent / f"{p.stem}_watermarked{p.suffix}")

    logger.info(f"Adding watermark to {video_path}")
    logger.info(f"  Watermark: {watermark_path} | Opacity: {opacity} | Interval: {teleport_interval}s")

    # Get video dimensions
    info = _get_video_info(video_path)
    vw, vh, duration = info["width"], info["height"], info["duration"]

    # Calculate watermark size
    wm_width = int(vw * scale_pct)

    # Generate random positions
    positions = _generate_random_positions(
        video_width=vw,
        video_height=vh,
        wm_width=wm_width,
        wm_height=wm_width,  # approximate square for position calc
        duration=duration,
        interval=teleport_interval,
    )

    logger.info(f"  Video: {vw}x{vh}, {duration:.1f}s | Positions: {len(positions)}")

    # Build the overlay x/y expressions using FFmpeg's between() and timeline
    # Each position is active for [t, t+interval)
    # We use nested if() expressions: if(between(t,start,end), x_val, next_if(...))
    x_expr_parts = []
    y_expr_parts = []

    for i, pos in enumerate(positions):
        t_start = pos["t"]
        t_end = t_start + teleport_interval
        x_expr_parts.append(f"between(t\\,{t_start:.2f}\\,{t_end:.2f})*{pos['x']}")
        y_expr_parts.append(f"between(t\\,{t_start:.2f}\\,{t_end:.2f})*{pos['y']}")

    # Sum of between() expressions - only one will be 1 at any time
    x_expr = "+".join(x_expr_parts)
    y_expr = "+".join(y_expr_parts)

    # FFmpeg filter: scale watermark, set opacity, overlay with dynamic position
    filter_complex = (
        f"[1:v]scale={wm_width}:-1,format=rgba,"
        f"colorchannelmixer=aa={opacity}[wm];"
        f"[0:v][wm]overlay=x='{x_expr}':y='{y_expr}':shortest=1"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", watermark_path,
        "-filter_complex", filter_complex,
        "-c:a", "copy",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-movflags", "+faststart",
        output_path
    ]

    logger.info(f"  Running FFmpeg watermark...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        logger.error(f"FFmpeg watermark failed:\n{result.stderr[-1000:]}")
        raise RuntimeError(f"Watermark failed: {result.stderr[-500:]}")

    logger.info(f"  Watermarked video saved: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

    parser = argparse.ArgumentParser(description="Add teleporting watermark to video")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("watermark", help="Watermark image path (PNG)")
    parser.add_argument("-o", "--output", default=None, help="Output path")
    parser.add_argument("--opacity", type=float, default=DEFAULT_OPACITY, help=f"Opacity (default: {DEFAULT_OPACITY})")
    parser.add_argument("--interval", type=float, default=DEFAULT_TELEPORT_INTERVAL, help=f"Teleport interval in seconds (default: {DEFAULT_TELEPORT_INTERVAL})")
    parser.add_argument("--scale", type=float, default=DEFAULT_WATERMARK_SCALE, help=f"Watermark size as %% of video width (default: {DEFAULT_WATERMARK_SCALE})")

    args = parser.parse_args()
    result = add_watermark(
        video_path=args.video,
        watermark_path=args.watermark,
        output_path=args.output,
        opacity=args.opacity,
        teleport_interval=args.interval,
        scale_pct=args.scale,
    )
    print(f"Done: {result}")
