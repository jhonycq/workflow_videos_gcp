#!/usr/bin/env python3
"""
smart_stitch.py — Smart video clip stitcher with automatic seam detection.

Finds the best visual splice point between clip A (end) and clip B (start)
using frame-by-frame SSIM comparison, then joins them with:
  - hard cut   (SSIM very high — frames nearly identical)
  - xfade      (SSIM moderate — short crossfade hides the seam)
  - RIFE       (SSIM low but motion coherent — interpolated bridge frames)

Supports single-pair and batch (folder of ordered clips) modes.

Usage examples:
  Single pair:
    python smart_stitch.py --clip-a scene01.mp4 --clip-b scene02.mp4 --output-dir out/

  Batch (alphabetical order):
    python smart_stitch.py --batch scenes/ --output-dir out/ --pattern "*_video.mp4"

  With RIFE fallback:
    python smart_stitch.py --clip-a a.mp4 --clip-b b.mp4 --output-dir out/ \\
        --rife --ssim-hard 0.90 --ssim-xfade 0.60 --window 24

Dependencies:
  pip install av numpy scikit-image pillow
  ffmpeg must be in PATH
  (optional) rife-ncnn-vulkan in PATH for --rife mode
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import av
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn

logger = logging.getLogger(__name__)


# ─── Data structures ────────────────────────────────────────────────────────

@dataclass
class ClipInfo:
    """Video stream metadata extracted via PyAV (no full decode required)."""
    path: str
    width: int
    height: int
    fps: float
    duration: float   # seconds
    nb_frames: int
    pix_fmt: str
    has_audio: bool


@dataclass
class StitchResult:
    """Outcome of a single stitch operation — saved as JSON alongside the output."""
    clip_a: str
    clip_b: str
    output: str
    best_frame_a: int     # absolute frame index in clip A where trim ends (inclusive)
    best_frame_b: int     # absolute frame index in clip B where trim starts (inclusive)
    ssim_score: float
    transition_type: str  # "hard_cut" | "xfade" | "rife"
    xfade_duration: float
    success: bool
    error: Optional[str] = None


@dataclass
class StitchConfig:
    """All tunable parameters for the stitch process."""
    window_frames: int = 30              # N frames compared at each boundary
    ssim_threshold_hard: float = 0.85   # SSIM >= this → hard cut
    ssim_threshold_xfade: float = 0.50  # SSIM >= this → xfade; below → rife / xfade
    xfade_duration: float = 0.5         # seconds for xfade overlap
    xfade_transition: str = "fade"      # FFmpeg xfade transition name
    use_rife: bool = False              # enable RIFE interpolation
    rife_executable: str = "rife-ncnn-vulkan"
    rife_intermediate_frames: int = 8   # frames RIFE generates between seam frames
    target_fps: Optional[float] = None  # None = use clip A fps
    target_width: Optional[int] = None  # None = use clip A width
    target_height: Optional[int] = None # None = use clip A height
    target_pix_fmt: str = "yuv420p"
    keep_temp: bool = False
    grayscale_ssim: bool = False        # faster; slightly less accurate


# ─── Clip metadata ──────────────────────────────────────────────────────────

def get_clip_info(path: str) -> ClipInfo:
    """Read video metadata with PyAV (header only, no full decode)."""
    container = av.open(path)
    try:
        vstream = next(s for s in container.streams if s.type == "video")
        has_audio = any(s.type == "audio" for s in container.streams)

        fps = float(vstream.average_rate or vstream.base_rate or 24)
        duration = (
            float(container.duration / av.time_base)
            if container.duration else 0.0
        )
        nb_frames = int(vstream.frames) or int(duration * fps)

        return ClipInfo(
            path=path,
            width=vstream.width,
            height=vstream.height,
            fps=fps,
            duration=duration,
            nb_frames=nb_frames,
            pix_fmt=vstream.pix_fmt or "yuv420p",
            has_audio=has_audio,
        )
    finally:
        container.close()


# ─── Frame extraction ────────────────────────────────────────────────────────

def extract_frames(
    path: str,
    info: ClipInfo,
    start_frame: int,
    count: int,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Decode up to `count` frames starting at `start_frame` using PyAV.

    Seeks backward to the nearest keyframe, then skips frames whose
    timestamps fall before the target start time.

    Returns:
        frames  — list of RGB numpy arrays (H, W, 3) uint8
        indices — corresponding absolute frame indices in the source clip
    """
    frames: List[np.ndarray] = []
    indices: List[int] = []
    target_time = start_frame / max(info.fps, 1)  # seconds

    container = av.open(path)
    vstream = container.streams.video[0]
    vstream.thread_type = "AUTO"

    try:
        if start_frame > 0:
            seek_ts = int(target_time / float(vstream.time_base))
            container.seek(seek_ts, stream=vstream, backward=True)

        for frame in container.decode(vstream):
            if frame.pts is None:
                continue

            frame_time = float(frame.pts) * float(vstream.time_base)

            # Skip pre-seek frames (keyframe may be before target)
            if frame_time < target_time - 1.0 / max(info.fps, 1):
                continue

            img = frame.to_ndarray(format="rgb24")
            frame_idx = int(round(frame_time * info.fps))
            frames.append(img)
            indices.append(frame_idx)

            if len(frames) >= count:
                break
    finally:
        container.close()

    return frames, indices


def extract_tail(path: str, n: int, info: ClipInfo) -> Tuple[List[np.ndarray], List[int]]:
    """Last N frames of a clip."""
    start = max(0, info.nb_frames - n)
    return extract_frames(path, info, start_frame=start, count=n)


def extract_head(path: str, n: int, info: ClipInfo) -> Tuple[List[np.ndarray], List[int]]:
    """First N frames of a clip."""
    return extract_frames(path, info, start_frame=0, count=n)


# ─── SSIM comparison ─────────────────────────────────────────────────────────

def _resize(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize frame using PIL (LANCZOS) when available, else numpy sampling."""
    if frame.shape[1] == w and frame.shape[0] == h:
        return frame
    try:
        from PIL import Image
        return np.array(Image.fromarray(frame).resize((w, h), Image.LANCZOS))
    except ImportError:
        ys = np.linspace(0, frame.shape[0] - 1, h, dtype=int)
        xs = np.linspace(0, frame.shape[1] - 1, w, dtype=int)
        return frame[np.ix_(ys, xs)]


def compute_ssim_matrix(
    frames_a: List[np.ndarray],
    frames_b: List[np.ndarray],
    grayscale: bool = False,
    comparison_size: Tuple[int, int] = (320, 180),
) -> np.ndarray:
    """
    Compute SSIM for every (i, j) pair across the two frame lists.

    Frames are downscaled to `comparison_size` (w, h) before comparison
    to keep computation fast regardless of source resolution.

    Returns shape (len(frames_a), len(frames_b)) float32 matrix.
    """
    cw, ch = comparison_size

    def prep(f: np.ndarray) -> np.ndarray:
        f = _resize(f, cw, ch)
        if grayscale:
            return np.dot(f[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        return f

    prepped_a = [prep(f) for f in frames_a]
    prepped_b = [prep(f) for f in frames_b]

    na, nb = len(prepped_a), len(prepped_b)
    matrix = np.zeros((na, nb), dtype=np.float32)

    for i, fa in enumerate(prepped_a):
        for j, fb in enumerate(prepped_b):
            if grayscale:
                score = ssim_fn(fa, fb, data_range=255)
            else:
                score = ssim_fn(fa, fb, data_range=255, channel_axis=-1)
            matrix[i, j] = score

    return matrix


def find_best_match(matrix: np.ndarray) -> Tuple[int, int, float]:
    """Return (win_i, win_j, ssim_score) of the highest-SSIM cell."""
    idx = int(np.argmax(matrix))
    i, j = np.unravel_index(idx, matrix.shape)
    return int(i), int(j), float(matrix[i, j])


# ─── FFmpeg helpers ──────────────────────────────────────────────────────────

def _ffmpeg(args: List[str], label: str = "") -> None:
    """Run FFmpeg with -y (overwrite), raising RuntimeError on failure."""
    cmd = ["ffmpeg", "-y", "-loglevel", "warning"] + args
    logger.debug("FFmpeg [%s]: %s", label, " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed [{label}]\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDERR: {r.stderr[-3000:]}"
        )


def normalize_clip(
    src: str, dst: str,
    width: int, height: int, fps: float, pix_fmt: str,
    has_audio: bool,
) -> None:
    """
    Re-encode to a standard format so both inputs to xfade are compatible.
    FFmpeg xfade requires identical resolution, FPS, timebase, and pixel format.
    """
    vf = f"scale={width}:{height}:force_original_aspect_ratio=disable,fps={fps}"
    args = [
        "-i", src,
        "-vf", vf,
        "-pix_fmt", pix_fmt,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    ]
    if has_audio:
        args += ["-c:a", "aac", "-ar", "44100", "-ac", "2"]
    else:
        args += ["-an"]
    args.append(dst)
    _ffmpeg(args, f"normalize {Path(src).name}")


def trim_clip(
    src: str, dst: str, fps: float, has_audio: bool,
    end_frame: Optional[int] = None,
    start_frame: Optional[int] = None,
) -> None:
    """
    Trim `src` to [start_frame, end_frame] (both inclusive) and re-encode.

    end_frame   — used for clip A (keep up to and including this frame)
    start_frame — used for clip B (keep from this frame onward)
    """
    args = ["-i", src]
    t_start = (start_frame / fps) if start_frame else 0.0

    if start_frame:
        args += ["-ss", f"{t_start:.6f}"]

    if end_frame is not None:
        t_end = (end_frame + 1) / fps  # +1 to include the match frame
        duration = t_end - t_start
        args += ["-t", f"{duration:.6f}"]

    args += ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]
    if has_audio:
        args += ["-c:a", "aac"]
    else:
        args += ["-an"]
    args.append(dst)
    _ffmpeg(args, f"trim {Path(src).name}")


def concat_clips(clip_a: str, clip_b: str, output: str, has_audio: bool) -> None:
    """
    Hard-cut concatenation via FFmpeg filter_complex concat.
    Both clips must share the same resolution / FPS / pix_fmt.
    """
    if has_audio:
        fc = "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[vout][aout]"
        maps = ["-map", "[vout]", "-map", "[aout]"]
    else:
        fc = "[0:v][1:v]concat=n=2:v=1[vout]"
        maps = ["-map", "[vout]"]

    args = (
        ["-i", clip_a, "-i", clip_b, "-filter_complex", fc]
        + maps
        + ["-c:v", "libx264", "-preset", "fast", "-crf", "18", output]
    )
    _ffmpeg(args, "concat hard_cut")


def xfade_clips(
    clip_a: str, clip_b: str, output: str,
    duration_a: float, xfade_dur: float,
    transition: str, has_audio: bool,
) -> None:
    """
    Apply FFmpeg xfade transition between two clips.
    offset = duration_a - xfade_dur positions the fade at the tail of clip A.
    """
    offset = max(0.0, duration_a - xfade_dur)
    fv = (
        f"[0:v][1:v]xfade=transition={transition}:"
        f"duration={xfade_dur:.4f}:offset={offset:.4f}[vout]"
    )

    if has_audio:
        fc = f"{fv};[0:a][1:a]acrossfade=d={xfade_dur:.4f}[aout]"
        maps = ["-map", "[vout]", "-map", "[aout]"]
    else:
        fc = fv
        maps = ["-map", "[vout]"]

    args = (
        ["-i", clip_a, "-i", clip_b, "-filter_complex", fc]
        + maps
        + ["-c:v", "libx264", "-preset", "fast", "-crf", "18", output]
    )
    _ffmpeg(args, f"xfade {transition} {xfade_dur}s")


# ─── RIFE interpolation (optional) ──────────────────────────────────────────

def rife_stitch(
    clip_a: str, clip_b: str, output: str,
    duration_a: float, xfade_dur: float,
    rife_exe: str, n_frames: int,
    has_audio: bool, tmp_dir: str,
) -> None:
    """
    Generate interpolated bridge frames between the seam using rife-ncnn-vulkan,
    then concat: trimmed_A + bridge + trimmed_B.

    Falls back to xfade automatically if RIFE fails or is not installed.
    """
    logger.info("Attempting RIFE interpolation...")

    frame_a = os.path.join(tmp_dir, "rife_last_a.png")
    frame_b = os.path.join(tmp_dir, "rife_first_b.png")
    rife_dir = os.path.join(tmp_dir, "rife_frames")
    os.makedirs(rife_dir, exist_ok=True)

    # Extract last frame of A and first frame of B
    try:
        _ffmpeg(["-sseof", "-0.1", "-i", clip_a, "-vframes", "1", frame_a], "rife: last frame A")
        _ffmpeg(["-i", clip_b, "-vframes", "1", frame_b], "rife: first frame B")
    except RuntimeError as e:
        logger.warning("RIFE setup failed: %s — falling back to xfade.", e)
        xfade_clips(clip_a, clip_b, output, duration_a, xfade_dur, "fade", has_audio)
        return

    rife_cmd = [rife_exe, "-0", frame_a, "-1", frame_b, "-o", rife_dir, "-n", str(n_frames)]
    logger.info("Running RIFE: %s", " ".join(rife_cmd))
    r = subprocess.run(rife_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.warning("RIFE failed (rc=%d): %s — falling back to xfade.", r.returncode, r.stderr[:400])
        xfade_clips(clip_a, clip_b, output, duration_a, xfade_dur, "fade", has_audio)
        return

    png_files = sorted(Path(rife_dir).glob("*.png"))
    if not png_files:
        logger.warning("RIFE produced no PNG files — falling back to xfade.")
        xfade_clips(clip_a, clip_b, output, duration_a, xfade_dur, "fade", has_audio)
        return

    # Assemble RIFE frames → bridge video
    bridge = os.path.join(tmp_dir, "rife_bridge.mp4")
    frame_dur = xfade_dur / max(len(png_files), 1)
    png_list = os.path.join(tmp_dir, "rife_frames.txt")
    with open(png_list, "w") as f:
        for p in png_files:
            f.write(f"file '{p}'\n")
            f.write(f"duration {frame_dur:.6f}\n")

    _ffmpeg(
        ["-f", "concat", "-safe", "0", "-i", png_list,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", bridge],
        "rife: assemble bridge",
    )

    # Concat: A + bridge + B
    concat_list = os.path.join(tmp_dir, "rife_concat.txt")
    with open(concat_list, "w") as f:
        f.write(f"file '{clip_a}'\n")
        f.write(f"file '{bridge}'\n")
        f.write(f"file '{clip_b}'\n")

    args = ["-f", "concat", "-safe", "0", "-i", concat_list,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18"]
    if has_audio:
        args += ["-c:a", "aac"]
    else:
        args += ["-an"]
    args.append(output)
    _ffmpeg(args, "rife: final concat")


# ─── Core stitch logic ───────────────────────────────────────────────────────

def stitch_pair(
    clip_a_path: str,
    clip_b_path: str,
    output_dir: str,
    config: StitchConfig,
    output_name: Optional[str] = None,
) -> StitchResult:
    """
    Stitch two clips with automatic seam detection.

    Pipeline:
      1. Read metadata (PyAV, header only)
      2. Extract window frames from tail of A and head of B
      3. Compute SSIM matrix  →  find best (i, j) match
      4. Decide transition type from SSIM score thresholds
      5. Trim A to frame i, trim B from frame j
      6. Normalize to common format if needed (required for xfade)
      7. Apply transition (hard_cut / xfade / rife)
      8. Write output video + JSON result
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem_a = Path(clip_a_path).stem
    stem_b = Path(clip_b_path).stem
    name = output_name or f"{stem_a}__{stem_b}_stitched"
    output_path = str(out_dir / f"{name}.mp4")
    json_path = str(out_dir / f"{name}.json")

    logger.info("=" * 60)
    logger.info("Stitch: %s  +  %s", Path(clip_a_path).name, Path(clip_b_path).name)
    logger.info("=" * 60)

    # ── 1. Metadata ──────────────────────────────────────────────────────────
    info_a = get_clip_info(clip_a_path)
    info_b = get_clip_info(clip_b_path)
    logger.info(
        "  Clip A: %dx%d @ %.2ffps  frames=%d  dur=%.2fs  audio=%s",
        info_a.width, info_a.height, info_a.fps, info_a.nb_frames, info_a.duration, info_a.has_audio,
    )
    logger.info(
        "  Clip B: %dx%d @ %.2ffps  frames=%d  dur=%.2fs  audio=%s",
        info_b.width, info_b.height, info_b.fps, info_b.nb_frames, info_b.duration, info_b.has_audio,
    )

    # ── 2. Target format (clip A drives defaults) ─────────────────────────────
    tw = config.target_width or info_a.width
    th = config.target_height or info_a.height
    tfps = config.target_fps or info_a.fps
    tpix = config.target_pix_fmt
    has_audio = info_a.has_audio or info_b.has_audio

    needs_norm = (
        info_a.width != tw or info_a.height != th or abs(info_a.fps - tfps) > 0.01
        or info_b.width != tw or info_b.height != th or abs(info_b.fps - tfps) > 0.01
    )
    if needs_norm:
        logger.info("  Normalization needed → %dx%d @ %.2ffps %s", tw, th, tfps, tpix)

    # ── 3. Extract window frames ──────────────────────────────────────────────
    n = config.window_frames
    logger.info("Extracting %d window frames from each boundary...", n)
    frames_a, indices_a = extract_tail(clip_a_path, n, info_a)
    frames_b, indices_b = extract_head(clip_b_path, n, info_b)
    logger.info("  A tail: %d frames  |  B head: %d frames", len(frames_a), len(frames_b))

    if not frames_a or not frames_b:
        raise ValueError(
            f"Could not extract frames. A tail={len(frames_a)} B head={len(frames_b)}"
        )

    # ── 4. SSIM matrix ────────────────────────────────────────────────────────
    cmp_w = min(tw, 320)
    cmp_h = min(th, 180)
    logger.info(
        "Computing %dx%d SSIM matrix (comparison at %dx%d)...",
        len(frames_a), len(frames_b), cmp_w, cmp_h,
    )
    matrix = compute_ssim_matrix(
        frames_a, frames_b,
        grayscale=config.grayscale_ssim,
        comparison_size=(cmp_w, cmp_h),
    )

    win_i, win_j, score = find_best_match(matrix)
    abs_frame_a = indices_a[win_i]
    abs_frame_b = indices_b[win_j]
    logger.info(
        "  Best match: A[%d] ↔ B[%d]  SSIM=%.4f  (window position A[%d] B[%d])",
        abs_frame_a, abs_frame_b, score, win_i, win_j,
    )

    # ── 5. Decide transition ──────────────────────────────────────────────────
    if score >= config.ssim_threshold_hard:
        transition = "hard_cut"
    elif score >= config.ssim_threshold_xfade:
        transition = "xfade"
    else:
        transition = "rife" if config.use_rife else "xfade"

    logger.info(
        "  Transition → %s  (SSIM=%.4f, thresholds: hard≥%.2f, xfade≥%.2f)",
        transition, score, config.ssim_threshold_hard, config.ssim_threshold_xfade,
    )

    # ── 6. Temp workspace ────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory(dir=str(out_dir), prefix="stitch_tmp_") as tmp:

        a_trimmed = os.path.join(tmp, "a_trimmed.mp4")
        b_trimmed = os.path.join(tmp, "b_trimmed.mp4")
        a_norm = os.path.join(tmp, "a_norm.mp4")
        b_norm = os.path.join(tmp, "b_norm.mp4")

        # Trim A: keep frames 0 … abs_frame_a (inclusive)
        logger.info("Trimming A → frame %d of %d...", abs_frame_a, info_a.nb_frames - 1)
        trim_clip(clip_a_path, a_trimmed, info_a.fps, info_a.has_audio, end_frame=abs_frame_a)

        # Trim B: keep frames abs_frame_b … end
        logger.info("Trimming B from frame %d of %d...", abs_frame_b, info_b.nb_frames - 1)
        trim_clip(clip_b_path, b_trimmed, info_b.fps, info_b.has_audio, start_frame=abs_frame_b)

        # Normalize if formats differ (always required before xfade)
        if needs_norm or transition in ("xfade", "rife"):
            logger.info("Normalizing to %dx%d @ %.2ffps %s...", tw, th, tfps, tpix)
            normalize_clip(a_trimmed, a_norm, tw, th, tfps, tpix, info_a.has_audio)
            normalize_clip(b_trimmed, b_norm, tw, th, tfps, tpix, info_b.has_audio)
            src_a, src_b = a_norm, b_norm
        else:
            src_a, src_b = a_trimmed, b_trimmed

        info_a_trim = get_clip_info(src_a)
        logger.info("  Trimmed A duration: %.3fs", info_a_trim.duration)

        # ── 7. Apply transition ───────────────────────────────────────────────
        if transition == "hard_cut":
            logger.info("Applying hard cut concat...")
            concat_clips(src_a, src_b, output_path, has_audio)

        elif transition == "xfade":
            logger.info("Applying xfade '%s' %.2fs...", config.xfade_transition, config.xfade_duration)
            xfade_clips(
                src_a, src_b, output_path,
                info_a_trim.duration, config.xfade_duration,
                config.xfade_transition, has_audio,
            )

        elif transition == "rife":
            rife_stitch(
                src_a, src_b, output_path,
                info_a_trim.duration, config.xfade_duration,
                config.rife_executable, config.rife_intermediate_frames,
                has_audio, tmp,
            )

        # Preserve temp files for debugging if requested
        if config.keep_temp:
            keep = out_dir / f"{name}_tmp"
            shutil.copytree(tmp, str(keep), dirs_exist_ok=True)
            logger.info("Temp files kept at: %s", keep)

    # ── 8. Write result JSON ──────────────────────────────────────────────────
    result = StitchResult(
        clip_a=clip_a_path,
        clip_b=clip_b_path,
        output=output_path,
        best_frame_a=abs_frame_a,
        best_frame_b=abs_frame_b,
        ssim_score=score,
        transition_type=transition,
        xfade_duration=config.xfade_duration if transition != "hard_cut" else 0.0,
        success=True,
    )
    with open(json_path, "w") as f:
        json.dump(asdict(result), f, indent=2)

    logger.info("Result JSON : %s", json_path)
    logger.info("Output video: %s", output_path)
    return result


# ─── Batch mode ──────────────────────────────────────────────────────────────

def batch_stitch(
    input_folder: str,
    output_dir: str,
    config: StitchConfig,
    pattern: str = "*.mp4",
) -> List[StitchResult]:
    """
    Stitch all clips in `input_folder` (sorted alphabetically) into one video.

    Strategy: chain pairs sequentially — stitched output of pair N becomes
    clip A of pair N+1.  Per-pair results go into output_dir/pairs/.
    Final video is copied to output_dir/final_stitched.mp4.
    A batch_results.json summary is written to output_dir/.
    """
    folder = Path(input_folder)
    clips = sorted(folder.glob(pattern))

    if len(clips) < 2:
        raise ValueError(
            f"Need ≥ 2 clips matching '{pattern}' in {input_folder}, found {len(clips)}"
        )

    logger.info("Batch mode: %d clips in %s", len(clips), input_folder)
    for c in clips:
        logger.info("  %s", c.name)

    out_dir = Path(output_dir)
    pairs_dir = out_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    results: List[StitchResult] = []
    current = str(clips[0])

    for idx in range(1, len(clips)):
        nxt = str(clips[idx])
        pair_name = f"pair_{idx:03d}"
        logger.info(
            "\n── Pair %d/%d: %s  →  %s",
            idx, len(clips) - 1, Path(current).name, Path(nxt).name,
        )
        try:
            r = stitch_pair(current, nxt, str(pairs_dir), config, output_name=pair_name)
            results.append(r)
            current = r.output  # chain: stitched result becomes next A

        except Exception as exc:
            logger.error("Pair %d failed: %s", idx, exc, exc_info=True)
            results.append(StitchResult(
                clip_a=current, clip_b=nxt, output="",
                best_frame_a=-1, best_frame_b=-1, ssim_score=0.0,
                transition_type="failed", xfade_duration=0.0,
                success=False, error=str(exc),
            ))
            current = nxt  # skip failed pair, continue from next clip

    # Copy final stitched video for easy access
    if results and results[-1].success:
        final_dst = str(out_dir / "final_stitched.mp4")
        shutil.copy2(results[-1].output, final_dst)
        logger.info("\nBatch complete. Final video: %s", final_dst)

    # Summary JSON
    batch_json = str(out_dir / "batch_results.json")
    with open(batch_json, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    ok = sum(1 for r in results if r.success)
    logger.info("Pairs: %d/%d succeeded, %d failed.", ok, len(results), len(results) - ok)
    return results


# ─── Pipeline integration ────────────────────────────────────────────────────

def analyze_seams(
    clips: List[Path],
    window: int = 15,
    comparison_size: Tuple[int, int] = (160, 90),
    grayscale: bool = True,
) -> List[float]:
    """
    Fast seam scan: compute max SSIM at the boundary of each consecutive clip pair.

    Uses a smaller window, smaller comparison resolution, and grayscale to run
    quickly before committing to a full stitch.  Returns one score per pair.

    A score >= 0.50 means consecutive clips share visually similar frames at the
    seam and will benefit from smart stitching.  A score < 0.50 means the clips
    are visually independent and regular hard concat is fine.
    """
    scores: List[float] = []
    for i in range(len(clips) - 1):
        try:
            info_a = get_clip_info(str(clips[i]))
            info_b = get_clip_info(str(clips[i + 1]))
            fa, _ = extract_tail(str(clips[i]), window, info_a)
            fb, _ = extract_head(str(clips[i + 1]), window, info_b)
            if not fa or not fb:
                scores.append(0.0)
                continue
            matrix = compute_ssim_matrix(fa, fb, grayscale=grayscale, comparison_size=comparison_size)
            _, _, best = find_best_match(matrix)
            scores.append(best)
        except Exception as exc:
            logger.warning("Seam scan failed for pair %d/%d: %s", i + 1, len(clips) - 1, exc)
            scores.append(0.0)
    return scores


def smart_stitch_scenes(
    run_id: str,
    scenes_dir: Path,
    output_dir: Path,
    config: Optional[StitchConfig] = None,
    pattern: str = "*_video.mp4",
    min_ssim_to_apply: float = 0.50,
) -> Optional[Path]:
    """
    Pipeline integration point called after all scene videos are generated.

    1. Discovers scene video files in `scenes_dir` sorted alphabetically.
    2. Runs a fast SSIM seam scan on consecutive pairs.
    3. If ANY pair's max SSIM >= `min_ssim_to_apply` → full smart stitch is
       applied to the entire sequence and the stitched silent video path is returned.
    4. If NO pair benefits (all SSIM < threshold) → returns None so the caller
       falls back to the regular concat pipeline.

    The stitched output is placed at:
        output_dir / run_id / stitched / final_stitched.mp4

    Args:
        run_id:             Pipeline run identifier (used for logging).
        scenes_dir:         Folder containing `*_video.mp4` scene files.
        output_dir:         Root output directory (e.g. `out/`).
        config:             StitchConfig; defaults to conservative pipeline settings.
        pattern:            Glob to find scene files inside `scenes_dir`.
        min_ssim_to_apply:  Minimum SSIM at any seam to trigger stitching.

    Returns:
        Path to the stitched silent video, or None to skip.
    """
    if config is None:
        config = StitchConfig(
            window_frames=24,
            ssim_threshold_hard=0.85,
            ssim_threshold_xfade=0.50,
            xfade_duration=0.5,
            xfade_transition="fade",
            grayscale_ssim=False,
        )

    clips = sorted(scenes_dir.glob(pattern))
    if len(clips) < 2:
        logger.info("[SmartStitch] Only %d clip(s) found — skipping stitch.", len(clips))
        return None

    logger.info(
        "[SmartStitch] Scanning %d scene pairs for overlapping frames (run=%s)...",
        len(clips) - 1, run_id,
    )

    # Fast pre-scan: tiny resolution, grayscale, small window
    seam_scores = analyze_seams(clips, window=15, comparison_size=(160, 90), grayscale=True)
    max_score = max(seam_scores) if seam_scores else 0.0

    pair_summary = "  |  ".join(
        f"Pair {i+1}: {s:.3f}" for i, s in enumerate(seam_scores)
    )
    logger.info("[SmartStitch] Seam SSIM scores — %s", pair_summary)
    logger.info("[SmartStitch] Max SSIM=%.4f  threshold=%.2f", max_score, min_ssim_to_apply)

    if max_score < min_ssim_to_apply:
        logger.info(
            "[SmartStitch] No overlapping frames detected (max SSIM %.3f < %.2f). "
            "Skipping — regular pipeline will be used.",
            max_score, min_ssim_to_apply,
        )
        return None

    # Full stitch
    stitch_out = output_dir / run_id / "stitched"
    logger.info("[SmartStitch] Overlapping frames detected. Applying full stitch → %s", stitch_out)

    try:
        results = batch_stitch(
            input_folder=str(scenes_dir),
            output_dir=str(stitch_out),
            config=config,
            pattern=pattern,
        )
    except Exception as exc:
        logger.error("[SmartStitch] Batch stitch failed: %s — falling back to regular pipeline.", exc)
        return None

    # Check all pairs succeeded
    ok = sum(1 for r in results if r.success)
    if ok == 0:
        logger.error("[SmartStitch] All pairs failed — falling back to regular pipeline.")
        return None

    if ok < len(results):
        logger.warning(
            "[SmartStitch] %d/%d pairs failed. Stitched video covers partial sequence.",
            len(results) - ok, len(results),
        )

    final_path = stitch_out / "final_stitched.mp4"
    if not final_path.exists():
        logger.error("[SmartStitch] Expected output not found: %s", final_path)
        return None

    logger.info("[SmartStitch] ✅ Stitched silent video: %s", final_path)
    return final_path


# ─── CLI ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Smart video stitcher — SSIM seam detection + hard cut / xfade / RIFE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--clip-a", metavar="PATH", help="First clip (pair mode)")
    mode.add_argument("--batch", metavar="FOLDER", help="Folder of clips to stitch in order")

    p.add_argument("--clip-b", metavar="PATH", help="Second clip (required with --clip-a)")
    p.add_argument("--output-dir", required=True, metavar="PATH", help="Output directory")
    p.add_argument("--output-name", metavar="NAME", help="Output filename stem (pair mode)")
    p.add_argument("--pattern", default="*.mp4", metavar="GLOB",
                   help="Glob pattern for clip discovery in batch mode")

    g = p.add_argument_group("Seam detection")
    g.add_argument("--window", type=int, default=30, metavar="N",
                   help="Number of frames to compare at each boundary")
    g.add_argument("--ssim-hard", type=float, default=0.85, metavar="T",
                   help="SSIM ≥ T → hard cut")
    g.add_argument("--ssim-xfade", type=float, default=0.50, metavar="T",
                   help="SSIM ≥ T → xfade; below → rife or xfade fallback")
    g.add_argument("--grayscale-ssim", action="store_true",
                   help="Compare in grayscale (faster, slightly less accurate)")

    g2 = p.add_argument_group("Transition")
    g2.add_argument("--xfade-duration", type=float, default=0.5, metavar="SECS",
                    help="xfade overlap duration in seconds")
    g2.add_argument("--xfade-transition", default="fade", metavar="NAME",
                    help="FFmpeg xfade transition type (fade, dissolve, wipeleft, …)")

    g3 = p.add_argument_group("RIFE (optional)")
    g3.add_argument("--rife", action="store_true",
                    help="Enable RIFE for low-SSIM seams (requires rife-ncnn-vulkan)")
    g3.add_argument("--rife-exe", default="rife-ncnn-vulkan", metavar="PATH",
                    help="Path or name of RIFE binary")
    g3.add_argument("--rife-frames", type=int, default=8, metavar="N",
                    help="Intermediate frames RIFE generates per seam")

    g4 = p.add_argument_group("Output format override")
    g4.add_argument("--width", type=int, metavar="PX", help="Force output width")
    g4.add_argument("--height", type=int, metavar="PX", help="Force output height")
    g4.add_argument("--fps", type=float, metavar="N", help="Force output FPS")
    g4.add_argument("--pix-fmt", default="yuv420p", help="Output pixel format")

    p.add_argument("--keep-temp", action="store_true", help="Keep intermediate temp files")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    config = StitchConfig(
        window_frames=args.window,
        ssim_threshold_hard=args.ssim_hard,
        ssim_threshold_xfade=args.ssim_xfade,
        xfade_duration=args.xfade_duration,
        xfade_transition=args.xfade_transition,
        use_rife=args.rife,
        rife_executable=args.rife_exe,
        rife_intermediate_frames=args.rife_frames,
        target_width=args.width,
        target_height=args.height,
        target_fps=args.fps,
        target_pix_fmt=args.pix_fmt,
        keep_temp=args.keep_temp,
        grayscale_ssim=args.grayscale_ssim,
    )

    # ── Batch ────────────────────────────────────────────────────────────────
    if args.batch:
        results = batch_stitch(
            input_folder=args.batch,
            output_dir=args.output_dir,
            config=config,
            pattern=args.pattern,
        )
        ok = sum(1 for r in results if r.success)
        print(f"\nBatch done: {ok}/{len(results)} pairs succeeded.")
        sys.exit(0 if ok == len(results) else 1)

    # ── Single pair ───────────────────────────────────────────────────────────
    if not args.clip_b:
        parser.error("--clip-b is required when using --clip-a")

    try:
        result = stitch_pair(
            clip_a_path=args.clip_a,
            clip_b_path=args.clip_b,
            output_dir=args.output_dir,
            config=config,
            output_name=args.output_name,
        )
    except Exception as exc:
        logger.error("Stitch failed: %s", exc, exc_info=True)
        sys.exit(1)

    print("\n── Result ──────────────────────────────────────")
    print(f"  Output video : {result.output}")
    print(f"  Best frame A : {result.best_frame_a}")
    print(f"  Best frame B : {result.best_frame_b}")
    print(f"  SSIM score   : {result.ssim_score:.4f}")
    print(f"  Transition   : {result.transition_type}")
    if result.transition_type != "hard_cut":
        print(f"  xfade dur    : {result.xfade_duration}s")
    sys.exit(0)


if __name__ == "__main__":
    main()
