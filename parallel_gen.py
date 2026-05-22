"""
Parallel generation module for images and videos.

This module provides parallelized generation with rate limiting
to respect Vertex AI API quotas.
"""

import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime

from config import (
    PARALLEL_IMAGE_WORKERS,
    PARALLEL_VIDEO_WORKERS,
    IMAGE_RATE_LIMIT_RPM,
    VIDEO_RATE_LIMIT_RPM,
    SCENES_DIR,
    BUCKET_NAME,
    DEFAULT_STYLE,
)

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter to enforce API rate limits.
    
    Thread-safe implementation that allows N requests per minute.
    """
    
    def __init__(self, requests_per_minute: int):
        self.rate = requests_per_minute
        self.interval = 60.0 / requests_per_minute  # seconds between requests
        self.lock = threading.Lock()
        self.last_request_time = 0.0
    
    def acquire(self) -> float:
        """
        Acquire permission to make a request.
        
        Returns:
            The time waited in seconds
        """
        with self.lock:
            now = time.time()
            time_since_last = now - self.last_request_time
            
            if time_since_last < self.interval:
                wait_time = self.interval - time_since_last
                time.sleep(wait_time)
            else:
                wait_time = 0.0
            
            self.last_request_time = time.time()
            return wait_time


@dataclass
class GenerationResult:
    """Result of a single generation task."""
    scene_idx: int
    success: bool
    output_path: Optional[str] = None
    gcs_uri: Optional[str] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    data: Optional[bytes] = None
    final_prompt: Optional[str] = None


@dataclass
class ParallelGenerationStats:
    """Statistics for parallel generation."""
    total_tasks: int = 0
    completed: int = 0
    failed: int = 0
    total_wait_time: float = 0.0
    total_generation_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def elapsed_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.completed / self.total_tasks


def generate_images_parallel(
    scenes: List[Any],
    generate_fn: Callable,
    global_style: Optional[dict] = None,
    run_id: str = "",
    style_id: str = "ghibli_dark",
    max_workers: int = PARALLEL_IMAGE_WORKERS,
    rate_limit_rpm: int = IMAGE_RATE_LIMIT_RPM,
    reference_image_bytes: Optional[bytes] = None,
    reference_images_list: Optional[list] = None  # List of (name, bytes) tuples
) -> Tuple[Dict[int, GenerationResult], ParallelGenerationStats]:
    """
    Generate images for multiple scenes in parallel.
    
    Args:
        scenes: List of Scene objects from manifest
        generate_fn: Function to generate a single image (generate_scene_image)
        global_style: Global style settings
        run_id: Run identifier
        style_id: Style preset ID
        max_workers: Maximum concurrent workers
        rate_limit_rpm: Rate limit in requests per minute
        
    Returns:
        Tuple of (results dict by scene_idx, stats)
    """
    stats = ParallelGenerationStats(total_tasks=len(scenes))
    stats.start_time = datetime.now()
    
    results: Dict[int, GenerationResult] = {}
    rate_limiter = RateLimiter(rate_limit_rpm)
    
    logger.info(f"Starting parallel image generation: {len(scenes)} scenes, "
                f"{max_workers} workers, {rate_limit_rpm} RPM limit")
    
    def generate_single_image(scene) -> GenerationResult:
        """Generate a single image with rate limiting."""
        start_time = time.time()
        
        try:
            # Wait for rate limiter
            wait_time = rate_limiter.acquire()
            stats.total_wait_time += wait_time
            
            logger.info(f"[Scene {scene.idx}] Generating image...")
            
            image_bytes, local_path, final_prompt = generate_fn(
                scene_idx=scene.idx,
                image_prompt=scene.image_prompt_en,
                image_avoid=scene.image_avoid,
                global_style=global_style,
                output_dir=SCENES_DIR,
                run_id=run_id,
                style_id=style_id,
                visual_context=scene.visual_context,
                reference_image_bytes=reference_image_bytes,
                shot_type=getattr(scene, 'shot_type', None),
                reference_images_list=reference_images_list
            )
            
            duration = time.time() - start_time
            stats.total_generation_time += duration
            
            logger.info(f"[Scene {scene.idx}] Image generated: {len(image_bytes)} bytes in {duration:.1f}s")
            
            return GenerationResult(
                scene_idx=scene.idx,
                success=True,
                output_path=str(local_path) if local_path else None,
                duration_seconds=duration,
                data=image_bytes,
                final_prompt=final_prompt
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            logger.error(f"[Scene {scene.idx}] Image generation failed: {error_msg}")
            
            return GenerationResult(
                scene_idx=scene.idx,
                success=False,
                error=error_msg,
                duration_seconds=duration
            )
    
    # Execute in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_scene = {
            executor.submit(generate_single_image, scene): scene
            for scene in scenes
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_scene):
            scene = future_to_scene[future]
            try:
                result = future.result()
                results[result.scene_idx] = result
                
                if result.success:
                    stats.completed += 1
                else:
                    stats.failed += 1
                    
            except Exception as e:
                logger.error(f"[Scene {scene.idx}] Unexpected error: {e}")
                results[scene.idx] = GenerationResult(
                    scene_idx=scene.idx,
                    success=False,
                    error=str(e)
                )
                stats.failed += 1
    
    stats.end_time = datetime.now()
    
    logger.info(f"Parallel image generation complete: "
                f"{stats.completed}/{stats.total_tasks} succeeded, "
                f"{stats.failed} failed, "
                f"{stats.elapsed_seconds:.1f}s total")
    
    return results, stats


def generate_videos_parallel(
    scenes_with_images: List[Tuple[Any, bytes, Optional[bytes]]],  # (scene, first_frame, last_frame)
    generate_fn: Callable,
    run_id: str = "",
    bucket_name: str = BUCKET_NAME,
    max_workers: int = PARALLEL_VIDEO_WORKERS,
    rate_limit_rpm: int = VIDEO_RATE_LIMIT_RPM,
    style_id: str = DEFAULT_STYLE,
    global_style: Optional[dict] = None,
    subject_reference_gcs_uri: Optional[str] = None,  # Primary character reference for Veo consistency
) -> Tuple[Dict[int, GenerationResult], ParallelGenerationStats]:
    """
    Generate videos for multiple scenes in parallel.
    
    Args:
        scenes_with_images: List of (Scene, first_frame_bytes, last_frame_bytes) tuples
                           last_frame_bytes enables seamless scene-to-scene transitions
        generate_fn: Function to generate a single video (generate_scene_video)
        run_id: Run identifier
        bucket_name: GCS bucket name
        max_workers: Maximum concurrent workers
        rate_limit_rpm: Rate limit in requests per minute
        style_id: Style preset ID to use
        
    Returns:
        Tuple of (results dict by scene_idx, stats)
    """
    stats = ParallelGenerationStats(total_tasks=len(scenes_with_images))
    stats.start_time = datetime.now()
    
    results: Dict[int, GenerationResult] = {}
    rate_limiter = RateLimiter(rate_limit_rpm)
    
    logger.info(f"Starting parallel video generation: {len(scenes_with_images)} scenes, "
                f"{max_workers} workers, {rate_limit_rpm} RPM limit")
    
    # Build style context from global style (Character Bible)
    style_context = None
    if global_style:
        style_context = f"Color: {global_style.get('lut')}, Rules: {global_style.get('rules')}"

    def generate_single_video(scene, image_bytes: bytes, last_frame_bytes: Optional[bytes] = None) -> GenerationResult:
        """Generate a single video with rate limiting. Uses internal retries of generate_fn."""
        start_time = time.time()
        # Underlying generate_fn (generate_scene_video) already has a 10-retry loop.
        # We don't want to double-dip, so we set local retries to 0.
        max_retries = 0 
        
        for attempt in range(max_retries + 1):
            try:
                # Wait for rate limiter
                wait_time = rate_limiter.acquire()
                stats.total_wait_time += wait_time
                
                # Extra backoff wait on retries
                if attempt > 0:
                    backoff_wait = 15 * attempt  # 15s, 30s for retries
                    logger.info(f"[Scene {scene.idx}] Retry {attempt}/{max_retries} after {backoff_wait}s backoff...")
                    time.sleep(backoff_wait)
                
                transition_mode = "seamless" if last_frame_bytes else "standard"
                logger.info(f"[Scene {scene.idx}] Generating video ({scene.seconds}s, {transition_mode})...")
                
                # Generate the video
                video_gcs_uri, final_prompt = generate_fn(
                    scene_idx=scene.idx,
                    image_bytes=image_bytes,
                    video_prompt=scene.video_prompt_en,
                    run_id=run_id,
                    bucket_name=bucket_name,
                    duration_seconds=scene.seconds,
                    style_id=style_id,  # Pass style_id
                    style_context=style_context,  # Pass style_context for retries
                    last_frame_bytes=last_frame_bytes,  # Pass last frame for seamless transitions
                    shot_type=getattr(scene, 'shot_type', None),  # Pass shot_type for frame-anchor
                    extend_scene=getattr(scene, 'extend_scene', False),
                    extension_prompt=getattr(scene, 'extension_prompt_en', None),
                    subject_reference_gcs_uri=subject_reference_gcs_uri,  # Character reference for Veo consistency
                )
                
                duration = time.time() - start_time
                stats.total_generation_time += duration
                
                logger.info(f"[Scene {scene.idx}] Video generated in {duration:.1f}s: {video_gcs_uri}")
                
                return GenerationResult(
                    scene_idx=scene.idx,
                    success=True,
                    gcs_uri=video_gcs_uri,
                    duration_seconds=duration,
                    final_prompt=final_prompt
                )
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if we should retry (rate limit or empty response errors)
                is_retryable = any(phrase in error_msg.lower() for phrase in [
                    "rate", "quota", "no video generated", "limit", "429"
                ])
                
                if is_retryable and attempt < max_retries:
                    logger.warning(f"[Scene {scene.idx}] Retryable error on attempt {attempt + 1}: {error_msg}")
                    continue  # Try again
                
                # Final failure
                duration = time.time() - start_time
                logger.error(f"[Scene {scene.idx}] Video generation failed: {error_msg}")
                
                return GenerationResult(
                    scene_idx=scene.idx,
                    success=False,
                    error=error_msg,
                    duration_seconds=duration
                )
    
    # Execute in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_scene = {
            executor.submit(generate_single_video, scene, first_frame, last_frame): scene
            for scene, first_frame, last_frame in scenes_with_images
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_scene):
            scene = future_to_scene[future]
            try:
                result = future.result()
                results[result.scene_idx] = result
                
                if result.success:
                    stats.completed += 1
                else:
                    stats.failed += 1
                    
            except Exception as e:
                logger.error(f"[Scene {scene.idx}] Unexpected error: {e}")
                results[scene.idx] = GenerationResult(
                    scene_idx=scene.idx,
                    success=False,
                    error=str(e)
                )
                stats.failed += 1
    
    stats.end_time = datetime.now()
    
    logger.info(f"Parallel video generation complete: "
                f"{stats.completed}/{stats.total_tasks} succeeded, "
                f"{stats.failed} failed, "
                f"{stats.elapsed_seconds:.1f}s total")
    
    return results, stats


if __name__ == "__main__":
    # Quick test of rate limiter
    logging.basicConfig(level=logging.INFO)
    
    limiter = RateLimiter(10)  # 10 requests per minute = 6 seconds between requests
    
    print("Testing rate limiter (10 RPM = 6s between requests)...")
    for i in range(3):
        start = time.time()
        wait = limiter.acquire()
        print(f"Request {i+1}: waited {wait:.2f}s (total: {time.time()-start:.2f}s)")
