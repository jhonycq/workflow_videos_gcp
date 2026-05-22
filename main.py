"""
Main Orchestration Module for Video Generation Pipeline.

Coordinates the full pipeline: story generation → image → video → assembly.
Includes logging, retries, cost tracking, and error handling.
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from config import (
    PROJECT_ID,
    BUCKET_NAME,
    NUM_SCENES,
    SECONDS_PER_SCENE,
    MAX_SECONDS_BUDGET,
    COST_PER_SECOND_720P_VEO,
    COST_PER_IMAGE_NANO,
    BUDGET_WARNING_THRESHOLD,
    MAX_RETRIES,
    RETRY_BACKOFF,
    RETRY_INITIAL_WAIT,
    OUTPUT_DIR,
    SCENES_DIR,
    USE_TEXT_TO_VIDEO,
    DEFAULT_STYLE,
    USE_DETERMINISTIC_SEED,
    USE_CONSISTENCY_QA,
)
from schemas import Manifest, GenerationStats
from llm_story import generate_story, regenerate_scene
from gen_image import generate_scene_image
from gen_video import generate_scene_video, generate_scene_video_text
from storage import upload_manifest, upload_scene_image, upload_file
from assemble import assemble_final_video, verify_final_video, assemble_with_tts, assemble_with_single_tts
from styles import get_style_preset, get_available_styles, parse_style_map, generate_seed_from_run, resolve_style_id
from parallel_gen import (
    generate_images_parallel,
    generate_videos_parallel,
    GenerationResult,
    ParallelGenerationStats,
)
from smart_stitch import smart_stitch_scenes, StitchConfig
from add_subtitles import add_subtitles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def generate_run_id() -> str:
    """Generate a unique run ID based on timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logging(run_id: str) -> None:
    """Setup file logging for this run."""
    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_dir / f"{run_id}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    ))
    
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging to {log_dir / f'{run_id}.log'}")


def check_budget(stats: GenerationStats) -> bool:
    """
    Check if we're within budget and warn if approaching limit.
    
    Returns:
        True if within budget, False if exceeded
    """
    budget_usage = stats.total_seconds_generated / MAX_SECONDS_BUDGET
    
    if budget_usage >= 1.0:
        logger.error(f"BUDGET EXCEEDED: {stats.total_seconds_generated}s / {MAX_SECONDS_BUDGET}s")
        logger.error(f"Estimated cost: ${stats.estimated_cost_usd:.2f}")
        return False
    
    if budget_usage >= BUDGET_WARNING_THRESHOLD:
        logger.warning(
            f"BUDGET WARNING: {budget_usage * 100:.1f}% of limit "
            f"({stats.total_seconds_generated}s / {MAX_SECONDS_BUDGET}s)"
        )
    
    return True


def process_scene(
    scene_idx: int,
    manifest: Manifest,
    run_id: str,
    stats: GenerationStats,
    dry_run: bool = False,
    style_id: str = DEFAULT_STYLE
) -> bool:
    """
    Process a single scene: generate image, then video.
    
    Args:
        scene_idx: Scene index (1-based)
        manifest: The manifest object
        run_id: Run identifier
        stats: Generation stats tracker
        dry_run: If True, skip actual generation
        
    Returns:
        True if successful, False if failed
    """
    scene = manifest.get_scene_by_idx(scene_idx)
    if not scene:
        logger.error(f"Scene {scene_idx} not found in manifest")
        return False
    
    logger.info(f"=" * 60)
    logger.info(f"Processing Scene {scene_idx}/{len(manifest.scenes)}")
    logger.info(f"=" * 60)
    
    if dry_run:
        logger.info("[DRY RUN] Skipping actual generation")
        scene.status = "video_done"
        return True
    
    # Check budget
    if not check_budget(stats):
        logger.error("Budget exceeded, stopping generation")
        scene.status = "failed"
        scene.error_message = "Budget exceeded"
        return False
    
    try:
        # Step 1: Generate image
        logger.info(f"Step 1: Generating image for scene {scene_idx}...")
        
        global_style = manifest.global_style.model_dump() if manifest.global_style else None
        
        image_bytes, local_image_path, final_image_prompt = generate_scene_image(
            scene_idx=scene_idx,
            image_prompt=scene.image_prompt_en,
            image_avoid=scene.image_avoid,
            global_style=global_style,
            output_dir=SCENES_DIR,
            run_id=run_id,
            style_id=style_id,  # Pass the style_id to image generation
            shot_type=getattr(scene, 'shot_type', None)
        )
        
        # Update manifest with refined image prompt if it changed
        if final_image_prompt != scene.image_prompt_en:
            logger.info(f"Updating scene {scene_idx} manifest with refined image prompt.")
            scene.image_prompt_en = final_image_prompt
        
        # Build style context for video/retries
        style_context = None
        if global_style:
            style_context = f"Color: {global_style.get('lut', '')}, Rules: {global_style.get('rules', '')}"
        
        # Upload image to GCS
        image_gcs_uri = upload_scene_image(
            image_bytes=image_bytes,
            scene_idx=scene_idx,
            run_id=run_id,
            bucket_name=BUCKET_NAME
        )
        
        scene.image_gcs_path = image_gcs_uri
        scene.status = "image_done"
        
        # Update stats
        stats.estimated_cost_usd += COST_PER_IMAGE_NANO
        
        logger.info(f"Image generated and uploaded: {image_gcs_uri}")
        
        # Step 2: Generate video
        if USE_TEXT_TO_VIDEO:
            # Text-to-video mode: skip image, generate video from prompt only
            logger.info(f"Step 2: Generating video (text-to-video mode) for scene {scene_idx}...")
            
            video_gcs_uri, final_video_prompt = generate_scene_video_text(
                scene_idx=scene_idx,
                video_prompt=scene.video_prompt_en,
                run_id=run_id,
                bucket_name=BUCKET_NAME,
                duration_seconds=scene.seconds,
                style_id=style_id,
                style_context=style_context,
                shot_type=getattr(scene, 'shot_type', None),
                extend_scene=getattr(scene, 'extend_scene', False),
                extension_prompt=getattr(scene, 'extension_prompt_en', None),
            )
        else:
            # Image-to-video mode: generate video from image
            logger.info(f"Step 2: Generating video (image-to-video mode) for scene {scene_idx}...")
            
            video_gcs_uri, final_video_prompt = generate_scene_video(
                scene_idx=scene_idx,
                image_bytes=image_bytes,
                video_prompt=scene.video_prompt_en,
                run_id=run_id,
                bucket_name=BUCKET_NAME,
                duration_seconds=scene.seconds,
                style_id=style_id,
                style_context=style_context,
                shot_type=getattr(scene, 'shot_type', None),
                extend_scene=getattr(scene, 'extend_scene', False),
                extension_prompt=getattr(scene, 'extension_prompt_en', None),
            )
        
        # Update manifest with refined video prompt if it changed
        if final_video_prompt != scene.video_prompt_en:
            logger.info(f"Updating scene {scene_idx} manifest with refined video prompt.")
            scene.video_prompt_en = final_video_prompt
        
        scene.video_gcs_path = video_gcs_uri
        scene.status = "video_done"
        
        if getattr(scene, 'extend_scene', False) and scene.seconds < 13:
            scene.seconds += 7
            
        # Update stats
        stats.update_progress(
            completed=1,
            seconds=scene.seconds,
            cost=scene.seconds * COST_PER_SECOND_720P_VEO
        )
        
        logger.info(f"Video generated: {video_gcs_uri}")
        logger.info(
            f"Progress: {stats.completed_scenes}/{stats.total_scenes} scenes, "
            f"{stats.total_seconds_generated}s generated, "
            f"${stats.estimated_cost_usd:.2f} estimated cost"
        )
        
        return True
        
    except ValueError as e:
        # Non-retryable error (safety, approval needed)
        error_msg = str(e)
        logger.error(f"Scene {scene_idx} failed (non-retryable): {error_msg}")
        
        scene.status = "failed"
        scene.error_message = error_msg
        stats.failed_scenes += 1
        
        # Suggest fixes
        if "person" in error_msg.lower() or "child" in error_msg.lower():
            logger.warning(
                "SUGGESTION: Person generation may require approval. "
                "Try regenerating the scene with 'avoid people' or use "
                "adult silhouettes/distant shots instead."
            )
        
        return False
        
    except Exception as e:
        # Retryable error
        error_msg = str(e)
        logger.error(f"Scene {scene_idx} failed: {error_msg}")
        
        scene.status = "failed"
        scene.error_message = error_msg
        stats.failed_scenes += 1
        
        return False


def run_pipeline(
    theme: Optional[str] = None,
    num_scenes: int = NUM_SCENES,
    dry_run: bool = False,
    resume_run_id: Optional[str] = None,
    style_id: str = DEFAULT_STYLE,
    style_map: Optional[Dict[int, str]] = None,
    base_seed: Optional[int] = None,
    images_only: bool = False,
    skip_assembly: bool = False,
    regenerate_scenes: Optional[str] = None,
    skip_image_gen: bool = False,
    skip_audio_gen: bool = False,
    no_transitions: bool = False,
    no_qa: bool = False,
    videos_only: bool = False,
    max_duration: Optional[int] = None,
    no_subtitles: bool = False,
) -> Dict[str, Any]:
    """
    Run the full video generation pipeline.
    
    Args:
        theme: Optional theme/topic for the story
        num_scenes: Number of scenes to generate
        dry_run: If True, generate manifest only
        resume_run_id: If provided, resume a previous run
        style_id: Default style preset ID
        style_map: Optional per-scene style overrides {scene_idx: style_id}
        base_seed: Optional base seed for deterministic generation
        
    Returns:
        Dictionary with results summary
    """
    # Generate or use existing run ID
    run_id = resume_run_id or generate_run_id()
    setup_logging(run_id)
    
    logger.info("=" * 70)
    logger.info("VIDEO GENERATION PIPELINE START")
    logger.info("=" * 70)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Theme: {theme or 'random'}")
    logger.info(f"Scenes: {num_scenes or 'Dynamic (determined by LLM)'}")
    if num_scenes:
        logger.info(f"Duration: {num_scenes * SECONDS_PER_SCENE}s")
    else:
        logger.info("Duration: TBD (depends on story)")
    logger.info(f"Style: {style_id}")
    if style_map:
        logger.info(f"Style overrides: {style_map}")
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"No QA: {no_qa}")
    logger.info(f"Videos only: {videos_only}")
    logger.info(f"Project: {PROJECT_ID}")
    logger.info(f"Bucket: {BUCKET_NAME}")
    logger.info("=" * 70)
    
    # Runtime override: disable QA if --no-qa flag is used
    if no_qa:
        import config as cfg
        cfg.USE_VISUAL_QA = False
        cfg.USE_CONSISTENCY_QA = False
        logger.info("⚡ QA disabled via --no-qa flag (faster generation, fewer API calls)")
    
    stats = GenerationStats(
        run_id=run_id,
        total_scenes=num_scenes or 0,
        start_time=datetime.now().isoformat()
    )
    
    # Parse selective regeneration indices
    indices_to_regen = []
    if regenerate_scenes:
        try:
            indices_to_regen = [int(i.strip()) for i in regenerate_scenes.split(",")]
            logger.info(f"Targeting scenes for regeneration: {indices_to_regen}")
        except Exception as e:
            logger.error(f"Failed to parse regenerate_scenes '{regenerate_scenes}': {e}")
            sys.exit(1)
    
    try:
        # Step 1: Generate story and manifest (or load if resuming)
        if resume_run_id:
            logger.info(f"\n[RESUME] Loading existing manifest for run {resume_run_id}...")
            local_manifest_path = OUTPUT_DIR / resume_run_id / "manifest.json"
            if local_manifest_path.exists():
                with open(local_manifest_path, 'r', encoding='utf-8') as f:
                    manifest_data = json.load(f)
                    manifest = Manifest(**manifest_data)
                # Restore the original style_id from the manifest (don't re-suggest)
                if manifest.style_id:
                    style_id = manifest.style_id
                    logger.info(f"Restored style from manifest: {style_id}")
                logger.info("Manifest loaded successfully.")
            else:
                logger.warning(f"Local manifest not found at {local_manifest_path}. Generating new story.")
                manifest = generate_story(
                    theme=theme,
                    num_scenes=num_scenes,
                    seconds_per_scene=SECONDS_PER_SCENE,
                    style_id=style_id,
                    max_duration=max_duration
                )
        else:
            # ─── AUDIO-FIRST: Generate story once, audio once, adapt video ───

            # Phase 1: Generate initial story
            logger.info(f"\n[PHASE 1] Generating story and prompts...")
            if max_duration:
                logger.info(f"Max duration target: {max_duration}s")
            
            manifest = generate_story(
                theme=theme,
                num_scenes=num_scenes,
                seconds_per_scene=SECONDS_PER_SCENE,
                style_id=style_id,
                max_duration=max_duration
            )
            manifest.style_id = style_id
            
            # V3: Audio-first approach - generate audio ONCE, adapt video to audio.
            # No retry loop. ElevenLabs credits are expensive, never regenerate audio.
            # Phase 1B below will scale scene durations to match the actual audio length.
        
        # Update stats if total_scenes was dynamic
        stats.total_scenes = len(manifest.scenes)
        
        logger.info(f"Story generated: '{manifest.title}'")
        logger.info(f"Total scenes: {len(manifest.scenes)}")
        logger.info(f"Total duration: {manifest.total_duration_seconds}s")
        
        # Save initial manifest to GCS
        manifest_uri = upload_manifest(manifest.model_dump(), run_id, BUCKET_NAME)
        logger.info(f"Manifest saved: {manifest_uri}")
        
        # Audio-First Generation (Phase 1B) - V3: ElevenLabs Dual-Narrator
        logger.info("\n[PHASE 1B] Generating Audio-First Dual-Narrator Dialogue & Timestamp Scaling...")
        run_folder = OUTPUT_DIR / run_id
        run_folder.mkdir(parents=True, exist_ok=True)

        from gen_tts import generate_dialogue_audio

        num_scenes = len(manifest.scenes)
        raw_audio_path = run_folder / "narration.wav"

        if not skip_audio_gen:
            # V3: Generate dialogue audio ONCE from per-scene dialogue (narrator + child interleaved)
            # Audio drives everything - scene durations will be adapted to match audio length.
            audio_path, total_audio_dur = generate_dialogue_audio(
                scenes=manifest.scenes,
                output_path=raw_audio_path,
                content_type=getattr(manifest, 'content_type', 'general'),
            )

            if audio_path and Path(audio_path).exists():
                logger.info(f"Audio generated: {total_audio_dur:.2f}s. Mathematical Scene Scaling starting...")

                # Proportional exact distribution (4s, 6s, or 8s blocks to match Veo constraints)
                # Algorithm: Start all scenes at 6s.
                # If sum < audio, upgrade sequentially to 8s.
                # If sum > audio + margin, downgrade sequentially to 4s.

                for scene in manifest.scenes:
                    scene.seconds = 6

                current_sum = sum(s.seconds for s in manifest.scenes)

                if current_sum < total_audio_dur:
                    # Upgrade to 8s to cover the extra audio
                    idx_to_upgrade = 0
                    while current_sum < total_audio_dur and idx_to_upgrade < num_scenes:
                        manifest.scenes[idx_to_upgrade].seconds = 8
                        current_sum += 2
                        idx_to_upgrade += 1
                elif current_sum > total_audio_dur + 1.5:
                    # Downgrade to 4s if we are vastly overshooting the audio (margin of 1.5s tolerance)
                    idx_to_downgrade = num_scenes - 1
                    while (current_sum - 2) >= total_audio_dur and idx_to_downgrade >= 0:
                        manifest.scenes[idx_to_downgrade].seconds = 4
                        current_sum -= 2
                        idx_to_downgrade -= 1

                for scene in manifest.scenes:
                    logger.info(f"Scene {scene.idx} duration mathematically set to {scene.seconds}s based on audio length")
            else:
                logger.warning("Audio generation failed. Defaulting to 6s scenes.")
                for scene in manifest.scenes:
                    scene.seconds = 6
        else:
             logger.info("Skipping audio generation (--skip-audio-gen). Defaulting all scenes to 6s.")
             for scene in manifest.scenes:
                 scene.seconds = 6
                     
        # Recalculate total duration based on generated audio
        manifest.total_duration_seconds = sum(scene.seconds for scene in manifest.scenes)
        logger.info(f"New Total Duration (Audio-First): {manifest.total_duration_seconds}s")
        
        # Update Stats with new total duration
        stats.total_seconds_generated = 0 # reset because it's calculated in the loop later
        
        # Save local manifest with calculated seconds
        local_manifest_path = run_folder / "manifest.json"
        with open(local_manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest.model_dump(), f, indent=2, ensure_ascii=False)
        logger.info(f"Local manifest (with timestamps): {local_manifest_path}")
        
        # Save story separately for easy reference
        story_data = {
            "title": manifest.title,
            "theme": theme,
            "global_style": manifest.global_style.model_dump() if manifest.global_style else None,
            "total_scenes": len(manifest.scenes),
            "total_duration_seconds": manifest.total_duration_seconds,
            "scenes_summary": [
                {
                    "idx": scene.idx,
                    "seconds": scene.seconds,
                    "image_prompt_en": scene.image_prompt_en,
                    "video_prompt_en": scene.video_prompt_en,
                }
                for scene in manifest.scenes
            ]
        }
        story_path = run_folder / "story.json"
        with open(story_path, 'w', encoding='utf-8') as f:
            json.dump(story_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Story saved: {story_path}")
        
        if dry_run:
            logger.info("\n[DRY RUN] Skipping image and video generation.")
            return {
                "status": "dry_run_complete",
                "run_id": run_id,
                "manifest_uri": manifest_uri,
                "local_manifest": str(local_manifest_path),
                "title": manifest.title,
                "scenes": num_scenes,
            }
        
        # Step 2: Generate reference images for consistency (Phase 1C)
        ref_images_bytes = []  # List of (name, bytes) for reference images
        primary_ref_gcs_uri: Optional[str] = None  # GCS URI of first character reference for Veo
        if hasattr(manifest, 'reference_sheets') and manifest.reference_sheets and not dry_run:
            logger.info(f"\n[PHASE 1C] Generating {len(manifest.reference_sheets)} reference images for consistency...")
            ref_dir = run_folder / "references"
            ref_dir.mkdir(parents=True, exist_ok=True)

            for ref_sheet in manifest.reference_sheets:
                ref_name_slug = ref_sheet.name.replace(' ', '_').lower()
                ref_path = ref_dir / f"{ref_name_slug}.png"
                ref_gcs_path = f"mvp/{run_id}/references/{ref_name_slug}.png"
                ref_gcs_uri = f"gs://{BUCKET_NAME}/{ref_gcs_path}"

                # Check if reference image already exists (resume support)
                if ref_path.exists():
                    with open(ref_path, "rb") as f:
                        ref_data = f.read()
                    ref_images_bytes.append((ref_sheet.name, ref_data))
                    logger.info(f"  Reusing existing reference: {ref_sheet.name} ({ref_sheet.type})")
                    # Set primary ref for Veo (first character reference)
                    if primary_ref_gcs_uri is None and ref_sheet.type == "character":
                        primary_ref_gcs_uri = ref_gcs_uri
                    continue

                try:
                    # Generate reference image using Gemini image generation
                    style = get_style_preset(style_id)
                    ref_prompt = f"""{style['style_positive']}

{ref_sheet.description}

Single subject portrait/reference sheet. Clean background. Vertical 9:16 composition.
High detail, consistent lighting, 8k resolution."""

                    ref_data, _, _ = generate_scene_image(
                        scene_idx=0,  # 0 = reference, not a real scene
                        image_prompt=ref_prompt,
                        image_avoid=style.get('style_negative', ''),
                        global_style=manifest.global_style.model_dump() if manifest.global_style else None,
                        output_dir=ref_dir,
                        run_id=run_id,
                        style_id=style_id,
                        visual_context=f"Reference sheet for: {ref_sheet.name} ({ref_sheet.type})",
                        shot_type="EYE-LEVEL MS" if ref_sheet.type == "character" else "EYE-LEVEL LS"
                    )

                    # Save reference image
                    with open(ref_path, "wb") as f:
                        f.write(ref_data)
                    ref_images_bytes.append((ref_sheet.name, ref_data))
                    logger.info(f"  ✅ Reference generated: {ref_sheet.name} ({ref_sheet.type}) - {len(ref_data)} bytes")

                    # Upload to GCS and capture URI for Veo
                    upload_file(ref_path, ref_gcs_path, BUCKET_NAME)
                    if primary_ref_gcs_uri is None and ref_sheet.type == "character":
                        primary_ref_gcs_uri = ref_gcs_uri
                        logger.info(f"  Primary reference for Veo set: {primary_ref_gcs_uri}")

                except Exception as e:
                    logger.warning(f"  ⚠️ Failed to generate reference for {ref_sheet.name}: {e}. Continuing without it.")

            logger.info(f"Reference images ready: {len(ref_images_bytes)}/{len(manifest.reference_sheets)}")
            if primary_ref_gcs_uri:
                logger.info(f"Primary Veo reference: {primary_ref_gcs_uri}")
        
        # Step 3: Generate scene images (Phase 2A)
        logger.info("\n[PHASE 2A] Generating ALL images in parallel...")
        
        global_style = manifest.global_style.model_dump() if manifest.global_style else None
        
        # Consistency Anchor Strategy:
        # If consistency QA is enabled, generate Scene 1 first sequentially as a reference
        # BUT skip if not in indices_to_regen AND regenerate_scenes is active
        # OR if skip_image_gen is enabled
        reference_image_bytes = None
        
        # Load existing images to reuse (auto-resume capability)
        scene_images: Dict[int, bytes] = {}
        logger.info("Checking for existing images to reuse...")
        for s in manifest.scenes:
            # Potential paths:
            # 1. out/{run_id}/scene_{idx:02d}.jpg
            # 2. out/scenes/{run_id}/{idx:02d}/image.png (or jpg)
            potential_paths = [
                OUTPUT_DIR / run_id / f"scene_{s.idx:02d}.jpg",
                OUTPUT_DIR / run_id / f"scene_{s.idx:02d}.png",
                SCENES_DIR / run_id / f"{s.idx:02d}" / "image.png",
                SCENES_DIR / run_id / f"{s.idx:02d}" / "image.jpg",
            ]
            
            for img_path in potential_paths:
                if img_path.exists():
                    with open(img_path, "rb") as f:
                        data = f.read()
                        scene_images[s.idx] = data
                        if s.idx == 1:
                            reference_image_bytes = data
                    logger.info(f"Reusing existing image for scene {s.idx}: {img_path.name}")
                    break
        logger.info(f"Found {len(scene_images)} existing images to reuse.")

        if indices_to_regen:
            scenes_to_generate = [s for s in manifest.scenes if s.idx in indices_to_regen]
        elif skip_image_gen:
            logger.info("Skipping image generation as requested. Reusing existing images.")
            scenes_to_generate = []
        else:
            # Default resume behavior: only generate what is missing
            scenes_to_generate = [s for s in manifest.scenes if s.idx not in scene_images]

        image_results = {}
        
        # Sequentially generate anchor if targeted and NOT skipped
        if USE_CONSISTENCY_QA and len(scenes_to_generate) > 0 and scenes_to_generate[0].idx == 1 and not skip_image_gen:
            scene_1 = scenes_to_generate[0]
            logger.info(f"\n[CONSISTENCY ANCHOR] Generating Scene 1 first as reference...")
            try:
                img_data, _, final_img_prompt = generate_scene_image(
                    scene_idx=scene_1.idx,
                    image_prompt=scene_1.image_prompt_en,
                    image_avoid=scene_1.image_avoid,
                    global_style=global_style,
                    output_dir=SCENES_DIR,
                    run_id=run_id,
                    style_id=style_id,
                    visual_context=scene_1.visual_context,
                    shot_type=getattr(scene_1, 'shot_type', None),
                    reference_images_list=ref_images_bytes if ref_images_bytes else None
                )
                # Sync refined prompt
                scene_1.image_prompt_en = final_img_prompt
                reference_image_bytes = img_data
                image_results[scene_1.idx] = GenerationResult(
                    scene_idx=scene_1.idx,
                    success=True,
                    data=img_data
                )
                scenes_to_generate = scenes_to_generate[1:]
                logger.info("Scene 1 reference secured.")
            except Exception as e:
                logger.error(f"Failed to generate Scene 1 reference: {e}. Proceeding without anchor.")
        
        # Generate remaining images in parallel (passing reference if available)
        if scenes_to_generate:
            parallel_results, image_stats = generate_images_parallel(
                scenes=scenes_to_generate,
                generate_fn=generate_scene_image,
                global_style=global_style,
                run_id=run_id,
                style_id=style_id,
                reference_image_bytes=reference_image_bytes,
                reference_images_list=ref_images_bytes if ref_images_bytes else None
            )
            image_results.update(parallel_results)
            logger.info(f"Parallel image generation complete: {len(parallel_results)} tasks in {image_stats.elapsed_seconds:.1f}s")
        else:
            # All done in sequential step (unlikely but possible if only 1 scene)
            image_stats = ParallelGenerationStats(total_tasks=len(manifest.scenes), completed=len(image_results))
        
        # Format: (scene, first_frame_bytes, last_frame_bytes)
        # last_frame of scene N = first_frame of scene N+1 (for seamless transitions)
        # Update manifest only for scenes we actually processed
        for scene in (manifest.scenes if not indices_to_regen else [s for s in manifest.scenes if s.idx in indices_to_regen]):
            if skip_image_gen and scene.idx in scene_images:
                 if scene.status != "video_done":
                     scene.status = "image_done"
                 continue
                 
            result = image_results.get(scene.idx)
            if result and result.success and result.data:
                # Upload to GCS
                image_gcs_uri = upload_scene_image(
                    image_bytes=result.data,
                    scene_idx=scene.idx,
                    run_id=run_id,
                    bucket_name=BUCKET_NAME
                )
                scene.image_gcs_path = image_gcs_uri
                scene.status = "image_done"
                scene_images[scene.idx] = result.data
                # Sync refined prompt from parallel result
                if result.final_prompt:
                    scene.image_prompt_en = result.final_prompt
                stats.estimated_cost_usd += COST_PER_IMAGE_NANO
            elif scene.idx in scene_images:
                # Reusing existing image
                if scene.status != "video_done":
                    scene.status = "image_done"
            else:
                scene.status = "failed"
                scene.error_message = result.error if result else "No image generated"
                stats.failed_scenes += 1
        
        # Build scenes_with_images with seamless transitions
        # Format: (scene, first_frame_bytes, last_frame_bytes)
        # last_frame of scene N = first_frame of scene N+1 (for seamless transitions)
        scenes_with_images = []
        sorted_scenes = sorted([s for s in manifest.scenes if s.idx in scene_images], key=lambda x: x.idx)
        
        for i, scene in enumerate(sorted_scenes):
            first_frame = scene_images[scene.idx]
            
            # Determine if we should inject the last frame of the PREVIOUS scene
            # Note: The logic in Veo 3.1 is that `last_frame_bytes` given to Scene N
            # is actually the end frame of Scene N.
            # *Correction*: In this pipeline, `last_frame_bytes` acts as the *start frame modifier*
            # to make it seamlessly continue from the previous video.
            # So, if Scene N says 'interpolated', it MUST receive Scene N-1's image.
            
            # Wait, looking at the existing code:
            # `if i < len(sorted_scenes) - 1 and not no_transitions: next_scene = sorted_scenes[i + 1]; last_frame = scene_images.get(next_scene.idx)`
            # This means `last_frame_bytes` passed to Scene N is actually the FIRST frame of Scene N+1!
            # Veo 3.1 interpolate: "Start at `image_bytes`, end at `last_frame_bytes`".
            
            # Therefore, if the NEXT scene (N+1) is `image_to_video` (hard cut), 
            # we should NOT pass the next scene's frame as the end frame for the CURRENT scene.
            
            if i < len(sorted_scenes) - 1 and not no_transitions:
                next_scene = sorted_scenes[i + 1]
                
                # NEW LOGIC: Look ahead at the next scene's motion_type
                if getattr(next_scene, 'motion_type', 'image_to_video') == 'image_to_video':
                    # The next scene wants a hard cut, so don't force the current scene to morph into it.
                    last_frame = None
                    logger.info(f"Scene {scene.idx} -> {next_scene.idx}: HARD CUT (motion_type='image_to_video'). No end-frame interpolation.")
                else:
                    # The next scene wants a seamless transition, so use its first frame as our end frame.
                    last_frame = scene_images.get(next_scene.idx)
                    logger.info(f"Scene {scene.idx} -> {next_scene.idx}: SEAMLESS INTERPOLATION (motion_type='interpolated').")
            else:
                # Last scene OR no_transitions enabled
                last_frame = None
            
            scenes_with_images.append((scene, first_frame, last_frame))
        
        logger.info(f"Prepared {len(scenes_with_images)} scenes for video generation")
        
        # [STRICT RULE] Verify all scenes have images before proceeding to video generation
        # This prevents generating fragmented videos.
        if not indices_to_regen and len(scenes_with_images) < len(manifest.scenes):
             missing_indices = [s.idx for s in manifest.scenes if s.idx not in scene_images]
             logger.error(f"❌ ERROR: Cannot proceed to video generation. Missing images for scenes: {missing_indices}")
             logger.error("All scenes in the manifest must have images before generating the final video.")
             return {
                 "status": "error_missing_images",
                 "missing_scenes": missing_indices,
                 "run_id": run_id,
                 "manifest_uri": manifest_uri
             }

        # Update manifest after all images
        upload_manifest(manifest.model_dump(), run_id, BUCKET_NAME)
        
        if images_only:
            logger.info("\n[IMAGES ONLY] skipping video generation and assembly.")
            return {
                "status": "images_only_complete",
                "run_id": run_id,
                "manifest_uri": manifest_uri,
                "title": manifest.title,
                "completed_images": image_stats.completed
            }

        logger.info(f"\n[PHASE 2B] Generating ALL videos in parallel (with seamless transitions)...")
        
        # Filter for regeneration if requested
        if indices_to_regen:
            scenes_with_images = [swi for swi in scenes_with_images if swi[0].idx in indices_to_regen]
            logger.info(f"Filtered to {len(scenes_with_images)} scenes for selective video regeneration.")
        else:
            # If no explicit regeneration requested, skip scenes that already have videos
            already_done = []
            to_generate = []
            for swi in scenes_with_images:
                if getattr(swi[0], 'video_gcs_path', None) and swi[0].status == "video_done":
                    already_done.append(swi)
                else:
                    to_generate.append(swi)
            if already_done:
                logger.info(f"Skipping {len(already_done)} scenes that already have videos generated.")
                scenes_with_images = to_generate
                # update progress based on the ones we are skipping
                for done_swi in already_done:
                    stats.update_progress(completed=1, seconds=done_swi[0].seconds, cost=0)

        if scenes_with_images:
            video_results, video_stats = generate_videos_parallel(
                scenes_with_images=scenes_with_images,
                generate_fn=generate_scene_video,
                run_id=run_id,
                bucket_name=BUCKET_NAME,
                style_id=style_id,  # Pass the resolved style_id (e.g. ghibli_whimsical)
                global_style=manifest.global_style.model_dump(),
                subject_reference_gcs_uri=primary_ref_gcs_uri,  # Primary character ref for Veo consistency
            )
            
            logger.info(f"Video generation complete: {video_stats.completed}/{video_stats.total_tasks} in {video_stats.elapsed_seconds:.1f}s")
            
            # Collect failed scenes for retry
            failed_scenes = []
            
            # Update scenes with video results
            for scene, first_frame, last_frame in scenes_with_images:
                result = video_results.get(scene.idx)
                if result and result.success:
                    scene.video_gcs_path = result.gcs_uri
                    scene.status = "video_done"
                    # Sync refined prompt
                    if result.final_prompt:
                        scene.video_prompt_en = result.final_prompt
                    
                    if getattr(scene, 'extend_scene', False) and scene.seconds < 13:
                        scene.seconds += 7
                        
                    stats.update_progress(
                        completed=1,
                        seconds=scene.seconds,
                        cost=scene.seconds * COST_PER_SECOND_720P_VEO
                    )
                else:
                    scene.status = "failed"
                    scene.error_message = result.error if result else "No video generated"
                    stats.failed_scenes += 1
                    # Save for retry
                    failed_scenes.append((scene, first_frame, last_frame))
            
            # [RETRY PASS] Try to regenerate failed videos one more time
            if failed_scenes:
                logger.info(f"\n[PHASE 2C] RETRY PASS: Attempting to regenerate {len(failed_scenes)} failed videos...")
                
                for scene, first_frame, last_frame in failed_scenes:
                    logger.info(f"[Scene {scene.idx}] Retrying video generation after 30s cooldown...")
                    time.sleep(30)  # Cool down before retry
                    
                    try:
                        video_gcs_uri, final_v_prompt = generate_scene_video(
                            scene_idx=scene.idx,
                            image_bytes=first_frame,
                            video_prompt=scene.video_prompt_en,
                            run_id=run_id,
                            bucket_name=BUCKET_NAME,
                            duration_seconds=scene.seconds,
                            style_id=style_id,
                            style_context=f"Color: {manifest.global_style.lut}, Rules: {manifest.global_style.rules}",
                            last_frame_bytes=last_frame,
                            shot_type=getattr(scene, 'shot_type', None),
                            extend_scene=getattr(scene, 'extend_scene', False),
                            extension_prompt=getattr(scene, 'extension_prompt_en', None)
                        )
                        
                        # Success on retry!
                        scene.video_gcs_path = video_gcs_uri
                        scene.video_prompt_en = final_v_prompt
                        scene.status = "video_done"
                        scene.error_message = None
                        
                        if getattr(scene, 'extend_scene', False) and scene.seconds < 13:
                            scene.seconds += 7
                            
                        stats.failed_scenes -= 1  # Remove from failed count
                        stats.update_progress(
                            completed=1,
                            seconds=scene.seconds,
                            cost=scene.seconds * COST_PER_SECOND_720P_VEO
                        )
                        logger.info(f"[Scene {scene.idx}] RETRY SUCCESS: {video_gcs_uri}")
                        
                    except Exception as e:
                        logger.error(f"[Scene {scene.idx}] RETRY FAILED: {e}")
                        scene.error_message = f"Retry also failed: {e}"
        
        # Final manifest update
        upload_manifest(manifest.model_dump(), run_id, BUCKET_NAME)

        logger.info(f"\n[PARALLEL SUMMARY] Images: {image_stats.completed} done, Videos: {video_stats.completed if scenes_with_images else 0} done")

        # ── Phase 2D: Smart Seam Stitch ──────────────────────────────────────
        # Scan consecutive scene video pairs for overlapping/duplicate frames
        # (common when Veo I2V anchors each clip to the previous clip's last frame).
        # If overlap is detected, smart_stitch eliminates redundant frames and
        # produces a single seamless silent video.  If no overlap → skip.
        stitched_video_path = None
        scenes_dir_for_run = SCENES_DIR / run_id
        if scenes_dir_for_run.exists() and not skip_assembly:
            try:
                stitch_cfg = StitchConfig(
                    window_frames=24,
                    ssim_threshold_hard=0.85,
                    ssim_threshold_xfade=0.50,
                    xfade_duration=0.5,
                    xfade_transition="fade",
                    grayscale_ssim=False,
                )
                logger.info(f"\n[PHASE 2D] Smart Seam Detection on {scenes_dir_for_run}...")
                stitched_video_path = smart_stitch_scenes(
                    run_id=run_id,
                    scenes_dir=scenes_dir_for_run,
                    output_dir=OUTPUT_DIR,
                    config=stitch_cfg,
                    pattern="*_video.mp4",
                    min_ssim_to_apply=0.50,
                )
                if stitched_video_path:
                    logger.info(f"[PHASE 2D] ✅ Smart stitch applied → {stitched_video_path}")
                else:
                    logger.info("[PHASE 2D] No overlapping frames detected — regular pipeline continues.")
            except Exception as e:
                logger.warning(f"[PHASE 2D] Smart stitch error (non-fatal, continuing): {e}")
                stitched_video_path = None

        # Step 3: Assemble final video with TTS narration
        final_video_subtitles_path = None
        final_video_watermarked_path = None
        if skip_assembly:
            logger.info("\n[SKIP ASSEMBLY] Skipping Phase 3 as requested.")
            final_video_path = None
        elif stats.completed_scenes > 0:
            try:
                # Extract narration texts and parameters ONLY from SUCCESSFUL scenes
                # This ensures TTS duration matches actual video duration
                successful_scenes = [
                    scene for scene in manifest.scenes 
                    if scene.status == "video_done" and scene.video_gcs_path
                ]
                
                scene_durations = [
                    float(scene.seconds) for scene in successful_scenes
                ]
                # Extract TTS style parameters
                tts_emotions = [
                    getattr(scene, 'tts_emotion', 'neutral') or 'neutral'
                    for scene in successful_scenes
                ]
                tts_paces = [
                    getattr(scene, 'tts_pace', 'normal') or 'normal'
                    for scene in successful_scenes
                ]
                tts_intensities = [
                    getattr(scene, 'tts_intensity', 0.5) or 0.5
                    for scene in successful_scenes
                ]

                # V3: Audio is generated in Phase 1B from dialogue arrays.
                # Check if narration.wav exists (V3 dialogue-based audio).
                # Fallback to global_narration (V2 legacy) if narration.wav is missing.
                has_dialogue_audio = raw_audio_path.exists() and raw_audio_path.stat().st_size > 0
                global_narr = getattr(manifest, "global_narration", "").strip()
                narration_texts = [global_narr] if global_narr else []
                
                if has_dialogue_audio or any(narration_texts):
                    logger.info(f"Using SINGLE-AUDIO TTS assembly (dialogue_audio={has_dialogue_audio}, legacy_narration={bool(narration_texts)})")
                    logger.info(f"Total duration: {sum(scene_durations)}s")
                    logger.info(f"Content type: {getattr(manifest, 'content_type', 'general')}, Voice: {getattr(manifest, 'narrator_voice', '')}")
                    
                    final_video_path = assemble_with_single_tts(
                        run_id=run_id,
                        narration_texts=narration_texts,
                        scene_durations=scene_durations,
                        tts_emotions=tts_emotions,
                        tts_paces=tts_paces,
                        tts_intensities=tts_intensities,
                        content_type=getattr(manifest, 'content_type', 'general'),
                        narrator_archetype=getattr(manifest, 'narrator_archetype', ''),
                        narrator_voice=getattr(manifest, 'narrator_voice', ''),
                        bucket_name=BUCKET_NAME,
                        upload_to_gcs=True,
                        reuse_existing_audio=has_dialogue_audio or skip_audio_gen,
                        pre_stitched_path=stitched_video_path,  # Phase 2D output (None = skip)
                    )
                else:
                    # Fallback to regular assembly (no TTS)
                    logger.info("No narration audio found, using regular assembly (no audio)")
                    final_video_path = assemble_final_video(
                        run_id=run_id,
                        bucket_name=BUCKET_NAME,
                        upload_to_gcs=True
                    )
                
                # Verify
                verification = verify_final_video(
                    final_video_path,
                    stats.completed_scenes
                )
                
                logger.info(f"Final video: {final_video_path}")
                logger.info(f"Verification: {verification}")

                # Phase 3B: Add subtitles with word highlighting
                final_video_subtitles_path = None
                if final_video_path and not no_subtitles:
                    logger.info("\n[PHASE 3B] Generating subtitles with word highlighting...")
                    try:
                        final_video_subtitles_path = add_subtitles(run_id, language="es")
                        logger.info(f"[PHASE 3B] ✅ Subtitles burned → {final_video_subtitles_path}")
                    except Exception as sub_e:
                        logger.warning(f"[PHASE 3B] Subtitle generation failed (non-fatal): {sub_e}")

                # Phase 3C: Add teleporting watermark
                from config import WATERMARK_ENABLED, WATERMARK_IMAGE, WATERMARK_OPACITY, WATERMARK_TELEPORT_INTERVAL, WATERMARK_SCALE
                final_video_watermarked_path = None
                best_video = final_video_subtitles_path or final_video_path
                if best_video and WATERMARK_ENABLED and WATERMARK_IMAGE and Path(WATERMARK_IMAGE).exists():
                    logger.info("\n[PHASE 3C] Adding teleporting watermark...")
                    try:
                        from add_watermark import add_watermark
                        wm_output = Path(best_video).parent / f"{Path(best_video).stem}_wm.mp4"
                        final_video_watermarked_path = add_watermark(
                            video_path=str(best_video),
                            watermark_path=WATERMARK_IMAGE,
                            output_path=str(wm_output),
                            opacity=WATERMARK_OPACITY,
                            teleport_interval=WATERMARK_TELEPORT_INTERVAL,
                            scale_pct=WATERMARK_SCALE,
                        )
                        logger.info(f"[PHASE 3C] ✅ Watermark added → {final_video_watermarked_path}")
                    except Exception as wm_e:
                        logger.warning(f"[PHASE 3C] Watermark failed (non-fatal): {wm_e}")

            except Exception as e:
                logger.error(f"Assembly failed: {e}")
                final_video_path = None
                final_video_subtitles_path = None
                final_video_watermarked_path = None
        else:
            logger.error("No scenes completed, skipping assembly")
            final_video_path = None
            final_video_watermarked_path = None

        # Final stats
        stats.end_time = datetime.now().isoformat()
        
        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Title: {manifest.title}")
        logger.info(f"Completed: {stats.completed_scenes}/{stats.total_scenes} scenes")
        logger.info(f"Failed: {stats.failed_scenes} scenes")
        logger.info(f"Duration generated: {stats.total_seconds_generated}s")
        logger.info(f"Estimated cost: ${stats.estimated_cost_usd:.2f}")
        if final_video_path:
            logger.info(f"Final video: {final_video_path}")
        logger.info("=" * 70)
        
        return {
            "status": "completed" if stats.failed_scenes == 0 else "partial",
            "run_id": run_id,
            "title": manifest.title,
            "completed_scenes": stats.completed_scenes,
            "failed_scenes": stats.failed_scenes,
            "total_seconds": stats.total_seconds_generated,
            "estimated_cost_usd": stats.estimated_cost_usd,
            "final_video": str(final_video_path) if final_video_path else None,
            "final_video_subtitles": str(final_video_subtitles_path) if final_video_subtitles_path else None,
            "final_video_watermarked": str(final_video_watermarked_path) if final_video_watermarked_path else None,
            "manifest_uri": manifest_uri,
        }
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return {
            "status": "failed",
            "run_id": run_id,
            "error": str(e),
        }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI Video Generation Pipeline using Gemini and Veo 3.1 Fast"
    )
    
    parser.add_argument(
        "--theme",
        type=str,
        default=None,
        help="Theme or topic for the video story"
    )
    
    parser.add_argument(
        "--story-file",
        type=str,
        default=None,
        help="Path to a text file containing a dense, detailed story to use as the theme"
    )
    
    parser.add_argument(
        "--scenes",
        type=int,
        default=None,  # None = LLM decides (4-8 scenes)
        help="Number of scenes to generate. If not specified, LLM decides (4-8 scenes based on story needs)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate manifest only, skip image/video generation"
    )
    
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Generate images only, skip video generation and assembly"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume a previous run by its run_id"
    )

    parser.add_argument(
        "--no-assemble",
        "--skip-assembly",
        dest="skip_assembly",
        action="store_true",
        help="Generate all scene videos but do NOT create the final concatenated video"
    )
    
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Override GCP project ID"
    )
    
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="Override GCS bucket name"
    )
    
    parser.add_argument(
        "--style",
        type=str,
        default=DEFAULT_STYLE,
        help=f"Style preset ID (default: {DEFAULT_STYLE}). Use 'auto' for LLM to suggest based on theme. Available: auto, {', '.join(get_available_styles())}"
    )
    
    parser.add_argument(
        "--style-map",
        type=str,
        default=None,
        help="Per-scene style overrides in format '1:style1,2:style2,3:style3'"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed for deterministic generation"
    )

    parser.add_argument(
        "--regenerate-scenes",
        type=str,
        default=None,
        help="Target specific scene indices for regeneration (e.g., '1,2,4'). Use with --resume."
    )

    parser.add_argument(
        "--skip-image-gen",
        action="store_true",
        help="Skip image generation for targeted scenes and reuse existing images"
    )

    parser.add_argument(
        "--skip-audio-gen",
        action="store_true",
        help="Skip TTS narration generation and reuse existing narration.wav"
    )
    
    parser.add_argument(
        "--no-transitions",
        action="store_true",
        help="Disable seamless transitions (last_frame) between scenes"
    )
    
    parser.add_argument(
        '--no-qa',
        action='store_true',
        help='Disable visual QA during image generation (faster, fewer API calls)'
    )
    
    parser.add_argument(
        '--max-duration',
        type=int,
        default=None,
        help='Maximum total audio/video duration in seconds. If audio exceeds this, scenes are reduced and story is regenerated.'
    )
    
    parser.add_argument(
        '--videos-only',
        action='store_true',
        help='Skip image generation, generate videos from --resume run. Requires --resume.'
    )

    parser.add_argument(
        '--no-subtitles',
        action='store_true',
        help='Skip subtitle generation after final video assembly.'
    )

    args = parser.parse_args()
    
    # Read story from file if provided
    story_theme = args.theme
    if args.story_file:
        try:
            with open(args.story_file, 'r', encoding='utf-8') as f:
                story_theme = f.read().strip()
                print(f"📄 Loaded story from {args.story_file} ({len(story_theme)} characters)")
        except Exception as e:
            print(f"❌ Error reading story file {args.story_file}: {e}")
            sys.exit(1)
            
    # Override config if provided
    if args.project:
        import config
        config.PROJECT_ID = args.project
    
    if args.bucket:
        import config
        config.BUCKET_NAME = args.bucket
    
    # Parse style map if provided
    style_map_dict = parse_style_map(args.style_map) if args.style_map else None
    
    # Resolve style (handle 'auto' by asking LLM to suggest based on theme)
    resolved_style = resolve_style_id(
        style_id=args.style,
        theme=story_theme or "",
        project_id=PROJECT_ID
    )
    
    if args.style.lower() == "auto":
        print(f"\n[AUTO] LLM suggested style for theme: {resolved_style}")
    
    # Validate --videos-only requires --resume
    if args.videos_only and not args.resume:
        print("❌ Error: --videos-only requires --resume to specify which run to use")
        sys.exit(1)
    
    # Run pipeline
    result = run_pipeline(
        theme=story_theme,
        num_scenes=args.scenes,
        dry_run=args.dry_run,
        resume_run_id=args.resume,
        style_id=resolved_style,
        style_map=style_map_dict,
        base_seed=args.seed,
        images_only=args.images_only,
        skip_assembly=args.skip_assembly,
        regenerate_scenes=args.regenerate_scenes,
        skip_image_gen=args.skip_image_gen or args.videos_only,  # videos-only skips images
        skip_audio_gen=args.skip_audio_gen,
        no_transitions=args.no_transitions,
        no_qa=args.no_qa,
        videos_only=args.videos_only,
        max_duration=args.max_duration,
        no_subtitles=args.no_subtitles,
    )
    
    # Print result summary
    print("\n" + "=" * 50)
    print("RESULT SUMMARY")
    print("=" * 50)
    print(json.dumps(result, indent=2))
    
    # Exit code based on status
    if result["status"] == "failed":
        sys.exit(1)
    elif result["status"] == "partial":
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
