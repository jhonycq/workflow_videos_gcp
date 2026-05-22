#!/usr/bin/env python3
"""
Standalone QA Verification Script.

Runs visual QA on images from a previous run to verify quality
without regenerating images in the same pipeline.

Usage:
    python qa_standalone.py --run-id 20260207_161734
    python qa_standalone.py --run-id 20260207_161734 --fix  # Regenerate failed images
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from config import OUTPUT_DIR, SCENES_DIR
import config
from schemas import Manifest
from gen_qa import verify_image_vision
from gen_image import generate_scene_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def find_scene_image(run_id: str, scene_idx: int) -> Tuple[Path, bytes]:
    """Find and load an image for a scene."""
    potential_paths = [
        SCENES_DIR / run_id / f"{scene_idx:02d}" / "image.png",
        SCENES_DIR / run_id / f"{scene_idx:02d}" / "image.jpg",
        OUTPUT_DIR / run_id / f"scene_{scene_idx:02d}.png",
        OUTPUT_DIR / run_id / f"scene_{scene_idx:02d}.jpg",
    ]
    
    for img_path in potential_paths:
        if img_path.exists():
            with open(img_path, "rb") as f:
                return img_path, f.read()
    
    raise FileNotFoundError(f"No image found for scene {scene_idx}")


def run_qa_verification(run_id: str, fix: bool = False) -> Dict[int, dict]:
    """
    Run QA verification on all images from a run.
    
    Returns:
        Dict mapping scene_idx to QA results
    """
    # Load manifest
    manifest_path = OUTPUT_DIR / run_id / "manifest.json"
    if not manifest_path.exists():
        logger.error(f"Manifest not found at {manifest_path}")
        sys.exit(1)
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest_data = json.load(f)
        manifest = Manifest(**manifest_data)
    
    logger.info(f"Loaded manifest: {manifest.title}")
    logger.info(f"Scenes to verify: {len(manifest.scenes)}")
    
    results = {}
    passed = 0
    failed = 0
    missing = 0
    
    for scene in manifest.scenes:
        logger.info(f"\n[Scene {scene.idx}] Verifying image...")
        
        should_fix = False
        image_bytes = None
        img_path = None
        
        try:
            img_path, image_bytes = find_scene_image(run_id, scene.idx)
            logger.info(f"  Image found: {img_path.name}")
        except FileNotFoundError:
            logger.warning(f"  ⚠️ MISSING: No image found for scene {scene.idx}")
            if fix:
                should_fix = True
            else:
                missing += 1
                results[scene.idx] = {
                    "passed": False,
                    "status": "missing",
                    "image_path": None
                }
                continue

        # Get style info for QA context
        style_name = "custom"  # GlobalStyle object doesn't have a name field
        style_notes = manifest.global_style.rendering_notes if hasattr(manifest.global_style, 'rendering_notes') else str(manifest.global_style)
        
        if not should_fix:
            # Run QA verification on existing image
            is_valid, fail_reason, recommended_fix = verify_image_vision(
                image_bytes=image_bytes,
                prompt=scene.image_prompt_en,
                visual_context=scene.visual_context or "",
                style_name=style_name,
                style_description=style_notes
            )
            
            if is_valid:
                logger.info(f"  ✅ PASSED")
                passed += 1
                results[scene.idx] = {
                    "passed": True,
                    "image_path": str(img_path),
                    "status": "passed"
                }
            else:
                logger.warning(f"  ❌ FAILED: {fail_reason}")
                if fix:
                    should_fix = True
                else:
                    failed += 1
                    results[scene.idx] = {
                        "passed": False,
                        "image_path": str(img_path),
                        "status": "failed",
                        "fail_reason": fail_reason,
                        "recommended_fix": recommended_fix
                    }

        if should_fix:
            logger.info(f"  🛠️ [FIX] Regenerating image for scene {scene.idx}...")
            try:
                # Get consistency anchor if available (Scene 1)
                reference_image_bytes = None
                if scene.idx > 1:
                    try:
                        _, reference_image_bytes = find_scene_image(run_id, 1)
                    except:
                        pass

                # Disable QA during regeneration as we will manually verify it next
                import config as cfg
                original_qa_val = cfg.USE_VISUAL_QA
                cfg.USE_VISUAL_QA = False
                
                new_image_bytes, local_path, final_prompt = generate_scene_image(
                    scene_idx=scene.idx,
                    image_prompt=scene.image_prompt_en,
                    image_avoid=scene.image_avoid or "",
                    global_style=manifest.global_style.model_dump(),
                    output_dir=SCENES_DIR, 
                    run_id=run_id,
                    style_id=manifest.style_id or config.DEFAULT_STYLE,
                    visual_context=scene.visual_context or "",
                    reference_image_bytes=reference_image_bytes
                )
                
                # Restore QA setting
                cfg.USE_VISUAL_QA = original_qa_val
                
                # Re-verify after fix
                is_valid, fail_reason, recommended_fix = verify_image_vision(
                    image_bytes=new_image_bytes,
                    prompt=scene.image_prompt_en,
                    visual_context=scene.visual_context or "",
                    style_name=style_name,
                    style_description=style_notes
                )
                
                if is_valid:
                    logger.info(f"  ✅ [FIX] Successfully fixed scene {scene.idx}!")
                    passed += 1
                else:
                    logger.error(f"  ❌ [FIX] Regeneration for scene {scene.idx} also failed QA: {fail_reason}")
                    failed += 1
                
                results[scene.idx] = {
                    "passed": is_valid,
                    "image_path": str(local_path),
                    "status": "fixed" if is_valid else "fix_failed",
                    "fail_reason": fail_reason if not is_valid else None
                }
            except Exception as e:
                logger.error(f"  ❌ [FIX] Critical error during regeneration of scene {scene.idx}: {e}")
                failed += 1
                results[scene.idx] = {"passed": False, "status": "fix_error", "error": str(e)}

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("QA VERIFICATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total scenes: {len(manifest.scenes)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Missing: {missing}")

    if failed > 0 or missing > 0:
        incomplete = [idx for idx, res in results.items() if not res["passed"]]
        logger.info(f"\nIncomplete scenes: {incomplete}")
        logger.info(f"To manually fix: python main.py --resume {run_id} --regenerate-scenes {','.join(map(str, incomplete))}")

    # Save report
    report_path = OUTPUT_DIR / run_id / "qa_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            "run_id": run_id,
            "title": manifest.title,
            "summary": {
                "total": len(manifest.scenes),
                "passed": passed,
                "failed": failed,
                "missing": missing
            },
            "scenes": results
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nReport saved: {report_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Standalone QA verification for generated images"
    )
    
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID to verify (e.g., 20260207_161734)"
    )
    
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Regenerate failed or missing images after verification"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("STANDALONE QA VERIFICATION")
    logger.info("=" * 50)
    logger.info(f"Run ID: {args.run_id}")
    if args.fix:
        logger.info("🛠️ Auto-Fix mode ENABLED")
    
    results = run_qa_verification(args.run_id, fix=args.fix)
    
    # Exit code based on results
    all_passed = all(r["passed"] for r in results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
