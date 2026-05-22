#!/usr/bin/env python3
"""
Quick audio-only test script.
Generates story + TTS audio without images or video.

Usage:
    python gen_audio_test.py --theme "la liebre y el conejo"
    python gen_audio_test.py --theme "..." --max-duration 90
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate story + audio only (no images/video)")
    parser.add_argument("--theme", type=str, required=True, help="Story theme")
    parser.add_argument("--max-duration", type=int, default=None, help="Max audio duration in seconds")
    parser.add_argument("--style", type=str, default="ultra_real", help="Style preset")
    args = parser.parse_args()

    run_id = datetime.now().strftime("audio_%Y%m%d_%H%M%S")
    from config import OUTPUT_DIR
    out_dir = OUTPUT_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate story
    logger.info(f"\n{'='*60}")
    logger.info(f"GENERATING STORY: {args.theme}")
    logger.info(f"{'='*60}")

    from llm_story import generate_story
    manifest = generate_story(
        theme=args.theme,
        num_scenes=None,
        style_id=args.style,
        max_duration=args.max_duration,
    )

    # Save manifest for inspection
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(), f, ensure_ascii=False, indent=2)
    logger.info(f"Manifest saved: {manifest_path}")

    # Print dialogue preview
    logger.info(f"\n{'='*60}")
    logger.info(f"STORY: {manifest.title}")
    logger.info(f"Scenes: {len(manifest.scenes)} | Duration: {manifest.total_duration_seconds}s")
    logger.info(f"{'='*60}")
    for i, scene in enumerate(manifest.scenes, 1):
        logger.info(f"\n--- Scene {i} ---")
        for line in scene.dialogue:
            speaker = "NORAH" if line.speaker == "narrator" else "DANIELA"
            logger.info(f"  [{speaker}] {line.text}")

    # Step 2: Generate audio
    logger.info(f"\n{'='*60}")
    logger.info("GENERATING AUDIO...")
    logger.info(f"{'='*60}")

    from gen_tts import generate_dialogue_audio
    audio_path, duration = generate_dialogue_audio(
        scenes=manifest.scenes,
        output_path=out_dir / "narration.wav",
        content_type=getattr(manifest, "content_type", "general"),
    )

    if audio_path:
        logger.info(f"\n✅ Audio ready: {audio_path}")
        logger.info(f"   Duration: {duration:.2f}s")
    else:
        logger.error("❌ Audio generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
