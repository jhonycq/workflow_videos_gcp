import argparse
import sys
import logging
from pathlib import Path

# Add current directory to path to import local modules
sys.path.append(str(Path(__file__).parent))

from assemble import assemble_with_single_tts, assemble_final_video
from config import BUCKET_NAME
from schemas import Manifest
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="n8n Pipeline Helper")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Assemble command
    assemble_parser = subparsers.add_parser("assemble", help="Assemble video from GCS clips")
    assemble_parser.add_argument("--run_id", required=True, help="Run ID of the generation")
    assemble_parser.add_argument("--manifest_json", help="Optional JSON string of the manifest")
    assemble_parser.add_argument("--manifest_file", help="Optional path to a manifest JSON file")

    args = parser.parse_args()

    if args.command == "assemble":
        run_id = args.run_id
        
        # Try to load manifest if provided
        manifest = None
        if args.manifest_json:
            manifest_data = json.loads(args.manifest_json)
            manifest = Manifest(**manifest_data)
        elif args.manifest_file:
            with open(args.manifest_file, 'r') as f:
                manifest_data = json.load(f)
                manifest = Manifest(**manifest_data)
        
        if manifest:
            logger.info(f"Assembling with manifest for run_id: {run_id}")
            narration_texts = [s.narration_text for s in manifest.scenes]
            scene_durations = [float(s.seconds) for s in manifest.scenes]
            tts_emotions = [s.tts_emotion for s in manifest.scenes]
            tts_paces = [s.tts_pace for s in manifest.scenes]
            tts_intensities = [s.tts_intensity for s in manifest.scenes]
            
            final_path = assemble_with_single_tts(
                run_id=run_id,
                narration_texts=narration_texts,
                scene_durations=scene_durations,
                tts_emotions=tts_emotions,
                tts_paces=tts_paces,
                tts_intensities=tts_intensities,
                content_type=manifest.content_type,
                narrator_archetype=manifest.narrator_archetype,
                narrator_voice=manifest.narrator_voice,
                bucket_name=BUCKET_NAME,
                upload_to_gcs=True
            )
        else:
            logger.info(f"Assembling without manifest for run_id: {run_id} (standard concat)")
            final_path = assemble_final_video(run_id, bucket_name=BUCKET_NAME, upload_to_gcs=True)
            
        print(f"SUCCESS: {final_path}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
