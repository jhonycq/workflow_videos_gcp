import json
import logging
from pathlib import Path
from assemble import assemble_with_single_tts
from config import BUCKET_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def resume_run(run_id: str):
    manifest_path = Path(f"out/{run_id}/manifest.json")
    if not manifest_path.exists():
        logger.error(f"Manifest not found for run {run_id}")
        return

    with open(manifest_path, "r") as f:
        data = json.load(f)
    
    scenes = data["scenes"]
    narration_texts = [s["narration_text"] for s in scenes]
    scene_durations = [float(s["seconds"]) for s in scenes]
    tts_emotions = [s.get("tts_emotion", "wonder") for s in scenes]
    tts_paces = [s.get("tts_pace", "normal") for s in scenes]
    tts_intensities = [s.get("tts_intensity", 0.8) for s in scenes]
    
    content_type = data.get("content_type", "fable")
    narrator_archetype = data.get("narrator_archetype", "")
    narrator_voice = data.get("narrator_voice", "storyteller")

    logger.info(f"Resuming assembly for run: {run_id}")
    
    final_video = assemble_with_single_tts(
        run_id=run_id,
        narration_texts=narration_texts,
        scene_durations=scene_durations,
        tts_emotions=tts_emotions,
        tts_paces=tts_paces,
        tts_intensities=tts_intensities,
        content_type=content_type,
        narrator_archetype=narrator_archetype,
        narrator_voice=narrator_voice,
        bucket_name=BUCKET_NAME,
        upload_to_gcs=True,
        reuse_existing_audio=True
    )
    
    logger.info(f"DONE! Final video at: {final_video}")

if __name__ == "__main__":
    resume_run("20260503_235255")
