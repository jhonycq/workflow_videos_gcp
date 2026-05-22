"""
Reassemble Video with New Scene 4 and Regenerated Scene 2 Audio
"""
import json
import subprocess
from pathlib import Path

from config import OUTPUT_DIR, BUCKET_NAME
from gen_tts import generate_single_audio, get_audio_duration
from storage import download_file

RUN_ID = "20260124_152119"
RUN_DIR = OUTPUT_DIR / RUN_ID
SCENES_DIR = OUTPUT_DIR / "scenes" / RUN_ID
MANIFEST_PATH = RUN_DIR / "manifest.json"
TEMP_AUDIO_DIR = RUN_DIR / "temp_audio_fixed"

def reassemble():
    print("=== REASSEMBLY SCRIPT ===")
    
    # 1. Download new Scene 4 video
    print("[1/5] Downloading new Scene 4 video from GCS...")
    new_scene_4_gcs = "gs://mvp-scenes1/mvp/20260124_152119/scenes/04/video.mp4/3276426880966873965/sample_0.mp4"
    local_scene_4 = SCENES_DIR / "04_video.mp4"
    
    try:
        download_file(new_scene_4_gcs, local_scene_4, BUCKET_NAME)
        print(f"Downloaded to: {local_scene_4}")
    except Exception as e:
        print(f"ERROR downloading Scene 4: {e}")
        return
    print(f"Downloaded to: {local_scene_4}")

    # 2. Regenerate Scene 2 audio with proper pacing
    print("[2/5] Regenerating Scene 2 audio...")
    
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    
    scene_2 = next(s for s in manifest["scenes"] if s["idx"] == 2)
    
    # Scene 2 has 6 seconds duration
    scene_2_audio = TEMP_AUDIO_DIR / "scene_02_fixed.wav"
    
    generate_single_audio(
        narration_texts=[scene_2["narration_text"]],
        tts_emotions=["calm"],
        tts_paces=["slow"],
        tts_intensities=[0.5],
        total_duration=6.0,
        output_path=scene_2_audio,
        content_type=manifest.get("content_type", "fable"),
        narrator_voice=manifest.get("narrator_voice", "storyteller")
    )
    print(f"Regenerated Scene 2 audio: {scene_2_audio}")
    
    # 3. Re-concatenate all videos
    print("[3/5] Concatenating all videos...")
    video_segments = []
    for i in range(1, 17):
        src_vid = SCENES_DIR / f"{i:02d}_video.mp4"
        silent_vid = RUN_DIR / f"scene_{i:02d}_silent_v2.mp4"
        
        # Strip audio
        subprocess.run([
            "ffmpeg", "-y", "-i", str(src_vid),
            "-an", "-c:v", "copy", str(silent_vid)
        ], capture_output=True, check=True)
        video_segments.append(silent_vid)
    
    concat_video_list = RUN_DIR / "concat_video_v2.txt"
    with open(concat_video_list, "w") as f:
        for p in video_segments:
            f.write(f"file '{p.resolve()}'\n")
    
    final_video_silent = RUN_DIR / "video_silent_v2.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_video_list),
        "-c", "copy", str(final_video_silent)
    ], check=True)
    print(f"Concatenated video: {final_video_silent}")
    
    # 4. Re-concatenate all audio
    print("[4/5] Concatenating all audio segments...")
    audio_segments = []
    for i in range(1, 17):
        p = TEMP_AUDIO_DIR / f"scene_{i:02d}_fixed.wav"
        if not p.exists():
            print(f"ERROR: Missing audio for scene {i}")
            return
        audio_segments.append(p)
    
    concat_audio_list = RUN_DIR / "concat_audio_v2.txt"
    with open(concat_audio_list, "w") as f:
        for p in audio_segments:
            f.write(f"file '{p.resolve()}'\n")
    
    final_audio = RUN_DIR / "narration_v2.wav"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_audio_list),
        "-c:a", "pcm_s16le", str(final_audio)
    ], check=True)
    
    audio_dur = get_audio_duration(str(final_audio))
    print(f"Concatenated audio: {final_audio} ({audio_dur:.2f}s)")
    
    # 5. Final mux
    print("[5/5] Muxing final video...")
    final_output = RUN_DIR / "final_v2.mp4"
    
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(final_video_silent),
        "-i", str(final_audio),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        str(final_output)
    ], check=True)
    
    print(f"\n=== SUCCESS ===")
    print(f"Final video saved: {final_output}")

if __name__ == "__main__":
    reassemble()
