"""
Script to add TTS narration to Cheonyeogwishin ghost video.
"""
import logging
from pathlib import Path
from assemble import assemble_with_tts

logging.basicConfig(level=logging.INFO)

# Run ID from the generated video
run_id = "20251223_011305"

# Scene narrations in Spanish (Cheonyeogwishin ghost story)
# Using phonetic pronunciations: Ieiu, Cheon-yeo-gwi-shin
narration_texts = [
    # Scene 1: Fisherman sees something
    "En los bosques oscuros de Ieiu, un viejo pescador camina solo. A lo lejos, una figura blanca lo observa en silencio.",
    
    # Scene 2: Ghost approaches
    "La Cheon-yeo-gwi-shin se acerca flotando. Su cabello negro ondula como si estuviera bajo el agua. Sus ojos... están vacíos.",
    
    # Scene 3: Confrontation
    "El pescador intenta huir, pero el fantasma aparece frente a él. Los ojos blancos de la doncella miran directamente su alma.",
    
    # Scene 4: Dawn - petals
    "Con el amanecer, la Cheon-yeo-gwi-shin desaparece. Solo quedan pétalos de cerezo donde ella estaba. Su espíritu aún busca justicia.",
]

# 8 seconds per scene
scene_durations = [8.0, 8.0, 8.0, 8.0]

print("=" * 60)
print(f"Adding TTS narration to run: {run_id}")
print("=" * 60)

for i, text in enumerate(narration_texts, 1):
    print(f"\nScene {i}: {text}")

print("\n" + "=" * 60)

# Run assembly with TTS
final_video = assemble_with_tts(
    run_id=run_id,
    narration_texts=narration_texts,
    scene_durations=scene_durations,
    upload_to_gcs=True
)

print(f"\n✅ Final video with TTS: {final_video}")
