"""
Test script for Google Cloud Text-to-Speech setup.
"""
import logging
from google.cloud import texttospeech
from config import NARRATOR_VOICE_ID, PROJECT_ID, TTS_MODEL

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

def test_tts():
    logger.info("🧪 Testing Google Cloud TTS API connection...")
    try:
        client = texttospeech.TextToSpeechClient(client_options={"quota_project_id": PROJECT_ID})
        input_text = texttospeech.SynthesisInput(text="Probando la voz en español de Google Cloud.")
        voice = texttospeech.VoiceSelectionParams(
            language_code="es-US",
            name=NARRATOR_VOICE_ID,
            model_name=TTS_MODEL
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        )
        response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
        if response.audio_content:
            logger.info(f"  ✓ API connection successful! Audio generated with {NARRATOR_VOICE_ID} ({TTS_MODEL}).")
            return True
        else:
            logger.error("  ✗ Failed to generate audio.")
            return False
    except Exception as e:
        logger.error(f"  ✗ API connection failed: {e}")
        return False

if __name__ == "__main__":
    test_tts()
