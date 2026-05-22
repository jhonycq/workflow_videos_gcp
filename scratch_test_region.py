from google.cloud import texttospeech
import os

os.environ["GOOGLE_CLOUD_PROJECT"] = "celtic-hub-492704-g2"
client = texttospeech.TextToSpeechClient(client_options={"quota_project_id": "celtic-hub-492704-g2"})

input_text = texttospeech.SynthesisInput(text="Hola, soy un zorro curioso.")
voice = texttospeech.VoiceSelectionParams(
    language_code="es-US",
    name="Aoede",
    model_name="gemini-2.5-flash-tts"
)
audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
try:
    response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
    print("SUCCESS: Spanish generation worked!")
except Exception as e:
    print(f"FAILED: {e}")
