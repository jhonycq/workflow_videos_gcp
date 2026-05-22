import os
from google import genai
from google.genai import types
import numpy as np
import wave

# Initialize Vertex AI client
PROJECT_ID = "celtic-hub-492704-g2"
client = genai.Client(vertexai=True, project=PROJECT_ID, location="us-central1")

text_to_generate = "Hola, esta es una prueba de voz con el nuevo modelo."

try:
    response = client.models.generate_content(
        model="gemini-3.1-flash-tts-preview",
        contents=text_to_generate,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name='Aoede',
                    )
                )
            ),
        )
    )
    
    audio_data = response.candidates[0].content.parts[0].inline_data.data
    mime_type = response.candidates[0].content.parts[0].inline_data.mime_type
    
    print(f"Success! Mime Type: {mime_type}")
    print(f"Generated audio bytes: {len(audio_data)}")
except Exception as e:
    print(f"Error: {e}")
