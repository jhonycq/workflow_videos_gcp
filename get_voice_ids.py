#!/usr/bin/env python3
"""
Helper script to find and display ElevenLabs Voice IDs.
Usage: python get_voice_ids.py
"""
import sys
from elevenlabs import ElevenLabs

def get_voice_ids():
    """Fetch and display available ElevenLabs voices."""

    api_key = input("Ingresa tu ElevenLabs API Key: ").strip()
    if not api_key:
        print("❌ API Key requerida")
        sys.exit(1)

    try:
        client = ElevenLabs(api_key=api_key)
        voices = client.voices.get_all()

        print("\n" + "=" * 80)
        print("🎤 VOCES DISPONIBLES EN ELEVENLABS")
        print("=" * 80)

        norah_found = False
        daniela_found = False

        # Show all voices
        for voice in voices.voices:
            name = voice.name.lower()

            # Highlight Norah and Daniela
            if name == "norah":
                print(f"\n✓✓✓ NARRADORA ENCONTRADA: {voice.name}")
                print(f"    Voice ID: {voice.voice_id}")
                print(f"    Category: {voice.category}")
                norah_found = True

            elif name == "daniela":
                print(f"\n✓✓✓ NIÑO ENCONTRADO: {voice.name}")
                print(f"    Voice ID: {voice.voice_id}")
                print(f"    Category: {voice.category}")
                daniela_found = True

        # If not found, show all voices
        if not norah_found or not daniela_found:
            print("\n⚠️  No se encontraron Norah y/o Daniela. Voces disponibles:\n")
            for voice in voices.voices:
                print(f"  • {voice.name:30} | ID: {voice.voice_id}")

        # Show config instructions
        if norah_found and daniela_found:
            print("\n" + "=" * 80)
            print("✅ ACTUALIZA config.py:")
            print("=" * 80)
            print("\nAbre config.py y reemplaza:")
            print('  NARRATOR_VOICE_ID = os.getenv("NARRATOR_VOICE_ID", "")')
            print("  CHILD_VOICE_ID = os.getenv("CHILD_VOICE_ID", "")')
            print("\nCon:")
            norah_id = next(v.voice_id for v in voices.voices if v.name.lower() == "norah")
            daniela_id = next(v.voice_id for v in voices.voices if v.name.lower() == "daniela")
            print(f'  NARRATOR_VOICE_ID = os.getenv("NARRATOR_VOICE_ID", "{norah_id}")')
            print(f'  CHILD_VOICE_ID = os.getenv("CHILD_VOICE_ID", "{daniela_id}")')

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    get_voice_ids()
