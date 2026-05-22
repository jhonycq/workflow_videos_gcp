#!/usr/bin/env python3
"""
Test script to validate ElevenLabs v3 setup and configuration.
Runs before any video generation to catch configuration issues early.

Usage: python test_elevenlabs_setup.py
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required packages are installed."""
    logger.info("🧪 Testing imports...")
    try:
        from elevenlabs.client import ElevenLabs
        from elevenlabs import VoiceSettings
        logger.info("  ✓ elevenlabs SDK installed")
        return True
    except ImportError as e:
        logger.error(f"  ✗ elevenlabs not installed: {e}")
        logger.error("     Run: pip install elevenlabs>=1.0.0")
        return False


def test_config():
    """Test that config.py is properly set up."""
    logger.info("🧪 Testing config.py...")
    try:
        from config import (
            ELEVENLABS_API_KEY,
            ELEVENLABS_MODEL,
            NARRATOR_VOICE_ID,
            CHILD_VOICE_ID,
        )

        checks = [
            ("ELEVENLABS_API_KEY", ELEVENLABS_API_KEY, "api_key", None),
            ("ELEVENLABS_MODEL", ELEVENLABS_MODEL, "expected", "eleven_v3"),
            ("NARRATOR_VOICE_ID", NARRATOR_VOICE_ID, "voice_id", None),
            ("CHILD_VOICE_ID", CHILD_VOICE_ID, "voice_id", None),
        ]

        all_ok = True
        for name, value, check_type, expected_val in checks:
            if check_type == "api_key":
                if not value or len(value) < 10:
                    logger.error(f"  ✗ {name}: Not set or too short")
                    all_ok = False
                else:
                    logger.info(f"  ✓ {name}: {value[:10]}...{'*' * 20}")
            elif check_type == "voice_id":
                if not value or len(value) < 10:
                    logger.error(f"  ✗ {name}: Not set")
                    all_ok = False
                else:
                    logger.info(f"  ✓ {name}: {value}")
            elif check_type == "expected":
                if value == expected_val:
                    logger.info(f"  ✓ {name}: {value}")
                else:
                    logger.error(f"  ✗ {name}: Expected {expected_val}, got {value}")
                    all_ok = False

        return all_ok

    except Exception as e:
        logger.error(f"  ✗ Failed to import config: {e}")
        return False


def test_api_connection():
    """Test actual connection to ElevenLabs API by generating a tiny TTS sample."""
    logger.info("🧪 Testing ElevenLabs API connection...")
    try:
        from elevenlabs.client import ElevenLabs
        from elevenlabs import VoiceSettings
        from config import ELEVENLABS_API_KEY, NARRATOR_VOICE_ID, ELEVENLABS_MODEL

        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

        # Test TTS endpoint directly (voices_read not required)
        audio_iter = client.text_to_speech.convert(
            voice_id=NARRATOR_VOICE_ID,
            text="Hola.",
            model_id=ELEVENLABS_MODEL,
            output_format="pcm_24000",
            voice_settings=VoiceSettings(
                stability=0.70,
                similarity_boost=0.75,
                style=0.15,
                use_speaker_boost=True,
            ),
        )
        # Consume a few chunks to confirm the connection works
        chunks = []
        for chunk in audio_iter:
            chunks.append(chunk)
            if sum(len(c) for c in chunks) > 1000:
                break

        logger.info(f"  ✓ Connected to ElevenLabs API (TTS endpoint)")
        logger.info(f"  ✓ Narrator voice ({NARRATOR_VOICE_ID}) is accessible")
        logger.info(f"  ✓ Received {sum(len(c) for c in chunks)} bytes of PCM audio")
        return True

    except Exception as e:
        logger.error(f"  ✗ API connection failed: {e}")
        logger.error("     Check your ELEVENLABS_API_KEY and NARRATOR_VOICE_ID")
        return False


def test_gen_tts_module():
    """Test that gen_tts.py loads correctly."""
    logger.info("🧪 Testing gen_tts module...")
    try:
        from gen_tts import (
            generate_dialogue_audio,
            generate_continuous_narration,
            get_audio_duration,
            paraphrase_text,
            _get_voice_settings,
            EMOTION_TO_VOICE_SETTINGS,
        )

        logger.info(f"  ✓ gen_tts module loaded")
        logger.info(f"  ✓ Found {len(EMOTION_TO_VOICE_SETTINGS)} emotion profiles")

        # Check emotion profiles have required keys
        required_keys = {"stability", "similarity_boost", "style"}
        for emotion, settings in EMOTION_TO_VOICE_SETTINGS.items():
            if not all(key in settings for key in required_keys):
                logger.error(f"  ✗ Emotion '{emotion}' missing required keys: {required_keys}")
                return False

        logger.info(f"  ✓ All emotion profiles valid")
        return True

    except Exception as e:
        logger.error(f"  ✗ gen_tts import failed: {e}")
        return False


def test_voice_settings():
    """Test voice settings generation for different emotions."""
    logger.info("🧪 Testing voice settings generation...")
    try:
        from gen_tts import _get_voice_settings, EMOTION_TO_VOICE_SETTINGS

        test_emotions = ["neutral", "fear", "excited", "calm"]

        for emotion in test_emotions:
            if emotion not in EMOTION_TO_VOICE_SETTINGS:
                logger.error(f"  ✗ Missing emotion: {emotion}")
                return False

            # Test narrator voice
            vs_narrator = _get_voice_settings(emotion, intensity=0.5, is_child=False)
            assert 0.0 <= vs_narrator.stability <= 1.0, "Invalid stability"
            assert 0.0 <= vs_narrator.similarity_boost <= 1.0, "Invalid similarity_boost"
            assert 0.0 <= vs_narrator.style <= 1.0, "Invalid style"
            assert vs_narrator.use_speaker_boost == True, "Speaker boost should be enabled"

            # Test child voice
            vs_child = _get_voice_settings(emotion, intensity=0.5, is_child=True)
            assert vs_child.stability < vs_narrator.stability, "Child should be less stable"
            assert vs_child.style > vs_narrator.style, "Child should have more style"

            logger.info(
                f"  ✓ {emotion:12} → Narrator: s={vs_narrator.stability:.2f} "
                f"sim={vs_narrator.similarity_boost:.2f} "
                f"st={vs_narrator.style:.2f} | "
                f"Child: s={vs_child.stability:.2f} st={vs_child.style:.2f}"
            )

        return True

    except Exception as e:
        logger.error(f"  ✗ Voice settings test failed: {e}")
        return False


def test_ffmpeg():
    """Test that ffprobe is available (needed for audio duration)."""
    logger.info("🧪 Testing ffprobe availability...")
    try:
        import subprocess
        result = subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            logger.info("  ✓ ffprobe found and working")
            return True
        else:
            logger.error("  ✗ ffprobe error")
            return False
    except Exception as e:
        logger.error(f"  ✗ ffprobe not found: {e}")
        logger.error("     Install FFmpeg: brew install ffmpeg")
        return False


def test_schemas():
    """Test that schemas.py has V3 structures."""
    logger.info("🧪 Testing schemas (V3 structures)...")
    try:
        from schemas import DialogueLine, Scene, Manifest

        # Test DialogueLine
        line = DialogueLine(
            speaker="narrator",
            text="Test text",
            emotion="neutral",
            intensity=0.5
        )
        logger.info("  ✓ DialogueLine model works")

        # Test Scene with dialogue
        scene = Scene(
            idx=1,
            image_prompt_en="A test image" * 20,  # 20+ words
            image_avoid="ugly, blurry",
            video_prompt_en="A test video motion" * 10,  # 100+ chars
            dialogue=[line],
        )
        logger.info("  ✓ Scene with dialogue works")

        # Test Manifest
        from schemas import GlobalStyle
        manifest = Manifest(
            title="Test Story",
            global_style=GlobalStyle(
                lut="cinematic",
                palette="warm tones",
                rules="consistent lighting"
            ),
            scenes=[scene],
            content_type="general",
            narrator_archetype="warm storyteller",
            child_archetype="curious child"
        )
        logger.info("  ✓ Manifest with dual-narrator works")

        return True

    except Exception as e:
        logger.error(f"  ✗ Schemas test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 70)
    logger.info("🚀 ElevenLabs V3 Setup Validation")
    logger.info("=" * 70)
    logger.info("")

    tests = [
        ("Imports", test_imports),
        ("Config File", test_config),
        ("Schemas (V3)", test_schemas),
        ("Voice Settings", test_voice_settings),
        ("Gen TTS Module", test_gen_tts_module),
        ("FFmpeg/FFprobe", test_ffmpeg),
        ("ElevenLabs API", test_api_connection),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            logger.error(f"❌ {name} test crashed: {e}")
            results.append((name, False))
        logger.info("")

    # Summary
    logger.info("=" * 70)
    logger.info("📊 Test Summary")
    logger.info("=" * 70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {status:8} | {name}")

    logger.info(f"\n  Result: {passed}/{total} tests passed")
    logger.info("=" * 70)

    if passed == total:
        logger.info("✅ All tests passed! Ready to run V3 pipeline.")
        logger.info("\nNext steps:")
        logger.info("  1. python main.py --theme 'your theme' --dry-run --no-qa")
        logger.info("  2. Check out/{run_id}/manifest.json for dialogue")
        logger.info("  3. python main.py --theme 'your theme' --scenes 2 --no-qa")
        return 0
    else:
        logger.error("❌ Some tests failed. Fix issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
