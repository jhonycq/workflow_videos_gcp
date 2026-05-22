"""
Automatic Subtitle Generator with Active Word Highlighting.
Shows all words but highlights the currently spoken word with darker background.
Uses Faster-Whisper for transcription and ASS format for animated subtitles.

Usage:
    python add_subtitles.py <run_id>
"""

import argparse
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

WHISPER_MODEL = "medium"
from config import OUTPUT_DIR
BASE_DIR = OUTPUT_DIR

def get_video_dimensions(video_path: Path) -> tuple:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    width, height = map(int, result.stdout.strip().split(','))
    logger.info(f"Video dimensions: {width}x{height}")
    return width, height


def calculate_subtitle_position(width: int, height: int) -> dict:
    margin_bottom = int(height * 0.22)
    margin_right = int(width * 0.10)
    margin_left = int(width * 0.05)
    font_size = max(16, int(height * 0.03))
    return {
        "margin_bottom": margin_bottom,
        "margin_left": margin_left,
        "margin_right": margin_right,
        "font_size": font_size,
    }


def format_ass_time(seconds: float) -> str:
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    secs = int(seconds % 60)
    centis = int((seconds - int(seconds)) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def extract_audio(video_path: Path, audio_path: Path) -> bool:
    logger.info(f"Extracting audio...")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(audio_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def transcribe_audio_with_words(audio_path: Path, language: str = "es") -> list:
    from faster_whisper import WhisperModel
    
    logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    
    logger.info(f"Transcribing with word timestamps...")
    segments, info = model.transcribe(
        str(audio_path), language=language,
        word_timestamps=True, vad_filter=True,
    )
    
    result = []
    for segment in segments:
        words = []
        if segment.words:
            for word in segment.words:
                words.append({
                    "word": word.word.strip(),
                    "start": word.start,
                    "end": word.end
                })
        result.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "words": words
        })
    
    total_words = sum(len(seg["words"]) for seg in result)
    logger.info(f"Transcription: {len(result)} segments, {total_words} words")
    return result


def generate_highlight_ass(segments: list, output_path: Path, width: int, height: int) -> bool:
    """
    Generate ASS with word-by-word highlighting.
    Shows entire sentence, but the active word has a darker/highlighted background.
    
    Approach: For each word timing, show the full sentence with that word highlighted.
    """
    pos = calculate_subtitle_position(width, height)
    
    # Colors (ASS format: &HAABBGGRR - Alpha, Blue, Green, Red)
    # Normal: white text, semi-transparent dark background
    # Highlighted: yellow text, more opaque dark background
    
    ass_header = f"""[Script Info]
Title: Word-highlight subtitles
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Normal,Arial,{pos['font_size']},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,4,1,0,2,{pos['margin_left']},{pos['margin_right']},{pos['margin_bottom']},1
Style: Highlight,Arial,{pos['font_size']},&H0000FFFF,&H000000FF,&H00000000,&HFF000000,1,0,0,0,100,100,0,0,4,2,0,2,{pos['margin_left']},{pos['margin_right']},{pos['margin_bottom']},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    ass_lines = [ass_header]
    
    for seg in segments:
        if not seg["words"]:
            # No word timestamps, show full sentence
            start = format_ass_time(seg['start'])
            end = format_ass_time(seg['end'])
            ass_lines.append(f"Dialogue: 0,{start},{end},Normal,,0,0,0,,{seg['text']}")
            continue
        
        words = seg["words"]
        
        # For each word, create a dialogue line showing the full sentence
        # with inline override to highlight just that word
        for i, active_word in enumerate(words):
            start = format_ass_time(active_word['start'])
            end = format_ass_time(active_word['end'])
            
            # Build the sentence with the active word highlighted
            # Use inline style overrides: {\c&H0000FFFF&\3c&H00000000&\4c&HFF000000&} for highlight
            sentence_parts = []
            for j, w in enumerate(words):
                if j == i:
                    # Highlighted word: yellow text with darker background
                    sentence_parts.append(f"{{\\c&H0000FFFF&\\b1}}{w['word']}{{\\c&H00FFFFFF&\\b1}}")
                else:
                    # Normal word: white text
                    sentence_parts.append(w['word'])
            
            full_text = " ".join(sentence_parts)
            ass_lines.append(f"Dialogue: 0,{start},{end},Normal,,0,0,0,,{full_text}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(ass_lines))
    
    logger.info(f"ASS file with word highlighting: {output_path}")
    return True


def burn_subtitles(video_path: Path, ass_path: Path, output_path: Path) -> bool:
    logger.info(f"Burning subtitles...")
    ass_escaped = str(ass_path).replace(":", "\\:").replace("'", "\\'")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"ass='{ass_escaped}'",
        "-c:a", "copy", "-c:v", "libx264", "-crf", "23", "-preset", "medium",
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"FFmpeg failed: {result.stderr}")
        return False
    logger.info(f"Video created: {output_path}")
    return True


def add_subtitles(run_id: str, language: str = "es") -> Path:
    run_dir = BASE_DIR / run_id
    
    input_video = run_dir / "final_with_narration.mp4"
    if not input_video.exists():
        input_video = run_dir / "final.mp4"
    
    if not input_video.exists():
        raise FileNotFoundError(f"No video found in {run_dir}")
    
    width, height = get_video_dimensions(input_video)
    
    audio_path = run_dir / "audio_for_transcription.wav"
    ass_path = run_dir / "subtitles_highlight.ass"
    output_video = run_dir / "final_with_subtitles.mp4"
    
    if not extract_audio(input_video, audio_path):
        raise RuntimeError("Audio extraction failed")
    
    segments = transcribe_audio_with_words(audio_path, language)
    
    if not segments:
        raise RuntimeError("Transcription returned no segments")
    
    generate_highlight_ass(segments, ass_path, width, height)
    
    if not burn_subtitles(input_video, ass_path, output_video):
        raise RuntimeError("Subtitle burning failed")
    
    audio_path.unlink(missing_ok=True)
    
    logger.info(f"✅ Done! {output_video}")
    return output_video


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add subtitles with word highlighting.")
    parser.add_argument("run_id", help="The run ID of the video")
    parser.add_argument("--language", "-l", default="es")
    parser.add_argument("--model", "-m", default="medium")
    
    args = parser.parse_args()
    if args.model != "medium":
        WHISPER_MODEL = args.model
    
    try:
        output = add_subtitles(args.run_id, args.language)
        print(f"\n🎬 Video ready: {output}")
    except Exception as e:
        logger.error(f"Failed: {e}")
        exit(1)
