import json
import logging
from pathlib import Path
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from config import PROJECT_ID, TEXT_LOCATION, TEXT_MODEL

logger = logging.getLogger(__name__)

def transcribe_with_timestamps(audio_path: Path, expected_segments: int = None, narration_hints: list[str] = None) -> list[dict]:
    """
    Uses Gemini (via Vertex AI) to transcribe the audio and return precise timestamps.
    
    Args:
        audio_path: Path to the WAV audio file.
        expected_segments: If provided, instructs the model to produce exactly this many segments.
        narration_hints: Optional list of narration texts (one per scene) to help alignment.
    
    Returns:
        A list of dicts with "text", "start_time", and "end_time" keys.
    """
    vertexai.init(project=PROJECT_ID, location=TEXT_LOCATION)
    model = GenerativeModel(TEXT_MODEL)
    
    logger.info(f"Transcribing {audio_path} for scene duration calculations...")
    
    with open(audio_path, "rb") as f:
        audio_data = f.read()
        
    audio_part = Part.from_data(data=audio_data, mime_type="audio/wav")
    
    # Build a dynamic prompt based on whether we know the expected segment count
    if expected_segments and narration_hints:
        hints_block = "\n".join([f"  Segment {i+1}: \"{h[:80]}...\"" for i, h in enumerate(narration_hints)])
        prompt = f"""
You are a precise audio transcription assistant. This audio contains a storytelling narration 
that was generated from exactly {expected_segments} text segments spoken continuously.

Here are the FIRST WORDS of each segment to help you identify boundaries:
{hints_block}

Your task: Find the exact start and end timestamps for each of these {expected_segments} segments.

Return ONLY a JSON array with exactly {expected_segments} objects. Each object must have:
1. "text": a brief summary of the spoken words in that segment
2. "start_time": the start time in seconds (float, precision to 0.1)
3. "end_time": the end time in seconds (float, precision to 0.1)

Rules:
- You MUST return exactly {expected_segments} segments, no more, no less.
- Segments must be contiguous (end_time of segment N = start_time of segment N+1).
- The first segment must start at 0.0.
- Do not include any Markdown formatting or code blocks. Just the raw JSON array.
"""
    else:
        prompt = """
Please transcribe this storytelling audio. Since this audio will be split into video scenes, 
I need you to break the transcript down into sentences or logical pauses. 

Return ONLY a JSON array where each object has:
1. "text": the spoken words for that chunk
2. "start_time": the start time in seconds (float)
3. "end_time": the end time in seconds (float)

Do not include any Markdown formatting or code blocks. Just the raw JSON array.
"""
    
    response = model.generate_content(
        [audio_part, prompt],
        generation_config={"temperature": 0.0}
    )
    
    text = response.text.replace("```json", "").replace("```", "").strip()
    result = json.loads(text)
    
    logger.info(f"Transcription returned {len(result)} segments")
    return result
