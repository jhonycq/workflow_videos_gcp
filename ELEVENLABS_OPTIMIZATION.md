# ElevenLabs v3 Optimization Guide for Video Generator V3

## Overview

This document explains how **Video Generator V3** leverages **ElevenLabs eleven_multilingual_v3** with optimized voice settings, audio tags, and Spanish language processing based on **official ElevenLabs best practices 2026**.

**Sources:**
- [ElevenLabs Best Practices](https://elevenlabs.io/docs/overview/capabilities/text-to-speech/best-practices)
- [Eleven v3 Audio Tags](https://elevenlabs.io/blog/eleven-v3-audio-tags-expressing-emotional-context-in-speech)
- [Voice Settings Guide](https://elevenlabs-sdk.mintlify.app/speech-synthesis/voice-settings)

---

## 1. Voice Settings Optimization

### Parameters Explained

#### Stability (0.0 - 1.0)
- **What it does**: Controls consistency vs. dynamism in speech delivery
- **Lower values (0.30-0.50)**: More emotional, dynamic delivery (but may sound unstable if < 0.30)
- **Higher values (0.70-0.85)**: More consistent but risks monotonous delivery
- **Best practice**: 35-40% for narrative passages to avoid monotony

**V3 Implementation:**
```python
EMOTION_TO_VOICE_SETTINGS = {
    "calm": {"stability": 0.80},        # High = consistent, peaceful
    "neutral": {"stability": 0.70},     # Medium = balanced
    "suspense": {"stability": 0.40},    # Low = dynamic tension
    "fear": {"stability": 0.35},        # Very low = maximum expression
    "excited": {"stability": 0.32},     # Very low = high energy
}
```

#### Similarity Boost (0.0 - 1.0)
- **What it does**: Clarity and consistency of the voice
- **Recommended range**: 75-80% (sweet spot for clarity + naturalness)
- **Warning**: Values above 80% can introduce audio artifacts/distortion

**V3 Implementation:**
```python
# Always keep between 72-82% depending on emotion
"calm": {"similarity_boost": 0.72},       # Slightly softer
"fear": {"similarity_boost": 0.80},       # Higher for clarity
"excited": {"similarity_boost": 0.81},    # Maximum for intense emotion
```

#### Style Exaggeration (0.0 - 1.0)
- **What it does**: Amplifies the speaker's natural style/personality
- **Computation cost**: Higher values increase latency (computationally intensive)
- **For narration**: 10-50% is the sweet spot
- **Range:**
  - 0-10%: Minimal style (neutral, informative)
  - 10-50%: Good for storytelling (this is our range)
  - 50%+: Very expressive (intense but slower)

**V3 Implementation:**
```python
"neutral": {"style": 0.15},      # 15% minimal style
"calm": {"style": 0.10},         # 10% very subtle
"excited": {"style": 0.60},      # 60% maximum expressiveness
"fear": {"style": 0.55},         # 55% high expression
```

#### Use Speaker Boost (Boolean)
- **Always TRUE for v3**: Enables Audio Tags processing
- Allows [excited], [whispers], [gasps] tags to be interpreted
- Essential for emotional expression in Eleven v3

---

## 2. Eleven v3 Audio Tags (Emotional Context)

### What Are Audio Tags?

Audio Tags are **square brackets** that wrap words/phrases to direct emotional expression. They're a key feature of Eleven v3 that allows text-based emotional control.

**Syntax:** `[tag_name] text here`

### Supported Tags (Common for Spanish Storytelling)

#### Emotional States
```
[calm]        → Peaceful, reflective delivery
[whispers]    → Soft, intimate speaking
[nervous]     → Uncertain, anxious tone
[excited]     → High energy, enthusiastic
[frustrated]  → Annoyed, exasperated
[sorrowful]   → Sad, mournful
```

#### Reactions (Perfect for Child Voice)
```
[gasps]       → Surprised intake of breath
[laughs]      → Laughter (can repeat: [laughs] [laughs])
[sigh]        → Deep breath or sigh
[cries]       → Crying, tears
[shouts]      → Loud yelling
[gulps]       → Nervous gulp (swallowing)
```

#### Pauses (v3-Specific - NO SSML)
```
[pause]       → Medium pause (~0.5s)
[short pause] → Brief pause (~0.25s)
[long pause]  → Extended pause (~1s)
```

### Using Audio Tags in Spanish

#### Example 1: Narrator (Norah)
```spanish
"[calm] En el bosque oscuro, el viento murmuraba entre los árboles."
→ Delivers with calm, peaceful tone, natural pacing
```

#### Example 2: Child (Daniela) - Reaction
```spanish
"[gasps] ¡Escuchas eso! [nervous] ¿Qué fue ese sonido?"
→ Gasps first, then delivers nervously
```

#### Example 3: Dramatic Moment
```spanish
"[excited] ¡Mira allá! [long pause] ¿Ves esa luz?"
→ Excited exclamation, long pause for effect, then curious question
```

### V3 Implementation

In `gen_tts.py`, the `_enhance_text_with_audio_tags()` function:

1. **Maps emotion to tag:**
   ```python
   emotion = "fear" → audio_tag = "[gasps]" (for child) or "[nervous]" (for narrator)
   ```

2. **Prepends tag to text:**
   ```python
   text = "¿Escuchas eso?"
   emotion = "fear"
   is_child = True

   → Enhanced: "[nervous] ¿Escuchas eso?"
   ```

3. **Child voice gets special treatment:**
   - More reactive tags ([gasps], [laughs], [cries])
   - Additional style boost (+0.10) for higher expressiveness
   - Lower stability for more dynamic delivery

---

## 3. Spanish Language Optimization

### Auto-Detection vs. Explicit Language

**Eleven v3 multilingual:**
- **Auto-detects language** from text content (Spanish detected automatically)
- No language_code parameter needed (unlike older models)
- Supports Spanish variants: es-ES (Spain), es-MX (Mexico), es-AR (Argentina)

### Punctuation Controls Prosody

**Critical for Spanish TTS natural pacing:**

| Punctuation | Effect | Example |
|---|---|---|
| `.` (period) | Creates pause | "Se fue al bosque. Silencio." |
| `,` (comma) | Brief pause/breath | "Lentamente, caminaba hacia adelante." |
| `!` (exclamation) | Adds energy/emphasis | "¡Corre! ¡Ahora!" |
| `...` (ellipsis) | Extended dramatic pause | "Y entonces... apareció." |
| `?` (question) | Raises pitch | "¿Quién eres?" |

**V3 doesn't modify these** - they're preserved for natural pacing.

### Text Normalization (Built-In)

ElevenLabs automatically handles:
- Numbers: "1000" → "mil"
- Currency: "$100" → "cien dólares"
- Dates: "2026-03-28" → "veintiocho de marzo de dos mil veintiséis"
- Abbreviations: "Sr." → "Señor"

No special action needed - just write natural Spanish!

---

## 4. Emotion-to-Voice-Settings Mapping

### Design Principles

**V3 uses a combined approach:**

1. **Voice Settings** (stability, similarity, style) → Overall tone/personality
2. **Audio Tags** ([excited], [whispers], etc.) → Specific emotional moments
3. **Text Content** ("ánimo", "miedo", "alegría") → Context clues
4. **Intensity Parameter** (0.0-1.0) → Modulates expressiveness

### Emotion Profiles

#### Neutral
```
Usage: Informative narration, baseline delivery
Settings: High stability (0.70), low style (0.15)
Tags: None (or [calm] for subtlety)
Example: "En el corazón del bosque, una casa antigua aguardaba."
```

#### Calm / Peaceful
```
Usage: Reflective moments, peaceful scenes
Settings: Highest stability (0.80), minimal style (0.10)
Tags: [calm] for extra softness
Example: "[calm] Las aves descansaban en las ramas, observando el atardecer."
```

#### Wonder / Awe
```
Usage: Magical moments, discoveries
Settings: Mid stability (0.45), medium style (0.40)
Tags: None (let the text convey wonder)
Example: "Un destello dorado iluminó la gruta, revelando tesoros antiguos."
```

#### Suspense
```
Usage: Building tension, mystery
Settings: Low stability (0.40), high style (0.45)
Tags: [pause] or [short pause] for effect
Example: "[short pause] Una sombra se movía entre los árboles... muy lentamente."
```

#### Fear / Ominous
```
Usage: Frightening moments, danger
Settings: Very low stability (0.35), high style (0.55)
Tags: [nervous], [gasps] (especially for child)
Example: "[nervous] Los gruñidos se acercaban. Cada paso más cerca."
```

#### Dramatic / Intense
```
Usage: Climax, crucial moments
Settings: Low stability (0.38), maximum style (0.50)
Tags: [excited] or [shouts]
Example: "¡El dragón extendió sus alas! [excited] ¡Era más grande que el cielo!"
```

#### Excited / Energetic
```
Usage: Action, enthusiasm, joy
Settings: Very low stability (0.32), high style (0.60)
Tags: [excited], [laughs]
Example: "[excited] ¡Ganamos! ¡Lo hicimos!"
```

### Intensity Modifier (0.0 - 1.0)

**How intensity scales emotion:**

- **0.5 (default)**: Base emotion settings used as-is
- **> 0.5 (more intense)**: Decreases stability, increases style
  - Makes delivery more dynamic, expressive
  - Example: intensity=0.9 on "fear" → even lower stability (≈0.30), higher style (≈0.60)
- **< 0.5 (less intense)**: Increases stability, decreases style
  - Makes delivery more controlled, understated
  - Example: intensity=0.3 on "excited" → higher stability (≈0.50), lower style (≈0.40)

**Modulation formula in code:**
```python
intensity_factor = (intensity - 0.5) * 2  # Range: -1.0 to 1.0

if intensity > 0.5:
    stability *= (1 - intensity_factor * 0.2)  # Decrease (more dynamic)
    style = min(style + (intensity_factor * 0.15), 0.65)  # Increase
```

---

## 5. Dual-Narrator Voice Differentiation

### Norah (Narrator - Storyteller)
- **Voice ID:** kcQkGnn0HAT2JRDQ4Ljp
- **Characteristics:** Warm, friendly, clear
- **Role:** Establishes scene, guides narrative, describes action
- **Stability:** Higher (less emotional, more consistent)
- **Style:** Moderate (10-50% range for expressiveness)
- **Tags:** Subtle ([calm], [pause] for pacing)
- **Typical text:** "En el bosque oscuro, bajo las estrellas plateadas..."

### Daniela (Child - Companion)
- **Voice ID:** ajOR9IDAaubDK5qtLUqQ
- **Characteristics:** Warm, clear, positive
- **Role:** Reacts emotionally, asks questions, complements narrator
- **Stability:** Lower (more emotional, dynamic)
- **Style:** Higher (30-65% range for more expression)
- **Style Boost:** +0.10 additional for kid voice expressiveness
- **Tags:** Reactive ([gasps], [excited], [nervous], [laughs])
- **Typical text:** "[gasps] ¿Escuchas eso? ¡Me da mucho miedo!"

### Key Differences Applied in Code

```python
if is_child:
    stability = stability - 0.08  # More dynamic
    style = style + 0.10          # More expressive
    similarity = similarity + 0.03 # Slightly clearer
```

---

## 6. API Call Flow

### Request Structure

```python
client.text_to_speech.convert(
    voice_id="kcQkGnn0HAT2JRDQ4Ljp",          # Norah or Daniela
    text="[calm] Texto en español aquí.",     # Enhanced with audio tags
    model_id="eleven_multilingual_v3",
    output_format="pcm_24000",                # 24kHz PCM 16-bit mono
    voice_settings=VoiceSettings(
        stability=0.70,
        similarity_boost=0.75,
        style=0.15,
        use_speaker_boost=True                # MUST be true for audio tags
    )
)
```

### Response Processing

1. **Stream response**: ElevenLabs returns audio as chunks
2. **Collect PCM**: Concatenate all chunks into single PCM bytes buffer
3. **Calculate duration**: `num_samples / sample_rate`
4. **Add silence**: Insert 200ms gaps between lines (natural pauses)
5. **Build WAV**: Add WAV header + PCM data
6. **Write to disk**: Save as `narration.wav`

---

## 7. Performance & Latency

### Voice Settings Impact on Speed

From official ElevenLabs documentation:

| Setting | Impact | Note |
|---------|--------|------|
| `stability` | Minimal | Doesn't affect latency |
| `similarity_boost` | Minimal | Doesn't affect latency |
| `style = 0` | Fastest | Computationally cheapest |
| `style = 0.5` | Medium | Standard latency |
| `style > 0.5` | Slower | Higher computational cost |

**V3 Strategy:**
- Keep style in 10-60% range for acceptable latency
- Batch requests if possible (processed by ElevenLabs in queue)
- Typical latency: 1-3 seconds per dialogue line

### Retry Strategy

V3 implements **exponential backoff** for failures:
- Attempt 1: Immediate
- Attempt 2: Wait 2s
- Attempt 3: Wait 4s
- Attempt 4: Wait 8s
- Attempt 5: Wait 16s
- Max: 30s wait

---

## 8. Spanish-Specific Best Practices

### Pronunciation Hints

For words that might be mispronounced:

```spanish
# ElevenLabs auto-detects, but you can hint with context:
"Zorro" (ambiguous - could be "thorro" in Spain or "sorro" in Latin America)
→ Add context: "El zorro rojo saltó sobre el muro."  # Clearer

# Names/proper nouns
"Mateo" (could rhyme with "cuate-o" or "mate-o")
→ Add context: "Mateo caminaba lentamente."
```

### Regional Spanish

If using es-MX (Mexico) or es-AR (Argentina) Spanish:
- Vos / tú differences handled automatically
- Pronunciation adapts to region
- ElevenLabs detects context

---

## 9. Troubleshooting

### Emotion Not Applied
**Symptom:** Audio sounds monotone despite emotion settings
**Causes:**
- `use_speaker_boost = False` (must be True for v3)
- `style = 0` (needs to be > 0 for expression)
- Stability too high (try lowering to 0.40-0.50)

**Fix:** Increase style to 0.30+, ensure use_speaker_boost=True

### Audio Tags Not Working
**Symptom:** [excited], [whispers], etc. not interpreted
**Causes:**
- Model is not `eleven_multilingual_v3`
- use_speaker_boost is False
- Tags placed incorrectly (must be at start of phrase)

**Fix:** Check model_id, enable use_speaker_boost, verify tag syntax

### Spanish Pronunciation Issues
**Symptom:** Words mispronounced
**Causes:**
- Text contains English mixed with Spanish
- Numbers/dates not normalized
- Abbreviations unclear

**Fix:** Use pure Spanish, add context, spell out numbers if critical

### Silence Not Working
**Symptom:** No pauses between dialogue lines
**Causes:**
- Silence being generated but lost in concatenation
- Audio player skipping short silences

**Fix:** Increase silence duration (from 200ms to 300-400ms), test playback

---

## 10. Production Checklist

- [ ] ELEVENLABS_API_KEY configured
- [ ] NARRATOR_VOICE_ID = "kcQkGnn0HAT2JRDQ4Ljp" (Norah)
- [ ] CHILD_VOICE_ID = "ajOR9IDAaubDK5qtLUqQ" (Daniela)
- [ ] Audio tags working in test: `python main.py --dry-run`
- [ ] Voice settings applied correctly (check logs)
- [ ] Spanish text clean and properly punctuated
- [ ] Intensity values within 0.0-1.0 range
- [ ] Generated narration.wav plays correctly
- [ ] Dual voices clearly differentiated in final video
- [ ] Subtitles sync with audio

---

## References

1. [ElevenLabs Best Practices](https://elevenlabs.io/docs/overview/capabilities/text-to-speech/best-practices)
2. [Eleven v3 Audio Tags - Emotional Context](https://elevenlabs.io/blog/eleven-v3-audio-tags-expressing-emotional-context-in-speech)
3. [Voice Settings Documentation](https://elevenlabs-sdk.mintlify.app/speech-synthesis/voice-settings)
4. [ElevenLabs Cheat Sheet 2025](https://www.webfuse.com/elevenlabs-cheat-sheet)
5. [Spanish Text to Speech Guide](https://elevenlabs.io/text-to-speech/spanish)
