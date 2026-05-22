# Proyecto Python V3 - SETUP COMPLETO ✅

## 📋 Estado del Proyecto

**Proyecto V3 está 100% creado y configurado con:**
- ✅ Arquitectura de dual-narrador (Norah + Daniela)
- ✅ ElevenLabs eleven_v3 optimizado
- ✅ Voice Settings basados en best practices 2026
- ✅ Audio Tags para emoción
- ✅ Spanish language optimization
- ✅ Schemas con `DialogueLine` y per-scene `dialogue`
- ✅ gen_tts.py completamente reescrito
- ✅ main.py actualizado (Phase 1B)
- ✅ Documentación técnica completa

---

## 🔧 Instalación Rápida (5 minutos)

### 1. Instalar dependencias
```bash
cd "Proyecto Python V3"
pip install elevenlabs>=1.0.0
pip install google-genai google-cloud-vertex  # Si no lo tienes de V2
```

### 2. Voice IDs ya configurados
```python
# En config.py (YA CONFIGURADOS):
NARRATOR_VOICE_ID = "kcQkGnn0HAT2JRDQ4Ljp"  # Norah ✓
CHILD_VOICE_ID = "ajOR9IDAaubDK5qtLUqQ"     # Daniela ✓
ELEVENLABS_API_KEY = "sk_2e91af4d3359e7a96c89c5523a3f1f1b7bac305644c9c476"  # ✓
```

### 3. Validar setup
```bash
python test_elevenlabs_setup.py
```

Deberías ver:
```
✓ PASS | Imports
✓ PASS | Config File
✓ PASS | Schemas (V3)
✓ PASS | Voice Settings
✓ PASS | Gen TTS Module
✓ PASS | FFmpeg/FFprobe
✓ PASS | ElevenLabs API

Result: 7/7 tests passed
✅ All tests passed! Ready to run V3 pipeline.
```

---

## 🎬 Primer Run

### Paso 1: Prueba de manifestación (sin audio aún)
```bash
python main.py --theme "un zorro en el bosque" --dry-run --no-qa
```

**Resultado esperado:**
```
out/{run_id}/
├── manifest.json          ← Verifica: contiene "dialogue" arrays
├── story.json
└── narration.wav          ← NO generado (dry-run)
```

**Verificar manifest.json:**
```json
{
  "scenes": [
    {
      "dialogue": [
        {
          "speaker": "narrator",
          "text": "En el bosque oscuro...",
          "emotion": "calm",
          "intensity": 0.6
        },
        {
          "speaker": "child",
          "text": "[susurra] ¿Escuchas eso?",
          "emotion": "fear",
          "intensity": 0.7
        }
      ]
    }
  ]
}
```

### Paso 2: Generar solo audio
```bash
# Edita main.py temporalmente para skipear image/video gen
# O usa máquina menos potente para test

python main.py --theme "un zorro" --scenes 1 --no-qa --images-only
# Luego ejecuta solo Phase 1B manualmente (si quieres test más rápido)
```

### Paso 3: Full run (final)
```bash
python main.py --theme "un zorro en el bosque" --scenes 3 --no-qa
```

**Pasos que verás:**
1. LLM genera story + dialogue
2. ElevenLabs TTS → `narration.wav` (Norah + Daniela interleazados)
3. Audio-first scaling (distribuye escenas 4/6/8s)
4. Genera imágenes (Gemini 3 Pro Image)
5. Genera videos (Veo 3.1 Fast)
6. Ensamble FFmpeg
7. Subtítulos automáticos

---

## 📁 Estructura de Archivos

```
Proyecto Python V3/
├── config.py                          ✅ ElevenLabs config
├── schemas.py                         ✅ V3 schemas (DialogueLine, dialogue)
├── llm_story.py                       ✅ Dual-narrator prompts
├── gen_tts.py                         ✅ ElevenLabs integration (NEW)
├── main.py                            ✅ Pipeline (Phase 1B updated)
├── gen_image.py                       (copied from V2)
├── gen_video.py                       (copied from V2)
├── assemble.py                        (copied from V2)
├── ... [otros archivos V2]
│
├── get_voice_ids.py                   (helper script)
├── test_elevenlabs_setup.py           (validation script)
│
├── README_V3.md                       📖 User guide
├── ELEVENLABS_OPTIMIZATION.md         📖 Technical reference
└── SETUP_COMPLETE.md                  📖 This file
```

---

## 🎤 Cómo Funciona la Narración Dual

### Flujo de Generación

```
Escena 1:
  LLM genera dialogue array:
    [
      {speaker: "narrator", text: "En el bosque...", emotion: "calm"},
      {speaker: "child", text: "[gasps] ¿Qué es eso?", emotion: "fear"}
    ]

  Gen TTS:
    1. Norah: "En el bosque..."
       → Voice ID: kcQkGnn0HAT2JRDQ4Ljp
       → Settings: stability=0.70, similarity=0.75, style=0.15
       → Audio: ~1.5s PCM

    2. [200ms silence]

    3. Daniela: "[gasps] ¿Qué es eso?"
       → Voice ID: ajOR9IDAaubDK5qtLUqQ
       → Settings: stability=0.62 (lower), style=0.55 (higher), use_speaker_boost=true
       → [gasps] tag interpreted as emotional reaction
       → Audio: ~0.8s PCM

Escena 2:
  (Similar flow)

Final:
  Concatenar todos los PCM chunks con silencios
  → narration.wav (dual-narrator interleazado)
```

### Emociones Soportadas

| Emoción | Stability | Style | Uso |
|---------|-----------|-------|-----|
| neutral | 0.70 | 0.15 | Narración base |
| calm | 0.80 | 0.10 | Momentos pacíficos |
| wonder | 0.45 | 0.40 | Descubrimientos mágicos |
| suspense | 0.40 | 0.45 | Misterio, tensión |
| fear | 0.35 | 0.55 | Momentos aterradores |
| dramatic | 0.38 | 0.50 | Climax |
| excited | 0.32 | 0.60 | Energía, acción |

### Audio Tags Disponibles

**Para Norah (naradora):**
- `[calm]` - Entrega pacífica
- `[pause]`, `[short pause]`, `[long pause]` - Control de pacing
- Raramente emotivos

**Para Daniela (niño):**
- `[gasps]` - Sorpresa, miedo
- `[excited]` - Emoción, entusiasmo
- `[nervous]` - Incertidumbre
- `[laughs]` - Risa
- `[cries]` - Llorando
- `[whispers]` - Susurro secreto
- `[sigh]` - Suspiro

---

## 🔍 Debugging & Logs

### Ver logs detallados
```bash
python main.py --theme "cuento" --no-qa 2>&1 | tee output.log
```

### Específicamente gen_tts logs
```bash
# En main.py o en scripts, aumenta debug level:
logging.basicConfig(level=logging.DEBUG)
```

### Verificar audio generado
```bash
# Reproducir narración.wav generada
ffplay "out/{run_id}/narration.wav"

# Ver duración
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1:nokey=1 "out/{run_id}/narration.wav"

# Ver metadata
ffprobe "out/{run_id}/narration.wav"
```

---

## ⚙️ Configuración Avanzada

### Cambiar Voice IDs (si quieres otras voces)

En config.py:
```python
# Obtén otros voice IDs desde https://elevenlabs.io/voice-library
NARRATOR_VOICE_ID = "otro_id_aqui"
CHILD_VOICE_ID = "otro_id_aqui"
```

### Ajustar Voice Settings

En gen_tts.py, función `EMOTION_TO_VOICE_SETTINGS`:
```python
EMOTION_TO_VOICE_SETTINGS = {
    "custom_emotion": {
        "stability": 0.45,       # 0.0-1.0 (lower = more dynamic)
        "similarity_boost": 0.77, # 75-80% recommended
        "style": 0.35,            # 10-50% for narration
    }
}
```

### Modificar silencio entre líneas

En gen_tts.py, función `generate_dialogue_audio()`:
```python
# Línea ~180: aumentar de 200ms a 300ms para más pausa
silence = _silence_pcm(300)  # ms entre líneas

# Línea ~190: aumentar de 400ms a 500ms entre escenas
scene_silence = _silence_pcm(500)  # ms entre escenas
```

---

## 🚀 Comandos Útiles

```bash
# Dry run (solo manifest)
python main.py --theme "tema" --dry-run --no-qa

# Full pipeline
python main.py --theme "tema" --scenes 4 --no-qa

# Con duración máxima
python main.py --theme "tema" --max-duration 60 --no-qa
# (Reduce automáticamente escenas si narración es muy larga)

# Reanudar run anterior
python main.py --resume 20260328_142530

# Regenerar escenas específicas
python main.py --resume 20260328_142530 --regenerate-scenes 2,3,4

# Sin subtítulos
python main.py --theme "tema" --no-subtitles

# Solo imágenes (sin videos)
python main.py --theme "tema" --images-only

# Validar setup
python test_elevenlabs_setup.py
```

---

## ✅ Checklist Final

- [ ] elevenlabs>=1.0.0 instalado
- [ ] config.py tiene Voice IDs correctos
- [ ] test_elevenlabs_setup.py pasa 7/7
- [ ] ffprobe disponible (ffmpeg instalado)
- [ ] Primer dry-run genera manifest con dialogue
- [ ] Segundo run genera narration.wav (escúchalo)
- [ ] Tercer run genera video completo
- [ ] Dual-narrator audible en final_with_narration.mp4
- [ ] Subtítulos sincronizados en final_with_subtitles.mp4

---

## 📞 Support

### Si algo falla:

1. **Revisa los logs:**
   ```bash
   tail -100 out/{run_id}/logs/*.log
   ```

2. **Valida setup:**
   ```bash
   python test_elevenlabs_setup.py
   ```

3. **Documentación técnica:**
   - `README_V3.md` — User guide
   - `ELEVENLABS_OPTIMIZATION.md` — Technical reference
   - `gen_tts.py` — Inline comments

4. **ElevenLabs docs:**
   - https://elevenlabs.io/docs/overview/capabilities/text-to-speech/best-practices
   - https://elevenlabs.io/blog/eleven-v3-audio-tags-expressing-emotional-context-in-speech

---

## 🎉 ¡Listo!

**Proyecto V3 está completamente configurado y listo para generar videos con narración dual-narrador usando ElevenLabs.**

Próximo paso: `python main.py --theme "tu tema" --dry-run --no-qa`

¡Que disfrutes! 🎬
