# Video Generator V3 - Gemini Dual-Narrator Edition

## 🎬 Descripción

V3 introduce una **arquitectura de dual-narrador** usando **Gemini 3.1 Flash TTS**:
- **Narradora**: `Aoede` — Describe la escena, guía la narrativa
- **Niño**: `Puck` — Reacciona emocionalmente, hace preguntas, complementa

El audio **interleazado** entre ambas voces crea una **experiencia de cuento más dinámica y envolvente**.

### 🔬 Basado en Documentación Oficial

V3 implementa **best practices de ElevenLabs 2026:**
- Voice Settings optimization (stability 35-40%, similarity 75-80%, style 10-50%)
- Eleven v3 Audio Tags para emoción ([excited], [whispers], [gasps], etc.)
- Spanish language optimization con procesamiento de puntuación
- Dual-voice differentiation para máximo contraste

**Ver documentación técnica en:** `ELEVENLABS_OPTIMIZATION.md`

---

## ⚙️ Setup Inicial

### 1. Instalar Dependencias

```bash
cd "Proyecto Python V3"
pip install elevenlabs>=1.0.0
pip install google-genai google-cloud-vertex
# ... otras dependencias según V2
```

### 2. Obtener Voice IDs de ElevenLabs

**Opción A: Script Helper (Recomendado)**
```bash
python get_voice_ids.py
# Ingresa tu API Key cuando se solicite
# El script mostrará los Voice IDs de Norah y Daniela
```

**Opción B: Manual (Dashboard)**
1. Ve a https://elevenlabs.io/app/voice-lab
2. Busca "Norah" → Click → "More actions" (⋮) → "Copy voice ID"
3. Guarda el ID
4. Repite para "Daniela"

### 3. Configurar config.py

Actualiza `config.py` con los Voice IDs obtenidos:

```python
ELEVENLABS_API_KEY = "sk_2e91af4d3359e7a96c89c5523a3f1f1b7bac305644c9c476"

# Reemplaza con los IDs reales de Norah y Daniela
NARRATOR_VOICE_ID = "xxxxxxxxxxxxxxxx"  # Norah
CHILD_VOICE_ID    = "xxxxxxxxxxxxxxxx"  # Daniela
```

**Alternativa (Variables de Entorno - MÁS SEGURO):**
```bash
export ELEVENLABS_API_KEY="tu_api_key"
export NARRATOR_VOICE_ID="norah_voice_id"
export CHILD_VOICE_ID="daniela_voice_id"
python main.py --theme "cuento"
```

---

## 🚀 Uso Básico

### Dry Run (Generar Manifest)
```bash
python main.py --theme "un zorro en el bosque" --dry-run --no-qa
```
Genera un manifest con las escenas y diálogos **sin** generar imágenes/videos.

### Full Pipeline
```bash
python main.py --theme "un zorro en el bosque" --scenes 4 --no-qa
```
Executa: LLM Story → Dialogue Audio (ElevenLabs) → Images → Videos → Assembly

### Con Duración Máxima
```bash
python main.py --theme "cuento" --max-duration 60 --no-qa
```
Genera story que **cabe exactamente en 60 segundos**. Si la narración es muy larga, reduce escenas automáticamente.

### Con Estilo Específico
```bash
python main.py --theme "cuento" --style ghibli_dark --scenes 5 --no-qa
```

### Reanudar Run Anterior
```bash
python main.py --resume 20260328_142530 --regenerate-scenes 2,3,4
```

---

## 📋 Flujo de Audio V3

```
1. LLM genera story con:
   - Per-scene dialogue arrays (narrator ↔ child)
   - Emotions + intensities por línea

2. gen_tts.py::generate_dialogue_audio():
   - Para cada scene → para cada dialogue line:
     * NARRATOR_VOICE_ID + voice_settings(emotion, intensity)
     * CHILD_VOICE_ID + voice_settings(emotion, intensity) + boost
     * +200ms silence entre líneas
     * +400ms silence entre escenas

3. Output: narration.wav (PCM 24kHz mono)

4. Audio-First Scaling:
   - Mide duración de audio
   - Distribuye escenas 4/6/8s para que video = audio

5. Phase 2-3: Imagen + Video + Assembly (sin cambios V2)
```

---

## 🎯 Estructura del Diálogo Generado

El LLM genera por cada escena:

```json
{
  "idx": 1,
  "dialogue": [
    {
      "speaker": "narrator",
      "text": "En el bosque oscuro, el viento murmuraba entre los árboles...",
      "emotion": "calm",
      "intensity": 0.6
    },
    {
      "speaker": "child",
      "text": "[susurra] ¿Escuchas eso? ¿Qué fue ese sonido?",
      "emotion": "fear",
      "intensity": 0.7
    },
    {
      "speaker": "narrator",
      "text": "De repente, una sombra se movió entre las ramas...",
      "emotion": "suspense",
      "intensity": 0.8
    }
  ],
  "image_prompt_en": "...",
  "video_prompt_en": "..."
}
```

**Reglas de Diálogo:**
- **Narradora**: Prosa descriptiva, 40-100 palabras, establece contexto
- **Niño**: Reacciones cortas (1-3 frases), puede usar tags: `[susurra]`, `[gasps]`, `[excited]`, `[scared]`, `[laughs]`
- **Alternancia**: Típicamente Narrator → Child → Narrator (no dos narradores seguidos)

---

## 🎤 Mapeo Emociones → Voice Settings

El sistema mapea emociones a parámetros de ElevenLabs:

```python
{
    "neutral":  (stability=0.50, similarity_boost=0.75, style=0.20),
    "suspense": (stability=0.30, similarity_boost=0.80, style=0.60),
    "fear":     (stability=0.20, similarity_boost=0.85, style=0.80),
    "wonder":   (stability=0.40, similarity_boost=0.75, style=0.50),
    "dramatic": (stability=0.30, similarity_boost=0.80, style=0.70),
    "calm":     (stability=0.70, similarity_boost=0.70, style=0.10),
    "excited":  (stability=0.20, similarity_boost=0.85, style=0.90),
}
```

- **intensity (0.0-1.0)** escala los valores (ej: intensity=1.0 → más expresivo)
- **Child speaker** recibe +0.15 style boost para más expresividad

---

## 🔧 Configuración Avanzada

### Cambiar Voces
En `config.py`:
```python
NARRATOR_VOICE_ID = "voice_id_otra_narradora"
CHILD_VOICE_ID = "voice_id_otro_nino"
```

### Ajustar Silence Padding
En `gen_tts.py`, función `generate_dialogue_audio()`:
```python
# +200ms silence entre líneas (cambiar a 300ms para más pausa)
silence = _silence_pcm(300)  # ms

# +400ms silence entre escenas (cambiar a 500ms para más separación)
scene_silence = _silence_pcm(500)  # ms
```

### Deshabilitar Subtítulos
```bash
python main.py --theme "cuento" --no-subtitles
```

---

## 📊 Diferencias V2 → V3

| Aspecto | V2 | V3 |
|---|---|---|
| **Narración** | `global_narration` (monolítico) | Per-scene `dialogue` array |
| **TTS Engine** | Gemini 2.5 Pro TTS | ElevenLabs eleven_multilingual_v3 |
| **Voces** | 1 voz fija ("Leda") | 2 voces fijas (Norah + Daniela) |
| **Emoción** | Prompt directives a Gemini | ElevenLabs VoiceSettings |
| **Personajes Audio** | Narrador invisible | Narrador + Niño complementario |
| **Interacción** | Narración pasiva | Diálogo interleazado, reacciones |
| **Audio Sync** | Audio-first (igual) | Audio-first (igual) |

---

## ⚠️ Troubleshooting

### Error: "ELEVENLABS_API_KEY not set"
- Verifica que config.py tiene la API key
- O usa variable de entorno: `export ELEVENLABS_API_KEY="tu_key"`

### Error: "NARRATOR_VOICE_ID or CHILD_VOICE_ID not set"
- Ejecuta `python get_voice_ids.py`
- Copia los IDs a config.py

### Audio generado pero muy corto/largo
- Ajusta `intensity` en diálogos generados (LLM puede modificar)
- Usa `--max-duration` para constrainir exactamente

### Voz del niño no suena diferente
- Verifica que CHILD_VOICE_ID es diferente de NARRATOR_VOICE_ID
- Aumenta `intensity` en líneas del niño para más expresividad

---

## 📞 Support

Para problemas con:
- **ElevenLabs API**: https://help.elevenlabs.io
- **Voice Library**: https://elevenlabs.io/voice-library
- **Video Generator V3**: Ver `out/{run_id}/manifest.json` y logs

---

## ✅ Checklist de Setup

- [ ] Instalar `elevenlabs>=1.0.0`
- [ ] Obtener API Key de ElevenLabs
- [ ] Obtener Voice IDs de Norah y Daniela
- [ ] Actualizar `config.py` con API Key + Voice IDs
- [ ] Prueba dry run: `python main.py --theme "test" --dry-run --no-qa`
- [ ] Prueba full run: `python main.py --theme "test" --scenes 2 --no-qa`

---

**¡Listo! V3 está configurado y listo para generar videos con diálogo dual-narrador.**
