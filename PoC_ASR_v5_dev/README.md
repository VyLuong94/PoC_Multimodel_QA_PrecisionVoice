---
title: PrecisionVoice
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app/main.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# PrecisionVoice - STT & Speaker Diarization

A production-ready Speech-to-Text and Speaker Diarization web application using FastAPI, faster-whisper, and pyannote.audio.

## Features

- 🎙️ Speech-to-Text using `erax-ai/EraX-WoW-Turbo-V1.1-CT2` (8x faster, 8 Vietnamese dialects)
- 👥 Speaker Diarization using `pyannote/speaker-diarization-3.1`
- 🧼 Speech Enhancement using `SpeechBrain SepFormer DNS4` (noise + reverb removal)
- 🔇 Voice Activity Detection using `Silero VAD v5` (prevents hallucination)
- 🎤 Vocal Isolation using `MDX-Net` (UVR-MDX-NET-Voc_FT)
- 🔄 Automatic speaker-transcript alignment
- 📥 Download results in TXT or SRT format
- 🐳 Docker-ready with persistent model caching and GPU support
- 🐳 Docker-ready with persistent model caching and GPU support

## Quick Start

### Prerequisites

1. Docker and Docker Compose
2. (Optional) NVIDIA GPU with CUDA support
3. HuggingFace account with access to pyannote models

### Setup

1. Clone and configure:
   ```bash
   cp .env.example .env
   # Edit .env and add your HuggingFace token
   ```

2. Build and run:
   ```bash
   docker compose up --build
   ```

3. Open http://localhost:8000

## Audio Processing Pipeline

The system uses a state-of-the-art multi-stage pipeline to ensure maximum accuracy:

1. **Speech Enhancement**: Background noise and reverb are removed using `SpeechBrain SepFormer` (DNS4 Challenge winner).
2. **Vocal Isolation**: Vocals are separated from background music using `MDX-Net`.
3. **VAD Filtering**: Silence is removed using `Silero VAD v5` to prevent ASR hallucination.
4. **Refinement**: Highpass filtering and EBU R128 loudness normalization.
5. **Transcription**: High-precision Vietnamese transcription using `PhoWhisper`.
6. **Diarization**: Segmenting audio by speaker using `Pyannote 3.1`.
7. **Alignment**: Merging transcripts with speaker segments + timestamp reconstruction.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | - | Required for Pyannote models |
| `ENABLE_SPEECH_ENHANCEMENT` | `True` | Toggle SpeechBrain speech enhancement |
| `ENHANCEMENT_MODEL` | `speechbrain/sepformer-dns4-16k-enhancement` | Model for speech enhancement |
| `ENABLE_SILERO_VAD` | `True` | Toggle Silero VAD for hallucination prevention |
| `ENABLE_VOCAL_SEPARATION` | `True` | Toggle MDX-Net vocal isolation |
| `MDX_MODEL` | `UVR-MDX-NET-Voc_FT` | Model for vocal separation |
| `DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |

## Development

### Local Setup (without Docker)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/transcribe` | POST | Upload and transcribe audio |
| `/api/download/{filename}` | GET | Download result files |

## Supported Audio Formats

- MP3
- WAV
- M4A
- OGG

## License

MIT
