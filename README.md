# Audiobook and Dubbing Generator

[![Watch the video](https://img.youtube.com/vi/nhQ0yTyRaj4/maxresdefault.jpg)](https://www.youtube.com/watch?v=nhQ0yTyRaj4)

A desktop application for AI-powered text-to-speech synthesis, Lektor-style audio generation, audiobook creation from eBooks, and multi-speaker dubbing for both audiobooks and video content.

<img width="1527" height="998" alt="dr" src="https://github.com/user-attachments/assets/1a124398-332b-4b5e-a995-d0e3413820e6" />

---

## What it does

The app has three main modes, accessible as tabs:

### 📄 SRT Fragments
Load an `.srt` or `.txt` subtitle file and synthesize each fragment into a separate audio clip using a TTS model. Once all fragments are generated, you can export a video with the synthesized lektor track mixed on top of the original audio (requires ffmpeg).

Supports multi-speaker dubbing: load a video, run automatic speaker diarization (via `pyannote/speaker-diarization-3.1`, requires a Hugging Face token), assign reference voices to each detected speaker, and let the app synthesize each line with the correct voice.

### 📚 Ebook Fragments
Load an ebook and synthesize it as an audiobook. Supported input formats: `.epub`, `.pdf`, `.mobi`, `.azw`, `.azw3`, `.fb2`, `.txt`.

Synthesized audiobook can be exported as WAV, FLAC, MP3, OGG, or OPUS. Sessions can be saved and resumed.

### ⚡ Quick TTS
Type or paste any text and synthesize it immediately with the currently loaded model. Useful for testing voices or generating one-off audio clips.

---

## Installation

### Prerequisites

Before running any installer, make sure the following are available on your system:

- **Python 3.10+** — [python.org/downloads](https://www.python.org/downloads/) — check *"Add Python to PATH"* during installation
- **Git** — [git-scm.com/download/win](https://git-scm.com/download/win)
- **ffmpeg** — [ffmpeg.org/download.html](https://ffmpeg.org/download.html) — must be accessible from PATH (required for video export and some audio format conversions)

### Installing a TTS backend

Each TTS model has its own isolated Python virtual environment. You only need to install the models you want to use — you can always install more later.

1. Run **`install.bat`** from the application root folder.
2. Choose a model from the menu (enter a number).
3. The corresponding installer will run, create a virtual environment under `venvs\`, download PyTorch, and install all dependencies.
4. During installation you will be asked to choose between **CPU** or **GPU (NVIDIA CUDA)** mode. The installer automatically detects your GPU and selects the appropriate CUDA version.

> Installing a model for the first time downloads PyTorch (~2.5–3.5 GB) plus model-specific packages. This may take several minutes depending on your internet connection.

### Launching the app

Run **`start.bat`** from the application root folder. It will list all installed environments — select one to activate it and launch the application.

### Notes

**TADA TTS** — requires a Hugging Face account token and **Meta Llama license approval**. During installation a dialog will ask for your token and prompt you to accept the Llama 3.2 license at [huggingface.co/meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B). Meta approval typically takes 10–20 minutes — do not proceed with installation until you receive the confirmation email.

**Speaker diarization** (the *"I want dubbing"* feature in the SRT tab) requires a Hugging Face token and acceptance of the pyannote model terms. Before using it:
1. Accept terms at [huggingface.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
2. Accept terms at [huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Create a token (Classic, Read) at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

The token is entered inside the application when you first activate dubbing mode. It is saved in `.hf_token` in the application root for subsequent sessions.

---

## TTS Backends

The app supports multiple TTS backends. Available backends are detected automatically at startup based on which virtual environments are installed. Each backend has its own installer.

| Backend | Model page |
|---|---|
| Fish Audio S2 Pro | [huggingface.co/fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) |
| MOSS TTS | [huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer) |
| Chatterbox | [huggingface.co/ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) |
| OmniVoice | [huggingface.co/k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) |
| Qwen3 TTS | [huggingface.co/collections/Qwen/qwen3-tts](https://huggingface.co/collections/Qwen/qwen3-tts) |
| TADA | [huggingface.co/HumeAI/tada-3b-ml](https://huggingface.co/HumeAI/tada-3b-ml) · [tada-1b](https://huggingface.co/HumeAI/tada-1b) |
| VoxCPM2 | [huggingface.co/openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) |
| Supertonic 3 | [huggingface.co/Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3) |

Most backends support **voice cloning** via a reference audio file. You can upload a reference `.wav`, optionally pre-process it (resample, convert to mono, normalize, isolate vocals via Demucs), and optionally transcribe it using Whisper before synthesis.

---

## Whisper Integration

The app can transcribe reference audio using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Available model sizes: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`. Models are downloaded from Hugging Face and stored locally under `models/whisper/`.

Supported transcription languages include English, Polish, Japanese, Chinese, Korean, German, French, Spanish, Russian, Arabic, Portuguese, Italian, Turkish, Dutch, Ukrainian, Swedish, Finnish, and more (auto-detect available).

---

## Audio Preprocessing

Before using a reference audio file for voice cloning, you can run it through the built-in preprocessor, which supports:

- Resampling to a target sample rate (8 kHz – 48 kHz)
- Converting stereo to mono
- Vocal isolation using [Demucs](https://github.com/facebookresearch/demucs)
- Normalization
- Bit depth selection (PCM output)

## License

[GNU General Public License v3.0](LICENSE)
