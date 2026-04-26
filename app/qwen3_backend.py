from __future__ import annotations

from tts_backends import TTSBackend, SynthesisRequest, SynthesisResult, register_backend, is_rocm, safe_compute_dtype

import gc
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

_ROOT_DIR = Path(__file__).parent.parent
DEFAULT_SR = 24000


class _Qwen3Base(TTSBackend):

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._loaded = False

    @property
    def default_sample_rate(self) -> int:
        return DEFAULT_SR

    @property
    def auth_required(self) -> bool:
        return False

    @property
    def venv_names(self) -> List[str]:
        return ["venv_qwen3"]

    @property
    def header_icon(self) -> str:
        return "🧬"

    @property
    def header_title(self) -> str:
        return "Qwen3 TTS"

    def is_available(self) -> bool:
        return self.model_dir.exists() and any(self.model_dir.iterdir())

    def load(self, progress_cb: Optional[Callable] = None) -> None:
        self.load_model(progress_cb)

    def unload(self) -> None:
        self.unload_model()

    def synthesize(self, request: SynthesisRequest, progress_cb: Optional[Callable] = None) -> SynthesisResult:
        audio, sr = self.generate(
            text=request.text,
            reference_audio_path=request.reference_audio,
            reference_text=request.reference_text,
            progress_cb=progress_cb,
        )
        return SynthesisResult(audio=audio, sample_rate=sr, duration_s=len(audio) / sr if audio is not None else 0)

    def download(self, model_dir: Path, progress_cb: Optional[Callable] = None) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        if progress_cb:
            progress_cb(f"Downloading {self.download_repo}…")
        snapshot_download(repo_id=self.download_repo, local_dir=str(model_dir))
        if progress_cb:
            progress_cb("Model downloaded successfully!")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device_info(self) -> str:
        if self.device == "cuda":
            try:
                name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
                return f"CUDA — {name} ({mem} GB)"
            except Exception:
                return "CUDA"
        return "CPU (no GPU — generation will be slow)"

    def load_model(self, progress_cb: Optional[Callable] = None):
        def status(msg):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        status("Importing Qwen3-TTS…")
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError as e:
            raise Exception(f"Failed to import qwen_tts: {e}\nMake sure 'qwen-tts' is installed in venv_qwen3")

        status(f"Loading Qwen3TTS from {self.model_dir} on {self.device}")

        dtype      = safe_compute_dtype(self.device)
        device_map = "cuda:0" if self.device == "cuda" else "cpu"

        if self.device == "cuda":
            if is_rocm():
                attn_impl = "sdpa"
            else:
                try:
                    import flash_attn
                    attn_impl = "flash_attention_2"
                except ImportError:
                    attn_impl = "sdpa"
        else:
            attn_impl = None

        self._model = Qwen3TTSModel.from_pretrained(
            str(self.model_dir),
            device_map=device_map,
            dtype=dtype,
            attn_implementation=attn_impl,
        )

        self._loaded = True
        status("✓ Qwen3TTS model loaded!")
    
    def unload_model(self, progress_cb: Optional[Callable] = None):
        def status(msg):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        if not self._loaded:
            return

        status("Releasing Qwen3TTS model…")
        self._model = None
        self._loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        status("✓ Qwen3TTS model unloaded")

    def generate(self, text: str, reference_audio_path: Optional[str] = None,
                 reference_text: Optional[str] = None, progress_cb: Optional[Callable] = None, **kwargs) -> Tuple[np.ndarray, int]:
        raise NotImplementedError


# ==================== REGISTERED BACKENDS ====================

@register_backend
class Qwen3VoiceDesign17B(_Qwen3Base):

    def __init__(self):
        super().__init__(_ROOT_DIR / "models" / "qwen3-voicedesign-1.7b")

    @property
    def name(self) -> str:
        return "qwen3-voicedesign-1.7b"

    @property
    def model_id(self) -> str:
        return "qwen3-voicedesign-1.7b"

    @property
    def display_name(self) -> str:
        return "🧬 Qwen3 VoiceDesign 1.7B"

    @property
    def download_repo(self) -> str:
        return "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

    @property
    def download_size(self) -> str:
        return "~4.2 GB"

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "language",
                "label": "Language",
                "type": "combo",
                "options": [
                    ("Auto", "Auto"), ("Chinese", "Chinese"), ("English", "English"),
                    ("Japanese", "Japanese"), ("Korean", "Korean"), ("German", "German"),
                    ("French", "French"), ("Russian", "Russian"), ("Portuguese", "Portuguese"),
                    ("Spanish", "Spanish"), ("Italian", "Italian"),
                ],
                "default": "Auto",
            },
            {"key": "temperature", "type": "slider", "label": "Temperature", "min": 0.1, "max": 1.5, "default": 0.7},
            {"key": "top_p", "type": "slider", "label": "Top-P", "min": 0.1, "max": 1.0, "default": 0.8},
        ]

    def generate(self, text: str, reference_audio_path: Optional[str] = None,
                 reference_text: Optional[str] = None, progress_cb: Optional[Callable] = None, **kwargs) -> Tuple[np.ndarray, int]:
        if not self._loaded or self._model is None:
            raise Exception("Model not loaded")

        language = kwargs.get("language")
        if language == "Auto":
            language = None
        instruct = reference_text or ""
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.8)

        wavs, sr = self._model.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
            temperature=temperature,
            top_p=top_p,
        )
        wav = wavs[0] if isinstance(wavs, list) else wavs
        audio = wav.squeeze().detach().cpu().float().numpy() if torch.is_tensor(wav) else np.asarray(wav, dtype=np.float32)
        return audio, sr


@register_backend
class Qwen3CustomVoice17B(_Qwen3Base):

    def __init__(self):
        super().__init__(_ROOT_DIR / "models" / "qwen3-customvoice-1.7b")

    @property
    def name(self) -> str:
        return "qwen3-customvoice-1.7b"

    @property
    def model_id(self) -> str:
        return "qwen3-customvoice-1.7b"

    @property
    def display_name(self) -> str:
        return "🧬 Qwen3 CustomVoice 1.7B"

    @property
    def download_repo(self) -> str:
        return "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

    @property
    def download_size(self) -> str:
        return "~4.2 GB"

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "language",
                "label": "Language",
                "type": "combo",
                "options": [("Auto", "Auto"), ("Chinese", "Chinese"), ("English", "English"), ("Japanese", "Japanese"), ("Korean", "Korean"), ("German", "German"), ("French", "French"), ("Russian", "Russian"), ("Portuguese", "Portuguese"), ("Spanish", "Spanish"), ("Italian", "Italian")],
                "default": "Auto",
            },
            {
                "key": "speaker",
                "label": "Speaker",
                "type": "combo",
                "options": [("Vivian", "Vivian"), ("Serena", "Serena"), ("Uncle_Fu", "Uncle_Fu"), ("Dylan", "Dylan"), ("Eric", "Eric"), ("Ryan", "Ryan"), ("Aiden", "Aiden"), ("Ono_Anna", "Ono_Anna"), ("Sohee", "Sohee")],
                "default": "Vivian",
            },
            {"key": "temperature", "type": "slider", "label": "Temperature", "min": 0.1, "max": 1.5, "default": 0.7},
            {"key": "top_p", "type": "slider", "label": "Top-P", "min": 0.1, "max": 1.0, "default": 0.8},
        ]

    def generate(self, text: str, reference_audio_path: Optional[str] = None,
                 reference_text: Optional[str] = None, progress_cb: Optional[Callable] = None, **kwargs) -> Tuple[np.ndarray, int]:
        if not self._loaded or self._model is None:
            raise Exception("Model not loaded")

        language = kwargs.get("language")
        if language == "Auto":
            language = None
        speaker = kwargs.get("speaker", "Vivian")
        instruct = reference_text or ""
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.8)

        wavs, sr = self._model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            temperature=temperature,
            top_p=top_p,
        )
        wav = wavs[0] if isinstance(wavs, list) else wavs
        audio = wav.squeeze().detach().cpu().float().numpy() if torch.is_tensor(wav) else np.asarray(wav, dtype=np.float32)
        return audio, sr


@register_backend
class Qwen3CustomVoice06B(_Qwen3Base):

    def __init__(self):
        super().__init__(_ROOT_DIR / "models" / "qwen3-customvoice-0.6b")

    @property
    def name(self) -> str:
        return "qwen3-customvoice-0.6b"

    @property
    def model_id(self) -> str:
        return "qwen3-customvoice-0.6b"

    @property
    def display_name(self) -> str:
        return "🧬 Qwen3 CustomVoice 0.6B (lighter)"

    @property
    def download_repo(self) -> str:
        return "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

    @property
    def download_size(self) -> str:
        return "~2.1 GB"

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "language",
                "label": "Language",
                "type": "combo",
                "options": [("Auto", "Auto"), ("Chinese", "Chinese"), ("English", "English"), ("Japanese", "Japanese"), ("Korean", "Korean"), ("German", "German"), ("French", "French"), ("Russian", "Russian"), ("Portuguese", "Portuguese"), ("Spanish", "Spanish"), ("Italian", "Italian")],
                "default": "Auto",
            },
            {
                "key": "speaker",
                "label": "Speaker",
                "type": "combo",
                "options": [("Vivian", "Vivian"), ("Serena", "Serena"), ("Uncle_Fu", "Uncle_Fu"), ("Dylan", "Dylan"), ("Eric", "Eric"), ("Ryan", "Ryan"), ("Aiden", "Aiden"), ("Ono_Anna", "Ono_Anna"), ("Sohee", "Sohee")],
                "default": "Vivian",
            },
            {"key": "temperature", "type": "slider", "label": "Temperature", "min": 0.1, "max": 1.5, "default": 0.7},
            {"key": "top_p", "type": "slider", "label": "Top-P", "min": 0.1, "max": 1.0, "default": 0.8},
        ]

    def generate(self, text: str, reference_audio_path: Optional[str] = None,
                 reference_text: Optional[str] = None, progress_cb: Optional[Callable] = None, **kwargs) -> Tuple[np.ndarray, int]:
        if not self._loaded or self._model is None:
            raise Exception("Model not loaded")

        language = kwargs.get("language")
        if language == "Auto":
            language = None
        speaker = kwargs.get("speaker", "Vivian")
        instruct = reference_text or ""
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.8)

        wavs, sr = self._model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            temperature=temperature,
            top_p=top_p,
        )
        wav = wavs[0] if isinstance(wavs, list) else wavs
        audio = wav.squeeze().detach().cpu().float().numpy() if torch.is_tensor(wav) else np.asarray(wav, dtype=np.float32)
        return audio, sr


@register_backend
class Qwen3VoiceClone17B(_Qwen3Base):

    def __init__(self):
        super().__init__(_ROOT_DIR / "models" / "qwen3-voiceclone-1.7b")

    @property
    def name(self) -> str:
        return "qwen3-voiceclone-1.7b"

    @property
    def model_id(self) -> str:
        return "qwen3-voiceclone-1.7b"

    @property
    def display_name(self) -> str:
        return "🧬 Qwen3 VoiceClone 1.7B"

    @property
    def download_repo(self) -> str:
        return "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

    @property
    def download_size(self) -> str:
        return "~4.2 GB"

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "language",
                "label": "Language",
                "type": "combo",
                "options": [("Auto", "Auto"), ("Chinese", "Chinese"), ("English", "English"), ("Japanese", "Japanese"), ("Korean", "Korean"), ("German", "German"), ("French", "French"), ("Russian", "Russian"), ("Portuguese", "Portuguese"), ("Spanish", "Spanish"), ("Italian", "Italian")],
                "default": "Auto",
            },
            {"key": "temperature", "type": "slider", "label": "Temperature", "min": 0.1, "max": 1.5, "default": 0.7},
            {"key": "top_p", "type": "slider", "label": "Top-P", "min": 0.1, "max": 1.0, "default": 0.8},
        ]

    def generate(self, text: str, reference_audio_path: Optional[str] = None,
                 reference_text: Optional[str] = None, progress_cb: Optional[Callable] = None, **kwargs) -> Tuple[np.ndarray, int]:
        if not self._loaded or self._model is None:
            raise Exception("Model not loaded")

        language = kwargs.get("language")
        if language == "Auto":
            language = None
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.8)

        wavs, sr = self._model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=reference_audio_path,
            ref_text=reference_text or "",
            temperature=temperature,
            top_p=top_p,
        )
        wav = wavs[0] if isinstance(wavs, list) else wavs
        audio = wav.squeeze().detach().cpu().float().numpy() if torch.is_tensor(wav) else np.asarray(wav, dtype=np.float32)
        return audio, sr


@register_backend
class Qwen3VoiceClone06B(_Qwen3Base):

    def __init__(self):
        super().__init__(_ROOT_DIR / "models" / "qwen3-voiceclone-0.6b")

    @property
    def name(self) -> str:
        return "qwen3-voiceclone-0.6b"

    @property
    def model_id(self) -> str:
        return "qwen3-voiceclone-0.6b"

    @property
    def display_name(self) -> str:
        return "🧬 Qwen3 VoiceClone 0.6B (lighter)"

    @property
    def download_repo(self) -> str:
        return "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

    @property
    def download_size(self) -> str:
        return "~2.1 GB"

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "language",
                "label": "Language",
                "type": "combo",
                "options": [("Auto", "Auto"), ("Chinese", "Chinese"), ("English", "English"), ("Japanese", "Japanese"), ("Korean", "Korean"), ("German", "German"), ("French", "French"), ("Russian", "Russian"), ("Portuguese", "Portuguese"), ("Spanish", "Spanish"), ("Italian", "Italian")],
                "default": "Auto",
            },
            {"key": "temperature", "type": "slider", "label": "Temperature", "min": 0.1, "max": 1.5, "default": 0.7},
            {"key": "top_p", "type": "slider", "label": "Top-P", "min": 0.1, "max": 1.0, "default": 0.8},
        ]

    def generate(self, text: str, reference_audio_path: Optional[str] = None,
                 reference_text: Optional[str] = None, progress_cb: Optional[Callable] = None, **kwargs) -> Tuple[np.ndarray, int]:
        if not self._loaded or self._model is None:
            raise Exception("Model not loaded")

        language = kwargs.get("language")
        if language == "Auto":
            language = None
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.8)

        wavs, sr = self._model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=reference_audio_path,
            ref_text=reference_text or "",
            temperature=temperature,
            top_p=top_p,
        )
        wav = wavs[0] if isinstance(wavs, list) else wavs
        audio = wav.squeeze().detach().cpu().float().numpy() if torch.is_tensor(wav) else np.asarray(wav, dtype=np.float32)
        return audio, sr
