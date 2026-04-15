from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from tts_backends import (
    TTSBackend, SynthesisRequest, SynthesisResult, register_backend,
)

_ROOT_DIR = Path(__file__).parent.parent
_CB_CACHE = _ROOT_DIR / "models" / "chatterbox"

CHATTERBOX_LANGUAGES: List[Tuple[str, str]] = [
    ("ar", "Arabic"),    ("da", "Danish"),      ("de", "German"),
    ("el", "Greek"),     ("en", "English"),     ("es", "Spanish"),
    ("fi", "Finnish"),   ("fr", "French"),      ("he", "Hebrew"),
    ("hi", "Hindi"),     ("it", "Italian"),     ("ja", "Japanese"),
    ("ko", "Korean"),    ("ms", "Malay"),       ("nl", "Dutch"),
    ("no", "Norwegian"), ("pl", "Polish"),      ("pt", "Portuguese"),
    ("ru", "Russian"),   ("sv", "Swedish"),     ("sw", "Swahili"),
    ("tr", "Turkish"),   ("zh", "Chinese"),
]

_EXAGGERATION_PARAM: Dict[str, Any] = {
    "key":     "exaggeration",
    "label":   "Exaggeration",
    "type":    "slider",
    "min":     0.0,
    "max":     1.0,
    "default": 0.5,
    "tip": (
        "Controls expressiveness of speech. Higher values produce more expressive output. "
        "Increase for dramatic speech. Higher exaggeration tends to speed up speech — "
        "lower cfg_weight to compensate."
    ),
}

_CFG_WEIGHT_PARAM: Dict[str, Any] = {
    "key":     "cfg_weight",
    "label":   "CFG Weight",
    "type":    "slider",
    "min":     0.0,
    "max":     1.0,
    "default": 0.5,
    "tip": (
        "Pacing control. Lower values (~0.3) produce slower, more deliberate pacing. "
        "If the reference clip language differs from the target language, set to 0."
    ),
}

_LANGUAGE_PARAM: Dict[str, Any] = {
    "key":     "language",
    "label":   "Language",
    "type":    "combo",
    "options": CHATTERBOX_LANGUAGES,
    "default": "en",
    "tip":     "Target language for multilingual synthesis.",
}


def _ensure_cache() -> None:
    _CB_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(_CB_CACHE)


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _tensor_to_numpy(wav: torch.Tensor) -> np.ndarray:
    return wav.squeeze().cpu().float().numpy().astype(np.float32)


class _ChatterboxBase(TTSBackend):

    def __init__(self):
        _ensure_cache()
        self._model    = None
        self._device   = _best_device()
        self.model_dir = _CB_CACHE

    @property
    def auth_required(self) -> bool:
        return False

    @property
    def venv_names(self) -> List[str]:
        return ["venv_chatterbox"]

    @property
    def default_sample_rate(self) -> int:
        return 24000

    @property
    def header_icon(self) -> str:
        return "🗣"

    @property
    def header_title(self) -> str:
        return "Chatterbox TTS"

    @property
    def whisper_incompatible(self) -> bool:
        return False

    @property
    def device(self) -> str:
        return self._device

    @property
    def device_info(self) -> str:
        return self._device.upper()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def _marker(self) -> Path:
        return _CB_CACHE / f".{self.name}_ok"

    def is_available(self) -> bool:
        return self._marker.exists()

    def download(self, model_dir: Path, progress_cb: Optional[Callable] = None) -> None:
        _ensure_cache()
        if progress_cb:
            progress_cb("Downloading Chatterbox model from HuggingFace…")
        self._do_download(progress_cb)
        self._marker.touch(exist_ok=True)
        if progress_cb:
            progress_cb("Download complete.")

    def _do_download(self, progress_cb: Optional[Callable] = None) -> None:
        raise NotImplementedError

    def load(self, progress_cb: Optional[Callable] = None) -> None:
        self.load_model(progress_cb)

    def load_model(self, progress_cb: Optional[Callable] = None) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        self.unload_model()

    def unload_model(self, progress_cb: Optional[Callable] = None) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if progress_cb:
            progress_cb("Model unloaded.")

    def synthesize(
        self,
        request: SynthesisRequest,
        progress_cb: Optional[Callable] = None,
    ) -> SynthesisResult:
        audio, sr = self.generate(
            text=request.text,
            reference_audio_path=request.reference_audio,
            reference_text=request.reference_text,
            progress_cb=progress_cb,
        )
        return SynthesisResult(audio=audio, sample_rate=sr, duration_s=len(audio) / sr)

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        progress_cb: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        raise NotImplementedError


@register_backend
class ChatterboxBackend(_ChatterboxBase):

    @property
    def name(self) -> str:
        return "chatterbox"

    @property
    def model_id(self) -> str:
        return "chatterbox"

    @property
    def display_name(self) -> str:
        return "Chatterbox TTS"

    @property
    def download_repo(self) -> str:
        return "resemble-ai/chatterbox"

    @property
    def download_size(self) -> str:
        return "~2 GB"

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return [_EXAGGERATION_PARAM, _CFG_WEIGHT_PARAM]

    def _do_download(self, progress_cb: Optional[Callable] = None) -> None:
        _ensure_cache()
        from chatterbox.tts import ChatterboxTTS
        if progress_cb:
            progress_cb("Downloading ChatterboxTTS — please wait…")
        ChatterboxTTS.from_pretrained(device="cpu")

    def load_model(self, progress_cb: Optional[Callable] = None) -> None:
        _ensure_cache()
        from chatterbox.tts import ChatterboxTTS
        if progress_cb:
            progress_cb("Loading ChatterboxTTS…")
        self._model = ChatterboxTTS.from_pretrained(device=self._device)
        self._marker.touch(exist_ok=True)
        if progress_cb:
            progress_cb("ChatterboxTTS ready.")

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        progress_cb: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        if progress_cb:
            progress_cb("Generating audio…")
        kw: Dict[str, Any] = {
            "exaggeration": float(kwargs.get("exaggeration", 0.5)),
            "cfg_weight":   float(kwargs.get("cfg_weight",   0.5)),
        }
        if reference_audio_path and Path(reference_audio_path).exists():
            kw["audio_prompt_path"] = reference_audio_path
        wav = self._model.generate(text, **kw)
        return _tensor_to_numpy(wav), self._model.sr


@register_backend
class ChatterboxTurboBackend(_ChatterboxBase):

    @property
    def name(self) -> str:
        return "chatterbox-turbo"

    @property
    def model_id(self) -> str:
        return "chatterbox-turbo"

    @property
    def display_name(self) -> str:
        return "Chatterbox-Turbo TTS"

    @property
    def download_repo(self) -> str:
        return "resemble-ai/chatterbox-turbo"

    @property
    def download_size(self) -> str:
        return "~2 GB"

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return []

    def _do_download(self, progress_cb: Optional[Callable] = None) -> None:
        _ensure_cache()
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        if progress_cb:
            progress_cb("Downloading ChatterboxTurboTTS — please wait…")
        ChatterboxTurboTTS.from_pretrained(device="cpu")

    def load_model(self, progress_cb: Optional[Callable] = None) -> None:
        _ensure_cache()
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        if progress_cb:
            progress_cb("Loading ChatterboxTurboTTS…")
        self._model = ChatterboxTurboTTS.from_pretrained(device=self._device)
        self._marker.touch(exist_ok=True)
        if progress_cb:
            progress_cb("ChatterboxTurboTTS ready.")

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        progress_cb: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        if progress_cb:
            progress_cb("Generating audio…")
        kw: Dict[str, Any] = {}
        if reference_audio_path and Path(reference_audio_path).exists():
            kw["audio_prompt_path"] = reference_audio_path
        wav = self._model.generate(text, **kw)
        return _tensor_to_numpy(wav), self._model.sr


@register_backend
class ChatterboxMultilingualBackend(_ChatterboxBase):

    @property
    def name(self) -> str:
        return "chatterbox-multilingual"

    @property
    def model_id(self) -> str:
        return "chatterbox-multilingual"

    @property
    def display_name(self) -> str:
        return "Chatterbox-Multilingual TTS"

    @property
    def download_repo(self) -> str:
        return "resemble-ai/chatterbox-multilingual"

    @property
    def download_size(self) -> str:
        return "~3 GB"

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return [_EXAGGERATION_PARAM, _CFG_WEIGHT_PARAM, _LANGUAGE_PARAM]

    def _do_download(self, progress_cb: Optional[Callable] = None) -> None:
        _ensure_cache()
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        if progress_cb:
            progress_cb("Downloading ChatterboxMultilingualTTS — please wait…")
        ChatterboxMultilingualTTS.from_pretrained(device="cpu")

    def load_model(self, progress_cb: Optional[Callable] = None) -> None:
        _ensure_cache()
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        if progress_cb:
            progress_cb("Loading ChatterboxMultilingualTTS…")
        self._model = ChatterboxMultilingualTTS.from_pretrained(device=self._device)
        self._marker.touch(exist_ok=True)
        if progress_cb:
            progress_cb("ChatterboxMultilingualTTS ready.")

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        progress_cb: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        if progress_cb:
            progress_cb("Generating audio…")
        kw: Dict[str, Any] = {
            "exaggeration": float(kwargs.get("exaggeration", 0.5)),
            "cfg_weight":   float(kwargs.get("cfg_weight",   0.5)),
            "language_id":  str(kwargs.get("language", "en")),
        }
        if reference_audio_path and Path(reference_audio_path).exists():
            kw["audio_prompt_path"] = reference_audio_path
        wav = self._model.generate(text, **kw)
        return _tensor_to_numpy(wav), self._model.sr
