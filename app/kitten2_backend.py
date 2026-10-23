from __future__ import annotations

from tts_backends import (
    TTSBackend, SynthesisRequest, SynthesisResult, register_backend,
)

import gc
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

_ROOT_DIR  = Path(__file__).parent.parent
_MODEL_DIR = _ROOT_DIR / "models" / "kitten2"
_HF_REPO   = "KittenML/kitten-tts-2"
DEFAULT_SR = 24000

_WEIGHTS_SIZES: Dict[str, str] = {
    "packed": "~1.0 GB",
    "emb4":   "~0.5 GB",
    "full":   "~3.5 GB",
}
_DECODERS = ("default", "student_w4")


def _env_choice(name: str, allowed, default: str) -> str:
    val = os.environ.get(name, default).strip().lower()
    return val if val in allowed else default


KITTEN2_WEIGHTS = _env_choice("KITTEN2_WEIGHTS", _WEIGHTS_SIZES.keys(), "packed")
KITTEN2_DECODER = _env_choice("KITTEN2_DECODER", _DECODERS, "default")

KITTEN2_LANGUAGE_VOICES: List[str] = [
    "Arabic", "Chinese", "French", "German", "Hindi",
    "Italian", "Portuguese", "Russian", "Spanish",
]

KITTEN2_VOICES: List[Tuple[str, str]] = (
    [
        ("Bella",  "Bella"),
        ("Jasper", "Jasper"),
        ("Luna",   "Luna"),
        ("Bruno",  "Bruno"),
        ("Rosie",  "Rosie"),
        ("Hugo",   "Hugo"),
        ("Kiki",   "Kiki"),
        ("Leo",    "Leo"),
        ("Victor", "Victor"),
    ]
    + [(v, f"{v} (language voice)") for v in KITTEN2_LANGUAGE_VOICES]
)

KITTEN2_PRESETS: List[Tuple[str, str]] = [
    ("default",    "Default"),
    ("expressive", "Expressive"),
]

_REF_MIN_S = 5.0
_REF_MAX_S = 30.0

_KITTEN2_PARAMS: List[Dict[str, Any]] = [
    {
        "key":     "kitten2_voice",
        "type":    "combo",
        "label":   "Voice",
        "options": KITTEN2_VOICES,
        "default": "Bella",
        "tip": (
            "Built-in voice. Ignored when a reference audio file is loaded "
            "(the voice is then cloned from it). Language voices (German, French, …) "
            "select the target language and accent."
        ),
    },
    {
        "key":     "kitten2_preset",
        "type":    "combo",
        "label":   "Decoding preset",
        "options": KITTEN2_PRESETS,
        "default": "default",
        "tip": (
            "Expressive enables stronger emotional variation. Works best with "
            "[emotion], <event> and (((emphasis))) markup in the text."
        ),
    },
    {
        "key":     "kitten2_normalize",
        "type":    "combo",
        "label":   "Text normalizer",
        "options": [(True, "On (English)"), (False, "Off (non-English)")],
        "default": True,
        "tip": (
            "English-tuned normalizer for numbers, dates and currencies. "
            "Turn off for non-English text (forced off for language voices)."
        ),
    },
]


def _ensure_cache() -> None:
    hub = _MODEL_DIR / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(_MODEL_DIR)
    os.environ["HF_HUB_CACHE"] = str(hub)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        from huggingface_hub import constants as _hf_const
        _hf_const.HF_HOME = str(_MODEL_DIR)
        _hf_const.HF_HUB_CACHE = str(hub)
        _hf_const.HUGGINGFACE_HUB_CACHE = str(hub)
    except Exception:
        pass


def _to_numpy(audio: Any) -> np.ndarray:
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().float().numpy()
    arr = np.asarray(audio)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    return arr.astype(np.float32, copy=False).reshape(-1)


def _wav_duration(path: str) -> Optional[float]:
    try:
        import soundfile as sf
        info = sf.info(path)
        return info.frames / float(info.samplerate)
    except Exception:
        return None


@register_backend
class Kitten2Backend(TTSBackend):

    def __init__(self):
        _ensure_cache()
        self.model_dir = _MODEL_DIR
        self._model    = None
        self._loaded   = False

    @property
    def name(self) -> str:
        return "KittenTTS 2"

    @property
    def model_id(self) -> str:
        return "kitten2"

    @property
    def display_name(self) -> str:
        return "🐱 KittenTTS 2"

    @property
    def default_sample_rate(self) -> int:
        return DEFAULT_SR

    @property
    def auth_required(self) -> bool:
        return False

    @property
    def venv_names(self) -> List[str]:
        return ["venv_kitten2"]

    @property
    def download_repo(self) -> str:
        return _HF_REPO

    @property
    def download_size(self) -> str:
        return _WEIGHTS_SIZES.get(KITTEN2_WEIGHTS, "~1.0 GB")

    @property
    def header_icon(self) -> str:
        return "🐱"

    @property
    def header_title(self) -> str:
        return "KittenTTS 2"

    @property
    def whisper_incompatible(self) -> bool:
        return True

    @property
    def pyannote_incompatible(self) -> bool:
        return False

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return _KITTEN2_PARAMS

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def device_info(self) -> str:
        if torch.cuda.is_available():
            try:
                n = torch.cuda.get_device_name(0)
                m = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
                return f"CUDA (auto) — {n} ({m} GB)"
            except Exception:
                return "CUDA (auto)"
        return "CPU (auto) — slower, 1.7B model"

    @property
    def _marker(self) -> Path:
        return _MODEL_DIR / f".kitten2_{KITTEN2_WEIGHTS}_{KITTEN2_DECODER}_ok"

    def is_available(self) -> bool:
        return self._marker.exists()

    @staticmethod
    def _construct():
        _ensure_cache()
        try:
            from kittenml import KittenTTS
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import kittenml:\n{e}\n\n"
                "Make sure 'kittenml' is installed in venv_kitten2 (run kitten2_install.bat)."
            )
        return KittenTTS(
            _HF_REPO,
            weights=KITTEN2_WEIGHTS,
            decoder=KITTEN2_DECODER,
        )

    def download(self, model_dir: Path, progress_cb: Optional[Callable] = None) -> None:
        def status(msg: str) -> None:
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        status(
            f"Downloading {_HF_REPO} (weights={KITTEN2_WEIGHTS}, "
            f"decoder={KITTEN2_DECODER}, {self.download_size})…"
        )
        model = self._construct()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._marker.touch(exist_ok=True)
        status("KittenTTS 2 downloaded successfully!")

    def load_model(self, progress_cb: Optional[Callable] = None) -> None:
        def status(msg: str) -> None:
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        if self._loaded and self._model is not None:
            status("✓ KittenTTS 2 already loaded.")
            return

        status(
            f"Loading KittenTTS 2 (weights={KITTEN2_WEIGHTS}, "
            f"decoder={KITTEN2_DECODER}, device: {self.device})…"
        )
        self._model = self._construct()
        self._loaded = True
        self._marker.touch(exist_ok=True)
        status("✓ KittenTTS 2 loaded!")

    def load(self, progress_cb: Optional[Callable] = None) -> None:
        self.load_model(progress_cb)

    def unload_model(self, progress_cb: Optional[Callable] = None) -> None:
        def status(msg: str) -> None:
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        if not self._loaded:
            return
        status("Releasing KittenTTS 2 from memory…")
        self._model  = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        status("✓ KittenTTS 2 unloaded.")

    def unload(self) -> None:
        self.unload_model()

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
        if not self._loaded or self._model is None:
            raise RuntimeError("KittenTTS 2 model not loaded. Click 'Load model' first.")

        def status(msg: str) -> None:
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        text = (text or "").strip()
        if not text:
            raise ValueError("Empty text passed to KittenTTS 2.")

        voice     = str(kwargs.get("kitten2_voice") or "Bella")
        preset    = str(kwargs.get("kitten2_preset") or "default")
        normalize = bool(kwargs.get("kitten2_normalize", True))

        gen_kw: Dict[str, Any] = {}
        if preset != "default":
            gen_kw["preset"] = preset

        if reference_audio_path and Path(reference_audio_path).exists():
            dur = _wav_duration(reference_audio_path)
            if dur is not None and not (_REF_MIN_S <= dur <= _REF_MAX_S):
                status(
                    f"⚠ Reference is {dur:.1f}s — KittenTTS 2 expects "
                    f"{_REF_MIN_S:.0f}–{_REF_MAX_S:.0f}s of a single speaker."
                )
            gen_kw["reference"] = str(reference_audio_path)
            status(f"Cloning voice from {Path(reference_audio_path).name}…")
        else:
            if reference_audio_path:
                status("⚠ Reference audio not found — using the built-in voice.")
            gen_kw["voice"] = voice
            if voice in KITTEN2_LANGUAGE_VOICES:
                normalize = False
            status(f"Synthesizing with voice '{voice}'…")

        gen_kw["normalize"] = normalize

        try:
            audio = self._model.generate(text, **gen_kw)
        except Exception as e:
            raise RuntimeError(f"KittenTTS 2 inference failed:\n{e}")

        if audio is None:
            raise RuntimeError("KittenTTS 2 returned no audio.")

        arr = _to_numpy(audio)
        if arr.size == 0:
            raise RuntimeError("KittenTTS 2 returned empty audio.")

        sr = int(getattr(self._model, "sample_rate", DEFAULT_SR) or DEFAULT_SR)
        status(f"✓ Generated {len(arr) / sr:.1f}s of audio")
        return arr, sr
