from __future__ import annotations

from tts_backends import TTSBackend, SynthesisRequest, SynthesisResult, register_backend

import gc
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

_ROOT_DIR = Path(__file__).parent.parent
DEFAULT_SR = 16000
_MODEL_DIR = _ROOT_DIR / "models" / "supertonic"

SUPERTONIC_VOICES: List[Tuple[str, str]] = [
    ("M1", "M1 — Male 1"),
    ("M2", "M2 — Male 2"),
    ("M3", "M3 — Male 3"),
    ("M4", "M4 — Male 4"),
    ("M5", "M5 — Male 5"),
    ("F1", "F1 — Female 1"),
    ("F2", "F2 — Female 2"),
    ("F3", "F3 — Female 3"),
    ("F4", "F4 — Female 4"),
    ("F5", "F5 — Female 5"),
]

_VOICE_CODES: frozenset = frozenset(v for v, _ in SUPERTONIC_VOICES)

_SUPERTONIC_LANGS: List[Tuple[str, str]] = [
    ("en", "English"),
    ("ar", "Arabic"),
    ("bg", "Bulgarian"),
    ("cs", "Czech"),
    ("da", "Danish"),
    ("de", "German"),
    ("el", "Greek"),
    ("es", "Spanish"),
    ("et", "Estonian"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("hi", "Hindi"),
    ("hr", "Croatian"),
    ("hu", "Hungarian"),
    ("id", "Indonesian"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("lt", "Lithuanian"),
    ("lv", "Latvian"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("sk", "Slovak"),
    ("sl", "Slovenian"),
    ("sv", "Swedish"),
    ("tr", "Turkish"),
    ("uk", "Ukrainian"),
    ("vi", "Vietnamese"),
]

_GENERATION_PARAMS: List[Dict[str, Any]] = [
    {
        "key": "voice_name",
        "type": "combo",
        "label": "Voice",
        "options": SUPERTONIC_VOICES,
        "default": "M1",
    },
    {
        "key": "lang",
        "type": "combo",
        "label": "Language",
        "options": _SUPERTONIC_LANGS,
        "default": "en",
    },
]


@register_backend
class SupertonicBackend(TTSBackend):

    def __init__(self):
        self.model_dir = _MODEL_DIR
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tts = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "Supertonic 3"

    @property
    def model_id(self) -> str:
        return "supertonic_3"

    @property
    def display_name(self) -> str:
        return "⚡ Supertonic 3"

    @property
    def default_sample_rate(self) -> int:
        return DEFAULT_SR

    @property
    def auth_required(self) -> bool:
        return False

    @property
    def venv_names(self) -> List[str]:
        return ["venv_supertonic"]

    @property
    def download_repo(self) -> str:
        return "Supertone/supertonic-3"

    @property
    def download_size(self) -> str:
        return "~400 MB"

    @property
    def header_icon(self) -> str:
        return "⚡"

    @property
    def header_title(self) -> str:
        return "Supertonic 3"

    @property
    def whisper_incompatible(self) -> bool:
        return True

    @property
    def pyannote_incompatible(self) -> bool:
        return False

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return _GENERATION_PARAMS

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device_info(self) -> str:
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
                return f"CUDA — {name} ({mem} GB)"
            except Exception:
                return "CUDA"
        return "CPU"

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
            voice_name="M1",
            lang="en",
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

    def load_model(self, progress_cb: Optional[Callable] = None) -> None:
        def status(msg: str) -> None:
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        status("Importing Supertonic…")
        try:
            from supertonic import TTS
        except ImportError as e:
            raise Exception(
                f"Failed to import supertonic: {e}\n"
                "Make sure 'supertonic' is installed in venv_supertonic"
            )

        status(f"Loading Supertonic 3 (device: {self.device})…")

        if self.model_dir.exists() and any(self.model_dir.iterdir()):
            status(f"  Using local model from {self.model_dir}")
            try:
                self._tts = TTS(model_path=str(self.model_dir), auto_download=False)
            except TypeError:
                try:
                    self._tts = TTS(auto_download=False)
                except Exception:
                    status("  Falling back to auto-download…")
                    self._tts = TTS(auto_download=True)
            except Exception as e:
                status(f"  Local path load failed ({e}), falling back to auto-download…")
                self._tts = TTS(auto_download=True)
        else:
            status("  Model directory empty — using auto-download from Hugging Face…")
            self._tts = TTS(auto_download=True)

        self._loaded = True
        status("✓ Supertonic 3 loaded!")

    def unload_model(self, progress_cb: Optional[Callable] = None) -> None:
        def status(msg: str) -> None:
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        if not self._loaded:
            return

        status("Releasing Supertonic 3 model…")
        self._tts = None
        self._loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        status("✓ Supertonic 3 model unloaded")

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        voice_name: str = "M1",
        lang: str = "en",
        progress_cb: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        if not self._loaded or self._tts is None:
            raise Exception("Model not loaded. Call load_model() first.")

        def status(msg: str) -> None:
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        effective_voice = voice_name or "M1"
        if reference_text and reference_text.strip().upper() in _VOICE_CODES:
            effective_voice = reference_text.strip().upper()

        if effective_voice not in _VOICE_CODES:
            logger.warning(f"Unknown voice '{effective_voice}', falling back to M1")
            effective_voice = "M1"

        status(f"Synthesizing with voice {effective_voice}, lang={lang}…")

        try:
            voice_style = self._tts.get_voice_style(voice_name=effective_voice)
            wav, duration = self._tts.synthesize(text, voice_style=voice_style, lang=lang)
        except Exception as e:
            raise Exception(f"Supertonic synthesis failed: {e}")

        try:
            sr = int(getattr(self._tts, "sample_rate", DEFAULT_SR))
        except Exception:
            sr = DEFAULT_SR

        audio = _wav_to_float32(wav)

        if audio is None or len(audio) == 0:
            raise Exception("Supertonic returned empty audio.")

        status(f"✓ Generated {len(audio) / max(1, sr):.1f}s of audio")
        return audio, sr


def _wav_to_float32(wav: Any) -> np.ndarray:
    if hasattr(wav, "numpy"):
        arr = wav.numpy()
    elif torch.is_tensor(wav):
        arr = wav.detach().cpu().numpy()
    else:
        arr = np.asarray(wav)

    arr = arr.astype(np.float32).flatten()

    if arr.size > 0 and (arr.max() > 1.5 or arr.min() < -1.5):
        arr = arr / 32768.0

    return arr
