from __future__ import annotations

from tts_backends import (
    TTSBackend, SynthesisRequest, SynthesisResult, InferenceError,
    register_backend, is_rocm,
)

import gc
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

_ROOT_DIR = Path(__file__).parent.parent

MAYA2_REPO = "maya-research/maya2-native"
MAYA2_DIR = _ROOT_DIR / "models" / "maya2_native"
SAMPLE_RATE = 24000
MAYA2_CACHE_DIR = _ROOT_DIR / "models" / "maya2_cache"
MAYA2_MIMI_REPO = "kyutai/mimi"
MAYA2_ECAPA_REPO = "speechbrain/spkrec-ecapa-voxceleb"

MAYA2_VOICES: List[Tuple[str, str]] = [
    ("Ahana", "Ahana — young female, bright"),
    ("Vaidehi", "Vaidehi — adult female, calm"),
    ("Malini", "Malini — mature female, narrator"),
    ("Charvi", "Charvi — young female, energetic"),
    ("Jahnavi", "Jahnavi — adult female, warm"),
    ("Anirban", "Anirban — adult male, narrator"),
    ("Nakul", "Nakul — young male, bright"),
    ("Vasant", "Vasant — mature male, narrator"),
    ("Siddhant", "Siddhant — adult male, professional"),
    ("Satvik", "Satvik — young male, warm"),
    ("Abhay", "Abhay — adult male, conversational"),
    ("Nivedita", "Nivedita — adult female, conversational"),
    ("Ishwari", "Ishwari — adult female, conversational"),
    ("Ritvik", "Ritvik — adult male, conversational"),
    ("Vrinda", "Vrinda — young female, conversational"),
    ("Sreekar", "Sreekar — adult male, conversational"),
    ("Kayal", "Kayal — adult female, conversational"),
    ("Shaunak", "Shaunak — young male, conversational"),
    ("Akshat", "Akshat — adult male, conversational"),
    ("Tanushree", "Tanushree — adult female, conversational"),
]

MAYA2_LANGUAGES: List[Tuple[str, str]] = [
    ("as", "Assamese"), ("bn", "Bengali"), ("en", "English"),
    ("gu", "Gujarati"), ("hi", "Hindi"), ("kn", "Kannada"),
    ("ml", "Malayalam"), ("mr", "Marathi"), ("or", "Odia"),
    ("pa", "Punjabi"), ("ta", "Tamil"), ("te", "Telugu"),
]

_SCRIPT_RANGES: List[Tuple[str, int, int]] = [
    ("hi", 0x0900, 0x097F),
    ("bn", 0x0980, 0x09FF),
    ("pa", 0x0A00, 0x0A7F),
    ("gu", 0x0A80, 0x0AFF),
    ("or", 0x0B00, 0x0B7F),
    ("ta", 0x0B80, 0x0BFF),
    ("te", 0x0C00, 0x0C7F),
    ("kn", 0x0C80, 0x0CFF),
    ("ml", 0x0D00, 0x0D7F),
]

_VOICE_KEY = "maya2_voice"
_LANGUAGE_KEY = "maya2_language"

_MAYA2_PARAMS: List[Dict[str, Any]] = [
    {
        "key": _VOICE_KEY, "type": "combo", "label": "Voice",
        "options": MAYA2_VOICES, "default": MAYA2_VOICES[0][0],
        "tip": "Built-in voice (ignored when a reference audio is provided)",
    },
    {
        "key": _LANGUAGE_KEY, "type": "combo", "label": "Language",
        "options": [("auto", "Auto-detect (by script)")] + MAYA2_LANGUAGES,
        "default": "auto",
        "tip": "Output language; auto-detect cannot tell Hindi from Marathi (both use Devanagari)",
    },
]


def _configure_cache() -> None:
    hub_dir = MAYA2_CACHE_DIR / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(MAYA2_CACHE_DIR)
    os.environ["HF_HUB_CACHE"] = str(hub_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_dir)
    try:
        from huggingface_hub import constants
        constants.HF_HOME = str(MAYA2_CACHE_DIR)
        constants.HF_HUB_CACHE = str(hub_dir)
        constants.HUGGINGFACE_HUB_CACHE = str(hub_dir)
    except Exception as e:
        logger.warning(f"Could not update huggingface_hub cache constants: {e}")

def detect_language(text: str) -> str:
    counts: Dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for code, lo, hi in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if not counts:
        return "en"
    best = max(counts, key=counts.get)
    if best == "bn" and any(ch in text for ch in "\u09F0\u09F1"):
        return "as"
    return best


@register_backend
class Maya2NativeBackend(TTSBackend):

    def __init__(self):
        self.model_dir = MAYA2_DIR
        self._tts = None
        self._loaded = False
        self._registered: Dict[str, str] = {}

    @property
    def name(self) -> str:
        return "Maya 2 Native 2.5B"

    @property
    def model_id(self) -> str:
        return "maya2_native"

    @property
    def display_name(self) -> str:
        return "🌏 Maya 2 Native (12 languages)"

    @property
    def default_sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def auth_required(self) -> bool:
        return False

    @property
    def venv_names(self) -> List[str]:
        return ["venv_maya2"]

    @property
    def download_repo(self) -> str:
        return MAYA2_REPO

    @property
    def download_size(self) -> str:
        return "~5 GB"

    @property
    def header_icon(self) -> str:
        return "🌏"

    @property
    def header_title(self) -> str:
        return "Maya 2 Native"

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return _MAYA2_PARAMS

    @property
    def speaker_voice_options(self) -> List[Tuple[str, str]]:
        return MAYA2_VOICES

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device_info(self) -> str:
        if torch.cuda.is_available() and not is_rocm():
            try:
                name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
                return f"CUDA — {name} ({mem} GB)"
            except Exception:
                return "CUDA"
        return "No NVIDIA GPU — Maya 2 Native requires CUDA"

    @property
    def _marker(self) -> Path:
        return self.model_dir.parent / f".{self.model_id}_ok"

    def is_available(self) -> bool:
        return self._marker.exists()

    def load(self, progress_cb: Optional[Callable] = None) -> None:
        self.load_model(progress_cb)

    def unload(self) -> None:
        self.unload_model()

    def download(
        self,
        model_dir: Path,
        progress_cb: Optional[Callable] = None,
    ) -> None:
        _configure_cache()
        model_dir.mkdir(parents=True, exist_ok=True)
        if progress_cb:
            progress_cb(f"Downloading {MAYA2_REPO}…", 0.0)
        snapshot_download(repo_id=MAYA2_REPO, local_dir=str(model_dir))
        for i, repo in enumerate((MAYA2_MIMI_REPO, MAYA2_ECAPA_REPO)):
            if progress_cb:
                progress_cb(f"Downloading {repo}…", 0.6 + 0.2 * i)
            try:
                snapshot_download(
                    repo_id=repo,
                    cache_dir=str(MAYA2_CACHE_DIR / "hub"),
                )
            except Exception as e:
                logger.warning(f"Pre-download of {repo} failed: {e}")
                if progress_cb:
                    progress_cb(f"Warning: could not pre-download {repo}: {e}", 0.9)
        self._marker.touch(exist_ok=True)
        if progress_cb:
            progress_cb("Model downloaded successfully!", 1.0)

    def load_model(self, progress_cb: Optional[Callable] = None):
        def status(msg):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        if not torch.cuda.is_available() or is_rocm():
            raise InferenceError(
                "Maya 2 Native requires an NVIDIA GPU with CUDA 12.8 support."
            )

        _configure_cache()

        status("Importing maya3…")
        try:
            from maya3 import Maya3TTS
        except ImportError as e:
            raise InferenceError(
                f"Failed to import maya3:\n{e}\n\n"
                "Run maya2_install.bat to set up venv_maya2."
            )

        sources = [MAYA2_REPO]
        if self.model_dir.exists():
            sources.insert(0, str(self.model_dir))

        status(f"Loading {self.name} | device: cuda")
        last_error: Optional[Exception] = None
        self._tts = None
        for source in sources:
            try:
                self._tts = Maya3TTS.from_pretrained(source, max_seats=1)
                break
            except Exception as e:
                last_error = e
                logger.warning(f"Maya 2 Native load from {source} failed: {e}")
        if self._tts is None:
            raise InferenceError(f"Failed to load {self.name}:\n{last_error}")

        self._registered = {}
        self._loaded = True
        status(f"✓ {self.name} loaded!")

    def unload_model(self, progress_cb: Optional[Callable] = None):
        def status(msg):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        if not self._loaded:
            status("Model is not loaded — nothing to release.")
            return

        status(f"Releasing {self.name} from memory…")
        try:
            self._tts.close()
        except Exception as e:
            logger.warning(f"Maya 2 Native close failed: {e}")
        self._tts = None
        self._registered = {}
        self._loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        status(f"✓ {self.name} unloaded")

    def _register_reference(self, audio_path: str, ref_text: str) -> str:
        import soundfile as sf

        try:
            duration = float(sf.info(audio_path).duration)
        except Exception:
            duration = 0.0
        if duration > 30.0:
            raise InferenceError(
                f"Reference audio is {duration:.1f} s long. Maya 2 Native accepts at most 30 s; "
                "3-10 s works best. Trim the reference audio and transcribe it again."
            )
        if duration > 10.0:
            logger.warning(
                f"Reference audio is {duration:.1f} s long; 3-10 s usually gives the best clone."
            )

        stat = os.stat(audio_path)
        digest = hashlib.sha1(
            f"{os.path.abspath(audio_path)}|{stat.st_mtime_ns}|{ref_text}".encode("utf-8")
        ).hexdigest()[:12]
        voice_name = f"ref_{digest}"
        if voice_name not in self._registered:
            self._tts.register_voice(
                voice_name,
                ref_audio=audio_path,
                ref_text=ref_text,
                lang=detect_language(ref_text),
            )
            self._registered[voice_name] = audio_path
        return voice_name

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        progress_cb: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        if not self._loaded or self._tts is None:
            raise InferenceError("Model is not loaded. Call load_model() first.")

        def status(msg):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        language_codes = {code for code, _ in MAYA2_LANGUAGES}
        requested_language = kwargs.get(_LANGUAGE_KEY)
        language = requested_language if requested_language in language_codes else detect_language(text)

        has_ref_audio = bool(reference_audio_path and os.path.exists(reference_audio_path))
        has_ref_text = bool(reference_text and reference_text.strip())

        if has_ref_audio and not has_ref_text:
            raise InferenceError(
                "Maya 2 Native requires the exact transcript of the reference audio "
                "(reference text) to perform voice cloning. Provide the reference "
                "text or remove the reference audio to use a built-in voice."
            )

        try:
            if has_ref_audio:
                status(f"Voice Cloning mode — ref: {Path(reference_audio_path).name}")
                voice_name = self._register_reference(reference_audio_path, reference_text.strip())
            else:
                names = {code.lower(): code for code, _ in MAYA2_VOICES}
                candidate = str(reference_text or "").strip().lower()
                chosen = candidate if candidate in names else str(kwargs.get(_VOICE_KEY) or "").lower()
                voice_name = names.get(chosen, MAYA2_VOICES[0][0])
                status(f"Built-in voice: {voice_name}")
        except Exception as e:
            logger.error(f"Maya 2 Native voice preparation error: {e}", exc_info=True)
            raise InferenceError(f"Failed to prepare voice for Maya 2 Native:\n{e}")

        status(f"Generating speech… (language: {language})")
        try:
            audio = self._tts.generate(text, lang=language, voice=voice_name, normalize=True)
        except Exception as e:
            logger.error(f"Maya 2 Native inference error: {e}", exc_info=True)
            raise InferenceError(f"Inference failed:\n{e}")

        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        if arr.size < int(SAMPLE_RATE * 0.05):
            raise InferenceError(
                f"Maya 2 Native returned audio that is too short ({arr.size} samples)."
            )

        status(f"✓ Generated {len(arr) / SAMPLE_RATE:.1f}s of audio")
        return arr, SAMPLE_RATE

    def synthesize(
        self,
        request: SynthesisRequest,
        progress_cb: Optional[Callable] = None,
    ) -> SynthesisResult:
        extra = getattr(request, "extra", None) or {}
        audio, sr = self.generate(
            text=request.text,
            reference_audio_path=request.reference_audio,
            reference_text=request.reference_text,
            progress_cb=progress_cb,
            **{
                _VOICE_KEY: request.speaker_id or extra.get(_VOICE_KEY),
                _LANGUAGE_KEY: extra.get(_LANGUAGE_KEY),
            },
        )
        return SynthesisResult(audio=audio, sample_rate=sr, duration_s=len(audio) / sr)
