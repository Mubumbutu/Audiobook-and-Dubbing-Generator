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
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

_ROOT_DIR  = Path(__file__).parent.parent
_MODEL_DIR = _ROOT_DIR / "models" / "xtts_v2"
_HF_REPO   = "coqui/XTTS-v2"
DEFAULT_SR = 24000

XTTS_LANGUAGES: List[Tuple[str, str]] = [
    ("en",    "English"),
    ("pl",    "Polish"),
    ("de",    "German"),
    ("fr",    "French"),
    ("es",    "Spanish"),
    ("it",    "Italian"),
    ("pt",    "Portuguese"),
    ("ru",    "Russian"),
    ("nl",    "Dutch"),
    ("cs",    "Czech"),
    ("tr",    "Turkish"),
    ("ar",    "Arabic"),
    ("zh-cn", "Chinese"),
    ("ja",    "Japanese"),
    ("ko",    "Korean"),
    ("hu",    "Hungarian"),
    ("hi",    "Hindi"),
]

_XTTS_PARAMS: List[Dict[str, Any]] = [
    {
        "key":     "language",
        "type":    "combo",
        "label":   "Language",
        "options": XTTS_LANGUAGES,
        "default": "en",
        "tip":     "Target language for synthesis (must match input text)",
    },
    {
        "key":     "temperature",
        "type":    "slider",
        "label":   "Temperature",
        "min":     0.1,
        "max":     1.0,
        "default": 0.7,
        "tip":     "Generation randomness / expressiveness (0.7 = default)",
    },
    {
        "key":     "speed",
        "type":    "slider",
        "label":   "Speed",
        "min":     0.5,
        "max":     2.0,
        "default": 1.0,
        "tip":     "Speech speed (1.0 = normal, <1 slower, >1 faster)",
    },
    {
        "key":     "repetition_penalty",
        "type":    "slider",
        "label":   "Repetition penalty",
        "min":     1.0,
        "max":     20.0,
        "default": 10.0,
        "tip":     "Penalizes repeated tokens (default 10.0 for XTTS)",
    },
    {
        "key":     "top_p",
        "type":    "slider",
        "label":   "Top-P",
        "min":     0.0,
        "max":     1.0,
        "default": 0.85,
        "tip":     "Nucleus sampling probability (0.85 = default)",
    },
    {
        "key":     "gpt_cond_len",
        "type":    "spinbox",
        "label":   "Conditioning length (s)",
        "min":     3,
        "max":     30,
        "default": 6,
        "step":    1,
        "tip":     "Seconds of reference audio used for voice conditioning (6-12 recommended)",
    },
]


@register_backend
class XttsV2Backend(TTSBackend):

    def __init__(self):
        self.model_dir   = _MODEL_DIR
        self._model      = None
        self._config     = None
        self._loaded     = False
        self._device     = "cuda" if torch.cuda.is_available() else "cpu"
        self._latent_cache: Optional[Tuple[str, int, Any, Any]] = None

    @property
    def name(self) -> str:
        return "XTTS v2"

    @property
    def model_id(self) -> str:
        return "xtts_v2"

    @property
    def display_name(self) -> str:
        return "🐸 XTTS v2"

    @property
    def default_sample_rate(self) -> int:
        return DEFAULT_SR

    @property
    def auth_required(self) -> bool:
        return False

    @property
    def venv_names(self) -> List[str]:
        return ["venv_xttsv2"]

    @property
    def download_repo(self) -> str:
        return _HF_REPO

    @property
    def download_size(self) -> str:
        return "~1.8 GB"

    @property
    def header_icon(self) -> str:
        return "🐸"

    @property
    def header_title(self) -> str:
        return "XTTS v2"

    @property
    def whisper_incompatible(self) -> bool:
        return False

    @property
    def pyannote_incompatible(self) -> bool:
        return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return self._device

    @property
    def device_info(self) -> str:
        if self._device == "cuda" and torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
                mem  = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
                return f"CUDA — {name} ({mem} GB)"
            except Exception:
                return "CUDA"
        return "CPU (slow — GPU strongly recommended)"

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return _XTTS_PARAMS

    def is_available(self) -> bool:
        return (
            (_MODEL_DIR / "config.json").exists()
            and (_MODEL_DIR / "model.pth").exists()
        )

    def download(self, model_dir: Path, progress_cb: Optional[Callable] = None) -> None:
        def status(msg: str) -> None:
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        model_dir.mkdir(parents=True, exist_ok=True)
        status(f"Downloading {_HF_REPO} (~1.8 GB)…")
        os.environ["COQUI_TOS_AGREED"] = "1"
        snapshot_download(repo_id=_HF_REPO, local_dir=str(model_dir))
        status("XTTS v2 model downloaded successfully!")

    def load_model(self, progress_cb: Optional[Callable] = None) -> None:
        def status(msg: str) -> None:
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        if not self.is_available():
            raise RuntimeError(
                f"XTTS v2 model files not found in:\n{_MODEL_DIR}\n\n"
                "Click 'Download model' to download (~1.8 GB) from HuggingFace."
            )

        status("Importing XTTS v2 modules…")
        os.environ["COQUI_TOS_AGREED"] = "1"
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import coqui-tts:\n{e}\n\n"
                "Make sure 'coqui-tts' is installed in venv_xttsv2."
            )

        config_path = _MODEL_DIR / "config.json"
        status(f"Loading config from {config_path}…")
        self._config = XttsConfig()
        self._config.load_json(str(config_path))

        status(f"Loading XTTS v2 model (device: {self._device})…")
        status("  (first load may take 1–3 min and uses ~3 GB VRAM)")
        self._model = Xtts.init_from_config(self._config)
        self._model.load_checkpoint(
            self._config,
            checkpoint_dir=str(_MODEL_DIR),
            eval=True,
            use_deepspeed=False,
        )

        if self._device == "cuda":
            self._model.cuda()
        else:
            self._model.cpu()

        self._loaded        = True
        self._latent_cache  = None
        status("✓ XTTS v2 model loaded!")

    def load(self, progress_cb: Optional[Callable] = None) -> None:
        self.load_model(progress_cb)

    def unload_model(self, progress_cb: Optional[Callable] = None) -> None:
        def status(msg: str) -> None:
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        if not self._loaded:
            return

        status("Releasing XTTS v2 model from memory…")
        self._model         = None
        self._config        = None
        self._latent_cache  = None
        self._loaded        = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        status("✓ XTTS v2 model unloaded.")

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
        language: str = "en",
        temperature: float = 0.7,
        speed: float = 1.0,
        repetition_penalty: float = 10.0,
        top_p: float = 0.85,
        gpt_cond_len: int = 6,
        progress_cb: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        if not self._loaded or self._model is None:
            raise RuntimeError("XTTS v2 model not loaded. Click 'Load model' first.")

        def status(msg: str) -> None:
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        if not reference_audio_path or not Path(reference_audio_path).exists():
            raise RuntimeError(
                "XTTS v2 requires a reference audio file for voice cloning.\n"
                "Upload a reference WAV file (6–30 seconds of clean speech)."
            )

        gpt_cond_len = int(gpt_cond_len)
        cache_key    = (reference_audio_path, gpt_cond_len)

        if (
            self._latent_cache is None
            or self._latent_cache[0] != reference_audio_path
            or self._latent_cache[1] != gpt_cond_len
        ):
            status(f"Computing speaker latents from {Path(reference_audio_path).name}…")
            try:
                gpt_cond_latent, speaker_embedding = (
                    self._model.get_conditioning_latents(
                        audio_path=[reference_audio_path],
                        gpt_cond_len=gpt_cond_len,
                    )
                )
                self._latent_cache = (
                    reference_audio_path,
                    gpt_cond_len,
                    gpt_cond_latent,
                    speaker_embedding,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to compute speaker latents:\n{e}")
        else:
            status("Using cached speaker latents…")

        _, _, gpt_cond_latent, speaker_embedding = self._latent_cache

        status(f"Synthesizing in {language}…")
        try:
            result = self._model.inference(
                text,
                language,
                gpt_cond_latent,
                speaker_embedding,
                temperature=float(temperature),
                speed=float(speed),
                repetition_penalty=float(repetition_penalty),
                top_p=float(top_p),
            )
        except Exception as e:
            raise RuntimeError(f"XTTS v2 inference failed:\n{e}")

        wav = result.get("wav")
        if wav is None:
            raise RuntimeError("XTTS v2 returned no audio.")

        if isinstance(wav, torch.Tensor):
            audio = wav.squeeze().detach().cpu().float().numpy()
        else:
            audio = np.array(wav, dtype=np.float32).flatten()

        if audio.size == 0:
            raise RuntimeError("XTTS v2 returned empty audio.")

        status(f"✓ Generated {len(audio) / DEFAULT_SR:.1f}s of audio")
        return audio, DEFAULT_SR
