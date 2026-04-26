from __future__ import annotations

from tts_backends import (
    TTSBackend, SynthesisRequest, SynthesisResult, register_backend, is_rocm,
)

import gc
import logging
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

_ROOT_DIR = Path(__file__).parent.parent
MODEL_REPO = "k2-fsa/OmniVoice"
DEFAULT_SR  = 24000

_OMNIVOICE_PARAMS: List[Dict[str, Any]] = [
    {
        "key": "num_step", "type": "spinbox", "label": "Diffusion Steps",
        "min": 4, "max": 100, "default": 32, "step": 4,
        "tip": "Diffusion steps (32 = quality, 16 = faster)",
    },
    {
        "key": "speed", "type": "slider", "label": "Speed",
        "min": 0.5, "max": 2.0, "default": 1.0,
        "tip": "Speaking speed (1.0 = normal, >1 faster, <1 slower)",
    },
    {
        "key": "duration", "type": "dspinbox", "label": "Duration (0=auto)",
        "min": 0.0, "max": 60.0, "default": 0.0, "step": 0.5,
        "tip": "Fixed output duration in seconds (0 = automatic)",
    },
    {
        "key": "seed", "type": "spinbox", "label": "Seed (0=random)",
        "min": 0, "max": 99999, "default": 0, "step": 1,
        "tip": "Random seed for reproducibility (0 = random each time)",
    },
]


class InferenceError(Exception):
    pass


class _OmniVoiceBase(TTSBackend):

    def __init__(self, model_dir: Path, dtype: torch.dtype):
        self.model_dir = Path(model_dir)
        self.dtype     = dtype
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype in (torch.float16, torch.bfloat16) and (self.device == "cpu" or is_rocm()):
            self.dtype = torch.float32
        self._model  = None
        self._loaded = False

    @property
    def default_sample_rate(self) -> int:
        return DEFAULT_SR

    @property
    def auth_required(self) -> bool:
        return False

    @property
    def venv_names(self) -> List[str]:
        return ["venv_omnivoice"]

    @property
    def header_icon(self) -> str:
        return "🌐"

    @property
    def header_title(self) -> str:
        return "OmniVoice"

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return _OMNIVOICE_PARAMS

    def is_available(self) -> bool:
        return self._marker.exists()

    def load(self, progress_cb: Optional[Callable] = None) -> None:
        self.load_model(progress_cb)

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

    def download(
        self,
        model_dir: Path,
        progress_cb: Optional[Callable] = None,
    ) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        if progress_cb:
            progress_cb(f"Downloading {MODEL_REPO}…", 0.0)
        snapshot_download(repo_id=MODEL_REPO, local_dir=str(model_dir))
        self._marker.touch(exist_ok=True)
        if progress_cb:
            progress_cb("Model downloaded successfully!", 1.0)

    @property
    def _marker(self) -> Path:
        return self.model_dir.parent / f".{self.model_id}_ok"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device_info(self) -> str:
        if self.device == "cuda":
            try:
                name = torch.cuda.get_device_name(0)
                mem  = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
                return f"CUDA — {name} ({mem} GB)"
            except Exception:
                return "CUDA"
        return "CPU (no GPU — generation will be slow)"

    def load_model(self, progress_cb: Optional[Callable] = None):
        def status(msg):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        if not self.model_dir.exists():
            raise InferenceError(
                f"Model directory does not exist:\n{self.model_dir}\n"
                "Download the model using the 'Download model' button."
            )

        status("Importing OmniVoice…")
        try:
            from omnivoice import OmniVoice
        except ImportError as e:
            raise InferenceError(
                f"Failed to import omnivoice:\n{e}\n\n"
                "Run: pip install omnivoice"
            )

        if self.device == "cuda":
            device_map = "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_map = "mps"
        else:
            device_map = "cpu"

        status(
            f"Loading OmniVoice from {self.model_dir} "
            f"| dtype: {self.dtype} | device: {device_map}"
        )
        status("  (first load may take 1–3 min)")
        try:
            self._model = OmniVoice.from_pretrained(
                str(self.model_dir),
                device_map=device_map,
                dtype=self.dtype,
            )
        except Exception as e:
            raise InferenceError(f"Failed to load OmniVoice model:\n{e}")

        self._loaded = True
        status("✓ OmniVoice model loaded!")

    def unload_model(self, progress_cb: Optional[Callable] = None):
        def status(msg):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        if not self._loaded:
            status("Model is not loaded — nothing to release.")
            return

        status("Releasing OmniVoice model from memory…")
        self._model  = None
        self._loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        status("✓ OmniVoice model unloaded")

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        voice_mode: str = "cloning",
        instruct: str = "",
        num_step: int = 32,
        speed: float = 1.0,
        duration: float = 0.0,
        seed: int = 0,
        progress_cb: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        if not self._loaded or self._model is None:
            raise InferenceError("Model is not loaded. Call load_model() first.")

        def status(msg):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        actual_seed = seed if seed > 0 else random.randint(1, 99999)
        torch.manual_seed(actual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(actual_seed)

        generate_kwargs: Dict[str, Any] = {
            "text":     text,
            "num_step": int(num_step),
        }

        actual_speed = float(speed)
        if abs(actual_speed - 1.0) > 0.01:
            generate_kwargs["speed"] = actual_speed

        actual_duration = float(duration)
        if actual_duration > 0.0:
            generate_kwargs["duration"] = actual_duration

        if voice_mode == "design" and instruct:
            status(f"Voice Design mode — attributes: {instruct[:80]}")
            generate_kwargs["instruct"] = instruct
        elif voice_mode == "cloning" and reference_audio_path and os.path.exists(reference_audio_path):
            status(f"Voice Cloning mode — ref: {Path(reference_audio_path).name}")
            generate_kwargs["ref_audio"] = reference_audio_path
            if reference_text:
                generate_kwargs["ref_text"] = reference_text
        else:
            status("Auto Voice mode — model selects voice automatically")

        status("Generating speech…")
        try:
            audio_tensors = self._model.generate(**generate_kwargs)
        except Exception as e:
            logger.error(f"OmniVoice inference error: {e}", exc_info=True)
            raise InferenceError(f"Inference failed:\n{e}")

        if not audio_tensors:
            raise InferenceError("OmniVoice returned no audio.")

        tensor = audio_tensors[0]
        if isinstance(tensor, torch.Tensor):
            arr = tensor.squeeze().detach().cpu().float().numpy()
        else:
            arr = np.asarray(tensor, dtype=np.float32).flatten()

        min_samples = int(DEFAULT_SR * 0.05)
        if arr.size < min_samples:
            raise InferenceError(
                f"OmniVoice returned audio that is too short "
                f"({arr.size} samples / {arr.size / DEFAULT_SR:.3f}s). "
                f"Try a different seed or increase diffusion steps."
            )

        if arr.max() > 1.5 or arr.min() < -1.5:
            arr = arr / 32768.0

        status(f"✓ Generated {len(arr) / DEFAULT_SR:.1f}s of audio")
        return arr, DEFAULT_SR


@register_backend
class OmniVoiceF16Backend(_OmniVoiceBase):

    def __init__(self):
        super().__init__(_ROOT_DIR / "models" / "omnivoice", torch.float16)

    @property
    def name(self) -> str:
        return "OmniVoice float16"

    @property
    def model_id(self) -> str:
        return "omnivoice_f16"

    @property
    def display_name(self) -> str:
        return "🌐 OmniVoice float16"

    @property
    def download_repo(self) -> str:
        return MODEL_REPO

    @property
    def download_size(self) -> str:
        return "~2.5 GB"


@register_backend
class OmniVoiceF32Backend(_OmniVoiceBase):

    def __init__(self):
        super().__init__(_ROOT_DIR / "models" / "omnivoice", torch.float32)

    @property
    def name(self) -> str:
        return "OmniVoice float32"

    @property
    def model_id(self) -> str:
        return "omnivoice_f32"

    @property
    def display_name(self) -> str:
        return "🔵 OmniVoice float32"

    @property
    def download_repo(self) -> str:
        return MODEL_REPO

    @property
    def download_size(self) -> str:
        return "~3.5 GB"
