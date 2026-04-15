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
DEFAULT_SR = 48000
_MODEL_DIR = _ROOT_DIR / "models" / "voxcpm2"


class _VoxCPM2Base(TTSBackend):

    def __init__(self):
        self.model_dir = _MODEL_DIR
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
        return ["venv_voxcpm2"]

    @property
    def header_icon(self) -> str:
        return "🔊"

    @property
    def header_title(self) -> str:
        return "VoxCPM2"

    @property
    def download_repo(self) -> str:
        return "openbmb/VoxCPM2"

    @property
    def download_size(self) -> str:
        return "~8 GB"

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

        status("Importing VoxCPM…")
        try:
            from voxcpm import VoxCPM
        except ImportError as e:
            raise Exception(f"Failed to import voxcpm: {e}\nMake sure 'voxcpm' is installed in venv_voxcpm2")

        status(f"Loading VoxCPM2 from {self.model_dir} on {self.device}")

        self._model = VoxCPM.from_pretrained(
            str(self.model_dir),
            load_denoiser=False,
        )

        self._loaded = True
        status("✓ VoxCPM2 model loaded!")

    def unload_model(self, progress_cb: Optional[Callable] = None):
        def status(msg):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        if not self._loaded:
            return

        status("Releasing VoxCPM2 model…")
        self._model = None
        self._loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        status("✓ VoxCPM2 model unloaded")

    def generate(self, text: str, reference_audio_path: Optional[str] = None,
                 reference_text: Optional[str] = None, progress_cb: Optional[Callable] = None, **kwargs) -> Tuple[np.ndarray, int]:
        raise NotImplementedError


# ==================== REGISTERED BACKENDS ====================

@register_backend
class VoxCPM2VoiceDesign(_VoxCPM2Base):

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "voxcpm2-voicedesign"

    @property
    def model_id(self) -> str:
        return "voxcpm2-voicedesign"

    @property
    def display_name(self) -> str:
        return "🔊 VoxCPM2 Voice Design"

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return [
            {"key": "cfg_value", "type": "slider", "label": "CFG Value", "min": 0.5, "max": 5.0, "default": 2.0},
            {"key": "inference_timesteps", "type": "slider", "label": "Inference Steps", "min": 5.0, "max": 50.0, "default": 10.0},
            {
                "key": "normalize",
                "label": "Text Normalization",
                "type": "combo",
                "options": [(True, "On"), (False, "Off")],
                "default": True,
            },
        ]

    def generate(self, text: str, reference_audio_path: Optional[str] = None,
                 reference_text: Optional[str] = None, progress_cb: Optional[Callable] = None, **kwargs) -> Tuple[np.ndarray, int]:
        if not self._loaded or self._model is None:
            raise Exception("Model not loaded")

        cfg_value = float(kwargs.get("cfg_value", 2.0))
        inference_timesteps = int(round(float(kwargs.get("inference_timesteps", 10.0))))
        normalize = kwargs.get("normalize", True)

        wav = self._model.generate(
            text=text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
        )
        sr = self._model.tts_model.sample_rate
        audio = np.asarray(wav, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()
        return audio, sr


@register_backend
class VoxCPM2VoiceClone(_VoxCPM2Base):

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "voxcpm2-voiceclone"

    @property
    def model_id(self) -> str:
        return "voxcpm2-voiceclone"

    @property
    def display_name(self) -> str:
        return "🔊 VoxCPM2 Voice Clone"

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return [
            {"key": "cfg_value", "type": "slider", "label": "CFG Value", "min": 0.5, "max": 5.0, "default": 2.0},
            {"key": "inference_timesteps", "type": "slider", "label": "Inference Steps", "min": 5.0, "max": 50.0, "default": 10.0},
            {
                "key": "normalize",
                "label": "Text Normalization",
                "type": "combo",
                "options": [(True, "On"), (False, "Off")],
                "default": True,
            },
        ]

    def generate(self, text: str, reference_audio_path: Optional[str] = None,
                 reference_text: Optional[str] = None, progress_cb: Optional[Callable] = None, **kwargs) -> Tuple[np.ndarray, int]:
        if not self._loaded or self._model is None:
            raise Exception("Model not loaded")

        cfg_value = float(kwargs.get("cfg_value", 2.0))
        inference_timesteps = int(round(float(kwargs.get("inference_timesteps", 10.0))))
        normalize = kwargs.get("normalize", True)

        gen_kwargs: Dict[str, Any] = dict(
            text=text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
        )
        if reference_audio_path:
            gen_kwargs["reference_wav_path"] = reference_audio_path

        wav = self._model.generate(**gen_kwargs)
        sr = self._model.tts_model.sample_rate
        audio = np.asarray(wav, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()
        return audio, sr


@register_backend
class VoxCPM2HiFiClone(_VoxCPM2Base):

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "voxcpm2-hificlone"

    @property
    def model_id(self) -> str:
        return "voxcpm2-hificlone"

    @property
    def display_name(self) -> str:
        return "🔊 VoxCPM2 Hi-Fi Clone"

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return [
            {"key": "cfg_value", "type": "slider", "label": "CFG Value", "min": 0.5, "max": 5.0, "default": 2.0},
            {"key": "inference_timesteps", "type": "slider", "label": "Inference Steps", "min": 5.0, "max": 50.0, "default": 10.0},
            {
                "key": "normalize",
                "label": "Text Normalization",
                "type": "combo",
                "options": [(True, "On"), (False, "Off")],
                "default": True,
            },
        ]

    def generate(self, text: str, reference_audio_path: Optional[str] = None,
                 reference_text: Optional[str] = None, progress_cb: Optional[Callable] = None, **kwargs) -> Tuple[np.ndarray, int]:
        if not self._loaded or self._model is None:
            raise Exception("Model not loaded")

        cfg_value = float(kwargs.get("cfg_value", 2.0))
        inference_timesteps = int(round(float(kwargs.get("inference_timesteps", 10.0))))
        normalize = kwargs.get("normalize", True)

        gen_kwargs: Dict[str, Any] = dict(
            text=text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
        )
        if reference_audio_path:
            gen_kwargs["prompt_wav_path"] = reference_audio_path
        if reference_text:
            gen_kwargs["prompt_text"] = reference_text

        wav = self._model.generate(**gen_kwargs)
        sr = self._model.tts_model.sample_rate
        audio = np.asarray(wav, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()
        return audio, sr
