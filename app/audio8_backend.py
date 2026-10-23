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
MODEL_REPO = "AutoArk-AI/Audio8-TTS-Preview-0.6b"
DEFAULT_SR = 44100

_AUDIO8_PARAMS: List[Dict[str, Any]] = [
    {
        "key": "temperature", "type": "slider", "label": "Temperature",
        "min": 0.1, "max": 1.5, "default": 0.8,
        "tip": "Sampling temperature (higher = more varied output)",
    },
    {
        "key": "top_p", "type": "slider", "label": "Top-P",
        "min": 0.0, "max": 1.0, "default": 0.95,
        "tip": "Top-p (nucleus) sampling",
    },
    {
        "key": "top_k", "type": "spinbox", "label": "Top-K",
        "min": 0, "max": 200, "default": 50, "step": 1,
        "tip": "Top-k sampling (0 = disabled)",
    },
    {
        "key": "max_new_tokens", "type": "spinbox", "label": "Max New Tokens",
        "min": 64, "max": 2048, "default": 1024, "step": 64,
        "tip": "Maximum number of generated audio tokens (model context limit: 2048)",
    },
    {
        "key": "seed", "type": "spinbox", "label": "Seed (0=random)",
        "min": 0, "max": 99999, "default": 0, "step": 1,
        "tip": "Random seed for reproducibility (0 = random each time)",
    },
]


class InferenceError(Exception):
    pass


class _Audio8Base(TTSBackend):

    def __init__(self, model_dir: Path, dtype: torch.dtype):
        self.model_dir = Path(model_dir)
        self.dtype     = dtype
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype in (torch.float16, torch.bfloat16) and (self.device == "cpu" or is_rocm()):
            self.dtype = torch.float32
        self._model     = None
        self._processor = None
        self._loaded    = False

    @property
    def default_sample_rate(self) -> int:
        return DEFAULT_SR

    @property
    def auth_required(self) -> bool:
        return False

    @property
    def venv_names(self) -> List[str]:
        return ["venv_audio8"]

    @property
    def header_icon(self) -> str:
        return "🎧"

    @property
    def header_title(self) -> str:
        return "Audio8 TTS"

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return _AUDIO8_PARAMS

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

        status("Importing transformers…")
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            raise InferenceError(
                f"Failed to import transformers:\n{e}\n\n"
                "Run: pip install \"transformers>=4.57.0,<5\""
            )

        status(
            f"Loading Audio8 TTS from {self.model_dir} "
            f"| dtype: {self.dtype} | device: {self.device}"
        )
        status("  (first load may take 1–3 min)")
        try:
            self._processor = AutoProcessor.from_pretrained(
                str(self.model_dir),
                trust_remote_code=True,
            )
            self._model = AutoModel.from_pretrained(
                str(self.model_dir),
                trust_remote_code=True,
                dtype=self.dtype,
            ).eval().to(self.device)
        except Exception as e:
            raise InferenceError(f"Failed to load Audio8 TTS model:\n{e}")

        self._loaded = True
        status("✓ Audio8 TTS model loaded!")

    def unload_model(self, progress_cb: Optional[Callable] = None):
        def status(msg):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        if not self._loaded:
            status("Model is not loaded — nothing to release.")
            return

        status("Releasing Audio8 TTS model from memory…")
        self._model     = None
        self._processor = None
        self._loaded    = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        status("✓ Audio8 TTS model unloaded")

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        max_new_tokens: int = 1024,
        seed: int = 0,
        progress_cb: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        if not self._loaded or self._model is None or self._processor is None:
            raise InferenceError("Model is not loaded. Call load_model() first.")

        def status(msg):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        actual_seed = seed if seed > 0 else random.randint(1, 99999)
        torch.manual_seed(actual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(actual_seed)

        has_ref_audio = bool(reference_audio_path and os.path.exists(reference_audio_path))
        has_ref_text  = bool(reference_text and reference_text.strip())

        processor_kwargs: Dict[str, Any] = {
            "text": [text],
            "return_tensors": "pt",
        }

        if has_ref_audio and has_ref_text:
            status(f"Voice Cloning mode — ref: {Path(reference_audio_path).name}")
            processor_kwargs["reference_audio"] = [reference_audio_path]
            processor_kwargs["reference_text"]  = [reference_text]
        elif has_ref_audio and not has_ref_text:
            raise InferenceError(
                "Audio8 TTS requires the exact transcript of the reference audio "
                "(reference text) to perform voice cloning. Provide the reference "
                "text or remove the reference audio to generate without cloning."
            )
        else:
            status("Auto mode — generating without a reference voice")

        try:
            inputs = self._processor(**processor_kwargs)
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
        except Exception as e:
            raise InferenceError(f"Failed to prepare inputs for Audio8 TTS:\n{e}")

        status("Generating speech…")
        try:
            with torch.inference_mode():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    do_sample=True,
                    return_dict_in_generate=True,
                )
                waveforms, waveform_lengths = self._model.decode_audio(output.codes)
        except Exception as e:
            logger.error(f"Audio8 TTS inference error: {e}", exc_info=True)
            raise InferenceError(f"Inference failed:\n{e}")

        try:
            length = int(waveform_lengths[0])
            arr = waveforms[0, :length].float().detach().cpu().numpy()
        except Exception as e:
            raise InferenceError(f"Failed to decode generated audio:\n{e}")

        sample_rate = int(getattr(self._model.config, "codec_sample_rate", DEFAULT_SR))

        min_samples = int(sample_rate * 0.05)
        if arr.size < min_samples:
            raise InferenceError(
                f"Audio8 TTS returned audio that is too short "
                f"({arr.size} samples / {arr.size / sample_rate:.3f}s). "
                f"Try a different seed or increase max_new_tokens."
            )

        if arr.max() > 1.5 or arr.min() < -1.5:
            arr = arr / 32768.0

        status(f"✓ Generated {len(arr) / sample_rate:.1f}s of audio")
        return arr.astype(np.float32), sample_rate


@register_backend
class Audio8Bf16Backend(_Audio8Base):

    def __init__(self):
        super().__init__(_ROOT_DIR / "models" / "audio8", torch.bfloat16)

    @property
    def name(self) -> str:
        return "Audio8 TTS Preview 0.6B bfloat16"

    @property
    def model_id(self) -> str:
        return "audio8_bf16"

    @property
    def display_name(self) -> str:
        return "🎧 Audio8 TTS bfloat16"

    @property
    def download_repo(self) -> str:
        return MODEL_REPO

    @property
    def download_size(self) -> str:
        return "~2 GB"


@register_backend
class Audio8Fp32Backend(_Audio8Base):

    def __init__(self):
        super().__init__(_ROOT_DIR / "models" / "audio8", torch.float32)

    @property
    def name(self) -> str:
        return "Audio8 TTS Preview 0.6B float32"

    @property
    def model_id(self) -> str:
        return "audio8_fp32"

    @property
    def display_name(self) -> str:
        return "🔵 Audio8 TTS float32"

    @property
    def download_repo(self) -> str:
        return MODEL_REPO

    @property
    def download_size(self) -> str:
        return "~3.5 GB"
