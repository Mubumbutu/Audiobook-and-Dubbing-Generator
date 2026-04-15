from __future__ import annotations

import gc
import importlib.util
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import snapshot_download

from tts_backends import TTSBackend, SynthesisRequest, SynthesisResult, register_backend

logger = logging.getLogger(__name__)

_ROOT_DIR = Path(__file__).parent.parent
_DEFAULT_SR = 24000


class InferenceError(Exception):
    pass


_MOSS_PARAMS_8B: List[Dict[str, Any]] = [
    {
        "key":     "audio_temperature",
        "type":    "slider",
        "label":   "Audio Temperature",
        "min":     0.1,
        "max":     3.0,
        "default": 1.7,
        "tip":     "Higher = more expressive variation; lower = more stable delivery",
    },
    {
        "key":     "audio_top_p",
        "type":    "slider",
        "label":   "Audio Top-P",
        "min":     0.1,
        "max":     1.0,
        "default": 0.8,
        "tip":     "Nucleus sampling cutoff; lower = more conservative",
    },
    {
        "key":     "audio_top_k",
        "type":    "spinbox",
        "label":   "Audio Top-K",
        "min":     1,
        "max":     500,
        "default": 25,
        "step":    5,
        "tip":     "Top-K sampling; lower = tighter token distribution",
    },
    {
        "key":     "audio_repetition_penalty",
        "type":    "slider",
        "label":   "Rep. Penalty",
        "min":     0.5,
        "max":     2.0,
        "default": 1.0,
        "tip":     ">1.0 discourages repeating token patterns",
    },
    {
        "key":     "max_new_tokens",
        "type":    "spinbox",
        "label":   "Max New Tokens",
        "min":     512,
        "max":     32768,
        "default": 4096,
        "step":    512,
        "tip":     "Maximum audio tokens to generate (higher = longer audio allowed)",
    },
    {
        "key":     "target_tokens",
        "type":    "spinbox",
        "label":   "Target Tokens (0=off)",
        "min":     0,
        "max":     45000,
        "default": 0,
        "step":    50,
        "tip":     "Duration control: ~12.5 tokens/second. 0 = disabled",
    },
]

_MOSS_PARAMS_17B: List[Dict[str, Any]] = [
    {
        "key":     "audio_temperature",
        "type":    "slider",
        "label":   "Audio Temperature",
        "min":     0.1,
        "max":     3.0,
        "default": 1.0,
        "tip":     "Higher = more expressive variation; lower = more stable delivery",
    },
    {
        "key":     "audio_top_p",
        "type":    "slider",
        "label":   "Audio Top-P",
        "min":     0.1,
        "max":     1.0,
        "default": 0.95,
        "tip":     "Nucleus sampling cutoff; lower = more conservative",
    },
    {
        "key":     "audio_top_k",
        "type":    "spinbox",
        "label":   "Audio Top-K",
        "min":     1,
        "max":     500,
        "default": 50,
        "step":    5,
        "tip":     "Top-K sampling; lower = tighter token distribution",
    },
    {
        "key":     "audio_repetition_penalty",
        "type":    "slider",
        "label":   "Rep. Penalty",
        "min":     0.5,
        "max":     2.0,
        "default": 1.1,
        "tip":     ">1.0 discourages repeating token patterns",
    },
    {
        "key":     "max_new_tokens",
        "type":    "spinbox",
        "label":   "Max New Tokens",
        "min":     512,
        "max":     32768,
        "default": 4096,
        "step":    512,
        "tip":     "Maximum audio tokens to generate (higher = longer audio allowed)",
    },
    {
        "key":     "target_tokens",
        "type":    "spinbox",
        "label":   "Target Tokens (0=off)",
        "min":     0,
        "max":     45000,
        "default": 0,
        "step":    50,
        "tip":     "Duration control: ~12.5 tokens/second. 0 = disabled",
    },
]


class _MossTTSBase(TTSBackend):

    def __init__(self, model_dir: Path, params: List[Dict[str, Any]]):
        self.model_dir  = Path(model_dir)
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"
        self._params    = params
        self._model     = None
        self._processor = None
        self._sr        = _DEFAULT_SR
        self._loaded    = False

    @property
    def default_sample_rate(self) -> int:
        return self._sr

    @property
    def auth_required(self) -> bool:
        return False

    @property
    def venv_names(self) -> List[str]:
        return ["venv_moss"]

    @property
    def header_icon(self) -> str:
        return "🌿"

    @property
    def header_title(self) -> str:
        return "MOSS-TTS"

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return []

    def is_available(self) -> bool:
        if not self.model_dir.exists():
            return False
        try:
            return any(self.model_dir.iterdir())
        except Exception:
            return False

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
            audio_temperature=request.temperature,
            audio_top_p=request.top_p,
            audio_repetition_penalty=request.repetition_penalty,
            max_new_tokens=request.max_new_tokens,
            target_tokens=0,
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
            progress_cb(f"Downloading {self.download_repo}…")
        snapshot_download(
            repo_id=self.download_repo,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
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
                mem  = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
                return f"CUDA — {name} ({mem} GB)"
            except Exception:
                return "CUDA"
        return "CPU (no GPU — generation will be very slow)"

    def _resolve_attn_impl(self, dtype) -> str:
        if (
            self.device == "cuda"
            and importlib.util.find_spec("flash_attn") is not None
            and dtype in {torch.float16, torch.bfloat16}
        ):
            try:
                major, _ = torch.cuda.get_device_capability()
                if major >= 8:
                    return "flash_attention_2"
            except Exception:
                pass
        if self.device == "cuda":
            return "sdpa"
        return "eager"

    def load_model(self, progress_cb: Optional[Callable] = None) -> None:
        def status(msg):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        if not self.model_dir.exists():
            raise InferenceError(
                f"Model directory does not exist:\n{self.model_dir}\n"
                "Download the model using the 'Download model' button."
            )
        if not self.is_available():
            raise InferenceError(
                "Model directory is empty — download the model first."
            )

        status("Importing transformers…")
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            raise InferenceError(
                f"Failed to import transformers:\n{e}\n\n"
                "Run: pip install transformers>=5.0.0"
            )

        dtype    = torch.bfloat16 if self.device == "cuda" else torch.float32
        attn_impl = self._resolve_attn_impl(dtype)
        status(f"Attention: {attn_impl} | Device: {self.device} | Dtype: {dtype}")

        if self.device == "cuda":
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)

        status("Loading processor…")
        try:
            processor = AutoProcessor.from_pretrained(
                str(self.model_dir),
                trust_remote_code=True,
            )
            processor.audio_tokenizer = processor.audio_tokenizer.to(self.device)
        except Exception as e:
            raise InferenceError(f"Failed to load processor:\n{e}")

        status("Loading model weights — this may take several minutes…")
        try:
            model = AutoModel.from_pretrained(
                str(self.model_dir),
                trust_remote_code=True,
                attn_implementation=attn_impl,
                torch_dtype=dtype,
            ).to(self.device)
            model.eval()
        except Exception as e:
            raise InferenceError(f"Failed to load model:\n{e}")

        try:
            self._sr = int(processor.model_config.sampling_rate)
        except Exception:
            self._sr = _DEFAULT_SR

        self._processor = processor
        self._model     = model
        self._loaded    = True
        status(f"✓ MOSS-TTS loaded! Sample rate: {self._sr} Hz | Device: {self.device}")

    def unload_model(self, progress_cb: Optional[Callable] = None) -> None:
        def status(msg):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        if not self._loaded:
            status("Model is not loaded — nothing to release.")
            return

        status("Releasing MOSS-TTS model from memory…")

        self._model     = None
        self._processor = None
        self._loaded    = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        status("✓ MOSS-TTS unloaded")

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        audio_temperature: float = 1.7,
        audio_top_p: float = 0.8,
        audio_top_k: int = 25,
        audio_repetition_penalty: float = 1.0,
        max_new_tokens: int = 4096,
        target_tokens: int = 0,
        progress_cb: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        if not self._loaded or self._model is None:
            raise InferenceError("Model is not loaded. Call load_model() first.")

        def status(msg):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        status("Preparing input…")
        try:
            msg_kwargs: Dict[str, Any] = {"text": text}
            if (
                reference_audio_path
                and os.path.exists(reference_audio_path)
            ):
                msg_kwargs["reference"] = [reference_audio_path]
            if target_tokens and target_tokens > 0:
                msg_kwargs["tokens"] = int(target_tokens)

            conversation = [self._processor.build_user_message(**msg_kwargs)]
            batch        = self._processor([conversation], mode="generation")
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
        except Exception as e:
            raise InferenceError(f"Failed to prepare input:\n{e}")

        status("Generating speech…")
        generate_kwargs: Dict[str, Any] = {
            "input_ids":                input_ids,
            "attention_mask":           attention_mask,
            "max_new_tokens":           int(max_new_tokens),
        }
        try:
            generate_kwargs["audio_temperature"]        = float(audio_temperature)
            generate_kwargs["audio_top_p"]              = float(audio_top_p)
            generate_kwargs["audio_top_k"]              = int(audio_top_k)
            generate_kwargs["audio_repetition_penalty"] = float(audio_repetition_penalty)
        except Exception:
            pass

        try:
            with torch.no_grad():
                outputs = self._model.generate(**generate_kwargs)
        except TypeError:
            basic = {
                "input_ids":     input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": int(max_new_tokens),
            }
            with torch.no_grad():
                outputs = self._model.generate(**basic)
        except Exception as e:
            raise InferenceError(f"Generation failed:\n{e}")

        status("Decoding audio…")
        try:
            messages = list(self._processor.decode(outputs))
            if not messages:
                raise InferenceError("Decoder returned no messages.")
            first = messages[0]
            if not hasattr(first, "audio_codes_list") or not first.audio_codes_list:
                raise InferenceError("Decoded message contains no audio.")
            audio_tensor = first.audio_codes_list[0]
            if audio_tensor.dim() > 1:
                audio_tensor = audio_tensor.squeeze(0)
            audio_np = audio_tensor.detach().cpu().float().numpy()
        except InferenceError:
            raise
        except Exception as e:
            raise InferenceError(f"Decoding failed:\n{e}")

        if audio_np.size == 0:
            raise InferenceError("Decoded audio is empty.")

        if audio_np.max() > 1.5 or audio_np.min() < -1.5:
            audio_np = audio_np / 32768.0

        dur = len(audio_np) / max(1, self._sr)
        status(f"✓ Generated {dur:.1f}s of audio")
        return audio_np, self._sr


@register_backend
class MossTTSDelay8BBackend(_MossTTSBase):

    def __init__(self):
        super().__init__(
            _ROOT_DIR / "models" / "moss-tts-8b",
            _MOSS_PARAMS_8B,
        )

    @property
    def name(self) -> str:
        return "MOSS-TTS Delay 8B"

    @property
    def model_id(self) -> str:
        return "moss_tts_8b"

    @property
    def display_name(self) -> str:
        return "🌿 MOSS-TTS Delay 8B"

    @property
    def download_repo(self) -> str:
        return "OpenMOSS-Team/MOSS-TTS"

    @property
    def download_size(self) -> str:
        return "~16 GB"

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return _MOSS_PARAMS_8B

@register_backend
class MossTTSLocal17BBackend(_MossTTSBase):

    def __init__(self):
        super().__init__(
            _ROOT_DIR / "models" / "moss-tts-1.7b",
            _MOSS_PARAMS_17B,
        )

    @property
    def name(self) -> str:
        return "MOSS-TTS Local 1.7B"

    @property
    def model_id(self) -> str:
        return "moss_tts_17b"

    @property
    def display_name(self) -> str:
        return "🌱 MOSS-TTS Local 1.7B"

    @property
    def download_repo(self) -> str:
        return "OpenMOSS-Team/MOSS-TTS"

    @property
    def download_size(self) -> str:
        return "~3.5 GB"

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return _MOSS_PARAMS_17B