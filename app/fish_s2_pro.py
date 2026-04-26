from __future__ import annotations

from tts_backends import (
    TTSBackend, SynthesisRequest, SynthesisResult, register_backend, safe_compute_dtype,
)

import gc
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import torch
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

_ROOT_DIR = Path(__file__).parent.parent
MODEL_REPO = "fishaudio/s2-pro"
DEFAULT_SR = 44100

_FISH_PARAMS: List[Dict[str, Any]] = [
    {"key": "temperature",        "type": "slider",  "label": "Temperature",         "min": 0.1, "max": 1.5, "default": 0.7,  "tip": "Generation randomness"},
    {"key": "top_p",              "type": "slider",  "label": "Top-P",               "min": 0.1, "max": 1.0, "default": 0.7,  "tip": "Nucleus sampling"},
    {"key": "repetition_penalty", "type": "slider",  "label": "Rep. Penalty",        "min": 0.9, "max": 2.0, "default": 1.2,  "tip": "Repetition penalty"},
    {"key": "max_new_tokens",     "type": "spinbox", "label": "Max tokens (0=auto)", "min": 0,   "max": 4096, "default": 0,   "step": 128},
    {"key": "chunk_length",       "type": "spinbox", "label": "Chunk length",        "min": 50,  "max": 1000, "default": 200, "step": 50},
]


class InferenceError(Exception):
    pass


# ─── FP8 support ─────────────────────────────────────────────────────────────

class FP8Linear(torch.nn.Module):
    def __init__(self, weight_fp8: torch.Tensor, scale: torch.Tensor,
                 bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("weight_fp8", weight_fp8)
        self.register_buffer("scale", scale)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        self.out_features, self.in_features = weight_fp8.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight_fp8.to(x.dtype) * self.scale.to(x.dtype)
        return torch.nn.functional.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, fp8=True"


def _is_fp8_model(model_dir: Path) -> bool:
    try:
        from safetensors import safe_open
        for f in model_dir.glob("*.safetensors"):
            with safe_open(str(f), framework="pt", device="cpu") as st:
                for key in st.keys():
                    if key.endswith(".weight.scale"):
                        return True
    except Exception:
        pass
    return False


def _load_fp8_scales(model_dir: Path, device: str) -> Dict[str, torch.Tensor]:
    scales: Dict[str, torch.Tensor] = {}
    try:
        from safetensors import safe_open
        for f in sorted(model_dir.glob("*.safetensors")):
            with safe_open(str(f), framework="pt", device="cpu") as st:
                for key in st.keys():
                    if key.endswith(".weight.scale"):
                        scales[key[: -len(".scale")]] = st.get_tensor(key).to(device)
    except Exception as e:
        logger.warning(f"Failed to load FP8 scales: {e}")
    return scales


def _patch_model_fp8(model: torch.nn.Module,
                     model_dir: Path,
                     device: str) -> int:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        logger.warning("torch.float8_e4m3fn unavailable (PyTorch < 2.1) — FP8 patch skipped")
        return 0
    try:
        from safetensors import safe_open
    except ImportError:
        logger.warning("safetensors unavailable — FP8 patch skipped")
        return 0

    fp8_weights: Dict[str, torch.Tensor] = {}
    fp8_scales: Dict[str, torch.Tensor] = {}

    for f in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as st:
            for key in st.keys():
                t = st.get_tensor(key)
                if t.dtype == fp8_dtype:
                    fp8_weights[key] = t.to(device)
                elif key.endswith(".weight.scale"):
                    fp8_scales[key[: -len(".scale")]] = t.to(device)

    if not fp8_weights:
        logger.warning("No FP8 tensors found in safetensors — FP8 patch skipped")
        return 0

    logger.info(f"  Found {len(fp8_weights)} FP8 layers in safetensors")

    patched = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        weight_key = name + ".weight"
        if weight_key not in fp8_weights:
            continue
        if weight_key not in fp8_scales:
            logger.warning(f"Missing scale for {weight_key} — skipping")
            continue
        fp8_layer = FP8Linear(
            weight_fp8=fp8_weights[weight_key],
            scale=fp8_scales[weight_key],
            bias=module.bias,
        )
        parts = name.split(".")
        parent: torch.nn.Module = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], fp8_layer)
        patched += 1

    return patched


def _install_fp8_hook() -> Tuple[list, Callable]:
    captured: list = []
    try:
        from fish_speech.models.text2semantic.llama import DualARTransformer

        original = DualARTransformer.from_pretrained

        def patched_func(cls, *args, **kwargs):
            m = original.__func__(cls, *args, **kwargs) if hasattr(original, '__func__') \
                else original(*args, **kwargs)
            captured.append(m)
            return m

        DualARTransformer.from_pretrained = classmethod(patched_func)

        def restore():
            DualARTransformer.from_pretrained = original
    except Exception as e:
        logger.warning(f"FP8 hook could not be installed: {e}")

        def restore():
            pass

    return captured, restore


# ─── helpers ─────────────────────────────────────────────────────────────────

def _to_float32(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().float().numpy()
    arr = np.asarray(x, dtype=np.float32).flatten()
    if arr.size > 0 and (arr.max() > 1.5 or arr.min() < -1.5):
        arr = arr / 32768.0
    return arr


def _inspect_serve_request() -> Dict[str, Any]:
    try:
        from fish_speech.inference_engine import ServeTTSRequest
        if hasattr(ServeTTSRequest, "model_fields"):
            return {k: v.default for k, v in ServeTTSRequest.model_fields.items()}
        if hasattr(ServeTTSRequest, "__fields__"):
            return {k: v.default for k, v in ServeTTSRequest.__fields__.items()}
    except Exception:
        pass
    return {}


# ─── Base class (not registered) ─────────────────────────────────────────────

class _FishS2ProBase(TTSBackend):

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self._engine        = None
        self._llama_queue   = None
        self._decoder_model = None
        self._loaded        = False

    # ── TTSBackend abstract properties shared by both variants ───────────────

    @property
    def default_sample_rate(self) -> int:
        return DEFAULT_SR

    @property
    def auth_required(self) -> bool:
        return False

    @property
    def venv_names(self) -> List[str]:
        return ["venv_fish_s2_pro"]

    @property
    def header_icon(self) -> str:
        return "🐟"

    @property
    def header_title(self) -> str:
        return "Fish Audio S2 Pro"

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return _FISH_PARAMS

    # ── TTSBackend abstract methods ───────────────────────────────────────────

    def is_available(self) -> bool:
        return self.model_dir.exists() and any(self.model_dir.iterdir())

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
            max_new_tokens=request.max_new_tokens,
            chunk_length=request.chunk_length,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            progress_cb=progress_cb,
        )
        return SynthesisResult(audio=audio, sample_rate=sr, duration_s=len(audio) / sr)

    def download(
        self,
        model_dir: Path,
        progress_cb: Optional[Callable] = None,
    ) -> None:
        download_model(model_dir, progress_cb, repo=self.download_repo)

    # ── Public interface ──────────────────────────────────────────────────────

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
                "Download the model using the 'Download model' button.")
        if not (self.model_dir / "config.json").exists():
            raise InferenceError("Missing config.json in model directory.")
        if not (self.model_dir / "codec.pth").exists():
            raise InferenceError("Missing codec.pth in model directory.")

        status("Importing fish_speech…")

        for _mod in [
            "lightning", "lightning.pytorch", "lightning.pytorch.callbacks",
            "lightning.pytorch.callbacks.callback", "lightning.pytorch.core",
            "lightning.pytorch.core.module", "lightning.pytorch.loggers",
            "lightning.pytorch.trainer", "lightning.pytorch.utilities",
            "lightning.pytorch.utilities.types", "lightning.pytorch.utilities.rank_zero",
            "pytorch_lightning", "pytorch_lightning.callbacks",
            "pytorch_lightning.callbacks.callback", "pytorch_lightning.core",
            "pytorch_lightning.core.module", "pytorch_lightning.loggers",
            "pytorch_lightning.trainer", "pytorch_lightning.utilities",
            "pytorch_lightning.utilities.types",
            "torchmetrics", "torchmetrics.functional",
            "torchmetrics.functional.audio", "torchmetrics.functional.audio.dnsmos",
        ]:
            if _mod not in sys.modules:
                sys.modules[_mod] = MagicMock()
        sys.modules["lightning.pytorch"].Callback = type("Callback", (), {})
        sys.modules["pytorch_lightning"].Callback = type("Callback", (), {})
        sys.modules["torchmetrics"].Metric = type("Metric", (), {})
        _rz = MagicMock()
        _rz.side_effect = lambda fn: fn
        sys.modules["lightning.pytorch.utilities.rank_zero"].rank_zero_only = _rz

        try:
            from fish_speech.models.text2semantic.inference import (
                launch_thread_safe_queue, load_codec_model)
            from fish_speech.inference_engine import TTSInferenceEngine
        except ImportError as e:
            raise InferenceError(
                f"Failed to import fish_speech:\n{e}\n\n"
                "Run: pip install git+https://github.com/fishaudio/fish-speech.git")

        precision = safe_compute_dtype(self.device)
        status(f"Precision: {precision} | device: {self.device}")

        is_fp8 = _is_fp8_model(self.model_dir)
        captured: list = []
        restore_hook: Optional[Callable] = None
        if is_fp8:
            status("FP8 model detected — installing dequantization hook…")
            captured, restore_hook = _install_fp8_hook()

        status("Launching text2semantic model in thread…")
        status("  (first load may take 1–3 min)")
        try:
            self._llama_queue = launch_thread_safe_queue(
                checkpoint_path=str(self.model_dir), device=self.device,
                precision=precision, compile=False)
        except Exception as e:
            if restore_hook:
                restore_hook()
            raise InferenceError(f"launch_thread_safe_queue failed:\n{e}")
        status("✓ LLaMA queue ready")

        if is_fp8:
            if restore_hook:
                restore_hook()
            if captured:
                status("Loading FP8 weights from safetensors and patching layers…")
                n = _patch_model_fp8(captured[0], self.model_dir, self.device)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if n > 0:
                    status(f"✓ FP8 patch applied — {n} Linear → FP8Linear layers")
                else:
                    status("⚠ FP8 patch replaced no layers — check logs")
            else:
                status("⚠ Hook did not capture model — FP8 patch skipped")

        status("Loading DAC codec (codec.pth)…")
        try:
            self._decoder_model = load_codec_model(
                codec_checkpoint_path=str(self.model_dir / "codec.pth"),
                device=self.device, precision=torch.float32)
        except Exception as e:
            raise InferenceError(f"load_codec_model failed:\n{e}")
        status("✓ DAC loaded")

        status("Initialising TTSInferenceEngine…")
        try:
            self._engine = TTSInferenceEngine(
                llama_queue=self._llama_queue, decoder_model=self._decoder_model,
                precision=precision, compile=False)
        except Exception as e:
            raise InferenceError(f"TTSInferenceEngine failed:\n{e}")

        self._loaded = True
        status("✓ Fish Audio S2 Pro model loaded!")

    def unload_model(self, progress_cb: Optional[Callable] = None):
        def status(msg):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        if not self._loaded:
            status("Model is not loaded — nothing to release.")
            return

        status("Releasing Fish S2 Pro model from memory…")

        if self._llama_queue is not None:
            try:
                self._llama_queue.put(None)
            except Exception:
                pass

        self._engine        = None
        self._llama_queue   = None
        self._decoder_model = None
        self._loaded        = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        status("✓ Fish S2 Pro model unloaded")

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        max_new_tokens: int = 0,
        chunk_length: int = 200,
        top_p: float = 0.7,
        repetition_penalty: float = 1.2,
        temperature: float = 0.7,
        progress_cb: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        if not self._loaded or self._engine is None:
            raise InferenceError("Model is not loaded. Call load_model() first.")

        def status(msg):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        try:
            from fish_speech.inference_engine import ServeTTSRequest
        except ImportError as e:
            raise InferenceError(f"Cannot import ServeTTSRequest:\n{e}")

        references = []
        if reference_audio_path and os.path.exists(reference_audio_path):
            with open(reference_audio_path, "rb") as fh:
                ref_bytes = fh.read()
            references = [{"audio": ref_bytes, "text": reference_text or ""}]

        known_fields = _inspect_serve_request()

        req_kwargs: Dict[str, Any] = {
            "text":               text,
            "references":         references,
            "max_new_tokens":     max_new_tokens,
            "chunk_length":       chunk_length,
            "top_p":              top_p,
            "repetition_penalty": repetition_penalty,
            "temperature":        temperature,
            "streaming":          False,
        }

        if known_fields:
            req_kwargs = {k: v for k, v in req_kwargs.items() if k in known_fields}

        status("Generating speech…")
        try:
            request = ServeTTSRequest(**req_kwargs)
            chunks  = list(self._engine.inference(request))
        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            raise InferenceError(f"Inference failed:\n{e}")

        audio_parts = []
        for chunk in chunks:
            if hasattr(chunk, "audio") and chunk.audio is not None:
                raw = chunk.audio
                if isinstance(raw, (bytes, bytearray)):
                    try:
                        import io as _io
                        import soundfile as _sf
                        arr, _ = _sf.read(_io.BytesIO(raw), dtype="float32")
                        audio_parts.append(arr.flatten())
                    except Exception:
                        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                        audio_parts.append(arr)
                elif isinstance(raw, tuple) and len(raw) == 2:
                    audio_parts.append(_to_float32(raw[1]))
                else:
                    audio_parts.append(_to_float32(raw))
            elif isinstance(chunk, tuple) and len(chunk) == 2:
                audio_parts.append(_to_float32(chunk[1]))

        if not audio_parts:
            raise InferenceError("Engine returned no audio chunks.")

        audio = np.concatenate(audio_parts)
        status(f"✓ Generated {len(audio) / DEFAULT_SR:.1f}s of audio")
        return audio, DEFAULT_SR


# ─── Registered backends ──────────────────────────────────────────────────────

@register_backend
class FishS2ProFP8Backend(_FishS2ProBase):

    def __init__(self):
        super().__init__(_ROOT_DIR / "models" / "s2-pro-fp8")

    @property
    def name(self) -> str:
        return "Fish S2 Pro FP8"

    @property
    def model_id(self) -> str:
        return "fish_s2_pro_fp8"

    @property
    def display_name(self) -> str:
        return "⚡ Fish S2 Pro FP8"

    @property
    def download_repo(self) -> str:
        return "drbaph/s2-pro-fp8"

    @property
    def download_size(self) -> str:
        return "~6.2 GB"


@register_backend
class FishS2ProFullBackend(_FishS2ProBase):

    def __init__(self):
        super().__init__(_ROOT_DIR / "models" / "s2-pro")

    @property
    def name(self) -> str:
        return "Fish S2 Pro Full bfloat16"

    @property
    def model_id(self) -> str:
        return "fish_s2_pro_full"

    @property
    def display_name(self) -> str:
        return "🔵 Fish S2 Pro Full bfloat16"

    @property
    def download_repo(self) -> str:
        return "fishaudio/s2-pro"

    @property
    def download_size(self) -> str:
        return "~11 GB"


# ─── Download Fish model ──────────────────────────────────────────────────────

def download_model(model_dir: Path, progress_cb: Optional[Callable] = None,
                   repo: str = MODEL_REPO) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    if progress_cb:
        progress_cb(f"Downloading {repo}…", 0.0)
    snapshot_download(repo_id=repo, local_dir=str(model_dir))
    if progress_cb:
        progress_cb("Model downloaded successfully!", 1.0)
