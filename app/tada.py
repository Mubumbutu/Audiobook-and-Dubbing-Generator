from __future__ import annotations

import gc
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import snapshot_download

from tts_backends import (
    TTSBackend, SynthesisRequest, SynthesisResult, register_backend,
)

logger = logging.getLogger(__name__)

_ROOT_DIR = Path(__file__).parent.parent
_APP_DIR  = Path(__file__).parent
ENCODER_REPO = "HumeAI/tada-codec"
DEFAULT_SR   = 24000

_ENCODER_DIR = _ROOT_DIR / "models" / "tada-codec"

TADA_SUPPORTED_LANGUAGES = [
    ("",   "English (default)"),
    ("ar", "Arabic"),
    ("zh", "Chinese"),
    ("de", "German"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
]

_TADA_PARAMS: List[Dict[str, Any]] = [
    {
        "key": "temperature", "type": "slider", "label": "Temperature",
        "min": 0.1, "max": 1.5, "default": 0.7,
        "tip": "Controls generation randomness. Lower = more deterministic.",
    },
    {
        "key": "top_p", "type": "slider", "label": "Top-P",
        "min": 0.1, "max": 1.0, "default": 0.7,
        "tip": "Nucleus sampling probability threshold.",
    },
    {
        "key": "repetition_penalty", "type": "slider", "label": "Rep. Penalty",
        "min": 0.9, "max": 2.0, "default": 1.2,
        "tip": "Penalizes repeated tokens. Higher = less repetition.",
    },
    {
        "key": "language", "type": "combo", "label": "Language",
        "options": TADA_SUPPORTED_LANGUAGES, "default": "",
        "tip": (
            "Target language for synthesis. For non-English prompts, "
            "provide the reference audio transcript — built-in ASR is English-only."
        ),
    },
    {
        "key": "num_extra_steps", "type": "spinbox", "label": "Extra steps",
        "min": 0, "max": 200, "default": 0, "step": 10,
        "tip": (
            "Additional autoregressive steps appended after the synthesized text "
            "(speech continuation). 0 = disabled."
        ),
    },
]

class TADAInferenceError(Exception):
    pass


def _win_add_torch_dll() -> None:
    if sys.platform != "win32":
        return
    venv = os.path.dirname(os.path.dirname(sys.executable))
    for sub in ("Lib", "lib"):
        p = os.path.join(venv, sub, "site-packages", "torch", "lib")
        if os.path.isdir(p):
            try:
                os.add_dll_directory(p)
            except Exception:
                pass

def download_tada_model(
    model_dir: Path,
    encoder_dir: Path,
    repo_id: str,
    progress_cb: Optional[Callable] = None,
) -> None:
    def status(msg: str) -> None:
        logger.info(msg)
        if progress_cb:
            progress_cb(msg, 0.0)

    hf_token: Optional[str] = None
    hf_token_file = _ROOT_DIR / ".hf_token"
    if hf_token_file.exists():
        try:
            hf_token = hf_token_file.read_text(encoding="utf-8").strip() or None
        except Exception:
            pass
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    hf_cache_dir = _ROOT_DIR / "models" / "hf_cache"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache_dir / "hub")

    status(f"Downloading TADA encoder ({ENCODER_REPO})…")
    encoder_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=ENCODER_REPO,
        local_dir=str(encoder_dir),
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
        token=hf_token,
    )
    status("✓ Encoder downloaded")

    status(f"Downloading TADA model ({repo_id})…")
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_dir),
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
        token=hf_token,
    )
    status("✓ Model downloaded")

    status("Downloading Wav2Vec2 model (facebook/wav2vec2-large)…")
    snapshot_download(
        repo_id="facebook/wav2vec2-large",
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
        token=hf_token,
    )
    status("✓ Wav2Vec2 model downloaded")

    status("Downloading Parakeet ASR model (nvidia/parakeet-ctc-1.1b) ~4.25 GB…")
    snapshot_download(
        repo_id="nvidia/parakeet-ctc-1.1b",
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
        token=hf_token,
    )
    status("✓ Parakeet ASR model downloaded")

    status("Caching TADA codec decoder (HumeAI/tada-codec)…")
    snapshot_download(
        repo_id=ENCODER_REPO,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
        token=hf_token,
    )
    status("✓ TADA codec decoder cached")


# ─── Base class (not registered) ─────────────────────────────────────────────

class _TADABase(TTSBackend):

    def __init__(self, model_dir: Path):
        self.model_dir          = Path(model_dir)
        self.encoder_dir        = _ENCODER_DIR
        self.device             = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype             = torch.float32
        self._encoder           = None
        self._codec_decoder     = None
        self._model             = None
        self._loaded            = False
        self._encoder_language  = ""
        self._Encoder_class     = None

    # ── TTSBackend abstract properties shared by all TADA variants ────────────

    @property
    def default_sample_rate(self) -> int:
        return DEFAULT_SR

    @property
    def auth_required(self) -> bool:
        return True

    @property
    def venv_names(self) -> List[str]:
        return ["venv_tada"]

    @property
    def header_icon(self) -> str:
        return "🎙"

    @property
    def header_title(self) -> str:
        return "TADA TTS"

    @property
    def whisper_incompatible(self) -> bool:
        return True

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return _TADA_PARAMS

    # ── TTSBackend abstract methods ───────────────────────────────────────────

    def is_available(self) -> bool:
        try:
            return (
                self.model_dir.exists()
                and any(self.model_dir.iterdir())
                and self.encoder_dir.exists()
                and any(self.encoder_dir.iterdir())
            )
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
            progress_cb=progress_cb,
        )
        return SynthesisResult(audio=audio, sample_rate=sr, duration_s=len(audio) / sr)

    def download(
        self,
        model_dir: Path,
        progress_cb: Optional[Callable] = None,
    ) -> None:
        download_tada_model(model_dir, self.encoder_dir, self.download_repo, progress_cb)

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

    def load_model(self, progress_cb: Optional[Callable] = None) -> None:
        _APP_DIR = Path(__file__).parent

        def status(msg: str) -> None:
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        _win_add_torch_dll()

        if not self.model_dir.exists() or not any(self.model_dir.iterdir()):
            raise TADAInferenceError(
                f"Model directory not found or empty:\n{self.model_dir}\n"
                "Download the model first."
            )
        if not self.encoder_dir.exists() or not any(self.encoder_dir.iterdir()):
            raise TADAInferenceError(
                f"Encoder directory not found or empty:\n{self.encoder_dir}\n"
                "Download the model first."
            )

        hf_token: Optional[str] = None
        hf_token_file = _ROOT_DIR / ".hf_token"   # ← katalog główny aplikacji
        if hf_token_file.exists():
            try:
                hf_token = hf_token_file.read_text(encoding="utf-8").strip() or None
            except Exception:
                pass
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

        hf_cache_dir = _ROOT_DIR / "models" / "hf_cache"   # ← katalog główny aplikacji
        hf_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_cache_dir)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache_dir / "hub")

        try:
            from transformers.modeling_utils import PreTrainedModel

            def _atw_get(self):
                return self.__dict__.get("_atw_store", {})

            def _atw_set(self, value):
                self.__dict__["_atw_store"] = value if value is not None else {}

            PreTrainedModel.all_tied_weights_keys = property(_atw_get, _atw_set)
        except Exception:
            pass

        try:
            from transformers.generation.utils import GenerationMixin

            _orig_prepare = GenerationMixin._prepare_generation_config.__wrapped__ \
                if hasattr(GenerationMixin._prepare_generation_config, "__wrapped__") \
                else GenerationMixin._prepare_generation_config

            def _patched_prepare(self, generation_config, *args, **kwargs):
                try:
                    result = _orig_prepare(self, generation_config)
                except TypeError:
                    result = _orig_prepare(self, generation_config, *args, **kwargs)
                if isinstance(result, tuple):
                    return result
                return result, {}

            GenerationMixin._prepare_generation_config = _patched_prepare
        except Exception:
            pass

        app_dir_str = str(_APP_DIR)
        _removed = []
        for entry in list(sys.path):
            if os.path.normcase(os.path.normpath(entry)) == os.path.normcase(os.path.normpath(app_dir_str)):
                sys.path.remove(entry)
                _removed.append(entry)

        try:
            _tada_mod = sys.modules.pop("tada", None)

            from tada.modules.encoder import Encoder
            from tada.modules.decoder import Decoder
            from tada.modules.tada import TadaForCausalLM

        except ImportError as e:
            raise TADAInferenceError(
                f"Failed to import tada:\n{e}\n\n"
                "Run: pip install git+https://github.com/HumeAI/tada.git"
            )
        finally:
            for entry in _removed:
                if entry not in sys.path:
                    sys.path.insert(0, entry)
            if _tada_mod is not None and "tada" not in sys.modules:
                sys.modules["tada"] = _tada_mod

        self._dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        status(f"Loading TADA encoder from {self.encoder_dir}…")
        try:
            self._encoder = (
                Encoder.from_pretrained(
                    str(self.encoder_dir),
                    subfolder="encoder",
                    token=hf_token,
                )
                .to(self.device)
                .to(self._dtype)
            )
            self._codec_decoder = (
                Decoder.from_pretrained(
                    str(self.encoder_dir),
                    subfolder="decoder",
                    token=hf_token,
                )
                .to(self.device)
                .to(self._dtype)
            )
        except Exception as e:
            raise TADAInferenceError(f"Encoder/decoder load failed:\n{e}")

        self._Encoder_class    = Encoder
        self._encoder_language = ""
        status("✓ Encoder loaded")

        status(f"Loading TADA model from {self.model_dir}…")
        status("  (first load may take 2–5 min)")
        try:
            self._model = (
                TadaForCausalLM.from_pretrained(
                    str(self.model_dir),
                    torch_dtype=self._dtype,
                    token=hf_token,
                )
                .to(self.device)
            )
            self._model.eval()
        except Exception as e:
            raise TADAInferenceError(f"Model load failed:\n{e}")

        try:
            self._model._decoder.load_state_dict(
                self._codec_decoder.state_dict(), strict=False
            )
            self._model._decoder.to(device=self.device, dtype=self._dtype)
        except Exception as e:
            raise TADAInferenceError(f"Codec decoder weight injection failed:\n{e}")

        self._loaded = True
        status(f"✓ TADA model loaded on {self.device}")

    def unload_model(self, progress_cb: Optional[Callable] = None) -> None:
        def status(msg: str) -> None:
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        self._encoder           = None
        self._codec_decoder     = None
        self._model             = None
        self._loaded            = False
        self._encoder_language  = ""
        self._Encoder_class     = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        status("TADA model unloaded")

    def _ensure_encoder_language(self, language: str, status_fn: Callable) -> None:
        target = language or ""
        if target == self._encoder_language:
            return
        if self._Encoder_class is None:
            return

        hf_token: Optional[str] = None
        hf_token_file = _ROOT_DIR / ".hf_token"   # ← _APP_DIR → _ROOT_DIR
        if hf_token_file.exists():
            try:
                hf_token = hf_token_file.read_text(encoding="utf-8").strip() or None
            except Exception:
                pass

        lang_label = target if target else "en"
        status_fn(f"Reloading encoder for language '{lang_label}'…")
        lang_kwargs: Dict[str, Any] = {"language": target} if target else {}
        try:
            self._encoder = (
                self._Encoder_class.from_pretrained(
                    str(self.encoder_dir),
                    subfolder="encoder",
                    token=hf_token,
                    **lang_kwargs,
                )
                .to(self.device)
                .to(self._dtype)
            )
            self._encoder_language = target
            status_fn(f"✓ Encoder ready for '{lang_label}'")
        except Exception as e:
            raise TADAInferenceError(
                f"Encoder reload for language '{lang_label}' failed:\n{e}"
            )

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        progress_cb: Optional[Callable] = None,
        language: str = "",
        num_extra_steps: int = 0,
        temperature: float = 0.7,
        top_p: float = 0.7,
        repetition_penalty: float = 1.2,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        if not self._loaded or self._encoder is None or self._model is None:
            raise TADAInferenceError("Model is not loaded. Call load_model() first.")

        try:
            import torchaudio
        except ImportError as e:
            raise TADAInferenceError(f"torchaudio not available:\n{e}")

        def status(msg: str) -> None:
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        if not reference_audio_path:
            raise TADAInferenceError(
                "TADA requires a reference audio file for voice cloning.\n"
                "Upload a reference audio file in the Voice Cloning section."
            )

        self._ensure_encoder_language(language, status)

        status("Loading reference audio…")
        ref_audio, ref_sr = torchaudio.load(reference_audio_path)
        ref_audio = ref_audio.to(device=self.device, dtype=self._dtype)

        ref_text_list = [reference_text] if reference_text else None

        status("Encoding reference prompt…")
        try:
            prompt = self._encoder(
                ref_audio,
                text=ref_text_list,
                sample_rate=ref_sr,
            )
        except Exception as e:
            raise TADAInferenceError(f"Encoder failed:\n{e}")

        status("Generating speech…")
        try:
            gen_kwargs: Dict[str, Any] = {
                "prompt":             prompt,
                "text":               text,
                "temperature":        temperature,
                "top_p":              top_p,
                "repetition_penalty": repetition_penalty,
            }
            if num_extra_steps > 0:
                gen_kwargs["num_extra_steps"] = num_extra_steps
            with torch.inference_mode():
                output = self._model.generate(**gen_kwargs)
        except TypeError:
            try:
                with torch.inference_mode():
                    output = self._model.generate(prompt=prompt, text=text)
            except Exception as e:
                raise TADAInferenceError(f"Generation failed:\n{e}")
        except Exception as e:
            raise TADAInferenceError(f"Generation failed:\n{e}")

        audio_np = None
        if hasattr(output, "audio"):
            val = output.audio
            if torch.is_tensor(val):
                cand = val.detach().cpu().float().numpy().flatten().astype(np.float32)
                if cand.size > 100:
                    audio_np = self._normalize(cand)
            elif isinstance(val, (list, tuple)) and len(val) > 0:
                first = val[0]
                if torch.is_tensor(first):
                    cand = first.detach().cpu().float().numpy().flatten().astype(np.float32)
                    if cand.size > 100:
                        audio_np = self._normalize(cand)
                elif isinstance(first, np.ndarray) and first.size > 100:
                    audio_np = self._normalize(first.flatten().astype(np.float32))

        if audio_np is None or audio_np.size <= 100:
            audio_np = self._decode_output(output, status)

        dur = len(audio_np) / DEFAULT_SR
        logger.info(f"[TADA] generated {dur:.2f}s of audio")

        if dur < 0.3:
            raise TADAInferenceError(
                f"Generated audio is suspiciously short ({dur:.3f}s)."
            )

        status(f"✓ Generated {dur:.1f}s of audio")
        return audio_np, DEFAULT_SR

    def _decode_output(
        self,
        output,
        status_fn: Callable,
    ) -> np.ndarray:
        audio_candidate = None

        if hasattr(output, "audio"):
            val = output.audio
            if torch.is_tensor(val):
                audio_candidate = val.detach().cpu().float().numpy()
            elif isinstance(val, np.ndarray):
                audio_candidate = val.astype(np.float32)
            elif isinstance(val, (list, tuple)) and len(val) > 0:
                first = val[0]
                if torch.is_tensor(first):
                    audio_candidate = first.detach().cpu().float().numpy()
                elif isinstance(first, np.ndarray):
                    audio_candidate = first.astype(np.float32)

        if audio_candidate is not None:
            arr = audio_candidate.flatten().astype(np.float32)
            if arr.size > 100:
                return self._normalize(arr)

        acoustic = getattr(output, "acoustic_features", None)
        if acoustic is not None and torch.is_tensor(acoustic):
            status_fn("Decoding acoustic features via codec decoder…")
            if self._codec_decoder is None:
                raise TADAInferenceError(
                    "Codec decoder not loaded — call load_model() first."
                )
            try:
                batch_size, time_steps, _ = acoustic.shape
                token_masks = torch.zeros(
                    (batch_size, time_steps),
                    dtype=torch.int32,
                    device=self.device,
                )
                with torch.inference_mode():
                    feats = acoustic.to(device=self.device, dtype=self._dtype)
                    decoded = self._codec_decoder(feats, token_masks)
                if torch.is_tensor(decoded):
                    arr = decoded.detach().cpu().float().numpy().flatten().astype(np.float32)
                    return self._normalize(arr)
                elif isinstance(decoded, np.ndarray):
                    return self._normalize(decoded.flatten().astype(np.float32))
                elif hasattr(decoded, "audio"):
                    val = decoded.audio
                    if torch.is_tensor(val):
                        arr = val.detach().cpu().float().numpy().flatten().astype(np.float32)
                        return self._normalize(arr)
            except TADAInferenceError:
                raise
            except Exception as e:
                raise TADAInferenceError(
                    f"Failed to decode acoustic_features via codec decoder:\n{e}\n"
                    f"acoustic_features shape: {tuple(acoustic.shape)}, dtype: {acoustic.dtype}"
                )

        raise TADAInferenceError(
            f"Cannot extract audio from GenerationOutput.\n"
            f"output.audio={getattr(output, 'audio', 'N/A')}, "
            f"output.acoustic_features={'present' if acoustic is not None else 'None'}."
        )
    
    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr
        max_abs = float(np.abs(arr).max())
        if max_abs > 1.5 and max_abs > 0:
            arr = arr / max_abs * 0.95
        return arr


# ─── Registered backends ──────────────────────────────────────────────────────

@register_backend
class TADA1BBackend(_TADABase):

    def __init__(self):
        super().__init__(_ROOT_DIR / "models" / "tada-1b")

    @property
    def name(self) -> str:
        return "TADA 1B · Llama 3.2 1B · ~2.5 GB"

    @property
    def model_id(self) -> str:
        return "tada-1b"

    @property
    def display_name(self) -> str:
        return "🟠 TADA 1B (Llama 3.2 1B)"

    @property
    def download_repo(self) -> str:
        return "HumeAI/tada-1b"

    @property
    def download_size(self) -> str:
        return "~2.5 GB"


@register_backend
class TADA3BMLBackend(_TADABase):

    def __init__(self):
        super().__init__(_ROOT_DIR / "models" / "tada-3b-ml")

    @property
    def name(self) -> str:
        return "TADA 3B Multilingual · Llama 3.2 3B · ~6.5 GB"

    @property
    def model_id(self) -> str:
        return "tada-3b-ml"

    @property
    def display_name(self) -> str:
        return "🟠 TADA 3B Multilingual (Llama 3.2 3B)"

    @property
    def download_repo(self) -> str:
        return "HumeAI/tada-3b-ml"

    @property
    def download_size(self) -> str:
        return "~6.5 GB"
