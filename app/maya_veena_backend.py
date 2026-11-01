from __future__ import annotations

from tts_backends import (
    TTSBackend, SynthesisRequest, SynthesisResult, InferenceError,
    register_backend, is_rocm, safe_compute_dtype,
)

import gc
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

_ROOT_DIR = Path(__file__).parent.parent

MAYA1_REPO = "maya-research/maya1"
VEENA_REPO = "maya-research/veena-tts"
SNAC_REPO = "hubertsiuzdak/snac_24khz"

MAYA1_DIR = _ROOT_DIR / "models" / "maya1"
VEENA_DIR = _ROOT_DIR / "models" / "veena"
SNAC_DIR = _ROOT_DIR / "models" / "snac_24khz"

SAMPLE_RATE = 24000

CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_TOKENS_PER_FRAME = 7
SOH_ID = 128259
EOH_ID = 128260
SOA_ID = 128261
EOA_ID = 128262
BOS_ID = 128000
TEXT_EOT_ID = 128009
_STOP_IDS = {CODE_END_TOKEN_ID, EOA_ID}

MAYA1_DESCRIPTIONS: Dict[str, str] = {
    "narrator_male": "Realistic male voice in the 30s age with american accent. Normal pitch, warm timbre, conversational pacing.",
    "narrator_warm": "A warm, low-pitched narrator with clear diction and an unhurried conversational pace.",
    "narrator_female": "Female voice in her 30s with a neutral accent. Normal pitch, warm timbre, clear diction, conversational pacing.",
    "host_female": "Female, in her 30s with an American accent and is an event host, energetic, clear diction",
    "villain_male": "Dark villain character, Male voice in their 40s with a British accent. low pitch, gravelly timbre, slow pacing, angry tone at high intensity.",
    "goddess_female": "Mythical godlike magical character, Female voice in their 30s slow pacing, curious tone at medium intensity.",
}

MAYA1_VOICES: List[Tuple[str, str]] = [
    ("narrator_male", "Narrator — male, warm"),
    ("narrator_warm", "Narrator — low-pitched, unhurried"),
    ("narrator_female", "Narrator — female, warm"),
    ("host_female", "Event host — female, energetic"),
    ("villain_male", "Villain — male, gravelly, angry"),
    ("goddess_female", "Goddess — female, slow, curious"),
    ("custom", "✏️ Custom description…"),
]

VEENA_SPEAKERS: List[Tuple[str, str]] = [
    ("kavya", "Kavya"),
    ("agastya", "Agastya"),
    ("maitri", "Maitri"),
    ("vinaya", "Vinaya"),
]

_MAYA1_VOICE_KEY = "maya1_voice"
_VEENA_VOICE_KEY = "veena_speaker"


def _sampling_params(temperature: float, top_p: float, repetition_penalty: float) -> List[Dict[str, Any]]:
    return [
        {
            "key": "temperature", "type": "slider", "label": "Temperature",
            "min": 0.1, "max": 1.5, "default": temperature,
            "tip": "Sampling temperature (higher = more varied output)",
        },
        {
            "key": "top_p", "type": "slider", "label": "Top-P",
            "min": 0.1, "max": 1.0, "default": top_p,
            "tip": "Top-p (nucleus) sampling",
        },
        {
            "key": "repetition_penalty", "type": "dspinbox", "label": "Repetition Penalty",
            "min": 1.0, "max": 2.0, "default": repetition_penalty, "step": 0.05,
            "tip": "Penalty for repeated audio tokens (too high can distort speech)",
        },
        {
            "key": "max_new_tokens", "type": "spinbox", "label": "Max New Tokens",
            "min": 64, "max": 2048, "default": 2048, "step": 64,
            "tip": "Maximum number of generated audio tokens (7 tokens per audio frame)",
        },
    ]


def _voice_param(key: str, label: str, options: List[Tuple[str, str]], tip: str) -> Dict[str, Any]:
    return {
        "key": key, "type": "combo", "label": label,
        "options": list(options), "default": options[0][0], "tip": tip,
    }


_MAYA1_PARAMS = [
    _voice_param(
        _MAYA1_VOICE_KEY, "Voice", MAYA1_VOICES,
        "Voice description preset used to design the voice",
    ),
    {
        "key": "maya1_description",
        "type": "text",
        "label": "Custom description",
        "default": "",
        "tip": "Free-form voice description, used when Voice = 'Custom description'. "
               "Example: 'Female, in her 30s with an American accent, energetic, clear diction'. "
               "Inline expressive tokens like <laugh>, <whisper>, <sigh>, <cry> are supported in the text.",
    },
    *_sampling_params(0.4, 0.9, 1.1),
]

_VEENA_PARAMS = [
    _voice_param(
        _VEENA_VOICE_KEY, "Speaker", VEENA_SPEAKERS,
        "Built-in Veena speaker",
    ),
    *_sampling_params(0.4, 0.9, 1.05),
]


def _extract_codes(token_ids: List[int]) -> List[int]:
    end = next((i for i, t in enumerate(token_ids) if t in _STOP_IDS), len(token_ids))
    codes = [t for t in token_ids[:end] if SNAC_MIN_ID <= t <= SNAC_MAX_ID]
    return codes[: len(codes) // SNAC_TOKENS_PER_FRAME * SNAC_TOKENS_PER_FRAME]


class _SnacLlamaBackend(TTSBackend):

    _REPO: str = ""
    _VOICES: List[Tuple[str, str]] = []
    _VOICE_KEY: str = ""
    _TRIM_WARMUP: int = 0
    _MIN_NEW_TOKENS: int = 0

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = safe_compute_dtype(self.device)
        self._model = None
        self._tokenizer = None
        self._snac = None
        self._loaded = False
        self._device_mapped = False

    @property
    def default_sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def auth_required(self) -> bool:
        return False

    @property
    def venv_names(self) -> List[str]:
        return ["venv_maya_veena"]

    @property
    def header_icon(self) -> str:
        return "🗣"

    @property
    def header_title(self) -> str:
        return "Maya-1 / Veena TTS"

    @property
    def download_repo(self) -> str:
        return self._REPO

    @property
    def speaker_voice_options(self) -> List[Tuple[str, str]]:
        return self._VOICES

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
        model_dir.mkdir(parents=True, exist_ok=True)
        if progress_cb:
            progress_cb(f"Downloading {self._REPO}…", 0.0)
        snapshot_download(repo_id=self._REPO, local_dir=str(model_dir))
        if progress_cb:
            progress_cb(f"Downloading {SNAC_REPO}…", 0.9)
        SNAC_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=SNAC_REPO, local_dir=str(SNAC_DIR))
        self._marker.touch(exist_ok=True)
        if progress_cb:
            progress_cb("Model downloaded successfully!", 1.0)

    def _model_kwargs(self, status: Callable[[str], None]) -> Dict[str, Any]:
        return {"dtype": self.dtype}

    def _load_snac(self):
        try:
            from snac import SNAC
        except ImportError as e:
            raise InferenceError(
                f"Failed to import snac:\n{e}\n\nRun: pip install snac"
            )
        sources = [str(SNAC_DIR), SNAC_REPO] if (SNAC_DIR / "config.json").exists() else [SNAC_REPO]
        last_error: Optional[Exception] = None
        for source in sources:
            try:
                return SNAC.from_pretrained(source).eval().to(self.device)
            except Exception as e:
                last_error = e
                logger.warning(f"SNAC load from {source} failed: {e}")
        raise InferenceError(f"Failed to load SNAC decoder:\n{last_error}")

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
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise InferenceError(
                f"Failed to import transformers:\n{e}\n\n"
                "Run: pip install \"transformers>=4.57.0,<5\" accelerate"
            )

        status(f"Loading {self.name} from {self.model_dir} | device: {self.device}")
        self._device_mapped = False
        try:
            kwargs = self._model_kwargs(status)
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_dir), trust_remote_code=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(self.model_dir), trust_remote_code=True, **kwargs,
            ).eval()
            if not self._device_mapped:
                model = model.to(self.device)
            self._model = model
        except InferenceError:
            raise
        except Exception as e:
            raise InferenceError(f"Failed to load {self.name}:\n{e}")

        status("Loading SNAC decoder…")
        try:
            self._snac = self._load_snac()
        except Exception:
            self._model = None
            self._tokenizer = None
            self._snac = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

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
        self._model = None
        self._tokenizer = None
        self._snac = None
        self._loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        status(f"✓ {self.name} unloaded")

    def _resolve_voice(self, reference_text: Optional[str], kwargs: Dict[str, Any]) -> str:
        codes = [code for code, _ in self._VOICES]
        candidate = (reference_text or "").strip()
        if candidate in codes:
            return candidate
        chosen = kwargs.get(self._VOICE_KEY)
        return chosen if chosen in codes else codes[0]

    def _build_input_ids(self, text: str, voice: str, custom_description: Optional[str] = None) -> List[int]:
        raise NotImplementedError

    def _token_budget(self, text: str, max_new_tokens: int) -> int:
        return int(max_new_tokens)

    def _decode_codes(self, codes: List[int]) -> np.ndarray:
        frames = len(codes) // SNAC_TOKENS_PER_FRAME
        grid = (
            np.asarray(codes[: frames * SNAC_TOKENS_PER_FRAME], dtype=np.int64)
            .reshape(frames, SNAC_TOKENS_PER_FRAME) - CODE_TOKEN_OFFSET
        ) % 4096
        levels = [
            grid[:, 0],
            grid[:, [1, 4]].reshape(-1),
            grid[:, [2, 3, 5, 6]].reshape(-1),
        ]
        tensors = [
            torch.from_numpy(np.ascontiguousarray(level)).long().unsqueeze(0).to(self.device)
            for level in levels
        ]
        with torch.inference_mode():
            audio = self._snac.decode(tensors)
        return audio.reshape(-1).float().detach().cpu().numpy()

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        temperature: float = 0.4,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        max_new_tokens: int = 2048,
        progress_cb: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        if not self._loaded or self._model is None or self._tokenizer is None or self._snac is None:
            raise InferenceError("Model is not loaded. Call load_model() first.")

        def status(msg):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        custom_desc = (kwargs.get("maya1_description") or "").strip() or None
        voice = self._resolve_voice(reference_text, kwargs)
        status(f"Voice: {voice}")

        try:
            input_ids = torch.tensor(
                [self._build_input_ids(text, voice, custom_desc)], device=self._model.device,
            )
        except Exception as e:
            raise InferenceError(f"Failed to prepare inputs for {self.name}:\n{e}")

        pad_id = self._tokenizer.pad_token_id
        status("Generating speech…")
        try:
            with torch.inference_mode():
                output = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=self._token_budget(text, max_new_tokens),
                    min_new_tokens=self._MIN_NEW_TOKENS,
                    do_sample=True,
                    temperature=max(float(temperature), 0.01),
                    top_p=float(top_p),
                    repetition_penalty=float(repetition_penalty),
                    eos_token_id=sorted(_STOP_IDS),
                    pad_token_id=pad_id if pad_id is not None else CODE_END_TOKEN_ID,
                )
        except Exception as e:
            logger.error(f"{self.name} inference error: {e}", exc_info=True)
            raise InferenceError(f"Inference failed:\n{e}")

        generated = output[0, input_ids.shape[1]:].tolist()
        codes = _extract_codes(generated)
        if len(codes) < SNAC_TOKENS_PER_FRAME:
            raise InferenceError(
                f"{self.name} did not generate any audio tokens. "
                "Try different sampling settings or a longer text."
            )

        try:
            arr = self._decode_codes(codes)
        except Exception as e:
            raise InferenceError(f"Failed to decode generated audio:\n{e}")

        if self._TRIM_WARMUP and arr.size > self._TRIM_WARMUP:
            arr = arr[self._TRIM_WARMUP:]
        arr = np.clip(arr, -1.0, 1.0)

        if arr.size < int(SAMPLE_RATE * 0.05):
            raise InferenceError(
                f"{self.name} returned audio that is too short "
                f"({arr.size} samples). Try different sampling settings."
            )

        status(f"✓ Generated {len(arr) / SAMPLE_RATE:.1f}s of audio")
        return arr.astype(np.float32), SAMPLE_RATE

    def synthesize(
        self,
        request: SynthesisRequest,
        progress_cb: Optional[Callable] = None,
    ) -> SynthesisResult:
        extra = getattr(request, "extra", None) or {}
        audio, sr = self.generate(
            text=request.text,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            max_new_tokens=request.max_new_tokens,
            progress_cb=progress_cb,
            maya1_description=extra.get("maya1_description"),
            **{self._VOICE_KEY: request.speaker_id},
        )
        return SynthesisResult(audio=audio, sample_rate=sr, duration_s=len(audio) / sr)


@register_backend
class Maya1Backend(_SnacLlamaBackend):

    _REPO = MAYA1_REPO
    _VOICES = MAYA1_VOICES
    _VOICE_KEY = _MAYA1_VOICE_KEY
    _TRIM_WARMUP = 2048
    _MIN_NEW_TOKENS = 28

    def __init__(self):
        super().__init__(MAYA1_DIR)

    @property
    def name(self) -> str:
        return "Maya-1 3.3B"

    @property
    def model_id(self) -> str:
        return "maya1"

    @property
    def display_name(self) -> str:
        return "🎭 Maya-1 (voice design)"

    @property
    def download_size(self) -> str:
        return "~7 GB"

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return _MAYA1_PARAMS

    def _build_input_ids(self, text: str, voice: str, custom_description: Optional[str] = None) -> List[int]:
        if voice == "custom":
            description = (custom_description or "").strip() or MAYA1_DESCRIPTIONS["narrator_male"]
        else:
            description = MAYA1_DESCRIPTIONS.get(voice, MAYA1_DESCRIPTIONS["narrator_male"])
        description = description.replace('"', "'")
        body = self._tokenizer.encode(
            f'<description="{description}"> {text}', add_special_tokens=False,
        )
        return [SOH_ID, BOS_ID, *body, TEXT_EOT_ID, EOH_ID, SOA_ID, CODE_START_TOKEN_ID]


@register_backend
class VeenaBackend(_SnacLlamaBackend):

    _REPO = VEENA_REPO
    _VOICES = VEENA_SPEAKERS
    _VOICE_KEY = _VEENA_VOICE_KEY

    def __init__(self):
        super().__init__(VEENA_DIR)

    @property
    def name(self) -> str:
        return "Veena 3B"

    @property
    def model_id(self) -> str:
        return "veena"

    @property
    def display_name(self) -> str:
        return "🎙 Veena (Hindi / English)"

    @property
    def download_size(self) -> str:
        return "~6 GB"

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return _VEENA_PARAMS

    def _model_kwargs(self, status: Callable[[str], None]) -> Dict[str, Any]:
        if self.device == "cuda" and not is_rocm():
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes  # noqa: F401
                status("Using 4-bit NF4 quantization")
                self._device_mapped = True
                return {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    ),
                    "device_map": "auto",
                }
            except Exception as e:
                status(f"4-bit quantization unavailable ({e}) — loading without quantization")
        return {"dtype": self.dtype}

    def _token_budget(self, text: str, max_new_tokens: int) -> int:
        estimate = int(len(text) * 1.3) * SNAC_TOKENS_PER_FRAME + 21
        return max(SNAC_TOKENS_PER_FRAME, min(estimate, int(max_new_tokens)))

    def _build_input_ids(self, text: str, voice: str, custom_description: Optional[str] = None) -> List[int]:
        body = self._tokenizer.encode(f"<spk_{voice}> {text}", add_special_tokens=False)
        return [SOH_ID, *body, EOH_ID, SOA_ID, CODE_START_TOKEN_ID]
