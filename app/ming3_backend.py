from __future__ import annotations

from tts_backends import TTSBackend, SynthesisRequest, SynthesisResult, register_backend

import gc
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

_ROOT_DIR = Path(__file__).parent.parent
_MODEL_DIR = _ROOT_DIR / "models" / "ming3-0.5b"
_REPO_DIR = _ROOT_DIR / "repo" / "ming3"
MODEL_REPO = "inclusionAI/Ming-3.0-tts-0.5B"
DEFAULT_SR = 44100

MING3_TASKS: List[Tuple[str, str]] = [
    ("TTS", "TTS — zero-shot voice cloning"),
    ("VOICE_DESIGN", "Voice Design — from description"),
    ("FREEFORM", "Freeform — natural-language control"),
    ("MULTILINGUAL", "Multilingual — language_id required"),
    ("BASIC", "Basic — speed / pitch / volume"),
    ("EMOTION", "Emotion — happy / sad / angry"),
    ("DIALECT", "Dialect — Sichuanese / Cantonese"),
    ("IP", "IP — built-in premium voice"),
]

MING3_LANGUAGES: List[Tuple[str, str]] = [
    ("Auto", "Auto-detect"),
    ("Chinese", "Chinese"),
    ("English", "English"),
    ("German", "German"),
    ("Spanish", "Spanish"),
    ("French", "French"),
    ("Korean", "Korean"),
    ("Japanese", "Japanese"),
]

MING3_EMOTIONS: List[Tuple[str, str]] = [
    ("", "None"),
    ("happy", "Happy"),
    ("sad", "Sad"),
    ("angry", "Angry"),
]

MING3_DIALECTS: List[Tuple[str, str]] = [
    ("", "None"),
    ("sichuanese", "Sichuanese"),
    ("cantonese", "Cantonese"),
]

_MING3_PARAMS: List[Dict[str, Any]] = [
    {
        "key": "ming_task",
        "type": "combo",
        "label": "Task",
        "options": MING3_TASKS,
        "default": "TTS",
        "tip": (
            "TTS: zero-shot voice cloning from reference audio.\n"
            "VOICE_DESIGN: describe the voice — put the description in the "
            "Reference audio transcription field.\n"
            "FREEFORM: free-form instruction — put the instruction in the "
            "Reference audio transcription field.\n"
            "MULTILINGUAL: like TTS but requires a Language.\n"
            "BASIC: speed / pitch / volume only.\n"
            "EMOTION: happy / sad / angry (reference audio optional).\n"
            "DIALECT: Sichuanese / Cantonese.\n"
            "IP: built-in premium voice — put the voice name in the "
            "Reference audio transcription field."
        ),
    },
    {
        "key": "ming_language",
        "type": "combo",
        "label": "Language",
        "options": MING3_LANGUAGES,
        "default": "Auto",
        "tip": "Target language. Required for MULTILINGUAL.",
    },
    {
        "key": "ming_emotion",
        "type": "combo",
        "label": "Emotion",
        "options": MING3_EMOTIONS,
        "default": "",
        "tip": "Emotion control for EMOTION task.",
    },
    {
        "key": "ming_dialect",
        "type": "combo",
        "label": "Dialect",
        "options": MING3_DIALECTS,
        "default": "",
        "tip": "Dialect for DIALECT task.",
    },
    {
        "key": "ming_speed",
        "type": "combo",
        "label": "Speed",
        "options": [("", "Default"), ("slow", "Slow"), ("normal", "Normal"), ("fast", "Fast")],
        "default": "",
        "tip": "Speed for BASIC task.",
    },
    {
        "key": "ming_pitch",
        "type": "combo",
        "label": "Pitch",
        "options": [("", "Default"), ("low", "Low"), ("normal", "Normal"), ("high", "High")],
        "default": "",
        "tip": "Pitch for BASIC task.",
    },
    {
        "key": "ming_volume",
        "type": "combo",
        "label": "Volume",
        "options": [("", "Default"), ("low", "Low"), ("normal", "Normal"), ("high", "High")],
        "default": "",
        "tip": "Volume for BASIC task.",
    },
    {
        "key": "ming_ip",
        "type": "combo",
        "label": "IP voice",
        "options": [("", "None")],
        "default": "",
        "tip": "Built-in premium voice name for IP task. Add names by editing _MING3_PARAMS.",
    },
    {
        "key": "ming_cfg",
        "type": "slider",
        "label": "CFG",
        "min": 1.0,
        "max": 5.0,
        "default": 2.0,
        "tip": "Classifier-free guidance scale.",
    },
    {
        "key": "ming_sigma",
        "type": "slider",
        "label": "Sigma",
        "min": 0.0,
        "max": 1.0,
        "default": 0.25,
        "tip": "Flow-matching sigma.",
    },
    {
        "key": "ming_temperature",
        "type": "slider",
        "label": "Temperature",
        "min": 0.0,
        "max": 1.5,
        "default": 0.0,
        "tip": "Sampling temperature (0 = deterministic).",
    },
    {
        "key": "ming_max_length",
        "type": "spinbox",
        "label": "Max length",
        "min": 10,
        "max": 500,
        "default": 50,
        "step": 10,
        "tip": "Semantic length each synthesis fragment is cut to.",
    },
    {
        "key": "ming_max_decode_steps",
        "type": "spinbox",
        "label": "Max decode steps",
        "min": 100,
        "max": 5000,
        "default": 1000,
        "step": 100,
        "tip": "Maximum number of decode steps.",
    },
    {
        "key": "ming_use_spk_emb",
        "type": "combo",
        "label": "Use speaker embedding",
        "options": [(False, "Off"), (True, "On")],
        "default": False,
        "tip": "Use campplus speaker embedding from reference audio instead of raw prompt features.",
    },
]


@register_backend
class Ming3TTSBackend(TTSBackend):

    def __init__(self):
        self.model_dir = _MODEL_DIR
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._loaded = False
        self._sample_rate = DEFAULT_SR

    @property
    def name(self) -> str:
        return "Ming-3.0-tts-0.5B"

    @property
    def model_id(self) -> str:
        return "ming3-0.5b"

    @property
    def display_name(self) -> str:
        return "🔮 Ming-3.0-tts 0.5B"

    @property
    def default_sample_rate(self) -> int:
        return DEFAULT_SR

    @property
    def auth_required(self) -> bool:
        return False

    @property
    def venv_names(self) -> List[str]:
        return ["venv_ming3"]

    @property
    def download_repo(self) -> str:
        return MODEL_REPO

    @property
    def download_size(self) -> str:
        return "~1 GB"

    @property
    def header_icon(self) -> str:
        return "🔮"

    @property
    def header_title(self) -> str:
        return "Ming-3.0-tts"

    @property
    def whisper_incompatible(self) -> bool:
        return False

    @property
    def pyannote_incompatible(self) -> bool:
        return False

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return _MING3_PARAMS

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
        return "CPU (no GPU — generation will be slow)"

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
            progress_cb=progress_cb,
        )
        return SynthesisResult(audio=audio, sample_rate=sr, duration_s=len(audio) / sr)

    def download(self, model_dir: Path, progress_cb: Optional[Callable] = None) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        if progress_cb:
            progress_cb(f"Downloading {MODEL_REPO}…", 0.0)
        snapshot_download(repo_id=MODEL_REPO, local_dir=str(model_dir))
        if progress_cb:
            progress_cb("Model downloaded successfully!", 1.0)

    def load_model(self, progress_cb: Optional[Callable] = None) -> None:
        def status(msg):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        if not self.model_dir.exists():
            raise RuntimeError(
                f"Model directory does not exist:\n{self.model_dir}\n"
                "Download the model using the 'Download model' button."
            )

        repo_str = str(_REPO_DIR)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        status("Importing ming_tts…")
        try:
            from ming_tts import MingTTS
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import ming_tts:\n{e}\n\n"
                f"Make sure the Ming-3.0-tts repository was cloned to:\n{_REPO_DIR}\n"
                "Run ming3_install.bat to clone it."
            )

        has_pynini = False
        try:
            import pynini
            has_pynini = True
        except ImportError:
            pass
        status(f"Text normalization available: {has_pynini}")

        status(f"Loading Ming-3.0-tts from {self.model_dir} on {self.device}")
        status("  (first load may take 1–3 min)")
        try:
            self._model = MingTTS(
                str(self.model_dir),
                device=self.device,
                text_normalization=has_pynini,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Ming-3.0-tts model:\n{e}")

        try:
            self._sample_rate = int(getattr(self._model, "sample_rate", DEFAULT_SR))
        except Exception:
            self._sample_rate = DEFAULT_SR

        self._loaded = True
        status(f"✓ Ming-3.0-tts loaded! Sample rate: {self._sample_rate} Hz")

    def unload_model(self, progress_cb: Optional[Callable] = None) -> None:
        def status(msg):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        if not self._loaded:
            return

        status("Releasing Ming-3.0-tts from memory…")
        self._model = None
        self._loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        status("✓ Ming-3.0-tts unloaded")

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        ming_task: str = "TTS",
        ming_language: str = "Auto",
        ming_emotion: str = "",
        ming_dialect: str = "",
        ming_speed: str = "",
        ming_pitch: str = "",
        ming_volume: str = "",
        ming_ip: str = "",
        ming_cfg: float = 2.0,
        ming_sigma: float = 0.25,
        ming_temperature: float = 0.0,
        ming_max_length: int = 50,
        ming_max_decode_steps: int = 1000,
        ming_use_spk_emb: bool = False,
        progress_cb: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        if not self._loaded or self._model is None:
            raise RuntimeError("Ming-3.0-tts model not loaded. Click 'Load model' first.")

        def status(msg):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        task = (ming_task or "TTS").upper()
        ref_text = (reference_text or "").strip()
        language_id = None if ming_language in ("", "Auto") else ming_language

        controls: Dict[str, Any] = {}
        if ming_speed:
            controls["speed"] = ming_speed
        if ming_pitch:
            controls["pitch"] = ming_pitch
        if ming_volume:
            controls["volume"] = ming_volume
        if ming_emotion:
            controls["emotion"] = ming_emotion
        if ming_dialect:
            controls["dialect"] = ming_dialect
        if ming_ip:
            controls["ip"] = ming_ip
        if task == "IP" and ref_text and "ip" not in controls:
            controls["ip"] = ref_text
        if task == "VOICE_DESIGN" and ref_text:
            controls["style"] = ref_text
        controls_arg = controls or None

        instruction = None
        if task == "FREEFORM" and ref_text:
            instruction = ref_text

        prompt_wav_path: Optional[str] = None
        prompt_text: Optional[str] = None
        use_spk_emb = False

        if task != "IP":
            if reference_audio_path and os.path.exists(reference_audio_path):
                prompt_wav_path = reference_audio_path
                prompt_text = ref_text or None
                use_spk_emb = bool(ming_use_spk_emb)
                status(f"Using reference audio: {Path(reference_audio_path).name}")

        status(f"Synthesizing ({task})…")

        fd, tmp_out = tempfile.mkstemp(suffix=".wav", prefix="ming3_")
        os.close(fd)
        waveform = None
        try:
            waveform = self._model.synthesize(
                task=task,
                text=text,
                language_id=language_id,
                instruction=instruction,
                controls=controls_arg,
                instruction_schema=None,
                prompt_wav_path=prompt_wav_path,
                prompt_text=prompt_text,
                spk_emb_path=None,
                use_spk_emb=use_spk_emb,
                use_zero_spk_emb=False,
                cfg=float(ming_cfg),
                sigma=float(ming_sigma),
                temperature=float(ming_temperature),
                max_length=int(ming_max_length),
                max_decode_steps=int(ming_max_decode_steps),
                trim_silence=True,
                raw_text=False,
                output_path=tmp_out,
            )
        except Exception as e:
            logger.error(f"Ming-3.0-tts inference error: {e}", exc_info=True)
            raise RuntimeError(f"Inference failed:\n{e}")
        finally:
            try:
                if os.path.exists(tmp_out):
                    os.remove(tmp_out)
            except Exception:
                pass

        if waveform is None:
            raise RuntimeError("Ming-3.0-tts returned no audio.")

        if isinstance(waveform, torch.Tensor):
            arr = waveform.detach().cpu().float().numpy()
        else:
            arr = np.asarray(waveform, dtype=np.float32)
        arr = arr.reshape(-1).astype(np.float32)

        sr = self._sample_rate
        min_samples = int(sr * 0.05)
        if arr.size < min_samples:
            raise RuntimeError(
                f"Ming-3.0-tts returned audio that is too short "
                f"({arr.size} samples / {arr.size / sr:.3f}s)."
            )

        if arr.max() > 1.5 or arr.min() < -1.5:
            arr = arr / 32768.0

        status(f"✓ Generated {len(arr) / sr:.1f}s of audio")
        return arr, sr