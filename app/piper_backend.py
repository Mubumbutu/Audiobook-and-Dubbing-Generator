from __future__ import annotations

from tts_backends import TTSBackend, SynthesisRequest, SynthesisResult, register_backend

import gc
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_ROOT_DIR   = Path(__file__).parent.parent
_VOICES_DIR = _ROOT_DIR / "models" / "piper" / "voices"
DEFAULT_SR  = 22050


def _scan_voices() -> List[Path]:
    if not _VOICES_DIR.exists():
        return []
    return sorted(p for p in _VOICES_DIR.rglob("*.onnx") if not p.name.endswith(".onnx.json"))


def _voice_options() -> List[Tuple[str, str]]:
    voices = _scan_voices()
    if not voices:
        return [("", "No voices found — place .onnx files in models/piper/voices/")]
    opts = []
    for p in voices:
        rel   = p.relative_to(_VOICES_DIR)
        parts = list(rel.parts)
        label = f"{p.stem}  [{'/'.join(parts[:-1])}]" if len(parts) > 1 else p.stem
        opts.append((p.stem, label))
    return opts


def _resolve_voice(stem: str) -> Optional[Path]:
    if not stem:
        voices = _scan_voices()
        return voices[0] if voices else None
    for p in _scan_voices():
        if p.stem == stem:
            return p
    return None


_PIPER_PARAMS: List[Dict[str, Any]] = [
    {
        "key":     "voice_model",
        "type":    "combo",
        "label":   "Voice",
        "options": _voice_options(),
        "default": (_voice_options()[0][0] if _voice_options() else ""),
        "tip":     f"Piper voice model (.onnx) from {_VOICES_DIR}",
    },
    {
        "key":     "length_scale",
        "type":    "slider",
        "label":   "Speech rate",
        "min":     0.25,
        "max":     4.0,
        "default": 1.0,
        "tip":     "Duration multiplier: 1.0 = normal, <1 faster, >1 slower",
    },
    {
        "key":     "noise_scale",
        "type":    "slider",
        "label":   "Variation",
        "min":     0.0,
        "max":     1.0,
        "default": 0.667,
        "tip":     "Speech variation/expressiveness (default 0.667)",
    },
    {
        "key":     "sentence_silence",
        "type":    "dspinbox",
        "label":   "Sentence pause (s)",
        "min":     0.0,
        "max":     3.0,
        "default": 0.3,
        "step":    0.1,
        "tip":     "Seconds of silence inserted between sentences",
    },
]


@register_backend
class PiperBackend(TTSBackend):

    def __init__(self):
        self.model_dir = _VOICES_DIR
        self._voice = None
        self._loaded_voice_path: Optional[str] = None
        self._loaded = False
        _VOICES_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "Piper"

    @property
    def model_id(self) -> str:
        return "piper"

    @property
    def display_name(self) -> str:
        return "🔊 Piper TTS"

    @property
    def default_sample_rate(self) -> int:
        return DEFAULT_SR

    @property
    def auth_required(self) -> bool:
        return False

    @property
    def venv_names(self) -> List[str]:
        return ["venv_piper"]

    @property
    def download_repo(self) -> str:
        return "rhasspy/piper-voices"

    @property
    def download_size(self) -> str:
        return "30–200 MB per voice"

    @property
    def header_icon(self) -> str:
        return "🔊"

    @property
    def header_title(self) -> str:
        return "Piper TTS"

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
    def device_info(self) -> str:
        return "CPU (ONNX)"

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        opts = _voice_options()
        params = list(_PIPER_PARAMS)
        params[0] = {
            **params[0],
            "options": opts,
            "default": opts[0][0] if opts else "",
        }
        return params

    def is_available(self) -> bool:
        return bool(_scan_voices())

    def load(self, progress_cb: Optional[Callable] = None) -> None:
        self.load_model(progress_cb)

    def load_model(self, progress_cb: Optional[Callable] = None) -> None:
        def status(msg: str) -> None:
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        _VOICES_DIR.mkdir(parents=True, exist_ok=True)

        voices = _scan_voices()
        if not voices:
            raise RuntimeError(
                f"No .onnx voice files found in:\n{_VOICES_DIR}\n\n"
                "Download voices from: https://huggingface.co/rhasspy/piper-voices\n\n"
                "Each voice needs two files:\n"
                "  <name>.onnx\n"
                "  <name>.onnx.json"
            )

        status(f"Piper TTS ready — {len(voices)} voice(s) available.")
        self._loaded = True

    def unload(self) -> None:
        self.unload_model()

    def unload_model(self, progress_cb: Optional[Callable] = None) -> None:
        def status(msg: str) -> None:
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        if not self._loaded:
            return

        status("Releasing Piper voice from memory…")
        self._voice = None
        self._loaded_voice_path = None
        self._loaded = False
        gc.collect()
        status("Piper TTS unloaded.")

    def download(self, model_dir: Path, progress_cb: Optional[Callable] = None) -> None:
        raise RuntimeError(
            "Piper voices are not downloaded automatically.\n\n"
            f"Place .onnx and .onnx.json files in:\n{_VOICES_DIR}\n\n"
            "Download from:\nhttps://huggingface.co/rhasspy/piper-voices\n\n"
            "Example for Polish:\n"
            "  pl_PL-bass-high.onnx\n"
            "  pl_PL-bass-high.onnx.json"
        )

    def synthesize(
        self,
        request: SynthesisRequest,
        progress_cb: Optional[Callable] = None,
    ) -> SynthesisResult:
        audio, sr = self.generate(
            text=request.text,
            progress_cb=progress_cb,
        )
        return SynthesisResult(audio=audio, sample_rate=sr, duration_s=len(audio) / sr)

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        voice_model: str = "",
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        sentence_silence: float = 0.3,
        progress_cb: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        if not self._loaded:
            raise RuntimeError("Piper model not loaded. Click 'Load model' first.")

        def status(msg: str) -> None:
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        effective_voice = str(voice_model) if voice_model else ""
        if reference_text and reference_text.strip() and _resolve_voice(reference_text.strip()) is not None:
            effective_voice = reference_text.strip()

        onnx_path = _resolve_voice(effective_voice)
        if onnx_path is None:
            raise RuntimeError(
                f"Voice not found: {effective_voice!r}\n"
                f"Voices directory: {_VOICES_DIR}"
            )

        if str(onnx_path) != self._loaded_voice_path:
            status(f"Loading voice: {onnx_path.name}…")
            try:
                from piper import PiperVoice
                self._voice = PiperVoice.load(str(onnx_path))
                self._loaded_voice_path = str(onnx_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load Piper voice {onnx_path.name}:\n{e}")

        status(f"Synthesizing with {onnx_path.stem}…")

        try:
            sr = int(self._voice.config.sample_rate)
        except Exception:
            sr = DEFAULT_SR

        audio_float = self._synthesize_audio(
            text=text,
            length_scale=float(length_scale),
            noise_scale=float(noise_scale),
            sentence_silence=float(sentence_silence),
            sr=sr,
        )

        if audio_float.size == 0:
            raise RuntimeError("Piper returned empty audio.")

        status(f"✓ Generated {len(audio_float) / max(1, sr):.1f}s of audio")
        return audio_float, sr

    def _synthesize_audio(
        self,
        text: str,
        length_scale: float,
        noise_scale: float,
        sentence_silence: float,
        sr: int,
    ) -> np.ndarray:
        if hasattr(self._voice, "synthesize_stream_raw"):
            chunks: List[bytes] = []
            for chunk in self._voice.synthesize_stream_raw(
                text,
                length_scale=length_scale,
                noise_scale=noise_scale,
                sentence_silence=sentence_silence,
            ):
                chunks.append(chunk)
            raw = b"".join(chunks)
            return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        silence_samples = int(sr * sentence_silence)
        silence_block = np.zeros(silence_samples, dtype=np.float32)
        segments: List[np.ndarray] = []

        if hasattr(self._voice, "synthesize"):
            try:
                from piper import SynthesisConfig
                syn_cfg = SynthesisConfig(
                    length_scale=length_scale,
                    noise_scale=noise_scale,
                )
                for chunk in self._voice.synthesize(text, syn_config=syn_cfg):
                    raw = getattr(chunk, "audio_int16_bytes", None)
                    if raw is None:
                        raw = bytes(chunk) if isinstance(chunk, (bytes, bytearray)) else b""
                    seg = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    if seg.size > 0:
                        segments.append(seg)
                        if silence_samples > 0:
                            segments.append(silence_block)
            except ImportError:
                for chunk in self._voice.synthesize(text):
                    raw = getattr(chunk, "audio_int16_bytes", None)
                    if raw is None:
                        raw = bytes(chunk) if isinstance(chunk, (bytes, bytearray)) else b""
                    seg = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    if seg.size > 0:
                        segments.append(seg)
                        if silence_samples > 0:
                            segments.append(silence_block)

            if segments:
                return np.concatenate(segments)
            return np.zeros(0, dtype=np.float32)

        import io
        import wave as _wave
        buf = io.BytesIO()
        with _wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            try:
                self._voice.synthesize(text, wf)
            except TypeError:
                self._voice.synthesize_wav(text, wf)
        buf.seek(0)
        with _wave.open(buf, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
