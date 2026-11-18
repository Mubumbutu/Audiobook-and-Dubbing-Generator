from __future__ import annotations

from tts_backends import TTSBackend, SynthesisRequest, SynthesisResult, register_backend

import atexit
import gc
import logging
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

_ROOT_DIR = Path(__file__).parent.parent
_MODEL_DIR = _ROOT_DIR / "models" / "breeze-tts-2"
_REPO_DIR = _ROOT_DIR / "repo" / "breeze-tts"
_GIT_URL = "https://github.com/breezeblue-ai/breeze-tts.git"
MODEL_REPO = "BreezeBlue/Breeze-TTS-2"
DEFAULT_SR = 24000
_HOST = "127.0.0.1"
_FAST_ALL = False
_LOAD_TIMEOUT_S = 900
_REQUEST_TIMEOUT = (10, 600)

BREEZE_MODES: List[Tuple[str, str]] = [
    ("clone", "Voice Clone — reference audio + exact transcript"),
    ("design", "Voice Design — from description (no reference)"),
    ("direction", "Voice Direction — reference audio + instruction"),
]

_BREEZE_PARAMS: List[Dict[str, Any]] = [
    {
        "key": "breeze_mode",
        "type": "combo",
        "label": "Mode",
        "options": BREEZE_MODES,
        "default": "clone",
        "tip": (
            "Voice Clone: needs reference audio and its exact transcript.\n"
            "Voice Design: describe the voice in the Instruction field, no reference audio.\n"
            "Voice Direction: reference audio + transcript, steered by the Instruction field."
        ),
    },
    {
        "key": "breeze_instruction",
        "type": "text",
        "label": "Instruction",
        "default": "",
        "tip": (
            "Natural-language instruction for Voice Design / Voice Direction.\n"
            "Write it in the same language as the target text (English or Chinese).\n"
            "Vocal events inline in text: (laugh), (cough), (clears throat), (sigh); "
            "Chinese: [笑], [咳嗽], [清嗓子], [叹气]."
        ),
    },
    {
        "key": "breeze_cfg_scale",
        "type": "slider",
        "label": "CFG scale",
        "min": 1.0,
        "max": 6.0,
        "default": 4.0,
        "tip": "Instruction-following strength (4 recommended). Ignored in Voice Clone.",
    },
    {
        "key": "breeze_seed",
        "type": "spinbox",
        "label": "Seed (-1 = random)",
        "min": -1,
        "max": 2147483647,
        "default": -1,
        "step": 1,
        "tip": "Fixed seed for reproducible output. -1 sends no seed.",
    },
]


def _report(cb: Optional[Callable], msg: str, frac: Optional[float] = None) -> None:
    if not cb:
        return
    try:
        if frac is None:
            cb(msg)
        else:
            cb(msg, frac)
    except TypeError:
        cb(msg)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((_HOST, 0))
        return s.getsockname()[1]


@register_backend
class Breeze2Backend(TTSBackend):

    def __init__(self):
        self.model_dir = _MODEL_DIR
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._proc: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        self._log_path: Optional[str] = None
        self._loaded = False
        atexit.register(self._kill_server)

    @property
    def name(self) -> str:
        return "Breeze TTS 2"

    @property
    def model_id(self) -> str:
        return "breeze_2"

    @property
    def display_name(self) -> str:
        return "🌬️ Breeze TTS 2"

    @property
    def default_sample_rate(self) -> int:
        return DEFAULT_SR

    @property
    def auth_required(self) -> bool:
        return False

    @property
    def venv_names(self) -> List[str]:
        return ["venv_breeze_2"]

    @property
    def download_repo(self) -> str:
        return MODEL_REPO

    @property
    def download_size(self) -> str:
        return "~7.7 GB"

    @property
    def header_icon(self) -> str:
        return "🌬️"

    @property
    def header_title(self) -> str:
        return "Breeze TTS 2"

    @property
    def whisper_incompatible(self) -> bool:
        return False

    @property
    def pyannote_incompatible(self) -> bool:
        return False

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return _BREEZE_PARAMS

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._proc is not None and self._proc.poll() is None

    @property
    def device_info(self) -> str:
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
                return f"CUDA — {name} ({mem} GB)"
            except Exception:
                return "CUDA"
        return "CPU (Breeze TTS 2 requires a CUDA GPU)"

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
            **(request.extra or {}),
        )
        return SynthesisResult(audio=audio, sample_rate=sr, duration_s=len(audio) / sr)

    def download(self, model_dir: Path, progress_cb: Optional[Callable] = None) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        _report(progress_cb, f"Downloading {MODEL_REPO}…", 0.0)
        snapshot_download(repo_id=MODEL_REPO, local_dir=str(model_dir))
        if not (_REPO_DIR / "breeze_infer").exists():
            _report(progress_cb, "Cloning breeze-tts inference code…", 0.9)
            try:
                _REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    ["git", "clone", _GIT_URL, str(_REPO_DIR)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except Exception as e:
                logger.warning(f"Could not clone breeze-tts repo: {e}")
                _report(
                    progress_cb,
                    f"Model downloaded, but cloning failed — clone {_GIT_URL} to {_REPO_DIR} manually.",
                )
                return
        _report(progress_cb, "Model downloaded successfully!", 1.0)

    def _log_tail(self, lines: int = 25) -> str:
        if not self._log_path:
            return ""
        try:
            with open(self._log_path, "r", encoding="utf-8", errors="replace") as fh:
                return "".join(fh.readlines()[-lines:])
        except Exception:
            return ""

    def _kill_server(self) -> None:
        proc = self._proc
        self._proc = None
        self._port = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        except Exception:
            pass

    def load_model(self, progress_cb: Optional[Callable] = None) -> None:
        def status(msg):
            logger.info(msg)
            _report(progress_cb, msg)

        if self.is_loaded:
            return

        if not self.model_dir.exists():
            raise RuntimeError(
                f"Model directory does not exist:\n{self.model_dir}\n"
                "Download the model using the 'Download model' button."
            )
        if not (_REPO_DIR / "breeze_infer").exists():
            raise RuntimeError(
                f"Breeze inference code not found in:\n{_REPO_DIR}\n\n"
                f"Clone it with:\ngit clone {_GIT_URL} \"{_REPO_DIR}\""
            )

        self._kill_server()
        self._port = _free_port()

        cmd = [
            sys.executable, "-m", "breeze_infer.api", str(self.model_dir),
            "--host", _HOST, "--port", str(self._port),
        ]
        if _FAST_ALL:
            cmd.append("--fast-all")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(_REPO_DIR) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"

        fd, self._log_path = tempfile.mkstemp(suffix=".log", prefix="breeze2_")
        os.close(fd)
        log_fh = open(self._log_path, "w", encoding="utf-8")

        popen_kwargs: Dict[str, Any] = {}
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        status("Starting Breeze TTS 2 server…")
        status("  (first load may take 1–3 min)")
        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(_REPO_DIR),
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                **popen_kwargs,
            )
        except Exception as e:
            log_fh.close()
            raise RuntimeError(f"Failed to start Breeze TTS 2 server:\n{e}")
        finally:
            log_fh.close()

        start = time.time()
        last_msg = 0.0
        while True:
            if self._proc.poll() is not None:
                tail = self._log_tail()
                self._kill_server()
                raise RuntimeError(f"Breeze TTS 2 server exited during startup:\n{tail}")
            try:
                with socket.create_connection((_HOST, self._port), timeout=1):
                    break
            except OSError:
                pass
            elapsed = time.time() - start
            if elapsed > _LOAD_TIMEOUT_S:
                tail = self._log_tail()
                self._kill_server()
                raise RuntimeError(f"Breeze TTS 2 server startup timed out:\n{tail}")
            if elapsed - last_msg >= 10:
                last_msg = elapsed
                status(f"Loading Breeze TTS 2… {int(elapsed)}s")
            time.sleep(1.0)

        self._loaded = True
        status(f"✓ Breeze TTS 2 loaded on {self.device}!")

    def unload_model(self, progress_cb: Optional[Callable] = None) -> None:
        def status(msg):
            logger.info(msg)
            _report(progress_cb, msg)

        if not self._loaded and self._proc is None:
            return

        status("Releasing Breeze TTS 2 from memory…")
        self._kill_server()
        self._loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        status("✓ Breeze TTS 2 unloaded")

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        breeze_mode: str = "clone",
        breeze_instruction: str = "",
        breeze_cfg_scale: float = 4.0,
        breeze_seed: int = -1,
        progress_cb: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        if not self.is_loaded or self._port is None:
            raise RuntimeError("Breeze TTS 2 model not loaded. Click 'Load model' first.")

        def status(msg):
            logger.info(msg)
            _report(progress_cb, msg)

        mode = (breeze_mode or "clone").lower()
        instruction = (breeze_instruction or "").strip()
        ref_text = (reference_text or "").strip()
        has_ref = bool(reference_audio_path) and os.path.exists(reference_audio_path)

        fields: Dict[str, str] = {"text": text}
        use_ref = False

        if mode == "clone":
            if not has_ref or not ref_text:
                raise RuntimeError(
                    "Voice Clone requires a reference audio file and its exact transcript."
                )
            use_ref = True
        elif mode == "design":
            if not instruction:
                raise RuntimeError("Voice Design requires a voice description in the Instruction field.")
            fields["instruction"] = instruction
            fields["cfg_scale"] = str(float(breeze_cfg_scale))
        elif mode == "direction":
            if not has_ref or not ref_text:
                raise RuntimeError(
                    "Voice Direction requires a reference audio file and its exact transcript."
                )
            if not instruction:
                raise RuntimeError("Voice Direction requires an instruction in the Instruction field.")
            use_ref = True
            fields["instruction"] = instruction
            fields["cfg_scale"] = str(float(breeze_cfg_scale))
        else:
            raise RuntimeError(f"Unknown Breeze mode: {mode}")

        if use_ref:
            fields["ref_text"] = ref_text
        if breeze_seed is not None and int(breeze_seed) >= 0:
            fields["seed"] = str(int(breeze_seed))

        files: Dict[str, Any] = {k: (None, v) for k, v in fields.items()}
        ref_fh = None
        if use_ref:
            ref_fh = open(reference_audio_path, "rb")
            files["ref_audio"] = (Path(reference_audio_path).name, ref_fh, "audio/wav")
            status(f"Using reference audio: {Path(reference_audio_path).name}")

        status(f"Synthesizing ({mode})…")
        url = f"http://{_HOST}:{self._port}/v1/audio/speech"
        chunks: List[bytes] = []
        try:
            with requests.post(url, files=files, stream=True, timeout=_REQUEST_TIMEOUT) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        chunks.append(chunk)
        except Exception as e:
            logger.error(f"Breeze TTS 2 inference error: {e}", exc_info=True)
            if self._proc is not None and self._proc.poll() is not None:
                self._loaded = False
                raise RuntimeError(f"Breeze TTS 2 server crashed:\n{self._log_tail()}")
            raise RuntimeError(f"Inference failed:\n{e}")
        finally:
            if ref_fh is not None:
                ref_fh.close()

        raw = b"".join(chunks)
        if len(raw) % 2:
            raw = raw[:-1]
        if not raw:
            raise RuntimeError("Breeze TTS 2 returned no audio.")

        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if audio.size < int(DEFAULT_SR * 0.05):
            raise RuntimeError(
                f"Breeze TTS 2 returned audio that is too short "
                f"({audio.size} samples / {audio.size / DEFAULT_SR:.3f}s)."
            )

        status(f"✓ Generated {len(audio) / DEFAULT_SR:.1f}s of audio")
        return audio, DEFAULT_SR
