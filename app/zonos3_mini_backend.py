from __future__ import annotations

from tts_backends import (
    TTSBackend, SynthesisRequest, SynthesisResult, InferenceError, register_backend,
)

import base64
import io
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import requests
import soundfile as sf

logger = logging.getLogger(__name__)

_ROOT_DIR = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WSL_DISTRO       = "Ubuntu"
IMAGE_NAME       = "zonos3-runtime:cuda12.8"
BASE_IMAGE       = "nvidia/cuda:12.8.1-devel-ubuntu22.04"
CONTAINER_NAME   = "zonos3_server"
LEGACY_CONTAINER = "sglang_omni_zonos3"
SERVER_PORT      = 8001
CONTAINER_PORT   = 1919
SERVE_MEMORY_RATIO = "0.25"
WARM_IMAGE_NAME  = "zonos3-runtime:warm"
SERVER_HOST      = "127.0.0.1"
MODEL_REPO       = "Zyphra/ZONOS3-Mini"
REPO_URL         = "https://github.com/Zyphra/ZONOS3.git"
DEFAULT_SR       = 44100

LOAD_TIMEOUT_S        = 2400
HEALTHCHECK_INTERVAL  = 3.0
HEALTHCHECK_TIMEOUT_S = 5

DOCKER_DESKTOP_STARTUP_TIMEOUT_S = 180
DOCKER_DESKTOP_POLL_INTERVAL_S   = 3.0
DOCKER_READY_CONSECUTIVE_CHECKS  = 2

_DOCKER_DESKTOP_EXE_CANDIDATES = [
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Docker" / "Docker" / "Docker Desktop.exe",
    Path(os.environ.get("ProgramW6432", r"C:\Program Files")) / "Docker" / "Docker" / "Docker Desktop.exe",
    Path(os.environ.get("LocalAppData", "")) / "Programs" / "DockerDesktop" / "Docker Desktop.exe",
]

_BUILD_DIR = _ROOT_DIR / "outputs" / "_zonos3_build"

_DOCKERFILE = (
    f"FROM {BASE_IMAGE}\n"
    "ENV DEBIAN_FRONTEND=noninteractive\n"
    "RUN apt-get update && apt-get install -y --no-install-recommends "
    "git curl ca-certificates build-essential ninja-build libnuma1 libnuma-dev "
    "libgomp1 libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*\n"
)

_VOLUMES = (
    "-v sglang_omni_hf_cache:/root/.cache/huggingface "
    "-v zonos3_workspace:/workspace "
)

_ENV_PREFIX = (
    "export PATH=/workspace/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "
    "UV_PYTHON_INSTALL_DIR=/workspace/uv-python "
    "UV_CACHE_DIR=/workspace/uv-cache "
    "UV_LINK_MODE=hardlink"
)

# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------

_ZONOS3_VOICES: List[Tuple[str, str]] = [
    ("AmericanFemale", "AmericanFemale — English (US), female"),
    ("AmericanMale", "AmericanMale — English (US), male"),
    ("BritishFemale", "BritishFemale — English (UK), female"),
    ("Savannah en American F 1", "Savannah — English (US), female"),
    ("Wyatt en American M 1", "Wyatt — English (US), male"),
    ("Imogen en British F 1", "Imogen — English (UK), female"),
    ("Rupert en British M 1", "Rupert — English (UK), male"),
    ("Cillian en Irish M 1", "Cillian — English (Irish), male"),
    ("Jett en energetic M 1", "Jett — English, energetic male"),
    ("Thaddeus en announcervoice M 1", "Thaddeus — English, announcer male"),
    ("Morgana en witch F 1", "Morgana — English, witch female"),
    ("Sakura en anime F 1", "Sakura — English, anime female"),
    ("Willow en asrm F 1", "Willow — English, ASMR female"),
    ("Liesel de F 1", "Liesel — German, female"),
    ("Wolfgang de M 1", "Wolfgang — German, male"),
    ("Gaspard fr M 1", "Gaspard — French, male"),
    ("Ginevra it F 1", "Ginevra — Italian, female"),
    ("Lorenzo it M 1", "Lorenzo — Italian, male"),
    ("Rodrigo es M 1", "Rodrigo — Spanish, male"),
    ("Ximena es F 1", "Ximena — Spanish, female"),
    ("Leonor pt F 1", "Leonor — Portuguese, female"),
    ("Thiago pt M 1", "Thiago — Portuguese, male"),
    ("Haoran zh energetic M 1", "Haoran — Chinese, energetic male"),
    ("Meilin zh energetic F 1", "Meilin — Chinese, energetic female"),
    ("Minjun ko M 1", "Minjun — Korean, male"),
    ("Seoyeon ko F 1", "Seoyeon — Korean, female"),
    ("Ananya hi F 1", "Ananya — Hindi, female"),
    ("Arjun hi M 1", "Arjun — Hindi, male"),
    ("Chaitanya te M 1", "Chaitanya — Telugu, male"),
    ("Sravya te F 1", "Sravya — Telugu, female"),
]

_ZONOS3_LANGUAGES: List[Tuple[str, str]] = [
    ("en_us", "English (US)"), ("en_gb", "English (UK)"), ("fr_fr", "French"),
    ("es", "Spanish"), ("de", "German"), ("it", "Italian"), ("pt_br", "Portuguese"),
    ("cmn", "Mandarin Chinese"), ("ja", "Japanese"), ("ko", "Korean"),
    ("nl", "Dutch"), ("el", "Greek"), ("sv", "Swedish"), ("da", "Danish"),
    ("no", "Norwegian"), ("fi", "Finnish"), ("pl", "Polish"), ("cs", "Czech"),
    ("sk", "Slovak"), ("hu", "Hungarian"), ("ro", "Romanian"),
    ("bg", "Bulgarian"), ("ru", "Russian"), ("uk", "Ukrainian"),
    ("tr", "Turkish"), ("ar", "Arabic"), ("he", "Hebrew"), ("ka", "Georgian"),
    ("vi", "Vietnamese"), ("id", "Indonesian"), ("ms", "Malay"), ("th", "Thai"),
    ("tl", "Tagalog"), ("hi", "Hindi"), ("bn", "Bengali"), ("ta", "Tamil"),
    ("te", "Telugu"), ("mr", "Marathi"), ("gu", "Gujarati"), ("kn", "Kannada"),
    ("ml", "Malayalam"), ("pa", "Punjabi"), ("or", "Odia"), ("ur", "Urdu"),
]

_ZONOS3_SERVER_LANGUAGES = frozenset(
    {"en_us", "en_gb", "fr_fr", "de", "es", "it", "pt_br", "ja", "cmn", "ko"}
)

_ZONOS3_PARAMS: List[Dict[str, Any]] = [
    {
        "key": "zonos3_voice", "type": "combo", "label": "Built-in voice",
        "options": _ZONOS3_VOICES, "default": "AmericanFemale",
        "tip": "Default voice used when no reference audio is loaded",
    },
    {
        "key": "zonos3_language", "type": "combo", "label": "Language",
        "options": _ZONOS3_LANGUAGES, "default": "en_us",
        "tip": "Language of the synthesized text",
    },
    {
        "key": "zonos3_temperature", "type": "slider", "label": "Temperature",
        "min": 0.1, "max": 1.5, "default": 0.75,
        "tip": "Sampling temperature",
    },
    {
        "key": "zonos3_top_p", "type": "slider", "label": "Top-P",
        "min": 0.1, "max": 1.0, "default": 0.70,
        "tip": "Nucleus sampling threshold",
    },
    {
        "key": "zonos3_topk", "type": "spinbox", "label": "Top-K",
        "min": 1, "max": 200, "default": 80, "step": 1,
        "tip": "Top-k sampling",
    },
    {
        "key": "zonos3_repetition_penalty", "type": "slider", "label": "Repetition penalty",
        "min": 1.0, "max": 4.0, "default": 2.4,
        "tip": "Penalty for repeated audio codes",
    },
    {
        "key": "zonos3_seed", "type": "spinbox", "label": "Seed (0=random)",
        "min": 0, "max": 99999, "default": 0, "step": 1,
        "tip": "Random seed (0 = random each time)",
    },
]

# ---------------------------------------------------------------------------
# WSL / Docker helpers
# ---------------------------------------------------------------------------


class WSLServerError(InferenceError):
    pass


def _wsl_available() -> bool:
    if shutil.which("wsl") is None:
        return False
    try:
        result = subprocess.run(
            ["wsl", "--status"],
            capture_output=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return result.returncode == 0
    except Exception:
        return False


def _distro_running(distro: str) -> bool:
    try:
        result = subprocess.run(
            ["wsl", "--list", "--running", "--quiet"],
            capture_output=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        if result.returncode != 0:
            return False

        output = result.stdout.decode("utf-16-le", errors="ignore")
        running_distros = {
            line.strip().lstrip("*").strip().lower()
            for line in output.replace("\ufeff", "").splitlines()
            if line.strip()
        }

        return distro.strip().lower() in running_distros

    except Exception:
        return False


def _run_wsl(
    command: str,
    distro: str = WSL_DISTRO,
    timeout: Optional[int] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    full_cmd = ["wsl", "-d", distro, "--", "bash", "-lc", command]
    kwargs: Dict[str, Any] = dict(
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        result = subprocess.run(full_cmd, **kwargs)
    except subprocess.TimeoutExpired:
        if check:
            raise WSLServerError(
                f"WSL command timed out after {timeout}s (Docker not responding):\n$ {command}"
            )
        return subprocess.CompletedProcess(full_cmd, 124, "", "timeout")
    if check and result.returncode != 0:
        raise WSLServerError(
            f"WSL command failed (code {result.returncode}):\n"
            f"$ {command}\n\n"
            f"stdout:\n{result.stdout[-3000:]}\n\nstderr:\n{result.stderr[-3000:]}"
        )
    return result


def _run_wsl_with_retry(
    command: str,
    status_cb: Optional[Callable[[str], None]] = None,
    distro: str = WSL_DISTRO,
    timeout: Optional[int] = 60,
    attempts: int = 3,
    retry_delay: float = 5.0,
) -> subprocess.CompletedProcess:
    last_error: Optional[WSLServerError] = None
    for attempt in range(1, attempts + 1):
        try:
            return _run_wsl(command, distro=distro, timeout=timeout)
        except WSLServerError as e:
            last_error = e
            if attempt < attempts:
                if status_cb:
                    status_cb(
                        f"Docker command failed (attempt {attempt}/{attempts}), "
                        f"retrying in {retry_delay:.0f}s…"
                    )
                time.sleep(retry_delay)
    assert last_error is not None
    raise last_error


def _windows_path_to_wsl(path: Path) -> str:
    p = str(Path(path).resolve())
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:].replace("\\", "/").lstrip("/")
        return f"/mnt/{drive}/{rest}"
    return p.replace("\\", "/")


def _docker_desktop_process_running() -> bool:
    if os.name != "nt":
        return True
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq Docker Desktop.exe"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return "Docker Desktop.exe" in (result.stdout or "")
    except Exception:
        return False


def _find_docker_desktop_exe() -> Optional[Path]:
    for candidate in _DOCKER_DESKTOP_EXE_CANDIDATES:
        try:
            if candidate.is_file():
                return candidate
        except Exception:
            continue
    return None


def _launch_docker_desktop() -> bool:
    exe = _find_docker_desktop_exe()
    if exe is None:
        return False
    try:
        popen_kwargs: Dict[str, Any] = {"close_fds": True}
        if os.name == "nt":
            popen_kwargs["creationflags"] = (
                subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            )
        subprocess.Popen([str(exe)], **popen_kwargs)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


@register_backend
class Zonos3MiniWSLBackend(TTSBackend):
    """
    ZONOS3-Mini (Zyphra) served by its own Mini-SGLang server in a Docker container on WSL2.
    Windows app talks to http://127.0.0.1:8001 (/v2/generate, /v2/speakers, /v2/capabilities).
    """

    def __init__(self):
        self.device    = "cpu"
        self._loaded   = False
        self._base_url = f"http://{SERVER_HOST}:{SERVER_PORT}"

    # -- metadata ----------------------------------------------------------

    @property
    def name(self) -> str:
        return "ZONOS3-Mini (WSL2 / Docker)"

    @property
    def model_id(self) -> str:
        return "zonos3_mini"

    @property
    def display_name(self) -> str:
        return "🗣 ZONOS3-Mini (WSL2)"

    @property
    def default_sample_rate(self) -> int:
        return DEFAULT_SR

    @property
    def auth_required(self) -> bool:
        return False

    @property
    def venv_names(self) -> List[str]:
        return ["venv_zonos3_mini"]

    @property
    def download_repo(self) -> str:
        return MODEL_REPO

    @property
    def download_size(self) -> str:
        return "~9 GB (model) + ~10 GB (CUDA image, Python env)"

    @property
    def header_icon(self) -> str:
        return "🗣"

    @property
    def header_title(self) -> str:
        return "ZONOS3-Mini"

    @property
    def whisper_incompatible(self) -> bool:
        return True
        
    @property
    def reference_text_supported(self) -> bool:
        return False

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return _ZONOS3_PARAMS

    # -- state ---------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device_info(self) -> str:
        if self._loaded:
            return "WSL2 / Docker — ZONOS3 server (GPU in container)"
        return "WSL2 / Docker (not running)"

    @property
    def model_dir(self) -> Path:
        return _ROOT_DIR / "models" / "zonos3_mini"

    @property
    def _marker(self) -> Path:
        return _ROOT_DIR / "models" / ".zonos3_mini_runtime_ok"

    def is_available(self) -> bool:
        return self._marker.exists()

    # -- container helpers -------------------------------------------------

    def _health_ok(self) -> bool:
        try:
            r = requests.get(f"{self._base_url}/v2/health", timeout=HEALTHCHECK_TIMEOUT_S)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def _container_exists(self) -> bool:
        result = _run_wsl(
            f"docker ps -a --filter name=^/{CONTAINER_NAME}$ --format '{{{{.Names}}}}'",
            check=False,
        )
        return CONTAINER_NAME in (result.stdout or "")

    def _container_running(self) -> bool:
        result = _run_wsl(
            f"docker ps --filter name=^/{CONTAINER_NAME}$ --format '{{{{.Names}}}}'",
            check=False,
        )
        return CONTAINER_NAME in (result.stdout or "")

    def _docker_daemon_ok(self) -> bool:
        try:
            result = _run_wsl("docker info", timeout=15, check=False)
        except Exception:
            return False
        if result.returncode != 0:
            return False
        return "could not be found" not in (result.stdout or "").lower()

    def _ensure_docker_ready(self, status: Callable[[str], None]) -> None:
        status("Checking Docker availability in WSL2…")

        if not _docker_desktop_process_running():
            status("Docker Desktop is not running — starting it automatically…")
            if not _launch_docker_desktop():
                raise InferenceError(
                    "Docker is not responding inside WSL2, and Docker Desktop could not "
                    "be started automatically (Docker Desktop.exe was not found in the "
                    "usual install locations).\n"
                    "Start Docker Desktop manually and try again."
                )

        consecutive_ok = 0
        start = time.monotonic()
        last_log = 0.0
        while time.monotonic() - start < DOCKER_DESKTOP_STARTUP_TIMEOUT_S:
            if self._docker_daemon_ok():
                consecutive_ok += 1
                if consecutive_ok >= DOCKER_READY_CONSECUTIVE_CHECKS:
                    status("✓ Docker is ready and responding in WSL2.")
                    return
            else:
                consecutive_ok = 0

            elapsed = time.monotonic() - start
            if elapsed - last_log > 10:
                status(
                    f"Waiting for Docker to become ready in WSL2… "
                    f"({elapsed:.0f}s / {DOCKER_DESKTOP_STARTUP_TIMEOUT_S}s)"
                )
                last_log = elapsed
            time.sleep(DOCKER_DESKTOP_POLL_INTERVAL_S)

        raise InferenceError(
            f"Docker did not become ready in WSL2 within {DOCKER_DESKTOP_STARTUP_TIMEOUT_S}s.\n"
            "Make sure Docker Desktop started correctly (the whale icon should appear in "
            "the system tray) and that WSL integration is enabled for the "
            f"'{WSL_DISTRO}' distribution under Settings → Resources → WSL Integration, "
            "then try again. If it's enabled and this still happens, try running "
            "'wsl --shutdown' in PowerShell, wait a few seconds, then retry."
        )

    def _container_outdated(self) -> bool:
        if not self._container_exists():
            return False

        result = _run_wsl(
            f"docker inspect -f '{{{{json .Config.Cmd}}}} {{{{json .Config.Env}}}}' {CONTAINER_NAME}",
            timeout=30,
            check=False,
        )

        config = result.stdout or ""

        required_flags = (
            f"--memory-ratio {SERVE_MEMORY_RATIO}",
            "--max-running-requests 1",
            "--cuda-graph-max-bs 1",
            "--max-prefill-length 128",
        )

        if any(flag not in config for flag in required_flags):
            return True

        return "ZONOS3_SPEAKER_EMBEDDER_DEVICE=cpu" not in config

    def _create_container(self, image: str, status: Callable[[str], None]) -> None:
        hf_token = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")

        env_flags = (
            f"-e ZONOS3_SPEAKER_EMBEDDER_DEVICE=cpu "
        )

        if hf_token:
            env_flags += f"-e HF_TOKEN={hf_token} "

        serve_cmd = (
            f"{_ENV_PREFIX}; "
            f"cd /workspace/ZONOS3 && "
            f"servers/sgl/scripts/serve.sh "
            f"--host 0.0.0.0 "
            f"--port {CONTAINER_PORT} "
            f"--memory-ratio {SERVE_MEMORY_RATIO} "
            f"--max-running-requests 1 "
            f"--cuda-graph-max-bs 1 "
            f"--max-prefill-length 128"
        )

        run_cmd = (
            f"docker run -d --name {CONTAINER_NAME} "
            f"--gpus all --shm-size 16g --ipc host "
            f"-p {SERVER_PORT}:{CONTAINER_PORT} "
            f"{env_flags}"
            f"{_VOLUMES}"
            f"{image} "
            f"bash -lc '{serve_cmd}'"
        )

        _run_wsl_with_retry(run_cmd, status)

    def _migrate_container(self, status: Callable[[str], None]) -> None:
        status("Saving the compiled-kernel cache of the current container…")
        _run_wsl(f"docker commit {CONTAINER_NAME} {WARM_IMAGE_NAME}", timeout=1800)
        status("Recreating the container with lower GPU memory use…")
        _run_wsl(f"docker rm -f {CONTAINER_NAME}", timeout=120, check=False)
        self._create_container(WARM_IMAGE_NAME, status)

    # -- load / unload -----------------------------------------------------

    def load(self, progress_cb: Optional[Callable] = None) -> None:
        self.load_model(progress_cb)

    def unload(self) -> None:
        self.unload_model()

    def load_model(self, progress_cb: Optional[Callable] = None) -> None:
        def status(msg: str):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        if not self._marker.exists():
            raise InferenceError(
                "Model has not been downloaded yet.\n"
                "Use the 'Download model' button to build the runtime image "
                "and download the ZONOS3-Mini weights inside WSL2."
            )

        status("Checking WSL2…")
        if not _wsl_available():
            raise InferenceError(
                "WSL2 is not installed or unavailable.\n"
                "Install it with: wsl --install\n"
                "and make sure you have the Ubuntu distribution."
            )

        if not _distro_running(WSL_DISTRO):
            status(f"Starting WSL2 distribution '{WSL_DISTRO}'…")
            try:
                subprocess.run(
                    ["wsl", "-d", WSL_DISTRO, "--", "true"],
                    timeout=60,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            except Exception as e:
                raise InferenceError(
                    f"Failed to start WSL2 ({WSL_DISTRO}):\n{e}"
                )

        status("Checking if the TTS server is already running…")

        if self._health_ok():
            if not self._container_outdated():
                status("✓ ZONOS3-Mini server is already running and responding.")
                self._loaded = True
                return

            status(
                "The running server uses old GPU memory settings — recreating the container…"
            )

        self._ensure_docker_ready(status)

        if self._container_exists() and self._container_outdated():
            self._migrate_container(status)
        elif self._container_running():
            status(
                "Container is already running, but the server isn't responding yet — waiting…"
            )
        elif self._container_exists():
            status(f"Starting existing container '{CONTAINER_NAME}'…")
            _run_wsl_with_retry(f"docker start {CONTAINER_NAME}", status)
        else:
            status(
                f"Creating and starting container '{CONTAINER_NAME}' (first run)…"
            )
            self._create_container(IMAGE_NAME, status)

        status(
            "Starting the server (first start JIT-compiles kernels, this can take several minutes)…"
        )

        start = time.monotonic()
        last_log = 0.0

        while time.monotonic() - start < LOAD_TIMEOUT_S:
            if self._health_ok():
                elapsed = time.monotonic() - start
                status(f"✓ ZONOS3-Mini loaded and ready ({elapsed:.0f}s)")
                self._loaded = True
                return

            if not self._container_running():
                logs = _run_wsl(
                    f"docker logs --tail 80 {CONTAINER_NAME}",
                    check=False,
                )

                raise InferenceError(
                    "The container stopped while the model was loading.\n\n"
                    f"Last logs:\n{logs.stdout}\n{logs.stderr}"
                )

            elapsed = time.monotonic() - start

            if elapsed - last_log > 10:
                status(f"…still loading ({elapsed:.0f}s / {LOAD_TIMEOUT_S}s)")
                last_log = elapsed

            time.sleep(HEALTHCHECK_INTERVAL)

        logs = _run_wsl(
            f"docker logs --tail 80 {CONTAINER_NAME}",
            check=False,
        )

        raise InferenceError(
            f"The ZONOS3-Mini server did not respond within {LOAD_TIMEOUT_S}s.\n\n"
            f"Last container logs:\n{logs.stdout}\n{logs.stderr}"
        )

    def unload_model(self, progress_cb: Optional[Callable] = None) -> None:
        def status(msg: str):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        if not self._loaded:
            status("Model is not loaded — nothing to release.")
            return

        status("Stopping the ZONOS3-Mini container (releasing VRAM)…")
        try:
            _run_wsl(f"docker stop -t 15 {CONTAINER_NAME}", timeout=30, check=False)
        except Exception as e:
            logger.warning(f"Could not stop the container cleanly: {e}")

        self._loaded = False
        status("✓ ZONOS3-Mini released (container stopped)")

    # -- download ------------------------------------------------------------

    def _run_job(
        self,
        name: str,
        inner: str,
        timeout: int,
        status: Callable[[str], None],
    ) -> None:
        _run_wsl(f"docker rm -f {name}", timeout=120, check=False)
        _run_wsl(
            f"docker run -d --name {name} --gpus all {_VOLUMES}{IMAGE_NAME} bash -lc '{inner}'",
            timeout=180,
        )
        start = time.monotonic()
        last_log = 0.0
        failures = 0
        while time.monotonic() - start < timeout:
            r = _run_wsl(
                f"docker inspect -f '{{{{.State.Status}}}} {{{{.State.ExitCode}}}}' {name}",
                timeout=30,
                check=False,
            )
            out = (r.stdout or "").strip()
            if not out or r.returncode != 0:
                failures += 1
                if failures >= 12:
                    raise InferenceError(
                        f"Lost contact with job container '{name}' (Docker daemon not responding)."
                    )
            else:
                failures = 0
                parts = out.split()
                if parts[0] in ("exited", "dead"):
                    code = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
                    if code != 0:
                        logs = _run_wsl(f"docker logs --tail 60 {name}", timeout=60, check=False)
                        raise InferenceError(
                            f"Job '{name}' failed (exit code {code}).\n\n"
                            f"{logs.stdout[-3000:]}\n{logs.stderr[-3000:]}"
                        )
                    _run_wsl(f"docker rm -f {name}", timeout=120, check=False)
                    return
            elapsed = time.monotonic() - start
            if elapsed - last_log > 30:
                status(f"…working ({elapsed / 60:.0f} min)")
                last_log = elapsed
            time.sleep(5)
        raise InferenceError(f"Job '{name}' did not finish within {timeout}s.")

    def download(
        self,
        model_dir: Path,
        progress_cb: Optional[Callable] = None,
    ) -> None:
        def status(msg: str, p: float = 0.0):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg, p)

        status("Checking WSL2…", 0.0)
        if not _wsl_available():
            raise InferenceError(
                "WSL2 is not installed. Install it with: wsl --install"
            )

        self._ensure_docker_ready(lambda m: status(m, 0.02))

        for stale in (LEGACY_CONTAINER, CONTAINER_NAME, "zonos3_setup_job", "zonos3_weights_job"):
            _run_wsl(f"docker rm -f {stale}", timeout=120, check=False)

        status("Building the CUDA 12.8 runtime image (this can take several minutes)…", 0.1)
        _BUILD_DIR.mkdir(parents=True, exist_ok=True)
        (_BUILD_DIR / "Dockerfile").write_text(_DOCKERFILE, encoding="utf-8")
        _run_wsl(
            f"docker build -t {IMAGE_NAME} {_windows_path_to_wsl(_BUILD_DIR)}",
            timeout=3600,
        )

        status("Cloning ZONOS3 and installing the Python environment (this can take a long time)…", 0.4)
        setup_inner = (
            "set -e; mkdir -p /workspace/bin; cd /workspace; "
            f"{_ENV_PREFIX}; "
            "(test -x /workspace/bin/uv || "
            "(curl -LsSf https://astral.sh/uv/install.sh | "
            "env UV_INSTALL_DIR=/workspace/bin UV_NO_MODIFY_PATH=1 sh)); "
            f"(test -d ZONOS3 || git clone --recurse-submodules {REPO_URL}); "
            "cd ZONOS3 && git submodule update --init --recursive && "
            "uv sync --locked --all-packages"
        )
        self._run_job("zonos3_setup_job", setup_inner, 7200, lambda m: status(m, 0.4))

        status(f"Downloading model weights {MODEL_REPO} (~9 GB)…", 0.7)
        weights_inner = (
            "set -e; "
            f"{_ENV_PREFIX}; "
            "cd /workspace/ZONOS3 && "
            f"uv run --locked --package zonos3-sgl hf download {MODEL_REPO} "
            "--include \"lm/*\" \"autoencoder/*\" \"speaker_encoders/*\""
        )
        self._run_job("zonos3_weights_job", weights_inner, 7200, lambda m: status(m, 0.7))

        model_dir.mkdir(parents=True, exist_ok=True)
        self._marker.parent.mkdir(parents=True, exist_ok=True)
        self._marker.touch(exist_ok=True)
        status("✓ Runtime image, ZONOS3 environment and weights are ready!", 1.0)

    # -- generation ----------------------------------------------------------

    def _resolve_speaker_id(self, wanted: str, status: Callable[[str], None]) -> Optional[str]:
        try:
            resp = requests.get(f"{self._base_url}/v2/speakers", timeout=30)
            resp.raise_for_status()
            speakers = resp.json().get("speakers", [])
        except Exception as e:
            raise InferenceError(f"Could not fetch /v2/speakers:\n{e}")

        if not speakers:
            return None

        wanted_l = (wanted or "").strip().lower()
        if wanted_l:
            for s in speakers:
                if str(s.get("id", "")).lower() == wanted_l:
                    return s["id"]
            for s in speakers:
                if str(s.get("label", "")).strip().lower() == wanted_l:
                    return s["id"]
            for s in speakers:
                haystack = f"{s.get('id', '')} {s.get('label', '')} {s.get('original_name', '')}".lower()
                if wanted_l in haystack:
                    return s["id"]
            available = ", ".join(str(s.get("label")) for s in speakers)
            status(f"Voice '{wanted}' not found — using the first default voice. Available: {available}")

        return speakers[0]["id"]


    def _warmup(self, status: Callable[[str], None]) -> None:
        status("Warming up the generation path (first run compiles kernels, this can take a long time)…")

        try:
            speaker_id = self._resolve_speaker_id("", lambda m: None)

            payload: Dict[str, Any] = {
                "text": "Warm up.",
                "language": "en_us",
                "stream": False,
                "format": "wav",
            }

            if speaker_id:
                payload["speaker_embedding_id"] = speaker_id

            response = requests.post(
                f"{self._base_url}/v2/generate",
                json=payload,
                timeout=(10, 3600),
            )
            response.raise_for_status()

            status("✓ Warm-up finished")

        except Exception as e:
            logger.warning(f"Warm-up failed: {e}")

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        zonos3_voice: str = "",
        zonos3_language: str = "en_us",
        zonos3_temperature: float = 0.75,
        zonos3_top_p: float = 0.70,
        zonos3_topk: int = 80,
        zonos3_repetition_penalty: float = 2.4,
        zonos3_seed: int = 0,
        progress_cb: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        def status(msg: str):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        if not self._loaded:
            raise InferenceError("Model is not loaded. Click 'Load model' first.")

        language = (zonos3_language or "en_us").strip()

        payload: Dict[str, Any] = {
            "text": text,
            "language": language,
            "stream": False,
            "format": "wav",
            "temperature": float(zonos3_temperature),
            "top_p": float(zonos3_top_p),
            "topk": int(zonos3_topk),
            "repetition_penalty": float(zonos3_repetition_penalty),
        }

        if language not in _ZONOS3_SERVER_LANGUAGES:
            status(
                f"Language '{language}' has no text normalizer on the server — "
                "sending the text as written."
            )
            payload["language"] = "en_us"
            payload["text_normalization"] = False

        if zonos3_seed and int(zonos3_seed) > 0:
            payload["seed"] = int(zonos3_seed)

        if reference_audio_path and os.path.exists(reference_audio_path):
            status(f"Voice Cloning — ref: {Path(reference_audio_path).name}")
            with open(reference_audio_path, "rb") as f:
                payload["speaker_audio_base64"] = base64.b64encode(f.read()).decode("ascii")
        else:
            speaker_id = self._resolve_speaker_id(zonos3_voice, status)
            if speaker_id:
                status(f"Built-in voice: {speaker_id}")
                payload["speaker_embedding_id"] = speaker_id
            else:
                status("Zero-shot — no voice reference")

        status("Sending request to the ZONOS3-Mini server…")
        try:
            resp = requests.post(
                f"{self._base_url}/v2/generate",
                json=payload,
                timeout=(10, 1800),
            )
        except requests.RequestException as e:
            raise InferenceError(
                f"Could not connect to the TTS server at {self._base_url}:\n{e}\n"
                "Check that the model is still loaded (the container may have stopped)."
            )

        if resp.status_code != 200:
            raise InferenceError(
                f"ZONOS3-Mini server returned error {resp.status_code}:\n{resp.text[:1500]}"
            )

        try:
            audio, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
        except Exception as e:
            raise InferenceError(f"Could not decode the audio response (WAV):\n{e}")

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        min_samples = int(sr * 0.05)
        if audio.size < min_samples:
            raise InferenceError(
                f"ZONOS3-Mini returned audio that is too short "
                f"({audio.size} samples / {audio.size / sr:.3f}s)."
            )

        status(f"✓ Generated {len(audio) / sr:.1f}s of audio")
        return audio.astype(np.float32), int(sr)

    def synthesize(
        self,
        request: SynthesisRequest,
        progress_cb: Optional[Callable] = None,
    ) -> SynthesisResult:
        audio, sr = self.generate(
            text=request.text,
            reference_audio_path=request.reference_audio,
            reference_text=request.reference_text,
            zonos3_voice=request.speaker_id or "",
            zonos3_language=request.extra.get("language", "en_us"),
            zonos3_temperature=request.temperature,
            zonos3_top_p=request.top_p,
            zonos3_topk=request.extra.get("topk", 80),
            zonos3_repetition_penalty=request.repetition_penalty,
            zonos3_seed=request.extra.get("seed", 0),
            progress_cb=progress_cb,
        )

        return SynthesisResult(
            audio=audio,
            sample_rate=sr,
            duration_s=len(audio) / sr,
        )