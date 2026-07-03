from __future__ import annotations

from tts_backends import (
    TTSBackend, SynthesisRequest, SynthesisResult, register_backend,
)

import io
import json
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

WSL_DISTRO       = "Ubuntu"                       # nazwa dystrybucji WSL2 (zmień jeśli inna)
DOCKER_IMAGE     = "lmsysorg/sglang-omni:dev"
CONTAINER_NAME   = "sglang_omni_higgs"
SERVER_PORT      = 8000
SERVER_HOST      = "127.0.0.1"
MODEL_REPO       = "bosonai/higgs-tts-3-4b"
DEFAULT_SR       = 24000

LOAD_TIMEOUT_S        = 1800   # pierwsze ładowanie modelu może trwać kilka minut
HEALTHCHECK_INTERVAL  = 2.0
HEALTHCHECK_TIMEOUT_S = 5

DOCKER_DESKTOP_STARTUP_TIMEOUT_S = 180
DOCKER_DESKTOP_POLL_INTERVAL_S   = 3.0
DOCKER_READY_CONSECUTIVE_CHECKS  = 2

_DOCKER_DESKTOP_EXE_CANDIDATES = [
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Docker" / "Docker" / "Docker Desktop.exe",
    Path(os.environ.get("ProgramW6432", r"C:\Program Files")) / "Docker" / "Docker" / "Docker Desktop.exe",
    Path(os.environ.get("LocalAppData", "")) / "Programs" / "DockerDesktop" / "Docker Desktop.exe",
]

# Katalog na Windowsie, w którym trzymane są pliki referencyjne dla cloningu,
# żeby kontener (z zamontowanym /mnt/host) mógł je odczytać.
_REF_AUDIO_DIR = _ROOT_DIR / "outputs" / "_higgs_ref_audio"

_HIGGS_PARAMS: List[Dict[str, Any]] = [
    {
        "key": "temperature", "type": "slider", "label": "Temperature",
        "min": 0.1, "max": 1.5, "default": 0.8,
        "tip": "Sampling temperature (wyżej = bardziej zróżnicowane wyjście)",
    },
    {
        "key": "top_k", "type": "spinbox", "label": "Top-K",
        "min": 0, "max": 200, "default": 50, "step": 1,
        "tip": "Top-k sampling (0 = wyłączone)",
    },
    {
        "key": "top_p", "type": "slider", "label": "Top-P",
        "min": 0.0, "max": 1.0, "default": 0.0,
        "tip": "Top-p sampling (0 = wyłączone)",
    },
    {
        "key": "max_new_tokens", "type": "spinbox", "label": "Max New Tokens",
        "min": 64, "max": 4096, "default": 1024, "step": 64,
        "tip": "Maksymalna liczba generowanych kroków audio",
    },
    {
        "key": "seed", "type": "spinbox", "label": "Seed (0=random)",
        "min": 0, "max": 99999, "default": 0, "step": 1,
        "tip": "Ziarno losowości (0 = losowe za każdym razem)",
    },
]


class InferenceError(Exception):
    pass


class WSLServerError(InferenceError):
    pass


def _wsl_available() -> bool:
    """Sprawdza czy komenda `wsl` istnieje i WSL2 jest zainstalowane."""
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
            ["wsl", "-l", "-v"],
            capture_output=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        out = result.stdout.decode("utf-16-le", errors="ignore") if result.stdout else ""
        return distro.lower() in out.lower()
    except Exception:
        return False


def _run_wsl(
    command: str,
    distro: str = WSL_DISTRO,
    timeout: Optional[int] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Wykonuje pojedynczą blokującą komendę bash wewnątrz WSL2 i czeka na wynik."""
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
    result = subprocess.run(full_cmd, **kwargs)
    if check and result.returncode != 0:
        raise WSLServerError(
            f"Komenda WSL zakończyła się błędem (kod {result.returncode}):\n"
            f"$ {command}\n\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
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
    """Konwertuje ścieżkę Windows (C:\\foo\\bar) na ścieżkę WSL (/mnt/c/foo/bar)."""
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


@register_backend
class HiggsWSLBackend(TTSBackend):
    """
    Backend dla Higgs TTS 3 (4B), serwowany przez SGLang-Omni wewnątrz
    kontenera Docker uruchomionego w WSL2 (dystrybucja Ubuntu).

    Architektura:
        Windows (ta aplikacja)
            │  HTTP localhost:8000
            ▼
        WSL2 (Ubuntu) → Docker Desktop backend
            │
            ▼
        Kontener `lmsysorg/sglang-omni:dev`
            └── sgl-omni serve --model-path bosonai/higgs-tts-3-4b

    "Load model" = upewnij się, że Docker Desktop działa, wystartuj
    (lub podłącz się do już działającego) kontener i poczekaj aż
    endpoint /health zacznie odpowiadać.

    "Unload model" = zatrzymaj i usuń kontener, zwalniając VRAM.
    """

    def __init__(self):
        self.device   = "cpu"   # proces Windows nie dotyka GPU bezpośrednio
        self._loaded  = False
        self._base_url = f"http://{SERVER_HOST}:{SERVER_PORT}"

    # -- metadane wymagane przez TTSBackend ---------------------------------

    @property
    def name(self) -> str:
        return "Higgs TTS 3 (WSL2 / Docker)"

    @property
    def model_id(self) -> str:
        return "higgs_tts3_wsl"

    @property
    def display_name(self) -> str:
        return "🐳 Higgs TTS 3 (WSL2)"

    @property
    def default_sample_rate(self) -> int:
        return DEFAULT_SR

    @property
    def auth_required(self) -> bool:
        # Pobranie wag z HF wymaga akceptacji licencji (token HF),
        # ale sam serwer po stronie WSL go potrzebuje, nie ta apka.
        return True

    @property
    def venv_names(self) -> List[str]:
        # Ten backend nie wymaga osobnego venv Windows poza tym co już jest
        # w głównym środowisku aplikacji (potrzebuje tylko `requests`),
        # ale rejestrujemy nazwę dla spójności z resztą projektu.
        return ["venv_higgs_wsl"]

    @property
    def download_repo(self) -> str:
        return MODEL_REPO

    @property
    def download_size(self) -> str:
        return "~9 GB (model) + ~12 GB (obraz Docker)"

    @property
    def header_icon(self) -> str:
        return "🐳"

    @property
    def header_title(self) -> str:
        return "Higgs TTS 3"

    @property
    def generation_params(self) -> List[Dict[str, Any]]:
        return _HIGGS_PARAMS

    # -- stan ------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device_info(self) -> str:
        if self._loaded:
            return "WSL2 / Docker — sgl-omni serve (GPU w kontenerze)"
        return "WSL2 / Docker (nie uruchomiony)"

    @property
    def _marker(self) -> Path:
        return _ROOT_DIR / "models" / f".{self.model_id}_ok"

    def is_available(self) -> bool:
        """
        'Dostępność' modelu sprawdzamy markerem na Windowsie ustawianym po
        udanym `download()` (czyli po pociągnięciu obrazu Docker + wagi HF
        wewnątrz kontenera). Plików modelu fizycznie NIE MA po stronie
        Windows, więc nie da się tego sprawdzić przez Path.exists() na
        katalogu modelu jak w innych backendach.
        """
        return self._marker.exists()

    # -- pomocnicze: health / start / stop kontenera ----------------------

    def _health_ok(self) -> bool:
        try:
            r = requests.get(f"{self._base_url}/health", timeout=HEALTHCHECK_TIMEOUT_S)
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

    # -- TTSBackend: load / unload -----------------------------------------

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
                "Use the 'Download model' button to download the Docker image "
                "and the Higgs TTS 3 weights inside WSL2."
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
                raise InferenceError(f"Failed to start WSL2 ({WSL_DISTRO}):\n{e}")

        status("Checking if the TTS server is already running…")
        if self._health_ok():
            status("✓ Higgs TTS 3 server is already running and responding.")
            self._loaded = True
            return

        self._ensure_docker_ready(status)

        if self._container_running():
            status("Container is already running, but the server isn't responding yet — waiting…")
        elif self._container_exists():
            status(f"Starting existing container '{CONTAINER_NAME}'…")
            _run_wsl_with_retry(f"docker start {CONTAINER_NAME}", status)
        else:
            status(f"Creating and starting container '{CONTAINER_NAME}' (first run)…")
            hf_token = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
            env_flag = f"-e HF_TOKEN={hf_token}" if hf_token else ""
            ref_dir_wsl = _windows_path_to_wsl(_REF_AUDIO_DIR)
            _REF_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

            run_cmd = (
                f"docker run -d --name {CONTAINER_NAME} "
                f"--gpus all --shm-size 32g --ipc host --privileged "
                f"-p {SERVER_PORT}:{SERVER_PORT} "
                f"{env_flag} "
                f"-v {ref_dir_wsl}:/refs "
                f"-v sglang_omni_hf_cache:/root/.cache/huggingface "
                f"-v sglang_omni_workspace:/workspace "
                f"-v sglang_omni_uv_cache:/root/.cache/uv "
                f"{DOCKER_IMAGE} "
                f"bash -lc \""
                f"cd /workspace/sglang-omni && "
                f"source .venv/bin/activate && "
                f"sgl-omni serve "
                f"--model-path {MODEL_REPO} "
                f"--allowed-local-media-path /refs "
                f"--host 0.0.0.0 --port {SERVER_PORT}"
                f"\""
            )
            _run_wsl_with_retry(run_cmd, status)

        status("Waiting for the model to load into VRAM (this can take a few minutes)…")
        start = time.monotonic()
        last_log = 0.0
        while time.monotonic() - start < LOAD_TIMEOUT_S:
            if self._health_ok():
                elapsed = time.monotonic() - start
                status(f"✓ Higgs TTS 3 loaded and ready ({elapsed:.0f}s)")
                self._loaded = True
                return

            if not self._container_running():
                logs = _run_wsl(f"docker logs --tail 60 {CONTAINER_NAME}", check=False)
                raise InferenceError(
                    "The container stopped while the model was loading.\n\n"
                    f"Last logs:\n{logs.stdout}\n{logs.stderr}"
                )

            elapsed = time.monotonic() - start
            if elapsed - last_log > 10:
                status(f"…still loading ({elapsed:.0f}s / {LOAD_TIMEOUT_S}s)")
                last_log = elapsed
            time.sleep(HEALTHCHECK_INTERVAL)

        logs = _run_wsl(f"docker logs --tail 60 {CONTAINER_NAME}", check=False)
        raise InferenceError(
            f"The Higgs TTS 3 server did not respond within {LOAD_TIMEOUT_S}s.\n\n"
            f"Last container logs:\n{logs.stdout}\n{logs.stderr}"
        )

    def unload_model(self, progress_cb: Optional[Callable] = None) -> None:
        def status(msg: str):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        if not self._loaded:
            status("Model nie jest załadowany — nic do zwolnienia.")
            return

        status("Zatrzymuję kontener Higgs TTS 3 (zwalniam VRAM)…")
        try:
            _run_wsl(f"docker stop -t 15 {CONTAINER_NAME}", timeout=30, check=False)
        except Exception as e:
            logger.warning(f"Nie udało się zatrzymać kontenera czysto: {e}")

        self._loaded = False
        status("✓ Higgs TTS 3 zwolniony (kontener zatrzymany)")

    # -- TTSBackend: download ------------------------------------------------

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

        status(f"Pulling Docker image {DOCKER_IMAGE} (this can take several minutes)…", 0.1)
        _run_wsl(f"docker pull {DOCKER_IMAGE}", timeout=3600)

        status(f"Downloading model weights {MODEL_REPO} (~9 GB)…", 0.4)
        hf_token = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
        token_export = f"export HF_TOKEN={hf_token} && " if hf_token else ""
        pull_cmd = (
            f"docker run --rm --gpus all "
            f"{('-e HF_TOKEN=' + hf_token) if hf_token else ''} "
            f"-v sglang_omni_hf_cache:/root/.cache/huggingface "
            f"{DOCKER_IMAGE} "
            f"bash -lc \"{token_export}hf download {MODEL_REPO}\""
        )
        _run_wsl(pull_cmd, timeout=3600)

        status("Setting up the sgl-omni serving environment (this can take several minutes)…", 0.6)
        setup_cmd = (
            f"docker run --rm --gpus all "
            f"-e UV_LINK_MODE=symlink "
            f"-v sglang_omni_workspace:/workspace "
            f"-v sglang_omni_uv_cache:/root/.cache/uv "
            f"{DOCKER_IMAGE} "
            f"bash -lc \""
            f"mkdir -p /workspace && cd /workspace && "
            f"(test -d sglang-omni || git clone https://github.com/sgl-project/sglang-omni.git) && "
            f"cd sglang-omni && "
            f"(test -d .venv || uv venv .venv -p 3.12 --system-site-packages) && "
            f"source .venv/bin/activate && "
            f"(sgl-omni --help >/dev/null 2>&1 </dev/null || uv pip install -v -e .)"
            f"\""
        )
        _run_wsl(setup_cmd, timeout=3600)

        model_dir.mkdir(parents=True, exist_ok=True)
        self._marker.parent.mkdir(parents=True, exist_ok=True)
        self._marker.touch(exist_ok=True)
        status("✓ Model, Docker image and serving environment downloaded successfully!", 1.0)

    # -- główna metoda generacji ---------------------------------------------
    #
    # UWAGA WAŻNA: main.py NIE woła backend.synthesize(SynthesisRequest(...)).
    # TTSWorker i GenerateWorker w main.py wołają bezpośrednio:
    #
    #     backend.generate(
    #         text=..., reference_audio_path=..., reference_text=...,
    #         progress_cb=..., **generation_settings,
    #     )
    #
    # gdzie generation_settings to dict zbudowany z kluczy "key" podanych
    # w generation_params (czyli tutaj: temperature, top_k, top_p,
    # max_new_tokens, seed). Dlatego — dokładnie jak w omnivoice_backend.py —
    # to .generate() jest właściwym entry-pointem, a synthesize() poniżej to
    # tylko cienki wrapper zgodności z abstrakcyjnym interfejsem TTSBackend.

    def generate(
        self,
        text: str,
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.0,
        max_new_tokens: int = 1024,
        seed: int = 0,
        progress_cb: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        def status(msg: str):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        if not self._loaded:
            raise InferenceError("Model nie jest załadowany. Najpierw kliknij 'Load model'.")

        payload: Dict[str, Any] = {
            "input": text,
            "response_format": "wav",
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
        }

        if top_p and float(top_p) > 0.0:
            payload["top_p"] = float(top_p)

        if top_k and int(top_k) > 0:
            payload["top_k"] = int(top_k)

        if seed and int(seed) > 0:
            payload["seed"] = int(seed)

        if reference_audio_path and os.path.exists(reference_audio_path):
            status(f"Voice Cloning — ref: {Path(reference_audio_path).name}")
            wsl_ref_path = self._stage_reference_audio(reference_audio_path)
            ref_entry: Dict[str, Any] = {"audio_path": wsl_ref_path}
            if reference_text:
                ref_entry["text"] = reference_text
            payload["references"] = [ref_entry]
        else:
            status("Auto / zero-shot — bez referencji głosu")

        status("Wysyłam żądanie do serwera Higgs TTS 3…")
        try:
            resp = requests.post(
                f"{self._base_url}/v1/audio/speech",
                json=payload,
                timeout=180,
            )
        except requests.RequestException as e:
            raise InferenceError(
                f"Nie udało się połączyć z serwerem TTS pod {self._base_url}:\n{e}\n"
                "Sprawdź czy model jest nadal załadowany (kontener mógł się zatrzymać)."
            )

        if resp.status_code != 200:
            raise InferenceError(
                f"Serwer Higgs TTS 3 zwrócił błąd {resp.status_code}:\n{resp.text[:1000]}"
            )

        try:
            audio, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
        except Exception as e:
            raise InferenceError(f"Nie udało się zdekodować odpowiedzi audio (WAV):\n{e}")

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        min_samples = int(sr * 0.05)
        if audio.size < min_samples:
            raise InferenceError(
                f"Higgs TTS 3 zwrócił zbyt krótkie audio "
                f"({audio.size} próbek / {audio.size / sr:.3f}s)."
            )

        status(f"✓ Wygenerowano {len(audio) / sr:.1f}s audio")
        return audio.astype(np.float32), int(sr)

    def synthesize(
        self,
        request: SynthesisRequest,
        progress_cb: Optional[Callable] = None,
    ) -> SynthesisResult:
        """Wrapper zgodności z abstrakcyjnym interfejsem TTSBackend.
        Realna ścieżka generacji w main.py używa generate() bezpośrednio."""
        audio, sr = self.generate(
            text=request.text,
            reference_audio_path=request.reference_audio,
            reference_text=request.reference_text,
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=request.max_new_tokens,
            progress_cb=progress_cb,
        )
        return SynthesisResult(audio=audio, sample_rate=sr, duration_s=len(audio) / sr)

    # -- pomocnicze ----------------------------------------------------------

    def _stage_reference_audio(self, windows_path: str) -> str:
        """
        Kopiuje plik referencyjny do katalogu udostępnionego kontenerowi
        (zamontowanego jako /refs) i zwraca ścieżkę widzianą przez kontener.
        """
        _REF_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        src = Path(windows_path)
        dst = _REF_AUDIO_DIR / src.name
        try:
            if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
                shutil.copy2(src, dst)
        except Exception as e:
            raise InferenceError(f"Nie udało się skopiować pliku referencyjnego:\n{e}")
        return f"/refs/{src.name}"