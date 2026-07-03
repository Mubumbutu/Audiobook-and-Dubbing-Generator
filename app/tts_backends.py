from __future__ import annotations

import importlib
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class SynthesisRequest:
    text: str
    reference_audio: Optional[str] = None
    reference_text: Optional[str] = None
    speaker_id: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.7
    repetition_penalty: float = 1.2
    max_new_tokens: int = 1024
    chunk_length: int = 200


@dataclass
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    duration_s: float


class TTSBackend(ABC):

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @property
    @abstractmethod
    def display_name(self) -> str: ...

    @property
    @abstractmethod
    def default_sample_rate(self) -> int: ...

    @property
    @abstractmethod
    def auth_required(self) -> bool: ...

    @property
    @abstractmethod
    def venv_names(self) -> List[str]: ...

    @property
    @abstractmethod
    def download_repo(self) -> str: ...

    @property
    @abstractmethod
    def download_size(self) -> str: ...

    @property
    @abstractmethod
    def header_icon(self) -> str: ...

    @property
    @abstractmethod
    def header_title(self) -> str: ...

    @property
    def whisper_incompatible(self) -> bool:
        return False

    @property
    def pyannote_incompatible(self) -> bool:
        return False

    @property
    @abstractmethod
    def generation_params(self) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def load(self, progress_cb: Optional[Callable] = None) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def synthesize(
        self,
        request: SynthesisRequest,
        progress_cb: Optional[Callable] = None,
    ) -> SynthesisResult: ...

    @abstractmethod
    def download(
        self,
        model_dir: Path,
        progress_cb: Optional[Callable] = None,
    ) -> None: ...


class InferenceError(Exception):
    pass


_BACKENDS: Dict[str, type[TTSBackend]] = {}
_LOADED_MODULES: Dict[str, Optional[ModuleType]] = {}

_NON_BACKEND_MODULES = {
    "main", "tts_backends", "input_formats",
    "srt_format", "epub_format", "txt_format",
    "pdf_format", "kindle_format", "fb2_format",
}


def register_backend(cls: type[TTSBackend]) -> type[TTSBackend]:
    prop = cls.__dict__.get("model_id")
    if not isinstance(prop, property):
        raise TypeError(f"{cls.__name__}.model_id must be a @property")
    sentinel = object.__new__(cls)
    key = prop.fget(sentinel)
    _BACKENDS[key] = cls
    return cls


def _current_venv_name() -> Optional[str]:
    for candidate in (
        os.environ.get("VIRTUAL_ENV"),
        os.environ.get("CONDA_DEFAULT_ENV"),
        sys.prefix if sys.prefix != sys.base_prefix else None,
    ):
        if candidate:
            return Path(candidate).name.lower()
    return None


def _discover_all_backend_module_names() -> List[str]:
    app_dir = Path(__file__).resolve().parent
    names = []
    for path in app_dir.glob("*.py"):
        stem = path.stem
        if stem in _NON_BACKEND_MODULES or stem.startswith("_"):
            continue
        names.append(stem)
    return names


def load_active_backend_modules() -> Dict[str, Optional[ModuleType]]:
    if _LOADED_MODULES:
        return _LOADED_MODULES

    venv_name = _current_venv_name()
    available_backends = _discover_all_backend_module_names()
    candidates: List[str] = []

    if venv_name:
        stem = venv_name[5:] if venv_name.startswith("venv_") else venv_name
        if f"{stem}_backend" in available_backends:
            candidates.append(f"{stem}_backend")
        if stem in available_backends:
            candidates.append(stem)

    found = False
    for mod_name in candidates:
        try:
            module = importlib.import_module(mod_name)
        except ModuleNotFoundError as e:
            if e.name == mod_name:
                continue
            _LOADED_MODULES[mod_name] = None
            found = True
            break
        except Exception:
            _LOADED_MODULES[mod_name] = None
            found = True
            break
        else:
            _LOADED_MODULES[mod_name] = module
            found = True
            break

    if not found:
        for mod_name in available_backends:
            try:
                _LOADED_MODULES[mod_name] = importlib.import_module(mod_name)
            except Exception:
                _LOADED_MODULES[mod_name] = None

    return _LOADED_MODULES


def get_loaded_module(module_name: str) -> Optional[ModuleType]:
    return _LOADED_MODULES.get(module_name)


def all_backends() -> List[type[TTSBackend]]:
    load_active_backend_modules()
    return list(_BACKENDS.values())


def get_backend(model_id: str) -> type[TTSBackend]:
    if model_id not in _BACKENDS:
        load_active_backend_modules()
    if model_id not in _BACKENDS:
        raise ValueError(f"Unknown TTS backend: {model_id}")
    return _BACKENDS[model_id]


def create_backend(model_id: str) -> TTSBackend:
    return get_backend(model_id)()


def detect_active_backends() -> List[type[TTSBackend]]:
    load_active_backend_modules()

    venv_name = _current_venv_name()
    matched: List[type[TTSBackend]] = []
    if venv_name:
        for cls in _BACKENDS.values():
            try:
                sentinel = object.__new__(cls)
                names = [n.lower() for n in sentinel.venv_names]
                if venv_name in names and cls not in matched:
                    matched.append(cls)
            except Exception:
                pass

    if matched:
        return matched
    return list(_BACKENDS.values())


def is_rocm() -> bool:
    try:
        import torch
        return torch.version.hip is not None
    except Exception:
        return False


def safe_compute_dtype(device: str) -> "torch.dtype":
    import torch
    if device == "cuda" and not is_rocm():
        return torch.bfloat16
    return torch.float32
