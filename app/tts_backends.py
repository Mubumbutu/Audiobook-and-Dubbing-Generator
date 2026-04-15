from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
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


_BACKENDS: Dict[str, type[TTSBackend]] = {}


def register_backend(cls: type[TTSBackend]) -> type[TTSBackend]:
    prop = cls.__dict__.get("model_id")
    if not isinstance(prop, property):
        raise TypeError(f"{cls.__name__}.model_id must be a @property")
    sentinel = object.__new__(cls)
    key = prop.fget(sentinel)
    _BACKENDS[key] = cls
    return cls


def all_backends() -> List[type[TTSBackend]]:
    return list(_BACKENDS.values())


def get_backend(model_id: str) -> type[TTSBackend]:
    if model_id not in _BACKENDS:
        raise ValueError(f"Unknown TTS backend: {model_id}")
    return _BACKENDS[model_id]


def create_backend(model_id: str) -> TTSBackend:
    return get_backend(model_id)()


def detect_active_backends() -> List[type[TTSBackend]]:
    candidates = [
        os.environ.get("VIRTUAL_ENV"),
        os.environ.get("CONDA_DEFAULT_ENV"),
        sys.prefix if sys.prefix != sys.base_prefix else None,
    ]

    matched: List[type[TTSBackend]] = []
    for candidate in candidates:
        if not candidate:
            continue
        venv_name = Path(candidate).name.lower()
        for cls in _BACKENDS.values():
            try:
                sentinel = object.__new__(cls)
                names = [n.lower() for n in sentinel.venv_names]
                if venv_name in names:
                    if cls not in matched:
                        matched.append(cls)
            except Exception:
                pass

    if matched:
        return matched
    return list(_BACKENDS.values())
