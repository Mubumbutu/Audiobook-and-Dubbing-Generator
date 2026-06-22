# main.py
"""
SRT Lektor/Dubbing Studio
"""
 
import sys
import os
import re
import json
import time
import gc
import shutil
import logging
import tempfile
import subprocess
import traceback
import threading
import hashlib
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable
 
import numpy as np
import soundfile as sf
import sounddevice as sd
import torch
import torchaudio
import torchaudio.functional as TAF
from huggingface_hub import snapshot_download
 
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QPlainTextEdit, QTreeWidget, QTreeWidgetItem,
    QSlider, QSpinBox, QGroupBox, QFileDialog, QProgressBar,
    QScrollArea, QFrame, QSplitter, QStatusBar, QSizePolicy,
    QMessageBox, QComboBox, QCheckBox, QMenu, QTabWidget, QLineEdit,
    QGridLayout, QDialog, QDialogButtonBox, QInputDialog, QDoubleSpinBox, QHeaderView,
    QRadioButton,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QUrl, QFileSystemWatcher
from PyQt6.QtGui import (
    QFont, QColor, QPainter, QIcon, QPixmap,
    QDragEnterEvent, QDropEvent, QKeySequence, QShortcut, QPen, QAction,
    QDesktopServices,
)
 
from lingua import LanguageDetectorBuilder
from num2words import num2words
import srt_format
import epub_format
import txt_format
import pdf_format
import kindle_format
import fb2_format
import fish_s2_pro
import moss_backend
import tada
import chatterbox_backend
import omnivoice_backend
import qwen3_backend
import voxcpm2_backend
import supertonic_backend
from supertonic_backend import SUPERTONIC_VOICES
import piper_backend
from piper_backend import _voice_options as _piper_voice_options
import xttsv2_backend
from input_formats import get_format
from tts_backends import (
    SynthesisRequest, SynthesisResult,
    detect_active_backends, create_backend,
)
from fish_s2_pro import InferenceError
from srt_format import _ms_to_srt_ts as _ms_to_ts
from txt_format import txt_srt_format, txt_ebook_format
 
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
 
APP_DIR     = Path(__file__).parent
ROOT_DIR    = APP_DIR.parent
WHISPER_DIR = ROOT_DIR / "models" / "whisper"
OUTPUTS_DIR = ROOT_DIR / "outputs"
PROC_DIR    = ROOT_DIR / "outputs" / "preprocessed"
_LAST_DIRS: Dict[str, str] = {}


def _get_last_dir(key: str, fallback: str = "") -> str:
    return _LAST_DIRS.get(key) or fallback or str(Path.home())


def _set_last_dir(key: str, path: str) -> None:
    d = path if os.path.isdir(path) else str(Path(path).parent)
    if os.path.isdir(d):
        _LAST_DIRS[key] = d

os.environ.setdefault("TORCH_HOME", str(ROOT_DIR / "models" / "torch_hub"))
OUTPUTS_DIR.mkdir(exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

HF_TOKEN_FILE = ROOT_DIR / ".hf_token"
 
_ACTIVE_BACKEND_CLASSES = detect_active_backends()
MODEL_OPTIONS: List[Tuple[str, str]] = [
    (cls().model_id, cls().display_name)
    for cls in _ACTIVE_BACKEND_CLASSES
]

C = {
    "bg":         "#141414",
    "surface":    "#1e1e1e",
    "panel":      "#252525",
    "border":     "#333333",
    "border2":    "#444444",
    "accent":     "#2a6aaa",
    "accent2":    "#1a4a7a",
    "accent_dim": "#2a6aaa22",
    "text":       "#cccccc",
    "text2":      "#888888",
    "text3":      "#555555",
    "success":    "#4a9aff",
    "warning":    "#ffb300",
    "error":      "#ff5555",
    "tag_bg":     "#1e1e1e",
    "tag_hover":  "#2a2a2a",
    "player":     "#161616",
    "whisper":    "#7c4dff",
    "whisper_dim":"#7c4dff22",
    "proc":       "#ff6d00",
    "proc_dim":   "#ff6d0022",
}

STYLE = f"""
QMainWindow, QWidget {{
    background-color: {C["bg"]};
    color: {C["text"]};
    font-family: "Segoe UI","Ubuntu",sans-serif;
    font-size: 13px;
}}
QGroupBox {{
    border: 1px solid {C["border"]};
    border-radius: 5px;
    margin-top: 18px;
    padding: 12px 10px 10px 10px;
    background: {C["panel"]};
    font-weight: bold;
    color: #aaaaaa;
    font-size: 11px;
    letter-spacing: 1px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px; top: 4px;
    padding: 0 6px;
    background: {C["panel"]};
    color: #888888;
    font-size: 11px;
}}
QTreeWidget {{
    background: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 4px;
    color: {C["text"]};
    alternate-background-color: {C["panel"]};
}}
QTreeWidget::item:selected {{
    background: {C["accent2"]};
    color: {C["text"]};
}}
QTreeWidget::item:hover {{ background: {C["accent_dim"]}; }}
QHeaderView::section {{
    background: {C["panel"]};
    color: {C["text2"]};
    border: none;
    border-right: 1px solid {C["border"]};
    padding: 4px 8px;
    font-size: 11px;
}}
QTextEdit, QPlainTextEdit {{
    background: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 4px;
    color: {C["text"]};
    padding: 8px;
    selection-background-color: {C["accent2"]};
}}
QTextEdit:focus, QPlainTextEdit:focus {{ border-color: {C["accent"]}; }}
QPushButton {{
    background: {C["surface"]};
    border: 1px solid {C["border2"]};
    border-radius: 4px;
    color: {C["text"]};
    padding: 5px 14px;
    font-size: 12px;
}}
QPushButton:hover {{
    background: #333333;
    border-color: #666666;
    color: white;
}}
QPushButton:pressed {{ background: #1a1a1a; }}
QPushButton:disabled {{ color: {C["text3"]}; border-color: {C["border"]}; }}
QCheckBox {{
    color: {C["text2"]};
    font-size: 12px;
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 2px solid #555555;
    border-radius: 3px;
    background: {C["surface"]};
}}
QCheckBox::indicator:hover {{
    border-color: {C["accent"]};
    background: #252535;
}}
QCheckBox::indicator:checked {{
    background: {C["accent2"]};
    border-color: {C["accent"]};
}}
QCheckBox::indicator:checked:hover {{
    background: #2a5a9a;
}}
QSlider::groove:horizontal {{
    height: 4px; background: {C["border2"]}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {C["accent"]}; border: none;
    width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
}}
QSlider::sub-page:horizontal {{ background: {C["accent"]}; border-radius: 2px; }}
QSpinBox, QComboBox, QLineEdit {{
    background: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 3px;
    color: {C["text"]};
    padding: 4px 8px;
    min-height: 24px;
}}
QSpinBox:focus, QComboBox:focus, QLineEdit:focus {{ border-color: {C["accent"]}; }}
QComboBox::drop-down {{ border: none; padding-right: 8px; }}
QComboBox QAbstractItemView {{
    background: {C["panel"]};
    color: {C["text"]};
    selection-background-color: {C["accent2"]};
    border: 1px solid {C["border2"]};
}}
QScrollBar:vertical {{
    background: {C["surface"]}; width: 8px; border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {C["border2"]}; border-radius: 4px; min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{ background: {C["accent"]}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ height: 0; }}
QProgressBar {{
    background: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 3px;
    text-align: center; color: transparent; height: 8px;
}}
QProgressBar::chunk {{
    background: {C["accent"]};
    border-radius: 2px;
}}
QStatusBar {{
    background: {C["surface"]};
    border-top: 1px solid {C["border"]};
    color: {C["text2"]}; font-size: 11px;
}}
QLabel {{ color: {C["text"]}; }}
QSplitter::handle {{ background: {C["border"]}; width: 2px; }}
QScrollArea {{ border: none; background: transparent; }}
QTabWidget::pane {{
    border: 1px solid {C["border"]};
    border-radius: 0 3px 3px 3px;
    background: {C["panel"]};
}}
QTabBar::tab {{
    background: {C["surface"]};
    border: 1px solid {C["border"]};
    border-bottom: none;
    border-top-left-radius: 3px;
    border-top-right-radius: 3px;
    color: #777777;
    padding: 5px 14px;
    font-size: 12px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background: {C["panel"]};
    color: #cccccc;
    border-color: {C["border2"]};
}}
QTabBar::tab:hover {{ color: #aaaaaa; background: #2a2a2a; }}
"""

def _btn(color):
    return f"""
QPushButton {{
    background:#2a2a2a; border:1px solid {color};
    border-radius:4px; color:{color}; font-size:12px; font-weight:600; padding:6px 14px;
}}
QPushButton:hover{{ background:#333333; border-color:{color}; color:white; }}
QPushButton:disabled{{ color:{C["text3"]}; border-color:{C["border"]}; }}
"""

SYNTH_BTN_STYLE = f"""
QPushButton {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #1e5fa8,stop:1 #133f74);
    border: 1px solid #2a6aaa; border-radius: 5px; color: white;
    font-size: 14px; font-weight: 700; padding: 12px 32px;
}}
QPushButton:hover {{
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #2a72c8,stop:1 #1a4e8c);
    border-color: #4a8acc;
}}
QPushButton:pressed {{ background: #0e3060; }}
QPushButton:disabled {{ background: #1a2a35; color: #555; border-color: #333; }}
"""

TAG_BTN = f"""
QPushButton {{
    background:{C["tag_bg"]}; border:1px solid {C["border"]};
    border-radius:4px; color:{C["accent"]};
    font-size:11px; font-family:"Consolas","Courier New",monospace; padding:3px 8px;
}}
QPushButton:hover{{ background:{C["tag_hover"]}; border-color:{C["accent"]}; }}
"""

WHISPER_REPOS: Dict[str, str] = {
    "tiny":     "Systran/faster-whisper-tiny",
    "base":     "Systran/faster-whisper-base",
    "small":    "Systran/faster-whisper-small",
    "medium":   "Systran/faster-whisper-medium",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
}

WHISPER_SIZES = ["tiny","base","small","medium","large-v2","large-v3"]
WHISPER_SIZE_MB = {"tiny":"~75 MB","base":"~150 MB","small":"~490 MB",
                   "medium":"~1.5 GB","large-v2":"~2.9 GB","large-v3":"~2.9 GB"}
WHISPER_LANGS = [
    ("auto","🌐 Auto-detect"),("en","🇬🇧 English"),("pl","🇵🇱 Polish"),
    ("ja","🇯🇵 Japanese"),("zh","🇨🇳 Chinese"),("ko","🇰🇷 Korean"),
    ("de","🇩🇪 German"),("fr","🇫🇷 French"),("es","🇪🇸 Spanish"),
    ("ru","🇷🇺 Russian"),("ar","🇸🇦 Arabic"),("pt","🇧🇷 Portuguese"),
    ("it","🇮🇹 Italian"),("tr","🇹🇷 Turkish"),("nl","🇳🇱 Dutch"),
    ("uk","🇺🇦 Ukrainian"),("sv","🇸🇪 Swedish"),("fi","🇫🇮 Finnish"),
]

TARGET_SR_OPTIONS = [
    (None,  "Keep original sample rate"),
    (8000,  "8 kHz"),
    (16000, "16 kHz"),
    (22000, "22 kHz"),
    (24000, "24 kHz"),
    (32000, "32 kHz"),
    (44100, "44.1 kHz"),
    (48000, "48 kHz"),
]

STATUS_WAITING = "⬜"
STATUS_RUNNING = "🔄"
STATUS_DONE    = "✅"
STATUS_ERROR   = "❌"
COL_STATUS   = 0
COL_FRAGMENT = 1
COL_SPEAKER  = 2
COL_TIMING   = 3


def _fmt(s: float) -> str:
    s = int(s)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"

def _fmt_ms(ms: int) -> str:
    ms  = int(ms)
    m   = ms // 60000
    s   = (ms % 60000) // 1000
    rem = ms % 1000
    return f"{m}:{s:02d}.{rem:03d}"

def _open_file(path: str) -> None:
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        logger.error(f"Cannot open file: {e}")


def _check_ffmpeg() -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def _get_wav_duration(path: str) -> Optional[float]:
    try:
        info = torchaudio.info(path)
        return info.num_frames / info.sample_rate
    except Exception:
        try:
            data, sr = sf.read(path)
            return len(data) / sr
        except Exception:
            return None


def _detect_srt_language(texts: List[str]) -> str:
    sample = " ".join(texts[:40])[:3000]
    try:
        detector = LanguageDetectorBuilder.from_all_languages().build()
        lang = detector.detect_language_of(sample)
        if lang is None:
            return "en"
        LINGUA_TO_NUM2WORDS = {
            "ENGLISH":    "en",
            "POLISH":     "pl",
            "GERMAN":     "de",
            "FRENCH":     "fr",
            "SPANISH":    "es",
            "ITALIAN":    "it",
            "PORTUGUESE": "pt",
            "RUSSIAN":    "ru",
            "DUTCH":      "nl",
            "CZECH":      "cs",
            "HUNGARIAN":  "hu",
            "TURKISH":    "tr",
            "ARABIC":     "ar",
            "JAPANESE":   "ja",
            "KOREAN":     "ko",
            "CHINESE":    "zh",
            "UKRAINIAN":  "uk",
            "SWEDISH":    "sv",
            "FINNISH":    "fi",
            "DANISH":     "da",
            "NORWEGIAN":  "no",
            "ROMANIAN":   "ro",
            "SLOVAK":     "sk",
            "SLOVENIAN":  "sl",
            "LATVIAN":    "lv",
            "LITHUANIAN": "lt",
            "SERBIAN":    "sr",
            "INDONESIAN": "id",
            "VIETNAMESE": "vi",
            "THAI":       "th",
            "HEBREW":     "he",
            "FARSI":      "fa",
            "CATALAN":    "ca",
        }
        lang_name = lang.name.upper()
        return LINGUA_TO_NUM2WORDS.get(lang_name, "en")
    except ImportError:
        logger.warning("lingua not installed — language detection skipped, defaulting to 'en'")
        return "en"
    except Exception as e:
        logger.warning(f"Language detection failed: {e} — defaulting to 'en'")
        return "en"


def _convert_numbers_in_text(text: str, lang: str) -> str:
    def _replace(match):
        raw   = match.group(0)
        clean = raw.replace(" ", "").replace(",", "").replace("\u00a0", "")
        try:
            if "." in clean:
                return num2words(float(clean), lang=lang)
            else:
                return num2words(int(clean), lang=lang)
        except (ValueError, NotImplementedError, OverflowError):
            try:
                return num2words(int(clean), lang="en")
            except Exception:
                return raw

    return re.sub(r"\b\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b", _replace, text)

def _normalize_text_for_tts(text: str) -> str:
    text = text.replace("\u2026", "...")
    def _fix_token(m: re.Match) -> str:
        token = m.group(0)
        alpha_chars = [c for c in token if c.isalpha()]
        if len(alpha_chars) >= 2 and all(c.isupper() for c in alpha_chars):
            return token.title()
        return token
    return re.sub(r"\S+", _fix_token, text)

def _trim_silence_wav(path: str) -> None:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    abs_audio = np.abs(audio)
    noise_floor = np.percentile(np.sort(abs_audio)[:max(1, len(abs_audio) // 10)], 95)
    threshold = max(noise_floor * 3, 0.005)
    above = np.where(abs_audio > threshold)[0]
    if len(above) == 0:
        return
    pad = int(sr * 0.01)
    start = max(0, above[0] - pad)
    end   = min(len(audio), above[-1] + pad)
    trimmed = audio[start:end]
    if len(trimmed) < sr * 0.1:
        return
    sf.write(path, trimmed, sr, subtype="PCM_16")

def _compute_trim_bounds(audio: np.ndarray, sr: int, aggressiveness: float):
    if aggressiveness <= 0.0 or len(audio) == 0:
        return 0, 0, 0, 0
    abs_audio   = np.abs(audio)
    noise_floor = np.percentile(np.sort(abs_audio)[:max(1, len(abs_audio) // 10)], 95)
    threshold   = max(noise_floor * (1.0 + aggressiveness * 0.5), aggressiveness * 0.001)
    above       = np.where(abs_audio > threshold)[0]
    if len(above) == 0:
        return 0, 0, 0, 0
    pad           = int(sr * 0.005)
    lead_samples  = max(0, above[0] - pad)
    trail_samples = max(0, len(audio) - 1 - above[-1] - pad)
    lead_ms       = int(lead_samples  / max(1, sr) * 1000)
    trail_ms      = int(trail_samples / max(1, sr) * 1000)
    return lead_ms, trail_ms, lead_samples, trail_samples

class BaseWorker(QThread):
    status  = pyqtSignal(str)
    error   = pyqtSignal(str)


class DownloadModelWorker(BaseWorker):
    finished = pyqtSignal()
 
    def __init__(self, backend, hf_token: Optional[str] = None):
        super().__init__()
        self.backend   = backend
        self.hf_token  = hf_token
 
    def run(self):
        try:
            if self.hf_token:
                os.environ["HF_TOKEN"]               = self.hf_token
                os.environ["HUGGING_FACE_HUB_TOKEN"] = self.hf_token
            self.backend.download(
                self.backend.model_dir,
                lambda m, p=0.0: self.status.emit(m),
            )
            self.finished.emit()
        except Exception:
            self.error.emit(traceback.format_exc())

class LoadModelWorker(BaseWorker):
    finished = pyqtSignal()

    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    def run(self):
        try:
            self.backend.load_model(lambda m: self.status.emit(m))
            self.finished.emit()
        except Exception:
            self.error.emit(traceback.format_exc())


class TTSWorker(QThread):
    progress  = pyqtSignal(int, str, bool)
    item_done = pyqtSignal(int, str, bool)
    finished  = pyqtSignal()
 
    def __init__(self, backend, fragments: List[Dict], output_dir: str,
                 reference_audio: Optional[str] = None,
                 reference_text: Optional[str] = None,
                 filename_prefix: str = "fragment",
                 generation_settings: Optional[Dict] = None,
                 normalize_audio: bool = False,
                 speaker_voices: Optional[Dict] = None,
                 reserved_paths: Optional[set] = None):
        super().__init__()
        self.backend             = backend
        self.fragments           = fragments
        self.output_dir          = output_dir
        self.reference_audio     = reference_audio
        self.reference_text      = reference_text
        self.filename_prefix     = filename_prefix
        self.generation_settings = generation_settings or {}
        self.normalize_audio     = normalize_audio
        self.speaker_voices      = speaker_voices or {}
        self.reserved_paths      = set(reserved_paths or set())
        self._cancelled          = False
 
    def request_cancel(self) -> None:
        self._cancelled = True
 
    def run(self) -> None:
        if not self.backend.is_loaded:
            self.progress.emit(-1, "Model not loaded — cannot start synthesis.", True)
            self.finished.emit()
            return

        os.makedirs(self.output_dir, exist_ok=True)

        for fragment in self.fragments:
            if self._cancelled:
                break

            idx    = fragment.get("index", 0)
            raw    = fragment.get("text", "").strip()
            prefix = (fragment.get("prefix") or "").strip()
            suffix = (fragment.get("suffix") or "").strip()
            parts  = [x for x in [prefix, raw, suffix] if x]
            text   = _normalize_text_for_tts(" ".join(parts))

            if not text:
                self.item_done.emit(idx, "", False)
                continue

            if fragment.get('output_path'):
                output_path = fragment['output_path']
                self.reserved_paths.add(output_path)
            else:
                n = idx + 1
                while True:
                    candidate = os.path.join(
                        self.output_dir, f"{self.filename_prefix}_{n:03d}.wav"
                    )
                    if candidate not in self.reserved_paths:
                        break
                    n += 1
                output_path = candidate
                self.reserved_paths.add(output_path)

            self.progress.emit(idx, f"Synthesizing fragment {idx + 1}…", False)

            if self.speaker_voices:
                speaker = fragment.get("speaker") or ""
                sv = self.speaker_voices.get(speaker) if speaker else None
                if sv:
                    ref_audio, ref_text = sv
                else:
                    ref_audio = self.reference_audio
                    ref_text  = self.reference_text
            else:
                ref_audio = self.reference_audio
                ref_text  = self.reference_text

            try:
                audio, sr = self.backend.generate(
                    text=text,
                    reference_audio_path=ref_audio,
                    reference_text=ref_text,
                    progress_cb=lambda m, _i=idx: self.progress.emit(_i, m, False),
                    **self.generation_settings,
                )

                min_samples = max(1, int(sr * 0.05))
                if audio is None or len(audio) < min_samples:
                    raise ValueError(
                        f"Generated audio is too short "
                        f"({0 if audio is None else len(audio)} samples / "
                        f"{0.0 if audio is None else len(audio) / max(1, sr):.3f}s) "
                        f"for fragment {idx + 1}"
                    )

                sf.write(output_path, audio, sr, subtype="PCM_16")

                self.progress.emit(idx, f"✓ Fragment {idx + 1} done", False)
                self.item_done.emit(idx, output_path, False)

            except Exception as e:
                msg = str(e)
                self.progress.emit(idx, f"❌ Fragment {idx + 1} failed: {msg}", True)
                self.item_done.emit(idx, msg, True)

            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if self._cancelled:
                break

        self.finished.emit()
        


class GenerateWorker(BaseWorker):
    finished = pyqtSignal(object, int)

    def __init__(self, backend, text: str, ref_audio: Optional[str],
                 ref_text: Optional[str], settings: Dict):
        super().__init__()
        self.backend   = backend
        self.text      = text
        self.ref_audio = ref_audio
        self.ref_text  = ref_text
        self.settings  = settings

    def run(self):
        try:
            audio, sr = self.backend.generate(
                text=_normalize_text_for_tts(self.text),
                reference_audio_path=self.ref_audio,
                reference_text=self.ref_text,
                progress_cb=lambda m: self.status.emit(m),
                **self.settings,
            )
            self.finished.emit(audio, sr)
        except Exception:
            self.error.emit(traceback.format_exc())


class WhisperDownloadWorker(BaseWorker):
    finished = pyqtSignal()

    def __init__(self, wb, size: str):
        super().__init__()
        self.wb   = wb
        self.size = size

    def run(self):
        try:
            self.wb.download(self.size, lambda m, p: self.status.emit(m))
            self.finished.emit()
        except Exception:
            self.error.emit(traceback.format_exc())

class WhisperBackend:

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"

    def model_path(self, size: str) -> Path:
        return self.model_dir / size

    def is_downloaded(self, size: str) -> bool:
        p = self.model_path(size)
        return p.exists() and any(p.iterdir())

    def download(self, size: str, progress_cb: Optional[Callable] = None) -> None:
        if size not in WHISPER_REPOS:
            raise InferenceError(f"Unknown Whisper size: {size}")
        dest = self.model_path(size)
        dest.mkdir(parents=True, exist_ok=True)
        if progress_cb:
            progress_cb(f"Downloading Whisper {size}…", 0.0)
        snapshot_download(repo_id=WHISPER_REPOS[size], local_dir=str(dest))
        if progress_cb:
            progress_cb(f"✓ Whisper {size} downloaded!", 1.0)

    def transcribe(
        self,
        audio_path: str,
        size: str = "large-v3",
        language: Optional[str] = None,
        progress_cb: Optional[Callable] = None,
    ) -> str:
        if not self.is_downloaded(size):
            raise InferenceError(
                f"Whisper model '{size}' is not downloaded.\n"
                "Click 'Download Whisper' first.")

        script = r"""
import sys, gc, json
import numpy as np
audio_path, model_path, device, lang_arg = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
import soundfile as sf
audio, sr = sf.read(audio_path, dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1)
if sr != 16000:
    try:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    except ImportError:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
sys.stderr.write(f"[whisper-proc] audio shape={audio.shape} sr=16000\n"); sys.stderr.flush()
from faster_whisper import WhisperModel
sys.stderr.write("[whisper-proc] loading model...\n"); sys.stderr.flush()
compute = "float16" if device == "cuda" else "int8"
model = WhisperModel(model_path, device=device, compute_type=compute)
sys.stderr.write("[whisper-proc] transcribing...\n"); sys.stderr.flush()
lang = None if lang_arg == "None" else lang_arg
segments, _ = model.transcribe(audio, language=lang, beam_size=5, vad_filter=False)
text = " ".join(s.text.strip() for s in list(segments)).strip()
del model; del audio; gc.collect()
sys.stderr.write("[whisper-proc] done\n"); sys.stderr.flush()
print(json.dumps({"text": text}))
"""
        fd, script_path = tempfile.mkstemp(suffix=".py", prefix="whisper_worker_")
        try:
            os.write(fd, script.encode("utf-8"))
            os.close(fd)
            lang_str = language if (language and language != "auto") else "None"
            cmd = [sys.executable, script_path,
                   audio_path, str(self.model_path(size)), self.device, lang_str]
            if progress_cb:
                progress_cb("Whisper working… (may take up to a minute)")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.stderr:
                for line in result.stderr.strip().splitlines():
                    logger.info(line)
            if result.returncode != 0:
                last = "\n".join(result.stderr.strip().splitlines()[-20:]) or "(no logs)"
                raise InferenceError(
                    f"Whisper subprocess error (code {result.returncode}).\n\n{last}")
            if not result.stdout.strip():
                raise InferenceError("Whisper subprocess returned no output.")
            text = json.loads(result.stdout.strip()).get("text", "")
        except subprocess.TimeoutExpired:
            raise InferenceError("Transcription timed out (>10 min).")
        finally:
            try:
                os.unlink(script_path)
            except Exception:
                pass
        if progress_cb:
            progress_cb("✓ Transcription complete")
        return text

class TranscribeWorker(BaseWorker):
    finished = pyqtSignal(str)

    def __init__(self, wb, path: str, size: str, lang: str):
        super().__init__()
        self.wb   = wb
        self.path = path
        self.size = size
        self.lang = lang

    def run(self):
        try:
            text = self.wb.transcribe(
                self.path, self.size, self.lang,
                lambda m: self.status.emit(m),
            )
            self.finished.emit(text)
        except Exception:
            self.error.emit(traceback.format_exc())

class AudioPreprocessor:

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process(
        self,
        input_path: str,
        target_sr: int = 44100,
        to_mono: bool = True,
        isolate_vocals: bool = False,
        normalize: bool = True,
        device: str = "cpu",
        progress_cb: Optional[Callable] = None,
        output_name: Optional[str] = None,
    ) -> str:
        def status(msg):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        status("Loading audio file…")
        audio, sr = sf.read(input_path, dtype="float32", always_2d=True)
        status(f"  Loaded: {sr}Hz, {audio.shape[1]}ch, {len(audio)/sr:.1f}s")

        if to_mono and audio.shape[1] > 1:
            status("Converting stereo → mono…")
            audio = audio.mean(axis=1, keepdims=True)

        if sr != target_sr:
            status(f"Resampling {sr}Hz → {target_sr}Hz…")
            audio, sr = self._resample(audio[:, 0], sr, target_sr)
            audio = audio[:, np.newaxis]

        if isolate_vocals:
            status("Isolating vocals (Demucs htdemucs)…")
            status("  Loading Demucs model — this may take 1–3 minutes…")
            audio_1d, sr = self._isolate_vocals(audio[:, 0], sr, device, status)
            audio = audio_1d[:, np.newaxis]

        if normalize:
            max_val = float(np.abs(audio).max())
            if max_val > 0:
                audio = audio * (0.92 / max_val)

        if output_name:
            out_path = str(self.output_dir / output_name)
        else:
            stem = Path(input_path).stem
            h = hashlib.md5(str(input_path).encode()).hexdigest()[:8]
            out_path = str(self.output_dir / f"reference_processed_{stem}_{h}.wav")

        sf.write(out_path, audio, sr, subtype="PCM_16")

        dur = len(audio) / sr
        status(f"✓ Done: WAV PCM_16 | {sr}Hz | mono | {dur:.1f}s → {out_path}")
        return out_path

    @staticmethod
    def _resample(audio_1d: np.ndarray, orig_sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
        try:
            import librosa
            resampled = librosa.resample(audio_1d, orig_sr=orig_sr, target_sr=target_sr)
            return resampled, target_sr
        except ImportError:
            pass
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(orig_sr, target_sr)
            resampled = resample_poly(audio_1d, target_sr // g, orig_sr // g)
            return resampled.astype(np.float32), target_sr
        except Exception as e:
            raise InferenceError(f"Resampling failed:\n{e}\nInstall librosa: pip install librosa")

    @staticmethod
    def _isolate_vocals(
        audio_1d: np.ndarray, sr: int, device: str,
        status_cb: Callable,
    ) -> Tuple[np.ndarray, int]:
        try:
            import torchaudio
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
        except ImportError as e:
            raise InferenceError(
                f"Demucs is not installed:\n{e}\n"
                "Run: pip install demucs")
        status_cb("  Loading htdemucs…")
        model = get_model("htdemucs")
        model.eval()
        use_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        model = model.to(use_device)
        demucs_sr = model.samplerate
        if sr != demucs_sr:
            status_cb(f"  Resampling for Demucs: {sr}Hz → {demucs_sr}Hz…")
            audio_1d, _ = AudioPreprocessor._resample(audio_1d, sr, demucs_sr)
        stereo = torch.from_numpy(
            np.stack([audio_1d, audio_1d], axis=0)
        ).unsqueeze(0).float().to(use_device)
        status_cb("  Separating vocals…")
        with torch.no_grad():
            sources = apply_model(model, stereo, device=use_device, progress=False)
        vocals_idx = model.sources.index("vocals")
        vocals = sources[0, vocals_idx].mean(0).cpu().numpy().astype(np.float32)
        if sr != demucs_sr:
            vocals, _ = AudioPreprocessor._resample(vocals, demucs_sr, sr)
        del model, sources, stereo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        status_cb("  ✓ Demucs released from memory")
        return vocals, sr

class AudioProcessWorker(BaseWorker):
    finished = pyqtSignal(str)

    def __init__(self, preprocessor, input_path: str, settings: Dict):
        super().__init__()
        self.preprocessor = preprocessor
        self.input_path   = input_path
        self.settings     = settings

    def run(self):
        try:
            settings = dict(self.settings)
            if "target_sr" in settings and settings["target_sr"] is None:
                try:
                    info = torchaudio.info(self.input_path)
                    settings["target_sr"] = info.sample_rate
                except Exception:
                    try:
                        _, sr = sf.read(self.input_path, dtype="float32", frames=1)
                        settings["target_sr"] = int(sr)
                    except Exception:
                        settings["target_sr"] = 44100
            output_subtype = settings.pop("output_subtype", "PCM_16")
            out = self.preprocessor.process(
                self.input_path,
                progress_cb=lambda m: self.status.emit(m),
                **settings,
            )
            if output_subtype and out and out.lower().endswith(".wav"):
                try:
                    audio, sr = sf.read(out, dtype="float32")
                    sf.write(out, audio, sr, subtype=output_subtype)
                except Exception:
                    pass
            self.finished.emit(out)
        except Exception:
            self.error.emit(traceback.format_exc())


class LektorExportThread(QThread):
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(str)

    def __init__(self, cmd: list, lektor_wav_path: str,
                 extra_tmp_paths: Optional[List[str]] = None):
        super().__init__()
        self.cmd              = cmd
        self.lektor_wav_path  = lektor_wav_path
        self.extra_tmp_paths  = extra_tmp_paths or []

    def run(self):
        try:
            result = subprocess.run(
                self.cmd, capture_output=True, timeout=3600,
            )
            stderr_text = result.stderr.decode(errors="replace")
            if stderr_text.strip():
                for line in stderr_text.strip().splitlines()[-20:]:
                    logger.info(f"[ffmpeg] {line}")

            try:
                if os.path.exists(self.lektor_wav_path):
                    os.remove(self.lektor_wav_path)
            except Exception:
                pass

            for p in self.extra_tmp_paths:
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

            if result.returncode == 0:
                self.finished.emit(True, "")
            else:
                self.finished.emit(False, stderr_text[-800:])
        except subprocess.TimeoutExpired:
            self.finished.emit(False, "Timeout — operation took too long.")
        except Exception as e:
            self.finished.emit(False, str(e))

class DubbingVocalExtractWorker(BaseWorker):
    finished = pyqtSignal(str)

    def __init__(self, video_path: str):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            tmp_dir    = Path(tempfile.mkdtemp(prefix="dubbing_"))
            audio_path = str(tmp_dir / "extracted_audio.wav")

            self.status.emit("Extracting audio from video (ffmpeg)…")
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", self.video_path,
                 "-ac", "1", "-ar", "16000", "-vn", audio_path],
                capture_output=True, timeout=600,
            )
            if r.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg audio extraction failed:\n{r.stderr.decode(errors='replace')[-2000:]}"
                )

            self.status.emit(
                "Isolating vocals — this may take several minutes…"
            )
            demucs_out_dir = str(tmp_dir / "demucs_output")

            demucs_cache_dir = ROOT_DIR / "models" / "demucs"
            demucs_cache_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["TORCH_HOME"] = str(demucs_cache_dir)

            demucs_wrapper = (
                "import sys; "
                "sys.modules.setdefault('torchcodec', type(sys)('torchcodec')); "
                "from demucs.__main__ import main; "
                "sys.exit(main() or 0)"
            )
            r = subprocess.run(
                [sys.executable, "-c", demucs_wrapper,
                 "--two-stems=vocals", "-n", "htdemucs_ft",
                 "--out", demucs_out_dir, audio_path],
                capture_output=True, timeout=3600,
                env=env,
            )
            if r.returncode != 0:
                raise RuntimeError(
                    f"Demucs failed:\n{r.stderr.decode(errors='replace')[-2000:]}"
                )

            stem        = Path(audio_path).stem
            vocals_path = Path(demucs_out_dir) / "htdemucs_ft" / stem / "vocals.wav"
            if not vocals_path.exists():
                found = list(Path(demucs_out_dir).rglob("vocals.wav"))
                if not found:
                    raise RuntimeError(
                        f"vocals.wav not found after Demucs in: {demucs_out_dir}"
                    )
                vocals_path = found[0]

            self.status.emit("Converting vocals to 16kHz mono PCM_16…")
            audio, sr = sf.read(str(vocals_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            if sr != 16000:
                try:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                except ImportError:
                    from scipy.signal import resample_poly
                    from math import gcd
                    g = gcd(sr, 16000)
                    audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)

            max_val = float(np.abs(audio).max())
            if max_val > 0:
                audio = audio / max_val * 0.92

            output_dir   = OUTPUTS_DIR / "dubbing"
            output_dir.mkdir(parents=True, exist_ok=True)
            final_vocals = output_dir / f"{Path(self.video_path).stem}_vocals.wav"
            sf.write(str(final_vocals), audio, 16000, subtype="PCM_16")

            try:
                shutil.rmtree(str(tmp_dir))
            except Exception:
                pass

            self.finished.emit(str(final_vocals))
        except Exception:
            self.error.emit(traceback.format_exc())
            
class VocalSuppressWorker(BaseWorker):
    finished = pyqtSignal(str, str)

    def __init__(self, video_path: str, output_dir: str):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir

    def run(self):
        try:
            tmp_dir    = Path(tempfile.mkdtemp(prefix="vocal_suppress_"))
            audio_path = str(tmp_dir / "extracted_audio.wav")

            self.status.emit("Vocal suppression: extracting audio from video (ffmpeg)…")
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", self.video_path,
                 "-ac", "2", "-ar", "44100", "-vn", audio_path],
                capture_output=True, timeout=600,
            )
            if r.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg audio extraction failed:\n{r.stderr.decode(errors='replace')[-2000:]}"
                )

            self.status.emit(
                "Vocal suppression: isolating vocals — this may take several minutes…"
            )
            demucs_out_dir = str(tmp_dir / "demucs_output")

            demucs_cache_dir = ROOT_DIR / "models" / "demucs"
            demucs_cache_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["TORCH_HOME"] = str(demucs_cache_dir)

            demucs_wrapper = (
                "import sys; "
                "sys.modules.setdefault('torchcodec', type(sys)('torchcodec')); "
                "from demucs.__main__ import main; "
                "sys.exit(main() or 0)"
            )
            r = subprocess.run(
                [sys.executable, "-c", demucs_wrapper,
                 "--two-stems=vocals", "-n", "htdemucs_ft",
                 "--out", demucs_out_dir, audio_path],
                capture_output=True, timeout=3600,
                env=env,
            )
            if r.returncode != 0:
                raise RuntimeError(
                    f"Demucs failed:\n{r.stderr.decode(errors='replace')[-2000:]}"
                )

            stem          = Path(audio_path).stem
            vocals_src    = Path(demucs_out_dir) / "htdemucs_ft" / stem / "vocals.wav"
            no_vocals_src = Path(demucs_out_dir) / "htdemucs_ft" / stem / "no_vocals.wav"

            if not vocals_src.exists():
                found = list(Path(demucs_out_dir).rglob("vocals.wav"))
                if not found:
                    raise RuntimeError(
                        f"vocals.wav not found after Demucs in: {demucs_out_dir}"
                    )
                vocals_src = found[0]

            if not no_vocals_src.exists():
                found = list(Path(demucs_out_dir).rglob("no_vocals.wav"))
                if not found:
                    raise RuntimeError(
                        f"no_vocals.wav not found after Demucs in: {demucs_out_dir}"
                    )
                no_vocals_src = found[0]

            out_dir       = Path(self.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            video_stem    = Path(self.video_path).stem
            vocals_dst    = str(out_dir / f"_vsup_vocals_{video_stem}.wav")
            no_vocals_dst = str(out_dir / f"_vsup_no_vocals_{video_stem}.wav")

            shutil.copy2(str(vocals_src), vocals_dst)
            shutil.copy2(str(no_vocals_src), no_vocals_dst)

            try:
                shutil.rmtree(str(tmp_dir))
            except Exception:
                pass

            self.finished.emit(vocals_dst, no_vocals_dst)
        except Exception:
            self.error.emit(traceback.format_exc())
            
class VideoAudioExtractWorker(BaseWorker):
    finished = pyqtSignal(str)

    def __init__(self, video_path: str):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            tmp = tempfile.mktemp(suffix=".wav", prefix="vidwave_")
            self.status.emit("Extracting audio from video…")
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", self.video_path,
                 "-ac", "1", "-ar", "22050", "-vn", tmp],
                capture_output=True, timeout=600,
            )
            if r.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed:\n{r.stderr.decode(errors='replace')[-400:]}"
                )
            self.finished.emit(tmp)
        except Exception:
            self.error.emit(traceback.format_exc())
            
class DiarizationWorker(BaseWorker):
    finished = pyqtSignal(dict)

    def __init__(self, audio_path: str, hf_token: str):
        super().__init__()
        self.audio_path = audio_path
        self.hf_token   = hf_token

    def _extract_annotation(self, raw_result):
        if hasattr(raw_result, "itertracks"):
            return raw_result
        for attr in ("diarization", "annotation", "output", "result"):
            candidate = getattr(raw_result, attr, None)
            if candidate is not None and hasattr(candidate, "itertracks"):
                return candidate
        for attr in vars(raw_result):
            candidate = getattr(raw_result, attr, None)
            if candidate is not None and hasattr(candidate, "itertracks"):
                return candidate
        raise RuntimeError(
            f"Cannot find Annotation in DiarizeOutput.\n"
            f"Type: {type(raw_result)}\n"
            f"Attributes: {list(vars(raw_result).keys())}"
        )

    def run(self):
        try:
            import warnings as _warnings
            _warnings.filterwarnings("ignore", message="In 2.9, this function")
            _warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
            from pyannote.audio import Pipeline

            self.status.emit(
                "Loading speaker diarization model (pyannote/speaker-diarization-3.1)…"
            )
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self.hf_token,
            )
            if torch.cuda.is_available():
                pipeline = pipeline.to(torch.device("cuda"))

            self.status.emit("Loading audio for diarization…")
            waveform, sample_rate = torchaudio.load(self.audio_path)
            audio_input = {"waveform": waveform, "sample_rate": sample_rate}

            self.status.emit("Running speaker diarization — please wait…")
            raw_result = pipeline(audio_input)

            self.status.emit(
                f"Processing diarization output "
                f"(type: {type(raw_result).__name__}, "
                f"attrs: {list(vars(raw_result).keys()) if hasattr(raw_result, '__dict__') else 'n/a'})…"
            )

            diarization = self._extract_annotation(raw_result)

            segments: List[Dict]                = []
            speaker_durations: Dict[str, float] = {}

            for segment, _, label in diarization.itertracks(yield_label=True):
                dur = segment.end - segment.start
                segments.append({
                    "start":   segment.start,
                    "end":     segment.end,
                    "speaker": label,
                })
                speaker_durations[label] = speaker_durations.get(label, 0.0) + dur

            sorted_spk = sorted(
                speaker_durations.items(), key=lambda x: x[1], reverse=True
            )
            speaker_map = {
                orig: f"Person {i + 1}"
                for i, (orig, _) in enumerate(sorted_spk)
            }

            self.finished.emit({"segments": segments, "speaker_map": speaker_map})
        except Exception:
            self.error.emit(traceback.format_exc())

class WaveformWidget(QWidget):
    seeked = pyqtSignal(float)

    def __init__(self, h: int = 64, parent=None):
        super().__init__(parent)
        self.setFixedHeight(h)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._peaks: list  = []
        self._pos:   float = 0.0
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_audio(self, audio: np.ndarray):
        n     = max(1, self.width() // 3)
        chunk = max(1, len(audio) // n)
        peaks = []
        for i in range(n):
            seg = audio[i * chunk:(i + 1) * chunk]
            if len(seg):
                peaks.append(float(np.max(np.abs(seg))))
        mx = max(peaks) if peaks and max(peaks) > 0 else 1.0
        self._peaks = [p / mx for p in peaks]
        self.update()

    def set_position(self, p: float):
        self._pos = p
        self.update()

    def clear(self):
        self._peaks = []
        self._pos   = 0.0
        self.update()

    def mousePressEvent(self, e):
        if self._peaks and e.button() == Qt.MouseButton.LeftButton:
            self.seeked.emit(max(0., min(1., e.position().x() / max(1, self.width()))))

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(C["player"]))

        f = p.font()
        if f.pointSize() <= 0:
            f.setPointSize(10)
        p.setFont(f)

        if not self._peaks:
            p.setPen(QColor(C["border"]))
            p.drawLine(0, h // 2, w, h // 2)
            p.setPen(QColor(C["text3"]))
            f2 = QFont(f)
            f2.setPointSize(9)
            p.setFont(f2)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No audio")
            return

        n   = len(self._peaks)
        bw  = w / n
        mid = h / 2
        ma  = mid - 3

        for i, pk in enumerate(self._peaks):
            x   = int(i * bw)
            bwi = max(1, int(bw) - 1)
            bh  = max(2, int(pk * ma))
            col = QColor(C["accent"] if (i / n) < self._pos else C["accent2"])
            col.setAlpha(220 if (i / n) < self._pos else 100)
            p.fillRect(x, int(mid - bh), bwi, int(bh * 2), col)

        if 0 < self._pos < 1:
            p.setPen(QPen(QColor(C["accent"]), 2))
            cx = int(w * self._pos)
            p.drawLine(cx, 0, cx, h)

class SelectionWaveformWidget(QWidget):
    seeked = pyqtSignal(float)
    delete_requested = pyqtSignal(float, float)
    mute_requested = pyqtSignal(float, float)

    def __init__(self, h: int = 64, readonly: bool = False, parent=None):
        super().__init__(parent)
        self.setFixedHeight(h)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._peaks:           list            = []
        self._pos:             float           = 0.0
        self._readonly:        bool            = readonly
        self._sel_start:       Optional[float] = None
        self._sel_end:         Optional[float] = None
        self._is_selecting:    bool            = False
        self._zoom:            float           = 1.0
        self._view_start:      float           = 0.0
        self._pan_last_x:      Optional[float] = None
        self._pan_started:     bool            = False
        self._duration:        float           = 0.0
        self._trim_lead_frac:  float           = 0.0
        self._trim_trail_frac: float           = 0.0
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if readonly:
            self.setMouseTracking(True)

    def set_audio(self, audio: np.ndarray):
        if len(audio) == 0:
            self._peaks = []
            self.update()
            return
        n     = max(1, self.width() // 3)
        chunk = max(1, len(audio) // n)
        peaks = []
        for i in range(n):
            seg = audio[i * chunk:(i + 1) * chunk]
            if len(seg):
                peaks.append(float(np.max(np.abs(seg))))
            else:
                peaks.append(0.0)
        mx = max(peaks) if peaks and max(peaks) > 0 else 1.0
        self._peaks = [p / mx for p in peaks]
        self.update()

    def set_duration(self, dur: float):
        self._duration = max(0.0, dur)
        self.update()

    def set_position(self, p: float):
        self._pos = p
        if self._readonly and self._zoom > 1.0:
            view_w = 1.0 / self._zoom
            margin = view_w * 0.15
            if p < self._view_start + margin:
                self._view_start = max(0.0, p - margin)
            elif p > self._view_start + view_w - margin:
                self._view_start = min(1.0 - view_w, p - view_w + margin)
        self.update()

    def set_trim_preview(self, lead_frac: float, trail_frac: float):
        self._trim_lead_frac  = max(0.0, min(1.0, lead_frac))
        self._trim_trail_frac = max(0.0, min(1.0, trail_frac))
        self.update()

    def clear_trim_preview(self):
        self._trim_lead_frac  = 0.0
        self._trim_trail_frac = 0.0
        self.update()

    def clear(self):
        self._peaks           = []
        self._pos             = 0.0
        self._sel_start       = None
        self._sel_end         = None
        self._zoom            = 1.0
        self._view_start      = 0.0
        self._duration        = 0.0
        self._trim_lead_frac  = 0.0
        self._trim_trail_frac = 0.0
        self.update()

    def get_selection(self) -> Optional[Tuple[float, float]]:
        if self._sel_start is None or self._sel_end is None:
            return None
        s = min(self._sel_start, self._sel_end)
        e = max(self._sel_start, self._sel_end)
        return (s, e) if e - s >= 0.002 else None

    def set_selection_fracs(self, start_frac: float, end_frac: float):
        self._sel_start = max(0.0, min(1.0, start_frac))
        self._sel_end   = max(0.0, min(1.0, end_frac))
        if self._readonly and self._zoom > 1.0:
            mid    = (self._sel_start + self._sel_end) / 2.0
            view_w = 1.0 / self._zoom
            self._view_start = max(0.0, min(1.0 - view_w, mid - view_w / 2.0))
        self.update()

    def clear_selection(self):
        self._sel_start = None
        self._sel_end   = None
        self.update()

    def _screen_to_frac(self, x: float) -> float:
        view_w = 1.0 / self._zoom
        return max(0.0, min(1.0, self._view_start + (x / max(1, self.width())) * view_w))

    def _frac_to_screen(self, frac: float) -> float:
        view_w = 1.0 / self._zoom
        return (frac - self._view_start) / view_w * self.width()

    def _clamp_view(self):
        view_w = 1.0 / self._zoom
        self._view_start = max(0.0, min(1.0 - view_w, self._view_start))

    def mousePressEvent(self, e):
        if not self._peaks:
            return
        if e.button() == Qt.MouseButton.LeftButton:
            self._pan_last_x  = e.position().x()
            self._pan_started = False
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif e.button() == Qt.MouseButton.RightButton:
            frac = self._screen_to_frac(e.position().x())
            self._is_selecting = True
            self._sel_start    = frac
            self._sel_end      = frac
            self.setFocus()
            self.setCursor(Qt.CursorShape.IBeamCursor)
            self.update()

    def mouseMoveEvent(self, e):
        if self._is_selecting:
            self._sel_end = self._screen_to_frac(e.position().x())
            self.update()
        elif (e.buttons() & Qt.MouseButton.LeftButton) and self._pan_last_x is not None:
            dx = e.position().x() - self._pan_last_x
            if abs(dx) > 3:
                self._pan_started = True
            if dx != 0:
                view_w = 1.0 / self._zoom
                delta  = -(dx / max(1, self.width())) * view_w
                self._view_start += delta
                self._clamp_view()
                self._pan_last_x = e.position().x()
                self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            if self._is_selecting:
                self._is_selecting = False
                frac = self._screen_to_frac(e.position().x())
                self._sel_end = frac
                if abs((self._sel_end or 0.0) - (self._sel_start or 0.0)) < 0.002:
                    self._sel_start = None
                    self._sel_end   = None
                self.setCursor(Qt.CursorShape.PointingHandCursor)
                self.update()
            elif self._pan_last_x is not None:
                self._pan_last_x = None
                self.setCursor(Qt.CursorShape.PointingHandCursor)
                if not self._pan_started:
                    frac = self._screen_to_frac(e.position().x())
                    self.seeked.emit(max(0., min(1., frac)))
                self._pan_started = False
        elif e.button() == Qt.MouseButton.RightButton:
            if self._is_selecting:
                self._is_selecting = False
                frac = self._screen_to_frac(e.position().x())
                self._sel_end = frac
                if abs((self._sel_end or 0.0) - (self._sel_start or 0.0)) < 0.002:
                    self._sel_start = None
                    self._sel_end   = None
                self.setCursor(Qt.CursorShape.PointingHandCursor)
                self.update()
            else:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocus()

    def wheelEvent(self, e):
        if not self._peaks or not self._readonly:
            super().wheelEvent(e)
            return
        delta      = e.angleDelta().y()
        factor     = 1.25 if delta > 0 else (1.0 / 1.25)
        new_zoom   = max(1.0, min(64.0, self._zoom * factor))
        mouse_frac = self._screen_to_frac(e.position().x())
        view_w_new = 1.0 / new_zoom
        self._view_start = mouse_frac - (e.position().x() / max(1, self.width())) * view_w_new
        self._zoom       = new_zoom
        self._clamp_view()
        if self._zoom == 1.0:
            self._view_start = 0.0
        self.update()
        e.accept()

    def keyPressEvent(self, e):
        if not self._readonly:
            if e.key() == Qt.Key.Key_Delete:
                sel = self.get_selection()
                if sel:
                    self.delete_requested.emit(sel[0], sel[1])
                    self._sel_start = None
                    self._sel_end   = None
                    self.update()
                    e.accept()
                    return
            elif e.modifiers() & Qt.KeyboardModifier.ControlModifier and e.key() == Qt.Key.Key_M:
                sel = self.get_selection()
                if sel:
                    self.mute_requested.emit(sel[0], sel[1])
                    self._sel_start = None
                    self._sel_end   = None
                    self.update()
                    e.accept()
                    return
        super().keyPressEvent(e)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(C["player"]))

        f = p.font()
        if f.pointSize() <= 0:
            f.setPointSize(10)
        p.setFont(f)

        if not self._peaks:
            p.setPen(QColor(C["border"]))
            p.drawLine(0, h // 2, w, h // 2)
            p.setPen(QColor(C["text3"]))
            f2 = QFont(f)
            f2.setPointSize(9)
            p.setFont(f2)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No audio")
            return

        ruler_h    = 14 if self._duration > 0.0 else 0
        wave_h     = h - ruler_h
        view_w     = 1.0 / self._zoom
        n          = len(self._peaks)
        mid        = wave_h / 2.0
        ma         = mid - 3

        live_sel = self.get_selection()
        if not live_sel and self._is_selecting and self._sel_start is not None and self._sel_end is not None:
            s  = min(self._sel_start, self._sel_end)
            ev = max(self._sel_start, self._sel_end)
            if ev - s >= 0.002:
                live_sel = (s, ev)

        if live_sel:
            sx = int(self._frac_to_screen(live_sel[0]))
            ex = int(self._frac_to_screen(live_sel[1]))
            p.fillRect(sx, 0, ex - sx, wave_h, QColor(0, 120, 215, 70))

        i_start = max(0, int(self._view_start * n) - 1)
        i_end   = min(n, int((self._view_start + view_w) * n) + 2)

        for i in range(i_start, i_end):
            pk     = self._peaks[i]
            i_frac = i / n
            x      = self._frac_to_screen(i_frac)
            bw_px  = max(1.0, (1.0 / n) / view_w * w)
            bwi    = max(1, int(bw_px) - 1)
            bh     = max(2, int(pk * ma))
            col    = QColor(C["accent"] if i_frac < self._pos else C["accent2"])
            col.setAlpha(220 if i_frac < self._pos else 100)
            p.fillRect(int(x), int(mid - bh), bwi, int(bh * 2), col)

        if self._trim_lead_frac > 0.0:
            ex_trim = max(0, min(w, int(self._frac_to_screen(self._trim_lead_frac))))
            if ex_trim > 0:
                p.fillRect(0, 0, ex_trim, wave_h, QColor(220, 50, 50, 100))
        if self._trim_trail_frac > 0.0:
            sx_trim = max(0, min(w, int(self._frac_to_screen(1.0 - self._trim_trail_frac))))
            if sx_trim < w:
                p.fillRect(sx_trim, 0, w - sx_trim, wave_h, QColor(220, 50, 50, 100))

        if live_sel:
            p.setPen(QPen(QColor(0, 160, 255, 220), 2))
            p.drawLine(int(self._frac_to_screen(live_sel[0])), 0,
                       int(self._frac_to_screen(live_sel[0])), wave_h)
            p.drawLine(int(self._frac_to_screen(live_sel[1])), 0,
                       int(self._frac_to_screen(live_sel[1])), wave_h)

        if 0 < self._pos < 1:
            p.setPen(QPen(QColor(C["accent"]), 2))
            cx = int(self._frac_to_screen(self._pos))
            if 0 <= cx <= w:
                p.drawLine(cx, 0, cx, wave_h)

        if self._zoom > 1.0:
            p.setPen(QColor(C["text3"]))
            f3 = QFont(f)
            f3.setPointSize(8)
            p.setFont(f3)
            p.drawText(4, wave_h - 3, f"{self._zoom:.1f}x")

        if self._duration > 0.0:
            ruler_y = wave_h
            p.fillRect(0, ruler_y, w, ruler_h, QColor(14, 14, 14, 220))
            p.setPen(QColor(50, 50, 50))
            p.drawLine(0, ruler_y, w, ruler_y)

            t_start      = self._view_start * self._duration
            t_end        = (self._view_start + view_w) * self._duration
            visible_span = t_end - t_start

            if visible_span <= 15:
                interval = 1
            elif visible_span <= 40:
                interval = 2
            elif visible_span <= 90:
                interval = 5
            elif visible_span <= 240:
                interval = 10
            elif visible_span <= 600:
                interval = 30
            elif visible_span <= 1800:
                interval = 60
            elif visible_span <= 5400:
                interval = 300
            else:
                interval = 600

            f_ruler = QFont("Consolas", 7)
            p.setFont(f_ruler)
            fm      = p.fontMetrics()
            last_lx = -999

            first_tick = int(t_start / interval) * interval
            t = first_tick
            while t <= t_end + interval:
                if t < 0:
                    t += interval
                    continue
                frac = t / self._duration
                x    = int(self._frac_to_screen(frac))
                if 0 <= x <= w:
                    p.setPen(QColor(80, 80, 80))
                    p.drawLine(x, ruler_y, x, ruler_y + 4)
                    mins  = int(t) // 60
                    secs  = int(t) % 60
                    label = f"{mins}:{secs:02d}"
                    lw    = fm.horizontalAdvance(label)
                    lx    = max(1, min(w - lw - 1, x - lw // 2))
                    if lx > last_lx + lw + 4:
                        p.setPen(QColor(C["text2"]))
                        p.drawText(lx, h - 2, label)
                        last_lx = lx
                t += interval

class TimingIssuesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_win = parent
        self.setWindowTitle("⏳ Select overlong fragments")
        self.resize(360, 230)
        self.setStyleSheet(parent.styleSheet() if parent else "")
        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        lay.setContentsMargins(14, 14, 14, 10)
        title = QLabel("Timing overflow levels")
        title.setStyleSheet(f"color:{C['text']};font-size:13px;font-weight:600;")
        lay.addWidget(title)
        desc = QLabel("Select which fragments should be marked based on how much audio exceeds its time slot.")
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color:{C['text3']};font-size:11px;")
        lay.addWidget(desc)
        self.yellow_check = QCheckBox("🟡 Yellow (1.0× – 2.0×)")
        self.orange_check = QCheckBox("🟠 Orange (2.0× – 4.0×)")
        self.red_check = QCheckBox("🔴 Red (4.0×+)")
        self.yellow_check.setChecked(True)
        self.orange_check.setChecked(True)
        self.red_check.setChecked(True)
        for cb in [self.yellow_check, self.orange_check, self.red_check]:
            cb.setStyleSheet(f"""
                QCheckBox {{
                    color: {C['text']};
                    font-size: 12px;
                    padding: 2px 0;
                }}
            """)
            cb.stateChanged.connect(self._update_preview_count)
            lay.addWidget(cb)
        self._count_label = QLabel("Matching fragments: –")
        self._count_label.setStyleSheet(f"""
            color:{C['accent']};
            font-size:12px;
            font-weight:600;
            padding-top:4px;
        """)
        lay.addWidget(self._count_label)
        hint = QLabel("Based on audio duration vs subtitle slot length")
        hint.setStyleSheet(f"color:{C['text3']};font-size:10px;font-style:italic;")
        lay.addWidget(hint)
        lay.addStretch()
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)
        self._update_preview_count()

    def _update_preview_count(self):
        if not self.parent_win:
            self._count_label.setText("–")
            return
        select_yellow = self.yellow_check.isChecked()
        select_orange = self.orange_check.isChecked()
        select_red = self.red_check.isChecked()
        y_count = 0
        o_count = 0
        r_count = 0
        for frag in getattr(self.parent_win, "_fragments", []):
            if frag.get('status') != 'done':
                continue
            dur = _get_wav_duration(frag.get('output_path', ''))
            if dur is None:
                continue
            slot_s = ((frag.get('end_ms', 0) or 0) - (frag.get('start_ms', 0) or 0)) / 1000.0
            if slot_s <= 0:
                continue
            ratio = dur / max(slot_s, 0.001)
            if 1.0 < ratio < 2.0:
                y_count += 1
            elif 2.0 <= ratio < 4.0:
                o_count += 1
            elif ratio >= 4.0:
                r_count += 1
        total = (
            (y_count if select_yellow else 0) +
            (o_count if select_orange else 0) +
            (r_count if select_red else 0)
        )
        self._count_label.setText(
            f"🟡 {y_count if select_yellow else 0} "
            f"🟠 {o_count if select_orange else 0} "
            f"🔴 {r_count if select_red else 0} "
            f"| Total: {total}"
        )

class RefAudioPlayer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._path:    Optional[str]        = None
        self._data:    Optional[np.ndarray] = None
        self._sr       = 44100
        self._playing  = False
        self._cursor   = 0
        self._stream   = None
        self._timer    = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 0)
        lay.setSpacing(4)

        self._wave = WaveformWidget(h=40)
        self._wave.seeked.connect(self._seek)
        lay.addWidget(self._wave)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)
        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setFixedHeight(26)
        self._play_btn.setEnabled(False)
        self._play_btn.clicked.connect(self._toggle)
        self._lbl = QLabel("—")
        self._lbl.setStyleSheet(f"color:{C['text3']};font-size:10px;font-family:'Consolas',monospace;")

        vol_icon = QLabel("🔊")
        vol_icon.setStyleSheet("font-size:11px;background:transparent;border:none;")
        self._vol = QSlider(Qt.Orientation.Horizontal)
        self._vol.setRange(0, 150)
        self._vol.setValue(100)
        self._vol.setFixedWidth(80)

        ctrl.addWidget(self._play_btn)
        ctrl.addWidget(self._lbl)
        ctrl.addStretch()
        ctrl.addWidget(vol_icon)
        ctrl.addWidget(self._vol)
        lay.addLayout(ctrl)

    def load(self, path: str):
        self._stop_now()
        try:
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            self._data = audio
            self._sr   = sr
            self._path = path
            self._cursor = 0
            self._wave.set_audio(audio)
            self._lbl.setText(f"0:00 / {_fmt(len(audio) / sr)}")
            self._lbl.setStyleSheet(f"color:{C['text2']};font-size:10px;font-family:'Consolas',monospace;")
            self._play_btn.setEnabled(True)
        except Exception as ex:
            self._lbl.setText(f"Error: {ex}")

    def clear(self):
        self._stop_now()
        self._data = None
        self._path = None
        self._wave.clear()
        self._play_btn.setEnabled(False)
        self._lbl.setText("—")
        self._lbl.setStyleSheet(f"color:{C['text3']};font-size:10px;font-family:'Consolas',monospace;")

    def _toggle(self):
        if self._playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        if self._data is None:
            return
        self._playing = True
        self._play_btn.setText("■  Pause")
        audio = self._data.astype(np.float32)
        chunk = audio[self._cursor:] if self._cursor < len(audio) else audio
        if not len(chunk):
            self._cursor = 0
            chunk = audio

        def cb(out, frames, ti, st):
            nonlocal chunk
            if not self._playing:
                raise sd.CallbackStop()
            n = min(frames, len(chunk))
            if n == 0:
                out[:] = 0
                raise sd.CallbackStop()
            vol = self._vol.value() / 100.0
            out[:n, 0] = chunk[:n] * vol
            if frames > n:
                out[n:] = 0
            chunk = chunk[n:]
            self._cursor += n

        try:
            self._stream = sd.OutputStream(
                samplerate=self._sr, channels=1, dtype="float32",
                callback=cb, finished_callback=self._on_end,
            )
            self._stream.start()
            self._timer.start()
        except Exception:
            self._playing = False
            self._play_btn.setText("▶  Play")

    def _pause(self):
        def _safe():
            self._playing = False
            self._play_btn.setText("▶  Play")
            self._timer.stop()
            if self._stream:
                try:
                    self._stream.stop()
                except Exception:
                    pass
        QTimer.singleShot(0, _safe)

    def _stop_now(self):
        def _safe():
            self._playing = False
            self._play_btn.setText("▶  Play")
            self._timer.stop()
            if self._stream:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            self._cursor = 0
            self._wave.set_position(0.0)
        QTimer.singleShot(0, _safe)

    def _on_end(self):
        def _safe():
            self._playing = False
            self._play_btn.setText("▶  Play")
            self._timer.stop()
            self._cursor = 0
            self._wave.set_position(0.0)
            if self._data is not None:
                self._lbl.setText(f"0:00 / {_fmt(len(self._data) / self._sr)}")
        QTimer.singleShot(0, _safe)

    def _tick(self):
        if self._data is None:
            return
        self._wave.set_position(self._cursor / max(1, len(self._data)))
        self._lbl.setText(
            f"{_fmt(self._cursor / max(1, self._sr))} / {_fmt(len(self._data) / max(1, self._sr))}"
        )

    def _seek(self, frac: float):
        if self._data is None:
            return
        was = self._playing
        self._pause()
        self._cursor = int(frac * len(self._data))
        self._wave.set_position(frac)
        if was:
            self._play()

class VideoAudioPlayer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._path:    Optional[str]        = None
        self._data:    Optional[np.ndarray] = None
        self._sr       = 22050
        self._playing  = False
        self._cursor   = 0
        self._play_end = 0
        self._stream   = None
        self._timer    = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._tick)
        self.setVisible(False)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 4)
        lay.setSpacing(3)

        hdr_row = QHBoxLayout()
        lbl = QLabel("VIDEO AUDIO")
        lbl.setStyleSheet(
            f"color:{C['text3']};font-size:9px;letter-spacing:2px;"
            f"background:transparent;border:none;"
        )
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color:{C['border']};")
        hdr_row.addWidget(lbl)
        hdr_row.addWidget(sep, 1)
        lay.addLayout(hdr_row)

        self._wave = SelectionWaveformWidget(h=66, readonly=True)
        self._wave.seeked.connect(self._seek)
        lay.addWidget(self._wave)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)
        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setFixedHeight(26)
        self._play_btn.setEnabled(False)
        self._play_btn.clicked.connect(self._toggle)
        self._lbl = QLabel("—")
        self._lbl.setStyleSheet(
            f"color:{C['text3']};font-size:10px;font-family:'Consolas',monospace;"
        )
        vol_icon = QLabel("🔊")
        vol_icon.setStyleSheet("font-size:11px;background:transparent;border:none;")
        self._vol = QSlider(Qt.Orientation.Horizontal)
        self._vol.setRange(0, 150)
        self._vol.setValue(100)
        self._vol.setFixedWidth(70)
        ctrl.addWidget(self._play_btn)
        ctrl.addWidget(self._lbl)
        ctrl.addStretch()
        ctrl.addWidget(vol_icon)
        ctrl.addWidget(self._vol)
        lay.addLayout(ctrl)

    def load(self, path: str):
        self._stop_now()
        try:
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            self._data   = audio
            self._sr     = sr
            self._path   = path
            self._cursor = 0
            dur = len(audio) / sr
            self._wave.set_audio(audio)
            self._wave.set_duration(dur)
            self._wave.set_position(0.0)
            self._lbl.setText(f"0:00 / {_fmt(dur)}")
            self._lbl.setStyleSheet(
                f"color:{C['text2']};font-size:10px;font-family:'Consolas',monospace;"
            )
            self._play_btn.setEnabled(True)
            self.setVisible(True)
        except Exception as ex:
            self._lbl.setText(f"Error: {ex}")

    def clear(self):
        self._stop_now()
        self._data = None
        self._path = None
        self._wave.clear()
        self._play_btn.setEnabled(False)
        self._lbl.setText("—")
        self.setVisible(False)

    def set_selection_by_time(self, start_s: float, end_s: float):
        if self._data is None:
            return
        dur = len(self._data) / max(1, self._sr)
        if dur <= 0:
            return
        self._wave.set_selection_fracs(start_s / dur, end_s / dur)

    def _toggle(self):
        if self._playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        if self._data is None:
            return
        self._playing = True
        self._play_btn.setText("■  Pause")

        audio = self._data.astype(np.float32)

        sel = self._wave.get_selection()
        if sel:
            s_sample = int(sel[0] * len(audio))
        else:
            s_sample = self._cursor if self._cursor < len(audio) else 0

        e_sample       = len(audio)
        self._cursor   = s_sample
        self._play_end = e_sample
        chunk = audio[s_sample:e_sample].copy()

        if not len(chunk):
            self._playing = False
            self._play_btn.setText("▶  Play")
            return

        def cb(out, frames, ti, st):
            nonlocal chunk
            if not self._playing:
                raise sd.CallbackStop()
            n = min(frames, len(chunk))
            if n == 0:
                out[:] = 0
                raise sd.CallbackStop()
            vol = self._vol.value() / 100.0
            out[:n, 0] = chunk[:n] * vol
            if frames > n:
                out[n:] = 0
            chunk = chunk[n:]
            self._cursor += n

        try:
            self._stream = sd.OutputStream(
                samplerate=self._sr, channels=1, dtype="float32",
                callback=cb, finished_callback=self._on_end,
            )
            self._stream.start()
            self._timer.start()
        except Exception:
            self._playing = False
            self._play_btn.setText("▶  Play")

    def _pause(self):
        self._playing = False
        self._play_btn.setText("▶  Play")
        self._timer.stop()
        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass

    def _stop_now(self):
        self._playing = False
        if hasattr(self, "_play_btn"):
            self._play_btn.setText("▶  Play")
        if hasattr(self, "_timer"):
            self._timer.stop()
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._cursor = 0
        if hasattr(self, "_wave"):
            self._wave.set_position(0.0)

    def _on_end(self):
        def _safe():
            self._playing = False
            self._play_btn.setText("▶  Play")
            self._timer.stop()
            if self._data is not None:
                sel = self._wave.get_selection()
                self._cursor = int(sel[0] * len(self._data)) if sel else 0
                self._wave.set_position(self._cursor / max(1, len(self._data)))
                dur = len(self._data) / max(1, self._sr)
                t   = self._cursor / max(1, self._sr)
                self._lbl.setText(f"{_fmt(t)} / {_fmt(dur)}")
        QTimer.singleShot(0, _safe)

    def _tick(self):
        if self._data is None:
            return
        total = len(self._data)
        self._wave.set_position(min(self._cursor, total) / max(1, total))
        self._lbl.setText(
            f"{_fmt(self._cursor / max(1, self._sr))} / "
            f"{_fmt(total / max(1, self._sr))}"
        )

    def _seek(self, frac: float):
        if self._data is None:
            return
        was = self._playing
        self._pause()
        self._cursor = int(frac * len(self._data))
        self._wave.set_position(frac)
        if was:
            self._play()

class DropAudioWidget(QFrame):
    file_dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFixedHeight(58)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._file: Optional[str] = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.setContentsMargins(8, 4, 8, 4)
        self._lbl  = QLabel("🎙  Drop audio or click to select")
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl.setStyleSheet(f"color:{C['text3']};font-size:12px;")
        self._name = QLabel("")
        self._name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name.setStyleSheet(f"color:{C['accent']};font-size:11px;")
        lay.addWidget(self._lbl)
        lay.addWidget(self._name)
        self._set_style_normal()

    def _set_style_normal(self):
        self.setStyleSheet("""
            QFrame{border:1px dashed #444444;border-radius:8px;background:#1e1e1e;}
            QFrame:hover{border-color:#666666;background:#252525;}
        """)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select audio file",
                _get_last_dir("audio"),
                "Audio files (*.wav *.mp3 *.flac *.ogg)"
            )
            if path:
                _set_last_dir("audio", path)
                self._file = path
                self._name.setText(os.path.basename(path))
                self.file_dropped.emit(path)

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.setStyleSheet(
                f"QFrame{{border:1px solid {C['accent']};border-radius:8px;background:{C['accent_dim']};}}"
            )

    def dragLeaveEvent(self, e):
        self._set_style_normal()

    def dropEvent(self, e: QDropEvent):
        self._set_style_normal()
        urls = e.mimeData().urls()
        if urls:
            self._set(urls[0].toLocalFile())

    def _set(self, path: str):
        self._file = path
        self._lbl.setText("✓  Reference audio:")
        self._name.setText(Path(path).name)
        self.file_dropped.emit(path)

    def clear_file(self):
        self._file = None
        self._lbl.setText("🎙  Drop audio or click to select")
        self._name.setText("")

    @property
    def file_path(self) -> Optional[str]:
        return self._file


class CollapsibleSection(QWidget):
    def __init__(self, title: str, accent_color: str, parent=None):
        super().__init__(parent)
        self._accent = accent_color
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._btn = QPushButton(f"▸  {title}")
        self._btn.setCheckable(True)
        self._btn.setStyleSheet(f"""
            QPushButton{{
                background:#2a2a2a; border:1px solid #444444;
                border-radius:4px; color:#cccccc;
                font-size:11px; font-weight:600;
                padding:6px 12px; text-align:left;
            }}
            QPushButton:hover{{ background:#333333; border-color:#666666; color:white; }}
            QPushButton:checked{{ background:#333333; border-color:#666666; color:white; }}
        """)
        self._btn.clicked.connect(self._toggle)
        outer.addWidget(self._btn)

        self._wrapper = QWidget()
        self._wrapper.setVisible(False)
        self._wrapper.setStyleSheet("""
            QWidget{
                background:#252525;
                border:1px solid #333333;
                border-top:none;
                border-radius:0 0 4px 4px;
            }
        """)
        self._inner_layout = QVBoxLayout(self._wrapper)
        self._inner_layout.setContentsMargins(10, 8, 10, 10)
        self._inner_layout.setSpacing(8)
        outer.addWidget(self._wrapper)

    def _toggle(self, checked: bool):
        self._wrapper.setVisible(checked)
        label = self._btn.text()[3:]
        self._btn.setText(f"{'▾' if checked else '▸'}  {label}")

    def expand(self):
        if not self._btn.isChecked():
            self._btn.setChecked(True)
            self._wrapper.setVisible(True)
            label = self._btn.text()[3:]
            self._btn.setText(f"▾  {label}")

    def add_widget(self, w: QWidget):
        self._inner_layout.addWidget(w)

    def add_layout(self, lay):
        self._inner_layout.addLayout(lay)

    def set_enabled(self, enabled: bool):
        self._btn.setEnabled(enabled)


class FlowLayout(QWidget):
    def __init__(self, parent=None, h_gap: int = 4, v_gap: int = 4):
        super().__init__(parent)
        self._items: list = []
        self._hg = h_gap
        self._vg = v_gap
        if parent:
            pl = QVBoxLayout(parent)
            pl.setContentsMargins(0, 0, 0, 0)
            pl.addWidget(self)

    def add(self, w: QWidget):
        w.setParent(self)
        self._items.append(w)

    def resizeEvent(self, e):
        self._relayout()

    def _relayout(self):
        x = y = rh = 0
        for w in self._items:
            w.adjustSize()
            ww, wh = w.sizeHint().width(), w.sizeHint().height()
            if x + ww > self.width() and x > 0:
                x = 0
                y += rh + self._vg
                rh = 0
            w.setGeometry(x, y, ww, wh)
            x += ww + self._hg
            rh = max(rh, wh)
        self.setMinimumHeight(y + rh + 4)

    def sizeHint(self):
        return QSize(200, 80)


class FragmentTreeWidget(QTreeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_checked_item: Optional[QTreeWidgetItem] = None
        self._ignore_change: bool = False
        self.itemChanged.connect(self._on_item_changed_internal)

    def mousePressEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            item = self.itemAt(event.pos())
            if (item is not None
                    and item.data(COL_STATUS, Qt.ItemDataRole.UserRole) is not None
                    and self._last_checked_item is not None
                    and not item.isHidden()):
                col = self.columnAt(event.pos().x())
                if col == COL_STATUS:
                    target_state = self._last_checked_item.checkState(COL_STATUS)
                    self._apply_range_check(self._last_checked_item, item, target_state)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def _on_item_changed_internal(self, item: QTreeWidgetItem, column: int):
        if self._ignore_change:
            return
        if column == COL_STATUS and item.data(COL_STATUS, Qt.ItemDataRole.UserRole) is not None:
            self._last_checked_item = item

    def _apply_range_check(self, item1, item2, state) -> None:
        visible = self._get_all_leaf_items()
        try:
            pos1 = visible.index(item1)
            pos2 = visible.index(item2)
        except ValueError:
            return
        start, end = min(pos1, pos2), max(pos1, pos2)
        self._ignore_change = True
        for it in visible[start:end + 1]:
            if not it.isHidden():
                it.setCheckState(COL_STATUS, state)
        self._ignore_change = False
        self._last_checked_item = item2

    def _get_all_leaf_items(self) -> list:
        result = []
        root = self.invisibleRootItem()
        for i in range(root.childCount()):
            chapter = root.child(i)
            if chapter.isHidden():
                continue
            for j in range(chapter.childCount()):
                leaf = chapter.child(j)
                if not leaf.isHidden():
                    result.append(leaf)
        return result


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audiobook and Dubbing Generator by Mubumbutu")
        self.resize(1440, 960)
        self.setMinimumSize(1280, 820)

        _icon_path = ROOT_DIR / "icon.ico"
        if _icon_path.exists():
            self.setWindowIcon(QIcon(str(_icon_path)))

        self._backend          = None
        self._whisper_backend  = None
        self._preprocessor     = None
        self._ffmpeg_ok        = _check_ffmpeg()

        self._fragments:        List[Dict]          = []
        self._frag_items:       Dict[int, QTreeWidgetItem] = {}
        self._chapter_item:     Optional[QTreeWidgetItem] = None
        self._srt_path:         Optional[str]       = None
        self._output_dir:       str                 = str(OUTPUTS_DIR)
        self._is_running:       bool                = False
        self._completed_count:  int                 = 0
        self._worker:           Optional[TTSWorker] = None
        self._lektor_export_thread: Optional[LektorExportThread] = None

        self._audio_data: Optional[np.ndarray] = None
        self._audio_sr    = 44100
        self._audio_tmp:  Optional[str]        = None
        self._playing     = False
        self._cursor      = 0
        self._stream      = None

        self._play_end_sample:      int            = 0
        self._current_fragment_idx: Optional[int]  = None
        self._video_source_path:    Optional[str]  = None
        self._audio_undo_stack:     list           = []
        self._audio_redo_stack:     list           = []

        self._dubbing_mode:         bool            = False
        self._dubbing_video_path:   Optional[str]   = None
        self._hf_token:             Optional[str]   = None
        self._speaker_list:         List[str]       = []
        self._speaker_voices:       Dict            = {}

        self._ebook_fragments:      List[Dict]      = []
        self._ebook_frag_items:     Dict[int, QTreeWidgetItem] = {}
        self._ebook_chapter_item:   Optional[QTreeWidgetItem] = None
        self._epub_path:            Optional[str]   = None
        self._ebook_output_dir:     str             = str(OUTPUTS_DIR)
        self._synthesis_source:     str             = 'srt'

        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._playback_tick)

        self._file_watcher = QFileSystemWatcher(self)
        self._file_watcher.fileChanged.connect(self._on_audio_file_changed)

        self._build_ui()
        self.setStyleSheet(STYLE)
        self._init_backends()

    def _init_backends(self):
        try:
            model_id = (
                self._model_combo.currentData()
                if hasattr(self, "_model_combo")
                else MODEL_OPTIONS[0][0]
            )
            self._backend         = create_backend(model_id)
            self._whisper_backend = WhisperBackend(WHISPER_DIR)
            self._preprocessor    = AudioPreprocessor(PROC_DIR)
            self._refresh_fish_ui()
            self._refresh_whisper_ui()
            self._update_device_label()
            self._update_params_visibility(model_id)
            self._update_whisper_visibility()
            self._update_voice_section_for_model(model_id)
            self._update_ref_text_visibility_for_model(model_id)
            self._set_status("Ready — select model and click 'Load model'.")
        except Exception as e:
            QMessageBox.critical(self, "Initialization error",
                f"Cannot initialize backend:\n{e}")

    def _build_ui(self):
        c = QWidget()
        self.setCentralWidget(c)
        root = QVBoxLayout(c)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.setHandleWidth(2)
        sp.addWidget(self._make_left_panel())
        sp.addWidget(self._make_right_panel())
        sp.setSizes([680, 560])
        root.addWidget(sp, 1)

        root.addWidget(self._make_player_bar())

        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self._on_wave_undo)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self._on_wave_redo)

        self._status_bar = QStatusBar()
        self._status_lbl = QLabel("…")
        self._device_lbl = QLabel()
        self._status_bar.addWidget(self._status_lbl, 1)
        self._status_bar.addPermanentWidget(self._device_lbl)
        self.setStatusBar(self._status_bar)

    def _make_header(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(56)
        w.setStyleSheet(f"""QWidget{{
            background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 {C['surface']},stop:0.5 #0d1e2a,stop:1 {C['surface']});
            border-bottom:1px solid {C['border']};}}""")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(20, 0, 20, 0)
        if _ACTIVE_BACKEND_CLASSES:
            _inst = object.__new__(_ACTIVE_BACKEND_CLASSES[0])
            try:
                _hicon = _inst.header_icon
            except Exception:
                _hicon = "🎙"
            try:
                _htitle = _inst.header_title
            except Exception:
                _htitle = "TTS Studio"
        else:
            _hicon  = "🎙"
            _htitle = "TTS Studio"
        logo  = QLabel(_hicon)
        logo.setStyleSheet("font-size:22px;background:transparent;border:none;")
        title = QLabel(_htitle)
        title.setStyleSheet(f"font-size:18px;font-weight:700;color:{C['text']};background:transparent;border:none;")
        sub   = QLabel("SRT Lektor Studio")
        sub.setStyleSheet(f"font-size:11px;color:{C['accent']};background:transparent;border:none;letter-spacing:2px;")
        col = QVBoxLayout()
        col.setSpacing(1)
        col.addWidget(title)
        col.addWidget(sub)
        lay.addWidget(logo)
        lay.addSpacing(8)
        lay.addLayout(col)
        lay.addStretch()
        model_lbl = QLabel("Model:")
        model_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;background:transparent;border:none;")
        self._model_combo = QComboBox()
        for key, name in MODEL_OPTIONS:
            self._model_combo.addItem(name, key)
        self._model_combo.setFixedWidth(290)
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        lay.addWidget(model_lbl)
        lay.addSpacing(4)
        lay.addWidget(self._model_combo)
        lay.addSpacing(12)
        self._model_lbl = QLabel("● Model not downloaded")
        self._model_lbl.setStyleSheet(f"color:{C['warning']};font-size:11px;background:transparent;border:none;")
        self._dl_fish_btn = QPushButton("⬇  Download model")
        self._dl_fish_btn.setStyleSheet(_btn(C["accent"]))
        self._dl_fish_btn.clicked.connect(self._start_model_download)
        self._load_btn = QPushButton("⚡  Load model")
        self._load_btn.setStyleSheet(_btn(C["accent"]))
        self._load_btn.clicked.connect(self._load_fish_model)
        self._unload_btn = QPushButton("🗑  Unload")
        self._unload_btn.setStyleSheet(_btn(C["error"]))
        self._unload_btn.setVisible(False)
        self._unload_btn.clicked.connect(self._unload_fish_model)
        lay.addWidget(self._model_lbl)
        lay.addSpacing(10)
        lay.addWidget(self._dl_fish_btn)
        lay.addSpacing(6)
        lay.addWidget(self._load_btn)
        lay.addSpacing(6)
        lay.addWidget(self._unload_btn)
        return w

    def _make_left_panel(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 14, 7, 14)
        lay.setSpacing(8)
 
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #333333;
                border-radius: 0 3px 3px 3px;
                background: #252525;
            }
            QTabBar::tab {
                background: #1e1e1e;
                border: 1px solid #333333;
                border-bottom: none;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
                color: #777777;
                padding: 5px 14px;
                font-size: 12px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #252525;
                color: #cccccc;
                border-color: #444444;
            }
            QTabBar::tab:hover {
                background: #2a2a2a;
                color: #aaaaaa;
            }
        """)
        self._tabs.tabBar().setStyleSheet("""
            QTabBar::tab {
                background: #1e1e1e;
                border: 1px solid #333333;
                border-bottom: none;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
                color: #777777;
                padding: 5px 14px;
                font-size: 12px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #252525;
                color: #cccccc;
                border-color: #444444;
            }
            QTabBar::tab:hover {
                background: #2a2a2a;
                color: #aaaaaa;
            }
        """)
        self._tabs.addTab(self._make_srt_tab(),    "📄  SRT Fragments")
        self._tabs.addTab(self._make_ebook_tab(),  "📚  Ebook Fragments")
        self._tabs.addTab(self._make_quick_tts_tab(), "⚡  Quick TTS")
        lay.addWidget(self._tabs, 1)
        return w

    def _make_srt_tab(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        top = QHBoxLayout()
        self._btn_load_srt = QPushButton("📂  Load SRT / TXT")
        self._btn_load_srt.clicked.connect(self._load_srt_file)
        self._btn_close_srt = QPushButton("✕  Close")
        self._btn_close_srt.setEnabled(False)
        self._btn_close_srt.clicked.connect(self._close_srt_file)

        self._btn_show_video_wave = QPushButton("🎬  Show Waveform from video")
        self._btn_show_video_wave.setToolTip(
            "Select a video file to display its audio waveform above the synthesized audio.\n"
            "Clicking an SRT fragment will auto-select its time range on the video waveform\n"
            "so you can listen to how it sounds in the original video.\n"
            "Also sets the video file for Lektor Export automatically."
        )
        self._btn_show_video_wave.clicked.connect(self._on_show_video_waveform)

        self._btn_dubbing = QPushButton("🎙  I want dubbing")
        self._btn_dubbing.setVisible(False)
        self._btn_dubbing.setEnabled(False)
        self._btn_dubbing.setToolTip(
            "Automatic multi-speaker dubbing:\n\n"
            "1. Extracts and isolates vocals from your video\n"
            "2. Detects all speakers via pyannote speaker-diarization-3.1\n"
            "   (speakers sorted by total speaking time: Person 1 = most speech)\n"
            "3. Assigns each SRT fragment to the correct speaker\n"
            "4. Adds a separate voice cloning panel for each detected speaker\n"
            "   (drop reference audio + auto-transcribe per speaker)\n"
            "5. During synthesis, TTS uses the matching speaker's reference voice\n\n"
            "(accept pyannote/speaker-diarization-3.1 terms on huggingface.co)"
        )
        self._btn_dubbing.clicked.connect(self._on_dubbing_clicked)

        self._btn_save_session = QPushButton("💾  Save session")
        self._btn_save_session.setEnabled(False)
        self._btn_save_session.clicked.connect(self._save_session)
        self._btn_load_session = QPushButton("📁  Load session")
        self._btn_load_session.clicked.connect(self._load_session)

        top.addWidget(self._btn_load_srt)
        top.addWidget(self._btn_close_srt)
        top.addWidget(self._btn_show_video_wave)
        top.addWidget(self._btn_dubbing)
        top.addStretch()
        top.addWidget(self._btn_save_session)
        top.addWidget(self._btn_load_session)
        lay.addLayout(top)

        self._srt_label = QLabel("No SRT file loaded")
        self._srt_label.setStyleSheet(f"color:{C['text3']};font-size:11px;font-style:italic;")
        lay.addWidget(self._srt_label)

        filter_row = QHBoxLayout()
        filter_lbl = QLabel("Filter:")
        filter_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        filter_lbl.setFixedWidth(38)
        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("Filter fragments…")
        self._filter_edit.setFixedHeight(26)
        self._filter_edit.textChanged.connect(self._apply_filter)
        filter_row.addWidget(filter_lbl)
        filter_row.addWidget(self._filter_edit)

        sel_all  = QPushButton("☑ All")
        sel_all.setFixedHeight(26)
        sel_none = QPushButton("☐ None")
        sel_none.setFixedHeight(26)
        sel_fail = QPushButton("❌ Failed")
        sel_fail.setFixedHeight(26)
        sel_fail.setCheckable(True)
        sel_pending = QPushButton("⬜ Pending")
        sel_pending.setFixedHeight(26)
        sel_pending.setCheckable(True)

        self._sel_fail_btn    = sel_fail
        self._sel_pending_btn = sel_pending

        sel_overlong = QPushButton("⏳ Overlong")
        sel_overlong.setFixedHeight(26)
        sel_overlong.setStyleSheet("font-size:11px;padding:2px 8px;")
        sel_overlong.setToolTip("Select fragments where generated audio is longer than the time slot\n(yellow / orange / red timing in column Timing)")
        sel_overlong.clicked.connect(self._show_timing_issues_dialog)

        btn_snap_timing = QPushButton("⏱ Snap")
        btn_snap_timing.setFixedHeight(26)
        btn_snap_timing.setStyleSheet("font-size:11px;padding:2px 8px;")
        btn_snap_timing.setToolTip(
            "Snap end times to next fragment's start time.\n\n"
            "Example:\n"
            "  Fragment 1: 9:24.100 → 9:29.000\n"
            "  Fragment 2: 9:29.200 → 9:31.100\n"
            "After snap:\n"
            "  Fragment 1: 9:24.100 → 9:29.200\n"
            "  Fragment 2: 9:29.200 → 9:31.200\n"
            "  (last fragment end time is not changed)\n\n"
            "Removes gaps between consecutive fragments."
        )
        btn_snap_timing.clicked.connect(self._snap_timing)

        for btn in [sel_all, sel_none, sel_fail, sel_pending, sel_overlong]:
            btn.setStyleSheet("font-size:11px;padding:2px 8px;")

        sel_all.clicked.connect(lambda: self._select_all_reset(True))
        sel_none.clicked.connect(lambda: self._select_all_reset(False))
        sel_fail.toggled.connect(lambda _: self._apply_status_filter())
        sel_pending.toggled.connect(lambda _: self._apply_status_filter())

        filter_row.addSpacing(4)
        filter_row.addWidget(sel_all)
        filter_row.addWidget(sel_none)
        filter_row.addWidget(sel_fail)
        filter_row.addWidget(sel_overlong)
        filter_row.addWidget(sel_pending)
        filter_row.addSpacing(8)
        filter_row.addWidget(btn_snap_timing)
        lay.addLayout(filter_row)

        self._tree = FragmentTreeWidget()
        self._tree.setColumnCount(4)
        self._tree.setHeaderLabels(["", "Text", "Speaker", "Timing"])
        self._tree.header().setStretchLastSection(False)
        self._tree.setColumnWidth(COL_STATUS,   28)
        self._tree.setColumnWidth(COL_FRAGMENT, 420)
        self._tree.setColumnWidth(COL_SPEAKER,  100)
        self._tree.setColumnWidth(COL_TIMING,   220)
        self._tree.setWordWrap(True)
        self._tree.setAlternatingRowColors(True)
        self._tree.setIndentation(0)
        self._tree.setMinimumHeight(160)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._show_context_menu)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        self._tree.itemDoubleClicked.connect(self._on_tree_item_double_clicked)
        self._tree.setStyleSheet(f"""
            QTreeWidget {{
                background: #1e1e1e;
                border: 1px solid {C["border"]};
                border-radius: 4px;
                color: {C["text"]};
                alternate-background-color: #252525;
                show-decoration-selected: 1;
            }}
            QTreeWidget::item {{
                padding-top: 4px;
                padding-bottom: 4px;
                min-height: 44px;
            }}
            QTreeWidget::item:selected {{
                background: {C["accent2"]};
                color: {C["text"]};
            }}
            QTreeWidget::item:hover:!selected {{
                background: #2a2a2a;
            }}
            QTreeWidget::indicator {{
                width: 16px;
                height: 16px;
                margin-left: 4px;
                border: 2px solid {C["border2"]};
                border-radius: 3px;
                background: #1e1e1e;
            }}
            QTreeWidget::indicator:checked {{
                border: 2px solid {C["accent"]};
                background: {C["accent2"]};
            }}
            QTreeWidget::indicator:unchecked:hover {{
                border: 2px solid {C["accent"]};
            }}
            QHeaderView::section {{
                background: #252525;
                color: {C["text2"]};
                border: none;
                border-right: 1px solid {C["border"]};
                padding: 4px 8px;
                font-size: 11px;
            }}
        """)

        self._tree.header().setSectionResizeMode(COL_STATUS,   QHeaderView.ResizeMode.Fixed)
        self._tree.header().setSectionResizeMode(COL_FRAGMENT, QHeaderView.ResizeMode.Interactive)
        self._tree.header().setSectionResizeMode(COL_SPEAKER,  QHeaderView.ResizeMode.Interactive)
        self._tree.header().setSectionResizeMode(COL_TIMING,   QHeaderView.ResizeMode.Stretch)

        lay.addWidget(self._tree, 1)

        preview_lbl = QLabel("Fragment preview:")
        preview_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        lay.addWidget(preview_lbl)
        self._preview_text = QPlainTextEdit()
        self._preview_text.setReadOnly(True)
        self._preview_text.setFixedHeight(124)
        self._preview_text.setFont(QFont("Segoe UI", 11))
        lay.addWidget(self._preview_text)

        return w
    
    def _snap_timing(self):
        if len(self._fragments) < 2:
            return

        checked_indices: set = set()
        if self._chapter_item:
            for i in range(self._chapter_item.childCount()):
                child = self._chapter_item.child(i)
                if (child.checkState(COL_STATUS) == Qt.CheckState.Checked
                        and not child.isHidden()):
                    idx = child.data(COL_STATUS, Qt.ItemDataRole.UserRole)
                    if idx is not None:
                        checked_indices.add(idx)

        checked_count = len(checked_indices)

        dlg = QDialog(self)
        dlg.setWindowTitle("Snap end times — confirm")
        dlg.resize(480, 0)
        dlg.setStyleSheet(self.styleSheet())

        lay = QVBoxLayout(dlg)
        lay.setSpacing(12)
        lay.setContentsMargins(16, 16, 16, 12)

        title_lbl = QLabel("⚠️  Snap end times to next fragment's start")
        title_lbl.setStyleSheet(f"color:{C['warning']};font-size:13px;font-weight:bold;")
        lay.addWidget(title_lbl)

        desc = QLabel(
            "This operation will overwrite the end time of each fragment "
            "to match the start time of the next fragment, removing all gaps.\n\n"
            "Recommended workflow before using Snap:\n"
            "  1. Merge all fragments that belong together into single entries.\n"
            "  2. Verify timing in the tree — yellow/orange/red indicators "
            "signal fragments whose synthesized audio already exceeds the slot.\n"
            "  3. Only then run Snap to close the remaining gaps.\n\n"
            "Running Snap before merging may cause closely spaced fragments to "
            "overlap in time after synthesis, leading to audio cutting off early."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color:{C['text']};font-size:12px;")
        lay.addWidget(desc)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet(f"color:{C['border']};")
        lay.addWidget(separator)

        warn_lbl = QLabel("This action cannot be undone.")
        warn_lbl.setStyleSheet(f"color:{C['error']};font-size:11px;font-style:italic;")
        lay.addWidget(warn_lbl)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        btn_all      = QPushButton(f"Snap all  ({len(self._fragments)})")
        btn_checked  = QPushButton(f"Snap checked  ({checked_count})")
        btn_cancel   = QPushButton("Cancel")

        btn_all.setStyleSheet(_btn(C["warning"]))
        btn_checked.setStyleSheet(_btn(C["accent"]))
        btn_checked.setEnabled(checked_count >= 1)

        result = {"action": None}
        btn_all.clicked.connect(lambda: (result.update({"action": "all"}), dlg.accept()))
        btn_checked.clicked.connect(lambda: (result.update({"action": "checked"}), dlg.accept()))
        btn_cancel.clicked.connect(dlg.reject)

        btn_row.addStretch()
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_checked)
        btn_row.addWidget(btn_cancel)
        lay.addLayout(btn_row)

        dlg.adjustSize()
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        if result["action"] == "all":
            targets = self._fragments
        elif result["action"] == "checked":
            targets = sorted(
                [f for f in self._fragments if f['index'] in checked_indices],
                key=lambda f: f['index'],
            )
        else:
            return

        target_set = {f['index'] for f in targets}
        changed = 0
        for i in range(len(self._fragments) - 1):
            current = self._fragments[i]
            nxt     = self._fragments[i + 1]
            if current['index'] not in target_set:
                continue
            next_start = nxt.get('start_ms', 0) or 0
            if (current.get('end_ms', 0) or 0) != next_start:
                current['end_ms']     = next_start
                current['srt_end_ms'] = next_start
                current['timestamp']  = (
                    f"{_ms_to_ts(current.get('start_ms', 0) or 0)} --> {_ms_to_ts(next_start)}"
                )
                self._update_tree_item(current['index'])
                changed += 1

        if changed:
            self._set_status(
                f"Snap timing: {changed} fragment(s) updated.", C["accent"]
            )
        else:
            self._set_status("Snap timing: nothing to change.")
    
    def _make_ebook_tab(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        top = QHBoxLayout()
        self._btn_load_ebook = QPushButton("📂  Load Ebook")
        self._btn_load_ebook.clicked.connect(self._load_ebook_file)
        self._btn_close_ebook = QPushButton("✕  Close")
        self._btn_close_ebook.setEnabled(False)
        self._btn_close_ebook.clicked.connect(self._close_ebook_file)
        self._btn_save_ebook_session = QPushButton("💾  Save session")
        self._btn_save_ebook_session.setEnabled(False)
        self._btn_save_ebook_session.clicked.connect(self._save_ebook_session)
        self._btn_load_ebook_session = QPushButton("📁  Load session")
        self._btn_load_ebook_session.clicked.connect(self._load_session)

        top.addWidget(self._btn_load_ebook)
        top.addWidget(self._btn_close_ebook)
        top.addStretch()
        top.addWidget(self._btn_save_ebook_session)
        top.addWidget(self._btn_load_ebook_session)
        lay.addLayout(top)

        self._epub_label = QLabel("No ebook loaded")
        self._epub_label.setStyleSheet(f"color:{C['text3']};font-size:11px;font-style:italic;")
        lay.addWidget(self._epub_label)

        filter_row = QHBoxLayout()
        filter_lbl = QLabel("Filter:")
        filter_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        filter_lbl.setFixedWidth(38)

        self._ebook_filter_edit = QLineEdit()
        self._ebook_filter_edit.setPlaceholderText("Filter fragments…")
        self._ebook_filter_edit.setFixedHeight(26)
        self._ebook_filter_edit.textChanged.connect(self._apply_ebook_filter)

        filter_row.addWidget(filter_lbl)
        filter_row.addWidget(self._ebook_filter_edit)

        sel_all  = QPushButton("☑ All")
        sel_all.setFixedHeight(26)
        sel_none = QPushButton("☐ None")
        sel_none.setFixedHeight(26)
        sel_fail = QPushButton("❌ Failed")
        sel_fail.setFixedHeight(26)
        sel_fail.setCheckable(True)
        sel_pending = QPushButton("⬜ Pending")
        sel_pending.setFixedHeight(26)
        sel_pending.setCheckable(True)

        self._ebook_sel_fail_btn    = sel_fail
        self._ebook_sel_pending_btn = sel_pending

        for btn in [sel_all, sel_none, sel_fail, sel_pending]:
            btn.setStyleSheet("font-size:11px;padding:2px 8px;")

        sel_all.clicked.connect(lambda: self._select_all_ebook_reset(True))
        sel_none.clicked.connect(lambda: self._select_all_ebook_reset(False))
        sel_fail.toggled.connect(lambda _: self._apply_ebook_status_filter())
        sel_pending.toggled.connect(lambda _: self._apply_ebook_status_filter())

        filter_row.addSpacing(4)
        filter_row.addWidget(sel_all)
        filter_row.addWidget(sel_none)
        filter_row.addWidget(sel_fail)
        filter_row.addWidget(sel_pending)
        lay.addLayout(filter_row)

        self._ebook_tree = FragmentTreeWidget()
        self._ebook_tree.setColumnCount(4)
        self._ebook_tree.setHeaderLabels(["", "Text", "Speaker", "Duration"])
        self._ebook_tree.header().setStretchLastSection(False)
        self._ebook_tree.setColumnWidth(COL_STATUS,   28)
        self._ebook_tree.setColumnWidth(COL_FRAGMENT, 370)
        self._ebook_tree.setColumnWidth(COL_SPEAKER,  90)
        self._ebook_tree.setColumnWidth(COL_TIMING,   160)
        self._ebook_tree.setWordWrap(True)
        self._ebook_tree.setAlternatingRowColors(True)
        self._ebook_tree.setIndentation(0)
        self._ebook_tree.setMinimumHeight(160)
        self._ebook_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._ebook_tree.customContextMenuRequested.connect(self._show_ebook_context_menu)
        self._ebook_tree.itemSelectionChanged.connect(self._on_ebook_selection_changed)
        self._ebook_tree.itemDoubleClicked.connect(self._on_ebook_tree_item_double_clicked)
        self._ebook_tree.itemChanged.connect(lambda item, col: self._update_preview_btn_state())

        self._ebook_tree.setStyleSheet(f"""
            QTreeWidget {{
                background: #1e1e1e;
                border: 1px solid {C["border"]};
                border-radius: 4px;
                color: {C["text"]};
                alternate-background-color: #252525;
                show-decoration-selected: 1;
            }}
            QTreeWidget::item {{
                padding-top: 4px;
                padding-bottom: 4px;
                min-height: 44px;
            }}
            QTreeWidget::item:selected {{
                background: {C["accent2"]};
                color: {C["text"]};
            }}
            QTreeWidget::item:hover:!selected {{
                background: #2a2a2a;
            }}
            QTreeWidget::indicator {{
                width: 16px;
                height: 16px;
                margin-left: 4px;
                border: 2px solid {C["border2"]};
                border-radius: 3px;
                background: #1e1e1e;
            }}
            QTreeWidget::indicator:checked {{
                border: 2px solid {C["accent"]};
                background: {C["accent2"]};
            }}
            QTreeWidget::indicator:unchecked:hover {{
                border: 2px solid {C["accent"]};
            }}
            QHeaderView::section {{
                background: #252525;
                color: {C["text2"]};
                border: none;
                border-right: 1px solid {C["border"]};
                padding: 4px 8px;
                font-size: 11px;
            }}
        """)

        self._ebook_tree.header().setSectionResizeMode(COL_STATUS,   QHeaderView.ResizeMode.Fixed)
        self._ebook_tree.header().setSectionResizeMode(COL_FRAGMENT, QHeaderView.ResizeMode.Interactive)
        self._ebook_tree.header().setSectionResizeMode(COL_SPEAKER,  QHeaderView.ResizeMode.Interactive)
        self._ebook_tree.header().setSectionResizeMode(COL_TIMING,   QHeaderView.ResizeMode.Stretch)

        lay.addWidget(self._ebook_tree, 1)

        preview_lbl = QLabel("Fragment preview:")
        preview_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        lay.addWidget(preview_lbl)

        self._ebook_preview_text = QPlainTextEdit()
        self._ebook_preview_text.setReadOnly(True)
        self._ebook_preview_text.setMinimumHeight(120)
        self._ebook_preview_text.setMaximumHeight(124)
        self._ebook_preview_text.setFont(QFont("Segoe UI", 11))
        lay.addWidget(self._ebook_preview_text)

        preview_btn_row = QHBoxLayout()
        self._preview_audiobook_btn = QPushButton("🔊  Preview selected fragments")
        self._preview_audiobook_btn.setEnabled(False)
        self._preview_audiobook_btn.setFixedHeight(30)
        self._preview_audiobook_btn.setStyleSheet(_btn(C["accent"]))
        self._preview_audiobook_btn.clicked.connect(self._preview_selected_fragments)
        preview_btn_row.addWidget(self._preview_audiobook_btn)
        preview_btn_row.addStretch()
        lay.addLayout(preview_btn_row)

        return w
    
    def _make_quick_tts_tab(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)
 
        tg  = QGroupBox("Input text")
        tgl = QVBoxLayout(tg)
        tgl.setSpacing(6)
        self._quick_edit = QTextEdit()
        self._quick_edit.setPlaceholderText(
            "Enter text to synthesize…\n\n"
            "Example: Hello! [excited] This sounds incredible! [chuckle]\n"
            "Shortcut: Ctrl+Enter = Generate"
        )
        self._quick_edit.setMinimumHeight(150)
        self._quick_edit.setFont(QFont("Segoe UI", 13))
        self._quick_edit.textChanged.connect(
            lambda: self._char_lbl.setText(f"{len(self._quick_edit.toPlainText())} chars")
        )
        tgl.addWidget(self._quick_edit)
        self._char_lbl = QLabel("0 chars")
        self._char_lbl.setStyleSheet(f"color:{C['text3']};font-size:10px;")
        cr = QHBoxLayout()
        cr.addWidget(self._char_lbl)
        cr.addStretch()
        tgl.addLayout(cr)
        lay.addWidget(tg, 2)
 
        btn_row = QHBoxLayout()
        self._quick_btn = QPushButton("  🚀  Synthesize")
        self._quick_btn.setStyleSheet(SYNTH_BTN_STYLE)
        self._quick_btn.setEnabled(False)
        self._quick_btn.clicked.connect(self._synthesize_quick_tts)
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(
            lambda: self._synthesize_quick_tts() if self._tabs.currentIndex() == 2 else None
        )
        self._save_wav_btn = QPushButton("💾  Save WAV")
        self._save_wav_btn.setEnabled(False)
        self._save_wav_btn.clicked.connect(self._save_audio)
        self._save_mp3_btn = QPushButton("🎵  Save MP3")
        self._save_mp3_btn.setEnabled(False)
        self._save_mp3_btn.clicked.connect(lambda: self._save_audio("mp3"))
        btn_row.addWidget(self._quick_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._save_wav_btn)
        btn_row.addWidget(self._save_mp3_btn)
        lay.addLayout(btn_row)
 
        return w

    def _make_right_panel(self) -> QWidget:
        outer = QWidget()
        outer_lay = QVBoxLayout(outer)
        outer_lay.setContentsMargins(0, 0, 0, 0)
        outer_lay.setSpacing(0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(7, 14, 14, 14)
        lay.setSpacing(10)
        lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        model_grp = QGroupBox("Model")
        model_lay = QVBoxLayout(model_grp)
        model_lay.setSpacing(8)
        model_row = QHBoxLayout()
        model_row.setSpacing(6)
        self._model_combo = QComboBox()
        self._model_combo.setToolTip("Select the voice model to use for synthesis")
        for mid, name in MODEL_OPTIONS:
            self._model_combo.addItem(name, mid)
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_row.addWidget(self._model_combo, 1)
        model_lay.addLayout(model_row)
        status_row = QHBoxLayout()
        status_row.setSpacing(6)
        self._model_lbl = QLabel("● Model not downloaded")
        self._model_lbl.setStyleSheet(f"color:{C['warning']};font-size:11px;")
        self._dl_fish_btn = QPushButton("⬇ Download model")
        self._dl_fish_btn.setToolTip("Download the selected model to your device")
        self._dl_fish_btn.setStyleSheet(_btn(C["accent"]))
        self._dl_fish_btn.setFixedHeight(28)
        self._dl_fish_btn.clicked.connect(self._start_model_download)
        self._load_btn = QPushButton("⚡ Load model")
        self._load_btn.setToolTip("Load the downloaded model into VRAM for synthesis")
        self._load_btn.setStyleSheet(_btn(C["accent"]))
        self._load_btn.setFixedHeight(28)
        self._load_btn.clicked.connect(self._load_fish_model)
        self._unload_btn = QPushButton("🗑 Unload")
        self._unload_btn.setToolTip("Unload the model from memory to free VRAM")
        self._unload_btn.setStyleSheet(_btn(C["error"]))
        self._unload_btn.setFixedHeight(28)
        self._unload_btn.setVisible(False)
        self._unload_btn.clicked.connect(self._unload_fish_model)
        status_row.addWidget(self._model_lbl)
        status_row.addStretch()
        status_row.addWidget(self._dl_fish_btn)
        status_row.addWidget(self._load_btn)
        status_row.addWidget(self._unload_btn)
        model_lay.addLayout(status_row)
        lay.addWidget(model_grp)

        voice_section = CollapsibleSection("🎙 Voice cloning", C["accent"])
        self._voice_single_container = QWidget()
        self._voice_single_container.setStyleSheet("background: transparent;")
        single_lay = QVBoxLayout(self._voice_single_container)
        single_lay.setContentsMargins(0, 0, 0, 0)
        single_lay.setSpacing(6)

        self._omnivoice_mode_widget = QWidget()
        self._omnivoice_mode_widget.setStyleSheet("background: transparent;")
        ov_lay = QVBoxLayout(self._omnivoice_mode_widget)
        ov_lay.setContentsMargins(0, 0, 0, 4)
        ov_lay.setSpacing(6)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        mode_lbl = QLabel("Voice mode:")
        mode_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        mode_lbl.setFixedWidth(76)
        self._voice_mode_combo = QComboBox()
        self._voice_mode_combo.addItem("🎤  Voice Cloning", "cloning")
        self._voice_mode_combo.addItem("🎨  Voice Design", "design")
        self._voice_mode_combo.addItem("🎲  Auto Voice", "auto")
        self._voice_mode_combo.setToolTip(
            "Voice Cloning: clone from reference WAV\n"
            "Voice Design: describe voice attributes in text\n"
            "Auto Voice: let the model pick its own voice"
        )
        self._voice_mode_combo.currentIndexChanged.connect(self._on_voice_mode_changed)
        mode_row.addWidget(mode_lbl)
        mode_row.addWidget(self._voice_mode_combo, 1)
        ov_lay.addLayout(mode_row)
        self._omnivoice_hint_lbl = QLabel("")
        self._omnivoice_hint_lbl.setStyleSheet(
            f"color:{C['text3']};font-size:10px;font-style:italic;"
        )
        self._omnivoice_hint_lbl.setWordWrap(True)
        ov_lay.addWidget(self._omnivoice_hint_lbl)
        self._instruct_widget = QWidget()
        self._instruct_widget.setStyleSheet("background: transparent;")
        instruct_lay_v = QVBoxLayout(self._instruct_widget)
        instruct_lay_v.setContentsMargins(0, 0, 0, 0)
        instruct_lay_v.setSpacing(4)
        instruct_lbl = QLabel("Voice attributes:")
        instruct_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        self._instruct_text = QPlainTextEdit()
        self._instruct_text.setPlaceholderText(
            "e.g. female, low pitch, British accent, whisper"
        )
        self._instruct_text.setFixedHeight(52)
        self._instruct_text.setFont(QFont("Segoe UI", 12))
        self._instruct_text.setToolTip(
            "Describe desired voice: gender, age, pitch, accent, style…\n"
            "Example: 'male, elderly, low pitch, American accent'"
        )
        instruct_lay_v.addWidget(instruct_lbl)
        instruct_lay_v.addWidget(self._instruct_text)
        ov_lay.addWidget(self._instruct_widget)
        self._instruct_widget.setVisible(False)
        self._omnivoice_mode_widget.setVisible(False)
        single_lay.addWidget(self._omnivoice_mode_widget)
        self._supertonic_single_hint = QLabel(
            "Select voice in ⚙️ Generation Parameters below.\n"
            "In dubbing mode each speaker gets its own voice."
        )
        self._supertonic_single_hint.setStyleSheet(
            f"color:{C['text3']};font-size:10px;font-style:italic;"
        )
        self._supertonic_single_hint.setWordWrap(True)
        self._supertonic_single_hint.setVisible(False)
        single_lay.addWidget(self._supertonic_single_hint)

        self._cloning_widget = QWidget()
        self._cloning_widget.setStyleSheet("background: transparent;")
        cloning_lay = QVBoxLayout(self._cloning_widget)
        cloning_lay.setContentsMargins(0, 0, 0, 0)
        cloning_lay.setSpacing(6)
        self._drop = DropAudioWidget()
        self._drop.setToolTip(
            "Drag and drop your reference audio file here (WAV, MP3, etc.)\n"
            "Recommended: 3–15 seconds of clean speech"
        )
        self._drop.file_dropped.connect(self._on_ref_dropped)
        cloning_lay.addWidget(self._drop)
        self._ref_player = RefAudioPlayer()
        self._ref_player.setToolTip("Playback controls for the loaded reference audio")
        cloning_lay.addWidget(self._ref_player)
        bar = QHBoxLayout()
        bar.setSpacing(6)
        clear_btn = QPushButton("✕ Remove reference")
        clear_btn.setToolTip("Remove the current reference audio")
        clear_btn.setFixedHeight(28)
        clear_btn.clicked.connect(self._clear_ref)
        bar.addWidget(clear_btn)
        bar.addStretch()
        cloning_lay.addLayout(bar)
        self._ref_audio_info_lbl = QLabel("")
        self._ref_audio_info_lbl.setStyleSheet(
            f"color:{C['text2']};font-size:11px;font-style:italic;"
        )
        cloning_lay.addWidget(self._ref_audio_info_lbl)
        self._whisper_ref_text_label = QLabel("Reference audio transcription:")
        self._whisper_ref_text_label.setStyleSheet(
            f"color:{C['text2']};font-size:11px;"
        )
        self._ref_text = QPlainTextEdit()
        self._ref_text.setToolTip("Edit or paste the transcription of the reference audio")
        self._ref_text.setPlaceholderText(
            "Enter or generate transcription of the reference audio…"
        )
        self._ref_text.setFixedHeight(68)
        self._ref_text.setFont(QFont("Segoe UI", 12))
        cloning_lay.addWidget(self._whisper_ref_text_label)
        cloning_lay.addWidget(self._ref_text)
        self._proc_section = CollapsibleSection("🔧 Prepare reference audio", C["accent"])
        hint = QLabel(
            "Converts audio to optimal format for voice cloning. "
            "Demucs v4 (~1.5 GB VRAM) isolates voice from background/music."
        )
        hint.setStyleSheet(f"color:{C['text3']};font-size:10px;")
        hint.setWordWrap(True)
        self._proc_section.add_widget(hint)
        self._mono_check = QCheckBox("Convert to mono")
        self._mono_check.setToolTip("Convert reference audio to single (mono) channel")
        self._mono_check.setChecked(False)
        self._proc_section.add_widget(self._mono_check)
        self._chk_demucs = QCheckBox("Vocal isolation")
        self._chk_demucs.setChecked(False)
        self._chk_demucs.setToolTip(
            "Removes music and background effects, keeps only voice."
        )
        self._chk_demucs.toggled.connect(self._on_demucs_toggled)
        self._proc_section.add_widget(self._chk_demucs)
        bd_row = QHBoxLayout()
        bd_row.setSpacing(8)
        self._chk_bit_depth = QCheckBox("Output bit depth:")
        self._chk_bit_depth.setToolTip(
            "Set a specific bit depth for the processed audio output"
        )
        self._chk_bit_depth.setChecked(False)
        self._bit_depth_combo = QComboBox()
        self._bit_depth_combo.setToolTip(
            "Choose the output bit depth for the processed reference audio"
        )
        self._bit_depth_combo.addItem("16-bit", "PCM_16")
        self._bit_depth_combo.addItem("24-bit", "PCM_24")
        self._bit_depth_combo.addItem("32-bit float", "FLOAT")
        self._bit_depth_combo.setCurrentIndex(0)
        self._bit_depth_combo.setEnabled(False)
        self._chk_bit_depth.toggled.connect(
            lambda checked: self._bit_depth_combo.setEnabled(checked)
        )
        bd_row.addWidget(self._chk_bit_depth)
        bd_row.addWidget(self._bit_depth_combo)
        bd_row.addStretch()
        self._proc_section.add_layout(bd_row)
        sr_row = QHBoxLayout()
        sr_row.setSpacing(6)
        sr_lbl = QLabel("Sample rate:")
        sr_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        self._sr_combo = QComboBox()
        self._sr_combo.setToolTip(
            "Select the target sample rate for the processed audio"
        )
        for val, name in TARGET_SR_OPTIONS:
            self._sr_combo.addItem(name, val)
        self._sr_combo.setCurrentIndex(0)
        sr_row.addWidget(sr_lbl)
        sr_row.addWidget(self._sr_combo)
        sr_row.addStretch()
        self._proc_section.add_layout(sr_row)
        self._proc_btn = QPushButton("🔧 Process audio")
        self._proc_btn.setToolTip(
            "Process the reference audio with the selected options"
        )
        self._proc_btn.setStyleSheet(_btn(C["accent"]))
        self._proc_btn.setEnabled(False)
        self._proc_btn.clicked.connect(self._process_audio)
        self._proc_section.add_widget(self._proc_btn)
        self._proc_lbl = QLabel("")
        self._proc_lbl.setStyleSheet(
            f"color:{C['text3']};font-size:10px;font-style:italic;"
        )
        self._proc_lbl.setWordWrap(True)
        self._proc_section.add_widget(self._proc_lbl)
        cloning_lay.addWidget(self._proc_section)

        whisper_section = CollapsibleSection(
            "🎤 Automatic transcription — Whisper", C["accent"]
        )
        self._whisper_section_widget = whisper_section
        mr = QHBoxLayout()
        mr.setSpacing(6)
        ml = QLabel("Model:")
        ml.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        ml.setFixedWidth(44)
        self._w_size = QComboBox()
        self._w_size.setToolTip(
            "Select the Whisper model size "
            "(larger models are more accurate but need more memory)"
        )
        for s in WHISPER_SIZES:
            self._w_size.addItem(f"{s} {WHISPER_SIZE_MB[s]}", s)
        self._w_size.setCurrentIndex(5)
        self._w_size.currentIndexChanged.connect(self._refresh_whisper_ui)
        ll = QLabel("Language:")
        ll.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        ll.setFixedWidth(56)
        self._w_lang = QComboBox()
        self._w_lang.setToolTip(
            "Language of the audio (or auto-detect) for better transcription accuracy"
        )
        for code, name in WHISPER_LANGS:
            self._w_lang.addItem(name, code)
        mr.addWidget(ml)
        mr.addWidget(self._w_size)
        mr.addSpacing(4)
        mr.addWidget(ll)
        mr.addWidget(self._w_lang)
        whisper_section.add_layout(mr)
        wr = QHBoxLayout()
        wr.setSpacing(6)
        self._w_dl_btn = QPushButton("⬇ Download Whisper")
        self._w_dl_btn.setToolTip("Download the selected Whisper model")
        self._w_dl_btn.setStyleSheet(_btn(C["accent"]))
        self._w_dl_btn.clicked.connect(self._start_whisper_download)
        self._w_tr_btn = QPushButton("🎤 Transcribe")
        self._w_tr_btn.setToolTip("Transcribe the reference audio using Whisper")
        self._w_tr_btn.setStyleSheet(_btn(C["accent"]))
        self._w_tr_btn.setEnabled(False)
        self._w_tr_btn.clicked.connect(self._transcribe)
        wr.addWidget(self._w_dl_btn)
        wr.addWidget(self._w_tr_btn)
        wr.addStretch()
        whisper_section.add_layout(wr)
        self._w_status = QLabel("")
        self._w_status.setStyleSheet(
            f"color:{C['text3']};font-size:10px;font-style:italic;"
        )
        whisper_section.add_widget(self._w_status)
        cloning_lay.addWidget(whisper_section)

        single_lay.addWidget(self._cloning_widget)

        voice_section.add_widget(self._voice_single_container)
        self._speakers_container = QWidget()
        self._speakers_container.setStyleSheet("background: transparent;")
        self._speakers_lay = QVBoxLayout(self._speakers_container)
        self._speakers_lay.setContentsMargins(0, 0, 0, 0)
        self._speakers_lay.setSpacing(10)
        self._speakers_container.setVisible(False)
        voice_section.add_widget(self._speakers_container)
        lay.addWidget(voice_section)
        self._voice_cloning_section = voice_section

        params_section = CollapsibleSection("⚙️ Generation parameters", C["accent"])
        params_inner = QWidget()
        params_inner_lay = QVBoxLayout(params_inner)
        params_inner_lay.setContentsMargins(0, 0, 0, 0)
        params_inner_lay.setSpacing(6)
        params_inner_lay.addWidget(self._make_params_grid())
        rb = QPushButton("↺ Reset to defaults")
        rb.setToolTip("Reset all generation parameters to their default values")
        rb.clicked.connect(self._reset_params)
        params_inner_lay.addWidget(rb)
        params_section.add_widget(params_inner)
        lay.addWidget(params_section)

        self._lektor_section = CollapsibleSection("🎬 Lektor Export", C["accent"])
        self._build_lektor_section()
        lay.addWidget(self._lektor_section)
        self._audiobook_section = CollapsibleSection("🎧 Audiobook Export", C["accent"])
        self._build_audiobook_section()
        lay.addWidget(self._audiobook_section)
        self._audiobook_section.setVisible(False)
        lay.addStretch()
        scroll.setWidget(content)
        outer_lay.addWidget(scroll, 1)
        outer_lay.addWidget(self._make_synth_bar())
        self._tabs.currentChanged.connect(self._on_tab_changed)
        self._on_tab_changed(self._tabs.currentIndex())
        return outer

    def _on_tab_changed(self, index: int):
        is_ebook = index == 1
        is_quick = index == 2
        self._btn_start.setVisible(not is_quick)
        self._btn_stop.setVisible(not is_quick)
        self._lektor_section.setVisible(not is_quick and not is_ebook)
        self._audiobook_section.setVisible(is_ebook)
        self._update_action_buttons()

    def _make_synth_bar(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(
            f"background:{C['surface']};border-top:1px solid {C['border']};"
        )
        lay = QHBoxLayout(w)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(8)

        self._btn_start = QPushButton("▶  Start synthesis")
        self._btn_start.setStyleSheet(SYNTH_BTN_STYLE)
        self._btn_start.setMinimumWidth(160)
        self._btn_start.setEnabled(False)
        self._btn_start.clicked.connect(self._start_synthesis)

        self._btn_stop = QPushButton("⏹  Stop")
        self._btn_stop.setStyleSheet(_btn(C["warning"]))
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_synthesis)

        self._synth_progress = QProgressBar()
        self._synth_progress.setFixedHeight(6)
        self._synth_progress.setTextVisible(False)
        self._synth_progress.setVisible(False)

        self._eta_label = QLabel()
        self._eta_label.setStyleSheet(
            f"color:{C['text2']};font-size:11px;min-width:110px;"
        )
        self._eta_label.setVisible(False)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setFixedHeight(6)
        self._progress.setTextVisible(False)
        self._progress.setVisible(False)

        lay.addWidget(self._btn_start)
        lay.addWidget(self._btn_stop)
        lay.addWidget(self._synth_progress, 1)
        lay.addWidget(self._eta_label)
        lay.addWidget(self._progress, 1)
        lay.addStretch()
        return w

    def _build_lektor_section(self):
        self._w_vocal_suppress = None
        self._pending_export: Dict = {}

        hint = QLabel(
            "Assembles synthesized fragments into a timeline WAV and mixes "
            "it with the original video track."
        )
        hint.setStyleSheet(f"color:{C['text3']};font-size:10px;")
        hint.setWordWrap(True)
        self._lektor_section.add_widget(hint)

        if not self._ffmpeg_ok:
            warn = QLabel("⚠  ffmpeg not found in PATH — video export disabled.")
            warn.setStyleSheet(f"color:{C['warning']};font-size:10px;")
            self._lektor_section.add_widget(warn)

        self._norm_check = QCheckBox("Normalize audio (ffmpeg)")
        self._norm_check.setEnabled(self._ffmpeg_ok)
        if not self._ffmpeg_ok:
            self._norm_check.setToolTip("ffmpeg not found in PATH")
        self._lektor_section.add_widget(self._norm_check)

        self._vid_row_widget = QWidget()
        self._vid_row_widget.setStyleSheet("background: transparent;")
        vid_row = QHBoxLayout(self._vid_row_widget)
        vid_row.setContentsMargins(0, 0, 0, 0)
        vid_row.setSpacing(6)
        self._vid_path_edit = QLineEdit()
        self._vid_path_edit.setPlaceholderText("No video file selected…")
        self._vid_path_edit.setFixedHeight(28)
        self._vid_path_edit.setReadOnly(True)
        self._vid_path_edit.setStyleSheet(f"""
            QLineEdit {{
                background: {C["surface"]};
                border: 1px solid {C["border"]};
                border-radius: 6px;
                color: {C["text2"]};
                padding: 4px 8px;
                font-size: 11px;
            }}
        """)
        btn_browse_vid = QPushButton("🎬  Select video file")
        btn_browse_vid.setStyleSheet(_btn(C["accent"]))
        btn_browse_vid.setFixedHeight(28)
        btn_browse_vid.setToolTip("Select the video file you want to add the lektor track to.")
        btn_browse_vid.clicked.connect(self._browse_video_file)
        vid_row.addWidget(btn_browse_vid)
        vid_row.addWidget(self._vid_path_edit, 1)
        self._lektor_section.add_widget(self._vid_row_widget)

        off_row = QHBoxLayout()
        off_row.setSpacing(8)
        off_lbl = QLabel("Offset (ms):")
        off_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        self._offset_spin = QSpinBox()
        self._offset_spin.setRange(-10000, 10000)
        self._offset_spin.setValue(0)
        self._offset_spin.setSingleStep(100)
        self._offset_spin.setMinimumWidth(100)
        self._offset_spin.setToolTip(
            "Global time shift of the entire lektor track in milliseconds.\n"
            "Use negative values to move the lektor earlier, positive to delay it.\n"
            "Example: -200 shifts all speech 200ms earlier relative to the video."
        )
        off_row.addWidget(off_lbl)
        off_row.addWidget(self._offset_spin)
        off_row.addStretch()
        self._lektor_section.add_layout(off_row)

        vol_row = QHBoxLayout()
        vol_row.setSpacing(8)

        lv_lbl = QLabel("Lektor vol:")
        lv_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        self._lektor_vol = QSlider(Qt.Orientation.Horizontal)
        self._lektor_vol.setRange(0, 200)
        self._lektor_vol.setValue(100)
        self._lektor_vol.setFixedWidth(90)
        self._lektor_vol.setToolTip(
            "Volume of the lektor (synthesized speech) track in the final video.\n"
            "100% = original level, 50% = half volume, 150% = louder.\n"
            "Increase if the lektor voice is too quiet against the original audio."
        )
        self._lektor_vol_lbl = QLabel("100%")
        self._lektor_vol_lbl.setStyleSheet(
            f"color:{C['accent']};font-size:11px;font-family:'Consolas';"
        )
        self._lektor_vol_lbl.setFixedWidth(36)
        self._lektor_vol.valueChanged.connect(
            lambda v: self._lektor_vol_lbl.setText(f"{v}%")
        )

        ov_lbl = QLabel("Original vol:")
        ov_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        self._orig_vol = QSlider(Qt.Orientation.Horizontal)
        self._orig_vol.setRange(0, 200)
        self._orig_vol.setValue(100)
        self._orig_vol.setFixedWidth(90)
        self._orig_vol.setToolTip(
            "Volume of the original video audio track (music, effects, original dialogue).\n"
            "Lower this so the lektor is clearly audible over the background.\n"
            "Example: 30% keeps ambience in the background while lektor stays in front."
        )
        self._orig_vol_lbl = QLabel("100%")
        self._orig_vol_lbl.setStyleSheet(
            f"color:{C['accent']};font-size:11px;font-family:'Consolas';"
        )
        self._orig_vol_lbl.setFixedWidth(36)
        self._orig_vol.valueChanged.connect(
            lambda v: self._orig_vol_lbl.setText(f"{v}%")
        )

        vol_row.addWidget(lv_lbl)
        vol_row.addWidget(self._lektor_vol)
        vol_row.addWidget(self._lektor_vol_lbl)
        vol_row.addSpacing(12)
        vol_row.addWidget(ov_lbl)
        vol_row.addWidget(self._orig_vol)
        vol_row.addWidget(self._orig_vol_lbl)
        vol_row.addStretch()
        self._lektor_section.add_layout(vol_row)

        fit_row = QHBoxLayout()
        fit_row.setSpacing(8)
        self._autofit_check = QCheckBox("Auto-fit to slot (atempo)")
        self._autofit_check.setChecked(False)
        self._autofit_check.setEnabled(self._ffmpeg_ok)
        self._autofit_check.setToolTip(
            "If a synthesized fragment exceeds its SRT slot by less than the threshold below,\n"
            "it will be automatically sped up using ffmpeg atempo to fit the slot.\n"
            "For larger overruns the next fragment simply waits — preserving voice quality.\n"
            "Requires ffmpeg. Works best for small overruns (up to ~2x speed). Pitch is preserved."
        )
        atempo_lbl = QLabel("Threshold (ms):")
        atempo_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        self._atempo_threshold = QSpinBox()
        self._atempo_threshold.setRange(50, 2000)
        self._atempo_threshold.setValue(300)
        self._atempo_threshold.setSingleStep(50)
        self._atempo_threshold.setMinimumWidth(100)
        self._atempo_threshold.setToolTip(
            "Maximum overrun in milliseconds that will be corrected with atempo.\n"
            "If the fragment exceeds its slot by more than this value,\n"
            "atempo is skipped and the next fragment waits instead."
        )
        self._autofit_check.toggled.connect(
            lambda checked: self._atempo_threshold.setEnabled(checked and self._ffmpeg_ok)
        )
        self._atempo_threshold.setEnabled(False)
        fit_row.addWidget(self._autofit_check)
        fit_row.addSpacing(4)
        fit_row.addWidget(atempo_lbl)
        fit_row.addWidget(self._atempo_threshold)
        fit_row.addStretch()
        self._lektor_section.add_layout(fit_row)

        self._duck_check = QCheckBox("Ducking (sidechaincompress)")
        self._duck_check.setToolTip(
            "Automatically lowers the original audio volume whenever the lektor is speaking.\n"
            "This is the classic TV dubbing effect — background audio dips under the voice\n"
            "and returns to normal volume during pauses in speech."
        )
        self._lektor_section.add_widget(self._duck_check)

        vsup_row = QHBoxLayout()
        vsup_row.setSpacing(8)
        self._vocal_suppress_check = QCheckBox("Suppress original vocals")
        self._vocal_suppress_check.setChecked(False)
        self._vocal_suppress_check.setEnabled(self._ffmpeg_ok)
        self._vocal_suppress_check.setToolTip(
            "Separates the video's original audio into vocals and background (music/effects)\n"
            "using Demucs htdemucs, then lets you independently control the volume of each.\n\n"
            "Use this to lower or silence the original speaker's voice without affecting\n"
            "ambient sound, music, or sound effects — while keeping the lektor clearly audible.\n\n"
            "The 'Original vol' slider above controls the background (non-vocal) track volume.\n"
            "The 'Vocal vol' spinbox below controls how loud the suppressed original voice is.\n\n"
            "Requires demucs installed. Processing may take several minutes."
        )

        vsup_vol_lbl = QLabel("Vocal vol:")
        vsup_vol_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        self._vocal_suppress_spin = QSpinBox()
        self._vocal_suppress_spin.setRange(0, 100)
        self._vocal_suppress_spin.setValue(20)
        self._vocal_suppress_spin.setSingleStep(5)
        self._vocal_suppress_spin.setMinimumWidth(70)
        self._vocal_suppress_spin.setEnabled(False)
        self._vocal_suppress_spin.setToolTip(
            "Volume of the original video vocals after suppression (0–100%).\n"
            "0% = completely silenced, 100% = original volume (no suppression).\n"
        )
        vsup_pct_lbl = QLabel("%")
        vsup_pct_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")

        self._vocal_suppress_check.toggled.connect(
            lambda checked: self._vocal_suppress_spin.setEnabled(
                checked and self._ffmpeg_ok
            )
        )

        vsup_row.addWidget(self._vocal_suppress_check)
        vsup_row.addSpacing(6)
        vsup_row.addWidget(vsup_vol_lbl)
        vsup_row.addWidget(self._vocal_suppress_spin)
        vsup_row.addWidget(vsup_pct_lbl)
        vsup_row.addStretch()
        self._lektor_section.add_layout(vsup_row)

        self._keep_original_track_check = QCheckBox("Keep original audio as separate track")
        self._keep_original_track_check.setChecked(False)
        self._keep_original_track_check.setEnabled(self._ffmpeg_ok)
        self._keep_original_track_check.setToolTip(
            "When enabled, the output video will contain TWO audio tracks instead of one:\n"
            "  Track 1 — the untouched original audio from the source video\n"
            "  Track 2 — the original audio mixed together with the lektor dubbing\n\n"
            "This lets you switch between the original and dubbed audio in your media\n"
            "player (e.g. VLC: Audio → Audio Track). The original track is set as default.\n\n"
            "When disabled (default), the output contains a single mixed audio track only\n"
            "and the original audio cannot be recovered from the exported file.\n\n"
            "Works best with MKV and MP4 containers. AVI has limited multi-track support."
        )
        self._lektor_section.add_widget(self._keep_original_track_check)

        self._dubbed_lang_widget = QWidget()
        self._dubbed_lang_widget.setStyleSheet("background: transparent;")
        dubbed_lang_inner = QHBoxLayout(self._dubbed_lang_widget)
        dubbed_lang_inner.setContentsMargins(0, 0, 0, 0)
        dubbed_lang_inner.setSpacing(8)
        dubbed_lang_lbl = QLabel("Dubbing language (ISO 639-2):")
        dubbed_lang_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        self._dubbed_lang_edit = QLineEdit()
        self._dubbed_lang_edit.setPlaceholderText("e.g. eng")
        self._dubbed_lang_edit.setMaxLength(3)
        self._dubbed_lang_edit.setFixedWidth(70)
        self._dubbed_lang_edit.setFixedHeight(24)
        self._dubbed_lang_edit.setStyleSheet(f"""
            QLineEdit {{
                background: {C["surface"]};
                border: 1px solid {C["border"]};
                border-radius: 4px;
                color: {C["text"]};
                padding: 2px 6px;
                font-size: 11px;
                font-family: 'Consolas';
            }}
        """)
        self._dubbed_lang_edit.setToolTip(
            "Three-letter ISO 639-2 language code for the dubbing audio track.\n\n"
            "Media servers such as Plex, Jellyfin and Kodi read this tag to automatically\n"
            "select the correct audio track for users whose preferred language matches.\n\n"
            "Common codes:\n"
            "  eng — English        pol — Polish         deu — German\n"
            "  fra — French         spa — Spanish        ita — Italian\n"
            "  rus — Russian        por — Portuguese     jpn — Japanese\n"
            "  kor — Korean         zho — Chinese        ara — Arabic\n"
            "  ukr — Ukrainian      hin — Hindi          tur — Turkish\n\n"
            "Full list: https://www.loc.gov/standards/iso639-2/php/code_list.php\n"
            "If left blank or invalid, 'und' (undetermined) will be written."
        )
        dubbed_lang_inner.addWidget(dubbed_lang_lbl)
        dubbed_lang_inner.addWidget(self._dubbed_lang_edit)
        dubbed_lang_inner.addStretch()
        self._dubbed_lang_widget.setVisible(False)
        self._keep_original_track_check.toggled.connect(
            lambda checked: self._dubbed_lang_widget.setVisible(checked)
        )
        self._lektor_section.add_widget(self._dubbed_lang_widget)

        fmt_row = QHBoxLayout()
        fmt_row.setSpacing(8)
        fmt_lbl = QLabel("Format:")
        fmt_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        self._lektor_vid_fmt_combo = QComboBox()
        self._lektor_vid_fmt_combo.addItem("Follow input format", "auto")
        self._lektor_vid_fmt_combo.addItem("MP4", "mp4")
        self._lektor_vid_fmt_combo.addItem("MKV", "mkv")
        self._lektor_vid_fmt_combo.addItem("AVI", "avi")
        self._lektor_vid_fmt_combo.addItem("MOV", "mov")
        self._lektor_vid_fmt_combo.addItem("WebM", "webm")
        fmt_row.addWidget(fmt_lbl)
        fmt_row.addWidget(self._lektor_vid_fmt_combo)
        fmt_row.addStretch()
        self._lektor_section.add_layout(fmt_row)

        exp_row = QHBoxLayout()
        self._export_btn = QPushButton("🎬  Export video with lektor")
        self._export_btn.setStyleSheet(_btn(C["accent"]))
        self._export_btn.setEnabled(self._ffmpeg_ok)
        self._export_btn.setToolTip(
            "Mix the lektor audio track with the selected video file using ffmpeg\n"
            "and save the result as a new video file."
        )
        self._export_btn.clicked.connect(self._export_lektor_video)
        exp_row.addWidget(self._export_btn)
        exp_row.addStretch()
        self._lektor_section.add_layout(exp_row)

        self._lektor_status = QLabel("")
        self._lektor_status.setStyleSheet(
            f"color:{C['text3']};font-size:10px;font-style:italic;"
        )
        self._lektor_section.add_widget(self._lektor_status)

    def _build_audiobook_section(self):
        hint = QLabel(
            "Concatenates all synthesized fragments into a single audiobook file."
        )
        hint.setStyleSheet(f"color:{C['text3']};font-size:10px;")
        hint.setWordWrap(True)
        self._audiobook_section.add_widget(hint)

        out_row = QHBoxLayout()
        out_row.setSpacing(6)
        out_lbl = QLabel("Output:")
        out_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        out_lbl.setFixedWidth(46)
        self._audiobook_output_edit = QLineEdit(self._ebook_output_dir)
        self._audiobook_output_edit.setFixedHeight(26)
        self._audiobook_output_edit.textChanged.connect(
            lambda t: setattr(self, "_ebook_output_dir", t)
        )
        btn_browse_ab = QPushButton("…")
        btn_browse_ab.setFixedSize(28, 26)
        btn_browse_ab.clicked.connect(self._browse_audiobook_output)
        out_row.addWidget(out_lbl)
        out_row.addWidget(self._audiobook_output_edit)
        out_row.addWidget(btn_browse_ab)
        self._audiobook_section.add_layout(out_row)

        opt_row = QHBoxLayout()
        opt_row.setSpacing(8)
        sil_lbl = QLabel("Silence between fragments (ms):")
        sil_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        self._audiobook_silence_spin = QSpinBox()
        self._audiobook_silence_spin.setRange(0, 5000)
        self._audiobook_silence_spin.setValue(500)
        self._audiobook_silence_spin.setSingleStep(100)
        self._audiobook_silence_spin.setMinimumWidth(90)
        opt_row.addWidget(sil_lbl)
        opt_row.addWidget(self._audiobook_silence_spin)
        opt_row.addStretch()
        self._audiobook_section.add_layout(opt_row)

        fmt_row = QHBoxLayout()
        fmt_row.setSpacing(8)
        fmt_lbl = QLabel("Format:")
        fmt_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        self._audiobook_fmt_combo = QComboBox()
        self._audiobook_fmt_combo.addItem("WAV", "wav")
        self._audiobook_fmt_combo.addItem("MP3  (requires ffmpeg)", "mp3")
        self._audiobook_fmt_combo.addItem("FLAC", "flac")
        self._audiobook_fmt_combo.addItem("OGG  (requires ffmpeg)", "ogg")
        self._audiobook_fmt_combo.addItem("OPUS  (requires ffmpeg)", "opus")
        fmt_row.addWidget(fmt_lbl)
        fmt_row.addWidget(self._audiobook_fmt_combo)
        fmt_row.addStretch()
        self._audiobook_section.add_layout(fmt_row)

        self._audiobook_norm_check = QCheckBox("Normalize audio (ffmpeg)")
        self._audiobook_norm_check.setEnabled(self._ffmpeg_ok)
        if not self._ffmpeg_ok:
            self._audiobook_norm_check.setToolTip("ffmpeg not found in PATH")
        self._audiobook_section.add_widget(self._audiobook_norm_check)

        exp_row = QHBoxLayout()
        self._audiobook_export_btn = QPushButton("🎧  Export audiobook")
        self._audiobook_export_btn.setStyleSheet(_btn(C["accent"]))
        self._audiobook_export_btn.clicked.connect(self._export_audiobook)
        exp_row.addWidget(self._audiobook_export_btn)
        exp_row.addStretch()
        self._audiobook_section.add_layout(exp_row)

        self._audiobook_status_lbl = QLabel("")
        self._audiobook_status_lbl.setStyleSheet(
            f"color:{C['text3']};font-size:10px;font-style:italic;"
        )
        self._audiobook_section.add_widget(self._audiobook_status_lbl)
 
    def _browse_audiobook_output(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select output folder",
            _get_last_dir("output", str(OUTPUTS_DIR))
        )
        if d:
            _set_last_dir("output", d)
            self._audiobook_output_edit.setText(d)
            self._ebook_output_dir = d

    def _update_preview_btn_state(self):
        if not hasattr(self, '_preview_audiobook_btn'):
            return
        if not hasattr(self, '_ebook_chapter_item') or not self._ebook_chapter_item:
            self._preview_audiobook_btn.setEnabled(False)
            return

        visible_items = []
        for i in range(self._ebook_chapter_item.childCount()):
            child = self._ebook_chapter_item.child(i)
            if not child.isHidden():
                visible_items.append(child)

        checked_positions = [
            i for i, child in enumerate(visible_items)
            if child.checkState(COL_STATUS) == Qt.CheckState.Checked
        ]

        if not checked_positions:
            self._preview_audiobook_btn.setEnabled(False)
            return

        first, last = checked_positions[0], checked_positions[-1]
        if last - first + 1 != len(checked_positions):
            self._preview_audiobook_btn.setEnabled(False)
            return

        has_done = False
        for i in checked_positions:
            idx = visible_items[i].data(COL_STATUS, Qt.ItemDataRole.UserRole)
            frag = next((f for f in self._ebook_fragments if f['index'] == idx), None)
            if frag and frag.get('status') == 'done' and frag.get('output_path') and os.path.exists(frag['output_path']):
                has_done = True
                break

        self._preview_audiobook_btn.setEnabled(has_done)

    def _ebook_fragment_silence_arrays(
        self, frag: Dict, audio: np.ndarray, sr: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        pre_ms = int(frag.get('pre_silence_ms') or 0)
        pre_arr = (
            np.zeros(int(sr * pre_ms / 1000), dtype=np.float32)
            if pre_ms > 0 else np.zeros(0, dtype=np.float32)
        )

        post_arr = np.zeros(0, dtype=np.float32)
        target_ms = frag.get('target_duration_ms')
        if target_ms is not None:
            audio_dur_ms = int(round(len(audio) / sr * 1000))
            extra_ms = max(0, int(target_ms) - audio_dur_ms)
            if extra_ms > 0:
                post_arr = np.zeros(int(sr * extra_ms / 1000), dtype=np.float32)

        return pre_arr, post_arr

    def _preview_selected_fragments(self):
        if not hasattr(self, '_ebook_chapter_item') or not self._ebook_chapter_item:
            return

        visible_items = []
        for i in range(self._ebook_chapter_item.childCount()):
            child = self._ebook_chapter_item.child(i)
            if not child.isHidden():
                visible_items.append(child)

        checked_indices = []
        for child in visible_items:
            if child.checkState(COL_STATUS) == Qt.CheckState.Checked:
                idx = child.data(COL_STATUS, Qt.ItemDataRole.UserRole)
                if idx is not None:
                    checked_indices.append(idx)

        done_frags = [
            f for f in self._ebook_fragments
            if f['index'] in checked_indices
            and f.get('status') == 'done'
            and f.get('output_path')
            and os.path.exists(f['output_path'])
        ]
        done_frags.sort(key=lambda f: f['index'])

        if not done_frags:
            QMessageBox.warning(
                self, "Nothing to preview",
                "No synthesized fragments in selection. Run synthesis first."
            )
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Preview selected fragments")
        dlg.resize(380, 150)
        dlg.setStyleSheet(self.styleSheet())
        lay = QVBoxLayout(dlg)
        lay.setSpacing(10)
        lay.setContentsMargins(14, 14, 14, 10)

        info = QLabel(f"Stitch and preview {len(done_frags)} synthesized fragment(s).")
        info.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        lay.addWidget(info)

        norm_chk = QCheckBox("Normalize audio (ffmpeg)")
        norm_chk.setEnabled(self._ffmpeg_ok)
        if not self._ffmpeg_ok:
            norm_chk.setToolTip("ffmpeg not found in PATH")
        norm_init = (
            self._audiobook_norm_check.isChecked()
            if hasattr(self, '_audiobook_norm_check')
            else False
        )
        norm_chk.setChecked(norm_init)
        lay.addWidget(norm_chk)

        sil_row = QHBoxLayout()
        sil_lbl = QLabel("Global silence between fragments (ms):")
        sil_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        sil_spin = QSpinBox()
        sil_spin.setRange(0, 5000)
        sil_spin.setSingleStep(100)
        sil_spin.setFixedWidth(90)
        sil_spin.setValue(
            self._audiobook_silence_spin.value()
            if hasattr(self, '_audiobook_silence_spin') else 500
        )
        sil_row.addWidget(sil_lbl)
        sil_row.addWidget(sil_spin)
        sil_row.addStretch()
        lay.addLayout(sil_row)

        hint = QLabel("Per-fragment extra silence (set via 'Edit duration') is applied additionally.")
        hint.setStyleSheet(f"color:{C['text3']};font-size:10px;font-style:italic;")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        normalize  = norm_chk.isChecked()
        silence_ms = sil_spin.value()

        target_sr = None
        for frag in done_frags:
            try:
                info_sf = sf.info(frag['output_path'])
                target_sr = int(info_sf.samplerate)
                break
            except Exception:
                pass
        if target_sr is None:
            target_sr = 44100

        silence_samples = int(target_sr * silence_ms / 1000)
        global_silence  = np.zeros(silence_samples, dtype=np.float32)

        self._set_status("Building preview…")
        QApplication.processEvents()

        parts: List[np.ndarray] = []
        for frag in done_frags:
            try:
                audio, sr = sf.read(frag['output_path'], dtype='float32')
                sr = int(sr)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                audio = np.ascontiguousarray(audio, dtype=np.float32)
                if sr != target_sr:
                    tensor = torch.from_numpy(audio).unsqueeze(0)
                    tensor = TAF.resample(tensor, sr, target_sr)
                    audio  = np.ascontiguousarray(tensor.squeeze(0).numpy(), dtype=np.float32)

                pre_arr, post_arr = self._ebook_fragment_silence_arrays(frag, audio, target_sr)
                if len(pre_arr):
                    parts.append(pre_arr)
                parts.append(audio)
                if len(post_arr):
                    parts.append(post_arr)
                if silence_ms > 0:
                    parts.append(global_silence.copy())
            except Exception as e:
                logger.warning(f"Preview: skipping fragment {frag['index']}: {e}")

        if not parts:
            QMessageBox.warning(self, "Preview failed", "Could not load any audio fragments.")
            return

        combined = np.concatenate(parts).astype(np.float32)

        if normalize:
            tmp_wav  = tempfile.mktemp(suffix="_preview.wav")
            norm_wav = tempfile.mktemp(suffix="_preview_norm.wav")
            try:
                sf.write(tmp_wav, combined, target_sr, subtype="PCM_16")
                if self._normalize_ffmpeg(tmp_wav, norm_wav):
                    norm_audio, norm_sr = sf.read(norm_wav, dtype="float32")
                    combined = np.ascontiguousarray(norm_audio, dtype=np.float32)
                    if int(norm_sr) != target_sr:
                        tensor   = torch.from_numpy(combined).unsqueeze(0)
                        tensor   = TAF.resample(tensor, int(norm_sr), target_sr)
                        combined = np.ascontiguousarray(tensor.squeeze(0).numpy(), dtype=np.float32)
                else:
                    logger.warning("Preview normalization failed, using unnormalized audio")
            except Exception as e:
                logger.warning(f"Preview normalization error: {e}")
            finally:
                for p in (tmp_wav, norm_wav):
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except Exception:
                        pass

        self._load_audio_to_player(combined, target_sr, fragment_idx=None)
        dur = len(combined) / target_sr
        self._set_status(
            f"Preview ready — {len(done_frags)} fragments | {_fmt(dur)}", C["success"]
        )
 
    def _export_audiobook(self):
        if not self._ebook_fragments:
            QMessageBox.warning(self, "No ebook loaded", "Load an EPUB or TXT first.")
            return

        done_frags = [
            f for f in self._ebook_fragments
            if f.get('status') == 'done'
            and f.get('output_path')
            and os.path.exists(f.get('output_path', ''))
        ]
        if not done_frags:
            QMessageBox.warning(self, "No audio", "No fragments have been synthesized yet.")
            return

        done_frags = sorted(done_frags, key=lambda f: f['index'])

        fmt = self._audiobook_fmt_combo.currentData() or 'wav'
        silence_ms = self._audiobook_silence_spin.value()

        base_name = Path(self._epub_path).stem if self._epub_path else "audiobook"
        default_path = str(
            Path(_get_last_dir("output", self._ebook_output_dir)) / f"{base_name}.{fmt}"
        )

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save audiobook",
            default_path,
            f"Audio {fmt.upper()} (*.{fmt});;All files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(f".{fmt}"):
            path += f".{fmt}"

        _set_last_dir("output", path)

        self._set_status("Exporting audiobook...")
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)

        try:
            all_audio = []
            sr = 44100
            for i, frag in enumerate(done_frags):
                audio, frag_sr = sf.read(frag['output_path'], dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if frag_sr != sr:
                    audio = np.ascontiguousarray(
                        torchaudio.functional.resample(
                            torch.from_numpy(audio), frag_sr, sr
                        ).numpy(),
                        dtype=np.float32,
                    )

                pre_arr, post_arr = self._ebook_fragment_silence_arrays(frag, audio, sr)
                if len(pre_arr):
                    all_audio.append(pre_arr)
                all_audio.append(audio)
                if len(post_arr):
                    all_audio.append(post_arr)

                if i < len(done_frags) - 1 and silence_ms > 0:
                    silence = np.zeros(int(sr * silence_ms / 1000), dtype=np.float32)
                    all_audio.append(silence)

            full_audio = np.concatenate(all_audio)

            if fmt == "wav":
                sf.write(path, full_audio, sr, subtype="PCM_16")
            else:
                tmp_wav = tempfile.mktemp(suffix=".wav")
                sf.write(tmp_wav, full_audio, sr, subtype="PCM_16")
                try:
                    from pydub import AudioSegment
                    AudioSegment.from_wav(tmp_wav).export(path, format="mp3", bitrate="192k")
                except ImportError:
                    path = path.replace(".mp3", ".wav")
                    sf.write(path, full_audio, sr, subtype="PCM_16")
                finally:
                    try:
                        os.unlink(tmp_wav)
                    except Exception:
                        pass

            self._progress.setVisible(False)
            self._set_status(f"Audiobook saved: {path}", C["success"])

            reply = QMessageBox.information(
                self, "Export complete",
                f"Audiobook saved successfully:\n{path}\n\nOpen output folder?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                _open_file(os.path.dirname(path))

        except Exception:
            self._progress.setVisible(False)
            self._on_error("Audiobook export error", traceback.format_exc())
        
    def _make_params_grid(self) -> QWidget:
        container = QWidget()
        layout    = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._param_widgets:     Dict[str, QWidget] = {}
        self._param_row_widgets: Dict[str, QWidget] = {}

        seen_keys: Dict[str, Dict] = {}
        for cls in _ACTIVE_BACKEND_CLASSES:
            sentinel = object.__new__(cls)
            try:
                params = sentinel.generation_params
            except Exception:
                continue
            for p in params:
                if p["key"] not in seen_keys:
                    seen_keys[p["key"]] = p

        for key, p in seen_keys.items():
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(8)

            lbl = QLabel(p["label"])
            lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
            lbl.setFixedWidth(140)
            if p.get("tip"):
                lbl.setToolTip(p["tip"])
            row_l.addWidget(lbl)

            if p["type"] == "slider":
                sl = QSlider(Qt.Orientation.Horizontal)
                sl.setRange(int(p["min"] * 100), int(p["max"] * 100))
                sl.setValue(int(p["default"] * 100))
                if p.get("tip"):
                    sl.setToolTip(p["tip"])
                vl = QLabel(f"{p['default']:.2f}")
                vl.setFixedWidth(38)
                vl.setStyleSheet(
                    f"color:{C['accent']};font-size:11px;font-family:'Consolas';"
                )
                sl.valueChanged.connect(lambda v, _vl=vl: _vl.setText(f"{v / 100:.2f}"))
                row_l.addWidget(sl, 1)
                row_l.addWidget(vl)
                self._param_widgets[key] = sl
            elif p["type"] == "combo":
                cb = QComboBox()
                for val, label in p["options"]:
                    cb.addItem(label, val)
                default_idx = next(
                    (i for i, (v, _) in enumerate(p["options"]) if v == p["default"]),
                    0,
                )
                cb.setCurrentIndex(default_idx)
                if p.get("tip"):
                    cb.setToolTip(p["tip"])
                row_l.addWidget(cb, 1)
                self._param_widgets[key] = cb
            elif p["type"] == "dspinbox":
                dsp = QDoubleSpinBox()
                dsp.setRange(p["min"], p["max"])
                dsp.setValue(p["default"])
                dsp.setSingleStep(p.get("step", 0.1))
                dsp.setDecimals(2)
                dsp.setMinimumWidth(100)
                if p.get("tip"):
                    dsp.setToolTip(p["tip"])
                row_l.addWidget(dsp)
                row_l.addStretch(1)
                self._param_widgets[key] = dsp
            else:
                sp = QSpinBox()
                sp.setRange(p["min"], p["max"])
                sp.setValue(p["default"])
                sp.setSingleStep(p.get("step", 1))
                sp.setMinimumWidth(100)
                if p.get("tip"):
                    sp.setToolTip(p["tip"])
                row_l.addWidget(sp)
                row_l.addStretch(1)
                self._param_widgets[key] = sp

            layout.addWidget(row_w)
            self._param_row_widgets[key] = row_w

        self._token_hint_lbl = QLabel("")
        self._token_hint_lbl.setStyleSheet(
            f"color:{C['text3']};font-size:10px;font-style:italic;"
        )
        self._token_hint_lbl.setWordWrap(True)
        self._token_hint_row = QWidget()
        th_lay = QHBoxLayout(self._token_hint_row)
        th_lay.setContentsMargins(0, 0, 0, 0)
        th_lay.addSpacing(148)
        th_lay.addWidget(self._token_hint_lbl, 1)
        self._token_hint_row.setVisible(False)
        layout.addWidget(self._token_hint_row)

        max_tok_w = self._param_widgets.get("max_new_tokens")
        tgt_tok_w = self._param_widgets.get("target_tokens")
        if isinstance(max_tok_w, QSpinBox) and isinstance(tgt_tok_w, QSpinBox):
            max_tok_w.valueChanged.connect(self._update_token_hint)
            tgt_tok_w.valueChanged.connect(self._update_token_hint)

        return container

    def _update_params_visibility(self, model_id: str):
        if not hasattr(self, "_param_row_widgets"):
            return
        params = self._backend.generation_params if self._backend else []
        active_keys = {p["key"] for p in params}
        for key, row_w in self._param_row_widgets.items():
            row_w.setVisible(key in active_keys)
        show_hint = (
            "max_new_tokens" in active_keys
            and "target_tokens" in active_keys
            and hasattr(self, "_token_hint_row")
        )
        if hasattr(self, "_token_hint_row"):
            self._token_hint_row.setVisible(show_hint)
        if show_hint:
            self._update_token_hint()

    def _update_token_hint(self):
        if not hasattr(self, "_token_hint_lbl"):
            return
        max_tok_w = self._param_widgets.get("max_new_tokens")
        tgt_tok_w = self._param_widgets.get("target_tokens")
        if not isinstance(max_tok_w, QSpinBox) or not isinstance(tgt_tok_w, QSpinBox):
            return
        max_t = max_tok_w.value()
        tgt_t = tgt_tok_w.value()
        max_s = max_t / 12.5
        parts = [f"Max output: {max_t} tokens (~{max_s:.0f} s)"]
        if tgt_t > 0:
            tgt_s = tgt_t / 12.5
            parts.append(f"Target: {tgt_t} tokens (~{tgt_s:.0f} s)")
        self._token_hint_lbl.setText("  ·  ".join(parts))

    def _make_player_bar(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f"background:{C['player']};border-top:1px solid {C['border']};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(16, 6, 16, 8)
        lay.setSpacing(4)

        self._video_player = VideoAudioPlayer()
        lay.addWidget(self._video_player)

        self._wave_out = SelectionWaveformWidget(h=64)
        self._wave_out.delete_requested.connect(self._delete_selected_audio_segment)
        self._wave_out.mute_requested.connect(self._mute_selected_audio_segment)
        self._wave_out.seeked.connect(self._seek)
        lay.addWidget(self._wave_out)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(6)

        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setFixedHeight(32)
        self._play_btn.setMinimumWidth(80)
        self._play_btn.setEnabled(False)
        self._play_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._play_btn.clicked.connect(self._toggle_play)

        self._stop_btn = QPushButton("⏹  Stop")
        self._stop_btn.setFixedHeight(32)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._stop_btn.clicked.connect(self._stop_play)

        self._time_lbl = QLabel("0:00 / 0:00")
        self._time_lbl.setStyleSheet(
            f"color:{C['text2']};font-size:11px;font-family:'Consolas',monospace;min-width:80px;"
        )

        vol_icon = QLabel("🔊")
        vol_icon.setStyleSheet("font-size:14px;background:transparent;")
        vol_icon.setFixedWidth(24)
        vol_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._vol = QSlider(Qt.Orientation.Horizontal)
        self._vol.setRange(0, 100)
        self._vol.setValue(80)
        self._vol.setFixedWidth(90)
        self._vol.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        bottom_row.addWidget(self._play_btn)
        bottom_row.addWidget(self._stop_btn)
        bottom_row.addSpacing(4)
        bottom_row.addWidget(self._time_lbl)
        bottom_row.addSpacing(8)
        bottom_row.addWidget(vol_icon)
        bottom_row.addWidget(self._vol)

        bottom_row.addStretch()

        self._trim_preview_lbl = QLabel("")
        self._trim_preview_lbl.setFixedWidth(230)
        self._trim_preview_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._trim_preview_lbl.setStyleSheet(
            f"color:{C['text2']};font-size:10px;font-family:'Consolas',monospace;"
        )

        trim_lbl = QLabel("Trim aggressiveness:")
        trim_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")

        self._trim_slider = QSlider(Qt.Orientation.Horizontal)
        self._trim_slider.setRange(0, 300)
        self._trim_slider.setValue(0)
        self._trim_slider.setFixedWidth(110)
        self._trim_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._trim_input = QDoubleSpinBox()
        self._trim_input.setRange(0.0, 3.0)
        self._trim_input.setSingleStep(0.1)
        self._trim_input.setDecimals(2)
        self._trim_input.setValue(0.0)
        self._trim_input.setFixedWidth(70)

        self._trim_apply_all_btn = QPushButton("Trim all selected")
        self._trim_apply_all_btn.setFixedHeight(28)
        self._trim_apply_all_btn.setEnabled(False)
        self._trim_apply_all_btn.setStyleSheet(_btn(C["warning"]))
        self._trim_apply_all_btn.clicked.connect(self._apply_trim_to_selected)

        def slider_changed(val: int):
            v = val / 100.0
            self._trim_input.blockSignals(True)
            self._trim_input.setValue(v)
            self._trim_input.blockSignals(False)
            self._on_trim_slider_changed(val)

        def input_changed(val: float):
            v = int(val * 100)
            self._trim_slider.blockSignals(True)
            self._trim_slider.setValue(v)
            self._trim_slider.blockSignals(False)
            self._on_trim_slider_changed(v)

        self._trim_slider.valueChanged.connect(slider_changed)
        self._trim_input.valueChanged.connect(input_changed)

        bottom_row.addWidget(self._trim_preview_lbl)
        bottom_row.addSpacing(8)
        bottom_row.addWidget(trim_lbl)
        bottom_row.addWidget(self._trim_slider)
        bottom_row.addWidget(self._trim_input)
        bottom_row.addSpacing(8)
        bottom_row.addWidget(self._trim_apply_all_btn)

        lay.addLayout(bottom_row)
        return w
        
    def _model_downloaded(self) -> bool:
        if not self._backend:
            return False
        return self._backend.is_available()

    def _refresh_fish_ui(self):
        dl     = self._model_downloaded()
        loaded = self._backend.is_loaded if self._backend else False
        self._dl_fish_btn.setVisible(not dl)
        self._load_btn.setVisible(dl and not loaded)
        self._load_btn.setEnabled(True)
        self._load_btn.setText("⚡  Load model")
        self._unload_btn.setVisible(loaded)
        if loaded:
            self._model_lbl.setText("● Model loaded")
            self._model_lbl.setStyleSheet(
                f"color:{C['success']};font-size:11px;background:transparent;border:none;"
            )
        elif dl:
            self._model_lbl.setText("● Model downloaded")
            self._model_lbl.setStyleSheet(
                f"color:{C['accent']};font-size:11px;background:transparent;border:none;"
            )
        else:
            self._model_lbl.setText("● Model not downloaded")
            self._model_lbl.setStyleSheet(
                f"color:{C['warning']};font-size:11px;background:transparent;border:none;"
            )
        self._update_action_buttons()

    def _refresh_whisper_ui(self):
        if not self._whisper_backend:
            return
        size = self._w_size.currentData()
        dl   = self._whisper_backend.is_downloaded(size)
        self._w_dl_btn.setVisible(not dl)
        if dl:
            self._w_status.setText(f"✓  Whisper {size} ready")
            self._w_status.setStyleSheet(f"color:{C['success']};font-size:10px;")
        else:
            self._w_status.setText(f"Not downloaded — {WHISPER_SIZE_MB.get(size, '')}")
            self._w_status.setStyleSheet(f"color:{C['text3']};font-size:10px;font-style:italic;")
        has_audio = bool(self._drop.file_path)
        self._w_tr_btn.setEnabled(dl and has_audio)
        
    def _update_whisper_visibility(self):
        if not hasattr(self, "_whisper_section_widget"):
            return
        incompatible = False
        if self._backend is not None:
            try:
                incompatible = self._backend.whisper_incompatible
            except Exception:
                pass
        self._whisper_section_widget.setVisible(not incompatible)

    def _update_dubbing_visibility(self):
        if not hasattr(self, "_btn_dubbing"):
            return
        srt_loaded = bool(self._fragments)
        incompatible = self._is_pyannote_incompatible()
        visible = srt_loaded and not incompatible
        self._btn_dubbing.setVisible(visible)
        self._btn_dubbing.setEnabled(visible)

    def _update_device_label(self):
        if torch.cuda.is_available():
            n = torch.cuda.get_device_name(0)
            m = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
            self._device_lbl.setText(f"  🟢 CUDA — {n} ({m}GB)")
            self._device_lbl.setStyleSheet(f"color:{C['success']};")
        else:
            self._device_lbl.setText("  🟡 CPU (slower)")
            self._device_lbl.setStyleSheet(f"color:{C['warning']};")

    def _update_action_buttons(self):
        model_ok = self._backend.is_loaded if self._backend else False
        tab = self._tabs.currentIndex() if hasattr(self, '_tabs') else 0
        if tab == 1:
            file_ok = bool(self._ebook_fragments)
        else:
            file_ok = bool(self._fragments)
        self._btn_start.setEnabled(model_ok and file_ok and not self._is_running)
        self._btn_stop.setEnabled(self._is_running)
        self._quick_btn.setEnabled(model_ok)

    def _set_status(self, msg: str, color=None):
        self._status_lbl.setText(msg)
        self._status_lbl.setStyleSheet(f"color:{color or C['text2']};")

    def _reset_params(self):
        params = self._backend.generation_params if self._backend else []
        for p in params:
            widget = self._param_widgets.get(p["key"]) if hasattr(self, "_param_widgets") else None
            if widget is None:
                continue
            if isinstance(widget, QSlider):
                widget.setValue(int(p["default"] * 100))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(p["default"]))
            elif isinstance(widget, QSpinBox):
                widget.setValue(p["default"])
            elif isinstance(widget, QComboBox):
                options = p.get("options", [])
                idx = next(
                    (i for i, (v, _) in enumerate(options) if v == p["default"]), 0
                )
                widget.setCurrentIndex(idx)

    def _get_generation_settings(self) -> Dict:
        params = self._backend.generation_params if self._backend else []
        active_keys = {p["key"] for p in params}
        result: Dict = {}
        if not hasattr(self, "_param_widgets"):
            return result
        for key, widget in self._param_widgets.items():
            if key not in active_keys:
                continue
            if isinstance(widget, QSlider):
                result[key] = widget.value() / 100
            elif isinstance(widget, QDoubleSpinBox):
                result[key] = widget.value()
            elif isinstance(widget, QSpinBox):
                result[key] = widget.value()
            elif isinstance(widget, QComboBox):
                result[key] = widget.currentData()
        if self._is_omnivoice_active():
            if hasattr(self, "_voice_mode_combo"):
                result["voice_mode"] = self._voice_mode_combo.currentData() or "cloning"
            if hasattr(self, "_instruct_text"):
                result["instruct"] = self._instruct_text.toPlainText().strip()
        return result

    def _is_moss_active(self) -> bool:
        if not hasattr(self, "_model_combo"):
            return False
        model_id = self._model_combo.currentData()
        return model_id is not None and model_id.startswith("moss_tts")

    def _is_pyannote_incompatible(self) -> bool:
        if self._backend is None:
            return False
        try:
            return self._backend.pyannote_incompatible
        except Exception:
            return False

    def _update_ref_text_visibility_for_model(self, model_id: str):
        is_moss = model_id.startswith("moss_tts")
        hide_ref_text = (
            is_moss
            or model_id == "voxcpm2-voicedesign"
            or model_id == "voxcpm2-voiceclone"
            or model_id.startswith("supertonic_")
            or model_id == "piper"
        )
        if hasattr(self, "_whisper_ref_text_label"):
            self._whisper_ref_text_label.setVisible(not hide_ref_text)
        if hasattr(self, "_ref_text"):
            self._ref_text.setVisible(not hide_ref_text)
        if hasattr(self, "_w_tr_btn"):
            self._w_tr_btn.setVisible(not hide_ref_text)

    def _on_model_changed(self):
        if not hasattr(self, "_model_combo"):
            return
        try:
            model_id       = self._model_combo.currentData()
            self._backend  = create_backend(model_id)
            self._refresh_fish_ui()
            self._update_params_visibility(model_id)
            self._update_whisper_visibility()
            self._update_voice_section_for_model(model_id)
            self._update_ref_text_visibility_for_model(model_id)
            self._update_dubbing_visibility()
            self._set_status(
                f"{self._backend.display_name} selected — click 'Load model'."
            )
        except Exception as e:
            self._set_status(f"Backend error: {e}", C["error"])

    def _is_omnivoice_active(self) -> bool:
        if not hasattr(self, "_model_combo"):
            return False
        model_id = self._model_combo.currentData()
        return model_id is not None and model_id.startswith("omnivoice_")

    def _is_supertonic_active(self) -> bool:
        if not hasattr(self, "_model_combo"):
            return False
        model_id = self._model_combo.currentData()
        return model_id is not None and model_id.startswith("supertonic_")

    def _is_piper_active(self) -> bool:
        if not hasattr(self, "_model_combo"):
            return False
        model_id = self._model_combo.currentData()
        return model_id == "piper"

    def _update_voice_section_for_model(self, model_id: str):
        if not hasattr(self, "_omnivoice_mode_widget"):
            return
        is_ov         = model_id.startswith("omnivoice_")
        is_supertonic = model_id.startswith("supertonic_")
        is_piper      = model_id == "piper"
        is_no_clone   = is_supertonic or is_piper

        self._omnivoice_mode_widget.setVisible(is_ov)
        if hasattr(self, "_supertonic_single_hint"):
            self._supertonic_single_hint.setVisible(is_no_clone and not is_ov)

        if is_ov:
            self._on_voice_mode_changed()
        else:
            is_voxcpm2_voicedesign = (model_id == "voxcpm2-voicedesign")
            is_qwen3_no_ref_audio = model_id in (
                "qwen3-voicedesign-1.7b",
                "qwen3-customvoice-1.7b",
                "qwen3-customvoice-0.6b",
            )

            hide_audio_drop = is_qwen3_no_ref_audio or is_no_clone

            for attr in ("_drop", "_ref_player", "_ref_audio_info_lbl", "_proc_section"):
                if hasattr(self, attr):
                    getattr(self, attr).setVisible(not hide_audio_drop)

            self._cloning_widget.setVisible(
                not is_voxcpm2_voicedesign and not is_no_clone
            )

            if hasattr(self, "_whisper_section_widget"):
                incompatible = False
                try:
                    if self._backend is not None:
                        incompatible = self._backend.whisper_incompatible
                except Exception:
                    pass
                self._whisper_section_widget.setVisible(
                    not incompatible
                    and not is_voxcpm2_voicedesign
                    and not is_qwen3_no_ref_audio
                    and not is_no_clone
                )
 
    def _on_voice_mode_changed(self):
        if not hasattr(self, "_voice_mode_combo"):
            return
        mode = self._voice_mode_combo.currentData()
        show_cloning  = (mode == "cloning")
        show_instruct = (mode == "design")
        self._cloning_widget.setVisible(show_cloning)
        self._instruct_widget.setVisible(show_instruct)
        if hasattr(self, "_whisper_section_widget"):
            self._whisper_section_widget.setVisible(show_cloning)
        hints = {
            "cloning": "Clone a voice from a reference WAV file. Recommended: 3–15 s of clean speech.",
            "design":  "Describe the desired voice with attributes (gender, age, pitch, accent…). No reference audio needed.",
            "auto":    "Let the model choose a voice automatically. No configuration required.",
        }
        if hasattr(self, "_omnivoice_hint_lbl"):
            self._omnivoice_hint_lbl.setText(hints.get(mode, ""))
 
    def _get_ref_audio(self) -> Optional[str]:
        if self._is_omnivoice_active():
            mode = (
                self._voice_mode_combo.currentData()
                if hasattr(self, "_voice_mode_combo")
                else "cloning"
            )
            if mode != "cloning":
                return None
        return self._drop.file_path or None
 
    def _get_ref_text(self) -> Optional[str]:
        if self._is_omnivoice_active():
            mode = (
                self._voice_mode_combo.currentData()
                if hasattr(self, "_voice_mode_combo")
                else "cloning"
            )
            if mode != "cloning":
                return None
        return self._ref_text.toPlainText().strip() or None

    def _start_model_download(self):
        model_id = self._model_combo.currentData()
        backend  = self._backend
        if backend is None:
            return
 
        hf_token  = None
        extra_msg = ""
        if backend.auth_required:
            extra_msg = (
                "\n\nThis model requires a HuggingFace token.\n"
                "Accept the license at huggingface.co before downloading."
            )
            hf_token = self._get_hf_token()
            if not hf_token:
                hf_token = self._show_model_token_dialog()
                if not hf_token:
                    return
 
        if QMessageBox.question(
            self, "Download model",
            f"Model: {backend.display_name}\n"
            f"Repo:  {backend.download_repo}\n"
            f"Size:  {backend.download_size}\n"
            f"Destination: {backend.model_dir}"
            f"{extra_msg}\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
 
        self._dl_fish_btn.setEnabled(False)
        self._dl_fish_btn.setText("⬇  Downloading…")
        self._progress.setVisible(True)
        self._progress.setRange(0, 100)
 
        self._w_dl = DownloadModelWorker(backend, hf_token)
        self._w_dl.status.connect(lambda m: self._set_status(m))
        self._w_dl.finished.connect(self._on_fish_dl_done)
        self._w_dl.error.connect(lambda e: self._on_error("Download error", e,
            reset_fn=lambda: (
                self._dl_fish_btn.setEnabled(True),
                self._dl_fish_btn.setText("⬇  Download model"),
                self._progress.setVisible(False),
            )))
        self._w_dl.start()

    def _on_fish_dl_done(self):
        self._progress.setVisible(False)
        self._set_status("Model downloaded! Click 'Load model'.", C["success"])
        self._refresh_fish_ui()

    def _load_fish_model(self):
        backend = self._backend
        if backend is None:
            return

        self._load_btn.setEnabled(False)
        self._load_btn.setText("⚡  Loading…")
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self._model_lbl.setText("● Loading…")
        self._model_lbl.setStyleSheet(
            f"color:{C['warning']};font-size:11px;background:transparent;border:none;"
        )

        self._w_load = LoadModelWorker(backend)
        self._w_load.status.connect(lambda m: self._set_status(m))
        self._w_load.finished.connect(self._on_model_loaded)
        self._w_load.error.connect(lambda e: self._on_error("Model load error", e,
            reset_fn=lambda: (
                self._load_btn.setEnabled(True),
                self._load_btn.setText("⚡  Load model"),
                self._progress.setVisible(False),
                self._refresh_fish_ui(),
            )))
        self._w_load.start()

    def _on_model_loaded(self):
        self._progress.setVisible(False)
        device_info = getattr(
            self._backend, "device_info",
            getattr(self._backend, "device", "")
        )
        self._set_status(f"Model ready — {device_info}", C["success"])
        self._refresh_fish_ui()

    def _unload_fish_model(self):
        if not self._backend or not self._backend.is_loaded:
            return
        model_name = self._backend.name
        if QMessageBox.question(
            self, "Unload model",
            f"Unload {model_name} from GPU memory?\n\nYou will need to reload it before generating.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
 
        self._unload_btn.setEnabled(False)
        self._unload_btn.setText("⏳  Unloading…")
        self._stop_play()
 
        def do_unload():
            self._backend.unload_model(lambda m: self._set_status(m))
 
        t = threading.Thread(target=do_unload, daemon=True)
        t.start()
 
        def poll():
            if t.is_alive():
                QTimer.singleShot(100, poll)
            else:
                self._unload_btn.setEnabled(True)
                self._unload_btn.setText("🗑  Unload")
                self._set_status("Model unloaded — reload before generating.", C["warning"])
                self._refresh_fish_ui()
 
        QTimer.singleShot(100, poll)

    @staticmethod
    def _audio_info_str(path: str) -> str:
        try:
            info = sf.info(path)
            channels = "Mono" if info.channels == 1 else "Stereo"
            sr = info.samplerate
            if sr >= 1000:
                khz = sr / 1000
                sr_str = f"{khz:.1f} kHz" if khz != int(khz) else f"{int(khz)} kHz"
            else:
                sr_str = f"{sr} Hz"
            sub = info.subtype.upper()
            if "16" in sub:
                bits = "16-bit"
            elif "24" in sub:
                bits = "24-bit"
            elif "32" in sub or "FLOAT" in sub or "DOUBLE" in sub:
                bits = "32-bit"
            else:
                bits = sub
            return f"{channels} • {bits} • {sr_str}"
        except Exception:
            return ""

    def _on_ref_dropped(self, path: str):
        self._ref_player.load(path)
        self._proc_btn.setEnabled(True)
        self._set_status(f"Reference audio: {Path(path).name}", C["accent"])
        self._ref_audio_info_lbl.setText(self._audio_info_str(path))
        self._refresh_whisper_ui()

    def _on_show_video_waveform(self):
        if not self._ffmpeg_ok:
            QMessageBox.warning(self, "ffmpeg not found",
                "This feature requires ffmpeg installed in PATH.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video file", _get_last_dir("video"),
            "Video (*.mp4 *.mkv *.avi *.mov *.webm *.wmv);;All files (*)",
        )
        if not path:
            return
        _set_last_dir("video", path)
        self._video_source_path = path
        self._btn_show_video_wave.setEnabled(False)
        self._btn_show_video_wave.setText("⏳  Extracting audio…")
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)

        self._w_vid_extract = VideoAudioExtractWorker(path)
        self._w_vid_extract.status.connect(lambda m: self._set_status(m))

    def _on_video_audio_extracted(self, wav_path: str):
        self._progress.setVisible(False)
        self._btn_show_video_wave.setEnabled(True)
        self._btn_show_video_wave.setText("🎬  Show Waveform from video")
        self._video_player.load(wav_path)
        if hasattr(self, '_vid_row_widget'):
            self._vid_row_widget.setVisible(False)
        self._set_status(
            f"Video audio loaded: {Path(self._video_source_path).name}", C["success"]
        )

    def _clear_ref(self):
        self._drop.clear_file()
        self._ref_player.clear()
        self._ref_text.clear()
        self._proc_btn.setEnabled(False)
        self._proc_lbl.setText("")
        self._ref_audio_info_lbl.setText("")
        self._set_status("Reference audio removed.")
        self._refresh_whisper_ui()

    def _demucs_model_downloaded(self) -> bool:
        torch_home = Path(os.environ.get("TORCH_HOME", str(ROOT_DIR / "models" / "torch_hub")))
        for search_root in [
            torch_home / "hub" / "checkpoints",
            torch_home / "checkpoints",
            Path.home() / ".cache" / "torch" / "hub" / "checkpoints",
        ]:
            if search_root.exists():
                for f in search_root.iterdir():
                    if "htdemucs" in f.name.lower():
                        return True
        return False

    def _on_demucs_toggled(self, checked: bool):
        if not checked:
            return
        if self._demucs_model_downloaded():
            return
        demucs_dir = ROOT_DIR / "models" / "torch_hub"
        reply = QMessageBox.question(
            self, "Download Demucs model?",
            (
                "Vocal isolation requires Demucs htdemucs_ft model (~300 MB).\n\n"
                f"Model will be downloaded to:\n{demucs_dir}\n\n"
                "Download happens automatically on first processing.\nContinue?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            self._chk_demucs.blockSignals(True)
            self._chk_demucs.setChecked(False)
            self._chk_demucs.blockSignals(False)

    def _on_speaker_demucs_toggled(self, checked: bool, chk: QCheckBox):
        if not checked:
            return
        if self._demucs_model_downloaded():
            return
        demucs_dir = ROOT_DIR / "models" / "torch_hub"
        reply = QMessageBox.question(
            self, "Download Demucs model?",
            (
                "Vocal isolation requires Demucs htdemucs_ft model (~300 MB).\n\n"
                f"Model will be downloaded to:\n{demucs_dir}\n\n"
                "Download happens automatically on first processing.\nContinue?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            chk.blockSignals(True)
            chk.setChecked(False)
            chk.blockSignals(False)

    def _process_audio(self):
        path = self._drop.file_path
        if not path:
            QMessageBox.warning(self, "No audio", "Upload reference audio first.")
            return
        if not any([
            self._mono_check.isChecked(),
            self._chk_demucs.isChecked(),
            self._chk_bit_depth.isChecked(),
            self._sr_combo.currentData() is not None,
        ]):
            QMessageBox.information(self, "No options", "Select at least one option.")
            return

        output_subtype = (
            self._bit_depth_combo.currentData()
            if self._chk_bit_depth.isChecked()
            else "PCM_16"
        )

        settings = {
            "target_sr":      self._sr_combo.currentData(),
            "to_mono":        self._mono_check.isChecked(),
            "isolate_vocals": self._chk_demucs.isChecked(),
            "normalize":      True,
            "output_subtype": output_subtype,
            "device":         "cuda" if (self._backend and self._backend.device == "cuda") else "cpu",
        }

        self._proc_btn.setEnabled(False)
        self._proc_btn.setText("⏳  Processing…")
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self._proc_lbl.setText("Processing…")
        self._proc_lbl.setStyleSheet(f"color:{C['warning']};font-size:10px;")

        self._w_proc = AudioProcessWorker(self._preprocessor, path, settings)
        self._w_proc.status.connect(lambda m: (self._set_status(m), self._proc_lbl.setText(m)))
        self._w_proc.finished.connect(self._on_proc_done)
        self._w_proc.error.connect(lambda e: self._on_error("Audio processing error", e,
            reset_fn=lambda: (self._proc_btn.setEnabled(True),
                              self._proc_btn.setText("🔧  Process audio"),
                              self._progress.setVisible(False),
                              self._proc_lbl.setText("Error — check logs"))))
        self._w_proc.start()

    def _on_proc_done(self, out: str):
        self._proc_btn.setEnabled(True)
        self._proc_btn.setText("🔧  Process audio")
        self._progress.setVisible(False)
        self._drop._set(out)
        self._ref_player.load(out)
        self._proc_lbl.setText(f"✓  {Path(out).name}")
        self._proc_lbl.setStyleSheet(f"color:{C['success']};font-size:10px;")
        self._set_status(f"Audio processed: {Path(out).name}", C["success"])
        self._refresh_whisper_ui()

    def _start_whisper_download(self):
        size = self._w_size.currentData()
        if QMessageBox.question(
            self, f"Download Whisper {size}",
            f"Model '{size}' ({WHISPER_SIZE_MB.get(size, '')}) will be downloaded to:\n"
            f"{WHISPER_DIR / size}\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return

        self._w_dl_btn.setEnabled(False)
        self._w_dl_btn.setText("⬇  Downloading…")
        self._progress.setVisible(True)
        self._progress.setRange(0, 100)

        self._w_wdl = WhisperDownloadWorker(self._whisper_backend, size)
        self._w_wdl.status.connect(lambda m: self._set_status(m))
        self._w_wdl.finished.connect(self._on_whisper_dl_done)
        self._w_wdl.error.connect(lambda e: self._on_error("Whisper download error", e,
            reset_fn=lambda: (self._w_dl_btn.setEnabled(True),
                              self._w_dl_btn.setText("⬇  Download Whisper"),
                              self._progress.setVisible(False))))
        self._w_wdl.start()

    def _on_whisper_dl_done(self):
        self._progress.setVisible(False)
        self._set_status(f"Whisper {self._w_size.currentData()} downloaded!", C["success"])
        self._refresh_whisper_ui()

    def _transcribe(self):
        path = self._drop.file_path
        if not path:
            QMessageBox.warning(self, "No audio", "Upload reference audio first.")
            return

        size = self._w_size.currentData()
        lang = self._w_lang.currentData()

        self._w_tr_btn.setEnabled(False)
        self._w_tr_btn.setText("⏳  Transcribing…")
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self._w_status.setText(f"Loading Whisper {size}…")
        self._w_status.setStyleSheet(f"color:{C['warning']};font-size:10px;")

        self._w_tr = TranscribeWorker(self._whisper_backend, path, size, lang)
        self._w_tr.status.connect(lambda m: (self._set_status(m), self._w_status.setText(m)))
        self._w_tr.finished.connect(self._on_transcribe_done)
        self._w_tr.error.connect(lambda e: self._on_error("Whisper transcription error", e,
            reset_fn=lambda: (self._w_tr_btn.setEnabled(True),
                              self._w_tr_btn.setText("🎤  Transcribe"),
                              self._progress.setVisible(False),
                              self._refresh_whisper_ui())))
        self._w_tr.start()

    def _on_transcribe_done(self, text: str):
        self._progress.setVisible(False)
        self._w_tr_btn.setText("🎤  Transcribe")
        self._ref_text.setPlainText(text)
        size = self._w_size.currentData()
        self._w_status.setText(f"✓  Whisper {size} released from memory")
        self._w_status.setStyleSheet(f"color:{C['success']};font-size:10px;")
        self._set_status(f"Transcription: {len(text)} chars", C["success"])
        self._refresh_whisper_ui()

    def _redistribute_split_timings(self, fragments: List[Dict]) -> None:
        i = 0
        while i < len(fragments):
            j = i + 1
            while (
                j < len(fragments)
                and fragments[j].get('start_ms') == fragments[i].get('start_ms')
                and fragments[j].get('end_ms') == fragments[i].get('end_ms')
            ):
                j += 1

            group_size = j - i
            if group_size > 1:
                group      = fragments[i:j]
                base_start = fragments[i].get('start_ms', 0)
                base_end   = fragments[i].get('end_ms', 0)
                total_ms   = base_end - base_start

                next_frag     = fragments[j] if j < len(fragments) else None
                next_start_ms = next_frag.get('start_ms', base_end) if next_frag else base_end
                gap_ms        = next_start_ms - base_end

                dynamic_limit   = int(total_ms * 0.25)
                borrow_limit_ms = min(500, max(0, dynamic_limit))
                bonus_ms        = min(max(0, gap_ms), borrow_limit_ms)

                lengths   = [max(1, len(f.get('text') or '')) for f in group]
                total_len = sum(lengths)

                bonus_parts = [int(bonus_ms * l / total_len) for l in lengths]
                bonus_parts[-1] += bonus_ms - sum(bonus_parts)

                if total_ms <= 0:
                    slot_ms = max(100, bonus_ms // group_size)
                    cursor  = base_start
                    for k, frag in enumerate(group):
                        frag_start = cursor
                        frag_end   = frag_start + slot_ms
                        if k == group_size - 1 and next_frag is not None:
                            frag_end = min(frag_end, next_start_ms)
                        frag['start_ms']  = frag_start
                        frag['end_ms']    = frag_end
                        frag['timestamp'] = f"{_ms_to_ts(frag_start)} --> {_ms_to_ts(frag_end)}"
                        cursor = frag_end
                else:
                    cursor = base_start
                    for k, frag in enumerate(group):
                        frag_start   = cursor
                        base_frag_ms = int(total_ms * lengths[k] / total_len)
                        frag_ms      = max(100, base_frag_ms + bonus_parts[k])
                        frag_end     = frag_start + frag_ms
                        if k == group_size - 1 and next_frag is not None:
                            frag_end = min(frag_end, next_start_ms)
                        frag['start_ms']  = frag_start
                        frag['end_ms']    = frag_end
                        frag['timestamp'] = f"{_ms_to_ts(frag_start)} --> {_ms_to_ts(frag_end)}"
                        cursor = frag_end

            i = j

        for frag in fragments:
            frag['srt_end_ms'] = frag['end_ms']
        
    def _load_srt_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open SRT / TXT file", _get_last_dir("srt"),
            "Subtitle files (*.srt *.txt);;SRT Subtitles (*.srt);;TXT Subtitles (*.txt);;All files (*)",
        )
        if not path:
            return
        _set_last_dir("srt", path)

        ext = Path(path).suffix.lower()

        try:
            if ext == '.txt':
                segments = txt_srt_format.load(path)
            else:
                segments = get_format('.srt').load(path)
            if not segments:
                QMessageBox.warning(self, "Empty file", "No valid subtitle blocks found.")
                return
            self._fragments = []
            for seg in segments:
                self._fragments.append({
                    'index':       seg.index,
                    'srt_id':      str(seg.index + 1),
                    'timestamp':   f"{_ms_to_ts(seg.start_ms)} --> {_ms_to_ts(seg.end_ms)}",
                    'text':        seg.text,
                    'start_ms':    seg.start_ms,
                    'end_ms':      seg.end_ms,
                    'speaker':     "",
                    'status':      'waiting',
                    'output_path': None,
                    'error_msg':   None,
                })
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return

        self._reset_dubbing_state()

        self._set_status("Detecting language…")
        QApplication.processEvents()

        raw_texts     = [f["text"] for f in self._fragments if f.get("text")]
        detected_lang = _detect_srt_language(raw_texts)
        logger.info(f"Detected language: {detected_lang}")

        try:
            self._set_status(f"Converting numbers to words [{detected_lang}]…")
            QApplication.processEvents()
            for frag in self._fragments:
                original  = frag.get("text", "")
                converted = _convert_numbers_in_text(original, detected_lang)
                if converted != original:
                    frag["text"] = converted
                    logger.debug(f"Fragment {frag['index']}: {original!r} → {converted!r}")
        except Exception as e:
            logger.warning(f"Number conversion failed: {e}")

        self._redistribute_split_timings(self._fragments)

        self._srt_path   = path
        self._output_dir = str(OUTPUTS_DIR / Path(path).stem)

        auto = self._auto_session_path()
        if auto and auto.exists():
            reply = QMessageBox.question(
                self, "Restore previous session",
                f"A saved session was found for this file.\n\n"
                f"Restore it? Previously synthesized fragments will be loaded automatically.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    with open(str(auto), "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    self._restore_session_data(data)
                    logger.info(f"Auto-session restored from: {auto}")
                    return
                except Exception as e:
                    logger.warning(f"Auto-session restore failed: {e}")
                    self._set_status("Could not restore previous session — starting fresh.", C["warning"])

        fname = Path(path).name
        self._srt_label.setText(
            f"📄  {fname}  •  {len(self._fragments)} fragments  [{detected_lang}]"
        )
        self._srt_label.setStyleSheet(
            f"color:{C['accent']};font-size:11px;font-weight:600;"
        )

        self._btn_close_srt.setEnabled(True)
        self._btn_save_session.setEnabled(True)
        self._update_dubbing_visibility()
        self._tabs.setCurrentIndex(0)

        self._populate_tree()
        self._update_action_buttons()
        self._set_status(
            f"Loaded: {fname} — {len(self._fragments)} fragments | language: {detected_lang}"
        )
        logger.info(
            f"Loaded: {path} → {len(self._fragments)} fragments, lang={detected_lang}"
        )

    def _close_srt_file(self):
        if self._is_running:
            QMessageBox.warning(self, "Busy", "Cannot close file during active synthesis.")
            return
        reply = QMessageBox.question(
            self, "Close SRT file",
            "Close current SRT file?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
 
        self._fragments.clear()
        self._frag_items.clear()
        self._chapter_item = None
        self._srt_path     = None
        self._tree.clear()
        self._preview_text.clear()
        self._srt_label.setText("No SRT file loaded")
        self._srt_label.setStyleSheet(
            f"color:{C['text3']};font-size:11px;font-style:italic;"
        )
        self._btn_close_srt.setEnabled(False)
        self._btn_save_session.setEnabled(False)
        self._reset_dubbing_state()
        self._update_action_buttons()
        self._set_status("SRT file closed.")

    def _load_ebook_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open ebook file", _get_last_dir("ebook"),
            "Ebook files (*.epub *.pdf *.mobi *.azw *.azw3 *.fb2 *.txt)"
            ";;EPUB (*.epub);;PDF (*.pdf);;Kindle (*.mobi *.azw *.azw3)"
            ";;FictionBook (*.fb2);;TXT (*.txt);;All files (*)",
        )
        if not path:
            return
        _set_last_dir("ebook", path)

        ext = Path(path).suffix.lower()

        try:
            if ext == '.txt':
                segments = txt_ebook_format.load(path)
            else:
                segments = get_format(ext).load(path)
            if not segments:
                QMessageBox.warning(self, "Empty file", "No text fragments found in this file.")
                return
            self._ebook_fragments = []
            for seg in segments:
                self._ebook_fragments.append({
                    'index':       seg.index,
                    'text':        seg.text,
                    'speaker':     seg.speaker or "",
                    'status':      'waiting',
                    'output_path': None,
                    'error_msg':   None,
                })
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return

        self._epub_path        = path
        self._ebook_output_dir = str(OUTPUTS_DIR / Path(path).stem)
        if hasattr(self, '_audiobook_output_edit'):
            self._audiobook_output_edit.setText(self._ebook_output_dir)

        auto = self._auto_ebook_session_path()
        if auto and auto.exists():
            reply = QMessageBox.question(
                self, "Restore previous session",
                f"A saved session was found for this file.\n\n"
                f"Restore it? Previously synthesized fragments will be loaded automatically.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    with open(str(auto), "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    self._restore_ebook_session_data(data)
                    logger.info(f"Auto ebook session restored from: {auto}")
                    return
                except Exception as e:
                    logger.warning(f"Auto ebook session restore failed: {e}")
                    self._set_status("Could not restore previous session — starting fresh.", C["warning"])

        fname = Path(path).name
        self._epub_label.setText(
            f"📚  {fname}  •  {len(self._ebook_fragments)} fragments"
        )
        self._epub_label.setStyleSheet(
            f"color:{C['accent']};font-size:11px;font-weight:600;"
        )
        self._btn_close_ebook.setEnabled(True)
        self._btn_save_ebook_session.setEnabled(True)
        self._tabs.setCurrentIndex(1)
        self._populate_ebook_tree()
        self._update_action_buttons()
        self._set_status(f"Loaded: {fname} — {len(self._ebook_fragments)} fragments")
 
    def _close_ebook_file(self):
        if self._is_running:
            QMessageBox.warning(self, "Busy", "Cannot close file during active synthesis.")
            return
        reply = QMessageBox.question(
            self, "Close ebook",
            "Close current ebook file?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._ebook_fragments.clear()
        self._ebook_frag_items.clear()
        self._ebook_chapter_item = None
        self._epub_path          = None
        self._ebook_tree.clear()
        self._ebook_preview_text.clear()
        self._epub_label.setText("No ebook loaded")
        self._epub_label.setStyleSheet(
            f"color:{C['text3']};font-size:11px;font-style:italic;"
        )
        self._btn_close_ebook.setEnabled(False)
        self._btn_save_ebook_session.setEnabled(False)
        self._update_action_buttons()
        self._set_status("Ebook file closed.")
 
    def _populate_ebook_tree(self):
        self._ebook_tree.clear()
        self._ebook_frag_items.clear()
        if not self._ebook_fragments:
            return

        epub_name = Path(self._epub_path).name if self._epub_path else "EPUB"

        ch_item = QTreeWidgetItem()
        ch_item.setText(COL_FRAGMENT, f"📚 {epub_name}")
        ch_item.setFont(COL_FRAGMENT, QFont("Segoe UI", 12, QFont.Weight.Bold))
        ch_item.setForeground(COL_FRAGMENT, QColor(C["accent"]))
        self._ebook_chapter_item = ch_item

        self._ebook_tree.setUpdatesEnabled(False)
        try:
            children = []
            for frag in self._ebook_fragments:
                item = self._create_ebook_tree_item(frag, parent=None)
                children.append(item)

            ch_item.addChildren(children)
            self._ebook_tree.addTopLevelItem(ch_item)
            ch_item.setExpanded(True)
        finally:
            self._ebook_tree.setUpdatesEnabled(True)


    def _create_ebook_tree_item(self, frag: Dict, parent=None) -> QTreeWidgetItem:
        status = frag.get('status', 'waiting')
        icon = {
            'waiting': STATUS_WAITING,
            'running': STATUS_RUNNING,
            'done': STATUS_DONE,
            'error': STATUS_ERROR,
        }.get(status, STATUS_WAITING)

        num = frag['index'] + 1

        item = QTreeWidgetItem(parent) if parent is not None else QTreeWidgetItem()

        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(COL_STATUS, Qt.CheckState.Checked)
        item.setText(COL_STATUS, "")
        item.setText(COL_FRAGMENT, f"{icon} #{num} {frag.get('text', '')[:75]}")
        item.setText(COL_SPEAKER, frag.get('speaker') or "")

        if status == 'error':
            item.setForeground(COL_FRAGMENT, QColor(C["error"]))

        if status == 'done':
            dur = _get_wav_duration(frag.get('output_path', ''))
            if dur is not None:
                dur_ms = int(dur * 1000)
                target_ms = frag.get('target_duration_ms')
                pre_ms = int(frag.get('pre_silence_ms') or 0)
                timing_txt = _fmt_ms(dur_ms)
                has_extra = bool(target_ms and target_ms > dur_ms)
                if has_extra:
                    extra_ms = target_ms - dur_ms
                    timing_txt += f" +{extra_ms} ms"
                if pre_ms > 0:
                    timing_txt = f"+{pre_ms} ms, " + timing_txt
                if has_extra or pre_ms > 0:
                    item.setForeground(COL_TIMING, QColor(C["accent"]))
                else:
                    item.setForeground(COL_TIMING, QColor("#55bb55"))
                item.setText(COL_TIMING, timing_txt)
            else:
                item.setText(COL_TIMING, "—")
                item.setForeground(COL_TIMING, QColor(C["text3"]))
        else:
            item.setText(COL_TIMING, "—")
            item.setForeground(COL_TIMING, QColor(C["text3"]))

        item.setData(COL_STATUS, Qt.ItemDataRole.UserRole, frag['index'])
        self._ebook_frag_items[frag['index']] = item
        return item
 
    def _update_ebook_tree_item(self, idx: int):
        item = self._ebook_frag_items.get(idx)
        if not item:
            return
        frag = next((f for f in self._ebook_fragments if f['index'] == idx), None)
        if not frag:
            return
        status = frag.get('status', 'waiting')
        icon = {
            'waiting': STATUS_WAITING,
            'running': STATUS_RUNNING,
            'done':    STATUS_DONE,
            'error':   STATUS_ERROR,
        }.get(status, STATUS_WAITING)

        num = frag['index'] + 1
        item.setText(COL_FRAGMENT, f"{icon}  #{num}  {frag.get('text', '')[:75]}")
        item.setText(COL_SPEAKER,  frag.get('speaker') or "")

        if status == 'error':
            item.setForeground(COL_FRAGMENT, QColor(C["error"]))
        else:
            item.setForeground(COL_FRAGMENT, QColor(C["text"]))

        if status == 'done':
            dur = _get_wav_duration(frag.get('output_path', ''))
            if dur is not None:
                dur_ms     = int(dur * 1000)
                target_ms  = frag.get('target_duration_ms')
                pre_ms     = int(frag.get('pre_silence_ms') or 0)
                timing_txt = _fmt_ms(dur_ms)
                has_extra  = bool(target_ms and target_ms > dur_ms)
                if has_extra:
                    extra_ms    = target_ms - dur_ms
                    timing_txt += f"  +{extra_ms} ms"
                if pre_ms > 0:
                    timing_txt = f"+{pre_ms} ms, " + timing_txt
                if has_extra or pre_ms > 0:
                    item.setForeground(COL_TIMING, QColor(C["accent"]))
                else:
                    item.setForeground(COL_TIMING, QColor("#55bb55"))
                item.setText(COL_TIMING, timing_txt)
            else:
                item.setText(COL_TIMING, "—")
                item.setForeground(COL_TIMING, QColor(C["text3"]))
        else:
            item.setText(COL_TIMING, "—")
            item.setForeground(COL_TIMING, QColor(C["text3"]))
 
    def _apply_ebook_filter(self, text: str):
        if not self._ebook_chapter_item:
            return
        txt = text.lower().strip()
        for i in range(self._ebook_chapter_item.childCount()):
            child = self._ebook_chapter_item.child(i)
            matches = (not txt or
                       txt in child.text(COL_FRAGMENT).lower() or
                       txt in child.text(COL_SPEAKER).lower())
            child.setHidden(not matches)
 
    def _select_all_ebook(self, state: bool):
        if not self._ebook_chapter_item:
            return
        cs = Qt.CheckState.Checked if state else Qt.CheckState.Unchecked
        self._ebook_tree.setUpdatesEnabled(False)
        self._ebook_tree.blockSignals(True)
        try:
            for i in range(self._ebook_chapter_item.childCount()):
                child = self._ebook_chapter_item.child(i)
                if not child.isHidden():
                    child.setCheckState(COL_STATUS, cs)
        finally:
            self._ebook_tree.blockSignals(False)
            self._ebook_tree.setUpdatesEnabled(True)
        self._update_preview_btn_state()

    def _select_failed_ebook(self):
        if not self._ebook_chapter_item:
            return
        frag_map = {f['index']: f for f in self._ebook_fragments}
        self._ebook_tree.setUpdatesEnabled(False)
        self._ebook_tree.blockSignals(True)
        try:
            for i in range(self._ebook_chapter_item.childCount()):
                child = self._ebook_chapter_item.child(i)
                idx  = child.data(COL_STATUS, Qt.ItemDataRole.UserRole)
                frag = frag_map.get(idx)
                cs   = Qt.CheckState.Checked if (frag and frag.get('status') == 'error') else Qt.CheckState.Unchecked
                child.setCheckState(COL_STATUS, cs)
        finally:
            self._ebook_tree.blockSignals(False)
            self._ebook_tree.setUpdatesEnabled(True)
        self._update_preview_btn_state()
 
    def _select_all_reset(self, state: bool):
        self._sel_fail_btn.blockSignals(True)
        self._sel_pending_btn.blockSignals(True)
        self._sel_fail_btn.setChecked(False)
        self._sel_pending_btn.setChecked(False)
        self._sel_fail_btn.blockSignals(False)
        self._sel_pending_btn.blockSignals(False)
        self._select_all(state)

    def _select_all_ebook_reset(self, state: bool):
        self._ebook_sel_fail_btn.blockSignals(True)
        self._ebook_sel_pending_btn.blockSignals(True)
        self._ebook_sel_fail_btn.setChecked(False)
        self._ebook_sel_pending_btn.setChecked(False)
        self._ebook_sel_fail_btn.blockSignals(False)
        self._ebook_sel_pending_btn.blockSignals(False)
        self._select_all_ebook(state)

    def _apply_status_filter(self):
        if not self._chapter_item:
            return
        want_failed  = self._sel_fail_btn.isChecked()
        want_pending = self._sel_pending_btn.isChecked()
        frag_map = {f['index']: f for f in self._fragments}
        self._tree.setUpdatesEnabled(False)
        self._tree.blockSignals(True)
        try:
            for i in range(self._chapter_item.childCount()):
                child = self._chapter_item.child(i)
                if child.isHidden():
                    continue
                idx    = child.data(COL_STATUS, Qt.ItemDataRole.UserRole)
                frag   = frag_map.get(idx)
                if not frag:
                    continue
                status = frag.get('status', 'waiting')
                if want_failed and want_pending:
                    match = status in ('error', 'waiting')
                elif want_failed:
                    match = status == 'error'
                elif want_pending:
                    match = status == 'waiting'
                else:
                    match = False
                child.setCheckState(COL_STATUS, Qt.CheckState.Checked if match else Qt.CheckState.Unchecked)
        finally:
            self._tree.blockSignals(False)
            self._tree.setUpdatesEnabled(True)

    def _apply_ebook_status_filter(self):
        if not self._ebook_chapter_item:
            return
        want_failed  = self._ebook_sel_fail_btn.isChecked()
        want_pending = self._ebook_sel_pending_btn.isChecked()
        frag_map = {f['index']: f for f in self._ebook_fragments}
        self._ebook_tree.setUpdatesEnabled(False)
        self._ebook_tree.blockSignals(True)
        try:
            for i in range(self._ebook_chapter_item.childCount()):
                child = self._ebook_chapter_item.child(i)
                if child.isHidden():
                    continue
                idx    = child.data(COL_STATUS, Qt.ItemDataRole.UserRole)
                frag   = frag_map.get(idx)
                if not frag:
                    continue
                status = frag.get('status', 'waiting')
                if want_failed and want_pending:
                    match = status in ('error', 'waiting')
                elif want_failed:
                    match = status == 'error'
                elif want_pending:
                    match = status == 'waiting'
                else:
                    match = False
                child.setCheckState(COL_STATUS, Qt.CheckState.Checked if match else Qt.CheckState.Unchecked)
        finally:
            self._ebook_tree.blockSignals(False)
            self._ebook_tree.setUpdatesEnabled(True)
        self._update_preview_btn_state()
 
    def _on_ebook_selection_changed(self):
        items = self._ebook_tree.selectedItems()
        if not items:
            return
        item = items[0]
        idx = item.data(COL_STATUS, Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        self._load_fragment_audio(idx)
        frag = next((f for f in self._ebook_fragments if f['index'] == idx), None)
        if not frag:
            return
        self._ebook_preview_text.setPlainText(frag.get('text', ''))
 
    def _load_fragment_audio(self, idx: int) -> bool:
        if self._tabs.currentIndex() == 1:
            fragments = self._ebook_fragments
        else:
            fragments = self._fragments

        frag = next((f for f in fragments if f.get('index') == idx), None)
        if not frag or frag.get('status') != 'done':
            return False

        path = frag.get('output_path')
        if not path or not os.path.exists(path):
            return False

        try:
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            self._load_audio_to_player(audio, sr, fragment_idx=idx)
            return True
        except Exception as e:
            self._set_status(f"Cannot load audio for fragment {idx}: {e}", C["error"])
            return False
 
    def _get_checked_ebook_fragments(self, retry_errors: bool = False) -> List[Dict]:
        result = []
        if not self._ebook_chapter_item:
            return result
        for i in range(self._ebook_chapter_item.childCount()):
            child = self._ebook_chapter_item.child(i)
            if child.checkState(COL_STATUS) != Qt.CheckState.Checked:
                continue
            if child.isHidden():
                continue
            idx  = child.data(COL_STATUS, Qt.ItemDataRole.UserRole)
            frag = next((f for f in self._ebook_fragments if f['index'] == idx), None)
            if not frag:
                continue
            if retry_errors:
                if frag.get('status') not in ('waiting', 'error'):
                    continue
            result.append(frag)
        return result
 
    def _start_ebook_synthesis(self, retry_errors: bool = False):
        if not self._backend or not self._backend.is_loaded:
            QMessageBox.warning(self, "Model not loaded", "Load the model first.")
            return

        to_process = self._get_checked_ebook_fragments(retry_errors=retry_errors)
        if not to_process:
            QMessageBox.information(self, "Nothing to synthesize", "No fragments selected.")
            return

        already_done = [f for f in to_process if f.get('status') == 'done']
        if already_done:
            reply = QMessageBox.question(
                self,
                "Re-synthesize completed fragments?",
                f"{len(already_done)} fragment(s) already have generated audio.\n\n"
                "Do you want to re-synthesize them anyway?\n"
                "Existing audio files will be overwritten.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.No:
                to_process = [f for f in to_process if f.get('status') != 'done']
                if not to_process:
                    QMessageBox.information(self, "Nothing to synthesize",
                        "No pending fragments left to process.")
                    return

        for frag in to_process:
            frag["status"] = "waiting"
            frag["error_msg"] = None
            self._update_ebook_tree_item(frag["index"])

        self._synthesis_source = 'ebook'
        self._is_running = True
        self._completed_count = 0
        self._synth_start_time = time.monotonic()
        self._synth_total = len(to_process)
        self._update_action_buttons()

        total = self._synth_total
        self._synth_progress.setMaximum(total)
        self._synth_progress.setValue(0)
        self._synth_progress.setVisible(True)
        self._eta_label.setText(f"0/{total}")
        self._eta_label.setVisible(True)

        os.makedirs(self._ebook_output_dir, exist_ok=True)
        reserved_paths = {f['output_path'] for f in self._ebook_fragments if f.get('output_path')}

        self._worker = TTSWorker(
            backend=self._backend,
            fragments=to_process,
            output_dir=self._ebook_output_dir,
            reference_audio=self._get_ref_audio(),
            reference_text=self._get_ref_text(),
            filename_prefix=Path(self._epub_path).stem if self._epub_path else "ebook_fragment",
            generation_settings=self._get_generation_settings(),
            normalize_audio=self._norm_check.isChecked(),
            speaker_voices=self._get_speaker_voices_dict(),
            reserved_paths=reserved_paths,
        )

        self._worker.progress.connect(self._on_progress)
        self._worker.item_done.connect(self._on_item_done)
        self._worker.finished.connect(self._on_synthesis_finished)
        self._worker.start()

        self._set_status(f"Ebook synthesis started — {total} fragments queued…")
        logger.info(f"Ebook synthesis started: {total} fragments")
 
    def _re_synthesize_ebook_fragment(self, frag: Dict):
        if not self._backend or not self._backend.is_loaded:
            QMessageBox.warning(self, "Model not loaded", "Load the model first.")
            return

        self._synthesis_source = 'ebook'

        frag["status"]    = "waiting"
        frag["error_msg"] = None
        self._update_ebook_tree_item(frag["index"])

        self._is_running       = True
        self._completed_count  = 0
        self._synth_start_time = time.monotonic()
        self._synth_total      = 1
        self._update_action_buttons()
        self._synth_progress.setMaximum(1)
        self._synth_progress.setValue(0)
        self._synth_progress.setVisible(True)
        self._eta_label.setText("0/1")
        self._eta_label.setVisible(True)

        reserved_paths = {
            f['output_path'] for f in self._ebook_fragments if f.get('output_path')
        }

        self._worker = TTSWorker(
            backend=self._backend,
            fragments=[frag],
            output_dir=self._ebook_output_dir,
            reference_audio=self._get_ref_audio(),
            reference_text=self._get_ref_text(),
            filename_prefix=Path(self._epub_path).stem if self._epub_path else "ebook_fragment",
            generation_settings=self._get_generation_settings(),
            normalize_audio=self._norm_check.isChecked(),
            speaker_voices=self._get_speaker_voices_dict(),
            reserved_paths=reserved_paths,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.item_done.connect(self._on_item_done)
        self._worker.finished.connect(self._on_synthesis_finished)
        self._worker.start()
 
    def _show_ebook_context_menu(self, pos):
        item = self._ebook_tree.itemAt(pos)
        if not item:
            return
        idx = item.data(COL_STATUS, Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        frag = next((f for f in self._ebook_fragments if f['index'] == idx), None)
        if not frag:
            return

        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{ background:{C['panel']}; border:1px solid {C['border2']}; color:{C['text']}; padding:4px 0; }}
            QMenu::item {{ padding:5px 20px; }}
            QMenu::item:selected {{ background:{C['accent2']}; }}
            QMenu::separator {{ background:{C['border']}; height:1px; margin:4px 0; }}
        """)

        has_audio = bool(
            frag.get('output_path') and os.path.exists(frag.get('output_path', ''))
        )
        model_ok = self._backend.is_loaded if self._backend else False

        act_open = menu.addAction("📁  Open output folder")
        act_open.setEnabled(has_audio)
        act_open.triggered.connect(lambda: _open_file(os.path.dirname(frag['output_path'])))

        menu.addSeparator()

        act_re = menu.addAction("🔄  Re-synthesize this fragment")
        act_re.setEnabled(model_ok and not self._is_running)
        act_re.triggered.connect(lambda: self._re_synthesize_ebook_fragment(frag))

        menu.addSeparator()

        act_txt = menu.addAction("✏️  Edit text")
        act_txt.triggered.connect(lambda: self._edit_ebook_fragment_text(frag))

        act_dur = menu.addAction("⏱  Edit duration / add silence")
        act_dur.setEnabled(has_audio)
        act_dur.setToolTip(
            "Add extra silence before and/or after this fragment during export/preview.\n"
            "Cannot set duration shorter than the actual audio."
        )
        act_dur.triggered.connect(lambda: self._edit_ebook_fragment_timing(frag))

        speaker = frag.get('speaker') or ""
        if speaker:
            act_spk = menu.addAction(f"✏️  Edit speaker  ({speaker})")
        else:
            act_spk = menu.addAction("➕  Add speaker")
        act_spk.triggered.connect(lambda: self._edit_ebook_speaker(frag))

        if speaker:
            act_clr = menu.addAction("✕  Remove speaker")
            act_clr.triggered.connect(lambda: self._edit_ebook_speaker(frag, clear=True))

        act_spk_sel = menu.addAction("✏️  Set speaker for selected")
        act_spk_sel.triggered.connect(self._set_speaker_for_selected_ebook)

        menu.addSeparator()

        pos_in_list = next(
            (i for i, f in enumerate(self._ebook_fragments) if f['index'] == idx), -1
        )

        act_add = menu.addAction("➕  Add fragment after")
        act_add.setEnabled(not self._is_running)
        act_add.triggered.connect(lambda: self._add_ebook_fragment_after(frag))

        act_del = menu.addAction("🗑  Remove fragment")
        act_del.setEnabled(not self._is_running)
        act_del.triggered.connect(lambda: self._remove_ebook_fragment(frag))

        act_up = menu.addAction("⬆  Move up")
        act_up.setEnabled(not self._is_running and pos_in_list > 0)
        act_up.triggered.connect(lambda: self._move_ebook_fragment(frag, -1))

        act_down = menu.addAction("⬇  Move down")
        act_down.setEnabled(
            not self._is_running and 0 <= pos_in_list < len(self._ebook_fragments) - 1
        )
        act_down.triggered.connect(lambda: self._move_ebook_fragment(frag, 1))

        menu.addSeparator()

        act_merge = menu.addAction("🔀  Merge fragments")
        act_merge.setEnabled(not self._is_running and len(self._ebook_fragments) > 1)
        act_merge.triggered.connect(lambda: self._merge_ebook_fragments(frag))

        menu.exec(self._ebook_tree.viewport().mapToGlobal(pos))
 
    def _edit_ebook_fragment_text(self, frag: Dict):
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Edit fragment #{frag['index'] + 1}")
        dlg.resize(560, 260)
        dlg.setStyleSheet(self.styleSheet())
 
        lay = QVBoxLayout(dlg)
        lay.setSpacing(8)
 
        info = QLabel(f"Speaker: {frag.get('speaker') or '—'}")
        info.setStyleSheet(f"color:{C['text3']};font-size:11px;")
        lay.addWidget(info)
 
        editor = QPlainTextEdit()
        editor.setFont(QFont("Segoe UI", 13))
        editor.setPlainText(frag.get('text', ''))
        editor.selectAll()
        lay.addWidget(editor, 1)
 
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)
        editor.setFocus()
 
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
 
        new_text = editor.toPlainText().strip()
        if not new_text or new_text == frag.get('text', ''):
            return
 
        frag['text']      = new_text
        frag['status']    = 'waiting'
        frag['error_msg'] = None
        self._update_ebook_tree_item(frag['index'])
        self._ebook_preview_text.setPlainText(new_text)
        self._set_status(f"Fragment #{frag['index'] + 1} text updated — status reset to waiting.")
 
    def _edit_ebook_speaker(self, frag: Dict, clear: bool = False):
        if clear:
            frag['speaker'] = None
            self._update_ebook_tree_item(frag['index'])
            self._sync_ebook_speaker_ui()
            return
 
        current = frag.get('speaker') or ""
        name, ok = QInputDialog.getText(
            self,
            "Speaker name",
            "Enter speaker name:",
            text=current,
        )
        if ok:
            frag['speaker'] = name.strip() or None
            self._update_ebook_tree_item(frag['index'])
            self._sync_ebook_speaker_ui()
 
    def _set_speaker_for_selected_ebook(self):
        if not self._ebook_fragments or not self._ebook_chapter_item:
            return

        checked_indices: set = set()
        for i in range(self._ebook_chapter_item.childCount()):
            child = self._ebook_chapter_item.child(i)
            if (not child.isHidden()
                    and child.checkState(COL_STATUS) == Qt.CheckState.Checked):
                idx = child.data(COL_STATUS, Qt.ItemDataRole.UserRole)
                if idx is not None:
                    checked_indices.add(idx)

        if not checked_indices:
            QMessageBox.information(
                self, "Nothing selected",
                "No fragments are checked.\nCheck at least one fragment first."
            )
            return

        checked_frags = [f for f in self._ebook_fragments if f['index'] in checked_indices]
        current = checked_frags[0].get('speaker') or ""

        name, ok = QInputDialog.getText(
            self,
            "Speaker name",
            f"Set speaker for {len(checked_frags)} selected fragment(s)\n"
            "(leave empty to clear / use default voice):",
            text=current,
        )
        if not ok:
            return

        new_speaker = name.strip() or None
        for frag in checked_frags:
            frag['speaker'] = new_speaker
            self._update_ebook_tree_item(frag['index'])

        self._sync_ebook_speaker_ui()
        self._set_status(
            f"Speaker {'set to ' + new_speaker if new_speaker else 'cleared'} "
            f"for {len(checked_frags)} fragment(s).",
            C["accent"],
        )
 
    def _edit_ebook_fragment_timing(self, frag: Dict):
        path = frag.get('output_path')
        if not path or not os.path.exists(path):
            QMessageBox.warning(
                self, "No audio",
                "Synthesize this fragment first before adjusting its duration."
            )
            return

        audio_dur_s = _get_wav_duration(path)
        if audio_dur_s is None:
            QMessageBox.warning(self, "Error", "Cannot read audio duration.")
            return

        audio_dur_ms     = int(audio_dur_s * 1000)
        current_target   = frag.get('target_duration_ms', audio_dur_ms)
        current_extra_ms = max(0, current_target - audio_dur_ms)
        current_pre_ms   = int(frag.get('pre_silence_ms') or 0)

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Edit duration — fragment #{frag['index'] + 1}")
        dlg.resize(400, 230)
        dlg.setStyleSheet(self.styleSheet())

        lay = QVBoxLayout(dlg)
        lay.setSpacing(10)
        lay.setContentsMargins(14, 14, 14, 10)

        audio_lbl = QLabel(f"Audio duration:  {_fmt_ms(audio_dur_ms)}")
        audio_lbl.setStyleSheet(f"color:{C['text2']};font-size:12px;")
        lay.addWidget(audio_lbl)

        pre_row = QHBoxLayout()
        pre_lbl = QLabel("Extra silence before (ms):")
        pre_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        pre_spin = QSpinBox()
        pre_spin.setRange(0, 120000)
        pre_spin.setValue(current_pre_ms)
        pre_spin.setSingleStep(100)
        pre_spin.setMinimumWidth(110)
        pre_spin.setToolTip(
            "Extra silence inserted before this fragment during audiobook export and preview.\n"
            "Useful for pauses before a new paragraph, title or chapter heading."
        )
        pre_row.addWidget(pre_lbl)
        pre_row.addWidget(pre_spin)
        pre_row.addStretch()
        lay.addLayout(pre_row)

        sil_row = QHBoxLayout()
        sil_lbl = QLabel("Extra silence after (ms):")
        sil_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        sil_spin = QSpinBox()
        sil_spin.setRange(0, 120000)
        sil_spin.setValue(current_extra_ms)
        sil_spin.setSingleStep(100)
        sil_spin.setMinimumWidth(110)
        sil_spin.setToolTip(
            "Extra silence appended after this fragment during audiobook export and preview.\n"
            "0 = no extra silence (use global silence setting only)."
        )
        sil_row.addWidget(sil_lbl)
        sil_row.addWidget(sil_spin)
        sil_row.addStretch()
        lay.addLayout(sil_row)

        total_lbl = QLabel(
            f"Total duration:  {_fmt_ms(current_pre_ms + audio_dur_ms + current_extra_ms)}"
        )
        total_lbl.setStyleSheet(f"color:{C['accent']};font-size:11px;font-weight:600;")
        lay.addWidget(total_lbl)

        def _update_total_lbl():
            total_lbl.setText(
                f"Total duration:  {_fmt_ms(pre_spin.value() + audio_dur_ms + sil_spin.value())}"
            )

        sil_spin.valueChanged.connect(lambda _v: _update_total_lbl())
        pre_spin.valueChanged.connect(lambda _v: _update_total_lbl())

        hint = QLabel(
            "Silence before/after is added during audiobook export/preview,\n"
            "in addition to the global 'Silence between fragments' setting."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{C['text3']};font-size:10px;font-style:italic;")
        lay.addWidget(hint)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        pre_ms = pre_spin.value()
        if pre_ms <= 0:
            frag.pop('pre_silence_ms', None)
        else:
            frag['pre_silence_ms'] = pre_ms

        extra_ms = sil_spin.value()
        if extra_ms <= 0:
            frag.pop('target_duration_ms', None)
        else:
            frag['target_duration_ms'] = audio_dur_ms + extra_ms

        self._update_ebook_tree_item(frag['index'])
        self._set_status(
            f"Fragment #{frag['index'] + 1} — audio: {_fmt_ms(audio_dur_ms)}, "
            f"silence before: {pre_ms} ms, silence after: {extra_ms} ms.",
            C["accent"],
        )
 
    def _sync_ebook_speaker_ui(self):
        if self._dubbing_video_path:
            return
 
        freq: Dict[str, int] = {}
        for f in self._ebook_fragments:
            spk = f.get("speaker")
            if spk and str(spk).strip():
                k = str(spk).strip()
                freq[k] = freq.get(k, 0) + 1
 
        speakers = sorted(freq.keys(), key=lambda s: freq[s], reverse=True)
 
        if not speakers:
            if self._dubbing_mode:
                self._dubbing_mode = False
                self._speaker_list = []
                self._speaker_voices.clear()
                while self._speakers_lay.count():
                    item = self._speakers_lay.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                self._speakers_container.setVisible(False)
                self._voice_single_container.setVisible(True)
                if hasattr(self, "_whisper_section_widget"):
                    self._whisper_section_widget.setVisible(True)
            return
 
        if set(speakers) == set(self._speaker_list) and self._dubbing_mode:
            return
 
        saved_paths: Dict[str, str] = {}
        saved_texts: Dict[str, str] = {}
        for spk, sv in self._speaker_voices.items():
            drop     = sv.get("drop")
            ref_text = sv.get("ref_text")
            if drop and drop.file_path:
                saved_paths[spk] = drop.file_path
            if ref_text:
                t = ref_text.toPlainText()
                if t:
                    saved_texts[spk] = t
 
        self._speaker_list = speakers
        self._dubbing_mode = True
        self._rebuild_voice_cloning_for_speakers()
 
        for spk, path in saved_paths.items():
            if spk in self._speaker_voices and os.path.exists(path):
                sv = self._speaker_voices[spk]
                if "drop" not in sv:
                    continue
                sv["drop"]._set(path)
                sv["player"].load(path)
                sv["proc_btn"].setEnabled(True)
                sv["tr_btn"].setEnabled(
                    self._whisper_backend is not None
                    and self._whisper_backend.is_downloaded(self._w_size.currentData())
                )
 
        for spk, text in saved_texts.items():
            if spk in self._speaker_voices and text:
                sv = self._speaker_voices[spk]
                if "ref_text" in sv:
                    sv["ref_text"].setPlainText(text)
 
        self._set_status(
            f"{len(speakers)} speaker(s) — add reference audio for each in Voice cloning.",
            C["accent"],
        )
 
    def _add_fragment_after(self, frag: Dict):
        dlg = QDialog(self)
        dlg.setWindowTitle("Add fragment")
        dlg.resize(520, 220)
        dlg.setStyleSheet(self.styleSheet())

        lay = QVBoxLayout(dlg)
        lay.setSpacing(8)

        lbl = QLabel("Enter text for the new fragment:")
        lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        lay.addWidget(lbl)

        editor = QPlainTextEdit()
        editor.setFont(QFont("Segoe UI", 13))
        lay.addWidget(editor, 1)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)
        editor.setFocus()

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        new_text = editor.toPlainText().strip()
        if not new_text:
            return

        pos = next((i for i, f in enumerate(self._fragments) if f['index'] == frag['index']), -1)
        if pos < 0:
            return

        end_ms = frag.get('end_ms', 0) or 0
        new_frag = {
            'index':       0,
            'srt_id':      '',
            'timestamp':   f"{_ms_to_ts(end_ms)} --> {_ms_to_ts(end_ms + 3000)}",
            'text':        new_text,
            'start_ms':    end_ms,
            'end_ms':      end_ms + 3000,
            'srt_end_ms':  end_ms + 3000,
            'speaker':     frag.get('speaker'),
            'status':      'waiting',
            'output_path': None,
            'error_msg':   None,
        }
        self._fragments.insert(pos + 1, new_frag)

        for i, f in enumerate(self._fragments):
            f['index']  = i
            f['srt_id'] = str(i + 1)

        self._rename_outputs_to_match_order()
        self._populate_tree()
        self._update_action_buttons()

        new_item = self._frag_items.get(pos + 1)
        if new_item:
            self._tree.setCurrentItem(new_item)
 
    def _remove_fragment(self, frag: Dict):
        checked_indices = set()
        if self._chapter_item:
            for i in range(self._chapter_item.childCount()):
                child = self._chapter_item.child(i)
                if child.checkState(COL_STATUS) == Qt.CheckState.Checked and not child.isHidden():
                    idx = child.data(COL_STATUS, Qt.ItemDataRole.UserRole)
                    if idx is not None:
                        checked_indices.add(idx)

        checked_count = len(checked_indices)
        is_checked    = frag['index'] in checked_indices

        dlg = QDialog(self)
        dlg.setWindowTitle("Remove fragment")
        dlg.setStyleSheet(self.styleSheet())
        dlg.resize(420, 160)
        lay = QVBoxLayout(dlg)
        lay.setSpacing(10)

        msg_lbl = QLabel(
            f"Remove fragment #{frag['index'] + 1}?\n\"{frag.get('text', '')[:70]}\""
        )
        msg_lbl.setWordWrap(True)
        msg_lbl.setStyleSheet(f"color:{C['text']};font-size:12px;")
        lay.addWidget(msg_lbl)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_this   = QPushButton("Remove this one")
        btn_all    = QPushButton(f"Remove all checked  ({checked_count})")
        btn_all.setEnabled(checked_count > 0 and is_checked)
        btn_all.setToolTip("Removes all fragments that have a checked checkbox")
        btn_cancel = QPushButton("Cancel")

        btn_this.setStyleSheet(_btn(C["error"]))
        btn_all.setStyleSheet(_btn(C["error"]))

        btn_row.addWidget(btn_this)
        btn_row.addWidget(btn_all)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        lay.addLayout(btn_row)

        result = {"action": None}
        btn_this.clicked.connect(lambda: (result.update({"action": "one"}), dlg.accept()))
        btn_all.clicked.connect(lambda: (result.update({"action": "all"}), dlg.accept()))
        btn_cancel.clicked.connect(dlg.reject)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        if result["action"] == "all":
            self._fragments = [f for f in self._fragments if f['index'] not in checked_indices]
        else:
            self._fragments = [f for f in self._fragments if f['index'] != frag['index']]

        for i, f in enumerate(self._fragments):
            f['index']  = i
            f['srt_id'] = str(i + 1)

        self._rename_outputs_to_match_order()
        self._populate_tree()
        self._update_action_buttons()
        self._set_status(f"Fragment(s) removed. {len(self._fragments)} fragments remaining.")
 
    def _rename_outputs_to_match_order(self):
        if not self._srt_path or not self._output_dir:
            return
        prefix = Path(self._srt_path).stem

        to_rename = [
            frag for frag in self._fragments
            if frag.get('output_path') and os.path.exists(frag['output_path'])
        ]
        if not to_rename:
            return

        desired = {
            frag['index']: os.path.join(
                self._output_dir, f"{prefix}_{frag['index'] + 1:03d}.wav"
            )
            for frag in to_rename
        }

        if all(frag['output_path'] == desired[frag['index']] for frag in to_rename):
            return

        for frag in to_rename:
            self._file_watcher.removePath(frag['output_path'])

        temp_map = {}
        for frag in to_rename:
            src = frag['output_path']
            tmp = src + '.__reorder__'
            try:
                os.rename(src, tmp)
                temp_map[frag['index']] = tmp
            except Exception as e:
                logger.warning(f"Reorder phase 1 failed for {src}: {e}")
                temp_map[frag['index']] = src

        for frag in to_rename:
            tmp = temp_map[frag['index']]
            dst = desired[frag['index']]
            if tmp == frag['output_path']:
                continue
            try:
                os.rename(tmp, dst)
                frag['output_path'] = dst
            except Exception as e:
                logger.warning(f"Reorder phase 2 failed {tmp} -> {dst}: {e}")
                try:
                    os.rename(tmp, frag['output_path'])
                except Exception:
                    pass

        for frag in to_rename:
            if frag.get('output_path') and os.path.exists(frag['output_path']):
                self._file_watcher.addPath(frag['output_path'])
 
    def _move_fragment(self, frag: Dict, direction: int):
        pos = next((i for i, f in enumerate(self._fragments) if f['index'] == frag['index']), -1)
        if pos < 0:
            return
        new_pos = pos + direction
        if new_pos < 0 or new_pos >= len(self._fragments):
            return

        self._fragments[pos], self._fragments[new_pos] = self._fragments[new_pos], self._fragments[pos]

        for i, f in enumerate(self._fragments):
            f['index']  = i
            f['srt_id'] = str(i + 1)

        self._rename_outputs_to_match_order()
        self._populate_tree()

        moved_item = self._frag_items.get(new_pos)
        if moved_item:
            self._tree.setCurrentItem(moved_item)
 
    def _merge_srt_fragments(self, frag: Dict):
        pos = next(
            (i for i, f in enumerate(self._fragments) if f['index'] == frag['index']), -1
        )
        if pos < 0:
            return

        checked_indices: set = set()
        if self._chapter_item:
            for i in range(self._chapter_item.childCount()):
                child = self._chapter_item.child(i)
                if (child.checkState(COL_STATUS) == Qt.CheckState.Checked
                        and not child.isHidden()):
                    idx = child.data(COL_STATUS, Qt.ItemDataRole.UserRole)
                    if idx is not None:
                        checked_indices.add(idx)

        can_merge_next    = pos + 1 < len(self._fragments)
        is_checked        = frag['index'] in checked_indices
        checked_count     = len(checked_indices)
        can_merge_checked = is_checked and checked_count >= 2

        dlg = QDialog(self)
        dlg.setWindowTitle("Merge fragments")
        dlg.resize(480, 220)
        dlg.setStyleSheet(self.styleSheet())
        lay = QVBoxLayout(dlg)
        lay.setSpacing(10)
        lay.setContentsMargins(14, 14, 14, 10)

        msg = QLabel(f"Fragment #{frag['index'] + 1}:  \"{frag.get('text', '')[:70]}\"")
        msg.setWordWrap(True)
        msg.setStyleSheet(f"color:{C['text']};font-size:12px;")
        lay.addWidget(msg)

        hint = QLabel(
            "Available audio will be concatenated. Fragments without audio are replaced by silence\n"
            "proportional to their time slot. Orphan audio files are deleted automatically."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{C['text3']};font-size:10px;font-style:italic;")
        lay.addWidget(hint)

        gap_chk = QCheckBox("Merge by timing gap")
        gap_chk.setStyleSheet(f"color:{C['text2']};font-size:12px;")
        gap_chk.setEnabled(not self._is_running)
        lay.addWidget(gap_chk)

        gap_container = QWidget()
        gap_row = QHBoxLayout(gap_container)
        gap_row.setContentsMargins(20, 0, 0, 0)
        gap_row.setSpacing(6)
        gap_lbl = QLabel("Max gap between fragments:")
        gap_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        gap_spin = QDoubleSpinBox()
        gap_spin.setRange(0.0, 60.0)
        gap_spin.setSingleStep(0.1)
        gap_spin.setDecimals(3)
        gap_spin.setValue(1.0)
        gap_spin.setSuffix(" s")
        gap_spin.setFixedWidth(100)
        gap_row.addWidget(gap_lbl)
        gap_row.addWidget(gap_spin)
        gap_row.addStretch()
        gap_container.setVisible(False)
        lay.addWidget(gap_container)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_next    = QPushButton("Merge with next fragment")
        btn_checked = QPushButton(f"Merge all checked  ({checked_count})")
        btn_cancel  = QPushButton("Cancel")

        btn_next.setEnabled(can_merge_next and not self._is_running)
        btn_checked.setEnabled(can_merge_checked and not self._is_running)
        btn_next.setStyleSheet(_btn(C["accent"]))
        btn_checked.setStyleSheet(_btn(C["accent"]))

        def _refresh_buttons(gap_mode: bool):
            gap_container.setVisible(gap_mode)
            btn_next.setEnabled(not gap_mode and can_merge_next and not self._is_running)
            if gap_mode:
                if checked_count:
                    btn_checked.setText(f"Merge by gap — checked  ({checked_count})")
                    btn_checked.setEnabled(not self._is_running)
                else:
                    btn_checked.setText(f"Merge by gap — all  ({len(self._fragments)})")
                    btn_checked.setEnabled(not self._is_running)
            else:
                btn_checked.setText(f"Merge all checked  ({checked_count})")
                btn_checked.setEnabled(can_merge_checked and not self._is_running)
            dlg.adjustSize()

        gap_chk.stateChanged.connect(
            lambda state: _refresh_buttons(state == Qt.CheckState.Checked.value)
        )

        result = {"action": None, "gap_ms": 1000}
        btn_next.clicked.connect(lambda: (result.update({"action": "next"}), dlg.accept()))
        btn_checked.clicked.connect(
            lambda: (
                result.update({
                    "action": "gap" if gap_chk.isChecked() else "checked",
                    "gap_ms": int(gap_spin.value() * 1000),
                }),
                dlg.accept(),
            )
        )
        btn_cancel.clicked.connect(dlg.reject)

        btn_row.addWidget(btn_next)
        btn_row.addWidget(btn_checked)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        lay.addLayout(btn_row)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        if result["action"] == "next" and can_merge_next:
            to_merge = [frag, self._fragments[pos + 1]]
            if len(to_merge) >= 2:
                self._do_merge_srt_fragments(to_merge)

        elif result["action"] == "checked" and can_merge_checked:
            to_merge = sorted(
                [f for f in self._fragments if f['index'] in checked_indices],
                key=lambda f: f['index'],
            )
            if len(to_merge) >= 2:
                self._do_merge_srt_fragments(to_merge)

        elif result["action"] == "gap":
            max_gap_ms = result["gap_ms"]
            if checked_indices:
                source_frags = sorted(
                    [f for f in self._fragments if f['index'] in checked_indices],
                    key=lambda f: f['index'],
                )
            else:
                source_frags = list(self._fragments)

            if not source_frags:
                return

            groups: List[List[Dict]] = []
            current_group = [source_frags[0]]
            for i in range(1, len(source_frags)):
                prev = source_frags[i - 1]
                curr = source_frags[i]
                gap  = (curr.get('start_ms', 0) or 0) - (prev.get('end_ms', 0) or 0)
                if gap < max_gap_ms:
                    current_group.append(curr)
                else:
                    if len(current_group) >= 2:
                        groups.append(current_group)
                    current_group = [curr]
            if len(current_group) >= 2:
                groups.append(current_group)

            if not groups:
                self._set_status("No fragment pairs found within the specified gap threshold.")
                return
            self._do_merge_gap_groups(groups)


    def _do_merge_srt_fragments(self, fragments: List[Dict]):
        if len(fragments) < 2:
            return

        self._stop_play()

        combined_text = " ".join(
            f.get('text', '').strip() for f in fragments if f.get('text', '').strip()
        )
        start_ms  = fragments[0].get('start_ms', 0) or 0
        end_ms    = fragments[-1].get('end_ms', 0) or 0
        srt_end   = fragments[-1].get('srt_end_ms', end_ms) or end_ms
        speaker   = fragments[0].get('speaker', '')

        target_sr     = 44100
        parts: List[np.ndarray] = []
        for df in fragments:
            ap = df.get('output_path', '')
            if df.get('status') == 'done' and ap and os.path.exists(ap):
                try:
                    audio, sr = sf.read(ap, dtype='float32')
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    if sr != target_sr:
                        t = torch.from_numpy(audio).unsqueeze(0)
                        t = TAF.resample(t, sr, target_sr)
                        audio = t.squeeze(0).numpy()
                    parts.append(audio.astype(np.float32))
                except Exception as e:
                    logger.warning(f"Merge: cannot load {ap}: {e}")

        merged_audio_path: Optional[str] = None
        merged_status = 'waiting'

        if parts:
            try:
                os.makedirs(self._output_dir, exist_ok=True)
                tmp_name  = f"__merge_tmp_{abs(hash(tuple(f['index'] for f in fragments))) % 10**9}.wav"
                tmp_path  = os.path.join(self._output_dir, tmp_name)
                combined  = np.concatenate(parts)
                sf.write(tmp_path, combined, target_sr, subtype='PCM_16')
                merged_audio_path = tmp_path
                merged_status     = 'done'
            except Exception as e:
                logger.warning(f"Merge audio write failed: {e}")

        paths_to_delete = set()
        for f in fragments:
            p = f.get('output_path', '')
            if p and os.path.exists(p):
                paths_to_delete.add(p)
                self._file_watcher.removePath(p)

        indices_to_remove = {f['index'] for f in fragments}
        insert_pos = min(
            i for i, f in enumerate(self._fragments) if f['index'] in indices_to_remove
        )

        merged_frag: Dict = {
            'index':       0,
            'srt_id':      '',
            'timestamp':   f"{_ms_to_ts(start_ms)} --> {_ms_to_ts(end_ms)}",
            'text':        combined_text,
            'start_ms':    start_ms,
            'end_ms':      end_ms,
            'srt_end_ms':  srt_end,
            'speaker':     speaker,
            'status':      merged_status,
            'output_path': merged_audio_path,
            'error_msg':   None,
        }

        self._fragments = [f for f in self._fragments if f['index'] not in indices_to_remove]
        self._fragments.insert(insert_pos, merged_frag)

        for i, f in enumerate(self._fragments):
            f['index']  = i
            f['srt_id'] = str(i + 1)

        for p in paths_to_delete:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as e:
                logger.warning(f"Could not delete orphan audio {p}: {e}")

        self._rename_outputs_to_match_order()

        if self._current_fragment_idx in indices_to_remove:
            self._current_fragment_idx = None
            self._audio_data = None
            if hasattr(self, '_wave_out'):
                self._wave_out.clear()
            self._play_btn.setEnabled(False)
            self._stop_btn.setEnabled(False)
            self._time_lbl.setText("0:00 / 0:00")

        self._populate_tree()
        self._update_action_buttons()
        self._set_status(
            f"Merged {len(fragments)} fragments into one  "
            f"({_fmt_ms(start_ms)} → {_fmt_ms(end_ms)}).",
            C["accent"],
        )
 
    def _do_merge_gap_groups(self, groups: List[List[Dict]]):
        if not groups:
            return

        self._stop_play()

        target_sr = 44100
        all_group_indices: set = set()
        for g in groups:
            for f in g:
                all_group_indices.add(f['index'])

        prepared: List[Dict] = []
        for group in groups:
            combined_text = " ".join(
                f.get('text', '').strip() for f in group if f.get('text', '').strip()
            )
            start_ms = group[0].get('start_ms', 0) or 0
            end_ms   = group[-1].get('end_ms', 0) or 0
            srt_end  = group[-1].get('srt_end_ms', end_ms) or end_ms
            speaker  = group[0].get('speaker', '')

            parts: List[np.ndarray] = []
            for df in group:
                ap = df.get('output_path', '')
                if df.get('status') == 'done' and ap and os.path.exists(ap):
                    try:
                        audio, sr = sf.read(ap, dtype='float32')
                        if audio.ndim > 1:
                            audio = audio.mean(axis=1)
                        if sr != target_sr:
                            t = torch.from_numpy(audio).unsqueeze(0)
                            t = TAF.resample(t, sr, target_sr)
                            audio = t.squeeze(0).numpy()
                        parts.append(audio.astype(np.float32))
                    except Exception as e:
                        logger.warning(f"Gap merge: cannot load {ap}: {e}")

            merged_audio_path: Optional[str] = None
            merged_status = 'waiting'
            if parts:
                try:
                    os.makedirs(self._output_dir, exist_ok=True)
                    tmp_name = f"__merge_tmp_{abs(hash(tuple(f['index'] for f in group))) % 10**9}.wav"
                    tmp_path = os.path.join(self._output_dir, tmp_name)
                    sf.write(tmp_path, np.concatenate(parts), target_sr, subtype='PCM_16')
                    merged_audio_path = tmp_path
                    merged_status     = 'done'
                except Exception as e:
                    logger.warning(f"Gap merge audio write failed: {e}")

            group_indices = {f['index'] for f in group}
            leader_idx    = min(group_indices)
            prepared.append({
                'group_indices': group_indices,
                'leader_idx':    leader_idx,
                'frag': {
                    'index':       0,
                    'srt_id':      '',
                    'timestamp':   f"{_ms_to_ts(start_ms)} --> {_ms_to_ts(end_ms)}",
                    'text':        combined_text,
                    'start_ms':    start_ms,
                    'end_ms':      end_ms,
                    'srt_end_ms':  srt_end,
                    'speaker':     speaker,
                    'status':      merged_status,
                    'output_path': merged_audio_path,
                    'error_msg':   None,
                },
            })

        leader_map: Dict[int, Dict] = {p['leader_idx']: p for p in prepared}

        for f in self._fragments:
            if f['index'] in all_group_indices:
                p = f.get('output_path', '')
                if p and os.path.exists(p):
                    self._file_watcher.removePath(p)

        new_fragments: List[Dict] = []
        for f in self._fragments:
            orig_idx = f['index']
            if orig_idx in all_group_indices:
                if orig_idx in leader_map:
                    new_fragments.append(leader_map[orig_idx]['frag'])
                p = f.get('output_path', '')
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception as e:
                        logger.warning(f"Could not delete orphan audio {p}: {e}")
            else:
                new_fragments.append(f)

        self._fragments = new_fragments
        for i, f in enumerate(self._fragments):
            f['index']  = i
            f['srt_id'] = str(i + 1)

        for p_item in prepared:
            ap = p_item['frag'].get('output_path', '')
            if ap and os.path.exists(ap):
                self._file_watcher.addPath(ap)

        if self._current_fragment_idx in all_group_indices:
            self._current_fragment_idx = None
            self._audio_data = None
            if hasattr(self, '_wave_out'):
                self._wave_out.clear()
            self._play_btn.setEnabled(False)
            self._stop_btn.setEnabled(False)
            self._time_lbl.setText("0:00 / 0:00")

        self._rename_outputs_to_match_order()
        self._populate_tree()
        self._update_action_buttons()
        self._set_status(
            f"Merged {len(all_group_indices)} fragments into {len(prepared)} groups by gap.",
            C["accent"],
        )
 
    def _add_ebook_fragment_after(self, frag: Dict):
        dlg = QDialog(self)
        dlg.setWindowTitle("Add fragment")
        dlg.resize(520, 220)
        dlg.setStyleSheet(self.styleSheet())
 
        lay = QVBoxLayout(dlg)
        lay.setSpacing(8)
 
        lbl = QLabel("Enter text for the new fragment:")
        lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        lay.addWidget(lbl)
 
        editor = QPlainTextEdit()
        editor.setFont(QFont("Segoe UI", 13))
        lay.addWidget(editor, 1)
 
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)
        editor.setFocus()
 
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
 
        new_text = editor.toPlainText().strip()
        if not new_text:
            return
 
        pos = next((i for i, f in enumerate(self._ebook_fragments) if f['index'] == frag['index']), -1)
        if pos < 0:
            return
 
        new_frag = {
            'index':       0,
            'text':        new_text,
            'speaker':     frag.get('speaker'),
            'status':      'waiting',
            'output_path': None,
            'error_msg':   None,
        }
        self._ebook_fragments.insert(pos + 1, new_frag)
 
        for i, f in enumerate(self._ebook_fragments):
            f['index'] = i
 
        self._populate_ebook_tree()
        self._update_action_buttons()
 
        new_item = self._ebook_frag_items.get(pos + 1)
        if new_item:
            self._ebook_tree.setCurrentItem(new_item)
 
    def _remove_ebook_fragment(self, frag: Dict):
        checked_indices = set()
        if self._ebook_chapter_item:
            for i in range(self._ebook_chapter_item.childCount()):
                child = self._ebook_chapter_item.child(i)
                if child.checkState(COL_STATUS) == Qt.CheckState.Checked and not child.isHidden():
                    idx = child.data(COL_STATUS, Qt.ItemDataRole.UserRole)
                    if idx is not None:
                        checked_indices.add(idx)
 
        checked_count = len(checked_indices)
        is_checked    = frag['index'] in checked_indices
 
        dlg = QDialog(self)
        dlg.setWindowTitle("Remove fragment")
        dlg.setStyleSheet(self.styleSheet())
        dlg.resize(420, 160)
        lay = QVBoxLayout(dlg)
        lay.setSpacing(10)
 
        msg_lbl = QLabel(
            f"Remove fragment #{frag['index'] + 1}?\n\"{frag.get('text', '')[:70]}\""
        )
        msg_lbl.setWordWrap(True)
        msg_lbl.setStyleSheet(f"color:{C['text']};font-size:12px;")
        lay.addWidget(msg_lbl)
 
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_this   = QPushButton("Remove this one")
        btn_all    = QPushButton(f"Remove all checked  ({checked_count})")
        btn_all.setEnabled(checked_count > 0 and is_checked)
        btn_all.setToolTip("Removes all fragments that have a checked checkbox")
        btn_cancel = QPushButton("Cancel")
 
        btn_this.setStyleSheet(_btn(C["error"]))
        btn_all.setStyleSheet(_btn(C["error"]))
 
        btn_row.addWidget(btn_this)
        btn_row.addWidget(btn_all)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        lay.addLayout(btn_row)
 
        result = {"action": None}
        btn_this.clicked.connect(lambda: (result.update({"action": "one"}), dlg.accept()))
        btn_all.clicked.connect(lambda: (result.update({"action": "all"}), dlg.accept()))
        btn_cancel.clicked.connect(dlg.reject)
 
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
 
        if result["action"] == "all":
            self._ebook_fragments = [f for f in self._ebook_fragments if f['index'] not in checked_indices]
        else:
            self._ebook_fragments = [f for f in self._ebook_fragments if f['index'] != frag['index']]
 
        for i, f in enumerate(self._ebook_fragments):
            f['index'] = i
 
        self._populate_ebook_tree()
        self._update_action_buttons()
        self._set_status(f"Fragment(s) removed. {len(self._ebook_fragments)} fragments remaining.")
 
    def _move_ebook_fragment(self, frag: Dict, direction: int):
        pos = next((i for i, f in enumerate(self._ebook_fragments) if f['index'] == frag['index']), -1)
        if pos < 0:
            return
        new_pos = pos + direction
        if new_pos < 0 or new_pos >= len(self._ebook_fragments):
            return
 
        self._ebook_fragments[pos], self._ebook_fragments[new_pos] = \
            self._ebook_fragments[new_pos], self._ebook_fragments[pos]
 
        for i, f in enumerate(self._ebook_fragments):
            f['index'] = i
 
        self._populate_ebook_tree()
 
        moved_item = self._ebook_frag_items.get(new_pos)
        if moved_item:
            self._ebook_tree.setCurrentItem(moved_item)

    def _merge_ebook_fragments(self, frag: Dict):
        pos = next(
            (i for i, f in enumerate(self._ebook_fragments) if f['index'] == frag['index']), -1
        )
        if pos < 0:
            return

        checked_indices: set = set()
        if self._ebook_chapter_item:
            for i in range(self._ebook_chapter_item.childCount()):
                child = self._ebook_chapter_item.child(i)
                if (child.checkState(COL_STATUS) == Qt.CheckState.Checked
                        and not child.isHidden()):
                    idx = child.data(COL_STATUS, Qt.ItemDataRole.UserRole)
                    if idx is not None:
                        checked_indices.add(idx)

        can_merge_next    = pos + 1 < len(self._ebook_fragments)
        is_checked        = frag['index'] in checked_indices
        checked_count     = len(checked_indices)
        can_merge_checked = is_checked and checked_count >= 2

        dlg = QDialog(self)
        dlg.setWindowTitle("Merge fragments")
        dlg.resize(460, 190)
        dlg.setStyleSheet(self.styleSheet())
        lay = QVBoxLayout(dlg)
        lay.setSpacing(10)
        lay.setContentsMargins(14, 14, 14, 10)

        msg = QLabel(f"Fragment #{frag['index'] + 1}:  \"{frag.get('text', '')[:70]}\"")
        msg.setWordWrap(True)
        msg.setStyleSheet(f"color:{C['text']};font-size:12px;")
        lay.addWidget(msg)

        hint = QLabel(
            "Available audio will be concatenated in order. Fragments without audio are skipped.\n"
            "Orphan audio files are deleted automatically."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{C['text3']};font-size:10px;font-style:italic;")
        lay.addWidget(hint)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_next    = QPushButton("Merge with next fragment")
        btn_checked = QPushButton(f"Merge all checked  ({checked_count})")
        btn_cancel  = QPushButton("Cancel")

        btn_next.setEnabled(can_merge_next and not self._is_running)
        btn_checked.setEnabled(can_merge_checked and not self._is_running)
        btn_next.setStyleSheet(_btn(C["accent"]))
        btn_checked.setStyleSheet(_btn(C["accent"]))

        result = {"action": None}
        btn_next.clicked.connect(lambda: (result.update({"action": "next"}), dlg.accept()))
        btn_checked.clicked.connect(lambda: (result.update({"action": "checked"}), dlg.accept()))
        btn_cancel.clicked.connect(dlg.reject)

        btn_row.addWidget(btn_next)
        btn_row.addWidget(btn_checked)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        lay.addLayout(btn_row)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        if result["action"] == "next" and can_merge_next:
            to_merge = [frag, self._ebook_fragments[pos + 1]]
        elif result["action"] == "checked" and can_merge_checked:
            to_merge = sorted(
                [f for f in self._ebook_fragments if f['index'] in checked_indices],
                key=lambda f: f['index'],
            )
        else:
            return

        if len(to_merge) < 2:
            return

        self._do_merge_ebook_fragments(to_merge)


    def _do_merge_ebook_fragments(self, fragments: List[Dict]):
        if len(fragments) < 2:
            return

        self._stop_play()

        combined_text = " ".join(
            f.get('text', '').strip() for f in fragments if f.get('text', '').strip()
        )
        speaker = fragments[0].get('speaker', '')

        target_sr     = 44100
        parts: List[np.ndarray] = []
        for df in fragments:
            ap = df.get('output_path', '')
            if df.get('status') == 'done' and ap and os.path.exists(ap):
                try:
                    audio, sr = sf.read(ap, dtype='float32')
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    if sr != target_sr:
                        t = torch.from_numpy(audio).unsqueeze(0)
                        t = TAF.resample(t, sr, target_sr)
                        audio = t.squeeze(0).numpy()
                    parts.append(audio.astype(np.float32))
                except Exception as e:
                    logger.warning(f"Ebook merge: cannot load {ap}: {e}")

        merged_audio_path: Optional[str] = None
        merged_status = 'waiting'

        if parts:
            try:
                os.makedirs(self._ebook_output_dir, exist_ok=True)
                tmp_name  = f"__merge_tmp_{abs(hash(tuple(f['index'] for f in fragments))) % 10**9}.wav"
                tmp_path  = os.path.join(self._ebook_output_dir, tmp_name)
                combined  = np.concatenate(parts)
                sf.write(tmp_path, combined, target_sr, subtype='PCM_16')
                merged_audio_path = tmp_path
                merged_status     = 'done'
            except Exception as e:
                logger.warning(f"Ebook merge audio write failed: {e}")

        for f in fragments:
            p = f.get('output_path', '')
            if p and os.path.exists(p):
                self._file_watcher.removePath(p)
                try:
                    os.remove(p)
                except Exception as e:
                    logger.warning(f"Could not delete orphan audio {p}: {e}")

        indices_to_remove = {f['index'] for f in fragments}
        insert_pos = min(
            i for i, f in enumerate(self._ebook_fragments) if f['index'] in indices_to_remove
        )

        merged_frag: Dict = {
            'index':       0,
            'text':        combined_text,
            'speaker':     speaker,
            'status':      merged_status,
            'output_path': merged_audio_path,
            'error_msg':   None,
        }

        self._ebook_fragments = [
            f for f in self._ebook_fragments if f['index'] not in indices_to_remove
        ]
        self._ebook_fragments.insert(insert_pos, merged_frag)

        for i, f in enumerate(self._ebook_fragments):
            f['index'] = i

        if merged_audio_path and os.path.exists(merged_audio_path):
            self._file_watcher.addPath(merged_audio_path)

        if self._current_fragment_idx in indices_to_remove:
            self._current_fragment_idx = None
            self._audio_data = None
            if hasattr(self, '_wave_out'):
                self._wave_out.clear()
            self._play_btn.setEnabled(False)
            self._stop_btn.setEnabled(False)
            self._time_lbl.setText("0:00 / 0:00")

        self._populate_ebook_tree()
        self._update_action_buttons()
        self._update_preview_btn_state()

        self._set_status(
            f"Merged {len(fragments)} ebook fragments into one.", C["accent"]
        )

    def _populate_tree(self):
        self._tree.clear()
        self._frag_items.clear()

        if not self._fragments:
            return

        srt_name = Path(self._srt_path).name if self._srt_path else "SRT"
        ch_item = QTreeWidgetItem()
        ch_item.setText(COL_FRAGMENT, f"📄  {srt_name}")
        ch_item.setFont(COL_FRAGMENT, QFont("Segoe UI", 12, QFont.Weight.Bold))
        ch_item.setForeground(COL_FRAGMENT, QColor(C["accent"]))
        self._chapter_item = ch_item

        self._tree.setUpdatesEnabled(False)
        try:
            children = []
            for frag in self._fragments:
                item = self._create_tree_item(frag, parent=None)
                children.append(item)
            ch_item.addChildren(children)
            self._tree.addTopLevelItem(ch_item)
            ch_item.setExpanded(True)
        finally:
            self._tree.setUpdatesEnabled(True)

        self._tree.resizeColumnToContents(COL_TIMING)

    def _create_tree_item(self, frag: Dict, parent=None) -> QTreeWidgetItem:
        status = frag.get('status', 'waiting')
        icon = {
            'waiting': STATUS_WAITING,
            'running': STATUS_RUNNING,
            'done':    STATUS_DONE,
            'error':   STATUS_ERROR,
        }.get(status, STATUS_WAITING)

        num = frag['index'] + 1

        item = QTreeWidgetItem(parent) if parent is not None else QTreeWidgetItem()
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(COL_STATUS, Qt.CheckState.Checked)
        item.setText(COL_STATUS,   "")
        item.setText(COL_FRAGMENT, f"{icon}  #{num}  {frag.get('text', '')[:75]}")
        item.setText(COL_SPEAKER,  frag.get('speaker') or "")

        start_ms = frag.get('start_ms', 0) or 0
        end_ms   = frag.get('end_ms',   0) or 0
        srt_end  = frag.get('srt_end_ms', end_ms)
        timing_text = f"{_fmt_ms(start_ms)} → {_fmt_ms(end_ms)}"

        if status == 'done':
            dur = _get_wav_duration(frag.get('output_path', ''))
            if dur is not None:
                slot_s   = (srt_end - start_ms) / 1000.0
                diff     = dur - slot_s
                diff_str = f" ({'+' if diff > 0 else ''}{diff:.1f}s)"
                timing_text += diff_str
                if dur <= slot_s:
                    timing_color = "#55bb55"
                elif dur < 2 * max(slot_s, 0.001):
                    timing_color = "#dddd00"
                elif dur < 4 * max(slot_s, 0.001):
                    timing_color = "#ff8800"
                else:
                    timing_color = "#ff4444"
                item.setForeground(COL_TIMING, QColor(timing_color))
        elif status == 'error':
            item.setForeground(COL_FRAGMENT, QColor(C["error"]))

        item.setText(COL_TIMING, timing_text)
        item.setData(COL_STATUS, Qt.ItemDataRole.UserRole, frag['index'])
        self._frag_items[frag['index']] = item
        return item

    def _update_tree_item(self, idx: int, known_dur_s: Optional[float] = None):
        item = self._frag_items.get(idx)
        if not item:
            return
        frag = next((f for f in self._fragments if f['index'] == idx), None)
        if not frag:
            return
        status = frag.get('status', 'waiting')
        icon = {
            'waiting': STATUS_WAITING,
            'running': STATUS_RUNNING,
            'done':    STATUS_DONE,
            'error':   STATUS_ERROR,
        }.get(status, STATUS_WAITING)

        num        = frag['index'] + 1
        base_text  = frag.get('text', '')
        prefix     = (frag.get('prefix') or '').strip()
        suffix     = (frag.get('suffix') or '').strip()
        parts      = [x for x in [prefix, base_text, suffix] if x]
        display    = " ".join(parts)
        item.setText(COL_FRAGMENT, f"{icon}  #{num}  {display[:75]}")
        item.setText(COL_SPEAKER, frag.get('speaker') or "")
        item.setForeground(COL_FRAGMENT, QColor(C["text"]))

        start_ms = frag.get('start_ms', 0) or 0
        end_ms   = frag.get('end_ms',   0) or 0
        srt_end  = frag.get('srt_end_ms', end_ms)
        timing_text = f"{_fmt_ms(start_ms)} → {_fmt_ms(end_ms)}"

        if status == 'done':
            dur = known_dur_s if known_dur_s is not None else _get_wav_duration(frag.get('output_path', ''))
            if dur is not None:
                slot_s   = (srt_end - start_ms) / 1000.0
                diff     = dur - slot_s
                diff_str = f" ({'+' if diff > 0 else ''}{diff:.1f}s)"
                timing_text += diff_str
                if dur <= slot_s:
                    timing_color = "#55bb55"
                elif dur < 2 * max(slot_s, 0.001):
                    timing_color = "#dddd00"
                elif dur < 4 * max(slot_s, 0.001):
                    timing_color = "#ff8800"
                else:
                    timing_color = "#ff4444"
                item.setForeground(COL_TIMING, QColor(timing_color))
            else:
                item.setForeground(COL_TIMING, QColor(C["text2"]))
        elif status == 'error':
            item.setForeground(COL_FRAGMENT, QColor(C["error"]))
            item.setForeground(COL_TIMING, QColor(C["text2"]))
        else:
            item.setForeground(COL_TIMING, QColor(C["text2"]))

        item.setText(COL_TIMING, timing_text)

    def _apply_filter(self, text: str):
        if not self._chapter_item:
            return
        txt = text.lower().strip()
        for i in range(self._chapter_item.childCount()):
            child = self._chapter_item.child(i)
            matches = (not txt or
                       txt in child.text(COL_FRAGMENT).lower() or
                       txt in child.text(COL_SPEAKER).lower())
            child.setHidden(not matches)

    def _select_all(self, state: bool):
        if not self._chapter_item:
            return
        cs = Qt.CheckState.Checked if state else Qt.CheckState.Unchecked
        self._tree.setUpdatesEnabled(False)
        self._tree.blockSignals(True)
        try:
            for i in range(self._chapter_item.childCount()):
                child = self._chapter_item.child(i)
                if not child.isHidden():
                    child.setCheckState(COL_STATUS, cs)
        finally:
            self._tree.blockSignals(False)
            self._tree.setUpdatesEnabled(True)

    def _select_failed(self):
        if not self._chapter_item:
            return
        frag_map = {f['index']: f for f in self._fragments}
        self._tree.setUpdatesEnabled(False)
        self._tree.blockSignals(True)
        try:
            for i in range(self._chapter_item.childCount()):
                child = self._chapter_item.child(i)
                idx  = child.data(COL_STATUS, Qt.ItemDataRole.UserRole)
                frag = frag_map.get(idx)
                cs   = Qt.CheckState.Checked if (frag and frag.get('status') == 'error') else Qt.CheckState.Unchecked
                child.setCheckState(COL_STATUS, cs)
        finally:
            self._tree.blockSignals(False)
            self._tree.setUpdatesEnabled(True)

    def _show_timing_issues_dialog(self):
        dlg = TimingIssuesDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        select_yellow = dlg.yellow_check.isChecked()
        select_orange = dlg.orange_check.isChecked()
        select_red    = dlg.red_check.isChecked()

        if not any([select_yellow, select_orange, select_red]):
            return

        if not self._chapter_item:
            return

        frag_map = {f['index']: f for f in self._fragments}
        self._tree.setUpdatesEnabled(False)
        self._tree.blockSignals(True)
        try:
            for i in range(self._chapter_item.childCount()):
                child = self._chapter_item.child(i)
                if child.isHidden():
                    continue
                idx = child.data(COL_STATUS, Qt.ItemDataRole.UserRole)
                if idx is None:
                    continue
                frag = frag_map.get(idx)
                if not frag or frag.get('status') != 'done':
                    continue

                dur = _get_wav_duration(frag.get('output_path', ''))
                if dur is None:
                    continue

                slot_s = ((frag.get('end_ms', 0) or 0) - (frag.get('start_ms', 0) or 0)) / 1000.0
                if slot_s <= 0:
                    continue

                ratio = dur / max(slot_s, 0.001)

                color_match = False
                if select_yellow and 1.0 < ratio < 2.0:
                    color_match = True
                elif select_orange and 2.0 <= ratio < 4.0:
                    color_match = True
                elif select_red and ratio >= 4.0:
                    color_match = True

                if color_match:
                    child.setCheckState(COL_STATUS, Qt.CheckState.Checked)
        finally:
            self._tree.blockSignals(False)
            self._tree.setUpdatesEnabled(True)

    def _on_selection_changed(self):
        items = self._tree.selectedItems()
        if not items:
            return
        item = items[0]
        idx = item.data(COL_STATUS, Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        self._load_fragment_audio(idx)
        frag = next((f for f in self._fragments if f['index'] == idx), None)
        if not frag:
            return
        self._preview_text.setPlainText(frag.get('text', ''))
        if hasattr(self, '_video_player') and self._video_player.isVisible():
            start_s = frag.get('start_ms', 0) / 1000.0
            end_s = frag.get('end_ms', 0) / 1000.0
            self._video_player.set_selection_by_time(start_s, end_s)

    def _on_tree_item_double_clicked(self, item, column):
        idx = item.data(COL_STATUS, Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        if self._load_fragment_audio(idx):
            self._start_play()

    def _on_ebook_tree_item_double_clicked(self, item, column):
        idx = item.data(COL_STATUS, Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        if self._load_fragment_audio(idx):
            self._start_play()

    def _get_checked_fragments(self, retry_errors: bool = False) -> List[Dict]:
        result = []
        if not self._chapter_item:
            return result
        for i in range(self._chapter_item.childCount()):
            child = self._chapter_item.child(i)
            if child.checkState(COL_STATUS) != Qt.CheckState.Checked:
                continue
            if child.isHidden():
                continue
            idx  = child.data(COL_STATUS, Qt.ItemDataRole.UserRole)
            frag = next((f for f in self._fragments if f['index'] == idx), None)
            if not frag:
                continue
            if retry_errors:
                if frag.get('status') not in ('waiting', 'error'):
                    continue
            result.append(frag)
        return result

    def _start_synthesis(self, retry_errors: bool = False):
        if hasattr(self, '_tabs') and self._tabs.currentIndex() == 1:
            self._start_ebook_synthesis(retry_errors=retry_errors)
            return

        if not self._backend or not self._backend.is_loaded:
            QMessageBox.warning(self, "Model not loaded", "Load the model first.")
            return

        to_process = self._get_checked_fragments(retry_errors=retry_errors)
        if not to_process:
            QMessageBox.information(self, "Nothing to synthesize", "No fragments selected.")
            return

        already_done = [f for f in to_process if f.get('status') == 'done']
        if already_done:
            reply = QMessageBox.question(
                self,
                "Re-synthesize completed fragments?",
                f"{len(already_done)} fragment(s) already have generated audio.\n\n"
                "Do you want to re-synthesize them anyway?\n"
                "Existing audio files will be overwritten.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.No:
                to_process = [f for f in to_process if f.get('status') != 'done']
                if not to_process:
                    QMessageBox.information(self, "Nothing to synthesize",
                        "No pending fragments left to process.")
                    return

        for frag in to_process:
            frag["status"] = "waiting"
            frag["error_msg"] = None
            self._update_tree_item(frag["index"])

        self._synthesis_source = 'srt'
        self._is_running = True
        self._completed_count = 0
        self._synth_start_time = time.monotonic()
        self._synth_total = len(to_process)
        self._update_action_buttons()

        total = self._synth_total
        self._synth_progress.setMaximum(total)
        self._synth_progress.setValue(0)
        self._synth_progress.setVisible(True)
        self._eta_label.setText(f"0/{total}")
        self._eta_label.setVisible(True)

        reserved_paths = {f['output_path'] for f in self._fragments if f.get('output_path')}
        self._worker = TTSWorker(
            backend=self._backend,
            fragments=to_process,
            output_dir=self._output_dir,
            reference_audio=self._get_ref_audio(),
            reference_text=self._get_ref_text(),
            filename_prefix=Path(self._srt_path).stem if self._srt_path else "fragment",
            generation_settings=self._get_generation_settings(),
            normalize_audio=self._norm_check.isChecked(),
            speaker_voices=self._get_speaker_voices_dict(),
            reserved_paths=reserved_paths,
        )

        self._worker.progress.connect(self._on_progress)
        self._worker.item_done.connect(self._on_item_done)
        self._worker.finished.connect(self._on_synthesis_finished)
        self._worker.start()

        self._set_status(f"Synthesis started — {total} fragments queued…")
        logger.info(f"Synthesis started: {total} fragments")

    def _stop_synthesis(self):
        if self._worker:
            self._worker.request_cancel()
        self._set_status("Stop requested — waiting for current fragment to finish…", C["warning"])

    def _on_progress(self, idx: int, msg: str, is_error: bool):
        if self._synthesis_source == 'ebook':
            if idx >= 0:
                frag = next((f for f in self._ebook_fragments if f['index'] == idx), None)
                if frag:
                    frag['status'] = 'running' if not is_error else 'error'
                    self._update_ebook_tree_item(idx)
        else:
            if idx >= 0:
                frag = next((f for f in self._fragments if f['index'] == idx), None)
                if frag:
                    frag['status'] = 'running' if not is_error else 'error'
                    self._update_tree_item(idx)
        color = C["error"] if is_error else C["text2"]
        self._set_status(msg, color)

    def _on_item_done(self, idx: int, result: str, is_error: bool):
        if self._synthesis_source == 'ebook':
            frag = next((f for f in self._ebook_fragments if f['index'] == idx), None)
            if not frag:
                return
            if is_error:
                frag['status']    = 'error'
                frag['error_msg'] = result
            else:
                frag['status']      = 'done'
                frag['output_path'] = result
                frag['error_msg']   = None
                item = self._ebook_frag_items.get(idx)
                if item:
                    item.setCheckState(COL_STATUS, Qt.CheckState.Unchecked)
                if result and os.path.exists(result):
                    watched = self._file_watcher.files()
                    if result not in watched:
                        self._file_watcher.addPath(result)

            self._update_ebook_tree_item(idx)
            
            self._update_preview_btn_state()

            if not is_error and frag.get('output_path') and os.path.exists(frag.get('output_path', '')):
                selected = self._ebook_tree.selectedItems()
                if selected:
                    sel_idx = selected[0].data(COL_STATUS, Qt.ItemDataRole.UserRole)
                    if sel_idx == idx:
                        try:
                            audio, sr = sf.read(frag['output_path'], dtype="float32")
                            if audio.ndim > 1:
                                audio = audio.mean(axis=1)
                            self._load_audio_to_player(audio, sr, fragment_idx=idx)
                        except Exception as e:
                            self._set_status(f"Cannot load audio: {e}", C["error"])

        else:
            frag = next((f for f in self._fragments if f['index'] == idx), None)
            if not frag:
                return
            if is_error:
                frag['status']    = 'error'
                frag['error_msg'] = result
            else:
                frag['status']      = 'done'
                frag['output_path'] = result
                frag['error_msg']   = None
                item = self._frag_items.get(idx)
                if item:
                    item.setCheckState(COL_STATUS, Qt.CheckState.Unchecked)
                if result and os.path.exists(result):
                    watched = self._file_watcher.files()
                    if result not in watched:
                        self._file_watcher.addPath(result)

            self._update_tree_item(idx)

            if not is_error and frag.get('output_path') and os.path.exists(frag.get('output_path', '')):
                selected = self._tree.selectedItems()
                if selected:
                    sel_idx = selected[0].data(COL_STATUS, Qt.ItemDataRole.UserRole)
                    if sel_idx == idx:
                        try:
                            audio, sr = sf.read(frag['output_path'], dtype="float32")
                            if audio.ndim > 1:
                                audio = audio.mean(axis=1)
                            self._load_audio_to_player(audio, sr, fragment_idx=idx)
                        except Exception as e:
                            self._set_status(f"Cannot load audio: {e}", C["error"])

        self._completed_count += 1
        self._synth_progress.setValue(self._completed_count)

        total     = getattr(self, '_synth_total', 0)
        remaining = total - self._completed_count
        if remaining > 0 and getattr(self, '_synth_start_time', None) is not None:
            elapsed = time.monotonic() - self._synth_start_time
            avg     = elapsed / self._completed_count
            eta_s   = avg * remaining
            self._eta_label.setText(
                f"{self._completed_count}/{total}  ~{_fmt(eta_s)} left"
            )
        else:
            self._eta_label.setText(f"{self._completed_count}/{total}")

    def _on_audio_file_changed(self, path: str):
        if not os.path.exists(path):
            return
        self._file_watcher.addPath(path)
        dur = _get_wav_duration(path)
        if dur is None:
            return
        for frag in self._fragments:
            if frag.get('output_path') == path and frag.get('status') == 'done':
                self._update_tree_item(frag['index'], known_dur_s=dur)
                return
        for frag in self._ebook_fragments:
            if frag.get('output_path') == path and frag.get('status') == 'done':
                self._update_ebook_tree_item(frag['index'])
                return

    def _on_synthesis_finished(self):
        self._is_running = False
        self._synth_progress.setVisible(False)
        self._eta_label.setVisible(False)

        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()

        if self._synthesis_source == 'ebook':
            done  = sum(1 for f in self._ebook_fragments if f.get("status") == "done")
            error = sum(1 for f in self._ebook_fragments if f.get("status") == "error")
        else:
            done  = sum(1 for f in self._fragments if f.get("status") == "done")
            error = sum(1 for f in self._fragments if f.get("status") == "error")

        self._set_status(
            f"Synthesis complete — {done} done, {error} errors.",
            C["success"] if error == 0 else C["warning"],
        )
        self._update_action_buttons()
        logger.info(f"Synthesis finished: {done} done, {error} errors")

        if self._synthesis_source == 'ebook':
            auto = self._auto_ebook_session_path()
            if auto:
                if self._write_ebook_session_to(str(auto)):
                    logger.info(f"Auto-saved ebook session: {auto}")
        else:
            auto = self._auto_session_path()
            if auto:
                if self._write_session_to(str(auto)):
                    logger.info(f"Auto-saved session: {auto}")

    def _fit_audio_to_slot(self, audio_path: str, slot_ms: int) -> Optional[str]:
        dur = _get_wav_duration(audio_path)
        if dur is None:
            return None
        slot_s = slot_ms / 1000.0
        if dur <= slot_s:
            return None
        ratio = min(dur / slot_s, 4.0)
        filters   = []
        remaining = ratio
        while remaining > 2.0:
            filters.append("atempo=2.0")
            remaining /= 2.0
        filters.append(f"atempo={remaining:.6f}")
        tmp = audio_path.replace(".wav", "_fitted.wav")
        try:
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, "-af", ",".join(filters), tmp],
                capture_output=True, timeout=60,
            )
            if r.returncode == 0 and os.path.exists(tmp):
                return tmp
        except Exception as e:
            logger.warning(f"atempo failed for {audio_path}: {e}")
        return None

    def _show_context_menu(self, pos):
        item = self._tree.itemAt(pos)
        if not item:
            return
        idx = item.data(COL_STATUS, Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        frag = next((f for f in self._fragments if f['index'] == idx), None)
        if not frag:
            return

        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{ background:{C['panel']}; border:1px solid {C['border2']}; color:{C['text']}; padding:4px 0; }}
            QMenu::item {{ padding:5px 20px; }}
            QMenu::item:selected {{ background:{C['accent2']}; }}
            QMenu::separator {{ background:{C['border']}; height:1px; margin:4px 0; }}
        """)

        has_audio = bool(
            frag.get('output_path') and os.path.exists(frag.get('output_path', ''))
        )
        model_ok = self._backend.is_loaded if self._backend else False

        act_open = menu.addAction("📁  Open output folder")
        act_open.setEnabled(has_audio)
        act_open.triggered.connect(lambda: _open_file(os.path.dirname(frag['output_path'])))

        menu.addSeparator()

        act_re = menu.addAction("🔄  Re-synthesize this fragment")
        act_re.setEnabled(model_ok and not self._is_running)
        act_re.triggered.connect(lambda: self._re_synthesize_fragment(frag))

        menu.addSeparator()

        act_txt = menu.addAction("✏️  Edit text")
        act_txt.triggered.connect(lambda: self._edit_fragment_text(frag))

        act_all = menu.addAction("✏️  Add text to selected")
        act_all.triggered.connect(self._add_text_to_selected)

        act_time = menu.addAction("🕐  Edit time")
        act_time.triggered.connect(lambda: self._edit_fragment_time(frag))

        speaker = frag.get('speaker') or ""
        if speaker:
            act_spk = menu.addAction(f"✏️  Edit speaker  ({speaker})")
        else:
            act_spk = menu.addAction("➕  Add speaker")
        act_spk.triggered.connect(lambda: self._edit_speaker(frag))

        if speaker:
            act_clr = menu.addAction("✕  Remove speaker")
            act_clr.triggered.connect(lambda: self._edit_speaker(frag, clear=True))

        act_spk_sel = menu.addAction("✏️  Set speaker for selected")
        act_spk_sel.triggered.connect(self._set_speaker_for_selected)

        menu.addSeparator()

        pos_in_list = next(
            (i for i, f in enumerate(self._fragments) if f['index'] == idx), -1
        )

        act_add = menu.addAction("➕  Add fragment after")
        act_add.setEnabled(not self._is_running)
        act_add.triggered.connect(lambda: self._add_fragment_after(frag))

        act_del = menu.addAction("🗑  Remove fragment")
        act_del.setEnabled(not self._is_running)
        act_del.triggered.connect(lambda: self._remove_fragment(frag))

        act_up = menu.addAction("⬆  Move up")
        act_up.setEnabled(not self._is_running and pos_in_list > 0)
        act_up.triggered.connect(lambda: self._move_fragment(frag, -1))

        act_down = menu.addAction("⬇  Move down")
        act_down.setEnabled(
            not self._is_running and 0 <= pos_in_list < len(self._fragments) - 1
        )
        act_down.triggered.connect(lambda: self._move_fragment(frag, 1))

        menu.addSeparator()

        act_merge = menu.addAction("🔀  Merge fragments")
        act_merge.setEnabled(not self._is_running and len(self._fragments) > 1)
        act_merge.triggered.connect(lambda: self._merge_srt_fragments(frag))

        menu.exec(self._tree.viewport().mapToGlobal(pos))

    def _add_text_to_selected(self):
        if not self._fragments or not self._chapter_item:
            return

        checked_indices: set = set()
        for i in range(self._chapter_item.childCount()):
            child = self._chapter_item.child(i)
            if (not child.isHidden()
                    and child.checkState(COL_STATUS) == Qt.CheckState.Checked):
                idx = child.data(COL_STATUS, Qt.ItemDataRole.UserRole)
                if idx is not None:
                    checked_indices.add(idx)

        if not checked_indices:
            QMessageBox.information(
                self, "Nothing selected",
                "No fragments are checked.\nCheck at least one fragment first."
            )
            return

        checked_frags = [f for f in self._fragments if f['index'] in checked_indices]

        dlg = QDialog(self)
        dlg.setWindowTitle("Add text to selected fragments")
        dlg.resize(540, 310)
        dlg.setStyleSheet(self.styleSheet())

        lay = QVBoxLayout(dlg)
        lay.setSpacing(10)

        radio_row = QHBoxLayout()
        rb_prepend = QRadioButton("Prepend  (add before text)")
        rb_append  = QRadioButton("Append  (add after text)")
        rb_prepend.setChecked(True)
        radio_row.addWidget(rb_prepend)
        radio_row.addWidget(rb_append)
        radio_row.addStretch()
        lay.addLayout(radio_row)

        state_lbl = QLabel()
        state_lbl.setStyleSheet(
            f"color:{C['text2']};font-size:11px;"
            f"background:{C['surface']};padding:6px 8px;"
            f"border:1px solid {C['border']};border-radius:3px;"
        )
        state_lbl.setWordWrap(True)
        lay.addWidget(state_lbl)

        field_lbl = QLabel()
        field_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        lay.addWidget(field_lbl)

        editor = QLineEdit()
        editor.setFont(QFont("Segoe UI", 12))
        lay.addWidget(editor)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        def _build_counts(use_prefix: bool) -> Dict[str, int]:
            key = 'prefix' if use_prefix else 'suffix'
            counts: Dict[str, int] = {}
            for f in checked_frags:
                v = (f.get(key) or '').strip()
                counts[v] = counts.get(v, 0) + 1
            return counts

        def _update_mode():
            is_prepend = rb_prepend.isChecked()
            mode_word  = "prefix" if is_prepend else "suffix"
            counts     = _build_counts(is_prepend)
            lines = []
            for v, cnt in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
                label = f'"{v}"' if v else "(none)"
                lines.append(f"  {label}  ·  {cnt} fragment(s)")
            state_lbl.setText(
                f"Current {mode_word} in {len(checked_indices)} checked fragment(s):\n"
                + "\n".join(lines)
            )
            field_lbl.setText(
                f"New {mode_word} for all checked fragments "
                f"(leave empty to clear):"
            )
            common = list(counts.keys())[0] if len(counts) == 1 else ""
            editor.setText(common)
            editor.selectAll()

        rb_prepend.toggled.connect(_update_mode)
        _update_mode()
        editor.setFocus()

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        new_text   = editor.text().strip()
        is_prepend = rb_prepend.isChecked()
        key        = 'prefix' if is_prepend else 'suffix'
        mode_word  = "prefix" if is_prepend else "suffix"

        for frag in self._fragments:
            if frag['index'] not in checked_indices:
                continue
            frag[key]         = new_text
            frag['status']    = 'waiting'
            frag['error_msg'] = None
            self._update_tree_item(frag['index'])

        if new_text:
            self._set_status(
                f"{mode_word.capitalize()} '{new_text}' applied to {len(checked_indices)} "
                f"fragment(s) — status reset to waiting."
            )
        else:
            self._set_status(
                f"{mode_word.capitalize()} cleared for {len(checked_indices)} "
                f"fragment(s) — status reset to waiting."
            )

    def _edit_fragment_text(self, frag: Dict):
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Edit fragment #{frag['index'] + 1}")
        dlg.resize(560, 260)
        dlg.setStyleSheet(self.styleSheet())

        lay = QVBoxLayout(dlg)
        lay.setSpacing(8)

        info = QLabel(f"Timing: {frag.get('timestamp', '')}    Speaker: {frag.get('speaker') or '—'}")
        info.setStyleSheet(f"color:{C['text3']};font-size:11px;")
        lay.addWidget(info)

        editor = QPlainTextEdit()
        editor.setFont(QFont("Segoe UI", 13))
        editor.setPlainText(frag.get('text', ''))
        editor.selectAll()
        lay.addWidget(editor, 1)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        editor.setFocus()

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        new_text = editor.toPlainText().strip()
        if not new_text or new_text == frag.get('text', ''):
            return

        frag['text']   = new_text
        frag['status'] = 'waiting'
        frag['error_msg'] = None
        self._update_tree_item(frag['index'])
        self._preview_text.setPlainText(new_text)
        self._set_status(f"Fragment #{frag['index'] + 1} text updated — status reset to waiting.")

    def _edit_fragment_time(self, frag: Dict):
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Edit time — fragment #{frag['index'] + 1}")
        dlg.resize(420, 210)
        dlg.setStyleSheet(self.styleSheet())

        lay = QVBoxLayout(dlg)
        lay.setSpacing(10)

        info = QLabel(f"Text: {frag.get('text', '')[:60]}")
        info.setStyleSheet(f"color:{C['text3']};font-size:11px;")
        lay.addWidget(info)

        def ms_to_parts(ms: int):
            ms  = max(0, int(ms))
            m   = ms // 60000
            s   = (ms % 60000) // 1000
            rem = ms % 1000
            return m, s, rem

        def make_time_row(label_text: str, ms_val: int):
            row   = QHBoxLayout()
            lbl   = QLabel(label_text)
            lbl.setFixedWidth(50)
            lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
            m, s, rem = ms_to_parts(ms_val)

            sp_m = QSpinBox()
            sp_m.setRange(0, 9999)
            sp_m.setValue(m)
            sp_m.setSuffix(" min")
            sp_m.setFixedWidth(80)

            sp_s = QSpinBox()
            sp_s.setRange(0, 59)
            sp_s.setValue(s)
            sp_s.setSuffix(" s")
            sp_s.setFixedWidth(70)

            sp_ms = QSpinBox()
            sp_ms.setRange(0, 999)
            sp_ms.setValue(rem)
            sp_ms.setSuffix(" ms")
            sp_ms.setFixedWidth(80)

            row.addWidget(lbl)
            row.addWidget(sp_m)
            row.addWidget(sp_s)
            row.addWidget(sp_ms)
            row.addStretch()
            return row, sp_m, sp_s, sp_ms

        start_row, sp_sm, sp_ss, sp_sms = make_time_row("Start:", frag.get('start_ms', 0) or 0)
        end_row,   sp_em, sp_es, sp_ems = make_time_row("End:",   frag.get('srt_end_ms') or frag.get('end_ms', 0) or 0)
        lay.addLayout(start_row)
        lay.addLayout(end_row)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        new_start_ms = sp_sm.value() * 60000 + sp_ss.value() * 1000 + sp_sms.value()
        new_end_ms   = sp_em.value() * 60000 + sp_es.value() * 1000 + sp_ems.value()

        if new_start_ms >= new_end_ms:
            QMessageBox.warning(self, "Invalid time", "Start time must be less than end time.")
            return

        frag['start_ms']   = new_start_ms
        frag['end_ms']     = new_end_ms
        frag['srt_end_ms'] = new_end_ms
        frag['timestamp']  = f"{_ms_to_ts(new_start_ms)} --> {_ms_to_ts(new_end_ms)}"
        self._update_tree_item(frag['index'])
        self._set_status(
            f"Fragment #{frag['index'] + 1} time updated: "
            f"{_fmt_ms(new_start_ms)} → {_fmt_ms(new_end_ms)}", C["accent"]
        )

    def _edit_speaker(self, frag: Dict, clear: bool = False):
        if clear:
            frag['speaker'] = None
            self._update_tree_item(frag['index'])
            self._sync_speaker_ui_from_fragments()
            return

        current = frag.get('speaker') or ""
        name, ok = QInputDialog.getText(
            self,
            "Speaker name",
            "Enter speaker name:",
            text=current,
        )
        if ok:
            frag['speaker'] = name.strip() or None
            self._update_tree_item(frag['index'])
            self._sync_speaker_ui_from_fragments()

    def _set_speaker_for_selected(self):
        if not self._fragments or not self._chapter_item:
            return

        checked_indices: set = set()
        for i in range(self._chapter_item.childCount()):
            child = self._chapter_item.child(i)
            if (not child.isHidden()
                    and child.checkState(COL_STATUS) == Qt.CheckState.Checked):
                idx = child.data(COL_STATUS, Qt.ItemDataRole.UserRole)
                if idx is not None:
                    checked_indices.add(idx)

        if not checked_indices:
            QMessageBox.information(
                self, "Nothing selected",
                "No fragments are checked.\nCheck at least one fragment first."
            )
            return

        checked_frags = [f for f in self._fragments if f['index'] in checked_indices]
        current = checked_frags[0].get('speaker') or ""

        name, ok = QInputDialog.getText(
            self,
            "Speaker name",
            f"Set speaker for {len(checked_frags)} selected fragment(s)\n"
            "(leave empty to clear / use default voice):",
            text=current,
        )
        if not ok:
            return

        new_speaker = name.strip() or None
        for frag in checked_frags:
            frag['speaker'] = new_speaker
            self._update_tree_item(frag['index'])

        self._sync_speaker_ui_from_fragments()
        self._set_status(
            f"Speaker {'set to ' + new_speaker if new_speaker else 'cleared'} "
            f"for {len(checked_frags)} fragment(s).",
            C["accent"],
        )

    def _sync_speaker_ui_from_fragments(self):
        if self._dubbing_video_path:
            return
 
        freq: Dict[str, int] = {}
        for f in self._fragments:
            spk = f.get("speaker")
            if spk and str(spk).strip():
                k = str(spk).strip()
                freq[k] = freq.get(k, 0) + 1
 
        speakers = sorted(freq.keys(), key=lambda s: freq[s], reverse=True)
 
        if not speakers:
            if self._dubbing_mode:
                self._dubbing_mode = False
                self._speaker_list = []
                self._speaker_voices.clear()
                while self._speakers_lay.count():
                    item = self._speakers_lay.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                self._speakers_container.setVisible(False)
                self._voice_single_container.setVisible(True)
                if hasattr(self, "_whisper_section_widget"):
                    self._whisper_section_widget.setVisible(True)
            return
 
        if set(speakers) == set(self._speaker_list) and self._dubbing_mode:
            return
 
        saved_paths: Dict[str, str] = {}
        saved_texts: Dict[str, str] = {}
        for spk, sv in self._speaker_voices.items():
            drop     = sv.get("drop")
            ref_text = sv.get("ref_text")
            if drop and drop.file_path:
                saved_paths[spk] = drop.file_path
            if ref_text:
                t = ref_text.toPlainText()
                if t:
                    saved_texts[spk] = t
 
        self._speaker_list = speakers
        self._dubbing_mode = True
        self._rebuild_voice_cloning_for_speakers()
 
        for spk, path in saved_paths.items():
            if spk in self._speaker_voices and os.path.exists(path):
                sv = self._speaker_voices[spk]
                if "drop" not in sv:
                    continue
                sv["drop"]._set(path)
                sv["player"].load(path)
                sv["proc_btn"].setEnabled(True)
                sv["tr_btn"].setEnabled(
                    self._whisper_backend is not None
                    and self._whisper_backend.is_downloaded(self._w_size.currentData())
                )
 
        for spk, text in saved_texts.items():
            if spk in self._speaker_voices and text:
                sv = self._speaker_voices[spk]
                if "ref_text" in sv:
                    sv["ref_text"].setPlainText(text)
 
        self._set_status(
            f"{len(speakers)} speaker(s) — add reference audio for each in Voice cloning.",
            C["accent"],
        )

    def _play_fragment_audio(self, frag: Dict):
        path = frag.get('output_path', '')
        if not path or not os.path.exists(path):
            return
        try:
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            self._load_audio_to_player(audio, sr, fragment_idx=frag['index'])
        except Exception as e:
            self._set_status(f"Playback error: {e}", C["error"])

    def _re_synthesize_fragment(self, frag: Dict):
        if not self._backend or not self._backend.is_loaded:
            QMessageBox.warning(self, "Model not loaded", "Load the model first.")
            return

        self._synthesis_source = 'srt'

        frag["status"]    = "waiting"
        frag["error_msg"] = None
        self._update_tree_item(frag["index"])

        self._is_running       = True
        self._completed_count  = 0
        self._synth_start_time = time.monotonic()
        self._synth_total      = 1
        self._update_action_buttons()
        self._synth_progress.setMaximum(1)
        self._synth_progress.setValue(0)
        self._synth_progress.setVisible(True)
        self._eta_label.setText("0/1")
        self._eta_label.setVisible(True)

        reserved_paths = {
            f['output_path'] for f in self._fragments if f.get('output_path')
        }

        self._worker = TTSWorker(
            backend=self._backend,
            fragments=[frag],
            output_dir=self._output_dir,
            reference_audio=self._get_ref_audio(),
            reference_text=self._get_ref_text(),
            filename_prefix=Path(self._srt_path).stem if self._srt_path else "fragment",
            generation_settings=self._get_generation_settings(),
            normalize_audio=self._norm_check.isChecked(),
            speaker_voices=self._get_speaker_voices_dict(),
            reserved_paths=reserved_paths,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.item_done.connect(self._on_item_done)
        self._worker.finished.connect(self._on_synthesis_finished)
        self._worker.start()

    def _synthesize_quick_tts(self):
        if not self._backend or not self._backend.is_loaded:
            QMessageBox.warning(self, "Model not loaded", "Load the model first.")
            return
 
        text = self._quick_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Empty text", "Enter text to synthesize.")
            self._quick_edit.setFocus()
            return
 
        self._quick_btn.setEnabled(False)
        self._quick_btn.setText("  ⏳  Synthesizing…")
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self._stop_play()
 
        self._w_gen = GenerateWorker(
            self._backend,
            text,
            self._get_ref_audio(),
            self._get_ref_text(),
            self._get_generation_settings(),
        )
        self._w_gen.status.connect(lambda m: self._set_status(m))
        self._w_gen.finished.connect(self._on_quick_tts_done)
        self._w_gen.error.connect(lambda e: self._on_error("Generation error", e,
            reset_fn=lambda: (self._quick_btn.setEnabled(True),
                              self._quick_btn.setText("  🚀  Synthesize"),
                              self._progress.setVisible(False))))
        self._w_gen.start()

    def _on_quick_tts_done(self, audio: np.ndarray, sr: int):
        self._quick_btn.setEnabled(True)
        self._quick_btn.setText("  🚀  Synthesize")
        self._progress.setVisible(False)
        self._load_audio_to_player(audio, sr)
        self._save_wav_btn.setEnabled(True)
        self._save_mp3_btn.setEnabled(True)
        dur = len(audio) / sr
        self._set_status(f"Generated — {dur:.1f}s | {sr}Hz", C["success"])
        if self._audio_tmp and Path(self._audio_tmp).exists():
            try:
                os.unlink(self._audio_tmp)
            except Exception:
                pass
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            self._audio_tmp = f.name
        sf.write(self._audio_tmp, audio, sr, subtype="PCM_16")

    def _load_audio_to_player(self, audio: np.ndarray, sr: int,
                              fragment_idx: Optional[int] = None):
        self._playing = False
        self._play_btn.setText("▶ Play")
        self._timer.stop()
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        self._audio_data = audio
        self._audio_sr = sr
        self._current_fragment_idx = fragment_idx
        self._play_end_sample = len(audio)

        self._wave_out.set_audio(audio)
        self._wave_out.clear_trim_preview()
        self._wave_out.set_position(0.0)
        self._cursor = 0

        dur = len(audio) / max(1, sr)
        self._time_lbl.setText(f"0:00 / {_fmt(dur)}")

        self._play_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)

        tab = self._tabs.currentIndex() if hasattr(self, '_tabs') else 0
        if tab == 1:
            self._ebook_tree.setFocus()
        else:
            self._tree.setFocus()

        self._update_trim_preview()

    def _on_trim_slider_changed(self, value: int):
        val = value / 100.0

        if abs(self._trim_input.value() - val) > 0.001:
            self._trim_input.blockSignals(True)
            self._trim_input.setValue(val)
            self._trim_input.blockSignals(False)

        self._update_trim_preview()

    def _update_trim_preview(self):
        if not hasattr(self, '_trim_slider'):
            return
        aggressiveness = self._trim_slider.value() / 10.0
        if self._audio_data is None or aggressiveness <= 0.0:
            if hasattr(self, '_wave_out'):
                self._wave_out.clear_trim_preview()
            if hasattr(self, '_trim_preview_lbl'):
                self._trim_preview_lbl.setText("")
            if hasattr(self, '_trim_apply_all_btn'):
                self._trim_apply_all_btn.setEnabled(False)
            return

        lead_ms, trail_ms, lead_s, trail_s = _compute_trim_bounds(
            self._audio_data, self._audio_sr, aggressiveness
        )
        total = len(self._audio_data)
        if total == 0:
            return

        self._wave_out.set_trim_preview(lead_s / total, trail_s / total)

        new_samples = max(0, total - lead_s - trail_s)
        new_dur_s   = new_samples / max(1, self._audio_sr)
        self._trim_preview_lbl.setText(
            f"Lead: {lead_ms} ms | Trail: {trail_ms} ms  →  {new_dur_s:.2f}s"
        )

        has_trim = lead_ms > 0 or trail_ms > 0
        self._trim_apply_all_btn.setEnabled(has_trim)

    def _apply_trim_to_fragment(self):
        if self._audio_data is None:
            return
        aggressiveness = self._trim_slider.value() / 10.0
        if aggressiveness <= 0.0:
            return
        lead_ms, trail_ms, lead_s, trail_s = _compute_trim_bounds(
            self._audio_data, self._audio_sr, aggressiveness
        )
        if lead_ms == 0 and trail_ms == 0:
            self._set_status("Nothing to trim at current aggressiveness.", C["text2"])
            return

        total     = len(self._audio_data)
        end_s     = total - trail_s if trail_s > 0 else total
        new_audio = self._audio_data[lead_s:end_s].astype(np.float32)

        if len(new_audio) < int(self._audio_sr * 0.05):
            self._set_status("Aggressiveness too high — would trim everything. Reduce it.", C["warning"])
            return

        self._audio_undo_stack.append(self._build_audio_undo_state())
        self._audio_redo_stack.clear()

        new_dur_s        = len(new_audio) / max(1, self._audio_sr)
        self._audio_data = new_audio
        self._wave_out.set_audio(self._audio_data)
        self._wave_out.clear_trim_preview()
        self._cursor = 0

        if self._current_fragment_idx is not None:
            frag     = None
            is_ebook = False
            frag = next(
                (f for f in self._fragments if f['index'] == self._current_fragment_idx), None
            )
            if frag is None:
                frag = next(
                    (f for f in self._ebook_fragments if f['index'] == self._current_fragment_idx), None
                )
                is_ebook = frag is not None
            if frag:
                if frag.get('output_path') and os.path.exists(frag['output_path']):
                    try:
                        sf.write(frag['output_path'], self._audio_data, self._audio_sr, subtype="PCM_16")
                    except Exception:
                        pass
                if is_ebook:
                    self._update_ebook_tree_item(self._current_fragment_idx)
                else:
                    self._update_tree_item(self._current_fragment_idx, known_dur_s=new_dur_s)

        if hasattr(self, '_audio_tmp') and self._audio_tmp and os.path.exists(self._audio_tmp):
            try:
                sf.write(self._audio_tmp, self._audio_data, self._audio_sr, subtype="PCM_16")
            except Exception:
                pass

        self._time_lbl.setText(f"0:00 / {_fmt(new_dur_s)}")
        self._update_trim_preview()
        self._set_status(
            f"Trimmed — lead: {lead_ms} ms, trail: {trail_ms} ms. Ctrl+Z to undo.", C["accent"]
        )

    def _apply_trim_to_selected(self):
        aggressiveness = self._trim_slider.value() / 10.0
        if aggressiveness <= 0.0:
            return

        tab      = self._tabs.currentIndex()
        is_ebook = tab == 1
        frags    = self._ebook_fragments if is_ebook else self._fragments
        chapter  = self._ebook_chapter_item if is_ebook else self._chapter_item

        checked_done: List[Dict] = []
        if chapter:
            for i in range(chapter.childCount()):
                child = chapter.child(i)
                if child.checkState(COL_STATUS) != Qt.CheckState.Checked or child.isHidden():
                    continue
                idx  = child.data(COL_STATUS, Qt.ItemDataRole.UserRole)
                frag = next((f for f in frags if f['index'] == idx), None)
                if (frag
                        and frag.get('status') == 'done'
                        and frag.get('output_path')
                        and os.path.exists(frag.get('output_path', ''))):
                    checked_done.append(frag)

        if not checked_done:
            QMessageBox.information(
                self, "Nothing to trim",
                "No synthesized (done) fragments are checked.\n"
                "Check at least one completed fragment first."
            )
            return

        reply = QMessageBox.question(
            self, "Apply trim to all selected",
            f"Apply trim (aggressiveness {aggressiveness:.1f}) to {len(checked_done)} selected fragment(s)?\n\n"
            "Audio files will be permanently modified on disk.\n"
            "Ctrl+Z restores only the currently displayed fragment in the player.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        trimmed = 0
        for frag in checked_done:
            try:
                audio, sr = sf.read(frag['output_path'], dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                lms, tms, ls, ts = _compute_trim_bounds(audio, sr, aggressiveness)
                if lms == 0 and tms == 0:
                    continue
                total     = len(audio)
                end_s     = total - ts if ts > 0 else total
                new_audio = audio[ls:end_s].astype(np.float32)
                if len(new_audio) < int(sr * 0.05):
                    continue
                sf.write(frag['output_path'], new_audio, sr, subtype="PCM_16")
                new_dur_s = len(new_audio) / max(1, sr)
                if frag.get('index') == self._current_fragment_idx:
                    self._audio_undo_stack.append(self._build_audio_undo_state())
                    self._audio_redo_stack.clear()
                    self._audio_data = new_audio
                    self._audio_sr   = sr
                    self._wave_out.set_audio(self._audio_data)
                    self._cursor = 0
                    self._time_lbl.setText(f"0:00 / {_fmt(new_dur_s)}")
                if is_ebook:
                    self._update_ebook_tree_item(frag['index'])
                else:
                    self._update_tree_item(frag['index'], known_dur_s=new_dur_s)
                trimmed += 1
            except Exception as e:
                logger.warning(f"Trim failed for fragment {frag.get('index')}: {e}")

        self._wave_out.clear_trim_preview()
        self._update_trim_preview()
        self._set_status(
            f"Trim applied to {trimmed}/{len(checked_done)} fragment(s).", C["success"]
        )

    def _build_audio_undo_state(self) -> dict:
        frag = None
        is_ebook = False
        if self._current_fragment_idx is not None:
            frag = next((f for f in self._fragments if f['index'] == self._current_fragment_idx), None)
            if frag is None:
                frag = next((f for f in self._ebook_fragments if f['index'] == self._current_fragment_idx), None)
                is_ebook = frag is not None
        return {
            'audio':         self._audio_data.copy(),
            'sr':            self._audio_sr,
            'frag_idx':      self._current_fragment_idx,
            'old_end_ms':    frag.get('end_ms')    if frag else None,
            'old_timestamp': frag.get('timestamp') if frag else None,
            'is_ebook':      is_ebook,
        }

    def _restore_audio_state(self, state: dict):
        audio    = state['audio']
        sr       = state['sr']
        frag_idx = state.get('frag_idx')
        is_ebook = state.get('is_ebook', False)
        old_end_ms    = state.get('old_end_ms')
        old_timestamp = state.get('old_timestamp')

        self._audio_data = audio
        self._audio_sr   = sr

        if hasattr(self, '_wave_out'):
            self._wave_out.set_audio(audio)
            self._wave_out.clear_selection()
            self._wave_out.clear_trim_preview()
        self._cursor = min(self._cursor, len(audio))

        if frag_idx is not None and old_end_ms is not None:
            frags = self._ebook_fragments if is_ebook else self._fragments
            frag  = next((f for f in frags if f['index'] == frag_idx), None)
            if frag:
                frag['end_ms'] = old_end_ms
                if old_timestamp is not None:
                    frag['timestamp'] = old_timestamp
                if frag.get('output_path') and os.path.exists(frag['output_path']):
                    try:
                        sf.write(frag['output_path'], audio, sr, subtype="PCM_16")
                    except Exception:
                        pass
                new_dur_s = len(audio) / max(1, sr)
                if is_ebook:
                    self._update_ebook_tree_item(frag_idx)
                else:
                    self._update_tree_item(frag_idx, known_dur_s=new_dur_s)

        if hasattr(self, '_audio_tmp') and self._audio_tmp and os.path.exists(self._audio_tmp):
            try:
                sf.write(self._audio_tmp, audio, sr, subtype="PCM_16")
            except Exception:
                pass

        if hasattr(self, '_time_lbl'):
            dur = len(audio) / max(1, sr)
            self._time_lbl.setText(f"0:00 / {_fmt(dur)}")

        self._update_trim_preview()

    def _delete_selected_audio_segment(self, start_frac: float, end_frac: float):
        if self._audio_data is None:
            return
        length       = len(self._audio_data)
        start_sample = int(start_frac * length)
        end_sample   = int(end_frac   * length)
        if end_sample - start_sample < 10:
            return

        self._audio_undo_stack.append(self._build_audio_undo_state())
        self._audio_redo_stack.clear()

        new_audio  = np.concatenate((
            self._audio_data[:start_sample],
            self._audio_data[end_sample:]
        )).astype(np.float32)
        new_dur_s  = len(new_audio) / max(1, self._audio_sr)
        deleted_s  = (end_sample - start_sample) / max(1, self._audio_sr)

        self._audio_data = new_audio

        if hasattr(self, '_wave_out'):
            self._wave_out.set_audio(self._audio_data)
            self._wave_out.clear_selection()
        self._cursor = min(self._cursor, len(self._audio_data))

        if self._current_fragment_idx is not None:
            frag     = None
            is_ebook = False
            frag = next((f for f in self._fragments if f['index'] == self._current_fragment_idx), None)
            if frag is None:
                frag     = next((f for f in self._ebook_fragments if f['index'] == self._current_fragment_idx), None)
                is_ebook = frag is not None
            if frag:
                if frag.get('output_path') and os.path.exists(frag['output_path']):
                    try:
                        sf.write(frag['output_path'], self._audio_data, self._audio_sr, subtype="PCM_16")
                    except Exception:
                        pass
                if is_ebook:
                    self._update_ebook_tree_item(self._current_fragment_idx)
                else:
                    self._update_tree_item(self._current_fragment_idx, known_dur_s=new_dur_s)

        if hasattr(self, '_audio_tmp') and self._audio_tmp and os.path.exists(self._audio_tmp):
            try:
                sf.write(self._audio_tmp, self._audio_data, self._audio_sr, subtype="PCM_16")
            except Exception:
                pass

        if hasattr(self, '_time_lbl'):
            dur = len(self._audio_data) / max(1, self._audio_sr)
            self._time_lbl.setText(f"0:00 / {_fmt(dur)}")

        self._update_trim_preview()

        self._set_status(
            f"Deleted {deleted_s:.2f}s — audio updated. Ctrl+Z to undo.", C["accent"]
        )

    def _mute_selected_audio_segment(self, start_frac: float, end_frac: float):
        if self._audio_data is None:
            return
        length       = len(self._audio_data)
        start_sample = int(start_frac * length)
        end_sample   = int(end_frac   * length)
        if end_sample - start_sample < 10:
            return

        self._audio_undo_stack.append(self._build_audio_undo_state())
        self._audio_redo_stack.clear()

        self._audio_data = self._audio_data.copy()
        self._audio_data[start_sample:end_sample] = 0.0
        muted_s = (end_sample - start_sample) / max(1, self._audio_sr)

        if hasattr(self, '_wave_out'):
            self._wave_out.set_audio(self._audio_data)
            self._wave_out.clear_selection()
        self._cursor = min(self._cursor, len(self._audio_data))

        if self._current_fragment_idx is not None:
            frag     = None
            is_ebook = False
            frag = next((f for f in self._fragments if f['index'] == self._current_fragment_idx), None)
            if frag is None:
                frag     = next((f for f in self._ebook_fragments if f['index'] == self._current_fragment_idx), None)
                is_ebook = frag is not None
            if frag and frag.get('output_path') and os.path.exists(frag['output_path']):
                try:
                    sf.write(frag['output_path'], self._audio_data, self._audio_sr, subtype="PCM_16")
                    if is_ebook:
                        self._update_ebook_tree_item(self._current_fragment_idx)
                    else:
                        self._update_tree_item(self._current_fragment_idx)
                except Exception:
                    pass

        if hasattr(self, '_audio_tmp') and self._audio_tmp and os.path.exists(self._audio_tmp):
            try:
                sf.write(self._audio_tmp, self._audio_data, self._audio_sr, subtype="PCM_16")
            except Exception:
                pass

        if hasattr(self, '_time_lbl'):
            dur = len(self._audio_data) / max(1, self._audio_sr)
            self._time_lbl.setText(f"0:00 / {_fmt(dur)}")

        self._update_trim_preview()

        self._set_status(
            f"Audio segment muted ({muted_s:.2f}s silenced). Ctrl+Z to undo.", C["accent"]
        )

    def _toggle_play(self):
        if self._playing:
            self._pause_play()
        else:
            self._start_play()

    def _start_play(self):
        if self._audio_data is None:
            return
        self._playing = True
        self._play_btn.setText("■  Pause")

        audio = self._audio_data.astype(np.float32)

        aggressiveness = self._trim_slider.value() / 10.0 if hasattr(self, '_trim_slider') else 0.0
        lead_s = 0
        trail_s = 0
        if aggressiveness > 0.0:
            _, _, lead_s, trail_s = _compute_trim_bounds(audio, self._audio_sr, aggressiveness)

        trimmed_end = len(audio) - trail_s

        sel = self._wave_out.get_selection()
        if sel:
            s_sample = int(sel[0] * len(audio))
            e_sample = int(sel[1] * len(audio))
        else:
            s_sample = max(lead_s, self._cursor) if self._cursor < trimmed_end else lead_s
            e_sample = trimmed_end

        if s_sample >= e_sample:
            s_sample = lead_s
            e_sample = trimmed_end

        self._cursor = s_sample
        self._play_end_sample = e_sample
        chunk = audio[s_sample:e_sample].copy()

        if not len(chunk):
            self._playing = False
            self._play_btn.setText("▶  Play")
            return

        def cb(out, frames, ti, st):
            nonlocal chunk
            if not self._playing:
                raise sd.CallbackStop()
            n = min(frames, len(chunk))
            if n == 0:
                out[:] = 0
                raise sd.CallbackStop()
            vol = self._vol.value() / 100.0
            out[:n, 0] = chunk[:n] * vol
            if frames > n:
                out[n:] = 0
            chunk = chunk[n:]
            self._cursor += n

        try:
            self._stream = sd.OutputStream(
                samplerate=self._audio_sr, channels=1, dtype="float32",
                callback=cb, finished_callback=self._on_play_end,
            )
            self._stream.start()
            self._timer.start()
        except Exception as e:
            self._playing = False
            self._play_btn.setText("▶  Play")
            self._set_status(f"Playback error: {e}", C["error"])
        
    def _pause_play(self):
        self._playing = False
        self._play_btn.setText("▶  Play")
        self._timer.stop()
        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass

    def _stop_play(self):
        self._playing = False
        self._play_btn.setText("▶  Play")
        self._timer.stop()
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._audio_data is not None and hasattr(self, '_wave_out'):
            sel = self._wave_out.get_selection()
            if sel:
                self._cursor = int(sel[0] * len(self._audio_data))
            else:
                aggressiveness = self._trim_slider.value() / 10.0 if hasattr(self, '_trim_slider') else 0.0
                lead_s = 0
                if aggressiveness > 0.0:
                    _, _, lead_s, _ = _compute_trim_bounds(
                        self._audio_data.astype(np.float32), self._audio_sr, aggressiveness
                    )
                self._cursor = lead_s
            self._wave_out.set_position(self._cursor / max(1, len(self._audio_data)))
            self._time_lbl.setText(f"0:00 / {_fmt(len(self._audio_data) / max(1, self._audio_sr))}")
        else:
            self._cursor = 0

    def _on_play_end(self):
        def _safe():
            self._playing = False
            self._play_btn.setText("▶  Play")
            self._timer.stop()
            if self._audio_data is not None and hasattr(self, '_wave_out'):
                sel = self._wave_out.get_selection()
                if sel:
                    self._cursor = int(sel[0] * len(self._audio_data))
                else:
                    aggressiveness = self._trim_slider.value() / 10.0 if hasattr(self, '_trim_slider') else 0.0
                    lead_s = 0
                    if aggressiveness > 0.0:
                        _, _, lead_s, _ = _compute_trim_bounds(
                            self._audio_data.astype(np.float32), self._audio_sr, aggressiveness
                        )
                    self._cursor = lead_s
                self._wave_out.set_position(self._cursor / max(1, len(self._audio_data)))
                self._time_lbl.setText(
                    f"{_fmt(self._cursor / max(1, self._audio_sr))} / "
                    f"{_fmt(len(self._audio_data) / max(1, self._audio_sr))}"
                )
        QTimer.singleShot(0, _safe)

    def _on_wave_undo(self):
        if not self._audio_undo_stack or self._audio_data is None:
            return
        self._audio_redo_stack.append(self._build_audio_undo_state())
        state = self._audio_undo_stack.pop()
        self._restore_audio_state(state)
        self._set_status("Undo: segment restored", C["accent"])

    def _on_wave_redo(self):
        if not self._audio_redo_stack or self._audio_data is None:
            return
        self._audio_undo_stack.append(self._build_audio_undo_state())
        state = self._audio_redo_stack.pop()
        self._restore_audio_state(state)
        self._set_status("Redo: segment applied", C["accent"])

    def _playback_tick(self):
        if self._audio_data is None:
            return
        total = len(self._audio_data)
        pos   = min(self._cursor, total) / max(1, total)
        if hasattr(self, '_wave_out'):
            self._wave_out.set_position(pos)
        self._time_lbl.setText(
            f"{_fmt(self._cursor / max(1, self._audio_sr))} / "
            f"{_fmt(total / max(1, self._audio_sr))}"
        )

    def _seek(self, frac: float):
        if self._audio_data is None:
            return
        self._pause_play()
        self._cursor = int(frac * len(self._audio_data))
        self._wave_out.set_position(frac)
        self._start_play()

    def _save_audio(self, fmt: str = "wav"):
        if isinstance(fmt, bool):
            fmt = "wav"
        if self._audio_data is None:
            return

        default_path = str(Path(_get_last_dir("output", str(OUTPUTS_DIR))) / f"audio.{fmt}")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save audio", default_path,
            f"Audio {fmt.upper()} (*.{fmt});;All files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(f".{fmt}"):
            path = f"{path}.{fmt}"

        _set_last_dir("output", path)

        audio = np.asarray(self._audio_data, dtype=np.float32)
        sr    = int(self._audio_sr)

        try:
            if fmt == "wav":
                sf.write(path, audio, sr, subtype="PCM_16")
            else:
                try:
                    from pydub import AudioSegment
                    t = tempfile.mktemp(suffix=".wav")
                    sf.write(t, audio, sr, subtype="PCM_16")
                    AudioSegment.from_wav(t).export(path, format="mp3", bitrate="192k")
                    os.unlink(t)
                except ImportError:
                    wav_path = path.replace(".mp3", ".wav")
                    sf.write(wav_path, audio, sr, subtype="PCM_16")
                    self._set_status(f"pydub not available — saved as WAV: {wav_path}", C["warning"])
                    return
            self._set_status(f"Saved: {path}", C["success"])
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    def _build_lektor_audio_track(self, output_path: str,
                                   sample_rate: int = 44100,
                                   offset_ms: int = 0) -> bool:
        done_frags = [
            f for f in self._fragments
            if f.get('status') == 'done'
            and f.get('output_path')
            and os.path.exists(f['output_path'])
            and f.get('start_ms') is not None
        ]

        if not done_frags:
            logger.warning("_build_lektor_audio_track: no done fragments")
            return False

        done_frags = sorted(done_frags, key=lambda f: f.get('start_ms', 0) or 0)

        autofit       = self._ffmpeg_ok and self._autofit_check.isChecked()
        atempo_thresh = self._atempo_threshold.value()
        tmp_files: List[str] = []

        max_end_ms = max((f.get('end_ms') or f.get('start_ms', 0)) for f in done_frags)

        total_audio_s    = sum(_get_wav_duration(f['output_path']) or 0.0 for f in done_frags)
        total_duration_s = max(max_end_ms / 1000.0, total_audio_s) + 10.0
        total_samples    = int(total_duration_s * sample_rate)
        track            = torch.zeros(1, total_samples)
        cursor_sample    = 0

        for frag in done_frags:
            raw_start_ms = frag.get('start_ms', 0) or 0
            raw_end_ms   = frag.get('end_ms',   0) or 0
            adj_start_ms = raw_start_ms + offset_ms
            adj_end_ms   = raw_end_ms   + offset_ms
            audio_file   = frag['output_path']

            slot_ms = max(0, adj_end_ms - adj_start_ms)

            audio_dur_s  = _get_wav_duration(audio_file) or 0.0
            audio_dur_ms = int(audio_dur_s * 1000)

            if slot_ms > 0 and audio_dur_ms > slot_ms:
                overshoot_ms = audio_dur_ms - slot_ms
                if autofit and overshoot_ms <= atempo_thresh:
                    fitted = self._fit_audio_to_slot(audio_file, slot_ms)
                    if fitted:
                        tmp_files.append(fitted)
                        audio_file   = fitted
                        audio_dur_ms = slot_ms

            srt_start_sample = max(0, int(adj_start_ms / 1000.0 * sample_rate))
            start_sample     = max(srt_start_sample, cursor_sample)

            try:
                frag_audio, frag_sr = sf.read(audio_file, dtype="float32", always_2d=False)
                if frag_audio.ndim > 1:
                    frag_audio = frag_audio.mean(axis=1)
                frag_max = float(np.abs(frag_audio).max()) if len(frag_audio) > 0 else 0.0
                if frag_max == 0.0:
                    logger.warning(f"Fragment {audio_file} has zero amplitude, skipping")
                    continue
                frag_audio = (frag_audio / frag_max * 0.92).astype(np.float32)
                waveform = torch.from_numpy(frag_audio).unsqueeze(0)
                if frag_sr != sample_rate:
                    waveform = TAF.resample(waveform, frag_sr, sample_rate)
            except Exception as e:
                logger.warning(f"Cannot load {audio_file}: {e}")
                continue

            length = min(waveform.shape[1], total_samples - start_sample)
            if start_sample < total_samples and length > 0:
                track[0, start_sample:start_sample + length] = waveform[0, :length]
                cursor_sample = start_sample + length

        for tmp in tmp_files:
            try:
                os.remove(tmp)
            except Exception:
                pass

        max_val = float(track.abs().max())
        logger.info(f"Lektor track peak amplitude before final normalize: {max_val:.6f}")
        if max_val == 0.0:
            logger.warning("_build_lektor_audio_track: track is completely silent — no audio was placed")
            return False
        track = track / max_val * 0.92

        sf.write(output_path, track[0].numpy(), sample_rate, subtype="PCM_16")
        logger.info(f"Lektor track saved: {output_path} ({len(done_frags)} frags, {total_duration_s:.1f}s)")
        return True

    def _browse_video_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video file", _get_last_dir("video"),
            "Video (*.mp4 *.mkv *.avi *.mov *.webm *.wmv);;All files (*)",
        )
        if path:
            _set_last_dir("video", path)
            self._vid_path_edit.setText(path)

    def _normalize_ffmpeg(self, input_path: str, output_path: str) -> bool:
        try:
            r1 = subprocess.run(
                ["ffmpeg", "-y", "-i", input_path,
                 "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
                 "-f", "null", "-"],
                capture_output=True, timeout=120,
            )
            stderr = r1.stderr.decode(errors="replace")
            json_start = stderr.rfind("{")
            json_end   = stderr.rfind("}") + 1
            if json_start == -1:
                raise ValueError("loudnorm: no JSON in stderr")
            info = json.loads(stderr[json_start:json_end])

            af = (
                f"loudnorm=I=-16:TP=-1.5:LRA=11:linear=true"
                f":measured_I={info['input_i']}"
                f":measured_LRA={info['input_lra']}"
                f":measured_TP={info['input_tp']}"
                f":measured_thresh={info['input_thresh']}"
                f":offset={info['target_offset']}"
            )
            r2 = subprocess.run(
                ["ffmpeg", "-y", "-i", input_path, "-af", af, output_path],
                capture_output=True, timeout=300,
            )
            return r2.returncode == 0
        except Exception as e:
            logger.warning(f"loudnorm two-pass failed: {e}")
            return False

    def _export_lektor_video(self):
        if not self._ffmpeg_ok:
            QMessageBox.warning(self, "ffmpeg not found",
                "Video export requires ffmpeg installed in PATH.")
            return

        if self._lektor_export_thread and self._lektor_export_thread.isRunning():
            QMessageBox.information(self, "Export in progress",
                "Video export is already running.")
            return

        if (self._w_vocal_suppress is not None
                and self._w_vocal_suppress.isRunning()):
            QMessageBox.information(self, "Processing in progress",
                "Vocal separation is already running.")
            return

        if self._dubbing_mode and self._dubbing_video_path:
            video_path = self._dubbing_video_path
        elif getattr(self, '_video_source_path', None) and os.path.exists(
                self._video_source_path or ''):
            video_path = self._video_source_path
        else:
            video_path = self._vid_path_edit.text().strip()

        if not video_path or not os.path.exists(video_path):
            QMessageBox.warning(self, "No video file",
                "Select a video file in the 'Video:' field.")
            return

        done_count = sum(
            1 for f in self._fragments
            if f.get("status") == "done"
            and f.get("start_ms") is not None
            and f.get("output_path")
            and os.path.exists(f.get("output_path", ""))
        )
        if done_count == 0:
            QMessageBox.warning(self, "No audio",
                "No SRT fragment has generated audio yet.\nRun synthesis before exporting.")
            return

        os.makedirs(self._output_dir, exist_ok=True)

        fmt = self._lektor_vid_fmt_combo.currentData()
        if fmt == "auto":
            video_ext = os.path.splitext(video_path)[1]
        else:
            video_ext = f".{fmt}"

        default_out = os.path.join(
            _get_last_dir("output", self._output_dir), f"lektor_output{video_ext}"
        )
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save video with lektor", default_out,
            f"Video (*{video_ext});;All files (*)",
        )
        if not out_path:
            return

        _set_last_dir("output", out_path)

        sample_rate         = 44100
        offset_ms           = self._offset_spin.value()
        lektor_wav          = os.path.join(self._output_dir, "_lektor_track_tmp.wav")
        lektor_vol          = self._lektor_vol.value() / 100.0
        orig_vol            = self._orig_vol.value() / 100.0
        use_ducking         = self._duck_check.isChecked()
        keep_original_track = self._keep_original_track_check.isChecked()
        if keep_original_track:
            _raw_lang = self._dubbed_lang_edit.text().strip().lower()
            dubbed_lang = _raw_lang if (len(_raw_lang) == 3 and _raw_lang.isalpha()) else "und"
        else:
            dubbed_lang = "und"

        self._set_status(f"Building lektor track from {done_count} fragments…")
        self._lektor_status.setText("Building lektor track…")
        self._lektor_status.setStyleSheet(f"color:{C['warning']};font-size:10px;")

        ok = self._build_lektor_audio_track(lektor_wav, sample_rate, offset_ms)
        if not ok:
            QMessageBox.critical(self, "Error",
                "Could not build lektor track.\nCheck that fragments have been synthesized.")
            return

        if self._norm_check.isChecked():
            self._set_status("Normalizing lektor track…")
            self._lektor_status.setText("Normalizing audio…")
            norm_tmp = lektor_wav.replace(".wav", "_norm.wav")
            if self._normalize_ffmpeg(lektor_wav, norm_tmp):
                try:
                    os.replace(norm_tmp, lektor_wav)
                    logger.info("Lektor track normalized (two-pass loudnorm)")
                except OSError as e:
                    logger.warning(f"Could not replace lektor WAV after normalization: {e}")
            else:
                logger.warning("Lektor track normalization failed, using unnormalized track")

        try:
            check_data, _ = sf.read(lektor_wav, dtype="float32")
            check_max = float(np.abs(check_data).max())
            logger.info(f"Lektor WAV amplitude validation: max={check_max:.6f}")
            if check_max < 0.01:
                logger.warning(f"Lektor WAV is near-silent (max={check_max:.6f})")
                QMessageBox.warning(self, "Warning",
                    f"Lektor track has very low amplitude ({check_max:.4f}).\n"
                    "Audio may be inaudible in the output video.")
        except Exception as e:
            logger.warning(f"Could not validate lektor WAV: {e}")

        has_video_audio = False
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_streams", "-select_streams", "a", video_path],
                capture_output=True, timeout=30,
            )
            if probe.returncode == 0:
                info = json.loads(probe.stdout.decode(errors="replace"))
                has_video_audio = len(info.get("streams", [])) > 0
            else:
                has_video_audio = True
        except Exception as e:
            logger.warning(f"ffprobe audio probe failed: {e} — assuming audio present")
            has_video_audio = True

        logger.info(f"Video has audio stream: {has_video_audio}")

        use_vocal_suppress = (
            self._vocal_suppress_check.isChecked()
            and has_video_audio
        )
        vocal_suppress_vol = self._vocal_suppress_spin.value() / 100.0

        if use_vocal_suppress:
            self._pending_export = {
                "video_path":          video_path,
                "lektor_wav":          lektor_wav,
                "out_path":            out_path,
                "has_video_audio":     has_video_audio,
                "lektor_vol":          lektor_vol,
                "orig_vol":            orig_vol,
                "use_ducking":         use_ducking,
                "vocal_suppress_vol":  vocal_suppress_vol,
                "keep_original_track": keep_original_track,
                "dubbed_lang":         dubbed_lang,
            }
            self._export_btn.setEnabled(False)
            self._progress.setVisible(True)
            self._progress.setRange(0, 0)
            self._lektor_status.setText("Running Demucs vocal separation…")
            self._lektor_status.setStyleSheet(f"color:{C['warning']};font-size:10px;")

            self._w_vocal_suppress = VocalSuppressWorker(video_path, self._output_dir)
            self._w_vocal_suppress.status.connect(lambda m: self._set_status(m))
            self._w_vocal_suppress.finished.connect(self._on_vocal_suppress_done)
            self._w_vocal_suppress.error.connect(
                lambda e: self._on_error("Vocal separation error", e,
                    reset_fn=lambda: (
                        self._export_btn.setEnabled(self._ffmpeg_ok),
                        self._progress.setVisible(False),
                        self._lektor_status.setText("Vocal separation failed."),
                        self._lektor_status.setStyleSheet(
                            f"color:{C['error']};font-size:10px;"
                        ),
                    ))
            )
            self._w_vocal_suppress.start()
        else:
            self._do_lektor_ffmpeg_export(
                video_path, lektor_wav, out_path,
                has_video_audio, lektor_vol, orig_vol,
                use_ducking,
                keep_original_track=keep_original_track,
                dubbed_lang=dubbed_lang,
            )

    def _do_lektor_ffmpeg_export(
        self,
        video_path: str,
        lektor_wav: str,
        out_path: str,
        has_video_audio: bool,
        lektor_vol: float,
        orig_vol: float,
        use_ducking: bool,
        vocals_wav: Optional[str] = None,
        no_vocals_wav: Optional[str] = None,
        vocal_suppress_vol: float = 0.0,
        keep_original_track: bool = False,
        dubbed_lang: str = "pol",
    ):
        use_vocal_suppress = vocals_wav is not None and no_vocals_wav is not None
        extra_tmp = [p for p in [vocals_wav, no_vocals_wav] if p]

        if has_video_audio:
            if use_vocal_suppress:
                if use_ducking:
                    audio_filter = (
                        f"[1:a]volume={orig_vol:.2f}[bg];"
                        f"[2:a]volume={vocal_suppress_vol:.2f}[vox];"
                        f"[bg][vox]amix=inputs=2:duration=longest:normalize=0[orig_mix];"
                        f"[3:a]volume={lektor_vol:.2f},asplit=2[lekt1][lekt2];"
                        f"[orig_mix][lekt1]sidechaincompress="
                        f"threshold=0.025:ratio=4:attack=10:release=400[ducked];"
                        f"[ducked][lekt2]amix=inputs=2:duration=first:normalize=0[aout]"
                    )
                else:
                    audio_filter = (
                        f"[1:a]volume={orig_vol:.2f}[bg];"
                        f"[2:a]volume={vocal_suppress_vol:.2f}[vox];"
                        f"[3:a]volume={lektor_vol:.2f}[lekt];"
                        f"[bg][vox][lekt]amix=inputs=3:duration=first:normalize=0[aout]"
                    )
                if keep_original_track:
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", video_path,
                        "-i", no_vocals_wav,
                        "-i", vocals_wav,
                        "-i", lektor_wav,
                        "-filter_complex", audio_filter,
                        "-map", "0:v:0",
                        "-map", "0:a:0",
                        "-map", "[aout]",
                        "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k",
                        "-metadata:s:a:0", "title=Original",
                        "-metadata:s:a:1", "title=Dubbing",
                        "-metadata:s:a:1", f"language={dubbed_lang}",
                        "-disposition:a:0", "default",
                        "-disposition:a:1", "0",
                        out_path,
                    ]
                else:
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", video_path,
                        "-i", no_vocals_wav,
                        "-i", vocals_wav,
                        "-i", lektor_wav,
                        "-filter_complex", audio_filter,
                        "-map", "0:v:0",
                        "-map", "[aout]",
                        "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k",
                        out_path,
                    ]
            else:
                if use_ducking:
                    audio_filter = (
                        f"[1:a]volume={lektor_vol:.2f},asplit=2[lekt1][lekt2];"
                        f"[0:a]volume={orig_vol:.2f}[orig];"
                        f"[orig][lekt1]sidechaincompress="
                        f"threshold=0.025:ratio=4:attack=10:release=400[ducked];"
                        f"[ducked][lekt2]amix=inputs=2:duration=first:normalize=0[aout]"
                    )
                else:
                    audio_filter = (
                        f"[0:a]volume={orig_vol:.2f}[orig];"
                        f"[1:a]volume={lektor_vol:.2f}[lekt];"
                        f"[orig][lekt]amix=inputs=2:duration=first:normalize=0[aout]"
                    )
                if keep_original_track:
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", video_path,
                        "-i", lektor_wav,
                        "-filter_complex", audio_filter,
                        "-map", "0:v:0",
                        "-map", "0:a:0",
                        "-map", "[aout]",
                        "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k",
                        "-metadata:s:a:0", "title=Original",
                        "-metadata:s:a:1", "title=Dubbing",
                        "-metadata:s:a:1", f"language={dubbed_lang}",
                        "-disposition:a:0", "default",
                        "-disposition:a:1", "0",
                        out_path,
                    ]
                else:
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", video_path,
                        "-i", lektor_wav,
                        "-filter_complex", audio_filter,
                        "-map", "0:v:0",
                        "-map", "[aout]",
                        "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k",
                        out_path,
                    ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", lektor_wav,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-af", f"volume={lektor_vol:.2f}",
                out_path,
            ]

        logger.info(f"ffmpeg cmd: {' '.join(cmd)}")
        self._set_status("Exporting video with lektor track…")
        self._lektor_status.setText("Exporting video…")
        self._lektor_status.setStyleSheet(f"color:{C['warning']};font-size:10px;")
        self._export_btn.setEnabled(False)

        self._lektor_export_thread = LektorExportThread(cmd, lektor_wav, extra_tmp)
        self._lektor_export_thread.progress.connect(
            lambda m: self._set_status(m)
        )
        self._lektor_export_thread.finished.connect(
            lambda ok, err: self._on_lektor_export_finished(ok, err, out_path)
        )
        self._lektor_export_thread.start()

    def _on_vocal_suppress_done(self, vocals_path: str, no_vocals_path: str):
        self._progress.setVisible(False)
        p = self._pending_export
        if not p:
            return
        self._pending_export = {}
        self._do_lektor_ffmpeg_export(
            video_path          = p["video_path"],
            lektor_wav          = p["lektor_wav"],
            out_path            = p["out_path"],
            has_video_audio     = p["has_video_audio"],
            lektor_vol          = p["lektor_vol"],
            orig_vol            = p["orig_vol"],
            use_ducking         = p["use_ducking"],
            vocals_wav          = vocals_path,
            no_vocals_wav       = no_vocals_path,
            vocal_suppress_vol  = p["vocal_suppress_vol"],
            keep_original_track = p.get("keep_original_track", False),
            dubbed_lang         = p.get("dubbed_lang", "pol"),
        )

    def _on_lektor_export_finished(self, success: bool, error_msg: str, out_path: str):
        self._lektor_export_thread = None
        self._export_btn.setEnabled(self._ffmpeg_ok)

        if success:
            self._lektor_status.setText(f"✓  Saved: {Path(out_path).name}")
            self._lektor_status.setStyleSheet(f"color:{C['success']};font-size:10px;")
            self._set_status(f"Video with lektor saved: {out_path}", C["success"])
            reply = QMessageBox.information(
                self, "Export complete",
                f"Video with lektor saved:\n{out_path}\n\nOpen output folder?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                _open_file(os.path.dirname(out_path))
        else:
            self._lektor_status.setText("✗  Export failed")
            self._lektor_status.setStyleSheet(f"color:{C['error']};font-size:10px;")
            self._set_status("Video export failed.", C["error"])
            QMessageBox.critical(self, "ffmpeg error",
                f"Export failed:\n\n{error_msg}")
        logger.info(f"Lektor export finished: success={success}")

    def _save_session(self):
        if not self._srt_path:
            return
        default_path = str(
            Path(_get_last_dir("session", str(OUTPUTS_DIR))) / "session.json"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save session", default_path,
            "JSON (*.json);;All files (*)",
        )
        if not path:
            return
        _set_last_dir("session", path)
        if self._write_session_to(path):
            auto = self._auto_session_path()
            if auto and str(auto) != path:
                self._write_session_to(str(auto))
            self._set_status(f"Session saved: {path}", C["success"])
        else:
            QMessageBox.critical(self, "Save error", f"Could not write session to:\n{path}")

    def _ref_audio_hash(self, path: Optional[str]) -> Optional[str]:
        if not path or not os.path.exists(path):
            return None
        h = hashlib.md5()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    def _auto_session_path(self) -> Optional[Path]:
        if not self._srt_path:
            return None
        return OUTPUTS_DIR / Path(self._srt_path).stem / "session.json"

    def _build_session_data(self) -> dict:
        freq: Dict[str, int] = {}
        for f in self._fragments:
            spk = f.get("speaker")
            if spk and str(spk).strip():
                k = str(spk).strip()
                freq[k] = freq.get(k, 0) + 1
        sorted_speaker_list = sorted(
            self._speaker_list,
            key=lambda s: freq.get(s, 0),
            reverse=True,
        )

        is_supertonic = self._is_supertonic_active()
        is_piper      = self._is_piper_active()
        speaker_voices_data = {}
        for spk in sorted_speaker_list:
            sv = self._speaker_voices.get(spk, {})
            if is_supertonic:
                voice_combo = sv.get("voice_combo")
                speaker_voices_data[spk] = {
                    "voice_name": voice_combo.currentData() if voice_combo else "M1",
                }
            elif is_piper:
                voice_combo = sv.get("voice_combo")
                speaker_voices_data[spk] = {
                    "voice_model": voice_combo.currentData() if voice_combo else "",
                }
            else:
                drop         = sv.get("drop")
                ref_text_wdg = sv.get("ref_text")
                demucs_chk   = sv.get("demucs_chk")
                audio_path   = drop.file_path if drop else None
                speaker_voices_data[spk] = {
                    "audio_path": audio_path,
                    "audio_hash": self._ref_audio_hash(audio_path),
                    "ref_text":   ref_text_wdg.toPlainText() if ref_text_wdg else "",
                    "demucs":     demucs_chk.isChecked() if demucs_chk else False,
                }

        ref_audio = self._drop.file_path
        return {
            "version":              3,
            "srt_path":             self._srt_path,
            "output_dir":           self._output_dir,
            "reference_audio":      ref_audio,
            "reference_audio_hash": self._ref_audio_hash(ref_audio),
            "reference_text":       self._ref_text.toPlainText(),
            "generation_params":    self._get_generation_settings(),
            "normalize_audio":      self._norm_check.isChecked(),
            "target_sr":            self._sr_combo.currentData(),
            "whisper_size":         self._w_size.currentData(),
            "whisper_lang":         self._w_lang.currentData(),
            "lektor": {
                "video_path":          self._vid_path_edit.text().strip() or None,
                "offset_ms":           self._offset_spin.value(),
                "lektor_vol":          self._lektor_vol.value(),
                "orig_vol":            self._orig_vol.value(),
                "autofit":             self._autofit_check.isChecked(),
                "atempo_threshold":    self._atempo_threshold.value(),
                "ducking":             self._duck_check.isChecked(),
                "vocal_suppress":      self._vocal_suppress_check.isChecked(),
                "vocal_suppress_vol":  self._vocal_suppress_spin.value(),
            },
            "dubbing_mode":       self._dubbing_mode,
            "dubbing_video_path": self._dubbing_video_path,
            "speaker_list":       sorted_speaker_list,
            "speaker_voices":     speaker_voices_data,
            "fragments": [
                {k: v for k, v in f.items()}
                for f in self._fragments
            ],
        }

    def _write_session_to(self, path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self._build_session_data(), fh, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.warning(f"Session write failed: {e}")
            return False

    def _restore_session_data(self, data: dict):
        self._srt_path   = data.get("srt_path", "")
        self._output_dir = data.get("output_dir", str(OUTPUTS_DIR))
        self._fragments  = data.get("fragments", [])
 
        for f in self._fragments:
            start_ms = f.get("start_ms") or 0
            end_ms   = f.get("end_ms") or 0
            f["timestamp"] = f"{_ms_to_ts(start_ms)} --> {_ms_to_ts(end_ms)}"
 
        missing = 0
        for f in self._fragments:
            if (f.get("status") == "done"
                    and f.get("output_path")
                    and not os.path.exists(f.get("output_path", ""))):
                f["status"]      = "waiting"
                f["output_path"] = None
                missing += 1
 
        ref_audio  = data.get("reference_audio")
        saved_hash = data.get("reference_audio_hash")
        if ref_audio and os.path.exists(ref_audio):
            current_hash = self._ref_audio_hash(ref_audio)
            if saved_hash and current_hash and current_hash != saved_hash:
                QMessageBox.warning(
                    self, "Reference audio changed",
                    f"The reference audio file has changed since the session was saved:\n"
                    f"{ref_audio}\n\nFragments marked as done may not match the current voice.",
                )
            self._drop._set(ref_audio)
            self._ref_player.load(ref_audio)
        elif ref_audio:
            self._set_status(
                f"Reference audio not found: {Path(ref_audio).name}", C["warning"]
            )
 
        self._ref_text.setPlainText(data.get("reference_text", ""))
 
        params = data.get("generation_params", {})
        for key, value in params.items():
            widget = self._param_widgets.get(key) if hasattr(self, "_param_widgets") else None
            if widget is None:
                continue
            if isinstance(widget, QSlider):
                widget.setValue(int(value * 100))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))
 
        self._norm_check.setChecked(data.get("normalize_audio", False))
 
        target_sr = data.get("target_sr")
        if target_sr is not None:
            for i in range(self._sr_combo.count()):
                if self._sr_combo.itemData(i) == target_sr:
                    self._sr_combo.setCurrentIndex(i)
                    break
 
        whisper_size = data.get("whisper_size")
        if whisper_size:
            for i in range(self._w_size.count()):
                if self._w_size.itemData(i) == whisper_size:
                    self._w_size.setCurrentIndex(i)
                    break
 
        whisper_lang = data.get("whisper_lang")
        if whisper_lang:
            for i in range(self._w_lang.count()):
                if self._w_lang.itemData(i) == whisper_lang:
                    self._w_lang.setCurrentIndex(i)
                    break
 
        lektor = data.get("lektor", {})
        vid_path = lektor.get("video_path") or ""
        self._vid_path_edit.setText(vid_path)
        self._offset_spin.setValue(lektor.get("offset_ms", 0))
        self._lektor_vol.setValue(lektor.get("lektor_vol", 100))
        self._orig_vol.setValue(lektor.get("orig_vol", 100))
        self._autofit_check.setChecked(lektor.get("autofit", True))
        self._atempo_threshold.setValue(lektor.get("atempo_threshold", 300))
        self._duck_check.setChecked(lektor.get("ducking", False))
        self._vocal_suppress_check.setChecked(lektor.get("vocal_suppress", False))
        self._vocal_suppress_spin.setValue(lektor.get("vocal_suppress_vol", 80))
        self._vocal_suppress_spin.setEnabled(
            lektor.get("vocal_suppress", False) and self._ffmpeg_ok
        )
 
        dubbing_mode = data.get("dubbing_mode", False)
        if dubbing_mode:
            self._dubbing_mode       = True
            self._dubbing_video_path = data.get("dubbing_video_path")
 
            saved_speaker_list = data.get("speaker_list", [])
            freq: Dict[str, int] = {}
            for f in self._fragments:
                spk = f.get("speaker")
                if spk and str(spk).strip():
                    k = str(spk).strip()
                    freq[k] = freq.get(k, 0) + 1
 
            if freq:
                self._speaker_list = sorted(
                    saved_speaker_list,
                    key=lambda s: freq.get(s, 0),
                    reverse=True,
                )
            else:
                self._speaker_list = saved_speaker_list
 
            self._rebuild_voice_cloning_for_speakers()
 
            speaker_voices_data = data.get("speaker_voices", {})
            whisper_ready = (
                self._whisper_backend is not None
                and self._whisper_backend.is_downloaded(self._w_size.currentData())
            )
            is_supertonic = self._is_supertonic_active()
            is_piper      = self._is_piper_active()
            for spk, sv_data in speaker_voices_data.items():
                sv = self._speaker_voices.get(spk)
                if not sv:
                    continue
                if is_supertonic:
                    voice_name  = sv_data.get("voice_name", "M1")
                    voice_combo = sv.get("voice_combo")
                    if voice_combo:
                        for i in range(voice_combo.count()):
                            if voice_combo.itemData(i) == voice_name:
                                voice_combo.setCurrentIndex(i)
                                break
                elif is_piper:
                    voice_model = sv_data.get("voice_model", "")
                    voice_combo = sv.get("voice_combo")
                    if voice_combo and voice_model:
                        for i in range(voice_combo.count()):
                            if voice_combo.itemData(i) == voice_model:
                                voice_combo.setCurrentIndex(i)
                                break
                else:
                    audio_path = sv_data.get("audio_path")
                    if audio_path and os.path.exists(audio_path):
                        sv["drop"]._set(audio_path)
                        sv["player"].load(audio_path)
                        sv["proc_btn"].setEnabled(True)
                        sv["tr_btn"].setEnabled(whisper_ready)
                        info = self._audio_info_str(audio_path)
                        sv["audio_info_lbl"].setText(info)
                    ref_text = sv_data.get("ref_text", "")
                    if ref_text:
                        sv["ref_text"].setPlainText(ref_text)
                    demucs = sv_data.get("demucs", False)
                    sv["demucs_chk"].setChecked(demucs)
 
            if hasattr(self, "_btn_dubbing"):
                self._btn_dubbing.setEnabled(True)
                self._btn_dubbing.setText("✓  Dubbing active")
                self._btn_dubbing.setStyleSheet(_btn(C["success"]))
                self._btn_dubbing.setVisible(True)
 
            if hasattr(self, "_vid_row_widget"):
                self._vid_row_widget.setVisible(False)
            if hasattr(self, "_whisper_section_widget"):
                self._whisper_section_widget.setVisible(False)
        else:
            self._update_dubbing_visibility()
 
        for f in self._fragments:
            ap = f.get("output_path")
            if ap and os.path.exists(ap):
                self._file_watcher.addPath(ap)
 
        fname = Path(self._srt_path).name if self._srt_path else "session"
        done  = sum(1 for f in self._fragments if f.get("status") == "done")
        self._srt_label.setText(
            f"📄  {fname}  •  {len(self._fragments)} fragments"
        )
        self._srt_label.setStyleSheet(
            f"color:{C['accent']};font-size:11px;font-weight:600;"
        )
        self._btn_close_srt.setEnabled(True)
        self._btn_save_session.setEnabled(True)
        self._tabs.setCurrentIndex(0)
        self._populate_tree()
        self._update_action_buttons()
 
        status_msg = f"Session loaded: {len(self._fragments)} fragments, {done} already synthesized."
        if missing:
            status_msg = f"Session loaded — {missing} audio file(s) missing on disk, reset to waiting."
            self._set_status(status_msg, C["warning"])
        else:
            self._set_status(status_msg, C["success"])

    def _auto_ebook_session_path(self) -> Optional[Path]:
        if not self._epub_path:
            return None
        return OUTPUTS_DIR / Path(self._epub_path).stem / "ebook_session.json"
 
    def _build_ebook_session_data(self) -> dict:
        ref_audio = self._drop.file_path
        return {
            "session_type":         "ebook",
            "version":              1,
            "epub_path":            self._epub_path,
            "ebook_output_dir":     self._ebook_output_dir,
            "reference_audio":      ref_audio,
            "reference_audio_hash": self._ref_audio_hash(ref_audio),
            "reference_text":       self._ref_text.toPlainText(),
            "generation_params":    self._get_generation_settings(),
            "normalize_audio":      self._norm_check.isChecked(),
            "audiobook_silence_ms": self._audiobook_silence_spin.value(),
            "audiobook_format":     self._audiobook_fmt_combo.currentData(),
            "fragments": [
                {k: v for k, v in f.items()}
                for f in self._ebook_fragments
            ],
        }
 
    def _write_ebook_session_to(self, path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self._build_ebook_session_data(), fh, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.warning(f"Ebook session write failed: {e}")
            return False
 
    def _save_ebook_session(self):
        if not self._epub_path:
            return
        default_path = str(
            Path(_get_last_dir("session", str(OUTPUTS_DIR))) / "ebook_session.json"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save ebook session", default_path,
            "JSON (*.json);;All files (*)",
        )
        if not path:
            return
        _set_last_dir("session", path)
        if self._write_ebook_session_to(path):
            auto = self._auto_ebook_session_path()
            if auto and str(auto) != path:
                self._write_ebook_session_to(str(auto))
            self._set_status(f"Ebook session saved: {path}", C["success"])
        else:
            QMessageBox.critical(self, "Save error", f"Could not write session to:\n{path}")
 
    def _restore_ebook_session_data(self, data: dict):
        self._epub_path = data.get("epub_path", "")
        self._ebook_output_dir = data.get("ebook_output_dir", str(OUTPUTS_DIR))
        if hasattr(self, '_audiobook_output_edit'):
            self._audiobook_output_edit.setText(self._ebook_output_dir)
        self._ebook_fragments = data.get("fragments", [])
        ref_audio = data.get("reference_audio")
        saved_hash = data.get("reference_audio_hash")
        if ref_audio and os.path.exists(ref_audio):
            current_hash = self._ref_audio_hash(ref_audio)
            if saved_hash and current_hash and current_hash != saved_hash:
                QMessageBox.warning(
                    self, "Reference audio changed",
                    f"The reference audio file has changed since the session was saved:\n"
                    f"{ref_audio}\n\nFragments marked as done may not match the current voice.",
                )
            self._drop._set(ref_audio)
            self._ref_player.load(ref_audio)
        elif ref_audio:
            self._set_status(
                f"Reference audio not found: {Path(ref_audio).name}", C["warning"]
            )
        self._ref_text.setPlainText(data.get("reference_text", ""))
        params = data.get("generation_params", {})
        for key, value in params.items():
            widget = self._param_widgets.get(key) if hasattr(self, "_param_widgets") else None
            if widget is None:
                continue
            if isinstance(widget, QSlider):
                widget.setValue(int(value * 100))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))
        self._norm_check.setChecked(data.get("normalize_audio", False))
        if hasattr(self, '_audiobook_silence_spin'):
            self._audiobook_silence_spin.setValue(data.get("audiobook_silence_ms", 500))
        if hasattr(self, '_audiobook_fmt_combo'):
            fmt = data.get("audiobook_format", "wav")
            for i in range(self._audiobook_fmt_combo.count()):
                if self._audiobook_fmt_combo.itemData(i) == fmt:
                    self._audiobook_fmt_combo.setCurrentIndex(i)
                    break
        missing = sum(
            1 for f in self._ebook_fragments
            if f.get("status") == "done"
            and f.get("output_path")
            and not os.path.exists(f.get("output_path", ""))
        )
        if missing:
            for f in self._ebook_fragments:
                if (f.get("status") == "done"
                        and f.get("output_path")
                        and not os.path.exists(f.get("output_path", ""))):
                    f["status"] = "waiting"
                    f["output_path"] = None
        fname = Path(self._epub_path).name if self._epub_path else "session"
        done = sum(1 for f in self._ebook_fragments if f.get("status") == "done")
        self._epub_label.setText(
            f"📚 {fname} • {len(self._ebook_fragments)} fragments"
        )
        self._epub_label.setStyleSheet(
            f"color:{C['accent']};font-size:11px;font-weight:600;"
        )
        self._btn_close_ebook.setEnabled(True)
        self._btn_save_ebook_session.setEnabled(True)
        self._tabs.setCurrentIndex(1)
        self._populate_ebook_tree()
        self._update_action_buttons()

        self._update_preview_btn_state()

        status_msg = f"Ebook session loaded: {len(self._ebook_fragments)} fragments, {done} already synthesized."
        if missing:
            status_msg = f"Ebook session loaded — {missing} audio file(s) missing on disk, reset to waiting."
            self._set_status(status_msg, C["warning"])
        else:
            self._set_status(status_msg, C["success"])

    def _load_session(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load session", _get_last_dir("session", str(OUTPUTS_DIR)),
            "JSON (*.json);;All files (*)",
        )
        if not path:
            return
        _set_last_dir("session", path)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return

        if data.get("session_type") == "ebook":
            self._restore_ebook_session_data(data)
        else:
            self._reset_dubbing_state()
            self._restore_session_data(data)

    def _get_hf_token(self) -> Optional[str]:
        if HF_TOKEN_FILE.exists():
            try:
                token = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
                if token:
                    return token
            except Exception:
                pass
        return None
 
    def _save_hf_token(self, token: str):
        try:
            HF_TOKEN_FILE.write_text(token.strip(), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Cannot save HF token: {e}")
 
    def _show_hf_token_dialog(self) -> Optional[str]:
        dlg = QDialog(self)
        dlg.setWindowTitle("HuggingFace Token Required")
        dlg.resize(540, 320)
        dlg.setStyleSheet(self.styleSheet())

        lay = QVBoxLayout(dlg)
        lay.setSpacing(10)

        info = QLabel(
            "Speaker diarization requires a HuggingFace token.<br><br>"
            "Steps:<br>"
            f"&nbsp;&nbsp;1. Accept "
            f"<a href='https://huggingface.co/pyannote/segmentation-3.0' style='color:{C['accent']};'>"
            f"pyannote/segmentation-3.0</a> user conditions<br>"
            f"&nbsp;&nbsp;2. Accept "
            f"<a href='https://huggingface.co/pyannote/speaker-diarization-3.1' style='color:{C['accent']};'>"
            f"pyannote/speaker-diarization-3.1</a> user conditions<br>"
            f"&nbsp;&nbsp;3. Accept "
            f"<a href='https://huggingface.co/pyannote/speaker-diarization-community-1' style='color:{C['accent']};'>"
            f"pyannote/speaker-diarization-community-1</a> user conditions<br>"
            f"&nbsp;&nbsp;4. Create a token at "
            f"<a href='https://huggingface.co/settings/tokens' style='color:{C['accent']};'>"
            f"hf.co/settings/tokens</a>:<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;• <b>Classic token</b> with <b>Read</b> scope — simplest option<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;• <b>Fine-grained token</b> — enable:<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<i>User permissions → Repositories →<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\"Read access to contents of all public gated repos you can access\"</i><br><br>"
            "Enter your token below:"
        )
        info.setWordWrap(True)
        info.setTextFormat(Qt.TextFormat.RichText)
        info.setOpenExternalLinks(False)
        info.setStyleSheet(f"color:{C['text2']};font-size:12px;")
        info.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        info.linkActivated.connect(
            lambda url: QDesktopServices.openUrl(QUrl(url))
        )

        token_edit = QLineEdit()
        token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        token_edit.setPlaceholderText("hf_…")
        token_edit.setFixedHeight(30)
        token_edit.setTextMargins(4, 0, 4, 0)

        show_chk = QCheckBox("Show token")
        show_chk.toggled.connect(
            lambda checked: token_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )

        save_chk = QCheckBox("Remember token (saved locally in app folder)")
        save_chk.setChecked(True)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        lay.addWidget(info)
        lay.addWidget(token_edit)
        lay.addWidget(show_chk)
        lay.addWidget(save_chk)
        lay.addWidget(btns)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            token = token_edit.text().strip()
            if token:
                if save_chk.isChecked():
                    self._save_hf_token(token)
                return token
        return None
 
    def _show_model_token_dialog(self) -> Optional[str]:
        dlg = QDialog(self)
        dlg.setWindowTitle("HuggingFace Token Required")
        dlg.resize(460, 240)
        dlg.setStyleSheet(self.styleSheet())

        lay = QVBoxLayout(dlg)
        lay.setSpacing(10)

        info = QLabel(
            "This model requires a HuggingFace token.<br><br>"
            "Steps:<br>"
            f"&nbsp;&nbsp;1. Accept the model license at "
            f"<a href='https://huggingface.co' style='color:{C['accent']};'>huggingface.co</a><br>"
            f"&nbsp;&nbsp;2. Create a token at "
            f"<a href='https://huggingface.co/settings/tokens' style='color:{C['accent']};'>"
            f"hf.co/settings/tokens</a> (Read scope)<br><br>"
            "Enter your token below:"
        )
        info.setWordWrap(True)
        info.setTextFormat(Qt.TextFormat.RichText)
        info.setOpenExternalLinks(False)
        info.setStyleSheet(f"color:{C['text2']};font-size:12px;")
        info.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        info.linkActivated.connect(lambda url: QDesktopServices.openUrl(QUrl(url)))

        token_edit = QLineEdit()
        token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        token_edit.setPlaceholderText("hf_…")
        token_edit.setFixedHeight(30)

        show_chk = QCheckBox("Show token")
        show_chk.toggled.connect(
            lambda checked: token_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )

        save_chk = QCheckBox("Remember token (saved locally in app folder)")
        save_chk.setChecked(True)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        lay.addWidget(info)
        lay.addWidget(token_edit)
        lay.addWidget(show_chk)
        lay.addWidget(save_chk)
        lay.addWidget(btns)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            token = token_edit.text().strip()
            if token:
                if save_chk.isChecked():
                    self._save_hf_token(token)
                return token
        return None
 
    def _on_dubbing_clicked(self):
        backend = self._backend
        is_tada = backend is not None and getattr(backend, "auth_required", False)

        if is_tada:
            hf_token = self._get_hf_token()
            if not hf_token:
                QMessageBox.critical(
                    self,
                    "HuggingFace token missing",
                    "Model requires a HuggingFace token.\n\n"
                    "Make sure the token was saved, or place your token "
                    "in the '.hf_token' file in the application folder.",
                )
                return
        else:
            hf_token = self._get_hf_token()
            if not hf_token:
                hf_token = self._show_hf_token_dialog()
                if not hf_token:
                    return

        path, _ = QFileDialog.getOpenFileName(
            self, "Select video file for dubbing", _get_last_dir("video"),
            "Video (*.mp4 *.mkv *.avi *.mov *.webm *.wmv);;All files (*)",
        )
        if not path:
            return

        _set_last_dir("video", path)
        self._dubbing_video_path = path
        self._hf_token           = hf_token
        self._btn_dubbing.setEnabled(False)
        self._btn_dubbing.setText("⏳  Processing…")

        self._start_vocal_extraction(path)
 
    def _start_vocal_extraction(self, video_path: str):
        self._set_status("Extracting audio and isolating vocals.")
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
 
        self._w_vocal_extract = DubbingVocalExtractWorker(video_path)
        self._w_vocal_extract.status.connect(lambda m: self._set_status(m))
        self._w_vocal_extract.finished.connect(self._on_vocal_extraction_done)
        self._w_vocal_extract.error.connect(
            lambda e: self._on_error("Vocal extraction error", e,
                reset_fn=lambda: (
                    self._btn_dubbing.setEnabled(True),
                    self._btn_dubbing.setText("🎙  I want dubbing"),
                    self._progress.setVisible(False),
                ))
        )
        self._w_vocal_extract.start()
 
    def _on_vocal_extraction_done(self, vocals_path: str):
        self._set_status(
            f"Vocals extracted: {Path(vocals_path).name} — starting diarization…"
        )
        self._start_diarization(vocals_path)
 
    def _start_diarization(self, audio_path: str):
        self._set_status("Running speaker diarization (pyannote)…")
 
        self._w_diarization = DiarizationWorker(audio_path, self._hf_token)
        self._w_diarization.status.connect(lambda m: self._set_status(m))
        self._w_diarization.finished.connect(self._on_diarization_done)
        self._w_diarization.error.connect(
            lambda e: self._on_error("Diarization error", e,
                reset_fn=lambda: (
                    self._btn_dubbing.setEnabled(True),
                    self._btn_dubbing.setText("🎙  I want dubbing"),
                    self._progress.setVisible(False),
                ))
        )
        self._w_diarization.start()
 
    def _on_diarization_done(self, result: dict):
        self._progress.setVisible(False)
 
        segments    = result.get("segments", [])
        speaker_map = result.get("speaker_map", {})
 
        if not speaker_map:
            QMessageBox.warning(self, "No speakers detected",
                "Speaker diarization found no speakers in the audio.")
            self._btn_dubbing.setEnabled(True)
            self._btn_dubbing.setText("🎙  I want dubbing")
            return
 
        self._assign_speakers_from_diarization(segments, speaker_map)
 
        self._speaker_list = sorted(
            set(speaker_map.values()),
            key=lambda s: int(s.split()[-1]) if s.split()[-1].isdigit() else 999,
        )
 
        self._dubbing_mode = True
        self._rebuild_voice_cloning_for_speakers()
 
        if hasattr(self, "_vid_row_widget"):
            self._vid_row_widget.setVisible(False)
 
        self._populate_tree()
 
        n = len(speaker_map)
        self._set_status(
            f"Diarization complete — {n} speaker{'s' if n != 1 else ''} detected. "
            f"Set reference audio for each speaker in 'Voice cloning'.",
            C["success"],
        )
        self._btn_dubbing.setEnabled(True)
        self._btn_dubbing.setText("✓  Dubbing active")
        self._btn_dubbing.setStyleSheet(_btn(C["success"]))
 
    def _assign_speakers_from_diarization(
        self, segments: List[Dict], speaker_map: Dict[str, str]
    ):
        for frag in self._fragments:
            start_s = (frag.get("start_ms") or 0) / 1000.0
            end_s   = (frag.get("end_ms")   or 0) / 1000.0
            if end_s <= start_s:
                continue
 
            overlap: Dict[str, float] = {}
            for seg in segments:
                ov_start = max(start_s, seg["start"])
                ov_end   = min(end_s,   seg["end"])
                if ov_end > ov_start:
                    label = seg["speaker"]
                    overlap[label] = overlap.get(label, 0.0) + (ov_end - ov_start)
 
            if overlap:
                best_label      = max(overlap, key=lambda k: overlap[k])
                frag["speaker"] = speaker_map.get(best_label, best_label)
 
    def _rebuild_voice_cloning_for_speakers(self):
        while self._speakers_lay.count():
            item = self._speakers_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._speaker_voices.clear()
        is_supertonic = self._is_supertonic_active()
        is_piper      = self._is_piper_active()

        for spk in self._speaker_list:
            spk_group = QGroupBox(f"🎤  {spk}")
            spk_group.setStyleSheet(f"""
                QGroupBox {{
                    border: 1px solid {C['border2']};
                    border-radius: 8px;
                    margin-top: 18px;
                    padding: 12px 8px 8px 8px;
                    background: {C['surface']};
                    font-weight: 600;
                    color: {C['accent']};
                    font-size: 11px;
                    letter-spacing: 1px;
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    subcontrol-position: top left;
                    left: 10px; top: 2px;
                    padding: 0 6px;
                    background: {C['surface']};
                }}
            """)
            spk_lay = QVBoxLayout(spk_group)
            spk_lay.setSpacing(6)
            spk_lay.setContentsMargins(4, 4, 4, 4)

            if is_supertonic:
                voice_row = QHBoxLayout()
                voice_row.setSpacing(8)
                voice_lbl = QLabel("Voice:")
                voice_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
                voice_lbl.setFixedWidth(44)
                voice_combo = QComboBox()
                for vcode, vdisplay in SUPERTONIC_VOICES:
                    voice_combo.addItem(vdisplay, vcode)
                voice_combo.setCurrentIndex(0)
                voice_row.addWidget(voice_lbl)
                voice_row.addWidget(voice_combo, 1)
                spk_lay.addLayout(voice_row)

                hint_lbl = QLabel("Language is set globally in Generation Settings.")
                hint_lbl.setStyleSheet(
                    f"color:{C['text3']};font-size:10px;font-style:italic;"
                )
                spk_lay.addWidget(hint_lbl)

                self._speaker_voices[spk] = {"voice_combo": voice_combo}

            elif is_piper:
                voice_row = QHBoxLayout()
                voice_row.setSpacing(8)
                voice_lbl = QLabel("Voice:")
                voice_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
                voice_lbl.setFixedWidth(44)
                voice_combo = QComboBox()
                piper_opts = _piper_voice_options()
                for vcode, vdisplay in piper_opts:
                    voice_combo.addItem(vdisplay, vcode)
                if piper_opts:
                    voice_combo.setCurrentIndex(0)
                voice_row.addWidget(voice_lbl)
                voice_row.addWidget(voice_combo, 1)
                spk_lay.addLayout(voice_row)

                hint_lbl = QLabel("Other generation settings (speed, variation) are set globally.")
                hint_lbl.setStyleSheet(
                    f"color:{C['text3']};font-size:10px;font-style:italic;"
                )
                spk_lay.addWidget(hint_lbl)

                self._speaker_voices[spk] = {"voice_combo": voice_combo}

            else:
                tab_widget = QTabWidget()
                tab_widget.setStyleSheet("""
                    QTabWidget::pane {
                        border: 1px solid #333333;
                        border-radius: 0 3px 3px 3px;
                        background: #252525;
                    }
                    QTabBar::tab {
                        background: #1e1e1e;
                        border: 1px solid #333333;
                        border-bottom: none;
                        border-top-left-radius: 3px;
                        border-top-right-radius: 3px;
                        color: #777777;
                        padding: 4px 10px;
                        font-size: 11px;
                        margin-right: 2px;
                    }
                    QTabBar::tab:selected {
                        background: #252525;
                        color: #cccccc;
                        border-color: #444444;
                    }
                    QTabBar::tab:hover {
                        background: #2a2a2a;
                        color: #aaaaaa;
                    }
                """)

                clone_tab = QWidget()
                clone_tab.setStyleSheet("background: transparent;")
                clone_lay = QVBoxLayout(clone_tab)
                clone_lay.setContentsMargins(6, 6, 6, 6)
                clone_lay.setSpacing(6)

                drop = DropAudioWidget()
                drop.file_dropped.connect(
                    lambda path, s=spk: self._on_speaker_ref_dropped(s, path)
                )
                clone_lay.addWidget(drop)

                player = RefAudioPlayer()
                clone_lay.addWidget(player)

                btn_row = QHBoxLayout()
                btn_row.setSpacing(6)
                clear_btn = QPushButton("✕  Remove reference")
                clear_btn.setFixedHeight(26)
                clear_btn.clicked.connect(lambda _, s=spk: self._clear_speaker_ref(s))
                btn_row.addWidget(clear_btn)
                btn_row.addStretch()
                clone_lay.addLayout(btn_row)

                audio_info_lbl = QLabel("")
                audio_info_lbl.setStyleSheet(
                    f"color:{C['text2']};font-size:11px;font-style:italic;"
                )
                clone_lay.addWidget(audio_info_lbl)

                rt_lbl = QLabel("Reference audio transcription:")
                rt_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
                ref_text = QPlainTextEdit()
                ref_text.setPlaceholderText(
                    "Enter or auto-transcribe reference audio for this speaker…"
                )
                ref_text.setFixedHeight(60)
                ref_text.setFont(QFont("Segoe UI", 11))
                clone_lay.addWidget(rt_lbl)
                clone_lay.addWidget(ref_text)

                tab_widget.addTab(clone_tab, "🎙 Voice Cloning")

                proc_tab = QWidget()
                proc_tab.setStyleSheet("background: transparent;")
                proc_lay = QVBoxLayout(proc_tab)
                proc_lay.setContentsMargins(6, 6, 6, 6)
                proc_lay.setSpacing(6)

                w_size_row = QHBoxLayout()
                w_size_lbl = QLabel("Whisper size:")
                w_size_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
                w_size_lbl.setFixedWidth(100)
                w_size = QComboBox()
                for s in WHISPER_SIZES:
                    w_size.addItem(f"{s}  ({WHISPER_SIZE_MB.get(s, '')})", s)
                w_size.setCurrentIndex(
                    WHISPER_SIZES.index(self._w_size.currentData())
                    if self._w_size.currentData() in WHISPER_SIZES else 0
                )
                w_size_row.addWidget(w_size_lbl)
                w_size_row.addWidget(w_size, 1)
                proc_lay.addLayout(w_size_row)

                w_lang_row = QHBoxLayout()
                w_lang_lbl = QLabel("Transcription language:")
                w_lang_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
                w_lang_lbl.setFixedWidth(100)
                w_lang = QComboBox()
                for code, label in WHISPER_LANGS:
                    w_lang.addItem(label, code)
                w_lang_row.addWidget(w_lang_lbl)
                w_lang_row.addWidget(w_lang, 1)
                proc_lay.addLayout(w_lang_row)

                mono_check  = QCheckBox("Convert to mono")
                mono_check.setChecked(self._mono_check.isChecked())
                sr_combo    = QComboBox()
                for val, label in TARGET_SR_OPTIONS:
                    sr_combo.addItem(label, val)
                for i in range(sr_combo.count()):
                    if sr_combo.itemData(i) == self._sr_combo.currentData():
                        sr_combo.setCurrentIndex(i)
                        break

                chk_bit_depth   = QCheckBox("Override bit depth")
                chk_bit_depth.setChecked(self._chk_bit_depth.isChecked())
                bit_depth_combo = QComboBox()
                for code, label in [("PCM_16","16-bit"),("PCM_24","24-bit"),("FLOAT","32-bit float")]:
                    bit_depth_combo.addItem(label, code)
                for i in range(bit_depth_combo.count()):
                    if bit_depth_combo.itemData(i) == self._bit_depth_combo.currentData():
                        bit_depth_combo.setCurrentIndex(i)
                        break

                sr_row = QHBoxLayout()
                sr_lbl = QLabel("Target sample rate:")
                sr_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
                sr_lbl.setFixedWidth(100)
                sr_row.addWidget(sr_lbl)
                sr_row.addWidget(sr_combo, 1)

                demucs_chk = QCheckBox("Isolate vocals (Demucs)")
                demucs_chk.setToolTip(
                    "Run Demucs source separation to strip background noise / music "
                    "from the reference audio before cloning."
                )

                proc_btn = QPushButton("🔧  Process audio")
                proc_btn.setFixedHeight(28)
                proc_btn.setEnabled(False)
                proc_btn.setStyleSheet(_btn(C["proc"]))
                proc_btn.clicked.connect(
                    lambda _, s=spk: self._process_speaker_audio(s)
                )

                tr_btn = QPushButton("🎤  Transcribe (Whisper)")
                tr_btn.setFixedHeight(28)
                tr_btn.setEnabled(False)
                tr_btn.setStyleSheet(_btn(C["whisper"]))
                tr_btn.clicked.connect(
                    lambda _, s=spk: self._transcribe_speaker_ref(s)
                )

                proc_lay.addWidget(mono_check)
                proc_lay.addLayout(sr_row)
                proc_lay.addWidget(chk_bit_depth)
                proc_lay.addWidget(bit_depth_combo)
                proc_lay.addWidget(demucs_chk)
                proc_lay.addWidget(proc_btn)
                proc_lay.addWidget(tr_btn)

                tab_widget.addTab(proc_tab, "🔧 Prepare Audio")

                spk_lay.addWidget(tab_widget)

                self._speaker_voices[spk] = {
                    "drop":            drop,
                    "player":          player,
                    "ref_text":        ref_text,
                    "proc_btn":        proc_btn,
                    "tr_btn":          tr_btn,
                    "demucs_chk":      demucs_chk,
                    "audio_info_lbl":  audio_info_lbl,
                    "w_size":          w_size,
                    "w_lang":          w_lang,
                    "mono_check":      mono_check,
                    "chk_bit_depth":   chk_bit_depth,
                    "bit_depth_combo": bit_depth_combo,
                    "sr_combo":        sr_combo,
                }

            self._speakers_lay.addWidget(spk_group)

        self._voice_single_container.setVisible(False)
        self._speakers_container.setVisible(True)

        if hasattr(self, "_whisper_section_widget"):
            self._whisper_section_widget.setVisible(False)

        if hasattr(self, "_voice_cloning_section"):
            self._voice_cloning_section.expand()
 
    def _get_speaker_voices_dict(self) -> Optional[Dict]:
        if not self._dubbing_mode or not self._speaker_list:
            return None
        result        = {}
        is_supertonic = self._is_supertonic_active()
        is_piper      = self._is_piper_active()
        for spk in self._speaker_list:
            sv = self._speaker_voices.get(spk, {})
            if is_supertonic:
                voice_combo = sv.get("voice_combo")
                voice_name  = voice_combo.currentData() if voice_combo else "M1"
                result[spk] = (None, voice_name)
            elif is_piper:
                voice_combo  = sv.get("voice_combo")
                voice_model  = voice_combo.currentData() if voice_combo else ""
                result[spk]  = (None, voice_model)
            else:
                drop         = sv.get("drop")
                ref_text_wdg = sv.get("ref_text")
                audio_path   = drop.file_path if drop else None
                text         = ref_text_wdg.toPlainText().strip() if ref_text_wdg else None
                result[spk]  = (audio_path, text or None)
        if is_supertonic or is_piper:
            return result if result else None
        has_any_audio = any(v[0] for v in result.values())
        return result if has_any_audio else None
 
    def _on_speaker_ref_dropped(self, speaker: str, path: str):
        sv = self._speaker_voices.get(speaker)
        if not sv:
            return
        sv["player"].load(path)
        sv["proc_btn"].setEnabled(True)
        w_size_combo = sv.get("w_size", self._w_size)
        sv["tr_btn"].setEnabled(
            self._whisper_backend is not None
            and self._whisper_backend.is_downloaded(w_size_combo.currentData())
        )
        info = self._audio_info_str(path)
        sv["audio_info_lbl"].setText(info)
        self._set_status(
            f"[{speaker}] Reference audio: {Path(path).name}", C["accent"]
        )
 
    def _clear_speaker_ref(self, speaker: str):
        sv = self._speaker_voices.get(speaker)
        if not sv:
            return
        sv["drop"].clear_file()
        sv["player"].clear()
        sv["ref_text"].clear()
        sv["proc_btn"].setEnabled(False)
        sv["tr_btn"].setEnabled(False)
        sv["audio_info_lbl"].setText("")
        self._set_status(f"[{speaker}] Reference audio removed.")
 
    def _transcribe_speaker_ref(self, speaker: str):
        sv = self._speaker_voices.get(speaker)
        if not sv:
            return
        path = sv["drop"].file_path
        if not path:
            QMessageBox.warning(
                self, "No audio",
                f"Upload reference audio for {speaker} first."
            )
            return

        w_size_combo = sv.get("w_size", self._w_size)
        size = w_size_combo.currentData()

        if not self._whisper_backend.is_downloaded(size):
            reply = QMessageBox.question(
                self, f"Download Whisper {size}",
                f"Whisper {size} ({WHISPER_SIZE_MB.get(size, '')}) is not downloaded.\n"
                f"Download now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._w_size.setCurrentIndex(w_size_combo.currentIndex())
                self._start_whisper_download()
            return

        w_lang_combo = sv.get("w_lang", self._w_lang)
        lang = w_lang_combo.currentData()

        sv["tr_btn"].setEnabled(False)
        sv["tr_btn"].setText("⏳  Transcribing…")
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)

        self._w_tr = TranscribeWorker(self._whisper_backend, path, size, lang)
        self._w_tr.status.connect(lambda m: self._set_status(m))
        self._w_tr.finished.connect(
            lambda text, s=speaker: self._on_speaker_transcribe_done(s, text)
        )
        self._w_tr.error.connect(
            lambda e, s=speaker: self._on_error("Transcription error", e,
                reset_fn=lambda: (
                    sv["tr_btn"].setEnabled(True),
                    sv["tr_btn"].setText("🎤  Transcribe (Whisper)"),
                    self._progress.setVisible(False),
                ))
        )
        self._w_tr.start()
 
    def _on_speaker_transcribe_done(self, speaker: str, text: str):
        sv = self._speaker_voices.get(speaker)
        if sv:
            sv["ref_text"].setPlainText(text)
            sv["tr_btn"].setEnabled(True)
            sv["tr_btn"].setText("🎤  Transcribe (Whisper)")
        self._progress.setVisible(False)
        self._set_status(
            f"[{speaker}] Transcription complete: {len(text)} chars", C["success"]
        )
 
    def _process_speaker_audio(self, speaker: str):
        sv = self._speaker_voices.get(speaker)
        if not sv:
            return
        path = sv["drop"].file_path
        if not path:
            QMessageBox.warning(
                self, "No audio",
                f"Upload reference audio for {speaker} first."
            )
            return

        chk_bit = sv.get("chk_bit_depth", self._chk_bit_depth)
        bd_combo = sv.get("bit_depth_combo", self._bit_depth_combo)
        output_subtype = (
            bd_combo.currentData()
            if chk_bit.isChecked()
            else "PCM_16"
        )

        safe_spk = "".join(c for c in speaker if c.isalnum() or c in " -_")
        output_name = f"speaker_{safe_spk}_processed.wav"

        mono_chk = sv.get("mono_check", self._mono_check)
        sr_cbo   = sv.get("sr_combo",   self._sr_combo)

        settings = {
            "target_sr":      sr_cbo.currentData(),
            "to_mono":        mono_chk.isChecked(),
            "isolate_vocals": sv["demucs_chk"].isChecked(),
            "normalize":      True,
            "output_subtype": output_subtype,
            "device":         "cuda" if (
                self._backend and self._backend.device == "cuda"
            ) else "cpu",
            "output_name":    output_name,
        }

        sv["proc_btn"].setEnabled(False)
        sv["proc_btn"].setText("⏳  Processing…")
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)

        self._w_proc = AudioProcessWorker(self._preprocessor, path, settings)
        self._w_proc.status.connect(lambda m: self._set_status(m))
        self._w_proc.finished.connect(
            lambda out, s=speaker: self._on_speaker_proc_done(s, out)
        )
        self._w_proc.error.connect(
            lambda e, s=speaker: self._on_error("Audio processing error", e,
                reset_fn=lambda: (
                    sv["proc_btn"].setEnabled(True),
                    sv["proc_btn"].setText("🔧  Process audio"),
                    self._progress.setVisible(False),
                ))
        )
        self._w_proc.start()
 
    def _on_speaker_proc_done(self, speaker: str, out: str):
        sv = self._speaker_voices.get(speaker)
        if not sv:
            return
        sv["proc_btn"].setEnabled(True)
        sv["proc_btn"].setText("🔧  Process audio")
        self._progress.setVisible(False)
        sv["drop"]._set(out)
        sv["player"].load(out)
        w_size_combo = sv.get("w_size", self._w_size)
        sv["tr_btn"].setEnabled(
            self._whisper_backend is not None
            and self._whisper_backend.is_downloaded(w_size_combo.currentData())
        )
        self._set_status(
            f"[{speaker}] Audio processed: {Path(out).name}", C["success"]
        )
 
    def _reset_dubbing_state(self):
        self._dubbing_mode       = False
        self._dubbing_video_path = None
        self._hf_token           = None
        self._speaker_list       = []
        self._speaker_voices     = {}

        if hasattr(self, "_btn_dubbing"):
            self._btn_dubbing.setText("🎙  I want dubbing")
            self._btn_dubbing.setStyleSheet("")
            self._btn_dubbing.setVisible(False)
            self._btn_dubbing.setEnabled(False)

        if hasattr(self, "_voice_single_container"):
            self._voice_single_container.setVisible(True)

        if hasattr(self, "_speakers_lay") and hasattr(self, "_speakers_container"):
            while self._speakers_lay.count():
                item = self._speakers_lay.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self._speakers_container.setVisible(False)

        if hasattr(self, "_whisper_section_widget"):
            self._whisper_section_widget.setVisible(True)

        if hasattr(self, "_vid_row_widget"):
            self._vid_row_widget.setVisible(True)

    def _on_error(self, title: str, tb: str, reset_fn=None):
        logger.error(f"{title}:\n{tb}")
        if reset_fn:
            try:
                reset_fn()
            except Exception:
                pass
        last_line = [l.strip() for l in tb.strip().splitlines() if l.strip()][-1]
        self._set_status(f"{title}: {last_line}", C["error"])
        QMessageBox.critical(self, title, f"{last_line}\n\nFull traceback in console.")

    def closeEvent(self, e):
        if self._worker and self._worker.isRunning():
            self._worker.request_cancel()
            self._worker.wait(10000)
            if self._worker.isRunning():
                self._worker.terminate()
                self._worker.wait(3000)
        self._stop_play()
        if self._audio_tmp and Path(self._audio_tmp).exists():
            try:
                os.unlink(self._audio_tmp)
            except Exception:
                pass
        e.accept()


def main():
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "TTSStudio.1"
        )

    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    if _ACTIVE_BACKEND_CLASSES:
        first = object.__new__(_ACTIVE_BACKEND_CLASSES[0])
        try:
            title_str = first.header_title
        except Exception:
            title_str = "TTS Studio"
    else:
        title_str = "TTS Studio"

    app.setApplicationName(f"{title_str} — SRT Lektor Studio")

    _icon_path = ROOT_DIR / "icon.ico"
    if _icon_path.exists():
        app_icon = QIcon(str(_icon_path))
    else:
        pm = QPixmap(64, 64)
        pm.fill(QColor(C["surface"]))
        p = QPainter(pm)
        p.setFont(QFont("Segoe UI Emoji", 32))
        p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "💀")
        p.end()
        app_icon = QIcon(pm)

    app.setWindowIcon(app_icon)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
