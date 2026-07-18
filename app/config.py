# config.py
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# --- import for backend detection (added) ---
from tts_backends import detect_active_backends, load_active_backend_modules, get_loaded_module

APP_NAME = "SRT Lektor/Dubbing Studio"

APP_DIR     = Path(__file__).parent
ROOT_DIR    = APP_DIR.parent
WHISPER_DIR = ROOT_DIR / "models" / "whisper"
OUTPUTS_DIR = ROOT_DIR / "outputs"
PROC_DIR    = ROOT_DIR / "outputs" / "preprocessed"
HF_TOKEN_FILE = ROOT_DIR / ".hf_token"

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

# ----------------------------------------------------------------------
# Dodane na końcu zgodnie z poprawkami – aktywne backendy TTS i listy głosów
# ----------------------------------------------------------------------

load_active_backend_modules()

_ACTIVE_BACKEND_CLASSES = detect_active_backends()
MODEL_OPTIONS = [
    (cls().model_id, cls().display_name)
    for cls in _ACTIVE_BACKEND_CLASSES
]

SUPERTONIC_VOICES = getattr(get_loaded_module("supertonic_backend"), "SUPERTONIC_VOICES", [])
_piper_voice_options = getattr(get_loaded_module("piper_backend"), "_voice_options", lambda *a, **k: [])

def _fmt(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    millis = int((secs - int(secs)) * 1000)
    secs_int = int(secs)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs_int:02d},{millis:03d}"
    return f"{minutes:02d}:{secs_int:02d},{millis:03d}"


def _fmt_ms(ms: int) -> str:
    """Format milliseconds as SRT timestamp: HH:MM:SS,mmm"""
    if ms < 0:
        ms = 0
    seconds = ms / 1000.0
    return _fmt(seconds)


def _open_file(path: str) -> None:
    """Open file with default system application"""
    import os
    import subprocess
    import sys
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.run(["open", path], check=True)
    else:
        subprocess.run(["xdg-open", path], check=True)


def _check_ffmpeg() -> bool:
    """Check if ffmpeg is available in PATH"""
    import shutil
    import subprocess
    if shutil.which("ffmpeg") is None:
        return False
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def _get_wav_duration(path: str) -> Optional[float]:
    """Get duration of WAV file in seconds, or None on error"""
    try:
        import soundfile as sf
        info = sf.info(path)
        return info.duration
    except Exception:
        return None


def _detect_srt_language(text: str) -> str:
    """Detect language code from text using lingua, fallback 'en'"""
    try:
        from lingua import LanguageDetectorBuilder
        detector = LanguageDetectorBuilder.from_all_languages().build()
        lang = detector.detect_language_of(text)
        if lang is None:
            return "en"
        return lang.iso_code_639_1.name.lower()
    except Exception:
        return "en"


def _convert_numbers_in_text(text: str, lang: str = "en") -> str:
    """
    Convert numeric digits in text to words using num2words.
    Supports simple integer numbers (separate tokens).
    """
    import re
    try:
        from num2words import num2words
        # Przetwarzamy tylko liczby całkowite oddzielone spacjami
        def repl(match: re.Match) -> str:
            num_str = match.group(0)
            try:
                num = int(num_str)
                return num2words(num, lang=lang)
            except Exception:
                return num_str
        # Wyrażenie regularne na liczby całkowite (nie będące częścią innych tokenów)
        pattern = r'\b\d+\b'
        return re.sub(pattern, repl, text)
    except Exception:
        return text


def _normalize_text_for_tts(text: str, lang: str = "en") -> str:
    """
    Basic normalization for TTS: convert numbers, expand abbreviations if needed.
    """
    # Prosta normalizacja – usuń nadmiarowe spacje, zamień liczby
    import re
    text = re.sub(r'\s+', ' ', text).strip()
    text = _convert_numbers_in_text(text, lang)
    # Dodatkowo można dodać inne reguły, np. rozszerzenie skrótów
    return text


def _compute_trim_bounds(audio: np.ndarray, sr: int, aggressiveness: float) -> Tuple[int, int]:
    """
    Compute start and end sample indices for trimming silence.
    aggressiveness: 0.0 = none, 1.0 = aggressive, values beyond 1.0 possible.
    """
    import numpy as np
    if audio is None or len(audio) == 0:
        return 0, len(audio)
    # Oblicz RMS dla małych okien
    window_ms = 20
    window_samples = int(sr * window_ms / 1000)
    if window_samples < 1:
        window_samples = 1
    # Podział na okna i RMS
    n_windows = len(audio) // window_samples
    if n_windows == 0:
        return 0, len(audio)
    rms = np.array([
        np.sqrt(np.mean(audio[i*window_samples:(i+1)*window_samples]**2))
        for i in range(n_windows)
    ])
    # Próg: średnia RMS razy współczynnik
    threshold = np.mean(rms) * (0.02 + 0.08 * aggressiveness)  # aggressiveness 0-3 daje zakres
    threshold = max(threshold, 1e-6)
    # Znajdź pierwsze i ostatnie okno powyżej progu
    above = np.where(rms > threshold)[0]
    if len(above) == 0:
        return 0, len(audio)
    start_idx = above[0] * window_samples
    end_idx = (above[-1] + 1) * window_samples
    # Bezpieczne ograniczenie
    start_idx = max(0, start_idx)
    end_idx = min(len(audio), end_idx)
    return int(start_idx), int(end_idx)


def _trim_silence_wav(path: str, aggressiveness: float) -> Optional[str]:
    """
    Trim silence from WAV file and save as new file with '_trimmed' suffix.
    Returns path to trimmed file or None on error.
    """
    import os
    import tempfile
    import soundfile as sf
    import numpy as np
    try:
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # do mono
        start, end = _compute_trim_bounds(audio, sr, aggressiveness)
        if start >= end:
            return None
        trimmed = audio[start:end]
        # Zapisz do pliku tymczasowego z sufiksem _trimmed
        base, ext = os.path.splitext(path)
        out_path = f"{base}_trimmed.wav"
        # Zapisz z tym samym sample rate
        sf.write(out_path, trimmed, sr, subtype='PCM_16')
        return out_path
    except Exception:
        return None


# Globalny słownik do przechowywania ostatnich ścieżek dla różnych kategorii
_last_dirs_cache: Dict[str, str] = {}
_last_dirs_file = APP_DIR / "last_dirs.json"


def _get_last_dir(category: str) -> str:
    """Get last directory for a category, falls back to str(ROOT_DIR)"""
    import json
    global _last_dirs_cache
    if category not in _last_dirs_cache:
        # Spróbuj wczytać z pliku
        try:
            if _last_dirs_file.exists():
                with open(_last_dirs_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    _last_dirs_cache.update(data)
        except Exception:
            pass
    return _last_dirs_cache.get(category, str(ROOT_DIR))


def _set_last_dir(category: str, path: str) -> None:
    """Save last directory for a category"""
    import json
    global _last_dirs_cache
    _last_dirs_cache[category] = path
    try:
        with open(_last_dirs_file, 'w', encoding='utf-8') as f:
            json.dump(_last_dirs_cache, f, indent=2)
    except Exception:
        pass