# core.py
"""
SRT Lektor/Dubbing Studio - Main Window
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

try:
    import torch
    import torchaudio
    import torchaudio.functional as TAF
    TORCH_AVAILABLE = True
except ImportError as _torch_import_error:
    torch = None
    torchaudio = None
    TAF = None
    TORCH_AVAILABLE = False
    print(f"[WARN] torch/torchaudio not available in this venv: {_torch_import_error}", file=sys.stderr)

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

from input_formats import get_format
from tts_backends import (
    SynthesisRequest, SynthesisResult,
    detect_active_backends, create_backend,
    load_active_backend_modules, get_loaded_module,
    InferenceError,
)

from config import (
    APP_DIR, ROOT_DIR, WHISPER_DIR, OUTPUTS_DIR, PROC_DIR,
    HF_TOKEN_FILE, MODEL_OPTIONS, C, STYLE, SYNTH_BTN_STYLE, TAG_BTN, _btn,
    WHISPER_REPOS, WHISPER_SIZES, WHISPER_SIZE_MB, WHISPER_LANGS,
    TARGET_SR_OPTIONS,
    STATUS_WAITING, STATUS_RUNNING, STATUS_DONE, STATUS_ERROR,
    COL_STATUS, COL_FRAGMENT, COL_SPEAKER, COL_TIMING,
    _ACTIVE_BACKEND_CLASSES,
    SUPERTONIC_VOICES,
    _piper_voice_options,
    _fmt, _fmt_ms, _open_file, _check_ffmpeg, _get_wav_duration,
    _detect_srt_language, _convert_numbers_in_text, _normalize_text_for_tts,
    _trim_silence_wav, _compute_trim_bounds, _get_last_dir, _set_last_dir,
)

from widgets import (
    CollapsibleSection, FlowLayout, FragmentTreeWidget,
    DropAudioWidget, RefAudioPlayer, VideoAudioPlayer, WaveformWidget,
    SelectionWaveformWidget, TimingIssuesDialog,
)

from workers import (
    BaseWorker, DownloadModelWorker, LoadModelWorker, TTSWorker,
    GenerateWorker, WhisperDownloadWorker, TranscribeWorker,
    AudioProcessWorker, LektorExportThread, DubbingVocalExtractWorker,
    VocalSuppressWorker, VideoAudioExtractWorker, DiarizationWorker,
    WhisperBackend, AudioPreprocessor,
)

from audio_utils import (
    load_audio_to_player, toggle_play, start_play, pause_play, stop_play,
    playback_tick, seek_audio, on_play_end, build_audio_undo_state,
    restore_audio_state, delete_selected_audio_segment, mute_selected_audio_segment,
    apply_trim_to_fragment, apply_trim_to_selected, update_trim_preview,
    on_trim_slider_changed,
)

from session_manager import (
    _write_session_to as write_session_to,
    _write_ebook_session_to as write_ebook_session_to,
    _restore_session_data as restore_session_data,
    _restore_ebook_session_data as restore_ebook_session_data,
    _build_session_data as build_session_data,
    _build_ebook_session_data as build_ebook_session_data,
    _auto_session_path as auto_session_path,
    _auto_ebook_session_path as auto_ebook_session_path,
)

from tabs.srt_tab import SrtTab
from tabs.ebook_tab import EbookTab
from tabs.quick_tts_tab import QuickTTSTab

load_active_backend_modules()
SUPERTONIC_VOICES = getattr(get_loaded_module("supertonic_backend"), "SUPERTONIC_VOICES", [])
_piper_voice_options = getattr(get_loaded_module("piper_backend"), "_voice_options", lambda *a, **k: [])

from srt_format import _ms_to_srt_ts as _ms_to_ts
from txt_format import txt_srt_format, txt_ebook_format

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("TORCH_HOME", str(ROOT_DIR / "models" / "torch_hub"))
OUTPUTS_DIR.mkdir(exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

_ACTIVE_BACKEND_CLASSES = detect_active_backends()


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audiobook and Dubbing Generator by Mubumbutu")
        self.resize(1440, 960)
        self.setMinimumSize(1280, 820)

        _icon_path = ROOT_DIR / "icon.ico"
        if _icon_path.exists():
            self.setWindowIcon(QIcon(str(_icon_path)))

        self._backend = None
        self._whisper_backend = None
        self._preprocessor = None
        self._ffmpeg_ok = _check_ffmpeg()

        self._output_dir: str = str(OUTPUTS_DIR)
        self._is_running: bool = False
        self._model_is_loaded: bool = False
        self._completed_count: int = 0
        self._worker: Optional[TTSWorker] = None
        self._lektor_export_thread: Optional[LektorExportThread] = None
        self._synthesis_source: str = "srt"

        # Audio playback state
        self._audio_data: Optional[np.ndarray] = None
        self._audio_sr = 44100
        self._audio_tmp: Optional[str] = None
        self._playing = False
        self._cursor = 0
        self._stream = None
        self._play_end_sample: int = 0
        self._current_fragment_idx: Optional[int] = None
        self._video_source_path: Optional[str] = None
        self._audio_undo_stack: list = []
        self._audio_redo_stack: list = []

        # Dubbing state
        self._dubbing_mode: bool = False
        self._dubbing_video_path: Optional[str] = None
        self._hf_token: Optional[str] = None
        self._speaker_list: List[str] = []
        self._speaker_voices: Dict = {}

        # Voice cloning widgets (shared)
        self._drop = None
        self._ref_player = None
        self._ref_text = None
        self._mono_check = None
        self._chk_demucs = None
        self._chk_bit_depth = None
        self._bit_depth_combo = None
        self._sr_combo = None
        self._proc_btn = None
        self._proc_lbl = None
        self._w_size = None
        self._w_lang = None
        self._w_dl_btn = None
        self._w_tr_btn = None
        self._w_status = None
        self._whisper_section_widget = None
        self._voice_single_container = None
        self._speakers_container = None
        self._speakers_lay = None
        self._voice_cloning_section = None
        self._omnivoice_mode_widget = None
        self._instruct_widget = None
        self._supertonic_single_hint = None
        self._cloning_widget = None
        self._voice_mode_combo = None
        self._instruct_text = None

        # Tworzony wcześniej niż UI — wymagany przez zakładki
        self._norm_check = QCheckBox("Normalize audio")

        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(lambda: playback_tick(self))

        self._file_watcher = QFileSystemWatcher(self)
        self._file_watcher.fileChanged.connect(self._on_audio_file_changed)

        self._build_ui()
        self.setStyleSheet(STYLE)
        self._init_backends()

    # ---------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------

    def _init_backends(self):
        try:
            model_id = (
                self._model_combo.currentData()
                if hasattr(self, "_model_combo")
                else MODEL_OPTIONS[0][0]
            )
            self._backend = create_backend(model_id)
            self._whisper_backend = WhisperBackend(WHISPER_DIR)
            self._preprocessor = AudioPreprocessor(PROC_DIR)
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

    # ---------------------------------------------------------------------
    # UI Construction
    # ---------------------------------------------------------------------

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

        # Po utworzeniu prawego panelu – wszystkie współdzielone widżety już istnieją
        self.srt_tab.set_shared_widgets(
            self._btn_start,
            self._btn_stop,
            self._synth_progress,
            self._eta_label,
            self._progress,
            self._norm_check,
        )

        self.ebook_tab.set_shared_widgets(
            self._btn_start,
            self._btn_stop,
            self._synth_progress,
            self._eta_label,
            self._progress,
            self._norm_check,
        )

        self.quick_tab.set_shared_widgets(
            self._btn_start,
            self._btn_stop,
            self._synth_progress,
            self._eta_label,
            self._progress,
            self._norm_check,
        )

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
            _hicon = "🎙"
            _htitle = "TTS Studio"
        logo = QLabel(_hicon)
        logo.setStyleSheet("font-size:22px;background:transparent;border:none;")
        title = QLabel(_htitle)
        title.setStyleSheet(f"font-size:18px;font-weight:700;color:{C['text']};background:transparent;border:none;")
        sub = QLabel("SRT Lektor Studio")
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
        w = QWidget()
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

        self.srt_tab = SrtTab(self)
        self.ebook_tab = EbookTab(self)
        self.quick_tab = QuickTTSTab(self)

        self._tabs.addTab(self.srt_tab.get_widget(), "📄  SRT Fragments")
        self._tabs.addTab(self.ebook_tab.get_widget(), "📚  Ebook Fragments")
        self._tabs.addTab(self.quick_tab.get_widget(), "⚡  Quick TTS")

        self._tabs.currentChanged.connect(self._on_tab_changed)
        lay.addWidget(self._tabs, 1)
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

        # Placeholder for Lektor section — managed by SrtTab
        self._lektor_section = CollapsibleSection("🎬 Lektor Export", C["accent"])
        self._lektor_section.add_widget(self.srt_tab.build_lektor_controls())
        lay.addWidget(self._lektor_section)

        self._video_preview_widget = self.srt_tab.build_video_preview_section()
        lay.addWidget(self._video_preview_widget)

        self._audiobook_section = CollapsibleSection("🎧 Audiobook Export", C["accent"])
        self._audiobook_section.add_widget(self.ebook_tab._build_audiobook_section())
        lay.addWidget(self._audiobook_section)
        self._audiobook_section.setVisible(False)

        lay.addStretch()
        scroll.setWidget(content)
        outer_lay.addWidget(scroll, 1)
        outer_lay.addWidget(self._make_synth_bar())
        self._on_tab_changed(self._tabs.currentIndex())
        return outer

    def _on_tab_changed(self, index: int):
        is_ebook = index == 1
        is_quick = index == 2
        self._btn_start.setVisible(not is_quick)
        self._btn_stop.setVisible(not is_quick)
        self._lektor_section.setVisible(not is_quick and not is_ebook)
        self.srt_tab.set_lektor_tab_active(not is_quick and not is_ebook)
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

    # ---------------------------------------------------------------------
    # Parameters Grid
    # ---------------------------------------------------------------------

    def _make_params_grid(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._param_widgets: Dict[str, QWidget] = {}
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

    # ---------------------------------------------------------------------
    # Player Bar
    # ---------------------------------------------------------------------

    def _make_player_bar(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f"background:{C['player']};border-top:1px solid {C['border']};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(16, 6, 16, 8)
        lay.setSpacing(4)

        self._video_player = VideoAudioPlayer()
        lay.addWidget(self._video_player)

        self._wave_out = SelectionWaveformWidget(h=64)
        self._wave_out.delete_requested.connect(
            lambda s, e: delete_selected_audio_segment(self, s, e)
        )
        self._wave_out.mute_requested.connect(
            lambda s, e: mute_selected_audio_segment(self, s, e)
        )
        self._wave_out.seeked.connect(lambda frac: seek_audio(self, frac))
        lay.addWidget(self._wave_out)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(6)

        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setFixedHeight(32)
        self._play_btn.setMinimumWidth(80)
        self._play_btn.setEnabled(False)
        self._play_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._play_btn.clicked.connect(lambda: toggle_play(self))

        self._stop_btn = QPushButton("⏹  Stop")
        self._stop_btn.setFixedHeight(32)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._stop_btn.clicked.connect(lambda: stop_play(self))

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
        self._trim_apply_all_btn.clicked.connect(
            lambda: apply_trim_to_selected(self)
        )

        def slider_changed(val: int):
            v = val / 100.0
            self._trim_input.blockSignals(True)
            self._trim_input.setValue(v)
            self._trim_input.blockSignals(False)
            on_trim_slider_changed(self, val)

        def input_changed(val: float):
            v = int(val * 100)
            self._trim_slider.blockSignals(True)
            self._trim_slider.setValue(v)
            self._trim_slider.blockSignals(False)
            on_trim_slider_changed(self, v)

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

    # ---------------------------------------------------------------------
    # Synthesis Control (delegated to tabs)
    # ---------------------------------------------------------------------

    def _start_synthesis(self):
        current_tab = self._tabs.currentIndex()
        if current_tab == 0:
            self.srt_tab.start_synthesis()
        elif current_tab == 1:
            self.ebook_tab.start_synthesis()
        else:
            self.quick_tab.synthesize()

    def _stop_synthesis(self):
        current_tab = self._tabs.currentIndex()
        if current_tab == 0:
            self.srt_tab.stop_synthesis()
        elif current_tab == 1:
            self.ebook_tab.stop_synthesis()
        else:
            # Quick TTS nie ma stop – ale możemy spróbować zatrzymać worker
            if hasattr(self.quick_tab, '_w_gen') and self.quick_tab._w_gen:
                self.quick_tab._w_gen.terminate()
        self._set_status("Stop requested — waiting for current fragment to finish…", C["warning"])

    # ---------------------------------------------------------------------
    # File Watcher
    # ---------------------------------------------------------------------

    def _on_audio_file_changed(self, path: str):
        if not os.path.exists(path):
            return
        self._file_watcher.addPath(path)
        dur = _get_wav_duration(path)
        if dur is None:
            return
        # Check SRT fragments
        for frag in self.srt_tab._fragments:
            if frag.get('output_path') == path and frag.get('status') == 'done':
                self.srt_tab.update_tree_item_duration(frag['index'], dur)
                return
        # Check ebook fragments
        for frag in self.ebook_tab._ebook_fragments:
            if frag.get('output_path') == path and frag.get('status') == 'done':
                self.ebook_tab.update_tree_item_duration(frag['index'], dur)
                return

    # ---------------------------------------------------------------------
    # Audio Playback (shared)
    # ---------------------------------------------------------------------

    def _on_wave_undo(self):
        if self._audio_undo_stack:
            state = self._audio_undo_stack.pop()
            self._audio_redo_stack.append({
                'audio': self._audio_data.copy() if self._audio_data is not None else None,
                'sr': self._audio_sr,
                'cursor': self._cursor,
            })
            restore_audio_state(self, state)

    def _on_wave_redo(self):
        if self._audio_redo_stack:
            state = self._audio_redo_stack.pop()
            self._audio_undo_stack.append({
                'audio': self._audio_data.copy() if self._audio_data is not None else None,
                'sr': self._audio_sr,
                'cursor': self._cursor,
            })
            restore_audio_state(self, state)

    def _load_audio_file(self, path: str, idx: int) -> bool:
        try:
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            load_audio_to_player(self, audio, sr, fragment_idx=idx)
            return True
        except Exception as e:
            self._set_status(f"Cannot load audio for fragment {idx}: {e}", C["error"])
            return False

    def _load_audio_to_player(self, audio: np.ndarray, sr: int, fragment_idx: Optional[int] = None):
        load_audio_to_player(self, audio, sr, fragment_idx=fragment_idx)

    def _start_play(self):
        start_play(self)

    def _pause_play(self):
        pause_play(self)

    def _stop_play(self):
        stop_play(self)

    def _on_play_end(self):
        on_play_end(self)

    def _update_trim_preview(self):
        update_trim_preview(self)

    # ---------------------------------------------------------------------
    # Model Management
    # ---------------------------------------------------------------------

    def _model_downloaded(self) -> bool:
        if not self._backend:
            return False
        return self._backend.is_available()

    def _refresh_fish_ui(self):
        dl = self._model_downloaded()
        loaded = self._model_is_loaded and self._backend is not None
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
        dl = self._whisper_backend.is_downloaded(size)
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

    def _update_voice_section_for_model(self, model_id: str):
        if not hasattr(self, "_omnivoice_mode_widget"):
            return
        is_ov = model_id.startswith("omnivoice_")
        is_supertonic = model_id.startswith("supertonic_")
        is_piper = model_id == "piper"
        is_no_clone = is_supertonic or is_piper

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

    def _update_device_label(self):
        if TORCH_AVAILABLE and torch.cuda.is_available():
            n = torch.cuda.get_device_name(0)
            m = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
            self._device_lbl.setText(f"  🟢 CUDA — {n} ({m}GB)")
            self._device_lbl.setStyleSheet(f"color:{C['success']};")
        else:
            self._device_lbl.setText("  🟡 CPU (slower)")
            self._device_lbl.setStyleSheet(f"color:{C['warning']};")

    def _update_action_buttons(self):
        model_ok = self._model_is_loaded and self._backend is not None
        tab = self._tabs.currentIndex() if hasattr(self, '_tabs') else 0
        if tab == 1:
            file_ok = self.ebook_tab.has_fragments()
            is_running = getattr(self.ebook_tab, '_is_running', False)
        else:
            file_ok = self.srt_tab.has_fragments()
            is_running = getattr(self.srt_tab, '_is_running', False)
        self._btn_start.setEnabled(model_ok and file_ok and not is_running)
        self._btn_stop.setEnabled(is_running)
        self.quick_tab._quick_btn.setEnabled(model_ok)

    def _is_moss_active(self) -> bool:
        if not hasattr(self, "_model_combo"):
            return False
        model_id = self._model_combo.currentData()
        return model_id is not None and model_id.startswith("moss_tts")

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

    def _is_pyannote_incompatible(self) -> bool:
        if self._backend is None:
            return False
        try:
            return self._backend.pyannote_incompatible
        except Exception:
            return False

    def _on_model_changed(self):
        if not hasattr(self, "_model_combo"):
            return
        self._model_is_loaded = False
        try:
            model_id = self._model_combo.currentData()
            self._backend = create_backend(model_id)
            self._refresh_fish_ui()
            self._update_params_visibility(model_id)
            self._update_whisper_visibility()
            self._update_voice_section_for_model(model_id)
            self._update_ref_text_visibility_for_model(model_id)
            self.srt_tab.update_dubbing_visibility()
            self._set_status(
                f"{self._backend.display_name} selected — click 'Load model'."
            )
        except Exception as e:
            self._set_status(f"Backend error: {e}", C["error"])

    def _on_voice_mode_changed(self):
        if not hasattr(self, "_voice_mode_combo"):
            return
        mode = self._voice_mode_combo.currentData()
        show_cloning = (mode == "cloning")
        show_instruct = (mode == "design")
        self._cloning_widget.setVisible(show_cloning)
        self._instruct_widget.setVisible(show_instruct)
        if hasattr(self, "_whisper_section_widget"):
            self._whisper_section_widget.setVisible(show_cloning)
        hints = {
            "cloning": "Clone a voice from a reference WAV file. Recommended: 3–15 s of clean speech.",
            "design": "Describe the desired voice with attributes (gender, age, pitch, accent…). No reference audio needed.",
            "auto": "Let the model choose a voice automatically. No configuration required.",
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

    def _get_speaker_voices_dict(self) -> Optional[Dict]:
        if not self._dubbing_mode or not self._speaker_list:
            return None
        result = {}
        is_supertonic = self._is_supertonic_active()
        is_piper = self._is_piper_active()
        for spk in self._speaker_list:
            sv = self._speaker_voices.get(spk, {})
            if is_supertonic:
                voice_combo = sv.get("voice_combo")
                voice_name = voice_combo.currentData() if voice_combo else "M1"
                result[spk] = (None, voice_name)
            elif is_piper:
                voice_combo = sv.get("voice_combo")
                voice_model = voice_combo.currentData() if voice_combo else ""
                result[spk] = (None, voice_model)
            else:
                drop = sv.get("drop")
                ref_text_wdg = sv.get("ref_text")
                audio_path = drop.file_path if drop else None
                text = ref_text_wdg.toPlainText().strip() if ref_text_wdg else None
                result[spk] = (audio_path, text or None)
        if is_supertonic or is_piper:
            return result if result else None
        has_any_audio = any(v[0] for v in result.values())
        return result if has_any_audio else None

    # ---------------------------------------------------------------------
    # Model Download / Load / Unload
    # ---------------------------------------------------------------------

    def _start_model_download(self):
        model_id = self._model_combo.currentData()
        backend = self._backend
        if backend is None:
            return

        hf_token = None
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
        self._model_is_loaded = True
        self._refresh_fish_ui()

    def _unload_fish_model(self):
        if not self._backend or not self._model_is_loaded:
            return

        model_name = self._backend.name
        if QMessageBox.question(
            self,
            "Unload model",
            f"Unload {model_name} from GPU memory?\n\nYou will need to reload it before generating.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return

        self._unload_btn.setEnabled(False)
        self._unload_btn.setText("⏳  Unloading…")
        stop_play(self)

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
                self._set_status(
                    "Model unloaded — reload before generating.",
                    C["warning"],
                )
                self._model_is_loaded = False
                self._refresh_fish_ui()

        QTimer.singleShot(100, poll)

    # ---------------------------------------------------------------------
    # Reference Audio
    # ---------------------------------------------------------------------

    @staticmethod
    def _audio_info_str(path: str) -> str:
        try:
            info = sf.info(path)
            channels = info.channels
            ch_str = "Mono" if channels == 1 else "Stereo" if channels == 2 else f"{channels}ch"
            sr = info.samplerate
            sr_str = f"{sr/1000:.1f} kHz" if sr >= 1000 else f"{sr} Hz"
            sub = info.subtype.upper()
            if "16" in sub:
                bits = "16-bit"
            elif "24" in sub:
                bits = "24-bit"
            elif "32" in sub or "FLOAT" in sub or "DOUBLE" in sub:
                bits = "32-bit"
            else:
                bits = sub
            dur = info.duration
            dur_str = _fmt(dur) if dur > 0 else ""
            return f"{ch_str} • {bits} • {sr_str} • {dur_str}"
        except Exception:
            return ""

    def _on_ref_dropped(self, path: str):
        self._ref_player.load(path)
        self._proc_btn.setEnabled(True)
        self._set_status(f"Reference audio: {Path(path).name}", C["accent"])
        self._ref_audio_info_lbl.setText(self._audio_info_str(path))
        self._refresh_whisper_ui()

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
            "target_sr": self._sr_combo.currentData(),
            "to_mono": self._mono_check.isChecked(),
            "isolate_vocals": self._chk_demucs.isChecked(),
            "normalize": True,
            "output_subtype": output_subtype,
            "device": "cuda" if (self._backend and self._backend.device == "cuda") else "cpu",
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

    # ---------------------------------------------------------------------
    # Whisper
    # ---------------------------------------------------------------------

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

    # ---------------------------------------------------------------------
    # Dubbing
    # ---------------------------------------------------------------------

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

    def _rebuild_voice_cloning_for_speakers(self):
        while self._speakers_lay.count():
            item = self._speakers_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._speaker_voices.clear()
        is_supertonic = self._is_supertonic_active()
        is_piper = self._is_piper_active()

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

                mono_check = QCheckBox("Convert to mono")
                mono_check.setChecked(self._mono_check.isChecked())
                sr_combo = QComboBox()
                for val, label in TARGET_SR_OPTIONS:
                    sr_combo.addItem(label, val)
                for i in range(sr_combo.count()):
                    if sr_combo.itemData(i) == self._sr_combo.currentData():
                        sr_combo.setCurrentIndex(i)
                        break

                chk_bit_depth = QCheckBox("Override bit depth")
                chk_bit_depth.setChecked(self._chk_bit_depth.isChecked())
                bit_depth_combo = QComboBox()
                for code, label in [("PCM_16", "16-bit"), ("PCM_24", "24-bit"), ("FLOAT", "32-bit float")]:
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
                    "drop": drop,
                    "player": player,
                    "ref_text": ref_text,
                    "proc_btn": proc_btn,
                    "tr_btn": tr_btn,
                    "demucs_chk": demucs_chk,
                    "audio_info_lbl": audio_info_lbl,
                    "w_size": w_size,
                    "w_lang": w_lang,
                    "mono_check": mono_check,
                    "chk_bit_depth": chk_bit_depth,
                    "bit_depth_combo": bit_depth_combo,
                    "sr_combo": sr_combo,
                }

            self._speakers_lay.addWidget(spk_group)

        self._voice_single_container.setVisible(False)
        self._speakers_container.setVisible(True)

        if hasattr(self, "_whisper_section_widget"):
            self._whisper_section_widget.setVisible(False)

        if hasattr(self, "_voice_cloning_section"):
            self._voice_cloning_section.expand()

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
        sr_cbo = sv.get("sr_combo", self._sr_combo)

        settings = {
            "target_sr": sr_cbo.currentData(),
            "to_mono": mono_chk.isChecked(),
            "isolate_vocals": sv["demucs_chk"].isChecked(),
            "normalize": True,
            "output_subtype": output_subtype,
            "device": "cuda" if (
                self._backend and self._backend.device == "cuda"
            ) else "cpu",
            "output_name": output_name,
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

    # ---------------------------------------------------------------------
    # Status & Helpers
    # ---------------------------------------------------------------------

    def _set_status(self, msg: str, color=None):
        self._status_lbl.setText(msg)
        self._status_lbl.setStyleSheet(f"color:{color or C['text2']};")

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

    # ---------------------------------------------------------------------
    # Close Event
    # ---------------------------------------------------------------------

    def closeEvent(self, e):
        workers = [
            getattr(self.srt_tab, "_worker", None),
            getattr(self.ebook_tab, "_worker", None),
            getattr(self.quick_tab, "_w_gen", None),
            getattr(self.srt_tab, "_lektor_export_thread", None),
        ]
        for worker in workers:
            if worker is None:
                continue
            try:
                if not worker.isRunning():
                    continue
                if hasattr(worker, "request_cancel"):
                    worker.request_cancel()
                worker.wait(10000)
                if worker.isRunning():
                    worker.terminate()
                    worker.wait(3000)
            except RuntimeError:
                pass
        stop_play(self)
        if self._audio_tmp and Path(self._audio_tmp).exists():
            try:
                os.unlink(self._audio_tmp)
            except Exception:
                pass
        e.accept()