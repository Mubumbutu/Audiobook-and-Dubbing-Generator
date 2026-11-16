# tabs/srt_tab.py
"""
SRT tab implementation – subtitle loading, fragment management,
synthesis, dubbing, and Lektor video export.
"""
import copy
import os
import re
import json
import time
import gc
import shutil
import subprocess
import tempfile
import threading
import hashlib
import logging
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable, Set

import numpy as np
import soundfile as sf

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTreeWidget, QTreeWidgetItem, QPlainTextEdit, QLineEdit,
    QGroupBox, QFileDialog, QProgressBar, QHeaderView,
    QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QSlider,
    QSplitter, QMenu, QDialog, QDialogButtonBox, QInputDialog,
    QMessageBox, QFrame, QSizePolicy, QScrollArea, QRadioButton,
    QApplication, QTabWidget, QAbstractItemView,
)
from PyQt6.QtCore import Qt, QTimer, QUrl, QFileSystemWatcher
from PyQt6.QtGui import (
    QFont, QColor, QIcon, QAction, QDesktopServices, QPen, QBrush,
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget

from .base_tab import BaseTab

# --- Imports from project modules ---
from config import (
    C, _btn,
    STATUS_WAITING, STATUS_RUNNING, STATUS_DONE, STATUS_ERROR,
    COL_STATUS, COL_FRAGMENT, COL_SPEAKER, COL_TIMING,
    OUTPUTS_DIR,
    WHISPER_SIZES, WHISPER_SIZE_MB, WHISPER_LANGS, TARGET_SR_OPTIONS,
    SUPERTONIC_VOICES, _piper_voice_options,
    MODEL_OPTIONS, _ACTIVE_BACKEND_CLASSES,
)
from widgets import (
    FragmentTreeWidget, DropAudioWidget, RefAudioPlayer, CollapsibleSection,
    ClickableVideoWidget, VideoFullscreenWindow,
)
from utils import (
    _get_last_dir, _set_last_dir,
    _detect_srt_language, _convert_numbers_in_text,
    _ms_to_ts, _fmt, _fmt_ms,
    _get_wav_duration, _open_file,
    _ref_audio_hash, _audio_info_str,
)
from workers import (
    TTSWorker,
    DubbingVocalExtractWorker, DiarizationWorker,
    TranscribeWorker, AudioProcessWorker,
    VideoAudioExtractWorker, LektorExportThread, VocalSuppressWorker,
)
from input_formats import get_format, supported_extensions
import ass_format
from txt_format import txt_srt_format
from tts_backends import InferenceError
from session_manager import _write_session_to as write_session_to, _restore_session_data as restore_session_data, choose_session_file
from undo_manager import snapshot_fragments, restore_snapshot_files, discard_snapshot, move_to_trash

logger = logging.getLogger(__name__)


class SrtTab(BaseTab):
    """SRT subtitle processing tab."""

    def __init__(self, main_window):
        super().__init__(main_window)

        self._vid_path_edit = None
        self._audio_path_edit = None
        self._offset_spin = None
        self._lektor_vol = None
        self._lektor_vol_lbl = None
        self._orig_vol = None
        self._orig_vol_lbl = None
        self._autofit_check = None
        self._atempo_threshold = None
        self._duck_check = None
        self._vocal_suppress_check = None
        self._vocal_suppress_spin = None
        self._keep_original_track_check = None
        self._dubbed_lang_edit = None
        self._lektor_vid_fmt_combo = None
        self._export_btn = None
        self._lektor_status = None
        self._lektor_norm_check = None
        self._vid_info_lbl = None
        self._audio_info_lbl = None
        self._vid_row_widget = None
        self._audio_row_widget = None

        self._current_sim_frag_idx = None
        self._last_sim_pos_ms = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Build the SRT tab UI."""
        self._fragments: List[Dict] = []
        self._frag_items: Dict[int, QTreeWidgetItem] = {}
        self._chapter_item: Optional[QTreeWidgetItem] = None
        self._srt_path: Optional[str] = None
        self._output_dir: str = ""
        self._is_running: bool = False
        self._completed_count: int = 0
        self._synth_start_time: float = 0.0
        self._synth_total: int = 0
        self._worker = None
        self._lektor_export_thread = None
        self._synthesis_source: str = "srt"
        self._dubbing_mode: bool = False
        self._dubbing_video_path: Optional[str] = None
        self._hf_token: Optional[str] = None
        self._speaker_list: List[str] = []
        self._speaker_voices: Dict = {}
        self._pending_export: Dict = {}
        self._w_vocal_extract = None
        self._w_diarization = None
        self._w_vocal_suppress = None
        self._w_proc = None
        self._w_tr = None
        self._sel_fail_btn = None
        self._sel_pending_btn = None
        self._filter_edit = None
        self._srt_label = None
        self._tree = None
        self._preview_text = None
        self._btn_close_srt = None
        self._btn_save_session = None
        self._btn_dubbing = None
        self._btn_show_video_wave = None
        self._btn_load_srt = None
        self._btn_start = None
        self._btn_stop = None
        self._synth_progress = None
        self._eta_label = None
        self._progress = None
        self._norm_check = None
        self._file_watcher = self.main._file_watcher

        # Główny odtwarzacz wideo
        self._media_player = QMediaPlayer()
        self._audio_output = QAudioOutput()
        self._media_player.setAudioOutput(self._audio_output)
        self._audio_output.setVolume(0.0)

        # Odtwarzacz pomocniczy do symulacji lektora
        self._dub_player = QMediaPlayer()
        self._dub_audio_output = QAudioOutput()
        self._dub_player.setAudioOutput(self._dub_audio_output)
        self._dub_audio_output.setVolume(1.0)
        self._current_sim_frag_idx = None
        self._media_player.positionChanged.connect(self._on_video_position_changed)

        self._video_widget = None
        self._video_section = None
        self._video_preview_widget = None
        self._video_loaded = False
        self._lektor_tab_active = True
        self._vid_info_lbl = None
        self._audio_info_lbl = None
        self._vid_row_widget = None
        self._audio_row_widget = None

        # Build the actual UI
        self._srt_tab_widget = self._make_srt_tab()
        self._media_player.mediaStatusChanged.connect(self._on_media_status_changed)

    def set_shared_widgets(self, btn_start, btn_stop, synth_progress, eta_label, progress, norm_check):
        """Set widgets shared with MainWindow."""
        self._btn_start = btn_start
        self._btn_stop = btn_stop
        self._synth_progress = synth_progress
        self._eta_label = eta_label
        self._progress = progress
        self._norm_check = norm_check
        self._tree.itemChanged.connect(self._on_item_check_changed)

    def get_widget(self) -> QWidget:
        """Return the built tab widget."""
        return self._srt_tab_widget

    # ------------------------------------------------------------------
    # SRT Tab UI
    # ------------------------------------------------------------------

    def _make_srt_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        top = QHBoxLayout()
        self._btn_load_srt = QPushButton("📂  Load File")
        self._btn_load_srt.clicked.connect(self._load_srt_file)
        self._btn_close_srt = QPushButton("✕  Close")
        self._btn_close_srt.setEnabled(False)
        self._btn_close_srt.clicked.connect(self._close_srt_file)

        self._btn_show_video_wave = QPushButton("🎬  Show video")
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

        sel_all = QPushButton("☑ All")
        sel_all.setFixedHeight(26)
        sel_none = QPushButton("☐ None")
        sel_none.setFixedHeight(26)
        sel_fail = QPushButton("❌ Failed")
        sel_fail.setFixedHeight(26)
        sel_fail.setCheckable(True)
        sel_pending = QPushButton("⬜ Pending")
        sel_pending.setFixedHeight(26)
        sel_pending.setCheckable(True)

        self._sel_fail_btn = sel_fail
        self._sel_pending_btn = sel_pending

        sel_overlong = QPushButton("⏳ Overlong")
        sel_overlong.setFixedHeight(26)
        sel_overlong.setStyleSheet("font-size:11px;padding:2px 8px;")
        sel_overlong.setToolTip(
            "Select fragments where generated audio is longer than the time slot\n"
            "(yellow / orange / red timing in column Timing)"
        )
        sel_overlong.clicked.connect(self._show_timing_issues_dialog)

        btn_time_corrector = QPushButton("⏱ Time corrector")
        btn_time_corrector.setFixedHeight(26)
        btn_time_corrector.setStyleSheet("font-size:11px;padding:2px 8px;")
        btn_time_corrector.setToolTip(
            "Review all overlapping fragments and choose for each one:\n"
            "  - merge the fragments,\n"
            "  - keep them separate.\n\n"
            "Then choose the subtitle timing:\n"
            "  - overlap allowed (good for dubbing) - gaps are closed, overlapping fragments are not changed,\n"
            "  - no overlap (good for lektor) - the end of every fragment is set to the start of the next one.\n"
            "The last correction can be undone from the Time corrector window."
        )
        btn_time_corrector.clicked.connect(self._time_corrector)
        self._time_corrector_undo: Optional[Dict] = None

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
        filter_row.addWidget(btn_time_corrector)
        lay.addLayout(filter_row)

        self._tree = FragmentTreeWidget()
        self._tree.setColumnCount(4)
        self._tree.setHeaderLabels(["", "Text", "Speaker", "Timing"])
        self._tree.header().setStretchLastSection(False)
        self._tree.setColumnWidth(COL_STATUS, 28)
        self._tree.setColumnWidth(COL_FRAGMENT, 420)
        self._tree.setColumnWidth(COL_SPEAKER, 100)
        self._tree.setColumnWidth(COL_TIMING, 220)
        self._tree.setWordWrap(True)
        self._tree.setAlternatingRowColors(True)
        self._tree.setIndentation(0)
        self._tree.setMinimumHeight(160)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._show_context_menu)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        self._tree.itemDoubleClicked.connect(self._on_tree_item_double_clicked)
        self._tree.itemClicked.connect(self._on_tree_item_clicked)
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

        self._tree.header().setSectionResizeMode(COL_STATUS, QHeaderView.ResizeMode.Fixed)
        self._tree.header().setSectionResizeMode(COL_FRAGMENT, QHeaderView.ResizeMode.Interactive)
        self._tree.header().setSectionResizeMode(COL_SPEAKER, QHeaderView.ResizeMode.Interactive)
        self._tree.header().setSectionResizeMode(COL_TIMING, QHeaderView.ResizeMode.Stretch)

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

    # ------------------------------------------------------------------
    # SRT Loading / Closing
    # ------------------------------------------------------------------

    def _load_srt_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self.main, "Open SRT / ASS / TXT file", _get_last_dir("srt"),
            "Subtitle files (*.srt *.ass *.ssa *.txt);;SRT Subtitles (*.srt);;ASS Subtitles (*.ass *.ssa);;TXT Subtitles (*.txt);;All files (*)",
        )
        if not path:
            return
        _set_last_dir("srt", path)

        ext = Path(path).suffix.lower()

        try:
            if ext == '.txt':
                segments = txt_srt_format.load(path)
            else:
                segments = get_format(ext if ext in supported_extensions() else '.srt').load(path)
            if not segments:
                QMessageBox.warning(self.main, "Empty file", "No valid subtitle blocks found.")
                return
            self._fragments = []
            for seg in segments:
                self._fragments.append({
                    'index': seg.index,
                    'srt_id': str(seg.index + 1),
                    'timestamp': f"{_ms_to_ts(seg.start_ms)} --> {_ms_to_ts(seg.end_ms)}",
                    'text': seg.text,
                    'start_ms': seg.start_ms,
                    'end_ms': seg.end_ms,
                    'speaker': "",
                    'status': 'waiting',
                    'output_path': None,
                    'error_msg': None,
                })
        except Exception as e:
            QMessageBox.critical(self.main, "Load error", str(e))
            return

        self.main.undo_manager.clear()
        self._reset_dubbing_state()

        self.main._set_status("Detecting language…")
        QApplication.processEvents()

        raw_texts = [f["text"] for f in self._fragments if f.get("text")]
        detected_lang = _detect_srt_language(raw_texts)
        logger.info(f"Detected language: {detected_lang}")

        try:
            self.main._set_status(f"Converting numbers to words [{detected_lang}]…")
            QApplication.processEvents()
            for frag in self._fragments:
                original = frag.get("text", "")
                converted = _convert_numbers_in_text(original, detected_lang)
                if converted != original:
                    frag["text"] = converted
                    logger.debug(f"Fragment {frag['index']}: {original!r} → {converted!r}")
        except Exception as e:
            logger.warning(f"Number conversion failed: {e}")

        self._redistribute_split_timings(self._fragments)

        self._srt_path = path
        self._output_dir = str(self._session_dirs(path)[1])

        chosen = choose_session_file(
            self.main, [self._auto_session_path(), self._user_session_path()]
        )
        if chosen:
            try:
                with open(str(chosen), "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self._restore_session_data(data)
                logger.info(f"Session restored from: {chosen}")
                return
            except Exception as e:
                logger.warning(f"Session restore failed: {e}")
                self.main._set_status("Could not restore previous session — starting fresh.", C["warning"])

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
        self.main._tabs.setCurrentIndex(0)

        self._populate_tree()
        self.main._update_action_buttons()
        self.main._set_status(
            f"Loaded: {fname} — {len(self._fragments)} fragments | language: {detected_lang}"
        )
        logger.info(f"Loaded: {path} → {len(self._fragments)} fragments, lang={detected_lang}")

    def _close_srt_file(self):
        if self._is_running:
            QMessageBox.warning(self.main, "Busy", "Cannot close file during active synthesis.")
            return
        reply = QMessageBox.question(
            self.main, "Close SRT file",
            "Close current SRT file?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.main.undo_manager.clear()
        self._fragments.clear()
        self._frag_items.clear()
        self._chapter_item = None
        self._srt_path = None
        self._tree.clear()
        self._preview_text.clear()
        self._srt_label.setText("No SRT file loaded")
        self._srt_label.setStyleSheet(
            f"color:{C['text3']};font-size:11px;font-style:italic;"
        )
        self._btn_close_srt.setEnabled(False)
        self._btn_save_session.setEnabled(False)
        self._reset_dubbing_state()
        self.main._update_action_buttons()
        self.main._set_status("SRT file closed.")

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
                group = fragments[i:j]
                base_start = fragments[i].get('start_ms', 0)
                base_end = fragments[i].get('end_ms', 0)
                total_ms = base_end - base_start

                next_frag = fragments[j] if j < len(fragments) else None
                next_start_ms = next_frag.get('start_ms', base_end) if next_frag else base_end
                gap_ms = next_start_ms - base_end

                dynamic_limit = int(total_ms * 0.25)
                borrow_limit_ms = min(500, max(0, dynamic_limit))
                bonus_ms = min(max(0, gap_ms), borrow_limit_ms)

                lengths = [max(1, len(f.get('text') or '')) for f in group]
                total_len = sum(lengths)

                bonus_parts = [int(bonus_ms * l / total_len) for l in lengths]
                bonus_parts[-1] += bonus_ms - sum(bonus_parts)

                if total_ms <= 0:
                    slot_ms = max(100, bonus_ms // group_size)
                    cursor = base_start
                    for k, frag in enumerate(group):
                        frag_start = cursor
                        frag_end = frag_start + slot_ms
                        if k == group_size - 1 and next_frag is not None:
                            frag_end = min(frag_end, next_start_ms)
                        frag['start_ms'] = frag_start
                        frag['end_ms'] = frag_end
                        frag['timestamp'] = f"{_ms_to_ts(frag_start)} --> {_ms_to_ts(frag_end)}"
                        cursor = frag_end
                else:
                    cursor = base_start
                    for k, frag in enumerate(group):
                        frag_start = cursor
                        base_frag_ms = int(total_ms * lengths[k] / total_len)
                        frag_ms = max(100, base_frag_ms + bonus_parts[k])
                        frag_end = frag_start + frag_ms
                        if k == group_size - 1 and next_frag is not None:
                            frag_end = min(frag_end, next_start_ms)
                        frag['start_ms'] = frag_start
                        frag['end_ms'] = frag_end
                        frag['timestamp'] = f"{_ms_to_ts(frag_start)} --> {_ms_to_ts(frag_end)}"
                        cursor = frag_end

            i = j

        for frag in fragments:
            frag['srt_end_ms'] = frag['end_ms']

    # ------------------------------------------------------------------
    # Tree Population
    # ------------------------------------------------------------------

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

        if self._filter_edit is not None:
            self._apply_filter(self._filter_edit.text())

    def _create_tree_item(self, frag: Dict, parent=None) -> QTreeWidgetItem:
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
        item.setCheckState(
            COL_STATUS,
            Qt.CheckState.Checked if frag.get('_checked', True) else Qt.CheckState.Unchecked,
        )
        item.setText(COL_STATUS, "")

        base_text = frag.get('text', '')
        prefix = (frag.get('prefix') or '').strip()
        suffix = (frag.get('suffix') or '').strip()
        parts = [x for x in [prefix, base_text, suffix] if x]
        display = " ".join(parts)
        item.setText(COL_FRAGMENT, f"{icon}  #{num}  {display}")
        item.setText(COL_SPEAKER, frag.get('speaker') or "")

        start_ms = frag.get('start_ms', 0) or 0
        end_ms = frag.get('end_ms', 0) or 0
        srt_end = frag.get('srt_end_ms', end_ms)
        timing_text = f"{_ms_to_ts(start_ms)} → {_ms_to_ts(end_ms)}"

        if status == 'done':
            dur = _get_wav_duration(frag.get('output_path', ''))
            if dur is not None:
                slot_s = (srt_end - start_ms) / 1000.0
                diff = dur - slot_s
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
            'done': STATUS_DONE,
            'error': STATUS_ERROR,
        }.get(status, STATUS_WAITING)

        num = frag['index'] + 1
        base_text = frag.get('text', '')
        prefix = (frag.get('prefix') or '').strip()
        suffix = (frag.get('suffix') or '').strip()
        parts = [x for x in [prefix, base_text, suffix] if x]
        display = " ".join(parts)
        item.setText(COL_FRAGMENT, f"{icon}  #{num}  {display}")
        item.setText(COL_SPEAKER, frag.get('speaker') or "")
        item.setForeground(COL_FRAGMENT, QColor(C["text"]))

        start_ms = frag.get('start_ms', 0) or 0
        end_ms = frag.get('end_ms', 0) or 0
        srt_end = frag.get('srt_end_ms', end_ms)
        timing_text = f"{_ms_to_ts(start_ms)} → {_ms_to_ts(end_ms)}"

        if status == 'done':
            dur = known_dur_s if known_dur_s is not None else _get_wav_duration(frag.get('output_path', ''))
            if dur is not None:
                slot_s = (srt_end - start_ms) / 1000.0
                diff = dur - slot_s
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

    # ------------------------------------------------------------------
    # Fragment Selection / Filtering
    # ------------------------------------------------------------------

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
            idx = child.data(COL_STATUS, Qt.ItemDataRole.UserRole)
            frag = next((f for f in self._fragments if f['index'] == idx), None)
            if not frag:
                continue
            if retry_errors:
                if frag.get('status') not in ('waiting', 'error'):
                    continue
            result.append(frag)
        return result

    def _on_item_check_changed(self, item, column):
        if column != COL_STATUS:
            return
        idx = item.data(COL_STATUS, Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        if 0 <= idx < len(self._fragments) and self._fragments[idx].get('index') == idx:
            frag = self._fragments[idx]
        else:
            frag = next((f for f in self._fragments if f['index'] == idx), None)
        if frag is not None:
            frag['_checked'] = item.checkState(COL_STATUS) == Qt.CheckState.Checked

    def _sync_checks_from_tree(self):
        if not self._chapter_item:
            return
        frag_map = {f['index']: f for f in self._fragments}
        for i in range(self._chapter_item.childCount()):
            child = self._chapter_item.child(i)
            idx = child.data(COL_STATUS, Qt.ItemDataRole.UserRole)
            frag = frag_map.get(idx)
            if frag is not None:
                frag['_checked'] = child.checkState(COL_STATUS) == Qt.CheckState.Checked

    def _apply_filter(self, text: str):
        if not self._chapter_item:
            return
        txt = " ".join(text.lower().split())
        frag_map = {f['index']: f for f in self._fragments}
        self._tree.setUpdatesEnabled(False)
        try:
            for i in range(self._chapter_item.childCount()):
                child = self._chapter_item.child(i)
                if not txt:
                    child.setHidden(False)
                    continue
                idx = child.data(COL_STATUS, Qt.ItemDataRole.UserRole)
                frag = frag_map.get(idx)
                if frag is None:
                    child.setHidden(True)
                    continue
                haystack = " ".join(
                    p for p in (
                        (frag.get('prefix') or '').strip(),
                        (frag.get('text') or '').strip(),
                        (frag.get('suffix') or '').strip(),
                        (frag.get('speaker') or '').strip(),
                    ) if p
                ).lower()
                haystack = " ".join(haystack.split())
                child.setHidden(txt not in haystack)
        finally:
            self._tree.setUpdatesEnabled(True)

    def _select_all(self, state: bool):
        if not self._chapter_item:
            return
        self._tree.setUpdatesEnabled(False)
        self._tree.blockSignals(True)
        try:
            for i in range(self._chapter_item.childCount()):
                child = self._chapter_item.child(i)
                checked = state and not child.isHidden()
                child.setCheckState(
                    COL_STATUS,
                    Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked,
                )
        finally:
            self._tree.blockSignals(False)
            self._tree.setUpdatesEnabled(True)
        self._sync_checks_from_tree()

    def _select_failed(self):
        if not self._chapter_item:
            return
        frag_map = {f['index']: f for f in self._fragments}
        self._tree.setUpdatesEnabled(False)
        self._tree.blockSignals(True)
        try:
            for i in range(self._chapter_item.childCount()):
                child = self._chapter_item.child(i)
                idx = child.data(COL_STATUS, Qt.ItemDataRole.UserRole)
                frag = frag_map.get(idx)
                cs = Qt.CheckState.Checked if (frag and frag.get('status') == 'error') else Qt.CheckState.Unchecked
                child.setCheckState(COL_STATUS, cs)
        finally:
            self._tree.blockSignals(False)
            self._tree.setUpdatesEnabled(True)

    def _select_all_reset(self, state: bool):
        self._sel_fail_btn.blockSignals(True)
        self._sel_pending_btn.blockSignals(True)
        self._sel_fail_btn.setChecked(False)
        self._sel_pending_btn.setChecked(False)
        self._sel_fail_btn.blockSignals(False)
        self._sel_pending_btn.blockSignals(False)
        self._select_all(state)

    def _apply_status_filter(self):
        if not self._chapter_item:
            return
        want_failed = self._sel_fail_btn.isChecked()
        want_pending = self._sel_pending_btn.isChecked()
        frag_map = {f['index']: f for f in self._fragments}
        self._tree.setUpdatesEnabled(False)
        self._tree.blockSignals(True)
        try:
            for i in range(self._chapter_item.childCount()):
                child = self._chapter_item.child(i)
                match = False
                if not child.isHidden():
                    idx = child.data(COL_STATUS, Qt.ItemDataRole.UserRole)
                    frag = frag_map.get(idx)
                    if frag is not None:
                        status = frag.get('status', 'waiting')
                        if want_failed and want_pending:
                            match = status in ('error', 'waiting')
                        elif want_failed:
                            match = status == 'error'
                        elif want_pending:
                            match = status == 'waiting'
                child.setCheckState(
                    COL_STATUS,
                    Qt.CheckState.Checked if match else Qt.CheckState.Unchecked,
                )
        finally:
            self._tree.blockSignals(False)
            self._tree.setUpdatesEnabled(True)
        self._sync_checks_from_tree()

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

        start_s = frag.get('start_ms', 0) / 1000.0
        end_s = frag.get('end_ms', 0) / 1000.0
        self._click_pos_ms = int(start_s * 1000)

        was_playing = False
        if hasattr(self.main, '_video_player'):
            was_playing = self.main._video_player.is_playing()

        self._suppress_scroll = not was_playing
        try:
            if hasattr(self.main, '_video_player') and self.main._video_player.isVisible():
                overflow_end_s = None
                if frag.get('status') == 'done':
                    dur = _get_wav_duration(frag.get('output_path', ''))
                    if dur is not None:
                        slot_s = end_s - start_s
                        if dur > slot_s:
                            overflow_end_s = start_s + dur
                self.main._video_player.set_selection_by_time(start_s, end_s, overflow_end_s)

            if getattr(self, '_media_player', None) is not None and self._media_player.source().isValid():
                self._media_player.setPosition(int(start_s * 1000))
        finally:
            self._suppress_scroll = False

        self._preview_text.setPlainText(frag.get('text', ''))
        self._last_preview_idx = idx

    def _on_tree_item_clicked(self, item, column):
        idx = item.data(COL_STATUS, Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        self._on_selection_changed()

    def _on_tree_item_double_clicked(self, item, column):
        idx = item.data(COL_STATUS, Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        if self._load_fragment_audio(idx):
            self.main._start_play()

    def _load_fragment_audio(self, idx: int) -> bool:
        frag = next((f for f in self._fragments if f.get('index') == idx), None)
        if not frag or frag.get('status') != 'done':
            return False

        path = frag.get('output_path')
        if not path or not os.path.exists(path):
            return False

        try:
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            self.main._load_audio_to_player(audio, sr, fragment_idx=idx)
            return True
        except Exception as e:
            self.main._set_status(f"Cannot load audio for fragment {idx}: {e}", C["error"])
            return False

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def _get_retry_settings(self) -> Tuple[bool, int, float]:
        check = getattr(self.main, "_retry_check", None)
        spin = getattr(self.main, "_retry_spin", None)
        tol_spin = getattr(self.main, "_retry_tol_spin", None)
        if check is None or spin is None or not check.isChecked():
            return False, 0, 0.0
        tol_s = float(tol_spin.value()) if tol_spin is not None else 0.0
        return True, int(spin.value()), tol_s

    def _release_audio_file_locks(self):
        self._dub_hold_until = max(
            getattr(self, '_dub_hold_until', 0.0), time.monotonic() + 2.0
        )
        players = []
        dub = getattr(self, '_dub_player', None)
        if dub is not None:
            players.append(dub)
        for pair in (getattr(self, '_extra_dub_players', None) or {}).values():
            players.append(pair[0])
        for player in players:
            try:
                player.stop()
                player.setSource(QUrl())
            except Exception as e:
                logger.warning(f"Could not release dub player: {e}")
        self._current_sim_frag_idx = None
        self._last_sim_pos_ms = None
        self._dub_schedule = []
        self._dub_schedule_sig = None
        extra_state = getattr(self, '_extra_sim_idx', None)
        if extra_state:
            extra_state.clear()
        for attempt in range(30):
            QApplication.processEvents()
            time.sleep(0.03)
            if attempt >= 4 and all(
                p.mediaStatus() == QMediaPlayer.MediaStatus.NoMedia for p in players
            ):
                break

    def _dub_sync_held(self) -> bool:
        return time.monotonic() < getattr(self, '_dub_hold_until', 0.0)

    def _start_synthesis(self, retry_errors: bool = False):
        if not self.main._model_is_loaded:
            QMessageBox.warning(self.main, "Model not loaded", "Load the model first.")
            return

        to_process = self._get_checked_fragments(retry_errors=retry_errors)
        if not to_process:
            QMessageBox.information(self.main, "Nothing to synthesize", "No fragments selected.")
            return

        already_done = [f for f in to_process if f.get('status') == 'done']
        if already_done:
            reply = QMessageBox.question(
                self.main,
                "Re-synthesize completed fragments?",
                f"{len(already_done)} fragment(s) already have generated audio.\n\n"
                "Do you want to re-synthesize them anyway?\n"
                "Existing audio files will be overwritten.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.No:
                to_process = [f for f in to_process if f.get('status') != 'done']
                if not to_process:
                    QMessageBox.information(self.main, "Nothing to synthesize",
                                            "No pending fragments left to process.")
                    return

        for frag in to_process:
            frag["status"] = "waiting"
            frag["error_msg"] = None
            self._update_tree_item(frag["index"])

        self._release_audio_file_locks()

        self._synthesis_source = 'srt'
        self._is_running = True
        self._completed_count = 0
        self._synth_start_time = time.monotonic()
        self._synth_total = len(to_process)
        self.main._update_action_buttons()

        total = self._synth_total
        self._synth_progress.setMaximum(total)
        self._synth_progress.setValue(0)
        self._synth_progress.setVisible(True)
        self._eta_label.setText(f"0/{total}")
        self._eta_label.setVisible(True)

        reserved_paths = {f['output_path'] for f in self._fragments if f.get('output_path')}
        retry_enabled, retry_max, retry_tol = self._get_retry_settings()

        self._worker = TTSWorker(
            backend=self.main._backend,
            fragments=to_process,
            output_dir=self._output_dir,
            reference_audio=self.main._get_ref_audio(),
            reference_text=self.main._get_ref_text(),
            filename_prefix=Path(self._srt_path).stem if self._srt_path else "fragment",
            generation_settings=self.main._get_generation_settings(),
            normalize_audio=self._norm_check.isChecked(),
            speaker_voices=self._get_speaker_voices_dict(),
            reserved_paths=reserved_paths,
            retry_until_fit=retry_enabled,
            max_retries=retry_max,
            retry_tolerance_s=retry_tol,
        )

        self._worker.progress.connect(self._on_progress)
        self._worker.item_done.connect(self._on_item_done)
        self._worker.finished.connect(self._on_synthesis_finished)
        self._worker.start()

        self.main._set_status(f"Synthesis started — {total} fragments queued…")
        logger.info(f"Synthesis started: {total} fragments")

    def _stop_synthesis(self):
        if self._worker:
            self._worker.request_cancel()
        self.main._set_status("Stop requested — waiting for current fragment to finish…", C["warning"])

    def _on_progress(self, idx: int, msg: str, is_error: bool):
        if self._synthesis_source == 'ebook':
            return
        if idx >= 0:
            frag = next((f for f in self._fragments if f['index'] == idx), None)
            if frag:
                frag['status'] = 'running' if not is_error else 'error'
                self._update_tree_item(idx)
        color = C["error"] if is_error else C["text2"]
        self.main._set_status(msg, color)

    def _on_item_done(self, idx: int, result: str, is_error: bool):
        if self._synthesis_source == 'ebook':
            return

        frag = next((f for f in self._fragments if f['index'] == idx), None)
        if not frag:
            return
        if is_error:
            frag['status'] = 'error'
            frag['error_msg'] = result
        else:
            frag['status'] = 'done'
            frag['output_path'] = result
            frag['error_msg'] = None
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
                        self.main._load_audio_to_player(audio, sr, fragment_idx=idx)
                    except Exception as e:
                        self.main._set_status(f"Cannot load audio: {e}", C["error"])

        self._completed_count += 1
        self._synth_progress.setValue(self._completed_count)

        total = getattr(self, '_synth_total', 0)
        remaining = total - self._completed_count
        if remaining > 0 and getattr(self, '_synth_start_time', None) is not None:
            elapsed = time.monotonic() - self._synth_start_time
            avg = elapsed / self._completed_count
            eta_s = avg * remaining
            self._eta_label.setText(f"{self._completed_count}/{total}  ~{_fmt(eta_s)} left")
        else:
            self._eta_label.setText(f"{self._completed_count}/{total}")

    def _on_synthesis_finished(self):
        self._is_running = False
        self._synth_progress.setVisible(False)
        self._eta_label.setVisible(False)

        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()

        done = sum(1 for f in self._fragments if f.get("status") == "done")
        error = sum(1 for f in self._fragments if f.get("status") == "error")

        self.main._set_status(
            f"Synthesis complete — {done} done, {error} errors.",
            C["success"] if error == 0 else C["warning"],
        )
        self.main._update_action_buttons()
        logger.info(f"Synthesis finished: {done} done, {error} errors")

        auto = self._auto_session_path()
        if auto:
            if self._write_session_to(str(auto)):
                logger.info(f"Auto-saved session: {auto}")

    def _re_synthesize_fragment(self, frag: Dict):
        if not self.main._model_is_loaded:
            QMessageBox.warning(self.main, "Model not loaded", "Load the model first.")
            return

        self._synthesis_source = 'srt'

        frag["status"] = "waiting"
        frag["error_msg"] = None
        self._update_tree_item(frag["index"])

        self._release_audio_file_locks()

        self._is_running = True
        self._completed_count = 0
        self._synth_start_time = time.monotonic()
        self._synth_total = 1
        self.main._update_action_buttons()
        self._synth_progress.setMaximum(1)
        self._synth_progress.setValue(0)
        self._synth_progress.setVisible(True)
        self._eta_label.setText("0/1")
        self._eta_label.setVisible(True)

        reserved_paths = {
            f['output_path'] for f in self._fragments if f.get('output_path')
        }
        retry_enabled, retry_max, retry_tol = self._get_retry_settings()

        self._worker = TTSWorker(
            backend=self.main._backend,
            fragments=[frag],
            output_dir=self._output_dir,
            reference_audio=self.main._get_ref_audio(),
            reference_text=self.main._get_ref_text(),
            filename_prefix=Path(self._srt_path).stem if self._srt_path else "fragment",
            generation_settings=self.main._get_generation_settings(),
            normalize_audio=self._norm_check.isChecked(),
            speaker_voices=self._get_speaker_voices_dict(),
            reserved_paths=reserved_paths,
            retry_until_fit=retry_enabled,
            max_retries=retry_max,
            retry_tolerance_s=retry_tol,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.item_done.connect(self._on_item_done)
        self._worker.finished.connect(self._on_synthesis_finished)
        self._worker.start()

    # ------------------------------------------------------------------
    # Context Menu
    # ------------------------------------------------------------------

    def _show_context_menu(self, pos):
        item = self._tree.itemAt(pos)
        if not item:
            return
        idx = item.data(COL_STATUS, Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        frag = next((f for f in self._fragments if f["index"] == idx), None)
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
            frag.get("output_path") and os.path.exists(frag.get("output_path", ""))
        )
        model_ok = self.main._model_is_loaded

        act_open = menu.addAction("📁  Open output folder")
        act_open.setEnabled(has_audio)
        act_open.triggered.connect(
            lambda: _open_file(os.path.dirname(frag["output_path"]))
        )

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

        speaker = frag.get("speaker") or ""
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
            (i for i, f in enumerate(self._fragments) if f["index"] == idx), -1
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

    def _undo_begin(self, with_files: bool = False) -> Optional[Dict]:
        try:
            return snapshot_fragments(self._fragments, with_files)
        except Exception as e:
            logger.warning(f"Undo snapshot failed: {e}")
            self.main.undo_manager.clear()
            return None

    def _undo_commit(self, label: str, before: Optional[Dict]):
        if before is not None:
            self.main.undo_manager.push('srt', label, before)

    def _undo_capture(self, with_files: bool = False) -> Dict:
        return snapshot_fragments(self._fragments, with_files)

    def _undo_restore(self, state: Dict) -> int:
        missing = restore_snapshot_files(state, self._fragments, self._file_watcher)
        missing_set = set(missing)
        fragments = copy.deepcopy(state['fragments'])
        for f in fragments:
            if f.get('output_path') in missing_set:
                f['status'] = 'waiting'
                f['output_path'] = None
        self._fragments = fragments
        if state.get('files') is not None:
            self._tc_reset_current_fragment()
        self._populate_tree()
        self._sync_speaker_ui_from_fragments()
        self.main._update_action_buttons()
        return len(missing)

    def _edit_fragment_text(self, frag: Dict):
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Edit fragment #{frag['index'] + 1}")
        dlg.resize(560, 260)
        dlg.setStyleSheet(self.main.styleSheet())

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

        before = self._undo_begin()
        frag['text'] = new_text
        frag['status'] = 'waiting'
        frag['error_msg'] = None
        self._update_tree_item(frag['index'])
        self._preview_text.setPlainText(new_text)
        self._undo_commit(f"text edit of fragment #{frag['index'] + 1}", before)
        self.main._set_status(f"Fragment #{frag['index'] + 1} text updated — status reset to waiting.")

    def _edit_fragment_time(self, frag: Dict):
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Edit time — fragment #{frag['index'] + 1}")
        dlg.resize(420, 210)
        dlg.setStyleSheet(self.main.styleSheet())

        lay = QVBoxLayout(dlg)
        lay.setSpacing(10)

        info = QLabel(f"Text: {frag.get('text', '')[:60]}")
        info.setStyleSheet(f"color:{C['text3']};font-size:11px;")
        lay.addWidget(info)

        def ms_to_parts(ms: int):
            ms = max(0, int(ms))
            m = ms // 60000
            s = (ms % 60000) // 1000
            rem = ms % 1000
            return m, s, rem

        def make_time_row(label_text: str, ms_val: int):
            row = QHBoxLayout()
            lbl = QLabel(label_text)
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
        end_row, sp_em, sp_es, sp_ems = make_time_row("End:", frag.get('srt_end_ms') or frag.get('end_ms', 0) or 0)
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
        new_end_ms = sp_em.value() * 60000 + sp_es.value() * 1000 + sp_ems.value()

        if new_start_ms >= new_end_ms:
            QMessageBox.warning(self.main, "Invalid time", "Start time must be less than end time.")
            return

        before = self._undo_begin()
        frag['start_ms'] = new_start_ms
        frag['end_ms'] = new_end_ms
        frag['srt_end_ms'] = new_end_ms
        frag['timestamp'] = f"{_ms_to_ts(new_start_ms)} --> {_ms_to_ts(new_end_ms)}"
        self._update_tree_item(frag['index'])
        self._undo_commit(f"time edit of fragment #{frag['index'] + 1}", before)
        self.main._set_status(
            f"Fragment #{frag['index'] + 1} time updated: "
            f"{_ms_to_ts(new_start_ms)} → {_ms_to_ts(new_end_ms)}", C["accent"]
        )

    def _add_fragment_after(self, frag: Dict):
        dlg = QDialog(self)
        dlg.setWindowTitle("Add fragment")
        dlg.resize(520, 220)
        dlg.setStyleSheet(self.main.styleSheet())

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

        before = self._undo_begin(True)

        end_ms = frag.get('end_ms', 0) or 0
        new_frag = {
            'index': 0,
            'srt_id': '',
            'timestamp': f"{_ms_to_ts(end_ms)} --> {_ms_to_ts(end_ms + 3000)}",
            'text': new_text,
            'start_ms': end_ms,
            'end_ms': end_ms + 3000,
            'srt_end_ms': end_ms + 3000,
            'speaker': frag.get('speaker'),
            'status': 'waiting',
            'output_path': None,
            'error_msg': None,
        }
        self._fragments.insert(pos + 1, new_frag)

        for i, f in enumerate(self._fragments):
            f['index'] = i
            f['srt_id'] = str(i + 1)

        self._rename_outputs_to_match_order()
        self._populate_tree()
        self.main._update_action_buttons()
        self._undo_commit("adding a fragment", before)

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
        is_checked = frag['index'] in checked_indices

        dlg = QDialog(self)
        dlg.setWindowTitle("Remove fragment")
        dlg.setStyleSheet(self.main.styleSheet())
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
        btn_this = QPushButton("Remove this one")
        btn_all = QPushButton(f"Remove all checked  ({checked_count})")
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

        before = self._undo_begin(True)

        if result["action"] == "all":
            removed_fragments = [f for f in self._fragments if f['index'] in checked_indices]
            self._fragments = [f for f in self._fragments if f['index'] not in checked_indices]
        else:
            removed_fragments = [f for f in self._fragments if f['index'] == frag['index']]
            self._fragments = [f for f in self._fragments if f['index'] != frag['index']]

        self._release_audio_file_locks()
        try:
            self.main._stop_play()
        except Exception:
            pass
        QApplication.processEvents()

        for f in removed_fragments:
            p = f.get('output_path', '')
            if not p or not os.path.exists(p):
                continue
            try:
                self._file_watcher.removePath(p)
            except Exception:
                pass
            try:
                dst = move_to_trash(p)
                logger.info(f"Moved removed audio to trash: {dst}")
            except Exception as e:
                logger.error(f"Cannot move {p} to trash: {e}")
                self.main._set_status(
                    f"Could not move file to trash: {Path(p).name} "
                    f"(close any player or Explorer window and retry)",
                    C["warning"],
                )

        for i, f in enumerate(self._fragments):
            f['index'] = i
            f['srt_id'] = str(i + 1)

        self._rename_outputs_to_match_order()
        self._populate_tree()
        self.main._update_action_buttons()
        self._undo_commit(f"removal of {len(removed_fragments)} fragment(s)", before)
        self.main._set_status(f"Fragment(s) removed. {len(self._fragments)} fragments remaining.")

    def _move_fragment(self, frag: Dict, direction: int):
        pos = next((i for i, f in enumerate(self._fragments) if f['index'] == frag['index']), -1)
        if pos < 0:
            return
        new_pos = pos + direction
        if new_pos < 0 or new_pos >= len(self._fragments):
            return

        label = f"moving fragment #{frag['index'] + 1} {'down' if direction > 0 else 'up'}"
        before = self._undo_begin(True)

        self._fragments[pos], self._fragments[new_pos] = self._fragments[new_pos], self._fragments[pos]

        for i, f in enumerate(self._fragments):
            f['index'] = i
            f['srt_id'] = str(i + 1)

        self._rename_outputs_to_match_order()
        self._populate_tree()
        self._undo_commit(label, before)

        moved_item = self._frag_items.get(new_pos)
        if moved_item:
            self._tree.setCurrentItem(moved_item)

    def _rename_outputs_to_match_order(self):
        if not self._srt_path or not self._output_dir:
            return

        self._release_audio_file_locks()
        self._dub_hold_until = float('inf')
        try:
            prefix = Path(self._srt_path).stem

            def _norm(p: str) -> str:
                return os.path.normcase(os.path.abspath(p))

            to_rename: List[Dict] = []
            seen_paths: Set[str] = set()
            for frag in self._fragments:
                p = frag.get('output_path')
                if not p or not os.path.exists(p):
                    continue
                key = _norm(p)
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                to_rename.append(frag)
            if not to_rename:
                return

            desired = {
                frag['index']: os.path.join(
                    self._output_dir, f"{prefix}_{frag['index'] + 1:03d}.wav"
                )
                for frag in to_rename
            }

            if all(
                _norm(frag['output_path']) == _norm(desired[frag['index']])
                for frag in to_rename
            ):
                return

            for frag in to_rename:
                try:
                    self._file_watcher.removePath(frag['output_path'])
                except Exception:
                    pass

            def _safe_rename(src: str, dst: str, retries: int = 15, delay: float = 0.1) -> Tuple[bool, Optional[Exception]]:
                last_exc: Optional[Exception] = None
                for attempt in range(retries):
                    if os.path.exists(dst):
                        return False, FileExistsError(dst)
                    try:
                        os.rename(src, dst)
                        return True, None
                    except FileNotFoundError as e:
                        return False, e
                    except OSError as e:
                        last_exc = e
                        if attempt < retries - 1:
                            QApplication.processEvents()
                            time.sleep(delay)
                return False, last_exc

            owned: Set[str] = {
                _norm(f['output_path']) for f in self._fragments if f.get('output_path')
            }

            def _temp_path(src: str) -> str:
                if '.__reorder__' in os.path.basename(src):
                    return src
                root, ext = os.path.splitext(src)
                candidate = f"{root}.__reorder__{ext}"
                n = 1
                while os.path.exists(candidate) or _norm(candidate) in owned:
                    candidate = f"{root}.__reorder__{n}{ext}"
                    n += 1
                return candidate

            def _owner_of(path: str, exclude_index: int) -> Optional[Dict]:
                key = _norm(path)
                for other in self._fragments:
                    other_path = other.get('output_path')
                    if other['index'] == exclude_index or not other_path:
                        continue
                    if _norm(other_path) == key:
                        return other
                return None

            for frag in to_rename:
                src = frag['output_path']
                tmp = _temp_path(src)
                if tmp == src:
                    continue
                ok, err = _safe_rename(src, tmp)
                if ok:
                    frag['output_path'] = tmp
                    owned.add(_norm(tmp))
                else:
                    logger.warning(f"Reorder phase 1 failed for {src}: {err}")

            backup_dir = os.path.join(self._output_dir, "__undo_backup")
            ts = int(time.time() * 1000)
            failed_count = 0
            pending = list(to_rename)

            while pending:
                progress = False
                next_pending: List[Dict] = []
                for frag in pending:
                    idx = frag['index']
                    cur = frag['output_path']
                    dst = desired[idx]

                    if _norm(cur) == _norm(dst):
                        frag['output_path'] = dst
                        progress = True
                        continue

                    if os.path.exists(dst):
                        if _owner_of(dst, idx) is not None:
                            next_pending.append(frag)
                            continue
                        cleared = False
                        try:
                            os.remove(dst)
                            cleared = True
                            logger.info(f"Removed orphan reorder target: {dst}")
                        except Exception as rm_err:
                            logger.warning(f"Cannot remove orphan target {dst}: {rm_err}")
                        if not cleared:
                            try:
                                os.makedirs(backup_dir, exist_ok=True)
                                bdst = os.path.join(backup_dir, f"{ts}_{os.path.basename(dst)}")
                                shutil.move(dst, bdst)
                                cleared = True
                                logger.info(f"Moved orphan target to backup: {bdst}")
                            except Exception as mv_err:
                                logger.error(f"Cannot move orphan target {dst}: {mv_err}")
                        if not cleared:
                            failed_count += 1
                            continue

                    ok, err = _safe_rename(cur, dst)
                    if ok:
                        frag['output_path'] = dst
                        progress = True
                    else:
                        failed_count += 1
                        logger.warning(
                            f"Reorder phase 2 failed {cur} -> {dst}: {err}; "
                            f"file kept at {cur}"
                        )

                if not next_pending:
                    break
                if not progress:
                    failed_count += len(next_pending)
                    for frag in next_pending:
                        logger.warning(
                            f"Reorder blocked for fragment {frag['index']}: "
                            f"target occupied, file kept at {frag['output_path']}"
                        )
                    break
                pending = next_pending

            for frag in self._fragments:
                p = frag.get('output_path')
                if p and os.path.exists(p):
                    try:
                        self._file_watcher.addPath(p)
                    except Exception:
                        pass

            if failed_count:
                logger.warning(f"Reorder completed with {failed_count} file(s) not renamed")
                self.main._set_status(
                    f"{failed_count} audio file(s) could not be renamed and were kept "
                    f"under temporary names; they will be fixed on the next reorder.",
                    C["warning"],
                )
        finally:
            self._dub_hold_until = 0.0
            self._sanitize_fragment_outputs()

    def _sanitize_fragment_outputs(self) -> int:
        prefix = Path(self._srt_path).stem if self._srt_path else ""

        def _norm(p: str) -> str:
            return os.path.normcase(os.path.abspath(p))

        def _expected(frag: Dict) -> str:
            return f"{prefix}_{frag['index'] + 1:03d}.wav"

        ordered = sorted(
            self._fragments,
            key=lambda f: 0 if os.path.basename(f.get('output_path') or "") == _expected(f) else 1,
        )

        claimed: Set[str] = set()
        reset_count = 0
        for frag in ordered:
            if frag.get('status') != 'done':
                continue
            path = frag.get('output_path')
            valid = False
            if path:
                try:
                    valid = os.path.isfile(path) and os.path.getsize(path) > 0
                except OSError:
                    valid = False
            if valid:
                key = _norm(path)
                if key in claimed:
                    valid = False
                else:
                    claimed.add(key)
            if not valid:
                logger.warning(
                    f"Fragment {frag['index'] + 1} marked done but audio is missing, "
                    f"empty or shared: {path}; reset to waiting"
                )
                frag['status'] = 'waiting'
                frag['output_path'] = None
                frag['error_msg'] = None
                reset_count += 1

        if reset_count:
            self.main._set_status(
                f"{reset_count} fragment(s) had missing or invalid audio and were reset to waiting.",
                C["warning"],
            )
        return reset_count

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

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

        can_merge_next = pos + 1 < len(self._fragments)
        is_checked = frag['index'] in checked_indices
        checked_count = len(checked_indices)
        can_merge_checked = is_checked and checked_count >= 2

        dlg = QDialog(self)
        dlg.setWindowTitle("Merge fragments")
        dlg.resize(480, 220)
        dlg.setStyleSheet(self.main.styleSheet())
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
        btn_next = QPushButton("Merge with next fragment")
        btn_checked = QPushButton(f"Merge all checked  ({checked_count})")
        btn_cancel = QPushButton("Cancel")

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
                gap = (curr.get('start_ms', 0) or 0) - (prev.get('end_ms', 0) or 0)
                if gap < max_gap_ms:
                    current_group.append(curr)
                else:
                    if len(current_group) >= 2:
                        groups.append(current_group)
                    current_group = [curr]
            if len(current_group) >= 2:
                groups.append(current_group)

            if not groups:
                self.main._set_status("No fragment pairs found within the specified gap threshold.")
                return
            self._do_merge_gap_groups(groups)

    def _do_merge_srt_fragments(
        self,
        fragments: List[Dict],
        text_override: Optional[str] = None,
        end_ms_override: Optional[int] = None,
        drop_audio: bool = False,
        refresh: bool = True,
    ):
        if len(fragments) < 2:
            return

        self._release_audio_file_locks()
        QApplication.processEvents()

        before = self._undo_begin(True) if refresh else None

        self.main._stop_play()

        if text_override is not None:
            combined_text = text_override
        else:
            combined_text = " ".join(
                f.get('text', '').strip() for f in fragments if f.get('text', '').strip()
            )
        start_ms = fragments[0].get('start_ms', 0) or 0
        if end_ms_override is not None:
            end_ms = end_ms_override
            srt_end = max(
                (f.get('srt_end_ms', f.get('end_ms', 0)) or 0) for f in fragments
            )
        else:
            end_ms = fragments[-1].get('end_ms', 0) or 0
            srt_end = fragments[-1].get('srt_end_ms', end_ms) or end_ms
        speaker = fragments[0].get('speaker', '')

        target_sr = 44100
        parts: List[np.ndarray] = []
        loaded_count = 0
        import torch
        import torchaudio.functional as TAF
        for df in ([] if drop_audio else fragments):
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
                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"Merge: cannot load {ap}: {e}")

        merged_audio_path: Optional[str] = None
        merged_status = 'waiting'

        if parts and loaded_count == len(fragments):
            try:
                os.makedirs(self._output_dir, exist_ok=True)
                stamp = time.time_ns()
                tmp_path = os.path.join(self._output_dir, f"__merge_tmp_{stamp}.wav")
                while os.path.exists(tmp_path):
                    stamp += 1
                    tmp_path = os.path.join(self._output_dir, f"__merge_tmp_{stamp}.wav")
                combined = np.concatenate(parts)
                sf.write(tmp_path, combined, target_sr, subtype='PCM_16')
                merged_audio_path = tmp_path
                merged_status = 'done'
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
            'index': 0,
            'srt_id': '',
            'timestamp': f"{_ms_to_ts(start_ms)} --> {_ms_to_ts(end_ms)}",
            'text': combined_text,
            'start_ms': start_ms,
            'end_ms': end_ms,
            'srt_end_ms': srt_end,
            'speaker': speaker,
            'status': merged_status,
            'output_path': merged_audio_path,
            'error_msg': None,
        }

        self._fragments = [f for f in self._fragments if f['index'] not in indices_to_remove]
        self._fragments.insert(insert_pos, merged_frag)

        for i, f in enumerate(self._fragments):
            f['index'] = i
            f['srt_id'] = str(i + 1)

        kept_paths = {
            os.path.normcase(os.path.abspath(f['output_path']))
            for f in self._fragments if f.get('output_path')
        }
        for p in paths_to_delete:
            if os.path.normcase(os.path.abspath(p)) in kept_paths:
                continue
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as e:
                logger.warning(f"Could not delete orphan audio {p}: {e}")

        if not refresh:
            return

        self._rename_outputs_to_match_order()

        if self.main._current_fragment_idx in indices_to_remove:
            self.main._current_fragment_idx = None
            self.main._audio_data = None
            if hasattr(self.main, '_wave_out'):
                self.main._wave_out.clear()
            self.main._play_btn.setEnabled(False)
            self.main._stop_btn.setEnabled(False)
            self.main._time_lbl.setText("0:00 / 0:00")

        self._populate_tree()
        self.main._update_action_buttons()
        self._undo_commit(f"merge of {len(fragments)} fragments", before)
        self.main._set_status(
            f"Merged {len(fragments)} fragments into one  "
            f"({_ms_to_ts(start_ms)} → {_ms_to_ts(end_ms)})."
            + ("" if merged_status == 'done' else "  Re-synthesis required."),
            C["accent"],
        )

    def _do_merge_gap_groups(self, groups: List[List[Dict]]):
        if not groups:
            return

        self._release_audio_file_locks()
        QApplication.processEvents()

        before = self._undo_begin(True)

        self.main._stop_play()

        target_sr = 44100
        all_group_indices: set = set()
        for g in groups:
            for f in g:
                all_group_indices.add(f['index'])

        prepared: List[Dict] = []
        import torch
        import torchaudio.functional as TAF
        for group in groups:
            combined_text = " ".join(
                f.get('text', '').strip() for f in group if f.get('text', '').strip()
            )
            start_ms = group[0].get('start_ms', 0) or 0
            end_ms = group[-1].get('end_ms', 0) or 0
            srt_end = group[-1].get('srt_end_ms', end_ms) or end_ms
            speaker = group[0].get('speaker', '')

            parts: List[np.ndarray] = []
            loaded_count = 0
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
                        loaded_count += 1
                    except Exception as e:
                        logger.warning(f"Gap merge: cannot load {ap}: {e}")

            merged_audio_path: Optional[str] = None
            merged_status = 'waiting'
            if parts and loaded_count == len(group):
                try:
                    os.makedirs(self._output_dir, exist_ok=True)
                    stamp = time.time_ns()
                    tmp_path = os.path.join(self._output_dir, f"__merge_tmp_{stamp}.wav")
                    while os.path.exists(tmp_path):
                        stamp += 1
                        tmp_path = os.path.join(self._output_dir, f"__merge_tmp_{stamp}.wav")
                    sf.write(tmp_path, np.concatenate(parts), target_sr, subtype='PCM_16')
                    merged_audio_path = tmp_path
                    merged_status = 'done'
                except Exception as e:
                    logger.warning(f"Gap merge audio write failed: {e}")

            group_indices = {f['index'] for f in group}
            leader_idx = min(group_indices)
            prepared.append({
                'group_indices': group_indices,
                'leader_idx': leader_idx,
                'frag': {
                    'index': 0,
                    'srt_id': '',
                    'timestamp': f"{_ms_to_ts(start_ms)} --> {_ms_to_ts(end_ms)}",
                    'text': combined_text,
                    'start_ms': start_ms,
                    'end_ms': end_ms,
                    'srt_end_ms': srt_end,
                    'speaker': speaker,
                    'status': merged_status,
                    'output_path': merged_audio_path,
                    'error_msg': None,
                },
            })

        leader_map: Dict[int, Dict] = {p['leader_idx']: p for p in prepared}

        kept_paths = {
            os.path.normcase(os.path.abspath(f['output_path']))
            for f in self._fragments
            if f['index'] not in all_group_indices and f.get('output_path')
        }

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
                if (
                    p
                    and os.path.exists(p)
                    and os.path.normcase(os.path.abspath(p)) not in kept_paths
                ):
                    try:
                        os.remove(p)
                    except Exception as e:
                        logger.warning(f"Could not delete orphan audio {p}: {e}")
            else:
                new_fragments.append(f)

        self._fragments = new_fragments
        for i, f in enumerate(self._fragments):
            f['index'] = i
            f['srt_id'] = str(i + 1)

        for p_item in prepared:
            ap = p_item['frag'].get('output_path', '')
            if ap and os.path.exists(ap):
                self._file_watcher.addPath(ap)

        if self.main._current_fragment_idx in all_group_indices:
            self.main._current_fragment_idx = None
            self.main._audio_data = None
            if hasattr(self.main, '_wave_out'):
                self.main._wave_out.clear()
            self.main._play_btn.setEnabled(False)
            self.main._stop_btn.setEnabled(False)
            self.main._time_lbl.setText("0:00 / 0:00")

        self._rename_outputs_to_match_order()
        self._populate_tree()
        self.main._update_action_buttons()
        self._undo_commit(f"merge by gap into {len(prepared)} group(s)", before)
        partial = sum(1 for p in prepared if p['frag'].get('status') != 'done')
        msg = f"Merged {len(all_group_indices)} fragments into {len(prepared)} groups by gap."
        if partial:
            msg += f"  {partial} group(s) require re-synthesis."
        self.main._set_status(msg, C["accent"])

    # ------------------------------------------------------------------
    # Timing Tools
    # ------------------------------------------------------------------

    @staticmethod
    def _tc_join_text(left: str, right: str) -> str:
        left = (left or "").strip()
        right = (right or "").strip()
        if not left:
            return right
        if not right:
            return left
        if not left[-1].isalnum():
            return f"{left} {right}"
        if right[0].isupper():
            return f"{left}. {right}"
        return f"{left} {right}"

    @staticmethod
    def _tc_speaker(frag: Dict) -> str:
        return str(frag.get('speaker') or "").strip()

    def _tc_find_overlaps(self) -> List[Dict]:
        result: List[Dict] = []
        for i in range(len(self._fragments) - 1):
            cur = self._fragments[i]
            nxt = self._fragments[i + 1]
            cur_end = cur.get('end_ms', 0) or 0
            nxt_start = nxt.get('start_ms', 0) or 0
            overlap = cur_end - nxt_start
            if overlap <= 0:
                continue
            result.append({
                'cur': cur,
                'nxt': nxt,
                'overlap_ms': overlap,
                'same_speaker': self._tc_speaker(cur) == self._tc_speaker(nxt),
            })
        return result

    def _tc_reset_current_fragment(self) -> None:
        self.main._current_fragment_idx = None
        self.main._audio_data = None
        if hasattr(self.main, '_wave_out'):
            self.main._wave_out.clear()
        self.main._play_btn.setEnabled(False)
        self.main._stop_btn.setEnabled(False)
        self.main._time_lbl.setText("0:00 / 0:00")

    def _tc_refresh_undo_button(self) -> None:
        btn = getattr(self, '_tc_undo_btn', None)
        if btn is not None:
            btn.setEnabled(getattr(self, '_time_corrector_undo', None) is not None)

    @staticmethod
    def _tc_char_count(frag: Dict) -> int:
        parts = [
            (frag.get('prefix') or '').strip(),
            (frag.get('text') or '').strip(),
            (frag.get('suffix') or '').strip(),
        ]
        return len(" ".join(p for p in parts if p))

    def _tc_speech_rate(self) -> float:
        total_chars = 0
        total_dur = 0.0
        for frag in self._fragments:
            path = frag.get('output_path')
            if frag.get('status') != 'done' or not path:
                continue
            chars = self._tc_char_count(frag)
            dur = _get_wav_duration(path)
            if chars <= 0 or not dur or dur <= 0:
                continue
            total_chars += chars
            total_dur += dur
        if total_dur <= 0:
            return 14.0
        return min(40.0, max(5.0, total_chars / total_dur))

    def _tc_fragment_duration(
        self,
        frag: Dict,
        rate: float,
        cache: Dict[int, Tuple[Optional[float], bool]],
    ) -> Tuple[Optional[float], bool]:
        key = id(frag)
        if key in cache:
            return cache[key]
        result: Tuple[Optional[float], bool] = (None, True)
        path = frag.get('output_path')
        if frag.get('status') == 'done' and path:
            dur = _get_wav_duration(path)
            if dur is not None:
                result = (dur, False)
        if result[0] is None:
            chars = self._tc_char_count(frag)
            if chars > 0 and rate > 0:
                result = (chars / rate, True)
        cache[key] = result
        return result

    def _tc_compute_fit(
        self,
        cur: Dict,
        nxt: Dict,
        action: str,
        allow_overlap: bool,
        cache: Dict[int, Tuple[Optional[float], bool]],
        rate: float,
    ) -> Tuple[Optional[int], str, str, Optional[float]]:
        def severity(dur_s: float, slot_s: float) -> int:
            if dur_s <= slot_s:
                return 0
            if dur_s < 2 * slot_s:
                return 1
            if dur_s < 4 * slot_s:
                return 2
            return 3

        cur_start = cur.get('start_ms', 0) or 0
        cur_end = cur.get('end_ms', 0) or 0
        nxt_start = nxt.get('start_ms', 0) or 0
        nxt_end = nxt.get('end_ms', 0) or 0

        cur_dur, cur_est = self._tc_fragment_duration(cur, rate, cache)
        nxt_dur, nxt_est = self._tc_fragment_duration(nxt, rate, cache)

        def dur_label(dur: float, est: bool) -> str:
            return f"{'~' if est else ''}{dur:.1f}s"

        estimate_note = f"\n~ = estimated from text length ({rate:.1f} chars/s)"

        if action == 'merge':
            if cur_dur is None or nxt_dur is None:
                return None, "-", "Merge: cannot calculate, a fragment has no text or audio.", None
            total_dur = cur_dur + nxt_dur
            slot_s = max(0.001, (max(cur_end, nxt_end) - cur_start) / 1000.0)
            sev = severity(total_dur, slot_s)
            diff = total_dur - slot_s
            estimated = cur_est or nxt_est
            mark = "~" if estimated else ""
            text = f"{mark}OK" if sev == 0 else f"{mark}+{diff:.1f}s"
            tip = (
                f"Merge fragments:\n"
                f"  Combined speech: {mark}{total_dur:.1f}s\n"
                f"  Available slot: {slot_s:.1f}s\n"
                f"  Result: {'fits' if sev == 0 else f'over by {diff:.1f}s'}"
            )
            if estimated:
                tip += estimate_note
            return sev, text, tip, diff

        if allow_overlap or nxt_start <= cur_start:
            cur_slot_s = max(0.001, (cur_end - cur_start) / 1000.0)
        else:
            cur_slot_s = max(0.001, (nxt_start - cur_start) / 1000.0)
        nxt_slot_s = max(0.001, (nxt_end - nxt_start) / 1000.0)

        entries = []
        for dur, est, slot in (
            (cur_dur, cur_est, cur_slot_s),
            (nxt_dur, nxt_est, nxt_slot_s),
        ):
            if dur is None:
                continue
            entries.append((severity(dur, slot), dur - slot, est))

        if not entries:
            return None, "-", "Keep separate: cannot calculate, fragments have no text or audio.", None

        sev, diff, _est = max(entries, key=lambda e: (e[0], e[1]))
        estimated = any(e[2] for e in entries)
        mark = "~" if estimated else ""
        text = f"{mark}OK" if sev == 0 else f"{mark}+{diff:.1f}s"

        mode_label = "overlap allowed" if allow_overlap else "no overlap"
        cur_str = (
            f"{dur_label(cur_dur, cur_est)} / slot {cur_slot_s:.1f}s"
            if cur_dur is not None else "no data"
        )
        nxt_str = (
            f"{dur_label(nxt_dur, nxt_est)} / slot {nxt_slot_s:.1f}s"
            if nxt_dur is not None else "no data"
        )
        tip = (
            f"Keep separate ({mode_label}):\n"
            f"  #{cur['index'] + 1}: {cur_str}\n"
            f"  #{nxt['index'] + 1}: {nxt_str}"
        )
        if estimated:
            tip += estimate_note
        return sev, text, tip, diff

    @staticmethod
    def _tc_text_suggests_merge(cur: Dict, nxt: Dict) -> bool:
        cur_text = (cur.get('text') or '').strip()
        nxt_text = (nxt.get('text') or '').strip()
        if not cur_text or not nxt_text:
            return False

        if nxt_text.startswith(('\u2026', '...')):
            return True

        cur_last = cur_text[-1]
        nxt_first = nxt_text[0]
        nxt_last = nxt_text[-1]

        if cur_text.endswith(('...', '\u2026', ',')) and nxt_first.islower():
            return True

        if cur_last not in '.!?' and nxt_first.islower() and nxt_last in '.!?':
            return True

        return False

    def _tc_suggest_action(
        self,
        cur: Dict,
        nxt: Dict,
        allow_overlap: bool,
        cache: Dict[int, Tuple[Optional[float], bool]],
        rate: float,
    ) -> str:
        if self._tc_text_suggests_merge(cur, nxt):
            return 'merge'

        keep_sev, _kt, _ktip, keep_diff = self._tc_compute_fit(
            cur, nxt, 'keep', allow_overlap, cache, rate
        )
        merge_sev, _mt, _mtip, merge_diff = self._tc_compute_fit(
            cur, nxt, 'merge', allow_overlap, cache, rate
        )
        if keep_sev is None or merge_sev is None:
            return 'keep'
        if keep_sev == 0:
            return 'keep'
        if merge_sev < keep_sev:
            return 'merge'
        if merge_sev == keep_sev and merge_diff < keep_diff - 0.05:
            return 'merge'
        return 'keep'

    def _time_corrector(self):
        if len(self._fragments) < 2:
            self.main._set_status("Time corrector: at least 2 fragments are required.")
            return
        if self._is_running:
            self.main._set_status("Time corrector: stop synthesis first.")
            return

        overlaps = self._tc_find_overlaps()
        gap_count = 0
        trim_count = 0
        for i in range(len(self._fragments) - 1):
            cur_start = self._fragments[i].get('start_ms', 0) or 0
            cur_end = self._fragments[i].get('end_ms', 0) or 0
            nxt_start = self._fragments[i + 1].get('start_ms', 0) or 0
            if nxt_start <= cur_start:
                continue
            if nxt_start > cur_end:
                gap_count += 1
            elif nxt_start < cur_end:
                trim_count += 1

        dlg = QDialog(self)
        dlg.setWindowTitle("Time corrector")
        dlg.resize(960, 700)
        dlg.setStyleSheet(self.main.styleSheet())
        lay = QVBoxLayout(dlg)
        lay.setSpacing(8)
        lay.setContentsMargins(16, 16, 16, 12)

        def _section(title: str):
            lbl = QLabel(title)
            lbl.setStyleSheet(f"color:{C['accent']};font-size:12px;font-weight:bold;")
            lay.addSpacing(4)
            lay.addWidget(lbl)

        def _short(text: str, limit: int = 40) -> str:
            t = " ".join((text or "").split())
            return t if len(t) <= limit else t[:limit].rstrip() + "\u2026"

        _section(f"A. Overlapping fragments  ({len(overlaps)})")
        hint_lbl = QLabel(
            "Choose what to do with each overlapping pair: merge the two fragments "
            "into one, or keep them as separate fragments.\n"
            "The best option is preselected for every pair (Keep separate when it fits, "
            "otherwise the option with the smaller overrun). Use Auto-select best to redo it.\n"
            "Pairs whose text looks like one sentence split across fragments "
            "(trailing ellipsis, comma, or no sentence-ending punctuation with a lowercase "
            "continuation) are preselected as Merge regardless of the fit.\n"
            "The Fit column shows whether the speech fits the available time for the "
            "selected action: green = fits, yellow = up to 2x the slot, "
            "orange = up to 4x the slot, red = more.\n"
            "Values with ~ are estimated from text length; generate the voices to get exact values."
            if overlaps else
            "No overlapping fragments found."
        )
        hint_lbl.setWordWrap(True)
        hint_lbl.setStyleSheet(f"color:{C['text3']};font-size:11px;")
        lay.addWidget(hint_lbl)

        tree = QTreeWidget()
        tree.setColumnCount(5)
        tree.setHeaderLabels(["Fragments", "Speakers", "Overlap", "Fit", "Action"])
        tree.setRootIsDecorated(False)
        tree.setWordWrap(True)
        tree.setMinimumHeight(200)
        header = tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        tree.setColumnWidth(4, 190)
        lay.addWidget(tree, 1)

        rows: List[Tuple[Dict, QComboBox, QTreeWidgetItem]] = []
        fit_cache: Dict[int, Tuple[Optional[float], bool]] = {}
        speech_rate = self._tc_speech_rate()
        fit_colors = ["#55bb55", "#dddd00", "#ff8800", "#ff4444"]

        def refresh_row_fit(item: QTreeWidgetItem, ov: Dict, combo: QComboBox):
            sev, text, tip, _diff = self._tc_compute_fit(
                ov['cur'], ov['nxt'], combo.currentData(),
                rb_overlap.isChecked(), fit_cache, speech_rate,
            )
            item.setText(3, text)
            item.setToolTip(3, tip)
            if sev is not None:
                item.setForeground(3, QColor(fit_colors[sev]))
            else:
                item.setForeground(3, QColor(C["text3"]))

        for ov in overlaps:
            cur = ov['cur']
            nxt = ov['nxt']
            label = (
                f"#{cur['index'] + 1} \"{_short(cur.get('text'))}\"  \u2194  "
                f"#{nxt['index'] + 1} \"{_short(nxt.get('text'))}\""
            )
            spk_cur = self._tc_speaker(cur) or "-"
            spk_nxt = self._tc_speaker(nxt) or "-"
            speakers = spk_cur if ov['same_speaker'] else f"{spk_cur} / {spk_nxt}"
            item = QTreeWidgetItem([label, speakers, f"{ov['overlap_ms']} ms", "", ""])
            tooltip_text = f"{cur.get('text', '')}\n---\n{nxt.get('text', '')}"
            if self._tc_text_suggests_merge(cur, nxt):
                tooltip_text += (
                    "\n\nText looks like one sentence split across fragments "
                    "- merge suggested."
                )
            item.setToolTip(0, tooltip_text)
            tree.addTopLevelItem(item)

            combo = QComboBox()
            combo.addItem("Merge fragments", "merge")
            combo.addItem("Keep separate", "keep")
            combo.setCurrentIndex(max(0, combo.findData("keep")))
            tree.setItemWidget(item, 4, combo)
            rows.append((ov, combo, item))
            combo.currentIndexChanged.connect(lambda _i: update_summary())

        def set_all(action: str):
            for _ov, combo, _item in rows:
                idx = combo.findData(action)
                if idx >= 0:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(idx)
                    combo.blockSignals(False)
            update_summary()

        def auto_select():
            allow_overlap = rb_overlap.isChecked()
            for ov, combo, _item in rows:
                action = self._tc_suggest_action(
                    ov['cur'], ov['nxt'], allow_overlap, fit_cache, speech_rate
                )
                idx = combo.findData(action)
                if idx >= 0:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(idx)
                    combo.blockSignals(False)
            update_summary()

        bulk_row = QHBoxLayout()
        bulk_row.setSpacing(6)
        for text, action in (
            ("Set all: Merge", "merge"),
            ("Set all: Keep separate", "keep"),
        ):
            b = QPushButton(text)
            b.setFixedHeight(26)
            b.setStyleSheet("font-size:11px;padding:2px 8px;")
            b.setEnabled(bool(rows))
            b.clicked.connect(lambda _checked=False, a=action: set_all(a))
            bulk_row.addWidget(b)
        auto_btn = QPushButton("Auto-select (best for lektor)")
        auto_btn.setFixedHeight(26)
        auto_btn.setStyleSheet("font-size:11px;padding:2px 8px;")
        auto_btn.setEnabled(bool(rows))
        auto_btn.setToolTip(
            "Choose the best action for every pair using the current timing option:\n"
            "Text-based merge is preferred when fragments look like one sentence; "
            "otherwise Keep separate when it fits, else the option with the smaller overrun."
        )
        auto_btn.clicked.connect(lambda _checked=False: auto_select())
        bulk_row.addWidget(auto_btn)
        bulk_row.addStretch()
        lay.addLayout(bulk_row)

        _section("B. Subtitle timing")
        rb_overlap = QRadioButton(
            "I want subtitles to overlap  (good for dubbing)"
        )
        rb_no_overlap = QRadioButton(
            "I don't want subtitles to overlap  (good for lektor)"
        )
        rb_no_overlap.setChecked(True)
        rb_overlap.setStyleSheet(f"color:{C['text']};font-size:12px;")
        rb_no_overlap.setStyleSheet(f"color:{C['text']};font-size:12px;")
        lay.addWidget(rb_overlap)
        lay.addWidget(rb_no_overlap)
        timing_lbl = QLabel(
            "Overlap: gaps are closed, but fragments that already overlap are not changed, "
            "e.g. 00:00:22,000 --> 00:00:23,200 and 00:00:23,000 --> 00:00:25,000 stay as they are.\n"
            "No overlap: the end of every fragment is set to the start of the next fragment, "
            "e.g. 00:00:22,000 --> 00:00:23,000 and 00:00:23,000 --> 00:00:25,000."
        )
        timing_lbl.setWordWrap(True)
        timing_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        lay.addWidget(timing_lbl)

        _section("C. Summary")
        summary_lbl = QLabel("")
        summary_lbl.setWordWrap(True)
        summary_lbl.setStyleSheet(f"color:{C['text']};font-size:12px;")
        lay.addWidget(summary_lbl)

        warn_lbl = QLabel("")
        warn_lbl.setWordWrap(True)
        warn_lbl.setStyleSheet(f"color:{C['warning']};font-size:11px;")
        warn_lbl.setVisible(False)
        lay.addWidget(warn_lbl)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        undo_btn = QPushButton("\u21b6 Undo last correction")
        undo_btn.setEnabled(getattr(self, '_time_corrector_undo', None) is not None)
        undo_btn.setToolTip("Undo the last time correction")
        apply_btn = QPushButton("Apply")
        cancel_btn = QPushButton("Cancel")
        apply_btn.setStyleSheet(_btn(C["accent"]))
        btn_row.addWidget(undo_btn)
        btn_row.addStretch()
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(cancel_btn)
        lay.addLayout(btn_row)

        result: Dict = {}

        def update_summary():
            counts = {"merge": 0, "keep": 0}
            merged_ids: set = set()
            for ov, combo, _item in rows:
                action = combo.currentData()
                counts[action] = counts.get(action, 0) + 1
                if action == "merge":
                    merged_ids.add(id(ov['cur']))
                    merged_ids.add(id(ov['nxt']))
            done_count = sum(
                1 for f in self._fragments
                if id(f) in merged_ids and f.get('status') == 'done'
            )
            allow_overlap = rb_overlap.isChecked()
            if allow_overlap:
                timing_changes = gap_count
                timing_text = f"{gap_count} gap(s) to close, overlapping fragments left unchanged"
            else:
                timing_changes = gap_count + trim_count
                timing_text = (
                    f"{gap_count} gap(s) to close, {trim_count} overlap(s) to shorten"
                )
            summary_lbl.setText(
                f"{counts['merge']} pair(s) to merge, "
                f"{counts['keep']} kept separate, {timing_text}"
            )
            if done_count:
                warn_lbl.setText(
                    f"{done_count} already synthesized fragment(s) will be merged "
                    f"\u2013 their audio will need to be regenerated"
                )
                warn_lbl.setVisible(True)
            else:
                warn_lbl.setVisible(False)
            apply_btn.setEnabled(bool(counts['merge'] or timing_changes))
            for ov, combo, item in rows:
                refresh_row_fit(item, ov, combo)

        def on_apply():
            result["decisions"] = [
                (ov['cur'], ov['nxt'], combo.currentData()) for ov, combo, _item in rows
            ]
            result["allow_overlap"] = rb_overlap.isChecked()
            dlg.accept()

        def on_undo():
            result["undo"] = True
            dlg.accept()

        rb_overlap.toggled.connect(lambda _c: update_summary())
        rb_no_overlap.toggled.connect(lambda _c: update_summary())
        apply_btn.clicked.connect(on_apply)
        undo_btn.clicked.connect(on_undo)
        cancel_btn.clicked.connect(dlg.reject)

        auto_select()

        accepted = dlg.exec() == QDialog.DialogCode.Accepted
        if accepted and result.get("undo"):
            self._time_corrector_undo_last()
            return
        if not accepted or "decisions" not in result:
            return

        self._time_corrector_apply(result["decisions"], result["allow_overlap"])

    def _time_corrector_apply(self, decisions: List[Tuple[Dict, Dict, str]], allow_overlap: bool = False):
        merge_pairs = [(c, n) for c, n, a in decisions if a == 'merge']
        kept = sum(1 for _c, _n, a in decisions if a == 'keep')

        position = {id(f): i for i, f in enumerate(self._fragments)}
        merge_pairs.sort(key=lambda p: position.get(id(p[0]), 0))
        member_groups: List[List[Dict]] = []
        for cur, nxt in merge_pairs:
            if member_groups and member_groups[-1][-1] is cur:
                member_groups[-1].append(nxt)
            else:
                member_groups.append([cur, nxt])

        has_merge = bool(member_groups)

        if has_merge:
            self.main._stop_play()
            self._release_audio_file_locks()

        try:
            safety = snapshot_fragments(self._fragments, has_merge)
        except Exception as e:
            logger.error(f"Time corrector backup failed:\n{traceback.format_exc()}")
            self.main._set_status(
                f"Time corrector: could not back up audio files, nothing was changed: {e}",
                C["error"],
            )
            return

        undo_before = self._undo_begin(has_merge)

        joined = 0
        changed: Dict[int, Dict] = {}

        try:
            if member_groups:
                infos: List[Dict] = []
                for members in member_groups:
                    text = ""
                    for f in members:
                        text = self._tc_join_text(text, f.get('text', ''))
                    infos.append({
                        'members': members,
                        'text': text,
                        'end_ms': max((f.get('end_ms', 0) or 0) for f in members),
                    })
                infos.sort(key=lambda g: g['members'][0]['index'], reverse=True)
                for gi in infos:
                    self._do_merge_srt_fragments(
                        gi['members'],
                        text_override=gi['text'],
                        end_ms_override=gi['end_ms'],
                        drop_audio=True,
                        refresh=False,
                    )

            for i in range(len(self._fragments) - 1):
                cur = self._fragments[i]
                nxt = self._fragments[i + 1]
                cur_start = cur.get('start_ms', 0) or 0
                cur_end = cur.get('end_ms', 0) or 0
                nxt_start = nxt.get('start_ms', 0) or 0
                if nxt_start <= cur_start or nxt_start == cur_end:
                    continue
                if allow_overlap and nxt_start < cur_end:
                    continue
                cur['end_ms'] = nxt_start
                cur['srt_end_ms'] = nxt_start
                cur['timestamp'] = f"{_ms_to_ts(cur_start)} --> {_ms_to_ts(nxt_start)}"
                joined += 1
                changed[id(cur)] = cur

            if member_groups:
                self._rename_outputs_to_match_order()
                self._tc_reset_current_fragment()
                self._populate_tree()
                self.main._update_action_buttons()
            else:
                for frag in changed.values():
                    self._update_tree_item(frag['index'])
        except Exception as e:
            logger.error(f"Time corrector failed:\n{traceback.format_exc()}")
            rolled_back = False
            try:
                self._release_audio_file_locks()
                self._undo_restore(safety)
                rolled_back = True
            except Exception:
                logger.error(f"Time corrector rollback failed:\n{traceback.format_exc()}")
            if rolled_back:
                discard_snapshot(safety)
                discard_snapshot(undo_before)
                self.main._set_status(
                    f"Time corrector failed, changes were rolled back: {e}", C["error"]
                )
            else:
                logger.error(f"Safety backup kept in: {safety.get('backup_dir')}")
                self.main._set_status(
                    f"Time corrector failed and rollback failed. Backup kept in: {safety.get('backup_dir')}",
                    C["error"],
                )
            return

        if not (member_groups or joined):
            discard_snapshot(safety)
            discard_snapshot(undo_before)
            self.main._set_status("Time corrector: nothing to change.")
            return

        old = getattr(self, '_time_corrector_undo', None)
        self._time_corrector_undo = {
            'state': safety,
            'srt_path': self._srt_path,
            'has_merge': has_merge,
            'post_refs': list(self._fragments),
        }
        if old is not None:
            discard_snapshot(old.get('state'))
        self._tc_refresh_undo_button()

        self._undo_commit("time correction", undo_before)
        self.main._set_status(
            f"Time corrector: {len(member_groups)} group(s) merged, "
            f"{joined} timestamp(s) joined, {kept} kept separate.",
            C["accent"],
        )

    def _time_corrector_undo_last(self):
        snapshot = getattr(self, '_time_corrector_undo', None)
        if snapshot is None:
            self._tc_refresh_undo_button()
            return
        if self._is_running:
            self.main._set_status("Time corrector: stop synthesis before undoing.")
            return

        post_refs = snapshot['post_refs']
        unchanged = (
            snapshot['srt_path'] == self._srt_path
            and len(post_refs) == len(self._fragments)
            and all(a is b for a, b in zip(post_refs, self._fragments))
        )
        if not unchanged:
            self._time_corrector_undo = None
            self._tc_refresh_undo_button()
            self.main._set_status(
                "Time corrector: cannot undo, fragments were changed after the correction.",
                C["warning"],
            )
            return

        undo_before = self._undo_begin(snapshot['has_merge'])

        self.main._stop_play()
        self._release_audio_file_locks()
        self._dub_hold_until = float('inf')
        try:
            try:
                self._undo_restore(snapshot['state'])
            except Exception as e:
                logger.error(f"Time corrector undo failed:\n{traceback.format_exc()}")
                self.main._set_status(
                    f"Time corrector undo failed, backup kept: {e}", C["error"]
                )
                return
            discard_snapshot(snapshot['state'])
            self._time_corrector_undo = None
            self._tc_refresh_undo_button()
            self._undo_commit("undo of time correction", undo_before)
            self.main._set_status("Time corrector: last correction undone.", C["accent"])
        finally:
            self._dub_hold_until = 0.0

    def _show_timing_issues_dialog(self):
        from widgets import TimingIssuesDialog

        dlg = TimingIssuesDialog(self.main)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        select_yellow = dlg.yellow_check.isChecked()
        select_orange = dlg.orange_check.isChecked()
        select_red = dlg.red_check.isChecked()

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
                idx = child.data(COL_STATUS, Qt.ItemDataRole.UserRole)
                if idx is None:
                    continue

                match = False
                if not child.isHidden():
                    frag = frag_map.get(idx)
                    if frag and frag.get('status') == 'done':
                        dur = _get_wav_duration(frag.get('output_path', ''))
                        if dur is not None:
                            start_ms = frag.get('start_ms', 0) or 0
                            end_ms = frag.get('srt_end_ms', frag.get('end_ms', 0)) or 0
                            slot_s = (end_ms - start_ms) / 1000.0
                            if dur > slot_s:
                                safe_slot = max(slot_s, 0.001)
                                if dur < 2 * safe_slot:
                                    match = select_yellow
                                elif dur < 4 * safe_slot:
                                    match = select_orange
                                else:
                                    match = select_red

                child.setCheckState(
                    COL_STATUS,
                    Qt.CheckState.Checked if match else Qt.CheckState.Unchecked,
                )
        finally:
            self._tree.blockSignals(False)
            self._tree.setUpdatesEnabled(True)
        self._sync_checks_from_tree()

    # ------------------------------------------------------------------
    # Speaker UI / Dubbing
    # ------------------------------------------------------------------

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
                while self.main._speakers_lay.count():
                    item = self.main._speakers_lay.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                self.main._speakers_container.setVisible(False)
                self.main._voice_single_container.setVisible(True)
                if hasattr(self.main, "_whisper_section_widget"):
                    self.main._whisper_section_widget.setVisible(True)
            return

        if set(speakers) == set(self._speaker_list) and self._dubbing_mode:
            return

        saved_paths: Dict[str, str] = {}
        saved_texts: Dict[str, str] = {}
        for spk, sv in self._speaker_voices.items():
            drop = sv.get("drop")
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
                    self.main._whisper_backend is not None
                    and self.main._whisper_backend.is_downloaded(self.main._w_size.currentData())
                )

        for spk, text in saved_texts.items():
            if spk in self._speaker_voices and text:
                sv = self._speaker_voices[spk]
                if "ref_text" in sv:
                    sv["ref_text"].setPlainText(text)

        self.main._set_status(
            f"{len(speakers)} speaker(s) — add reference audio for each in Voice cloning.",
            C["accent"],
        )

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
                self.main, "Nothing selected",
                "No fragments are checked.\nCheck at least one fragment first."
            )
            return

        checked_frags = [f for f in self._fragments if f['index'] in checked_indices]

        dlg = QDialog(self)
        dlg.setWindowTitle("Add text to selected fragments")
        dlg.resize(540, 310)
        dlg.setStyleSheet(self.main.styleSheet())

        lay = QVBoxLayout(dlg)
        lay.setSpacing(10)

        radio_row = QHBoxLayout()
        rb_prepend = QRadioButton("Prepend  (add before text)")
        rb_append = QRadioButton("Append  (add after text)")
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
            mode_word = "prefix" if is_prepend else "suffix"
            counts = _build_counts(is_prepend)
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

        new_text = editor.text().strip()
        is_prepend = rb_prepend.isChecked()
        key = 'prefix' if is_prepend else 'suffix'
        mode_word = "prefix" if is_prepend else "suffix"

        before = self._undo_begin()

        for frag in self._fragments:
            if frag['index'] not in checked_indices:
                continue
            frag[key] = new_text
            frag['status'] = 'waiting'
            frag['error_msg'] = None
            self._update_tree_item(frag['index'])

        self._undo_commit(f"{mode_word} change on {len(checked_indices)} fragment(s)", before)

        if new_text:
            self.main._set_status(
                f"{mode_word.capitalize()} '{new_text}' applied to {len(checked_indices)} "
                f"fragment(s) — status reset to waiting."
            )
        else:
            self.main._set_status(
                f"{mode_word.capitalize()} cleared for {len(checked_indices)} "
                f"fragment(s) — status reset to waiting."
            )

    def _prompt_speaker_name(self, title: str, message: str, current: str) -> Tuple[Optional[str], bool]:
        all_speakers = sorted(
            {
                (f.get('speaker') or "").strip()
                for f in self._fragments
                if (f.get('speaker') or "").strip()
            },
            key=str.lower,
        )

        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(380, 150)
        dlg.setStyleSheet(self.main.styleSheet())

        lay = QVBoxLayout(dlg)
        lay.setSpacing(10)
        lay.setContentsMargins(14, 14, 14, 10)

        lbl = QLabel(message)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        lay.addWidget(lbl)

        combo = QComboBox()
        combo.setEditable(True)
        combo.addItems(all_speakers)
        combo.setCurrentText(current)
        combo.setMinimumHeight(26)
        combo.lineEdit().setPlaceholderText("Leave empty to clear / use default voice")
        lay.addWidget(combo)

        hint = QLabel(
            "Choose an existing speaker from the list, type a new name, "
            "or leave the field empty to clear the speaker / use the default voice."
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

        combo.lineEdit().selectAll()
        combo.setFocus()

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None, False

        return combo.currentText(), True

    def _edit_speaker(self, frag: Dict, clear: bool = False):
        if clear:
            before = self._undo_begin()
            frag['speaker'] = None
            self._update_tree_item(frag['index'])
            self._sync_speaker_ui_from_fragments()
            self._undo_commit(f"speaker removal on fragment #{frag['index'] + 1}", before)
            return

        current = frag.get('speaker') or ""
        name, ok = self._prompt_speaker_name(
            "Speaker name",
            "Select or enter speaker name:",
            current,
        )
        if ok:
            before = self._undo_begin()
            frag['speaker'] = name.strip() or None
            self._update_tree_item(frag['index'])
            self._sync_speaker_ui_from_fragments()
            self._undo_commit(f"speaker change on fragment #{frag['index'] + 1}", before)

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
                self.main, "Nothing selected",
                "No fragments are checked.\nCheck at least one fragment first."
            )
            return

        checked_frags = [f for f in self._fragments if f['index'] in checked_indices]
        current = checked_frags[0].get('speaker') or ""

        name, ok = self._prompt_speaker_name(
            "Speaker name",
            f"Set speaker for {len(checked_frags)} selected fragment(s):",
            current,
        )
        if not ok:
            return

        new_speaker = name.strip() or None
        before = self._undo_begin()
        for frag in checked_frags:
            frag['speaker'] = new_speaker
            self._update_tree_item(frag['index'])

        self._sync_speaker_ui_from_fragments()
        self._undo_commit(f"speaker change on {len(checked_frags)} fragment(s)", before)
        self.main._set_status(
            f"Speaker {'set to ' + new_speaker if new_speaker else 'cleared'} "
            f"for {len(checked_frags)} fragment(s).",
            C["accent"],
        )

    # ------------------------------------------------------------------
    # Dubbing Pipeline
    # ------------------------------------------------------------------

    def _on_dubbing_clicked(self):
        backend = self.main._backend
        is_tada = backend is not None and getattr(backend, "auth_required", False)

        if is_tada:
            hf_token = self.main._get_hf_token()
            if not hf_token:
                QMessageBox.critical(
                    self.main,
                    "HuggingFace token missing",
                    "Model requires a HuggingFace token.\n\n"
                    "Make sure the token was saved, or place your token "
                    "in the '.hf_token' file in the application folder.",
                )
                return
        else:
            hf_token = self.main._get_hf_token()
            if not hf_token:
                hf_token = self.main._show_hf_token_dialog()
                if not hf_token:
                    return

        path, _ = QFileDialog.getOpenFileName(
            self.main, "Select video file for dubbing", _get_last_dir("video"),
            "Video (*.mp4 *.mkv *.avi *.mov *.webm *.wmv);;All files (*)",
        )
        if not path:
            return

        _set_last_dir("video", path)
        self._dubbing_video_path = path
        self._hf_token = hf_token
        self._btn_dubbing.setEnabled(False)
        self._btn_dubbing.setText("⏳  Processing…")

        self._start_vocal_extraction(path)

    def _start_vocal_extraction(self, video_path: str):
        self.main._set_status("Extracting audio and isolating vocals.")
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)

        self._w_vocal_extract = DubbingVocalExtractWorker(video_path)
        self._w_vocal_extract.status.connect(lambda m: self.main._set_status(m))
        self._w_vocal_extract.finished.connect(self._on_vocal_extraction_done)
        self._w_vocal_extract.error.connect(
            lambda e: self.main._on_error("Vocal extraction error", e,
                                           reset_fn=lambda: (
                                               self._btn_dubbing.setEnabled(True),
                                               self._btn_dubbing.setText("🎙  I want dubbing"),
                                               self._progress.setVisible(False),
                                           ))
        )
        self._w_vocal_extract.start()

    def _on_vocal_extraction_done(self, vocals_path: str):
        self.main._set_status(
            f"Vocals extracted: {Path(vocals_path).name} — starting diarization…"
        )
        self._start_diarization(vocals_path)

    def _start_diarization(self, audio_path: str):
        self.main._set_status("Running speaker diarization (pyannote)…")

        self._w_diarization = DiarizationWorker(audio_path, self._hf_token)
        self._w_diarization.status.connect(lambda m: self.main._set_status(m))
        self._w_diarization.finished.connect(self._on_diarization_done)
        self._w_diarization.error.connect(
            lambda e: self.main._on_error("Diarization error", e,
                                           reset_fn=lambda: (
                                               self._btn_dubbing.setEnabled(True),
                                               self._btn_dubbing.setText("🎙  I want dubbing"),
                                               self._progress.setVisible(False),
                                           ))
        )
        self._w_diarization.start()

    def _on_diarization_done(self, result: dict):
        self._progress.setVisible(False)

        segments = result.get("segments", [])
        speaker_map = result.get("speaker_map", {})

        if not speaker_map:
            QMessageBox.warning(self.main, "No speakers detected",
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

        self._vid_row_widget.setVisible(False)

        self._populate_tree()

        n = len(speaker_map)
        self.main._set_status(
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
            end_s = (frag.get("end_ms") or 0) / 1000.0
            if end_s <= start_s:
                continue

            overlap: Dict[str, float] = {}
            for seg in segments:
                ov_start = max(start_s, seg["start"])
                ov_end = min(end_s, seg["end"])
                if ov_end > ov_start:
                    label = seg["speaker"]
                    overlap[label] = overlap.get(label, 0.0) + (ov_end - ov_start)

            if overlap:
                best_label = max(overlap, key=lambda k: overlap[k])
                frag["speaker"] = speaker_map.get(best_label, best_label)

    def _rebuild_voice_cloning_for_speakers(self):
        while self.main._speakers_lay.count():
            item = self.main._speakers_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._speaker_voices.clear()
        is_supertonic = self.main._is_supertonic_active()
        is_piper = self.main._is_piper_active()

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
                    WHISPER_SIZES.index(self.main._w_size.currentData())
                    if self.main._w_size.currentData() in WHISPER_SIZES else 0
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
                mono_check.setChecked(self.main._mono_check.isChecked())
                sr_combo = QComboBox()
                for val, label in TARGET_SR_OPTIONS:
                    sr_combo.addItem(label, val)
                for i in range(sr_combo.count()):
                    if sr_combo.itemData(i) == self.main._sr_combo.currentData():
                        sr_combo.setCurrentIndex(i)
                        break

                chk_bit_depth = QCheckBox("Override bit depth")
                chk_bit_depth.setChecked(self.main._chk_bit_depth.isChecked())
                bit_depth_combo = QComboBox()
                for code, label in [("PCM_16", "16-bit"), ("PCM_24", "24-bit"), ("FLOAT", "32-bit float")]:
                    bit_depth_combo.addItem(label, code)
                for i in range(bit_depth_combo.count()):
                    if bit_depth_combo.itemData(i) == self.main._bit_depth_combo.currentData():
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

            self.main._speakers_lay.addWidget(spk_group)

        self.main._voice_single_container.setVisible(False)
        self.main._speakers_container.setVisible(True)

        if hasattr(self.main, "_whisper_section_widget"):
            self.main._whisper_section_widget.setVisible(False)

        if hasattr(self.main, "_voice_cloning_section"):
            self.main._voice_cloning_section.expand()

    def _get_speaker_voices_dict(self) -> Optional[Dict]:
        if not self._dubbing_mode or not self._speaker_list:
            return None
        result = {}
        is_supertonic = self.main._is_supertonic_active()
        is_piper = self.main._is_piper_active()
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

    def _on_speaker_ref_dropped(self, speaker: str, path: str):
        sv = self._speaker_voices.get(speaker)
        if not sv:
            return
        sv["player"].load(path)
        sv["proc_btn"].setEnabled(True)
        w_size_combo = sv.get("w_size", self.main._w_size)
        sv["tr_btn"].setEnabled(
            self.main._whisper_backend is not None
            and self.main._whisper_backend.is_downloaded(w_size_combo.currentData())
        )
        info = _audio_info_str(path)
        sv["audio_info_lbl"].setText(info)
        self.main._set_status(
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
        self.main._set_status(f"[{speaker}] Reference audio removed.")

    def _transcribe_speaker_ref(self, speaker: str):
        sv = self._speaker_voices.get(speaker)
        if not sv:
            return
        path = sv["drop"].file_path
        if not path:
            QMessageBox.warning(
                self.main, "No audio",
                f"Upload reference audio for {speaker} first."
            )
            return

        w_size_combo = sv.get("w_size", self.main._w_size)
        size = w_size_combo.currentData()

        if not self.main._whisper_backend.is_downloaded(size):
            reply = QMessageBox.question(
                self.main, f"Download Whisper {size}",
                f"Whisper {size} ({WHISPER_SIZE_MB.get(size, '')}) is not downloaded.\n"
                f"Download now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.main._w_size.setCurrentIndex(w_size_combo.currentIndex())
                self.main._start_whisper_download()
            return

        w_lang_combo = sv.get("w_lang", self.main._w_lang)
        lang = w_lang_combo.currentData()

        sv["tr_btn"].setEnabled(False)
        sv["tr_btn"].setText("⏳  Transcribing…")
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)

        self._w_tr = TranscribeWorker(self.main._whisper_backend, path, size, lang)
        self._w_tr.status.connect(lambda m: self.main._set_status(m))
        self._w_tr.finished.connect(
            lambda text, s=speaker: self._on_speaker_transcribe_done(s, text)
        )
        self._w_tr.error.connect(
            lambda e, s=speaker: self.main._on_error("Transcription error", e,
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
        self.main._set_status(
            f"[{speaker}] Transcription complete: {len(text)} chars", C["success"]
        )

    def _process_speaker_audio(self, speaker: str):
        sv = self._speaker_voices.get(speaker)
        if not sv:
            return
        path = sv["drop"].file_path
        if not path:
            QMessageBox.warning(
                self.main, "No audio",
                f"Upload reference audio for {speaker} first."
            )
            return

        chk_bit = sv.get("chk_bit_depth", self.main._chk_bit_depth)
        bd_combo = sv.get("bit_depth_combo", self.main._bit_depth_combo)
        output_subtype = (
            bd_combo.currentData()
            if chk_bit.isChecked()
            else "PCM_16"
        )

        safe_spk = "".join(c for c in speaker if c.isalnum() or c in " -_")
        output_name = f"speaker_{safe_spk}_processed.wav"

        mono_chk = sv.get("mono_check", self.main._mono_check)
        sr_cbo = sv.get("sr_combo", self.main._sr_combo)
        backend = self.main._backend

        settings = {
            "target_sr": sr_cbo.currentData(),
            "to_mono": mono_chk.isChecked(),
            "isolate_vocals": sv["demucs_chk"].isChecked(),
            "normalize": True,
            "output_subtype": output_subtype,
            "device": "cuda" if (
                backend and backend.device == "cuda"
            ) else "cpu",
            "output_name": output_name,
        }

        sv["proc_btn"].setEnabled(False)
        sv["proc_btn"].setText("⏳  Processing…")
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)

        self._w_proc = AudioProcessWorker(self.main._preprocessor, path, settings)
        self._w_proc.status.connect(lambda m: self.main._set_status(m))
        self._w_proc.finished.connect(
            lambda out, s=speaker: self._on_speaker_proc_done(s, out)
        )
        self._w_proc.error.connect(
            lambda e, s=speaker: self.main._on_error("Audio processing error", e,
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
        w_size_combo = sv.get("w_size", self.main._w_size)
        sv["tr_btn"].setEnabled(
            self.main._whisper_backend is not None
            and self.main._whisper_backend.is_downloaded(w_size_combo.currentData())
        )
        self.main._set_status(
            f"[{speaker}] Audio processed: {Path(out).name}", C["success"]
        )

    def _reset_dubbing_state(self):
        self._dubbing_mode = False
        self._dubbing_video_path = None
        self._hf_token = None
        self._speaker_list = []
        self._speaker_voices = {}

        if hasattr(self, "_btn_dubbing"):
            self._btn_dubbing.setText("🎙  I want dubbing")
            self._btn_dubbing.setStyleSheet("")
            self._btn_dubbing.setVisible(False)
            self._btn_dubbing.setEnabled(False)

        if hasattr(self.main, "_voice_single_container"):
            self.main._voice_single_container.setVisible(True)

        if hasattr(self.main, "_speakers_lay") and hasattr(self.main, "_speakers_container"):
            while self.main._speakers_lay.count():
                item = self.main._speakers_lay.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.main._speakers_container.setVisible(False)

        if hasattr(self.main, "_whisper_section_widget"):
            self.main._whisper_section_widget.setVisible(True)

        self._vid_row_widget.setVisible(True)

    def _update_dubbing_visibility(self):
        if not hasattr(self, "_btn_dubbing"):
            return
        srt_loaded = bool(self._fragments)
        incompatible = self.main._is_pyannote_incompatible()
        visible = srt_loaded and not incompatible
        self._btn_dubbing.setVisible(visible)
        self._btn_dubbing.setEnabled(visible)

    # ------------------------------------------------------------------
    # Video Waveform
    # ------------------------------------------------------------------

    def _on_media_status_changed(self, status):
        if status != QMediaPlayer.MediaStatus.LoadedMedia:
            return
        if self.main._video_player.is_playing():
            if self._media_player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
                self._media_player.play()
            return
        self._media_player.play()
        self._media_player.pause()

    def _on_show_video_waveform(self):
        if getattr(self, '_video_open', False):
            self._close_video()
            return

        external_audio = self._audio_path_edit.text().strip()
        if external_audio and os.path.exists(external_audio):
            self.main._video_player.load(external_audio)
            self.main._video_player.setVisible(True)
            self.main._video_source_path = None
            self._video_open = True
            self._btn_show_video_wave.setText("✕  Close video")
            self.main._set_status(f"Loaded external audio: {Path(external_audio).name}", C["success"])
            return

        if not self.main._ffmpeg_ok:
            QMessageBox.warning(self.main, "ffmpeg not found",
                                "This feature requires ffmpeg installed in PATH.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self.main, "Select video file", _get_last_dir("video"),
            "Video (*.mp4 *.mkv *.avi *.mov *.webm *.wmv);;All files (*)",
        )
        if not path:
            return
        _set_last_dir("video", path)
        self.main._video_source_path = path
        self._video_open = True

        if getattr(self, '_media_player', None) is not None:
            self._media_player.setSource(QUrl.fromLocalFile(path))
            if getattr(self, '_video_section', None) is not None:
                self._video_section.set_enabled(True)
                self._video_section.expand()
            self._video_loaded = True
            self._update_video_preview_visibility()

        try:
            self.main._video_player.playback_state_changed.disconnect(self._sync_video_playback)
            self.main._video_player.position_updated.disconnect(self._sync_video_position)
        except TypeError:
            pass

        self.main._video_player.playback_state_changed.connect(self._sync_video_playback)
        self.main._video_player.position_updated.connect(self._sync_video_position)

        self._btn_show_video_wave.setEnabled(False)
        self._btn_show_video_wave.setText("⏳  Extracting audio…")
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)

        self._w_vid_extract = VideoAudioExtractWorker(path)
        self._w_vid_extract.status.connect(lambda m: self.main._set_status(m))
        self._w_vid_extract.finished.connect(self._on_video_audio_extracted)
        self._w_vid_extract.error.connect(
            lambda e: self.main._on_error("Video audio extraction error", e,
                                           reset_fn=lambda: (
                                               self._close_video(),
                                               self._progress.setVisible(False),
                                           ))
        )
        self._w_vid_extract.start()

    def _on_video_audio_extracted(self, wav_path: str):
        self._progress.setVisible(False)
        self._btn_show_video_wave.setEnabled(True)
        self._btn_show_video_wave.setText("✕  Close video")
        self.main._video_player.load(wav_path)
        if self.main._video_source_path:
            self._vid_path_edit.setText(self.main._video_source_path)
            self._update_video_info()
        self._vid_row_widget.setVisible(False)
        self.main._set_status(
            f"Video audio loaded: {Path(self.main._video_source_path).name}", C["success"]
        )

    def _close_video(self):
        win = getattr(self, '_fs_window', None)
        if win is not None:
            win.leave()

        player = self.main._video_player
        for sig, slot in (
            (player.playback_state_changed, self._sync_video_playback),
            (player.position_updated, self._sync_video_position),
        ):
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass

        if getattr(self, '_dub_player', None) is not None:
            self._dub_player.stop()
            self._dub_player.setSource(QUrl())
        if getattr(self, '_media_player', None) is not None:
            self._media_player.stop()
            self._media_player.setSource(QUrl())

        self._current_sim_frag_idx = None
        self._last_sim_pos_ms = None
        self._dub_schedule_sig = None

        player.clear()

        prev = getattr(self, '_hl_item', None)
        if prev is not None:
            self._tree.blockSignals(True)
            try:
                for col in range(self._tree.columnCount()):
                    prev.setBackground(col, QBrush())
            except RuntimeError:
                pass
            finally:
                self._tree.blockSignals(False)
            self._hl_item = None

        self._video_loaded = False
        self._video_open = False
        self._update_video_preview_visibility()
        if getattr(self, '_video_section', None) is not None:
            self._video_section.set_enabled(False)

        src = self.main._video_source_path
        self.main._video_source_path = None
        if not self._dubbing_mode:
            if src and self._vid_path_edit.text().strip() == src:
                self._vid_path_edit.setText("")
                self._update_video_info()
            self._vid_row_widget.setVisible(True)

        self._btn_show_video_wave.setEnabled(True)
        self._btn_show_video_wave.setText("🎬  Show video")
        self.main._set_status("Video closed.")

    # ------------------------------------------------------------------
    # Lektor Export
    # ------------------------------------------------------------------

    def _update_video_info(self):
        """Update the video info label."""
        path = self._vid_path_edit.text().strip()
        if not path or not os.path.exists(path):
            self._vid_info_lbl.setText("")
            return
        try:
            info = self._video_audio_info_str(path)
            self._vid_info_lbl.setText(info)
        except Exception:
            self._vid_info_lbl.setText("Unknown audio format")

    def _update_audio_info(self):
        """Update the external audio info label."""
        path = self._audio_path_edit.text().strip()
        if not path or not os.path.exists(path):
            self._audio_info_lbl.setText("")
            return
        try:
            info = _audio_info_str(path)
            self._audio_info_lbl.setText(info)
        except Exception:
            self._audio_info_lbl.setText("Unknown audio format")

    def _video_audio_info_str(self, video_path: str) -> str:
        """Return audio stream info from a video file (ffprobe)."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", "-show_format", "-select_streams", "a", video_path
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            if result.returncode != 0:
                return ""
            data = json.loads(result.stdout.decode(errors="replace"))
            streams = data.get("streams", [])
            if not streams:
                return "No audio stream"
            s = streams[0]
            codec = s.get("codec_name", "unknown").upper()
            sr = int(s.get("sample_rate", 0))
            sr_str = f"{sr/1000:.1f} kHz" if sr >= 1000 else f"{sr} Hz"
            channels = s.get("channels", 0)
            ch_str = "Mono" if channels == 1 else "Stereo" if channels == 2 else f"{channels}ch"
            bit_depth = s.get("bits_per_raw_sample") or s.get("bits_per_sample") or ""
            if not bit_depth and codec in ("AAC", "MP3", "AC3", "EAC3"):
                bit_depth = "(lossy)"
            bit_str = f"{bit_depth}" if bit_depth else ""
            duration = float(s.get("duration", 0)) or float(data.get("format", {}).get("duration", 0))
            dur_str = _fmt(duration) if duration > 0 else ""
            bitrate = s.get("bit_rate") or data.get("format", {}).get("bit_rate")
            bitrate_str = ""
            if bitrate:
                br = int(bitrate)
                if br > 1000:
                    bitrate_str = f"{br/1000:.0f} kbps"
                else:
                    bitrate_str = f"{br} bps"
            parts = [ch_str, bit_str, sr_str, bitrate_str, codec, dur_str]
            return " • ".join(p for p in parts if p)
        except Exception:
            return ""

    def _get_overlap_mode(self) -> str:
        combo = getattr(self, '_overlap_combo', None)
        if combo is None:
            return "never"
        mode = combo.currentData()
        return mode if mode in ("never", "speakers", "always") else "never"

    def _fragment_start_with_overlap(self, mode: str, speaker, desired, cursors: Dict[str, float]):
        if mode == "always":
            return desired
        key = str(speaker or "").strip() if mode == "speakers" else ""
        return max(desired, cursors.get(key, 0))

    def _advance_overlap_cursor(self, mode: str, speaker, end, cursors: Dict[str, float]) -> None:
        if mode == "always":
            return
        key = str(speaker or "").strip() if mode == "speakers" else ""
        cursors[key] = end

    def _atempo_target_ms(self, slot_ms: float, audio_dur_ms: float) -> Optional[float]:
        if slot_ms <= 0 or audio_dur_ms <= slot_ms:
            return None
        overshoot_ms = audio_dur_ms - slot_ms
        if overshoot_ms <= self._atempo_threshold.value():
            return float(slot_ms)
        halve_check = getattr(self, '_atempo_halve_check', None)
        if halve_check is not None and halve_check.isChecked():
            return slot_ms + overshoot_ms / 2.0
        return None

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

        autofit = self.main._ffmpeg_ok and self._autofit_check.isChecked()
        tmp_files: List[str] = []

        max_end_ms = max((f.get('end_ms') or f.get('start_ms', 0)) for f in done_frags)

        total_audio_s = sum(_get_wav_duration(f['output_path']) or 0.0 for f in done_frags)
        total_duration_s = max(max_end_ms / 1000.0, total_audio_s) + abs(offset_ms) / 1000.0 + 10.0
        total_samples = int(total_duration_s * sample_rate)
        track = np.zeros(total_samples, dtype=np.float32)
        overlap_mode = self._get_overlap_mode()
        cursors: Dict[str, int] = {}

        for frag in done_frags:
            raw_start_ms = frag.get('start_ms', 0) or 0
            raw_end_ms = frag.get('end_ms', 0) or 0
            adj_start_ms = raw_start_ms + offset_ms
            adj_end_ms = raw_end_ms + offset_ms
            audio_file = frag['output_path']

            slot_ms = max(0, adj_end_ms - adj_start_ms)

            audio_dur_s = _get_wav_duration(audio_file) or 0.0
            audio_dur_ms = int(audio_dur_s * 1000)

            if autofit:
                target_ms = self._atempo_target_ms(slot_ms, audio_dur_ms)
                if target_ms is not None:
                    fitted = self._fit_audio_to_slot(audio_file, int(target_ms))
                    if fitted:
                        tmp_files.append(fitted)
                        audio_file = fitted
                        audio_dur_ms = int(target_ms)

            srt_start_sample = max(0, int(adj_start_ms / 1000.0 * sample_rate))
            start_sample = self._fragment_start_with_overlap(
                overlap_mode, frag.get('speaker'), srt_start_sample, cursors
            )

            try:
                frag_audio, frag_sr = sf.read(audio_file, dtype="float32", always_2d=False)
                if frag_audio.ndim > 1:
                    frag_audio = frag_audio.mean(axis=1)
                frag_max = float(np.abs(frag_audio).max()) if len(frag_audio) > 0 else 0.0
                if frag_max == 0.0:
                    logger.warning(f"Fragment {audio_file} has zero amplitude, skipping")
                    continue
                frag_audio = (frag_audio / frag_max * 0.92).astype(np.float32)
                waveform = frag_audio
                if frag_sr != sample_rate:
                    waveform, _ = self.main._preprocessor._resample(waveform, frag_sr, sample_rate)
            except Exception as e:
                logger.warning(f"Cannot load {audio_file}: {e}")
                continue

            if waveform.shape[0] == 0:
                continue

            required_samples = start_sample + waveform.shape[0]
            if required_samples > track.shape[0]:
                grown = np.zeros(required_samples, dtype=np.float32)
                grown[:track.shape[0]] = track
                track = grown
                logger.info(f"Lektor track buffer extended to {required_samples / sample_rate:.1f}s")

            track[start_sample:required_samples] += waveform
            self._advance_overlap_cursor(overlap_mode, frag.get('speaker'), required_samples, cursors)

        for tmp in tmp_files:
            try:
                os.remove(tmp)
            except Exception:
                pass

        if overlap_mode != "never":
            over = np.abs(track) > 0.92
            if over.any():
                excess = np.abs(track[over]) - 0.92
                track[over] = np.sign(track[over]) * (0.92 + 0.08 * np.tanh(excess / 0.08))

        max_val = float(np.abs(track).max())
        logger.info(f"Lektor track peak amplitude before final normalize: {max_val:.6f}")
        if max_val == 0.0:
            logger.warning("_build_lektor_audio_track: track is completely silent — no audio was placed")
            return False
        track = track / max_val * 0.92

        sf.write(output_path, track, sample_rate, subtype="PCM_16")
        logger.info(f"Lektor track saved: {output_path} ({len(done_frags)} frags, {len(track) / sample_rate:.1f}s)")
        return True

    def _fit_audio_to_slot(self, audio_path: str, slot_ms: int) -> Optional[str]:
        dur = _get_wav_duration(audio_path)
        if dur is None:
            return None
        slot_s = slot_ms / 1000.0
        if dur <= slot_s:
            return None
        ratio = min(dur / slot_s, 4.0)
        filters = []
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

    def _export_subtitles(self):
        if not self._fragments:
            QMessageBox.warning(self.main, "No SRT loaded", "Load an SRT or TXT file first.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Export subtitles")
        dlg.setStyleSheet(self.main.styleSheet())
        dlg.resize(420, 0)

        lay = QVBoxLayout(dlg)
        lay.setSpacing(10)
        lay.setContentsMargins(16, 16, 16, 12)

        title_lbl = QLabel("Export subtitles (.srt)")
        title_lbl.setStyleSheet(f"color:{C['text']};font-size:13px;font-weight:bold;")
        lay.addWidget(title_lbl)

        scope_box = QGroupBox("Fragments to include")
        scope_lay = QVBoxLayout(scope_box)
        rb_all = QRadioButton("All fragments (including not yet synthesized)")
        rb_done = QRadioButton("Only synthesized fragments (status: done)")
        rb_all.setChecked(True)
        scope_lay.addWidget(rb_all)
        scope_lay.addWidget(rb_done)
        lay.addWidget(scope_box)

        offset_val = self._offset_spin.value() if self._offset_spin is not None else 0
        offset_check = QCheckBox(f"Apply Lektor timing offset ({offset_val} ms)")
        offset_check.setChecked(False)
        offset_check.setToolTip(
            "If enabled, subtitle timestamps are shifted by the same offset\n"
            "used when building the lektor audio track, so subtitles stay in\n"
            "sync with the dubbed voice instead of the original SRT timing."
        )
        lay.addWidget(offset_check)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("Export")
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        only_done = rb_done.isChecked()
        apply_offset = offset_check.isChecked()

        if only_done:
            frags = [
                f for f in self._fragments
                if f.get('status') == 'done'
                and f.get('output_path')
                and os.path.exists(f.get('output_path', ''))
                and f.get('start_ms') is not None
            ]
            if not frags:
                QMessageBox.warning(self.main, "No audio", "No fragments have been synthesized yet.")
                return
        else:
            frags = list(self._fragments)

        frags = sorted(frags, key=lambda f: f.get('start_ms', 0) or 0)

        offset_ms = self._offset_spin.value() if (apply_offset and self._offset_spin is not None) else 0

        base_name = Path(self._srt_path).stem if self._srt_path else "subtitles"
        default_path = str(
            Path(_get_last_dir("output", self._output_dir)) / f"{base_name}.srt"
        )

        path, _ = QFileDialog.getSaveFileName(
            self.main,
            "Save subtitles",
            default_path,
            "SubRip subtitles (*.srt);;All files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".srt"):
            path += ".srt"

        _set_last_dir("output", path)

        try:
            lines = []
            for i, frag in enumerate(frags, start=1):
                start_ms = max(0, (frag.get('start_ms', 0) or 0) + offset_ms)
                end_ms = max(start_ms, (frag.get('srt_end_ms', frag.get('end_ms', 0)) or 0) + offset_ms)
                text = (frag.get('text') or '').strip()
                lines.append(str(i))
                lines.append(
                    f"{self._ms_to_srt_timestamp(start_ms)} --> {self._ms_to_srt_timestamp(end_ms)}"
                )
                lines.append(text if text else " ")
                lines.append("")

            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines))

            self.main._set_status(f"Subtitles saved: {path}", C["success"])

            reply = QMessageBox.information(
                self.main, "Export complete",
                f"Subtitles saved successfully:\n{path}\n\nOpen output folder?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                _open_file(os.path.dirname(path))

        except Exception:
            self.main._on_error("Subtitles export error", traceback.format_exc())

    def _ms_to_srt_timestamp(self, ms: float) -> str:
        total_ms = max(0, int(round(ms)))
        hours, rem = divmod(total_ms, 3600000)
        minutes, rem = divmod(rem, 60000)
        seconds, millis = divmod(rem, 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

    def _export_lektor_video(self):
        if not self.main._ffmpeg_ok:
            QMessageBox.warning(self.main, "ffmpeg not found", "Video export requires ffmpeg installed in PATH.")
            return

        if self._lektor_export_thread and self._lektor_export_thread.isRunning():
            QMessageBox.information(self.main, "Export in progress", "Video export is already running.")
            return

        external_audio_path = self._audio_path_edit.text().strip()
        if external_audio_path and not os.path.exists(external_audio_path):
            QMessageBox.warning(self.main, "External audio not found", "The selected external audio file no longer exists.")
            return

        if (self._w_vocal_suppress is not None and self._w_vocal_suppress.isRunning()):
            QMessageBox.information(self.main, "Processing in progress", "Vocal separation is already running.")
            return

        if self._dubbing_mode and self._dubbing_video_path:
            video_path = self._dubbing_video_path
        elif getattr(self.main, '_video_source_path', None) and os.path.exists(self.main._video_source_path or ''):
            video_path = self.main._video_source_path
        else:
            video_path = self._vid_path_edit.text().strip()

        if not video_path or not os.path.exists(video_path):
            QMessageBox.warning(self.main, "No video file", "Select a video file in the 'Video:' field.")
            return

        done_count = sum(1 for f in self._fragments if f.get("status") == "done" and f.get("start_ms") is not None)
        if done_count == 0:
            QMessageBox.warning(self.main, "No audio", "No SRT fragment has generated audio yet.\nRun synthesis before exporting.")
            return

        os.makedirs(self._output_dir, exist_ok=True)
        fmt = self._lektor_vid_fmt_combo.currentData()
        video_ext = os.path.splitext(video_path)[1] if fmt == "auto" else f".{fmt}"

        default_out = os.path.join(_get_last_dir("output", self._output_dir), f"lektor_output{video_ext}")
        out_path, _ = QFileDialog.getSaveFileName(self.main, "Save video with lektor", default_out, f"Video (*{video_ext});;All files (*)")
        if not out_path:
            return

        _set_last_dir("output", out_path)
        sample_rate = 44100
        offset_ms = self._offset_spin.value()
        lektor_wav = os.path.join(self._output_dir, "_lektor_track_tmp.wav")
        lektor_vol = self._lektor_vol.value() / 100.0
        orig_vol = self._orig_vol.value() / 100.0
        use_ducking = self._duck_check.isChecked()
        keep_original_track = self._keep_original_track_check.isChecked()
        if keep_original_track:
            dubbed_lang = self._dubbed_lang_edit.text().strip().lower()
            if not re.fullmatch(r"[a-z]{3}", dubbed_lang):
                dubbed_lang = "und"
        else:
            dubbed_lang = "und"

        self.main._set_status(f"Building lektor track from {done_count} fragments…")

        ok = self._build_lektor_audio_track(lektor_wav, sample_rate, offset_ms)
        if not ok:
            QMessageBox.warning(self.main, "Lektor track build failed", "Could not build the lektor audio track from the synthesized fragments.")
            self.main._set_status("Failed to build lektor track.", C["error"])
            return

        if self._lektor_norm_check and self._lektor_norm_check.isChecked():
            norm_tmp = lektor_wav.replace(".wav", "_norm.wav")
            if self._normalize_ffmpeg(lektor_wav, norm_tmp):
                os.replace(norm_tmp, lektor_wav)

        has_video_audio = bool(external_audio_path) or self._video_has_audio_stream(video_path)

        if self._vocal_suppress_check.isChecked() and has_video_audio:
            self._pending_export = {
                "video_path": video_path, "lektor_wav": lektor_wav, "out_path": out_path,
                "has_video_audio": has_video_audio, "lektor_vol": lektor_vol, "orig_vol": orig_vol,
                "use_ducking": use_ducking, "vocal_suppress_vol": self._vocal_suppress_spin.value() / 100.0,
                "keep_original_track": keep_original_track, "dubbed_lang": dubbed_lang,
                "external_audio_path": external_audio_path
            }
            self._export_btn.setEnabled(False)
            self._progress.setVisible(True)
            self._progress.setRange(0, 0)

            self._w_vocal_suppress = VocalSuppressWorker(
                video_path, self._output_dir,
                audio_path=external_audio_path if external_audio_path else None
            )
            self._w_vocal_suppress.status.connect(lambda m: self.main._set_status(m))
            self._w_vocal_suppress.finished.connect(self._on_vocal_suppress_done)
            self._w_vocal_suppress.error.connect(
                lambda e: self.main._on_error("Vocal suppression error", e,
                                               reset_fn=lambda: (
                                                   self._progress.setVisible(False),
                                                   self._export_btn.setEnabled(self.main._ffmpeg_ok),
                                                   self._pending_export.clear(),
                                               ))
            )
            self._w_vocal_suppress.start()
        else:
            self._do_lektor_ffmpeg_export(
                video_path, lektor_wav, out_path, has_video_audio, lektor_vol, orig_vol,
                use_ducking, keep_original_track=keep_original_track, dubbed_lang=dubbed_lang,
                external_audio_path=external_audio_path
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
            dubbed_lang: str = "und",
            external_audio_path: Optional[str] = None,
    ):
        use_vocal_suppress = vocals_wav is not None and no_vocals_wav is not None
        extra_tmp = [p for p in [vocals_wav, no_vocals_wav] if p]

        primary_audio = external_audio_path if external_audio_path else video_path

        cmd = ["ffmpeg", "-y"]

        if has_video_audio:
            if use_vocal_suppress:
                cmd.extend(["-i", video_path])
                cmd.extend(["-i", no_vocals_wav])
                cmd.extend(["-i", vocals_wav])
                cmd.extend(["-i", lektor_wav])

                if use_ducking:
                    audio_filter = (
                        f"[1:a]volume={orig_vol:.2f}[bg];"
                        f"[2:a]volume={vocal_suppress_vol:.2f}[vox];"
                        f"[bg][vox]amix=inputs=2:duration=longest:normalize=0[orig_mix];"
                        f"[3:a]volume={lektor_vol:.2f},asplit=2[lekt1][lekt2];"
                        f"[orig_mix][lekt1]sidechaincompress="
                        f"threshold=0.025:ratio=4:attack=10:release=400[ducked];"
                        f"[ducked][lekt2]amix=inputs=2:duration=longest:normalize=0[aout]"
                    )
                else:
                    audio_filter = (
                        f"[1:a]volume={orig_vol:.2f}[bg];"
                        f"[2:a]volume={vocal_suppress_vol:.2f}[vox];"
                        f"[3:a]volume={lektor_vol:.2f}[lekt];"
                        f"[bg][vox][lekt]amix=inputs=3:duration=longest:normalize=0[aout]"
                    )

                cmd.extend(["-filter_complex", audio_filter])
                cmd.extend(["-map", "0:v:0"])

                if keep_original_track:
                    cmd.extend(["-i", primary_audio])
                    cmd.extend(["-map", "4:a:0"])
                    cmd.extend(["-map", "[aout]"])
                    cmd.extend([
                        "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k",
                        "-metadata:s:a:0", "title=Original",
                        "-metadata:s:a:1", "title=Dubbing",
                        "-metadata:s:a:1", f"language={dubbed_lang}",
                        "-disposition:a:0", "default",
                        "-disposition:a:1", "0"
                    ])
                else:
                    cmd.extend(["-map", "[aout]"])
                    cmd.extend([
                        "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k"
                    ])

            else:
                cmd.extend(["-i", video_path])
                cmd.extend(["-i", primary_audio])
                cmd.extend(["-i", lektor_wav])

                if use_ducking:
                    audio_filter = (
                        f"[2:a]volume={lektor_vol:.2f},asplit=2[lekt1][lekt2];"
                        f"[1:a]volume={orig_vol:.2f}[orig];"
                        f"[orig][lekt1]sidechaincompress="
                        f"threshold=0.025:ratio=4:attack=10:release=400[ducked];"
                        f"[ducked][lekt2]amix=inputs=2:duration=longest:normalize=0[aout]"
                    )
                else:
                    audio_filter = (
                        f"[1:a]volume={orig_vol:.2f}[orig];"
                        f"[2:a]volume={lektor_vol:.2f}[lekt];"
                        f"[orig][lekt]amix=inputs=2:duration=longest:normalize=0[aout]"
                    )

                cmd.extend(["-filter_complex", audio_filter])
                cmd.extend(["-map", "0:v:0"])

                if keep_original_track:
                    cmd.extend(["-map", "1:a:0"])
                    cmd.extend(["-map", "[aout]"])
                    cmd.extend([
                        "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k",
                        "-metadata:s:a:0", "title=Original",
                        "-metadata:s:a:1", "title=Dubbing",
                        "-metadata:s:a:1", f"language={dubbed_lang}",
                        "-disposition:a:0", "default",
                        "-disposition:a:1", "0"
                    ])
                else:
                    cmd.extend(["-map", "[aout]"])
                    cmd.extend([
                        "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k"
                    ])
        else:
            cmd.extend(["-i", video_path])
            cmd.extend(["-i", lektor_wav])
            cmd.extend([
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-af", f"volume={lektor_vol:.2f}"
            ])

        cmd.append(out_path)

        logger.info(f"ffmpeg cmd: {' '.join(cmd)}")
        self.main._set_status("Exporting video with lektor track…")
        self._lektor_status.setText("Exporting video…")
        self._lektor_status.setStyleSheet(f"color:{C['warning']};font-size:10px;")
        self._export_btn.setEnabled(False)

        self._lektor_export_thread = LektorExportThread(cmd, lektor_wav, extra_tmp)
        self._lektor_export_thread.progress.connect(
            lambda m: self.main._set_status(m)
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
            video_path=p["video_path"],
            lektor_wav=p["lektor_wav"],
            out_path=p["out_path"],
            has_video_audio=p["has_video_audio"],
            lektor_vol=p["lektor_vol"],
            orig_vol=p["orig_vol"],
            use_ducking=p["use_ducking"],
            vocals_wav=vocals_path,
            no_vocals_wav=no_vocals_path,
            vocal_suppress_vol=p["vocal_suppress_vol"],
            keep_original_track=p.get("keep_original_track", False),
            dubbed_lang=p.get("dubbed_lang", "und"),
            external_audio_path=p.get("external_audio_path", None),
        )

    def _on_lektor_export_finished(self, success: bool, error_msg: str, out_path: str):
        if self._lektor_export_thread:
            self._lektor_export_thread.wait()
            self._lektor_export_thread = None

        self._export_btn.setEnabled(self.main._ffmpeg_ok)

        if success:
            self._lektor_status.setText(f"✓  Saved: {Path(out_path).name}")
            self._lektor_status.setStyleSheet(f"color:{C['success']};font-size:10px;")
            self.main._set_status(f"Video with lektor saved: {out_path}", C["success"])
            reply = QMessageBox.information(
                self.main, "Export complete",
                f"Video with lektor saved:\n{out_path}\n\nOpen output folder?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                _open_file(os.path.dirname(out_path))
        else:
            self._lektor_status.setText("✗  Export failed")
            self._lektor_status.setStyleSheet(f"color:{C['error']};font-size:10px;")
            self.main._set_status("Video export failed.", C["error"])
            QMessageBox.critical(self.main, "ffmpeg error",
                                 f"Export failed:\n\n{error_msg}")
        logger.info(f"Lektor export finished: success={success}")

    def _normalize_ffmpeg(self, input_path: str, output_path: str) -> bool:
        try:
            cmd_detect = [
                "ffmpeg", "-y", "-i", input_path,
                "-af", "volumedetect", "-f", "null", "-"
            ]
            result = subprocess.run(cmd_detect, capture_output=True, timeout=300)
            stderr = result.stderr.decode(errors="replace")
            match = re.search(r"max_volume:\s+(-?\d+\.?\d*)\s+dB", stderr)
            if not match:
                shutil.copy2(input_path, output_path)
                return True
            max_vol_db = float(match.group(1))
            target_db = -1.0
            gain_db = target_db - max_vol_db
            if abs(gain_db) < 0.1:
                shutil.copy2(input_path, output_path)
                return True
            cmd_gain = [
                "ffmpeg", "-y", "-i", input_path,
                "-af", f"volume={gain_db:.2f}dB",
                output_path
            ]
            r2 = subprocess.run(cmd_gain, capture_output=True, timeout=300)
            return r2.returncode == 0
        except Exception:
            return False

    @staticmethod
    def _video_has_audio_stream(video_path: str) -> bool:
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", "-select_streams", "a", video_path
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            if result.returncode != 0:
                return False
            data = json.loads(result.stdout.decode(errors="replace"))
            return bool(data.get("streams", []))
        except Exception:
            return False

    def _browse_video_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self.main, "Select video file", _get_last_dir("video"),
            "Video (*.mp4 *.mkv *.avi *.mov *.webm *.wmv);;All files (*)",
        )
        if path:
            _set_last_dir("video", path)
            self._vid_path_edit.setText(path)
            self._update_video_info()

    def _browse_audio_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self.main, "Select external audio file", _get_last_dir("audio"),
            "Audio files (*.wav *.mp3 *.flac *.ogg *.m4a);;All files (*)"
        )
        if path:
            _set_last_dir("audio", path)
            self._audio_path_edit.setText(path)
            self._update_audio_info()

    # ------------------------------------------------------------------
    # Session Management (SRT)
    # ------------------------------------------------------------------

    def _save_session(self):
        if not self._srt_path:
            return
        user_path = self._user_session_path()
        if user_path is not None:
            os.makedirs(os.path.dirname(str(user_path)), exist_ok=True)
            default_path = str(user_path)
        else:
            default_path = str(
                Path(_get_last_dir("session", str(OUTPUTS_DIR))) / "session.json"
            )
        path, _ = QFileDialog.getSaveFileName(
            self.main, "Save session", default_path,
            "JSON (*.json);;All files (*)",
        )
        if not path:
            return
        _set_last_dir("session", path)
        if self._write_manual_session_to(path):
            self.main._set_status(f"Session saved: {path}", C["success"])
        else:
            QMessageBox.critical(
                self.main, "Save error",
                f"Could not write session to:\n{path}\n\n"
                "A manual session cannot be saved into the autosession folder."
            )

    def _load_session(self):
        path, _ = QFileDialog.getOpenFileName(
            self.main, "Load session", _get_last_dir("session", str(OUTPUTS_DIR)),
            "JSON (*.json);;All files (*)",
        )
        if not path:
            return
        _set_last_dir("session", path)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            QMessageBox.critical(self.main, "Load error", str(e))
            return

        self.main.undo_manager.clear()
        if data.get("session_type") == "ebook":
            self.main.ebook_tab._restore_ebook_session_data(data)
        else:
            self._reset_dubbing_state()
            self._restore_session_data(data)

    @staticmethod
    def _session_dirs(srt_path: str) -> Tuple[Path, Path]:
        stem = Path(srt_path).stem
        root = Path(OUTPUTS_DIR) / stem
        return root / stem, root / f"{stem}_autosession"

    def _manual_session_dir(self) -> Optional[Path]:
        if not self._srt_path:
            return None
        return self._session_dirs(self._srt_path)[0]

    def _auto_session_dir(self) -> Optional[Path]:
        if not self._srt_path:
            return None
        return self._session_dirs(self._srt_path)[1]

    def _auto_session_path(self) -> Optional[Path]:
        auto_dir = self._auto_session_dir()
        if auto_dir is None:
            return None
        return auto_dir / "autosession.json"

    def _user_session_path(self) -> Optional[Path]:
        manual_dir = self._manual_session_dir()
        if manual_dir is None:
            return None
        return manual_dir / "session.json"

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

        is_supertonic = self.main._is_supertonic_active()
        is_piper = self.main._is_piper_active()
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
                drop = sv.get("drop")
                ref_text_wdg = sv.get("ref_text")
                demucs_chk = sv.get("demucs_chk")
                audio_path = drop.file_path if drop else None
                speaker_voices_data[spk] = {
                    "audio_path": audio_path,
                    "audio_hash": _ref_audio_hash(audio_path),
                    "ref_text": ref_text_wdg.toPlainText() if ref_text_wdg else "",
                    "demucs": demucs_chk.isChecked() if demucs_chk else False,
                }

        ref_audio = self.main._drop.file_path
        return {
            "version": 3,
            "srt_path": self._srt_path,
            "output_dir": self._output_dir,
            "reference_audio": ref_audio,
            "reference_audio_hash": _ref_audio_hash(ref_audio),
            "reference_text": self.main._ref_text.toPlainText(),
            "generation_params": self.main._get_generation_settings(),
            "normalize_audio": self._norm_check.isChecked(),
            "target_sr": self.main._sr_combo.currentData(),
            "whisper_size": self.main._w_size.currentData(),
            "whisper_lang": self.main._w_lang.currentData(),
            "lektor": {
                "video_path": self._vid_path_edit.text().strip() or None,
                "audio_path": self._audio_path_edit.text().strip() or None,
                "offset_ms": self._offset_spin.value(),
                "lektor_vol": self._lektor_vol.value(),
                "orig_vol": self._orig_vol.value(),
                "autofit": self._autofit_check.isChecked(),
                "atempo_threshold": self._atempo_threshold.value(),
                "ducking": self._duck_check.isChecked(),
                "vocal_suppress": self._vocal_suppress_check.isChecked(),
                "vocal_suppress_vol": self._vocal_suppress_spin.value(),
                "keep_original_track": self._keep_original_track_check.isChecked(),
                "dubbed_lang": self._dubbed_lang_edit.text().strip(),
                "video_format": self._lektor_vid_fmt_combo.currentData(),
                "overlap_mode": self._get_overlap_mode(),
            },
            "dubbing_mode": self._dubbing_mode,
            "dubbing_video_path": self._dubbing_video_path,
            "speaker_list": sorted_speaker_list,
            "speaker_voices": speaker_voices_data,
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

    def _write_manual_session_to(self, path: str) -> bool:
        def _norm(p: str) -> str:
            return os.path.normcase(os.path.abspath(p))

        try:
            target_dir = os.path.dirname(os.path.abspath(path))
            auto_dir = self._auto_session_dir()
            if auto_dir is not None and _norm(str(auto_dir)) == _norm(target_dir):
                logger.warning("Manual session cannot be written into the autosession folder")
                return False
            os.makedirs(target_dir, exist_ok=True)

            previous: Set[str] = set()
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        old_data = json.load(fh)
                    for old_frag in old_data.get("fragments", []):
                        old_path = old_frag.get("output_path")
                        if old_path and os.path.dirname(_norm(old_path)) == _norm(target_dir):
                            previous.add(_norm(old_path))
                except Exception:
                    previous = set()

            data = self._build_session_data()
            mapping: Dict[str, str] = {}
            used: Set[str] = set()
            for frag in data["fragments"]:
                src = frag.get("output_path")
                if not src or not os.path.isfile(src):
                    continue
                key = _norm(src)
                if key not in mapping:
                    dst = os.path.join(target_dir, os.path.basename(src))
                    root, ext = os.path.splitext(dst)
                    n = 1
                    while _norm(dst) in used:
                        dst = f"{root}_{n}{ext}"
                        n += 1
                    if _norm(dst) != key:
                        shutil.copy2(src, dst)
                    used.add(_norm(dst))
                    mapping[key] = dst
                frag["output_path"] = mapping[key]
            data["output_dir"] = target_dir

            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)

            prefix = Path(self._srt_path).stem if self._srt_path else ""
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}
            delete_targets: Set[str] = set(previous)
            if prefix:
                try:
                    for entry in os.scandir(target_dir):
                        if not entry.is_file():
                            continue
                        ext = os.path.splitext(entry.name)[1].lower()
                        if ext not in audio_exts:
                            continue
                        name = entry.name
                        if (
                            name.startswith(prefix + "_")
                            or name.startswith("__merge_tmp_")
                            or name.startswith("_lektor_track_tmp")
                        ):
                            delete_targets.add(_norm(entry.path))
                except OSError:
                    pass

            for stale in delete_targets - used:
                try:
                    os.remove(stale)
                except OSError:
                    pass
            return True
        except Exception as e:
            logger.warning(f"Manual session write failed: {e}")
            return False

    def _recover_missing_audio(self, data: dict) -> dict:
        srt_path = data.get("srt_path") or ""
        if not srt_path:
            return data

        def _norm(p: str) -> str:
            return os.path.normcase(os.path.abspath(p))

        stem = Path(srt_path).stem
        candidate_dirs: List[str] = []
        seen: Set[str] = set()
        for raw in (
            str(Path(OUTPUTS_DIR) / stem / stem),
            str(Path(OUTPUTS_DIR) / stem / f"{stem}_autosession"),
            data.get("output_dir") or "",
        ):
            if not raw:
                continue
            ad = os.path.abspath(raw)
            if not os.path.isdir(ad):
                continue
            key = _norm(ad)
            if key in seen:
                continue
            seen.add(key)
            candidate_dirs.append(ad)

        if not candidate_dirs:
            return data

        fragments = [dict(f) for f in data.get("fragments", [])]
        changed = False
        for frag in fragments:
            existing = frag.get("output_path")
            if existing:
                try:
                    if os.path.isfile(existing) and os.path.getsize(existing) > 0:
                        continue
                except OSError:
                    pass
            idx = frag.get("index")
            if idx is None:
                continue
            try:
                idx_int = int(idx)
            except (TypeError, ValueError):
                continue
            expected_name = f"{stem}_{idx_int + 1:03d}.wav"
            for d in candidate_dirs:
                candidate = os.path.join(d, expected_name)
                try:
                    if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
                        frag["output_path"] = candidate
                        frag["status"] = "done"
                        frag["error_msg"] = None
                        changed = True
                        logger.info(
                            f"Recovered missing audio for fragment #{idx_int + 1}: {candidate}"
                        )
                        break
                except OSError:
                    continue

        if not changed:
            return data

        result = dict(data)
        result["fragments"] = fragments
        return result

    def _relocate_session_data(self, data: dict) -> dict:
        srt_path = data.get("srt_path") or ""
        if not srt_path:
            return data
        data = self._recover_missing_audio(data)

        def _norm(p: str) -> str:
            return os.path.normcase(os.path.abspath(p))

        auto_dir = os.path.abspath(str(self._session_dirs(srt_path)[1]))
        auto_key = _norm(auto_dir)
        fragments = [dict(f) for f in data.get("fragments", [])]

        sources = [
            f["output_path"] for f in fragments
            if f.get("output_path") and os.path.isfile(f["output_path"])
        ]
        if any(os.path.dirname(_norm(s)) != auto_key for s in sources):
            self._release_audio_file_locks()

        os.makedirs(auto_dir, exist_ok=True)
        used: Set[str] = {_norm(s) for s in sources if os.path.dirname(_norm(s)) == auto_key}
        mapping: Dict[str, Optional[str]] = {}
        for frag in fragments:
            src = frag.get("output_path")
            if not src or not os.path.isfile(src):
                continue
            key = _norm(src)
            if key not in mapping:
                if os.path.dirname(key) == auto_key:
                    mapping[key] = src
                else:
                    dst = os.path.join(auto_dir, os.path.basename(src))
                    root, ext = os.path.splitext(dst)
                    n = 1
                    while _norm(dst) in used:
                        dst = f"{root}_{n}{ext}"
                        n += 1
                    try:
                        shutil.copy2(src, dst)
                        used.add(_norm(dst))
                        mapping[key] = dst
                    except OSError as e:
                        logger.warning(f"Could not copy {src} to {dst}: {e}")
                        mapping[key] = None
            frag["output_path"] = mapping[key]
            if mapping[key] is None:
                frag["status"] = "waiting"

        result = dict(data)
        result["fragments"] = fragments
        result["output_dir"] = auto_dir
        return result

    def _restore_session_data(self, data: dict):
        self._apply_session_data(self._relocate_session_data(data))

    def _apply_session_data(self, data: dict):
        self._srt_path = data.get("srt_path", "")
        self._output_dir = data.get("output_dir", str(OUTPUTS_DIR))
        self._fragments = data.get("fragments", [])

        for f in self._fragments:
            start_ms = f.get("start_ms") or 0
            end_ms = f.get("end_ms") or 0
            f["timestamp"] = f"{_ms_to_ts(start_ms)} --> {_ms_to_ts(end_ms)}"

        missing = 0
        for f in self._fragments:
            if (f.get("status") == "done"
                    and f.get("output_path")
                    and not os.path.exists(f.get("output_path", ""))):
                f["status"] = "waiting"
                f["output_path"] = None
                missing += 1

        ref_audio = data.get("reference_audio")
        saved_hash = data.get("reference_audio_hash")
        if ref_audio and os.path.exists(ref_audio):
            current_hash = _ref_audio_hash(ref_audio)
            if saved_hash and current_hash and current_hash != saved_hash:
                QMessageBox.warning(
                    self.main, "Reference audio changed",
                    f"The reference audio file has changed since the session was saved:\n"
                    f"{ref_audio}\n\nFragments marked as done may not match the current voice.",
                )
            self.main._drop._set(ref_audio)
            self.main._ref_player.load(ref_audio)
        elif ref_audio:
            self.main._set_status(
                f"Reference audio not found: {Path(ref_audio).name}", C["warning"]
            )

        self.main._ref_text.setPlainText(data.get("reference_text", ""))

        params = data.get("generation_params", {})
        for key, value in params.items():
            widget = self.main._param_widgets.get(key) if hasattr(self.main, "_param_widgets") else None
            if widget is None:
                continue
            if isinstance(widget, QSlider):
                widget.setValue(int(value * 100))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))

        self._norm_check.setChecked(data.get("normalize_audio", False))

        target_sr = data.get("target_sr")
        if target_sr is not None:
            for i in range(self.main._sr_combo.count()):
                if self.main._sr_combo.itemData(i) == target_sr:
                    self.main._sr_combo.setCurrentIndex(i)
                    break

        whisper_size = data.get("whisper_size")
        if whisper_size:
            for i in range(self.main._w_size.count()):
                if self.main._w_size.itemData(i) == whisper_size:
                    self.main._w_size.setCurrentIndex(i)
                    break

        whisper_lang = data.get("whisper_lang")
        if whisper_lang:
            for i in range(self.main._w_lang.count()):
                if self.main._w_lang.itemData(i) == whisper_lang:
                    self.main._w_lang.setCurrentIndex(i)
                    break

        lektor = data.get("lektor", {})
        vid_path = lektor.get("video_path") or ""
        self._vid_path_edit.setText(vid_path)
        self._update_video_info()
        ext_audio_path = lektor.get("audio_path") or ""
        self._audio_path_edit.setText(ext_audio_path)
        self._update_audio_info()
        self._offset_spin.setValue(lektor.get("offset_ms", 0))
        self._lektor_vol.setValue(lektor.get("lektor_vol", 100))
        self._orig_vol.setValue(lektor.get("orig_vol", 100))
        self._autofit_check.setChecked(lektor.get("autofit", False))
        self._atempo_threshold.setValue(lektor.get("atempo_threshold", 300))
        self._duck_check.setChecked(lektor.get("ducking", False))
        self._vocal_suppress_check.setChecked(lektor.get("vocal_suppress", False))
        self._vocal_suppress_spin.setValue(lektor.get("vocal_suppress_vol", 20))
        self._vocal_suppress_spin.setEnabled(
            lektor.get("vocal_suppress", False) and self.main._ffmpeg_ok
        )
        self._keep_original_track_check.setChecked(lektor.get("keep_original_track", False))
        self._dubbed_lang_edit.setText(lektor.get("dubbed_lang", ""))
        video_format = lektor.get("video_format")
        if video_format is not None:
            for i in range(self._lektor_vid_fmt_combo.count()):
                if self._lektor_vid_fmt_combo.itemData(i) == video_format:
                    self._lektor_vid_fmt_combo.setCurrentIndex(i)
                    break

        overlap_mode = lektor.get("overlap_mode", "never")
        for i in range(self._overlap_combo.count()):
            if self._overlap_combo.itemData(i) == overlap_mode:
                self._overlap_combo.setCurrentIndex(i)
                break

        dubbing_mode = data.get("dubbing_mode", False)
        if dubbing_mode:
            self._dubbing_mode = True
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
                self.main._whisper_backend is not None
                and self.main._whisper_backend.is_downloaded(self.main._w_size.currentData())
            )
            is_supertonic = self.main._is_supertonic_active()
            is_piper = self.main._is_piper_active()
            for spk, sv_data in speaker_voices_data.items():
                sv = self._speaker_voices.get(spk)
                if not sv:
                    continue
                if is_supertonic:
                    voice_name = sv_data.get("voice_name", "M1")
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
                        info = _audio_info_str(audio_path)
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
            if hasattr(self.main, "_whisper_section_widget"):
                self.main._whisper_section_widget.setVisible(False)
        else:
            self._update_dubbing_visibility()

        for f in self._fragments:
            ap = f.get("output_path")
            if ap and os.path.exists(ap):
                self._file_watcher.addPath(ap)

        fname = Path(self._srt_path).name if self._srt_path else "session"
        done = sum(1 for f in self._fragments if f.get("status") == "done")
        self._srt_label.setText(
            f"📄  {fname}  •  {len(self._fragments)} fragments"
        )
        self._srt_label.setStyleSheet(
            f"color:{C['accent']};font-size:11px;font-weight:600;"
        )
        self._btn_close_srt.setEnabled(True)
        self._btn_save_session.setEnabled(True)
        self.main._tabs.setCurrentIndex(0)
        self._populate_tree()
        self.main._update_action_buttons()

        status_msg = f"Session loaded: {len(self._fragments)} fragments, {done} already synthesized."
        if missing:
            status_msg = f"Session loaded — {missing} audio file(s) missing on disk, reset to waiting."
            self.main._set_status(status_msg, C["warning"])
        else:
            self.main._set_status(status_msg, C["success"])

    def build_lektor_controls(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self._vid_row_widget = QWidget()
        row = QHBoxLayout(self._vid_row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel("Video file:")
        lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        lbl.setFixedWidth(70)
        self._vid_path_edit = QLineEdit()
        self._vid_path_edit.setPlaceholderText("Select video file...")
        self._vid_path_edit.setReadOnly(True)
        browse_vid_btn = QPushButton("Browse...")
        browse_vid_btn.setFixedHeight(26)
        browse_vid_btn.clicked.connect(self._browse_video_file)
        row.addWidget(lbl)
        row.addWidget(self._vid_path_edit, 1)
        row.addWidget(browse_vid_btn)
        lay.addWidget(self._vid_row_widget)

        self._vid_info_lbl = QLabel("")
        self._vid_info_lbl.setStyleSheet(
            f"color:{C['text2']};font-size:11px;font-style:italic;"
        )
        self._vid_info_lbl.setWordWrap(True)
        lay.addWidget(self._vid_info_lbl)

        row = QHBoxLayout()
        lbl = QLabel("External audio:")
        lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        lbl.setFixedWidth(70)
        self._audio_path_edit = QLineEdit()
        self._audio_path_edit.setPlaceholderText("Optional external audio track...")
        self._audio_path_edit.setReadOnly(True)
        browse_audio_btn = QPushButton("Browse...")
        browse_audio_btn.setFixedHeight(26)
        browse_audio_btn.clicked.connect(self._browse_audio_file)
        row.addWidget(lbl)
        row.addWidget(self._audio_path_edit, 1)
        row.addWidget(browse_audio_btn)
        lay.addLayout(row)

        self._audio_info_lbl = QLabel("")
        self._audio_info_lbl.setStyleSheet(
            f"color:{C['text2']};font-size:11px;font-style:italic;"
        )
        self._audio_info_lbl.setWordWrap(True)
        lay.addWidget(self._audio_info_lbl)

        row = QHBoxLayout()
        lbl = QLabel("Offset (ms):")
        lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        lbl.setFixedWidth(70)
        self._offset_spin = QSpinBox()
        self._offset_spin.setRange(-10000, 10000)
        self._offset_spin.setSingleStep(100)
        self._offset_spin.setValue(0)
        self._offset_spin.setToolTip(
            "Global time shift of the entire dubbing track in milliseconds.\n"
            "Positive values delay the dubbing, negative values move it earlier.\n"
            "Example: -200 shifts all speech 200 ms earlier relative to the video."
        )
        row.addWidget(lbl)
        row.addWidget(self._offset_spin)
        row.addStretch()
        lay.addLayout(row)

        row = QHBoxLayout()
        lbl = QLabel("Overlap:")
        lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        lbl.setFixedWidth(70)
        self._overlap_combo = QComboBox()
        self._overlap_combo.addItem("Never", "never")
        self._overlap_combo.addItem("Different speakers only", "speakers")
        self._overlap_combo.addItem("Always", "always")
        self._overlap_combo.setToolTip(
            "Controls whether synthesized fragments may play at the same time.\n\n"
            "Never: a fragment is delayed until the previous one has finished (single narrator).\n"
            "Different speakers only: fragments of different speakers may overlap, "
            "but one speaker never overlaps themselves. Fragments without a speaker "
            "count as a single narrator.\n"
            "Always: every fragment starts exactly at its subtitle time, even if it "
            "overlaps another one.\n\n"
            "Applies to the exported video and to the dubbing simulation in the video preview."
        )
        row.addWidget(lbl)
        row.addWidget(self._overlap_combo, 1)
        lay.addLayout(row)

        vol_spin_style = (
            f"QSpinBox {{ color:{C['accent']}; font-size:11px; "
            f"border:1px solid {C['border']}; border-radius:3px; padding:1px 3px; }}"
        )

        row = QHBoxLayout()
        lbl = QLabel("Lektor volume:")
        lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        lbl.setFixedWidth(70)
        self._lektor_vol = QSlider(Qt.Orientation.Horizontal)
        self._lektor_vol.setRange(0, 200)
        self._lektor_vol.setValue(100)
        self._lektor_vol.setFixedWidth(120)
        self._lektor_vol_spin = QSpinBox()
        self._lektor_vol_spin.setRange(0, 200)
        self._lektor_vol_spin.setValue(100)
        self._lektor_vol_spin.setSuffix("%")
        self._lektor_vol_spin.setFixedWidth(64)
        self._lektor_vol_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self._lektor_vol_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lektor_vol_spin.setStyleSheet(vol_spin_style)
        self._lektor_vol_spin.setToolTip("Type the exact lektor volume in percent (0-200%).")
        self._lektor_vol.valueChanged.connect(self._lektor_vol_spin.setValue)
        self._lektor_vol_spin.valueChanged.connect(self._lektor_vol.setValue)
        self._lektor_vol.setToolTip(
            "Volume of the synthesized (lektor) speech track in the final video.\n"
            "100% = original level, 50% = half volume, 150% = louder.\n"
            "Increase if the lektor voice is too quiet compared to the original audio."
        )
        row.addWidget(lbl)
        row.addWidget(self._lektor_vol)
        row.addWidget(self._lektor_vol_spin)
        row.addStretch()
        lay.addLayout(row)

        row = QHBoxLayout()
        lbl = QLabel("Original volume:")
        lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        lbl.setFixedWidth(70)
        self._orig_vol = QSlider(Qt.Orientation.Horizontal)
        self._orig_vol.setRange(0, 200)
        self._orig_vol.setValue(100)
        self._orig_vol.setFixedWidth(120)
        self._orig_vol_spin = QSpinBox()
        self._orig_vol_spin.setRange(0, 200)
        self._orig_vol_spin.setValue(100)
        self._orig_vol_spin.setSuffix("%")
        self._orig_vol_spin.setFixedWidth(64)
        self._orig_vol_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self._orig_vol_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._orig_vol_spin.setStyleSheet(vol_spin_style)
        self._orig_vol_spin.setToolTip("Type the exact original volume in percent (0-200%).")
        self._orig_vol.valueChanged.connect(self._orig_vol_spin.setValue)
        self._orig_vol_spin.valueChanged.connect(self._orig_vol.setValue)
        self._orig_vol.setToolTip(
            "Volume of the original video audio track (music, effects, original dialogue).\n"
            "Lower this so the lektor is clearly audible over the background.\n"
            "Example: 30% keeps ambience while the lektor stays in front."
        )
        row.addWidget(lbl)
        row.addWidget(self._orig_vol)
        row.addWidget(self._orig_vol_spin)
        row.addStretch()
        lay.addLayout(row)

        self._autofit_check = QCheckBox("Autofit overshoot (atempo)")
        self._autofit_check.setToolTip(
            "If a synthesized fragment exceeds its time slot by less than the threshold below,\n"
            "it will be sped up using ffmpeg atempo to fit the slot.\n"
            "Works best for small overruns (up to ~2x speed). Pitch is preserved.\n"
            "Requires ffmpeg. Larger overruns are not corrected to preserve voice quality."
        )
        lay.addWidget(self._autofit_check)

        self._atempo_widget = QWidget()
        self._atempo_widget.setObjectName("atempoWrap")
        self._atempo_widget.setStyleSheet("QWidget#atempoWrap { background: transparent; }")
        atempo_outer = QVBoxLayout(self._atempo_widget)
        atempo_outer.setContentsMargins(0, 0, 0, 0)
        atempo_outer.setSpacing(4)

        atempo_row_wrap = QWidget()
        atempo_row_wrap.setStyleSheet("background: transparent;")
        atempo_row = QHBoxLayout(atempo_row_wrap)
        atempo_row.setContentsMargins(20, 0, 0, 0)
        atempo_row.setSpacing(8)
        atempo_lbl = QLabel("Atempo threshold (ms):")
        atempo_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        atempo_lbl.setFixedWidth(130)
        self._atempo_threshold = QSpinBox()
        self._atempo_threshold.setRange(50, 2000)
        self._atempo_threshold.setSingleStep(50)
        self._atempo_threshold.setValue(300)
        self._atempo_threshold.setMinimumWidth(90)
        self._atempo_threshold.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        self._atempo_threshold.setStyleSheet("""
            QSpinBox { padding-right: 14px; }
            QSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; width: 16px; height: 10px; }
            QSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; width: 16px; height: 10px; }
        """)
        atempo_row.addWidget(atempo_lbl)
        atempo_row.addWidget(self._atempo_threshold)
        atempo_row.addStretch()
        atempo_outer.addWidget(atempo_row_wrap)

        halve_row_wrap = QWidget()
        halve_row_wrap.setStyleSheet("background: transparent;")
        halve_row = QHBoxLayout(halve_row_wrap)
        halve_row.setContentsMargins(20, 0, 0, 0)
        halve_row.setSpacing(8)
        self._atempo_halve_check = QCheckBox("Halve overshoot above threshold")
        self._atempo_halve_check.setToolTip(
            "Fragments that exceed their slot by MORE than the threshold are sped up\n"
            "so that the remaining overshoot is half of the original one.\n"
            "Example: +0.8s becomes +0.4s, +1.0s becomes +0.5s.\n"
            "Fragments within the threshold are still fitted fully into their slot.\n"
            "Speed-up is limited to 4x."
        )
        halve_row.addWidget(self._atempo_halve_check)
        halve_row.addStretch()
        atempo_outer.addWidget(halve_row_wrap)

        lay.addWidget(self._atempo_widget)
        self._atempo_widget.setVisible(False)

        self._autofit_check.toggled.connect(lambda checked: self._atempo_widget.setVisible(checked))

        self._duck_check = QCheckBox("Ducking (sidechain compression)")
        lay.addWidget(self._duck_check)

        self._vocal_suppress_check = QCheckBox("Vocal suppression (remove original vocals)")
        lay.addWidget(self._vocal_suppress_check)

        self._vocal_suppress_widget = QWidget()
        self._vocal_suppress_widget.setObjectName("vocalSuppressWrap")
        self._vocal_suppress_widget.setStyleSheet("QWidget#vocalSuppressWrap { background: transparent; }")
        sup_row = QHBoxLayout(self._vocal_suppress_widget)
        sup_row.setContentsMargins(20, 0, 0, 0)
        sup_row.setSpacing(8)
        sup_lbl = QLabel("Suppress volume:")
        sup_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        sup_lbl.setFixedWidth(100)
        self._vocal_suppress_spin = QSpinBox()
        self._vocal_suppress_spin.setRange(0, 100)
        self._vocal_suppress_spin.setValue(20)
        self._vocal_suppress_spin.setSingleStep(5)
        self._vocal_suppress_spin.setSuffix("%")
        self._vocal_suppress_spin.setMinimumWidth(80)
        self._vocal_suppress_spin.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        self._vocal_suppress_spin.setStyleSheet("""
            QSpinBox { padding-right: 14px; }
            QSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; width: 16px; height: 10px; }
            QSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; width: 16px; height: 10px; }
        """)
        sup_row.addWidget(sup_lbl)
        sup_row.addWidget(self._vocal_suppress_spin)
        sup_row.addStretch()
        lay.addWidget(self._vocal_suppress_widget)
        self._vocal_suppress_widget.setVisible(False)
        self._vocal_suppress_check.toggled.connect(lambda checked: self._vocal_suppress_widget.setVisible(checked))

        self._lektor_norm_check = QCheckBox("Normalize audio (ffmpeg)")
        lay.addWidget(self._lektor_norm_check)

        self._keep_original_track_check = QCheckBox("Keep original track as separate audio stream")
        lay.addWidget(self._keep_original_track_check)

        self._dubbed_lang_widget = QWidget()
        self._dubbed_lang_widget.setStyleSheet("background: transparent;")
        dlang_row = QHBoxLayout(self._dubbed_lang_widget)
        dlang_row.setContentsMargins(20, 0, 0, 0)
        dlang_row.setSpacing(8)
        dlang_lbl = QLabel("Dubbed language (ISO 639-2):")
        dlang_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        dlang_lbl.setFixedWidth(160)
        self._dubbed_lang_edit = QLineEdit()
        self._dubbed_lang_edit.setPlaceholderText("e.g. eng")
        self._dubbed_lang_edit.setMaxLength(3)
        self._dubbed_lang_edit.setFixedWidth(70)
        self._dubbed_lang_edit.setFixedHeight(24)
        self._dubbed_lang_edit.setStyleSheet(f"QLineEdit {{ background: {C['surface']}; border: 1px solid {C['border']}; border-radius: 4px; color: {C['text']}; padding: 2px 6px; font-size: 11px; font-family: 'Consolas'; }}")
        dlang_row.addWidget(dlang_lbl)
        dlang_row.addWidget(self._dubbed_lang_edit)
        dlang_row.addStretch()
        lay.addWidget(self._dubbed_lang_widget)
        self._dubbed_lang_widget.setVisible(False)
        self._keep_original_track_check.toggled.connect(lambda checked: self._dubbed_lang_widget.setVisible(checked))

        row = QHBoxLayout()
        lbl = QLabel("Video format:")
        lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        lbl.setFixedWidth(70)
        self._lektor_vid_fmt_combo = QComboBox()
        self._lektor_vid_fmt_combo.addItem("Auto (same as input)", "auto")
        self._lektor_vid_fmt_combo.addItem("MP4", "mp4")
        self._lektor_vid_fmt_combo.addItem("MKV", "mkv")
        self._lektor_vid_fmt_combo.addItem("AVI", "avi")
        self._lektor_vid_fmt_combo.addItem("MOV", "mov")
        self._lektor_vid_fmt_combo.addItem("WebM", "webm")
        row.addWidget(lbl)
        row.addWidget(self._lektor_vid_fmt_combo, 1)
        lay.addLayout(row)

        exp_row = QHBoxLayout()
        self._export_btn = QPushButton("🎬  Export Lektor Video")
        self._export_btn.setStyleSheet(_btn(C["accent"]))
        self._export_btn.setEnabled(self.main._ffmpeg_ok)
        self._export_btn.clicked.connect(self._export_lektor_video)
        exp_row.addWidget(self._export_btn)

        self._subtitles_export_btn = QPushButton("📝  Export subtitles")
        self._subtitles_export_btn.setStyleSheet(_btn(C["accent"]))
        self._subtitles_export_btn.clicked.connect(self._export_subtitles)
        exp_row.addWidget(self._subtitles_export_btn)
        lay.addLayout(exp_row)

        self._lektor_status = QLabel("")
        self._lektor_status.setStyleSheet(f"color:{C['text3']};font-size:10px;font-style:italic;")
        lay.addWidget(self._lektor_status)

        return w

    def _sync_video_playback(self, is_playing: bool):
        if not getattr(self, '_media_player', None):
            return
        if is_playing:
            self._media_player.play()
            QTimer.singleShot(150, self._resync_dub_player)
        else:
            self._media_player.pause()
            if getattr(self, '_dub_player', None):
                self._dub_player.pause()
            self._pause_extra_dub_players()

    def _resync_dub_player(self):
        if not getattr(self, '_dub_player', None):
            return
        if self._dub_sync_held():
            return
        if not getattr(self, '_sim_dub_check', None) or not self._sim_dub_check.isChecked():
            return
        player = self.main._video_player
        if not player.has_media() or not player.is_playing():
            return

        pos_ms = player.position_seconds() * 1000.0
        schedule = self._build_dub_schedule()
        self._update_extra_dub_lanes(schedule, pos_ms, True, True)

        entry = None
        for candidate in schedule:
            if candidate['start_ms'] > pos_ms:
                break
            if candidate.get('lane', 0) != 0:
                continue
            if candidate['start_ms'] <= pos_ms < candidate['end_ms']:
                entry = candidate
                break
        if entry is None:
            return

        url = QUrl.fromLocalFile(entry['output_path'])
        offset_ms = int((pos_ms - entry['start_ms']) * entry['rate'])

        if (
            getattr(self, '_current_sim_frag_idx', None) != entry['index']
            or self._dub_player.source() != url
        ):
            self._current_sim_frag_idx = entry['index']
            self._dub_player.setSource(url)
            self._dub_player.setPlaybackRate(entry['rate'])
            self._dub_player.setPosition(offset_ms)
            self._dub_player.play()
            return

        state = self._dub_player.playbackState()
        if state == QMediaPlayer.PlaybackState.PlayingState:
            return
        if self._dub_player.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia:
            return
        if state == QMediaPlayer.PlaybackState.StoppedState:
            self._dub_player.setPosition(offset_ms)
        self._dub_player.setPlaybackRate(entry['rate'])
        self._dub_player.play()

    def _sync_video_position(self, pos_s: float):
        self._update_playing_highlight(int(pos_s * 1000))
        if not getattr(self, '_media_player', None):
            return
        target_ms = int(pos_s * 1000)
        if abs(self._media_player.position() - target_ms) > 300:
            self._media_player.setPosition(target_ms)

    def _build_dub_schedule(self) -> List[Dict]:
        offset_ms = self._offset_spin.value() if getattr(self, '_offset_spin', None) else 0
        autofit = bool(getattr(self, '_autofit_check', None) and self._autofit_check.isChecked())
        atempo_thresh = self._atempo_threshold.value() if getattr(self, '_atempo_threshold', None) else 300
        halve = bool(getattr(self, '_atempo_halve_check', None) and self._atempo_halve_check.isChecked())
        overlap_mode = self._get_overlap_mode()

        done_frags = [
            f for f in self._fragments
            if f.get('status') == 'done'
            and f.get('output_path')
            and f.get('start_ms') is not None
        ]
        done_frags.sort(key=lambda f: f.get('start_ms', 0) or 0)

        sig = (
            offset_ms, autofit, atempo_thresh, halve, overlap_mode,
            tuple(
                (f['index'], f['output_path'], f.get('start_ms', 0), f.get('end_ms', 0),
                 str(f.get('speaker') or '').strip())
                for f in done_frags
            ),
        )
        if getattr(self, '_dub_schedule_sig', None) == sig:
            return getattr(self, '_dub_schedule', [])

        schedule: List[Dict] = []
        cursors: Dict[str, float] = {}
        for frag in done_frags:
            if not os.path.exists(frag['output_path']):
                continue

            raw_start = frag.get('start_ms', 0) or 0
            raw_end = frag.get('end_ms', 0) or 0
            adj_start = raw_start + offset_ms
            adj_end = raw_end + offset_ms
            slot_ms = max(0, adj_end - adj_start)

            audio_dur_s = _get_wav_duration(frag['output_path']) or 0.0
            audio_dur_ms = audio_dur_s * 1000.0
            rate = 1.0
            played_ms = audio_dur_ms

            if autofit:
                target_ms = self._atempo_target_ms(slot_ms, audio_dur_ms)
                if target_ms is not None and target_ms > 0:
                    rate = min(audio_dur_ms / target_ms, 4.0)
                    played_ms = audio_dur_ms / rate

            start_ms = self._fragment_start_with_overlap(
                overlap_mode, frag.get('speaker'), adj_start, cursors
            )
            end_ms = start_ms + played_ms

            schedule.append({
                'index': frag['index'],
                'output_path': frag['output_path'],
                'start_ms': start_ms,
                'end_ms': end_ms,
                'rate': rate,
                'lane': 0,
            })
            self._advance_overlap_cursor(overlap_mode, frag.get('speaker'), end_ms, cursors)

        schedule.sort(key=lambda e: e['start_ms'])
        lane_ends: List[float] = []
        for entry in schedule:
            for lane, lane_end in enumerate(lane_ends):
                if lane_end <= entry['start_ms']:
                    entry['lane'] = lane
                    lane_ends[lane] = entry['end_ms']
                    break
            else:
                entry['lane'] = len(lane_ends)
                lane_ends.append(entry['end_ms'])

        self._dub_schedule = schedule
        self._dub_schedule_sig = sig
        return schedule

    def _on_video_position_changed(self, pos_ms: int):
        if self._dub_sync_held():
            return

        if not self.main._video_player.has_media():
            self._stop_extra_dub_players()
            return

        if not getattr(self, '_sim_dub_check', None) or not self._sim_dub_check.isChecked():
            if getattr(self, '_dub_player', None) and self._dub_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self._dub_player.stop()
            self._stop_extra_dub_players()
            self._current_sim_frag_idx = None
            self._last_sim_pos_ms = None
            return

        schedule = self._build_dub_schedule()

        last_pos = getattr(self, '_last_sim_pos_ms', None)
        is_seek = last_pos is None or abs(pos_ms - last_pos) > 700
        self._last_sim_pos_ms = pos_ms

        matching_entry = None
        for entry in schedule:
            if entry['start_ms'] > pos_ms:
                break
            if entry.get('lane', 0) != 0:
                continue
            if entry['start_ms'] <= pos_ms < entry['end_ms']:
                matching_entry = entry
                break

        is_video_playing = self.main._video_player.is_playing()

        if matching_entry:
            frag_idx = matching_entry['index']
            frag_offset_ms = (pos_ms - matching_entry['start_ms']) * matching_entry['rate']
            changed_fragment = getattr(self, '_current_sim_frag_idx', None) != frag_idx

            if changed_fragment:
                self._current_sim_frag_idx = frag_idx
                self._dub_player.setSource(
                    QUrl.fromLocalFile(matching_entry['output_path'])
                )
                self._dub_player.setPlaybackRate(matching_entry['rate'])
                self._dub_player.setPosition(int(frag_offset_ms))
                if is_video_playing:
                    self._dub_player.play()

            elif is_seek:
                self._dub_player.setPlaybackRate(matching_entry['rate'])
                self._dub_player.setPosition(int(frag_offset_ms))
                if is_video_playing:
                    self._dub_player.play()

            else:
                if self._dub_player.playbackRate() != matching_entry['rate']:
                    self._dub_player.setPlaybackRate(matching_entry['rate'])
                if (
                    is_video_playing
                    and self._dub_player.playbackState()
                    != QMediaPlayer.PlaybackState.PlayingState
                ):
                    if (
                        self._dub_player.mediaStatus()
                        != QMediaPlayer.MediaStatus.EndOfMedia
                    ):
                        self._dub_player.play()
                elif (
                    not is_video_playing
                    and self._dub_player.playbackState()
                    == QMediaPlayer.PlaybackState.PlayingState
                ):
                    self._dub_player.pause()
        else:
            if getattr(self, '_current_sim_frag_idx', None) is not None:
                self._dub_player.stop()
                self._current_sim_frag_idx = None

        self._update_extra_dub_lanes(schedule, pos_ms, is_seek, is_video_playing)

    def _get_extra_dub_player(self, lane: int) -> QMediaPlayer:
        if getattr(self, '_extra_dub_players', None) is None:
            self._extra_dub_players = {}
        pair = self._extra_dub_players.get(lane)
        if pair is None:
            player = QMediaPlayer()
            output = QAudioOutput()
            player.setAudioOutput(output)
            base_output = getattr(self, '_dub_audio_output', None)
            output.setVolume(base_output.volume() if base_output is not None else 1.0)
            pair = (player, output)
            self._extra_dub_players[lane] = pair
        return pair[0]

    def _pause_extra_dub_players(self) -> None:
        for player, _output in (getattr(self, '_extra_dub_players', None) or {}).values():
            player.pause()

    def _stop_extra_dub_players(self) -> None:
        for player, _output in (getattr(self, '_extra_dub_players', None) or {}).values():
            player.stop()
        state = getattr(self, '_extra_sim_idx', None)
        if state:
            state.clear()

    def _update_extra_dub_lanes(self, schedule: List[Dict], pos_ms: float,
                                is_seek: bool, is_video_playing: bool) -> None:
        if getattr(self, '_extra_dub_players', None) is None:
            self._extra_dub_players = {}
        if getattr(self, '_extra_sim_idx', None) is None:
            self._extra_sim_idx = {}
        state = self._extra_sim_idx

        lanes = {e.get('lane', 0) for e in schedule if e.get('lane', 0) > 0}
        for lane in sorted(lanes | set(self._extra_dub_players.keys())):
            match = None
            for entry in schedule:
                if entry['start_ms'] > pos_ms:
                    break
                if entry.get('lane', 0) != lane:
                    continue
                if entry['start_ms'] <= pos_ms < entry['end_ms']:
                    match = entry
                    break

            if match is None:
                if state.get(lane) is not None:
                    pair = self._extra_dub_players.get(lane)
                    if pair is not None:
                        pair[0].stop()
                    state[lane] = None
                continue

            player = self._get_extra_dub_player(lane)
            frag_idx = match['index']
            offset_ms = int((pos_ms - match['start_ms']) * match['rate'])

            if state.get(lane) != frag_idx:
                state[lane] = frag_idx
                player.setSource(QUrl.fromLocalFile(match['output_path']))
                player.setPlaybackRate(match['rate'])
                player.setPosition(offset_ms)
                if is_video_playing:
                    player.play()
            elif is_seek:
                player.setPlaybackRate(match['rate'])
                player.setPosition(offset_ms)
                if is_video_playing:
                    player.play()
            else:
                if player.playbackRate() != match['rate']:
                    player.setPlaybackRate(match['rate'])
                if (
                    is_video_playing
                    and player.playbackState() != QMediaPlayer.PlaybackState.PlayingState
                ):
                    if player.mediaStatus() != QMediaPlayer.MediaStatus.EndOfMedia:
                        player.play()
                elif (
                    not is_video_playing
                    and player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
                ):
                    player.pause()

    def build_video_preview_section(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self._video_section = CollapsibleSection("🎬 Video Preview", C["accent"])

        preview_top_lay = QHBoxLayout()
        self._sim_dub_check = QCheckBox("Simulate dubbed audio with video")
        self._sim_dub_check.setToolTip(
            "Automatically play generated audio fragments when watching the video preview. Syncs with offset."
        )
        self._sim_dub_check.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        preview_top_lay.addWidget(self._sim_dub_check)
        preview_top_lay.addStretch()

        self._fs_btn = QPushButton("⛶  Fullscreen")
        self._fs_btn.setFixedHeight(24)
        self._fs_btn.setStyleSheet("font-size:11px;padding:2px 8px;")
        self._fs_btn.setToolTip(
            "Open the video in fullscreen (or double-click the video).\n"
            "Space: play/pause\n"
            "Left/Right: seek 5 s (Shift: 1 s)\n"
            "Up/Down: volume\n"
            "F / Esc: exit fullscreen"
        )
        self._fs_btn.clicked.connect(self._enter_video_fullscreen)
        preview_top_lay.addWidget(self._fs_btn)

        self._dub_vol_widget = QWidget()
        self._dub_vol_widget.setObjectName("dubVolWrap")
        self._dub_vol_widget.setStyleSheet("QWidget#dubVolWrap { background: transparent; }")
        dub_vol_row = QHBoxLayout(self._dub_vol_widget)
        dub_vol_row.setContentsMargins(20, 0, 0, 0)
        dub_vol_row.setSpacing(8)
        dub_vol_title = QLabel("Lektor volume:")
        dub_vol_title.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        dub_vol_title.setFixedWidth(80)
        self._dub_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self._dub_vol_slider.setRange(0, 100)
        self._dub_vol_slider.setValue(100)
        self._dub_vol_slider.setFixedWidth(120)
        self._dub_vol_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._dub_vol_lbl = QLabel("100%")
        self._dub_vol_lbl.setStyleSheet(f"color:{C['accent']};font-size:11px;")
        self._dub_vol_slider.setToolTip(
            "Volume of the simulated lektor audio played over the video preview."
        )
        self._dub_vol_slider.valueChanged.connect(self._on_dub_vol_changed)
        dub_vol_row.addWidget(dub_vol_title)
        dub_vol_row.addWidget(self._dub_vol_slider)
        dub_vol_row.addWidget(self._dub_vol_lbl)
        dub_vol_row.addStretch()
        self._dub_vol_widget.setVisible(False)

        self._sim_dub_check.toggled.connect(self._dub_vol_widget.setVisible)

        self._video_widget = ClickableVideoWidget()
        self._video_widget.double_clicked.connect(self._enter_video_fullscreen)
        self._video_widget.setMinimumHeight(200)
        self._video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._media_player.setVideoOutput(self._video_widget)

        self._video_section.add_layout(preview_top_lay)
        self._video_section.add_widget(self._dub_vol_widget)
        self._video_section.add_widget(self._video_widget)
        self._video_section.set_enabled(False)
        lay.addWidget(self._video_section)

        self._on_dub_vol_changed(self._dub_vol_slider.value())

        self._video_preview_widget = w
        w.setVisible(False)
        return w

    def _find_fragment_at(self, pos_ms: int) -> Optional[int]:
        sim_on = bool(getattr(self, '_sim_dub_check', None) and self._sim_dub_check.isChecked())
        if sim_on:
            for entry in self._build_dub_schedule():
                if entry['start_ms'] > pos_ms:
                    break
                if pos_ms < entry['end_ms']:
                    return entry['index']
        for frag in self._fragments:
            start = frag.get('start_ms')
            end = frag.get('end_ms')
            if start is None or end is None:
                continue
            if start <= pos_ms < end:
                return frag['index']
        return None

    def _fragment_caption(self, idx: int) -> str:
        frag = next((f for f in self._fragments if f.get('index') == idx), None)
        if not frag:
            return ""
        text = " ".join((frag.get('text') or '').split())
        if len(text) > 90:
            text = text[:87].rstrip() + "…"
        return f"#{idx + 1}  {text}"

    def _set_playing_highlight(self, idx: int, scroll: bool = True):
        item = self._frag_items.get(idx)
        if item is None:
            return
        prev = getattr(self, '_hl_item', None)
        if prev is item:
            return
        color = QColor(C["accent"])
        color.setAlpha(80)
        brush = QBrush(color)
        cols = range(self._tree.columnCount())
        self._tree.blockSignals(True)
        try:
            if prev is not None:
                try:
                    for col in cols:
                        prev.setBackground(col, QBrush())
                except RuntimeError:
                    pass
            for col in cols:
                item.setBackground(col, brush)
        finally:
            self._tree.blockSignals(False)
        self._hl_item = item
        if scroll:
            self._tree.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtCenter)

    def _update_playing_highlight(self, pos_ms: int):
        if not self._fragments or not self._frag_items:
            return
        try:
            idx = self._find_fragment_at(pos_ms)
            if idx is None:
                return
            self._set_playing_highlight(
                idx, scroll=not getattr(self, '_suppress_scroll', False)
            )
            self._fs_update_fragment(idx)

            click_pos = getattr(self, '_click_pos_ms', None)
            if click_pos is not None and abs(pos_ms - click_pos) < 300:
                return

            if getattr(self, '_last_preview_idx', None) != idx:
                self._last_preview_idx = idx
                frag = next((f for f in self._fragments if f['index'] == idx), None)
                if frag and self._preview_text is not None:
                    self._preview_text.setPlainText(frag.get('text', ''))
        except Exception as e:
            logger.warning(f"Playing highlight failed: {e}")

    def _fs_update_fragment(self, idx: int):
        win = getattr(self, '_fs_window', None)
        if win is None or getattr(self, '_fs_frag_idx', None) == idx:
            return
        self._fs_frag_idx = idx
        win.set_fragment(self._fragment_caption(idx))

    def _refresh_video_frame(self):
        def _nudge():
            if self._media_player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
                self._media_player.setPosition(self._media_player.position())
        QTimer.singleShot(150, _nudge)

    def _enter_video_fullscreen(self):
        if getattr(self, '_fs_window', None) is not None:
            return
        if not self._video_loaded or self._video_widget is None:
            return
        player = self.main._video_player
        if not player.has_media():
            self.main._set_status("Video audio is still loading — try again in a moment", C["warning"])
            return

        win = VideoFullscreenWindow(self.main)
        win.toggle_play_requested.connect(player.toggle_play)
        win.stop_requested.connect(player.stop_playback)
        win.seek_requested.connect(player.seek_seconds)
        win.seek_relative_requested.connect(player.seek_relative)
        win.volume_changed.connect(player.set_volume)
        win.closed.connect(self._exit_video_fullscreen)
        player.position_updated.connect(self._on_fs_position)
        player.playback_state_changed.connect(self._on_fs_playing)

        self._fs_window = win
        self._fs_frag_idx = None
        win.set_volume(player.get_volume())
        win.set_playing(player.is_playing())
        win.set_position(player.position_seconds(), player.duration_seconds())

        self._media_player.setVideoOutput(win.video_widget)
        win.enter(self.main.screen())
        self._refresh_video_frame()
        self._update_playing_highlight(int(player.position_seconds() * 1000))

    def _exit_video_fullscreen(self):
        win = getattr(self, '_fs_window', None)
        if win is None:
            return
        self._fs_window = None
        self._fs_frag_idx = None
        player = self.main._video_player
        for sig, slot in (
            (player.position_updated, self._on_fs_position),
            (player.playback_state_changed, self._on_fs_playing),
        ):
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._media_player.setVideoOutput(self._video_widget)
        win.deleteLater()
        self._refresh_video_frame()

    def _on_fs_position(self, pos_s: float):
        win = getattr(self, '_fs_window', None)
        if win is not None:
            win.set_position(pos_s, self.main._video_player.duration_seconds())

    def _on_fs_playing(self, playing: bool):
        win = getattr(self, '_fs_window', None)
        if win is not None:
            win.set_playing(playing)

    def _on_dub_vol_changed(self, value: int):
        self._dub_vol_lbl.setText(f"{value}%")
        volume = value / 100.0
        if getattr(self, '_dub_audio_output', None):
            self._dub_audio_output.setVolume(volume)
        for _player, output in (getattr(self, '_extra_dub_players', None) or {}).values():
            output.setVolume(volume)

    def _update_video_preview_visibility(self):
        if getattr(self, '_video_preview_widget', None) is not None:
            self._video_preview_widget.setVisible(
                self._lektor_tab_active and self._video_loaded
            )

    def set_lektor_tab_active(self, active: bool):
        self._lektor_tab_active = active
        self._update_video_preview_visibility()

    # ------------------------------------------------------------------
    # Public methods for MainWindow
    # ------------------------------------------------------------------

    def start_synthesis(self, retry_errors: bool = False):
        """Public entry point for synthesis."""
        self._start_synthesis(retry_errors)

    def stop_synthesis(self):
        """Public entry point to stop synthesis."""
        if self._worker:
            self._worker.request_cancel()
        self.main._set_status("Stop requested — waiting for current fragment to finish…", C["warning"])

    def has_fragments(self) -> bool:
        """Return True if any fragments are loaded."""
        return bool(self._fragments)

    def populate_tree(self):
        """Public method to refresh tree."""
        self._populate_tree()

    def update_tree_item_duration(self, idx: int, dur: float):
        """Update tree item with new duration."""
        self._update_tree_item(idx, known_dur_s=dur)

    def update_dubbing_visibility(self):
        """Update dubbing button visibility."""
        self._update_dubbing_visibility()

    def set_dubbing_button_state(self, enabled: bool, text: str):
        """Set dubbing button state."""
        if hasattr(self, "_btn_dubbing"):
            self._btn_dubbing.setEnabled(enabled)
            self._btn_dubbing.setText(text)

    def set_dubbing_button_style(self, style: str):
        """Set dubbing button style."""
        if hasattr(self, "_btn_dubbing"):
            self._btn_dubbing.setStyleSheet(style)

    def reset_dubbing_ui(self):
        """Reset dubbing UI state."""
        self._reset_dubbing_state()
