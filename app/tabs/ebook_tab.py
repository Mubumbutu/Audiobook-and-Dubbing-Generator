# tabs/ebook_tab.py
"""
Ebook tab implementation – EPUB, PDF, TXT, Kindle, FB2 loading,
fragment management, synthesis, audiobook export and preview.
"""
import os
import json
import time
import gc
import subprocess
import tempfile
import hashlib
import logging
import shutil
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import soundfile as sf

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTreeWidget, QTreeWidgetItem, QPlainTextEdit, QLineEdit,
    QGroupBox, QFileDialog, QProgressBar, QHeaderView,
    QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QSlider,
    QDialog, QDialogButtonBox, QInputDialog, QMessageBox, QFrame,
    QSizePolicy, QApplication, QMenu
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QColor, QAction

from .base_tab import BaseTab

# Importy z wydzielonych modułów
from config import (
    C, _btn, OUTPUTS_DIR,
    STATUS_WAITING, STATUS_RUNNING, STATUS_DONE, STATUS_ERROR,
    COL_STATUS, COL_FRAGMENT, COL_SPEAKER, COL_TIMING
)
from widgets import FragmentTreeWidget
from utils import _get_last_dir, _set_last_dir, _fmt, _fmt_ms, _get_wav_duration, _open_file, _ref_audio_hash
from workers import TTSWorker

# Lokalne importy (pozostawione bez zmian)
from input_formats import get_format
from txt_format import txt_ebook_format

logger = logging.getLogger(__name__)


class EbookTab(BaseTab):
    """Ebook processing tab."""

    def __init__(self, main_window):
        super().__init__(main_window)
        self._ebook_fragments: List[Dict] = []
        self._ebook_frag_items: Dict[int, QTreeWidgetItem] = {}
        self._ebook_chapter_item: Optional[QTreeWidgetItem] = None
        self._epub_path: Optional[str] = None
        self._ebook_output_dir: str = ""
        self._is_running: bool = False
        self._completed_count: int = 0
        self._synth_start_time: float = 0.0
        self._synth_total: int = 0
        self._worker = None
        self._synthesis_source: str = "ebook"
        self._dubbing_mode: bool = False
        self._speaker_list: List[str] = []
        self._speaker_voices: Dict = {}

        self._btn_close_ebook = None
        self._btn_save_ebook_session = None
        self._epub_label = None
        self._ebook_tree = None
        self._ebook_preview_text = None
        self._preview_audiobook_btn = None
        self._ebook_filter_edit = None
        self._ebook_sel_fail_btn = None
        self._ebook_sel_pending_btn = None
        self._audiobook_output_edit = None
        self._audiobook_silence_spin = None
        self._audiobook_fmt_combo = None
        self._audiobook_norm_check = None
        self._audiobook_status_lbl = None
        self._audiobook_export_btn = None

        self._btn_start = None
        self._btn_stop = None
        self._synth_progress = None
        self._eta_label = None
        self._progress = None
        self._norm_check = None
        self._file_watcher = main_window._file_watcher

        self._build_ui()

    def set_shared_widgets(self, btn_start, btn_stop, synth_progress, eta_label, progress, norm_check):
        """Sets shared widgets from MainWindow."""
        self._btn_start = btn_start
        self._btn_stop = btn_stop
        self._synth_progress = synth_progress
        self._eta_label = eta_label
        self._progress = progress
        self._norm_check = norm_check

    def get_widget(self) -> QWidget:
        """Return the built tab widget."""
        return self._ebook_tab_widget

    def _build_ui(self):
        """Build the ebook tab UI."""
        self._ebook_tab_widget = self._make_ebook_tab()

    # ------------------------------------------------------------------
    # Ebook Tab UI
    # ------------------------------------------------------------------

    def _make_ebook_tab(self) -> QWidget:
        w = QWidget()
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

        self._ebook_sel_fail_btn = sel_fail
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
        self._ebook_tree.setColumnWidth(COL_STATUS, 28)
        self._ebook_tree.setColumnWidth(COL_FRAGMENT, 370)
        self._ebook_tree.setColumnWidth(COL_SPEAKER, 90)
        self._ebook_tree.setColumnWidth(COL_TIMING, 160)
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

        self._ebook_tree.header().setSectionResizeMode(COL_STATUS, QHeaderView.ResizeMode.Fixed)
        self._ebook_tree.header().setSectionResizeMode(COL_FRAGMENT, QHeaderView.ResizeMode.Interactive)
        self._ebook_tree.header().setSectionResizeMode(COL_SPEAKER, QHeaderView.ResizeMode.Interactive)
        self._ebook_tree.header().setSectionResizeMode(COL_TIMING, QHeaderView.ResizeMode.Stretch)

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

    # ------------------------------------------------------------------
    # Ebook Loading / Closing
    # ------------------------------------------------------------------

    def _load_ebook_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self.main, "Open ebook file", _get_last_dir("ebook"),
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
                QMessageBox.warning(self.main, "Empty file", "No text fragments found in this file.")
                return
            self._ebook_fragments = []
            for seg in segments:
                self._ebook_fragments.append({
                    'index': seg.index,
                    'text': seg.text,
                    'speaker': seg.speaker or "",
                    'status': 'waiting',
                    'output_path': None,
                    'error_msg': None,
                })
        except Exception as e:
            QMessageBox.critical(self.main, "Load error", str(e))
            return

        self._epub_path = path
        self._ebook_output_dir = str(OUTPUTS_DIR / Path(path).stem)
        if self._audiobook_output_edit is not None:
            self._audiobook_output_edit.setText(self._ebook_output_dir)

        auto = self._auto_ebook_session_path()
        if auto and auto.exists():
            reply = QMessageBox.question(
                self.main, "Restore previous session",
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
                    self.main._set_status("Could not restore previous session — starting fresh.", C["warning"])

        fname = Path(path).name
        self._epub_label.setText(
            f"📚  {fname}  •  {len(self._ebook_fragments)} fragments"
        )
        self._epub_label.setStyleSheet(
            f"color:{C['accent']};font-size:11px;font-weight:600;"
        )
        self._btn_close_ebook.setEnabled(True)
        self._btn_save_ebook_session.setEnabled(True)
        self.main._tabs.setCurrentIndex(1)
        self._populate_ebook_tree()
        self.main._update_action_buttons()
        self.main._set_status(f"Loaded: {fname} — {len(self._ebook_fragments)} fragments")

    def _close_ebook_file(self):
        if self._is_running:
            QMessageBox.warning(self.main, "Busy", "Cannot close file during active synthesis.")
            return
        reply = QMessageBox.question(
            self.main, "Close ebook",
            "Close current ebook file?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._ebook_fragments.clear()
        self._ebook_frag_items.clear()
        self._ebook_chapter_item = None
        self._epub_path = None
        self._ebook_tree.clear()
        self._ebook_preview_text.clear()
        self._epub_label.setText("No ebook loaded")
        self._epub_label.setStyleSheet(
            f"color:{C['text3']};font-size:11px;font-style:italic;"
        )
        self._btn_close_ebook.setEnabled(False)
        self._btn_save_ebook_session.setEnabled(False)
        self.main._update_action_buttons()
        self.main._set_status("Ebook file closed.")

    # ------------------------------------------------------------------
    # Tree Population
    # ------------------------------------------------------------------

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
            'done': STATUS_DONE,
            'error': STATUS_ERROR,
        }.get(status, STATUS_WAITING)

        num = frag['index'] + 1
        item.setText(COL_FRAGMENT, f"{icon}  #{num}  {frag.get('text', '')[:75]}")
        item.setText(COL_SPEAKER, frag.get('speaker') or "")

        if status == 'error':
            item.setForeground(COL_FRAGMENT, QColor(C["error"]))
        else:
            item.setForeground(COL_FRAGMENT, QColor(C["text"]))

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

    # ------------------------------------------------------------------
    # Fragment Selection / Filtering
    # ------------------------------------------------------------------

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
                idx = child.data(COL_STATUS, Qt.ItemDataRole.UserRole)
                frag = frag_map.get(idx)
                cs = Qt.CheckState.Checked if (frag and frag.get('status') == 'error') else Qt.CheckState.Unchecked
                child.setCheckState(COL_STATUS, cs)
        finally:
            self._ebook_tree.blockSignals(False)
            self._ebook_tree.setUpdatesEnabled(True)
        self._update_preview_btn_state()

    def _select_all_ebook_reset(self, state: bool):
        self._ebook_sel_fail_btn.blockSignals(True)
        self._ebook_sel_pending_btn.blockSignals(True)
        self._ebook_sel_fail_btn.setChecked(False)
        self._ebook_sel_pending_btn.setChecked(False)
        self._ebook_sel_fail_btn.blockSignals(False)
        self._ebook_sel_pending_btn.blockSignals(False)
        self._select_all_ebook(state)

    def _apply_ebook_status_filter(self):
        if not self._ebook_chapter_item:
            return
        want_failed = self._ebook_sel_fail_btn.isChecked()
        want_pending = self._ebook_sel_pending_btn.isChecked()
        frag_map = {f['index']: f for f in self._ebook_fragments}
        self._ebook_tree.setUpdatesEnabled(False)
        self._ebook_tree.blockSignals(True)
        try:
            for i in range(self._ebook_chapter_item.childCount()):
                child = self._ebook_chapter_item.child(i)
                if child.isHidden():
                    continue
                idx = child.data(COL_STATUS, Qt.ItemDataRole.UserRole)
                frag = frag_map.get(idx)
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
        frag = next((f for f in self._ebook_fragments if f.get('index') == idx), None)
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

    def _on_ebook_tree_item_double_clicked(self, item, column):
        idx = item.data(COL_STATUS, Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        if self._load_fragment_audio(idx):
            self.main._start_play()

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

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
            idx = child.data(COL_STATUS, Qt.ItemDataRole.UserRole)
            frag = next((f for f in self._ebook_fragments if f['index'] == idx), None)
            if not frag:
                continue
            if retry_errors:
                if frag.get('status') not in ('waiting', 'error'):
                    continue
            result.append(frag)
        return result

    def _start_ebook_synthesis(self, retry_errors: bool = False):
        if not self.main._model_is_loaded:
            QMessageBox.warning(self.main, "Model not loaded", "Load the model first.")
            return

        to_process = self._get_checked_ebook_fragments(retry_errors=retry_errors)
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
            self._update_ebook_tree_item(frag["index"])

        self._synthesis_source = 'ebook'
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

        os.makedirs(self._ebook_output_dir, exist_ok=True)
        reserved_paths = {f['output_path'] for f in self._ebook_fragments if f.get('output_path')}

        self._worker = TTSWorker(
            backend=self.main._backend,
            fragments=to_process,
            output_dir=self._ebook_output_dir,
            reference_audio=self.main._get_ref_audio(),
            reference_text=self.main._get_ref_text(),
            filename_prefix=Path(self._epub_path).stem if self._epub_path else "ebook_fragment",
            generation_settings=self.main._get_generation_settings(),
            normalize_audio=self._norm_check.isChecked(),
            speaker_voices=self._get_speaker_voices_dict(),
            reserved_paths=reserved_paths,
        )

        self._worker.progress.connect(self._on_progress)
        self._worker.item_done.connect(self._on_item_done)
        self._worker.finished.connect(self._on_synthesis_finished)
        self._worker.start()

        self.main._set_status(f"Ebook synthesis started — {total} fragments queued…")
        logger.info(f"Ebook synthesis started: {total} fragments")

    def _on_progress(self, idx: int, msg: str, is_error: bool):
        if idx >= 0:
            frag = next((f for f in self._ebook_fragments if f['index'] == idx), None)
            if frag:
                frag['status'] = 'running' if not is_error else 'error'
                self._update_ebook_tree_item(idx)
        color = C["error"] if is_error else C["text2"]
        self.main._set_status(msg, color)

    def _on_item_done(self, idx: int, result: str, is_error: bool):
        frag = next((f for f in self._ebook_fragments if f['index'] == idx), None)
        if not frag:
            return
        if is_error:
            frag['status'] = 'error'
            frag['error_msg'] = result
        else:
            frag['status'] = 'done'
            frag['output_path'] = result
            frag['error_msg'] = None
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

        done = sum(1 for f in self._ebook_fragments if f.get("status") == "done")
        error = sum(1 for f in self._ebook_fragments if f.get("status") == "error")

        self.main._set_status(
            f"Synthesis complete — {done} done, {error} errors.",
            C["success"] if error == 0 else C["warning"],
        )
        self.main._update_action_buttons()
        logger.info(f"Synthesis finished: {done} done, {error} errors")

        auto = self._auto_ebook_session_path()
        if auto:
            if self._write_ebook_session_to(str(auto)):
                logger.info(f"Auto-saved ebook session: {auto}")

    def _re_synthesize_ebook_fragment(self, frag: Dict):
        if not self.main._model_is_loaded:
            QMessageBox.warning(self.main, "Model not loaded", "Load the model first.")
            return

        self._synthesis_source = 'ebook'

        frag["status"] = "waiting"
        frag["error_msg"] = None
        self._update_ebook_tree_item(frag["index"])

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
            f['output_path'] for f in self._ebook_fragments if f.get('output_path')
        }

        self._worker = TTSWorker(
            backend=self.main._backend,
            fragments=[frag],
            output_dir=self._ebook_output_dir,
            reference_audio=self.main._get_ref_audio(),
            reference_text=self.main._get_ref_text(),
            filename_prefix=Path(self._epub_path).stem if self._epub_path else "ebook_fragment",
            generation_settings=self.main._get_generation_settings(),
            normalize_audio=self._norm_check.isChecked(),
            speaker_voices=self._get_speaker_voices_dict(),
            reserved_paths=reserved_paths,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.item_done.connect(self._on_item_done)
        self._worker.finished.connect(self._on_synthesis_finished)
        self._worker.start()

    # ------------------------------------------------------------------
    # Context Menu
    # ------------------------------------------------------------------

    def _show_ebook_context_menu(self, pos):
        item = self._ebook_tree.itemAt(pos)
        if not item:
            return
        idx = item.data(COL_STATUS, Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        frag = next((f for f in self._ebook_fragments if f["index"] == idx), None)
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

        speaker = frag.get("speaker") or ""
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
            (i for i, f in enumerate(self._ebook_fragments) if f["index"] == idx), -1
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

    # ------------------------------------------------------------------
    # Fragment Editing
    # ------------------------------------------------------------------

    def _edit_ebook_fragment_text(self, frag: Dict):
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Edit fragment #{frag['index'] + 1}")
        dlg.resize(560, 260)
        dlg.setStyleSheet(self.main.styleSheet())

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

        frag['text'] = new_text
        frag['status'] = 'waiting'
        frag['error_msg'] = None
        self._update_ebook_tree_item(frag['index'])
        self._ebook_preview_text.setPlainText(new_text)
        self.main._set_status(f"Fragment #{frag['index'] + 1} text updated — status reset to waiting.")
        
    def _prompt_ebook_speaker_name(self, title: str, message: str, current: str) -> Tuple[Optional[str], bool]:
        all_speakers = sorted(
            {
                (f.get('speaker') or "").strip()
                for f in self._ebook_fragments
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

    def _edit_ebook_speaker(self, frag: Dict, clear: bool = False):
        if clear:
            frag['speaker'] = None
            self._update_ebook_tree_item(frag['index'])
            self._sync_ebook_speaker_ui()
            return

        current = frag.get('speaker') or ""
        name, ok = self._prompt_ebook_speaker_name(
            "Speaker name",
            "Select or enter speaker name:",
            current,
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
                self.main, "Nothing selected",
                "No fragments are checked.\nCheck at least one fragment first."
            )
            return

        checked_frags = [f for f in self._ebook_fragments if f['index'] in checked_indices]
        current = checked_frags[0].get('speaker') or ""

        name, ok = self._prompt_ebook_speaker_name(
            "Speaker name",
            f"Set speaker for {len(checked_frags)} selected fragment(s):",
            current,
        )
        if not ok:
            return

        new_speaker = name.strip() or None
        for frag in checked_frags:
            frag['speaker'] = new_speaker
            self._update_ebook_tree_item(frag['index'])

        self._sync_ebook_speaker_ui()
        self.main._set_status(
            f"Speaker {'set to ' + new_speaker if new_speaker else 'cleared'} "
            f"for {len(checked_frags)} fragment(s).",
            C["accent"],
        )

    def _edit_ebook_fragment_timing(self, frag: Dict):
        path = frag.get('output_path')
        if not path or not os.path.exists(path):
            QMessageBox.warning(
                self.main, "No audio",
                "Synthesize this fragment first before adjusting its duration."
            )
            return

        audio_dur_s = _get_wav_duration(path)
        if audio_dur_s is None:
            QMessageBox.warning(self.main, "Error", "Cannot read audio duration.")
            return

        audio_dur_ms = int(audio_dur_s * 1000)
        current_target = frag.get('target_duration_ms', audio_dur_ms)
        current_extra_ms = max(0, current_target - audio_dur_ms)
        current_pre_ms = int(frag.get('pre_silence_ms') or 0)

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Edit duration — fragment #{frag['index'] + 1}")
        dlg.resize(400, 230)
        dlg.setStyleSheet(self.main.styleSheet())

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
        self.main._set_status(
            f"Fragment #{frag['index'] + 1} — audio: {_fmt_ms(audio_dur_ms)}, "
            f"silence before: {pre_ms} ms, silence after: {extra_ms} ms.",
            C["accent"],
        )

    # ------------------------------------------------------------------
    # Speaker UI (Ebook)
    # ------------------------------------------------------------------

    def _sync_ebook_speaker_ui(self):
        if self.main._dubbing_video_path:
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
                self.main._dubbing_mode = False
                self.main._speaker_list = []
                self.main._speaker_voices.clear()
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
        for spk, sv in self.main._speaker_voices.items():
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
        self.main._speaker_list = speakers
        self.main._dubbing_mode = True
        self._rebuild_voice_cloning_for_speakers()

        for spk, path in saved_paths.items():
            if spk in self.main._speaker_voices and os.path.exists(path):
                sv = self.main._speaker_voices[spk]
                if "drop" not in sv:
                    continue
                sv["drop"]._set(path)
                sv["player"].load(path)
                sv["proc_btn"].setEnabled(True)
                sv["tr_btn"].setEnabled(
                    self.whisper is not None
                    and self.whisper.is_downloaded(self.main._w_size.currentData())
                )

        for spk, text in saved_texts.items():
            if spk in self.main._speaker_voices and text:
                sv = self.main._speaker_voices[spk]
                if "ref_text" in sv:
                    sv["ref_text"].setPlainText(text)

        self.main._set_status(
            f"{len(speakers)} speaker(s) — add reference audio for each in Voice cloning.",
            C["accent"],
        )

    def _rebuild_voice_cloning_for_speakers(self):
        self.main._rebuild_voice_cloning_for_speakers()

    def _get_speaker_voices_dict(self) -> Optional[Dict]:
        return self.main._get_speaker_voices_dict()

    # ------------------------------------------------------------------
    # Fragment Management
    # ------------------------------------------------------------------

    def _add_ebook_fragment_after(self, frag: Dict):
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

        pos = next((i for i, f in enumerate(self._ebook_fragments) if f['index'] == frag['index']), -1)
        if pos < 0:
            return

        new_frag = {
            'index': 0,
            'text': new_text,
            'speaker': frag.get('speaker'),
            'status': 'waiting',
            'output_path': None,
            'error_msg': None,
        }
        self._ebook_fragments.insert(pos + 1, new_frag)

        for i, f in enumerate(self._ebook_fragments):
            f['index'] = i

        self._rename_ebook_outputs_to_match_order()
        self._populate_ebook_tree()
        self.main._update_action_buttons()

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

        if result["action"] == "all":
            removed_fragments = [f for f in self._ebook_fragments if f['index'] in checked_indices]
            self._ebook_fragments = [f for f in self._ebook_fragments if f['index'] not in checked_indices]
        else:
            removed_fragments = [f for f in self._ebook_fragments if f['index'] == frag['index']]
            self._ebook_fragments = [f for f in self._ebook_fragments if f['index'] != frag['index']]

        for f in removed_fragments:
            p = f.get('output_path', '')
            if p and os.path.exists(p):
                self._file_watcher.removePath(p)
                try:
                    os.remove(p)
                except Exception as e:
                    logger.warning(f"Could not delete orphan audio {p}: {e}")

        for i, f in enumerate(self._ebook_fragments):
            f['index'] = i

        self._rename_ebook_outputs_to_match_order()
        self._populate_ebook_tree()
        self.main._update_action_buttons()
        self.main._set_status(f"Fragment(s) removed. {len(self._ebook_fragments)} fragments remaining.")

    def _rename_ebook_outputs_to_match_order(self):
        if not self._ebook_output_dir:
            return
        prefix = Path(self._epub_path).stem if self._epub_path else "ebook_fragment"

        to_rename = [
            frag for frag in self._ebook_fragments
            if frag.get('output_path') and os.path.exists(frag['output_path'])
        ]
        if not to_rename:
            return

        desired = {
            frag['index']: os.path.join(
                self._ebook_output_dir, f"{prefix}_{frag['index'] + 1:03d}.wav"
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

        self._rename_ebook_outputs_to_match_order()
        self._populate_ebook_tree()

        moved_item = self._ebook_frag_items.get(new_pos)
        if moved_item:
            self._ebook_tree.setCurrentItem(moved_item)

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

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

        can_merge_next = pos + 1 < len(self._ebook_fragments)
        is_checked = frag['index'] in checked_indices
        checked_count = len(checked_indices)
        can_merge_checked = is_checked and checked_count >= 2

        dlg = QDialog(self)
        dlg.setWindowTitle("Merge fragments")
        dlg.resize(460, 190)
        dlg.setStyleSheet(self.main.styleSheet())
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
        btn_next = QPushButton("Merge with next fragment")
        btn_checked = QPushButton(f"Merge all checked  ({checked_count})")
        btn_cancel = QPushButton("Cancel")

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

        self.main._stop_play()

        combined_text = " ".join(
            f.get('text', '').strip() for f in fragments if f.get('text', '').strip()
        )
        speaker = fragments[0].get('speaker', '')

        target_sr = 44100
        parts: List[np.ndarray] = []
        import torch
        import torchaudio.functional as TAF
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
                tmp_name = f"__merge_tmp_{abs(hash(tuple(f['index'] for f in fragments))) % 10**9}.wav"
                tmp_path = os.path.join(self._ebook_output_dir, tmp_name)
                combined = np.concatenate(parts)
                sf.write(tmp_path, combined, target_sr, subtype='PCM_16')
                merged_audio_path = tmp_path
                merged_status = 'done'
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
            'index': 0,
            'text': combined_text,
            'speaker': speaker,
            'status': merged_status,
            'output_path': merged_audio_path,
            'error_msg': None,
        }

        self._ebook_fragments = [
            f for f in self._ebook_fragments if f['index'] not in indices_to_remove
        ]
        self._ebook_fragments.insert(insert_pos, merged_frag)

        for i, f in enumerate(self._ebook_fragments):
            f['index'] = i

        self._rename_ebook_outputs_to_match_order()

        if merged_audio_path and os.path.exists(merged_audio_path):
            self._file_watcher.addPath(merged_audio_path)

        if self.main._current_fragment_idx in indices_to_remove:
            self.main._current_fragment_idx = None
            self.main._audio_data = None
            if hasattr(self.main, '_wave_out'):
                self.main._wave_out.clear()
            self.main._play_btn.setEnabled(False)
            self.main._stop_btn.setEnabled(False)
            self.main._time_lbl.setText("0:00 / 0:00")

        self._populate_ebook_tree()
        self.main._update_action_buttons()
        self._update_preview_btn_state()

        self.main._set_status(
            f"Merged {len(fragments)} ebook fragments into one.", C["accent"]
        )

    # ------------------------------------------------------------------
    # Audiobook Export / Preview
    # ------------------------------------------------------------------

    def _build_audiobook_section(self):
        """Build the Audiobook export UI section."""
        hint = QLabel(
            "Concatenates all synthesized fragments into a single audiobook file."
        )
        hint.setStyleSheet(f"color:{C['text3']};font-size:10px;")
        hint.setWordWrap(True)
        self.main._audiobook_section.add_widget(hint)

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
        self.main._audiobook_section.add_layout(out_row)

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
        self.main._audiobook_section.add_layout(opt_row)

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
        self.main._audiobook_section.add_layout(fmt_row)

        self._audiobook_norm_check = QCheckBox("Normalize audio (ffmpeg)")
        self._audiobook_norm_check.setEnabled(self.main._ffmpeg_ok)
        if not self.main._ffmpeg_ok:
            self._audiobook_norm_check.setToolTip("ffmpeg not found in PATH")
        self.main._audiobook_section.add_widget(self._audiobook_norm_check)

        exp_row = QHBoxLayout()
        self._audiobook_export_btn = QPushButton("🎧  Export audiobook")
        self._audiobook_export_btn.setStyleSheet(_btn(C["accent"]))
        self._audiobook_export_btn.clicked.connect(self._export_audiobook)
        exp_row.addWidget(self._audiobook_export_btn)
        self._audiobook_subtitles_btn = QPushButton("📝  Export subtitles")
        self._audiobook_subtitles_btn.setStyleSheet(_btn(C["accent"]))
        self._audiobook_subtitles_btn.clicked.connect(self._export_subtitles)
        exp_row.addWidget(self._audiobook_subtitles_btn)
        exp_row.addStretch()
        self.main._audiobook_section.add_layout(exp_row)

        self._audiobook_status_lbl = QLabel("")
        self._audiobook_status_lbl.setStyleSheet(
            f"color:{C['text3']};font-size:10px;font-style:italic;"
        )
        self.main._audiobook_section.add_widget(self._audiobook_status_lbl)

    def _browse_audiobook_output(self):
        d = QFileDialog.getExistingDirectory(
            self.main, "Select output folder",
            _get_last_dir("output", str(OUTPUTS_DIR))
        )
        if d:
            _set_last_dir("output", d)
            self._audiobook_output_edit.setText(d)
            self._ebook_output_dir = d

    def _export_audiobook(self):
        if not self._ebook_fragments:
            QMessageBox.warning(self.main, "No ebook loaded", "Load an EPUB or TXT first.")
            return

        done_frags = [
            f for f in self._ebook_fragments
            if f.get('status') == 'done'
            and f.get('output_path')
            and os.path.exists(f.get('output_path', ''))
        ]
        if not done_frags:
            QMessageBox.warning(self.main, "No audio", "No fragments have been synthesized yet.")
            return

        done_frags = sorted(done_frags, key=lambda f: f['index'])

        fmt = (
            self._audiobook_fmt_combo.currentData()
            if self._audiobook_fmt_combo is not None
            else "wav"
        )
        fmt = fmt or "wav"

        silence_ms = (
            self._audiobook_silence_spin.value()
            if self._audiobook_silence_spin is not None
            else 500
        )

        base_name = Path(self._epub_path).stem if self._epub_path else "audiobook"
        default_path = str(
            Path(_get_last_dir("output", self._ebook_output_dir)) / f"{base_name}.{fmt}"
        )

        path, _ = QFileDialog.getSaveFileName(
            self.main,
            "Save audiobook",
            default_path,
            f"Audio {fmt.upper()} (*.{fmt});;All files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(f".{fmt}"):
            path += f".{fmt}"

        _set_last_dir("output", path)

        self.main._set_status("Exporting audiobook...")
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)

        try:
            all_audio = []
            sr = 44100
            import torch
            import torchaudio.functional as TAF
            for i, frag in enumerate(done_frags):
                audio, frag_sr = sf.read(frag['output_path'], dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if frag_sr != sr:
                    audio = np.ascontiguousarray(
                        TAF.resample(
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
            self.main._set_status(f"Audiobook saved: {path}", C["success"])

            reply = QMessageBox.information(
                self.main, "Export complete",
                f"Audiobook saved successfully:\n{path}\n\nOpen output folder?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                _open_file(os.path.dirname(path))

        except Exception:
            self._progress.setVisible(False)
            self.main._on_error("Audiobook export error", traceback.format_exc())

    def _export_subtitles(self):
        if not self._ebook_fragments:
            QMessageBox.warning(self.main, "No ebook loaded", "Load an EPUB or TXT first.")
            return

        done_frags = [
            f for f in self._ebook_fragments
            if f.get('status') == 'done'
            and f.get('output_path')
            and os.path.exists(f.get('output_path', ''))
        ]
        if not done_frags:
            QMessageBox.warning(self.main, "No audio", "No fragments have been synthesized yet.")
            return

        done_frags = sorted(done_frags, key=lambda f: f['index'])

        silence_ms = (
            self._audiobook_silence_spin.value()
            if self._audiobook_silence_spin is not None
            else 500
        )

        base_name = Path(self._epub_path).stem if self._epub_path else "audiobook"
        default_path = str(
            Path(_get_last_dir("output", self._ebook_output_dir)) / f"{base_name}.srt"
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
            entries = []
            cursor_ms = 0.0
            for i, frag in enumerate(done_frags):
                dur = _get_wav_duration(frag.get('output_path', ''))
                dur_ms = int(round(dur * 1000)) if dur is not None else 0
                pre_ms = int(frag.get('pre_silence_ms') or 0)
                target_ms = frag.get('target_duration_ms')
                extra_ms = 0
                if target_ms and target_ms > dur_ms:
                    extra_ms = int(target_ms) - dur_ms

                start_ms = cursor_ms + pre_ms
                end_ms = start_ms + dur_ms
                entries.append((frag.get('text', '').strip(), start_ms, end_ms))

                cursor_ms = end_ms + extra_ms
                if i < len(done_frags) - 1 and silence_ms > 0:
                    cursor_ms += silence_ms

            lines = []
            for i, (text, start_ms, end_ms) in enumerate(entries, start=1):
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
                self.main, "Nothing to preview",
                "No synthesized fragments in selection. Run synthesis first."
            )
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Preview selected fragments")
        dlg.resize(380, 150)
        dlg.setStyleSheet(self.main.styleSheet())
        lay = QVBoxLayout(dlg)
        lay.setSpacing(10)
        lay.setContentsMargins(14, 14, 14, 10)

        info = QLabel(f"Stitch and preview {len(done_frags)} synthesized fragment(s).")
        info.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        lay.addWidget(info)

        norm_chk = QCheckBox("Normalize audio (ffmpeg)")
        norm_chk.setEnabled(self.main._ffmpeg_ok)
        if not self.main._ffmpeg_ok:
            norm_chk.setToolTip("ffmpeg not found in PATH")
        norm_init = (
            self._audiobook_norm_check.isChecked()
            if self._audiobook_norm_check is not None
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
            if self._audiobook_silence_spin is not None
            else 500
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

        normalize = norm_chk.isChecked()
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
        global_silence = np.zeros(silence_samples, dtype=np.float32)

        self.main._set_status("Building preview…")
        QApplication.processEvents()

        parts: List[np.ndarray] = []
        import torch
        import torchaudio.functional as TAF
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
                    audio = np.ascontiguousarray(tensor.squeeze(0).numpy(), dtype=np.float32)

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
            QMessageBox.warning(self.main, "Preview failed", "Could not load any audio fragments.")
            return

        combined = np.concatenate(parts).astype(np.float32)

        if normalize:
            tmp_wav = tempfile.mktemp(suffix="_preview.wav")
            norm_wav = tempfile.mktemp(suffix="_preview_norm.wav")
            try:
                sf.write(tmp_wav, combined, target_sr, subtype="PCM_16")
                if self._normalize_ffmpeg(tmp_wav, norm_wav):
                    norm_audio, norm_sr = sf.read(norm_wav, dtype="float32")
                    combined = np.ascontiguousarray(norm_audio, dtype=np.float32)
                    if int(norm_sr) != target_sr:
                        tensor = torch.from_numpy(combined).unsqueeze(0)
                        tensor = TAF.resample(tensor, int(norm_sr), target_sr)
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

        self.main._load_audio_to_player(combined, target_sr, fragment_idx=None)
        dur = len(combined) / target_sr
        self.main._set_status(
            f"Preview ready — {len(done_frags)} fragments | {_fmt(dur)}", C["success"]
        )

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

    # ------------------------------------------------------------------
    # Session Management (Ebook)
    # ------------------------------------------------------------------

    def _save_ebook_session(self):
        if not self._epub_path:
            return
        default_path = str(
            Path(_get_last_dir("session", str(OUTPUTS_DIR))) / "ebook_session.json"
        )
        path, _ = QFileDialog.getSaveFileName(
            self.main, "Save ebook session", default_path,
            "JSON (*.json);;All files (*)",
        )
        if not path:
            return
        _set_last_dir("session", path)
        if self._write_ebook_session_to(path):
            auto = self._auto_ebook_session_path()
            if auto and str(auto) != path:
                self._write_ebook_session_to(str(auto))
            self.main._set_status(f"Ebook session saved: {path}", C["success"])
        else:
            QMessageBox.critical(self.main, "Save error", f"Could not write session to:\n{path}")

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

        if data.get("session_type") == "ebook":
            self._restore_ebook_session_data(data)
        else:
            self.main.srt_tab._restore_session_data(data)

    def _auto_ebook_session_path(self) -> Optional[Path]:
        if not self._epub_path:
            return None
        return OUTPUTS_DIR / Path(self._epub_path).stem / "ebook_session.json"

    def _build_ebook_session_data(self) -> dict:
        ref_audio = self.main._drop.file_path
        return {
            "session_type": "ebook",
            "version": 1,
            "epub_path": self._epub_path,
            "ebook_output_dir": self._ebook_output_dir,
            "reference_audio": ref_audio,
            "reference_audio_hash": _ref_audio_hash(ref_audio),
            "reference_text": self.main._ref_text.toPlainText(),
            "generation_params": self.main._get_generation_settings(),
            "normalize_audio": self._norm_check.isChecked(),
            "audiobook_silence_ms": self._audiobook_silence_spin.value() if self._audiobook_silence_spin is not None else 500,
            "audiobook_format": self._audiobook_fmt_combo.currentData() if self._audiobook_fmt_combo is not None else "wav",
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

    def _restore_ebook_session_data(self, data: dict):
        self._epub_path = data.get("epub_path", "")
        self._ebook_output_dir = data.get("ebook_output_dir", str(OUTPUTS_DIR))
        if self._audiobook_output_edit is not None:
            self._audiobook_output_edit.setText(self._ebook_output_dir)
        self._ebook_fragments = data.get("fragments", [])
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
        if self._audiobook_silence_spin is not None:
            self._audiobook_silence_spin.setValue(data.get("audiobook_silence_ms", 500))
        if self._audiobook_fmt_combo is not None:
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
        self.main._tabs.setCurrentIndex(1)
        self._populate_ebook_tree()
        self.main._update_action_buttons()

        self._update_preview_btn_state()

        status_msg = f"Ebook session loaded: {len(self._ebook_fragments)} fragments, {done} already synthesized."
        if missing:
            status_msg = f"Ebook session loaded — {missing} audio file(s) missing on disk, reset to waiting."
            self.main._set_status(status_msg, C["warning"])
        else:
            self.main._set_status(status_msg, C["success"])

    # ------------------------------------------------------------------
    # Public methods for MainWindow
    # ------------------------------------------------------------------

    def start_synthesis(self, retry_errors: bool = False):
        """Public entry point for synthesis."""
        self._start_ebook_synthesis(retry_errors)

    def stop_synthesis(self):
        """Public entry point to stop synthesis."""
        if self._worker:
            self._worker.request_cancel()
        self.main._set_status("Stop requested — waiting for current fragment to finish…", C["warning"])

    def has_fragments(self) -> bool:
        """Return True if any fragments are loaded."""
        return bool(self._ebook_fragments)

    def populate_tree(self):
        """Public method to refresh tree."""
        self._populate_ebook_tree()

    def update_tree_item_duration(self, idx: int, dur: float):
        """Update tree item with new duration."""
        self._update_ebook_tree_item(idx)

    def update_preview_btn_state(self):
        """Update preview button state."""
        self._update_preview_btn_state()