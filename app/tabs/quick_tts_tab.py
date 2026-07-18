# tabs/quick_tts_tab.py
import os
import tempfile
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QPlainTextEdit, QLineEdit, QGroupBox, QFileDialog,
    QProgressBar, QCheckBox, QComboBox, QDialog, QMessageBox,
    QDialogButtonBox, QSplitter, QScrollArea, QFrame,
    QTextEdit,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QIcon, QAction, QColor, QShortcut, QKeySequence

from .base_tab import BaseTab
from config import C, OUTPUTS_DIR, SYNTH_BTN_STYLE, _btn
from utils import _get_last_dir, _set_last_dir

logger = logging.getLogger(__name__)


class QuickTTSTab(BaseTab):
    """Quick single‑text TTS synthesis tab."""

    def __init__(self, main_window):
        super().__init__(main_window)

        self._audio_tmp: Optional[str] = None
        self._audio_data: Optional[np.ndarray] = None
        self._audio_sr: int = 44100
        self._w_gen = None

        # Współdzielone widżety – zostaną ustawione przez set_shared_widgets()
        self._progress = None
        self._btn_start = None
        self._btn_stop = None
        self._synth_progress = None
        self._eta_label = None
        self._norm_check = None

        self._quick_btn = None
        self._save_wav_btn = None
        self._save_mp3_btn = None
        self._quick_edit = None
        self._char_lbl = None

        self._build_ui()
        
    def set_shared_widgets(self, btn_start, btn_stop, synth_progress, eta_label, progress, norm_check):
        """Ustawia widżety współdzielone z MainWindow."""
        self._btn_start = btn_start
        self._btn_stop = btn_stop
        self._synth_progress = synth_progress
        self._eta_label = eta_label
        self._progress = progress
        self._norm_check = norm_check

    def get_widget(self) -> QWidget:
        return self._quick_tab_widget

    def _build_ui(self):
        self._quick_tab_widget = self._make_quick_tts_tab()

    # ------------------------------------------------------------------
    # Quick TTS UI
    # ------------------------------------------------------------------

    def _make_quick_tts_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        tg = QGroupBox("Input text")
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
            lambda: self._synthesize_quick_tts() if self.main._tabs.currentIndex() == 2 else None
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

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesize(self):
        """Public entry point for synthesis (called from core)."""
        self._synthesize_quick_tts()

    def _synthesize_quick_tts(self):
        if not self.main._model_is_loaded:
            QMessageBox.warning(self.main, "Model not loaded", "Load the model first.")
            return

        text = self._quick_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self.main, "Empty text", "Enter text to synthesize.")
            self._quick_edit.setFocus()
            return

        from workers import GenerateWorker

        self._quick_btn.setEnabled(False)
        self._quick_btn.setText("  ⏳  Synthesizing…")
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self.main._stop_play()

        self._w_gen = GenerateWorker(
            self.main._backend,
            text,
            self.main._get_ref_audio(),
            self.main._get_ref_text(),
            self.main._get_generation_settings(),
        )
        self._w_gen.status.connect(lambda m: self.main._set_status(m))
        self._w_gen.finished.connect(self._on_quick_tts_done)
        self._w_gen.error.connect(lambda e: self.main._on_error("Generation error", e,
                                                                 reset_fn=lambda: (
                                                                     self._quick_btn.setEnabled(True),
                                                                     self._quick_btn.setText("  🚀  Synthesize"),
                                                                     self._progress.setVisible(False),
                                                                 )))
        self._w_gen.start()

    def _on_quick_tts_done(self, audio: np.ndarray, sr: int):
        self._quick_btn.setEnabled(True)
        self._quick_btn.setText("  🚀  Synthesize")
        self._progress.setVisible(False)
        self.main._load_audio_to_player(audio, sr)
        self._save_wav_btn.setEnabled(True)
        self._save_mp3_btn.setEnabled(True)
        dur = len(audio) / sr
        self.main._set_status(f"Generated — {dur:.1f}s | {sr}Hz", C["success"])
        if self._audio_tmp and Path(self._audio_tmp).exists():
            try:
                os.unlink(self._audio_tmp)
            except Exception:
                pass
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            self._audio_tmp = f.name
        sf.write(self._audio_tmp, audio, sr, subtype="PCM_16")

    # ------------------------------------------------------------------
    # Save Audio
    # ------------------------------------------------------------------

    def _save_audio(self, fmt: str = "wav"):
        if self.main._audio_data is None:
            return

        default_path = str(Path(_get_last_dir("output", str(OUTPUTS_DIR))) / f"audio.{fmt}")
        path, _ = QFileDialog.getSaveFileName(
            self.main, "Save audio", default_path,
            f"Audio {fmt.upper()} (*.{fmt});;All files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(f".{fmt}"):
            path = f"{path}.{fmt}"

        _set_last_dir("output", path)

        audio = np.asarray(self.main._audio_data, dtype=np.float32)
        sr = int(self.main._audio_sr)

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
                    self.main._set_status(f"pydub not available — saved as WAV: {wav_path}", C["warning"])
                    return
            self.main._set_status(f"Saved: {path}", C["success"])
        except Exception as e:
            QMessageBox.critical(self.main, "Save error", str(e))