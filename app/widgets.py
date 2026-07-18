# widgets.py
import os
import numpy as np
import soundfile as sf
import sounddevice as sd
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from PyQt6.QtWidgets import (
    QWidget, QDialog, QFrame, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QSizePolicy, QCheckBox,
    QDialogButtonBox, QFileDialog, QTreeWidget, QTreeWidgetItem,
)
from PyQt6.QtCore import Qt, QTimer, QSize, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QDragEnterEvent, QDropEvent

from config import C, COL_STATUS, COL_FRAGMENT, COL_SPEAKER, COL_TIMING, STATUS_WAITING, STATUS_DONE, STATUS_ERROR
from utils import _fmt, _fmt_ms, _get_last_dir, _set_last_dir, _get_wav_duration, _compute_trim_bounds


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
        if len(audio) == 0:
            self._peaks = []
            self.update()
            return
        n     = min(len(audio) // 100, 8000)
        n     = max(1, n)
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
        self._overflow_end:    Optional[float] = None
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
        n     = min(len(audio) // 100, 8000)
        n     = max(1, n)
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
        self._overflow_end    = None
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

    def set_selection_fracs(self, start_frac: float, end_frac: float, overflow_end_frac: Optional[float] = None):
        self._sel_start = max(0.0, min(1.0, start_frac))
        self._sel_end   = max(0.0, min(1.0, end_frac))
        if overflow_end_frac is not None:
            self._overflow_end = max(0.0, min(1.0, overflow_end_frac))
        else:
            self._overflow_end = None
            
        if self._readonly and self._zoom > 1.0:
            mid    = (self._sel_start + self._sel_end) / 2.0
            view_w = 1.0 / self._zoom
            self._view_start = max(0.0, min(1.0 - view_w, mid - view_w / 2.0))
        self.update()

    def clear_selection(self):
        self._sel_start = None
        self._sel_end   = None
        self._overflow_end = None
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
            if self._readonly:
                return
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
            if self._overflow_end is not None:
                ox = int(self._frac_to_screen(self._overflow_end))
                if ox > ex:
                    p.fillRect(ex, 0, ox - ex, wave_h, QColor(255, 68, 68, 100))

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
        srt_tab = getattr(self.parent_win, "srt_tab", None)
        fragments = getattr(srt_tab, "_fragments", []) if srt_tab is not None else []
        for frag in fragments:
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
    playback_state_changed = pyqtSignal(bool)
    position_updated = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._path:    Optional[str]        = None
        self._data:    Optional[np.ndarray] = None
        self._sr       = 22050
        self._playing  = False
        self._cursor   = 0
        self._play_end = 0
        self._play_gen = 0
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
        self._play_btn.setMinimumWidth(80)
        self._play_btn.setEnabled(False)
        self._play_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._play_btn.clicked.connect(self._toggle)

        self._stop_btn = QPushButton("⏹  Stop")
        self._stop_btn.setFixedHeight(26)
        self._stop_btn.setMinimumWidth(80)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._stop_btn.clicked.connect(self._stop_now)

        self._lbl = QLabel("—")
        self._lbl.setStyleSheet(
            f"color:{C['text3']};font-size:10px;font-family:'Consolas',monospace;min-width:80px;"
        )
        
        vol_icon = QLabel("🔊")
        vol_icon.setStyleSheet("font-size:11px;background:transparent;border:none;")
        vol_icon.setFixedWidth(24)
        vol_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self._vol = QSlider(Qt.Orientation.Horizontal)
        self._vol.setRange(0, 150)
        self._vol.setValue(100)
        self._vol.setFixedWidth(90)
        self._vol.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        ctrl.addWidget(self._play_btn)
        ctrl.addWidget(self._stop_btn)
        ctrl.addSpacing(4)
        ctrl.addWidget(self._lbl)
        ctrl.addSpacing(8)
        ctrl.addWidget(vol_icon)
        ctrl.addWidget(self._vol)
        ctrl.addStretch()
        
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
            self._stop_btn.setEnabled(True)
            self.setVisible(True)
        except Exception as ex:
            self._lbl.setText(f"Error: {ex}")

    def clear(self):
        self._stop_now()
        self._data = None
        self._path = None
        self._wave.clear()
        self._play_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._lbl.setText("—")
        self.setVisible(False)

    def set_selection_by_time(self, start_s: float, end_s: float, overflow_end_s: Optional[float] = None):
        if self._data is None:
            return
        dur = len(self._data) / max(1, self._sr)
        if dur <= 0:
            return

        start_frac    = start_s / dur
        overflow_frac = overflow_end_s / dur if overflow_end_s is not None else None
        self._wave.set_selection_fracs(start_frac, end_s / dur, overflow_frac)

        self._seek_to_frac(start_frac)

        if overflow_end_s is not None and overflow_end_s > end_s:
            diff = overflow_end_s - end_s
            self._wave.setToolTip(f"Selection: {_fmt(start_s)} - {_fmt(end_s)}\nOverflow: +{diff:.1f}s")
        else:
            self._wave.setToolTip(f"Selection: {_fmt(start_s)} - {_fmt(end_s)}")

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

        s_sample = self._cursor if 0 <= self._cursor < len(audio) else 0
        e_sample = len(audio)
        self._cursor   = s_sample
        self._play_end = e_sample
        chunk = audio[s_sample:e_sample].copy()

        if not len(chunk):
            self._playing = False
            self._play_btn.setText("▶  Play")
            return

        self._play_gen += 1
        gen = self._play_gen

        def cb(out, frames, ti, st):
            nonlocal chunk
            if not self._playing or gen != self._play_gen:
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

        def _finished(gen=gen):
            self._on_end(gen)

        try:
            self._stream = sd.OutputStream(
                samplerate=self._sr, channels=1, dtype="float32",
                callback=cb, finished_callback=_finished,
            )
            self._stream.start()
            self._timer.start()
            self.playback_state_changed.emit(True)
        except Exception:
            self._playing = False
            self._play_btn.setText("▶  Play")

    def _pause(self):
        self._playing = False
        self._play_gen += 1
        self._play_btn.setText("▶  Play")
        self._timer.stop()
        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass
        self.playback_state_changed.emit(False)

    def _stop_now(self):
        self._playing = False
        self._play_gen += 1
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
        self.position_updated.emit(0.0)
        self.playback_state_changed.emit(False)

    def _on_end(self, gen: int = None):
        def _safe():
            if gen is not None and gen != self._play_gen:
                return
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
                self.position_updated.emit(t)
            self.playback_state_changed.emit(False)
        QTimer.singleShot(0, _safe)

    def _tick(self):
        if self._data is None:
            return
        total = len(self._data)
        self._wave.set_position(min(self._cursor, total) / max(1, total))
        t = self._cursor / max(1, self._sr)
        self._lbl.setText(f"{_fmt(t)} / {_fmt(total / max(1, self._sr))}")
        self.position_updated.emit(t)

    def _seek(self, frac: float):
        self._seek_to_frac(frac)

    def _seek_to_frac(self, frac: float):
        if self._data is None:
            return
        was_playing = self._playing
        if was_playing:
            self._pause()
        self._cursor = int(frac * len(self._data))
        self._wave.set_position(frac)
        self.position_updated.emit(self._cursor / max(1, self._sr))
        if was_playing:
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