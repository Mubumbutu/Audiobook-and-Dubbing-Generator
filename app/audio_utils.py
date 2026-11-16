# audio_utils.py
import os
import logging
import threading
from typing import Optional, List, Dict, Tuple, Callable

import numpy as np
import soundfile as sf
import sounddevice as sd
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QMessageBox

from config import C, _fmt, COL_STATUS
from utils import _compute_trim_bounds, _get_wav_duration

logger = logging.getLogger(__name__)

_FADE_SECONDS = 0.01
_STREAM_LATENCY = 0.05


class _PlaybackEngine:
    def __init__(self, sr: int):
        self.sr = int(sr)
        self.volume = 1.0
        self.finished = False
        self._lock = threading.Lock()
        self._silent = threading.Event()
        self._silent.set()
        n = max(8, int(self.sr * _FADE_SECONDS))
        t = (np.arange(n, dtype=np.float32) + 0.5) / n
        self._fade_in = (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)
        self._fade_out = self._fade_in[::-1].copy()
        self._fade_len = n
        self._audio = None
        self._pos = 0
        self._start = 0
        self._end = 0
        self._fade_in_pos = 0
        self._stop_pos = -1
        self._pending = None
        self._gain = 1.0
        self._latency = 0
        self._stream = None

    def open(self):
        last_error = None
        for latency in (_STREAM_LATENCY, None):
            kwargs = dict(samplerate=self.sr, channels=1, dtype="float32",
                          callback=self._callback)
            if latency is not None:
                kwargs["latency"] = latency
            stream = None
            try:
                stream = sd.OutputStream(**kwargs)
                stream.start()
            except Exception as e:
                last_error = e
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
                continue
            self._stream = stream
            try:
                self._latency = int(float(stream.latency) * self.sr)
            except Exception:
                self._latency = 0
            return
        raise last_error

    def is_alive(self) -> bool:
        s = self._stream
        try:
            return s is not None and not s.closed and s.active
        except Exception:
            return False

    def close(self):
        self.stop()
        self._silent.wait(0.05)
        s = self._stream
        self._stream = None
        if s is not None:
            try:
                s.stop()
            except Exception:
                pass
            try:
                s.close()
            except Exception:
                pass

    def play(self, audio: np.ndarray, start: int, end: int):
        end = max(0, min(int(end), len(audio)))
        start = max(0, min(int(start), end))
        with self._lock:
            job = (audio, start, end)
            if self._audio is None:
                self._gain = self.volume
                self._install(job)
            else:
                self._pending = job
                if self._stop_pos < 0:
                    self._stop_pos = 0

    def stop(self):
        with self._lock:
            self._pending = None
            if self._audio is not None and self._stop_pos < 0:
                self._stop_pos = 0

    def position(self) -> int:
        return max(self._start, min(self._pos, self._end) - self._latency)

    def _install(self, job):
        self._audio, self._pos, self._end = job
        self._start = self._pos
        self._fade_in_pos = 0
        self._stop_pos = -1
        self._pending = None
        self.finished = False
        self._silent.clear()

    def _finish(self):
        natural = self._stop_pos < 0
        pending = self._pending
        self._audio = None
        self._pending = None
        self._stop_pos = -1
        if pending is not None:
            self._install(pending)
        else:
            self.finished = natural
            self._silent.set()

    def _callback(self, outdata, frames, time_info, status):
        outdata.fill(0.0)
        with self._lock:
            filled = 0
            while filled < frames and self._audio is not None:
                n = min(frames - filled, self._end - self._pos)
                if self._stop_pos >= 0:
                    n = min(n, self._fade_len - self._stop_pos)
                if n <= 0:
                    self._finish()
                    continue
                seg = self._audio[self._pos:self._pos + n]
                env = np.empty(n, dtype=np.float32)
                target = self.volume
                if target != self._gain:
                    env[:] = np.linspace(self._gain, target, n + 1, dtype=np.float32)[1:]
                    self._gain = target
                else:
                    env[:] = target
                if self._fade_in_pos < self._fade_len:
                    k = min(n, self._fade_len - self._fade_in_pos)
                    env[:k] *= self._fade_in[self._fade_in_pos:self._fade_in_pos + k]
                    self._fade_in_pos += k
                remaining = self._end - self._pos
                if remaining - n < self._fade_len:
                    j0 = max(0, remaining - self._fade_len)
                    rem = remaining - np.arange(j0, n)
                    env[j0:] *= self._fade_in[rem - 1]
                if self._stop_pos >= 0:
                    env *= self._fade_out[self._stop_pos:self._stop_pos + n]
                    self._stop_pos += n
                outdata[filled:filled + n, 0] = seg * env
                self._pos += n
                filled += n


def _get_engine(main_window, sr: int) -> "_PlaybackEngine":
    engine = getattr(main_window, '_engine', None)
    if engine is not None and (engine.sr != int(sr) or not engine.is_alive()):
        engine.close()
        engine = None
        main_window._engine = None
    if engine is None:
        engine = _PlaybackEngine(sr)
        engine.open()
        main_window._engine = engine
        main_window._stream = engine._stream
    return engine


def _engine_stop(main_window):
    engine = getattr(main_window, '_engine', None)
    if engine is not None:
        engine.stop()


def release_player(main_window):
    engine = getattr(main_window, '_engine', None)
    main_window._engine = None
    main_window._stream = None
    if engine is not None:
        engine.close()

def load_audio_to_player(main_window, audio: np.ndarray, sr: int,
                          fragment_idx: Optional[int] = None):
    main_window._playing = False
    main_window._play_btn.setText("▶ Play")
    main_window._timer.stop()
    _engine_stop(main_window)

    main_window._audio_data = audio
    main_window._audio_sr = sr
    main_window._current_fragment_idx = fragment_idx
    main_window._play_end_sample = len(audio)

    main_window._wave_out.set_audio(audio)
    main_window._wave_out.clear_selection()
    main_window._wave_out.clear_trim_preview()
    main_window._wave_out.set_position(0.0)
    main_window._cursor = 0

    try:
        _get_engine(main_window, sr)
    except Exception:
        pass

    dur = len(audio) / max(1, sr)
    main_window._time_lbl.setText(f"0:00 / {_fmt(dur)}")

    main_window._play_btn.setEnabled(True)
    main_window._stop_btn.setEnabled(True)

    tab = main_window._tabs.currentIndex() if hasattr(main_window, '_tabs') else 0
    if tab == 1:
        main_window.ebook_tab._ebook_tree.setFocus()
    else:
        main_window.srt_tab._tree.setFocus()

    main_window._update_trim_preview()


def toggle_play(main_window):
    if main_window._playing:
        main_window._pause_play()
    else:
        main_window._start_play()


def start_play(main_window):
    if main_window._audio_data is None:
        return

    audio = main_window._audio_data.astype(np.float32)

    aggressiveness = main_window._trim_slider.value() / 10.0 if hasattr(main_window, '_trim_slider') else 0.0
    lead_s = 0
    trail_s = 0
    if aggressiveness > 0.0:
        _, _, lead_s, trail_s = _compute_trim_bounds(audio, main_window._audio_sr, aggressiveness)

    trimmed_end = len(audio) - trail_s

    sel = main_window._wave_out.get_selection()
    if sel:
        sel_start = int(sel[0] * len(audio))
        sel_end = int(sel[1] * len(audio))
        if sel_start <= main_window._cursor < sel_end:
            s_sample = main_window._cursor
        else:
            s_sample = sel_start
        e_sample = sel_end
    else:
        s_sample = max(lead_s, main_window._cursor) if main_window._cursor < trimmed_end else lead_s
        e_sample = trimmed_end

    if s_sample >= e_sample:
        s_sample = lead_s
        e_sample = trimmed_end

    if e_sample - s_sample <= 0:
        main_window._playing = False
        main_window._play_btn.setText("▶  Play")
        return

    try:
        engine = _get_engine(main_window, main_window._audio_sr)
        engine.volume = main_window._vol.value() / 100.0
        engine.play(audio, s_sample, e_sample)
    except Exception as e:
        main_window._playing = False
        main_window._play_btn.setText("▶  Play")
        main_window._set_status(f"Playback error: {e}", C["error"])
        return

    main_window._cursor = s_sample
    main_window._play_end_sample = e_sample
    main_window._playing = True
    main_window._play_btn.setText("■  Pause")
    main_window._timer.start()


def pause_play(main_window):
    engine = getattr(main_window, '_engine', None)
    if engine is not None and main_window._playing:
        main_window._cursor = engine.position()
        engine.stop()
    main_window._playing = False
    main_window._play_btn.setText("▶  Play")
    main_window._timer.stop()
    if main_window._audio_data is not None and hasattr(main_window, '_wave_out'):
        total = len(main_window._audio_data)
        main_window._wave_out.set_position(min(main_window._cursor, total) / max(1, total))


def stop_play(main_window):
    main_window._playing = False
    main_window._play_btn.setText("▶  Play")
    main_window._timer.stop()
    _engine_stop(main_window)
    if main_window._audio_data is not None and hasattr(main_window, '_wave_out'):
        sel = main_window._wave_out.get_selection()
        if sel:
            main_window._cursor = int(sel[0] * len(main_window._audio_data))
        else:
            aggressiveness = main_window._trim_slider.value() / 10.0 if hasattr(main_window, '_trim_slider') else 0.0
            lead_s = 0
            if aggressiveness > 0.0:
                _, _, lead_s, _ = _compute_trim_bounds(
                    main_window._audio_data.astype(np.float32), main_window._audio_sr, aggressiveness
                )
            main_window._cursor = lead_s
        main_window._wave_out.set_position(main_window._cursor / max(1, len(main_window._audio_data)))
        main_window._time_lbl.setText(f"0:00 / {_fmt(len(main_window._audio_data) / max(1, main_window._audio_sr))}")
    else:
        main_window._cursor = 0


def on_play_end(main_window):
    engine = getattr(main_window, '_engine', None)
    if engine is not None:
        engine.finished = False
    main_window._playing = False
    main_window._play_btn.setText("▶  Play")
    main_window._timer.stop()
    if main_window._audio_data is not None and hasattr(main_window, '_wave_out'):
        sel = main_window._wave_out.get_selection()
        if sel:
            main_window._cursor = int(sel[0] * len(main_window._audio_data))
        else:
            aggressiveness = main_window._trim_slider.value() / 10.0 if hasattr(main_window, '_trim_slider') else 0.0
            lead_s = 0
            if aggressiveness > 0.0:
                _, _, lead_s, _ = _compute_trim_bounds(
                    main_window._audio_data.astype(np.float32), main_window._audio_sr, aggressiveness
                )
            main_window._cursor = lead_s
        main_window._wave_out.set_position(main_window._cursor / max(1, len(main_window._audio_data)))
        main_window._time_lbl.setText(
            f"{_fmt(main_window._cursor / max(1, main_window._audio_sr))} / "
            f"{_fmt(len(main_window._audio_data) / max(1, main_window._audio_sr))}"
        )


def playback_tick(main_window):
    if main_window._audio_data is None:
        return
    engine = getattr(main_window, '_engine', None)
    if engine is not None and main_window._playing:
        engine.volume = main_window._vol.value() / 100.0
        if engine.finished:
            on_play_end(main_window)
            return
        main_window._cursor = engine.position()
    total = len(main_window._audio_data)
    pos   = min(main_window._cursor, total) / max(1, total)
    if hasattr(main_window, '_wave_out'):
        main_window._wave_out.set_position(pos)
    main_window._time_lbl.setText(
        f"{_fmt(main_window._cursor / max(1, main_window._audio_sr))} / "
        f"{_fmt(total / max(1, main_window._audio_sr))}"
    )


def seek_audio(main_window, frac: float):
    if main_window._audio_data is None:
        return
    main_window._pause_play()
    main_window._cursor = int(frac * len(main_window._audio_data))
    main_window._wave_out.set_position(frac)
    main_window._start_play()


def _get_fragment_list(main_window, is_ebook):
    if is_ebook:
        return main_window.ebook_tab._ebook_fragments
    else:
        return main_window.srt_tab._fragments


def _get_current_fragment(main_window):
    if main_window._current_fragment_idx is None:
        return None, False
    tab = main_window._tabs.currentIndex()
    if tab == 1:
        frag = next((f for f in main_window.ebook_tab._ebook_fragments if f['index'] == main_window._current_fragment_idx), None)
        return frag, True
    else:
        frag = next((f for f in main_window.srt_tab._fragments if f['index'] == main_window._current_fragment_idx), None)
        return frag, False


def delete_selected_audio_segment(main_window, start_frac: float, end_frac: float):
    if main_window._audio_data is None:
        return
    length       = len(main_window._audio_data)
    start_sample = int(start_frac * length)
    end_sample   = int(end_frac   * length)
    if end_sample - start_sample < 10:
        return

    main_window.undo_manager.push('audio', "audio segment deletion", build_audio_undo_state(main_window))

    new_audio  = np.concatenate((
        main_window._audio_data[:start_sample],
        main_window._audio_data[end_sample:]
    )).astype(np.float32)
    new_dur_s  = len(new_audio) / max(1, main_window._audio_sr)
    deleted_s  = (end_sample - start_sample) / max(1, main_window._audio_sr)

    main_window._audio_data = new_audio

    if hasattr(main_window, '_wave_out'):
        main_window._wave_out.set_audio(main_window._audio_data)
        main_window._wave_out.clear_selection()
    main_window._cursor = min(main_window._cursor, len(main_window._audio_data))

    frag, is_ebook = _get_current_fragment(main_window)
    if frag:
        if frag.get('output_path') and os.path.exists(frag['output_path']):
            try:
                sf.write(frag['output_path'], main_window._audio_data, main_window._audio_sr, subtype="PCM_16")
            except Exception:
                pass
        if is_ebook:
            main_window.ebook_tab._update_ebook_tree_item(main_window._current_fragment_idx)
        else:
            main_window.srt_tab._update_tree_item(main_window._current_fragment_idx, known_dur_s=new_dur_s)

    if hasattr(main_window, '_audio_tmp') and main_window._audio_tmp and os.path.exists(main_window._audio_tmp):
        try:
            sf.write(main_window._audio_tmp, main_window._audio_data, main_window._audio_sr, subtype="PCM_16")
        except Exception:
            pass

    if hasattr(main_window, '_time_lbl'):
        dur = len(main_window._audio_data) / max(1, main_window._audio_sr)
        main_window._time_lbl.setText(f"0:00 / {_fmt(dur)}")

    main_window._update_trim_preview()

    main_window._set_status(
        f"Deleted {deleted_s:.2f}s — audio updated. Ctrl+Z to undo.", C["accent"]
    )


def mute_selected_audio_segment(main_window, start_frac: float, end_frac: float):
    if main_window._audio_data is None:
        return
    length       = len(main_window._audio_data)
    start_sample = int(start_frac * length)
    end_sample   = int(end_frac   * length)
    if end_sample - start_sample < 10:
        return

    main_window.undo_manager.push('audio', "audio segment mute", build_audio_undo_state(main_window))

    main_window._audio_data = main_window._audio_data.copy()
    main_window._audio_data[start_sample:end_sample] = 0.0
    muted_s = (end_sample - start_sample) / max(1, main_window._audio_sr)

    if hasattr(main_window, '_wave_out'):
        main_window._wave_out.set_audio(main_window._audio_data)
        main_window._wave_out.clear_selection()
    main_window._cursor = min(main_window._cursor, len(main_window._audio_data))

    frag, is_ebook = _get_current_fragment(main_window)
    if frag and frag.get('output_path') and os.path.exists(frag['output_path']):
        try:
            sf.write(frag['output_path'], main_window._audio_data, main_window._audio_sr, subtype="PCM_16")
            if is_ebook:
                main_window.ebook_tab._update_ebook_tree_item(main_window._current_fragment_idx)
            else:
                main_window.srt_tab._update_tree_item(main_window._current_fragment_idx)
        except Exception:
            pass

    if hasattr(main_window, '_audio_tmp') and main_window._audio_tmp and os.path.exists(main_window._audio_tmp):
        try:
            sf.write(main_window._audio_tmp, main_window._audio_data, main_window._audio_sr, subtype="PCM_16")
        except Exception:
            pass

    if hasattr(main_window, '_time_lbl'):
        dur = len(main_window._audio_data) / max(1, main_window._audio_sr)
        main_window._time_lbl.setText(f"0:00 / {_fmt(dur)}")

    main_window._update_trim_preview()

    main_window._set_status(
        f"Audio segment muted ({muted_s:.2f}s silenced). Ctrl+Z to undo.", C["accent"]
    )


def apply_trim_to_fragment(main_window):
    if main_window._audio_data is None:
        return
    aggressiveness = main_window._trim_slider.value() / 10.0
    if aggressiveness <= 0.0:
        return
    lead_ms, trail_ms, lead_s, trail_s = _compute_trim_bounds(
        main_window._audio_data, main_window._audio_sr, aggressiveness
    )
    if lead_ms == 0 and trail_ms == 0:
        main_window._set_status("Nothing to trim at current aggressiveness.", C["text2"])
        return

    total     = len(main_window._audio_data)
    end_s     = total - trail_s if trail_s > 0 else total
    new_audio = main_window._audio_data[lead_s:end_s].astype(np.float32)

    if len(new_audio) < int(main_window._audio_sr * 0.05):
        main_window._set_status("Aggressiveness too high — would trim everything. Reduce it.", C["warning"])
        return

    main_window.undo_manager.push('audio', "audio trim", build_audio_undo_state(main_window))

    new_dur_s        = len(new_audio) / max(1, main_window._audio_sr)
    main_window._audio_data = new_audio
    main_window._wave_out.set_audio(main_window._audio_data)
    main_window._wave_out.clear_trim_preview()
    main_window._cursor = 0

    frag, is_ebook = _get_current_fragment(main_window)
    if frag:
        if frag.get('output_path') and os.path.exists(frag['output_path']):
            try:
                sf.write(frag['output_path'], main_window._audio_data, main_window._audio_sr, subtype="PCM_16")
            except Exception:
                pass
        if is_ebook:
            main_window.ebook_tab._update_ebook_tree_item(main_window._current_fragment_idx)
        else:
            main_window.srt_tab._update_tree_item(main_window._current_fragment_idx, known_dur_s=new_dur_s)

    if hasattr(main_window, '_audio_tmp') and main_window._audio_tmp and os.path.exists(main_window._audio_tmp):
        try:
            sf.write(main_window._audio_tmp, main_window._audio_data, main_window._audio_sr, subtype="PCM_16")
        except Exception:
            pass

    main_window._time_lbl.setText(f"0:00 / {_fmt(new_dur_s)}")
    main_window._update_trim_preview()
    main_window._set_status(
        f"Trimmed — lead: {lead_ms} ms, trail: {trail_ms} ms. Ctrl+Z to undo.", C["accent"]
    )


def apply_trim_to_selected(main_window):
    aggressiveness = main_window._trim_slider.value() / 10.0
    if aggressiveness <= 0.0:
        return

    tab      = main_window._tabs.currentIndex()
    is_ebook = tab == 1
    frags    = main_window.ebook_tab._ebook_fragments if is_ebook else main_window.srt_tab._fragments
    chapter  = main_window.ebook_tab._ebook_chapter_item if is_ebook else main_window.srt_tab._chapter_item

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
            main_window, "Nothing to trim",
            "No synthesized (done) fragments are checked.\n"
            "Check at least one completed fragment first."
        )
        return

    reply = QMessageBox.question(
        main_window, "Apply trim to all selected",
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
            if frag.get('index') == main_window._current_fragment_idx:
                main_window.undo_manager.push('audio', "audio trim", build_audio_undo_state(main_window))
                main_window._audio_data = new_audio
                main_window._audio_sr   = sr
                main_window._wave_out.set_audio(main_window._audio_data)
                main_window._cursor = 0
                main_window._time_lbl.setText(f"0:00 / {_fmt(new_dur_s)}")
            if is_ebook:
                main_window.ebook_tab._update_ebook_tree_item(frag['index'])
            else:
                main_window.srt_tab._update_tree_item(frag['index'], known_dur_s=new_dur_s)
            trimmed += 1
        except Exception as e:
            logger.warning(f"Trim failed for fragment {frag.get('index')}: {e}")

    main_window._wave_out.clear_trim_preview()
    main_window._update_trim_preview()
    main_window._set_status(
        f"Trim applied to {trimmed}/{len(checked_done)} fragment(s).", C["success"]
    )


def on_trim_slider_changed(main_window, value: int):
    val = value / 100.0

    if abs(main_window._trim_input.value() - val) > 0.001:
        main_window._trim_input.blockSignals(True)
        main_window._trim_input.setValue(val)
        main_window._trim_input.blockSignals(False)

    main_window._update_trim_preview()


def update_trim_preview(main_window):
    if not hasattr(main_window, '_trim_slider'):
        return
    aggressiveness = main_window._trim_slider.value() / 10.0
    if main_window._audio_data is None or aggressiveness <= 0.0:
        if hasattr(main_window, '_wave_out'):
            main_window._wave_out.clear_trim_preview()
        if hasattr(main_window, '_trim_preview_lbl'):
            main_window._trim_preview_lbl.setText("")
        if hasattr(main_window, '_trim_apply_all_btn'):
            main_window._trim_apply_all_btn.setEnabled(False)
        return

    lead_ms, trail_ms, lead_s, trail_s = _compute_trim_bounds(
        main_window._audio_data, main_window._audio_sr, aggressiveness
    )
    total = len(main_window._audio_data)
    if total == 0:
        return

    main_window._wave_out.set_trim_preview(lead_s / total, trail_s / total)

    new_samples = max(0, total - lead_s - trail_s)
    new_dur_s   = new_samples / max(1, main_window._audio_sr)
    main_window._trim_preview_lbl.setText(
        f"Lead: {lead_ms} ms | Trail: {trail_ms} ms  →  {new_dur_s:.2f}s"
    )

    has_trim = lead_ms > 0 or trail_ms > 0
    main_window._trim_apply_all_btn.setEnabled(has_trim)


def build_audio_undo_state(main_window) -> dict:
    frag, is_ebook = _get_current_fragment(main_window)
    return {
        'audio':         main_window._audio_data.copy(),
        'sr':            main_window._audio_sr,
        'frag_idx':      main_window._current_fragment_idx,
        'old_end_ms':    frag.get('end_ms')    if frag else None,
        'old_timestamp': frag.get('timestamp') if frag else None,
        'is_ebook':      is_ebook,
    }


def restore_audio_state(main_window, state: dict):
    audio    = state['audio']
    sr       = state['sr']
    frag_idx = state.get('frag_idx')
    is_ebook = state.get('is_ebook', False)
    old_end_ms    = state.get('old_end_ms')
    old_timestamp = state.get('old_timestamp')

    main_window._audio_data = audio
    main_window._audio_sr   = sr

    if hasattr(main_window, '_wave_out'):
        main_window._wave_out.set_audio(audio)
        main_window._wave_out.clear_selection()
        main_window._wave_out.clear_trim_preview()
    main_window._cursor = min(main_window._cursor, len(audio))

    if frag_idx is not None and old_end_ms is not None:
        frags = main_window.ebook_tab._ebook_fragments if is_ebook else main_window.srt_tab._fragments
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
                main_window.ebook_tab._update_ebook_tree_item(frag_idx)
            else:
                main_window.srt_tab._update_tree_item(frag_idx, known_dur_s=new_dur_s)

    if hasattr(main_window, '_audio_tmp') and main_window._audio_tmp and os.path.exists(main_window._audio_tmp):
        try:
            sf.write(main_window._audio_tmp, audio, sr, subtype="PCM_16")
        except Exception:
            pass

    if hasattr(main_window, '_time_lbl'):
        dur = len(audio) / max(1, sr)
        main_window._time_lbl.setText(f"0:00 / {_fmt(dur)}")

    main_window._update_trim_preview()


def play_fragment_audio(main_window, frag: Dict):
    path = frag.get('output_path', '')
    if not path or not os.path.exists(path):
        return
    try:
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        load_audio_to_player(main_window, audio, sr, fragment_idx=frag['index'])
    except Exception as e:
        main_window._set_status(f"Playback error: {e}", C["error"])
