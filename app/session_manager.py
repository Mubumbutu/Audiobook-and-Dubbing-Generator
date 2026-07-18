# session_manager.py
import logging
import os
import json
from pathlib import Path
from typing import Optional, Dict, List

from PyQt6.QtWidgets import QMessageBox, QSlider, QSpinBox

from config import OUTPUTS_DIR, C, _btn
from utils import _ref_audio_hash, _ms_to_ts

logger = logging.getLogger(__name__)


def _auto_session_path(main_window) -> Optional[Path]:
    if not main_window._srt_path:
        return None
    return OUTPUTS_DIR / Path(main_window._srt_path).stem / "session.json"


def _build_session_data(main_window) -> dict:
    freq: Dict[str, int] = {}
    for f in main_window.srt_tab._fragments:
        spk = f.get("speaker")
        if spk and str(spk).strip():
            k = str(spk).strip()
            freq[k] = freq.get(k, 0) + 1
    sorted_speaker_list = sorted(
        main_window._speaker_list,
        key=lambda s: freq.get(s, 0),
        reverse=True,
    )

    is_supertonic = main_window._is_supertonic_active()
    is_piper      = main_window._is_piper_active()
    speaker_voices_data = {}
    for spk in sorted_speaker_list:
        sv = main_window._speaker_voices.get(spk, {})
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
                "audio_hash": _ref_audio_hash(audio_path),
                "ref_text":   ref_text_wdg.toPlainText() if ref_text_wdg else "",
                "demucs":     demucs_chk.isChecked() if demucs_chk else False,
            }

    ref_audio = main_window._drop.file_path
    return {
        "version":              3,
        "srt_path":             main_window._srt_path,
        "output_dir":           main_window._output_dir,
        "reference_audio":      ref_audio,
        "reference_audio_hash": _ref_audio_hash(ref_audio),
        "reference_text":       main_window._ref_text.toPlainText(),
        "generation_params":    main_window._get_generation_settings(),
        "normalize_audio":      main_window._norm_check.isChecked(),
        "target_sr":            main_window._sr_combo.currentData(),
        "whisper_size":         main_window._w_size.currentData(),
        "whisper_lang":         main_window._w_lang.currentData(),
        "lektor": {
            "video_path":          main_window._vid_path_edit.text().strip() or None,
            "audio_path":          main_window._audio_path_edit.text().strip() or None,
            "offset_ms":           main_window._offset_spin.value(),
            "lektor_vol":          main_window._lektor_vol.value(),
            "orig_vol":            main_window._orig_vol.value(),
            "autofit":             main_window._autofit_check.isChecked(),
            "atempo_threshold":    main_window._atempo_threshold.value(),
            "ducking":             main_window._duck_check.isChecked(),
            "vocal_suppress":      main_window._vocal_suppress_check.isChecked(),
            "vocal_suppress_vol":  main_window._vocal_suppress_spin.value(),
            "keep_original_track": main_window._keep_original_track_check.isChecked(),
            "dubbed_lang":         main_window._dubbed_lang_edit.text().strip(),
            "video_format":        main_window._lektor_vid_fmt_combo.currentData(),
        },
        "dubbing_mode":       main_window._dubbing_mode,
        "dubbing_video_path": main_window._dubbing_video_path,
        "speaker_list":       sorted_speaker_list,
        "speaker_voices":     speaker_voices_data,
        "fragments": [
            {k: v for k, v in f.items()}
            for f in main_window.srt_tab._fragments
        ],
    }


def _write_session_to(main_window, path: str) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_build_session_data(main_window), fh, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.warning(f"Session write failed: {e}")
        return False


def _restore_session_data(main_window, data: dict):
    main_window._srt_path   = data.get("srt_path", "")
    main_window._output_dir = data.get("output_dir", str(OUTPUTS_DIR))
    main_window.srt_tab._fragments  = data.get("fragments", [])

    for f in main_window.srt_tab._fragments:
        start_ms = f.get("start_ms") or 0
        end_ms   = f.get("end_ms") or 0
        f["timestamp"] = f"{_ms_to_ts(start_ms)} --> {_ms_to_ts(end_ms)}"

    missing = 0
    for f in main_window.srt_tab._fragments:
        if (f.get("status") == "done"
                and f.get("output_path")
                and not os.path.exists(f.get("output_path", ""))):
            f["status"]      = "waiting"
            f["output_path"] = None
            missing += 1

    ref_audio  = data.get("reference_audio")
    saved_hash = data.get("reference_audio_hash")
    if ref_audio and os.path.exists(ref_audio):
        current_hash = _ref_audio_hash(ref_audio)
        if saved_hash and current_hash and current_hash != saved_hash:
            QMessageBox.warning(
                main_window, "Reference audio changed",
                f"The reference audio file has changed since the session was saved:\n"
                f"{ref_audio}\n\nFragments marked as done may not match the current voice.",
            )
        main_window._drop._set(ref_audio)
        main_window._ref_player.load(ref_audio)
    elif ref_audio:
        main_window._set_status(
            f"Reference audio not found: {Path(ref_audio).name}", C["warning"]
        )

    main_window._ref_text.setPlainText(data.get("reference_text", ""))

    params = data.get("generation_params", {})
    for key, value in params.items():
        widget = main_window._param_widgets.get(key) if hasattr(main_window, "_param_widgets") else None
        if widget is None:
            continue
        if isinstance(widget, QSlider):
            widget.setValue(int(value * 100))
        elif isinstance(widget, QSpinBox):
            widget.setValue(int(value))

    main_window._norm_check.setChecked(data.get("normalize_audio", False))

    target_sr = data.get("target_sr")
    if target_sr is not None:
        for i in range(main_window._sr_combo.count()):
            if main_window._sr_combo.itemData(i) == target_sr:
                main_window._sr_combo.setCurrentIndex(i)
                break

    whisper_size = data.get("whisper_size")
    if whisper_size:
        for i in range(main_window._w_size.count()):
            if main_window._w_size.itemData(i) == whisper_size:
                main_window._w_size.setCurrentIndex(i)
                break

    whisper_lang = data.get("whisper_lang")
    if whisper_lang:
        for i in range(main_window._w_lang.count()):
            if main_window._w_lang.itemData(i) == whisper_lang:
                main_window._w_lang.setCurrentIndex(i)
                break

    lektor = data.get("lektor", {})
    vid_path = lektor.get("video_path") or ""
    main_window._vid_path_edit.setText(vid_path)
    main_window._update_video_info()
    ext_audio_path = lektor.get("audio_path") or ""
    main_window._audio_path_edit.setText(ext_audio_path)
    main_window._update_audio_info()
    main_window._offset_spin.setValue(lektor.get("offset_ms", 0))
    main_window._lektor_vol.setValue(lektor.get("lektor_vol", 100))
    main_window._orig_vol.setValue(lektor.get("orig_vol", 100))
    main_window._autofit_check.setChecked(lektor.get("autofit", False))
    main_window._atempo_threshold.setValue(lektor.get("atempo_threshold", 300))
    main_window._duck_check.setChecked(lektor.get("ducking", False))
    main_window._vocal_suppress_check.setChecked(lektor.get("vocal_suppress", False))
    main_window._vocal_suppress_spin.setValue(lektor.get("vocal_suppress_vol", 20))
    main_window._vocal_suppress_spin.setEnabled(
        lektor.get("vocal_suppress", False) and main_window._ffmpeg_ok
    )
    main_window._keep_original_track_check.setChecked(lektor.get("keep_original_track", False))
    main_window._dubbed_lang_edit.setText(lektor.get("dubbed_lang", ""))
    video_format = lektor.get("video_format")
    if video_format is not None:
        for i in range(main_window._lektor_vid_fmt_combo.count()):
            if main_window._lektor_vid_fmt_combo.itemData(i) == video_format:
                main_window._lektor_vid_fmt_combo.setCurrentIndex(i)
                break

    dubbing_mode = data.get("dubbing_mode", False)
    if dubbing_mode:
        main_window._dubbing_mode       = True
        main_window._dubbing_video_path = data.get("dubbing_video_path")

        saved_speaker_list = data.get("speaker_list", [])
        freq: Dict[str, int] = {}
        for f in main_window.srt_tab._fragments:
            spk = f.get("speaker")
            if spk and str(spk).strip():
                k = str(spk).strip()
                freq[k] = freq.get(k, 0) + 1

        if freq:
            main_window._speaker_list = sorted(
                saved_speaker_list,
                key=lambda s: freq.get(s, 0),
                reverse=True,
            )
        else:
            main_window._speaker_list = saved_speaker_list

        main_window._rebuild_voice_cloning_for_speakers()

        speaker_voices_data = data.get("speaker_voices", {})
        whisper_ready = (
            main_window._whisper_backend is not None
            and main_window._whisper_backend.is_downloaded(main_window._w_size.currentData())
        )
        is_supertonic = main_window._is_supertonic_active()
        is_piper      = main_window._is_piper_active()
        for spk, sv_data in speaker_voices_data.items():
            sv = main_window._speaker_voices.get(spk)
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
                    info = main_window._audio_info_str(audio_path)
                    sv["audio_info_lbl"].setText(info)
                ref_text = sv_data.get("ref_text", "")
                if ref_text:
                    sv["ref_text"].setPlainText(ref_text)
                demucs = sv_data.get("demucs", False)
                sv["demucs_chk"].setChecked(demucs)

        if hasattr(main_window, "_btn_dubbing"):
            main_window._btn_dubbing.setEnabled(True)
            main_window._btn_dubbing.setText("✓  Dubbing active")
            main_window._btn_dubbing.setStyleSheet(_btn(C["success"]))
            main_window._btn_dubbing.setVisible(True)

        if hasattr(main_window, "_vid_row_widget"):
            main_window._vid_row_widget.setVisible(False)
        if hasattr(main_window, "_whisper_section_widget"):
            main_window._whisper_section_widget.setVisible(False)
    else:
        main_window._update_dubbing_visibility()

    for f in main_window.srt_tab._fragments:
        ap = f.get("output_path")
        if ap and os.path.exists(ap):
            main_window._file_watcher.addPath(ap)

    fname = Path(main_window._srt_path).name if main_window._srt_path else "session"
    done  = sum(1 for f in main_window.srt_tab._fragments if f.get("status") == "done")
    main_window._srt_label.setText(
        f"📄  {fname}  •  {len(main_window.srt_tab._fragments)} fragments"
    )
    main_window._srt_label.setStyleSheet(
        f"color:{C['accent']};font-size:11px;font-weight:600;"
    )
    main_window._btn_close_srt.setEnabled(True)
    main_window._btn_save_session.setEnabled(True)
    main_window._tabs.setCurrentIndex(0)
    main_window.srt_tab._populate_tree()
    main_window._update_action_buttons()

    status_msg = f"Session loaded: {len(main_window.srt_tab._fragments)} fragments, {done} already synthesized."
    if missing:
        status_msg = f"Session loaded — {missing} audio file(s) missing on disk, reset to waiting."
        main_window._set_status(status_msg, C["warning"])
    else:
        main_window._set_status(status_msg, C["success"])


def _auto_ebook_session_path(main_window) -> Optional[Path]:
    if not main_window._epub_path:
        return None
    return OUTPUTS_DIR / Path(main_window._epub_path).stem / "ebook_session.json"


def _build_ebook_session_data(main_window) -> dict:
    ref_audio = main_window._drop.file_path
    return {
        "session_type":         "ebook",
        "version":              1,
        "epub_path":            main_window._epub_path,
        "ebook_output_dir":     main_window._ebook_output_dir,
        "reference_audio":      ref_audio,
        "reference_audio_hash": _ref_audio_hash(ref_audio),
        "reference_text":       main_window._ref_text.toPlainText(),
        "generation_params":    main_window._get_generation_settings(),
        "normalize_audio":      main_window._norm_check.isChecked(),
        "audiobook_silence_ms": main_window._audiobook_silence_spin.value(),
        "audiobook_format":     main_window._audiobook_fmt_combo.currentData(),
        "fragments": [
            {k: v for k, v in f.items()}
            for f in main_window.ebook_tab._ebook_fragments
        ],
    }


def _write_ebook_session_to(main_window, path: str) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_build_ebook_session_data(main_window), fh, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.warning(f"Ebook session write failed: {e}")
        return False


def _restore_ebook_session_data(main_window, data: dict):
    main_window._epub_path = data.get("epub_path", "")
    main_window._ebook_output_dir = data.get("ebook_output_dir", str(OUTPUTS_DIR))
    if hasattr(main_window, '_audiobook_output_edit'):
        main_window._audiobook_output_edit.setText(main_window._ebook_output_dir)
    main_window.ebook_tab._ebook_fragments = data.get("fragments", [])
    ref_audio = data.get("reference_audio")
    saved_hash = data.get("reference_audio_hash")
    if ref_audio and os.path.exists(ref_audio):
        current_hash = _ref_audio_hash(ref_audio)
        if saved_hash and current_hash and current_hash != saved_hash:
            QMessageBox.warning(
                main_window, "Reference audio changed",
                f"The reference audio file has changed since the session was saved:\n"
                f"{ref_audio}\n\nFragments marked as done may not match the current voice.",
            )
        main_window._drop._set(ref_audio)
        main_window._ref_player.load(ref_audio)
    elif ref_audio:
        main_window._set_status(
            f"Reference audio not found: {Path(ref_audio).name}", C["warning"]
        )
    main_window._ref_text.setPlainText(data.get("reference_text", ""))
    params = data.get("generation_params", {})
    for key, value in params.items():
        widget = main_window._param_widgets.get(key) if hasattr(main_window, "_param_widgets") else None
        if widget is None:
            continue
        if isinstance(widget, QSlider):
            widget.setValue(int(value * 100))
        elif isinstance(widget, QSpinBox):
            widget.setValue(int(value))
    main_window._norm_check.setChecked(data.get("normalize_audio", False))
    if hasattr(main_window, '_audiobook_silence_spin'):
        main_window._audiobook_silence_spin.setValue(data.get("audiobook_silence_ms", 500))
    if hasattr(main_window, '_audiobook_fmt_combo'):
        fmt = data.get("audiobook_format", "wav")
        for i in range(main_window._audiobook_fmt_combo.count()):
            if main_window._audiobook_fmt_combo.itemData(i) == fmt:
                main_window._audiobook_fmt_combo.setCurrentIndex(i)
                break
    missing = sum(
        1 for f in main_window.ebook_tab._ebook_fragments
        if f.get("status") == "done"
        and f.get("output_path")
        and not os.path.exists(f.get("output_path", ""))
    )
    if missing:
        for f in main_window.ebook_tab._ebook_fragments:
            if (f.get("status") == "done"
                    and f.get("output_path")
                    and not os.path.exists(f.get("output_path", ""))):
                f["status"] = "waiting"
                f["output_path"] = None
    fname = Path(main_window._epub_path).name if main_window._epub_path else "session"
    done = sum(1 for f in main_window.ebook_tab._ebook_fragments if f.get("status") == "done")
    main_window._epub_label.setText(
        f"📚 {fname} • {len(main_window.ebook_tab._ebook_fragments)} fragments"
    )
    main_window._epub_label.setStyleSheet(
        f"color:{C['accent']};font-size:11px;font-weight:600;"
    )
    main_window._btn_close_ebook.setEnabled(True)
    main_window._btn_save_ebook_session.setEnabled(True)
    main_window._tabs.setCurrentIndex(1)
    main_window.ebook_tab._populate_ebook_tree()
    main_window._update_action_buttons()

    main_window.ebook_tab._update_preview_btn_state()

    status_msg = f"Ebook session loaded: {len(main_window.ebook_tab._ebook_fragments)} fragments, {done} already synthesized."
    if missing:
        status_msg = f"Ebook session loaded — {missing} audio file(s) missing on disk, reset to waiting."
        main_window._set_status(status_msg, C["warning"])
    else:
        main_window._set_status(status_msg, C["success"])