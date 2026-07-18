# utils.py
import sys, os, re, json, subprocess, logging, hashlib
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable

import numpy as np
import soundfile as sf
import torchaudio
from lingua import LanguageDetectorBuilder
from num2words import num2words
from srt_format import _ms_to_srt_ts as _ms_to_ts

logger = logging.getLogger(__name__)

_LAST_DIRS: Dict[str, str] = {}


def _get_last_dir(key: str, fallback: str = "") -> str:
    return _LAST_DIRS.get(key) or fallback or str(Path.home())


def _set_last_dir(key: str, path: str) -> None:
    d = path if os.path.isdir(path) else str(Path(path).parent)
    if os.path.isdir(d):
        _LAST_DIRS[key] = d


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


def _ref_audio_hash(path: Optional[str]) -> Optional[str]:
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


def _video_audio_info_str(video_path: str) -> str:
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
