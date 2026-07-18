# workers.py
import sys
import os
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
from PyQt6.QtCore import QThread, pyqtSignal
from huggingface_hub import snapshot_download

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

from config import WHISPER_REPOS, WHISPER_SIZE_MB, ROOT_DIR, OUTPUTS_DIR, WHISPER_DIR
from utils import _normalize_text_for_tts, _get_wav_duration, _fmt, _fmt_ms, _check_ffmpeg
from tts_backends import InferenceError  # <-- dodany brakujący import

logger = logging.getLogger(__name__)


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
        if self.backend is None or not self.backend.is_loaded:
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
                if TORCH_AVAILABLE and torch.cuda.is_available():
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
        if self.backend is None:
            self.error.emit("Backend is None")
            return
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
        self.device    = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

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

    def _cleanup_tmp_files(self):
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

    def run(self):
        try:
            result = subprocess.run(
                self.cmd, capture_output=True, timeout=3600,
            )
            stderr_text = result.stderr.decode(errors="replace")
            if stderr_text.strip():
                for line in stderr_text.strip().splitlines()[-20:]:
                    logger.info(f"[ffmpeg] {line}")

            if result.returncode == 0:
                self.finished.emit(True, "")
            else:
                self.finished.emit(False, stderr_text[-800:])
        except subprocess.TimeoutExpired:
            self.finished.emit(False, "Timeout — operation took too long.")
        except Exception as e:
            self.finished.emit(False, str(e))
        finally:
            self._cleanup_tmp_files()

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

            self.status.emit("Converting audio to 16kHz mono PCM_16…")
            audio, sr = sf.read(str(audio_path), dtype="float32")
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

            output_dir  = OUTPUTS_DIR / "dubbing"
            output_dir.mkdir(parents=True, exist_ok=True)
            final_audio = output_dir / f"{Path(self.video_path).stem}_audio.wav"
            sf.write(str(final_audio), audio, 16000, subtype="PCM_16")

            try:
                shutil.rmtree(str(tmp_dir))
            except Exception:
                pass

            self.finished.emit(str(final_audio))
        except Exception:
            self.error.emit(traceback.format_exc())
            
class VocalSuppressWorker(BaseWorker):
    finished = pyqtSignal(str, str)

    def __init__(self, video_path: str, output_dir: str, audio_path: Optional[str] = None):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.audio_path = audio_path  # zewnętrzny plik audio (jeśli podany)

    def run(self):
        try:
            tmp_dir = Path(tempfile.mkdtemp(prefix="vocal_suppress_"))

            # Jeśli podano zewnętrzny plik audio, używamy go bezpośrednio
            if self.audio_path and os.path.exists(self.audio_path):
                audio_path = self.audio_path
                self.status.emit("Vocal suppression: using external audio file…")
            else:
                # Ekstrakcja audio z wideo
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

            stem = Path(audio_path).stem
            vocals_src = Path(demucs_out_dir) / "htdemucs_ft" / stem / "vocals.wav"
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

            out_dir = Path(self.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            video_stem = Path(self.video_path).stem
            vocals_dst = str(out_dir / f"_vsup_vocals_{video_stem}.wav")
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
            # Import Pipeline tylko tutaj, aby uniknąć zbędnego importu na górze
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