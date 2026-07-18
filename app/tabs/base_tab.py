# tabs/base_tab.py
"""
Base class for all tab widgets in the application.
"""
import os
import re
import subprocess
import shutil

from PyQt6.QtWidgets import QWidget


class BaseTab(QWidget):
    """Base class for all tab widgets."""

    def __init__(self, main_window):
        super().__init__()
        self.main = main_window

    @property
    def backend(self):
        return self.main._backend

    @property
    def whisper(self):
        return self.main._whisper_backend

    @property
    def preprocessor(self):
        return self.main._preprocessor
        
    def _normalize_ffmpeg(self, input_path: str, output_path: str) -> bool:
        """Normalize audio using ffmpeg volumedetect."""
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