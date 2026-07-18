# main.py
"""
SRT Lektor/Dubbing Studio – entry point
"""
import sys
import os
import logging
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QFont
from PyQt6.QtCore import Qt

from config import (
    C,
    STYLE,
    ROOT_DIR,
    OUTPUTS_DIR,
    PROC_DIR,
    APP_NAME,
    MODEL_OPTIONS,
    _ACTIVE_BACKEND_CLASSES,
    SUPERTONIC_VOICES,
    _piper_voice_options,
)

from core import MainWindow

os.environ.setdefault("TORCH_HOME", str(ROOT_DIR / "models" / "torch_hub"))
OUTPUTS_DIR.mkdir(exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("TTSStudio.1")

    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    icon_path = ROOT_DIR / "icon.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    else:
        pm = QPixmap(64, 64)
        pm.fill(QColor(C["surface"]))
        p = QPainter(pm)
        p.setFont(QFont("Segoe UI Emoji", 32))
        p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "💀")
        p.end()
        app.setWindowIcon(QIcon(pm))

    app.setApplicationName(APP_NAME)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
