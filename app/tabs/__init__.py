# tabs/__init__.py
"""
Tab modules for the SRT Lektor Studio application.
"""
from .base_tab import BaseTab
from .srt_tab import SrtTab
from .ebook_tab import EbookTab
from .quick_tts_tab import QuickTTSTab

__all__ = ["BaseTab", "SrtTab", "EbookTab", "QuickTTSTab"]