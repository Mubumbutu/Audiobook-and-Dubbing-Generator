# undo_manager.py
"""
SRT Lektor/Dubbing Studio - Main Window
"""
import os
import copy
import shutil
import logging
import traceback
import uuid
from typing import Dict, List, Optional

from config import C
from audio_utils import build_audio_undo_state, restore_audio_state

logger = logging.getLogger(__name__)

MAX_UNDO_ENTRIES = 100
BACKUP_DIR_NAME = "__undo_backup"


def snapshot_fragments(fragments: List[Dict], with_files: bool) -> Dict:
    state: Dict = {
        'fragments': copy.deepcopy(fragments),
        'files': None,
        'backup_dir': None,
    }
    if not with_files:
        return state

    paths: List[str] = []
    seen = set()
    for frag in fragments:
        path = frag.get('output_path')
        if path and path not in seen and os.path.exists(path):
            seen.add(path)
            paths.append(path)

    state['files'] = []
    if not paths:
        return state

    backup_dir = os.path.join(os.path.dirname(paths[0]), BACKUP_DIR_NAME, uuid.uuid4().hex)
    try:
        os.makedirs(backup_dir, exist_ok=True)
        for n, src in enumerate(paths):
            dst = os.path.join(backup_dir, f"{n:05d}_{os.path.basename(src)}")
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            state['files'].append((src, dst))
    except Exception:
        shutil.rmtree(backup_dir, ignore_errors=True)
        raise
    state['backup_dir'] = backup_dir
    return state


def restore_snapshot_files(state: Dict, current_fragments: List[Dict], file_watcher) -> List[str]:
    files = state.get('files')
    if files is None:
        return []

    available = [(orig, backup) for orig, backup in files if os.path.exists(backup)]
    missing = [orig for orig, backup in files if not os.path.exists(backup)]
    current_paths = {f.get('output_path') for f in current_fragments if f.get('output_path')}

    for path in current_paths | {orig for orig, _ in files}:
        file_watcher.removePath(path)

    for path in current_paths:
        if os.path.exists(path):
            os.remove(path)

    for orig, backup in available:
        parent = os.path.dirname(orig)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if os.path.exists(orig):
            os.remove(orig)
        try:
            os.link(backup, orig)
        except OSError:
            shutil.copy2(backup, orig)
        file_watcher.addPath(orig)

    return missing


def discard_snapshot(state: Optional[Dict]) -> None:
    if not state:
        return
    backup_dir = state.get('backup_dir')
    if not backup_dir:
        return
    shutil.rmtree(backup_dir, ignore_errors=True)
    try:
        os.rmdir(os.path.dirname(backup_dir))
    except OSError:
        pass


class UndoManager:

    def __init__(self, main_window):
        self.main = main_window
        self._undo_stack: List[Dict] = []
        self._redo_stack: List[Dict] = []
        self._busy = False

    def push(self, scope: str, label: str, state: Dict) -> None:
        for entry in self._redo_stack:
            self._discard_entry(entry)
        self._redo_stack.clear()
        self._undo_stack.append({'scope': scope, 'label': label, 'state': state})
        while len(self._undo_stack) > MAX_UNDO_ENTRIES:
            self._discard_entry(self._undo_stack.pop(0))

    def clear(self) -> None:
        for entry in self._undo_stack + self._redo_stack:
            self._discard_entry(entry)
        self._undo_stack.clear()
        self._redo_stack.clear()

    def undo(self) -> None:
        self._swap(self._undo_stack, self._redo_stack, True)

    def redo(self) -> None:
        self._swap(self._redo_stack, self._undo_stack, False)

    @staticmethod
    def _discard_entry(entry: Dict) -> None:
        if entry['scope'] != 'audio':
            discard_snapshot(entry['state'])

    def _tab_for(self, scope: str):
        if scope == 'srt':
            return self.main.srt_tab
        if scope == 'ebook':
            return self.main.ebook_tab
        return None

    def _prepare(self, scope: str, state: Dict) -> None:
        if scope in ('srt', 'ebook'):
            self.main._stop_play()
            self.main._tabs.setCurrentIndex(0 if scope == 'srt' else 1)
            return
        frag_idx = state.get('frag_idx')
        if frag_idx is None:
            return
        is_ebook = state.get('is_ebook', False)
        self.main._tabs.setCurrentIndex(1 if is_ebook else 0)
        if self.main._current_fragment_idx != frag_idx:
            tab = self.main.ebook_tab if is_ebook else self.main.srt_tab
            tab._load_fragment_audio(frag_idx)

    def _capture(self, scope: str, state: Dict) -> Optional[Dict]:
        if scope == 'audio':
            if self.main._audio_data is None:
                return None
            return build_audio_undo_state(self.main)
        return self._tab_for(scope)._undo_capture(state.get('files') is not None)

    def _apply(self, scope: str, state: Dict) -> int:
        if scope == 'audio':
            restore_audio_state(self.main, state)
            return 0
        return self._tab_for(scope)._undo_restore(state)

    def _swap(self, source: List[Dict], target: List[Dict], is_undo: bool) -> None:
        if not source or self._busy:
            return

        entry = source[-1]
        scope = entry['scope']
        state = entry['state']
        verb = "Undone" if is_undo else "Redone"
        fail = "Undo failed" if is_undo else "Redo failed"

        tab = self._tab_for(scope)
        if tab is not None and getattr(tab, '_is_running', False):
            self.main._set_status("Cannot undo or redo while synthesis is running.", C["warning"])
            return

        self._busy = True
        try:
            self._prepare(scope, state)
            current = self._capture(scope, state)
            try:
                missing = self._apply(scope, state)
            except Exception as e:
                logger.error(f"{fail}:\n{traceback.format_exc()}")
                if current is not None:
                    try:
                        self._apply(scope, current)
                    except Exception:
                        logger.error(f"Rollback failed:\n{traceback.format_exc()}")
                    if scope != 'audio':
                        discard_snapshot(current)
                self.main._set_status(f"{fail}, previous state restored: {e}", C["error"])
                return

            source.pop()
            if scope != 'audio':
                discard_snapshot(state)
            if current is not None:
                target.append({'scope': scope, 'label': entry['label'], 'state': current})

            msg = f"{verb}: {entry['label']}"
            if missing:
                msg += f" ({missing} audio file(s) missing, reset to waiting)"
            self.main._set_status(msg, C["warning"] if missing else C["accent"])
        except Exception as e:
            logger.error(f"{fail}:\n{traceback.format_exc()}")
            self.main._set_status(f"{fail}: {e}", C["error"])
        finally:
            self._busy = False