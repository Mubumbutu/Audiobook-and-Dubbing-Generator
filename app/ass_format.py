from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from input_formats import (
    InputFormat, Segment, register_format,
    _clean_text, _merge_continuations,
)
from srt_format import _process_srt_text, _parse_speaker


SKIP_STYLES: Set[str] = set()

_OVERRIDE_BLOCK_RE = re.compile(r'\{[^}]*\}')
_DRAWING_RE = re.compile(r'\\p\s*([1-9]\d*)')
_TS_RE = re.compile(r'^\s*(\d+):(\d{1,2}):(\d{1,2})(?:[.,](\d+))?\s*$')


def _ass_ts_to_ms(ts: str) -> Optional[int]:
    m = _TS_RE.match(ts)
    if not m:
        return None
    h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
    frac = m.group(4) or "0"
    ms_frac = int(round(float("0." + frac) * 1000))
    return (h * 3600 + mi * 60 + s) * 1000 + ms_frac


def _ms_to_ass_ts(ms: int) -> str:
    cs_total = int(round(max(0, int(ms)) / 10.0))
    h = cs_total // 360000
    m = (cs_total % 360000) // 6000
    s = (cs_total % 6000) // 100
    cs = cs_total % 100
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _is_drawing(raw_text: str) -> bool:
    return any(_DRAWING_RE.search(block) for block in _OVERRIDE_BLOCK_RE.findall(raw_text))


def _ass_text_to_plain(raw: str) -> str:
    text = _OVERRIDE_BLOCK_RE.sub('', raw)
    text = text.replace('\\N', '\n')
    text = text.replace('\\n', ' ')
    text = text.replace('\\h', ' ')
    return text


def _read_text_file(path: str) -> str:
    for enc in ['utf-8-sig', 'utf-16', 'windows-1250', 'iso-8859-2', 'cp1252', 'latin1']:
        try:
            with open(path, 'r', encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Cannot decode ASS file: {path}")


class AssFormat(InputFormat):
    @property
    def name(self) -> str:
        return "ASS/SSA Subtitle"

    @property
    def extensions(self) -> List[str]:
        return [".ass", ".ssa"]

    @property
    def description(self) -> str:
        return "Advanced SubStation Alpha subtitle file (.ass / .ssa)"

    def load(self, path: str) -> List[Segment]:
        content = _read_text_file(path).replace('\r\n', '\n').replace('\r', '\n')

        in_events = False
        columns: List[str] = []
        raw_events: List[Dict] = []

        for line in content.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith('[') and stripped.endswith(']'):
                in_events = stripped.lower() == '[events]'
                continue
            if not in_events:
                continue

            low = stripped.lower()

            if low.startswith('format:'):
                columns = [c.strip().lower() for c in stripped[7:].split(',')]
                continue

            if not low.startswith('dialogue:'):
                continue

            if not columns:
                columns = ['layer', 'start', 'end', 'style', 'name',
                           'marginl', 'marginr', 'marginv', 'effect', 'text']

            try:
                text_idx = columns.index('text')
                start_idx = columns.index('start')
                end_idx = columns.index('end')
            except ValueError:
                continue
            style_idx = columns.index('style') if 'style' in columns else None
            name_idx = columns.index('name') if 'name' in columns else None

            fields = stripped[len('dialogue:'):].lstrip().split(',', len(columns) - 1)
            if len(fields) < len(columns):
                continue

            start_ms = _ass_ts_to_ms(fields[start_idx])
            end_ms = _ass_ts_to_ms(fields[end_idx])
            if start_ms is None or end_ms is None or end_ms <= start_ms:
                continue

            style = fields[style_idx].strip() if style_idx is not None else ''
            if style.lower() in SKIP_STYLES:
                continue

            raw_text = fields[text_idx]
            if _is_drawing(raw_text):
                continue

            actor = fields[name_idx].strip() if name_idx is not None else ''

            raw_events.append({
                'text': _ass_text_to_plain(raw_text),
                'start_ms': start_ms,
                'end_ms': end_ms,
                'actor': actor,
            })

        raw_events.sort(key=lambda e: (e['start_ms'], e['end_ms']))

        segments: List[Segment] = []
        idx = 0
        for ev in raw_events:
            for item in _process_srt_text(ev['text'], ev['start_ms'], ev['end_ms']):
                speaker, clean_text = _parse_speaker(item['text'])
                if ev['actor']:
                    speaker = ev['actor']
                clean_text = _clean_text(clean_text)
                if not clean_text:
                    continue
                segments.append(Segment(
                    index=idx,
                    start_ms=item['start_ms'],
                    end_ms=item['end_ms'],
                    text=clean_text,
                    speaker=speaker,
                ))
                idx += 1

        return _merge_continuations(segments)

    def save(self, segments: List[Segment], path: str) -> None:
        out = [
            "[Script Info]",
            "ScriptType: v4.00+",
            "PlayResX: 1280",
            "PlayResY: 720",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
            "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
            "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
            "0,0,0,0,100,100,0,0,1,2,1,2,40,40,40,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
        for seg in segments:
            text = seg.text.replace('{', '(').replace('}', ')')
            text = text.replace('\r\n', '\n').replace('\n', '\\N')
            name = (seg.speaker or '').replace(',', ' ')
            out.append(
                f"Dialogue: 0,{_ms_to_ass_ts(seg.start_ms)},{_ms_to_ass_ts(seg.end_ms)},"
                f"Default,{name},0,0,0,,{text}"
            )
        out.append("")
        with open(path, "w", encoding="utf-8-sig") as f:
            f.write("\n".join(out))


register_format(AssFormat())
