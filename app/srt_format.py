from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from input_formats import (
    InputFormat, Segment, register_format,
    _clean_text, _merge_continuations,
)


def _ts_to_ms(ts: str) -> int:
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) != 3:
        return 0
    try:
        h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
        return int((h * 3600 + m * 60 + s) * 1000)
    except ValueError:
        return 0


def _ms_to_srt_ts(ms: int) -> str:
    ms = int(ms)
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    c = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{c:03d}"


def _parse_speaker(text: str) -> Tuple[Optional[str], str]:
    m = re.match(r'^\[([^\]]+)\]\s*', text)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    m = re.match(
        r'^([A-ZÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÄËÏÖÜÃÕŇČŠŽ][A-ZÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÄËÏÖÜÃÕŇČŠŽa-z]*'
        r'(?:\s[A-ZÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÄËÏÖÜÃÕŇČŠŽ][A-Za-z]*)*):\s*',
        text
    )
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    return None, text


def _process_srt_text(text: str, start_ms: int, end_ms: int) -> List[Dict]:
    DASH_RE = re.compile(r'^[-–]\s*')
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    if not lines:
        return []

    lines_have_dash = [bool(DASH_RE.match(ln)) for ln in lines]

    if len(lines) == 1:
        clean = DASH_RE.sub('', lines[0]).strip()
        return [{
            'text': clean,
            'start_ms': start_ms,
            'end_ms': end_ms,
            'speaker_hint': None
        }]

    if all(lines_have_dash):
        clean_lines = [DASH_RE.sub('', ln).strip() for ln in lines]
        clean_lines = [ln for ln in clean_lines if ln]
        if not clean_lines:
            return []

        if len(clean_lines) == 1:
            return [{
                'text': clean_lines[0],
                'start_ms': start_ms,
                'end_ms': end_ms,
                'speaker_hint': None
            }]

        char_counts = [max(1, len(ln)) for ln in clean_lines]
        total_chars = sum(char_counts)
        duration_ms = max(0, end_ms - start_ms)

        result = []
        current_start = start_ms

        for i, (line, cc) in enumerate(zip(clean_lines, char_counts)):
            if i == len(clean_lines) - 1:
                frag_end = end_ms
            else:
                frag_end = current_start + int(duration_ms * cc / total_chars)

            result.append({
                'text': line,
                'start_ms': current_start,
                'end_ms': frag_end,
                'speaker_hint': None
            })

            current_start = frag_end

        return result

    joined_parts = []
    for ln in lines:
        clean = DASH_RE.sub('', ln).strip()
        if clean:
            joined_parts.append(clean)

    return [{
        'text': " ".join(joined_parts).strip(),
        'start_ms': start_ms,
        'end_ms': end_ms,
        'speaker_hint': None
    }]


class SrtFormat(InputFormat):
    @property
    def name(self) -> str:
        return "SRT Subtitle"

    @property
    def extensions(self) -> List[str]:
        return [".srt"]

    @property
    def description(self) -> str:
        return "SubRip subtitle file (.srt)"

    def load(self, path: str) -> List[Segment]:
        encodings = ['utf-8', 'utf-8-sig', 'windows-1250', 'iso-8859-2', 'cp1252', 'latin1']
        content = None
        for enc in encodings:
            try:
                with open(path, 'r', encoding=enc) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if content is None:
            raise ValueError(f"Cannot decode SRT file: {path}")

        blocks = [b.strip() for b in content.split('\n\n') if b.strip()]
        raw_blocks = []

        for block in blocks:
            lines = block.split('\n')
            if len(lines) < 3:
                continue
            timestamp = lines[1].strip()
            text_lines = lines[2:]

            ts_parts = timestamp.split('-->')
            if len(ts_parts) != 2:
                continue
            start_ms = _ts_to_ms(ts_parts[0])
            end_ms   = _ts_to_ms(ts_parts[1])

            clean_lines = [re.sub(r'<[^>]+>', '', ln) for ln in text_lines]
            raw_blocks.append({
                'text':     '\n'.join(clean_lines),
                'start_ms': start_ms,
                'end_ms':   end_ms,
            })

        segments = []
        idx = 0
        for block in raw_blocks:
            for sf_item in _process_srt_text(block['text'], block['start_ms'], block['end_ms']):
                speaker, clean_text = _parse_speaker(sf_item['text'])
                clean_text = _clean_text(clean_text)
                if not clean_text:
                    continue
                segments.append(Segment(
                    index=idx,
                    start_ms=sf_item['start_ms'],
                    end_ms=sf_item['end_ms'],
                    text=clean_text,
                    speaker=speaker,
                ))
                idx += 1

        return _merge_continuations(segments)

    def save(self, segments: List[Segment], path: str) -> None:
        lines = []
        for i, seg in enumerate(segments, 1):
            lines.append(str(i))
            lines.append(f"{_ms_to_srt_ts(seg.start_ms)} --> {_ms_to_srt_ts(seg.end_ms)}")
            text = f"[{seg.speaker}] {seg.text}" if seg.speaker else seg.text
            lines.append(text)
            lines.append("")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


register_format(SrtFormat())