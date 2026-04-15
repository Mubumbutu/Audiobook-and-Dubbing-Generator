from __future__ import annotations

import re
from typing import List

from input_formats import (
    InputFormat, Segment, register_format,
    _NOISE_RE, _clean_text, _split_sentences, _merge_continuations,
)


def _parse_bracket_line(line: str):
    m = re.match(r'^\[(\d+)\]\[(\d+)\](.*)', line.strip())
    if not m:
        return None
    start_ms = int(m.group(1)) * 10
    end_ms   = int(m.group(2)) * 10
    text     = m.group(3)
    return start_ms, end_ms, text


def _split_pipe_dialogue(text: str, start_ms: int, end_ms: int):
    parts = [p.strip() for p in text.split('|')]
    parts = [p for p in parts if p]

    if not parts:
        return []

    if len(parts) == 1:
        return [{
            'text': parts[0],
            'start_ms': start_ms,
            'end_ms': end_ms,
            'speaker': None
        }]

    char_counts = [max(1, len(p)) for p in parts]
    total_chars = sum(char_counts)
    duration_ms = max(0, end_ms - start_ms)

    result = []
    current_start = start_ms

    for i, (part, cc) in enumerate(zip(parts, char_counts)):
        if i == len(parts) - 1:
            frag_end = end_ms
        else:
            frag_end = current_start + int(duration_ms * cc / total_chars)

        result.append({
            'text': part,
            'start_ms': current_start,
            'end_ms': frag_end,
            'speaker': None,
        })

        current_start = frag_end

    return result


def _read_txt_file(path: str) -> str:
    encodings = ['utf-8', 'utf-8-sig', 'windows-1250', 'iso-8859-2', 'cp1252', 'latin1']
    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Cannot decode TXT file: {path}")


_CHAPTER_HEADING_RE = re.compile(
    r'^(chapter|rozdział|глава|kapitel|chapitre|capítulo|capitolo)\b.*$'
    r'|^[IVXLCDM]+\.?\s*$'
    r'|^\*\s*\*\s*\*$'
    r'|^#{1,6}\s',
    re.IGNORECASE,
)


class TxtSrtFormat(InputFormat):
    @property
    def name(self) -> str:
        return "TXT Subtitle"

    @property
    def extensions(self) -> List[str]:
        return [".txt_srt"]

    @property
    def description(self) -> str:
        return "TXT subtitle file in [cs][cs]text format"

    def load(self, path: str) -> List[Segment]:
        content  = _read_txt_file(path)
        segments: List[Segment] = []
        idx = 0

        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            parsed = _parse_bracket_line(line)
            if parsed is None:
                continue
            start_ms, end_ms, text = parsed
            frags = _split_pipe_dialogue(text, start_ms, end_ms)
            for frag in frags:
                if not frag['text']:
                    continue
                segments.append(Segment(
                    index=idx,
                    start_ms=frag['start_ms'],
                    end_ms=frag['end_ms'],
                    text=frag['text'],
                    speaker=frag['speaker'],
                ))
                idx += 1

        return _merge_continuations(segments)

    def save(self, segments: List[Segment], path: str) -> None:
        lines = []
        for seg in segments:
            start_cs = seg.start_ms // 10
            end_cs   = seg.end_ms   // 10
            text = f"[{seg.speaker}] {seg.text}" if seg.speaker else seg.text
            lines.append(f"[{start_cs}][{end_cs}]{text}")
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


class TxtEbookFormat(InputFormat):
    @property
    def name(self) -> str:
        return "TXT Book"

    @property
    def extensions(self) -> List[str]:
        return [".txt_ebook"]

    @property
    def description(self) -> str:
        return "Plain text file split into sentences"

    def load(self, path: str) -> List[Segment]:
        content = _read_txt_file(path)

        paragraphs = re.split(r'\n{2,}', content)
        if sum(1 for p in paragraphs if p.strip()) <= 3:
            paragraphs = content.splitlines()

        segments: List[Segment] = []
        idx = 0

        for para in paragraphs:
            para = _clean_text(re.sub(r'[ \t]+', ' ', para))
            if not para or len(para) < 5:
                continue
            if _NOISE_RE.fullmatch(para):
                continue
            if _CHAPTER_HEADING_RE.match(para):
                continue

            if re.search(r'[.!?…]', para):
                for sentence in _split_sentences(para):
                    segments.append(Segment(
                        index=idx,
                        start_ms=0,
                        end_ms=0,
                        text=sentence,
                        speaker=None,
                    ))
                    idx += 1
            else:
                segments.append(Segment(
                    index=idx,
                    start_ms=0,
                    end_ms=0,
                    text=para,
                    speaker=None,
                ))
                idx += 1

        return segments

    def save(self, segments: List[Segment], path: str) -> None:
        lines = [seg.text for seg in segments]
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


txt_srt_format = TxtSrtFormat()
txt_ebook_format = TxtEbookFormat()
register_format(txt_srt_format)
register_format(txt_ebook_format)