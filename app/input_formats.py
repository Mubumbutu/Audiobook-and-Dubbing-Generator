from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Segment:
    index: int
    start_ms: int
    end_ms: int
    text: str
    speaker: Optional[str] = None


_WHITESPACE_RE = re.compile(r'\s+')
_NOISE_RE = re.compile(r'^[\W\d\s]+$')
_FOOTNOTE_RE = re.compile(r'\[\d+\]|(?<=\w)[¹²³⁴⁵⁶⁷⁸⁹⁰]+')
_URL_RE = re.compile(r'https?://\S+|www\.\S+')
_SENTENCE_SPLIT_RE = re.compile(
    r'(?<=[.!?…])\s+'
    r'(?=[A-Z\u00C0-\u00D6\u00D8-\u00DE\u0410-\u042F\u0401„""«»\u2018\u2019])'
)


def _local_tag(tag: str) -> str:
    return tag.split('}', 1)[-1] if '}' in tag else tag


def _clean_text(text: str) -> str:
    text = _URL_RE.sub('', text)
    text = _FOOTNOTE_RE.sub('', text)
    return _WHITESPACE_RE.sub(' ', text).strip()


def _split_sentences(text: str) -> List[str]:
    if not text or len(text) < 5:
        return [text] if text else []

    parts = _SENTENCE_SPLIT_RE.split(text)
    result = []

    for part in parts:
        part = part.strip()
        if not part or _NOISE_RE.fullmatch(part):
            continue
        if len(part) >= 5:
            result.append(part)

    if len(result) > 1:
        final = [result[0]]
        for part in result[1:]:
            if len(final[-1]) < 8:
                final[-1] = final[-1] + ' ' + part
            else:
                final.append(part)
        result = final

    return result or [text]


def _is_continuation_start(ch: str) -> bool:
    if not ch or not ch.isalpha():
        return False
    if ch.lower() == ch.upper():
        return True
    return ch.islower()


def _merge_continuations(segments: List[Segment], max_gap_ms: int = 1000) -> List[Segment]:
    if not segments:
        return segments

    merged = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        while i + 1 < len(segments):
            nxt = segments[i + 1]
            gap = nxt.start_ms - seg.end_ms
            if (
                gap <= max_gap_ms
                and seg.text
                and nxt.text
                and seg.text[-1] not in '.?!…'
                and _is_continuation_start(nxt.text[0])
            ):
                seg = Segment(
                    index=seg.index,
                    start_ms=seg.start_ms,
                    end_ms=nxt.end_ms,
                    text=seg.text + ' ' + nxt.text,
                    speaker=seg.speaker if seg.speaker is not None else nxt.speaker,
                )
                i += 1
            else:
                break
        merged.append(seg)
        i += 1

    for new_idx, seg in enumerate(merged):
        merged[new_idx] = Segment(
            index=new_idx,
            start_ms=seg.start_ms,
            end_ms=seg.end_ms,
            text=seg.text,
            speaker=seg.speaker,
        )

    return merged


class InputFormat(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def extensions(self) -> List[str]: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def load(self, path: str) -> List[Segment]: ...

    @abstractmethod
    def save(self, segments: List[Segment], path: str) -> None: ...


_FORMATS: dict[str, InputFormat] = {}


def register_format(fmt: InputFormat) -> None:
    for ext in fmt.extensions:
        _FORMATS[ext] = fmt


def get_format(ext: str) -> InputFormat:
    if ext not in _FORMATS:
        raise ValueError(f"Unsupported format: {ext}")
    return _FORMATS[ext]


def all_formats() -> List[InputFormat]:
    return list({id(v): v for v in _FORMATS.values()}.values())


def supported_extensions() -> List[str]:
    return list(_FORMATS.keys())