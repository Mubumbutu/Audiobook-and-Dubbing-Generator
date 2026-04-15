from __future__ import annotations

import re
from typing import List

from input_formats import (
    InputFormat, Segment, register_format,
    _NOISE_RE, _clean_text, _split_sentences,
)

_SENTENCE_END_RE = re.compile(r'[.!?…]\s*$')
_SOFT_HYPHEN_RE = re.compile(r'\xad')


def _clean_pdf_text(text: str) -> str:
    text = _SOFT_HYPHEN_RE.sub('', text)
    return _clean_text(text)


def _is_header_block(text: str) -> bool:
    if len(text) > 80:
        return False
    stripped = text.rstrip()
    if stripped.endswith(':'):
        return True
    if stripped.isupper() and len(stripped) < 60:
        return True
    return False


def _merge_broken_sentences(blocks: List[str]) -> List[str]:
    if not blocks:
        return blocks
    result = [blocks[0]]
    for text in blocks[1:]:
        prev = result[-1]
        if prev.endswith('-') and text and text[0].islower():
            result[-1] = prev[:-1] + text
        elif (
            not _SENTENCE_END_RE.search(prev)
            and text
            and text[0].islower()
            and prev[-1] not in {'\u2014', '\u2013', '\u2012'}
            and not _is_header_block(prev)
        ):
            result[-1] = prev + ' ' + text
        else:
            result.append(text)
    return result


class PdfFormat(InputFormat):
    @property
    def name(self) -> str:
        return "PDF Document"

    @property
    def extensions(self) -> List[str]:
        return [".pdf"]

    @property
    def description(self) -> str:
        return "PDF document (.pdf)"

    def load(self, path: str) -> List[Segment]:
        try:
            import fitz
        except ImportError as e:
            raise ImportError(f"Install required packages: pip install pymupdf\n{e}")

        doc = fitz.open(path)
        raw_blocks: List[str] = []

        for page in doc:
            for block in page.get_text("blocks"):
                if block[6] != 0:
                    continue
                text = _clean_pdf_text(block[4])
                if not text or len(text) < 5:
                    continue
                if _NOISE_RE.fullmatch(text):
                    continue
                raw_blocks.append(text)

        doc.close()

        segments: List[Segment] = []
        idx = 0
        for text in _merge_broken_sentences(raw_blocks):
            for sentence in _split_sentences(text):
                segments.append(Segment(
                    index=idx,
                    start_ms=0,
                    end_ms=0,
                    text=sentence,
                    speaker=None,
                ))
                idx += 1

        return segments

    def save(self, segments: List[Segment], path: str) -> None:
        try:
            import fitz
        except ImportError as e:
            raise ImportError(f"Install required packages: pip install pymupdf\n{e}")

        doc = fitz.open()
        page = doc.new_page()
        y = 72
        for seg in segments:
            line = f"[{seg.speaker}] {seg.text}" if seg.speaker else seg.text
            page.insert_text((72, y), line, fontsize=11)
            y += 16
            if y > page.rect.height - 72:
                page = doc.new_page()
                y = 72
        doc.save(path)
        doc.close()


register_format(PdfFormat())