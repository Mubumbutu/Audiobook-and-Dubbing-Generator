from __future__ import annotations

import os
import shutil
from typing import List

from input_formats import (
    InputFormat, Segment, register_format,
    _NOISE_RE, _local_tag, _clean_text, _split_sentences,
)

_BLOCK_TAGS = frozenset({
    "p", "blockquote", "li",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "div",
})


def _parse_html_file(html_path: str) -> List[str]:
    from lxml import etree

    with open(html_path, 'rb') as fh:
        raw = fh.read()
    if not raw or len(raw) < 10:
        return []

    try:
        root = etree.fromstring(raw)
    except etree.XMLSyntaxError:
        try:
            root = etree.fromstring(raw, parser=etree.HTMLParser())
        except Exception:
            return []

    texts = []
    for element in root.iter():
        tag = element.tag
        if not isinstance(tag, str):
            continue
        if _local_tag(tag) not in _BLOCK_TAGS:
            continue
        text = _clean_text("".join(element.itertext()))
        if not text or len(text) < 5:
            continue
        if _NOISE_RE.fullmatch(text):
            continue
        texts.append(text)
    return texts


class KindleFormat(InputFormat):
    @property
    def name(self) -> str:
        return "Kindle Book"

    @property
    def extensions(self) -> List[str]:
        return [".mobi", ".azw", ".azw3"]

    @property
    def description(self) -> str:
        return "Kindle e-book file (.mobi, .azw, .azw3) – read only"

    def load(self, path: str) -> List[Segment]:
        try:
            import mobi
        except ImportError as e:
            raise ImportError(f"Install required packages: pip install mobi lxml\n{e}")

        extract_dir, _ = mobi.extract(path)
        try:
            html_files = []
            for root_dir, _, files in os.walk(extract_dir):
                for fname in sorted(files):
                    if fname.lower().endswith(('.html', '.htm', '.xhtml')):
                        html_files.append(os.path.join(root_dir, fname))

            segments: List[Segment] = []
            idx = 0
            for html_path in html_files:
                for text in _parse_html_file(html_path):
                    for sentence in _split_sentences(text):
                        segments.append(Segment(
                            index=idx,
                            start_ms=0,
                            end_ms=0,
                            text=sentence,
                            speaker=None,
                        ))
                        idx += 1
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)

        return segments

    def save(self, segments: List[Segment], path: str) -> None:
        raise NotImplementedError(
            "Kindle format (.mobi/.azw/.azw3) does not support saving. "
            "Export to EPUB or TXT instead."
        )


register_format(KindleFormat())