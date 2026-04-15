from __future__ import annotations

import zipfile
from typing import List

from input_formats import (
    InputFormat, Segment, register_format,
    _NOISE_RE, _local_tag, _clean_text, _split_sentences,
)

_BLOCK_TAGS = frozenset({
    "p", "blockquote", "li",
    "h1", "h2", "h3", "h4", "h5", "h6",
})


def _get_content_files(zf: zipfile.ZipFile) -> List[str]:
    try:
        from lxml import etree
    except ImportError:
        etree = None

    opf_path: str | None = None

    if etree is not None and 'META-INF/container.xml' in zf.namelist():
        try:
            root = etree.fromstring(zf.read('META-INF/container.xml'))
            el = root.find('.//{*}rootfile')
            if el is not None:
                opf_path = el.get('full-path')
        except Exception:
            pass

    if opf_path is None:
        for name in zf.namelist():
            if name.lower().endswith('.opf'):
                opf_path = name
                break

    if opf_path and etree is not None:
        try:
            opf_dir = opf_path.rsplit('/', 1)[0] + '/' if '/' in opf_path else ''
            opf_root = etree.fromstring(zf.read(opf_path))

            manifest: dict[str, str] = {}
            for item in opf_root.iter('{*}item'):
                item_id = item.get('id')
                href = item.get('href', '')
                media_type = item.get('media-type', '')
                if item_id and href and 'html' in media_type:
                    manifest[item_id] = opf_dir + href

            ordered = []
            for itemref in opf_root.iter('{*}itemref'):
                idref = itemref.get('idref')
                if idref and idref in manifest:
                    ordered.append(manifest[idref])

            if ordered:
                return ordered
        except Exception:
            pass

    return sorted(
        name for name in zf.namelist()
        if name.lower().endswith(('.html', '.xhtml', '.htm'))
    )


class EpubFormat(InputFormat):
    @property
    def name(self) -> str:
        return "EPUB Book"

    @property
    def extensions(self) -> List[str]:
        return [".epub"]

    @property
    def description(self) -> str:
        return "EPUB e-book file (.epub)"

    def load(self, path: str) -> List[Segment]:
        try:
            from lxml import etree
        except ImportError as e:
            raise ImportError(f"Install required packages: pip install lxml\n{e}")

        segments: List[Segment] = []
        idx = 0

        with zipfile.ZipFile(path, 'r') as zf:
            content_files = _get_content_files(zf)

            for filepath in content_files:
                try:
                    raw = zf.read(filepath)
                except KeyError:
                    continue
                if not raw or len(raw) < 10:
                    continue

                try:
                    root = etree.fromstring(raw)
                except etree.XMLSyntaxError:
                    try:
                        root = etree.fromstring(raw, parser=etree.HTMLParser())
                    except Exception:
                        continue

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
            from ebooklib import epub
        except ImportError as e:
            raise ImportError(f"Install ebooklib: pip install ebooklib\n{e}")

        book = epub.EpubBook()
        book.set_title("Exported")
        book.set_language("en")

        lines = []
        for seg in segments:
            if seg.speaker:
                lines.append(f"[{seg.speaker}] {seg.text}")
            else:
                lines.append(seg.text)

        chapter = epub.EpubHtml(title="Content", file_name="content.xhtml", lang="en")
        chapter.set_content(
            "<html><body>" +
            "".join(f"<p>{ln}</p>" for ln in lines) +
            "</body></html>"
        )
        book.add_item(chapter)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = ["nav", chapter]
        epub.write_epub(path, book)


register_format(EpubFormat())