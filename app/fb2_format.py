from __future__ import annotations

from typing import List

from input_formats import (
    InputFormat, Segment, register_format,
    _NOISE_RE, _local_tag, _clean_text, _split_sentences,
)

_BLOCK_TAGS = frozenset({
    "p", "v", "subtitle", "text-author", "th", "td",
    "epigraph", "cite",
})
_FB2_NS = "http://www.gribuser.ru/xml/fictionbook/2.0"


class Fb2Format(InputFormat):
    @property
    def name(self) -> str:
        return "FictionBook 2"

    @property
    def extensions(self) -> List[str]:
        return [".fb2"]

    @property
    def description(self) -> str:
        return "FictionBook 2 e-book (.fb2)"

    def load(self, path: str) -> List[Segment]:
        import xml.etree.ElementTree as ET

        try:
            tree = ET.parse(path)
        except ET.ParseError as e:
            raise ValueError(f"Invalid FB2 file (XML error): {e}")

        root = tree.getroot()
        bodies = [child for child in root if _local_tag(child.tag) == "body"]
        if not bodies:
            raise ValueError("FB2 file contains no <body> section.")

        raw_texts: List[str] = []
        seen: set[int] = set()

        for body in bodies:
            for element in body.iter():
                tag = _local_tag(element.tag)
                if tag not in _BLOCK_TAGS:
                    continue
                if id(element) in seen:
                    continue
                seen.add(id(element))
                text = _clean_text("".join(element.itertext()))
                if not text or len(text) < 5:
                    continue
                if _NOISE_RE.fullmatch(text):
                    continue
                raw_texts.append(text)

        segments: List[Segment] = []
        idx = 0
        for text in raw_texts:
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
        import xml.etree.ElementTree as ET

        ET.register_namespace('', _FB2_NS)
        root = ET.Element(f"{{{_FB2_NS}}}FictionBook")
        body = ET.SubElement(root, f"{{{_FB2_NS}}}body")
        section = ET.SubElement(body, f"{{{_FB2_NS}}}section")

        for seg in segments:
            p = ET.SubElement(section, f"{{{_FB2_NS}}}p")
            p.text = f"[{seg.speaker}] {seg.text}" if seg.speaker else seg.text

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        with open(path, 'wb') as fh:
            tree.write(fh, xml_declaration=True, encoding="utf-8")


register_format(Fb2Format())