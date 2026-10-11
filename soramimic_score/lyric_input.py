"""Validated lyric spelling, syntax, and lossless normalization coordinates.

Offsets are Python character offsets, with exclusive ends. Display and audit
use ``original``; reading engines use ``text``. Every analysis character owns a
nonempty source span, including expansions, compositions, and whitespace.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
import unicodedata


class LyricInputError(ValueError):
    """Unsupported input with coordinates in the unmodified source."""

    def __init__(self, text: str, index: int, reason: str):
        self.position = index + 1
        separators = tuple(re.finditer(r"\r\n|[\r\n\u2028\u2029]", text[:index]))
        self.line = len(separators) + 1
        self.column = index - (separators[-1].end() if separators else 0) + 1
        char = text[index:index + 1]
        codepoint = f" (U+{ord(char):04X})" if char else ""
        super().__init__(f"{reason} in lyric input at position {self.position}: "
                         f"{char!r}{codepoint} (line {self.line}, column {self.column})")


class LyricMappingError(ValueError):
    """An analyzer did not consume the complete validated lyric surface."""

    def __init__(self, reason: str, position: int, part_index: int | None = None):
        self.reason, self.position, self.part_index = reason, position, part_index
        super().__init__(f"{reason} at original position {position}"
                         + (f" (part {part_index + 1})" if part_index is not None else ""))


@dataclass(frozen=True, slots=True)
class LyricAnnotation:
    kind: str
    span: tuple[int, int]
    source_span: tuple[int, int]
    content_span: tuple[int, int]
    source_content_span: tuple[int, int]
    surface_span: tuple[int, int] | None = None
    source_surface_span: tuple[int, int] | None = None
    reading: str | None = None


@dataclass(frozen=True, slots=True)
class LyricInput:
    """A validated source and its derived analysis view; derived fields are read-only."""

    original: str
    text: str = field(init=False)
    source_spans: tuple[tuple[int, int], ...] = field(init=False)
    annotations: tuple[LyricAnnotation, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.original, str):
            raise ValueError("lyric input must be a string")
        text, spans = _normalize(self.original)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "source_spans", spans)
        # A normalization may compose or expand characters, but never discard
        # them. Overlapping spans are expected for one-to-many expansions.
        covered = 0
        for start, end in spans:
            if not 0 <= start < end <= len(self.original) or start > covered:
                raise ValueError("lyric normalization lost source coverage")
            covered = max(covered, end)
        if covered != len(self.original) or len(spans) != len(text):
            raise ValueError("lyric normalization did not consume the complete input")
        object.__setattr__(self, "annotations", _annotations(self))

    def original_boundary(self, offset: int) -> int:
        """Translate only boundaries that do not split an expansion/composition."""
        if not 0 <= offset <= len(self.text):
            raise ValueError("analysis offset is out of range")
        if offset == 0:
            return 0
        if offset == len(self.text):
            return len(self.original)
        left, right = self.source_spans[offset - 1], self.source_spans[offset]
        if left[1] != right[0]:
            raise ValueError("analysis boundary splits one source character sequence")
        return right[0]

    def original_span(self, start: int, end: int) -> tuple[int, int]:
        if not 0 <= start <= end <= len(self.text):
            raise ValueError("analysis span is out of range")
        return self.original_boundary(start), self.original_boundary(end)

    @property
    def line_spans(self) -> tuple[tuple[int, int, int], ...]:
        """Original line start, content end, and separator end, including blank lines."""
        output, cursor = [], 0
        for raw_line in self.original.splitlines(keepends=True):
            line = raw_line.rstrip("\r\n\u2028\u2029")
            output.append((cursor, cursor + len(line), cursor + len(raw_line)))
            cursor += len(raw_line)
        return tuple(output)

    @property
    def lyric_lines(self) -> tuple[str, ...]:
        return tuple(self.original[start:end] for start, end, _ in self.line_spans
                     if self.original[start:end].strip())

    def match_parts(self, parts: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
        """Locate all analyzer parts in order, allowing only whitespace omission.

        Returned spans refer to original visible text. Whitespace between spans
        remains in ``original`` and belongs to neither token. Non-whitespace
        deletion, insertion, reordering, or a split inside an expansion fails.
        """
        cursor, spans = 0, []

        def position() -> int:
            return self.source_spans[cursor][0] + 1 if cursor < len(self.text) else len(self.original) + 1

        for index, part in enumerate(parts):
            normalized, _ = _normalize(part)
            if not normalized.strip():
                raise LyricMappingError("empty-part", position(), index)
            start = None
            for char in normalized:
                if char.isspace():
                    continue
                while cursor < len(self.text) and self.text[cursor].isspace():
                    cursor += 1
                if cursor == len(self.text) or self.text[cursor] != char:
                    raise LyricMappingError("surface-mismatch", position(), index)
                if start is None:
                    start = cursor
                cursor += 1
            try:
                spans.append(self.original_span(start, cursor))
            except ValueError as exc:
                raise LyricMappingError("unsafe-normalization-boundary", position(), index) from exc
        while cursor < len(self.text) and self.text[cursor].isspace():
            cursor += 1
        if cursor != len(self.text):
            raise LyricMappingError("unconsumed-input", position())
        return tuple(spans)

    def to_dict(self) -> dict:
        return {"original_text": self.original, "normalized_text": self.text,
                "source_spans": [list(span) for span in self.source_spans],
                "annotations": [asdict(item) for item in self.annotations],
                "line_spans": [{"source_span": [start, end], "separator_span": [end, full_end],
                                "included": bool(self.original[start:end].strip()),
                                "reason": "lyric-line" if self.original[start:end].strip() else "blank-line"}
                               for start, end, full_end in self.line_spans],
                "complete": True}


def parse_lyric_input(text: str) -> LyricInput:
    return LyricInput(text)


def _normalize(text: str) -> tuple[str, tuple[tuple[int, int], ...]]:
    from .japanese import _BASE, _READING_ALIASES, katakana

    if not isinstance(text, str):
        raise ValueError("lyric input must be a string")
    cells: list[tuple[str, int, int]] = []
    for index, original in enumerate(text):
        category = unicodedata.category(original)
        if (category in {"Cs", "Co", "Cn"}
                or category == "Cc" and original not in "\t\n\r"
                or category == "Cf" and original not in {"\u200c", "\u200d"}):
            raise LyricInputError(text, index, "unsupported character")
        if original == "\n" and cells and cells[-1][0] == "\n" and text[index - 1] == "\r":
            cells[-1] = ("\n", cells[-1][1], index + 1)
            continue
        if original in "\r\n" or category in {"Zl", "Zp"}:
            char = "\n"
        elif original.isspace():
            char = " "
        elif "！" <= original <= "～" and original not in "｜（）［］｛｝＜＞":
            char = chr(ord(original) - 0xFEE0)
        elif "｡" <= original <= "ﾟ":
            char = unicodedata.normalize("NFKC", original)
        else:
            char = original
        char = {"゛": "\u3099", "゜": "\u309a", "ゟ": "より", "ヿ": "コト"}.get(char, char)
        if char in {"\u3099", "\u309a"}:
            previous = cells[-1][0] if cells else ""
            if len(previous) != 1 or not ("ぁ" <= previous <= "ゖ" or "ァ" <= previous <= "ヶ"):
                raise LyricInputError(text, index, "voicing mark has no kana base")
            voiced = unicodedata.normalize("NFC", previous + char)
            if len(voiced) != 1:
                voiced = unicodedata.normalize("NFC", katakana(previous) + char)
            if len(voiced) != 1:
                raise LyricInputError(text, index, "voicing mark cannot voice its base")
            cells[-1] = (voiced, cells[-1][1], index + 1)
        elif char in "ゝゞヽヾ":
            previous = cells[-1][0] if cells else ""
            base = unicodedata.normalize("NFD", previous)[:1]
            if katakana(base) not in _BASE and base not in {"ゐ", "ゑ", "ヰ", "ヱ"}:
                raise LyricInputError(text, index, "kana iteration mark has no repeatable base")
            repeated = unicodedata.normalize("NFC", base + ("\u3099" if char in "ゞヾ" else ""))
            if len(repeated) != 1:
                repeated = unicodedata.normalize("NFC", katakana(base) + "\u3099")
            if len(repeated) != 1:
                raise LyricInputError(text, index, "kana iteration mark cannot voice its base")
            cells.append((repeated, index, index + 1))
        else:
            cells.extend((value, index, index + 1) for value in char)

    # NFC may compose non-kana spelling too (e.g. Latin accents and Hangul).
    # Normalize clusters together and map all their output back to the cluster.
    clusters: list[tuple[str, int, int]] = []
    for value, start, end in cells:
        if clusters and (unicodedata.combining(value)
                         or unicodedata.normalize("NFC", clusters[-1][0] + value)
                         != clusters[-1][0] + value):
            prior, left, _ = clusters.pop()
            clusters.append((unicodedata.normalize("NFC", prior + value), left, end))
        else:
            clusters.append((value, start, end))
    output, spans = [], []
    for value, start, end in clusters:
        value = value.translate(_READING_ALIASES)
        output.extend(value)
        spans.extend([(start, end)] * len(value))
    return "".join(output), tuple(spans)


def _annotations(source: LyricInput) -> tuple[LyricAnnotation, ...]:
    from .japanese import normalize_reading

    text = source.text
    pairs = {"(": ")", "[": "]", "{": "}", "（": "）", "［": "］", "｛": "｝",
             "＜": "＞", "「": "」", "『": "』", "【": "】", "《": "》", "〈": "〉",
             "≪": "≫", "“": "”"}
    source_pairs = {**pairs, "（": "）", "［": "］", "｛": "｝", "＜": "＞", "｢": "｣"}
    stack: list[tuple[str, int]] = []
    output = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            index += 2
            continue
        if char in "|｜":
            start = index
            opening = text.find("《", start + 1)
            closing = text.find("》", opening + 1) if opening >= 0 else -1
            if (opening <= start + 1 or closing <= opening + 1
                    or any(c in "\n|｜《》" for c in text[start + 1:opening])
                    or any(c in "\n|｜《" for c in text[opening + 1:closing])):
                raise LyricInputError(source.original, source.source_spans[start][0],
                                      "malformed explicit ruby; use ｜surface《kana》")
            try:
                reading = normalize_reading(text[opening + 1:closing])
            except ValueError as exc:
                offset = opening + getattr(exc, "position", 1)
                raise LyricInputError(source.original, source.source_spans[offset][0],
                                      f"explicit ruby must contain a kana pronunciation: {exc}") from exc
            output.append(LyricAnnotation(
                "ruby", (start, closing + 1), source.original_span(start, closing + 1),
                (opening + 1, closing), source.original_span(opening + 1, closing),
                (start + 1, opening), source.original_span(start + 1, opening), reading,
            ))
            index = closing + 1
            continue
        if char in pairs:
            stack.append((char, index))
        elif (stack and pairs[stack[-1][0]] == char
              and source_pairs.get(source.original[source.source_spans[stack[-1][1]][0]])
              == source.original[source.source_spans[index][0]]):
            opener, start = stack.pop()
            if opener in "([（［":
                output.append(LyricAnnotation(
                    "parenthetical", (start, index + 1), source.original_span(start, index + 1),
                    (start + 1, index), source.original_span(start + 1, index),
                ))
        index += 1
    return tuple(sorted(output, key=lambda item: item.span))
