"""Japanese lyric segmentation and pronunciation primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import re
import unicodedata
from typing import Callable, Iterable, Sequence

import jasyllablesep


SMALL_KANA = frozenset("ァィゥェォャュョヮヵヶ")
SPECIAL_MORAS = frozenset({"ン", "ッ", "ー"})


def strip_ruby(text: str) -> str:
    """Keep lyric spelling, removing explicit pronunciation annotations."""
    from .lyric_input import parse_lyric_input
    source = parse_lyric_input(text)
    parts, cursor = [], 0
    for item in source.annotations:
        if item.kind == "ruby":
            start, end = item.source_span
            parts.extend((text[cursor:start], text[slice(*item.source_surface_span)]))
            cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


@dataclass(frozen=True)
class ReadingCandidate:
    kana: str
    source: str
    score: float
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class LyricSpan:
    surface: str
    surface_span: tuple[int, int]
    reading_candidates: tuple[ReadingCandidate, ...]


def katakana(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(chr(ord(char) + 96) if "ぁ" <= char <= "ゖ" else char
                   for char in normalized)


def kana_to_moras(reading: str | ValidatedReading) -> tuple[str, ...]:
    """Validate a reading at the public boundary, then return all of its morae."""
    return validate_reading(reading).moras


def _split_kana(kana: str) -> tuple[str, ...]:
    """Internal construction step, used only after the character allowlist check."""
    result: list[str] = []
    for char in kana:
        if char in _SMALL_VOWELS and result and result[-1] in _BASE:
            result[-1] += char
        else:
            result.append(char)
    return tuple(result)


def kana_to_syllables(reading: str | ValidatedReading) -> tuple[str, ...]:
    """Split normalized kana with the pipeline's explicit syllable definition.

    ``jasyllablesep`` attaches long marks, closures, and moraic nasals to the
    preceding onset.  If it cannot consume a supported kana sequence in full,
    retain every mora as its own conservative syllable instead of dropping
    unmatched text.
    """
    kana = "".join(kana_to_moras(reading))
    if not kana:
        return ()
    syllables = tuple(jasyllablesep.parse(kana))
    return syllables if "".join(syllables) == kana else kana_to_moras(kana)


_ROWS = {
    "": "アイウエオ", "k": "カキクケコ", "s": "サシスセソ", "t": "タチツテト",
    "n": "ナニヌネノ", "h": "ハヒフヘホ", "m": "マミムメモ", "r": "ラリルレロ",
    "g": "ガギグゲゴ", "z": "ザジズゼゾ", "d": "ダヂヅデド", "b": "バビブベボ",
    "p": "パピプペポ",
}
_VOWELS = "aiueo"
_BASE: dict[str, tuple[str, str]] = {
    kana: (consonant, _VOWELS[index])
    for consonant, row in _ROWS.items() for index, kana in enumerate(row)
}
_BASE.update({"ヤ": ("y", "a"), "ユ": ("y", "u"), "ヨ": ("y", "o"),
              "ワ": ("w", "a"), "ヲ": ("", "o"), "ヴ": ("v", "u")})
_IRREGULAR = {"シ": "sh", "チ": "ch", "ツ": "ts", "フ": "f", "ジ": "j", "ヂ": "j"}
_SMALL_VOWELS = {"ァ": "a", "ィ": "i", "ゥ": "u", "ェ": "e", "ォ": "o",
                  "ャ": "a", "ュ": "u", "ョ": "o", "ヮ": "a"}
_STANDALONE_SMALL_KANA = str.maketrans("ァィゥェォャュョヮヵヶ", "アイウエオヤユヨワカケ")
_READING_ALIASES = str.maketrans({
    "ヰ": "イ", "ヱ": "エ", "ヷ": "ヴァ", "ヸ": "ヴィ", "ヹ": "ヴェ", "ヺ": "ヴォ",
})


def normalize_lyric_input(text: str) -> str:
    """Return the validated analysis view; use parse_lyric_input for source offsets."""
    from .lyric_input import parse_lyric_input
    return parse_lyric_input(text).text


def phonemes_for_mora(
    mora: str | ValidatedReading, previous_vowel: str | None = None,
) -> tuple[str, ...]:
    """Validate and normalize exactly one mora before phoneme conversion."""
    reading = validate_reading(mora)
    if len(reading.moras) != 1:
        raise ValueError(f"expected one Japanese mora: {reading.kana!r}")
    return _phonemes_for_normalized_mora(reading.moras[0], previous_vowel)


def _phonemes_for_normalized_mora(mora: str, previous_vowel: str | None = None) -> tuple[str, ...]:
    # A retained small kana can stand alone at a reading boundary or after a
    # special mora. Normalize only its sound; reading and mora text stay intact.
    if len(mora) == 1:
        mora = mora.translate(_STANDALONE_SMALL_KANA)
    if mora == "ー":
        return (previous_vowel or "a",)
    if mora == "ン":
        return ("N",)
    if mora == "ッ":
        return ("q",)
    if not mora or mora[0] not in _BASE:
        raise ValueError(f"unsupported Japanese mora: {mora!r}")
    consonant, vowel = _BASE[mora[0]]
    consonant = _IRREGULAR.get(mora[0], consonant)
    if len(mora) == 2 and mora[1] in _SMALL_VOWELS:
        vowel = _SMALL_VOWELS[mora[1]]
        if mora[1] in "ャュョ" and consonant:
            consonant = consonant + "y" if consonant not in {"sh", "ch", "j"} else consonant
    elif len(mora) != 1:
        raise ValueError(f"unsupported Japanese mora: {mora!r}")
    return tuple(value for value in (consonant, vowel) if value)


class ReadingValidationError(ValueError):
    def __init__(self, text: str, index: int):
        self.position = index + 1
        char = text[index]
        super().__init__(f"unsupported character in kana reading at position {self.position}: "
                         f"{char!r} (U+{ord(char):04X})")


@dataclass(frozen=True, slots=True)
class ValidatedReading:
    """A complete pronounceable kana sequence; derived values cannot be supplied."""

    original: str
    kana: str = field(init=False)
    moras: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.original, str):
            raise ValueError("reading must be a string containing kana")
        from .lyric_input import _normalize

        text, spans = _normalize(self.original)
        characters = []
        for index, char in enumerate(text):
            if char.isspace():
                continue
            for normalized in katakana(char).translate(_READING_ALIASES):
                if (normalized not in _BASE and normalized not in SMALL_KANA
                        and normalized not in SPECIAL_MORAS):
                    position = spans[index][0]
                    raise ReadingValidationError(self.original, position)
                characters.append(normalized)
        kana = "".join(characters)
        if not kana:
            raise ValueError("reading must contain a kana pronunciation")
        moras = _split_kana(kana)
        if "".join(moras) != kana:
            raise ValueError("mora segmentation did not consume the complete reading")
        for mora in moras:
            _phonemes_for_normalized_mora(mora)
        object.__setattr__(self, "kana", kana)
        object.__setattr__(self, "moras", moras)


@lru_cache(maxsize=4096)
def _validated_reading(reading: str) -> ValidatedReading:
    return ValidatedReading(reading)


def validate_reading(reading: str | ValidatedReading) -> ValidatedReading:
    if isinstance(reading, ValidatedReading):
        return reading
    if not isinstance(reading, str):
        raise ValueError("reading must be a string containing kana")
    return _validated_reading(reading)


def normalize_reading(reading: str) -> str:
    """Normalize and validate the entire generated reading, never lyric spelling."""
    return validate_reading(reading).kana


def mora_vowel(mora: str, previous_vowel: str | None = None) -> str | None:
    phonemes = phonemes_for_mora(mora, previous_vowel)
    value = phonemes[-1]
    return value if value in {*_VOWELS, "N"} else None


def mora_distance(expected: str, observed: str) -> float:
    """Pronunciation cost with vowel disagreement weighted above consonants."""
    left = phonemes_for_mora(expected)
    right = phonemes_for_mora(observed)
    left_vowel = left[-1] if left[-1] in {*_VOWELS, "N"} else None
    right_vowel = right[-1] if right[-1] in {*_VOWELS, "N"} else None
    vowel_cost = 0.0 if left_vowel == right_vowel else 3.0
    left_consonants = left[:-1] if left_vowel else left
    right_consonants = right[:-1] if right_vowel else right
    consonant_cost = 0.0 if left_consonants == right_consonants else 1.0
    return vowel_cost + consonant_cost


G2P = Callable[[str], Sequence[ReadingCandidate]]


def simple_kana_reading(surface: str) -> tuple[ReadingCandidate, ...]:
    """Fallback for kana-only spans; callers may supply a morphological G2P."""
    # Punctuation separates a kana-only lyric; non-kana words need a real G2P.
    try:
        kana = normalize_reading("".join(char for char in surface
                                        if not unicodedata.category(char).startswith("P")))
    except ValueError:
        return ()
    # An isolated grammatical particle is unambiguous even without a tokenizer.
    # Embedded particles are deliberately left to a morphological G2P callback.
    kana = {"ハ": "ワ", "ヘ": "エ", "ヲ": "オ"}.get(kana, kana)
    return ((ReadingCandidate(kana, "known-lyrics-kana", 1.0),) if kana else ())


def spans_from_ruby_text(marked_text: str, g2p: G2P = simple_kana_reading) -> tuple[str, tuple[LyricSpan, ...]]:
    """Parse Aozora ``｜surface《reading》`` while returning plain-text spans.

    Non-ruby chunks are passed to ``g2p``. A ruby reading is a strong candidate,
    while distinct G2P alternatives are retained for later acoustic selection.
    """
    from .lyric_input import parse_lyric_input
    source = parse_lyric_input(marked_text)
    spans: list[LyricSpan] = []
    plain_parts: list[str] = []
    plain_offset = 0

    def add(surface: str, candidates: Iterable[ReadingCandidate]) -> None:
        nonlocal plain_offset
        if not surface:
            return
        unique: dict[str, ReadingCandidate] = {}
        for candidate in candidates:
            kana = normalize_reading(candidate.kana)
            if not kana or not candidate.source or not 0 <= candidate.score <= 1:
                raise ValueError("G2P candidates need kana, source, and a score in [0, 1]")
            normalized = ReadingCandidate(kana, candidate.source, candidate.score,
                                          tuple(candidate.evidence_ids))
            current = unique.get(kana)
            if current is None or normalized.score > current.score:
                unique[kana] = normalized
        if not unique:
            raise ValueError(f"no pronunciation candidate for lyric span {surface!r}")
        start = plain_offset
        plain_parts.append(surface)
        plain_offset += len(surface)
        spans.append(LyricSpan(surface, (start, plain_offset), tuple(unique.values())))

    cursor = 0
    for item in source.annotations:
        if item.kind != "ruby":
            continue
        start, end = item.source_span
        add(marked_text[cursor:start], g2p(normalize_lyric_input(marked_text[cursor:start])))
        surface = marked_text[slice(*item.source_surface_span)]
        add(surface, (ReadingCandidate(item.reading, "explicit-ruby", 1.0),
                      *g2p(normalize_lyric_input(surface))))
        cursor = end
    add(marked_text[cursor:], g2p(normalize_lyric_input(marked_text[cursor:])))
    return "".join(plain_parts), tuple(spans)
