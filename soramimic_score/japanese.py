"""Japanese lyric segmentation and pronunciation primitives."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Callable, Iterable, Sequence

import jasyllablesep


SMALL_KANA = frozenset("ァィゥェォャュョヮヵヶ")
SPECIAL_MORAS = frozenset({"ン", "ッ", "ー"})
_RUBY = re.compile(r"｜([^《｜]+)《([^《》]+)》")


def strip_ruby(text: str) -> str:
    """Keep lyric spelling, removing explicit pronunciation annotations."""
    return _RUBY.sub(lambda match: match[1], text)


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


def kana_to_moras(reading: str) -> tuple[str, ...]:
    """Split a kana pronunciation while retaining long marks and special morae."""
    result: list[str] = []
    for char in katakana(reading):
        # Only a base kana and one small vowel form a combined mora. Extra
        # small kana retain their own sound instead of creating an unsupported
        # cluster such as ウォォォ; standalone small kana are handled by G2P.
        if char in _SMALL_VOWELS and result and result[-1] in _BASE:
            result[-1] += char
        elif "ァ" <= char <= "ヶ" or char == "ー":
            result.append(char)
    return tuple(result)


def kana_to_syllables(reading: str) -> tuple[str, ...]:
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
_READING_MARKS = str.maketrans({"゛": "\u3099", "゜": "\u309a", "ゝ": "ヽ", "ゞ": "ヾ"})


def normalize_lyric_input(text: str) -> str:
    """Prepare a copy for G2P without replacing the displayed lyric spelling.

    Keep kanji, Latin letters, digits, punctuation and ruby delimiters. Only
    kana width, voicing, iteration and ligature spellings are expanded here;
    applying NFKC to the whole text would also change the ruby delimiter ｜.
    Errors refer to one-based positions in the original string, before any
    composition. Tabs and line breaks are allowed; other controls and lone
    surrogates must never reach native reading engines.
    """
    if not isinstance(text, str):
        raise ValueError("lyric input must be a string")
    result: list[str] = []

    def invalid(index: int, char: str, reason: str) -> None:
        raise ValueError(f"{reason} in lyric input at position {index + 1}: "
                         f"{char!r} (U+{ord(char):04X})")

    for index, original in enumerate(text):
        if (unicodedata.category(original) == "Cs"
                or (unicodedata.category(original) == "Cc" and original not in "\t\n\r")):
            invalid(index, original, "unsupported character")
        char = (unicodedata.normalize("NFKC", original)
                if "ｦ" <= original <= "ﾟ" else original)
        char = {"゛": "\u3099", "゜": "\u309a", "ゟ": "より", "ヿ": "コト"}.get(char, char)
        if char in {"\u3099", "\u309a"}:
            previous = result[-1] if result else ""
            if len(previous) != 1 or not ("ぁ" <= previous <= "ゖ" or "ァ" <= previous <= "ヶ"):
                invalid(index, original, "voicing mark has no kana base")
            voiced = unicodedata.normalize("NFC", previous + char)
            if len(voiced) != 1:
                # Hiragana わ/ゐ/ゑ/を have no precomposed voiced forms.
                voiced = unicodedata.normalize("NFC", katakana(previous) + char)
            if len(voiced) != 1:
                invalid(index, original, "voicing mark cannot voice its base")
            result[-1] = voiced
        elif char in "ゝゞヽヾ":
            previous = result[-1] if result else ""
            base = unicodedata.normalize("NFD", previous)[:1]
            if katakana(base) not in _BASE and base not in {"ゐ", "ゑ", "ヰ", "ヱ"}:
                invalid(index, original, "kana iteration mark has no repeatable base")
            repeated = unicodedata.normalize("NFC", base + ("\u3099" if char in "ゞヾ" else ""))
            if len(repeated) != 1:
                repeated = unicodedata.normalize("NFC", katakana(base) + "\u3099")
            if len(repeated) != 1:
                invalid(index, original, "kana iteration mark cannot voice its base")
            result.append(repeated)
        else:
            result.extend(char)
    return unicodedata.normalize("NFC", "".join(result).translate(_READING_ALIASES))


def phonemes_for_mora(mora: str, previous_vowel: str | None = None) -> tuple[str, ...]:
    """Return a compact Japanese phoneme sequence for one retained mora."""
    mora = katakana(mora)
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


def normalize_reading(reading: str) -> str:
    """Normalize a generated kana reading, rejecting unresolved characters.

    Apply after G2P, never to lyric spelling. Width, script, voicing marks,
    whitespace, historical kana and kana iteration marks have explicit rules.
    All retained moras must be pronounceable; unknown text is never filtered
    out to make a partly readable candidate appear complete.
    """
    if not isinstance(reading, str):
        raise ValueError("reading must be a string containing kana")
    compact = "".join(char for char in reading if not char.isspace())
    # Compose again after hiragana -> katakana (わ + combining dakuten -> ヷ).
    kana = unicodedata.normalize("NFC", katakana(compact.translate(_READING_MARKS)))
    expanded: list[str] = []
    for index, char in enumerate(kana):
        if char in "ヽヾ":
            previous = expanded[-1] if expanded else ""
            base = unicodedata.normalize("NFD", previous)[:1]
            if base not in _BASE and base not in {"ヰ", "ヱ"}:
                raise ValueError(f"kana iteration mark has no repeatable base at position {index + 1}")
            char = unicodedata.normalize("NFC", base + ("\u3099" if char == "ヾ" else ""))
            if len(char) != 1:
                raise ValueError(f"kana iteration mark cannot voice its base at position {index + 1}")
        expanded.append(char)
    normalized = "".join(expanded).translate(_READING_ALIASES)
    if not normalized:
        raise ValueError("reading must contain a kana pronunciation")
    for index, char in enumerate(normalized):
        if char not in _BASE and char not in SMALL_KANA and char not in SPECIAL_MORAS:
            raise ValueError(f"unsupported character in kana reading at position {index + 1}: "
                             f"{char!r} (U+{ord(char):04X})")
    for mora in kana_to_moras(normalized):
        phonemes_for_mora(mora)
    return normalized


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
    normalize_lyric_input(marked_text)  # Validate original positions before splitting.
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
    for match in _RUBY.finditer(marked_text):
        add(marked_text[cursor:match.start()], g2p(normalize_lyric_input(marked_text[cursor:match.start()])))
        surface, ruby = match.groups()
        add(surface, (ReadingCandidate(ruby, "explicit-ruby", 1.0), *g2p(normalize_lyric_input(surface))))
        cursor = match.end()
    add(marked_text[cursor:], g2p(normalize_lyric_input(marked_text[cursor:])))
    return "".join(plain_parts), tuple(spans)
