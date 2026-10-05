"""Find omitted phrase repetitions using independently timed pronunciations.

An aligner can stretch one occurrence over several repetitions.  Neither its
occupied interval nor a high alignment score is evidence for the count.  This
module instead requires distinct phonetic matches. Approximate matches also
need melody support; exact matches can recover singing missed by the note model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import unicodedata

from .japanese import kana_to_moras, mora_vowel
from .phonetic_fallback import PhoneticMora


@dataclass(frozen=True)
class PhraseOccurrence:
    start_sec: float
    end_sec: float
    distance: float
    exact_moras: int
    observed_kana: str
    note_support: float


@dataclass(frozen=True)
class RepeatedLine:
    line_index: int
    text: str
    kana: str
    original_count: int
    occurrences: tuple[PhraseOccurrence, ...]


def _elongation(previous: str, current: str) -> bool:
    return (current in {"ア", "イ", "ウ", "エ", "オ"}
            and (mora_vowel(previous) == mora_vowel(current)
                 or (mora_vowel(previous), mora_vowel(current))
                 in {("o", "u"), ("e", "i")}))


def _phrase_moras(kana: str) -> tuple[str, ...]:
    result = []
    for mora in kana_to_moras(kana):
        if mora == "ー" or result and _elongation(result[-1], mora):
            continue
        result.append(mora)
    return tuple(result)


def _repeat_unit(text, kana):
    """Split only when both the actual surface and reading repeat exactly."""
    surface = "".join(char for char in unicodedata.normalize("NFKC", text)
                      if char.isalnum() or char == "ー")
    moras = kana_to_moras(kana)
    for count in range(min(16, len(surface), len(moras)), 1, -1):
        if len(surface) % count or len(moras) % count:
            continue
        word, reading = surface[:len(surface) // count], moras[:len(moras) // count]
        if word * count == surface and reading * count == moras:
            return word, "".join(reading), count
    return text, kana, 1


def repetition_windows(lines, readings):
    """Inspect short phrases even when their alignment fills the whole window."""
    windows = []
    for line, reading in zip(lines, readings, strict=True):
        _text, kana, _count = _repeat_unit(line.text, reading.kana)
        phrase = _phrase_moras(kana)
        if (line.start_sec is None or line.end_sec is None
                or not 4 <= len(phrase) <= 24
                or len(set(phrase) - {"ン", "ッ"}) < 3
                or not .8 <= line.end_sec - line.start_sec <= 60):
            continue
        windows.append((line.start_sec, line.end_sec))
    return tuple(windows)


def _distance_prefixes(target, observed):
    """Weighted edit distance with an exact-match count, for every prefix."""
    row = [(float(index), 0) for index in range(len(observed) + 1)]
    for index, expected in enumerate(target):
        following = [(float(index + 1), 0)]
        for offset, actual in enumerate(observed):
            same = expected == actual
            vowel = mora_vowel(expected)
            substitution = (0. if same else .5 if vowel is not None
                            and vowel == mora_vowel(actual) else 1.)
            cost, exact = row[offset]
            following.append(min(
                ((cost + substitution, exact + int(same)),
                 (row[offset + 1][0] + 1., row[offset + 1][1]),
                 (following[offset][0] + 1., following[offset][1])),
                key=lambda item: (item[0], -item[1]),
            ))
        row = following
    return row


def _occurrences(target, events, notes):
    units = [event for event in events if event.kana != "ー"]
    candidates = []
    slack = max(1, math.floor(len(target) * .2))
    for first in range(len(units)):
        # Normalize within each candidate, not across its left boundary: a
        # preceding phrase ending in /a/ must not swallow an initial ア.
        available = []
        for event in units[first:first + 3 * len(target) + slack]:
            if available and event.start_sec - available[-1].end_sec > 1.:
                break
            if available and _elongation(available[-1].kana, event.kana):
                continue
            available.append(event)
            if len(available) == len(target) + slack:
                break
        distances = _distance_prefixes(target, [event.kana for event in available])
        for count in range(max(3, len(target) - slack), len(available) + 1):
            cost, exact = distances[count]
            if cost / len(target) > .2 or exact < max(3, math.ceil(len(target) * .7)):
                continue
            group = available[:count]
            if (group[-1].end_sec - group[0].start_sec < .25
                    or any(b.start_sec - a.end_sec > 1.
                           for a, b in zip(group, group[1:]))):
                continue
            support = sum(any(note.start_sec - .12 <= event.start_sec <= note.end_sec + .12
                              for note in notes) for event in group)
            if cost > 0 and support / len(group) < .8:
                continue
            candidates.append(PhraseOccurrence(
                group[0].start_sec, group[-1].end_sec, cost / len(target), exact,
                "".join(event.kana for event in group), support / len(group),
            ))
    selected = []
    for candidate in sorted(candidates, key=lambda item: (item.distance, -item.exact_moras,
                                                         item.start_sec, item.end_sec)):
        if any(candidate.start_sec < item.end_sec and candidate.end_sec > item.start_sec
               for item in selected):
            continue
        selected.append(candidate)
    return tuple(sorted(selected, key=lambda item: item.start_sec))


def find_repeated_lines(lines, readings, notes, events):
    """Propose copies of an existing line, never a guessed new lexical phrase."""
    events = tuple(events)
    for event in events:
        if (not isinstance(event, PhoneticMora)
                or not math.isfinite(event.start_sec + event.end_sec)
                or not 0 <= event.start_sec < event.end_sec
                or kana_to_moras(event.kana) != (event.kana,) or not event.source):
            raise ValueError("phonetic recognition needs finite, timed kana morae")
    events = sorted(set(events), key=lambda event: (event.start_sec, event.end_sec))
    windows = set(repetition_windows(lines, readings))
    result = []
    for index, (line, reading) in enumerate(zip(lines, readings, strict=True)):
        if (line.start_sec, line.end_sec) not in windows:
            continue
        local = [event for event in events
                 if line.start_sec <= event.start_sec and event.end_sec <= line.end_sec]
        # Competing overlapping decodes are not independent occurrences.
        if any(a.end_sec > b.start_sec for a, b in zip(local, local[1:])):
            continue
        text, kana, count = _repeat_unit(line.text, reading.kana)
        found = _occurrences(_phrase_moras(kana), local, notes)
        if len(found) > count:
            result.append(RepeatedLine(index, text, kana, count, found))
    return tuple(result)
