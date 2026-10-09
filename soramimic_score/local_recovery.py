"""Bounded local Whisper retries for likely missing sung lyrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import re
import math
import statistics
from typing import TYPE_CHECKING
import unicodedata

from .japanese import kana_to_moras, katakana

if TYPE_CHECKING:
    from .audio import LyricLine, MelodyNote


def _normalized(text: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKC", text).casefold()
                   if unicodedata.category(char)[0] not in {"P", "Z"})


def coalesce_repeated_suffix_fragments(lines: Sequence[LyricLine]
                                       ) -> tuple[tuple[LyricLine, ...], tuple[tuple[int, int], ...]]:
    """Join a short Whisper suffix only if another refrain contains that suffix."""
    normalized = [_normalized(line.text) for line in lines]
    candidates = set()
    for index in range(1, len(lines)):
        line, previous = lines[index], lines[index - 1]
        fragment = normalized[index]
        if not re.fullmatch(r"[ぁ-んァ-ヶ一-龯々〆ヵヶー]{2,4}", fragment):
            continue
        if (line.start_sec is None or line.end_sec is None
                or previous.start_sec is None or previous.end_sec is None
                or line.end_sec - line.start_sec > 1.25
                or abs(line.start_sec - previous.end_sec) > .15
                or line.end_sec - previous.start_sec > 6):
            continue
        combined = len(normalized[index - 1]) + len(fragment)
        if any(other not in {index - 1, index} and len(surface) > len(fragment)
               and surface.endswith(fragment)
               and .67 <= combined / len(surface) <= 1.5
               for other, surface in enumerate(normalized)):
            candidates.add(index)
    merged, evidence = [], []
    index = 0
    while index < len(lines):
        if index + 1 in candidates:
            left, right = lines[index:index + 2]
            separator = " " if left.text[-1:].isascii() and right.text[:1].isascii() else ""
            merged.append(replace(left, text=left.text.rstrip() + separator + right.text.lstrip(),
                                  end_sec=right.end_sec))
            evidence.append((index, index + 1))
            index += 2
        else:
            merged.append(lines[index])
            index += 1
    return tuple(merged), tuple(evidence)


_VOCALIZATION_KANA = frozenset("アァイィウゥエェオォラナダファハヤワンー")
_LATIN_VOCALIZATION = re.compile(r"wow|la|na|da|fa|ha|ya|a+h*|i+|u+h*|e+|o+h*")
_LATIN_MORAS = {"wow": ("ワ", "ウ"), "la": ("ラ",), "na": ("ナ",),
                "da": ("ダ",), "fa": ("ファ",), "ha": ("ハ",), "ya": ("ヤ",)}


def _vocalization_moras(text: str) -> tuple[str, ...] | None:
    normalized = katakana(_normalized(text)).replace("ー", "")
    if not normalized:
        return None
    if re.fullmatch(r"[ァ-ヶ]+", normalized):
        if any(char not in _VOCALIZATION_KANA for char in normalized):
            return None
        moras = list(kana_to_moras(normalized))
    elif normalized.isascii():
        moras = []
        cursor = 0
        while cursor < len(normalized):
            match = _LATIN_VOCALIZATION.match(normalized, cursor)
            if match is None:
                return None
            token = match.group()
            moras.extend(_LATIN_MORAS[token] if token in _LATIN_MORAS else
                         ({"a": "ア", "i": "イ", "u": "ウ",
                           "e": "エ", "o": "オ"}[token[0]],))
            cursor = match.end()
    else:
        return None
    return tuple(moras)


def repeated_vocalization_period(text: str) -> tuple[str, ...] | None:
    """Find a short repeating pure vocalization, excluding ordinary lyrics."""
    moras = _vocalization_moras(text)
    if not moras:
        return None
    for width in range(1, min(3, len(moras) // 2) + 1):
        if width > 1 and len(moras) < width * 3:
            continue
        if all(mora == moras[index % width] for index, mora in enumerate(moras)):
            return tuple(moras[:width])
    return None


def normalize_repeated_vocalization(line: LyricLine) -> LyricLine:
    """Preserve Whisper's observed repeat count in kana, including Latin ``la``."""
    if repeated_vocalization_period(line.text) is None:
        return line
    moras = _vocalization_moras(line.text)
    return replace(line, text="".join(moras)) if moras else line


def adjacent_repeat_groups(lines: Sequence[LyricLine]
                           ) -> tuple[tuple[int, int, tuple[str, ...]], ...]:
    """Group touching Whisper fragments of the same nonlexical refrain."""
    groups = []
    start = 0
    while start < len(lines):
        period = repeated_vocalization_period(lines[start].text)
        if period is None:
            start += 1
            continue
        end = start + 1
        while end < len(lines):
            following = repeated_vocalization_period(lines[end].text)
            if (following != period or lines[end].start_sec is None
                    or lines[end - 1].end_sec is None
                    or lines[end].end_sec is None
                    or lines[start].start_sec is None
                    or lines[end].start_sec - lines[end - 1].end_sec > .25
                    or lines[end].end_sec - lines[start].start_sec > 16):
                break
            end += 1
        if end - start >= 2:
            groups.append((start, end, period))
        start = end
    return tuple(groups)


def duration_repeated_vocalization_candidate(
    source: LyricLine, recovered: Sequence[LyricLine], count: int,
) -> LyricLine | None:
    """Repeat a short phrase only when every local retry hears its mora family."""
    period = repeated_vocalization_period(source.text)
    if (count < 2 or not recovered or period is None
            or {repeated_vocalization_period(item.text) for item in recovered} != {period}):
        return None
    return replace(source, text=source.text * count)


def has_tandem_repeat_note_support(text: str, *, source_moras: int,
                                   recovered_moras: int, note_count: int,
                                   median_notes_per_mora: float) -> bool:
    """Permit a weak CTC retry only when a repeated phrase explains missing notes."""
    normalized = _normalized(text)
    minimum = max(4, math.ceil(len(normalized) * .45))
    tandem = any(normalized[start:start + width]
                 == normalized[start + width:start + width * 2]
                 for width in range(len(normalized) // 2, minimum - 1, -1)
                 for start in range(len(normalized) - width * 2 + 1))
    required = source_moras + max(2, math.ceil(source_moras * .25))
    if (not tandem or recovered_moras < required
            or recovered_moras > note_count * 2):
        return False
    return (abs(note_count - median_notes_per_mora * recovered_moras)
            < abs(note_count - median_notes_per_mora * source_moras))


def is_pathological_repeated_vocalization(
    line: LyricLine, notes: Sequence[MelodyNote],
) -> bool:
    """Reject decoder runaway, preserving ordinary observed repetitions."""
    moras = _vocalization_moras(line.text)
    if (not moras or repeated_vocalization_period(line.text) is None
            or line.start_sec is None or line.end_sec is None):
        return False
    note_count = sum(line.start_sec <= (note.start_sec + note.end_sec) / 2 < line.end_sec
                     for note in notes)
    duration = max(1e-6, line.end_sec - line.start_sec)
    return len(moras) > max(64, note_count * 2) or len(moras) / duration > 8


def expand_repeated_vocalization_from_kana(
    line: LyricLine, evidence_text: str, notes: Sequence[MelodyNote],
) -> LyricLine | None:
    """Use KanaWhisper for count only; spell every mora from Whisper's period."""
    period = repeated_vocalization_period(line.text)
    if period is None or line.start_sec is None or line.end_sec is None:
        return None
    source = _vocalization_moras(line.text) or ()
    evidence = kana_to_moras("".join(re.findall(r"[ァ-ヶー]+",
                              katakana(evidence_text).replace("ヲ", "オ"))).replace("ー", ""))
    note_count = sum(line.start_sec <= (note.start_sec + note.end_sec) / 2 < line.end_sec
                     for note in notes)
    if not evidence or note_count < 2:
        return None
    gain = len(source) + max(2, math.ceil(len(source) * .25))
    if (len(evidence) < gain or len(evidence) < math.ceil(note_count * .25)
            or len(evidence) > math.ceil(note_count * 1.5)
            or len(evidence) / max(1e-6, line.end_sec - line.start_sec) > 8):
        return None
    similarity = max(
        sum(mora == period[(index + phase) % len(period)]
            for index, mora in enumerate(evidence)) / len(evidence)
        for phase in range(len(period))
    )
    if similarity < .65:
        return None
    return replace(line, text="".join(period[index % len(period)]
                                      for index in range(len(evidence))))


def deficit_windows(lines: Sequence[LyricLine], notes: Sequence[MelodyNote],
                    mora_counts: Sequence[int]) -> tuple[tuple[int, float, float], ...]:
    """Find long note rich lines whose lyric detail is short for this song."""
    if len(lines) != len(mora_counts):
        raise ValueError("one mora count is required for every line")
    rows = []
    for index, (line, count) in enumerate(zip(lines, mora_counts, strict=True)):
        if line.start_sec is None or line.end_sec is None:
            continue
        in_line = [note for note in notes
                   if line.start_sec <= (note.start_sec + note.end_sec) / 2 < line.end_sec]
        effective = count + len(re.findall(r"[A-Za-z]+", line.text))
        rows.append((index, line, count, effective, in_line))
    ratios = [len(found) / effective for _, _, _, effective, found in rows
              if effective >= 4 and len(found) >= 2]
    if not ratios:
        return ()
    median = statistics.median(ratios)
    windows = []
    for index, line, count, effective, found in rows:
        period = repeated_vocalization_period(line.text)
        if _vocalization_moras(line.text) is not None:
            if period is None or len(period) == 1:
                continue
            peer_durations = [other.end_sec - other.start_sec
                              for other_index, other in enumerate(lines)
                              if other_index != index
                              and other.start_sec is not None and other.end_sec is not None
                              and _normalized(other.text) == _normalized(line.text)]
            if (not peer_durations or
                    (line.end_sec - line.start_sec) / statistics.median(peer_durations) < 1.5):
                continue
        if (len(found) < 7 or len(found) / max(effective, 1) < 1.5
                or len(found) - median * effective < 4):
            continue
        groups: list[list[MelodyNote]] = []
        for note in sorted(found, key=lambda item: item.start_sec):
            if not groups or note.start_sec - groups[-1][-1].end_sec > .32:
                groups.append([note])
            else:
                groups[-1].append(note)
        groups = [group for group in groups
                  if len(group) >= 4 and group[-1].end_sec - group[0].start_sec >= .6]
        for group_index, group in enumerate(groups):
            start = max(line.start_sec, group[0].start_sec - .5)
            end = min(line.end_sec, group[-1].end_sec + .5)
            if group_index:
                previous = groups[group_index - 1]
                start = max(start, (previous[-1].end_sec + group[0].start_sec) / 2)
            if group_index + 1 < len(groups):
                following = groups[group_index + 1]
                end = min(end, (group[-1].end_sec + following[0].start_sec) / 2)
            if end > start:
                windows.append((index, start, end))
    return tuple(windows)


def uncovered_note_windows(lines: Sequence[LyricLine], notes: Sequence[MelodyNote]
                           ) -> tuple[tuple[float, float], ...]:
    """Find long melody runs with no overlapping recognized lyric line."""
    uncovered = [note for note in notes if not any(
        line.start_sec is not None and line.end_sec is not None
        and line.start_sec < note.end_sec and line.end_sec > note.start_sec
        for line in lines)]
    groups: list[list[MelodyNote]] = []
    for note in uncovered:
        crosses_line = bool(groups) and any(
            line.start_sec is not None and line.end_sec is not None
            and line.start_sec < note.start_sec
            and line.end_sec > groups[-1][-1].end_sec
            for line in lines)
        if not groups or note.start_sec - groups[-1][-1].end_sec > .35 or crosses_line:
            groups.append([note])
        else:
            groups[-1].append(note)
    windows = []
    for group in groups:
        start, end = group[0].start_sec, group[-1].end_sec
        if len(group) < 5 or end - start < 1.5 or end - start > 20:
            continue
        left = max((line.end_sec for line in lines if line.end_sec is not None
                    and line.end_sec <= start), default=0.)
        right = min((line.start_sec for line in lines if line.start_sec is not None
                     and line.start_sec >= end), default=end + .3)
        windows.append((max(left, start - .3), min(right, end + .3)))
    return tuple(windows[:4])


def unowned_note_windows(document, lines: Sequence[LyricLine]
                         ) -> tuple[tuple[float, float, int], ...]:
    """Find long lyric-free runs from Stage 3 note ownership, as Video does."""
    notes_by_id = {note.id: note for note in document.note_candidates}
    unowned_ids = {note_id for link in document.links
                   if link.operation == "note_only" and not link.singing_unit_ids
                   for note_id in link.note_candidate_ids}
    unowned = sorted((notes_by_id[note_id] for note_id in unowned_ids
                      if note_id in notes_by_id),
                     key=lambda note: (note.start_sec, note.end_sec, note.id))
    unowned = [note for note in unowned if not any(
        line.start_sec is not None and line.end_sec is not None
        and line.start_sec < note.end_sec and line.end_sec > note.start_sec
        for line in lines)]
    clusters = []
    for note in unowned:
        crosses_line = bool(clusters) and any(
            line.start_sec is not None and line.end_sec is not None
            and line.start_sec < note.start_sec
            and line.end_sec > clusters[-1][-1].end_sec for line in lines)
        if not clusters or note.start_sec - clusters[-1][-1].end_sec > .32 or crosses_line:
            clusters.append([note])
        else:
            clusters[-1].append(note)
    seeds = [cluster for cluster in clusters if len(cluster) >= 4
             and cluster[-1].end_sec - cluster[0].start_sec >= .6]
    merged = []
    for seed in seeds:
        start = merged[-1][0].start_sec if merged else seed[0].start_sec
        crosses_line = any(line.start_sec is not None and line.end_sec is not None
                           and line.start_sec < seed[-1].end_sec
                           and line.end_sec > start for line in lines)
        if not merged or seed[0].start_sec - merged[-1][-1].end_sec > 2 or crosses_line:
            merged.append(list(seed))
        else:
            merged[-1].extend(seed)
    windows = []
    for group in merged:
        start, end = group[0].start_sec, group[-1].end_sec
        count = sum(note.start_sec >= start and note.end_sec <= end for note in unowned)
        if count >= 8 and end - start >= 4:
            windows.append((start, end, count))
    return tuple(windows)
