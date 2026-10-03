"""Keep independently recognized pronunciations in uncovered singing spans."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from .japanese import kana_to_moras, phonemes_for_mora


@dataclass(frozen=True)
class PhoneticMora:
    kana: str
    start_sec: float
    end_sec: float
    source: str = "romaji-asr-ctc"


def romaji_to_kana(token: str) -> str | None:
    """Translate supported mora tokens, leaving unknown phonemes unresolved."""
    if token == "N":
        return "ン"
    if token == "cl":
        return "ッ"
    bases = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポヴ"
    compounds = ([base + small for base in "キギシジチヂニヒビピミリ" for small in "ャュョ"]
                 + "イェ ウァ ウィ ウェ ウォ ヴァ ヴィ ヴェ ヴォ ヴュ クァ クィ クェ クォ グァ グィ グェ グォ シェ ジェ チェ ツァ ツィ ツェ ツォ ティ トゥ ディ ドゥ テュ デュ ファ フィ フェ フォ フュ".split())
    for kana in [*bases, *compounds]:
        if "".join(phonemes_for_mora(kana)) == token:
            return kana
    return None


def uncovered_note_windows(moras, notes):
    """Find notes outside the aligned lyric spans, including broad ASR windows."""
    coverage = {}
    for mora in moras:
        a, b = coverage.get(mora.line_index, (mora.start_sec, mora.end_sec))
        coverage[mora.line_index] = (min(a, mora.start_sec), max(b, mora.end_sec))
    missing = [note for note in notes if not any(
        a - .12 <= (note.start_sec + note.end_sec) / 2 <= b + .12
        for a, b in coverage.values())]
    groups = []
    for note in missing:
        if not groups or note.start_sec - groups[-1][-1].end_sec > .5:
            groups.append([note])
        else:
            groups[-1].append(note)
    return tuple((group[0].start_sec, group[-1].end_sec)
                 for group in groups if len(group) >= 2
                 and group[-1].end_sec - group[0].start_sec >= .25)


def add_phonetic_fallback(lines, readings, moras, notes, events):
    """Preserve known lyrics and append only note-supported, uncovered mora runs."""
    from .audio import AlignedMora, LyricLine, ReadingSelection
    from .ir import Evidence

    covered = {}
    for m in moras:
        a, b = covered.get(m.line_index, (m.start_sec, m.end_sec))
        covered[m.line_index] = (min(a, m.start_sec), max(b, m.end_sec))
    events = tuple(events)
    for e in events:
        if (not isinstance(e, PhoneticMora) or not math.isfinite(e.start_sec + e.end_sec)
                or e.start_sec < 0 or e.end_sec <= e.start_sec or not e.source
                or kana_to_moras(e.kana) != (e.kana,)):
            raise ValueError("phonetic recognition needs finite, timed kana morae")
    selected = []
    previous_end = 0.
    for e in sorted(events, key=lambda e: (e.start_sec, e.end_sec)):
        if any(e.start_sec < b + .12 and e.end_sec > a - .12
               for a, b in covered.values()):
            continue
        if not any(e.start_sec < n.end_sec + .12 and e.end_sec > n.start_sec - .12
                   for n in notes):
            continue
        if e.start_sec < previous_end:
            # Overlapping context windows can decode the same mora twice.
            continue
        selected.append(e)
        previous_end = e.end_sec
    groups = []
    for e in selected:
        crosses_lyrics = bool(groups) and any(
            groups[-1][-1].end_sec < b and e.start_sec > a
            for a, b in covered.values())
        if not groups or e.start_sec - groups[-1][-1].end_sec > .5 or crosses_lyrics:
            groups.append([e])
        else:
            groups[-1].append(e)
    additions = [g for g in groups if len(g) >= 3
                 and any(e.kana not in {"ン", "ッ", "ー"} for e in g)
                 and g[-1].end_sec - g[0].start_sec >= .25]
    if not additions:
        return tuple(lines), tuple(readings), tuple(moras), ()
    entries = []
    for index, (line, reading) in enumerate(zip(lines, readings, strict=True)):
        owned = tuple(m for m in moras if m.line_index == index)
        if index in covered:
            a, b = covered[index]
            start = a if any(g[0].start_sec < a and g[-1].end_sec > line.start_sec
                             for g in additions) else line.start_sec
            end = b if any(g[-1].end_sec > b and g[0].start_sec < line.end_sec
                           for g in additions) else line.end_sec
            line = replace(line, start_sec=start, end_sec=end)
        entries.append((line, reading, owned, None))
    for group in additions:
        text = "".join(e.kana for e in group)
        line = LyricLine(text, group[0].start_sec, group[-1].end_sec)
        reading = ReadingSelection(text, "romaji-asr-katakana", 0., (text,),
                                   {"lyric_kind": "phonetic-fallback",
                                    "confidence_available": False})
        aligned = tuple(AlignedMora(0, i, e.kana, e.start_sec, e.end_sec,
                                   0., e.source) for i, e in enumerate(group))
        entries.append((line, reading, aligned, group))
    entries.sort(key=lambda e: e[0].start_sec)
    output_moras, evidence = [], []
    for index, (line, reading, owned, group) in enumerate(entries):
        output_moras.extend(replace(m, line_index=index) for m in owned)
        if group is not None:
            evidence.append(Evidence(
                f"audio-phonetic-fallback-{index}", group[0].source,
                "lyric-phonetic-fallback", 0.,
                {"utterance_id": f"u{index}", "surface": line.text,
                 "start_sec": line.start_sec, "end_sec": line.end_sec,
                 "confidence_available": False, "timing": "greedy-ctc-token-spans",
                 "semantic_lyrics_available": False, "mora_count": len(group)},
            ))
    return (tuple(e[0] for e in entries), tuple(e[1] for e in entries),
            tuple(output_moras), tuple(evidence))
