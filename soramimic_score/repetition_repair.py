"""Apply acoustically measured repetitions without erasing adjacent singing."""
from __future__ import annotations

from dataclasses import replace

from .japanese import kana_to_moras, mora_vowel


def _foreign_text(text):
    letters = [char for char in text if char.isalpha()]
    latin = sum(char.isascii() for char in letters)
    return latin >= 4 and latin >= len(letters) * .5


def has_foreign_transcript(repetition, lines):
    """Use the relaxed repair for a motif ASR has turned into Latin prose.

    Repeated melody is also common under different, valid Japanese sentences.
    Those need a lexical recognizer, not pronunciation copied from a refrain.
    """
    for line in lines:
        if not _foreign_text(line.text):
            continue
        if any(max(0., min(line.end_sec, copy.end_sec) - max(line.start_sec, copy.start_sec))
               >= (copy.end_sec - copy.start_sec) * .5 for copy in repetition.occurrences):
            return True
    return False


def copies_needing_repair(group, lines, moras):
    missing = tuple(needs_pronunciation_repair(group.kana, copy, moras)
                    for copy in group.occurrences)
    if not any(missing):
        return ()
    # Once a collapsed foreign transcript needs repair, give its other copies
    # the same pronunciation and measured timing. Mixing repaired and old
    # foreign readings would leave the refrain's syllable count inconsistent.
    return tuple(copy for copy, repair in zip(group.occurrences, missing, strict=True)
                 if repair or has_foreign_transcript(replace(group, occurrences=(copy,)), lines))


def phase_aligned_moras(group, copies):
    """Transfer observed pronunciation phases through measured acoustic copies."""
    from .audio import AlignedMora
    moras = kana_to_moras(group.kana)
    if len(group.mora_offsets_sec) != len(moras):
        raise ValueError("each repeated mora needs an observed acoustic phase")
    if any(b <= a for a, b in zip(group.mora_offsets_sec, group.mora_offsets_sec[1:])):
        raise ValueError("repeated mora phases must be strictly ordered")
    return tuple(AlignedMora(
        line_index, index, kana, copy.start_sec + offset,
        min(copy.end_sec, copy.start_sec + offset + .04,
            copy.start_sec + group.mora_offsets_sec[index + 1]
            if index + 1 < len(moras) else copy.end_sec),
        0., "acoustic-repetition-phase",
    ) for line_index, copy in enumerate(copies)
      for index, (kana, offset) in enumerate(zip(moras, group.mora_offsets_sec, strict=True)))


def vowel_distance(first, second):
    """Pronunciation comparison ignores consonants and non-vocalic morae."""
    def vowels(kana):
        result = []
        for mora in kana_to_moras(kana):
            vowel = mora_vowel(mora)
            if vowel in {"a", "i", "u", "e", "o"} and mora != "ー":
                result.append(vowel)
        return result
    a, b = vowels(first), vowels(second)
    row = list(range(len(b) + 1))
    for i, x in enumerate(a):
        following = [i + 1]
        for j, y in enumerate(b):
            following.append(min(row[j] + int(x != y), row[j + 1] + 1, following[-1] + 1))
        row = following
    return row[-1] / max(1, len(a), len(b))


def needs_pronunciation_repair(kana, occurrence, moras):
    owned = [m for m in moras if occurrence.start_sec <= (m.start_sec + m.end_sec) / 2
             < occurrence.end_sec]
    current = "".join(m.kana for m in sorted(owned, key=lambda m: (m.start_sec, m.end_sec)))
    # Already usable vowel coverage is sufficient. Do not rewrite a recognized
    # word just because a phonetic decoder chose different consonants.
    return not owned or vowel_distance(kana, current) > .35


def replace_pronunciation_spans(lines, readings, moras, replacements):
    """Keep untouched lines verbatim and retain any uncovered line fragments."""
    from .audio import LyricLine, ReadingSelection
    entries = []
    spans = []
    for start, end in sorted((line.start_sec, line.end_sec) for line, _, _ in replacements):
        if spans and start <= spans[-1][1] + .10:
            spans[-1] = (spans[-1][0], max(end, spans[-1][1]))
        else:
            spans.append((start,end))
    for index, (line, reading) in enumerate(zip(lines, readings, strict=True)):
        original = tuple(m for m in moras if m.line_index == index)
        covered = sum(max(0., min(b,line.end_sec)-max(a,line.start_sec)) for a,b in spans)
        duration = line.end_sec - line.start_sec
        if (_foreign_text(line.text) and covered >= duration * .8
                and duration - covered <= .25):
            continue
        kept = []
        for mora in original:
            midpoint = (mora.start_sec + mora.end_sec) / 2
            if any(a <= midpoint < b for a, b in spans):
                continue
            start, end = mora.start_sec, mora.end_sec
            for a, b in spans:
                if start < b and end > a:
                    if midpoint < a:
                        end = min(end, a)
                    else:
                        start = max(start, b)
            if end > start:
                kept.append(replace(mora, start_sec=start, end_sec=end))
        if len(kept) == len(original):
            if original and any(line.start_sec < b and line.end_sec > a for a, b in spans):
                line = replace(line, start_sec=min(m.start_sec for m in kept),
                               end_sec=max(m.end_sec for m in kept))
            entries.append((line, reading, tuple(kept)))
            continue
        runs = []
        for mora in kept:
            if not runs or mora.mora_index != runs[-1][-1].mora_index + 1:
                runs.append([mora])
            else:
                runs[-1].append(mora)
        for run in runs:
            kana = "".join(m.kana for m in run)
            entries.append((LyricLine(kana, min(m.start_sec for m in run),
                                      max(m.end_sec for m in run)),
                            ReadingSelection(kana, "retained-pronunciation", 0., (kana,),
                                             {"original_surface": line.text,
                                              "confidence_available": False}),
                            tuple(replace(m, mora_index=i) for i, m in enumerate(run))))
    entries.extend(replacements)
    entries.sort(key=lambda e: (e[0].start_sec, e[0].end_sec))
    return (tuple(e[0] for e in entries), tuple(e[1] for e in entries),
            tuple(replace(m, line_index=i) for i, e in enumerate(entries) for m in e[2]))


def missing_note_windows(occurrences, notes):
    """Retry melody only where a measured repeated voice lacks note coverage."""
    windows = []
    for occurrence in sorted(occurrences, key=lambda c: c.start_sec):
        start, end = occurrence.start_sec, occurrence.end_sec
        covered = sum(max(0., min(end, n.end_sec) - max(start, n.start_sec)) for n in notes)
        if covered / (end - start) >= .5:
            continue
        if windows and start <= windows[-1][1] + .15:
            windows[-1] = (windows[-1][0], max(end, windows[-1][1]))
        else:
            windows.append((start, end))
    return tuple(windows)


def merge_recovered_notes(notes, recovered, start, end):
    """Never overwrite existing pitches or fill beyond the confirmed voice."""
    additions = []
    for note in recovered:
        fragments = [(max(start, note.start_sec), min(end, note.end_sec))]
        for existing in (*notes, *additions):
            next_fragments = []
            for a, b in fragments:
                if a < existing.end_sec and b > existing.start_sec:
                    next_fragments.extend(((a, min(b, existing.start_sec)),
                                           (max(a, existing.end_sec), b)))
                else:
                    next_fragments.append((a, b))
            fragments = next_fragments
        additions.extend(replace(note, start_sec=a, end_sec=b) for a, b in fragments if b-a >= .04)
    return tuple(sorted((*notes, *additions), key=lambda n: (n.start_sec, n.end_sec)))
