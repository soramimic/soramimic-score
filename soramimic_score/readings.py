"""Yomi pronunciations, dictionary alternatives and acoustic selection."""
from __future__ import annotations

import csv
from dataclasses import replace
from functools import lru_cache
from itertools import islice, product
import math
import re
from threading import Lock
import unicodedata

from .audio import LyricLine, ReadingSelection
from .japanese import _RUBY, kana_to_moras, katakana, mora_vowel, normalize_reading


_YOMI_LOCK = Lock()
_LATIN_OR_DIGIT = re.compile(r"[0-9０-９A-Za-zＡ-Ｚａ-ｚ]")


def _vowels(reading):
    """Use Video's vowel-sequence guard for automatic readings."""
    result = []
    for mora in kana_to_moras(_comparison_kana(reading)):
        if mora == "ー":
            continue
        try:
            vowel = mora_vowel(mora)
        except ValueError:
            continue
        if vowel is not None and vowel in "aiueo":
            result.append(vowel)
    return tuple(result)


def _automatic_candidates(text, candidates):
    """Keep bounded, vowel-distinct automatic readings, as Video does."""
    if not candidates:
        return ()
    default = candidates[0]
    yomi = [candidate for candidate in candidates
            if candidate[1] == "soramimic-yomi"]
    diverse_yomi = yomi[:1]
    seen_mora_counts = {len(kana_to_moras(yomi[0][0]))} if yomi else set()
    for candidate in yomi[1:]:
        count = len(kana_to_moras(candidate[0]))
        if count not in seen_mora_counts and len(diverse_yomi) < 8:
            diverse_yomi.append(candidate)
            seen_mora_counts.add(count)
    for candidate in yomi[1:]:
        if len(diverse_yomi) >= 8:
            break
        if candidate not in diverse_yomi:
            diverse_yomi.append(candidate)
    unidic = [candidate for candidate in candidates
              if candidate[1] == "unidic-lite"][:2]
    linguistic = tuple(diverse_yomi if _LATIN_OR_DIGIT.search(text)
                       else [*diverse_yomi, *unidic])
    if not linguistic:
        linguistic = candidates[:1]
    selected = [default]
    default_vowels = _vowels(default[0])
    for candidate in linguistic[1:]:
        if _vowels(candidate[0]) != default_vowels:
            selected.append(candidate)
    unique = []
    seen = set()
    for candidate in selected:
        key = _comparison_kana(candidate[0])
        if key and key not in seen:
            seen.add(key)
            unique.append(candidate)
    return tuple(unique)


def _node_reading(node):
    parts, surfaces = [], []
    while node is not None:
        if node.surface:
            fields = next(csv.reader([node.feature]))
            if fields[0] != "補助記号":
                pronunciation = fields[9] if len(fields) > 9 else "*"
                if pronunciation in {"", "*"}:
                    pronunciation = katakana(node.surface)
                try:
                    kana = normalize_reading(pronunciation)
                except ValueError as exc:
                    raise ValueError(f"Cannot determine Japanese pronunciation: {node.surface!r}") from exc
                parts.append(kana)
                surfaces.append(node.surface)
        node = node.next
    if not parts:
        raise ValueError("No Japanese pronunciation in lyric line")
    return "".join(parts), tuple(surfaces)


def dictionary_candidates(lines):
    """Retain distinct readings from eight dictionary analyses, including alternate splits."""
    import MeCab
    import unidic_lite

    tagger = MeCab.Tagger(f'-d "{unidic_lite.DICDIR}"')
    output = []
    for line in lines:
        primary, _ = _node_reading(tagger.parseToNode(line.text))
        candidates = [primary]
        tagger.parseNBestInit(line.text)
        for _ in range(8):
            node = tagger.nextNode()
            if node is None:
                break
            try:
                kana, _ = _node_reading(node)
            except ValueError:
                # An unknown alternative must not invalidate the primary reading.
                continue
            if kana not in candidates:
                candidates.append(kana)
        output.append(tuple(candidates))
    return tuple(output)


def dictionary_readings(_path, lines, *, automatic=False):
    """Use Yomi first, supplement with UniDic, and retain each candidate's origin.

    Import failures and engine errors propagate instead of silently disabling
    Yomi. UniDic may not read English/numbers that Yomi can pronounce.
    """
    from soramimic_yomi import get_yomi_candidates

    output = []
    for line in lines:
        if _RUBY.search(line.text):
            parts = []
            cursor = 0
            for match in [*_RUBY.finditer(line.text), None]:
                end = match.start() if match is not None else len(line.text)
                plain = line.text[cursor:end]
                if any(not c.isspace() and unicodedata.category(c)[0] not in "PZC"
                       for c in plain):
                    parts.append(dictionary_readings(_path, (replace(line, text=plain),),
                                                     automatic=automatic)[0].candidates)
                if match is not None:
                    try:
                        kana = normalize_reading(match[2])
                    except ValueError as exc:
                        raise ValueError(f"Explicit ruby must contain a kana pronunciation: {exc}") from exc
                    parts.append((kana,))
                    cursor = match.end()
            # Bound the combinatorial generator by order, never by mora count.
            candidates = tuple(dict.fromkeys("".join(row) for row in islice(product(*parts), 32)))
            output.append(ReadingSelection(candidates[0], "explicit-ruby", 1.0, candidates, {
                "reason": "supplied-ruby", "confidence_available": False,
            }))
            continue
        provenance = {}
        # Yomi initializes a process-wide OpenJTalk user dictionary lazily.
        with _YOMI_LOCK:
            yomi_candidates = get_yomi_candidates(line.text, nbest=32)
        for candidate in yomi_candidates:
            try:
                kana = normalize_reading(candidate.reading)
            except ValueError:
                continue
            entry = provenance.setdefault(kana, {
                "kana": kana, "sources": ["soramimic-yomi"], "yomi_candidates": [],
            })
            entry["yomi_candidates"].append(candidate.to_dict())
        yomi_status = "ok" if provenance else "no-pronunciation"
        try:
            alternatives = dictionary_candidates((line,))[0]
            unidic_status = "ok"
        except ValueError:
            alternatives = ()
            unidic_status = "no-pronunciation"
        for kana in alternatives:
            entry = provenance.setdefault(kana, {"kana": kana, "sources": []})
            entry["sources"].append("unidic-lite")
        if not provenance:
            raise ValueError("Neither soramimic-yomi nor UniDic could pronounce the lyric line")
        candidates = tuple(provenance)
        if automatic:
            selected = _automatic_candidates(
                line.text,
                tuple((candidate, provenance[candidate]["sources"][0])
                      for candidate in candidates),
            )
            candidates = tuple(candidate for candidate, _source in selected)
            for candidate, candidate_source in selected:
                provenance.setdefault(candidate, {"kana": candidate,
                                                  "sources": [candidate_source]})
        source = provenance[candidates[0]]["sources"][0]
        output.append(ReadingSelection(candidates[0], source, 1.0, candidates, {
            "reason": "dictionary", "confidence_available": False,
            "candidate_provenance": [provenance[candidate] for candidate in candidates],
            "yomi_status": yomi_status, "unidic_status": unidic_status,
        }))
    return tuple(output)


def _comparison_kana(text, *, evidence=False):
    """Apply Video's kana comparison normalization to a closed reading."""
    kana = "".join(re.findall(r"[ァ-ヶー]+", katakana(text))).replace("ヲ", "オ")
    small_vowels = {"ァ": "ア", "ィ": "イ", "ゥ": "ウ", "ェ": "エ",
                    "ォ": "オ", "ヮ": "ワ"}
    small_moras = {**small_vowels, "ャ": "ヤ", "ュ": "ユ", "ョ": "ヨ"}
    def char_vowel(char):
        try:
            return mora_vowel(small_moras.get(char, char))
        except ValueError:
            return None

    def open_small(value):
        output = []
        previous_vowel = None
        for char in value:
            large = small_vowels.get(char)
            if large is not None and previous_vowel in (None, char_vowel(large)):
                char = large
            if char != "ー":
                previous_vowel = char_vowel(char)
            if char != "ー" or not output or output[-1] != "ー":
                output.append(char)
        return "".join(output)

    def normalize_long(value):
        output = []
        for char in value:
            prior = char_vowel(output[-1]) if output and output[-1] != "ー" else None
            if (prior is not None and char in "アイウエオ"
                    and ((char == "ウ" and prior in {"o", "u"})
                         or (char == "イ" and prior in {"e", "i"})
                         or char_vowel(char) == prior)):
                char = "ー"
            output.append(char)
        return "".join(output)

    return (open_small(normalize_long(kana)) if evidence else
            open_small(normalize_long(open_small(kana))))


def token_reading_proposals(surface_text, default_reading, evidence=None):
    """Video-style single-token alternatives with two-mora local context."""
    from soramimic_yomi import get_tokens

    with _YOMI_LOCK:
        tokens = [(token.get("surface_form", ""),
                   katakana(token.get("pronunciation") or token.get("reading") or ""))
                  for token in get_tokens(surface_text)]
    pieces = [reading for surface, reading in tokens if surface]
    tokens = [(surface, reading) for surface, reading in tokens if surface]
    if _comparison_kana("".join(pieces)) != _comparison_kana(default_reading):
        return ()
    default_vowels = _vowels(default_reading)
    keys = (tuple("".join(_kana_distance().preprocess_func(_comparison_kana(text, evidence=True)))
                  for text in evidence) if evidence is not None else None)
    proposals = []
    seen = {_comparison_kana(default_reading)}
    for index, (surface, reading) in enumerate(tokens):
        if not re.search(r"[\u3400-\u9fff々〆ヵヶ]", surface) or not reading:
            continue
        left = _kana_distance().preprocess_func(
            _comparison_kana("".join(pieces[:index])))[-2:]
        right = _kana_distance().preprocess_func(
            _comparison_kana("".join(pieces[index + 1:])) )[:2]
        if len(left) + len(right) < 2:
            continue
        try:
            alternatives = dictionary_readings(None, (LyricLine(surface),))[0].candidates
        except ValueError:
            continue
        for alternative in alternatives:
            candidate = "".join((*pieces[:index], alternative, *pieces[index + 1:]))
            key = _comparison_kana(candidate)
            if key in seen or _vowels(candidate) == default_vowels:
                continue
            seen.add(key)
            local = "".join((*left, *_kana_distance().preprocess_func(
                _comparison_kana(alternative)), *right))
            full_key = "".join(_kana_distance().preprocess_func(key))
            if keys is None or any(local in text or full_key in text for text in keys):
                proposals.append(candidate)
    return tuple(proposals)


@lru_cache(maxsize=1)
def _kana_distance():
    from kanasim import create_kana_distance_calculator
    return create_kana_distance_calculator(symmetric=True, normalize=True)


def _substring_distance(candidate, evidence):
    """Match with Video's Kanasim costs and free evidence prefix/suffix."""
    distance = _kana_distance()
    candidate = distance.preprocess_func(re.sub("ー+", "ー", candidate))
    evidence = distance.preprocess_func(re.sub("ー+", "ー", evidence))
    previous = [0.0] * (len(evidence) + 1)
    for expected in candidate:
        delete = (distance.delete_cost_func(expected) if distance.delete_cost_func
                  else distance.delete_cost)
        current = [previous[0] + delete]
        for index, observed in enumerate(evidence, 1):
            insert = (distance.insert_cost_func(observed) if distance.insert_cost_func
                      else distance.insert_cost)
            replace = (distance.replace_cost_func(expected, observed)
                       if distance.replace_cost_func else distance.replace_cost)
            current.append(min(previous[index] + delete, current[-1] + insert,
                               previous[index - 1] + replace))
        previous = current
    return min(previous)


def select_acoustic_reading(candidates, transcripts):
    """Change the dictionary choice only when available audio views agree.

    The 1.0 score is a selection prior, not a calibrated acoustic probability.
    Alternative readings and raw distances are retained as evidence.
    """
    if not candidates:
        raise ValueError("At least one dictionary candidate is required")
    detail = {"transcripts": dict(transcripts), "confidence_available": False,
              "distance": "kanasim-weighted-substring-0.0.11", "reason": "no-acoustic-evidence"}
    evidence = {}
    for view, text in transcripts.items():
        try:
            moras = _comparison_kana(text, evidence=True)
        except ValueError:
            continue
        if moras:
            evidence[view] = moras
    distances = []
    for candidate in candidates:
        try:
            moras = _comparison_kana(candidate)
            row = {view: _substring_distance(moras, text) for view, text in evidence.items()}
        except ValueError:
            row = {view: None for view in evidence}
        distances.append(row)
    detail["distances"] = distances
    selected = 0
    if evidence and all(value is not None for value in distances[0].values()):
        totals = [sum(row.values()) if all(value is not None for value in row.values())
                  else math.inf for row in distances]
        best = min(range(len(candidates)), key=lambda index: totals[index])
        tied = sum(math.isclose(total, totals[best], abs_tol=1e-9) for total in totals) > 1
        detail["reason"] = "dictionary-supported"
        if tied:
            detail["reason"] = "ambiguous-evidence"
        elif best != 0:
            gains = [distances[0][view] - distances[best][view] for view in evidence]
            if min(gains) < -1e-9:
                detail["reason"] = "conflicting-evidence"
            elif sum(gains) >= .25 or any(value == 0. for value in distances[best].values()):
                selected = best
                detail["reason"] = "acoustic-agreement"
            else:
                detail["reason"] = "weak-evidence"
    return ReadingSelection(candidates[selected], "kana-whisper", 1.0,
                            tuple(candidates), detail)


def acoustic_windows(start, end, duration):
    """Cover a lyric interval in <=24-second windows without dropping its tail."""
    if not math.isfinite(start + end + duration) or not 0 <= start < end <= duration + .01:
        raise ValueError("Reading window must lie within the audio")
    cursor, stop = max(0.0, start - 1.5), min(duration, end + 1.5)
    result = []
    while cursor < stop:
        last = min(stop, cursor + 24.0)
        result.append((cursor, last))
        cursor = last
    return tuple(result)


def grouped_acoustic_windows(line_windows, indices, duration):
    """Share bounded KanaWhisper context across nearby lines, as Video does."""
    windows = []
    assignments = {}
    group = []

    def publish():
        if not group:
            return
        first = line_windows[group[0]][0]
        last = line_windows[group[-1]][1]
        if last - first > 21:
            for index in group:
                chunks = acoustic_windows(*line_windows[index], duration)
                assignments[index] = tuple(range(len(windows), len(windows) + len(chunks)))
                windows.extend(chunks)
            return
        window_index = len(windows)
        windows.append((max(0., first - 1.5), min(duration, last + 1.5)))
        for index in group:
            assignments[index] = (window_index,)

    for index in indices:
        start, end = line_windows[index]
        if not math.isfinite(start + end) or end <= start or start < 0:
            raise ValueError("Invalid acoustic reading window")
        if group and end - line_windows[group[0]][0] > 9:
            publish()
            group = []
        group.append(index)
    publish()
    if len(windows) > 256:
        raise ValueError("KanaWhisper context count exceeds 256")
    return tuple(windows), assignments
