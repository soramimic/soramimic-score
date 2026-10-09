"""Audio proposals confined to preserved symbol intervals."""
from __future__ import annotations

from functools import lru_cache
from dataclasses import replace
from collections import Counter
import math
import re

from .japanese import _RUBY, kana_to_moras, katakana
from .mora_gap import align_mora_gap


def symbol_slots(text, reading):
    """Map Yomi's symbol intervals onto an unchanged complete pronunciation.

    Failure to map a span is retained as evidence.  It is never permission to
    rewrite the surrounding lexical reading.
    """
    try:
        import soramimic_yomi as yomi
    except ImportError:
        return ()
    from .readings import _YOMI_LOCK
    find = getattr(yomi, "get_symbol_spans", None)
    if find is None:
        return ()
    ruby = tuple((match.start(), match.end()) for match in _RUBY.finditer(text))
    output = []
    for span in find(text):
        if any(start <= span.start < end for start, end in ruby):
            continue
        with _YOMI_LOCK:
            left = katakana(yomi.get_yomi(text[:span.start]))
            right = katakana(yomi.get_yomi(text[span.end:]))
        mapped = (reading.startswith(left) and reading.endswith(right) and
                  len(left) + len(right) <= len(reading))
        output.append({**span.to_dict(), "mapped": mapped,
                       "kana_start": len(left) if mapped else None,
                       "kana_end": len(reading) - len(right) if mapped else None,
                       "left": left, "right": right})
    locations = Counter((row["kana_start"], row["kana_end"]) for row in output if row["mapped"])
    for row in output:
        if row["mapped"] and locations[row["kana_start"], row["kana_end"]] > 1:
            row["mapped"] = False
    return tuple(output)


def _moras(text):
    kana = "".join(re.findall(r"[ァ-ヶー]+", katakana(text)))
    return tuple(kana_to_moras(kana))


@lru_cache(maxsize=4096)
def _unit_key(mora):
    # Keep original mora boundaries for extracting the proposed reading.
    from .readings import _comparison_kana
    return _comparison_kana(mora)


def _edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[-1] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def _anchor_edges(anchor, evidence, *, left):
    """Return the best bounded anchor matches; ambiguous matches stay explicit."""
    if not anchor:
        return [(0 if left else len(evidence), 0)]
    anchor = tuple(map(_unit_key, anchor))
    observed = tuple(map(_unit_key, evidence))
    matches = []
    tolerance = 1 if len(anchor) >= 4 else 0
    for start in range(len(observed)):
        for length in range(max(1, len(anchor) - tolerance), len(anchor) + tolerance + 1):
            end = start + length
            if end > len(observed):
                continue
            cost = _edit_distance(anchor, observed[start:end])
            if cost <= tolerance:
                matches.append((end if left else start, cost))
    return list(dict.fromkeys(matches))


def symbol_reading_proposals(reading, slots, transcripts):
    """Propose only the kana between independently recognized local anchors.

    Both audio views must locate the same bounded gap without an equally good
    competing interpretation.  Conventional readings are supplied by Yomi;
    this function contains no song-specific or symbol-name lookup.
    """
    proposals, evidence = [], []
    edits = []
    for slot in slots:
        row = {"start": slot["start"], "end": slot["end"],
               "surface": slot["surface"], "views": {}, "reason": "unmapped"}
        evidence.append(row)
        if not slot["mapped"]:
            continue
        if len(transcripts) < 2:
            row["reason"] = "insufficient-audio-views"
            continue
        lo, hi = slot["kana_start"], slot["kana_end"]
        left, right = _moras(reading[:lo])[-6:], _moras(reading[hi:])[:6]
        if len(left) < 2 or len(right) < 2:
            row["reason"] = "insufficient-anchors"
            continue
        view_choices = []
        for view, text in transcripts.items():
            observed = _moras(text)
            matches = []
            if len(observed) > 512:
                row["views"][view] = {"anchor_cost": None, "readings": []}
                view_choices.append([])
                continue
            for start, left_cost in _anchor_edges(left, observed, left=True):
                for end, right_cost in _anchor_edges(right, observed, left=False):
                    if not 0 <= end - start <= 12:
                        continue
                    replacement = "".join(observed[start:end])
                    matches.append((left_cost + right_cost, replacement, start, end))
            if matches:
                best = min(cost for cost, *_ in matches)
                choices = sorted({replacement for cost, replacement, *_ in matches if cost == best})
            else:
                best, choices = None, []
            row["views"][view] = {"anchor_cost": best, "readings": choices}
            view_choices.append(choices)
        if not view_choices or any(len(choices) != 1 for choices in view_choices):
            row["reason"] = "ambiguous-or-missing-anchors"
            continue
        keys = {_unit_key(choices[0]) for choices in view_choices}
        if len(keys) != 1:
            row["reason"] = "conflicting-audio-views"
            continue
        replacement = view_choices[0][0]
        candidate = reading[:lo] + replacement + reading[hi:]
        row.update(reason="anchored-audio-proposal", reading=replacement,
                   candidate=candidate, kana_start=lo, kana_end=hi)
        edits.append((lo, hi, replacement))
        if candidate and candidate != reading and candidate not in proposals:
            proposals.append(candidate)
    # A line may have several disjoint symbol spans.  Retain the jointly
    # supported realization as well as each local proposal.
    combined, offset = [], 0
    for lo, hi, replacement in sorted(edits):
        if lo < offset:
            continue
        combined.extend((reading[offset:lo], replacement))
        offset = hi
    combined.append(reading[offset:])
    candidate = "".join(combined)
    if candidate and candidate != reading and candidate not in proposals:
        proposals.insert(0, candidate)
    return tuple(proposals), tuple(evidence)


@lru_cache(maxsize=8192)
def _phonetic_distance(left, right):
    from .readings import _comparison_kana, _kana_distance
    try:
        return _kana_distance().calculate(_comparison_kana(left), _comparison_kana(right))
    except ValueError:
        # A sliced context can begin inside a long vowel, without its nucleus.
        return math.inf


def _phonetic_anchor_edges(anchor, observed, *, left):
    """Match bounded context while tolerating similar sung consonants/vowels."""
    target = "".join(anchor)
    matches = []
    for start in range(len(observed)):
        for length in range(max(2, len(anchor) - 1), len(anchor) + 2):
            end = start + length
            if end > len(observed):
                continue
            heard = "".join(observed[start:end])
            cost = _phonetic_distance(target, heard)
            scale = _phonetic_distance(target, "") + _phonetic_distance("", heard)
            if math.isfinite(cost + scale) and scale and cost <= .3 * scale:
                matches.append((end if left else start, cost))
    return matches


def _gap_readings(left, right, text):
    """Locate an interval before comparing its possible pronunciations.

    Short context is only a fallback when a longer anchor cannot be located.
    At least nine surrounding moras remain; candidate names never influence
    where the anchors are placed.
    """
    observed = _moras(text)
    if not observed or len(observed) > 512:
        return {"readings": [], "reason": "missing-or-excessive-transcript"}
    aligned = align_mora_gap(left, right, text)
    if aligned["readings"]:
        return aligned
    left, right = _moras(left), _moras(right)
    for width_left, width_right, phonetic in ((6, 6, False), (6, 6, True),
                                             (6, 3, True), (3, 6, True)):
        before, after = left[-width_left:], right[:width_right]
        if min(len(before), len(after)) < 2 or len(before) + len(after) < 9:
            continue
        matcher = _phonetic_anchor_edges if phonetic else _anchor_edges
        matches = []
        for start, a in matcher(before, observed, left=True):
            for end, b in matcher(after, observed, left=False):
                if 0 <= end - start <= 12:
                    matches.append((a + b, start, end))
        if not matches:
            continue
        best = min(cost for cost, _, _ in matches)
        spans = [(start, end) for cost, start, end in matches
                 if math.isclose(cost, best, abs_tol=1e-9)]
        readings = sorted({"".join(observed[start:end]) for start, end in spans})
        return {"readings": readings, "spans": spans, "anchor_cost": best,
                "anchor_moras": [len(before), len(after)],
                "reason": "phonetic-anchors" if phonetic else "exact-mora-anchors"}
    return {"readings": [], "reason": "missing-anchors"}


def _lexical_transcript(text):
    """Retain recognized token boundaries, without giving lyrics to Whisper."""
    from soramimic_yomi import get_tokens, get_yomi
    from .readings import _YOMI_LOCK
    with _YOMI_LOCK:
        kana = katakana(get_yomi(text))
        tokens = get_tokens(text)
    pieces, spans, cursor = [], [], 0
    for token in tokens:
        pronunciation = katakana(token.get("pronunciation") or token.get("reading") or "")
        moras = _moras(pronunciation)
        if not moras:
            continue
        pieces.extend(moras)
        spans.append((cursor, cursor + len(moras), token))
        cursor += len(moras)
    # English/numeric normalization can change token boundaries. Closed
    # dictionary comparisons still work; novel token proposals must abstain.
    return kana, spans if _moras(kana) == tuple(pieces) else []


def _content_word_gap(gap, token_spans):
    if len(gap["readings"]) != 1 or not gap["readings"][0]:
        return False
    for start, end in gap.get("spans", ()):
        inside = [(a, b, token) for a, b, token in token_spans if a >= start and b <= end]
        if not inside or inside[0][0] != start or inside[-1][1] != end:
            continue
        if all(token.get("pos") in {"名詞", "動詞", "形容詞", "副詞", "感動詞"}
               and token.get("pos_detail_1") not in {"非自立", "接尾", "代名詞"}
               for _, _, token in inside):
            return True
    return False


def refine_symbol_reading(text, selection, kana_views, whisper_views, *,
                          previous_reading="", next_reading=""):
    """Combine localized phonetic evidence with audio-only lexical recognition.

    Conventional names need corroborating views or models. Novel readings
    come from aligned lexical tokens and require phonetic-model support.
    Each observed spelling remains a separate candidate; no result can
    replace surrounding supplied words.
    """
    slots = symbol_slots(text, selection.kana)
    if not slots:
        return selection
    lexical = {view: _lexical_transcript(value) for view, value in whisper_views.items()}
    rows, edits = [], []
    for slot in slots:
        row = {"surface": slot["surface"], "start": slot["start"], "end": slot["end"],
               "reason": "unmapped", "views": {}}
        rows.append(row)
        if not slot["mapped"]:
            continue
        lo, hi = slot["kana_start"], slot["kana_end"]
        current = selection.kana[lo:hi]
        before, after = selection.kana[:lo], selection.kana[hi:]
        if len(_moras(before)) < 6:
            before = "".join(_moras(previous_reading)[-6:]) + before
        if len(_moras(after)) < 6:
            after += "".join(_moras(next_reading)[:6])
        kana_gaps = {view: _gap_readings(before, after, value)
                     for view, value in kana_views.items()}
        word_gaps = {view: _gap_readings(before, after, value[0])
                     for view, value in lexical.items()}
        row["views"] = {"kana_whisper": kana_gaps, "whisper": word_gaps}
        names = list(dict.fromkeys((current, *slot["readings"])))
        novel_support = {}
        for view, gap in word_gaps.items():
            if _content_word_gap(gap, lexical[view][1]):
                novel_support.setdefault(gap["readings"][0], []).append(view)
        novel = list(novel_support)
        names.extend(name for name in novel if name not in names)
        # Preserve each recognized spelling as its own candidate. Compare
        # pronunciation continuously; do not merge names by vowel identity.
        groups = {}
        for family, gaps in (("kana_whisper", kana_gaps), ("whisper", word_gaps)):
            for view, gap in gaps.items():
                if len(gap["readings"]) != 1:
                    continue
                group = (family, view.split(":", 1)[0])
                quality = (0 if gap["reason"] == "whole-context-mora-alignment" else 1,
                           gap.get("alignment_cost", gap.get("anchor_cost", 0.)) /
                           max(1, sum(gap.get("context_moras", gap.get("anchor_moras", (6, 6))))))
                if group not in groups or quality < groups[group][0]:
                    groups[group] = (quality, view, gap)
        scores = []
        for name in names:
            losses, support = {}, []
            for (family, view), (_, context, gap) in groups.items():
                heard = gap["readings"][0]
                distance = _phonetic_distance(name, heard)
                scale = _phonetic_distance(name, "") + _phonetic_distance("", heard)
                loss = (distance / scale if scale else 0.) if math.isfinite(distance + scale) else 1.
                losses.setdefault(family, []).append(loss)
                if loss <= .5:
                    support.append((family, view))
                gap.setdefault("candidate_losses", {})[name] = loss
            corroborated = (len({family for family, _ in support}) >= 2
                            or len({view for _, view in support}) >= 2)
            novel_name = name not in slot["readings"] and name != current
            phonetic = losses.get("kana_whisper", [])
            usable = (bool(losses) and corroborated and
                      (not novel_name or (name in novel_support and phonetic
                                          and max(phonetic) <= .5
                                          and "whisper" in losses)))
            # Whisper proposes lexical readings. Phonetic recognition ranks
            # them, so a word omitted by Whisper is not treated as silence.
            # Repeated contexts of one view never get additional weight.
            ranking = phonetic or losses.get("whisper", [])
            loss = sum(ranking) / len(ranking) if ranking else 1.
            scores.append({"reading": name, "loss": loss, "supported": bool(usable),
                           "models": losses, "support": support})
        row.update(candidates=names, novel_proposals=novel, candidate_scores=scores,
                   contexts_used={family + ":" + view: context
                                  for (family, view), (_, context, _) in groups.items()})
        supported = [item for item in scores if item["supported"] and item["loss"] < .5]
        if not supported:
            row["reason"] = "insufficient-aligned-audio-evidence"
            continue
        best = min(item["loss"] for item in supported)
        chosen = next(item["reading"] for item in supported
                      if math.isclose(item["loss"], best, abs_tol=1e-9))
        row.update(reason="aligned-audio-selection", selected=chosen)
        if chosen != current:
            edits.append((lo, hi, chosen))
    kana = selection.kana
    for lo, hi, replacement in sorted(edits, reverse=True):
        kana = kana[:lo] + replacement + kana[hi:]
    detail = {**selection.detail, "symbol_refinement": rows,
              "whisper_transcripts": dict(whisper_views),
              "whisper_kana": {view: value[0] for view, value in lexical.items()}}
    if kana and kana != selection.kana:
        detail.update(reason="localized-symbol-agreement", prior_reading=selection.kana)
        return replace(selection, kana=kana, source="whisper+kana-whisper",
                       candidates=tuple(dict.fromkeys((*selection.candidates, kana))), detail=detail)
    return replace(selection, detail=detail)
