"""Acoustic alternatives confined to English spelling-model guesses."""
from __future__ import annotations

from .japanese import normalize_lyric_input, normalize_reading


def inferred_candidates(selection):
    """Identify readings whose retained English pronunciation is only inferred."""
    if "mapped_inferred_spans" in selection.detail:
        return frozenset(selection.detail["inferred_candidates"])
    return frozenset(
        row["kana"] for row in selection.detail.get("candidate_provenance", ())
        if "unidic-lite" not in row.get("sources", ())
        and row.get("yomi_candidates")
        and all(candidate.get("inferred_spans") for candidate in row["yomi_candidates"])
    )


def inferred_reading_slots(text, selection):
    """Map inferred surface spans to the canonical reading without altering it."""
    if "mapped_inferred_spans" in selection.detail:
        return tuple(selection.detail["mapped_inferred_spans"])
    if selection.kana not in inferred_candidates(selection):
        return ()
    from soramimic_yomi import get_yomi
    from .readings import _YOMI_LOCK

    text = normalize_lyric_input(text)
    row = next(row for row in selection.detail["candidate_provenance"]
               if row["kana"] == selection.kana)
    spans = {(span["start"], span["end"]): span
             for candidate in row["yomi_candidates"]
             for span in candidate.get("inferred_spans", ())}
    output = []
    for (start, end), span in sorted(spans.items()):
        with _YOMI_LOCK:
            left = get_yomi(text[:start])
            right = get_yomi(text[end:])
        left = normalize_reading(left) if left else ""
        right = normalize_reading(right) if right else ""
        mapped = (selection.kana.startswith(left) and selection.kana.endswith(right)
                  and len(left) + len(right) <= len(selection.kana))
        output.append({**span, "mapped": mapped,
                       "kana_start": len(left) if mapped else None,
                       "kana_end": len(selection.kana) - len(right) if mapped else None})
    return tuple(output)


def inferred_acoustic_windows(line_windows, index, duration):
    """Keep padded recognition context inside the neighboring lyric lines."""
    from .readings import acoustic_windows

    start, end = line_windows[index]
    previous_end = line_windows[index - 1][1] if index else 0.
    next_start = line_windows[index + 1][0] if index + 1 < len(line_windows) else duration
    return tuple((max(a, previous_end), min(b, next_start))
                 for a, b in acoustic_windows(start, end, duration)
                 if max(a, previous_end) < min(b, next_start))


def inferred_reading_proposals(reading, slots, transcripts):
    """Retain a local audio alternative when available views agree on its bounds.

    Known surrounding kana never become replacement text. Unlike substring
    scoring, the complete recognized gap is retained, including repeated sounds.
    A span at a line edge uses that recognition window's corresponding edge.
    """
    from .symbol_readings import _anchor_edges, _moras, _unit_key

    proposals, evidence, edits = [], [], []
    for slot in slots:
        row = {"start": slot["start"], "end": slot["end"],
               "surface": slot["surface"], "source": "english-g2p",
               "views": {}, "reason": "unmapped"}
        evidence.append(row)
        if not slot["mapped"]:
            continue
        lo, hi = slot["kana_start"], slot["kana_end"]
        left, right = _moras(reading[:lo])[-6:], _moras(reading[hi:])[:6]
        if (left and len(left) < 2) or (right and len(right) < 2):
            row["reason"] = "insufficient-anchors"
            continue
        choices_by_view = []
        for view, text in transcripts.items():
            heard = _moras(text)
            matches = []
            if 0 < len(heard) <= 512:
                for start, a in _anchor_edges(left, heard, left=True):
                    for end, b in _anchor_edges(right, heard, left=False):
                        if start < end:
                            matches.append((a + b, "".join(heard[start:end])))
            best = min((cost for cost, _ in matches), default=None)
            choices = sorted({kana for cost, kana in matches if cost == best})
            row["views"][view] = {"anchor_cost": best, "readings": choices}
            choices_by_view.append(choices)
        if not choices_by_view or any(len(choices) != 1 for choices in choices_by_view):
            row["reason"] = "ambiguous-or-missing-audio"
            continue
        if len({_unit_key(choices[0]) for choices in choices_by_view}) != 1:
            row["reason"] = "conflicting-audio-views"
            continue
        replacement = choices_by_view[0][0]
        candidate = reading[:lo] + replacement + reading[hi:]
        row.update(reason="agreed-audio-proposal", reading=replacement,
                   candidate=candidate, kana_start=lo, kana_end=hi)
        edits.append((lo, hi, replacement))
        if candidate != reading and candidate not in proposals:
            proposals.append(candidate)
    combined, offset = [], 0
    for lo, hi, replacement in sorted(edits):
        if lo < offset:
            continue
        combined.extend((reading[offset:lo], replacement))
        offset = hi
    combined.append(reading[offset:])
    candidate = "".join(combined)
    if candidate != reading and candidate not in proposals:
        proposals.insert(0, candidate)
    return tuple(proposals), tuple(evidence)
