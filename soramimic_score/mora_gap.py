"""Align supplied pronunciation around a symbol to recognized mora sequences."""
from __future__ import annotations

import math

from .japanese import kana_to_moras, katakana, phonemes_for_mora


def mora_units(reading):
    """Keep a sound and a duration-bearing vowel for every retained mora."""
    units, previous = [], None
    for mora in kana_to_moras(katakana(reading)):
        sounds = phonemes_for_mora(mora, previous)
        vowel = sounds[-1]
        previous = vowel if vowel in "aiueo" else None
        units.append((mora, sounds, vowel))
    return tuple(units)


def _cost(left, right):
    if left[1] == right[1]:
        return 0.
    return .25 if left[2] == right[2] else 1.


def _prefix_alignment(target, observed):
    """Align the complete target, with free leading observed context."""
    rows = [[0.] * (len(observed) + 1)]
    steps = [[None] * (len(observed) + 1)]
    for i, expected in enumerate(target, 1):
        current, operations = [float(i)], ["delete"]
        for j, heard in enumerate(observed, 1):
            alternatives = ((rows[-1][j - 1] + _cost(expected, heard), "match"),
                            (rows[-1][j] + 1., "delete"),
                            (current[-1] + 1., "insert"))
            value, operation = min(alternatives, key=lambda item: item[0])
            current.append(value)
            operations.append(operation)
        rows.append(current)
        steps.append(operations)
    return rows[-1], steps


def _pairs(steps, end):
    i, j = len(steps) - 1, end
    pairs = []
    while i:
        operation = steps[i][j]
        if operation == "match":
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif operation == "insert":
            j -= 1
        else:
            i -= 1
    return list(reversed(pairs))


def _supported_edge(target, observed, pairs):
    width = min(6, len(target))
    edge = [(i, j) for i, j in pairs if i >= len(target) - width]
    matched = sum(_cost(target[i], observed[j]) <= .25 for i, j in edge)
    return matched >= max(2, math.ceil(width * .6))


def align_mora_gap(left, right, recognized):
    """Find an unnamed 0–12-mora interval using the whole surrounding sequence.

    Both supplied sides are consumed in order. Leading/trailing recognition
    context is free, while missing or different supplied moras have a cost.
    Candidate symbol names never affect alignment. The returned paths are
    mora correspondences, not inferred timestamps.
    """
    before, after, observed = mora_units(left), mora_units(right), mora_units(recognized)
    if (min(len(before), len(after)) < 2 or len(before) + len(after) < 9
            or max(len(before) + len(after), len(observed)) > 512 or not observed):
        return {"readings": [], "reason": "insufficient-mora-context"}
    forward, forward_steps = _prefix_alignment(before, observed)
    reverse, reverse_steps = _prefix_alignment(after[::-1], observed[::-1])
    starts, ends = {}, {}
    for start in range(len(observed) + 1):
        pairs = _pairs(forward_steps, start)
        if _supported_edge(before, observed, pairs):
            starts[start] = pairs
    for end in range(len(observed) + 1):
        pairs = _pairs(reverse_steps, len(observed) - end)
        if _supported_edge(after[::-1], observed[::-1], pairs):
            ends[end] = [(len(after) - 1 - i, len(observed) - 1 - j)
                         for i, j in reversed(pairs)]
    candidates = []
    for start in starts:
        for end in range(start, min(len(observed), start + 12) + 1):
            if end not in ends:
                continue
            mismatch = forward[start] + reverse[len(observed) - end]
            if mismatch <= .35 * (len(before) + len(after)):
                candidates.append((mismatch + .02 * (end - start), start, end))
    if not candidates:
        return {"readings": [], "reason": "no-supported-mora-alignment"}
    best = min(cost for cost, _, _ in candidates)
    spans = [(start, end) for cost, start, end in candidates
             if math.isclose(cost, best, abs_tol=1e-9)]
    return {
        "readings": sorted({"".join(unit[0] for unit in observed[start:end])
                            for start, end in spans}),
        "spans": spans, "alignment_cost": best,
        "reason": "whole-context-mora-alignment",
        "alignments": [{"left": starts[start], "right": ends[end]} for start, end in spans],
        "context_moras": [len(before), len(after)],
    }
