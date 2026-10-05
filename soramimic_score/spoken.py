"""Audible fallback for aligned automatic lyrics without a synthesis note.

The neutral MIDI value is a renderer hint, never a melody observation. All
pitched slots, observed boundaries, and canonical lyrics remain unchanged.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace

from .document import ScoreDocument
from .ir import Evidence, has_usable_timing
from .realization import SynthesisSlot
from .vocal_activity import VocalActivity


def add_spoken_fallback(
    document: ScoreDocument,
    line_windows: Mapping[str, tuple[float, float]],
    vocal_activity: Callable[[Sequence[tuple[float, float]]], Sequence[VocalActivity]],
) -> ScoreDocument:
    """Fill only unresolved, timed units with locally supported vocal activity.

    Text-conditioned token ends can be very short. A spoken slot therefore
    reaches the next measured onset in its line, bounded by the ASR line and
    existing synthesis slots. This allocation is labelled separately from
    observed timing. Missing/weak timing is not invented from line duration.
    """
    score = document.score
    unresolved = set(score.unresolved_unit_ids)
    if not unresolved:
        return document
    observations = document.observations
    units = {unit.id: unit for unit in observations.singing_units}
    moras = {mora.id: mora.text for mora in observations.moras}
    utterances = {mora: line.utterance_id for line in score.canonical
                  for mora in line.mora_ids}
    plan = score.synthesis_plan
    starts = [slot.start_sec for slot in plan]
    candidates = []
    for index, performed in enumerate(score.performed):
        unit = units[performed.singing_unit_id]
        if (unit.id not in unresolved or unit.status == "unobserved"
                or not has_usable_timing(unit.consonant_start)
                or not has_usable_timing(unit.end)):
            continue
        utterance = utterances[unit.mora_ids[0]]
        if utterance not in line_windows:
            continue
        lo, hi = line_windows[utterance]
        start = max(lo, unit.consonant_start.time_sec)
        end = hi
        if index + 1 < len(score.performed):
            following = units[score.performed[index + 1].singing_unit_id]
            if utterances[following.mora_ids[0]] == utterance:
                end = (following.consonant_start.time_sec
                       if has_usable_timing(following.consonant_start) else unit.end.time_sec)
        end = min(end, hi)
        position = bisect_right(starts, start)
        if position and plan[position - 1].end_sec > start:
            # Never move an aligned syllable to the other side of a sung note.
            continue
        if position < len(plan):
            end = min(end, plan[position].start_sec)
        if end > start:
            candidates.append((performed, utterance, start, end))
    if not candidates:
        return document
    observed_windows = tuple((lo, min(hi, units[p.singing_unit_id].end.time_sec))
                             for p, _, lo, hi in candidates)
    extended_windows = tuple((lo, hi) for _, _, lo, hi in candidates)
    activity = tuple(vocal_activity(observed_windows + extended_windows))
    count = len(candidates)
    if len(activity) != 2 * count:
        raise ValueError("spoken fallback needs one vocal activity result per interval")
    added = []
    evidence = []
    for (performed, utterance, start, end), support, extension in zip(
            candidates, activity[:count], activity[count:], strict=True):
        if not support.supported:
            continue
        unit_id = performed.singing_unit_id
        # A short real voice must survive a long following rest. Extend token
        # peaks only through intervals with a majority of active vocal frames.
        if not extension.supported or extension.active_frame_ratio < .5:
            end = min(end, units[unit_id].end.time_sec)
        if end <= start:
            continue
        evidence_id = f"spoken-fallback-{unit_id}"
        evidence.append(Evidence(
            evidence_id, "soramimic_score.spoken", "spoken-synthesis-fallback", 0.0,
            {"singing_unit_id": unit_id, "start_sec": start, "end_sec": end,
             "vocal_activity": asdict(support),
             "extension_vocal_activity": asdict(extension), "render_midi_pitch": 60,
             "measured_pitch": False, "condition": "automatic-aligned-voiced-unresolved"},
        ))
        added.append(SynthesisSlot(
            f"spoken-slot-{unit_id}", utterance, unit_id, performed.mora_ids,
            f"spoken-{unit_id}", performed.link_ids,
            "".join(moras[mora] for mora in performed.mora_ids), start, end, 60,
            "spoken_neutral_pitch", "spoken_onset_interval", 0.0,
            performed.evidence_ids + (evidence_id,), ("spoken",), None,
        ))
    if not added:
        return document
    rendered = {slot.singing_unit_id for slot in added}
    combined_evidence = observations.evidence + tuple(evidence)
    return replace(
        document,
        observations=replace(observations, evidence=combined_evidence),
        score=replace(
            score, synthesis_plan=tuple(sorted((*plan, *added), key=lambda s: s.start_sec)),
            unresolved_unit_ids=tuple(u for u in score.unresolved_unit_ids if u not in rendered),
            diagnostics=score.diagnostics + (f"spoken_fallback_units:{len(added)}",),
            evidence=combined_evidence,
        ),
    )
