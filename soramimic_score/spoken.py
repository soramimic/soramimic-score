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
    *, fill_unpitched_lines: bool = False,
) -> ScoreDocument:
    """Fill only unresolved, timed units with locally supported vocal activity.

    Text-conditioned token ends can be very short. A spoken slot therefore
    reaches the next measured onset in its line, bounded by the ASR line and
    existing synthesis slots. This allocation is labelled separately from
    observed timing. Entirely unpitched automatic lines can instead use their
    recognized line window, with proportional render timing explicitly labelled.
    """
    if fill_unpitched_lines:
        document = _fill_unpitched_lines(document, line_windows, vocal_activity)
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


def _fill_unpitched_lines(document, line_windows, vocal_activity):
    """Speak a recognized, voiced line even when CTC has no usable anchors.

    This is an ASR-timed reading, not recovered melody or measured mora timing.
    Only wholly unresolved lines with no overlapping sung slots qualify.
    """
    score = document.score
    unresolved = set(score.unresolved_unit_ids)
    unit_by_mora = {m: unit for unit in score.performed for m in unit.mora_ids}
    moras = {m.id: m.text for m in document.observations.moras}
    candidates = []
    for line in score.canonical:
        window = line_windows.get(line.utterance_id)
        units = list({unit_by_mora[m].singing_unit_id: unit_by_mora[m]
                      for m in line.mora_ids}.values())
        if (window is None or not units
                or any(unit.singing_unit_id not in unresolved for unit in units)):
            continue
        start, end = window
        if end <= start or any(slot.start_sec < end and slot.end_sec > start
                               for slot in score.synthesis_plan):
            continue
        candidates.append((line, units, start, end))
    if not candidates:
        return document
    activity = tuple(vocal_activity(tuple((lo, hi) for _, _, lo, hi in candidates)))
    if len(activity) != len(candidates):
        raise ValueError("spoken fallback needs one vocal activity result per interval")
    slots = []
    evidence = []
    for (line, units, start, end), support in zip(candidates, activity, strict=True):
        if not support.supported:
            continue
        evidence_id = f"spoken-line-{line.utterance_id}"
        evidence.append(Evidence(
            evidence_id, "soramimic_score.spoken", "spoken-synthesis-fallback", 0.,
            {"utterance_id": line.utterance_id, "start_sec": start, "end_sec": end,
             "vocal_activity": asdict(support), "render_midi_pitch": 60,
             "measured_pitch": False, "measured_mora_timing": False,
             "condition": "automatic-recognized-voiced-unpitched-line"},
        ))
        total = sum(len(unit.mora_ids) for unit in units)
        cursor = 0
        for unit in units:
            lo = start + (end - start) * cursor / total
            cursor += len(unit.mora_ids)
            hi = start + (end - start) * cursor / total
            uid = unit.singing_unit_id
            slots.append(SynthesisSlot(
                f"spoken-slot-{uid}", line.utterance_id, uid, unit.mora_ids,
                f"spoken-{uid}", unit.link_ids,
                "".join(moras[m] for m in unit.mora_ids), lo, hi, 60,
                "spoken_neutral_pitch", "spoken_line_proportional", 0.,
                unit.evidence_ids + (evidence_id,), ("spoken",), None,
            ))
    if not slots:
        return document
    rendered = {slot.singing_unit_id for slot in slots}
    combined_evidence = document.observations.evidence + tuple(evidence)
    return replace(document,
        observations=replace(document.observations, evidence=combined_evidence),
        score=replace(score,
            synthesis_plan=tuple(sorted((*score.synthesis_plan, *slots), key=lambda s: s.start_sec)),
            unresolved_unit_ids=tuple(u for u in score.unresolved_unit_ids if u not in rendered),
            evidence=combined_evidence,
            diagnostics=score.diagnostics + (f"spoken_fallback_line_units:{len(slots)}",),
        ),
    )
