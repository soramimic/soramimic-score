"""Compile correspondence into a reversible, evidence-preserving vocal plan.

This is synthesis allocation, not another acoustic estimator. A missing match
never proves an omitted performance. Repairs and interpolated boundaries are
labelled explicitly and do not modify the input representation.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass, replace
import json
import math
from typing import Any, Sequence

from .ir import (Evidence, IntermediateRepresentation, NoteCandidate, SingingUnit,
                 has_usable_timing)
from .japanese import kana_to_syllables, validate_reading


@dataclass(frozen=True)
class PerformanceOmission:
    singing_unit_id: str
    reason: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class CanonicalLine:
    utterance_id: str
    text: str
    kana: str
    mora_ids: tuple[str, ...]


@dataclass(frozen=True)
class PerformedUnit:
    singing_unit_id: str
    mora_ids: tuple[str, ...]
    status: str
    confidence: float
    start_sec: float | None
    end_sec: float | None
    link_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class SynthesisSlot:
    id: str
    utterance_id: str
    singing_unit_id: str
    mora_ids: tuple[str, ...]
    note_candidate_id: str
    link_ids: tuple[str, ...]
    kana: str
    start_sec: float
    end_sec: float
    midi_pitch: int
    operation: str
    timing_source: str
    confidence: float
    evidence_ids: tuple[str, ...]
    pitch_sources: tuple[str, ...]
    pitch_confidence: float | None
    continuation: bool = False


@dataclass(frozen=True)
class Realization:
    schema_version: int
    canonical_text: str
    canonical: tuple[CanonicalLine, ...]
    performed: tuple[PerformedUnit, ...]
    synthesis_plan: tuple[SynthesisSlot, ...]
    omissions: tuple[PerformanceOmission, ...]
    unresolved_unit_ids: tuple[str, ...]
    diagnostics: tuple[str, ...]
    evidence: tuple[Evidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, allow_nan=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Realization:
        def record(kind: type, value: dict[str, Any], tuples: tuple[str, ...]) -> Any:
            return kind(**{key: tuple(item) if key in tuples else item
                           for key, item in value.items()})

        result = cls(
            schema_version=data["schema_version"], canonical_text=data["canonical_text"],
            canonical=tuple(record(CanonicalLine, x, ("mora_ids",)) for x in data["canonical"]),
            performed=tuple(record(PerformedUnit, x, ("mora_ids", "link_ids", "evidence_ids"))
                            for x in data["performed"]),
            synthesis_plan=tuple(record(SynthesisSlot, x, (
                "mora_ids", "link_ids", "evidence_ids", "pitch_sources"))
                                 for x in data["synthesis_plan"]),
            omissions=tuple(record(PerformanceOmission, x, ("evidence_ids",))
                            for x in data["omissions"]),
            unresolved_unit_ids=tuple(data["unresolved_unit_ids"]),
            diagnostics=tuple(data["diagnostics"]),
            evidence=tuple(Evidence(**x) for x in data["evidence"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported realization version")
        known_units = {x.singing_unit_id for x in self.performed}
        if len(known_units) != len(self.performed):
            raise ValueError("duplicate performed unit")
        seen: set[str] = set()
        known_lines = {x.utterance_id for x in self.canonical}
        for line in self.canonical:
            reading = validate_reading(line.kana)
            if reading.kana != line.kana or len(reading.moras) != len(line.mora_ids):
                raise ValueError("canonical kana must contain one normalized mora per mora ID")
        known_moras = {m for x in self.canonical for m in x.mora_ids}
        canonical_moras = [m for x in self.canonical for m in x.mora_ids]
        performed_moras = [m for x in self.performed for m in x.mora_ids]
        if (len(canonical_moras) != len(known_moras)
                or sorted(canonical_moras) != sorted(performed_moras)):
            raise ValueError("every canonical mora needs exactly one performed-unit membership")
        by_unit = {x.singing_unit_id: x for x in self.performed}
        known_evidence = {x.id for x in self.evidence}
        if len(known_evidence) != len(self.evidence):
            raise ValueError("duplicate evidence")
        for unit in self.performed:
            if (not unit.mora_ids or set(unit.mora_ids) - known_moras
                    or set(unit.evidence_ids) - known_evidence
                    or unit.status not in {"observed", "weak", "unobserved"}):
                raise ValueError("invalid performed unit")
        previous_end = 0.0
        mora_order = {mid: i for i, mid in enumerate(canonical_moras)}
        previous_unit = None
        previous_mora = -1
        for slot in self.synthesis_plan:
            if validate_reading(slot.kana).kana != slot.kana:
                raise ValueError("synthesis kana must be normalized")
            if (slot.id in seen or slot.singing_unit_id not in known_units
                    or slot.utterance_id not in known_lines
                    or set(slot.mora_ids) - known_moras
                    or slot.mora_ids != by_unit[slot.singing_unit_id].mora_ids
                    or set(slot.evidence_ids) - known_evidence or not slot.pitch_sources):
                raise ValueError("invalid synthesis slot ownership")
            positions = [mora_order[mid] for mid in slot.mora_ids]
            if slot.singing_unit_id != previous_unit and min(positions) <= previous_mora:
                raise ValueError("synthesis slots must follow canonical lyric order")
            previous_unit = slot.singing_unit_id
            previous_mora = max(positions)
            seen.add(slot.id)
            if (not slot.kana or not math.isfinite(slot.start_sec + slot.end_sec)
                    or slot.start_sec < previous_end - 1e-9
                    or slot.end_sec <= slot.start_sec
                    or type(slot.midi_pitch) is not int or not 0 <= slot.midi_pitch <= 127
                    or not math.isfinite(slot.confidence) or not 0 <= slot.confidence <= 1):
                raise ValueError("invalid synthesis slot timing, pitch, or confidence")
            if slot.pitch_confidence is not None and (
                    not math.isfinite(slot.pitch_confidence) or not 0 <= slot.pitch_confidence <= 1):
                raise ValueError("invalid pitch confidence")
            previous_end = slot.end_sec
        omitted = {x.singing_unit_id for x in self.omissions}
        by_evidence = {x.id: x for x in self.evidence}
        for omission in self.omissions:
            if (not omission.reason.strip() or not omission.evidence_ids
                    or any(eid not in by_evidence or by_evidence[eid].kind != "performance-omission"
                           or by_evidence[eid].confidence <= 0 for eid in omission.evidence_ids)):
                raise ValueError("omission requires explicit performance-omission evidence")
        unresolved = set(self.unresolved_unit_ids)
        rendered = {x.singing_unit_id for x in self.synthesis_plan}
        if (omitted & rendered or unresolved & rendered or omitted & unresolved
                or omitted | unresolved | rendered != known_units):
            raise ValueError("every canonical unit must be rendered, explicitly omitted, or unresolved")
        by_note: dict[str, list[SynthesisSlot]] = {}
        for slot in self.synthesis_plan:
            by_note.setdefault(slot.note_candidate_id, []).append(slot)
        for note_id, slots in by_note.items():
            kana = "".join(slot.kana for slot in slots if not slot.continuation)
            if kana and len(kana_to_syllables(kana)) > 1:
                raise ValueError(f"note {note_id} is assigned more than one syllable")


def split_stack_interval(
    start_sec: float, end_sec: float, units: Sequence[SingingUnit],
) -> tuple[tuple[float, float, str], ...]:
    """Use ordered measured onsets; fill only unanchored runs proportionally.

    No minimum-duration rule deletes a unit or lengthens the enclosing note.
    The timing label describes the interval, not an acoustic confidence claim.
    """
    if not units or not math.isfinite(start_sec + end_sec) or not 0 <= start_sec < end_sec:
        raise ValueError("a split needs units and a positive finite interval")
    count = len(units)
    anchors = {0: start_sec, count: end_sec}
    last = start_sec
    for index, unit in enumerate(units[1:], 1):
        boundary = unit.consonant_start
        if (has_usable_timing(boundary)
                and last < boundary.time_sec < end_sec):
            anchors[index] = boundary.time_sec
            last = boundary.time_sec
    positions = [0.0] * (count + 1)
    for left, right in zip(sorted(anchors), sorted(anchors)[1:]):
        weights = [max(1, len(unit.mora_ids)) for unit in units[left:right]]
        total = sum(weights)
        accumulated = 0
        for index, weight in zip(range(left, right), weights, strict=True):
            positions[index] = anchors[left] + (anchors[right] - anchors[left]) * accumulated / total
            accumulated += weight
        positions[right] = anchors[right]
    return tuple((positions[i], positions[i + 1],
                  "aligned_boundary" if i in anchors and i + 1 in anchors and count > 1
                  else "note_interval" if count == 1 else "proportional_fallback")
                 for i in range(count))


def _extend_observed_utterance_tails(
    plan: Sequence[SynthesisSlot], units: Sequence[SingingUnit],
    utterance: dict[str, str],
) -> tuple[tuple[SynthesisSlot, ...], int]:
    """Use a measured phrase-final release time in otherwise empty space.

    A melody candidate can end before the source-aligned final vowel. Keeping
    that boundary makes a short final mora sound omitted even though its
    observed performance extends into the following rest. Extend only the
    last rendered slot of the last unit in each utterance, and never cross the
    next synthesis slot.
    """
    if not plan:
        return tuple(plan), 0
    by_id = {unit.id: unit for unit in units}
    final_unit = {utterance[unit.id]: unit.id for unit in units}
    last_slot = {slot.singing_unit_id: i for i, slot in enumerate(plan)}
    result = list(plan)
    changed = 0
    for i, slot in enumerate(plan):
        unit_id = final_unit.get(slot.utterance_id)
        if unit_id != slot.singing_unit_id or last_slot.get(unit_id) != i:
            continue
        boundary = by_id[unit_id].end
        if not has_usable_timing(boundary) or boundary.time_sec <= slot.end_sec:
            continue
        limit = plan[i + 1].start_sec if i + 1 < len(plan) else boundary.time_sec
        end = min(boundary.time_sec, limit)
        if end > slot.end_sec + 1e-9:
            result[i] = replace(slot, end_sec=end, timing_source="observed_utterance_tail")
            changed += 1
    return tuple(result), changed


def _repair_unmatched(
    units: Sequence[SingingUnit], notes: dict[str, NoteCandidate],
    utterance: dict[str, str], allocation: dict[str, list[str]], omitted: set[str],
    mora_text: dict[str, str],
) -> set[str]:
    """Partition a missing run where it still fits neighboring syllable slots.

    Measured consonant starts locate synthesis repairs, not vowel/note onsets.
    The one-cut allocation preserves lyric order even when measurements disagree.
    Missing timing falls back to canonical distance; repairs never become anchors
    for another repair, promote an unlinked pitch candidate, or add a second
    syllable to a note.
    """
    order = {unit.id: i for i, unit in enumerate(units)}
    original: dict[str, list[str]] = {}
    for note_id, owners in allocation.items():
        for owner in owners:
            original.setdefault(owner, []).append(note_id)
    anchors: dict[str, list[int]] = {}
    for unit in units:
        if unit.id in original:
            anchors.setdefault(utterance[unit.id], []).append(order[unit.id])
    groups: dict[tuple[int | None, int | None], list[SingingUnit]] = {}
    for unit in units:
        if unit.id in original or unit.id in omitted:
            continue
        nearby = anchors.get(utterance[unit.id], [])
        position = bisect_left(nearby, order[unit.id])
        left = nearby[position - 1] if position else None
        right = nearby[position] if position < len(nearby) else None
        if left is not None or right is not None:
            groups.setdefault((left, right), []).append(unit)

    def endpoint(owner: int, preceding: bool) -> str:
        candidates = original[units[owner].id]
        key = lambda note_id: (notes[note_id].start_sec, notes[note_id].end_sec, note_id)
        return (max if preceding else min)(candidates, key=key)

    def distance(unit: SingingUnit, note_id: str) -> float:
        boundary = unit.consonant_start
        if unit.status == "unobserved" or not has_usable_timing(boundary):
            return 0.0
        note = notes[note_id]
        return max(note.start_sec - boundary.time_sec, boundary.time_sec - note.end_sec, 0.0)

    repaired: set[str] = set()

    def fits(note_id: str, candidate: SingingUnit) -> bool:
        owners = sorted((*allocation[note_id], candidate.id), key=order.__getitem__)
        kana = "".join(mora_text[mora_id] for owner in owners
                       for mora_id in units[order[owner]].mora_ids)
        return len(kana_to_syllables(kana)) <= 1

    for (left, right), missing in groups.items():
        left_note = endpoint(left, True) if left is not None else None
        right_note = endpoint(right, False) if right is not None else None
        if left_note is None:
            chosen = [right_note] * len(missing)
        elif right_note is None or left_note == right_note:
            chosen = [left_note] * len(missing)
        else:
            # Evaluate every prefix/suffix split with additive acoustic distance.
            # Canonical distance breaks acoustic ties and is the entire fallback
            # when none of the units has a usable measured start.
            prefix = [(0.0, 0)]
            suffix = [(0.0, 0)]
            for unit in missing:
                prefix.append((prefix[-1][0] + distance(unit, left_note),
                               prefix[-1][1] + order[unit.id] - left))
            for unit in reversed(missing):
                suffix.append((suffix[-1][0] + distance(unit, right_note),
                               suffix[-1][1] + right - order[unit.id]))
            count = len(missing)
            cut = min(range(count + 1), key=lambda i: (
                prefix[i][0] + suffix[count - i][0],
                prefix[i][1] + suffix[count - i][1], -i,
            ))
            chosen = [left_note] * cut + [right_note] * (count - cut)
        for unit, note_id in zip(missing, chosen, strict=True):
            assert note_id is not None
            if not fits(note_id, unit):
                continue
            allocation[note_id].append(unit.id)
            repaired.add(unit.id)
    return repaired


def compile_realization(
    document: IntermediateRepresentation, *,
    omissions: Sequence[PerformanceOmission] = (),
) -> Realization:
    """Compile Stage 3 links without confusing alignment misses with omissions.

    Unmatched units may subdivide a neighboring linked note in the same utterance
    only when the combined text remains one syllable. Repairs use measured timing
    without reversing lyrics and retain ``unmatched_stack_repair`` provenance.
    If no such pitched slot exists they remain unresolved (never deleted).
    """
    document.validate()
    readings = {x.id: x for x in document.readings}
    moras = {x.id: x for x in document.moras}
    notes = {x.id: x for x in document.note_candidates}
    evidence = {x.id: x for x in document.evidence}
    canonical = tuple(CanonicalLine(x.id, x.surface, readings[x.selected_reading_id].kana,
                                    readings[x.selected_reading_id].mora_ids)
                      for x in document.utterances)
    mora_order = {mora_id: i for i, mora_id in enumerate(
        mora_id for line in canonical for mora_id in line.mora_ids)}
    units = sorted(document.singing_units, key=lambda x: min(mora_order[m] for m in x.mora_ids))
    unit_by_id = {x.id: x for x in units}
    unit_order = {x.id: i for i, x in enumerate(units)}
    utterance = {x.id: readings[moras[x.mora_ids[0]].reading_id].utterance_id for x in units}
    omission_ids = {x.singing_unit_id for x in omissions}
    if len(omission_ids) != len(omissions):
        raise ValueError("duplicate omission classification")
    for omission in omissions:
        if (omission.singing_unit_id not in unit_by_id or not omission.reason.strip()
                or not omission.evidence_ids
                or any(eid not in evidence or evidence[eid].kind != "performance-omission"
                       or evidence[eid].confidence <= 0 for eid in omission.evidence_ids)):
            raise ValueError("omission requires explicit performance-omission evidence")
    links_by_unit: dict[str, list[str]] = {x.id: [] for x in units}
    allocation: dict[str, list[str]] = {}
    operations: dict[str, str] = {}
    used_notes: set[str] = set()
    for link in document.links:
        for unit_id in link.singing_unit_ids:
            if links_by_unit[unit_id]:
                raise ValueError("correspondence links must not consume a unit twice")
            links_by_unit[unit_id].append(link.id)
            operations[unit_id] = link.operation
        for note_index, note_id in enumerate(link.note_candidate_ids):
            if note_id in used_notes:
                raise ValueError("correspondence links must not consume a note twice")
            used_notes.add(note_id)
            if notes[note_id].midi_pitch is not None:
                if link.operation == "sokuon":
                    allocated = [link.singing_unit_ids[0]]
                    if note_index == len(link.note_candidate_ids) - 1:
                        allocated.append(link.singing_unit_ids[1])
                    allocated = [unit_id for unit_id in allocated
                                 if unit_id not in omission_ids]
                else:
                    allocated = [u for u in link.singing_unit_ids if u not in omission_ids]
                if allocated:
                    allocation[note_id] = allocated
    for note_id, owners in allocation.items():
        kana = "".join(moras[mora_id].text for owner in owners
                       for mora_id in unit_by_id[owner].mora_ids)
        if len(kana_to_syllables(kana)) > 1:
            raise ValueError(f"note {note_id} is assigned more than one syllable")
    performed = tuple(PerformedUnit(
        x.id, x.mora_ids, x.status, x.confidence,
        x.consonant_start.time_sec if x.consonant_start else None,
        x.end.time_sec if x.end else None, tuple(links_by_unit[x.id]), _unit_evidence_ids(x),
    ) for x in units)
    note_run_unit_only = {
        unit_id
        for link in document.links if link.operation == "unit_only"
        if any(evidence[evidence_id].kind == "note-run-unit-only"
               for evidence_id in link.evidence_ids)
        for unit_id in link.singing_unit_ids
    }
    repaired = _repair_unmatched(
        units, notes, utterance, allocation, omission_ids | note_run_unit_only,
        {mora_id: mora.text for mora_id, mora in moras.items()},
    )
    plan: list[SynthesisSlot] = []
    diagnostics: list[str] = []
    emitted: set[str] = set()
    ordered_notes = sorted(allocation, key=lambda key: (
        notes[key].start_sec, min(unit_order[u] for u in allocation[key]), key))
    effective_times = {}
    coincident = set()
    first = 0
    while first < len(ordered_notes):
        last = first + 1
        start = notes[ordered_notes[first]].start_sec
        while last < len(ordered_notes) and notes[ordered_notes[last]].start_sec == start:
            last += 1
        group = ordered_notes[first:last]
        end = max(notes[key].end_sec for key in group)
        if last < len(ordered_notes):
            end = min(end, notes[ordered_notes[last]].start_sec)
        if len(group) > 1:
            # An adapter may provide several targets at one timestamp. Quantize
            # the shared interval in canonical order, retaining every pitch/lyric.
            weighted_units = [replace(unit_by_id[allocation[key][0]],
                                      mora_ids=tuple(m for u in allocation[key]
                                                     for m in unit_by_id[u].mora_ids)) for key in group]
            spans = split_stack_interval(start, end, weighted_units)
            for key, (lo, hi, _) in zip(group, spans, strict=True):
                effective_times[key] = (lo, hi)
                coincident.add(key)
            diagnostics.append(f"coincident_notes_partitioned:{len(group)}")
        else:
            key = group[0]
            effective_times[key] = (start, end)
            if end < notes[key].end_sec:
                diagnostics.append(f"overlapping_note_clipped:{key}")
        first = last
    for index, note_id in enumerate(ordered_notes):
        note = notes[note_id]
        start, end = effective_times[note_id]
        owners = sorted(allocation[note_id], key=unit_order.__getitem__)
        if end <= start:
            diagnostics.append(f"unrenderable_note:{note_id}")
            continue
        spans = split_stack_interval(start, end, [unit_by_id[u] for u in owners])
        for unit_id, (lo, hi, timing) in zip(owners, spans, strict=True):
            unit = unit_by_id[unit_id]
            continuation = unit_id in emitted
            kana = "".join(moras[mid].text for mid in unit.mora_ids)
            if continuation:
                kana = "ー"
            operation = ("unmatched_stack_repair" if unit_id in repaired
                         else "coincident_note_split" if note_id in coincident
                         else "sokuon_split" if len(owners) > 1
                         and any(operations[owner] == "sokuon" for owner in owners)
                         else "stack_split" if len(owners) > 1 else operations[unit_id])
            plan.append(SynthesisSlot(
                f"slot-{len(plan)}", utterance[unit_id], unit_id, unit.mora_ids, note_id,
                tuple(links_by_unit[unit_id]), kana, lo, hi, int(note.midi_pitch),
                operation, "proportional_fallback" if note_id in coincident else timing,
                min(unit.confidence, note.confidence),
                tuple(dict.fromkeys(_unit_evidence_ids(unit) + note.evidence_ids)),
                note.sources,
                None if any(evidence[eid].detail.get("confidence_available") is False
                            for eid in note.evidence_ids) else note.confidence,
                continuation,
            ))
            emitted.add(unit_id)
    plan, tail_extensions = _extend_observed_utterance_tails(plan, units, utterance)
    unresolved = [x.id for x in units if x.id not in emitted and x.id not in omission_ids]
    if repaired:
        diagnostics.append(f"unmatched_units_retained:{len(repaired)}")
    if tail_extensions:
        diagnostics.append(f"observed_utterance_tails_extended:{tail_extensions}")
    result = Realization(1, document.canonical_text, canonical, performed, tuple(plan),
                         tuple(omissions), tuple(unresolved), tuple(diagnostics), document.evidence)
    result.validate()
    return result


def _unit_evidence_ids(unit: SingingUnit) -> tuple[str, ...]:
    return tuple(dict.fromkeys(unit.evidence_ids + tuple(
        evidence_id for boundary in (unit.consonant_start, unit.end) if boundary is not None
        for evidence_id in boundary.evidence_ids
    )))
