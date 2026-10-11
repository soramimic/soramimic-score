"""CTC-guided lyric allocation that preserves SheetSage note runs.

Unlike :mod:`soramimic_score.syllable_notes`, this experiment does not collapse every
syllable to one representative pitch.  Source notes remain distinct when their
pitch changes.  Adjacent same-pitch fragments are coalesced only when one
syllable consumes them and there is no audible-size gap; assigning another CTC
syllable at the boundary keeps them separate.

The score intentionally consumes only lyric-onset observations (normally raw
mora CTC, or explicitly derived repeated-vocalization anchors), known-lyric
segment IDs supplied by the caller, optional Whisper line windows, and pitched
SheetSage intervals. Reference/XF timing, singing-unit intervals, F0, and note
confidence values are not inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import math
from typing import Mapping, Sequence

from .ir import (MINIMUM_USABLE_TIMING_CONFIDENCE, IntermediateRepresentation,
                 NoteCandidate, SingingUnit)
from .japanese import SPECIAL_MORAS, kana_to_syllables


@dataclass(frozen=True)
class NoteRunConfig:
    """Nonnegative costs for the note-preserving dynamic program."""

    ctc_onset_weight: float = 0.11258317857582528
    ctc_onset_scale_sec: float = 0.10
    whisper_boundary_cost_per_sec2: float = 0.0
    pitch_change_weight: float = 0.30469269715364344
    source_note_split_weight: float = 0.2971988578273897
    mora_omission_weight: float = 0.7357166836930735
    note_only_open_weight: float = 1.50
    note_only_duration_weight: float = 0.75
    within_syllable_rest_open_weight: float = 1.00
    within_syllable_rest_duration_weight: float = 2.00
    segment_rest_open_weight: float = 0.60
    segment_rest_duration_weight: float = 0.6299605249474366
    rest_gap_threshold_sec: float = 0.04
    same_pitch_merge_gap_sec: float = 0.02

    def __post_init__(self) -> None:
        values = tuple(
            value for key, value in asdict(self).items()
            if key != "ctc_onset_scale_sec"
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("note-run weights and thresholds must be finite and nonnegative")
        if (not math.isfinite(self.ctc_onset_scale_sec)
                or self.ctc_onset_scale_sec <= 0):
            raise ValueError("ctc_onset_scale_sec must be finite and positive")


@dataclass(frozen=True)
class NoteRunCost:
    ctc_onset_distance: float = 0.0
    whisper_line_ownership: float = 0.0
    pitch_change: float = 0.0
    source_note_split: float = 0.0
    mora_omission: float = 0.0
    note_only_open: float = 0.0
    note_only_duration: float = 0.0
    within_syllable_rest: float = 0.0
    segment_rest: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.ctc_onset_distance
            + self.whisper_line_ownership
            + self.pitch_change
            + self.source_note_split
            + self.mora_omission
            + self.note_only_open
            + self.note_only_duration
            + self.within_syllable_rest
            + self.segment_rest
        )

    def __add__(self, other: "NoteRunCost") -> "NoteRunCost":
        return NoteRunCost(
            self.ctc_onset_distance + other.ctc_onset_distance,
            self.whisper_line_ownership + other.whisper_line_ownership,
            self.pitch_change + other.pitch_change,
            self.source_note_split + other.source_note_split,
            self.mora_omission + other.mora_omission,
            self.note_only_open + other.note_only_open,
            self.note_only_duration + other.note_only_duration,
            self.within_syllable_rest + other.within_syllable_rest,
            self.segment_rest + other.segment_rest,
        )


@dataclass(frozen=True)
class PreservedNote:
    source_note_ids: tuple[str, ...]
    start_sec: float
    end_sec: float
    midi_pitch: int
    continuation: bool


@dataclass(frozen=True)
class NoteRunAssignment:
    singing_unit_ids: tuple[str, ...]
    mora_ids: tuple[str, ...]
    kana: str
    segment_id: str | None
    ctc_onset_sec: float
    notes: tuple[PreservedNote, ...]
    source_note_split: bool
    cost: NoteRunCost
    timing_adjustment: str | None = None


@dataclass(frozen=True)
class NoteOnlySpan:
    source_note_ids: tuple[str, ...]
    position: str
    start_sec: float
    end_sec: float
    cost: NoteRunCost


@dataclass(frozen=True)
class NoteRunResult:
    assignments: tuple[NoteRunAssignment, ...]
    omitted_mora_ids: tuple[str, ...]
    note_only_spans: tuple[NoteOnlySpan, ...]
    cost: NoteRunCost

    @property
    def total_cost(self) -> float:
        return self.cost.total

    @property
    def omitted_source_note_ids(self) -> tuple[str, ...]:
        return tuple(note_id for span in self.note_only_spans
                     for note_id in span.source_note_ids)

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["omitted_source_note_ids"] = self.omitted_source_note_ids
        value["total_cost"] = self.total_cost
        return value


@dataclass(frozen=True)
class _Unit:
    source: SingingUnit
    mora_ids: tuple[str, ...]
    kana: str
    base_syllable: int
    utterance_id: str
    segment_id: str | None
    ctc_onset_sec: float | None
    must_assign: bool
    independent_phase: bool = False
    onset_supports_timing: bool = True

    @property
    def is_special(self) -> bool:
        return bool(self.kana) and all(char in SPECIAL_MORAS for char in self.kana)


@dataclass(frozen=True)
class _Step:
    assignment: NoteRunAssignment | None = None
    omitted_mora_ids: tuple[str, ...] = ()
    note_only_span: NoteOnlySpan | None = None
    cost: NoteRunCost = NoteRunCost()


@dataclass(frozen=True)
class _Path:
    cost: NoteRunCost
    steps: tuple[_Step, ...]
    tie_key: tuple[object, ...]


def _ctc_onsets(
    document: IntermediateRepresentation, unit: SingingUnit,
) -> tuple[tuple[float, bool], ...]:
    evidence = {item.id: item for item in document.evidence}
    values = []
    for evidence_id in unit.evidence_ids:
        item = evidence.get(evidence_id)
        if item is None or item.kind not in {
            "mora-ctc-anchor", "repeated-vocalization-anchor",
            "repeated-vocalization-reattack",
        }:
            continue
        value = item.detail.get("time_sec")
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError("lyric onset evidence needs a finite nonnegative time_sec")
        # A posterior floor is not evidence for the exact onset. Keep the
        # canonical mora and its raw observation, but do not pull its assignment
        # toward that peak. Derived acoustic anchors have no CTC posterior.
        usable = (item.kind != "mora-ctc-anchor"
                  or item.detail.get("confidence_available") is False
                  or item.confidence >= MINIMUM_USABLE_TIMING_CONFIDENCE)
        values.append((item.id, float(value), usable))
    return tuple((value, usable) for _identifier, value, usable in sorted(
        values, key=lambda item: (item[1], item[0]),
    ))


def _units(
    document: IntermediateRepresentation,
    segment_ids_by_mora: Mapping[str, str] | None,
) -> tuple[tuple[str, tuple[_Unit, ...]], ...]:
    readings = {item.id: item for item in document.readings}
    moras = {item.id: item for item in document.moras}
    evidence = {item.id: item for item in document.evidence}
    owner = {
        unit.id: readings[moras[unit.mora_ids[0]].reading_id].utterance_id
        for unit in document.singing_units
    }
    result = []
    for utterance in document.utterances:
        source_units = tuple(unit for unit in document.singing_units
                             if owner[unit.id] == utterance.id)
        atoms: list[tuple[SingingUnit, tuple[str, ...], str, float | None, bool]] = []
        for unit in source_units:
            onsets = _ctc_onsets(document, unit)
            if len(unit.mora_ids) > 1 and len(onsets) == len(unit.mora_ids):
                atoms.extend((unit, (mora_id,), moras[mora_id].text, onset, usable)
                             for mora_id, (onset, usable) in zip(
                                 unit.mora_ids, onsets, strict=True))
            else:
                atoms.append((
                    unit, unit.mora_ids,
                    "".join(moras[mid].text for mid in unit.mora_ids),
                    onsets[0][0] if len(onsets) == 1 else None,
                    onsets[0][1] if len(onsets) == 1 else False,
                ))
        syllables = kana_to_syllables("".join(item[2] for item in atoms))
        base_indices = []
        syllable_index = 0
        accumulated = ""
        for _unit, _moras, text, _onset, _usable in atoms:
            if syllable_index >= len(syllables):
                raise ValueError("singing units do not preserve the selected syllables")
            accumulated += text
            base_indices.append(syllable_index)
            if accumulated == syllables[syllable_index]:
                syllable_index += 1
                accumulated = ""
            elif not syllables[syllable_index].startswith(accumulated):
                raise ValueError("singing units split a selected syllable inconsistently")
        if accumulated or syllable_index != len(syllables):
            raise ValueError("singing units do not cover every selected syllable")
        converted = []
        for (unit, atom_moras, text, onset, usable), base_index in zip(
                atoms, base_indices, strict=True):
            segments = ({segment_ids_by_mora[mid] for mid in atom_moras}
                        if segment_ids_by_mora is not None else set())
            if len(segments) > 1:
                raise ValueError("one singing unit cannot cross lyric segments")
            phases = [evidence[identifier] for identifier in unit.evidence_ids
                      if identifier in evidence and evidence[identifier].detail.get(
                          "timing_method") == "acoustic-repetition-phase"]
            converted.append(_Unit(
                unit, atom_moras, text, base_index, utterance.id,
                next(iter(segments)) if segments else None, onset,
                any(
                    evidence.get(identifier) is not None
                    and evidence[identifier].kind == "repeated-vocalization-reattack"
                    for identifier in unit.evidence_ids
                ) or (bool(phases) and not all(char in SPECIAL_MORAS for char in text)
                      and any(item.detail.get("pitched_note_support") is True for item in phases)),
                bool(phases), usable,
            ))
        result.append((utterance.id, tuple(converted)))
    return tuple(result)


def _group_ends(units: Sequence[_Unit], start: int) -> range:
    base = units[start].base_syllable
    utterance_id = units[start].utterance_id
    stop = start + 1
    if not units[start].is_special:
        while (stop < len(units) and units[stop].base_syllable == base
               and units[stop].utterance_id == utterance_id):
            stop += 1
    return range(start + 1, stop + 1)


def _coalesce(
    notes: Sequence[NoteCandidate], *, start_sec: float,
    config: NoteRunConfig,
) -> tuple[PreservedNote, ...]:
    output: list[PreservedNote] = []
    for index, note in enumerate(notes):
        if note.midi_pitch is None:
            raise ValueError("note-run inputs must contain pitched SheetSage notes")
        begin = start_sec if index == 0 else note.start_sec
        if (output and output[-1].midi_pitch == note.midi_pitch
                and begin - output[-1].end_sec <= config.same_pitch_merge_gap_sec):
            previous = output[-1]
            output[-1] = replace(
                previous,
                source_note_ids=previous.source_note_ids + (note.id,),
                end_sec=max(previous.end_sec, note.end_sec),
            )
        else:
            output.append(PreservedNote(
                (note.id,), begin, note.end_sec, note.midi_pitch, bool(output),
            ))
    return tuple(output)


def _note_only(
    notes: Sequence[NoteCandidate], position: str, config: NoteRunConfig,
) -> NoteOnlySpan | None:
    if not notes:
        return None
    duration = sum(note.end_sec - note.start_sec for note in notes)
    cost = NoteRunCost(
        note_only_open=config.note_only_open_weight,
        note_only_duration=config.note_only_duration_weight * duration,
    )
    return NoteOnlySpan(
        tuple(note.id for note in notes), position,
        notes[0].start_sec, notes[-1].end_sec, cost,
    )


def _assignment_cost(
    units: Sequence[_Unit], source_notes: Sequence[NoteCandidate],
    *, start_sec: float, split: bool,
    previous_note: NoteCandidate | None, previous_segment: str | None,
    config: NoteRunConfig,
    line_windows_by_utterance: Mapping[str, tuple[float, float]] | None,
) -> NoteRunCost:
    onsets = tuple(unit.ctc_onset_sec for unit in units)
    if any(value is None for value in onsets):
        raise ValueError("an assigned syllable requires a raw mora-CTC onset")
    gaps = [max(0.0, right.start_sec - left.end_sec)
            for left, right in zip(source_notes, source_notes[1:])]
    audible_gaps = [gap for gap in gaps if gap > config.rest_gap_threshold_sec]
    # Coalescing adjacent equal pitches cannot change the pitch-change count.
    pitch_changes = sum(left.midi_pitch != right.midi_pitch
                        for left, right in zip(source_notes, source_notes[1:]))
    segments = {unit.segment_id for unit in units}
    if len(segments) > 1:
        raise ValueError("one syllable cannot cross lyric segments")
    segment = next(iter(segments))
    utterances = {unit.utterance_id for unit in units}
    if len(utterances) > 1:
        raise ValueError("one syllable cannot cross utterances")
    segment_rest = _segment_rest_cost(
        source_notes[0], previous_note, segment, previous_segment, config,
    )
    within_rest = 0.0
    if audible_gaps:
        within_rest = (config.within_syllable_rest_open_weight * len(audible_gaps)
                       + config.within_syllable_rest_duration_weight * sum(audible_gaps))
    ownership = 0.0
    utterance_id = next(iter(utterances))
    if line_windows_by_utterance is not None:
        window = line_windows_by_utterance.get(utterance_id)
        if window is None:
            raise ValueError(f"missing Whisper line window for {utterance_id}")
        line_start, line_end = window
        note_start = source_notes[0].start_sec
        if note_start < line_start or note_start > line_end:
            first_ctc = float(onsets[0])
            ctc_supports_crossing = (
                note_start < line_start and first_ctc < line_start
            ) or (
                note_start > line_end and first_ctc > line_end
            )
            boundary_distance = max(line_start - note_start, note_start - line_end, 0.0)
            outside_duration = sum(
                max(0.0, note.end_sec - note.start_sec
                    - max(0.0, min(note.end_sec, line_end)
                          - max(note.start_sec, line_start)))
                for note in source_notes
            )
            if not ctc_supports_crossing:
                ownership = (
                    config.whisper_boundary_cost_per_sec2
                    * (boundary_distance + outside_duration) ** 2
                )
    return NoteRunCost(
        ctc_onset_distance=(
            config.ctc_onset_weight
            * sum(abs(float(unit.ctc_onset_sec) - start_sec)
                  for unit in units if unit.onset_supports_timing)
            / config.ctc_onset_scale_sec
        ),
        whisper_line_ownership=ownership,
        pitch_change=config.pitch_change_weight * pitch_changes,
        source_note_split=config.source_note_split_weight if split else 0.0,
        within_syllable_rest=within_rest,
        segment_rest=segment_rest,
    )


def _segment_rest_cost(
    first_note: NoteCandidate, previous_note: NoteCandidate | None,
    segment: str | None, previous_segment: str | None, config: NoteRunConfig,
) -> float:
    gap = 0.0
    if (previous_note is not None and segment is not None and segment == previous_segment
            and first_note.start_sec >= previous_note.end_sec):
        gap = first_note.start_sec - previous_note.end_sec
    if gap > config.rest_gap_threshold_sec:
        return config.segment_rest_open_weight + config.segment_rest_duration_weight * gap
    return 0.0


def _extend(path: _Path, step: _Step, signature: tuple[object, ...]) -> _Path:
    return _Path(path.cost + step.cost, path.steps + (step,), path.tie_key + signature)


def _keep(
    states: dict[tuple[int, int, int | None, str | None], _Path],
    state: tuple[int, int, int | None, str | None], path: _Path,
    step: _Step, signature: tuple[object, ...],
) -> None:
    proposal = _extend(path, step, signature)
    current = states.get(state)
    if current is None or (proposal.cost.total, proposal.tie_key) < (
            current.cost.total, current.tie_key):
        states[state] = proposal


def _optimize_phrase(
    units: Sequence[_Unit], notes: Sequence[NoteCandidate], config: NoteRunConfig,
    line_windows_by_utterance: Mapping[str, tuple[float, float]] | None = None,
    required_note_ids: frozenset[str] = frozenset(),
) -> _Path:
    states: dict[tuple[int, int, int | None, str | None], _Path] = {
        (0, 0, None, None): _Path(NoteRunCost(), (), ())
    }
    for unit_index in range(len(units)):
        # Predecessor paths repeatedly consider the same lyric/note interval.
        # Cache only its scalar cost, scoped to one lyric position and bounded
        # independently of the phrase length. Materialize notes only for a path
        # that actually replaces a state below.
        @lru_cache(maxsize=65536)
        def assignment_cost(
            end_unit: int, start_note: int, end_note: int, split: bool,
        ) -> NoteRunCost:
            group = units[unit_index:end_unit]
            selected = notes[start_note:end_note]
            return _assignment_cost(
                group, selected,
                start_sec=(float(group[0].ctc_onset_sec)
                           if split else selected[0].start_sec),
                split=split, previous_note=None, previous_segment=None,
                config=config, line_windows_by_utterance=line_windows_by_utterance,
            )

        current = [(state, path) for state, path in states.items()
                   if state[0] == unit_index]
        for (unused_unit, cursor, previous_index, previous_segment), path in current:
            unit = units[unused_unit]
            supported_phase = (unit.ctc_onset_sec is not None and any(
                note.start_sec - .12 <= unit.ctc_onset_sec <= note.end_sec + .12 for note in notes))
            if not unit.must_assign or unit.independent_phase and not supported_phase:
                omitted_cost = NoteRunCost(
                    mora_omission=config.mora_omission_weight * len(unit.mora_ids)
                )
                _keep(
                    states,
                    (unused_unit + 1, cursor, previous_index, previous_segment),
                    path,
                    _Step(omitted_mora_ids=unit.mora_ids, cost=omitted_cost),
                    ("omit-mora", unit.source.id),
                )
            for end_unit in _group_ends(units, unused_unit):
                group = units[unused_unit:end_unit]
                if any(item.ctc_onset_sec is None for item in group):
                    continue
                segments = {item.segment_id for item in group}
                if len(segments) > 1:
                    continue
                segment = next(iter(segments))
                starts = list(range(cursor, len(notes)))
                if previous_index is not None and cursor == previous_index + 1:
                    onset = group[0].ctc_onset_sec
                    previous = notes[previous_index]
                    if onset is not None and previous.start_sec < onset < previous.end_sec:
                        starts.insert(0, previous_index)
                for start_note in starts:
                    split = start_note < cursor
                    if (group[0].independent_phase and not notes[start_note].start_sec - .12
                            <= float(group[0].ctc_onset_sec) <= notes[start_note].end_sec + .12):
                        continue
                    skipped_notes = () if split else notes[cursor:start_note]
                    if any(note.id in required_note_ids for note in skipped_notes):
                        continue
                    position = "leading" if previous_index is None else "internal"
                    skipped = _note_only(skipped_notes, position, config)
                    previous_note = (notes[previous_index]
                                     if previous_index is not None else None)
                    segment_rest = _segment_rest_cost(
                        notes[start_note], previous_note, segment, previous_segment, config,
                    )
                    for end_note in range(start_note + 1, len(notes) + 1):
                        cost = assignment_cost(end_unit, start_note, end_note, split)
                        if segment_rest or math.copysign(1.0, segment_rest) < 0:
                            cost = replace(cost, segment_rest=segment_rest)
                        if skipped is not None:
                            cost += skipped.cost
                        next_cursor = max(cursor, end_note)
                        state = (end_unit, next_cursor, end_note - 1, segment)
                        candidate_cost = path.cost + cost
                        candidate_total = candidate_cost.total
                        incumbent = states.get(state)
                        incumbent_total = incumbent.cost.total if incumbent is not None else math.inf
                        if incumbent is not None and candidate_total > incumbent_total:
                            continue
                        selected = notes[start_note:end_note]
                        mora_ids = tuple(mid for item in group for mid in item.mora_ids)
                        tie_key = path.tie_key + (
                            "assign", mora_ids, tuple(note.id for note in selected), split,
                        )
                        if (incumbent is not None and not (
                            (candidate_total, tie_key) < (incumbent_total, incumbent.tie_key)
                        )):
                            continue
                        output = _coalesce(
                            selected,
                            start_sec=(float(group[0].ctc_onset_sec)
                                       if split else selected[0].start_sec),
                            config=config,
                        )
                        assignment = NoteRunAssignment(
                            tuple(dict.fromkeys(item.source.id for item in group)),
                            mora_ids, "".join(item.kana for item in group), segment,
                            float(group[0].ctc_onset_sec), output, split, cost,
                        )
                        step = _Step(assignment=assignment,
                                     note_only_span=skipped, cost=cost)
                        states[state] = _Path(candidate_cost, path.steps + (step,), tie_key)
    finalists = []
    for (unit_index, cursor, _previous, _segment), path in states.items():
        if unit_index != len(units):
            continue
        if any(note.id in required_note_ids for note in notes[cursor:]):
            continue
        trailing = _note_only(notes[cursor:], "trailing", config)
        step = (_Step(note_only_span=trailing, cost=trailing.cost)
                if trailing is not None else _Step())
        finalists.append(_extend(path, step, (
            "terminal", () if trailing is None else trailing.source_note_ids,
        )))
    if not finalists:
        raise RuntimeError("note-run optimizer did not reach a terminal state")
    return min(finalists, key=lambda path: (path.cost.total, path.tie_key))


def _resolve_splits(
    assignments: Sequence[NoteRunAssignment],
    notes_by_id: Mapping[str, NoteCandidate],
) -> tuple[NoteRunAssignment, ...]:
    result = list(assignments)
    for index in range(1, len(result)):
        right = result[index]
        if not right.source_note_split:
            continue
        left = result[index - 1]
        shared = right.notes[0].source_note_ids[0]
        if shared not in left.notes[-1].source_note_ids:
            raise RuntimeError("a split assignment does not reuse the preceding source note")
        note = notes_by_id[shared]
        split_start = max(left.notes[-1].start_sec, note.start_sec)
        split_end = min(right.notes[0].end_sec, note.end_sec)
        if split_end <= split_start:
            raise RuntimeError("a reused source note has no remaining duration to split")
        requested = right.ctc_onset_sec
        adjustment = None
        if requested <= split_start + 1e-6:
            # CTC timestamps are frame-quantized.  If several consecutive
            # syllables reuse one source note at the same timestamp, an
            # epsilon clamp would leave the middle syllables effectively
            # inaudible.  Share the uncertain interval up to the next distinct
            # onset (or the source-note end) across every remaining syllable.
            remaining_coincident = 1
            redistribution_end = split_end
            for following in result[index + 1:]:
                if (not following.source_note_split
                        or following.notes[0].source_note_ids[0] != shared):
                    break
                if following.ctc_onset_sec <= requested + 1e-6:
                    remaining_coincident += 1
                    continue
                redistribution_end = min(redistribution_end, following.ctc_onset_sec)
                break
            boundary = split_start + (
                (redistribution_end - split_start) / (remaining_coincident + 1)
            )
            adjustment = "coincident_ctc_proportional"
        else:
            margin = min(1e-6, (split_end - split_start) / 3)
            boundary = min(max(requested, split_start + margin), split_end - margin)
        if not split_start < boundary < split_end:
            raise RuntimeError("a reused source note is too short to split")
        left_notes = left.notes[:-1] + (replace(left.notes[-1], end_sec=boundary),)
        right_notes = (replace(right.notes[0], start_sec=boundary),) + right.notes[1:]
        result[index - 1] = replace(
            left, notes=left_notes,
            timing_adjustment=adjustment or left.timing_adjustment,
        )
        result[index] = replace(
            right, notes=right_notes,
            timing_adjustment=adjustment or right.timing_adjustment,
        )
    return tuple(result)


def optimize_note_runs(
    document: IntermediateRepresentation,
    config: NoteRunConfig | None = None,
    *,
    segment_ids_by_mora: Mapping[str, str] | None = None,
    line_windows_by_utterance: Mapping[str, tuple[float, float]] | None = None,
) -> NoteRunResult:
    """Allocate lyric syllables while retaining source-note pitch boundaries.

    ``segment_ids_by_mora`` should be derived from the selected lyrics (for
    example bunsetsu-like groups), never from reference/XF note truth. Optional
    line windows must be raw Whisper segment bounds keyed by utterance ID.
    """
    if not isinstance(document, IntermediateRepresentation):
        raise TypeError("document must be an IntermediateRepresentation")
    if document.links:
        raise ValueError("note-run input must not contain decoded links")
    if any(note.midi_pitch is None for note in document.note_candidates):
        raise ValueError("note-run inputs must contain pitched notes only")
    if any((left.start_sec, left.end_sec) > (right.start_sec, right.end_sec)
           for left, right in zip(document.note_candidates, document.note_candidates[1:])):
        raise ValueError("note candidates must be supplied in chronological order")
    settings = config or NoteRunConfig()
    phrases = _units(document, segment_ids_by_mora)
    required_notes_by_utterance: dict[str, frozenset[str]] = {}
    for item in document.evidence:
        if item.kind != "repeated-vocalization-expansion":
            continue
        utterance_id = item.detail.get("utterance_id")
        note_ids = item.detail.get("required_note_candidate_ids")
        if not isinstance(utterance_id, str) or not isinstance(note_ids, list):
            raise ValueError("repeated vocalization coverage evidence is malformed")
        if any(not isinstance(note_id, str) for note_id in note_ids):
            raise ValueError("repeated vocalization note IDs must be strings")
        required_notes_by_utterance[utterance_id] = frozenset(note_ids)
    if line_windows_by_utterance is not None:
        expected = {utterance_id for utterance_id, _units_for_phrase in phrases}
        if set(line_windows_by_utterance) != expected:
            raise ValueError("Whisper line windows must cover every utterance exactly")
        previous_end = -math.inf
        for utterance_id, _units_for_phrase in phrases:
            start_sec, end_sec = line_windows_by_utterance[utterance_id]
            if (not math.isfinite(start_sec) or not math.isfinite(end_sec)
                    or start_sec < 0 or end_sec < start_sec):
                raise ValueError("Whisper line windows must be finite ordered intervals")
            if (start_sec < previous_end
                    and not math.isclose(start_sec, previous_end, abs_tol=1e-9)):
                raise ValueError("Whisper line windows must not overlap or go backward")
            previous_end = end_sec
    anchors = []
    for phrase_index, (utterance_id, units) in enumerate(phrases):
        times = [unit.ctc_onset_sec for unit in units if unit.ctc_onset_sec is not None
                 and not (unit.independent_phase and unit.is_special)]
        if times:
            anchors.append((phrase_index, utterance_id, min(times), max(times)))
    if any(left[2] > right[2] or left[3] > right[3]
           for left, right in zip(anchors, anchors[1:])):
        raise ValueError("utterance CTC onsets must follow chronological order")
    cuts = [(left[3] + right[2]) / 2 for left, right in zip(anchors, anchors[1:])]
    note_groups = {utterance_id: [] for utterance_id, _units_for_phrase in phrases}
    if anchors:
        owner_index = 0
        for note in document.note_candidates:
            while owner_index < len(cuts) and note.start_sec >= cuts[owner_index]:
                owner_index += 1
            note_groups[anchors[owner_index][1]].append(note)
    elif phrases:
        # With no CTC evidence there is no lyric-bearing owner to infer. Keep
        # the notes explicit as note-only observations in the first phrase and
        # let every canonical unit remain unresolved.
        note_groups[phrases[0][0]].extend(document.note_candidates)
    paths = tuple(_optimize_phrase(
        units, note_groups[utterance_id], settings,
        line_windows_by_utterance=line_windows_by_utterance,
        required_note_ids=required_notes_by_utterance.get(
            utterance_id, frozenset(),
        ),
    ) for utterance_id, units in phrases)
    assignments = tuple(step.assignment for path in paths for step in path.steps
                        if step.assignment is not None)
    assignments = _resolve_splits(
        assignments, {note.id: note for note in document.note_candidates},
    )
    omitted_moras = tuple(mid for path in paths for step in path.steps
                          for mid in step.omitted_mora_ids)
    spans = tuple(step.note_only_span for path in paths for step in path.steps
                  if step.note_only_span is not None)
    total = NoteRunCost()
    for path in paths:
        total += path.cost
    return NoteRunResult(assignments, omitted_moras, spans, total)
