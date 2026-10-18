"""The single production Stage 3 execution path.

Text recognition, reading selection, CTC alignment, and melody transcription
belong to adapters supplied by the caller. This package accepts their immutable
observation document, decodes a note-preserving correspondence once, and compiles
one realization.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

from .correspondence import (
    CorrespondenceConfig,
    CorrespondenceResult,
    align_correspondence,
)
from .ir import (
    Evidence,
    IntermediateRepresentation,
    Link,
    LinkCost,
    NoteCandidate,
    SingingUnit,
    has_usable_timing,
)
from .note_runs import NoteRunConfig, NoteRunCost, NoteRunResult, optimize_note_runs
from .realization import Realization, compile_realization
from .vocalization import (
    VocalizationReattack,
    expand_repeated_vocalizations,
)


@dataclass(frozen=True)
class Stage3Run:
    document: IntermediateRepresentation
    realization: Realization
    correspondences: tuple[CorrespondenceResult, ...]
    note_run: NoteRunResult | None = None


def run_correspondence_document(
    document: IntermediateRepresentation, *,
    config: CorrespondenceConfig | None = None,
) -> Stage3Run:
    """Run the retired semi-Markov decoder for research comparisons.

    Notes are partitioned by onset at the midpoint between neighboring observed
    utterance ranges. This keeps a boundary-straddling note with the lyric that
    was already sounding when it began. An utterance without timing gets no
    inferred time or note ownership.
    Each note is consumed once, and the common compiler handles overlaps across
    utterance joins. Existing links are rejected rather than silently reused.
    """
    document.validate()
    if document.links:
        raise ValueError("Stage 3 input must not contain previously decoded links")
    if any((a.start_sec, a.end_sec) > (b.start_sec, b.end_sec)
           for a, b in zip(document.note_candidates, document.note_candidates[1:])):
        raise ValueError("note candidates must be supplied in chronological order")
    by_reading = {r.id: r.utterance_id for r in document.readings}
    by_mora = {m.id: by_reading[m.reading_id] for m in document.moras}
    unit_owner = {u.id: by_mora[u.mora_ids[0]] for u in document.singing_units}
    observed = []
    for utterance in document.utterances:
        timed = [u for u in document.singing_units if unit_owner[u.id] == utterance.id
                 and has_usable_timing(u.consonant_start) and has_usable_timing(u.end)]
        if timed:
            observed.append((utterance.id, min(u.consonant_start.time_sec for u in timed),
                             max(u.end.time_sec for u in timed)))
    if any(left[1] > right[1] or left[2] > right[2]
           for left, right in zip(observed, observed[1:])):
        raise ValueError("observed utterances must follow chronological order")
    cuts = [(left[2] + right[1]) / 2 for left, right in zip(observed, observed[1:])]
    owners: dict[str | None, list[NoteCandidate]] = {}
    for note in document.note_candidates:
        owner = observed[bisect_right(cuts, note.start_sec)][0] if observed else None
        owners.setdefault(owner, []).append(note)
    all_links = []
    results = []
    occupied = {item.id for values in (
        document.utterances, document.readings, document.moras, document.phonemes,
        document.vowel_nuclei, document.singing_units, document.note_candidates, document.evidence,
    ) for item in values}
    prefix = "stage3-link-"
    while any(item.startswith(prefix) for item in occupied):
        prefix = "_" + prefix
    for owner in [u.id for u in document.utterances] + ([None] if None in owners else []):
        readings = tuple(r for r in document.readings if r.utterance_id == owner)
        reading_ids = {r.id for r in readings}
        moras = tuple(m for m in document.moras if m.reading_id in reading_ids)
        mora_ids = {m.id for m in moras}
        units = tuple(u for u in document.singing_units if unit_owner[u.id] == owner)
        unit_ids = {u.id for u in units}
        part = replace(
            document, utterances=tuple(u for u in document.utterances if u.id == owner),
            readings=readings, moras=moras,
            phonemes=tuple(p for p in document.phonemes if p.mora_ids and set(p.mora_ids) <= mora_ids),
            vowel_nuclei=tuple(v for v in document.vowel_nuclei if v.singing_unit_id in unit_ids),
            singing_units=units, note_candidates=tuple(owners.get(owner, ())),
        )
        result = align_correspondence(part, config)
        results.append(result)
        for link in result.links:
            all_links.append(replace(link, id=f"{prefix}{len(all_links)}"))
    linked = replace(document, links=tuple(all_links))
    realization = compile_realization(linked)
    return Stage3Run(linked, realization, tuple(results))


def _available_prefix(occupied: set[str], stem: str) -> str:
    prefix = stem
    while any(identifier.startswith(prefix) for identifier in occupied):
        prefix = "_" + prefix
    return prefix


def _without_note_only(cost: NoteRunCost) -> NoteRunCost:
    return replace(cost, note_only_open=0.0, note_only_duration=0.0)


def _link_cost(cost: NoteRunCost) -> LinkCost:
    structural = cost.total - cost.ctc_onset_distance
    return LinkCost(
        ctc_anchor_timing=cost.ctc_onset_distance,
        candidate_confidence=0.0,
        complexity=max(0.0, structural),
    )


def _operation(
    unit_ids: tuple[str, ...], note_ids: tuple[str, ...],
    units: Mapping[str, SingingUnit], document: IntermediateRepresentation,
) -> str:
    if len(unit_ids) == 1:
        return "match" if len(note_ids) == 1 else "melisma"
    if len(note_ids) == 1:
        return "stack"
    phonemes = {item.id: item for item in document.phonemes}
    if len(unit_ids) == 2:
        closure = tuple(phonemes[identifier].symbol
                        for identifier in units[unit_ids[1]].phoneme_ids)
        if closure and all(symbol == "q" for symbol in closure):
            return "sokuon"
    raise ValueError(
        "note-run materialization needs one lexical singing unit per syllable"
    )


def _default_segment_ids(document: IntermediateRepresentation) -> dict[str, str]:
    readings = {item.id: item for item in document.readings}
    moras = {item.id: item for item in document.moras}
    return {
        mora_id: readings[moras[mora_id].reading_id].utterance_id
        for unit in document.singing_units for mora_id in unit.mora_ids
    }


def _split_units_for_note_run(
    document: IntermediateRepresentation,
    result: NoteRunResult,
) -> tuple[IntermediateRepresentation, tuple[tuple[str, ...], ...], tuple[str, ...]]:
    """Materialize mora-level CTC decisions as replayable singing units.

    The canonical IR groups special morae such as ン and ー with the preceding
    lexical syllable. Raw mora CTC can nevertheless place those parts on
    distinct melody notes. Split only the linked output's unit structure when
    the DP made such a decision; the caller's observation document stays
    unchanged and every derived unit points back to its source unit.
    """
    source_units = {item.id: item for item in document.singing_units}
    phonemes = {item.id: item for item in document.phonemes}
    nuclei = {item.id: item for item in document.vowel_nuclei}
    evidence = {item.id: item for item in document.evidence}
    occupied = {
        item.id for collection in (
            document.utterances, document.readings, document.moras,
            document.phonemes, document.vowel_nuclei, document.singing_units,
            document.evidence, document.note_candidates, document.links,
        ) for item in collection
    }
    unit_prefix = _available_prefix(occupied, "stage3-note-run-unit-")
    evidence_prefix = _available_prefix(occupied, "stage3-note-run-unit-evidence-")
    pieces: dict[str, list[tuple[str, int | None, tuple[str, ...]]]] = {
        identifier: [] for identifier in source_units
    }
    for assignment_index, assignment in enumerate(result.assignments):
        selected = set(assignment.mora_ids)
        for source_id in assignment.singing_unit_ids:
            source = source_units[source_id]
            mora_ids = tuple(mid for mid in source.mora_ids if mid in selected)
            if mora_ids:
                pieces[source_id].append(("assignment", assignment_index, mora_ids))
    omitted = set(result.omitted_mora_ids)
    for source in document.singing_units:
        mora_ids = tuple(mid for mid in source.mora_ids if mid in omitted)
        if mora_ids:
            pieces[source.id].append(("omitted", None, mora_ids))

    derived_evidence: list[Evidence] = []
    rebuilt_units: list[SingingUnit] = []
    piece_for_mora: dict[tuple[str, str], str] = {}
    assignment_units: list[list[str]] = [[] for _ in result.assignments]
    omitted_units: list[str] = []
    for source in document.singing_units:
        source_pieces = pieces[source.id]
        covered = [mid for _kind, _index, mids in source_pieces for mid in mids]
        if (len(covered) != len(set(covered))
                or set(covered) != set(source.mora_ids)):
            raise ValueError("note-run decisions must partition every singing unit")
        order = {mid: index for index, mid in enumerate(source.mora_ids)}
        source_pieces.sort(key=lambda item: min(order[mid] for mid in item[2]))
        ctc = sorted(
            (evidence[eid] for eid in source.evidence_ids
             if eid in evidence and evidence[eid].kind == "mora-ctc-anchor"),
            key=lambda item: (float(item.detail.get("time_sec", 0.0)), item.id),
        )
        ctc_by_mora = (dict(zip(source.mora_ids, (item.id for item in ctc), strict=True))
                       if len(ctc) == len(source.mora_ids) else {})
        shared_evidence = tuple(
            eid for eid in source.evidence_ids
            if eid not in {item.id for item in ctc}
        )
        for kind, assignment_index, mora_ids in source_pieces:
            is_identity = len(source_pieces) == 1 and mora_ids == source.mora_ids
            if is_identity:
                unit_id = source.id
                unit_evidence_ids = source.evidence_ids
            else:
                unit_id = f"{unit_prefix}{len(rebuilt_units)}"
                derivation_id = f"{evidence_prefix}{len(derived_evidence)}"
                derived_evidence.append(Evidence(
                    derivation_id, "soramimic_score.note_runs", "note-run-unit-derivation",
                    1.0, {
                        "source_singing_unit_id": source.id,
                        "mora_ids": list(mora_ids),
                        "decision": kind,
                        "assignment_index": assignment_index,
                    },
                ))
                selected_ctc = tuple(
                    ctc_by_mora[mid] for mid in mora_ids if mid in ctc_by_mora
                )
                unit_evidence_ids = shared_evidence + selected_ctc + (derivation_id,)
            selected_moras = set(mora_ids)
            phoneme_ids = tuple(
                pid for pid in source.phoneme_ids
                if selected_moras.intersection(phonemes[pid].mora_ids)
            )
            nucleus_ids = tuple(
                nid for nid in source.vowel_nucleus_ids
                if set(nuclei[nid].phoneme_ids).intersection(phoneme_ids)
            )
            rebuilt_units.append(replace(
                source, id=unit_id, mora_ids=mora_ids,
                phoneme_ids=phoneme_ids, vowel_nucleus_ids=nucleus_ids,
                evidence_ids=unit_evidence_ids,
            ))
            for mora_id in mora_ids:
                piece_for_mora[source.id, mora_id] = unit_id
            if kind == "assignment":
                assert assignment_index is not None
                assignment_units[assignment_index].append(unit_id)
            else:
                omitted_units.append(unit_id)

    # A CTC syllable can span source units (most commonly a base mora followed
    # by a standalone sokuon unit). The note-run output carries one combined
    # pronunciation, so materialize one combined unit rather than asking the
    # generic stack compiler to create multiple synthesis slots on one note.
    for assignment_index, unit_ids in enumerate(assignment_units):
        if len(unit_ids) <= 1:
            continue
        by_id = {unit.id: unit for unit in rebuilt_units}
        members = [by_id[unit_id] for unit_id in unit_ids]
        positions = [index for index, unit in enumerate(rebuilt_units)
                     if unit.id in set(unit_ids)]
        merged_id_index = len(rebuilt_units)
        used_ids = {unit.id for unit in rebuilt_units}
        merged_id = f"{unit_prefix}{merged_id_index}"
        while merged_id in used_ids:
            merged_id_index += 1
            merged_id = f"{unit_prefix}{merged_id_index}"
        derivation_id = f"{evidence_prefix}{len(derived_evidence)}"
        assignment = result.assignments[assignment_index]
        derived_evidence.append(Evidence(
            derivation_id, "soramimic_score.note_runs", "note-run-unit-derivation",
            1.0, {
                "source_singing_unit_ids": list(assignment.singing_unit_ids),
                "mora_ids": list(assignment.mora_ids),
                "decision": "assignment-merge",
                "assignment_index": assignment_index,
            },
        ))
        status = ("weak" if any(unit.status == "weak" for unit in members)
                  else members[0].status)
        starts = [unit.consonant_start for unit in members
                  if unit.consonant_start is not None]
        ends = [unit.end for unit in members if unit.end is not None]
        merged = replace(
            members[0],
            id=merged_id,
            mora_ids=assignment.mora_ids,
            phoneme_ids=tuple(dict.fromkeys(
                pid for unit in members for pid in unit.phoneme_ids
            )),
            vowel_nucleus_ids=tuple(dict.fromkeys(
                nid for unit in members for nid in unit.vowel_nucleus_ids
            )),
            status=status,
            confidence=min(unit.confidence for unit in members),
            consonant_start=(min(starts, key=lambda item: item.time_sec)
                             if starts else None),
            end=max(ends, key=lambda item: item.time_sec) if ends else None,
            evidence_ids=tuple(dict.fromkeys(
                eid for unit in members for eid in unit.evidence_ids
            )) + (derivation_id,),
        )
        first_position = min(positions)
        rebuilt_units = [unit for unit in rebuilt_units if unit.id not in set(unit_ids)]
        rebuilt_units.insert(first_position, merged)
        assignment_units[assignment_index] = [merged_id]
        selected_moras = set(assignment.mora_ids)
        for source_id in assignment.singing_unit_ids:
            for mora_id in source_units[source_id].mora_ids:
                if mora_id in selected_moras:
                    piece_for_mora[source_id, mora_id] = merged_id

    rebuilt_moras = tuple(replace(
        mora,
        singing_unit_ids=tuple(
            piece_for_mora[unit_id, mora.id] for unit_id in mora.singing_unit_ids
        ),
    ) for mora in document.moras)
    rebuilt_nuclei = tuple(replace(
        nucleus,
        singing_unit_id=piece_for_mora[
            nucleus.singing_unit_id,
            next(mora_id for phoneme_id in nucleus.phoneme_ids
                 for mora_id in phonemes[phoneme_id].mora_ids),
        ],
    ) for nucleus in document.vowel_nuclei)
    rebuilt = replace(
        document,
        moras=rebuilt_moras,
        vowel_nuclei=rebuilt_nuclei,
        singing_units=tuple(rebuilt_units),
        evidence=document.evidence + tuple(derived_evidence),
    )
    return rebuilt, tuple(tuple(ids) for ids in assignment_units), tuple(omitted_units)


def _materialize_note_run(
    document: IntermediateRepresentation,
    result: NoteRunResult,
    config: NoteRunConfig,
) -> tuple[IntermediateRepresentation, tuple[CorrespondenceResult, ...]]:
    """Turn the note-run decision into replayable IR links and derived notes.

    Raw SheetSage candidates remain untouched in the document.  Each final
    coalesced or split interval gets a new candidate with explicit derivation
    evidence, so saving and reloading the linked IR reproduces the same
    realization without rerunning the optimizer.
    """
    document, assignment_units, omitted_unit_ids = _split_units_for_note_run(
        document, result,
    )
    source_notes = {item.id: item for item in document.note_candidates}
    units = {item.id: item for item in document.singing_units}
    readings = {item.id: item for item in document.readings}
    moras = {item.id: item for item in document.moras}
    unit_owner = {
        unit.id: readings[moras[unit.mora_ids[0]].reading_id].utterance_id
        for unit in document.singing_units
    }
    occupied = {
        item.id for collection in (
            document.utterances, document.readings, document.moras,
            document.phonemes, document.vowel_nuclei, document.singing_units,
            document.evidence, document.note_candidates, document.links,
        ) for item in collection
    }
    note_prefix = _available_prefix(occupied, "stage3-note-run-")
    evidence_prefix = _available_prefix(occupied, "stage3-note-run-evidence-")
    derived_notes: list[NoteCandidate] = []
    derived_evidence: list[Evidence] = []
    pending_links: list[tuple[tuple[str, ...], tuple[str, ...], str,
                              NoteRunCost, tuple[str, ...]]] = []
    assigned_units: set[str] = set()

    for assignment_index, assignment in enumerate(result.assignments):
        unit_ids = assignment_units[assignment_index]
        if not unit_ids or any(identifier not in units for identifier in unit_ids):
            raise ValueError("note-run assignment references a missing singing unit")
        if assigned_units.intersection(unit_ids):
            raise ValueError("note-run assignment consumes a singing unit twice")
        expected_moras = tuple(
            mora_id for identifier in unit_ids for mora_id in units[identifier].mora_ids
        )
        if set(expected_moras) != set(assignment.mora_ids):
            raise ValueError("note-run assignment must consume a complete lexical syllable")
        assigned_units.update(unit_ids)
        final_note_ids: list[str] = []
        final_evidence_ids: list[str] = []
        for note_index, preserved in enumerate(assignment.notes):
            sources = tuple(source_notes[identifier]
                            for identifier in preserved.source_note_ids)
            if not sources:
                raise ValueError("a derived note needs source SheetSage candidates")
            note_id = f"{note_prefix}{len(derived_notes)}"
            evidence_id = f"{evidence_prefix}{len(derived_evidence)}"
            detail = {
                "source_note_ids": list(preserved.source_note_ids),
                "coalesced": len(preserved.source_note_ids) > 1,
                "source_note_split": assignment.source_note_split,
                "timing_adjustment": assignment.timing_adjustment,
                "continuation": preserved.continuation,
                "assignment_index": assignment_index,
                "note_index": note_index,
            }
            derived_evidence.append(Evidence(
                evidence_id, "soramimic_score.note_runs", "note-run-derivation", 1.0, detail,
            ))
            evidence_ids = tuple(dict.fromkeys(
                evidence_id for source in sources for evidence_id in source.evidence_ids
            )) + (evidence_id,)
            derived_notes.append(NoteCandidate(
                note_id, preserved.start_sec, preserved.end_sec,
                preserved.midi_pitch, min(source.confidence for source in sources),
                tuple(dict.fromkeys(source_name for source in sources
                                    for source_name in source.sources)),
                evidence_ids,
            ))
            final_note_ids.append(note_id)
            final_evidence_ids.append(evidence_id)
        if not final_note_ids:
            raise ValueError("a note-run assignment must produce at least one note")
        operation = _operation(unit_ids, tuple(final_note_ids), units, document)
        decision_cost = _without_note_only(assignment.cost)
        evidence_ids = tuple(dict.fromkeys(
            evidence_id for identifier in unit_ids
            for evidence_id in units[identifier].evidence_ids
        )) + tuple(final_evidence_ids)
        pending_links.append((
            unit_ids, tuple(final_note_ids), operation, decision_cost, evidence_ids,
        ))

    for unit_id in omitted_unit_ids:
        unit = units[unit_id]
        if unit.id in assigned_units:
            raise ValueError("note-run unit cannot be assigned and omitted")
        decision_evidence_id = f"{evidence_prefix}{len(derived_evidence)}"
        derived_evidence.append(Evidence(
            decision_evidence_id, "soramimic_score.note_runs", "note-run-unit-only",
            1.0, {"mora_ids": list(unit.mora_ids)},
        ))
        cost = NoteRunCost(
            mora_omission=config.mora_omission_weight * len(unit.mora_ids)
        )
        pending_links.append((
            (unit.id,), (), "unit_only", cost,
            tuple(unit.evidence_ids) + (decision_evidence_id,),
        ))

    for span in result.note_only_spans:
        for note_index, note_id in enumerate(span.source_note_ids):
            note = source_notes[note_id]
            cost = span.cost if note_index == 0 else NoteRunCost()
            pending_links.append((
                (), (note_id,), "note_only", cost, tuple(note.evidence_ids),
            ))

    # Unpitched candidates are explicit melody-model rests. They are outside
    # the pitched note-run DP but stay replayable in the linked observation IR.
    for note in document.note_candidates:
        if note.midi_pitch is None:
            pending_links.append((
                (), (note.id,), "rest", NoteRunCost(), tuple(note.evidence_ids),
            ))

    occupied.update(item.id for item in derived_notes)
    occupied.update(item.id for item in derived_evidence)
    link_prefix = _available_prefix(occupied, "stage3-link-")
    links = tuple(
        Link(
            f"{link_prefix}{index}", unit_ids, note_ids, operation,
            cost.total, max(0.0, min(1.0, math.exp(-cost.total))),
            _link_cost(cost), evidence_ids,
        )
        for index, (unit_ids, note_ids, operation, cost, evidence_ids)
        in enumerate(pending_links)
    )
    linked = replace(
        document,
        evidence=document.evidence + tuple(derived_evidence),
        note_candidates=tuple(sorted(
            document.note_candidates + tuple(derived_notes),
            key=lambda item: (item.start_sec, item.end_sec, item.id),
        )),
        links=links,
    )

    links_by_owner: dict[str, list[Link]] = {
        utterance.id: [] for utterance in document.utterances
    }
    timed_owner = []
    evidence = {item.id: item for item in document.evidence}
    for utterance in document.utterances:
        owned_units = [unit for unit in document.singing_units
                       if unit_owner[unit.id] == utterance.id]
        anchors = [
            float(evidence[evidence_id].detail["time_sec"])
            for unit in owned_units for evidence_id in unit.evidence_ids
            if evidence_id in evidence
            and evidence[evidence_id].kind == "mora-ctc-anchor"
            and isinstance(evidence[evidence_id].detail.get("time_sec"), (int, float))
        ]
        if anchors:
            timed_owner.append((utterance.id, min(anchors), max(anchors)))
    for link in links:
        if not document.utterances:
            continue
        if link.singing_unit_ids:
            owner = unit_owner[link.singing_unit_ids[0]]
        elif timed_owner:
            note = next(item for item in linked.note_candidates
                        if item.id == link.note_candidate_ids[0])
            owner = min(timed_owner, key=lambda item: (
                max(item[1] - note.start_sec, note.start_sec - item[2], 0.0),
                item[1], item[0],
            ))[0]
        else:
            owner = document.utterances[0].id
        links_by_owner[owner].append(link)
    correspondences = tuple(
        CorrespondenceResult(
            tuple(links_by_owner[utterance.id]),
            sum(link.cost for link in links_by_owner[utterance.id]),
            math.exp(-sum(link.cost for link in links_by_owner[utterance.id])),
            (),
        )
        for utterance in document.utterances
    )
    return linked, correspondences


def run_stage3_document(
    document: IntermediateRepresentation, *,
    config: NoteRunConfig | None = None,
    segment_ids_by_mora: Mapping[str, str] | None = None,
    line_windows_by_utterance: Mapping[str, tuple[float, float]] | None = None,
    utterance_groups: Sequence[Sequence[str]] | None = None,
    vocalization_reattacks_by_utterance: Mapping[
        str, Sequence[VocalizationReattack]
    ] | None = None,
) -> Stage3Run:
    """Run the production CTC-guided, note-preserving Stage 3 decoder.

    The score consumes raw mora-CTC observations, caller lyric segments,
    optional Whisper line windows, and pitched SheetSage candidates only.
    Automatic-transcription callers may supply transcript-independent acoustic
    re-attacks for a Whisper-rounded homogeneous vocalization. Whisper supplies
    the mora identity, re-attacks supply the count, and overlapping SheetSage
    notes supply pitch/duration coverage.
    Same-pitch fragments may coalesce inside one
    syllable, while CTC-separated syllables and pitch-change continuations stay
    distinct in the final synthesis plan. When no finer lyric segmentation is
    supplied, each utterance is used as one segment.
    """
    document.validate()
    if document.links:
        raise ValueError("Stage 3 input must not contain previously decoded links")
    if config is not None and not isinstance(config, NoteRunConfig):
        raise TypeError(
            "production Stage 3 requires NoteRunConfig; "
            "use run_correspondence_document for the retired comparison decoder"
        )
    if any((left.start_sec, left.end_sec) > (right.start_sec, right.end_sec)
           for left, right in zip(document.note_candidates, document.note_candidates[1:])):
        raise ValueError("note candidates must be supplied in chronological order")
    settings = config or NoteRunConfig()
    expansion = None
    if vocalization_reattacks_by_utterance is not None:
        if line_windows_by_utterance is None:
            raise ValueError(
                "repeated vocalization expansion requires raw Whisper line windows"
            )
        expansion = expand_repeated_vocalizations(
            document, line_windows_by_utterance,
            vocalization_reattacks_by_utterance,
        )
        document = expansion.document
    if segment_ids_by_mora is None:
        segments = _default_segment_ids(document)
    else:
        segments = dict(segment_ids_by_mora)
        if expansion is not None:
            for added, source in expansion.added_mora_sources:
                if source not in segments:
                    raise ValueError(
                        "segment IDs must cover the source repeated vocalization mora"
                    )
                segments[added] = segments[source]
    pitched = replace(
        document,
        note_candidates=tuple(note for note in document.note_candidates
                              if note.midi_pitch is not None),
    )
    note_run = optimize_note_runs(
        pitched, settings, segment_ids_by_mora=segments,
        line_windows_by_utterance=line_windows_by_utterance,
        utterance_groups=utterance_groups,
    )
    linked, correspondences = _materialize_note_run(document, note_run, settings)
    realization = compile_realization(linked)
    return Stage3Run(linked, realization, correspondences, note_run)
