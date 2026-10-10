import io
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from tests.test_correspondence import _document, _note
from tests.test_realization import document
from soramimic_score import (
    Boundary,
    Evidence,
    LyricSpan,
    NoteRunConfig,
    ObservedSingingUnit,
    ReadingCandidate,
    VocalizationReattack,
    build_known_lyrics_document,
    load,
)
from soramimic_score.pipeline import run_stage3_document
from soramimic_score.realization import compile_realization


def with_ctc(document, onsets):
    anchors = tuple(
        Evidence(f"raw-ctc-{index}", "synthetic-ctc", "mora-ctc-anchor", .1,
                 {"time_sec": onset})
        for index, onset in enumerate(onsets)
    )
    return replace(
        document,
        evidence=document.evidence + anchors,
        singing_units=tuple(
            replace(unit, evidence_ids=unit.evidence_ids + (anchor.id,))
            for unit, anchor in zip(document.singing_units, anchors, strict=True)
        ),
    )


def reattacks(*onsets):
    return {
        "u0": tuple(
            VocalizationReattack(onset, onset + .04, .8, "synthetic-raw-ctc")
            for onset in onsets
        )
    }


class PipelineTests(unittest.TestCase):
    def preserving_config(self):
        return NoteRunConfig(
            ctc_onset_weight=1,
            ctc_onset_scale_sec=1,
            pitch_change_weight=.3,
            source_note_split_weight=.3,
            mora_omission_weight=20,
            note_only_open_weight=10,
            note_only_duration_weight=10,
            within_syllable_rest_open_weight=1,
            within_syllable_rest_duration_weight=1,
            segment_rest_open_weight=1,
            segment_rest_duration_weight=1,
        )

    def test_fresh_observation_runner_rejects_reused_links(self):
        doc = with_ctc(document("カ", 1), (.0,))
        decoded = run_stage3_document(replace(doc, note_candidates=(_note("n", 0, .3),)))
        with self.assertRaisesRegex(ValueError, "previously decoded"):
            run_stage3_document(decoded.document)

    def test_utterance_partition_consumes_notes_once_and_compiles_line_joins_together(self):
        evidence = (Evidence("ctc", "synthetic", "phoneme-alignment", .9),)
        boundary = lambda t: Boundary(t, .9, ("ctc",))
        doc = build_known_lyrics_document("カ\nキ", (
            LyricSpan("カ", (0, 1), (ReadingCandidate("カ", "synthetic", 1),)),
            LyricSpan("キ", (2, 3), (ReadingCandidate("キ", "synthetic", 1),)),
        ), (
            ObservedSingingUnit(("カ",), boundary(0), boundary(.1), boundary(.4), .9, ("ctc",)),
            ObservedSingingUnit(("キ",), boundary(.5), boundary(.6), boundary(.9), .9, ("ctc",)),
        ), evidence)
        doc = with_ctc(doc, (.1, .6))
        doc = replace(doc, note_candidates=(_note("n0", .1, .7), _note("n1", .6, .9)))
        result = run_stage3_document(doc)
        self.assertEqual([s.kana for s in result.realization.synthesis_plan], ["カ", "キ"])
        self.assertEqual(result.realization.synthesis_plan[0].end_sec, .6)
        self.assertEqual(
            [note_id for item in result.note_run.assignments for note in item.notes
             for note_id in note.source_note_ids],
            ["n0", "n1"],
        )
        self.assertEqual(len({link.id for link in result.document.links}), len(result.document.links))
        by_id = {note.id: note for note in result.document.note_candidates}
        self.assertEqual(tuple(by_id[note.id] for note in doc.note_candidates), doc.note_candidates)

    def test_note_owner_uses_the_midpoint_between_utterance_ctc_anchors(self):
        evidence = (Evidence("ctc", "synthetic", "phoneme-alignment", .9),)
        boundary = lambda t: Boundary(t, .9, ("ctc",))
        doc = build_known_lyrics_document("カ\nキ", (
            LyricSpan("カ", (0, 1), (ReadingCandidate("カ", "synthetic", 1),)),
            LyricSpan("キ", (2, 3), (ReadingCandidate("キ", "synthetic", 1),)),
        ), (
            ObservedSingingUnit(("カ",), boundary(0), boundary(.1), boundary(.4), .9, ("ctc",)),
            ObservedSingingUnit(("キ",), boundary(.6), boundary(.7), boundary(.9), .9, ("ctc",)),
        ), evidence)
        doc = with_ctc(doc, (.1, .65))
        doc = replace(doc, note_candidates=(
            _note("straddling", .35, .65), _note("next", .65, .9),
        ))
        result = run_stage3_document(doc)
        owner_by_source = {
            source_id: assignment.singing_unit_ids
            for assignment in result.note_run.assignments
            for note in assignment.notes for source_id in note.source_note_ids
        }
        self.assertEqual(owner_by_source["straddling"], ("s0",))
        self.assertEqual(owner_by_source["next"], ("s1",))

    def test_unknown_timing_remains_unresolved_and_unowned_notes_are_retained(self):
        doc = document("カ", 0)
        doc = replace(doc, note_candidates=(_note("n", 0, .3),))
        run = run_stage3_document(doc)
        self.assertEqual(run.realization.unresolved_unit_ids, ("s0",))
        self.assertEqual({link.operation for link in run.document.links}, {"unit_only", "note_only"})
        self.assertEqual(len({link.id for link in run.document.links}), 2)

    def test_no_input_mutation_and_unknown_scores_are_not_replaced(self):
        doc = _document(((0, .1, .4),), (_note("n", .1, .4, confidence=0, evidence_ids=("unknown",)),),
                        extra_evidence=(Evidence("unknown", "synthetic", "model-note", 0,
                                                 {"confidence_available": False}),))
        doc = with_ctc(doc, (.1,))
        before = doc.to_json()
        run = run_stage3_document(doc)
        self.assertEqual(before, doc.to_json())
        by_id = {note.id: note for note in run.document.note_candidates}
        self.assertEqual(tuple(by_id[note.id] for note in doc.note_candidates), doc.note_candidates)
        self.assertIsNone(run.realization.synthesis_plan[0].pitch_confidence)

    def test_cli_uses_the_same_fresh_observation_runner(self):
        from soramimic_score.__main__ import main
        doc = with_ctc(_document(((0, .1, .4),), (_note("n", .1, .4),)), (.1,))
        expected = run_stage3_document(doc)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "observations.json", root / "score.json"
            source.write_text(doc.to_json())
            with patch("sys.argv", ["soramimic-score", "--input", str(source),
                                    "--output", str(output)]), redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            actual = load(output)
            self.assertEqual(actual.observations, expected.document)
            self.assertEqual(actual.score, expected.realization)

    def test_main_runner_coalesces_same_pitch_fragments_and_replays_linked_ir(self):
        doc = with_ctc(document("カ", 1), (.0,))
        doc = replace(doc, note_candidates=(
            _note("n0", 0, .2), _note("n1", .2, .5), _note("n2", .5, .8),
        ))
        run = run_stage3_document(doc, config=self.preserving_config())
        self.assertEqual(len(run.realization.synthesis_plan), 1)
        slot = run.realization.synthesis_plan[0]
        self.assertEqual((slot.start_sec, slot.end_sec, slot.kana), (0, .8, "カ"))
        derived = next(note for note in run.document.note_candidates
                       if note.id == slot.note_candidate_id)
        detail = next(item.detail for item in run.document.evidence
                      if item.id in derived.evidence_ids
                      and item.kind == "note-run-derivation")
        self.assertEqual(detail["source_note_ids"], ["n0", "n1", "n2"])
        self.assertEqual(compile_realization(run.document), run.realization)

    def test_main_runner_keeps_ctc_separated_same_pitch_syllables(self):
        doc = with_ctc(document("カキ", 2), (0, .4))
        doc = replace(doc, note_candidates=(_note("n0", 0, .4), _note("n1", .4, .8)))
        run = run_stage3_document(doc, config=self.preserving_config())
        self.assertEqual(
            [(slot.kana, slot.midi_pitch, slot.continuation)
             for slot in run.realization.synthesis_plan],
            [("カ", 60, False), ("キ", 60, False)],
        )
        self.assertNotEqual(
            run.realization.synthesis_plan[0].note_candidate_id,
            run.realization.synthesis_plan[1].note_candidate_id,
        )

    def test_main_runner_keeps_pitch_changes_as_continuation_notes(self):
        doc = with_ctc(document("カ", 1), (0,))
        doc = replace(doc, note_candidates=(_note("n0", 0, .3, 60),
                                             _note("n1", .3, .7, 62)))
        run = run_stage3_document(doc, config=self.preserving_config())
        self.assertEqual(
            [(slot.kana, slot.midi_pitch, slot.continuation)
             for slot in run.realization.synthesis_plan],
            [("カ", 60, False), ("ー", 62, True)],
        )

    def test_acoustic_reattacks_expand_rounded_whisper_vocalization(self):
        doc = with_ctc(document("ラララ", 3), (0, .4, .8))
        doc = replace(doc, note_candidates=(
            _note("n0", 0, .12, 60),
            _note("n1", .12, .39, 62),
            _note("n2", .39, .66, 64),
            _note("n3", .66, .93, 62),
            _note("n4", .93, 1.20, 65),
        ))
        before = doc.to_json()

        run = run_stage3_document(
            doc,
            config=self.preserving_config(),
            line_windows_by_utterance={"u0": (0, 1.20)},
            vocalization_reattacks_by_utterance=reattacks(0, .12, .39, .66, .93),
        )

        self.assertEqual(before, doc.to_json())
        self.assertEqual(
            [(slot.kana, slot.midi_pitch, slot.continuation)
             for slot in run.realization.synthesis_plan],
            [("ラ", 60, False), ("ラ", 62, False), ("ラ", 64, False),
             ("ラ", 62, False), ("ラ", 65, False)],
        )
        selected = next(
            reading for reading in run.document.readings
            if reading.id == run.document.utterances[0].selected_reading_id
        )
        self.assertEqual(selected.kana, "ラララララ")
        decision = next(
            item for item in run.document.evidence
            if item.kind == "repeated-vocalization-expansion"
        )
        self.assertEqual(decision.detail["recognized_repetitions"], 3)
        self.assertEqual(decision.detail["expanded_repetitions"], 5)
        self.assertEqual(
            decision.detail["required_note_candidate_ids"],
            ["n0", "n1", "n2", "n3", "n4"],
        )
        self.assertEqual(compile_realization(run.document), run.realization)

    def test_acoustic_reattacks_are_opt_in(self):
        doc = with_ctc(document("ラララ", 3), (0, .4, .8))
        doc = replace(doc, note_candidates=tuple(
            _note(f"n{index}", index * .2, (index + 1) * .2, 60 + index)
            for index in range(5)
        ))
        run = run_stage3_document(
            doc,
            config=self.preserving_config(),
            line_windows_by_utterance={"u0": (0, 1)},
        )
        selected = next(
            reading for reading in run.document.readings
            if reading.id == run.document.utterances[0].selected_reading_id
        )
        self.assertEqual(selected.kana, "ラララ")
        self.assertFalse(any(
            item.kind == "repeated-vocalization-expansion"
            for item in run.document.evidence
        ))

    def test_acoustic_repetition_uses_whisper_mora_and_rejects_mixed_line(self):
        notes = tuple(
            _note(f"n{index}", index * .2, (index + 1) * .2, 60 + index)
            for index in range(4)
        )
        for kana in ("カキ", "ナ"):
            with self.subTest(kana=kana):
                doc = with_ctc(
                    document(kana, len(kana)),
                    tuple(index * .2 for index in range(len(kana))),
                )
                doc = replace(doc, note_candidates=notes)
                run = run_stage3_document(
                    doc,
                    config=self.preserving_config(),
                    line_windows_by_utterance={"u0": (0, .8)},
                    vocalization_reattacks_by_utterance=reattacks(0, .2, .4, .6),
                )
                selected = next(
                    reading for reading in run.document.readings
                    if reading.id == run.document.utterances[0].selected_reading_id
                )
                self.assertEqual(
                    selected.kana,
                    kana if kana == "カキ" else "ナナナナ",
                )

    def test_acoustic_count_has_no_fixed_cap_and_preserves_segment(self):
        doc = with_ctc(document("ララ", 2), (0, .2))
        doc = replace(doc, note_candidates=tuple(
            _note(f"n{index}", index * .1, (index + 1) * .1, 60 + index)
            for index in range(18)
        ))
        source_segments = {mora.id: "rounded-la" for mora in doc.moras}
        run = run_stage3_document(
            doc,
            config=self.preserving_config(),
            segment_ids_by_mora=source_segments,
            line_windows_by_utterance={"u0": (0, 1.8)},
            vocalization_reattacks_by_utterance=reattacks(
                *(index * .1 for index in range(18))
            ),
        )
        selected = next(
            reading for reading in run.document.readings
            if reading.id == run.document.utterances[0].selected_reading_id
        )
        self.assertEqual(selected.kana, "ラ" * 18)
        self.assertEqual(
            [slot.midi_pitch for slot in run.realization.synthesis_plan
             if not slot.continuation],
            list(range(60, 78)),
        )
        self.assertAlmostEqual(run.realization.synthesis_plan[-1].end_sec, 1.8)
        decision = next(
            item for item in run.document.evidence
            if item.kind == "repeated-vocalization-expansion"
        )
        self.assertEqual(decision.detail["expanded_repetitions"], 18)
        self.assertEqual(
            {assignment.segment_id for assignment in run.note_run.assignments},
            {"rounded-la"},
        )

    def test_required_sheetsage_notes_become_continuations_not_note_only(self):
        doc = with_ctc(document("ララ", 2), (0, .5))
        doc = replace(doc, note_candidates=(
            _note("n0", 0, .2, 60),
            _note("n1", .2, .4, 62),
            _note("n2", .5, .7, 64),
            _note("n3", .7, .9, 65),
        ))
        run = run_stage3_document(
            doc,
            config=NoteRunConfig(
                mora_omission_weight=.01,
                note_only_open_weight=0,
                note_only_duration_weight=0,
            ),
            line_windows_by_utterance={"u0": (0, .9)},
            vocalization_reattacks_by_utterance=reattacks(0, .5),
        )
        self.assertEqual(run.note_run.omitted_mora_ids, ())
        self.assertEqual(run.note_run.note_only_spans, ())
        self.assertEqual(
            [(slot.kana, slot.midi_pitch) for slot in run.realization.synthesis_plan],
            [("ラ", 60), ("ー", 62), ("ラ", 64), ("ー", 65)],
        )
        self.assertAlmostEqual(run.realization.synthesis_plan[-1].end_sec, .9)

    def test_main_runner_materializes_mora_ctc_split_units_for_replay(self):
        doc = document("カン", 2)
        anchors = (
            Evidence("raw-ctc-0", "synthetic-ctc", "mora-ctc-anchor", .1,
                     {"time_sec": 0.0}),
            Evidence("raw-ctc-1", "synthetic-ctc", "mora-ctc-anchor", .1,
                     {"time_sec": 0.4}),
        )
        self.assertEqual(len(doc.singing_units), 1)
        doc = replace(
            doc,
            evidence=doc.evidence + anchors,
            singing_units=(replace(
                doc.singing_units[0],
                evidence_ids=doc.singing_units[0].evidence_ids
                + tuple(item.id for item in anchors),
            ),),
            note_candidates=(_note("n0", 0, .4), _note("n1", .4, .8)),
        )
        run = run_stage3_document(doc, config=self.preserving_config())
        self.assertEqual(
            [(slot.kana, slot.continuation) for slot in run.realization.synthesis_plan],
            [("カ", False), ("ン", False)],
        )
        self.assertEqual(len(run.document.singing_units), 2)
        self.assertEqual(len(doc.singing_units), 1)
        self.assertEqual(compile_realization(run.document), run.realization)

    def test_main_runner_redistributes_duplicate_ctc_splits(self):
        doc = with_ctc(document("カキク", 3), (0.1, 0.5, 0.5))
        doc = replace(doc, note_candidates=(_note("shared", 0, 1),))

        run = run_stage3_document(doc, config=self.preserving_config())

        intervals = [
            (assignment.notes[0].start_sec, assignment.notes[0].end_sec)
            for assignment in run.note_run.assignments
        ]
        self.assertEqual(len(intervals), 3)
        self.assertEqual(intervals, [(0, 0.5), (0.5, 0.75), (0.75, 1)])
        self.assertEqual(
            [assignment.timing_adjustment for assignment in run.note_run.assignments],
            [None, "coincident_ctc_proportional", "coincident_ctc_proportional"],
        )
        derivations = [
            evidence for evidence in run.document.evidence
            if evidence.kind == "note-run-derivation"
        ]
        self.assertEqual(
            [item.detail["timing_adjustment"] for item in derivations],
            [None, "coincident_ctc_proportional", "coincident_ctc_proportional"],
        )
        self.assertEqual(compile_realization(run.document), run.realization)

    def test_note_run_unit_only_is_not_repaired_back_into_a_note(self):
        doc = document("カン", 2)
        anchors = (
            Evidence("raw-ctc-0", "synthetic-ctc", "mora-ctc-anchor", .1,
                     {"time_sec": 0.0}),
            Evidence("raw-ctc-1", "synthetic-ctc", "mora-ctc-anchor", .1,
                     {"time_sec": 1.0}),
        )
        doc = replace(
            doc,
            evidence=doc.evidence + anchors,
            singing_units=(replace(
                doc.singing_units[0],
                evidence_ids=doc.singing_units[0].evidence_ids
                + tuple(item.id for item in anchors),
            ),),
            note_candidates=(_note("n0", 0, .2),),
        )
        config = replace(
            self.preserving_config(),
            mora_omission_weight=.1,
            note_only_open_weight=0,
            note_only_duration_weight=0,
        )
        run = run_stage3_document(doc, config=config)
        self.assertEqual([slot.kana for slot in run.realization.synthesis_plan], ["カ"])
        self.assertEqual(len(run.realization.unresolved_unit_ids), 1)
        self.assertEqual(compile_realization(run.document), run.realization)


if __name__ == "__main__":
    unittest.main()
