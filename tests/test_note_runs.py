from dataclasses import replace
import math
import unittest

from soramimic_score.alignment import ObservedSingingUnit, build_known_lyrics_document
from soramimic_score.ir import Boundary, Evidence, NoteCandidate
from soramimic_score.japanese import LyricSpan, ReadingCandidate
from soramimic_score.note_runs import NoteRunConfig, optimize_note_runs


def document(kana, onsets, notes):
    boundary = lambda value: Boundary(value, 0.9, ("alignment",))
    observed = tuple(
        ObservedSingingUnit(
            (mora,), boundary(max(0, onset - 0.02)), boundary(onset),
            boundary(onset + 0.12), 0.9, ("alignment", f"ctc-{index}"),
        )
        for index, (mora, onset) in enumerate(zip(kana, onsets, strict=True))
    )
    evidence = (Evidence("alignment", "test", "phoneme-alignment", 0.9),) + tuple(
        Evidence(f"ctc-{index}", "test-ctc", "mora-ctc-anchor", 0.1,
                 {"time_sec": onset})
        for index, onset in enumerate(onsets)
    )
    result = build_known_lyrics_document(
        kana,
        (LyricSpan(kana, (0, len(kana)), (ReadingCandidate(kana, "test", 1),)),),
        observed,
        evidence,
    )
    candidates = tuple(NoteCandidate(
        identifier, start, end, pitch, 0.5, ("sheetsage2-vocal",), (),
    ) for identifier, start, end, pitch in notes)
    return replace(result, note_candidates=candidates)


def multiline_document(lines, onsets, notes, confidences=None):
    kana = "".join(lines)
    confidences = confidences or (0.1,) * len(onsets)
    boundary = lambda value: Boundary(value, 0.9, ("alignment",))
    observed = tuple(
        ObservedSingingUnit(
            (mora,), boundary(max(0, onset - 0.02)), boundary(onset),
            boundary(onset + 0.12), 0.9, ("alignment", f"ctc-{index}"),
        )
        for index, (mora, onset) in enumerate(zip(kana, onsets, strict=True))
    )
    evidence = (Evidence("alignment", "test", "phoneme-alignment", 0.9),) + tuple(
        Evidence(f"ctc-{index}", "test-ctc", "mora-ctc-anchor", confidence,
                 {"time_sec": onset})
        for index, (onset, confidence) in enumerate(
            zip(onsets, confidences, strict=True))
    )
    spans = []
    offset = 0
    for line in lines:
        spans.append(LyricSpan(
            line, (offset, offset + len(line)),
            (ReadingCandidate(line, "test", 1),),
        ))
        offset += len(line)
    result = build_known_lyrics_document(kana, tuple(spans), observed, evidence)
    candidates = tuple(NoteCandidate(
        identifier, start, end, pitch, 0.5, ("sheetsage2-vocal",), (),
    ) for identifier, start, end, pitch in notes)
    return replace(result, note_candidates=candidates)


def note(identifier, start, end, pitch=60):
    return identifier, start, end, pitch


def preserving_config(**changes):
    values = dict(
        ctc_onset_weight=1,
        ctc_onset_scale_sec=1,
        pitch_change_weight=0.3,
        source_note_split_weight=0.3,
        mora_omission_weight=20,
        internal_note_only_open_weight=10,
        edge_note_only_open_weight=10,
        note_only_duration_weight=10,
        within_syllable_rest_open_weight=1,
        within_syllable_rest_duration_weight=1,
        segment_rest_open_weight=1,
        segment_rest_duration_weight=1,
    )
    values.update(changes)
    return NoteRunConfig(**values)


class NoteRunOptimizationTests(unittest.TestCase):
    def test_attached_pitch_change_is_not_treated_as_detached_phrase_edge(self):
        source = document("カ", (0.05,), (
            note("body", 0, .3), note("tail", .3, .5, 65),
            note("detached", 1.5, 1.7, 67),
        ))
        result = optimize_note_runs(source)
        self.assertEqual([n.source_note_ids for n in result.assignments[0].notes],
                         [("body",), ("tail",)])
        self.assertEqual(result.omitted_source_note_ids, ("detached",))
        self.assertFalse(result.omitted_mora_ids)

    def test_ctc_posterior_floor_does_not_pull_lyrics_to_early_notes(self):
        source = multiline_document(("カキク",), (.05, .31, .34), (
            note("first", 0, .3), note("middle", .3, .6),
            note("last", .6, .9, 65),
        ), confidences=(.1, .1, 1e-8))
        result = optimize_note_runs(source)
        self.assertEqual([a.kana for a in result.assignments], ["カ", "キ", "ク"])
        self.assertEqual(result.assignments[-1].notes[0].source_note_ids, ("last",))
        self.assertEqual(result.assignments[-1].cost.ctc_onset_distance, 0)
        self.assertFalse(result.omitted_mora_ids)
        self.assertFalse(result.omitted_source_note_ids)
        self.assertEqual(source.evidence[-1].confidence, 1e-8)

    def test_whisper_ownership_prevents_fully_external_note_steal(self):
        source = multiline_document(
            ("カ", "キ"), (0.98, 1.20),
            (note("inside-first", 0.80, 0.95),
             note("external", 1.02, 1.12, 65),
             note("next-line", 1.20, 1.90)),
            confidences=(0.00004, 0.8),
        )
        result = optimize_note_runs(
            source,
            preserving_config(
                ctc_onset_weight=1,
                whisper_boundary_cost_per_sec2=100,
                pitch_change_weight=20,
            ),
            line_windows_by_utterance={"u0": (0.0, 1.0), "u1": (1.01, 2.0)},
        )
        self.assertEqual([item.kana for item in result.assignments], ["カ", "キ"])
        self.assertEqual(
            [item.notes[0].source_note_ids for item in result.assignments],
            [("inside-first",), ("next-line",)],
        )
        self.assertIn(
            ("external",),
            [item.source_note_ids for item in result.note_only_spans],
        )
        self.assertEqual(result.cost.whisper_line_ownership, 0)

    def test_note_starting_inside_line_keeps_tail_without_ownership_cost(self):
        source = multiline_document(
            ("カ", "キ"), (0.80, 1.30),
            (note("long-tail", 0.80, 1.20), note("next-line", 1.30, 1.60)),
        )
        result = optimize_note_runs(
            source,
            preserving_config(
                whisper_boundary_cost_per_sec2=10_000,
            ),
            line_windows_by_utterance={"u0": (0.0, 1.0), "u1": (1.10, 2.0)},
        )
        self.assertEqual(result.assignments[0].notes[0].source_note_ids, ("long-tail",))
        self.assertEqual(result.assignments[0].notes[0].end_sec, 1.20)
        self.assertEqual(result.assignments[0].cost.whisper_line_ownership, 0)

    def test_ctc_crossing_support_is_not_gated_by_low_confidence(self):
        notes = (note("external", 1.10, 1.30),)
        inside = multiline_document(("カ",), (0.95,), notes, confidences=(0.9,))
        crossing = multiline_document(("カ",), (1.10,), notes, confidences=(0.0,))
        config = preserving_config(
            whisper_boundary_cost_per_sec2=100,
        )
        windows = {"u0": (0.0, 1.0)}
        inside_result = optimize_note_runs(
            inside, config, line_windows_by_utterance=windows,
        )
        crossing_result = optimize_note_runs(
            crossing, config, line_windows_by_utterance=windows,
        )
        self.assertGreater(inside_result.cost.whisper_line_ownership, 0)
        self.assertEqual(crossing_result.cost.whisper_line_ownership, 0)

    def test_line_window_validation_tolerates_serialized_boundary_roundoff(self):
        source = multiline_document(
            ("カ", "キ"), (0.5, 1.5),
            (note("first", 0.5, 0.9), note("second", 1.5, 1.9)),
        )
        result = optimize_note_runs(
            source,
            preserving_config(whisper_boundary_cost_per_sec2=100),
            line_windows_by_utterance={
                "u0": (0.0, 1.00000000000002),
                "u1": (1.0, 2.0),
            },
        )
        self.assertEqual([item.kana for item in result.assignments], ["カ", "キ"])

    def test_whisper_boundary_cost_uses_one_per_second_squared_coefficient(self):
        source = multiline_document(
            ("カ",), (0.95,), (note("external", 1.10, 1.30),),
        )

        result = optimize_note_runs(
            source,
            preserving_config(whisper_boundary_cost_per_sec2=2),
            line_windows_by_utterance={"u0": (0.0, 1.0)},
        )

        # 0.10 seconds beyond the boundary plus 0.20 seconds outside it.
        self.assertAlmostEqual(result.cost.whisper_line_ownership, 2 * 0.30**2)

    def test_same_pitch_fragments_coalesce_inside_one_syllable(self):
        result = optimize_note_runs(
            document("カ", (0.0,), (
                note("n0", 0.0, 0.2),
                note("n1", 0.2, 0.5),
                note("n2", 0.5, 0.8),
            )),
            preserving_config(),
        )
        self.assertEqual(len(result.assignments), 1)
        self.assertEqual(len(result.assignments[0].notes), 1)
        self.assertEqual(
            result.assignments[0].notes[0].source_note_ids,
            ("n0", "n1", "n2"),
        )
        self.assertEqual(result.assignments[0].notes[0].end_sec, 0.8)

    def test_ctc_boundary_keeps_same_pitch_fragments_as_two_syllables(self):
        result = optimize_note_runs(
            document("カキ", (0.0, 0.4), (
                note("n0", 0.0, 0.4), note("n1", 0.4, 0.8),
            )),
            preserving_config(),
        )
        self.assertEqual([item.kana for item in result.assignments], ["カ", "キ"])
        self.assertEqual(
            [item.notes[0].source_note_ids for item in result.assignments],
            [("n0",), ("n1",)],
        )

    def test_pitch_changes_remain_separate_continuation_notes(self):
        result = optimize_note_runs(
            document("カ", (0.0,), (
                note("n0", 0.0, 0.3, 60), note("n1", 0.3, 0.7, 62),
            )),
            preserving_config(),
        )
        assignment = result.assignments[0]
        self.assertEqual([item.midi_pitch for item in assignment.notes], [60, 62])
        self.assertEqual([item.continuation for item in assignment.notes], [False, True])
        self.assertAlmostEqual(result.cost.pitch_change, 0.3)

    def test_small_gap_does_not_merge_when_above_threshold(self):
        result = optimize_note_runs(
            document("カ", (0.0,), (
                note("n0", 0.0, 0.2), note("n1", 0.23, 0.5),
            )),
            preserving_config(same_pitch_merge_gap_sec=0.02),
        )
        self.assertEqual(len(result.assignments[0].notes), 2)

    def test_one_source_note_can_split_at_second_ctc(self):
        result = optimize_note_runs(
            document("カキ", (0.1, 0.5), (note("shared", 0.0, 1.0),)),
            preserving_config(ctc_onset_weight=0),
        )
        self.assertEqual(len(result.assignments), 2)
        self.assertTrue(result.assignments[1].source_note_split)
        self.assertAlmostEqual(result.assignments[0].notes[-1].end_sec, 0.5)
        self.assertAlmostEqual(result.assignments[1].notes[0].start_sec, 0.5)

    def test_coincident_ctc_run_uses_next_distinct_onset_as_its_bound(self):
        result = optimize_note_runs(
            document(
                "カキクケ", (0.1, 0.5, 0.5, 0.8),
                (note("shared", 0.0, 1.0),),
            ),
            preserving_config(ctc_onset_weight=0),
        )

        self.assertEqual(
            [(item.notes[0].start_sec, item.notes[0].end_sec)
             for item in result.assignments],
            [(0.0, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.0)],
        )
        self.assertEqual(
            [item.timing_adjustment for item in result.assignments],
            [None, "coincident_ctc_proportional",
             "coincident_ctc_proportional", None],
        )

    def test_three_coincident_ctc_splits_share_the_remaining_note(self):
        result = optimize_note_runs(
            document(
                "カキクケ", (0.1, 0.5, 0.5, 0.5),
                (note("shared", 0.0, 1.0),),
            ),
            preserving_config(ctc_onset_weight=0),
        )

        intervals = [
            (item.notes[0].start_sec, item.notes[0].end_sec)
            for item in result.assignments
        ]
        self.assertEqual(intervals[0], (0.0, 0.5))
        self.assertTrue(all(
            math.isclose(end - start, 1 / 6)
            for start, end in intervals[1:]
        ))

    def test_internal_note_only_has_larger_opening_cost_than_edge(self):
        source = document("カキ", (0.0, 1.0), (
            note("n0", 0.0, 0.2), note("middle", 0.4, 0.6, 65),
            note("n1", 1.0, 1.2), note("tail", 1.4, 1.6, 67),
        ))
        result = optimize_note_runs(source, preserving_config(
            ctc_onset_weight=100,
            internal_note_only_open_weight=2,
            edge_note_only_open_weight=0.1,
            note_only_duration_weight=0,
            pitch_change_weight=20,
        ))
        spans = {span.source_note_ids: span for span in result.note_only_spans}
        self.assertEqual(spans[("middle",)].position, "internal")
        self.assertEqual(spans[("tail",)].position, "trailing")
        self.assertEqual(spans[("middle",)].cost.note_only_open, 2)
        self.assertEqual(spans[("tail",)].cost.note_only_open, 0.1)

    def test_segment_rest_penalty_only_applies_inside_same_segment(self):
        source = document("カキ", (0.0, 1.0), (
            note("n0", 0.0, 0.2), note("n1", 1.0, 1.2),
        ))
        mora_ids = [item.id for item in source.moras]
        same = optimize_note_runs(
            source, preserving_config(),
            segment_ids_by_mora={mora_id: "segment" for mora_id in mora_ids},
        )
        separate = optimize_note_runs(
            source, preserving_config(),
            segment_ids_by_mora={mora_id: f"segment-{index}"
                                 for index, mora_id in enumerate(mora_ids)},
        )
        self.assertGreater(same.cost.segment_rest, 0)
        self.assertEqual(separate.cost.segment_rest, 0)

    def test_usable_confidences_and_singing_intervals_do_not_rescale_cost(self):
        original = document("カキ", (0.1, 0.4), (
            note("n0", 0.1, 0.3), note("n1", 0.4, 0.6),
        ))
        changed = replace(
            original,
            evidence=tuple(replace(item, confidence=1 - item.confidence)
                           for item in original.evidence),
            singing_units=tuple(replace(
                unit, confidence=0,
                consonant_start=Boundary(50, 0, ("alignment",)),
                end=Boundary(51, 0, ("alignment",)),
            ) for unit in original.singing_units),
            note_candidates=tuple(replace(item, confidence=0.99)
                                  for item in original.note_candidates),
        )
        self.assertEqual(
            optimize_note_runs(original, preserving_config()),
            optimize_note_runs(changed, preserving_config()),
        )

    def test_raw_ctc_can_assign_part_of_a_multi_mora_lexical_syllable(self):
        source = document("カン", (0.0, 1.0), (note("n0", 0.0, 0.2),))
        result = optimize_note_runs(source, preserving_config(
            ctc_onset_weight=1,
            mora_omission_weight=0.1,
            edge_note_only_open_weight=0,
            note_only_duration_weight=0,
        ))
        unit_moras = set(source.singing_units[0].mora_ids)
        assigned = {mora_id for item in result.assignments for mora_id in item.mora_ids}
        omitted = set(result.omitted_mora_ids)
        self.assertEqual(len(assigned), 1)
        self.assertEqual(len(omitted), 1)
        self.assertEqual(assigned | omitted, unit_moras)

    def test_nearby_phrase_tail_without_a_note_remains_omitted(self):
        result = optimize_note_runs(
            document("カス", (0.0, 0.38), (note("n0", 0.0, 0.2, 64),)),
            preserving_config(),
        )
        self.assertEqual([item.kana for item in result.assignments], ["カ"])
        self.assertEqual(len(result.omitted_mora_ids), 1)


if __name__ == "__main__":
    unittest.main()
