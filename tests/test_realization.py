from dataclasses import replace
import unittest

from soramimic_score import (
    Boundary, Evidence, Link, LinkCost, LyricSpan, NoteCandidate,
    ObservedSingingUnit, ReadingCandidate, build_known_lyrics_document,
)
from soramimic_score.realization import (
    PerformanceOmission, Realization, compile_realization, split_stack_interval,
)


def document(kana="カキク", observed=3):
    evidence = (Evidence("ctc", "synthetic", "token-alignment", 0.9),)
    bounds = lambda t: Boundary(t, 0.9, ("ctc",))
    observations = tuple(ObservedSingingUnit(
        (m,), bounds(i * 0.3), None, bounds((i + 1) * 0.3), 0.9, ("ctc",),
    ) for i, m in enumerate(kana[:observed]))
    return build_known_lyrics_document(
        kana, (LyricSpan(kana, (0, len(kana)), (ReadingCandidate(kana, "synthetic", 1.0),)),),
        observations, evidence,
    )


def linked(doc, links, notes=None):
    return replace(doc, note_candidates=tuple(notes or (
        NoteCandidate("n0", 0, 0.9, 60, 0.8, ("synthetic",)),
    )), links=tuple(Link(f"l{i}", tuple(units), tuple(notes), operation, 0, 0.8, LinkCost())
                    for i, (operation, units, notes) in enumerate(links)))


class RealizationTests(unittest.TestCase):
    def test_time_order_alone_does_not_make_reversed_lyrics_valid(self):
        doc = linked(document(), [
            ("match", ("s0",), ("n0",)), ("match", ("s1",), ("n1",)),
            ("match", ("s2",), ("n2",)),
        ], [NoteCandidate("n0", 0, .3, 60, .8, ("synthetic",)),
            NoteCandidate("n1", .3, .6, 62, .8, ("synthetic",)),
            NoteCandidate("n2", .6, .9, 64, .8, ("synthetic",))])
        score = compile_realization(doc)
        first, middle, last = score.synthesis_plan
        invalid = replace(score, synthesis_plan=(
            replace(last, start_sec=0, end_sec=.3), middle,
            replace(first, start_sec=.6, end_sec=.9),
        ))
        with self.assertRaisesRegex(ValueError, "canonical lyric order"):
            invalid.validate()
        with self.assertRaisesRegex(ValueError, "canonical lyric order"):
            Realization.from_dict(invalid.to_dict())

    def test_multi_syllable_stack_is_rejected(self):
        doc = linked(document(), [("stack", ("s0", "s1", "s2"), ("n0",))])
        before = doc.to_json()
        with self.assertRaisesRegex(ValueError, "more than one syllable"):
            compile_realization(doc)
        self.assertEqual(before, doc.to_json())

    def test_missing_vowel_boundary_is_not_fabricated(self):
        self.assertTrue(all(x.start is None and x.end is None
                            for x in document().vowel_nuclei))

    def test_unobserved_unit_is_not_an_omission(self):
        doc = linked(document(observed=2), [
            ("match", ("s0",), ("n0",)), ("match", ("s1",), ("n1",)),
            ("unit_only", ("s2",), ()),
        ], [NoteCandidate("n0", 0, .3, 60, .8, ("synthetic",)),
            NoteCandidate("n1", .3, .6, 62, .8, ("synthetic",))])
        result = compile_realization(doc)
        self.assertEqual(result.performed[-1].status, "unobserved")
        self.assertEqual(result.canonical[0].kana, "カキク")
        self.assertEqual("".join(s.kana for s in result.synthesis_plan), "カキ")
        self.assertEqual(result.unresolved_unit_ids, ("s2",))
        self.assertFalse(result.omissions)

    def test_explicit_performance_omission_preserves_canonical(self):
        doc = linked(document(), [
            ("match", ("s0",), ("n0",)), ("match", ("s1",), ("n1",)),
            ("match", ("s2",), ("n2",)),
        ], [NoteCandidate("n0", 0, .3, 60, .8, ("synthetic",)),
            NoteCandidate("n1", .3, .6, 62, .8, ("synthetic",)),
            NoteCandidate("n2", .6, .9, 64, .8, ("synthetic",))])
        doc = replace(doc, evidence=doc.evidence + (
            Evidence("omission", "synthetic-review", "performance-omission", 1),
        ))
        result = compile_realization(doc, omissions=(PerformanceOmission(
            "s1", "synthetic explicit omission", ("omission",)),))
        self.assertEqual(result.canonical_text, "カキク")
        self.assertEqual(result.canonical[0].kana, "カキク")
        self.assertEqual("".join(s.kana for s in result.synthesis_plan), "カク")

    def test_alignment_evidence_does_not_authorize_omission(self):
        with self.assertRaisesRegex(ValueError, "performance-omission"):
            compile_realization(document(), omissions=(PerformanceOmission(
                "s0", "no match", ("ctc",)),))

    def test_no_notes_leaves_every_unit_explicitly_unresolved(self):
        result = compile_realization(document())
        self.assertEqual(result.unresolved_unit_ids, ("s0", "s1", "s2"))
        self.assertFalse(result.synthesis_plan)
        self.assertEqual(result.canonical[0].kana, "カキク")

    def test_melisma_marks_continuation_not_repeated_consonant(self):
        doc = linked(document("カ", 1), [("melisma", ("s0",), ("n0", "n1"))], [
            NoteCandidate("n0", 0, 0.4, 60, 0.8, ("synthetic",)),
            NoteCandidate("n1", 0.4, 0.9, 62, 0.8, ("synthetic",)),
        ])
        result = compile_realization(doc)
        self.assertEqual([s.kana for s in result.synthesis_plan], ["カ", "ー"])
        self.assertTrue(result.synthesis_plan[1].continuation)

    def test_phrase_final_vowel_extends_to_its_observed_release(self):
        doc = linked(document("カ", 1), [("match", ("s0",), ("n0",))], [
            NoteCandidate("n0", 0, 0.1, 60, 0.8, ("synthetic",)),
        ])
        result = compile_realization(doc)
        self.assertEqual(result.synthesis_plan[0].end_sec, .3)
        self.assertEqual(result.synthesis_plan[0].timing_source, "observed_utterance_tail")
        self.assertIn("observed_utterance_tails_extended:1", result.diagnostics)

    def test_near_zero_phrase_final_boundary_cannot_create_a_long_tail(self):
        doc = document("カ", 1)
        unit = replace(doc.singing_units[0], end=Boundary(99.0, 1e-6, ("ctc",)))
        doc = linked(replace(doc, singing_units=(unit,)),
                     [("match", ("s0",), ("n0",))], [
                         NoteCandidate("n0", 0, 0.1, 60, 0.8, ("synthetic",)),
                     ])
        result = compile_realization(doc)
        self.assertEqual(result.synthesis_plan[0].end_sec, .1)
        self.assertNotIn("observed_utterance_tails_extended:1", result.diagnostics)

    def test_nonfinal_unit_does_not_extend_into_the_following_mora(self):
        doc = document("カキ", 2)
        units = list(doc.singing_units)
        units[0] = replace(units[0], end=Boundary(.8, .9, ("ctc",)))
        doc = replace(doc, singing_units=tuple(units))
        doc = linked(doc, [
            ("match", ("s0",), ("n0",)), ("match", ("s1",), ("n1",)),
        ], [NoteCandidate("n0", 0, .1, 60, .8, ("synthetic",)),
            NoteCandidate("n1", .2, .5, 62, .8, ("synthetic",))])
        result = compile_realization(doc)
        self.assertEqual(result.synthesis_plan[0].end_sec, .1)

    def test_sokuon_follows_preceding_melisma_on_only_its_final_note(self):
        doc = linked(document("カッ", 2), [("melisma", ("s0",), ("n0", "n1"))], [
            NoteCandidate("n0", 0, 0.4, 60, 0.8, ("synthetic",)),
            NoteCandidate("n1", 0.4, 0.9, 62, 0.8, ("synthetic",)),
        ])
        result = compile_realization(doc)
        self.assertEqual([s.kana for s in result.synthesis_plan], ["カッ", "ー"])
        self.assertEqual([s.note_candidate_id for s in result.synthesis_plan],
                         ["n0", "n1"])
        self.assertEqual([s.operation for s in result.synthesis_plan],
                         ["melisma", "melisma"])

    def test_micro_note_partition_never_drops_or_extends(self):
        units = document(observed=0).singing_units
        spans = split_stack_interval(1.0, 1.015, units)
        self.assertEqual(len(spans), 3)
        self.assertTrue(all(hi > lo for lo, hi, _ in spans))
        self.assertEqual(spans[0][0], 1.0)
        self.assertEqual(spans[-1][1], 1.015)

    def test_duplicate_unit_consumption_is_rejected(self):
        doc = linked(document("カ"), [
            ("match", ("s0",), ("n0",)), ("unit_only", ("s0",), ()),
        ])
        with self.assertRaisesRegex(ValueError, "unit twice"):
            compile_realization(doc)

    def test_unmatched_unit_cannot_borrow_from_another_utterance(self):
        doc = build_known_lyrics_document("カ\nキ", (
            LyricSpan("カ", (0, 1), (ReadingCandidate("カ", "synthetic", 1),)),
            LyricSpan("キ", (2, 3), (ReadingCandidate("キ", "synthetic", 1),)),
        ))
        doc = linked(doc, [("match", ("s0",), ("n0",)), ("unit_only", ("s1",), ())])
        self.assertEqual(compile_realization(doc).unresolved_unit_ids, ("s1",))

    def test_unmatched_after_melisma_remains_unresolved(self):
        doc = linked(document("カキ", 2), [
            ("melisma", ("s0",), ("n0", "n1")), ("unit_only", ("s1",), ()),
        ], [NoteCandidate("n0", 0, 0.4, 60, 0.8, ("synthetic",)),
            NoteCandidate("n1", 0.4, 0.9, 62, 0.8, ("synthetic",))])
        result = compile_realization(doc)
        self.assertEqual([s.kana for s in result.synthesis_plan], ["カ", "ー"])
        self.assertEqual(result.unresolved_unit_ids, ("s1",))

    def _missing_run(self, starts):
        doc = document("カ" + "キ" * len(starts) + "ク", len(starts) + 2)
        units = list(doc.singing_units)
        for i, start in enumerate(starts, 1):
            units[i] = replace(units[i],
                               consonant_start=Boundary(start, .9, ("ctc",)) if start is not None else None,
                               end=Boundary(start + .1, .9, ("ctc",)) if start is not None else None,
                               status="observed" if start is not None else "unobserved",
                               confidence=.9 if start is not None else 0,
                               evidence_ids=("ctc",) if start is not None else ())
        doc = replace(doc, singing_units=tuple(units))
        links = [("match", ("s0",), ("n0",))]
        links.extend(("unit_only", (f"s{i}",), ()) for i in range(1, len(starts) + 1))
        links.append(("match", (f"s{len(starts) + 1}",), ("n1",)))
        return linked(doc, links, [
            NoteCandidate("n0", 0, .3, 60, .8, ("synthetic",)),
            NoteCandidate("n1", .58, 1.8, 62, .8, ("synthetic",)),
        ])

    def test_unmatched_observed_near_following_note_remains_unresolved(self):
        doc = self._missing_run((.59,))
        before = doc.to_json()
        result = compile_realization(doc)
        self.assertEqual(result.unresolved_unit_ids, ("s1",))
        self.assertEqual(doc.to_json(), before)
        self.assertEqual([s.singing_unit_id for s in result.synthesis_plan], ["s0", "s2"])

    def test_unmatched_run_does_not_overfill_neighbor_notes(self):
        doc = self._missing_run((.1, .2, .7, .8))
        result = compile_realization(doc)
        self.assertEqual([s.note_candidate_id for s in result.synthesis_plan],
                         ["n0", "n1"])
        self.assertEqual([s.singing_unit_id for s in result.synthesis_plan],
                         ["s0", "s5"])
        self.assertEqual(result.unresolved_unit_ids, ("s1", "s2", "s3", "s4"))

    def test_conflicting_measurements_leave_missing_run_unresolved(self):
        result = compile_realization(self._missing_run((.7, .1)))
        self.assertEqual([s.singing_unit_id for s in result.synthesis_plan],
                         ["s0", "s3"])
        self.assertEqual(result.unresolved_unit_ids, ("s1", "s2"))

    def test_missing_or_zero_confidence_timing_does_not_overfill_notes(self):
        for start in (None, .7):
            doc = self._missing_run((start,))
            if start is not None:
                units = list(doc.singing_units)
                units[1] = replace(units[1], consonant_start=Boundary(start, 0, ("ctc",)))
                doc = replace(doc, singing_units=tuple(units))
            result = compile_realization(doc)
            self.assertEqual(result.unresolved_unit_ids, ("s1",))

    def test_unmatched_repair_does_not_promote_note_only_pitch(self):
        doc = self._missing_run((.4,))
        unused = NoteCandidate("unlinked", .35, .55, 84, .99, ("sheetsage2-vocal",))
        doc = replace(doc, note_candidates=(doc.note_candidates[0], unused, doc.note_candidates[1]),
                      links=doc.links + (Link("unused-link", (), ("unlinked",), "note_only", 0, .9, LinkCost()),))
        result = compile_realization(doc)
        self.assertNotIn("unlinked", {s.note_candidate_id for s in result.synthesis_plan})

    def test_unresolved_unit_retains_separate_boundary_evidence(self):
        doc = self._missing_run((.59,))
        timing = Evidence("onset", "synthetic-aligner", "consonant-boundary", .9)
        units = list(doc.singing_units)
        units[1] = replace(units[1], consonant_start=Boundary(.59, .9, ("onset",)))
        doc = replace(doc, evidence=doc.evidence + (timing,), singing_units=tuple(units))
        result = compile_realization(doc)
        self.assertEqual(result.unresolved_unit_ids, ("s1",))
        self.assertIn("onset", result.performed[1].evidence_ids)

    def test_unmatched_before_melisma_remains_unresolved(self):
        doc = linked(document("カキ", 2), [
            ("unit_only", ("s0",), ()), ("melisma", ("s1",), ("n0", "n1")),
        ], [NoteCandidate("n0", .3, .6, 60, .8, ("synthetic",)),
            NoteCandidate("n1", .6, .9, 62, .8, ("synthetic",))])
        result = compile_realization(doc)
        self.assertEqual([s.kana for s in result.synthesis_plan], ["キ", "ー"])
        self.assertEqual(result.unresolved_unit_ids, ("s0",))

    def test_missing_performed_membership_cannot_hide_canonical_deletion(self):
        doc = linked(document(), [
            ("match", ("s0",), ("n0",)), ("match", ("s1",), ("n1",)),
            ("match", ("s2",), ("n2",)),
        ], [NoteCandidate("n0", 0, .3, 60, .8, ("synthetic",)),
            NoteCandidate("n1", .3, .6, 62, .8, ("synthetic",)),
            NoteCandidate("n2", .6, .9, 64, .8, ("synthetic",))])
        data = compile_realization(doc).to_dict()
        data["performed"] = tuple(x for x in data["performed"] if x["singing_unit_id"] != "s1")
        data["synthesis_plan"] = tuple(x for x in data["synthesis_plan"]
                                        if x["singing_unit_id"] != "s1")
        with self.assertRaisesRegex(ValueError, "every canonical mora"):
            Realization.from_dict(data)

if __name__ == "__main__":
    unittest.main()
