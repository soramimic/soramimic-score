from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from soramimic_score import (AlignedMora, AudioAdapters, LyricLine, MelodyNote,
                             ReadingSelection, analyze_audio, lyric_surface)
from soramimic_score.supplied_alignment import verify_supplied_alignment
from soramimic_score.vocal_activity import VocalActivity
from soramimic_score.japanese import kana_to_moras


class SuppliedAlignmentTests(unittest.TestCase):
    def test_weak_child_of_joint_alignment_keeps_observations_without_speech_permission(self):
        lines = (LyricLine("カキ", 0, 3),)
        readings = (ReadingSelection("カキ", "test", 1),)
        moras = (AlignedMora(0, 0, "カ", 1, 1.1, 1e-8),
                 AlignedMora(0, 1, "キ", 1.3, 1.4, 1e-8))
        adapters = AudioAdapters(lambda *_: readings, lambda *_: moras, lambda _: ())
        overlay = {"groups": [{"operation": "retain_supplied", "supplied_indices": [0],
                                "shared_acoustic_window": True}]}
        result = verify_supplied_alignment(Path("input.wav"), lines, readings, moras,
                                           overlay, (), adapters)
        self.assertEqual(result[2], moras)
        self.assertEqual(result[3], ())
        self.assertFalse(overlay["groups"][0]["ctc_verified"])

    def test_local_search_compares_closed_reading_alternatives(self):
        lines = (LyricLine("11", 0, 10),)
        readings = (ReadingSelection("ジュウイチ", "test", 1, ("ジュウイチ", "イチイチ")),)
        def align(_path, ls, rs):
            start = 8 if ls[0].start_sec == 8 else 1
            return tuple(AlignedMora(0, i, m, start + i * .1, start + i * .1 + .05, .01)
                         for i, m in enumerate(kana_to_moras(rs[0].kana)))
        initial = align(None, lines, readings)
        adapters = AudioAdapters(lambda *_: readings, align, lambda _: (),
            supplied_mora_aligner=align, lyric_reading=lambda text: text,
            repetition_evidence=lambda _p, ws: tuple("イチイチ" if lo >= 7.5 else "カキ"
                                                     for lo, _hi in ws))
        overlay = {"groups": [{"operation": "retain_supplied", "supplied_indices": [0],
                                "asr_indices": [0, 1]}],
                   "recognized_lines": [{"text": "カキ", "start_sec": 0, "end_sec": 5},
                                        {"text": "イチイチ", "start_sec": 8, "end_sec": 9}]}
        result = verify_supplied_alignment(Path("input.wav"), lines, readings, initial,
                                           overlay, (), adapters)
        self.assertEqual(result[1][0].kana, "イチイチ")
        self.assertEqual(result[0][0].text, "11")
        self.assertEqual(result[0][0].start_sec, 8)
        self.assertEqual(result[1][0].candidates, readings[0].candidates)

    def test_local_phrase_agreement_outweighs_a_higher_scoring_wrong_interval(self):
        lines = (LyricLine("カキ", 0, 10),)
        readings = (ReadingSelection("カキ", "test", 1),)
        wrong = (AlignedMora(0, 0, "カ", 1, 1.1, .9),
                 AlignedMora(0, 1, "キ", 1.3, 1.4, .9))
        right = (AlignedMora(0, 0, "カ", 8, 8.1, .0001),
                 AlignedMora(0, 1, "キ", 8.3, 8.4, .0001))
        adapters = AudioAdapters(lambda *_: readings, lambda *_: wrong, lambda _: (),
            supplied_mora_aligner=lambda _p, ls, _rs: right if ls[0].start_sec == 8 else wrong,
            repetition_evidence=lambda _p, ws: tuple("カキ" if lo >= 7.5 else "サシス"
                                                     for lo, _hi in ws))
        overlay = {"groups": [{"operation": "retain_supplied", "supplied_indices": [0],
                                "asr_indices": [0, 1]}],
                   "recognized_lines": [{"start_sec": 0, "end_sec": 5},
                                        {"start_sec": 8, "end_sec": 9}]}
        result = verify_supplied_alignment(Path("input.wav"), lines, readings, wrong,
                                           overlay, (), adapters)
        self.assertEqual(result[2], right)
        self.assertEqual(result[0][0].start_sec, 8)
        self.assertTrue(overlay["groups"][0]["ctc_verified"])

    def test_independent_phrase_agreement_can_verify_low_ctc_posteriors(self):
        lines = (LyricLine("カキ", 0, 3),)
        readings = (ReadingSelection("カキ", "test", 1),)
        moras = (AlignedMora(0, 0, "カ", 1, 1.1, .0001),
                 AlignedMora(0, 1, "キ", 1.3, 1.4, .0001))
        adapters = AudioAdapters(lambda *_: readings, lambda *_: moras, lambda _: (),
                                 repetition_evidence=lambda *_: ("カキ",))
        overlay = {"groups": [{"operation": "retain_supplied", "supplied_indices": [0]}]}
        result = verify_supplied_alignment(Path("input.wav"), lines, readings, moras,
                                           overlay, (), adapters)
        self.assertEqual(result[3], ())
        self.assertEqual(result[2], moras)
        self.assertTrue(overlay["groups"][0]["ctc_verified"])

    def test_confident_forced_ctc_cannot_override_independent_phrase_disagreement(self):
        lines = (LyricLine("カキ", 0, 3),)
        readings = (ReadingSelection("カキ", "test", 1),)
        moras = (AlignedMora(0, 0, "カ", 1, 1.1, .9),
                 AlignedMora(0, 1, "キ", 1.3, 1.4, .9))
        adapters = AudioAdapters(lambda *_: readings, lambda *_: moras, lambda _: (),
                                 repetition_evidence=lambda *_: ("サシスセソ",))
        overlay = {"groups": [{"operation": "retain_supplied", "supplied_indices": [0]}]}
        result = verify_supplied_alignment(Path("input.wav"), lines, readings, moras,
                                           overlay, (), adapters)
        self.assertEqual(result[3], (0,))
        self.assertFalse(result[2])
        self.assertFalse(overlay["groups"][0]["ctc_verified"])

    def test_unmatched_weak_phrase_stays_canonical_without_proportional_speech(self):
        def readings(_path, lines):
            return tuple(ReadingSelection(l.text, "test", 1) for l in lines)

        adapters = AudioAdapters(readings,
            lambda *_: (AlignedMora(0, 0, "カ", 0, .02, 1e-8),
                        AlignedMora(0, 1, "キ", 10, 10.02, .001)),
            lambda _: (), lambda _: (LyricLine("サシ", 0, 20),),
            audio_duration=lambda _: 20,
            vocal_activity=lambda _p, windows: tuple(
                VocalActivity(-10, 0, 1, True) for _ in windows))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            path.touch()
            for adjust in (False, True):
                with self.subTest(adjust=adjust):
                    score = analyze_audio(path, adapters, lyrics=("カキ",), adjust_lyrics=adjust)
                    self.assertEqual(score.score.canonical_text, "カキ")
                    self.assertFalse(score.score.synthesis_plan)
                    self.assertEqual(len(score.score.unresolved_unit_ids), 2)
                    group = lyric_surface(score)["groups"][0]
                    self.assertEqual(group["alignment_status"], "ctc-unresolved")
                    self.assertEqual(len(group["ctc_alignment"]["initial_moras"]), 2)

    def test_locator_skips_unrelated_audio_and_preserves_audit(self):
        lines = (LyricLine("カキ", 0, 20),)
        readings = (ReadingSelection("カキ", "test", 1),)
        weak = (AlignedMora(0, 0, "カ", 0, .02, 1e-8),
                AlignedMora(0, 1, "キ", 10, 10.02, .001))
        good = (AlignedMora(0, 0, "カ", 15, 15.1, .9),
                AlignedMora(0, 1, "キ", 15.3, 15.4, .8))
        adapters = AudioAdapters(lambda *_: readings, lambda *_: weak, lambda _: (),
                                 supplied_mora_aligner=lambda *_: good)
        overlay = {"groups": [{"operation": "retain_supplied", "supplied_indices": [0]}]}
        result = verify_supplied_alignment(Path("input.wav"), lines, readings, weak,
                                           overlay, (), adapters)
        self.assertEqual(result[0], (replace(lines[0], start_sec=15, end_sec=15.4),))
        self.assertEqual(result[2], good)
        self.assertEqual(result[3], ())
        self.assertTrue(overlay["groups"][0]["ctc_verified"])
        self.assertEqual(overlay["groups"][0]["ctc_alignment"]["initial_moras"][0]["start_sec"], 0)

    def test_verified_unpitched_phrase_uses_observed_ctc_not_asr_proportions(self):
        readings = lambda _p, ls: tuple(ReadingSelection(l.text, "test", 1) for l in ls)
        adapters = AudioAdapters(readings,
            lambda *_: (AlignedMora(0, 0, "カ", 2, 2.1, .9),
                        AlignedMora(0, 1, "キ", 2.3, 2.4, .9)),
            lambda _: (), lambda _: (LyricLine("カキ", 0, 10),),
            audio_duration=lambda _: 10,
            vocal_activity=lambda _p, ws: tuple(VocalActivity(-10, 0, 1, True) for _ in ws))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            path.touch()
            score = analyze_audio(path, adapters, lyrics=("カキ",))
            self.assertEqual([s.start_sec for s in score.score.synthesis_plan], [2, 2.3])
            self.assertEqual(score.score.synthesis_plan[-1].end_sec, 2.4)
            self.assertTrue(all(s.timing_source == "spoken_onset_interval"
                                for s in score.score.synthesis_plan))
            self.assertTrue(all(s.pitch_confidence is None for s in score.score.synthesis_plan))


if __name__ == "__main__":
    unittest.main()
