import tempfile
import unittest
from pathlib import Path

from soramimic_score import (AlignedMora, AudioAdapters, LyricLine, MelodyNote,
                             ReadingSelection, analyze_audio, lyric_surface)
from soramimic_score.japanese import kana_to_moras
from soramimic_score.supplied_lyrics import plan_supplied_lyrics, locate_supplied_groups
from soramimic_score.vocal_activity import VocalActivity


class SuppliedLyricsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "input.wav"
        self.path.touch()
        self.aligned_text = []

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def readings(_path, lines):
        return tuple(ReadingSelection(line.text.replace("\n", ""), "synthetic", 1)
                     for line in lines)

    def align(self, _path, lines, readings):
        self.aligned_text.extend(line.text for line in lines)
        output = []
        for index, (line, reading) in enumerate(zip(lines, readings, strict=True)):
            moras = kana_to_moras(reading.kana)
            step = (line.end_sec - line.start_sec) / len(moras)
            output.extend(AlignedMora(index, i, kana, line.start_sec + i * step,
                                       line.start_sec + (i + 1) * step, .9)
                          for i, kana in enumerate(moras))
        return tuple(output)

    def adapters(self, recognized, notes=(), *, duration=6., vocal=None):
        return AudioAdapters(
            self.readings, self.align, lambda _: notes, lambda _: recognized,
            lyric_reading=lambda text: text, audio_duration=lambda _: duration,
            vocal_activity=vocal, dictionary_reading_selector=self.readings,
        )

    def test_wrong_recognition_never_replaces_supplied_ctc_targets(self):
        supplied = ["カキ", "サシサシサシ", "タチ"]
        heard = (LyricLine("カキ", 0, 1), LyricLine("ウア", 1, 5), LyricLine("タチ", 5, 6))
        result = analyze_audio(self.path, self.adapters(
            heard, (MelodyNote(0, 6, 60),)), lyrics=supplied)
        self.assertEqual(result.score.canonical_text, "\n".join(supplied))
        self.assertEqual(self.aligned_text, supplied)
        self.assertNotIn("ウア", [u.surface for u in result.observations.utterances])
        self.assertEqual(lyric_surface(result)["unused_supplied_indices"], [])

    def test_missing_intro_middle_and_ending_use_ordered_acoustic_windows(self):
        supplied = ["アア", "カキ", "サシ", "タチ", "ナニ"]
        heard = (LyricLine("カキ", 1, 2), LyricLine("タチ", 4, 5))
        result = analyze_audio(self.path, self.adapters(
            heard, (MelodyNote(0, 6, 60),)), lyrics=supplied)
        groups = lyric_surface(result)["groups"]
        self.assertEqual([(g["start_sec"], g["end_sec"]) for g in groups],
                         [(0, 1), (1, 2), (2, 4), (4, 5), (5, 6)])
        self.assertEqual(self.aligned_text, supplied)

    def test_touching_anchors_align_missing_text_jointly(self):
        plan = plan_supplied_lyrics(["カキ", "サシ", "タチ"],
                                   [LyricLine("カキ", 0, 2), LyricLine("タチ", 2, 4)])
        groups = locate_supplied_groups(plan, [LyricLine("カキ", 0, 2),
                                               LyricLine("タチ", 2, 4)], 4)
        self.assertEqual([g["display_text"] for g in groups], ["カキ", "サシ\nタチ"])
        self.assertEqual([(g["start_sec"], g["end_sec"]) for g in groups], [(0, 2), (2, 4)])

    def test_empty_recognition_still_aligns_supplied_lyrics_to_notes(self):
        result = analyze_audio(self.path, self.adapters(
            (), (MelodyNote(0, 6, 60),)), lyrics=["カキ", "サシ"])
        self.assertEqual(result.score.canonical_text, "カキ\nサシ")
        self.assertEqual(self.aligned_text, ["カキ\nサシ"])

    def test_only_joint_absence_of_support_leaves_input_unobserved(self):
        def silent(_path, windows):
            return tuple(VocalActivity(-100, -90, 0, False) for _ in windows)
        result = analyze_audio(self.path, self.adapters(
            (LyricLine("カキ", 0, 2),), (MelodyNote(0, 2, 60),), vocal=silent),
            lyrics=["カキ", "サシ"])
        self.assertEqual(result.score.canonical_text, "カキ\nサシ")
        self.assertEqual(self.aligned_text, ["カキ"])
        missing = lyric_surface(result)["groups"][-1]
        self.assertEqual(missing["alignment_status"], "no-performance-evidence")
        self.assertEqual(missing["support"],
                         {"melody": False, "recognition": False, "vocal_activity": False})
        self.assertTrue(result.score.unresolved_unit_ids)

    def test_each_support_source_prevents_rejecting_input(self):
        for source in ["melody", "recognition", "vocal"]:
            with self.subTest(source=source):
                self.aligned_text.clear()
                def activity(_path, windows):
                    return tuple(VocalActivity(-20, 0, 1, source == "vocal") for _ in windows)
                notes = (MelodyNote(0, 6, 60),) if source == "melody" else ()
                heard = (LyricLine("ウア", 0, 6),) if source == "recognition" else ()
                result = analyze_audio(self.path, self.adapters(heard, notes, vocal=activity),
                                       lyrics=["カキ"])
                self.assertEqual(result.score.canonical_text, "カキ")
                self.assertEqual(self.aligned_text, ["カキ"])
                self.assertEqual(lyric_surface(result)["unobserved_supplied_indices"], [])

    def test_missing_activity_adapter_is_not_silence_evidence(self):
        result = analyze_audio(self.path, self.adapters(()), lyrics=["カキ"])
        self.assertEqual(self.aligned_text, ["カキ"])
        self.assertIsNone(lyric_surface(result)["groups"][0]["support"]["vocal_activity"])

    def test_completion_only_adds_in_unambiguous_input_gaps(self):
        heard = (LyricLine("カキ", 0, 1), LyricLine("サシ", 1, 2), LyricLine("タチ", 2, 3))
        for enabled, expected in [(False, "カキ\nタチ"), (True, "カキ\nサシ\nタチ")]:
            with self.subTest(enabled=enabled):
                result = analyze_audio(self.path, self.adapters(
                    heard, (MelodyNote(0, 3, 60),)), lyrics=["カキ", "タチ"],
                    adjust_lyrics=enabled)
                self.assertEqual(result.score.canonical_text, expected)
                self.assertEqual(lyric_surface(result)["unused_supplied_indices"], [])

    def test_completion_does_not_append_conflicting_recognition_to_input(self):
        plan = plan_supplied_lyrics(["カキ", "サシ", "タチ"],
                                   [LyricLine("カキ", 0, 1), LyricLine("ウア", 1, 2),
                                    LyricLine("タチ", 2, 3)], add_missing=True)
        self.assertEqual(plan["display_text"], "カキ\nサシ\nタチ")
        self.assertEqual(plan["unused_supplied_indices"], [])

    def test_unsupported_addition_does_not_change_supplied_text(self):
        heard = (LyricLine("カキ", 0, 1), LyricLine("サシ", 1, 2), LyricLine("タチ", 2, 3))
        def activity(_path, windows):
            return tuple(VocalActivity(-100, -90, 0, False) for _ in windows)
        result = analyze_audio(self.path, self.adapters(
            heard, (MelodyNote(0, 1, 60), MelodyNote(2, 3, 62)), vocal=activity),
            lyrics=["カキ", "タチ"], adjust_lyrics=True)
        self.assertEqual(result.score.canonical_text, "カキ\nタチ")
        self.assertEqual(len(lyric_surface(result)["rejected_additions"]), 1)


if __name__ == "__main__":
    unittest.main()
