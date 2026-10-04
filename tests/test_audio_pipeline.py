import tempfile
import importlib.util
import unittest
from pathlib import Path

from soramimic_score import (
    AlignedMora,
    AudioAdapters,
    AudioPipelineError,
    LyricLine,
    MelodyNote,
    ReadingSelection,
    analyze_audio,
    build_audio_observations,
    lyric_surface,
)
from soramimic_score.audio import CTCWindowCapacityError, is_credit_hallucination
from soramimic_score.semantic import (contextual_non_lyric_template_families,
                                      credit_recovery_windows,
                                      non_lyric_template_family)
from soramimic_score.vocal_activity import VocalActivity


class AudioPipelineTests(unittest.TestCase):
    def test_automatic_ctc_capacity_drops_only_the_unalignable_line(self):
        def readings(_path, lines):
            return tuple(ReadingSelection(line.text, "test", 1) for line in lines)

        def align(_path, lines, selected):
            for index, line in enumerate(lines):
                if line.text == "ア":
                    raise CTCWindowCapacityError(index, 1, 4)
            return tuple(AlignedMora(0, index, char, 1 + index * .2,
                                     1 + (index + 1) * .2, .9)
                         for index, char in enumerate(selected[0].kana))

        adapters = AudioAdapters(readings, align,
                                 lambda _: (MelodyNote(1, 1.4, 60),),
                                 lambda _: (LyricLine("ア", 0, .1),
                                            LyricLine("カキ", 1, 1.4)))
        score = analyze_audio(self.audio, adapters)
        self.assertEqual(score.score.canonical_text, "カキ")
        self.assertTrue(any(item.detail.get("reason")
                            == "ctc-window-capacity-insufficient"
                            for item in score.observations.evidence))
        with self.assertRaises(CTCWindowCapacityError):
            analyze_audio(self.audio, adapters, lyrics=("ア", "カキ"))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.audio = Path(self.temporary.name) / "input.wav"
        self.audio.write_bytes(b"test adapter boundary")

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _readings(_path, lines):
        table = {"空": "ソラ", "耳": "ミミ"}
        return tuple(ReadingSelection(table[line.text], "test-reading", .9) for line in lines)

    @staticmethod
    def _moras(_path, _lines, readings):
        result = []
        time = 0.0
        for line_index, reading in enumerate(readings):
            for mora_index, kana in enumerate(reading.kana):
                result.append(AlignedMora(
                    line_index, mora_index, kana, time, time + .2, .8,
                    "test-ctc", time + .03,
                ))
                time += .2
        return tuple(result)

    @staticmethod
    def _melody(_path):
        return tuple(
            MelodyNote(index * .2, (index + 1) * .2, 60 + index, "test-melody", .7)
            for index in range(4)
        )

    def test_automatic_lyrics_run_all_four_adapters_to_score(self):
        calls = []

        def recognize(path):
            calls.append(("lyrics", path))
            return (LyricLine("空", 0, .4, .85), LyricLine("耳", .4, .8, .8))

        def readings(path, lines):
            calls.append(("readings", path, tuple(line.text for line in lines)))
            return self._readings(path, lines)

        def align(path, lines, selected):
            calls.append(("moras", path, tuple(item.kana for item in selected)))
            return self._moras(path, lines, selected)

        def melody(path):
            calls.append(("melody", path))
            return self._melody(path)

        score = analyze_audio(
            self.audio,
            AudioAdapters(readings, align, melody, recognize),
        )

        self.assertEqual(score.score.canonical_text, "空\n耳")
        self.assertEqual([item.surface for item in score.observations.utterances], ["空", "耳"])
        self.assertEqual(
            [item.kana for item in score.observations.readings], ["ソラ", "ミミ"],
        )
        self.assertTrue(all(
            any(note.id == f"audio-note-{index}"
                for note in score.observations.note_candidates)
            for index in range(4)
        ))
        self.assertEqual([item[0] for item in calls], ["lyrics", "melody", "readings", "moras"])
        self.assertTrue(any(
            item.kind == "mora-ctc-anchor" and item.source == "test-ctc"
            for item in score.observations.evidence
        ))

    def test_progress_reports_the_active_analysis_stage(self):
        stages = []
        adapters = AudioAdapters(self._readings, self._moras, self._melody,
                                 lambda _: (LyricLine("空", 0, .4),))
        analyze_audio(self.audio, adapters, on_progress=stages.append)
        self.assertEqual(stages, ["歌詞を認識しています", "音符と音高を推定しています",
                                  "歌詞の読みを確認しています", "モーラの時刻を推定しています",
                                  "楽譜データを組み立てています"])

    def test_credit_like_whisper_line_is_excluded_before_alignment(self):
        self.assertTrue(is_credit_hallucination("作詞・作曲・編曲 初音ミク"))
        self.assertTrue(is_credit_hallucination("字幕制作: Example"))
        self.assertFalse(is_credit_hallucination("作曲家になりたい"))
        aligned = []

        def align(_path, lines, readings):
            aligned.extend(line.text for line in lines)
            return self._moras(_path, lines, readings)

        adapters = AudioAdapters(self._readings, align, self._melody,
                                 lambda _: (LyricLine("作詞・作曲・編曲 初音ミク", 0, .3),
                                            LyricLine("空", .4, .8)))
        score = analyze_audio(self.audio, adapters)
        self.assertEqual(aligned, ["空"])
        self.assertEqual(score.score.canonical_text, "空")

    def test_contextual_credit_block_matches_video_gate(self):
        lines = (LyricLine("歌 ABC Studio", 0, .2),
                 LyricLine("映像 ABC Studio", .2, .4),
                 LyricLine("空", .4, .8),
                 LyricLine("歌 ABC Studio", 1, 1.2))
        self.assertEqual(contextual_non_lyric_template_families(lines),
                         ("credits", "credits", None, None))
        score = analyze_audio(self.audio, AudioAdapters(
            self._readings, self._moras, self._melody, lambda _: lines[:3]))
        self.assertEqual(score.score.canonical_text, "空")

    @unittest.skipUnless(importlib.util.find_spec("MeCab") and
                         importlib.util.find_spec("unidic_lite"), "Japanese entity tagger unavailable")
    def test_japanese_creator_name_is_a_contextual_credit(self):
        lines = (LyricLine("歌 初音ミク", 0, .2), LyricLine("映像 初音ミク", .2, .4))
        self.assertEqual(contextual_non_lyric_template_families(lines),
                         ("credits", "credits"))

    def test_soft_template_requires_its_own_ctc_support(self):
        self.assertEqual(non_lyric_template_family("お疲れさま"), "closing-greeting")
        lines = (LyricLine("お疲れさま", 0, .4), LyricLine("空", .4, .8))
        passes = []
        low_score = True

        def readings(_path, chosen):
            return tuple(ReadingSelection("オツカレサマ" if line.text == "お疲れさま" else "ソラ",
                                          "test", 1) for line in chosen)

        def align(_path, chosen, selected):
            passes.append(tuple(line.text for line in chosen))
            return tuple(AlignedMora(index, offset, kana,
                                     line.start_sec + offset * .03,
                                     line.start_sec + (offset + 1) * .03,
                                     .0001 if low_score and line.text == "お疲れさま" else .8)
                         for index, (line, reading) in enumerate(zip(chosen, selected))
                         for offset, kana in enumerate(reading.kana))

        score = analyze_audio(self.audio, AudioAdapters(
            readings, align, self._melody, lambda _: lines))
        self.assertEqual(passes, [("お疲れさま", "空"), ("空",)])
        self.assertEqual(score.score.canonical_text, "空")
        low_score = False
        supported = analyze_audio(self.audio, AudioAdapters(
            readings, align, self._melody, lambda _: lines))
        self.assertEqual(supported.score.canonical_text, "お疲れさま\n空")

    def test_silent_vocal_stem_rejects_unresolved_whisper_line(self):
        lines = (LyricLine("空", 0, .4), LyricLine("何もない", 1, 2))
        activity = (VocalActivity(-15, 0, 1, True),
                    VocalActivity(-90, -75, 0, False))
        score = analyze_audio(self.audio, AudioAdapters(
            self._readings, self._moras, self._melody, lambda _: lines,
            vocal_activity=lambda _path, windows: activity))
        self.assertEqual(score.score.canonical_text, "空")
        self.assertTrue(any(item.kind == "lyric-semantic-gate"
                            and item.detail.get("vocal_activity_relative_db") == -75
                            for item in score.observations.evidence))

    def test_credit_gate_recovers_singing_island_by_short_whisper_retry(self):
        self.assertFalse(is_credit_hallucination("作詞・作曲・君へ歌う"))
        self.assertEqual(credit_recovery_windows(
            LyricLine("作詞・作曲・編曲 初音ミク", 0, 5),
            (MelodyNote(0, .2, 60), MelodyNote(1.3, 2, 60),
             MelodyNote(2.1, 3, 61))), ((1.3, 3),))
        recovered = []
        def retry(_path, start, end):
            recovered.append((start, end))
            return (LyricLine("空", start, start + .4),)
        def notes(_path):
            return (MelodyNote(0, 1, 60), MelodyNote(1, 2, 62),
                    MelodyNote(2.2, 2.6, 64))
        adapters = AudioAdapters(self._readings, self._moras, notes,
                                 lambda _: (LyricLine("作詞・作曲・編曲 初音ミク", 0, 2),
                                            LyricLine("耳", 2.2, 2.6)),
                                 lyric_recoverer=retry)
        score = analyze_audio(self.audio, adapters)
        self.assertEqual(recovered, [(0, 2)])
        self.assertEqual(score.score.canonical_text, "空\n耳")
        self.assertTrue(any(item.kind == "lyric-semantic-gate"
                            and item.detail["status"] == "recovered"
                            for item in score.observations.evidence))

    def test_unpronounceable_credit_retry_does_not_fail_the_song(self):
        def notes(_path):
            return (MelodyNote(0, 2, 60), MelodyNote(2.2, 2.6, 62))

        def reading(text):
            if text == "未知記号":
                raise ValueError("no reading")
            return "ミミ"

        score = analyze_audio(self.audio, AudioAdapters(
            self._readings, self._moras, notes,
            lambda _: (LyricLine("作詞 初音ミク", 0, 2), LyricLine("耳", 2.2, 2.6)),
            lyric_reading=reading,
            lyric_recoverer=lambda _path, start, end:
                (LyricLine("未知記号", start, start + .4),),
        ))
        self.assertEqual(score.score.canonical_text, "耳")

    def test_weak_ctc_rejects_credit_retry_before_final_score(self):
        def notes(_path):
            return (MelodyNote(0, 2, 60), MelodyNote(2.2, 2.6, 62))

        passes = []

        def align(_path, lines, readings):
            passes.append(tuple(line.text for line in lines))
            return tuple(AlignedMora(index, offset, kana,
                                     line.start_sec + offset * .05,
                                     line.start_sec + (offset + 1) * .05,
                                     .0001 if line.text == "空" else .8)
                         for index, (line, reading) in enumerate(zip(lines, readings))
                         for offset, kana in enumerate(reading.kana))

        score = analyze_audio(self.audio, AudioAdapters(
            self._readings, align, notes,
            lambda _: (LyricLine("作詞 初音ミク", 0, 2), LyricLine("耳", 2.2, 2.6)),
            lyric_reading=lambda text: "ソラ" if text == "空" else "ミミ",
            lyric_recoverer=lambda _path, start, end:
                (LyricLine("空", start, start + .4),),
        ))
        self.assertEqual(score.score.canonical_text, "耳")
        self.assertEqual(passes, [("空", "耳"), ("耳",)])

    def test_supplied_credit_text_remains_authoritative(self):
        credit = "作詞・作曲・編曲 初音ミク"
        def reading(_path, lines):
            return tuple(ReadingSelection("サクシ", "test-reading", .9) for _ in lines)
        adapters = AudioAdapters(reading, self._moras, self._melody,
                                 lambda _: (LyricLine(credit, 0, .8),))
        score = analyze_audio(self.audio, adapters, lyrics=(credit,))
        self.assertEqual(score.score.canonical_text, credit)

    def test_known_lyrics_run_recognition_and_preserve_acoustic_result(self):
        def reject(_path):
            return (LyricLine("空", 0, .4), LyricLine("耳", .4, .8))

        score = analyze_audio(
            self.audio,
            AudioAdapters(self._readings, self._moras, self._melody, reject),
            lyrics=("空", "耳", "遠い星"),
        )
        self.assertEqual(score.score.canonical_text, "空\n耳")
        self.assertEqual(lyric_surface(score)["supplied_lines"], ["空", "耳", "遠い星"])
        self.assertEqual(lyric_surface(score)["unused_supplied_indices"], [2])

    def test_observation_builder_marks_uncalibrated_note_confidence(self):
        document = build_audio_observations(
            (LyricLine("空", 0, .4),),
            (ReadingSelection("ソラ", "test-reading", 1),),
            (
                AlignedMora(0, 0, "ソ", 0, .2, .8),
                AlignedMora(0, 1, "ラ", .2, .4, .8),
            ),
            (MelodyNote(0, .4, 60),),
        )
        note = document.note_candidates[0]
        evidence = next(item for item in document.evidence if item.id in note.evidence_ids)
        self.assertEqual(note.confidence, 0)
        self.assertEqual(evidence.detail, {"confidence_available": False})

    def test_known_lyrics_adjustment_is_opt_in_and_keeps_audit_evidence(self):
        score = analyze_audio(
            self.audio,
            AudioAdapters(self._readings, self._moras, self._melody,
                          lambda _: (LyricLine("空", 0, .4), LyricLine("空", .4, .8))),
            lyrics=("耳", "空"), adjust_lyrics=True,
        )
        self.assertEqual(score.score.canonical_text, "空\n空")
        audit = next(item for item in score.observations.evidence if item.kind == "lyric-adjustment")
        self.assertEqual(audit.detail["supplied_lines"], ["耳", "空"])
        self.assertEqual([row["operation"] for row in audit.detail["decisions"]],
                         ["keep", "repeat", "remove"])

    def test_adjustment_requires_known_lyrics_and_a_recognizer(self):
        adapters = AudioAdapters(self._readings, self._moras, self._melody)
        with self.assertRaisesRegex(ValueError, "requires supplied lyrics"):
            analyze_audio(self.audio, adapters, adjust_lyrics=True)
        with self.assertRaisesRegex(AudioPipelineError, "requires a recognizer"):
            analyze_audio(self.audio, adapters, lyrics=("空",), adjust_lyrics=True)

    def test_inconsistent_mora_result_is_rejected_before_compilation(self):
        def bad_moras(_path, _lines, _readings):
            return (AlignedMora(0, 0, "ソ", 0, .2, .8),)

        with self.assertRaisesRegex(AudioPipelineError, "selected reading"):
            analyze_audio(
                self.audio,
                AudioAdapters(self._readings, bad_moras, self._melody,
                              lambda _: (LyricLine("空", 0, .4),)),
                lyrics=("空",),
            )

    def test_automatic_mode_requires_timed_nonoverlapping_lines(self):
        adapters = AudioAdapters(
            self._readings,
            self._moras,
            self._melody,
            lambda _path: (LyricLine("空", 0, .4), LyricLine("耳", .3, .8)),
        )
        with self.assertRaisesRegex(AudioPipelineError, "non-overlapping"):
            analyze_audio(self.audio, adapters)


if __name__ == "__main__":
    unittest.main()
