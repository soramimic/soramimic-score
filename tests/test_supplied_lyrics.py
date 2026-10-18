import tempfile
import unittest
import math
from dataclasses import replace
from pathlib import Path

from soramimic_score import (AlignedMora, AudioAdapters, LyricLine, MelodyNote,
                             ReadingSelection, analyze_audio, lyric_surface)
from soramimic_score.japanese import kana_to_moras
from soramimic_score.supplied_lyrics import (plan_supplied_lyrics, locate_supplied_groups,
                                            prepare_supplied_audio)
from soramimic_score.vocal_activity import VocalActivity


class SuppliedLyricsTests(unittest.TestCase):
    def test_roundoff_between_touching_anchors_keeps_missing_lyrics_in_a_real_window(self):
        for start in (.3, .1 + .2, math.nextafter(.3, math.inf)):
            with self.subTest(start=start):
                recognized = (LyricLine("カキ", 0., .3), LyricLine("サシ", start, 1.))
                supplied = ("カキ", "タチ", "サシ")
                plan = plan_supplied_lyrics(supplied, recognized)
                groups = locate_supplied_groups(plan, recognized, 1.)
                self.assertEqual([g["display_text"] for g in groups], ["カキ", "タチ\nサシ"])
                self.assertEqual([g["supplied_indices"] for g in groups], [[0], [1, 2]])
                self.assertEqual([(g["start_sec"], g["end_sec"]) for g in groups], [(0., .3), (.3, 1.)])

    def test_a_real_gap_is_not_merged_by_the_roundoff_tolerance(self):
        recognized = (LyricLine("カキ", 0., .3), LyricLine("サシ", .31, 1.))
        plan = plan_supplied_lyrics(("カキ", "タチ", "サシ"), recognized)
        groups = locate_supplied_groups(plan, recognized, 1.)
        self.assertEqual([g["display_text"] for g in groups], ["カキ", "タチ", "サシ"])
        self.assertEqual((groups[1]["start_sec"], groups[1]["end_sec"]), (.3, .31))

    def test_a_single_roundoff_sized_window_is_unresolved_without_alignment_or_removal(self):
        for adjust in (False, True):
            with self.subTest(adjust=adjust):
                adapters = self.adapters((), duration=1e-12)
                lines, plan, unobserved = prepare_supplied_audio(
                    self.path, ("カキ",), (), (), adapters, adjust_lyrics=adjust,
                    recognition_evidence=(),
                )
                self.assertEqual(unobserved, (0,))
                self.assertEqual(lines[0].text, "カキ")
                self.assertIsNone(lines[0].start_sec)
                self.assertEqual(plan["groups"][0]["alignment_status"], "window-unresolved")
                self.assertEqual(plan["removed_supplied_indices"], [])

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

    def test_ctc_expands_both_sides_after_unchanged_reading_selection(self):
        calls = []
        supplied = ["カキ", "空（そら）", "タチ"]
        def select(_path, lines, contexts):
            calls.append(("read", [(l.text, l.start_sec, l.end_sec) for l in lines], contexts))
            return (ReadingSelection("カキ", "test", .9),
                    ReadingSelection("ソラ", "test", .8, ("ソラ",), {
                        "resolved_text": "空", "parenthetical_readings": [{"decision": "ruby"}],
                    }), ReadingSelection("タチ", "test", .9))
        def align(_path, lines, readings):
            calls.append(("align", [(l.text, l.start_sec, l.end_sec) for l in lines],
                          [r.kana for r in readings]))
            times = ((0, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 4))
            return tuple(AlignedMora(0, i, kana, start, end, .9)
                         for i, (kana, (start, end)) in enumerate(zip("カキソラタチ", times)))
        adapters = replace(self.adapters((LyricLine("カキ", 0, 2), LyricLine("タチ", 2, 4)),
                                         (MelodyNote(0, 4, 60),), duration=4),
                           supplied_reading_selector=select, mora_aligner=align)
        result = analyze_audio(self.path, adapters, lyrics=supplied)
        self.assertEqual(calls, [
            ("read", [("カキ", 0, 2), ("空（そら）", 2, 4), ("タチ", 2, 4)], ("カキ", None, None)),
            ("align", [("カキ\n空\nタチ", 0, 4)], ["カキソラタチ"]),
        ])
        self.assertEqual([l.text for l in result.score.canonical], ["カキ", "空", "タチ"])
        self.assertEqual([r.kana for r in result.observations.readings], ["カキ", "ソラ", "タチ"])
        overlay = lyric_surface(result)
        self.assertEqual([(g["start_sec"], g["end_sec"]) for g in overlay["groups"]],
                         [(0, 1), (1, 2), (2, 4)])
        self.assertEqual([g["supplied_indices"] for g in overlay["groups"]], [[0], [1], [2]])
        self.assertEqual(overlay["groups"][1]["supplied_display_text"], "空（そら）")
        self.assertEqual(overlay["groups"][1]["parenthetical_readings"], [{"decision": "ruby"}])
        self.assertNotIn("parenthetical_readings", overlay["groups"][0])
        self.assertEqual([(g["start_sec"], g["end_sec"]) for g in overlay["reading_stage_groups"]],
                         [(0, 2), (2, 4)])
        self.assertNotIn("vocal_activity_measurement", overlay["acoustic_groups"][0])
        self.assertEqual(lyric_surface(type(result).from_json(result.to_json())), overlay)

    def test_expansion_preserves_silent_lines_and_opt_in_removal(self):
        silent = lambda _path, windows: tuple(
            VocalActivity(-100, -90, 0, False, silence_confirmed=True) for _ in windows)
        for adjust in (False, True):
            with self.subTest(adjust=adjust):
                self.aligned_text.clear()
                result = analyze_audio(self.path, self.adapters(
                    (LyricLine("カキ", 0, 2), LyricLine("タチ", 2, 4)),
                    (MelodyNote(0, 4, 60),), vocal=silent),
                    lyrics=["カキ", "サシ", "タチ", "ナニ"], adjust_lyrics=adjust)
                self.assertEqual(self.aligned_text, ["カキ\nサシ\nタチ"])
                overlay = lyric_surface(result)
                self.assertEqual([g["supplied_indices"] for g in overlay["groups"]],
                                 [[0], [1], [2]] + ([] if adjust else [[3]]))
                self.assertEqual(overlay["removed_supplied_indices"], [3] if adjust else [])
                if not adjust:
                    self.assertEqual(overlay["groups"][-1]["alignment_status"], "no-performance-evidence")
                    self.assertEqual(overlay["groups"][-1]["support"], {
                        "melody": False, "recognition": False, "vocal_activity": False})

    def test_expansion_keeps_automatic_addition_provenance(self):
        heard = (LyricLine("カキ", 0, 1), LyricLine("サシ", 1, 2),
                 LyricLine("タチ", 2, 3), LyricLine("ハヒ", 3, 4))
        result = analyze_audio(self.path, self.adapters(heard, (MelodyNote(0, 4, 60),), duration=4),
                               lyrics=["カキ", "タチ", "ナニ", "ハヒ"], adjust_lyrics=True)
        self.assertEqual(self.aligned_text, ["カキ", "サシ", "タチ\nナニ\nハヒ"])
        overlay = lyric_surface(result)
        self.assertEqual([g["supplied_indices"] for g in overlay["groups"]], [[0], [], [1], [2], [3]])
        self.assertEqual(overlay["groups"][1]["operation"], "add")
        self.assertEqual(overlay["groups"][1]["reading_source"], "automatic-addition")
        self.assertEqual([l.text for l in result.score.canonical], ["カキ", "サシ", "タチ", "ナニ", "ハヒ"])

    def test_ctc_context_does_not_bridge_an_unobserved_group(self):
        from soramimic_score.supplied_lyrics import expand_alignment_windows
        lines = (LyricLine("カキ", 0, 2), LyricLine("サシ", None, None), LyricLine("タチ", 4, 6))
        groups = [{"location_group_indices": [i]} for i in range(3)]
        overlay = {"groups": groups, "ctc_context_edges": [[0, 1], [1, 2]]}
        readings = self.readings(self.path, lines)
        self.assertEqual(expand_alignment_windows(lines, readings, overlay, (1,)),
                         (lines, readings, (1,)))
        self.assertNotIn("reading_stage_groups", overlay)

    def test_repeated_missing_anchors_have_bounded_ctc_context(self):
        from soramimic_score.supplied_lyrics import expand_alignment_windows, select_authored_readings
        recognized = tuple(LyricLine(f"カ{i}キ", i * 2, (i + 1) * 2) for i in range(64))
        supplied = [text for i, line in enumerate(recognized)
                    for text in ([line.text, "サシ"] if i < 63 else [line.text])]
        lines, overlay, unobserved = prepare_supplied_audio(
            self.path, supplied, recognized, (MelodyNote(0, 128, 60),),
            self.adapters(recognized, duration=128), adjust_lyrics=False, recognition_evidence=recognized)
        readings = select_authored_readings(lines, lambda expanded, _: tuple(
            ReadingSelection("カキ", "test", 1) for _ in expanded))
        expanded, _, _ = expand_alignment_windows(lines, readings, overlay, unobserved)
        self.assertGreater(len(expanded), 1)
        self.assertTrue(all(line.end_sec - line.start_sec <= 60 for line in expanded))
        self.assertEqual("\n".join(l.text for l in expanded), "\n".join(supplied))

    def test_touching_recording_edges_keep_ordered_ctc_targets(self):
        for supplied, heard, expected in [
            (["サシ", "カキ"], (LyricLine("カキ", 0, 4),), ["サシ\nカキ"]),
            (["カキ", "サシ"], (LyricLine("カキ", 0, 4),), ["カキ\nサシ"]),
        ]:
            with self.subTest(supplied=supplied):
                self.aligned_text.clear()
                result = analyze_audio(self.path, self.adapters(heard, (MelodyNote(0, 4, 60),), duration=4),
                                       lyrics=supplied)
                self.assertEqual(self.aligned_text, expected)
                self.assertEqual([g["supplied_indices"] for g in lyric_surface(result)["groups"]], [[0], [1]])

    def test_empty_recognition_still_aligns_supplied_lyrics_to_notes(self):
        result = analyze_audio(self.path, self.adapters(
            (), (MelodyNote(0, 6, 60),)), lyrics=["カキ", "サシ"])
        self.assertEqual(result.score.canonical_text, "カキ\nサシ")
        self.assertEqual(self.aligned_text, ["カキ\nサシ"])

    def test_only_joint_absence_of_support_leaves_input_unobserved(self):
        def silent(_path, windows):
            return tuple(VocalActivity(-100, -90, 0, False, silence_confirmed=True) for _ in windows)
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
                    return tuple(VocalActivity(-20, 0, 1, True) if source == "vocal"
                                 else VocalActivity(-100, -90, 0, False, silence_confirmed=True) for _ in windows)
                notes = (MelodyNote(0, 6, 60),) if source == "melody" else ()
                heard = (LyricLine("ウア", 0, 6),) if source == "recognition" else ()
                result = analyze_audio(self.path, self.adapters(heard, notes, vocal=activity),
                                       lyrics=["カキ"], adjust_lyrics=True)
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
            return tuple(VocalActivity(-100, -90, 0, False, silence_confirmed=True) for _ in windows)
        result = analyze_audio(self.path, self.adapters(
            heard, (MelodyNote(0, 1, 60), MelodyNote(2, 3, 62)), vocal=activity),
            lyrics=["カキ", "タチ"], adjust_lyrics=True)
        self.assertEqual(result.score.canonical_text, "カキ\nタチ")
        self.assertEqual(len(lyric_surface(result)["rejected_additions"]), 1)

    def test_removal_is_opt_in_and_audits_missing_intro_middle_and_ending(self):
        supplied = ["アア", "カキ", "サシ", "タチ", "ナニ"]
        heard = (LyricLine("カキ", 1, 2), LyricLine("タチ", 4, 5))
        def silent(_path, windows):
            return tuple(VocalActivity(-100, -90, 0, False, silence_confirmed=True) for _ in windows)
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                self.aligned_text.clear()
                result = analyze_audio(self.path, self.adapters(
                    heard, (MelodyNote(1, 2, 60), MelodyNote(4, 5, 62)), vocal=silent),
                    lyrics=supplied, adjust_lyrics=enabled)
                expected = ["カキ", "タチ"] if enabled else supplied
                self.assertEqual(result.score.canonical_text, "\n".join(expected))
                self.assertEqual(self.aligned_text, ["カキ", "タチ"])
                overlay = lyric_surface(result)
                self.assertEqual(overlay["supplied_lines"], supplied)
                self.assertEqual(overlay["removed_supplied_indices"], [0, 2, 4] if enabled else [])
                if enabled:
                    audit = next(e.detail for e in result.observations.evidence
                                 if e.kind == "lyric-adjustment")
                    removals = [d for d in audit["decisions"] if d["operation"] == "remove"]
                    self.assertEqual([(d["start_sec"], d["end_sec"]) for d in removals],
                                     [(0, 1), (2, 4), (5, 6)])
                    self.assertTrue(all(d["support"] == {
                        "melody": False, "recognition": False, "vocal_activity": False,
                    } and d["vocal_activity_measurement"]["active_frame_ratio"] == 0
                        for d in removals))
                    self.assertEqual([g["utterance_ids"] for g in overlay["groups"]],
                                     [["u0"], ["u1"]])

    def test_brief_vocal_activity_prevents_removal_even_below_percentile_gate(self):
        def brief(_path, windows):
            return tuple(VocalActivity(-70, -45, .01, False) for _ in windows)
        result = analyze_audio(self.path, self.adapters((), vocal=brief),
                               lyrics=["カキ"], adjust_lyrics=True)
        self.assertEqual(result.score.canonical_text, "カキ")
        self.assertEqual(self.aligned_text, ["カキ"])
        self.assertEqual(lyric_surface(result)["removed_supplied_indices"], [])

    def test_unmeasured_sources_cannot_authorize_removal(self):
        def silent(_path, windows):
            return tuple(VocalActivity(-100, -90, 0, False, silence_confirmed=True) for _ in windows)
        heard = (LyricLine("カキ", 0, 1), LyricLine("タチ", 5, 6))
        base = self.adapters(heard, vocal=silent)
        for adapter_name in ("vocal_activity", "lyric_recognizer", "audio_duration"):
            with self.subTest(adapter=adapter_name):
                result = analyze_audio(self.path, replace(base, **{adapter_name: None}),
                                       lyrics=["カキ", "サシ", "タチ"], adjust_lyrics=True)
                self.assertEqual(result.score.canonical_text, "カキ\nサシ\nタチ")
                self.assertEqual(lyric_surface(result)["removed_supplied_indices"], [])

    def test_non_silent_or_unmeasured_vocals_cannot_authorize_removal(self):
        for covered in (False, None):
            with self.subTest(covered=covered):
                def incomplete(_path, windows):
                    return tuple(VocalActivity(-100, -90, 0, False, silence_confirmed=covered)
                                 for _ in windows)
                result = analyze_audio(self.path, self.adapters((), vocal=incomplete),
                                       lyrics=["カキ"], adjust_lyrics=True)
                self.assertEqual(result.score.canonical_text, "カキ")
                self.assertEqual(lyric_surface(result)["removed_supplied_indices"], [])

    def test_discarded_recognition_still_prevents_claiming_absence(self):
        def silent(_path, windows):
            return tuple(VocalActivity(-100, -90, 0, False, silence_confirmed=True) for _ in windows)
        raw = (LyricLine("ウア", 0, 6),)
        lines, plan, unobserved = prepare_supplied_audio(
            self.path, ["カキ"], (), (), self.adapters(raw, vocal=silent),
            adjust_lyrics=True, recognition_evidence=raw,
        )
        self.assertEqual([line.text for line in lines], ["カキ"])
        self.assertEqual(plan["removed_supplied_indices"], [])
        self.assertEqual(unobserved, ())

    def test_unresolved_empty_window_is_not_absence_evidence(self):
        result = analyze_audio(self.path, self.adapters((), duration=0., vocal=lambda *_: ()),
                               lyrics=["カキ"], adjust_lyrics=True)
        self.assertEqual(result.score.canonical_text, "カキ")
        self.assertEqual(lyric_surface(result)["removed_supplied_indices"], [])
        self.assertEqual(lyric_surface(result)["groups"][0]["alignment_status"], "window-unresolved")

    def test_removing_all_silent_input_returns_an_empty_score_with_original_evidence(self):
        def silent(_path, windows):
            return tuple(VocalActivity(-100, -90, 0, False, silence_confirmed=True) for _ in windows)
        result = analyze_audio(self.path, self.adapters((), vocal=silent),
                               lyrics=["カキ", "サシ"], adjust_lyrics=True)
        self.assertEqual(result.score.canonical_text, "")
        self.assertEqual(result.score.synthesis_plan, ())
        self.assertEqual(self.aligned_text, [])
        restored = type(result).from_json(result.to_json())
        self.assertEqual(lyric_surface(restored)["supplied_lines"], ["カキ", "サシ"])
        self.assertEqual(lyric_surface(restored)["removed_supplied_indices"], [0, 1])

    def test_coarse_recognition_keeps_short_and_repeated_input_lines_separate(self):
        supplied = ["カキクケコ", "サシスセソ", "カキクケコ"]
        for adjust in (False, True):
            with self.subTest(adjust=adjust):
                self.aligned_text.clear()
                result = analyze_audio(self.path, self.adapters(
                    (LyricLine("".join(supplied), 0, 6),),
                    tuple(MelodyNote(i * .4, (i + 1) * .4, 60 + i % 5)
                          for i in range(15))), lyrics=supplied, adjust_lyrics=adjust)
                # Align jointly, but never expose the acoustic grouping as a phrase.
                self.assertEqual(self.aligned_text, ["\n".join(supplied)])
                self.assertEqual([line.text for line in result.score.canonical], supplied)
                self.assertEqual([line.kana for line in result.score.canonical], supplied)
                self.assertEqual([u.surface for u in result.observations.utterances], supplied)
                overlay = lyric_surface(result)
                self.assertEqual([g["supplied_indices"] for g in overlay["groups"]], [[0], [1], [2]])
                self.assertEqual([g["utterance_ids"] for g in overlay["groups"]],
                                 [["u0"], ["u1"], ["u2"]])
                self.assertEqual([(g["start_sec"], g["end_sec"]) for g in overlay["groups"]],
                                 [(0, 2), (2, 4), (4, 6)])
                self.assertEqual(len(overlay["acoustic_groups"]), 1)
                self.assertEqual("".join(slot.kana for slot in result.score.synthesis_plan),
                                 "".join(supplied))
                restored = type(result).from_json(result.to_json())
                self.assertEqual(restored, result)

    def test_authored_boundaries_use_measured_moras_instead_of_proportional_times(self):
        timings = ((.1, .2), (.2, .3), (2., 2.4), (2.4, 2.6))
        def align(_path, lines, readings):
            self.assertEqual(len(lines), 1)
            return tuple(AlignedMora(0, i, kana, *timing, .9)
                         for i, (kana, timing) in enumerate(zip("カキサシ", timings, strict=True)))
        adapters = replace(self.adapters((LyricLine("カキサシ", 0, 6),),
            tuple(MelodyNote(start, end, 60) for start, end in timings)), mora_aligner=align)
        result = analyze_audio(self.path, adapters, lyrics=["カキ", "サシ"])
        self.assertEqual([(g["start_sec"], g["end_sec"]) for g in lyric_surface(result)["groups"]],
                         [(.1, .3), (2., 2.6)])

    def test_unobserved_group_retains_every_authored_line(self):
        silent = lambda _path, windows: tuple(
            VocalActivity(-100, -90, 0, False, silence_confirmed=True) for _ in windows)
        result = analyze_audio(self.path, self.adapters((), vocal=silent),
                               lyrics=["カキ", "サシ", "タチ"])
        self.assertEqual([line.text for line in result.score.canonical], ["カキ", "サシ", "タチ"])
        self.assertEqual(self.aligned_text, [])
        self.assertEqual(len(result.score.unresolved_unit_ids), 6)
        self.assertEqual([g["alignment_status"] for g in lyric_surface(result)["groups"]],
                         ["no-performance-evidence"] * 3)

    def test_line_readings_and_annotations_survive_joint_alignment(self):
        supplied = ["朝の音", "Hello moon", "空（そら）"]
        observed = []
        def select(_path, lines, contexts):
            observed.extend((line.text, context) for line, context in zip(lines, contexts, strict=True))
            return (ReadingSelection("アサノオト", "test", .9, ("アサノオト", "アサノネ")),
                    ReadingSelection("ハロームーン", "test", .8),
                    ReadingSelection("ソラ", "test", .9, ("ソラ",), {
                        "resolved_text": "空", "parenthetical_readings": [{"decision": "ruby"}],
                    }))
        adapters = replace(self.adapters(
            (LyricLine("".join(supplied), 0, 6),), (MelodyNote(0, 6, 60),)),
            supplied_reading_selector=select)
        result = analyze_audio(self.path, adapters, lyrics=supplied)
        self.assertEqual([line.text for line in result.score.canonical], ["朝の音", "Hello moon", "空"])
        self.assertEqual([line.kana for line in result.score.canonical], ["アサノオト", "ハロームーン", "ソラ"])
        self.assertEqual([text for text, _ in observed], supplied)
        self.assertTrue(all(context == "".join(supplied) for _, context in observed))
        evidence = [e.detail for e in result.score.evidence if e.kind == "reading-selection"]
        self.assertEqual(evidence[0]["candidates"], ["アサノオト", "アサノネ"])
        self.assertEqual(evidence[1]["parenthetical_readings"], [{"decision": "ruby"}])


if __name__ == "__main__":
    unittest.main()
