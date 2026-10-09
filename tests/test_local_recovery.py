import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from soramimic_score.audio import (AlignedMora, AudioAdapters, LyricLine,
                                   MelodyNote, ReadingSelection, analyze_audio)
from soramimic_score.local_recovery import (adjacent_repeat_groups,
                                            coalesce_repeated_suffix_fragments,
                                            deficit_windows,
                                            duration_repeated_vocalization_candidate,
                                            expand_repeated_vocalization_from_kana,
                                            has_tandem_repeat_note_support,
                                            is_pathological_repeated_vocalization,
                                            normalize_repeated_vocalization,
                                            repeated_vocalization_period,
                                            uncovered_note_windows,
                                            unowned_note_windows)
from soramimic_score.vocal_activity import VocalActivity


class LocalRecoveryTests(unittest.TestCase):
    def test_adjacent_repeat_group_uses_one_acoustic_count(self):
        lines = (LyricLine("ララ", 0, 2), LyricLine("ララ", 2, 4))
        self.assertEqual(adjacent_repeat_groups(lines), ((0, 2, ("ラ",)),))
        notes = tuple(MelodyNote(i * .5, (i + 1) * .5, 60) for i in range(8))
        calls = []

        def readings(_path, chosen):
            return tuple(ReadingSelection(line.text, "test", 1) for line in chosen)

        def align(_path, chosen, selected):
            return tuple(AlignedMora(index, offset, char,
                                     line.start_sec + offset * .2,
                                     line.start_sec + (offset + 1) * .2, .9)
                         for index, (line, reading) in enumerate(zip(chosen, selected))
                         for offset, char in enumerate(reading.kana))

        def evidence(_path, windows):
            calls.append(windows)
            return ("ラ" * 6,) if len(windows) == 1 else ("ラ" * 4,) * len(windows)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.wav"
            path.write_bytes(b"model adapter fixture")
            score = analyze_audio(path, AudioAdapters(
                readings, align, lambda _: notes, lambda _: lines,
                repetition_evidence=evidence))
        self.assertEqual(score.score.canonical_text, "ラ" * 6)
        self.assertEqual(calls, [((0, 4),)])

    def test_deficit_retry_combines_separated_note_groups(self):
        lines = (LyricLine("カキ", 0, 5), LyricLine("サシスセ", 6, 6.8),
                 LyricLine("タチツテ", 7, 7.8))
        notes = tuple(MelodyNote(start + i * .2, start + (i + 1) * .2, 60)
                      for start in (1, 3, 6, 7) for i in range(4))
        calls = []

        def readings(_path, chosen):
            return tuple(ReadingSelection(line.text, "test", 1) for line in chosen)

        def align(_path, chosen, selected):
            return tuple(AlignedMora(index, offset, char,
                                     line.start_sec + offset * .1,
                                     line.start_sec + (offset + 1) * .1, .9)
                         for index, (line, reading) in enumerate(zip(chosen, selected))
                         for offset, char in enumerate(reading.kana))

        def retry(_path, start, end):
            calls.append((start, end))
            onset = 1 if start < 2 else 3
            return (LyricLine("カキクケ", onset, onset + .8),)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.wav"
            path.write_bytes(b"model adapter fixture")
            score = analyze_audio(path, AudioAdapters(
                readings, align, lambda _: notes, lambda _: lines,
                lyric_reading=lambda text: text, lyric_recoverer=retry))
        self.assertEqual(calls, [(.5, 2.3), (2.5, 4.3)])
        self.assertEqual(score.score.canonical_text,
                         "カキクケ\nカキクケ\nサシスセ\nタチツテ")
        self.assertIsNone(duration_repeated_vocalization_candidate(
            LyricLine("ラナラナ", 0, 2), (LyricLine("ララ", 0, 2),), 2))

    def test_latin_refrain_preserves_whisper_count_in_kana(self):
        self.assertEqual(normalize_repeated_vocalization(
            LyricLine("la la la", 0, 1)).text, "ラララ")
        self.assertEqual(normalize_repeated_vocalization(
            LyricLine("明日は晴れ", 0, 1)).text, "明日は晴れ")

        def readings(_path, chosen):
            return tuple(ReadingSelection(line.text, "test", 1) for line in chosen)

        def align(_path, _chosen, selected):
            return tuple(AlignedMora(0, index, char, index * .25,
                                     (index + 1) * .25, .9)
                         for index, char in enumerate(selected[0].kana))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.wav"
            path.write_bytes(b"model adapter fixture")
            score = analyze_audio(path, AudioAdapters(
                readings, align,
                lambda _: tuple(MelodyNote(i * .25, (i + 1) * .25, 60)
                                for i in range(3)),
                lambda _: (LyricLine("la la la", 0, .75),),
            ))
        self.assertEqual(score.score.canonical_text, "ラララ")

    def test_tandem_retry_needs_repetition_and_better_note_fit(self):
        options = dict(source_moras=4, recovered_moras=8,
                       note_count=9, median_notes_per_mora=1.)
        self.assertTrue(has_tandem_repeat_note_support("カキクケカキクケ", **options))
        self.assertFalse(has_tandem_repeat_note_support("カキクケサシスセ", **options))
        self.assertFalse(has_tandem_repeat_note_support(
            "カキクケカキクケ", **{**options, "note_count": 4}))

    def test_stage3_note_ownership_and_retained_line_bound_recovery(self):
        notes = [SimpleNamespace(id=f"n{i}", start_sec=i * .25,
                                 end_sec=(i + 1) * .25) for i in range(24)]
        links = [SimpleNamespace(operation="note_only", singing_unit_ids=(),
                                 note_candidate_ids=(note.id,)) for note in notes]
        document = SimpleNamespace(note_candidates=notes, links=links)
        self.assertEqual(unowned_note_windows(document, ()), ((0, 6, 24),))
        self.assertEqual(unowned_note_windows(document, (LyricLine("歌詞", 2, 4),)), ())
        links[0] = SimpleNamespace(operation="match", singing_unit_ids=("u0",),
                                   note_candidate_ids=("n0",))
        self.assertEqual(unowned_note_windows(document, ()), ((.25, 6, 23),))

    def test_decoder_runaway_is_rejected_without_dropping_short_repetition(self):
        notes = tuple(MelodyNote(i * .25, (i + 1) * .25, 60) for i in range(8))
        self.assertFalse(is_pathological_repeated_vocalization(LyricLine("ラ" * 6, 0, 2), notes))
        self.assertTrue(is_pathological_repeated_vocalization(LyricLine("ラ" * 70, 0, 2), notes))
        self.assertFalse(is_pathological_repeated_vocalization(LyricLine("カ" * 70, 0, 2), notes))
        self.assertFalse(is_pathological_repeated_vocalization(
            LyricLine("だん" * 3, 0, 2), notes))
        self.assertTrue(is_pathological_repeated_vocalization(
            LyricLine("だん" * 70 + "だ", 0, 2), notes))

    def test_nasal_repetition_retry_is_bounded_before_alignment(self):
        notes = (MelodyNote(0, 2, 60), MelodyNote(2.5, 3, 62))
        for recovered, expected in (("だん" * 3, "ダンダンダン\nカ"),
                                    ("だん" * 70 + "だ", "カ")):
            with self.subTest(recovered_moras=len(recovered)):
                def readings(_path, lines):
                    return tuple(ReadingSelection(line.text, "test", 1) for line in lines)

                def align(_path, lines, selected):
                    self.assertTrue(all(len(item.kana) <= 6 for item in selected))
                    return tuple(AlignedMora(i, j, char, line.start_sec + j * .25,
                                            line.start_sec + (j + 1) * .25, .0001)
                                 for i, (line, reading) in enumerate(zip(lines, selected))
                                 for j, char in enumerate(reading.kana))

                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "input.wav"
                    path.write_bytes(b"model adapter fixture")
                    document = analyze_audio(path, AudioAdapters(
                        readings, align, lambda _: notes, lambda _: (
                            LyricLine("作詞・作曲・編曲 初音ミク", 0, 2),
                            LyricLine("カ", 2.5, 3)),
                        lyric_recoverer=lambda _path, start, end:
                            (LyricLine(recovered, start, end),),
                    ))
                self.assertEqual(document.score.canonical_text, expected)

    def test_decoder_runaway_is_omitted_from_automatic_score(self):
        notes = tuple(MelodyNote(i * .25, (i + 1) * .25, 60) for i in range(12))
        lines = (LyricLine("ラ" * 70, 0, 2), LyricLine("カキクケ", 2, 3))

        def readings(_path, chosen):
            return tuple(ReadingSelection(item.text, "test", 1) for item in chosen)

        def align(_path, chosen, selected):
            return tuple(AlignedMora(0, index, char, 2 + index * .25,
                                     2 + (index + 1) * .25, .9)
                         for index, char in enumerate(selected[0].kana))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.wav"
            path.write_bytes(b"model adapter fixture")
            document = analyze_audio(path, AudioAdapters(
                readings, align, lambda _: notes, lambda _: lines,
            ))
        self.assertEqual(document.score.canonical_text, "カキクケ")
        self.assertTrue(any(e.kind == "lyric-repetition-rejection"
                            for e in document.observations.evidence))

    def test_kana_evidence_expands_only_whisper_repetition_family_and_count(self):
        line = LyricLine("ララ", 0, 2)
        notes = tuple(MelodyNote(i * .25, (i + 1) * .25, 60) for i in range(8))
        self.assertEqual(repeated_vocalization_period("ララ"), ("ラ",))
        self.assertEqual(repeated_vocalization_period("la la"), ("ラ",))
        self.assertIsNone(repeated_vocalization_period("カキカキ"))
        expanded = expand_repeated_vocalization_from_kana(line, "ララララララ", notes)
        self.assertEqual(expanded.text, "ララララララ")
        self.assertIsNone(expand_repeated_vocalization_from_kana(line, "ナナナナナナ", notes))
        self.assertIsNone(expand_repeated_vocalization_from_kana(line, "ラ" * 20, notes))

    def test_kana_repetition_evidence_reaches_the_score_without_using_its_words(self):
        line = LyricLine("ララ", 0, 2)
        notes = tuple(MelodyNote(i * .25, (i + 1) * .25, 60) for i in range(8))
        calls = []

        def readings(_path, chosen):
            return tuple(ReadingSelection(item.text, "test", 1) for item in chosen)

        def align(_path, chosen, selected):
            return tuple(AlignedMora(0, index, char, index * .3, (index + 1) * .3, .9)
                         for index, char in enumerate(selected[0].kana))

        def evidence(_path, windows):
            calls.extend(windows)
            return ("ララララララ",)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.wav"
            path.write_bytes(b"model adapter fixture")
            document = analyze_audio(path, AudioAdapters(
                readings, align, lambda _: notes, lambda _: (line,),
                repetition_evidence=evidence,
            ))
        self.assertEqual(calls, [(0, 2)])
        self.assertEqual(document.score.canonical_text, "ララララララ")
        self.assertTrue(any(e.kind == "lyric-repetition-expansion"
                            for e in document.observations.evidence))

    def test_short_suffix_joins_only_with_parallel_refrain_evidence(self):
        lines = (LyricLine("君に届け", 0, 2), LyricLine("君に", 3, 4),
                 LyricLine("届け", 4.02, 4.5))
        merged, evidence = coalesce_repeated_suffix_fragments(lines)
        self.assertEqual([line.text for line in merged], ["君に届け", "君に届け"])
        self.assertEqual(evidence, ((1, 2),))
        merged, evidence = coalesce_repeated_suffix_fragments(lines[1:])
        self.assertEqual(merged, lines[1:])
        self.assertFalse(evidence)

    def test_note_rich_short_line_gets_bounded_retry_window(self):
        lines = (LyricLine("カキ", 0, 1.6), LyricLine("カキクケ", 2, 2.8),
                 LyricLine("サシスセ", 3, 3.8))
        notes = tuple(MelodyNote(i * .2, (i + 1) * .2, 60) for i in range(8))
        notes += tuple(MelodyNote(2 + i * .2, 2 + (i + 1) * .2, 62)
                       for i in range(4))
        notes += tuple(MelodyNote(3 + i * .2, 3 + (i + 1) * .2, 64)
                       for i in range(4))
        self.assertEqual(deficit_windows(lines, notes, (2, 4, 4)), ((0, 0, 1.6),))

    def test_deficit_retry_splits_at_long_internal_rest(self):
        lines = (LyricLine("カキ", 0, 5), LyricLine("サシスセ", 6, 6.8),
                 LyricLine("タチツテ", 7, 7.8))
        notes = tuple(MelodyNote(start + i * .2, start + (i + 1) * .2, 60)
                      for start in (1, 3, 6, 7) for i in range(4))
        self.assertEqual(deficit_windows(lines, notes, (2, 4, 4)),
                         ((0, .5, 2.3), (0, 2.5, 4.3)))

    def test_long_uncovered_note_run_is_retried(self):
        lines = (LyricLine("ア", 0, .4), LyricLine("イ", 3, 3.4))
        notes = tuple(MelodyNote(1 + i * .2, 1 + (i + 1) * .2, 60)
                      for i in range(8))
        self.assertEqual(uncovered_note_windows(lines, notes), ((.7, 2.9),))

    def test_local_retry_recovers_missing_syllables_without_changing_peers(self):
        lines = (LyricLine("カキ", 0, 1.6), LyricLine("カキクケ", 2, 2.8),
                 LyricLine("サシスセ", 3, 3.8))
        notes = tuple(MelodyNote(i * .2, (i + 1) * .2, 60) for i in range(8))
        notes += tuple(MelodyNote(2 + i * .2, 2 + (i + 1) * .2, 62)
                       for i in range(4))
        notes += tuple(MelodyNote(3 + i * .2, 3 + (i + 1) * .2, 64)
                       for i in range(4))
        retries = []

        def retry(_path, start, end):
            retries.append((start, end))
            return (LyricLine("カキクケ", 0, 1.6),)

        def readings(_path, chosen):
            return tuple(ReadingSelection(line.text, "test", 1) for line in chosen)

        def align(_path, chosen, selected):
            return tuple(AlignedMora(index, offset, char,
                                     line.start_sec + (line.end_sec-line.start_sec)*offset/len(line.text),
                                     line.start_sec + (line.end_sec-line.start_sec)*(offset+1)/len(line.text),
                                     .9)
                         for index, (line, reading) in enumerate(zip(chosen, selected))
                         for offset, char in enumerate(reading.kana))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.wav"
            path.write_bytes(b"model adapter fixture")
            document = analyze_audio(path, AudioAdapters(
                readings, align, lambda _: notes, lambda _: lines,
                lyric_reading=lambda text: text, lyric_recoverer=retry,
            ))
        self.assertEqual(retries, [(0, 1.6)])
        self.assertEqual(document.score.canonical_text, "カキクケ\nカキクケ\nサシスセ")
        self.assertTrue(any(e.kind == "lyric-local-retry" for e in document.observations.evidence))

    def test_unowned_note_retry_requires_vocal_and_ctc_support(self):
        notes = tuple(MelodyNote(i * .25, (i + 1) * .25, 60) for i in range(24))
        notes += (MelodyNote(6, 6.25, 62), MelodyNote(6.25, 6.5, 62))
        calls = []

        def readings(_path, chosen):
            return tuple(ReadingSelection(line.text, "test", 1) for line in chosen)

        def align(_path, chosen, selected):
            return tuple(AlignedMora(index, offset, char,
                                     line.start_sec + (line.end_sec-line.start_sec)*offset/len(reading.kana),
                                     line.start_sec + (line.end_sec-line.start_sec)*(offset+1)/len(reading.kana),
                                     .001 if line.text == "カキクケサシスセ" else .9)
                         for index, (line, reading) in enumerate(zip(chosen, selected))
                         for offset, char in enumerate(reading.kana))

        def retry(_path, start, end):
            calls.append((start, end))
            return (LyricLine("カキクケサシスセ", 0, 6),)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.wav"
            path.write_bytes(b"model adapter fixture")
            def run(supported):
                return analyze_audio(path, AudioAdapters(
                    readings, align, lambda _: notes,
                    lambda _: (LyricLine("ア", 6, 6.5),),
                    lyric_recoverer=retry,
                    vocal_activity=lambda _path, windows: tuple(
                        VocalActivity(-20, 0, 1, supported) for _ in windows),
                ))
            self.assertEqual(run(False).score.canonical_text, "ア")
            self.assertEqual(calls, [])
            # An acoustically supported candidate is admitted only in a sung window.
            self.assertEqual(run(True).score.canonical_text, "カキクケサシスセ\nア")
            self.assertEqual(calls, [(0, 6), (0, 6.5)])

    def test_unowned_repeat_uses_kana_count_only_after_ctc_check(self):
        notes = tuple(MelodyNote(i * .25, (i + 1) * .25, 60) for i in range(24))
        notes += (MelodyNote(6, 6.5, 62),)

        def readings(_path, chosen):
            return tuple(ReadingSelection(line.text, "test", 1) for line in chosen)

        def align(_path, chosen, selected):
            return tuple(AlignedMora(index, offset, char,
                                     line.start_sec + offset * .1,
                                     line.start_sec + (offset + 1) * .1,
                                     .0001 if line.text == "ララ" else .9)
                         for index, (line, reading) in enumerate(zip(chosen, selected))
                         for offset, char in enumerate(reading.kana))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.wav"
            path.write_bytes(b"model adapter fixture")
            score = analyze_audio(path, AudioAdapters(
                readings, align, lambda _: notes,
                lambda _: (LyricLine("ア", 6, 6.5),),
                lyric_recoverer=lambda _path, _start, _end: (LyricLine("ララ", 0, 6),),
                repetition_evidence=lambda _path, _windows: ("ラ" * 8,),
                vocal_activity=lambda _path, windows: tuple(
                    VocalActivity(-20, 0, 1, True) for _ in windows),
            ))
        self.assertEqual(score.score.canonical_text, "ラ" * 8 + "\nア")
        recovery = [item for item in score.observations.evidence
                    if item.kind == "lyric-local-retry"]
        self.assertEqual(recovery[0].detail["source"], "kana-repeat-vocals")

    def test_unowned_vowels_need_two_agreeing_retries_and_strong_ctc(self):
        notes = tuple(MelodyNote(i * .25, (i + 1) * .25, 60) for i in range(24))
        notes += (MelodyNote(6, 6.5, 62),)

        def readings(_path, chosen):
            return tuple(ReadingSelection(line.text, "test", 1) for line in chosen)

        def align(_path, chosen, selected):
            return tuple(AlignedMora(index, offset, char,
                                     line.start_sec + offset * .1,
                                     line.start_sec + (offset + 1) * .1,
                                     .0001 if line.text == "カキクケサシスセ" else .9)
                         for index, (line, reading) in enumerate(zip(chosen, selected))
                         for offset, char in enumerate(reading.kana))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.wav"
            path.write_bytes(b"model adapter fixture")
            score = analyze_audio(path, AudioAdapters(
                readings, align, lambda _: notes,
                lambda _: (LyricLine("ア", 6, 6.5),),
                lyric_recoverer=lambda _path, _start, _end:
                    (LyricLine("カキクケサシスセ", 0, 6),),
                vocal_activity=lambda _path, windows: tuple(
                    VocalActivity(-20, 0, 1, True) for _ in windows),
            ))
        self.assertEqual(score.score.canonical_text, "アイウエアイウエ\nア")
        recovery = [item for item in score.observations.evidence
                    if item.kind == "lyric-local-retry"]
        self.assertEqual(recovery[0].detail["source"], "vowel-continuation")

    def test_deficit_retry_rejects_weak_acoustic_candidate(self):
        lines = (LyricLine("カキ", 0, 1.6), LyricLine("カキクケ", 2, 2.8),
                 LyricLine("サシスセ", 3, 3.8))
        notes = tuple(MelodyNote(start + i * .2, start + (i + 1) * .2, 60)
                      for start, count in ((0, 8), (2, 4), (3, 4))
                      for i in range(count))

        def readings(_path, chosen):
            return tuple(ReadingSelection(line.text, "test", 1) for line in chosen)

        def align(_path, chosen, selected):
            return tuple(AlignedMora(index, offset, char,
                                     line.start_sec + offset * .1,
                                     line.start_sec + (offset + 1) * .1,
                                     .0001 if line.text == "カキクケ" and line.start_sec == 0 else .9)
                         for index, (line, reading) in enumerate(zip(chosen, selected))
                         for offset, char in enumerate(reading.kana))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.wav"
            path.write_bytes(b"model adapter fixture")
            document = analyze_audio(path, AudioAdapters(
                readings, align, lambda _: notes, lambda _: lines,
                lyric_reading=lambda text: text,
                lyric_recoverer=lambda _path, _start, _end:
                    (LyricLine("カキクケ", 0, 1.6),),
            ))
        self.assertEqual(document.score.canonical_text, "カキ\nカキクケ\nサシスセ")
