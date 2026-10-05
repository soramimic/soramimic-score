from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

from soramimic_score import (AlignedMora, AudioAdapters, LyricLine, MelodyNote,
                             PhoneticMora, ReadingSelection, analyze_audio)
from soramimic_score.japanese import kana_to_moras
from soramimic_score.phonetic_repeats import find_repeated_lines, repetition_windows
from soramimic_score.romaji import context_windows


class PhoneticRepetitionTests(unittest.TestCase):
    def setUp(self):
        self.line = LyricLine('青い空', 0., 4.)
        self.reading = ReadingSelection('アオイソラ', 'test', .9)
        self.notes = tuple(MelodyNote(i * .2, (i + 1) * .2, 60 + i % 3)
                           for i in range(20))
        self.events = tuple(PhoneticMora(kana, offset + i * .2,
                                        offset + i * .2 + .08)
                            for offset in (.2, 2.2)
                            for i, kana in enumerate('アオイソラ'))

    def find(self, events=None, *, line=None, reading=None, notes=None):
        return find_repeated_lines((line or self.line,), (reading or self.reading,),
                                   self.notes if notes is None else notes,
                                   self.events if events is None else events)

    def test_finds_two_acoustic_occurrences_inside_one_stretched_line(self):
        self.assertEqual(repetition_windows((self.line,), (self.reading,)),
                         ((0., 4.),))
        found, = self.find()
        self.assertEqual(len(found.occurrences), 2)
        self.assertEqual(found.occurrences[0].start_sec, .2)
        self.assertEqual(found.occurrences[1].start_sec, 2.2)

    def test_single_phrase_silence_instruments_and_unrelated_vowels_do_not_expand(self):
        for events in ((), self.events[:5], tuple(replace(e, kana='カ') for e in self.events)):
            with self.subTest(events=events):
                self.assertEqual(self.find(events), ())
        vowels_only = tuple(replace(e, kana='カキクケコ'[i % 5])
                            for i, e in enumerate(self.events))
        self.assertEqual(self.find(vowels_only), ())

    def test_exact_phonetic_repetition_can_survive_a_note_model_miss(self):
        found, = self.find(notes=())
        self.assertEqual(len(found.occurrences), 2)
        approximate = tuple(replace(e, kana='ホ') if e.kana == 'ソ' else e
                            for e in self.events)
        self.assertEqual(self.find(approximate, notes=()), ())
        self.assertEqual(len(self.find(approximate)[0].occurrences), 2)

    def test_existing_repeats_are_counted_as_part_of_the_original_phrase(self):
        self.assertEqual(self.find(line=replace(self.line, text='青い空 青い空'),
                                   reading=replace(self.reading, kana='アオイソラアオイソラ')), ())

    def test_recovers_a_third_occurrence_after_two_recognized_copies(self):
        line = replace(self.line, text='青い空、青い空。', end_sec=6.)
        reading = replace(self.reading, kana='アオイソラアオイソラ')
        third = tuple(replace(e, start_sec=e.start_sec + 4, end_sec=e.end_sec + 4)
                      for e in self.events[:5])
        notes = self.notes + tuple(replace(n, start_sec=n.start_sec + 4, end_sec=n.end_sec + 4)
                                   for n in self.notes[:10])
        found, = self.find(self.events + third, line=line, reading=reading, notes=notes)
        self.assertEqual((found.text, found.kana, found.original_count),
                         ('青い空', 'アオイソラ', 2))
        self.assertEqual(len(found.occurrences), 3)

    def test_duplicate_or_overlapping_decodes_cannot_create_repetitions(self):
        self.assertEqual(self.find(self.events[:5] * 2), ())
        competing = tuple(replace(e, start_sec=e.start_sec + .02, end_sec=e.end_sec + .02)
                          for e in self.events)
        self.assertEqual(self.find(self.events + competing), ())

    def test_long_vowels_do_not_count_as_rearticulated_phrases(self):
        extended = tuple(PhoneticMora(k, i * .15, i * .15 + .02)
                         for i, k in enumerate('ソラアアアアアアアアア'))
        self.assertEqual(self.find(extended), ())
        self.assertEqual(self.find(reading=replace(self.reading, kana='アアアアア')), ())

    def test_events_outside_the_line_cannot_be_borrowed(self):
        shifted = tuple(replace(e, start_sec=e.start_sec + 10, end_sec=e.end_sec + 10)
                        for e in self.events)
        self.assertEqual(self.find(shifted), ())

    def test_rejects_invalid_phonetic_evidence(self):
        for event in (object(), PhoneticMora('カ', float('nan'), 1.),
                      PhoneticMora('カキ', 0., 1.), PhoneticMora('カ', 0., 1., '')):
            with self.subTest(event=event), self.assertRaises(ValueError):
                self.find((event,))

    def test_pipeline_realigns_each_copy_and_preserves_other_lines(self):
        other = LyricLine('白い雲', 4., 6.)
        aligned_calls = []

        def readings(_path, lines):
            return tuple(ReadingSelection('アオイソラ' if line.text == '青い空' else 'シロイクモ',
                                          'test', .9) for line in lines)

        def align(_path, lines, selected):
            aligned_calls.append(tuple(lines))
            return tuple(AlignedMora(index, i, kana,
                                      line.start_sec + i * (line.end_sec - line.start_sec) / 5,
                                      line.start_sec + (i + 1) * (line.end_sec - line.start_sec) / 5,
                                      .9)
                         for index, (line, reading) in enumerate(zip(lines, selected))
                         for i, kana in enumerate(kana_to_moras(reading.kana)))

        adapters = AudioAdapters(readings, align, lambda _: self.notes,
                                 lambda _: (self.line, other),
                                 phonetic_recognizer=lambda _p, _w: self.events)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.wav'
            path.write_bytes(b'adapter fixture')
            document = analyze_audio(path, adapters)
        self.assertEqual(document.score.canonical_text, '青い空\n青い空\n白い雲')
        self.assertEqual(len(aligned_calls), 2)
        self.assertEqual(len(aligned_calls[1]), 2)
        self.assertEqual(aligned_calls[1][0].end_sec, aligned_calls[1][1].start_sec)
        self.assertEqual(document.observations.readings[-1].kana, 'シロイクモ')
        evidence, = (e for e in document.observations.evidence
                      if e.kind == 'lyric-phonetic-repetition')
        self.assertEqual(evidence.detail['recovered_count'], 2)
        self.assertFalse(evidence.detail['confidence_available'])

    def test_failed_realignment_keeps_original(self):
        def align(_path, lines, _readings):
            if len(lines) > 1:
                raise ValueError('candidate alignment failed')
            return tuple(AlignedMora(0, i, k, i * .8, (i + 1) * .8, .9)
                         for i, k in enumerate('アオイソラ'))
        adapters = AudioAdapters(lambda _p, _l: (self.reading,), align,
                                 lambda _: self.notes, lambda _: (self.line,),
                                 phonetic_recognizer=lambda _p, _w: self.events)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.wav'
            path.write_bytes(b'adapter fixture')
            document = analyze_audio(path, adapters)
        self.assertEqual(document.score.canonical_text, self.line.text)
        self.assertFalse(any(e.kind == 'lyric-phonetic-repetition'
                             for e in document.observations.evidence))

    def test_pipeline_expands_two_to_three_copies_using_the_unit_reading(self):
        line = LyricLine('青い空、青い空。', 0., 6.)
        reading = replace(self.reading, kana=self.reading.kana * 2,
                          candidates=(self.reading.kana * 2,))
        events = self.events + tuple(replace(e, start_sec=e.start_sec + 4, end_sec=e.end_sec + 4)
                                     for e in self.events[:5])

        def align(_path, lines, selected):
            result = []
            for index, (item, pronunciation) in enumerate(zip(lines, selected)):
                kana = kana_to_moras(pronunciation.kana)
                step = (item.end_sec - item.start_sec) / len(kana)
                result.extend(AlignedMora(index, i, k, item.start_sec + i * step,
                                           item.start_sec + (i + 1) * step, .9)
                              for i, k in enumerate(kana))
            return tuple(result)

        adapters = AudioAdapters(lambda _p, _l: (reading,), align,
                                 lambda _: (MelodyNote(6., 7., 60),), lambda _: (line,),
                                 phonetic_recognizer=lambda _p, _w: events)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.wav'
            path.write_bytes(b'adapter fixture')
            document = analyze_audio(path, adapters)
        self.assertEqual(document.score.canonical_text, '青い空\n青い空\n青い空')
        self.assertEqual([r.kana for r in document.observations.readings],
                         ['アオイソラ'] * 3)


class PhoneticContextTests(unittest.TestCase):
    def test_overlapping_requests_share_one_decode_per_owned_interval(self):
        whole = context_windows(((5., 28.),), 50., 20.)
        split = context_windows(((20., 28.), (5., 15.), (10., 22.)), 50., 20.)
        self.assertEqual(whole, split)
        self.assertEqual([row[2:] for row in whole], [(0., 10.), (10., 20.), (20., 30.)])
        for first, last, start, end in whole:
            self.assertLessEqual(first, start)
            self.assertGreaterEqual(last, end)

    def test_short_audio_and_terminal_partial_cell(self):
        self.assertEqual(context_windows(((0., 3.),), 3., 20.), ((0., 3., 0., 3.),))
        self.assertEqual(context_windows(((23., 25.),), 25., 20.), ((5., 25., 20., 25.),))
        self.assertEqual(context_windows((), 10., 20.), ())

    def test_invalid_window_is_rejected_before_inference(self):
        for bounds in ((-1., 2.), (1., 1.), (1., 12.), (float('nan'), 2.)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                context_windows((bounds,), 10., 20.)
