import tempfile
import unittest
from pathlib import Path

from soramimic_score import (AlignedMora, AudioAdapters, LyricLine, MelodyNote,
                             PhoneticMora, ReadingSelection, analyze_audio)
from soramimic_score.phonetic_fallback import (add_phonetic_fallback,
                                              romaji_to_kana, uncovered_note_windows)
from soramimic_score.romaji import decode_romaji_ids


class PhoneticFallbackTests(unittest.TestCase):
    def test_ctc_repeats_blank_separation_unknown_tokens_and_padding(self):
        decoded = decode_romaji_ids([1, 1, 0, 1, 2, 3, 4, 9, 1],
                                   {1: 'ka', 2: 'sh', 3: 'a', 4: 'N', 9: '<unk>'},
                                   0, 10, .16)
        self.assertEqual([m.kana for m in decoded], ['カ', 'カ', 'シャ', 'ン'])
        self.assertEqual(decoded[0].start_sec, 10)
        self.assertAlmostEqual(decoded[2].end_sec, 10.12)
        self.assertLessEqual(decoded[-1].end_sec, 10.16)

    def test_rejects_invalid_adapter_events_before_sorting(self):
        for events in [(object(),), (PhoneticMora('カ', float('nan'), 1),),
                       (PhoneticMora('カキ', 0, 1),)]:
            with self.assertRaises(ValueError):
                add_phonetic_fallback((), (), (), (), events)

    def test_token_conversion_preserves_compounds_and_special_moras(self):
        self.assertEqual([romaji_to_kana(t) for t in
                          ('ka', 'ke', 'kya', 'sha', 'che', 'fa', 'N', 'cl')],
                         ['カ', 'ケ', 'キャ', 'シャ', 'チェ', 'ファ', 'ン', 'ッ'])
        self.assertIsNone(romaji_to_kana('<unk>'))
        self.assertIsNone(romaji_to_kana('k'))

    def test_preserves_known_lyrics_but_trims_unused_asr_window(self):
        line = LyricLine('空', 0, 2)
        reading = ReadingSelection('ソラ', 'dictionary', .9)
        moras = (AlignedMora(0, 0, 'ソ', 0, .2, .9),
                 AlignedMora(0, 1, 'ラ', .2, .4, .9))
        notes = tuple(MelodyNote(i * .2, (i + 1) * .2, 60) for i in range(10))
        events = (PhoneticMora('ソ', 0, .2), PhoneticMora('ラ', .2, .4),
                  PhoneticMora('カ', 1, 1.15), PhoneticMora('キ', 1.2, 1.35),
                  PhoneticMora('ク', 1.4, 1.55))
        lines, readings, aligned, evidence = add_phonetic_fallback(
            (line,), (reading,), moras, notes, events)
        self.assertEqual([l.text for l in lines], ['空', 'カキク'])
        self.assertEqual(lines[0].end_sec, .4)
        self.assertEqual(readings[0], reading)
        self.assertEqual(readings[1].kana, 'カキク')
        self.assertEqual([m.line_index for m in aligned], [0, 0, 1, 1, 1])
        self.assertFalse(evidence[0].detail['semantic_lyrics_available'])

    def test_no_instrumental_or_single_spike_fill(self):
        notes = (MelodyNote(0, 1, 60), MelodyNote(1, 2, 62))
        for events in [(), (PhoneticMora('ア', 0, .1),),
                       tuple(PhoneticMora('カ', 3 + i * .2, 3.1 + i * .2)
                             for i in range(3))]:
            self.assertEqual(add_phonetic_fallback((), (), (), notes, events),
                             ((), (), (), ()))

    def test_ctc_blanks_inside_known_line_are_not_missing_lyrics(self):
        moras = (AlignedMora(0, 0, 'カ', 0, .1, .9),
                 AlignedMora(0, 1, 'キ', .9, 1, .9))
        notes = tuple(MelodyNote(i * .2, (i + 1) * .2, 60) for i in range(5))
        self.assertEqual(uncovered_note_windows(moras, notes), ())

    def test_all_whisper_lyrics_missing_can_use_timed_katakana(self):
        notes = tuple(MelodyNote(i * .2, (i + 1) * .2, 60 + i) for i in range(3))
        events = tuple(PhoneticMora(kana, i * .2, (i + 1) * .2)
                       for i, kana in enumerate('カキク'))
        def unexpected(*args):
            raise AssertionError('fallback pronunciation must not be reinterpreted')
        adapters = AudioAdapters(unexpected, unexpected, lambda _: notes,
                                 lambda _: (), phonetic_recognizer=lambda _p, _w: events)
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / 'input.wav'
            audio.write_bytes(b'adapter test')
            score = analyze_audio(audio, adapters)
        self.assertEqual(score.score.canonical_text, 'カキク')
        self.assertEqual(score.observations.readings[0].kana, 'カキク')
        self.assertTrue(any(e.kind == 'lyric-phonetic-fallback'
                            for e in score.observations.evidence))
        self.assertTrue(any(link.singing_unit_ids for link in score.observations.links))

    def test_supplied_lyrics_never_trigger_phonetic_fallback(self):
        adapters = AudioAdapters(
            lambda _p, lines: tuple(ReadingSelection('ソラ', 'test', .9) for _ in lines),
            lambda _p, _l, _r: (AlignedMora(0, 0, 'ソ', 0, .2, .9),
                                AlignedMora(0, 1, 'ラ', .2, .4, .9)),
            lambda _: (MelodyNote(0, .4, 60),),
            lambda _: (LyricLine('空', 0, .4),),
            phonetic_recognizer=lambda *_: self.fail('supplied lyrics must remain authoritative'))
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / 'input.wav'
            audio.write_bytes(b'adapter test')
            score = analyze_audio(audio, adapters, lyrics=('空',))
        self.assertEqual(score.score.canonical_text, '空')
