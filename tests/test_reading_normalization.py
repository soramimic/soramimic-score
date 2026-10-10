from itertools import product
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from soramimic_score import (
    AlignedMora, AudioAdapters, AudioPipelineError, LyricLine, LyricSpan,
    MelodyNote, ReadingCandidate, ReadingSelection, analyze_audio,
    build_audio_observations, build_known_lyrics_document, kana_to_moras,
    kana_to_syllables, normalize_reading, phonemes_for_mora, spans_from_ruby_text,
)
from soramimic_score.audio import _validate_readings
from soramimic_score.japanese import simple_kana_reading


# Readings after G2P, not arbitrary lyric spelling. Keep the expected result
# explicit so acceptance changes cannot silently turn into character deletion.
NORMALIZATION_CASES = {
    "halfwidth-voicing": ("ｶﾞｯﾂﾎﾟｰｽﾞ", "ガッツポーズ"),
    "halfwidth-small-run": ("ｳｫｫｫ", "ウォォォ"),
    "decomposed-hiragana": ("か\u3099は\u309aう\u3099", "ガパヴ"),
    "spacing-voicing": ("カ゛ハ゜", "ガパ"),
    "combining-wa": ("わ\u3099", "ヴァ"),
    "historical-kana": ("ゐゑヰヱ", "イエイエ"),
    "voiced-historical-kana": ("ヷヸヹヺ", "ヴァヴィヴェヴォ"),
    "unvoiced-iteration": ("くゝゝ", "ククク"),
    "voiced-iteration": ("すゞき", "スズキ"),
    "iteration-voicing-reset": ("ガヽハヾパヽ", "ガカハバパハ"),
    "compatibility-ligature": ("ゟヿ", "ヨリコト"),
    "whitespace": (" \tカ\u3000ナ\n", "カナ"),
    "mixed-small-kana": ("ァィゥキャァキュゥゥ", "ァィゥキャァキュゥゥ"),
    "leading-and-repeated-specials": ("ーーンッッアーー", "ーーンッッアーー"),
    "foreign-combinations": ("ティファヴァクヮシェ", "ティファヴァクヮシェ"),
}

REJECTED_READINGS = {
    "empty": "", "whitespace-only": " \t\n\u3000",
    "mixed-symbol": "カ★ナ", "mixed-emoji": "カ👩‍🎤ナ",
    "mixed-latin": "カABCナ", "mixed-digits": "カ123ナ",
    "mixed-kanji": "カ漢字ナ", "punctuation-in-reading": "カ、ナ",
    "zero-width-space": "カ\u200bナ", "bom": "\ufeffカナ",
    "variation-selector": "カ\ufe0fナ", "bidi-control": "カ\u202eナ",
    "nul": "カ\x00ナ", "escape": "カ\x1bナ", "surrogate": "カ\ud800ナ",
    "orphan-dakuten": "\u3099", "orphan-handakuten": "゜",
    "impossible-dakuten": "ヌ\u3099", "duplicate-dakuten": "ガ\u3099",
    "impossible-handakuten": "カ\u309a", "orphan-iteration": "ヽカ",
    "orphan-voiced-iteration": "ヾ", "unvoiceable-iteration": "アヾ",
    "iteration-after-special": "カーヽ", "iteration-after-small": "キャヽ",
    "ainu-extension": "カㇰ", "hentaigana": "カ\U0001b002",
    "unexpanded-kanji-iteration": "カ々", "emoji-only": "😀",
}


class ReadingNormalizationTests(unittest.TestCase):
    def assert_pronounceable(self, normalized):
        moras = kana_to_moras(normalized)
        self.assertTrue(moras)
        self.assertEqual("".join(moras), normalized)
        self.assertEqual("".join(kana_to_syllables(normalized)), normalized)
        self.assertEqual(normalize_reading(normalized), normalized)
        for mora in moras:
            self.assertTrue(phonemes_for_mora(mora))

    def test_named_normalization_cases(self):
        for name, (raw, expected) in NORMALIZATION_CASES.items():
            with self.subTest(case=name):
                self.assertEqual(normalize_reading(raw), expected)
                self.assert_pronounceable(expected)

    def test_unknown_characters_are_not_silently_removed(self):
        for name, raw in REJECTED_READINGS.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    normalize_reading(raw)
        for value in (None, 123, b"kana"):
            with self.subTest(type=type(value).__name__):
                with self.assertRaisesRegex(ValueError, "string"):
                    normalize_reading(value)

    def test_kana_unicode_blocks_never_accept_unpronounceable_output(self):
        ranges = ((0x3040, 0x3100), (0x31F0, 0x3200), (0xFF61, 0xFFA0),
                  (0x1B000, 0x1B170))
        for start, end in ranges:
            for codepoint in range(start, end):
                for raw in (chr(codepoint), "カ" + chr(codepoint) + "ァー"):
                    with self.subTest(input=ascii(raw)):
                        try:
                            normalized = normalize_reading(raw)
                        except ValueError:
                            continue
                        self.assert_pronounceable(normalized)

    def test_small_kana_pairs_and_long_runs_preserve_every_character(self):
        for prefix, first, second in product(
            ("", "ア", "キャ", "ファ", "ン", "ッ", "ー"),
            "ァィゥェォャュョヮヵヶ", "ァィゥェォャュョヮヵヶ",
        ):
            raw = prefix + first + second + "ー"
            with self.subTest(input=raw):
                self.assertEqual(normalize_reading(raw), raw)
                self.assert_pronounceable(raw)
        raw = "ウォ" + "ォ" * 1024 + "ー" * 1024
        self.assertEqual(normalize_reading(raw), raw)
        self.assertEqual(len(kana_to_moras(raw)), 2049)

    def test_kana_only_fallback_does_not_hide_unreadable_words(self):
        self.assertEqual(simple_kana_reading("カナ、カナ！")[0].kana, "カナカナ")
        for raw in ("空カナ", "カABCナ", "カ★ナ", "カ😀ナ"):
            with self.subTest(input=raw):
                self.assertEqual(simple_kana_reading(raw), ())
                with self.assertRaisesRegex(ValueError, "no pronunciation"):
                    spans_from_ruby_text(raw)

    def test_direct_alignment_normalizes_all_candidates_and_keeps_surface(self):
        surface = "声😀"
        candidates = (ReadingCandidate("ｶﾞ", "first", 1),
                      ReadingCandidate("ヷ", "second", .5))
        document = build_known_lyrics_document(surface, (LyricSpan(surface, (0, 2), candidates),))
        self.assertEqual(document.canonical_text, surface)
        self.assertEqual([r.kana for r in document.readings], ["ガ", "ヴァ"])
        self.assertTrue(all(m.phoneme_ids for m in document.moras))
        bad = (*candidates, ReadingCandidate("カ★ナ", "bad-alternative", .1))
        with self.assertRaisesRegex(ValueError, "U\\+2605"):
            build_known_lyrics_document(surface, (LyricSpan(surface, (0, 2), bad),))

    def test_selected_and_alternative_readings_are_normalized_together(self):
        original = ReadingSelection("ｶﾞ", "test", .9, ("か\u3099", "ガ", "ヷ"), {"source": "kept"})
        selected, = _validate_readings((LyricLine("声"),), (original,))
        self.assertEqual(selected.kana, "ガ")
        self.assertEqual(selected.candidates, ("ガ", "ヴァ"))
        self.assertEqual(selected.detail["source"], "kept")
        self.assertEqual(selected.detail["reading_normalization"]["selected_before"], "ｶﾞ")
        self.assertEqual(_validate_readings((LyricLine("声"),), (selected,)), (selected,))
        self.assertEqual(original.kana, "ｶﾞ")
        for item in (ReadingSelection("ガ", "test", 1, ("ガ", "ヌ\u3099")),
                     ReadingSelection("ガ", "test", 1, ("カ",))):
            with self.subTest(candidates=item.candidates):
                with self.assertRaises(AudioPipelineError):
                    _validate_readings((LyricLine("声"),), (item,))


class AudioReadingBoundaryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.audio = Path(temp.name) / "input.wav"
        self.audio.write_bytes(b"synthetic adapter boundary")

    def adapters(self, reading, align):
        return AudioAdapters(
            lambda *_: (ReadingSelection(reading, "test", 1),), align,
            lambda _: (MelodyNote(0, .6, 60, confidence=.9),),
            lambda _: (LyricLine("声😀", 0, .6),),
        )

    def test_normalization_precedes_alignment_and_preserves_lyric_surface(self):
        for supplied in (False, True):
            seen = []
            def align(_path, _lines, selected):
                seen.append(selected[0].kana)
                self.assertEqual(selected[0].kana, "ガイ")
                return (AlignedMora(0, 0, "ガ", 0, .3, .9),
                        AlignedMora(0, 1, "イ", .3, .6, .9))
            with self.subTest(supplied=supplied):
                score = analyze_audio(self.audio, self.adapters("ｶﾞゐ", align),
                                      lyrics=("声😀",) if supplied else None)
                self.assertEqual(seen, ["ガイ"])
                self.assertEqual(score.score.canonical_text, "声😀")
                evidence = next(e for e in score.observations.evidence
                                if e.kind == "reading-selection")
                self.assertEqual(evidence.detail["reading_normalization"]["selected_before"], "ｶﾞゐ")

    def test_mixed_invalid_reading_fails_before_calling_aligner(self):
        for supplied, raw in product((False, True), REJECTED_READINGS.values()):
            align = Mock(side_effect=AssertionError("invalid reading reached the aligner"))
            with self.subTest(supplied=supplied, input=ascii(raw)):
                with self.assertRaises(AudioPipelineError) as raised:
                    analyze_audio(self.audio, self.adapters(raw, align),
                                  lyrics=("声😀",) if supplied else None)
                self.assertEqual(raised.exception.stage, "readings")
                self.assertIn("line 1:", str(raised.exception))
                align.assert_not_called()

    def test_aligner_output_cannot_smuggle_unresolved_characters(self):
        with self.assertRaisesRegex(AudioPipelineError, "mora alignment.*U\\+2605"):
            build_audio_observations(
                (LyricLine("声"),), (ReadingSelection("カナ", "test", 1),),
                (AlignedMora(0, 0, "カ★ナ", 0, .6, .9),),
                (MelodyNote(0, .6, 60),))


if __name__ == "__main__":
    unittest.main()
