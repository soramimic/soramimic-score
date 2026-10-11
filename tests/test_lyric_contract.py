"""Public-boundary and round-trip tests using synthetic spelling only."""
from dataclasses import FrozenInstanceError, replace
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest

from soramimic_score import (
    AlignedMora, AudioAdapters, MelodyNote, ReadingSelection, analyze_audio,
    LyricInput, LyricInputError, LyricMappingError, LyricSpan, ReadingCandidate,
    ScoreDocument, ValidatedReading, ValidationError, build_known_lyrics_document,
    compile_score, kana_to_moras, kana_to_syllables, parse_lyric_input, validate_reading,
)
from soramimic_score.japanese import strip_ruby
from soramimic_score.parenthetical import parentheses
from tests.test_reading_normalization import NORMALIZATION_CASES, REJECTED_READINGS


class LyricInputContractTests(unittest.TestCase):
    def test_whole_file_input_survives_analysis_and_json_roundtrip(self):
        original = "\r\n カ \t\r\n\r\n|声《こえ》\n"
        supplied = parse_lyric_input(original)
        adapters = AudioAdapters(
            lambda _p, _lines: (ReadingSelection("カコエ", "synthetic", 1.),),
            lambda _p, _lines, _readings: tuple(
                AlignedMora(0, i, kana, i * .3, (i + 1) * .3, .9)
                for i, kana in enumerate("カコエ")),
            lambda _p: tuple(MelodyNote(i * .3, (i + 1) * .3, 60) for i in range(3)),
            audio_duration=lambda _p: 1.,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.wav"
            path.touch()
            result = analyze_audio(path, adapters, lyrics=supplied)
        restored = ScoreDocument.from_json(result.to_json())
        receipt = next(item.detail["input"] for item in restored.observations.evidence
                       if item.kind == "lyric-input" and item.detail["stage"] == "supplied")
        self.assertEqual(receipt["original_text"], original)
        self.assertEqual(restored.score.canonical_text, " カ \t\n声")
        self.assertEqual([row["included"] for row in receipt["line_spans"]], [False, True, False, True])
        self.assertEqual(supplied.lyric_lines, (" カ \t", "|声《こえ》"))

    def test_normalization_maps_every_character_to_the_unchanged_source(self):
        raw = " \tｶﾞか\u3099Ａ１\u3000ヷ\r\n声"
        parsed = parse_lyric_input(raw)
        self.assertEqual(parsed.original, raw)
        self.assertEqual(parsed.text, "  ガがA1 ヴァ\n声")
        self.assertEqual(parsed.source_spans, (
            (0, 1), (1, 2), (2, 4), (4, 6), (6, 7), (7, 8), (8, 9),
            (9, 10), (9, 10), (10, 12), (12, 13),
        ))
        self.assertEqual(json.loads(json.dumps(parsed.to_dict()))["original_text"], raw)
        self.assertEqual(parsed.original_span(2, 4), (2, 6))
        with self.assertRaisesRegex(ValueError, "splits one source"):
            parsed.original_boundary(8)
        with self.assertRaises(FrozenInstanceError):
            parsed.text = "改変"
        with self.assertRaises(ValueError):
            replace(parsed, text="改変")

    def test_ruby_and_parenthetical_structures_keep_original_coordinates(self):
        raw = "ｶﾞ｜声《ｺｴ》\u3000夢（ゆめ）"
        parsed = parse_lyric_input(raw)
        ruby, parenthetical = parsed.annotations
        self.assertEqual((ruby.kind, ruby.reading), ("ruby", "コエ"))
        self.assertEqual(raw[slice(*ruby.source_surface_span)], "声")
        self.assertEqual(raw[slice(*ruby.source_content_span)], "ｺｴ")
        self.assertEqual(raw[slice(*ruby.source_span)], "｜声《ｺｴ》")
        self.assertEqual(parenthetical.kind, "parenthetical")
        self.assertIsNone(parenthetical.reading)  # Syntax does not decide what is sung.
        self.assertEqual(raw[slice(*parenthetical.source_span)], "（ゆめ）")
        self.assertEqual(strip_ruby(raw), "ｶﾞ声\u3000夢（ゆめ）")
        self.assertEqual(strip_ruby("|声《こえ》"), "声")
        self.assertEqual(strip_ruby(r"\｜声《こえ》"), r"\｜声《こえ》")

    def test_width_voicing_whitespace_and_brackets_combine_without_offset_drift(self):
        for base, gap, pair, reading in product(
            ("ｶﾞ", "カ゛", "か\u3099", "ガ"), (" ", "\t", "\u3000", " \t　"),
            (("(", ")"), ("（", "）"), ("[", "]"), ("［", "］")),
            ("ゆめ", "ユメ", "ﾕﾒ"),
        ):
            raw = base + gap + "夢" + pair[0] + reading + pair[1] + "声"
            with self.subTest(raw=raw):
                parsed = parse_lyric_input(raw)
                self.assertEqual(parse_lyric_input(parsed.text).text, parsed.text)
                self.assertEqual(parsed.original, raw)
                span, = parentheses(raw)
                self.assertEqual(raw[span.base_start:span.base_end], "夢")
                self.assertEqual(raw[span.start:span.end], pair[0] + reading + pair[1])
                self.assertEqual(span.reading, "ユメ")
                self.assertEqual(parsed.match_parts((base, "夢", pair[0] + reading + pair[1], "声"))[-1],
                                 (len(raw) - 1, len(raw)))

    def test_invalid_characters_and_syntax_report_original_positions(self):
        for prefix, bad in product(("", "ｶﾞ", "か\u3099\n"), ("\0", "\u202e", "\ud800", "\ue000")):
            raw = prefix + bad + "声"
            with self.subTest(raw=ascii(raw)), self.assertRaises(LyricInputError) as raised:
                parse_lyric_input(raw)
            self.assertEqual(raised.exception.position, len(prefix) + 1)
        for malformed in ("｜声", "｜《こえ》", "｜声《》", "｜声《こえ", "｜声《こ\nえ》"):
            with self.subTest(malformed=malformed), self.assertRaises(LyricInputError):
                parse_lyric_input(malformed)
        for newline in ("\r", "\r\n", "\n", "\u2028", "\u2029"):
            with self.subTest(newline=repr(newline)), self.assertRaises(LyricInputError) as raised:
                parse_lyric_input("声" + newline + "ｶﾞ\0")
            self.assertEqual((raised.exception.line, raised.exception.column), (2, 3))
        with self.assertRaises(LyricInputError) as raised:
            parse_lyric_input("ｶﾞ｜声《カ★ナ》")
        self.assertEqual(raised.exception.position, 7)
        for mixed in ("夢（ゆめ)", "夢[ゆめ］"):
            self.assertEqual(parentheses(mixed), ())
            self.assertEqual(parse_lyric_input(mixed).annotations, ())

    def test_complete_consumption_is_not_the_same_as_an_empty_boundary_list(self):
        parsed = LyricInput("\tか\u3099声\u3000Ａ１ ")
        self.assertEqual(parsed.match_parts(("が声", "A1")), ((1, 4), (5, 7)))
        for parts, reason in (((), "unconsumed-input"), (("が",), "unconsumed-input"),
                              (("が", "A1"), "surface-mismatch"),
                              (("声", "がA1"), "surface-mismatch"),
                              (("が声", "A1", "増"), "surface-mismatch"),
                              (("",), "empty-part")):
            with self.subTest(parts=parts), self.assertRaises(LyricMappingError) as raised:
                parsed.match_parts(parts)
            self.assertEqual(raised.exception.reason, reason)
        with self.assertRaises(LyricMappingError) as raised:
            LyricInput("ヷ声").match_parts(("ヴ", "ァ声"))
        self.assertEqual(raised.exception.reason, "unsafe-normalization-boundary")


class PronunciationContractTests(unittest.TestCase):
    def test_optional_acoustic_evidence_cannot_salvage_a_partial_reading(self):
        from soramimic_score import LyricLine
        from soramimic_score.local_recovery import expand_repeated_vocalization_from_kana
        from soramimic_score.readings import _comparison_kana, select_acoustic_reading

        with self.assertRaisesRegex(ValueError, "unsupported character"):
            _comparison_kana("ア★シタ")
        selection = select_acoustic_reading(("アス", "アシタ"), {"mix": "ア★シタ"})
        self.assertEqual(selection.kana, "アス")
        self.assertIn("position 2", selection.detail["invalid_transcripts"]["mix"])
        with self.assertRaisesRegex(ValueError, "unsupported character"):
            select_acoustic_reading(("ア★ス",), {})
        notes = tuple(MelodyNote(i / 3, (i + 1) / 3, 60) for i in range(6))
        with self.assertRaisesRegex(ValueError, "unsupported character"):
            expand_repeated_vocalization_from_kana(LyricLine("ララ", 0., 2.), "ララ★ララララ", notes)

    def test_model_vocabulary_separators_are_not_treated_as_readings(self):
        from soramimic_score.models import _ctc_kana_aliases

        self.assertEqual(_ctc_kana_aliases({"<pad>": 0, "ア": 1, "あ": 2, "|": 3,
                                           " ": 4, "★": 5, "ン": 6}),
                         {"ア": [1, 2], "ン": [6]})

    def test_no_public_segmentation_entry_point_can_drop_invalid_characters(self):
        for name, raw in REJECTED_READINGS.items():
            for entry in (kana_to_moras, kana_to_syllables, ValidatedReading, validate_reading):
                with self.subTest(name=name, entry=entry.__name__), self.assertRaises(ValueError):
                    entry(raw)

    def test_typed_readings_and_direct_calls_consume_the_same_complete_input(self):
        for name, (raw, expected) in NORMALIZATION_CASES.items():
            with self.subTest(name=name):
                parsed = ValidatedReading(raw)
                self.assertEqual(parsed.original, raw)
                self.assertEqual(parsed.kana, expected)
                self.assertEqual("".join(parsed.moras), expected)
                self.assertEqual(kana_to_moras(parsed), kana_to_moras(raw))
                self.assertEqual("".join(kana_to_syllables(raw)), expected)
                with self.assertRaises(FrozenInstanceError):
                    parsed.kana = "カ★ナ"

    def test_json_readings_moras_and_synthesis_cannot_bypass_validation(self):
        original = build_known_lyrics_document(
            "声", (LyricSpan("声", (0, 1), (ReadingCandidate("カ", "synthetic", 1.),)),))
        for bad in ("カ★", "ｶ", "カカ"):
            raw = original.to_dict()
            raw["readings"][0]["kana"] = bad
            raw["moras"][0]["text"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                type(original).from_dict(raw)
        document = compile_score(original)
        raw = document.to_dict()
        raw["score"]["canonical"][0]["kana"] = "キ"
        with self.assertRaisesRegex(ValidationError, "do not match observations"):
            ScoreDocument.from_dict(raw)
        raw["score"]["canonical"][0]["kana"] = "カ★"
        with self.assertRaisesRegex(ValueError, "unsupported character"):
            ScoreDocument.from_dict(raw)


if __name__ == "__main__":
    unittest.main()
