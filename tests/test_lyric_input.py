import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from soramimic_score import (
    AlignedMora, AudioAdapters, LyricLine, MelodyNote, ReadingCandidate,
    analyze_audio, normalize_lyric_input, spans_from_ruby_text,
)
from soramimic_score.audio import AudioPipelineError, _validate_lines
from soramimic_score.japanese import kana_to_moras
from soramimic_score.readings import dictionary_candidates, dictionary_readings, token_reading_proposals
from soramimic_score.ruby import ruby_segments
from soramimic_score.symbol_readings import _lexical_transcript, symbol_slots


class LyricInputTests(unittest.TestCase):
    def test_normalization_is_limited_and_idempotent(self):
        cases = (
            ("か\u3099くせい", "がくせい"), ("カ゛クセイ", "ガクセイ"),
            ("は\u309aとﾎﾟｯﾎﾟ", "ぱとポッポ"), ("わ\u3099ヷ", "ヴァヴァ"),
            ("すゞき くゝゝ ガヽハヾパヽ", "すずき くくく ガカハバパハ"),
            ("ゟヿ", "よりコト"), ("ｳｫｫｫ", "ウォォォ"),
            ("今日は AI 4443 café Ａ１ ①&｜声《こえ》 👩‍🎤", "今日は AI 4443 café Ａ１ ①&｜声《こえ》 👩‍🎤"),
            (" \t歌\r\n声\u3000", " \t歌\r\n声\u3000"),
        )
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(normalize_lyric_input(source), expected)
                self.assertEqual(normalize_lyric_input(expected), expected)

    def test_rejection_positions_refer_to_the_original_input(self):
        for char in ("\0", "\x1b", "\x7f", "\x85", "\ud800", "\udfff"):
            with self.subTest(char=ascii(char)):
                with self.assertRaisesRegex(ValueError, f"position 3:.*U\\+{ord(char):04X}"):
                    normalize_lyric_input("か\u3099" + char + "くせい")
        for source, position in (("゙", 1), ("カ゜", 2), ("ヌ゙", 2), ("ガ゙", 2),
                                 ("ヽカ", 1), ("アヾ", 2), ("キャヽ", 3)):
            with self.subTest(source=source):
                with self.assertRaisesRegex(ValueError, f"position {position}"):
                    normalize_lyric_input(source)

    def test_dictionary_input_is_a_copy_with_provenance(self):
        line = LyricLine("か\u3099くせい")
        candidate = SimpleNamespace(reading="ガクセイ", to_dict=lambda: {"reading": "ガクセイ"})
        yomi = Mock(return_value=[candidate])
        with patch.dict(sys.modules, soramimic_yomi=SimpleNamespace(get_yomi_candidates=yomi)), \
                patch("soramimic_score.readings.dictionary_candidates", return_value=(("ガクセイ",),)):
            selected, = dictionary_readings(None, (line,))
        yomi.assert_called_once_with("がくせい", nbest=32)
        self.assertEqual(line.text, "か\u3099くせい")
        self.assertEqual(selected.kana, "ガクセイ")
        self.assertEqual(selected.detail["lyric_input_normalization"], {
            "original_text": line.text, "normalized_text": "がくせい",
        })

    def test_unidic_uses_the_same_normalized_input(self):
        tagger = Mock()
        tagger.nextNode.return_value = None
        with patch.dict(sys.modules, MeCab=SimpleNamespace(Tagger=lambda _: tagger),
                        unidic_lite=SimpleNamespace(DICDIR="/dictionary")), \
                patch("soramimic_score.readings._node_reading", return_value=("ガクセイ", ())):
            self.assertEqual(dictionary_candidates((LyricLine("カ゛クセイ"),)), (("ガクセイ",),))
        tagger.parseToNode.assert_called_once_with("ガクセイ")
        tagger.parseNBestInit.assert_called_once_with("ガクセイ")

    def test_invalid_input_never_reaches_yomi_or_an_alternative_dictionary(self):
        yomi, unidic = Mock(), Mock()
        with patch.dict(sys.modules, soramimic_yomi=SimpleNamespace(get_yomi_candidates=yomi)), \
                patch("soramimic_score.readings.dictionary_candidates", unidic):
            with self.assertRaisesRegex(ValueError, "position 2.*U\\+0000"):
                dictionary_readings(None, (LyricLine("カ\0ナ"),))
        yomi.assert_not_called()
        unidic.assert_not_called()

    def test_ruby_delimiters_surfaces_and_spans_are_preserved(self):
        source = "か\u3099｜声《こえ》"
        g2p = Mock(side_effect=lambda text: (ReadingCandidate({"が": "ガ", "声": "コエ"}[text], "test", 1),)
                   if text else ())
        plain, spans = spans_from_ruby_text(source, g2p)
        self.assertEqual(plain, "か\u3099声")
        self.assertEqual([span.surface_span for span in spans], [(0, 2), (2, 3)])
        self.assertEqual([span.surface for span in spans], ["か\u3099", "声"])
        self.assertEqual([call.args[0] for call in g2p.call_args_list], ["が", "声", ""])
        self.assertEqual("".join(s["text"] for s in ruby_segments(plain, "ガコエ")), plain)

    def test_symbol_offsets_use_original_text_with_normalized_context(self):
        source = "か\u3099♡空"
        span = SimpleNamespace(start=2, end=3, to_dict=lambda: {"surface": "♡", "start": 2, "end": 3})
        find = Mock(return_value=[span])
        yomi = Mock(side_effect={"が": "ガ", "空": "ソラ"}.__getitem__)
        with patch.dict(sys.modules, soramimic_yomi=SimpleNamespace(get_symbol_spans=find, get_yomi=yomi)):
            slot, = symbol_slots(source, "ガソラ")
        find.assert_called_once_with(source)
        self.assertEqual((slot["start"], slot["end"], slot["kana_start"], slot["kana_end"]), (2, 3, 1, 1))
        self.assertTrue(slot["mapped"])

    def test_token_and_lexical_proposals_normalize_before_reading(self):
        tokens = Mock(return_value=[{"surface_form": "が", "pronunciation": "ガ"}])
        yomi = Mock(return_value="ガ")
        with patch.dict(sys.modules, soramimic_yomi=SimpleNamespace(get_tokens=tokens, get_yomi=yomi)):
            token_reading_proposals("か\u3099", "ガ")
            self.assertEqual(_lexical_transcript("か\u3099")[0], "ガ")
        self.assertTrue(all(call.args[0] == "が" for call in tokens.call_args_list))
        yomi.assert_called_once_with("が")

    def test_invalid_lyrics_fail_before_model_preparation_or_custom_adapters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.wav"
            path.touch()
            with patch("soramimic_score.models.prepared_adapters") as prepare:
                with self.assertRaisesRegex(AudioPipelineError, "lyrics: line 2:.*position 3.*U\\+0000"):
                    analyze_audio(path, lyrics=["歌", "か\u3099\0くせい"], model_config=object())
            prepare.assert_not_called()
        with self.assertRaisesRegex(AudioPipelineError, "line 1:.*position 2.*U\\+0000"):
            _validate_lines((LyricLine("カ\0ナ", 0, 1),), timed=True)


@unittest.skipUnless(importlib.util.find_spec("soramimic_yomi") and importlib.util.find_spec("MeCab"),
                     "reading engines not installed")
class RealLyricInputTests(unittest.TestCase):
    def test_spacing_voicing_marks_do_not_become_symbol_slots(self):
        self.assertEqual(symbol_slots("カ゛ヽ", "ガカ"), ())
        for source, reading in (("カ゛♡空", "ガソラ"), ("ハ゜♡空", "パソラ"),
                                ("カ゛ヽ♡空", "ガカソラ")):
            with self.subTest(source=source):
                slot, = symbol_slots(source, reading)
                self.assertEqual(slot["surface"], "♡")
                self.assertEqual((slot["start"], slot["end"]), (source.index("♡"), source.index("♡") + 1))
                self.assertTrue(slot["mapped"])

    def test_real_reading_engines_keep_voicing_and_ligatures(self):
        for source, expected in (("か\u3099くせい", "ガクセイ"), ("カ゛クセイ", "ガクセイ"),
                                 ("は\u309a", "パ"), ("わ\u3099", "ヴァ"), ("ゟ", "ヨリ"),
                                 ("ｶﾞ", "ガ")):
            with self.subTest(source=source):
                selected, = dictionary_readings(None, (LyricLine(source),))
                self.assertEqual(selected.kana, expected)
        for source in ("カ\0ナ", "カ\ud800ナ"):
            with self.assertRaisesRegex(ValueError, "position 2"):
                dictionary_readings(None, (LyricLine(source),))

    def test_real_readings_reach_alignment_without_rewriting_canonical_lyrics(self):
        source = "か\u3099くせい"
        moras = kana_to_moras("ガクセイ")

        def align(_path, lines, readings):
            self.assertEqual(lines[0].text, source)
            self.assertEqual(readings[0].kana, "ガクセイ")
            return tuple(AlignedMora(0, i, mora, i * .3, (i + 1) * .3, .9)
                         for i, mora in enumerate(moras))

        adapters = AudioAdapters(
            reading_selector=dictionary_readings, mora_aligner=align,
            melody_transcriber=lambda _: tuple(MelodyNote(i * .3, (i + 1) * .3, 60)
                                                for i in range(len(moras))),
            lyric_recognizer=lambda _: (LyricLine(source, 0, 1.2),),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.wav"
            path.touch()
            for supplied in (None, [source]):
                with self.subTest(supplied=supplied):
                    document = analyze_audio(path, adapters, lyrics=supplied).to_dict()
                    self.assertEqual(document["score"]["canonical_text"], source)
                    self.assertEqual(document["score"]["canonical"][0]["kana"], "ガクセイ")


if __name__ == "__main__":
    unittest.main()
