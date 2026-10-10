from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from soramimic_score import LyricLine
from soramimic_score.readings import (dictionary_candidates, dictionary_readings,
                                     select_acoustic_reading, token_reading_proposals)


@dataclass
class Candidate:
    reading: str
    rank: int = 0
    cost: float = 0.0
    sources: tuple[str, ...] = ("canonical",)
    spans: tuple = ()

    def to_dict(self):
        return asdict(self)


class YomiReadingsTests(unittest.TestCase):
    def setUp(self):
        self.yomi = Mock(return_value=[Candidate("アス"), Candidate("アシタ", rank=1)])
        self.module = patch.dict(sys.modules, {
            "soramimic_yomi": SimpleNamespace(get_yomi_candidates=self.yomi),
        })
        self.module.start()
        self.addCleanup(self.module.stop)

    def test_automatic_reading_with_sokuon(self):
        self.yomi.return_value = [Candidate("ヤッタ")]
        with patch("soramimic_score.readings.dictionary_candidates",
                   return_value=(("ヤッタ",),)):
            selection, = dictionary_readings(None, (LyricLine("やった"),), automatic=True)
        self.assertEqual(selection.kana, "ヤッタ")

    def test_dictionary_normalizes_candidates_before_acoustic_selection(self):
        self.yomi.return_value = [Candidate("ｶﾞ"), Candidate("ヰ"),
                                 Candidate("カ★ナ"), Candidate("ヌ\u3099")]
        with patch("soramimic_score.readings.dictionary_candidates",
                   return_value=(("ガ", "エ"),)):
            selected, = dictionary_readings(None, (LyricLine("声"),))
        self.assertEqual(selected.candidates, ("ガ", "イ", "エ"))
        provenance = selected.detail["candidate_provenance"]
        self.assertEqual(provenance[0]["yomi_candidates"][0]["reading"], "ｶﾞ")
        self.assertEqual(provenance[0]["sources"], ["soramimic-yomi", "unidic-lite"])

    def test_unreadable_candidates_cannot_leave_a_partial_reading(self):
        self.yomi.return_value = [Candidate("カ★ナ"), Candidate("ヌ\u3099")]
        with patch("soramimic_score.readings.dictionary_candidates", side_effect=ValueError):
            with self.assertRaisesRegex(ValueError, "Neither"):
                dictionary_readings(None, (LyricLine("声"),))

    def test_ruby_normalizes_voicing_and_rejects_mixed_unknown_text(self):
        selected, = dictionary_readings(None, (LyricLine("｜声《ｶﾞゐ》"),))
        self.assertEqual(selected.kana, "ガイ")
        with self.assertRaisesRegex(ValueError, "kana pronunciation.*U\\+2605"):
            dictionary_readings(None, (LyricLine("｜声《カ★ナ》"),))

    @unittest.skipUnless(importlib.util.find_spec("MeCab"), "audio dependencies not installed")
    def test_alternate_word_segmentation_keeps_imayoru(self):
        candidates, = dictionary_candidates((LyricLine("二人今夜に駆け出してく"),))
        self.assertEqual(candidates[0], "フタリコンヤニカケダシテク")
        self.assertIn("フタリイマヨルニカケダシテク", candidates)
        choice = select_acoustic_reading(candidates, {
            "mix": "フタリイマヨルニカケダシテク",
            "vocals": "フタリイマヨルニカケダシテク",
        })
        self.assertEqual(choice.kana, "フタリイマヨルニカケダシテク")

    @unittest.skipUnless(importlib.util.find_spec("MeCab"), "audio dependencies not installed")
    def test_automatic_candidates_retain_imayoru_and_spoken_symbol(self):
        self.yomi.return_value = [Candidate("フタリコンヤニカケダシテク")]
        selected, = dictionary_readings(None, (LyricLine("二人今夜に駆け出してく"),),
                                        automatic=True)
        self.assertIn("フタリイマヨルニカケダシテク", selected.candidates)
        self.yomi.return_value = [Candidate("ウタアイ"), Candidate("ウタアンドアイ", rank=1)]
        selected, = dictionary_readings(None, (LyricLine("歌&愛"),), automatic=True)
        self.assertIn("ウタアンドアイ", selected.candidates)

    def test_yomi_first_with_unidic_alternatives_and_both_origins_on_duplicates(self):
        with patch("soramimic_score.readings.dictionary_candidates",
                   return_value=(("アシタ", "ミョーニチ", "アス"),)):
            selection, = dictionary_readings(None, (LyricLine("明日"),))
        self.yomi.assert_called_once_with("明日", nbest=32)
        self.assertEqual(selection.kana, "アス")
        self.assertEqual(selection.source, "soramimic-yomi")
        self.assertEqual(selection.candidates, ("アス", "アシタ", "ミョーニチ"))
        provenance = selection.detail["candidate_provenance"]
        self.assertEqual(provenance[0]["sources"], ["soramimic-yomi", "unidic-lite"])
        self.assertEqual(provenance[1]["yomi_candidates"][0]["rank"], 1)
        self.assertEqual(provenance[2]["sources"], ["unidic-lite"])
        json.dumps(selection.detail, allow_nan=False)

    def test_unidic_unreadable_line_does_not_discard_yomi(self):
        self.yomi.return_value = [Candidate("ハロー")]
        with patch("soramimic_score.readings.dictionary_candidates", side_effect=ValueError):
            selection, = dictionary_readings(None, (LyricLine("hello"),))
        self.assertEqual(selection.kana, "ハロー")
        self.assertEqual(selection.detail["unidic_status"], "no-pronunciation")

    def test_unreadable_yomi_falls_back_to_unidic_with_honest_origin(self):
        self.yomi.return_value = [Candidate(""), Candidate("unreadable")]
        with patch("soramimic_score.readings.dictionary_candidates", return_value=(("ソラ",),)):
            selection, = dictionary_readings(None, (LyricLine("空"),))
        self.assertEqual(selection.source, "unidic-lite")
        self.assertEqual(selection.detail["yomi_status"], "no-pronunciation")

    def test_no_pronunciation_is_an_error(self):
        self.yomi.return_value = [Candidate("")]
        with patch("soramimic_score.readings.dictionary_candidates", side_effect=ValueError):
            with self.assertRaisesRegex(ValueError, "Neither"):
                dictionary_readings(None, (LyricLine("…"),))

    def test_explicit_ruby_excludes_other_pronunciations(self):
        selection, = dictionary_readings(None, (LyricLine("｜明日《あした》"),))
        self.assertEqual(selection.candidates, ("アシタ",))
        self.yomi.assert_not_called()

    def test_partial_ruby_keeps_ambiguity_only_in_unannotated_text(self):
        with patch("soramimic_score.readings.dictionary_candidates", return_value=(("アス",),)):
            selection, = dictionary_readings(None, (LyricLine("明日｜空《そら》"),))
        self.assertEqual(selection.candidates, ("アスソラ", "アシタソラ"))

    def test_ruby_rejects_non_kana(self):
        with self.assertRaisesRegex(ValueError, "kana pronunciation"):
            dictionary_readings(None, (LyricLine("｜空《sky》"),))

    def test_yomi_failure_is_not_silently_replaced_by_unidic(self):
        self.yomi.side_effect = RuntimeError("engine failed")
        with patch("soramimic_score.readings.dictionary_candidates") as unidic:
            with self.assertRaisesRegex(RuntimeError, "engine failed"):
                dictionary_readings(None, (LyricLine("空"),))
        unidic.assert_not_called()

    def test_missing_yomi_dependency_does_not_silently_disable_it(self):
        with patch.dict(sys.modules, {"soramimic_yomi": None}):
            with self.assertRaises(ImportError):
                dictionary_readings(None, (LyricLine("空"),))


@unittest.skipUnless(importlib.util.find_spec("soramimic_yomi") and importlib.util.find_spec("MeCab"),
                     "audio dependencies not installed")
class RealYomiTests(unittest.TestCase):
    def test_local_dictionary_proposal_requires_kana_context(self):
        surface = "二人今夜に駆け出してく"
        default = "フタリコンヤニカケダシテク"
        selected = "フタリイマヨルニカケダシテク"
        self.assertIn(selected, token_reading_proposals(surface, default, (selected,)))
        self.assertNotIn(selected, token_reading_proposals(surface, default, (default,)))

    def test_user_dictionary_numbers_and_latin_candidates(self):
        from soramimic_yomi import get_yomi, get_yomi_candidates

        for text in ("夕焼小焼", "AI 4443", "hello world", "明日"):
            with self.subTest(text=text):
                selection, = dictionary_readings(None, (LyricLine(text),))
                self.assertEqual(selection.kana, get_yomi(text))
                self.assertEqual(selection.source, "soramimic-yomi")
                for candidate in get_yomi_candidates(text, nbest=32):
                    self.assertIn(candidate.reading, selection.candidates)
        selection, = dictionary_readings(None, (LyricLine("明日"),))
        self.assertTrue({"アス", "アシタ", "ミョーニチ"}.issubset(selection.candidates))

    def test_standard_analysis_keeps_yomi_provenance_in_json_without_acoustic_selection(self):
        import tempfile
        from soramimic_score import AlignedMora, MelodyNote, ModelConfig, analyze_audio, dump, load
        from soramimic_score.japanese import kana_to_moras
        from soramimic_score.models import create_adapters

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "input.wav"
            import wave
            with wave.open(str(audio), "wb") as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
                wav.writeframes(b"\0\0" * 1600)
            for name in ("config.json", "model.safetensors", "LICENSE"):
                (root / name).touch()
            config = ModelConfig(root, root, separate_vocals=False, acoustic_readings=False)
            adapters = create_adapters(config)
            readings_seen = []

            def align(path, lines, readings):
                readings_seen.extend(readings)
                return tuple(AlignedMora(0, i, kana, i * .1, (i + 1) * .1, .9, "test")
                             for i, kana in enumerate(kana_to_moras(readings[0].kana)))

            from dataclasses import replace
            adapters = replace(adapters, mora_aligner=align,
                               lyric_recognizer=lambda _: (LyricLine("hello", 0, 1),),
                               melody_transcriber=lambda _: (MelodyNote(0, 1, 60, "test"),))
            with patch("soramimic_score.models.create_adapters", return_value=adapters):
                result = analyze_audio(audio, model_config=config)
            self.assertEqual(readings_seen[0].source, "soramimic-yomi")
            dump(result, root / "score.json")
            restored = load(root / "score.json")
            self.assertEqual(restored.score.canonical_text, "hello")
            evidence = next(e for e in restored.observations.evidence if e.kind == "reading-selection")
            self.assertEqual(evidence.detail["selected"], "ハロー")
            self.assertEqual(evidence.detail["candidate_provenance"][0]["sources"], ["soramimic-yomi"])


if __name__ == "__main__":
    unittest.main()
