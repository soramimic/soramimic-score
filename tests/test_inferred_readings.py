from dataclasses import replace
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from soramimic_score import LyricLine, ModelConfig, ReadingSelection
from soramimic_score.inferred_readings import (
    inferred_acoustic_windows, inferred_candidates, inferred_reading_proposals,
    inferred_reading_slots,
)
from soramimic_score.models import create_adapters
from soramimic_score.readings import dictionary_readings, select_acoustic_reading


def selection(kana="ウタウズクズク"):
    span = {"start": 3, "end": 9, "surface": "zzqzzq", "reading": "ズクズク",
            "source": "english-g2p", "rule": "spelling-model"}
    return ReadingSelection(kana, "soramimic-yomi", 1., (kana,), {
        "candidate_provenance": [{"kana": kana, "sources": ["soramimic-yomi"],
                                  "yomi_candidates": [{"inferred_spans": [span]}]}],
    })


def slot(start=3, end=7):
    return {"start": 3, "end": 9, "surface": "zzqzzq", "mapped": True,
            "kana_start": start, "kana_end": end}


class InferredReadingTests(unittest.TestCase):
    def model_config(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        for name in ("config.json", "model.safetensors", "LICENSE"):
            (root / name).touch()
        return ModelConfig(root, root)

    def test_only_model_only_origins_are_downgraded(self):
        guessed = selection()
        self.assertEqual(inferred_candidates(guessed), {guessed.kana})
        for extra in ("unidic", "yomi-dictionary"):
            detail = json.loads(json.dumps(guessed.detail))
            row = detail["candidate_provenance"][0]
            if extra == "unidic":
                row["sources"].append("unidic-lite")
            else:
                row["yomi_candidates"].append({"inferred_spans": []})
            self.assertFalse(inferred_candidates(replace(guessed, detail=detail)))
        self.assertFalse(inferred_candidates(ReadingSelection("アス", "explicit-ruby", 1.)))

    def test_surface_span_maps_without_rewriting_surrounding_kana(self):
        yomi = SimpleNamespace(get_yomi=lambda text: {"歌う ": "ウタウ", "": ""}[text])
        with patch.dict("sys.modules", {"soramimic_yomi": yomi}):
            spans = inferred_reading_slots("歌う zzqzzq", selection())
        self.assertTrue(spans[0]["mapped"])
        self.assertEqual((spans[0]["kana_start"], spans[0]["kana_end"]), (3, 7))

    def test_audio_repeat_count_wins_a_substring_tie_over_a_guess(self):
        base, heard = "ウタウズクズク", "ウタウズクズクズク"
        views = {"mix": heard, "vocals": heard}
        proposals, evidence = inferred_reading_proposals(base, (slot(),), views)
        self.assertEqual(proposals, (heard,))
        selected = select_acoustic_reading((base, *proposals), views,
                                           inferred_candidates={base})
        self.assertEqual(selected.kana, heard)
        self.assertEqual(selected.detail["reason"], "acoustic-over-inferred")
        self.assertEqual(selected.detail["distances"], [{"mix": 0., "vocals": 0.}] * 2)
        self.assertEqual(evidence[0]["reading"], "ズクズクズク")
        self.assertFalse(selected.detail["confidence_available"])
        json.dumps(selected.detail, allow_nan=False)

    def test_dictionary_tie_and_no_audio_keep_existing_behavior(self):
        base, heard = "ウタウズクズク", "ウタウズクズクズク"
        self.assertEqual(select_acoustic_reading((base, heard), {"mix": heard}).kana, base)
        selected = select_acoustic_reading((base, heard), {}, inferred_candidates={base})
        self.assertEqual(selected.kana, base)
        self.assertEqual(selected.detail["reason"], "no-acoustic-evidence")

    def test_conflicting_or_missing_audio_cannot_propose_a_replacement(self):
        for views in ({"mix": "ウタウズクズクズク", "vocals": "ウタウズクズク"},
                      {"mix": "ウタウズクズクズク", "vocals": ""}, {}):
            with self.subTest(views=views):
                proposals, _ = inferred_reading_proposals("ウタウズクズク", (slot(),), views)
                self.assertEqual(proposals, ())

    def test_internal_span_preserves_known_prefix_suffix_and_long_candidates(self):
        base = "ミンナデズクコエアワセ"
        replacement = "ズク" * 10
        heard = "マエノウタミンナデ" + replacement + "コエアワセツギノウタ"
        proposals, _ = inferred_reading_proposals(
            base, (slot(4, 6),), {"mix": heard, "vocals": heard})
        self.assertEqual(proposals, ("ミンナデ" + replacement + "コエアワセ",))

    def test_ambiguous_anchors_do_not_rewrite_a_known_word(self):
        heard = "ミンナデズクコエアワセミンナデズクズクコエアワセ"
        proposals, rows = inferred_reading_proposals(
            "ミンナデズクコエアワセ", (slot(4, 6),), {"mix": heard, "vocals": heard})
        self.assertFalse(proposals)
        self.assertEqual(rows[0]["reason"], "ambiguous-or-missing-audio")

    def test_local_audio_context_does_not_include_neighboring_lines(self):
        windows = ((0., 2.), (3., 5.), (6., 7.))
        self.assertEqual(inferred_acoustic_windows(windows, 1, 10.), ((2., 6.),))

    def test_partial_ruby_keeps_its_reading_while_unknown_english_can_use_audio(self):
        span = {"start": 0, "end": 6, "surface": "zzqzzq", "reading": "ズクズク",
                "source": "english-g2p", "rule": "spelling-model"}
        candidate = SimpleNamespace(reading="ズクズク", to_dict=lambda: {
            "reading": "ズクズク", "inferred_spans": [span]})
        yomi = SimpleNamespace(get_yomi=lambda _: "",
                               get_yomi_candidates=lambda *_args, **_kwargs: [candidate])
        text = "｜歌う《うたう》zzqzzq｜声《こえ》"
        with patch.dict("sys.modules", {"soramimic_yomi": yomi}), \
             patch("soramimic_score.readings.dictionary_candidates", side_effect=ValueError):
            base, = dictionary_readings(None, (LyricLine(text),))
            slots = inferred_reading_slots(text, base)
        self.assertEqual(base.kana, "ウタウズクズクコエ")
        self.assertEqual(inferred_candidates(base), {base.kana})
        self.assertEqual(slots[0]["surface"], text[slots[0]["start"]:slots[0]["end"]])
        proposals, _ = inferred_reading_proposals(base.kana, slots, {
            "mix": "ウタウズクズクズクコエ", "vocals": "ウタウズクズクズクコエ"})
        self.assertEqual(proposals, ("ウタウズクズクズクコエ",))

    def test_single_guessed_candidate_still_uses_kana_whisper(self):
        base, heard = "ウタウズクズク", "ウタウズクズクズク"
        config = self.model_config()
        with patch("soramimic_score.models.dictionary_readings", return_value=(selection(),)), \
             patch("soramimic_score.models.inferred_reading_slots", return_value=(slot(),)), \
             patch("soramimic_score.models.symbol_slots", return_value=()), \
             patch.dict("sys.modules", {"librosa": SimpleNamespace(get_duration=lambda **_: 10.)}), \
             patch("soramimic_score.models.transcribe_kana_views",
                   return_value={"mix": (heard,), "vocals": (heard,)}) as transcribe:
            result, = create_adapters(config, vocals_path=Path("vocals.wav")).reading_selector(
                Path("mix.wav"), (LyricLine("歌う zzqzzq", 2., 5.),))
        transcribe.assert_called_once()
        self.assertEqual(result.kana, heard)
        self.assertEqual(result.candidates, (base, heard))
        self.assertEqual(result.detail["inferred_windows_sec"], [[.5, 6.5]])

    def test_proposals_use_line_context_instead_of_a_shared_group(self):
        guessed = selection()
        heard = "ウタウズクズクズク"
        first, last = ReadingSelection("マエノウタ", "dictionary", 1., ("マエノウタ",)), \
                      ReadingSelection("ツギノウタ", "dictionary", 1., ("ツギノウタ",))
        shared = "マエノウタ" + heard + "ツギノウタ"
        def transcribe(_paths, windows, _config, **kwargs):
            self.assertIn((2., 6.), windows)
            return {"mix": tuple(heard if w == (2., 6.) else shared for w in windows),
                    "vocals": tuple(heard if w == (2., 6.) else shared for w in windows)}
        config = self.model_config()
        with patch("soramimic_score.models.dictionary_readings", return_value=(first, guessed, last)), \
             patch("soramimic_score.models.inferred_reading_slots", side_effect=[(), (slot(),), ()]), \
             patch("soramimic_score.models.symbol_slots", return_value=()), \
             patch.dict("sys.modules", {"librosa": SimpleNamespace(get_duration=lambda **_: 10.)}), \
             patch("soramimic_score.models.transcribe_kana_views", side_effect=transcribe):
            result = create_adapters(config, vocals_path=Path("vocals.wav")).reading_selector(
                Path("mix.wav"), (LyricLine("前の歌", 0., 2.), LyricLine("歌う zzqzzq", 3., 5.),
                                  LyricLine("次の歌", 6., 7.)))
        self.assertEqual([r.kana for r in result], [first.kana, heard, last.kana])
