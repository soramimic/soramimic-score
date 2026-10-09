from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import patch
import importlib.util
import json

from soramimic_score import ReadingSelection
from soramimic_score.symbol_readings import (symbol_reading_proposals, symbol_slots,
                                           refine_symbol_reading)


class SymbolProposalTests(TestCase):
    def slot(self, start=5, end=5):
        return {"start": 3, "end": 4, "surface": "♡", "mapped": True,
                "kana_start": start, "kana_end": end}

    def test_audio_can_supply_a_name_absent_from_the_symbol_dictionary(self):
        base = "アシタモキミトアルコウ"
        wanted = "アシタモステキキミトアルコウ"
        choices, evidence = symbol_reading_proposals(base, (self.slot(4, 4),), {
            "mix": wanted, "vocals": wanted})
        self.assertEqual(choices, (wanted,))
        self.assertEqual(evidence[0]["reading"], "ステキ")

    def test_conflicting_missing_and_ambiguous_anchors_abstain(self):
        base = "アシタモキミトアルコウ"
        heard = "アシタモステキキミトアルコウ"
        for views in ({"mix": heard}, {"mix": heard, "vocals": base},
                      {"mix": heard, "vocals": "ナンノカンケイモナイ"},
                      {"mix": heard + base, "vocals": heard + base}):
            with self.subTest(views=views):
                choices, _ = symbol_reading_proposals(base, (self.slot(4, 4),), views)
                self.assertEqual(choices, ())

    def test_replaces_only_the_symbol_interval(self):
        base = "アシタモハートキミトアルコウ"
        wanted = "アシタモステキキミトアルコウ"
        choices, _ = symbol_reading_proposals(base, (self.slot(4, 7),), {
            "mix": wanted, "vocals": wanted})
        self.assertEqual(choices, (wanted,))

    def test_line_edges_do_not_turn_unrelated_context_into_a_reading(self):
        choices, evidence = symbol_reading_proposals("キミトアルコウ", (self.slot(0, 0),), {
            "mix": "ステキキミトアルコウ", "vocals": "ステキキミトアルコウ"})
        self.assertEqual(choices, ())
        self.assertEqual(evidence[0]["reason"], "insufficient-anchors")

    def test_explicit_ruby_is_not_reinterpreted(self):
        span = SimpleNamespace(start=1, end=2)
        yomi = SimpleNamespace(get_symbol_spans=lambda _: (span,))
        with patch.dict("sys.modules", {"soramimic_yomi": yomi}):
            self.assertEqual(symbol_slots("｜♡《すてき》", "ステキ"), ())

    def test_unmappable_span_remains_explicit(self):
        span = SimpleNamespace(start=1, end=2, to_dict=lambda: {
            "start": 1, "end": 2, "surface": "♡", "readings": ("", "ハート")})
        yomi = SimpleNamespace(get_symbol_spans=lambda _: (span,), get_yomi=lambda _: "フメイ")
        with patch.dict("sys.modules", {"soramimic_yomi": yomi}):
            slots = symbol_slots("空♡雲", "ソラクモ")
        self.assertFalse(slots[0]["mapped"])
        self.assertEqual(slots[0]["surface"], "♡")


@skipUnless(importlib.util.find_spec("soramimic_yomi"), "Yomi unavailable")
class LexicalSymbolTests(TestCase):
    text = "あしたも🧭きみとあるこう"
    base = "アシタモキミトアルコウ"
    recognized = "あしたも冒険きみとあるこう"
    spoken = "アシタモボーケンキミトアルコウ"

    def refine(self, kana, lexical):
        initial = ReadingSelection(self.base, "dictionary", 1, (self.base,))
        return refine_symbol_reading(self.text, initial, kana, lexical)

    def test_lexical_proposal_needs_independent_phonetic_support(self):
        result = self.refine({"mix": "認識不能", "vocals": self.spoken},
                             {"mix": self.recognized, "vocals": self.recognized})
        self.assertEqual(result.kana, self.spoken)
        self.assertIn(self.spoken, result.candidates)
        self.assertEqual(result.source, "whisper+kana-whisper")
        json.dumps(result.detail, allow_nan=False)

    def test_missing_or_conflicting_phonetic_support_cannot_invent_a_reading(self):
        for kana, lexical in (
            ({"mix": "", "vocals": ""}, {"mix": self.recognized, "vocals": self.recognized}),
            ({"mix": self.base, "vocals": self.spoken},
             {"mix": self.recognized, "vocals": self.recognized}),
        ):
            with self.subTest(kana=kana, lexical=lexical):
                self.assertEqual(self.refine(kana, lexical).kana, self.base)

    def test_different_models_and_audio_views_can_confirm_a_novel_word(self):
        result = self.refine({"mix": "", "vocals": self.spoken},
                             {"mix": self.recognized, "vocals": "ご視聴ありがとうございました"})
        self.assertEqual(result.kana, self.spoken)

    def test_competing_recognizer_words_remain_separate_candidates(self):
        result = self.refine({"mix": "アシタモステキキミトアルコウ",
                              "vocals": "アシタモムテキキミトアルコウ"},
                             {"mix": "あしたも素敵きみとあるこう",
                              "vocals": "あしたも無敵きみとあるこう"})
        self.assertIn(result.kana, ("アシタモステキキミトアルコウ",
                                   "アシタモムテキキミトアルコウ"))
        row = result.detail["symbol_refinement"][0]
        self.assertIn("ステキ", row["candidates"])
        self.assertIn("ムテキ", row["candidates"])
        self.assertEqual({x["reading"] for x in row["candidate_scores"]} & {"ステキ", "ムテキ"},
                         {"ステキ", "ムテキ"})

    def test_two_contexts_of_one_audio_view_are_not_two_views(self):
        result = self.refine({},
                             {"mix": self.recognized, "mix:full-context": self.recognized})
        self.assertEqual(result.kana, self.base)

    def test_whole_sequence_and_two_models_can_work_without_a_vocal_stem(self):
        result = self.refine({"mix": self.spoken}, {"mix": self.recognized})
        self.assertEqual(result.kana, self.spoken)

    def test_different_models_can_confirm_a_conventional_name(self):
        initial = ReadingSelection(self.base, "dictionary", 1, (self.base,))
        heard = "アシタモハートキミトアルコウ"
        result = refine_symbol_reading("あしたも♡きみとあるこう", initial,
            {"mix": heard, "vocals": ""}, {"mix": "", "vocals": "あしたもハートきみとあるこう"})
        self.assertEqual(result.kana, heard)

    def test_a_clear_conflicting_vote_blocks_cross_model_support(self):
        result = self.refine({"mix": self.base, "vocals": self.spoken},
                             {"mix": self.recognized, "vocals": ""})
        self.assertEqual(result.kana, self.base)

    def test_unrelated_recognition_and_function_words_are_not_novel_symbol_names(self):
        for heard in ("ご視聴ありがとうございました", "あしたもはずきみとあるこう"):
            result = self.refine({"mix": self.spoken, "vocals": self.spoken},
                                 {"mix": heard, "vocals": heard})
            self.assertEqual(result.kana, self.base)

    def test_explicit_ruby_cannot_be_changed_by_recognition(self):
        initial = ReadingSelection("アシタモミチキミトアルコー", "explicit-ruby", 1,
                                   ("アシタモミチキミトアルコー",))
        result = refine_symbol_reading("あしたも｜🧭《みち》きみとあるこう", initial,
                                       {"mix": self.spoken, "vocals": self.spoken},
                                       {"mix": self.recognized, "vocals": self.recognized})
        self.assertEqual(result, initial)

    def test_two_dropped_spans_cannot_both_claim_the_same_audio_interval(self):
        slots = symbol_slots("空♡ ♡雨", "ソラアメ")
        self.assertEqual(len(slots), 2)
        self.assertFalse(any(slot["mapped"] for slot in slots))
