import unittest

from soramimic_score.mora_gap import align_mora_gap


class MoraGapTests(unittest.TestCase):
    def test_whole_sequence_locates_an_unlisted_reading(self):
        result = align_mora_gap("アシタモ", "キミトアルコウ", "キョウハアシタモボーケンキミトアルコウネ")
        self.assertEqual(result["readings"], ["ボーケン"])
        self.assertEqual(result["reason"], "whole-context-mora-alignment")
        start, end = result["spans"][0]
        self.assertEqual(end - start, 4)
        self.assertTrue(result["alignments"][0]["left"])
        self.assertTrue(result["alignments"][0]["right"])

    def test_silent_symbol_has_an_empty_aligned_interval(self):
        result = align_mora_gap("アシタモ", "キミトアルコウ", "アシタモキミトアルコウ")
        self.assertEqual(result["readings"], [""])

    def test_distant_words_resolve_a_repeated_local_context(self):
        result = align_mora_gap("アカイソラニキミト", "アルコウシロイクモ",
            "アオイソラニキミトナミアルコウクロイクモアカイソラニキミトユメアルコウシロイクモ")
        self.assertEqual(result["readings"], ["ユメ"])

    def test_unrelated_credits_and_missing_sides_abstain(self):
        for left, right, heard in (("アシタモ", "キミトアルコウ", "ゴシチョウアリガトウゴザイマシタ"),
                                   ("", "キミトアルコウ", "ユメキミトアルコウ"),
                                   ("アシタモ", "", "アシタモユメ")):
            with self.subTest(heard=heard):
                self.assertEqual(align_mora_gap(left, right, heard)["readings"], [])
