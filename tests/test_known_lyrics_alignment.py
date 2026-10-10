import json
from pathlib import Path
import unittest

from soramimic_score import (
    Boundary, Evidence, LyricSpan, ObservedSingingUnit, ReadingCandidate,
    build_known_lyrics_document, kana_to_moras, kana_to_syllables, mora_distance,
    phonemes_for_mora, spans_from_ruby_text,
)
from soramimic_score.japanese import simple_kana_reading


FIXTURE = Path(__file__).parent / "fixtures" / "known_lyrics_unobserved.json"


class JapaneseReadingTests(unittest.TestCase):
    def test_isolated_small_kana_have_full_size_phonemes(self):
        for small, full in zip("ァィゥェォャュョヮヵヶ", "アイウエオヤユヨワカケ", strict=True):
            with self.subTest(kana=small):
                self.assertEqual(phonemes_for_mora(small), phonemes_for_mora(full))
                self.assertEqual(mora_distance(small, full), 0)
        self.assertEqual(phonemes_for_mora("ぁ"), ("a",))
        self.assertEqual(kana_to_moras("ンァ"), ("ン", "ァ"))
        self.assertEqual(phonemes_for_mora("ファ"), ("f", "a"))
        self.assertEqual(phonemes_for_mora("キャ"), ("ky", "a"))
        with self.assertRaises(ValueError):
            phonemes_for_mora("★")

    def test_long_vowel_contracted_nasal_and_sokuon_are_distinct_moras(self):
        self.assertEqual(kana_to_moras("カーキャンッ"), ("カ", "ー", "キャ", "ン", "ッ"))
        self.assertEqual(phonemes_for_mora("ー", "a"), ("a",))
        self.assertEqual(phonemes_for_mora("キャ"), ("ky", "a"))
        self.assertEqual(phonemes_for_mora("ン"), ("N",))
        self.assertEqual(phonemes_for_mora("ッ"), ("q",))

    def test_repeated_small_kana_preserve_text_and_pronounce_every_mora(self):
        cases = {
            "ウォォォ": ("ウォ", "ォ", "ォ"),
            "きゅぅぅー": ("キュ", "ゥ", "ゥ", "ー"),
            "ファァァ": ("ファ", "ァ", "ァ"),
            "ァィゥ": ("ァ", "ィ", "ゥ"),
            "アヵヶ": ("ア", "ヵ", "ヶ"),
        }
        for reading, expected in cases.items():
            with self.subTest(reading=reading):
                self.assertEqual(kana_to_moras(reading), expected)
                self.assertTrue(all(phonemes_for_mora(mora) for mora in expected))
                self.assertEqual("".join(kana_to_syllables(reading)), "".join(expected))

    def test_small_kana_runs_after_supported_moras_are_pronounceable(self):
        for prefix in ("", "ア", "ウォ", "キャ", "ファ", "ン", "ッ", "ー"):
            for small in "ァィゥェォャュョヮヵヶ":
                reading = prefix + small * 3
                with self.subTest(reading=reading):
                    moras = kana_to_moras(reading)
                    self.assertEqual("".join(moras), reading)
                    self.assertTrue(all(phonemes_for_mora(mora) for mora in moras))

    def test_syllables_attach_codas_and_long_marks(self):
        self.assertEqual(kana_to_syllables("シンシュンシャンソンショー"),
                         ("シン", "シュン", "シャン", "ソン", "ショー"))
        self.assertEqual(kana_to_syllables("カッ"), ("カッ",))

    def test_isolated_particles_use_pronunciation_form(self):
        self.assertEqual(simple_kana_reading("は")[0].kana, "ワ")
        self.assertEqual(simple_kana_reading("へ")[0].kana, "エ")
        self.assertEqual(simple_kana_reading("を")[0].kana, "オ")

    def test_ruby_is_strong_but_g2p_alternative_is_retained(self):
        def g2p(surface):
            return ((ReadingCandidate("コンニチ", "synthetic-g2p", 0.7),)
                    if surface == "今日" else simple_kana_reading(surface))

        plain, spans = spans_from_ruby_text("｜今日《きょう》は", g2p)
        self.assertEqual(plain, "今日は")
        self.assertEqual(spans[0].surface_span, (0, 2))
        self.assertEqual([item.kana for item in spans[0].reading_candidates],
                         ["キョウ", "コンニチ"])
        self.assertEqual(spans[1].surface_span, (2, 3))
        self.assertEqual(spans[1].reading_candidates[0].kana, "ワ")

    def test_vowel_mismatch_costs_more_than_consonant_mismatch(self):
        self.assertGreater(mora_distance("カ", "キ"), mora_distance("カ", "ガ"))


class KnownLyricsAlignmentTests(unittest.TestCase):
    def test_repeated_small_kana_align_and_keep_alternate_readings(self):
        evidence = Evidence("e0", "synthetic-ctc", "alignment", .9)
        boundary = lambda t: Boundary(t, .9, ("e0",))
        observations = tuple(
            ObservedSingingUnit((mora,), boundary(i), boundary(i),
                               boundary(i + .8), .9, ("e0",))
            for i, mora in enumerate(("ウォ", "オ", "オ", "ー"))
        )
        span = LyricSpan("声", (0, 1), (
            ReadingCandidate("ウォォォー", "synthetic", 1),
            ReadingCandidate("ファァァー", "alternative", .5),
        ))
        document = build_known_lyrics_document("声", (span,), observations, (evidence,))
        self.assertEqual(document.canonical_text, "声")
        self.assertEqual([r.kana for r in document.readings], ["ウォォォー", "ファァァー"])
        selected = next(r for r in document.readings
                        if r.id == document.utterances[0].selected_reading_id)
        self.assertEqual(selected.kana, "ウォォォー")
        self.assertEqual([m.text for m in document.moras if m.id in selected.mora_ids],
                         ["ウォ", "ォ", "ォ", "ー"])
        self.assertTrue(all(m.phoneme_ids for m in document.moras))
        self.assertTrue(all(u.status == "observed" for u in document.singing_units))
        self.assertEqual(document.singing_units[0].consonant_start.time_sec, 0)
        self.assertEqual(document.singing_units[-1].end.time_sec, 3.8)

    def test_isolated_small_kana_keep_original_reading_and_long_vowel(self):
        span = LyricSpan("声", (0, 1), (
            ReadingCandidate("ァー", "synthetic", 1),
            ReadingCandidate("アー", "alternative", .5),
        ))
        document = build_known_lyrics_document("声", (span,))
        self.assertEqual([r.kana for r in document.readings], ["ァー", "アー"])
        self.assertEqual([m.text for m in document.moras], ["ァ", "ー", "ア", "ー"])
        self.assertEqual([p.symbol for p in document.phonemes], ["a", "a", "a", "a"])
        self.assertTrue(all(m.phoneme_ids for m in document.moras))

    def test_explicit_observation_owner_does_not_fill_silent_repeated_input(self):
        evidence = Evidence("e0", "synthetic-ctc", "alignment", .9)
        boundary = lambda t: Boundary(t, .9, ("e0",))
        observation = ObservedSingingUnit(("カ",), boundary(2), boundary(2.01),
                                          boundary(2.2), .9, ("e0",))
        spans = (LyricSpan("か", (0, 1), (ReadingCandidate("カ", "test", 1),)),
                 LyricSpan("か", (2, 3), (ReadingCandidate("カ", "test", 1),)))
        result = build_known_lyrics_document("か\nか", spans, (observation,), (evidence,),
                                             observation_span_indices=(1,))
        self.assertEqual([u.status for u in result.singing_units], ["unobserved", "observed"])
        self.assertIsNone(result.singing_units[0].consonant_start)
        self.assertEqual(result.singing_units[1].consonant_start.time_sec, 2)

    def _fixture_document(self):
        raw = json.loads(FIXTURE.read_text(encoding="utf-8"))

        def g2p(surface):
            return tuple(ReadingCandidate(**item) for item in raw["g2p"].get(surface, ()))

        canonical, spans = spans_from_ruby_text(raw["marked_text"], g2p)
        evidence = Evidence("ctc0", "synthetic-ctc", "phoneme-alignment", 0.91)
        observations = []
        for item in raw["observations"]:
            boundary = lambda key: Boundary(item[key], 0.85, ("ctc0",))
            observations.append(ObservedSingingUnit(
                tuple(item["moras"]), boundary("consonant_start_sec"),
                boundary("vowel_start_sec"), boundary("end_sec"),
                item["confidence"], ("ctc0",)))
        return build_known_lyrics_document(canonical, spans, observations, (evidence,))

    def test_fixture_retains_unobserved_canonical_unit(self):
        document = self._fixture_document()
        selected = [reading for reading in document.readings
                    if any(u.selected_reading_id == reading.id for u in document.utterances)]
        self.assertEqual([reading.kana for reading in selected], ["キョー", "モ"])
        self.assertEqual([unit.status for unit in document.singing_units],
                         ["observed", "unobserved"])
        missing = document.singing_units[-1]
        self.assertEqual([next(m.text for m in document.moras if m.id == mora_id)
                          for mora_id in missing.mora_ids], ["モ"])
        self.assertIsNone(missing.consonant_start)
        self.assertEqual(missing.confidence, 0.0)

    def test_multiple_morae_can_belong_to_one_singing_unit(self):
        document = self._fixture_document()
        first = document.singing_units[0]
        self.assertEqual(len(first.mora_ids), 2)
        self.assertEqual([m.singing_unit_ids for m in document.moras
                          if m.id in first.mora_ids], [(first.id,), (first.id,)])
        self.assertEqual(len(first.vowel_nucleus_ids), 2)

    def test_coarse_observation_is_split_at_lexical_syllable_boundaries(self):
        evidence = (Evidence("e0", "synthetic", "alignment", 0.9),)
        boundary = lambda value: Boundary(value, 0.9, ("e0",))
        observation = ObservedSingingUnit(
            ("カキ",), boundary(0), boundary(.05), boundary(.4), .9, ("e0",),
        )
        span = LyricSpan("かき", (0, 2), (ReadingCandidate("カキ", "synthetic", 1),))
        document = build_known_lyrics_document("かき", (span,), (observation,), evidence)
        moras = {mora.id: mora.text for mora in document.moras}
        self.assertEqual([tuple(moras[mora_id] for mora_id in unit.mora_ids)
                          for unit in document.singing_units], [("カ",), ("キ",)])

    def test_alternate_reading_keeps_moras_and_phonemes(self):
        document = self._fixture_document()
        alternate = next(reading for reading in document.readings if reading.kana == "キョウ")
        self.assertTrue(alternate.mora_ids)
        self.assertTrue(all(next(m for m in document.moras if m.id == mora_id).phoneme_ids
                            for mora_id in alternate.mora_ids))

    def test_vowel_evidence_overrides_a_stronger_candidate_prior(self):
        evidence = Evidence("e0", "synthetic-ctc", "alignment", 0.9)
        boundary = lambda value: Boundary(value, 0.9, ("e0",))
        observation = ObservedSingingUnit(("カ",), boundary(0), boundary(0.01),
                                          boundary(0.2), 0.9, ("e0",))
        span = LyricSpan("仮", (0, 1), (
            ReadingCandidate("カ", "candidate-a", 0.2),
            ReadingCandidate("キ", "candidate-b", 1.0),
        ))
        document = build_known_lyrics_document("仮", (span,), (observation,), (evidence,))
        selected_id = document.utterances[0].selected_reading_id
        self.assertEqual(next(r.kana for r in document.readings if r.id == selected_id), "カ")

    def test_candidate_selection_consumes_acoustic_evidence_in_order(self):
        evidence = Evidence("e0", "synthetic-ctc", "alignment", 0.9)
        boundary = lambda value: Boundary(value, 0.9, ("e0",))
        observations = (
            ObservedSingingUnit(("カ",), boundary(0), boundary(0.01), boundary(0.2), 0.9, ("e0",)),
            ObservedSingingUnit(("キ",), boundary(0.2), boundary(0.21), boundary(0.4), 0.9, ("e0",)),
        )
        candidates = (ReadingCandidate("カ", "a", 0.5), ReadingCandidate("キ", "b", 0.5))
        spans = (LyricSpan("甲", (0, 1), candidates), LyricSpan("乙", (1, 2), candidates))
        document = build_known_lyrics_document("甲乙", spans, observations, (evidence,))
        selected = [next(r.kana for r in document.readings if r.id == u.selected_reading_id)
                    for u in document.utterances]
        self.assertEqual(selected, ["カ", "キ"])

    def test_missing_audio_uses_candidate_prior_not_shorter_reading(self):
        span = LyricSpan("仮", (0, 1), (
            ReadingCandidate("カ", "short", 0.1),
            ReadingCandidate("カナ", "preferred", 0.9),
        ))
        document = build_known_lyrics_document("仮", (span,))
        selected_id = document.utterances[0].selected_reading_id
        self.assertEqual(next(r.kana for r in document.readings if r.id == selected_id), "カナ")
        self.assertTrue(all(unit.status == "unobserved" for unit in document.singing_units))

    def test_weak_observation_remains_explicit(self):
        evidence = Evidence("e0", "synthetic-ctc", "alignment", 0.4)
        boundary = lambda value: Boundary(value, 0.4, ("e0",))
        observation = ObservedSingingUnit(("ア",), boundary(0), boundary(0.02),
                                          boundary(0.2), 0.4, ("e0",))
        span = LyricSpan("あ", (0, 1), (ReadingCandidate("ア", "synthetic", 1),))
        document = build_known_lyrics_document("あ", (span,), (observation,), (evidence,))
        self.assertEqual(document.singing_units[0].status, "weak")


if __name__ == "__main__":
    unittest.main()
