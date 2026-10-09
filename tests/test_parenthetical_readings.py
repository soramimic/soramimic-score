from dataclasses import replace
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unicodedata
import unittest
from unittest.mock import patch

from soramimic_score import (AlignedMora, LyricLine, MelodyNote, ModelConfig,
                            ReadingSelection, analyze_audio, dump, load)
from soramimic_score.japanese import kana_to_moras, katakana, strip_ruby
from soramimic_score.models import create_adapters
from soramimic_score.parenthetical import (choose_reading, matching_forms, parentheses,
                                           reading_options, resolved_text)
from soramimic_score.supplied_lyrics import plan_supplied_lyrics


def fixture_readings(text):
    """A deterministic linguistic adapter; no acoustic decision is mocked here."""
    text = re.sub(r'[|｜][^《]+《([^》]+)》', lambda m: m[1], text)
    for surface, reading in {'運命': 'ウンメー', '未来': 'ミライ', '思い': 'オモイ',
                             '決意': 'ケツイ', '明日': 'アス', '誰': 'ダレ',
                             '夢': 'ユメ', '杜': 'モリ', '僕': 'ボク', '願い': 'ネガイ'}.items():
        text = text.replace(surface, reading)
    text = ''.join(c for c in katakana(text)
                   if not c.isspace() and unicodedata.category(c)[0] not in 'PZC')
    return (text,)


def fixture_selector(_path, lines, **_kwargs):
    return tuple(ReadingSelection(fixture_readings(line.text)[0], 'test-dictionary', 1.,
                                  fixture_readings(line.text)) for line in lines)


def fixture_starts(text):
    # Known morphological boundaries for the scope tests, otherwise one word.
    return {'この杜': (0, 2), '僕の願い': (0, 1, 2)}.get(text, (0,))


class ParentheticalCandidateTests(unittest.TestCase):
    def options(self, text):
        return reading_options(text, fixture_readings, word_starts=fixture_starts)

    def test_no_parenthesis_never_invokes_dictionary_or_tokenizer(self):
        def forbidden(_):
            self.fail('plain lyrics must not invoke parenthetical processing')
        for text in ('未来へ', 'hello', '', '｜未来《あす》'):
            self.assertEqual(reading_options(text, forbidden, word_starts=forbidden), ())
            self.assertEqual(matching_forms(text), (text, text))

    def test_four_bracket_forms_admit_dictionary_external_kana(self):
        for left, right in [('(', ')'), ('（', '）'), ('[', ']'), ('［', '］')]:
            with self.subTest(left=left):
                options, = self.options(f'運命{left}さだめ{right}')
                self.assertEqual({c['kana'] for c in options['choices']},
                                 {'ウンメー', 'サダメ', 'ウンメーサダメ'})
                self.assertEqual({c['mode'] for c in options['choices']},
                                 {'base', 'annotation', 'literal'})

    def test_escaped_nested_whitespace_and_explicit_ruby_are_preserved(self):
        for text in ('夢 （ゆめ）', '夢( ゆめ)', '夢(ゆめ！)', '夢(Oh)', '(ゆめ)',
                     'ゆめ(ゆめ)', '夢（ゆめ)', '夢(ゆめ', '（夢(ゆめ)）',
                     '夢((ゆめ))', r'夢\(ゆめ)', '｜夢(ゆめ)《ねがい》',
                     '｜夢《ゆめ》(ゆめ)', '夢{ゆめ}', '夢【ゆめ】'):
            with self.subTest(text=text):
                self.assertFalse(parentheses(text))

    def test_quoted_and_decorated_bases_keep_their_delimiters(self):
        for text in ('「明日(あした)へ」', '≪明日≫(あした)へ', '『明日』［あした］へ'):
            options = self.options(text)
            decisions = [choose_reading(o, {'mix': 'アシタヘ', 'vocals': 'アシタヘ'})
                         for o in options]
            result = resolved_text(text, options, decisions)
            self.assertIn('｜明日《アシタ》', result)
            self.assertEqual(strip_ruby(result), re.sub(r'\(あした\)|［あした］', '', text))

    def test_exact_normal_whisper_can_finish_without_phonetic_recognition(self):
        text = '決意(おもい)だ'
        options = self.options(text)
        decisions = [choose_reading(o, {'whisper': 'オモイダ'}, recognition_text='思いだ')
                     for o in options]
        self.assertEqual(decisions[0]['source'], 'whisper')
        self.assertEqual(decisions[0]['status'], 'resolved')
        self.assertEqual(resolved_text(text, options, decisions), '｜決意《オモイ》だ')

    def test_same_kanji_is_not_a_pronunciation_observation(self):
        options, = self.options('運命(さだめ)だ')
        decision = choose_reading(options, {'whisper': 'ウンメーダ'}, recognition_text='運命だ')
        self.assertEqual(decision['status'], 'unresolved')
        self.assertEqual(decision['reason'], 'kanji-reading-not-observed')
        decided = choose_reading(options, {'mix': 'サダメダ', 'vocals': 'サダメダ'})
        self.assertEqual(options['choices'][decided['selected']]['reading'], 'サダメ')

    def test_normal_whisper_can_confirm_one_shared_dictionary_reading(self):
        options, = self.options('未来(みらい)へ')
        decision = choose_reading(options, {'whisper': 'ミライヘ'}, recognition_text='未来へ')
        self.assertEqual(decision['status'], 'resolved')
        self.assertEqual(options['choices'][decision['selected']]['mode'], 'annotation')

    def test_context_reading_disagreement_does_not_hide_annotation_agreement(self):
        def alternatives(text):
            kana, = fixture_readings(text)
            return tuple(dict.fromkeys((kana, kana.replace('ボク', 'ワタシ'))))
        text = '僕の未来(みらい)'
        options = reading_options(text, alternatives, word_starts=lambda _: (0, 1, 2))
        decisions = [choose_reading(o, {'mix': 'ボクノミライ', 'vocals': 'ワタシノミライ'})
                     for o in options]
        self.assertEqual(resolved_text(text, options, decisions), '僕の｜未来《ミライ》')

    def test_neighboring_annotations_remain_context_alternatives(self):
        text = '運命(さだめ)だ　未来(あす)へ'
        options = self.options(text)
        decisions = [choose_reading(o, {'mix': 'サダメダアスヘ', 'vocals': 'サダメダアスヘ'})
                     for o in options]
        self.assertEqual(resolved_text(text, options, decisions), '｜運命《サダメ》だ　｜未来《アス》へ')

    def test_pronounced_base_and_repeated_words_remain_real_candidates(self):
        for text, heard, expected in [('運命(さだめ)だ', 'ウンメーダ', '｜運命《ウンメー》だ'),
                                      ('誰だ(だれだ)', 'ダレダダレダ', '｜誰だ《ダレダ》(だれだ)')]:
            options = self.options(text)
            decisions = [choose_reading(o, {'mix': heard, 'vocals': heard}) for o in options]
            self.assertEqual(resolved_text(text, options, decisions), expected)

    def test_sung_parenthesis_keeps_the_selected_base_pronunciation(self):
        def alternatives(text):
            return ('アス', 'アシタ') if text == '明日' else fixture_readings(text)
        text = '明日(あす)'
        options = reading_options(text, alternatives, word_starts=lambda _: (0,))
        decisions = [choose_reading(o, {'mix': 'アシタアス', 'vocals': 'アシタアス'})
                     for o in options]
        self.assertEqual(decisions[0]['status'], 'retained-sung')
        result = resolved_text(text, options, decisions)
        self.assertEqual(strip_ruby(result), text)
        self.assertEqual(fixture_readings(result), ('アシタアス',))

    def test_context_and_scope_are_compared_without_mora_length_cutoffs(self):
        for text, heard, expected in [('この杜(ここ)だ', 'ココダ', '｜この杜《ココ》だ'),
                                      ('僕の願い(ゆめ)だ', 'ボクノユメダ', '僕の｜願い《ユメ》だ')]:
            options = self.options(text)
            decisions = [choose_reading(o, {'mix': heard, 'vocals': heard}) for o in options]
            self.assertEqual(resolved_text(text, options, decisions), expected)
        def alternatives(text):
            return ('アス', 'アシタ', 'ミョーニチ') if text == '明日' else fixture_readings(text)
        options, = reading_options('明日(あした)', alternatives, word_starts=lambda _: (0,))
        self.assertTrue({'アス', 'アシタ', 'ミョーニチ'}.issubset(
            {x['reading'] for x in options['choices'] if x['mode'] == 'base'}))

    def test_noisy_context_is_not_discarded_to_make_a_short_annotation_match(self):
        text = '僕の願い(ゆめ)がある'
        options = self.options(text)
        decisions = [choose_reading(o, {'mix': 'ボグノユメガアル', 'vocals': 'ボクノユメガアル'})
                     for o in options]
        self.assertEqual(resolved_text(text, options, decisions), '僕の｜願い《ユメ》がある')

    def test_location_hint_keeps_words_before_the_annotated_word(self):
        with patch('soramimic_score.parenthetical._word_starts', side_effect=fixture_starts):
            self.assertEqual(matching_forms('僕の願い(ゆめ)'),
                             ('僕の願い', '僕の｜願い《ユメ》'))

    def test_conflicting_empty_and_weak_evidence_keep_the_complete_input(self):
        text = '運命(さだめ)だ'
        options = self.options(text)
        for views in ({'mix': 'サダメダ', 'vocals': 'ウンメーダ'},
                      {'mix': 'サダメダ', 'vocals': ''}, {},
                      {'mix': 'アカサタナ', 'vocals': 'アカサタナ'}):
            decisions = [choose_reading(o, views) for o in options]
            self.assertEqual(decisions[0]['status'], 'unresolved')
            self.assertEqual(resolved_text(text, options, decisions), text)

    def test_multiple_annotations_are_applied_at_original_character_offsets(self):
        text = '運命(さだめ)だ　未来(あす)へ'
        options = self.options(text)
        decisions = []
        for option in options:
            chosen = next(i for i, c in enumerate(option['choices']) if c['mode'] == 'annotation')
            decisions.append({'status': 'resolved', 'selected': chosen})
        self.assertEqual(resolved_text(text, options, decisions), '｜運命《サダメ》だ　｜未来《アス》へ')

    def test_location_hints_preserve_the_original_supplied_text(self):
        source = '運命(さだめ)だ'
        plan = plan_supplied_lyrics([source], [LyricLine('運命だ', 1., 2.)],
                                    reading=lambda text: fixture_readings(text)[0])
        self.assertEqual(plan['groups'][0]['operation'], 'match')
        self.assertEqual(plan['groups'][0]['display_text'], source)
        self.assertEqual(plan['supplied_lines'], [source])


class ParentheticalModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('config.json', 'model.safetensors', 'LICENSE'):
            (self.root / name).touch()
        self.config = ModelConfig(self.root, self.root, separate_vocals=False)
        self.audio = self.root / 'input.wav'
        self.audio.write_bytes(b'models replaced at the adapter boundary')
        self.stack = []
        for mock in (patch('soramimic_score.models.dictionary_readings', side_effect=fixture_selector),
                     patch('soramimic_score.parenthetical._word_starts', side_effect=fixture_starts),
                     patch.dict(sys.modules, {'librosa': SimpleNamespace(get_duration=lambda **_: 4.)})):
            mock.start()
            self.addCleanup(mock.stop)

    def test_clear_normal_recognition_skips_kana_model(self):
        adapters = create_adapters(self.config)
        with patch('soramimic_score.models.transcribe_kana_views') as kana:
            result, = adapters.supplied_reading_selector(
                self.audio, (LyricLine('決意(おもい)だ', 0., 1.),), ('思いだ',))
        kana.assert_not_called()
        self.assertEqual(result.kana, 'オモイダ')
        self.assertEqual(result.detail['parenthetical_readings'][0]['source'], 'whisper')

    def test_plain_lyrics_keep_the_existing_selector_result(self):
        adapters = create_adapters(self.config)
        lines = (LyricLine('未来へ', 0., 1.),)
        with patch('soramimic_score.models.transcribe_kana_views') as kana:
            ordinary = adapters.reading_selector(self.audio, lines)
            supplied = adapters.supplied_reading_selector(self.audio, lines, ('未来へ',))
        kana.assert_not_called()
        self.assertEqual(supplied, ordinary)

    def test_dictionary_only_mode_does_not_remove_parenthetical_text(self):
        adapters = create_adapters(replace(self.config, acoustic_readings=False))
        self.assertIsNone(adapters.supplied_reading_selector)
        result, = adapters.reading_selector(self.audio, (LyricLine('運命(さだめ)'),))
        self.assertEqual(result.kana, 'ウンメーサダメ')

    def test_same_kanji_invokes_kana_and_preserves_decision_evidence(self):
        adapters = create_adapters(self.config, vocals_path=self.root / 'vocals.wav')
        with patch('soramimic_score.models.transcribe_kana_views', return_value={
            'mix': ('サダメダ',), 'vocals': ('サダメダ',),
        }) as kana:
            result, = adapters.supplied_reading_selector(
                self.audio, (LyricLine('運命(さだめ)だ', 0., 1.),), ('運命だ',))
        kana.assert_called_once()
        self.assertEqual(result.kana, 'サダメダ')
        self.assertEqual(result.detail['resolved_text'], '｜運命《サダメ》だ')
        decision, = result.detail['parenthetical_readings']
        self.assertEqual(decision['source'], 'kana-whisper')
        self.assertEqual(decision['whisper']['reason'], 'kanji-reading-not-observed')
        json.dumps(result.detail, ensure_ascii=False, allow_nan=False)

    def test_complete_pipeline_keeps_reading_and_original_input_after_save(self):
        source = '運命(さだめ)だ'
        adapters = create_adapters(self.config)
        def align(_path, lines, readings):
            self.assertEqual(lines[0].text, '運命だ')
            self.assertEqual(readings[0].kana, 'サダメダ')
            return tuple(AlignedMora(0, i, mora, i*.3, (i+1)*.3, .9)
                         for i, mora in enumerate(kana_to_moras(readings[0].kana)))
        adapters = replace(adapters, lyric_recognizer=lambda _: (LyricLine('運命だ', 0., 1.2),),
                           lyric_reading=lambda text: fixture_readings(text)[0],
                           mora_aligner=align,
                           melody_transcriber=lambda _: tuple(MelodyNote(i*.3, (i+1)*.3, 60)
                                                              for i in range(4)),
                           audio_duration=lambda _: 1.2)
        with patch('soramimic_score.models.transcribe_kana_views', return_value={'mix': ('サダメダ',)}):
            document = analyze_audio(self.audio, adapters, lyrics=[source])
        dump(document, self.root / 'score.json')
        restored = load(self.root / 'score.json')
        self.assertEqual(restored.score.canonical_text, '運命だ')
        evidence = next(e.detail for e in restored.observations.evidence if e.kind == 'reading-selection')
        self.assertEqual(evidence['supplied_text'], source)
        self.assertEqual(evidence['selected'], 'サダメダ')
        self.assertEqual(evidence['parenthetical_readings'][0]['status'], 'resolved')


if __name__ == '__main__':
    unittest.main()
