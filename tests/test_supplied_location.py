"""Local phonetic hints cannot replace supplied lyrics or establish acoustics."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from soramimic_score import analyze_audio
from soramimic_score.audio import AlignedMora, AudioAdapters, LyricLine, MelodyNote, ReadingSelection
from soramimic_score.supplied_location import propose_phonetic_locations
from soramimic_score.supplied_lyrics import (expand_alignment_windows,
                                            prepare_supplied_audio,
                                            select_authored_readings)


class SuppliedLocationTests(unittest.TestCase):
    def test_automatic_and_already_matched_input_never_request_candidates(self):
        candidates = Mock(side_effect=AssertionError('unnecessary candidate lookup'))
        adapters = AudioAdapters(
            lambda *_: (ReadingSelection('カキ', 'test', .9),),
            lambda *_: (AlignedMora(0, 0, 'カ', 0, .4, .9),
                        AlignedMora(0, 1, 'キ', .5, .9, .9)),
            lambda *_: (MelodyNote(0, .4, 60), MelodyNote(.5, .9, 62)),
            lambda *_: (LyricLine('カキ', 0, 1),),
            lyric_reading=lambda text: text, lyric_reading_candidates=candidates,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.wav'
            path.touch()
            for lyrics in (None, ('カキ',)):
                self.assertEqual(analyze_audio(path, adapters, lyrics=lyrics).score.canonical_text,
                                 'カキ')
        candidates.assert_not_called()

    def plan(self, sources, *, target='567がひかる'):
        return {'minimum_similarity': .72, 'groups': [
            {'operation': 'match', 'asr_indices': [0], 'display_text': 'カキ'},
            {'operation': 'retain_supplied', 'asr_indices': list(range(1, len(sources) - 1)),
             'display_text': target},
            {'operation': 'match', 'asr_indices': [len(sources) - 1], 'display_text': 'タチ'},
        ]}

    def test_number_alternatives_locate_phonetics_without_selecting_a_reading(self):
        sources = [LyricLine('カキ', 0, 1), LyricLine('アオゾラ', 1, 4),
                   LyricLine('ゴーロクナナガヒカル', 4, 6), LyricLine('タチ', 6, 7)]
        plan = self.plan(sources)
        options = ('ゴヒャクロクジューナナバンゴーガヒカル', 'ゴーロクナナガヒカル')
        calls = []
        def candidates(text):
            calls.append(text)
            return options
        self.assertEqual(propose_phonetic_locations(plan, sources, reading=lambda x: x,
                                                   candidates=candidates), [])
        group = plan['groups'][1]
        self.assertEqual(group['phonetic_location']['window_sec'], [4, 6])
        self.assertEqual(group['phonetic_location']['reading'], options[1])
        self.assertEqual(group['asr_indices'], [2])
        self.assertEqual(group['phonetic_location']['prior_asr_indices'], [1, 2])
        self.assertEqual(group['display_text'], '567がひかる')
        self.assertEqual(group['operation'], 'retain_supplied')
        self.assertNotIn('ctc_verified', group)
        self.assertEqual(calls, ['567がひかる'])
        self.assertNotIn('phonetic_location', plan['groups'][0])

    def test_disjoint_equal_repetitions_abstain(self):
        sources = [LyricLine('カキ', 0, 1), LyricLine('ソラエ', 1, 3),
                   LyricLine('ソラエ', 3, 5), LyricLine('タチ', 5, 6)]
        plan = self.plan(sources)
        self.assertEqual(propose_phonetic_locations(plan, sources, reading=lambda x: x,
                                                   candidates=lambda _: ('ソラエ',)), [])
        self.assertFalse(plan['groups'][1]['phonetic_location_proposal']['accepted'])
        self.assertNotIn('phonetic_location', plan['groups'][1])
        self.assertEqual(plan['groups'][1]['asr_indices'], [1, 2])

    def test_weak_overlap_only_proposes_joint_context(self):
        sources = [LyricLine('カキ', 0, 3), LyricLine('サシ', 3, 5), LyricLine('タチ', 5, 7)]
        plan = self.plan(sources)
        def distance(target, evidence):
            return 1. if not evidence else (.4 if evidence == 'カキサシ' else 1.)
        with patch('soramimic_score.supplied_location._substring_distance', side_effect=distance):
            edges = propose_phonetic_locations(plan, sources, reading=lambda x: x,
                                               candidates=lambda _: ('ナニ',))
        self.assertEqual(edges, [[1, 0]])
        group = plan['groups'][1]
        self.assertEqual(group['phonetic_location']['window_sec'], [0, 5])
        self.assertEqual(group['operation'], 'retain_supplied')
        self.assertNotIn('ctc_verified', group)

    def test_weak_gap_without_shared_anchor_is_not_narrowed(self):
        sources = [LyricLine('カキ', 0, 3), LyricLine('サシ', 3, 5), LyricLine('タチ', 5, 7)]
        plan = self.plan(sources)
        def distance(target, evidence):
            return 1. if not evidence else (.4 if evidence == 'サシ' else 1.)
        with patch('soramimic_score.supplied_location._substring_distance', side_effect=distance):
            self.assertEqual(propose_phonetic_locations(plan, sources, reading=lambda x: x,
                                                       candidates=lambda _: ('ナニ',)), [])
        self.assertNotIn('phonetic_location', plan['groups'][1])

    def test_large_gaps_and_unavailable_candidates_do_no_new_work(self):
        for count, duration in ((34, 34), (3, 90)):
            sources = [LyricLine('カキ', i * duration / count, (i + 1) * duration / count)
                       for i in range(count)]
            plan = self.plan(sources)
            with patch('soramimic_score.supplied_location._substring_distance') as distance:
                self.assertEqual(propose_phonetic_locations(plan, sources, reading=lambda x: x,
                                                           candidates=lambda _: ('ナニ',)), [])
                distance.assert_not_called()
        self.assertEqual(propose_phonetic_locations({}, [], reading=None, candidates=None), [])

    def test_shared_segment_reaches_joint_ctc_after_preserving_authored_readings(self):
        supplied = ('カキカキカキ', '567', 'タチタチ')
        heard = (LyricLine('カキカキカキナニ', 0, 4), LyricLine('ソラエ', 4, 7),
                 LyricLine('タチタチ', 9, 12))
        # Force the initial matcher to leave the middle spelling unmatched;
        # alternatives are only used by the local phonetic proposal.
        def reading(text):
            return 'ヌヌヌヌヌヌヌ' if text == supplied[1] else text
        adapters = AudioAdapters(
            lambda *_: (), lambda *_: (), lambda *_: (),
            lyric_reading=reading, lyric_reading_candidates=lambda _: ('ナニソラエ',),
            audio_duration=lambda _: 12.,
        )
        lines, overlay, missing = prepare_supplied_audio(
            Path('unused.wav'), supplied, heard, (), adapters,
            adjust_lyrics=False, recognition_evidence=heard)
        self.assertEqual([l.text for l in lines], list(supplied))
        self.assertEqual((lines[1].start_sec, lines[1].end_sec), (0, 7))
        self.assertEqual(overlay['ctc_context_edges'], [[1, 0]])
        readings = select_authored_readings(lines, lambda expanded, _: tuple(
            ReadingSelection('ナニソラエ' if line.text == '567' else line.text, 'test', 1.) for line in expanded))
        expanded, _, _ = expand_alignment_windows(lines, readings, overlay, missing)
        self.assertEqual([(l.text, l.start_sec, l.end_sec) for l in expanded],
                         [('カキカキカキ\n567', 0, 7), ('タチタチ', 9, 12)])
        self.assertEqual([g['display_text'] for g in overlay['reading_stage_groups']], list(supplied))
