from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest

from soramimic_score import (AlignedMora, AudioAdapters, LyricLine, MelodyNote,
                             PhoneticMora, ReadingSelection, analyze_audio, build_audio_observations)
from soramimic_score.document import compile_score
from soramimic_score.acoustic_repeats import (
    AcousticOccurrence, AcousticRepetition, consensus_pronunciation,
    pronunciation_agreement, repetitions_from_features)
from soramimic_score.japanese import kana_to_moras
from soramimic_score.repetition_repair import (
    copies_needing_repair, has_foreign_transcript, merge_recovered_notes, missing_note_windows,
    needs_pronunciation_repair, phase_aligned_moras, replace_pronunciation_spans, vowel_distance)


class RepetitionRepairTests(unittest.TestCase):
    def test_vowel_agreement_tolerates_consonants_but_requires_matching_times(self):
        first = tuple(zip((0., .3, .6, .9, 1.2), 'カキクケコ'))
        second = tuple(zip((.02, .32, .62, .92, 1.22), 'ガギグゲゴ'))
        self.assertEqual(pronunciation_agreement(first, second, 1.5), (1., 1., 5))
        self.assertLess(pronunciation_agreement(first, ((.2, 'ガ'), (.5, 'ギ')), 1.5)[2], 3)
        self.assertEqual(vowel_distance('カキクケコー', 'ガギグゲゴ'), 0.)

    def test_consensus_keeps_one_copy_length_instead_of_concatenating_alternatives(self):
        one = ((0., 'カ'), (.3, 'キ'), (.6, 'ク'), (.9, 'ケ'), (1.2, 'コ'))
        two = ((.02, 'ガ'), (.32, 'ギ'), (.62, 'グ'), (.92, 'ゲ'), (1.22, 'ゴ'))
        consensus = consensus_pronunciation((one, two, one), 1.5)
        self.assertEqual(len(consensus), 5)
        self.assertEqual(''.join(k for _, k in consensus), 'カキクケコ')
        self.assertEqual(consensus_pronunciation((one,), 1.5), ())

    def test_relaxed_repair_does_not_rewrite_japanese_verses_with_repeated_melody(self):
        occurrence = AcousticOccurrence(1., 3., .8, 'カキクケコ')
        group = AcousticRepetition('カキクケコ', 1., 2., (occurrence,) * 4)
        self.assertFalse(has_foreign_transcript(group, (LyricLine('青い空を見上げた', 0., 4.),)))
        self.assertTrue(has_foreign_transcript(group, (LyricLine('Look into the sky', 0., 4.),)))
        self.assertFalse(has_foreign_transcript(group, (LyricLine('Look into the sky', 8., 10.),)))

    def test_usable_vowels_are_preserved_and_missing_copy_is_repaired(self):
        occurrence = AcousticOccurrence(0., 2., .8, 'カキクケコ')
        moras = tuple(AlignedMora(0, i, k, i*.3, i*.3+.1, .9)
                      for i, k in enumerate('ガギグゲゴ'))
        self.assertFalse(needs_pronunciation_repair('カキクケコ', occurrence, moras))
        self.assertTrue(needs_pronunciation_repair('カキクケコ', occurrence, moras[:2]))

    def test_partly_collapsed_foreign_refrain_uses_a_consistent_pronunciation(self):
        copies = tuple(AcousticOccurrence(float(i*2), float((i+1)*2), .8, 'カキクケコ') for i in range(4))
        group = AcousticRepetition('カキクケコ', 0., 2., copies)
        lines = tuple(LyricLine('Look into the sky', c.start_sec, c.end_sec) for c in copies)
        moras = tuple(AlignedMora(j,i,k,j*2+i*.3,j*2+i*.3+.1,.9)
                      for j in range(4) for i,k in enumerate('カキクケコ'))
        self.assertEqual(copies_needing_repair(group, lines, moras), ())
        self.assertEqual(copies_needing_repair(group, lines, moras[:15]), copies)

    def test_replacement_retains_neighbor_inside_a_broad_asr_window(self):
        lines = (LyricLine('Some phrase', 0., 2.), LyricLine('青い空', 2., 5.))
        readings = (ReadingSelection('カキクケコ', 'test', .9),
                    ReadingSelection('アオイソラ', 'test', .9))
        moras = tuple(AlignedMora(0, i, k, i*.3, i*.3+.1, .9) for i, k in enumerate('カキクケコ'))
        moras += tuple(AlignedMora(1, i, k, 4.+i*.15, 4.1+i*.15, .9) for i, k in enumerate('アオイソラ'))
        new = LyricLine('ガギグゲゴ', 2., 3.9)
        selected = ReadingSelection('ガギグゲゴ', 'test', .9)
        aligned = tuple(AlignedMora(0, i, k, 2.+i*.3, 2.1+i*.3, .9) for i, k in enumerate('ガギグゲゴ'))
        updated, _, _ = replace_pronunciation_spans(lines, readings, moras, ((new, selected, aligned),))
        self.assertEqual([l.text for l in updated], ['Some phrase', 'ガギグゲゴ', '青い空'])
        self.assertEqual(updated[-1].start_sec, 4.)
        self.assertEqual(updated[0], lines[0])

    def test_replacement_preserves_fragments_on_both_sides(self):
        line = LyricLine('アオイソラ', 0., 5.)
        reading = ReadingSelection(line.text, 'test', .9)
        moras = tuple(AlignedMora(0, i, k, float(i), float(i)+.5, .9) for i,k in enumerate(line.text))
        replacement = (LyricLine('ウ', 2., 3.), ReadingSelection('ウ', 'test', .9),
                       (AlignedMora(0, 0, 'ウ', 2., 2.5, .9),))
        lines, readings, moras = replace_pronunciation_spans((line,), (reading,), moras, (replacement,))
        self.assertEqual([l.text for l in lines], ['アオ', 'ウ', 'ソラ'])
        self.assertEqual([(m.line_index, m.mora_index) for m in moras], [(0,0),(0,1),(1,0),(2,0),(2,1)])

    def test_foreign_line_is_not_removed_when_only_half_is_replaced(self):
        line = LyricLine('Looking into the sky', 0., 4.)
        reading = ReadingSelection('アオイソラ', 'test', .9)
        moras = tuple(AlignedMora(0, i, k, i*.7, i*.7+.4, .9) for i,k in enumerate(reading.kana))
        replacement = (LyricLine('ウ', 0., 2.), ReadingSelection('ウ', 'test', .9),
                       (AlignedMora(0, 0, 'ウ', 0., .4, .9),))
        lines, _, _ = replace_pronunciation_spans((line,), (reading,), moras, (replacement,))
        self.assertEqual([l.text for l in lines], ['ウ', 'ソラ'])

    def test_observed_phases_preserve_the_first_vowel_in_every_copy(self):
        copies = (AcousticOccurrence(1., 3., .8, 'カキクケコ'),
                  AcousticOccurrence(3., 5., .8, 'カキクケコ'))
        group = AcousticRepetition('カキクケコ', 1., 2., copies, (.06,.4,.8,1.2,1.6), 1.)
        aligned = phase_aligned_moras(group, copies)
        self.assertEqual([m.start_sec for m in aligned if m.mora_index==0], [1.06,3.06])
        self.assertTrue(all(m.source=='acoustic-repetition-phase' for m in aligned))

    def test_note_allocation_keeps_a_measured_initial_vowel_across_a_rest(self):
        line = LyricLine('ラカタ', 0., 2.)
        reading = ReadingSelection(line.text, 'test', 0.)
        aligned = tuple(AlignedMora(0,i,k,t,t+.04,0.,'acoustic-repetition-phase')
                        for i,(k,t) in enumerate(zip(line.text,(.04,.8,1.4))))
        notes = (MelodyNote(.0,.3,60),MelodyNote(.45,.6,60),MelodyNote(.6,1.,60),
                 MelodyNote(1.2,1.4,62),MelodyNote(1.4,1.8,64))
        observations = build_audio_observations((line,),(reading,),aligned,notes)
        result = compile_score(observations)
        first = result.score.synthesis_plan[0]
        self.assertEqual(first.kana, 'ラ')
        self.assertEqual(first.start_sec, 0.)
        phase, = (e for e in observations.evidence if e.id=='audio-mora-0-0')
        self.assertFalse(phase.detail['conditioned_on_text'])
        self.assertFalse(phase.detail['confidence_available'])

    def test_local_melody_retry_fills_only_confirmed_missing_intervals(self):
        copies = (AcousticOccurrence(1., 3., .8, 'カキクケコ'),
                  AcousticOccurrence(3., 5., .8, 'カキクケコ'),
                  AcousticOccurrence(6., 8., .8, 'カキクケコ'))
        notes = (MelodyNote(6., 8., 70),)
        self.assertEqual(missing_note_windows(copies, notes), ((1., 5.),))
        recovered = (MelodyNote(0., 2., 60), MelodyNote(2., 7., 61), MelodyNote(7., 9., 62))
        merged = merge_recovered_notes(notes, recovered, 1., 7.)
        self.assertEqual(merged, (MelodyNote(1., 2., 60), MelodyNote(2., 6., 61), notes[0]))

    def test_pipeline_repairs_each_copy_and_retries_its_missing_notes(self):
        line = LyricLine('Look into the sky', 0., 4.)
        copies = tuple(AcousticOccurrence(float(i), float(i+1), .8, 'カキクケコ') for i in range(4))
        group = AcousticRepetition('カキクケコ', 0., 1., copies)
        calls = []
        def readings(_p, ls):
            return tuple(ReadingSelection('カキクケコ', 'test', .9) for _ in ls)
        def align(_p, ls, rs):
            return tuple(AlignedMora(i,j,k,l.start_sec+j*(l.end_sec-l.start_sec)/5,
                                     l.start_sec+(j+1)*(l.end_sec-l.start_sec)/5,.9)
                         for i,(l,r) in enumerate(zip(ls,rs)) for j,k in enumerate(kana_to_moras(r.kana)))
        def recover(_p, start, end):
            calls.append((start,end))
            return (MelodyNote(start,end,60),)
        adapters = AudioAdapters(readings,align,lambda _: (MelodyNote(3.,4.,70),),lambda _: (line,),
                                 phonetic_recognizer=lambda _p,_w: (),
                                 acoustic_repetition_recognizer=lambda _p,_e,_n: (group,),
                                 melody_recoverer=recover)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'test.wav';p.write_bytes(b'adapter fixture')
            result=analyze_audio(p,adapters)
            supplied=analyze_audio(p,adapters,lyrics=(line.text,))
        self.assertEqual(result.score.canonical_text, '\n'.join(['カキクケコ']*4))
        self.assertEqual(calls, [(0.,3.)])
        self.assertTrue(any(e.kind=='melody-local-retry' for e in result.observations.evidence))
        self.assertFalse(any(e.kind=='lyric-acoustic-repetition' for e in supplied.observations.evidence))


@unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('scipy'),
                     'optional audio feature dependencies')
class AcousticFeatureRepetitionTests(unittest.TestCase):
    def fixture(self, copies=4, repeat_audio=True, repeat_vowels=True):
        import numpy as np
        rng=np.random.default_rng(17)
        features=rng.normal(size=(24,round((4+2*copies)/.02)))
        pattern=rng.normal(size=(24,100))
        events=[]
        for i in range(copies):
            start=2.+i*2
            if repeat_audio:features[:,100+i*100:200+i*100]=pattern
            kana='カキクケコ' if repeat_vowels else ['カキクケコ','キクケコカ','クケコカキ','ケコカキク'][i%4]
            events.extend(PhoneticMora(k,start+t,start+t+.02) for t,k in zip((.1,.45,.8,1.2,1.6),kana))
        features/=np.linalg.norm(features,axis=0)
        return features,features.shape[1]*.02,tuple(events)

    def test_four_measured_copies_are_found(self):
        groups=repetitions_from_features(*self.fixture())
        self.assertEqual(len(groups),1)
        self.assertEqual(len(groups[0].occurrences),4)

    def test_vowels_without_repeating_audio_are_insufficient(self):
        self.assertEqual(repetitions_from_features(*self.fixture(repeat_audio=False)),())

    def test_repeating_audio_without_matching_vowels_is_insufficient(self):
        self.assertEqual(repetitions_from_features(*self.fixture(repeat_vowels=False)),())

    def test_two_copies_and_sustained_vowels_are_insufficient(self):
        self.assertEqual(repetitions_from_features(*self.fixture(copies=2)),())
        f,d,e=self.fixture()
        self.assertEqual(repetitions_from_features(f,d,tuple(replace(x,kana='ア') for x in e)),())

    def test_repeated_backing_with_only_two_phonetic_decodes_is_insufficient(self):
        f,d,e=self.fixture()
        self.assertEqual(repetitions_from_features(f,d,e[:10]),())


if __name__=='__main__':
    unittest.main()
