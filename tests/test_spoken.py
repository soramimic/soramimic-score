from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest

from soramimic_score import (AlignedMora, AudioAdapters, AudioPipelineError,
                             LyricLine, MelodyNote, ReadingSelection, analyze_audio)
from soramimic_score.document import from_linked_observations, loads
from soramimic_score.exports import export_midi, export_musicxml
from soramimic_score.spoken import add_spoken_fallback
from soramimic_score.vocal_activity import VocalActivity, measure_vocal_activity
from test_realization import document, linked
from soramimic_score import Boundary, NoteCandidate


def voiced(windows):
    return tuple(VocalActivity(-20, -3, 1, True) for _ in windows)


class SpokenTests(unittest.TestCase):
    def test_short_token_peaks_become_audible_intervals_without_changing_observations(self):
        ir = document()
        ir = replace(ir, singing_units=tuple(
            replace(u, end=replace(u.end, time_sec=u.consonant_start.time_sec + .02))
            for u in ir.singing_units))
        before = from_linked_observations(ir)
        result = add_spoken_fallback(before, {"u0": (0, 1)}, voiced)
        self.assertEqual([(s.start_sec, s.end_sec) for s in result.score.synthesis_plan],
                         [(0, .3), (.3, .6), (.6, 1)])
        self.assertEqual(result.observations.singing_units, ir.singing_units)
        self.assertEqual(result.observations.note_candidates, ())
        self.assertEqual(result.score.canonical, before.score.canonical)
        self.assertFalse(result.score.unresolved_unit_ids)
        self.assertTrue(all(s.pitch_sources == ("spoken",) and s.pitch_confidence is None
                            for s in result.score.synthesis_plan))
        self.assertEqual(loads(result.to_json()), result)
        self.assertEqual(add_spoken_fallback(result, {"u0": (0, 1)}, voiced), result)

    def test_existing_sung_slots_and_empty_space_bound_fallback(self):
        before = from_linked_observations(linked(document(), [
            ("unit_only", ("s0",), ()), ("match", ("s1",), ("n0",)),
            ("unit_only", ("s2",), ()),
        ], [NoteCandidate("n0", .25, .5, 67, .8, ("test",))]))
        result = add_spoken_fallback(before, {"u0": (0, .8)}, voiced)
        pitched = tuple(s for s in result.score.synthesis_plan if s.pitch_sources != ("spoken",))
        self.assertEqual(pitched, before.score.synthesis_plan)
        self.assertEqual([(s.start_sec, s.end_sec) for s in result.score.synthesis_plan],
                         [(0, .25), (.25, .5), (.6, .8)])

    def test_silent_and_unobserved_units_remain_unresolved(self):
        before = from_linked_observations(document(observed=2))
        seen = []
        def activity(windows):
            seen.extend(windows)
            return (VocalActivity(-90, -70, 0, False), voiced(windows)[1]) * 2
        result = add_spoken_fallback(before, {"u0": (0, .9)}, activity)
        self.assertEqual(len(seen), 4)
        self.assertEqual([s.kana for s in result.score.synthesis_plan], ["キ"])
        self.assertEqual(result.score.unresolved_unit_ids, ("s0", "s2"))

    def test_near_zero_timing_is_not_used(self):
        ir = document()
        ir = replace(ir, singing_units=tuple(replace(u, consonant_start=Boundary(
            u.consonant_start.time_sec, 1e-8, ("ctc",))) for u in ir.singing_units))
        before = from_linked_observations(ir)
        self.assertIs(add_spoken_fallback(before, {"u0": (0, .9)},
                                         lambda _: self.fail("must not query activity")), before)

    def test_fallback_does_not_cross_a_line_or_a_sung_note(self):
        before = from_linked_observations(linked(document(), [
            ("match", ("s0",), ("n0",)), ("unit_only", ("s1",), ()),
            ("unit_only", ("s2",), ()),
        ], [NoteCandidate("n0", 0, .5, 67, .8, ("test",))]))
        result = add_spoken_fallback(before, {"u0": (0, .7)}, voiced)
        self.assertEqual(result.score.unresolved_unit_ids, ("s1",))
        self.assertEqual(result.score.synthesis_plan[-1].end_sec, .7)

    def test_activity_count_must_match(self):
        with self.assertRaisesRegex(ValueError, "one vocal activity result"):
            add_spoken_fallback(from_linked_observations(document()), {"u0": (0, 1)},
                                lambda _: ())

    def test_exports_distinguish_render_pitch_from_measured_melody(self):
        from xml.etree import ElementTree as ET
        result = add_spoken_fallback(from_linked_observations(document()),
                                     {"u0": (0, 1)}, voiced)
        self.assertIn(b"spoken: neutral render pitch", export_midi(result))
        xml = ET.fromstring(export_musicxml(result))
        self.assertEqual(len(xml.findall(".//unpitched")), 3)
        self.assertFalse(xml.findall(".//pitch"))

    def test_automatic_pipeline_accepts_entirely_unpitched_voice(self):
        adapters = AudioAdapters(
            lambda _p, lines: tuple(ReadingSelection(l.text, "test", 1) for l in lines),
            lambda _p, _l, _r: (AlignedMora(0, 0, "カ", 0, .02, .9),
                                AlignedMora(0, 1, "キ", .3, .5, .9)),
            lambda _: (), lambda _: (LyricLine("カキ", 0, .5),),
            vocal_activity=lambda _p, windows: voiced(windows),
        )
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "test.wav"
            audio.touch()
            result = analyze_audio(audio, adapters)
            self.assertEqual([s.kana for s in result.score.synthesis_plan], ["カ", "キ"])
            self.assertFalse(result.observations.note_candidates)
            with self.assertRaisesRegex(AudioPipelineError, "no melody"):
                analyze_audio(audio, replace(adapters, vocal_activity=None))
            with self.assertRaisesRegex(AudioPipelineError, "no melody"):
                analyze_audio(audio, adapters, lyrics=("カキ",))

    def test_partial_melody_fallback_is_only_for_automatic_lyrics(self):
        adapters = AudioAdapters(
            lambda _p, lines: tuple(ReadingSelection(l.text, "test", 1) for l in lines),
            lambda _p, _l, _r: (AlignedMora(0, 0, "カ", 0, .02, .9),
                                AlignedMora(0, 1, "キ", .3, .5, .9),
                                AlignedMora(1, 0, "ク", 1, 1.2, .9)),
            lambda _: (MelodyNote(1, 1.2, 67),),
            lambda _: (LyricLine("カキ", 0, .5), LyricLine("ク", 1, 1.2)),
            vocal_activity=lambda _p, windows: voiced(windows),
        )
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "test.wav"
            audio.touch()
            automatic = analyze_audio(audio, adapters)
            supplied = analyze_audio(audio, adapters, lyrics=("カキ", "ク"))
            self.assertEqual([s.kana for s in automatic.score.synthesis_plan], ["カ", "キ", "ク"])
            self.assertEqual([s.kana for s in supplied.score.synthesis_plan], ["ク"])
            self.assertEqual(automatic.score.synthesis_plan[-1], supplied.score.synthesis_plan[-1])

    @unittest.skipUnless(importlib.util.find_spec("soundfile"), "audio dependencies unavailable")
    def test_absolute_silence_is_not_vocal_activity(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "silence.wav"
            sf.write(path, np.zeros(16000), 16000)
            self.assertFalse(measure_vocal_activity(path, ((0, 1),))[0].supported)

    @unittest.skipUnless(importlib.util.find_spec("soundfile"), "audio dependencies unavailable")
    def test_short_voice_before_a_long_rest_is_kept_without_stretching(self):
        import numpy as np
        import soundfile as sf
        ir = document()
        times = ((0, .05), (.8, .85), (1.1, 1.15))
        ir = replace(ir, singing_units=tuple(replace(
            u, consonant_start=replace(u.consonant_start, time_sec=start),
            end=replace(u.end, time_sec=end))
            for u, (start, end) in zip(ir.singing_units, times)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice-rest.wav"
            t = np.arange(24000) / 16000
            wave = .2 * np.sin(2 * np.pi * 220 * t)
            wave[(t >= .05) & (t < .8)] = 0
            sf.write(path, wave, 16000)
            result = add_spoken_fallback(from_linked_observations(ir), {"u0": (0, 1.5)},
                                         lambda windows: measure_vocal_activity(path, windows))
        self.assertFalse(result.score.unresolved_unit_ids)
        self.assertEqual(result.score.synthesis_plan[0].end_sec, .05)
        self.assertEqual(result.score.synthesis_plan[-1].end_sec, 1.5)
