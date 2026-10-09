import contextlib
from dataclasses import replace
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from soramimic_score import (
    AudioAdapters, AudioPipelineError, LyricLine, ModelConfig, ReadingSelection,
    analyze_audio, dump, load,
)
from soramimic_score.__main__ import analyze_main
from soramimic_score.models import create_adapters, prepared_adapters
from soramimic_score.readings import (
    acoustic_windows, dictionary_candidates, grouped_acoustic_windows,
    select_acoustic_reading,
)
from tests import test_audio_pipeline as fixtures


class ReadingChoiceTests(unittest.TestCase):
    def test_acoustics_can_choose_a_longer_dictionary_reading(self):
        result = select_acoustic_reading(("アス", "アシタ", "ミョーニチ"),
                                         {"mix": "アシタ", "vocals": "アシタ"})
        self.assertEqual(result.kana, "アシタ")
        self.assertEqual(result.candidates, ("アス", "アシタ", "ミョーニチ"))
        self.assertEqual(result.detail["reason"], "acoustic-agreement")
        self.assertFalse(result.detail["confidence_available"])

    def test_context_prefix_and_suffix_do_not_obscure_a_reading(self):
        result = select_acoustic_reading(("アス", "アシタ"), {"mix": "ソレトアシタデス"})
        self.assertEqual(result.kana, "アシタ")

    def test_conflicting_views_and_equal_distances_preserve_default(self):
        for transcripts in ({"mix": "アオイ", "vocals": "アカイ"}, {"mix": "アシイ"}):
            result = select_acoustic_reading(("アオイ", "アカイ"), transcripts)
            self.assertEqual(result.kana, "アオイ")

    def test_empty_evidence_keeps_dictionary_and_records_reason(self):
        result = select_acoustic_reading(("アス", "アシタ"), {"mix": "", "vocals": "…"})
        self.assertEqual(result.kana, "アス")
        self.assertEqual(result.detail["reason"], "no-acoustic-evidence")

    def test_long_vowel_equivalence_is_a_tie_not_a_reason_to_switch(self):
        result = select_acoustic_reading(("コー", "コオ"), {"mix": "コオ"})
        self.assertEqual(result.kana, "コー")
        self.assertEqual(result.detail["reason"], "ambiguous-evidence")

    def test_small_vowel_evidence_uses_video_normalization_order(self):
        result = select_acoustic_reading(("アス", "セェ"), {"mix": "セェ"})
        self.assertEqual(result.kana, "アス")
        self.assertEqual(result.detail["reason"], "weak-evidence")

    def test_small_kana_alternative_is_retained_without_nonfinite_json(self):
        result = select_acoustic_reading(("アス", "ァ"), {"mix": "アス"})
        self.assertEqual(result.candidates, ("アス", "ァ"))
        self.assertEqual(result.detail["distances"][1]["mix"], 0.0)

    def test_windows_bound_context_and_cover_long_lines(self):
        self.assertEqual(acoustic_windows(2, 5, 10), ((.5, 6.5),))
        windows = acoustic_windows(0, 65, 65)
        self.assertEqual(windows, ((0, 24), (24, 48), (48, 65)))
        for window in ((-1, 1, 2), (1, 1, 2), (0, 3, 2), (0, float("nan"), 3)):
            with self.assertRaises(ValueError):
                acoustic_windows(*window)

    def test_nearby_lines_share_video_style_kana_context(self):
        windows, assigned = grouped_acoustic_windows(
            [(2, 3), (4, 5), (14, 15)], range(3), 20)
        self.assertEqual(windows, ((.5, 6.5), (12.5, 16.5)))
        self.assertEqual(assigned, {0: (0,), 1: (0,), 2: (1,)})

    @unittest.skipUnless(importlib.util.find_spec("MeCab"), "audio dependencies not installed")
    def test_real_dictionary_keeps_unequal_mora_lengths(self):
        candidates = dictionary_candidates((LyricLine("明日"),))[0]
        self.assertTrue({"アス", "アシタ", "ミョーニチ"}.issubset(candidates))
        self.assertNotIn("アキラヒ", candidates)  # Different token segmentation.


class PreparedAudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio = self.root / "input.wav"
        import wave
        with wave.open(str(self.audio), "wb") as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
            wav.writeframes(b"\0\0" * 1600)
        self.config = ModelConfig(self.root / "model", self.root / "base")
        for folder in (self.config.sheetsage_model, self.config.sheetsage_base):
            folder.mkdir()
            for name in ("config.json", "model.safetensors", "LICENSE"):
                (folder / name).touch()

    def test_stem_is_scoped_to_one_analysis_and_cleaned_on_exception(self):
        def separate(path, output, config):
            self.assertEqual(path, self.audio)
            output.write_bytes(b"private temporary stem")
        with patch("soramimic_score.models.separate_vocals", side_effect=separate), \
             patch("soramimic_score.models.create_adapters") as factory:
            with self.assertRaisesRegex(RuntimeError, "downstream"):
                with prepared_adapters(self.audio, self.config):
                    stem = factory.call_args.kwargs["vocals_path"]
                    self.assertTrue(stem.is_file())
                    raise RuntimeError("downstream")
        self.assertFalse(stem.parent.exists())

    def test_separation_failure_is_labeled_and_does_not_silently_use_mix(self):
        with patch("soramimic_score.models.separate_vocals", side_effect=ValueError("failed")), \
             patch("soramimic_score.models.create_adapters") as factory:
            with self.assertRaisesRegex(AudioPipelineError, "vocal separation: failed"):
                with prepared_adapters(self.audio, self.config):
                    self.fail("failed separator cannot continue")
        factory.assert_not_called()

    def test_separation_can_be_disabled_independently(self):
        with patch("soramimic_score.models.separate_vocals") as separator, \
             patch("soramimic_score.models.create_adapters") as factory:
            with prepared_adapters(self.audio, replace(self.config, separate_vocals=False)):
                self.assertIsNone(factory.call_args.kwargs["vocals_path"])
        separator.assert_not_called()

    def test_cli_switches_reach_configuration(self):
        adapters = AudioAdapters(fixtures.AudioPipelineTests._readings,
                                 fixtures.AudioPipelineTests._moras, fixtures.AudioPipelineTests._melody,
                                 lambda _: (LyricLine("空", 0, .4), LyricLine("耳", .4, .8)))
        with patch("soramimic_score.models.separate_vocals") as separator, \
             patch("soramimic_score.models.create_adapters", return_value=adapters) as factory, \
             contextlib.redirect_stdout(io.StringIO()):
            analyze_main([str(self.audio), "--output", str(self.root / "score.json"),
                          "--sheetsage-model", str(self.config.sheetsage_model),
                          "--sheetsage-base", str(self.config.sheetsage_base),
                          "--no-vocal-separation", "--dictionary-readings", "--local-files-only",
                          "--kana-model", "local-kana", "--demucs-checkpoint", "local.th"])
        config = factory.call_args.args[0]
        self.assertFalse(config.separate_vocals)
        self.assertFalse(config.acoustic_readings)
        self.assertTrue(config.local_files_only)
        self.assertEqual(config.kana_model, "local-kana")
        self.assertEqual(config.demucs_checkpoint, Path("local.th"))
        separator.assert_not_called()

    def test_reading_evidence_is_linked_and_survives_json_roundtrip(self):
        def readings(_path, lines):
            return (select_acoustic_reading(("カラ", "ソラ"), {"mix": "ソラ", "vocals": "ソラ"}),
                    ReadingSelection("ミミ", "dictionary", 1))
        def reject(_):
            return (LyricLine("空", 0, .4), LyricLine("耳", .4, .8))
        result = analyze_audio(self.audio, AudioAdapters(readings, fixtures.AudioPipelineTests._moras,
                                                       fixtures.AudioPipelineTests._melody, reject),
                               lyrics=("空", "耳"))
        path = self.root / "score.json"
        dump(result, path)
        result = load(path)
        self.assertEqual(result.score.canonical_text, "空\n耳")
        evidence = next(e for e in result.observations.evidence if e.kind == "reading-selection")
        self.assertEqual(evidence.detail["candidates"], ["カラ", "ソラ"])
        self.assertEqual(evidence.detail["selected"], "ソラ")
        self.assertIn(evidence.id, result.observations.readings[0].evidence_ids)

    def test_single_dictionary_candidate_skips_kana_model(self):
        with patch("soramimic_score.models.dictionary_readings", return_value=(
                ReadingSelection("ソラ", "soramimic-yomi", 1, ("ソラ",)),)), \
             patch("soramimic_score.models.transcribe_kana_views") as transcribe:
            result = create_adapters(self.config).reading_selector(self.audio, (LyricLine("空"),))
        self.assertEqual(result[0].kana, "ソラ")
        self.assertEqual(result[0].source, "soramimic-yomi")
        transcribe.assert_not_called()

    def test_single_candidate_with_symbol_still_uses_audio(self):
        slot = {"start": 3, "end": 4, "surface": "🧭", "mapped": True,
                "kana_start": 4, "kana_end": 4}
        base, spoken = "アシタモキミトアルコウ", "アシタモステキキミトアルコウ"
        with patch("soramimic_score.models.dictionary_readings", return_value=(
                ReadingSelection(base, "soramimic-yomi", 1, (base,)),)), \
             patch("soramimic_score.models.symbol_slots", return_value=(slot,)), \
             patch("soramimic_score.models.refine_symbol_reading",
                   side_effect=lambda _text, selected, *_args, **_kwargs: selected), \
             patch.dict("sys.modules", {"librosa": SimpleNamespace(get_duration=lambda **_: 10)}), \
             patch("soramimic_score.models.transcribe_whisper_views",
                   return_value={"mix": (spoken,), "vocals": (spoken,)}) as lexical, \
             patch("soramimic_score.models.transcribe_kana_views",
                   return_value={"mix": (spoken,), "vocals": (spoken,)}) as transcribe:
            result = create_adapters(self.config, vocals_path=self.root / "vocals.wav").reading_selector(
                self.audio, (LyricLine("明日も🧭君と歩こう", 2, 5),))
        transcribe.assert_called_once()
        lexical.assert_called_once()
        self.assertEqual(result[0].kana, spoken)
        self.assertEqual(result[0].detail["symbol_proposals"], [spoken])

    def test_full_recognition_context_is_scoped_to_matching_audio_and_time(self):
        slot = {"start": 3, "end": 4, "surface": "🧭", "mapped": True,
                "kana_start": 4, "kana_end": 4}
        base = "アシタモキミトアルコウ"
        shared = SimpleNamespace(run=lambda *_args, **_kwargs: {
            "requested_language": "ja", "lines": [
                {"start_sec": 1., "end_sec": 4., "text": "あしたも冒険きみとあるこう"},
                {"start_sec": 8., "end_sec": 9., "text": "別の区間"},
            ]})
        with patch("soramimic_score.models.dictionary_readings", return_value=(
                ReadingSelection(base, "soramimic-yomi", 1, (base,)),)), \
             patch("soramimic_score.models.symbol_slots", return_value=(slot,)), \
             patch("soramimic_score.models.symbol_reading_proposals", return_value=((), ())), \
             patch("soramimic_score.models.refine_symbol_reading",
                   side_effect=lambda _text, selected, *_args, **_kwargs: selected) as refine, \
             patch.dict("sys.modules", {"librosa": SimpleNamespace(get_duration=lambda **_: 10)}), \
             patch("soramimic_score.models._transcribe_shared_kana", return_value=(base,)), \
             patch("soramimic_score.models.transcribe_whisper_views", return_value={"mix": ("字幕",)}):
            adapters = create_adapters(self.config, shared=shared)
            adapters.lyric_recognizer(self.audio)
            lines = (LyricLine("明日も🧭君と歩こう", 2, 5),)
            adapters.reading_selector(self.audio, lines)
            views = refine.call_args.args[3]
            self.assertEqual(views["mix:full-context"], "あしたも冒険きみとあるこう")
            self.assertEqual(views["mix"], "字幕")
            other = self.root / "other.wav"
            other.write_bytes(self.audio.read_bytes())
            adapters.reading_selector(other, lines)
            self.assertNotIn("mix:full-context", refine.call_args.args[3])

    @unittest.skipUnless(importlib.util.find_spec("librosa"), "audio dependencies not installed")
    def test_audio_views_and_windows_are_used_for_selection(self):
        vocals = self.root / "vocals.wav"
        provenance = [{"kana": "アス", "sources": ["soramimic-yomi"]},
                      {"kana": "アシタ", "sources": ["unidic-lite"]}]
        with patch("soramimic_score.models.dictionary_readings", return_value=(
                ReadingSelection("アス", "soramimic-yomi", 1, ("アス", "アシタ"),
                                 {"candidate_provenance": provenance}),)), \
             patch("librosa.get_duration", return_value=10), \
             patch("soramimic_score.models.transcribe_kana_views",
                   return_value={"mix": ("アシタ",), "vocals": ("アシタ",)}) as transcribe:
            result = create_adapters(self.config, vocals_path=vocals).reading_selector(
                self.audio, (LyricLine("明日", 2, 5),))
        self.assertEqual(transcribe.call_args.args[0], {"mix": self.audio, "vocals": vocals})
        self.assertEqual(transcribe.call_args.args[1], ((.5, 6.5),))
        self.assertEqual(result[0].kana, "アシタ")
        self.assertEqual(result[0].source, "kana-whisper")
        self.assertEqual(result[0].detail["candidate_provenance"], provenance)
        self.assertEqual(result[0].detail["vocal_separator"], "demucs-htdemucs")


@unittest.skipUnless(importlib.util.find_spec("demucs"), "audio dependencies not installed")
class AcousticBackendTests(unittest.TestCase):
    def test_kana_model_uses_audio_only_and_forwards_offline_options(self):
        import numpy as np
        import soundfile as sf
        import torch
        from soramimic_score.acoustic import KANA_MODEL, KANA_REVISION, transcribe_kana_views
        class Model:
            def eval(self):
                return self
            def to(self, device):
                return self
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.wav"
            sf.write(source, np.zeros(16000 * 3), 16000)
            processor = SimpleNamespace(tokenizer=object(), feature_extractor=object())
            with patch("transformers.AutoModelForSpeechSeq2Seq.from_pretrained", return_value=Model()) as loader, \
                 patch("transformers.AutoProcessor.from_pretrained", return_value=processor) as processor_loader, \
                 patch("transformers.pipeline") as pipeline:
                pipeline.return_value.return_value = {"text": "アシタ"}
                result = transcribe_kana_views({"mix": source, "vocals": source}, [(1, 2)],
                    SimpleNamespace(kana_model=KANA_MODEL, local_files_only=True, device="cpu"))
                self.assertEqual(result, {"mix": ("アシタ",), "vocals": ("アシタ",)})
                self.assertEqual(pipeline.return_value.call_count, 2)
                args, kwargs = pipeline.return_value.call_args
                self.assertEqual(len(args[0]), 16000)
                self.assertEqual(kwargs, {"generate_kwargs": {"language": "ja", "task": "transcribe"}})
                for call in (loader.call_args, processor_loader.call_args):
                    self.assertTrue(call.kwargs["local_files_only"])
                    self.assertEqual(call.kwargs["revision"], KANA_REVISION)
                self.assertEqual(loader.call_args.kwargs["torch_dtype"], torch.float32)

    def test_offline_demucs_never_downloads_missing_weights(self):
        from soramimic_score.acoustic import _demucs_package
        with tempfile.TemporaryDirectory() as directory, \
             patch("torch.hub.get_dir", return_value=directory), \
             patch("torch.hub.load_state_dict_from_url") as download:
            config = SimpleNamespace(demucs_checkpoint=None, local_files_only=True)
            with self.assertRaisesRegex(FileNotFoundError, "not available locally"):
                _demucs_package(config)
        download.assert_not_called()

    def test_demucs_rejects_unverified_checkpoint_before_unpickling(self):
        from soramimic_score.acoustic import _demucs_package
        with tempfile.TemporaryDirectory() as directory, patch("torch.load") as loader:
            path = Path(directory) / "model.th"
            path.write_bytes(b"not the official checkpoint")
            config = SimpleNamespace(demucs_checkpoint=path, local_files_only=True)
            with self.assertRaisesRegex(RuntimeError, "Invalid checksum"):
                _demucs_package(config)
        loader.assert_not_called()

    def test_separator_preserves_stereo_length_and_handles_silence(self):
        import numpy as np
        import soundfile as sf
        import torch
        from soramimic_score.acoustic import separate_vocals
        class Model:
            samplerate, audio_channels, sources = 16000, 2, ["other", "vocals"]
            def eval(self):
                return self
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.wav", Path(directory) / "vocals.wav"
            sf.write(source, np.zeros(16001), 16000)
            def apply(model, wave, **options):
                self.assertEqual(options["shifts"], 0)
                return torch.stack((wave, wave), dim=1)
            with patch("soramimic_score.acoustic._demucs_package", return_value={}), \
                 patch("demucs.states.load_model", return_value=Model()), \
                 patch("demucs.apply.apply_model", side_effect=apply):
                separate_vocals(source, output, SimpleNamespace(device="cpu"))
            samples, rate = sf.read(output)
            self.assertEqual(rate, 16000)
            self.assertEqual(samples.shape, (16001, 2))
            self.assertTrue(np.isfinite(samples).all())
            self.assertEqual(sf.info(output).subtype, "FLOAT")


if __name__ == "__main__":
    unittest.main()
