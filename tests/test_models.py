import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import wave
from unittest.mock import patch
import requests

from soramimic_score import ModelConfig, LyricLine, analyze_audio, load
from soramimic_score.audio import CTCWindowCapacityError
from soramimic_score.__main__ import analyze_main
from soramimic_score.models import (_ctc_token_id, _transcribe_shared_kana, create_adapters,
                                    dictionary_readings, read_melody_lab)
from soramimic_score.shared_inference import SharedInference
from tests import test_audio_pipeline as fixtures


class ModelTests(unittest.TestCase):
    def test_ctc_aliases_require_an_available_sound_and_prefer_exact_tokens(self):
        vocab = {"ワ": 1, "カ": 2, "ヮ": 3}
        self.assertEqual(_ctc_token_id("ヮ", vocab), 3)
        self.assertEqual(_ctc_token_id("ヮ", {"ワ": 1}), 1)
        self.assertEqual(_ctc_token_id("ヵ", vocab), 2)
        for char, vocabulary in (("ヮ", {"カ": 1}), ("★", vocab), ("ヵ", {})):
            with self.subTest(char=char):
                with self.assertRaisesRegex(ValueError, "CTC vocabulary"):
                    _ctc_token_id(char, vocabulary)

    def test_shared_inference_preserves_worker_rejection_reason(self):
        class Response:
            def __init__(self, body, rejected=False):
                self.body = body
                self.rejected = rejected
            def raise_for_status(self):
                if self.rejected:
                    raise requests.HTTPError("422")
            def json(self):
                return self.body

        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "audio.wav"
            audio.write_bytes(b"audio")
            with patch("soramimic_score.shared_inference.requests.get", return_value=Response({
                "status": "ok", "api": {"name": "soramimic-audio-inference", "version": 1},
                "capabilities": {"demucs": True, "whisper": True,
                                 "kana_whisper": True, "sheetsage2": True},
            })), patch("soramimic_score.shared_inference.requests.post",
                       return_value=Response({"detail": "window is invalid"}, True)):
                client = SharedInference("http://localhost:8320")
                with self.assertRaisesRegex(RuntimeError, "window is invalid"):
                    client.run("kana-whisper", audio, {"windows": [[0, 1]]})

    def test_shared_kana_windows_respect_worker_limit_and_restore_order(self):
        class Shared:
            def run(self, kind, audio, parameters):
                self.windows = parameters["windows"]
                return {"texts": ["早い", "長い", "遅い"]}

        shared = Shared()
        windows = ((40., 41.), (1.25, 25.250000000000004), (30., 31.))
        self.assertEqual(_transcribe_shared_kana(shared, Path("audio.wav"), windows),
                         ("遅い", "早い", "長い"))
        self.assertEqual([window[0] for window in shared.windows], [1.25, 30., 40.])
        self.assertLess(shared.windows[0][1] - shared.windows[0][0], 24.)

    def test_shared_inference_job_is_deleted_after_result(self):
        calls = []
        class Response:
            def __init__(self, body):
                self.body = body
            def raise_for_status(self):
                pass
            def json(self):
                return self.body
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "audio.wav"
            audio.write_bytes(b"audio")
            with patch("soramimic_score.shared_inference.requests.get", side_effect=[
                Response({"status": "ok", "api": {"name": "soramimic-audio-inference",
                                                 "version": 1},
                          "capabilities": {"demucs": True, "whisper": True,
                                           "kana_whisper": True, "sheetsage2": True}}),
                Response({"status": "done", "result": {"lines": []}}),
            ]), patch("soramimic_score.shared_inference.requests.post",
                     return_value=Response({"id": "job1"})) as post, \
                 patch("soramimic_score.shared_inference.requests.delete",
                       side_effect=lambda url, **kwargs: calls.append(url)):
                client = SharedInference("http://localhost:8320", "dev")
                self.assertEqual(client.run("whisper", audio, {"device": "auto"}),
                                 {"lines": []})
            self.assertEqual(post.call_args.kwargs["data"]["priority"], "dev")
            self.assertEqual(calls, ["http://localhost:8320/v1/jobs/job1"])

    def test_model_adapters_use_shared_worker_for_expensive_models(self):
        calls = []
        class Shared:
            def run(self, kind, audio, parameters, artifacts=None):
                calls.append((kind, audio, parameters))
                if kind == "whisper":
                    return {"requested_language": "ja",
                            "requested_temperature": parameters.get("temperature"),
                            "lines": [{"text": " 空 ", "start_sec": .01,
                                       "end_sec": .05}]}
                if kind == "sheetsage":
                    return {"notes": [{"start_sec": 0, "end_sec": .1,
                                       "midi_note": 60}]}
                if kind == "kana-whisper":
                    return {"texts": ["ソラ"] * len(parameters["windows"])}
                raise AssertionError(kind)
        audio = self.root / "audio.wav"
        self.write_audio(audio)
        adapters = create_adapters(self.config, vocals_path=audio, shared=Shared())
        self.assertEqual(adapters.lyric_recognizer(audio), (LyricLine("空", .01, .05),))
        self.assertEqual(adapters.melody_transcriber(audio)[0].midi_pitch, 60)
        with patch.dict(sys.modules, {
            "librosa": SimpleNamespace(load=lambda *args, **kwargs: ([0.] * 1600, 16000)),
            "soundfile": SimpleNamespace(write=lambda path, *args, **kwargs:
                                         Path(path).write_bytes(b"window")),
        }):
            self.assertEqual(adapters.lyric_recoverer(audio, 0, .1),
                             (LyricLine("空", .01, .05),))
        self.assertEqual(adapters.repetition_evidence(audio, ((0, .1),)), ("ソラ",))
        self.assertEqual([kind for kind, *_ in calls],
                         ["whisper", "sheetsage", "whisper", "kana-whisper"])
        self.assertEqual(calls[2][2]["temperature"], 0.)

    @staticmethod
    def write_audio(path):
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
            wav.writeframes(b"\0\0" * 1600)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = ModelConfig(self.root / "model", self.root / "base")
        for folder in (self.config.sheetsage_model, self.config.sheetsage_base):
            folder.mkdir()
            for filename in ("config.json", "model.safetensors", "LICENSE"):
                (folder / filename).touch()

    def test_missing_local_models_fail_before_any_model_import(self):
        (self.config.sheetsage_base / "LICENSE").unlink()
        with self.assertRaisesRegex(ValueError, "LICENSE"):
            create_adapters(self.config)

    def test_cli_adjustment_requires_a_lyrics_file(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            analyze_main(["input.wav", "--output", "score.json", "--sheetsage-model", "model",
                          "--sheetsage-base", "base", "--adjust-lyrics"])
        self.assertEqual(error.exception.code, 2)

    def test_cli_forwards_adjustment_and_preserves_input_file(self):
        from soramimic_score import AudioAdapters
        audio, lyrics, output = self.root / "input.wav", self.root / "lyrics.txt", self.root / "score.json"
        self.write_audio(audio)
        lyrics.write_text("耳\n空", encoding="utf-8")
        adapters = AudioAdapters(
            fixtures.AudioPipelineTests._readings, fixtures.AudioPipelineTests._moras,
            fixtures.AudioPipelineTests._melody, lambda _: (LyricLine("空", 0, .4),),
        )
        with patch("soramimic_score.models.create_adapters", return_value=adapters), \
             patch("soramimic_score.models.separate_vocals"), \
             contextlib.redirect_stdout(io.StringIO()):
            result = analyze_main([str(audio), "--output", str(output),
                                   "--sheetsage-model", str(self.config.sheetsage_model),
                                   "--sheetsage-base", str(self.config.sheetsage_base),
                                   "--lyrics", str(lyrics), "--adjust-lyrics"])
        self.assertEqual(result, 0)
        self.assertEqual(load(output).score.canonical_text, "耳\n空")
        self.assertEqual(lyrics.read_text(encoding="utf-8"), "耳\n空")

    def test_lab_normalizes_overlap_and_preserves_pitch(self):
        path = self.root / "melody.lab"
        path.write_text("0.2\t0.5\t62\n0\t0.3\t60\n")
        notes = read_melody_lab(path)
        self.assertEqual([(n.start_sec, n.end_sec, n.midi_pitch) for n in notes],
                         [(0, .2, 60), (.2, .5, 62)])
        self.assertIsNone(notes[0].confidence)

    def test_lab_rejects_nonfinite_pitch_and_simultaneous_conflicts(self):
        path = self.root / "melody.lab"
        for text in ("nan 1 60", "0 1 128", "0 1 60\n0 2 62", "1 0 60"):
            with self.subTest(text=text):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_melody_lab(path)

    def test_whisper_is_lazy_and_uses_singing_settings(self):
        calls = []
        class Whisper:
            def __init__(self, name, **options):
                calls.append((name, options))
            def transcribe(self, path, **options):
                calls.append((path, options))
                return iter([SimpleNamespace(text=" あ ", start=0, end=1.2),
                             SimpleNamespace(text="", start=1.2, end=2)]), SimpleNamespace(duration=1)
        with patch.dict(sys.modules, {"faster_whisper": SimpleNamespace(WhisperModel=Whisper)}):
            adapters = create_adapters(self.config)
            self.assertEqual(calls, [])
            with patch("soramimic_score.models._release"):
                lines = adapters.lyric_recognizer(self.root / "audio.wav")
        self.assertEqual(lines, (LyricLine("あ", 0, 1),))
        self.assertFalse(calls[1][1]["vad_filter"])
        self.assertFalse(calls[1][1]["condition_on_previous_text"])

    def test_credit_retry_uses_short_audio_window_and_original_clock(self):
        calls = []
        def load_audio(path, **kwargs):
            calls.append(("load", path, kwargs))
            return [0.0] * 32000, 16000
        class Whisper:
            def __init__(self, name, **options):
                calls.append(("model", name))
            def transcribe(self, samples, **options):
                calls.append(("samples", len(samples), options))
                return iter([SimpleNamespace(text=" 空 ", start=.2, end=.7)]), None
        with patch.dict(sys.modules, {"faster_whisper": SimpleNamespace(WhisperModel=Whisper),
                                      "librosa": SimpleNamespace(load=load_audio)}), \
             patch("soramimic_score.models._release"):
            lines = create_adapters(self.config).lyric_recoverer(self.root / "audio.wav", 10, 12)
        self.assertIn(("load", str(self.root / "audio.wav"),
                       {"sr": 16000, "mono": True, "offset": 10, "duration": 2}), calls)
        self.assertEqual(lines, (LyricLine("空", 10.2, 10.7),))
        self.assertIn(("samples", 32000, {"language": "ja", "vad_filter": False,
                                          "condition_on_previous_text": False,
                                          "temperature": 0.0}), calls)

    def test_template_retry_detects_language_and_preserves_local_clock(self):
        calls = []
        class Whisper:
            def __init__(self, *_args, **_kwargs):
                pass
            def transcribe(self, samples, **options):
                calls.append(options)
                return iter([SimpleNamespace(text=" Follow the light ", start=.2, end=.7)]), None
        with patch.dict(sys.modules, {
            "faster_whisper": SimpleNamespace(WhisperModel=Whisper),
            "librosa": SimpleNamespace(load=lambda *_args, **_kwargs: ([0.] * 32000, 16000)),
        }), patch("soramimic_score.models._release"):
            lines = create_adapters(self.config).template_lyric_recoverer(
                self.root / "audio.wav", 10, 12)
        self.assertEqual(lines, (LyricLine("Follow the light", 10.2, 10.7),))
        self.assertIsNone(calls[0]["language"])
        self.assertFalse(calls[0]["condition_on_previous_text"])
        self.assertEqual(calls[0]["temperature"], 0.)

    def test_shared_template_retry_requires_explicit_language_detection_acknowledgment(self):
        calls = []
        class Shared:
            def run(self, kind, audio, parameters):
                calls.append(parameters)
                return response

        response = {"requested_language": None, "requested_temperature": 0.,
                    "lines": [{"text": "Follow the light", "start_sec": .2, "end_sec": .7}]}
        with patch.dict(sys.modules, {
            "librosa": SimpleNamespace(load=lambda *_args, **_kwargs: ([0.] * 32000, 16000)),
            "soundfile": SimpleNamespace(write=lambda path, *_args, **_kwargs:
                                         Path(path).write_bytes(b"window")),
        }):
            adapters = create_adapters(self.config, shared=Shared())
            self.assertEqual(adapters.template_lyric_recoverer(self.root / "audio.wav", 10, 12),
                             (LyricLine("Follow the light", 10.2, 10.7),))
            self.assertIsNone(calls[0]["language"])
            for rejected in ({"requested_language": "ja", "requested_temperature": 0.},
                             {"requested_temperature": 0.},
                             {"requested_language": None}):
                with self.subTest(rejected=rejected):
                    response = rejected | {"lines": []}
                    with self.assertRaisesRegex(RuntimeError, "settings are unsupported"):
                        adapters.template_lyric_recoverer(self.root / "audio.wav", 10, 12)

    @unittest.skipUnless(importlib.util.find_spec("soramimic_yomi") and importlib.util.find_spec("MeCab"),
                         "audio dependencies not installed")
    def test_real_pronunciation_uses_yomi_and_handles_english(self):
        readings = dictionary_readings(None, [LyricLine("春が来た。")])
        self.assertEqual(readings[0].kana, "ハルガキタ")
        self.assertEqual(readings[0].source, "soramimic-yomi")
        readings = dictionary_readings(None, [LyricLine("hello")])
        self.assertEqual(readings[0].kana, "ハロー")
        self.assertEqual(readings[0].source, "soramimic-yomi")

    def test_cli_runs_configured_audio_pipeline_and_writes_loadable_json(self):
        from soramimic_score import AudioAdapters
        audio = self.root / "input.wav"
        self.write_audio(audio)
        output = self.root / "score.json"
        adapters = AudioAdapters(
            fixtures.AudioPipelineTests._readings, fixtures.AudioPipelineTests._moras,
            fixtures.AudioPipelineTests._melody,
            lambda path: (LyricLine("空", 0, .4), LyricLine("耳", .4, .8)),
        )
        with patch("soramimic_score.models.create_adapters", return_value=adapters) as factory, \
             patch("soramimic_score.models.separate_vocals") as separator:
            with contextlib.redirect_stdout(io.StringIO()):
                result = analyze_main([str(audio), "--output", str(output),
                                       "--sheetsage-model", str(self.config.sheetsage_model),
                                       "--sheetsage-base", str(self.config.sheetsage_base)])
        self.assertEqual(result, 0)
        self.assertEqual(factory.call_args.args[0], self.config)
        self.assertEqual(separator.call_count, 1)
        self.assertIsNotNone(factory.call_args.kwargs["vocals_path"])
        self.assertEqual(load(output).score.canonical_text, "空\n耳")

    def test_cli_failure_does_not_create_output(self):
        output = self.root / "missing.json"
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            analyze_main([str(self.root / "absent.wav"), "--output", str(output),
                          "--sheetsage-model", "missing", "--sheetsage-base", "missing"])
        self.assertEqual(caught.exception.code, 1)
        self.assertFalse(output.exists())

    def test_cli_rejects_overwriting_audio(self):
        audio = self.root / "input.wav"
        audio.write_bytes(b"preserve me")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            analyze_main([str(audio), "--output", str(audio),
                          "--sheetsage-model", "missing", "--sheetsage-base", "missing"])
        self.assertEqual(audio.read_bytes(), b"preserve me")

    def test_model_configuration_and_custom_adapters_are_exclusive(self):
        path = self.root / "input.wav"
        path.touch()
        with self.assertRaisesRegex(ValueError, "not both"):
            analyze_audio(path, adapters=object(), model_config=self.config)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("librosa"),
                         "audio dependencies not installed")
    def test_ctc_missing_small_kana_keeps_mora_text_and_timing(self):
        import numpy as np
        import soundfile as sf
        import torch
        from soramimic_score import ReadingSelection

        path = self.root / "small-kana.wav"
        sf.write(path, np.zeros(16000), 16000)
        class Processor:
            tokenizer = SimpleNamespace(get_vocab=lambda: {"<pad>": 0, "ク": 1, "ワ": 2, "カ": 3})
            def __call__(self, samples, **kwargs):
                return SimpleNamespace(input_values=torch.tensor(samples).unsqueeze(0))
        class Model:
            config = SimpleNamespace(conv_stride=[320], pad_token_id=0)
            def eval(self):
                return self
            def to(self, device):
                return self
            def __call__(self, values):
                logits = torch.full((1, values.shape[1] // 320, 4), -10.)
                logits[:, :, 0] = 0
                for frame, token in ((30, 1), (40, 2), (50, 3)):
                    logits[0, frame, token] = 10
                return SimpleNamespace(logits=logits)
        with patch("transformers.AutoProcessor.from_pretrained", return_value=Processor()), \
             patch("transformers.Wav2Vec2ForCTC.from_pretrained", return_value=Model()) as factory:
            align = create_adapters(self.config).mora_aligner
            moras = align(path, (LyricLine("声", 0, 1),),
                          (ReadingSelection("クヮヵ", "test", 1),))
            self.assertEqual([m.kana for m in moras], ["クヮ", "ヵ"])
            self.assertEqual([m.mora_index for m in moras], [0, 1])
            self.assertAlmostEqual(moras[0].start_sec, .1)
            self.assertAlmostEqual(moras[1].start_sec, .5)
            with self.assertRaisesRegex(ValueError, "CTC vocabulary"):
                create_adapters(self.config).mora_aligner(
                    path, (LyricLine("声", 0, 1),), (ReadingSelection("ア", "test", 1),))
            self.assertEqual(factory.call_count, 1)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("librosa"),
                         "audio dependencies not installed")
    def test_ctc_inference_boundary_uses_real_forced_alignment(self):
        import numpy as np
        import soundfile as sf
        import torch
        from soramimic_score import ReadingSelection

        path = self.root / "audio.wav"
        sf.write(path, np.zeros(16000), 16000)
        class Processor:
            tokenizer = SimpleNamespace(get_vocab=lambda: {"<pad>": 0, "ア": 1, "あ": 2})
            def __call__(self, samples, **kwargs):
                return SimpleNamespace(input_values=torch.tensor(samples).unsqueeze(0))
        class Model:
            config = SimpleNamespace(conv_stride=[320], pad_token_id=0)
            def eval(self):
                return self
            def to(self, device):
                return self
            def __call__(self, values):
                logits = torch.full((1, values.shape[1] // 320, 3), -10.0)
                logits[:, :, 0] = 0
                # The second event uses a hiragana alias of the same target.
                logits[0, 30, 1] = 10
                logits[0, 40, 2] = 10
                return SimpleNamespace(logits=logits)
        with patch("transformers.AutoProcessor.from_pretrained", return_value=Processor()) as processor_factory, \
             patch("transformers.Wav2Vec2ForCTC.from_pretrained", return_value=Model()) as model_factory:
            align = create_adapters(self.config).mora_aligner
            readings = (ReadingSelection("アア", "test", 1),)
            moras = align(path, (LyricLine("ああ", 0, 1),), readings)
            self.assertEqual([m.kana for m in moras], ["ア", "ア"])
            self.assertAlmostEqual(moras[0].start_sec, .1)
            self.assertAlmostEqual(moras[1].start_sec, .3)
            self.assertGreater(moras[1].confidence, .99)
            known = align(path, (LyricLine("ああ"),), readings)
            self.assertEqual(moras, known)
            self.assertEqual(processor_factory.call_count, 1)
            self.assertEqual(model_factory.call_count, 1)

            # Supplied text gets a coarse CTC window, then the selected reading
            # is aligned afresh. No lyric recognizer is involved.
            with patch("soramimic_score.models.dictionary_readings", return_value=(
                    ReadingSelection("アア", "soramimic-yomi", 1, ("アア", "ア")),)), \
                 patch("soramimic_score.models.transcribe_kana_views", return_value={"mix": ("ア",)}) as kana:
                adapters = create_adapters(self.config)
                selected = adapters.reading_selector(path, (LyricLine("ああ"),))
                self.assertEqual(selected[0].kana, "ア")
                final = adapters.mora_aligner(path, (LyricLine("ああ"),), selected)
                self.assertEqual(len(final), 1)
                self.assertEqual(kana.call_args.args[1], ((0.0, 1.0),))
                self.assertEqual(model_factory.call_count, 2)

            # Forced alignment uses the separated stem on the original clock.
            import librosa
            vocal_path = self.root / "vocals.wav"
            sf.write(vocal_path, np.zeros(16000), 16000)
            with patch("librosa.load", wraps=librosa.load) as audio_loader:
                vocal_moras = create_adapters(self.config, vocals_path=vocal_path).mora_aligner(
                    path, (LyricLine("ああ", 0, 1),), readings)
            self.assertEqual(audio_loader.call_args.args[0], str(vocal_path))
            self.assertIn("demucs-htdemucs", vocal_moras[0].source)
            self.assertEqual([m.start_sec for m in vocal_moras], [m.start_sec for m in moras])
            with self.assertRaises(CTCWindowCapacityError) as error:
                align(path, (LyricLine("ああ", 0, .02),), readings)
            self.assertEqual(error.exception.line_index, 0)

            # A 20.5s clip places EOF in the first core's right context.
            # The second core must not repeat those frames on the song clock.
            sf.write(path, np.zeros(328000), 16000)
            chunk_lengths = []
            original_cat = torch.cat
            def record_cat(chunks, *args, **kwargs):
                chunk_lengths.extend(len(chunk) for chunk in chunks)
                return original_cat(chunks, *args, **kwargs)
            with patch("torch.cat", side_effect=record_cat):
                align(path, (LyricLine("ああ", 0, 1),), readings)
            self.assertEqual(chunk_lengths, [1000, 75])

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("librosa"),
                         "audio dependencies not installed")
    def test_sheetsage_uses_local_models_and_parses_real_output_file(self):
        import numpy as np
        import soundfile as sf
        path = self.root / "audio.wav"
        sf.write(path, np.zeros(16000), 16000)
        class Model:
            def eval(self):
                return self
            def to(self, device):
                return self
            def transcribe(self, samples, **options):
                self_options.update(options)
                (Path(options["output_dir"]) / "melody_vocal.lab").write_text("0\t1\t60\n")
        self_options = {}
        with patch("transformers.AutoModel.from_pretrained", return_value=Model()) as loader:
            notes = create_adapters(self.config).melody_transcriber(path)
        self.assertTrue(loader.call_args.kwargs["local_files_only"])
        self.assertEqual(loader.call_args.kwargs["base_model_path"], str(self.config.sheetsage_base))
        self.assertTrue(self_options["melody_only"])
        self.assertEqual(notes[0].midi_pitch, 60)
        self.assertFalse(Path(self_options["output_dir"]).exists())
