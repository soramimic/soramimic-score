import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("librosa"), "audio dependencies not installed")
class WhisperReadingBackendTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import soundfile as sf
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source.wav"
        sf.write(self.source, np.arange(48000, dtype=np.float32) / 48000, 16000,
                 subtype="FLOAT")
        self.config = SimpleNamespace(whisper_model="large-v3", device="cpu",
                                      local_files_only=True)

    def test_local_model_is_shared_across_views_and_receives_audio_only(self):
        from soramimic_score.acoustic import transcribe_whisper_views
        model = SimpleNamespace(transcribe=lambda *a, **kw: None)
        with patch("faster_whisper.WhisperModel", return_value=model) as loader, \
             patch.object(model, "transcribe", return_value=(
                 iter(()), None)) as transcribe, \
             patch("soramimic_score.acoustic.release_memory") as release:
            transcribe.side_effect = lambda *a, **kw: (iter([SimpleNamespace(text="読み")]), None)
            result = transcribe_whisper_views({"mix": self.source, "vocals": self.source},
                                              [(2, 3), (0, 1)], self.config)
        loader.assert_called_once_with("large-v3", device="cpu", compute_type="int8",
                                       local_files_only=True)
        self.assertEqual(result, {"mix": ("読み", "読み"), "vocals": ("読み", "読み")})
        self.assertEqual(transcribe.call_count, 4)
        for call in transcribe.call_args_list:
            self.assertEqual(len(call.args[0]), 16000)
            self.assertEqual(call.kwargs, dict(language="ja", vad_filter=False,
                condition_on_previous_text=False, temperature=0.))
        self.assertGreater(transcribe.call_args_list[0].args[0][0],
                           transcribe.call_args_list[1].args[0][0])
        release.assert_called_once()

    def test_shared_windows_preserve_order_and_remove_temporary_audio(self):
        import soundfile as sf
        from soramimic_score.acoustic import transcribe_whisper_views
        excerpts = []
        def run(kind, path, parameters):
            excerpts.append(path)
            samples, rate = sf.read(path)
            self.assertEqual((kind, rate, len(samples)), ("whisper", 16000, 16000))
            self.assertEqual(parameters, dict(model_size="large-v3", device="auto",
                language="ja", vad_filter=False, condition_on_previous_text=False,
                temperature=0.))
            return dict(requested_language="ja", requested_temperature=0.,
                        lines=[{"text": "後" if samples[0] > .5 else "前"}])
        result = transcribe_whisper_views({"mix": self.source}, [(2, 3), (0, 1)],
                    self.config, shared=SimpleNamespace(run=run))
        self.assertEqual(result, {"mix": ("後", "前")})
        self.assertTrue(all(not path.exists() for path in excerpts))

    def test_invalid_shared_reply_fails_and_cleans_excerpt(self):
        from soramimic_score.acoustic import transcribe_whisper_views
        for reply in ({}, {"requested_language": "en", "requested_temperature": 0., "lines": []},
                      {"requested_language": "ja", "requested_temperature": 1., "lines": []},
                      {"requested_language": "ja", "requested_temperature": 0., "lines": [{"text": 1}]}):
            excerpts = []
            def run(_kind, path, _parameters):
                excerpts.append(path)
                return reply
            with self.subTest(reply=reply), self.assertRaisesRegex(RuntimeError, "response is invalid"):
                transcribe_whisper_views({"mix": self.source}, [(0, 1)], self.config,
                                        shared=SimpleNamespace(run=run))
            self.assertTrue(all(not path.exists() for path in excerpts))

    def test_invalid_windows_never_reach_inference(self):
        from soramimic_score.acoustic import transcribe_whisper_views
        from unittest.mock import Mock
        shared = Mock()
        for window in ((-1, 1), (2, 2), (0, 4), (0, float("nan"))):
            with self.subTest(window=window), self.assertRaises(ValueError):
                transcribe_whisper_views({"mix": self.source}, [window], self.config,
                                        shared=shared)
        shared.run.assert_not_called()
