import io
from threading import Event
import shutil
import sqlite3
import subprocess
import tempfile
import time
import unittest
import wave
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

from fastapi.testclient import TestClient

from soramimic_score.audio import AudioPipelineError
from soramimic_score.web import _prune, create_app
from soramimic_score.exports import export_musicxml
from tests.test_document import ScoreDocumentTests


def wav_bytes():
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 1600)
    return out.getvalue()


class ScoreWebTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.document = ScoreDocumentTests().score_document()
        self.received = []

        def analyze(path, *, model_config, lyrics):
            self.received.append((Path(path).name, lyrics))
            return self.document

        self.client = TestClient(create_app(data_root=Path(self.temporary.name),
                                            analyzer=analyze, public=True))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temporary.cleanup()

    def submit(self, lyrics=""):
        return self.client.post("/api/jobs", files={"audio": ("song.wav", wav_bytes(), "audio/wav")},
                                data={"lyrics": lyrics})

    def test_upload_analysis_preview_and_exports(self):
        with patch.dict("os.environ", {"SORAMIMIC_SCORE_SHEETSAGE_MODEL": "a",
                                    "SORAMIMIC_SCORE_SHEETSAGE_BASE": "b"}):
            response = self.submit("カ")
            self.assertEqual(response.status_code, 200)
            job = response.json()["id"]
            for _ in range(100):
                state = self.client.get(f"/api/jobs/{job}").json()["state"]
                if state == "done":
                    break
                time.sleep(.02)
            self.assertEqual(state, "done")
        self.assertEqual(self.received, [("input.wav", ("カ",))])
        score = self.client.get(f"/api/jobs/{job}/score").json()
        self.assertEqual(score["lines"][0]["text"], "カ")
        self.assertTrue(score["notes"])
        self.assertEqual(len({m["id"] for m in score["moras"]}), len(score["moras"]))
        self.assertEqual(self.client.get(f"/api/jobs/{job}/audio").status_code, 200)
        for format in ("json", "mid", "musicxml", "srt", "lrc"):
            response = self.client.get(f"/api/jobs/{job}/download/{format}")
            self.assertEqual(response.status_code, 200, format)
            self.assertTrue(response.content)
        ElementTree.fromstring(self.client.get(f"/api/jobs/{job}/download/musicxml").content)
        self.assertTrue(self.client.get(f"/api/jobs/{job}/download/mid").content.startswith(b"MThd"))

    def test_invalid_upload_does_not_consume_quota(self):
        with patch("soramimic_score.web.QUOTA_PER_DAY", 1):
            bad = self.client.post("/api/jobs", files={"audio": ("song.wav", b"broken")})
            self.assertEqual(bad.status_code, 400)
            with patch.dict("os.environ", {"SORAMIMIC_SCORE_SHEETSAGE_MODEL": "a",
                                        "SORAMIMIC_SCORE_SHEETSAGE_BASE": "b"}):
                self.assertEqual(self.submit().status_code, 200)
                self.assertEqual(self.submit().status_code, 429)

    def test_spoken_caption_uses_render_timing_with_estimated_provenance(self):
        from soramimic_score.document import from_linked_observations
        from soramimic_score.spoken import add_spoken_fallback
        from soramimic_score.vocal_activity import VocalActivity
        from tests.test_realization import document
        self.document = add_spoken_fallback(
            from_linked_observations(document("カ", 1)), {"u0": (0, 1)},
            lambda windows: tuple(VocalActivity(-20, -3, 1, True) for _ in windows),
            fill_unpitched_lines=True,
        )
        with patch.dict("os.environ", {"SORAMIMIC_SCORE_SHEETSAGE_MODEL": "a",
                                     "SORAMIMIC_SCORE_SHEETSAGE_BASE": "b"}):
            job = self.submit().json()["id"]
            for _ in range(100):
                if self.client.get(f"/api/jobs/{job}").json()["state"] == "done":
                    break
                time.sleep(.02)
        score = self.client.get(f"/api/jobs/{job}/score").json()
        self.assertTrue(score["notes"][0]["spoken"])
        self.assertEqual(score["moras"][0]["end"], 1)
        self.assertEqual(score["moras"][0]["source"], "estimated")

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
    def test_mp3_upload_is_decoded_before_analysis(self):
        encoded = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "wav",
                                  "-i", "pipe:0", "-f", "mp3", "pipe:1"],
                                 input=wav_bytes(), capture_output=True, check=True).stdout
        with patch.dict("os.environ", {"SORAMIMIC_SCORE_SHEETSAGE_MODEL": "a",
                                    "SORAMIMIC_SCORE_SHEETSAGE_BASE": "b"}):
            response = self.client.post("/api/jobs",
                                        files={"audio": ("song.mp3", encoded, "audio/mpeg")})
            self.assertEqual(response.status_code, 200)
            job = response.json()["id"]
            for _ in range(100):
                state = self.client.get(f"/api/jobs/{job}").json()["state"]
                if state in ("done", "failed"):
                    break
                time.sleep(.02)
            self.assertEqual(state, "done")
        self.assertEqual(self.client.get(f"/api/jobs/{job}/audio").status_code, 200)
        self.assertTrue(self.client.get(f"/api/jobs/{job}/audio-wav").content.startswith(b"RIFF"))

    def test_running_stage_is_visible(self):
        started, release = Event(), Event()
        def slow_analyzer(*args, **kwargs):
            started.set()
            release.wait(5)
            return self.document
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(data_root=Path(directory),
                                       analyzer=slow_analyzer)) as client:
                with patch.dict("os.environ", {"SORAMIMIC_SCORE_SHEETSAGE_MODEL": "a",
                                            "SORAMIMIC_SCORE_SHEETSAGE_BASE": "b"}):
                    job = client.post("/api/jobs", files={"audio": ("song.wav", wav_bytes())}).json()["id"]
                    self.assertTrue(started.wait(5))
                    state = client.get(f"/api/jobs/{job}").json()
                    self.assertEqual(state["state"], "running")
                    self.assertEqual(state["stage"], "音源を解析しています")
                    release.set()

    def test_resinging_is_started_only_by_request(self):
        with patch.dict("os.environ", {"SORAMIMIC_SCORE_SHEETSAGE_MODEL": "a",
                                    "SORAMIMIC_SCORE_SHEETSAGE_BASE": "b"}):
            job = self.submit().json()["id"]
            for _ in range(100):
                if self.client.get(f"/api/jobs/{job}").json()["state"] == "done":
                    break
                time.sleep(.02)
        self.assertIsNone(self.client.get(f"/api/jobs/{job}/resing").json()["state"])
        def synthesize(_document, output, **kwargs):
            output.write_bytes(wav_bytes())
            kwargs["on_progress"](1, 1)
        with patch("soramimic_score.resing.available", return_value=True), patch(
            "soramimic_score.resing.synthesize", side_effect=synthesize
        ):
            self.assertEqual(self.client.post(f"/api/jobs/{job}/resing").status_code, 200)
            for _ in range(100):
                state = self.client.get(f"/api/jobs/{job}/resing").json()["state"]
                if state == "done":
                    break
                time.sleep(.02)
            self.assertEqual(state, "done")
            self.assertTrue(self.client.get(f"/api/jobs/{job}/resing/audio").content.startswith(b"RIFF"))
            with sqlite3.connect(Path(self.temporary.name) / "jobs.sqlite3") as connection:
                connection.execute("UPDATE jobs SET synth_backend=NULL WHERE id=?", (job,))
            self.assertIsNone(self.client.get(f"/api/jobs/{job}/resing").json()["state"])
            self.assertEqual(self.client.get(f"/api/jobs/{job}/resing/audio").status_code, 409)
            self.assertEqual(self.client.post(f"/api/jobs/{job}/resing").status_code, 200)
            for _ in range(100):
                if self.client.get(f"/api/jobs/{job}/resing").json()["state"] == "done":
                    break
                time.sleep(.02)
            self.assertEqual(self.client.get(f"/api/jobs/{job}/resing").json()["state"], "done")
            with sqlite3.connect(Path(self.temporary.name) / "jobs.sqlite3") as connection:
                connection.execute("UPDATE jobs SET synth_state='failed', synth_error='歌唱合成に失敗しました' "
                                   "WHERE id=?", (job,))
            self.assertEqual(self.client.post(f"/api/jobs/{job}/resing").json()["state"], "queued")
            for _ in range(100):
                if self.client.get(f"/api/jobs/{job}/resing").json()["state"] == "done":
                    break
                time.sleep(.02)
            self.assertEqual(self.client.get(f"/api/jobs/{job}/resing").json()["state"], "done")

    def test_auto_resinging_runs_after_analysis_when_enabled(self):
        def synthesize(_document, output, **kwargs):
            output.write_bytes(wav_bytes())
            kwargs["on_progress"](1, 1)
        with patch.dict("os.environ", {"SORAMIMIC_SCORE_SHEETSAGE_MODEL": "a",
                                    "SORAMIMIC_SCORE_SHEETSAGE_BASE": "b",
                                    "SORAMIMIC_SCORE_AUTO_RESING": "1"}), patch(
            "soramimic_score.resing.available", return_value=True
        ), patch("soramimic_score.resing.synthesize", side_effect=synthesize) as render:
            job = self.submit().json()["id"]
            for _ in range(100):
                if self.client.get(f"/api/jobs/{job}").json()["state"] == "done":
                    break
                time.sleep(.02)
            for _ in range(100):
                state = self.client.get(f"/api/jobs/{job}/resing").json()
                if state["state"] == "done":
                    break
                time.sleep(.02)
            self.assertEqual(state["state"], "done")
            render.assert_called_once()
            self.assertEqual(self.client.get(f"/api/jobs/{job}/resing/audio").status_code, 200)

    def test_resinging_is_unavailable_without_renderer(self):
        with patch.dict("os.environ", {"SORAMIMIC_SCORE_SHEETSAGE_MODEL": "a",
                                    "SORAMIMIC_SCORE_SHEETSAGE_BASE": "b"}):
            job = self.submit().json()["id"]
            for _ in range(100):
                if self.client.get(f"/api/jobs/{job}").json()["state"] == "done":
                    break
                time.sleep(.02)
        with patch("soramimic_score.resing.available", return_value=False):
            self.assertFalse(self.client.get("/api/capabilities").json()["resing"])
            self.assertEqual(self.client.post(f"/api/jobs/{job}/resing").status_code, 503)

    def test_guidelines_explain_retention(self):
        response = self.client.get("/guidelines")
        self.assertEqual(response.status_code, 200)
        self.assertIn("約1時間", response.text)

    def test_finished_job_can_be_deleted_immediately(self):
        with patch.dict("os.environ", {"SORAMIMIC_SCORE_SHEETSAGE_MODEL": "a",
                                    "SORAMIMIC_SCORE_SHEETSAGE_BASE": "b"}):
            job = self.submit().json()["id"]
            for _ in range(100):
                if self.client.get(f"/api/jobs/{job}").json()["state"] == "done":
                    break
                time.sleep(.02)
        self.assertEqual(self.client.delete(f"/api/jobs/{job}").json(), {"deleted": True})
        self.assertFalse((Path(self.temporary.name) / job).exists())
        self.assertEqual(self.client.get(f"/api/jobs/{job}").status_code, 404)
        self.assertEqual(self.client.get(f"/api/jobs/{job}/audio").status_code, 404)

    def test_running_job_cannot_be_deleted(self):
        root = Path(self.temporary.name)
        job = "a" * 32
        (root / job).mkdir()
        with sqlite3.connect(root / "jobs.sqlite3") as connection:
            connection.execute("INSERT INTO jobs(id,state,created,ip) VALUES(?,?,?,?)",
                               (job, "running", "2026-09-24T00:00:00+00:00", "127.0.0.1"))
        self.assertEqual(self.client.delete(f"/api/jobs/{job}").status_code, 409)
        self.assertTrue((root / job).exists())

    def test_expired_job_waits_for_running_singing_synthesis(self):
        root = Path(self.temporary.name)
        job = "b" * 32
        (root / job).mkdir()
        (root / job / "resung.wav").write_bytes(wav_bytes())
        db = root / "jobs.sqlite3"
        with sqlite3.connect(db) as connection:
            connection.execute("INSERT INTO jobs(id,state,created,ip,finished,synth_state) "
                               "VALUES(?,?,?,?,?,?)",
                               (job, "done", "2000-01-01T00:00:00+00:00", "127.0.0.1",
                                "2000-01-01T00:00:00+00:00", "running"))
        _prune(root, db)
        self.assertTrue((root / job).exists())
        with sqlite3.connect(db) as connection:
            connection.execute("UPDATE jobs SET synth_state='done' WHERE id=?", (job,))
        _prune(root, db)
        self.assertFalse((root / job).exists())

    def test_expired_result_and_audio_are_removed(self):
        with patch.dict("os.environ", {"SORAMIMIC_SCORE_SHEETSAGE_MODEL": "a",
                                    "SORAMIMIC_SCORE_SHEETSAGE_BASE": "b"}):
            job = self.submit().json()["id"]
            for _ in range(100):
                if self.client.get(f"/api/jobs/{job}").json()["state"] == "done":
                    break
                time.sleep(.02)
        self.assertEqual(self.client.get(f"/api/jobs/{job}").json()["state"], "done")
        root = Path(self.temporary.name)
        db = root / "jobs.sqlite3"
        with sqlite3.connect(db) as connection:
            connection.execute("UPDATE jobs SET finished='2000-01-01T00:00:00+00:00' "
                               "WHERE id=?", (job,))
        _prune(root, db)
        self.assertFalse((root / job).exists())
        self.assertEqual(self.client.get(f"/api/jobs/{job}/audio").status_code, 404)

    def test_private_job_requires_unguessable_id(self):
        self.assertEqual(self.client.get("/api/jobs/missing").status_code, 404)
        self.assertEqual(self.client.get("/api/jobs/" + "0" * 32 + "/audio").status_code, 404)

    def test_missing_melody_reports_a_useful_error(self):
        def no_melody(*args, **kwargs):
            raise AudioPipelineError("melody", "no melody notes were produced")

        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(data_root=Path(directory), analyzer=no_melody)) as client:
                with patch.dict("os.environ", {"SORAMIMIC_SCORE_SHEETSAGE_MODEL": "a",
                                            "SORAMIMIC_SCORE_SHEETSAGE_BASE": "b"}):
                    response = client.post("/api/jobs", files={"audio": ("song.wav", wav_bytes())})
                    job = response.json()["id"]
                    for _ in range(100):
                        status = client.get(f"/api/jobs/{job}").json()
                        if status["state"] == "failed":
                            break
                        time.sleep(.02)
                self.assertEqual(status["state"], "failed")
                self.assertIn("音符を検出できませんでした", status["error"])

    def test_musicxml_splits_long_note_across_measures(self):
        slot = replace(self.document.score.synthesis_plan[0], end_sec=4.5)
        document = replace(self.document,
                           score=replace(self.document.score, synthesis_plan=(slot,)))
        root = ElementTree.fromstring(export_musicxml(document))
        measures = root.findall("./part/measure")
        self.assertEqual(len(measures), 2)
        self.assertEqual([n.findtext("duration") for m in measures
                          for n in m.findall("note")], ["4000", "500"])
        self.assertEqual(measures[0].find("note/tie").get("type"), "start")
        self.assertEqual(measures[1].find("note/tie").get("type"), "stop")


if __name__ == "__main__":
    unittest.main()
