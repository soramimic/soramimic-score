"""Private-by-default job API for Soramimic Score."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import logging
import os
import secrets
import shutil
import sqlite3
from threading import Event, Thread
import time

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response

from .audio import AudioPipelineError
from .document import load
from .exports import EXPORTS
from .models import ModelConfig
from .ir import has_usable_timing
from .media import AUDIO_SUFFIXES, decode_audio, probe_audio


MAX_WAV_BYTES = 100 * 1024 * 1024
MAX_DURATION_SEC = 15 * 60
QUOTA_PER_DAY = 100
RETENTION_SEC = 3600
logger = logging.getLogger(__name__)


def _now_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _job_id(value: str) -> str:
    if len(value) != 32 or any(char not in "0123456789abcdef" for char in value):
        raise HTTPException(404)
    return value


def _request_ip(request: Request) -> str:
    peer = request.client.host if request.client else ""
    # Cloudflare Tunnel terminates at loopback. Never trust headers from other peers.
    if peer in ("127.0.0.1", "::1"):
        forwarded = request.headers.get("cf-connecting-ip", "")
        if forwarded:
            from ipaddress import ip_address
            try:
                return str(ip_address(forwarded))
            except ValueError:
                pass
    return peer


def _prune(root: Path, db: Path) -> None:
    cutoff = datetime.fromtimestamp(time.time() - RETENTION_SEC, timezone.utc).isoformat()
    with sqlite3.connect(db) as conn:
        old = conn.execute("SELECT id FROM jobs WHERE finished<? AND state IN ('done','failed') "
                           "AND COALESCE(synth_state,'') NOT IN ('queued','running')",
                           (cutoff,)).fetchall()
        conn.executemany("DELETE FROM jobs WHERE id=?", old)
        conn.execute("DELETE FROM quota WHERE day<?", (_now_day(),))
    for (job,) in old:
        shutil.rmtree(root / job, ignore_errors=True)


def create_app(*, data_root: Path | None = None, analyzer=None, public: bool | None = None) -> FastAPI:
    """Create a single-worker app. Keep its data directory outside the Git checkout."""
    root = data_root or Path(os.environ.get("SORAMIMIC_SCORE_DATA", "work/web"))
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    db = root / "jobs.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, state TEXT NOT NULL, "
                     "error TEXT, created TEXT NOT NULL, ip TEXT NOT NULL, "
                     "stage TEXT, finished TEXT)")
        if "stage" not in {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}:
            conn.execute("ALTER TABLE jobs ADD COLUMN stage TEXT")
        if "finished" not in {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}:
            conn.execute("ALTER TABLE jobs ADD COLUMN finished TEXT")
        for column in ("synth_state", "synth_stage", "synth_error", "synth_backend"):
            if column not in {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} TEXT")
        conn.execute("CREATE TABLE IF NOT EXISTS quota (day TEXT NOT NULL, ip TEXT NOT NULL, "
                     "used INTEGER NOT NULL, PRIMARY KEY(day, ip))")
        conn.execute("UPDATE jobs SET state='failed', error='処理が中断されました' "
                     "WHERE state IN ('queued', 'running')")
        conn.execute("UPDATE jobs SET finished=COALESCE(finished, created) "
                     "WHERE state IN ('done', 'failed')")
        conn.execute("UPDATE jobs SET synth_state='failed', synth_error='合成が中断されました' "
                     "WHERE synth_state IN ('queued', 'running')")
    db.chmod(0o600)
    is_public = public if public is not None else os.environ.get("SORAMIMIC_SCORE_PUBLIC") == "1"
    pool = ThreadPoolExecutor(max_workers=1)
    synth_pool = ThreadPoolExecutor(max_workers=1)
    stop_cleanup = Event()

    def cleanup_periodically() -> None:
        while not stop_cleanup.is_set():
            try:
                _prune(root, db)
            except Exception:
                logger.exception("score cleanup failed")
            stop_cleanup.wait(300)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        cleaner = Thread(target=cleanup_periodically, daemon=True)
        cleaner.start()
        try:
            yield
        finally:
            stop_cleanup.set()
            cleaner.join(timeout=5)
            pool.shutdown(wait=True)
            synth_pool.shutdown(wait=True)

    app = FastAPI(title="Soramimic Score", docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)

    def analyze(job: str, supplied: bool, source_name: str) -> None:
        from .audio import analyze_audio
        from .document import dump
        job_dir = root / job
        def progress(stage: str) -> None:
            with sqlite3.connect(db) as conn:
                conn.execute("UPDATE jobs SET state='running', stage=? WHERE id=?", (stage, job))
        try:
            progress("音声を読み込んでいます")
            decode_audio(job_dir / source_name, job_dir / "input.wav",
                         max_duration=MAX_DURATION_SEC)
            config = ModelConfig(
                sheetsage_model=Path(os.environ["SORAMIMIC_SCORE_SHEETSAGE_MODEL"]),
                sheetsage_base=Path(os.environ["SORAMIMIC_SCORE_SHEETSAGE_BASE"]),
                device=os.environ.get("SORAMIMIC_SCORE_DEVICE", "cpu"),
                local_files_only=os.environ.get("SORAMIMIC_SCORE_LOCAL_ONLY") == "1",
                shared_inference_url=os.environ.get("SORAMIMIC_AUDIO_INFERENCE_URL"),
                shared_inference_priority="public" if is_public else "dev",
                romaji_model=Path(os.environ["SORAMIMIC_SCORE_ROMAJI_MODEL"])
                    if os.environ.get("SORAMIMIC_SCORE_ROMAJI_MODEL") else None,
            )
            lyrics = tuple(line for line in (job_dir / "lyrics.txt").read_text(
                encoding="utf-8").splitlines() if line.strip()) if supplied else None
            if analyzer is None:
                result = analyze_audio(job_dir / "input.wav", model_config=config,
                                       lyrics=lyrics, on_progress=progress,
                                       accompaniment_path=job_dir / "accompaniment.flac")
            else:
                progress("音源を解析しています")
                result = analyzer(job_dir / "input.wav", model_config=config, lyrics=lyrics)
            progress("書き出しを準備しています")
            dump(result, job_dir / "score.json")
            from .resing import available
            auto_resing = os.environ.get("SORAMIMIC_SCORE_AUTO_RESING") == "1" and available()
            with sqlite3.connect(db) as conn:
                conn.execute("UPDATE jobs SET state='done', stage=NULL, finished=?, "
                             "synth_state=?, synth_backend=?, synth_stage=? WHERE id=?",
                             (datetime.now(timezone.utc).isoformat(),
                              "queued" if auto_resing else None,
                              "prettypitch" if auto_resing else None,
                              "順番を待っています" if auto_resing else None, job))
            if auto_resing:
                synth_pool.submit(resing, job)
        except Exception as exc:
            logger.exception("score analysis failed for job %s", job)
            message = "解析に失敗しました。音源と設定を確認してください"
            if isinstance(exc, AudioPipelineError):
                if exc.stage == "melody" and "no melody notes were produced" in str(exc):
                    message = "歌唱の音符を検出できませんでした。別の歌唱音源をお試しください"
                elif exc.stage == "melody" and "At least two decoded beats" in str(exc):
                    message = "音符の推定に必要な長さが足りません。長めの歌唱音源をお試しください"
                elif exc.stage == "lyrics" and "no lyric lines were produced" in str(exc):
                    message = "歌詞を認識できませんでした。歌声が聞こえる音源をお試しください"
            with sqlite3.connect(db) as conn:
                conn.execute("UPDATE jobs SET state='failed', stage=NULL, error=?, finished=? "
                             "WHERE id=?", (message, datetime.now(timezone.utc).isoformat(), job))

    def resing(job: str) -> None:
        from .resing import synthesize
        from .audio import is_credit_hallucination
        job_dir = root / job
        output = job_dir / "resung.wav"
        try:
            document = load(job_dir / "score.json")
            duration = probe_audio(job_dir / "input.wav", max_duration=MAX_DURATION_SEC)
            excluded = (frozenset(line.utterance_id for line in document.score.canonical
                                  if is_credit_hallucination(line.text))
                        if not (job_dir / "lyrics.txt").exists() else frozenset())
            def on_progress(done: int, total: int) -> None:
                with sqlite3.connect(db) as conn:
                    conn.execute("UPDATE jobs SET synth_state='running', synth_stage=? WHERE id=?",
                                 (f"PrettyPitchで歌唱を合成中 ({done}/{total})", job))
            synthesize(document, output,
                       duration_sec=duration, on_progress=on_progress,
                       excluded_utterance_ids=excluded)
            accompaniment = job_dir / "accompaniment.flac"
            if accompaniment.is_file():
                from .resing import mix_accompaniment
                with sqlite3.connect(db) as conn:
                    conn.execute("UPDATE jobs SET synth_stage=? WHERE id=?",
                                 ("伴奏を重ねています", job))
                mix_accompaniment(output, accompaniment)
            with sqlite3.connect(db) as conn:
                conn.execute("UPDATE jobs SET synth_state='done', synth_stage=NULL, finished=? WHERE id=?",
                             (datetime.now(timezone.utc).isoformat(), job))
        except Exception:
            logger.exception("score resinging failed for job %s", job)
            output.unlink(missing_ok=True)
            with sqlite3.connect(db) as conn:
                conn.execute("UPDATE jobs SET synth_state='failed', synth_stage=NULL, "
                             "synth_error='歌唱合成に失敗しました', finished=? WHERE id=?",
                             (datetime.now(timezone.utc).isoformat(), job))

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.get("/api/capabilities")
    def capabilities():
        from .resing import available
        return {"resing": available()}

    @app.get("/", response_class=HTMLResponse)
    def home():
        return FileResponse(Path(__file__).with_name("score.html"), media_type="text/html")

    @app.get("/guidelines", response_class=HTMLResponse)
    def guidelines():
        return FileResponse(Path(__file__).with_name("guidelines.html"), media_type="text/html")

    @app.post("/api/jobs")
    async def submit(request: Request, audio: UploadFile = File(...), lyrics: str = Form("")):
        _prune(root, db)
        from .japanese import normalize_lyric_input
        try:
            normalize_lyric_input(lyrics)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if len(lyrics.encode("utf-8")) > 200_000:
            raise HTTPException(413, "歌詞が大きすぎます")
        suffix = Path(audio.filename or "").suffix.lower()
        if suffix not in AUDIO_SUFFIXES:
            raise HTTPException(400, "MP3、M4A、WAVなどの音声ファイルを選んでください")
        job = secrets.token_hex(16)
        job_dir = root / job
        job_dir.mkdir(mode=0o700)
        source_name = f"source{suffix}"
        path = job_dir / source_name
        size = 0
        try:
            with path.open("wb") as output:
                while chunk := await audio.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_WAV_BYTES:
                        raise HTTPException(413, "WAVは100 MB以下にしてください")
                    output.write(chunk)
            try:
                probe_audio(path, max_duration=MAX_DURATION_SEC)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            if lyrics.strip():
                (job_dir / "lyrics.txt").write_text(lyrics, encoding="utf-8")
            ip = _request_ip(request)
            with sqlite3.connect(db, timeout=30, isolation_level=None) as conn:
                conn.execute("BEGIN IMMEDIATE")
                if is_public:
                    count = conn.execute("SELECT used FROM quota WHERE day=? AND ip=?",
                                         (_now_day(), ip)).fetchone()
                    if count and count[0] >= QUOTA_PER_DAY:
                        raise HTTPException(429, "本日の解析回数は100回に達しました")
                    conn.execute("INSERT INTO quota(day, ip, used) VALUES(?, ?, 1) "
                                 "ON CONFLICT(day, ip) DO UPDATE SET used=used+1",
                                 (_now_day(), ip))
                conn.execute("INSERT INTO jobs(id,state,created,ip,stage) VALUES(?,?,?,?,?)",
                             (job, "queued", datetime.now(timezone.utc).isoformat(), ip, None))
                conn.commit()
            pool.submit(analyze, job, bool(lyrics.strip()), source_name)
        except Exception:
            if not path.exists() or not _known_job(db, job):
                for item in job_dir.iterdir():
                    item.unlink()
                job_dir.rmdir()
            raise
        return {"id": job, "state": "queued"}

    @app.get("/api/jobs/{job}")
    def status(job: str):
        job = _job_id(job)
        with sqlite3.connect(db) as conn:
            row = conn.execute("SELECT state,error,stage,created FROM jobs WHERE id=?", (job,)).fetchone()
        if not row:
            raise HTTPException(404)
        return {"id": job, "state": row[0], "error": row[1], "stage": row[2],
                "created": row[3]}

    @app.delete("/api/jobs/{job}")
    def delete_job(job: str):
        job = _job_id(job)
        with sqlite3.connect(db, timeout=30, isolation_level=None) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT state,synth_state FROM jobs WHERE id=?",
                               (job,)).fetchone()
            if row is None:
                raise HTTPException(404)
            if row[0] in ("queued", "running") or row[1] in ("queued", "running"):
                raise HTTPException(409, "処理中は削除できません。完了後にもう一度お試しください")
            try:
                shutil.rmtree(root / job)
            except FileNotFoundError:
                pass
            conn.execute("DELETE FROM jobs WHERE id=?", (job,))
            conn.commit()
        return {"deleted": True}

    @app.get("/api/jobs/{job}/score")
    def score(job: str):
        document = _completed(root, db, _job_id(job))
        from .audio import is_credit_hallucination
        from .ruby import ruby_segments
        excluded = (frozenset(line.utterance_id for line in document.score.canonical
                              if is_credit_hallucination(line.text))
                    if not (root / job / "lyrics.txt").exists() else frozenset())
        slots = [slot for slot in document.score.synthesis_plan
                 if slot.utterance_id not in excluded]
        units = {x.singing_unit_id: x for x in document.score.performed}
        observed = {x.id: x for x in document.observations.singing_units}
        moras = {x.id: x.text for x in document.observations.moras}
        by_unit = {}
        spoken_lines = {slot.singing_unit_id for slot in slots
                        if slot.timing_source == "spoken_line_proportional"}
        for slot in slots:
            interval = by_unit.setdefault(slot.singing_unit_id,
                                          [slot.start_sec, slot.end_sec, slot.utterance_id])
            interval[0] = min(interval[0], slot.start_sec)
            interval[1] = max(interval[1], slot.end_sec)
        timeline = []
        for unit_id, (start, end, line) in by_unit.items():
            ids = units[unit_id].mora_ids
            observation = observed.get(unit_id)
            aligned = (unit_id not in spoken_lines and observation is not None
                       and has_usable_timing(observation.consonant_start)
                       and has_usable_timing(observation.end)
                       and observation.consonant_start.time_sec < observation.end.time_sec)
            if aligned:
                start = observation.consonant_start.time_sec
                end = observation.end.time_sec
            duration = (end - start) / len(ids)
            for index, mora_id in enumerate(ids):
                timeline.append({"id": mora_id, "line": line,
                                 "text": moras[mora_id],
                                 "start": start + duration * index,
                                 "end": start + duration * (index + 1),
                                 "source": "aligned" if aligned else "estimated"})
        timeline.sort(key=lambda x: (x["start"], x["end"]))
        return {"lines": [{"id": line.utterance_id, "text": line.text,
                           "kana": line.kana, "ruby": ruby_segments(line.text, line.kana)}
                          for line in document.score.canonical
                          if line.utterance_id not in excluded],
                "notes": [{"start": s.start_sec, "end": s.end_sec, "pitch": s.midi_pitch,
                           "line": s.utterance_id, "kana": s.kana,
                           "spoken": "spoken" in s.pitch_sources} for s in slots],
                "moras": timeline}

    @app.get("/api/jobs/{job}/audio")
    def audio(job: str):
        _completed(root, db, _job_id(job))
        source = next(root.joinpath(job).glob("source.*"), None)
        if source is not None:
            return FileResponse(source)
        return FileResponse(root / job / "input.wav", media_type="audio/wav")

    @app.get("/api/jobs/{job}/audio-wav")
    def audio_wav(job: str):
        _completed(root, db, _job_id(job))
        return FileResponse(root / job / "input.wav", media_type="audio/wav")

    @app.post("/api/jobs/{job}/resing")
    def start_resing(job: str):
        job = _job_id(job)
        _completed(root, db, job)
        from .resing import available
        if not available():
            raise HTTPException(503, "このサーバーではPrettyPitchを利用できません")
        with sqlite3.connect(db) as conn:
            state, backend = conn.execute(
                "SELECT synth_state,synth_backend FROM jobs WHERE id=?",
                (job,)).fetchone()
            if state in (None, "failed") or backend != "prettypitch":
                conn.execute("UPDATE jobs SET synth_state='queued', synth_backend='prettypitch', "
                             "synth_stage='順番を待っています', "
                             "synth_error=NULL WHERE id=?", (job,))
                synth_pool.submit(resing, job)
                state = "queued"
        return {"state": state}

    @app.get("/api/jobs/{job}/resing")
    def resing_status(job: str):
        job = _job_id(job)
        _completed(root, db, job)
        with sqlite3.connect(db) as conn:
            row = conn.execute("SELECT synth_state,synth_stage,synth_error,synth_backend FROM jobs WHERE id=?",
                               (job,)).fetchone()
        if row[3] != "prettypitch":
            return {"state": None, "stage": None, "error": None}
        return {"state": row[0], "stage": row[1], "error": row[2]}

    @app.get("/api/jobs/{job}/resing/audio")
    def resing_audio(job: str):
        job = _job_id(job)
        status = resing_status(job)
        if status["state"] != "done":
            raise HTTPException(409, "歌唱合成が完了していません")
        return FileResponse(root / job / "resung.wav", media_type="audio/wav")

    @app.get("/api/jobs/{job}/download/{format}")
    def download(job: str, format: str):
        document = _completed(root, db, _job_id(job))
        if format not in EXPORTS:
            raise HTTPException(404)
        media_type, exporter = EXPORTS[format]
        return Response(exporter(document), media_type=media_type,
                        headers={"Content-Disposition": f'attachment; filename="score.{format}"'})

    return app


def _known_job(db: Path, job: str) -> bool:
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT 1 FROM jobs WHERE id=?", (job,)).fetchone() is not None


def _completed(root: Path, db: Path, job: str):
    with sqlite3.connect(db) as conn:
        state = conn.execute("SELECT state FROM jobs WHERE id=?", (job,)).fetchone()
    if not state:
        raise HTTPException(404)
    if state[0] != "done":
        raise HTTPException(409, "解析が完了していません")
    return load(root / job / "score.json")


app = create_app()
