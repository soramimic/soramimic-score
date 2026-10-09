"""Optional local inference backends; importing this module loads no models."""
from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import math
import logging
from pathlib import Path
import tempfile

from .audio import (AudioAdapters, AlignedMora, CTCWindowCapacityError,
                    LyricLine, MelodyNote)
from .audio import _run_adapter
from .acoustic import KANA_MODEL, release_memory as _release, separate_vocals, transcribe_kana_views
from .japanese import kana_to_moras, katakana
from .parenthetical import (choose_reading, reading_options, resolved_text,
                            selection_detail)
from .readings import (dictionary_readings, grouped_acoustic_windows,
                       select_acoustic_reading, token_reading_proposals)

logger = logging.getLogger(__name__)


def _transcribe_shared_kana(shared, source: Path, windows):
    """Match the worker's ordered, 24-second window contract without changing callers' order."""
    if any(not math.isfinite(start + end) or start < 0 or end <= start
           or end - start > 24.001 for start, end in windows):
        raise ValueError("Invalid shared KanaWhisper window")
    order = sorted(range(len(windows)), key=lambda index: (windows[index][0], index))
    ordered_windows = [[windows[index][0],
                        min(windows[index][1], windows[index][0] + 23.999)]
                       for index in order]
    response = shared.run("kana-whisper", source, {
        "device": "auto", "windows": ordered_windows,
    })
    if (not isinstance(response, dict)
            or not isinstance(response.get("texts"), list)
            or len(response["texts"]) != len(windows)
            or not all(isinstance(text, str) for text in response["texts"])):
        raise RuntimeError("shared KanaWhisper response is invalid")
    texts = [""] * len(windows)
    for index, value in zip(order, response["texts"], strict=True):
        texts[index] = value
    return tuple(texts)


@dataclass(frozen=True)
class ModelConfig:
    sheetsage_model: Path
    sheetsage_base: Path
    whisper_model: str = "large-v3"
    ctc_model: str = "reazon-research/japanese-wav2vec2-base-rs35kh"
    device: str = "cpu"
    local_files_only: bool = False
    separate_vocals: bool = True
    acoustic_readings: bool = True
    demucs_checkpoint: Path | None = None
    kana_model: str = KANA_MODEL
    shared_inference_url: str | None = None
    shared_inference_priority: str = "dev"
    romaji_model: Path | None = None

    def validate(self):
        if self.romaji_model is not None:
            for filename in ("model.onnx", "phoneme_vocab.json"):
                if not (Path(self.romaji_model) / filename).is_file():
                    raise ValueError(f"RomajiASR directory is missing {filename}")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        for directory in (self.sheetsage_model, self.sheetsage_base):
            for filename in ("config.json", "model.safetensors", "LICENSE"):
                if not (Path(directory) / filename).is_file():
                    raise ValueError(f"Model directory is missing {filename}: {directory}")


def read_melody_lab(path: Path) -> tuple[MelodyNote, ...]:
    notes = []
    for number, row in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not row.strip():
            continue
        fields = row.split()
        if len(fields) != 3:
            raise ValueError(f"Invalid melody LAB row {number}")
        start, end, pitch = float(fields[0]), float(fields[1]), int(fields[2])
        if not math.isfinite(start + end) or not 0 <= start < end or not 0 <= pitch <= 127:
            raise ValueError(f"Invalid melody LAB values on row {number}")
        notes.append(MelodyNote(start, end, pitch, "sheetsage2-vocal"))
    notes.sort(key=lambda n: (n.start_sec, n.end_sec, n.midi_pitch))
    normalized = []
    for note in notes:
        if normalized and normalized[-1].end_sec > note.start_sec:
            if normalized[-1].start_sec >= note.start_sec:
                raise ValueError("Conflicting simultaneous melody notes")
            normalized[-1] = replace(normalized[-1], end_sec=note.start_sec)
        normalized.append(note)
    return tuple(normalized)


@contextmanager
def prepared_adapters(path: Path, config: ModelConfig, *, accompaniment_path: Path | None = None):
    """Keep a private temporary vocal stem alive for exactly one analysis."""
    config.validate()
    shared = None
    if config.shared_inference_url:
        from .shared_inference import SharedInference
        shared = SharedInference(config.shared_inference_url, config.shared_inference_priority)
    with tempfile.TemporaryDirectory(prefix="soramimic-score-vocals-") as directory:
        vocals = None
        if config.separate_vocals:
            vocals = Path(directory) / "vocals.wav"
            if shared is not None:
                accompaniment = accompaniment_path or Path(directory) / "no_vocals.wav"
                result = shared.run("demucs", path, {"model": "htdemucs", "device": "auto"},
                                    {"vocals.wav": vocals,
                                     "no_vocals.wav": accompaniment})
                if not isinstance(result, dict) or set(result.get("artifacts", ())) != {
                        "vocals.wav", "no_vocals.wav"}:
                    raise RuntimeError("shared Demucs response is invalid")
            elif accompaniment_path is None:
                _run_adapter("vocal separation", separate_vocals, path, vocals, config)
            else:
                _run_adapter("vocal separation", separate_vocals, path, vocals, config,
                             accompaniment_path)
        yield create_adapters(config, vocals_path=vocals, shared=shared)


def create_adapters(config: ModelConfig, *, vocals_path: Path | None = None,
                    shared=None) -> AudioAdapters:
    """Low-level model adapters; use prepared_adapters to manage separation."""
    config.validate()

    def recognize(path):
        logger.info("歌詞を認識しています")
        if shared is not None:
            result = shared.run("whisper", path, {
                "model_size": config.whisper_model, "device": "auto", "language": "ja",
                "vad_filter": False, "condition_on_previous_text": False,
            })
            if not isinstance(result, dict) or not isinstance(result.get("lines"), list):
                raise RuntimeError("shared Whisper response is invalid")
            if result.get("requested_language") != "ja":
                raise RuntimeError("shared Whisper language response is invalid")
            previous_end = 0.
            lines = []
            for item in result["lines"]:
                start, end = max(previous_end, float(item["start_sec"]), 0.), float(item["end_sec"])
                if str(item["text"]).strip() and end > start:
                    lines.append(LyricLine(str(item["text"]).strip(), start, end))
                    previous_end = end
            return tuple(lines)
        from faster_whisper import WhisperModel
        model = WhisperModel(config.whisper_model, device=config.device,
                             compute_type="int8" if config.device == "cpu" else "float16",
                             local_files_only=config.local_files_only)
        try:
            segments, info = model.transcribe(str(path), language="ja", vad_filter=False,
                                               condition_on_previous_text=False)
            lines = []
            previous_end = 0.0
            for segment in segments:
                start = max(previous_end, float(segment.start), 0.0)
                end = min(float(info.duration), float(segment.end))
                if segment.text.strip() and end > start:
                    lines.append(LyricLine(segment.text.strip(), start, end))
                    previous_end = end
            return tuple(lines)
        finally:
            del model
            _release()

    def recover_window(path, start, end):
        """Retry a melody-supported credit span without the surrounding song."""
        import librosa
        if shared is not None:
            import soundfile as sf
            samples, _ = librosa.load(str(vocals_path or path), sr=16000, mono=True,
                                      offset=start, duration=end - start)
            if len(samples) == 0:
                return ()
            with tempfile.TemporaryDirectory(prefix="soramimic-score-retry-") as directory:
                excerpt = Path(directory) / "window.wav"
                sf.write(excerpt, samples, 16000)
                result = shared.run("whisper", excerpt, {
                    "model_size": config.whisper_model, "device": "auto",
                    "language": "ja", "vad_filter": False,
                    "condition_on_previous_text": False, "temperature": 0.,
                })
            if not isinstance(result, dict) or not isinstance(result.get("lines"), list):
                raise RuntimeError("shared Whisper retry response is invalid")
            if (result.get("requested_language") != "ja"
                    or result.get("requested_temperature") != 0.):
                raise RuntimeError("shared Whisper retry settings are unsupported")
            return tuple(LyricLine(str(item["text"]).strip(),
                                   max(start, start + float(item["start_sec"])),
                                   min(end, start + float(item["end_sec"])))
                         for item in result["lines"]
                         if str(item["text"]).strip()
                         and min(end, start + float(item["end_sec"]))
                         > max(start, start + float(item["start_sec"])))
        from faster_whisper import WhisperModel

        samples, _ = librosa.load(str(vocals_path or path), sr=16000, mono=True,
                                  offset=start, duration=end - start)
        if len(samples) == 0:
            return ()
        model = WhisperModel(config.whisper_model, device=config.device,
                             compute_type="int8" if config.device == "cpu" else "float16",
                             local_files_only=config.local_files_only)
        try:
            segments, _ = model.transcribe(samples, language="ja", vad_filter=False,
                                           condition_on_previous_text=False,
                                           temperature=0.0)
            lines = []
            for segment in segments:
                onset = max(start, start + float(segment.start))
                offset = min(end, start + float(segment.end))
                if segment.text.strip() and offset > onset:
                    lines.append(LyricLine(segment.text.strip(), onset, offset))
            return tuple(lines)
        finally:
            del model
            _release()

    emission_cache = {}

    def align(path, lines, readings):
        logger.info("発音時刻を推定しています")
        import torch
        import torchaudio.functional as taf
        source = vocals_path or path
        stat = source.stat()
        source_key = (source.resolve(), stat.st_size, stat.st_mtime_ns)
        if emission_cache.get("source_key") != source_key:
            import librosa
            import numpy as np
            from transformers import AutoProcessor, Wav2Vec2ForCTC

            audio, rate = librosa.load(str(source), sr=16000, mono=True)
            if not len(audio) or not np.isfinite(audio).all():
                raise ValueError("Audio must contain finite samples")
            processor = AutoProcessor.from_pretrained(config.ctc_model,
                                                      local_files_only=config.local_files_only)
            model = Wav2Vec2ForCTC.from_pretrained(config.ctc_model,
                                                  local_files_only=config.local_files_only).eval().to(config.device)
            try:
                stride = math.prod(model.config.conv_stride)
                padded = np.pad(audio, (8000, 8000))
                logits = []
                # Keep 20-second cores with two seconds of acoustic context at joins.
                for pos in range(0, len(padded), 320000):
                    first, last = max(0, pos - 32000), min(len(padded), pos + 352000)
                    values = processor(padded[first:last], sampling_rate=rate,
                                       return_tensors="pt").input_values.to(config.device)
                    with torch.inference_mode():
                        chunk = model(values).logits[0].cpu()
                    lo = (pos - first) // stride
                    # Even when right context reaches EOF, keep only this core.
                    # Otherwise the final context is duplicated by the next core.
                    hi = min(len(chunk), lo + 320000 // stride)
                    logits.append(chunk[lo:hi])
                probs = torch.log_softmax(torch.cat(logits), dim=-1)
                vocab = processor.tokenizer.get_vocab()
                blank = model.config.pad_token_id
                # Both scripts denote the same acoustic token; combine their mass.
                aliases = {}
                for char, index in vocab.items():
                    normalized = katakana(char)
                    if len(normalized) == 1 and kana_to_moras(normalized):
                        aliases.setdefault(normalized, []).append(index)
                token_ids = {}
                for char, indices in aliases.items():
                    target = vocab.get(char, indices[0])
                    combined = torch.logsumexp(probs[:, indices], dim=1)
                    probs[:, indices] = -torch.inf
                    probs[:, target] = combined
                    token_ids[char] = target
            finally:
                del model
                _release()
            emission_cache.clear()
            emission_cache.update(source_key=source_key, probs=probs, rate=rate,
                                  stride=stride, token_ids=token_ids, blank=blank,
                                  duration=len(audio) / rate)
        probs = emission_cache["probs"]
        rate = emission_cache["rate"]
        stride = emission_cache["stride"]
        token_ids = emission_cache["token_ids"]
        blank = emission_cache["blank"]
        moras = [kana_to_moras(reading.kana) for reading in readings]
        timed = all(line.start_sec is not None for line in lines)
        groups = ([([index], line.start_sec, line.end_sec)
                   for index, line in enumerate(lines)] if timed
                  else [(list(range(len(lines))), 0.0, emission_cache["duration"])])
        output = []
        for indices, start, end in groups:
            lo = max(0, math.ceil((start + .5) * rate / stride - 1e-9))
            hi = min(len(probs), math.ceil((end + .5) * rate / stride - 1e-9))
            targets, owners = [], []
            for li in indices:
                for mi, mora in enumerate(moras[li]):
                    for char in mora:
                        if char not in token_ids:
                            raise ValueError(f"CTC vocabulary does not support {char!r}")
                        targets.append(token_ids[char])
                        owners.append((li, mi))
            required = len(targets) + sum(a == b for a, b in zip(targets, targets[1:]))
            if not targets or hi - lo < required:
                raise CTCWindowCapacityError(indices[0] if len(indices) == 1 else None,
                                             hi - lo, required)
            alignment, scores = taf.forced_align(probs[lo:hi].unsqueeze(0).float(),
                                                 torch.tensor([targets]), blank=blank)
            spans = taf.merge_tokens(alignment[0], scores[0].exp(), blank=blank)
            grouped = {}
            for span, owner in zip(spans, owners, strict=True):
                grouped.setdefault(owner, []).append(span)
            for (li, mi), spans in grouped.items():
                onset = max(start, (spans[0].start + lo) * stride / rate - .5)
                offset = min(end, (spans[-1].end + lo) * stride / rate - .5)
                if offset <= onset:
                    raise ValueError("CTC produced an empty mora interval")
                confidence = min(float(span.score) for span in spans)
                output.append(AlignedMora(li, mi, moras[li][mi], onset, offset,
                                          confidence, "reazon-kana-ctc" +
                                          ("/demucs-htdemucs" if vocals_path else "")))
        return tuple(output)

    def melody(path):
        logger.info("音高を推定しています")
        if shared is not None:
            result = shared.run("sheetsage", path, {"device": "auto"})
            if not isinstance(result, dict) or not isinstance(result.get("notes"), list):
                raise RuntimeError("shared SheetSage response is invalid")
            return tuple(MelodyNote(float(item["start_sec"]), float(item["end_sec"]),
                                    int(item["midi_note"]), "sheetsage2-vocal")
                         for item in result["notes"])
        import librosa
        import numpy as np
        import torch
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            str(Path(config.sheetsage_model).resolve()),
            base_model_path=str(Path(config.sheetsage_base).resolve()),
            trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16 if config.device == "cuda" else torch.float32,
        ).eval().to(config.device)
        try:
            samples, rate = librosa.load(str(path), sr=None, mono=True)
            if not len(samples) or not np.isfinite(samples).all():
                raise ValueError("Audio must contain finite samples")
            with tempfile.TemporaryDirectory(prefix="soramimic-score-") as directory:
                with torch.inference_mode():
                    model.transcribe(samples, sampling_rate=rate, output_dir=directory,
                                     melody_only=True)
                return read_melody_lab(Path(directory) / "melody_vocal.lab")
        finally:
            del model
            _release()

    def select_readings(path, lines, *, automatic=False, recognition=None):
        originals = tuple(lines)
        options = [()] * len(lines)
        decisions = [[] for _ in lines]
        if recognition is not None:
            if len(recognition) != len(lines):
                raise ValueError("one recognition context is required per supplied line")

            @lru_cache(maxsize=None)
            def convert(text):
                return dictionary_readings(path, (LyricLine(text),))[0].candidates

            for index, (line, context) in enumerate(zip(lines, recognition, strict=True)):
                options[index] = reading_options(line.text, convert)
                if not options[index]:
                    continue
                try:
                    observed = {f"whisper-reading-{i}": reading
                                for i, reading in enumerate(convert(context))} if context else {}
                except ValueError:
                    observed = {}
                decisions[index] = [choose_reading(option, observed,
                                                   recognition_text=context or "")
                                    for option in options[index]]
            lines = tuple(replace(line, text=resolved_text(line.text, opts, choices))
                          for line, opts, choices in zip(lines, options, decisions, strict=True))

        def finish(readings):
            return tuple(replace(reading, detail={
                **reading.detail,
                **selection_detail(original.text, line.text, opts, choices),
            }) if opts else reading
                for original, line, reading, opts, choices in zip(
                    originals, lines, readings, options, decisions, strict=True))

        defaults = dictionary_readings(path, lines, automatic=automatic)
        candidates = tuple(reading.candidates for reading in defaults)
        potential = (tuple(token_reading_proposals(line.text, candidates[index][0])
                           for index, line in enumerate(lines)) if automatic else
                     ((),) * len(lines))
        ambiguous = [index for index, row in enumerate(candidates)
                     if len(row) > 1 or potential[index]
                     or any(choice["status"] == "unresolved" for choice in decisions[index])]
        if not ambiguous:
            return finish(defaults)
        if all(line.start_sec is not None and line.end_sec is not None for line in lines):
            line_windows = [(line.start_sec, line.end_sec) for line in lines]
        else:
            # Locate supplied lyrics without recognizing or rewriting their text.
            # Re-align after pronunciation selection; these times are only context.
            coarse = align(path, lines, defaults)
            line_windows = [(min(m.start_sec for m in coarse if m.line_index == index),
                             max(m.end_sec for m in coarse if m.line_index == index))
                            for index in range(len(lines))]
        import librosa
        duration = librosa.get_duration(path=str(path))
        windows, assignments = grouped_acoustic_windows(
            line_windows, ambiguous if any(options) else range(len(lines)), duration)
        paths = {"mix": path}
        if vocals_path is not None:
            paths["vocals"] = vocals_path
        transcripts = kana_views(paths, windows)
        changed = False
        for index in ambiguous:
            views = {view: "".join(rows[i] for i in assignments[index])
                     for view, rows in transcripts.items()}
            for i, option in enumerate(options[index]):
                previous = decisions[index][i]
                if previous["status"] == "unresolved":
                    decisions[index][i] = choose_reading(option, views) | {"whisper": previous}
            resolved = resolved_text(originals[index].text, options[index], decisions[index])
            if resolved != lines[index].text:
                updated = list(lines)
                updated[index] = replace(lines[index], text=resolved)
                lines = tuple(updated)
                changed = True
        if changed:
            defaults = dictionary_readings(path, lines, automatic=automatic)
            candidates = tuple(reading.candidates for reading in defaults)
        result = list(defaults)
        for index in ambiguous:
            if len(candidates[index]) == 1 and not potential[index]:
                continue
            views = {view: "".join(rows[i] for i in assignments[index])
                     for view, rows in transcripts.items()}
            proposals = (token_reading_proposals(
                lines[index].text, candidates[index][0], tuple(views.values()))
                if automatic and potential[index] else ())
            choices = tuple(dict.fromkeys((*candidates[index], *proposals)))
            selection = select_acoustic_reading(choices, views)
            result[index] = replace(selection, detail={
                **defaults[index].detail, **selection.detail,
                "dictionary_proposals": list(proposals),
                "windows_sec": [list(windows[i]) for i in assignments[index]],
                "model": config.kana_model,
                "vocal_separator": "demucs-htdemucs" if vocals_path else None,
            })
        return finish(result)

    def lyric_reading(text):
        return dictionary_readings(None, (LyricLine(text),))[0].kana

    def repeat_evidence(path, windows):
        source = vocals_path or path
        return kana_views({"vocals": source}, windows)["vocals"]

    def repeat_evidence_mix(path, windows):
        return kana_views({"mix": path}, windows)["mix"]

    def kana_views(paths, windows):
        if shared is None:
            return transcribe_kana_views(paths, windows, config)
        def transcribe(source):
            return _transcribe_shared_kana(shared, source, windows)
        with ThreadPoolExecutor(max_workers=len(paths)) as executor:
            pending = {view: executor.submit(transcribe, source)
                       for view, source in paths.items()}
            return {view: task.result() for view, task in pending.items()}

    def vocal_activity(path, windows):
        from .vocal_activity import measure_vocal_activity
        return measure_vocal_activity(vocals_path or path, windows)

    def vocalization_reattacks(mora, start, end):
        from .ctc_reattacks import decode_repeated_mora_reattacks
        if not emission_cache:
            raise RuntimeError("CTC emissions must be computed before reattack detection")
        return decode_repeated_mora_reattacks(
            emission_cache["probs"], emission_cache["token_ids"], mora, start, end,
            stride=emission_cache["stride"], rate=emission_cache["rate"],
        )

    def phonetic_recognizer(path, windows):
        from .romaji import transcribe_romaji
        return transcribe_romaji(vocals_path or path, windows, config.romaji_model)

    def phonetic_repetition_recognizer(path, windows):
        from .romaji import transcribe_romaji
        return transcribe_romaji(vocals_path or path, windows, config.romaji_model,
                                 fixed_grid=True)

    def acoustic_repetition_recognizer(path, events, notes):
        from .acoustic_repeats import find_acoustic_repetitions
        return find_acoustic_repetitions(vocals_path or path, events, notes)

    def melody_recoverer(path, start, end):
        import librosa
        import soundfile as sf
        duration = librosa.get_duration(path=str(path))
        first, last = max(0., start - 3.), min(duration, end + 3.)
        samples, rate = librosa.load(str(path), sr=None, mono=True,
                                    offset=first, duration=last - first)
        if not len(samples):
            return ()
        with tempfile.TemporaryDirectory(prefix="soramimic-score-repeat-melody-") as directory:
            excerpt = Path(directory) / "window.wav"
            sf.write(excerpt, samples, rate)
            return tuple(replace(n, start_sec=n.start_sec + first, end_sec=n.end_sec + first,
                                 source=n.source + "/local-repeat") for n in melody(excerpt))

    if config.acoustic_readings:
        automatic_selector = lambda path, lines: select_readings(path, lines, automatic=True)
    else:
        automatic_selector = lambda path, lines: dictionary_readings(path, lines, automatic=True)

    def audio_duration(path):
        import librosa
        return float(librosa.get_duration(path=str(path)))

    def supplied_readings(path, lines, recognition):
        return select_readings(path, lines, recognition=recognition)

    return AudioAdapters(select_readings if config.acoustic_readings else dictionary_readings,
                         align, melody, recognize, lyric_reading, recover_window,
                         repeat_evidence, vocal_activity if vocals_path is not None else None,
                         repeat_evidence_mix if vocals_path is not None else None,
                         vocalization_reattacks, automatic_selector,
                         phonetic_recognizer if config.romaji_model is not None else None,
                         phonetic_repetition_recognizer if config.romaji_model is not None else None,
                         acoustic_repetition_recognizer
                         if config.romaji_model is not None and vocals_path is not None else None,
                         melody_recoverer
                         if config.romaji_model is not None and vocals_path is not None else None,
                         audio_duration, dictionary_readings,
                         supplied_readings if config.acoustic_readings else None)
