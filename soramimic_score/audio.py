"""Model-neutral orchestration from singing audio to a score document.

The heavy models stay behind small callables so applications can decide where
and how they run (locally, in a worker, or through a private inference service).
This module owns the stable data passed between those models and the conversion
into Soramimic Score's canonical JSON document.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
import math
import re
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import ModelConfig

from .alignment import ObservedSingingUnit, build_known_lyrics_document
from .document import ScoreDocument, compile_score
from .ir import Boundary, Evidence, IntermediateRepresentation, NoteCandidate
from .japanese import LyricSpan, ReadingCandidate, kana_to_moras, mora_vowel
from .line_windows import snap_line_windows_to_rests
from .local_recovery import (adjacent_repeat_groups,
                             coalesce_repeated_suffix_fragments, deficit_windows,
                             duration_repeated_vocalization_candidate,
                             expand_repeated_vocalization_from_kana,
                             has_tandem_repeat_note_support,
                             is_pathological_repeated_vocalization,
                             normalize_repeated_vocalization,
                             repeated_vocalization_period, unowned_note_windows)
from .note_runs import NoteRunConfig
from .semantic import (MIN_CTC_MEDIAN_SCORE, contextual_non_lyric_template_families,
                       credit_recovery_windows, has_melodic_support,
                       is_credit_hallucination, non_lyric_template_family)
from .vocal_activity import VocalActivity


@dataclass(frozen=True)
class LyricLine:
    """One recognized lyric line on the original audio clock."""

    text: str
    start_sec: float | None = None
    end_sec: float | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class ReadingSelection:
    """The pronunciation selected for one lyric line."""

    kana: str
    source: str
    confidence: float
    candidates: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlignedMora:
    """One mora located by a text-conditioned acoustic aligner."""

    line_index: int
    mora_index: int
    kana: str
    start_sec: float
    end_sec: float
    confidence: float
    source: str = "mora-alignment"
    vowel_start_sec: float | None = None


@dataclass(frozen=True)
class MelodyNote:
    """One monophonic melody observation on the original audio clock."""

    start_sec: float
    end_sec: float
    midi_pitch: int
    source: str = "melody-transcription"
    confidence: float | None = None


LyricRecognizer = Callable[[Path], Sequence[LyricLine]]
ReadingSelector = Callable[[Path, Sequence[LyricLine]], Sequence[ReadingSelection]]
MoraAligner = Callable[
    [Path, Sequence[LyricLine], Sequence[ReadingSelection]],
    Sequence[AlignedMora],
]
MelodyTranscriber = Callable[[Path], Sequence[MelodyNote]]
LyricRecoverer = Callable[[Path, float, float], Sequence[LyricLine]]
RepetitionEvidence = Callable[[Path, Sequence[tuple[float, float]]], Sequence[str]]
VocalActivityEvidence = Callable[[Path, Sequence[tuple[float, float]]], Sequence[VocalActivity]]


@dataclass(frozen=True)
class AudioAdapters:
    """Model boundaries needed by the audio pipeline.

    ``lyric_recognizer`` is required by the ASR-first audio pipeline.
    ``lyric_reading`` optionally supplies linguistic readings for whole-line
    comparison (otherwise normalized surface text is used). ``lyric_recoverer``
    can retry one singing window when Whisper emits a credit template. The
    reading, mora, and melody adapters are required in both modes.
    """

    reading_selector: ReadingSelector
    mora_aligner: MoraAligner
    melody_transcriber: MelodyTranscriber
    lyric_recognizer: LyricRecognizer | None = None
    lyric_reading: Callable[[str], str] | None = None
    lyric_recoverer: LyricRecoverer | None = None
    repetition_evidence: RepetitionEvidence | None = None
    vocal_activity: VocalActivityEvidence | None = None
    repetition_evidence_mix: RepetitionEvidence | None = None
    vocalization_reattacks: Callable[[str, float, float], Sequence[object]] | None = None
    automatic_reading_selector: ReadingSelector | None = None
    phonetic_recognizer: Callable[[Path, Sequence[tuple[float, float]]], Sequence[object]] | None = None
    phonetic_repetition_recognizer: Callable[[Path, Sequence[tuple[float, float]]], Sequence[object]] | None = None
    acoustic_repetition_recognizer: Callable[[Path, Sequence[object], Sequence[MelodyNote]], Sequence[object]] | None = None
    melody_recoverer: Callable[[Path, float, float], Sequence[MelodyNote]] | None = None
    audio_duration: Callable[[Path], float] | None = None
    dictionary_reading_selector: ReadingSelector | None = None


class AudioPipelineError(RuntimeError):
    """An audio adapter failed or returned an inconsistent result."""

    def __init__(self, stage: str, detail: str):
        self.stage = stage
        super().__init__(f"{stage}: {detail}")


class CTCWindowCapacityError(AudioPipelineError):
    """One timed lyric line has more CTC targets than its acoustic window."""

    def __init__(self, line_index: int | None, available: int, required: int):
        self.line_index = line_index
        self.available = available
        self.required = required
        super().__init__("mora alignment",
                         f"CTC window has {available} frames for {required} targets")


def _run_adapter(stage: str, adapter: Callable[..., Sequence[object]], *args: object):
    try:
        return adapter(*args)
    except AudioPipelineError:
        raise
    except Exception as exc:
        raise AudioPipelineError(stage, str(exc) or type(exc).__name__) from exc


def _confidence(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise AudioPipelineError(label, "confidence must be in [0, 1]")
    return number


def _validate_lines(lines: Sequence[LyricLine], *, timed: bool) -> tuple[LyricLine, ...]:
    result = tuple(lines)
    if not result:
        raise AudioPipelineError("lyrics", "no lyric lines were produced")
    previous_end = 0.0
    normalized = []
    for line in result:
        if not isinstance(line.text, str) or not line.text.strip():
            raise AudioPipelineError("lyrics", "lyric lines must contain text")
        if line.confidence is not None:
            _confidence(line.confidence, "lyrics")
        if timed:
            if line.start_sec is None or line.end_sec is None:
                raise AudioPipelineError("lyrics", "recognized lines need start and end times")
            if (not math.isfinite(line.start_sec + line.end_sec) or line.start_sec < 0
                    or line.start_sec < previous_end - 1e-6
                    or line.end_sec <= max(previous_end, line.start_sec)):
                raise AudioPipelineError(
                    "lyrics", "recognized line times must be finite and non-overlapping",
                )
            if line.start_sec < previous_end:
                line = replace(line, start_sec=previous_end)
            previous_end = line.end_sec
        normalized.append(line)
    return tuple(normalized)


def _validate_readings(
    lines: Sequence[LyricLine], readings: Sequence[ReadingSelection],
) -> tuple[ReadingSelection, ...]:
    result = tuple(readings)
    if len(result) != len(lines):
        raise AudioPipelineError("readings", "one reading is required for every lyric line")
    for item in result:
        if not item.source or not kana_to_moras(item.kana):
            raise AudioPipelineError("readings", "every reading needs kana and a source")
        _confidence(item.confidence, "readings")
        if item.candidates and (item.kana not in item.candidates or any(
            not kana or "".join(kana_to_moras(kana)) != kana for kana in item.candidates
        )):
            raise AudioPipelineError("readings", "candidates must contain the selected kana reading")
    return result


def _validate_moras(
    readings: Sequence[ReadingSelection], aligned: Sequence[AlignedMora],
    *, unobserved_line_indices: Sequence[int] = (),
) -> tuple[AlignedMora, ...]:
    result = tuple(aligned)
    unobserved = set(unobserved_line_indices)
    if any(type(index) is not int or not 0 <= index < len(readings) for index in unobserved):
        raise AudioPipelineError("mora alignment", "invalid unobserved lyric line")
    if tuple((item.line_index, item.mora_index) for item in result) != tuple(sorted(
        (item.line_index, item.mora_index) for item in result
    )):
        raise AudioPipelineError("mora alignment", "moras must be in lyric order")
    for line_index, reading in enumerate(readings):
        items = tuple(item for item in result if item.line_index == line_index)
        if line_index in unobserved:
            if items:
                raise AudioPipelineError("mora alignment", "unobserved line has acoustic moras")
            continue
        if tuple(item.mora_index for item in items) != tuple(range(len(items))):
            raise AudioPipelineError("mora alignment", "mora indices must be contiguous")
        if "".join(item.kana for item in items) != "".join(kana_to_moras(reading.kana)):
            raise AudioPipelineError(
                "mora alignment", "aligned moras do not match the selected reading",
            )
    if any(item.line_index < 0 or item.line_index >= len(readings) for item in result):
        raise AudioPipelineError("mora alignment", "a mora refers to an unknown lyric line")
    previous_end = 0.0
    for item in result:
        values = [item.start_sec, item.end_sec, item.confidence]
        if item.vowel_start_sec is not None:
            values.append(item.vowel_start_sec)
        if (not all(math.isfinite(value) for value in values)
                or item.start_sec < previous_end or item.end_sec <= item.start_sec
                or (item.vowel_start_sec is not None
                    and not item.start_sec <= item.vowel_start_sec <= item.end_sec)
                or not item.source):
            raise AudioPipelineError(
                "mora alignment", "mora times must be finite, ordered, and positive",
            )
        _confidence(item.confidence, "mora alignment")
        previous_end = item.end_sec
    return result


def _validate_notes(notes: Sequence[MelodyNote], *, allow_empty: bool = False) -> tuple[MelodyNote, ...]:
    result = tuple(notes)
    if not result and not allow_empty:
        raise AudioPipelineError("melody", "no melody notes were produced")
    previous_end = 0.0
    for item in result:
        if (not math.isfinite(item.start_sec + item.end_sec)
                or item.start_sec < previous_end or item.end_sec <= item.start_sec
                or type(item.midi_pitch) is not int or not 0 <= item.midi_pitch <= 127
                or not item.source):
            raise AudioPipelineError(
                "melody", "notes must be finite, monophonic, ordered, and valid MIDI",
            )
        if item.confidence is not None:
            _confidence(item.confidence, "melody")
        previous_end = item.end_sec
    return result


def build_audio_observations(
    lines: Sequence[LyricLine],
    readings: Sequence[ReadingSelection],
    aligned_moras: Sequence[AlignedMora],
    melody_notes: Sequence[MelodyNote],
    *, allow_empty_melody: bool = False, unobserved_line_indices: Sequence[int] = (),
    allow_empty_lyrics: bool = False,
) -> IntermediateRepresentation:
    """Normalize adapter results into the versioned observation document."""
    lines = _validate_lines(lines, timed=False) if lines or not allow_empty_lyrics else ()
    readings = _validate_readings(lines, readings)
    aligned_moras = _validate_moras(readings, aligned_moras,
                                   unobserved_line_indices=unobserved_line_indices)
    melody_notes = _validate_notes(melody_notes, allow_empty=allow_empty_melody)

    canonical_text = "\n".join(line.text for line in lines)
    spans: list[LyricSpan] = []
    evidence: list[Evidence] = []
    offset = 0
    for index, (line, reading) in enumerate(zip(lines, readings, strict=True)):
        evidence_ids = ()
        if reading.candidates or reading.detail:
            evidence_id = f"audio-reading-{index}"
            evidence_ids = (evidence_id,)
            evidence.append(Evidence(
                evidence_id, reading.source, "reading-selection", reading.confidence,
                {**reading.detail, "candidates": list(reading.candidates or (reading.kana,)),
                 "selected": reading.kana},
            ))
        spans.append(LyricSpan(
            line.text,
            (offset, offset + len(line.text)),
            (ReadingCandidate(
                reading.kana, reading.source, reading.confidence, evidence_ids,
            ),),
        ))
        offset += len(line.text) + 1

    observations: list[ObservedSingingUnit] = []
    for item in aligned_moras:
        evidence_id = f"audio-mora-{item.line_index}-{item.mora_index}"
        center = (item.start_sec + item.end_sec) / 2
        independent_phase = item.source == "acoustic-repetition-phase"
        timing_detail = ({
            "timing_method": "acoustic-repetition-phase", "confidence_available": False,
            "pitched_note_support": any(n.start_sec - .12 <= center <= n.end_sec + .12
                                        for n in melody_notes),
        } if independent_phase else {})
        evidence.append(Evidence(
            evidence_id,
            item.source,
            "mora-ctc-anchor",
            item.confidence,
            {
                "time_sec": center,
                "start_sec": item.start_sec,
                "end_sec": item.end_sec,
                "conditioned_on_text": not independent_phase,
                **timing_detail,
            },
        ))
        observations.append(ObservedSingingUnit(
            (item.kana,),
            Boundary(item.start_sec, item.confidence, (evidence_id,)),
            (Boundary(item.vowel_start_sec, item.confidence, (evidence_id,))
             if item.vowel_start_sec is not None else None),
            Boundary(item.end_sec, item.confidence, (evidence_id,)),
            item.confidence,
            (evidence_id,),
        ))

    document = build_known_lyrics_document(
        canonical_text, tuple(spans), tuple(observations), tuple(evidence),
        observation_span_indices=tuple(item.line_index for item in aligned_moras),
    )
    note_evidence: list[Evidence] = []
    candidates: list[NoteCandidate] = []
    for index, item in enumerate(melody_notes):
        evidence_id = f"audio-note-evidence-{index}"
        confidence = item.confidence if item.confidence is not None else 0.0
        note_evidence.append(Evidence(
            evidence_id,
            item.source,
            "model-note",
            confidence,
            {"confidence_available": item.confidence is not None},
        ))
        candidates.append(NoteCandidate(
            f"audio-note-{index}",
            item.start_sec,
            item.end_sec,
            item.midi_pitch,
            confidence,
            (item.source,),
            (evidence_id,),
        ))
    return replace(
        document,
        evidence=document.evidence + tuple(note_evidence),
        note_candidates=tuple(candidates),
    )


def analyze_audio(
    audio_path: str | Path,
    adapters: AudioAdapters | None = None,
    *,
    lyrics: Sequence[str] | None = None,
    model_config: ModelConfig | None = None,
    adjust_lyrics: bool = False,
    on_progress: Callable[[str], None] | None = None,
    accompaniment_path: Path | None = None,
) -> ScoreDocument:
    """Keep supplied lyrics, locate them acoustically, then align their reading.

    Supplied text is authoritative even when recognition disagrees. Recognition
    supplies interval hints; neighboring anchors bound unresolved input. Optional
    whole-line additions require ``adjust_lyrics=True``. With that option, a
    fully measured window with no melody, recognition or vocal evidence may be
    removed; original text and removal evidence remain in the audit.
    """
    from .japanese import strip_ruby
    from .surface import attach_lyric_surface

    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if adapters is not None and model_config is not None:
        raise ValueError("Specify adapters or model_config, not both")
    if adjust_lyrics and lyrics is None:
        raise ValueError("adjust_lyrics requires supplied lyrics")
    if adapters is None:
        from .models import prepared_adapters
        from .media import decoded_audio
        if model_config is None:
            raise ValueError("model_config with local SheetSage2 directories is required")
        if on_progress:
            on_progress("音声を読み込んでいます")
        with decoded_audio(path) as prepared_path:
            if on_progress and model_config.separate_vocals:
                on_progress("歌声を分離しています")
            preparation = (prepared_adapters(prepared_path, model_config)
                           if accompaniment_path is None else
                           prepared_adapters(prepared_path, model_config,
                                             accompaniment_path=accompaniment_path))
            with preparation as prepared:
                return analyze_audio(prepared_path, prepared, lyrics=lyrics,
                                     adjust_lyrics=adjust_lyrics, on_progress=on_progress)

    if lyrics is not None:
        if isinstance(lyrics, (str, bytes)):
            raise TypeError("lyrics must be a sequence of lines, not one string")
        _validate_lines(tuple(LyricLine(text) for text in lyrics), timed=False)
    reading_selector = (adapters.automatic_reading_selector
                        if lyrics is None and adapters.automatic_reading_selector is not None
                        else adapters.reading_selector)
    if adapters.lyric_recognizer is None and lyrics is None:
        raise AudioPipelineError("lyrics", "ASR-first analysis requires a recognizer")
    if on_progress:
        on_progress("歌詞を認識しています")
    def validate_recognized(current):
        if not current and (lyrics is not None or adapters.phonetic_recognizer is not None):
            return ()
        return _validate_lines(current, timed=True)

    raw_recognized = validate_recognized(tuple(
        _run_adapter("lyrics", adapters.lyric_recognizer, path))
        if adapters.lyric_recognizer is not None else ())
    if on_progress:
        on_progress("音符と音高を推定しています")
    allow_spoken = adapters.vocal_activity is not None
    notes = _validate_notes(_run_adapter("melody", adapters.melody_transcriber, path),
                            allow_empty=allow_spoken or lyrics is not None)
    recognized_lines = []
    semantic_evidence = []
    credit_recovered: set[LyricLine] = set()

    def readable(text: str) -> bool:
        if adapters.lyric_reading is None:
            return True
        try:
            return bool(kana_to_moras(adapters.lyric_reading(text)))
        except (ValueError, TypeError):
            return False

    if lyrics is None:
        raw_recognized, merges = coalesce_repeated_suffix_fragments(raw_recognized)
        raw_recognized = tuple(normalize_repeated_vocalization(line)
                               for line in raw_recognized)
        for left, right in merges:
            semantic_evidence.append(Evidence(
                f"audio-repeated-suffix-{left}", "soramimic_score.local_recovery",
                "lyric-boundary-merge", 0.0,
                {"source_segment_indices": [left, right]},
            ))
    supplied_surfaces = {strip_ruby(text).strip() for text in lyrics or ()}
    template_families = contextual_non_lyric_template_families(raw_recognized)
    activity = None
    if lyrics is None and adapters.vocal_activity is not None:
        activity = tuple(_run_adapter("vocal activity", adapters.vocal_activity, path,
                                      tuple((line.start_sec, line.end_sec)
                                            for line in raw_recognized)))
        if len(activity) != len(raw_recognized):
            raise AudioPipelineError("vocal activity", "one result is required per line")
    for index, (line, family) in enumerate(zip(raw_recognized, template_families, strict=True)):
        if (lyrics is None and is_pathological_repeated_vocalization(line, notes)):
            semantic_evidence.append(Evidence(
                f"audio-repetition-runaway-{index}", "soramimic_score.local_recovery",
                "lyric-repetition-rejection", 0.0,
                {"source_segment_index": index, "surface": line.text},
            ))
            continue
        if lyrics is None and family is None and not readable(line.text):
            semantic_evidence.append(Evidence(
                f"audio-unreadable-asr-{index}", "soramimic_score.readings",
                "lyric-semantic-gate", 0.0,
                {"source_segment_index": index, "surface": line.text,
                 "status": "rejected", "reason": "unpronounceable"},
            ))
            continue
        if (lyrics is None and family is None and activity is not None
                and not has_melodic_support(line, notes) and not activity[index].supported):
            semantic_evidence.append(Evidence(
                f"audio-vocal-silence-{index}", "soramimic_score.vocal_activity",
                "lyric-semantic-gate", 0.0,
                {"source_segment_index": index, "surface": line.text,
                 "status": "rejected", "vocal_activity_relative_db":
                 activity[index].relative_db},
            ))
            continue
        if (line.text.strip() in supplied_surfaces or family is None
                or (family not in {"credits", "stock-media-credit"}
                    and has_melodic_support(line, notes))):
            recognized_lines.append(line)
            continue
        windows = credit_recovery_windows(line, notes, template_family=family)
        recovered = []
        if windows and adapters.lyric_recoverer is not None:
            if on_progress:
                on_progress("歌詞の誤認識区間を再確認しています")
            for start, end in windows:
                try:
                    candidates = _validate_lines(
                        adapters.lyric_recoverer(path, start, end), timed=True,
                    )
                except Exception:
                    continue
                candidates = tuple(normalize_repeated_vocalization(item)
                                   for item in candidates)
                recovered.extend(candidate for candidate in candidates
                                 if candidate.start_sec is not None
                                 and candidate.end_sec is not None
                                 and start <= candidate.start_sec < candidate.end_sec <= end
                                 and non_lyric_template_family(candidate.text) is None
                                 and not is_pathological_repeated_vocalization(candidate, notes)
                                 and readable(candidate.text)
                                 and has_melodic_support(candidate, notes))
        recognized_lines.extend(recovered)
        credit_recovered.update(recovered)
        semantic_evidence.append(Evidence(
            f"audio-credit-gate-{index}", "soramimic_score.semantic",
            "lyric-semantic-gate", 0.0,
            {"source_segment_index": index, "surface": line.text,
             "template_family": family,
             "status": "recovered" if recovered else "rejected",
             "recovery_windows": [list(window) for window in windows],
             "recovered_count": len(recovered)},
        ))
    recognized_lines.sort(key=lambda item: (item.start_sec, item.end_sec))
    recognized = validate_recognized(recognized_lines)
    if lyrics is None and (adapters.repetition_evidence is not None
                           or adapters.repetition_evidence_mix is not None):
        def repeat_ctc(line: LyricLine) -> float:
            try:
                selected = _validate_readings((line,), _run_adapter(
                    "readings", reading_selector, path, (line,)))
                aligned = _validate_moras(selected, _run_adapter(
                    "mora alignment", adapters.mora_aligner, path, (line,), selected))
            except Exception:
                return 0.
            return statistics.median(item.confidence for item in aligned) if aligned else 0.

        providers = (("mix", adapters.repetition_evidence_mix),
                     ("vocals", adapters.repetition_evidence))
        grouped_recovered = set()
        group_replacements = {}
        for first, last, period in adjacent_repeat_groups(recognized):
            source_count = sum(len(kana_to_moras(item.text))
                               for item in recognized[first:last])
            grouped = LyricLine("".join(period[index % len(period)]
                                         for index in range(source_count)),
                                recognized[first].start_sec, recognized[last - 1].end_sec)
            local_notes = [note for note in notes
                           if grouped.start_sec <= (note.start_sec + note.end_sec) / 2
                           < grouped.end_sec]
            if local_notes:
                grouped = replace(grouped,
                                  start_sec=max(grouped.start_sec, local_notes[0].start_sec),
                                  end_sec=min(grouped.end_sec, local_notes[-1].end_sec))
            if grouped.end_sec - grouped.start_sec > 24:
                continue
            best = None
            for view, adapter in providers:
                if adapter is None:
                    continue
                try:
                    evidence_texts = adapter(path, ((grouped.start_sec, grouped.end_sec),))
                except Exception:
                    continue
                if len(evidence_texts) != 1:
                    continue
                expanded = expand_repeated_vocalization_from_kana(
                    grouped, evidence_texts[0], notes)
                if expanded is None:
                    continue
                ctc = repeat_ctc(expanded)
                if ctc >= MIN_CTC_MEDIAN_SCORE and (best is None or ctc > best[0]):
                    best = (ctc, expanded, view)
            if best is not None:
                group_replacements[first] = (last, best)
        if group_replacements:
            updated = []
            index = 0
            while index < len(recognized):
                if index in group_replacements:
                    last, (ctc, expanded, view) = group_replacements[index]
                    updated.append(expanded)
                    grouped_recovered.add(expanded)
                    semantic_evidence.append(Evidence(
                        f"audio-adjacent-repeat-{index}", "soramimic_score.local_recovery",
                        "lyric-repetition-expansion", ctc,
                        {"source_segment_indices": list(range(index, last)),
                         "source": view, "expanded_moras": len(kana_to_moras(expanded.text))},
                    ))
                    index = last
                else:
                    updated.append(recognized[index])
                    index += 1
            recognized = _validate_lines(updated, timed=True)
        candidate_indices = [index for index, line in enumerate(recognized)
                             if line not in grouped_recovered
                             and repeated_vocalization_period(line.text) is not None
                             and line.start_sec is not None and line.end_sec is not None
                             and line.end_sec - line.start_sec <= 24][:4]
        if candidate_indices:
            if on_progress:
                on_progress("繰り返し歌詞を確認しています")
            windows = tuple((recognized[index].start_sec, recognized[index].end_sec)
                            for index in candidate_indices)
            chosen = {}
            for view, adapter in providers:
                if adapter is None:
                    continue
                try:
                    evidence_texts = adapter(path, windows)
                except Exception:
                    continue
                if len(evidence_texts) != len(candidate_indices):
                    continue
                for index, evidence_text in zip(candidate_indices, evidence_texts, strict=True):
                    expanded = expand_repeated_vocalization_from_kana(
                        recognized[index], evidence_text, notes)
                    if expanded is None:
                        continue
                    ctc = repeat_ctc(expanded)
                    if ctc >= MIN_CTC_MEDIAN_SCORE and (index not in chosen
                                                        or ctc > chosen[index][0]):
                        chosen[index] = (ctc, expanded, view)
            if chosen:
                updated = list(recognized)
                for index, (ctc, expanded, view) in chosen.items():
                    updated[index] = expanded
                    semantic_evidence.append(Evidence(
                        f"audio-kana-repeat-{index}", "soramimic_score.local_recovery",
                        "lyric-repetition-expansion", ctc,
                        {"source_segment_index": index, "source": view,
                         "source_moras": len(kana_to_moras(recognized[index].text)),
                         "expanded_moras": len(kana_to_moras(expanded.text))},
                    ))
                recognized = tuple(updated)
    if lyrics is None and adapters.lyric_recoverer is not None:
        def count_moras(text: str) -> int:
            try:
                reading = adapters.lyric_reading(text) if adapters.lyric_reading else text
            except ValueError:
                reading = text
            return len(kana_to_moras(reading))

        def retry(start: float, end: float) -> tuple[LyricLine, ...]:
            try:
                candidates = _validate_lines(
                    adapters.lyric_recoverer(path, start, end), timed=True)
            except Exception:
                return ()
            candidates = tuple(normalize_repeated_vocalization(item)
                               for item in candidates)
            return tuple(candidate for candidate in candidates
                         if candidate.start_sec is not None and candidate.end_sec is not None
                         and start <= candidate.start_sec < candidate.end_sec <= end
                         and non_lyric_template_family(candidate.text) is None
                         and not is_pathological_repeated_vocalization(candidate, notes)
                         and readable(candidate.text)
                         and has_melodic_support(candidate, notes))

        if on_progress:
            on_progress("歌詞の欠落区間を再確認しています")
        replacements: dict[int, tuple[LyricLine, ...]] = {}
        counts = tuple(count_moras(line.text) for line in recognized)
        ratios = []
        for line, count in zip(recognized, counts, strict=True):
            effective = count + len(re.findall(r"[A-Za-z]+", line.text))
            found = sum(line.start_sec <= (note.start_sec + note.end_sec) / 2 < line.end_sec
                        for note in notes)
            if effective >= 4 and found >= 2:
                ratios.append(found / effective)
        median_ratio = statistics.median(ratios) if ratios else 1.
        windows_by_line = {}
        for index, start, end in deficit_windows(recognized, notes, counts):
            windows_by_line.setdefault(index, []).append((start, end))
        for index, windows in windows_by_line.items():
            recovered = sorted((item for start, end in windows
                                for item in retry(start, end)),
                               key=lambda item: (item.start_sec, item.end_sec))
            try:
                candidates = _validate_lines(recovered, timed=True)
            except AudioPipelineError:
                continue
            source = recognized[index]
            period = repeated_vocalization_period(source.text)
            if period is not None and len(period) > 1:
                peers = [other.end_sec - other.start_sec
                         for peer_index, other in enumerate(recognized)
                         if peer_index != index and other.text == source.text]
                if peers:
                    repetition_count = math.floor(
                        (source.end_sec - source.start_sec)
                        / statistics.median(peers) + .5)
                    repeated = duration_repeated_vocalization_candidate(
                        source, candidates, repetition_count)
                    if repeated is not None:
                        candidates = (repeated,)
            effective_source = counts[index] + len(re.findall(r"[A-Za-z]+", source.text))
            required_moras = effective_source + max(2, math.ceil(effective_source * .25))
            if candidates and sum(count_moras(item.text) for item in candidates) >= required_moras:
                # More morae alone can be a Whisper hallucination. Compare the
                # local retry with the original line on the same vocal audio.
                try:
                    source_readings = _validate_readings((source,), _run_adapter(
                        "readings", reading_selector, path, (source,)))
                    source_moras = _validate_moras(source_readings, _run_adapter(
                        "mora alignment", adapters.mora_aligner,
                        path, (source,), source_readings))
                    candidate_readings = _validate_readings(candidates, _run_adapter(
                        "readings", reading_selector, path, candidates))
                    candidate_moras = _validate_moras(candidate_readings, _run_adapter(
                        "mora alignment", adapters.mora_aligner,
                        path, candidates, candidate_readings))
                except Exception:
                    continue
                source_ctc = statistics.median(item.confidence for item in source_moras)
                candidate_ctc = statistics.median(item.confidence for item in candidate_moras)
                recovered_count = sum(len(kana_to_moras(item.kana))
                                      for item in candidate_readings)
                note_count = sum(source.start_sec <= (note.start_sec + note.end_sec) / 2
                                 < source.end_sec for note in notes)
                tandem_support = has_tandem_repeat_note_support(
                    "".join(item.text for item in candidates),
                    source_moras=effective_source, recovered_moras=recovered_count,
                    note_count=note_count, median_notes_per_mora=median_ratio)
                weak_tandem_supported = (tandem_support and source_ctc > 0
                                         and candidate_ctc >= source_ctc)
                if (recovered_count < required_moras
                        or recovered_count > note_count * 3
                        or (candidate_ctc < MIN_CTC_MEDIAN_SCORE
                            and not weak_tandem_supported)
                        or candidate_ctc < source_ctc * .5):
                    continue
                replacements[index] = candidates
                semantic_evidence.append(Evidence(
                    f"audio-deficit-recovery-{index}", "soramimic_score.local_recovery",
                    "lyric-local-retry", candidate_ctc,
                    {"source_segment_index": index,
                     "windows": [list(window) for window in windows],
                     "original_moras": counts[index], "recovered_moras":
                     sum(count_moras(item.text) for item in candidates),
                     "source_ctc": source_ctc, "recovered_ctc": candidate_ctc,
                     "tandem_repeat_support": tandem_support},
                ))
        retained = tuple(item for index, line in enumerate(recognized)
                         for item in replacements.get(index, (line,)))
        recognized = validate_recognized(retained)
    overlay = None
    unobserved_lines = ()
    lines = recognized
    if lyrics is not None:
        from .supplied_lyrics import prepare_supplied_audio
        lines, overlay, unobserved_lines = prepare_supplied_audio(
            path, lyrics, recognized, notes, adapters, adjust_lyrics=adjust_lyrics,
            recognition_evidence=raw_recognized if adapters.lyric_recognizer is not None else None,
        )

    # Only the final text reaches the selector. Its closed candidates contain
    # no pronunciation copied from a different recognized surface.
    if on_progress:
        on_progress("歌詞の読みを確認しています")
    active_indices = [i for i in range(len(lines)) if i not in unobserved_lines]
    active_lines = tuple(lines[i] for i in active_indices)
    active_readings = _validate_readings(active_lines, _run_adapter(
        "readings", reading_selector, path, active_lines,
    )) if active_lines else ()
    selected = dict(zip(active_indices, active_readings, strict=True))
    if unobserved_lines:
        silent_lines = tuple(lines[i] for i in unobserved_lines)
        defaults = _validate_readings(silent_lines, _run_adapter(
            "readings", adapters.dictionary_reading_selector or reading_selector,
            path, silent_lines,
        ))
        selected.update(zip(unobserved_lines, defaults, strict=True))
    readings = tuple(selected[i] for i in range(len(lines)))
    if overlay is not None:
        for group, reading in zip(overlay["groups"], readings, strict=True):
            group["acoustic_reading"] = reading.kana
            group["reading_candidates"] = list(reading.candidates)
        overlay.pop("acoustic_changes", None)
        overlay["readings_fixed_before_alignment"] = True
    lines = tuple(replace(line, text=strip_ruby(line.text)) for line in lines)
    # A failure is reported. Do not fall back to the rejected ASR pronunciation.
    if on_progress:
        on_progress("モーラの時刻を推定しています")

    def align_retained(current_lines, current_readings):
        if not current_lines:
            return (), (), ()
        while True:
            try:
                aligned = _validate_moras(current_readings, _run_adapter(
                    "mora alignment", adapters.mora_aligner,
                    path, current_lines, current_readings))
                return current_lines, current_readings, aligned
            except CTCWindowCapacityError as exc:
                if lyrics is not None or exc.line_index is None:
                    raise
                index = exc.line_index
                if not 0 <= index < len(current_lines):
                    raise
                semantic_evidence.append(Evidence(
                    f"audio-ctc-capacity-{len(semantic_evidence)}",
                    "soramimic_score.models", "lyric-semantic-gate", 0.,
                    {"surface": current_lines[index].text, "status": "rejected",
                     "reason": "ctc-window-capacity-insufficient",
                     "available_frames": exc.available, "required_frames": exc.required},
                ))
                current_lines = current_lines[:index] + current_lines[index + 1:]
                current_readings = current_readings[:index] + current_readings[index + 1:]
                if not current_lines:
                    if adapters.phonetic_recognizer is not None:
                        return (), (), ()
                    raise AudioPipelineError("lyrics", "no acoustically alignable lyric lines")

    if unobserved_lines:
        _, _, active_moras = align_retained(
            tuple(lines[i] for i in active_indices),
            tuple(readings[i] for i in active_indices),
        )
        moras = tuple(replace(mora, line_index=active_indices[mora.line_index])
                      for mora in active_moras)
        moras = _validate_moras(readings, moras, unobserved_line_indices=unobserved_lines)
    else:
        lines, readings, moras = align_retained(lines, readings)
    if overlay is not None:
        for i, group in enumerate(overlay["groups"]):
            if i not in unobserved_lines:
                group["alignment_status"] = "aligned"
    if lyrics is None:
        rejected = []
        for index, line in enumerate(lines):
            family = non_lyric_template_family(line.text)
            if (family is None and line not in credit_recovered
                    or family in {"credits", "stock-media-credit"}):
                continue
            scores = [mora.confidence for mora in moras if mora.line_index == index]
            median_score = statistics.median(scores) if scores else 0.0
            if median_score >= MIN_CTC_MEDIAN_SCORE:
                continue
            rejected.append(index)
            semantic_evidence.append(Evidence(
                f"audio-ctc-template-{index}", "soramimic_score.semantic",
                "lyric-semantic-gate", 0.0,
                {"source_segment_index": index, "surface": line.text,
                 "template_family": family, "credit_recovery": line in credit_recovered,
                 "ctc_median_score": median_score,
                 "status": "rejected"},
            ))
        if rejected:
            if on_progress:
                on_progress("字幕候補の発音を再確認しています")
            retained = [line for index, line in enumerate(lines) if index not in rejected]
            for index in rejected:
                source = lines[index]
                if adapters.lyric_recoverer is None:
                    continue
                for start, end in credit_recovery_windows(source, notes):
                    try:
                        candidates = _validate_lines(
                            adapters.lyric_recoverer(path, start, end), timed=True)
                    except Exception:
                        continue
                    retained.extend(candidate for candidate in candidates
                                    if candidate.start_sec is not None
                                    and candidate.end_sec is not None
                                    and start <= candidate.start_sec < candidate.end_sec <= end
                                    and non_lyric_template_family(candidate.text) is None
                                    and not is_pathological_repeated_vocalization(candidate, notes)
                                    and readable(candidate.text)
                                    and has_melodic_support(candidate, notes))
            lines = validate_recognized(sorted(retained, key=lambda item: item.start_sec))
            readings = _validate_readings(
                lines, _run_adapter("readings", reading_selector, path, lines)) if lines else ()
            lines, readings, moras = align_retained(lines, readings)
    if lyrics is None and adapters.lyric_recoverer is not None and lines:
        # Stage 3 owns note assignment. Probe once before retrying truly unowned
        # note runs; a raw gap between Whisper lines is not sufficient evidence.
        provisional = compile_score(
            build_audio_observations(lines, readings, moras, notes,
                                     allow_empty_melody=allow_spoken),
            config=NoteRunConfig(whisper_boundary_cost_per_sec2=.1),
            line_windows_by_utterance={
                f"u{index}": (line.start_sec, line.end_sec)
                for index, line in enumerate(lines)
            } if lines else None,
        )
        windows = unowned_note_windows(provisional.observations, lines)
        if windows and adapters.vocal_activity is not None:
            if on_progress:
                on_progress("歌詞に未対応の音符を再確認しています")
            activity = tuple(_run_adapter(
                "vocal activity", adapters.vocal_activity, path,
                tuple((start, end) for start, end, _ in windows)))
            if len(activity) != len(windows):
                raise AudioPipelineError("vocal activity", "one result is required per window")
            additions = []
            for index, ((start, end, note_count), evidence) in enumerate(
                    zip(windows, activity, strict=True)):
                if not evidence.supported:
                    continue
                best = None
                fallback = []
                for source, retry_start, retry_end in (
                        ("exact", start, end),
                        ("padded", max(0., start - .5), end + .5)):
                    try:
                        raw = _validate_lines(adapters.lyric_recoverer(
                            path, retry_start, retry_end), timed=True)
                    except Exception:
                        continue
                    raw = tuple(normalize_repeated_vocalization(item) for item in raw)
                    candidates = tuple(replace(candidate,
                                               start_sec=max(start, candidate.start_sec),
                                               end_sec=min(end, candidate.end_sec))
                                       for candidate in raw
                                       if candidate.start_sec < end and candidate.end_sec > start)
                    candidates = tuple(candidate for candidate in candidates
                                       if candidate.end_sec > candidate.start_sec
                                       and non_lyric_template_family(candidate.text) is None
                                       and not is_pathological_repeated_vocalization(candidate, notes)
                                       and readable(candidate.text)
                                       and has_melodic_support(candidate, notes)
                                       and not any(candidate.start_sec < line.end_sec
                                                   and candidate.end_sec > line.start_sec
                                                   for line in (*lines, *additions)))
                    if not candidates:
                        continue
                    try:
                        selected = _validate_readings(candidates, _run_adapter(
                            "readings", reading_selector, path, candidates))
                        aligned = _validate_moras(selected, _run_adapter(
                            "mora alignment", adapters.mora_aligner,
                            path, candidates, selected))
                    except Exception:
                        continue
                    mora_count = sum(len(kana_to_moras(item.kana)) for item in selected)
                    ctc = statistics.median(item.confidence for item in aligned) if aligned else 0.
                    reasons = []
                    if (mora_count < max(4, math.ceil(note_count * .25))
                            or mora_count > note_count * 2):
                        reasons.append("detail")
                    if ctc < MIN_CTC_MEDIAN_SCORE:
                        reasons.append("ctc")
                    fallback.append((source, candidates, selected, tuple(reasons)))
                    if reasons:
                        continue
                    if best is None or ctc > best[0]:
                        best = (ctc, candidates, source)
                if best is None and fallback and (adapters.repetition_evidence is not None
                                                   or adapters.repetition_evidence_mix is not None):
                    exact = next((item for item in fallback if item[0] == "exact"), None)
                    repeated = (tuple(item for item in exact[1]
                                      if repeated_vocalization_period(item.text) is not None)
                                if exact else ())
                    if repeated:
                        windows_for_kana = tuple((item.start_sec, item.end_sec)
                                                 for item in repeated)
                        for view, kana_adapter in (("mix", adapters.repetition_evidence_mix),
                                                   ("vocals", adapters.repetition_evidence)):
                            if kana_adapter is None:
                                continue
                            try:
                                evidence_texts = kana_adapter(path, windows_for_kana)
                            except Exception:
                                continue
                            if len(evidence_texts) != len(repeated):
                                continue
                            for source_line, evidence_text in zip(
                                    repeated, evidence_texts, strict=True):
                                expanded = expand_repeated_vocalization_from_kana(
                                    source_line, evidence_text, notes)
                                if expanded is None:
                                    continue
                                try:
                                    selected = _validate_readings((expanded,), _run_adapter(
                                        "readings", reading_selector,
                                        path, (expanded,)))
                                    aligned = _validate_moras(selected, _run_adapter(
                                        "mora alignment", adapters.mora_aligner,
                                        path, (expanded,), selected))
                                except Exception:
                                    continue
                                ctc = statistics.median(item.confidence for item in aligned)
                                if ctc >= MIN_CTC_MEDIAN_SCORE and (best is None or ctc > best[0]):
                                    best = (ctc, (expanded,), f"kana-repeat-{view}")
                if best is None and len(fallback) == 2:
                    exact, padded = fallback
                    if exact[3] == padded[3] == ("ctc",):
                        try:
                            exact_vowels = tuple(mora_vowel(mora) for reading in exact[2]
                                                 for mora in kana_to_moras(reading.kana))
                            padded_vowels = tuple(mora_vowel(mora) for reading in padded[2]
                                                  for mora in kana_to_moras(reading.kana))
                        except ValueError:
                            exact_vowels, padded_vowels = (), ()
                        bounds_agree = (abs(exact[1][0].start_sec - padded[1][0].start_sec) <= .6
                                        and abs(exact[1][-1].end_sec - padded[1][-1].end_sec) <= .6)
                        if (bounds_agree and exact_vowels == padded_vowels
                                and len(exact_vowels) >= max(4, math.ceil(note_count * .25))
                                and all(vowel in {"a", "i", "u", "e", "o"}
                                        for vowel in exact_vowels)):
                            vowels = {"a": "ア", "i": "イ", "u": "ウ",
                                      "e": "エ", "o": "オ"}
                            vowel_line = LyricLine("".join(vowels[item]
                                                       for item in exact_vowels), start, end)
                            try:
                                selected = _validate_readings((vowel_line,), _run_adapter(
                                    "readings", reading_selector,
                                    path, (vowel_line,)))
                                aligned = _validate_moras(selected, _run_adapter(
                                    "mora alignment", adapters.mora_aligner,
                                    path, (vowel_line,), selected))
                            except Exception:
                                aligned = ()
                            ctc = statistics.median(item.confidence for item in aligned) if aligned else 0.
                            if ctc >= MIN_CTC_MEDIAN_SCORE:
                                best = (ctc, (vowel_line,), "vowel-continuation")
                if best is None:
                    continue
                additions.extend(best[1])
                semantic_evidence.append(Evidence(
                    f"audio-gap-recovery-{index}", "soramimic_score.local_recovery",
                    "lyric-local-retry", best[0],
                    {"window": [start, end], "note_count": note_count,
                     "recovered_count": len(best[1]), "source": best[2]},
                ))
            if additions:
                lines = _validate_lines(sorted((*lines, *additions),
                                               key=lambda item: item.start_sec), timed=True)
                readings = _validate_readings(lines, _run_adapter(
                    "readings", reading_selector, path, lines))
                lines, readings, moras = align_retained(lines, readings)
    if lyrics is None and adapters.phonetic_recognizer is not None:
        from .phonetic_fallback import add_phonetic_fallback, uncovered_note_windows
        from .phonetic_repeats import find_repeated_lines, repetition_windows
        windows = repetition_windows(lines, readings)
        if adapters.acoustic_repetition_recognizer is not None:
            last = max((item.end_sec for item in (*lines, *notes)), default=0.)
            windows = ((0., last),) if last > 0 else ()
        if windows:
            if on_progress:
                on_progress("独立した発音認識で反復と歌詞の欠損を確認しています")
            recognizer = adapters.phonetic_repetition_recognizer or adapters.phonetic_recognizer
            events = tuple(_run_adapter("phonetic repetition recognition", recognizer,
                                        path, windows))
            replacements = {}
            for proposal in find_repeated_lines(lines, readings, notes, events):
                index = proposal.line_index
                original = lines[index]
                found = proposal.occurrences
                # Each copy gets its own acoustic interval. Never force the
                # repeated target through the original, stretched alignment.
                boundaries = (original.start_sec,
                              *((a.end_sec + b.start_sec) / 2
                                for a, b in zip(found, found[1:])),
                              original.end_sec)
                # The phonetic recognizer may recover only some occurrences.
                # Keep unmatched audio available to gap recovery instead of
                # stretching the nearest copy across that unrecognized span.
                copies = tuple(replace(
                    original, text=proposal.text,
                    start_sec=max(a, occurrence.start_sec - .25),
                    end_sec=min(b, occurrence.end_sec + .25),
                ) for a, b, occurrence in zip(boundaries, boundaries[1:], found))
                unit_reading = readings[index]
                if proposal.original_count > 1:
                    unit_reading = replace(
                        unit_reading, kana=proposal.kana, candidates=(proposal.kana,),
                        detail={"reason": "recognized-phrase-repetition",
                                "original_reading": readings[index].kana,
                                "original_count": proposal.original_count},
                    )
                selected = tuple(unit_reading for _ in copies)
                try:
                    aligned = _validate_moras(selected, _run_adapter(
                        "mora alignment", adapters.mora_aligner, path, copies, selected))
                except Exception:
                    continue
                replacements[index] = (copies, selected, aligned)
                semantic_evidence.append(Evidence(
                    f"audio-phonetic-repeat-{index}", "soramimic_score.phonetic_repeats",
                    "lyric-phonetic-repetition", 0.,
                    {"source_line_index": index, "surface": original.text,
                     "reading": readings[index].kana,
                     "original_count": proposal.original_count,
                     "recovered_count": len(copies), "confidence_available": False,
                     "phonetic_source": sorted({event.source for event in events}),
                     "occurrences": [
                         {"start_sec": item.start_sec, "end_sec": item.end_sec,
                          "distance": item.distance, "exact_moras": item.exact_moras,
                          "observed_kana": item.observed_kana,
                          "note_support": item.note_support} for item in found]},
                ))
            if replacements:
                updated_lines, updated_readings, updated_moras = [], [], []
                for index, (line, reading) in enumerate(zip(lines, readings, strict=True)):
                    copies, selected, aligned = replacements.get(index, (
                        (line,), (reading,),
                        tuple(replace(mora, line_index=0) for mora in moras
                              if mora.line_index == index)))
                    updated_moras.extend(replace(mora, line_index=mora.line_index
                                                + len(updated_lines)) for mora in aligned)
                    updated_lines.extend(copies)
                    updated_readings.extend(selected)
                lines, readings, moras = (tuple(updated_lines), tuple(updated_readings),
                                          tuple(updated_moras))
            if adapters.acoustic_repetition_recognizer is not None:
                from .repetition_repair import (
                    copies_needing_repair, has_foreign_transcript, merge_recovered_notes,
                    missing_note_windows, phase_aligned_moras, preserve_neighbor_readings,
                    replace_pronunciation_spans)
                groups = _run_adapter("acoustic repetition recognition",
                                      adapters.acoustic_repetition_recognizer, path, events, notes)
                acoustic_replacements = []
                proposed_groups = []
                for group_index, group in enumerate(groups):
                    if not has_foreign_transcript(group, lines):
                        continue
                    copies = tuple(LyricLine(group.kana, copy.start_sec, copy.end_sec)
                                   for copy in copies_needing_repair(group, lines, moras))
                    if not copies:
                        continue
                    selection = ReadingSelection(
                        group.kana, "acoustic-phonetic-repetition", 0., (group.kana,),
                        {"lyric_kind": "phonetic-fallback", "confidence_available": False})
                    selected = (selection,) * len(copies)
                    try:
                        aligned = _validate_moras(selected, phase_aligned_moras(group, copies)
                                                  if group.mora_offsets_sec else _run_adapter(
                                                      "mora alignment", adapters.mora_aligner,
                                                      path, copies, selected))
                    except Exception:
                        continue
                    acoustic_replacements.extend(
                        (copy, selection, tuple(replace(m, line_index=0) for m in aligned
                                                if m.line_index == i)) for i, copy in enumerate(copies))
                    proposed_groups.append((group_index, group, copies))
                if acoustic_replacements:
                    def realign_neighbor(line, reading):
                        return _run_adapter("repetition neighbor alignment", adapters.mora_aligner,
                                            path, (line,), (reading,))
                    lines, moras, acoustic_replacements = preserve_neighbor_readings(
                        lines, readings, moras, acoustic_replacements, realign_neighbor)
                confirmed = tuple(line for line, _, _ in acoustic_replacements)
                for group_index, group, copies in proposed_groups:
                    applied = tuple(copy for copy in copies if copy in confirmed)
                    if not applied:
                        continue
                    semantic_evidence.append(Evidence(
                        f"audio-acoustic-repeat-{group_index}", "soramimic_score.acoustic_repeats",
                        "lyric-acoustic-repetition", 0.,
                        {"reading": group.kana, "confidence_available": False,
                         "semantic_lyrics_available": False, "period_sec": group.period_sec,
                         "template_start_sec": group.template_start_sec,
                         "seed_start_sec": group.seed_start_sec,
                         "mora_offsets_sec": list(group.mora_offsets_sec),
                         "observed_count": len(group.occurrences), "repaired_count": len(applied),
                         "applied_windows_sec": [[copy.start_sec, copy.end_sec] for copy in applied],
                         "occurrences": [{"start_sec": c.start_sec, "end_sec": c.end_sec,
                                          "spectral_similarity": c.similarity,
                                          "observed_kana": c.observed_kana} for c in group.occurrences]},
                    ))
                if acoustic_replacements:
                    lines, readings, moras = replace_pronunciation_spans(
                        lines, readings, moras, acoustic_replacements)
                    lines = _validate_lines(lines, timed=True)
                    readings = _validate_readings(lines, readings)
                    moras = _validate_moras(readings, moras)
                    if adapters.melody_recoverer is not None:
                        for index, (start, end) in enumerate(missing_note_windows(confirmed, notes)):
                            recovered = tuple(_run_adapter("repeated melody recovery",
                                                           adapters.melody_recoverer, path, start, end))
                            if not recovered:
                                continue
                            recovered = _validate_notes(recovered)
                            previous_count = len(notes)
                            notes = _validate_notes(merge_recovered_notes(notes, recovered, start, end))
                            semantic_evidence.append(Evidence(
                                f"audio-repeated-melody-{index}", "soramimic_score.models",
                                "melody-local-retry", 0.,
                                {"start_sec": start, "end_sec": end,
                                 "added_note_count": len(notes) - previous_count,
                                 "confidence_available": False},
                            ))
        windows = uncovered_note_windows(moras, notes)
        if windows:
            events = _run_adapter("phonetic recognition", adapters.phonetic_recognizer,
                                  path, windows)
            lines, readings, moras, fallback_evidence = add_phonetic_fallback(
                lines, readings, moras, notes, events)
            semantic_evidence.extend(fallback_evidence)
            lines = _validate_lines(lines, timed=True)
            readings = _validate_readings(lines, readings)
            moras = _validate_moras(readings, moras)
    if on_progress:
        on_progress("楽譜データを組み立てています")
    observations = build_audio_observations(lines, readings, moras, notes,
                                             allow_empty_melody=allow_spoken or lyrics is not None,
                                             unobserved_line_indices=unobserved_lines,
                                             allow_empty_lyrics=bool(
                                                 overlay and overlay["removed_supplied_indices"]))
    if semantic_evidence:
        observations = replace(observations,
                               evidence=observations.evidence + tuple(semantic_evidence))
    if adjust_lyrics and overlay is not None:
        observations = replace(observations, evidence=observations.evidence + (Evidence(
            "audio-lyric-adjustment", "soramimic_score.lyrics", "lyric-adjustment", 0.0,
            {"mode": "conservative-audio-adjustment", "supplied_lines": list(lyrics),
             "decisions": [{"operation": g["operation"],
                            "supplied_line_indices": g["supplied_indices"],
                            "recognized_line_indices": g["asr_indices"]}
                           for g in overlay["groups"]] + [
                               {"operation": "remove", "supplied_line_indices": g["supplied_indices"],
                                "recognized_line_indices": g["asr_indices"],
                                "start_sec": g["start_sec"], "end_sec": g["end_sec"],
                                "reason": g["reason"], "support": g["support"],
                                "vocal_activity_measurement": g["vocal_activity_measurement"]}
                               for g in overlay["removed_groups"]]},
        ),))
    line_windows = (
        {f"u{index}": (line.start_sec, line.end_sec)
         for index, line in enumerate(lines)
         if line.start_sec is not None and line.end_sec is not None} or None
    )
    if overlay is not None:
        line_windows = {f"u{i}": (g["start_sec"], g["end_sec"])
                        for i, g in enumerate(overlay["groups"])}
    if line_windows and len(line_windows) == len(lines) and not unobserved_lines:
        snapped = snap_line_windows_to_rests(
            tuple(line_windows[f"u{index}"] for index in range(len(lines))), notes,
        )
        line_windows = {f"u{index}": window
                        for index, window in enumerate(snapped)}
    reattacks = {}
    if lyrics is None and line_windows is not None and adapters.vocalization_reattacks:
        for index, reading in enumerate(readings):
            components = kana_to_moras(reading.kana)
            if (len(components) < 2 or len(set(components)) != 1
                    or components[0] in {"ン", "ッ", "ー"}):
                continue
            start, end = line_windows[f"u{index}"]
            events = tuple(adapters.vocalization_reattacks(components[0], start, end))
            if len(events) >= len(components):
                reattacks[f"u{index}"] = events
    result = compile_score(
        observations,
        config=NoteRunConfig(whisper_boundary_cost_per_sec2=.1),
        line_windows_by_utterance=line_windows,
        vocalization_reattacks_by_utterance=reattacks if reattacks else None,
    )
    if allow_spoken:
        from .spoken import add_spoken_fallback
        result = add_spoken_fallback(
            result, {f"u{i}": (line.start_sec, line.end_sec) for i, line in enumerate(lines)
                     if line.start_sec is not None and line.end_sec is not None},
            lambda windows: _run_adapter("vocal activity", adapters.vocal_activity, path, windows),
            fill_unpitched_lines=True,
        )
    return attach_lyric_surface(result, overlay) if overlay is not None else result
