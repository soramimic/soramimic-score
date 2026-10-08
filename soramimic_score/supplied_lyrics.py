"""Preserve supplied lyrics while using recognition as ordered location hints."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
import math
from typing import TYPE_CHECKING, Any

from .japanese import strip_ruby
from .surface import SurfaceLine, align_lyric_surface, normalize_surface

if TYPE_CHECKING:
    from .audio import LyricLine


def plan_supplied_lyrics(
    supplied: Sequence[str], recognized: Sequence[LyricLine], *,
    reading: Callable[[str], str] | None = None,
    add_missing: bool = False, minimum_similarity: float = .72,
    max_group_lines: int = 4,
) -> dict[str, Any]:
    """Keep every input occurrence; only empty input gaps can accept additions.

    Matching supplies anchors, never permission to discard input. A transcript
    disagreement inside an unmatched input span cannot be distinguished from an
    ASR error, so it is not appended as a second, competing lyric. No character
    times or proportional line times are inferred here.
    """
    if isinstance(supplied, (str, bytes)) or not supplied or any(
        not isinstance(text, str) or not text.strip() for text in supplied
    ):
        raise ValueError("supplied lyrics must be a nonempty sequence of lines")
    convert = reading or (lambda _: "")
    surfaces = [SurfaceLine(strip_ruby(text), convert(text)) for text in supplied]
    sources = [SurfaceLine(strip_ruby(line.text), convert(line.text)) for line in recognized]
    matched = (align_lyric_surface(
        sources, surfaces, minimum_similarity=minimum_similarity,
        max_group_lines=max_group_lines,
    ) if sources else {"groups": [], "unused_supplied_indices": list(range(len(supplied)))})
    anchors = [g for g in matched["groups"] if g["operation"] == "match"]
    groups: list[dict[str, Any]] = []
    ignored_asr: list[int] = []

    def append(indices, asr_indices, operation, *, text=None, similarity=None):
        groups.append({
            "operation": operation, "supplied_indices": list(indices),
            "asr_indices": list(asr_indices),
            "display_text": text if text is not None else "\n".join(supplied[i] for i in indices),
            "original_text": "\n".join(recognized[i].text for i in asr_indices),
            "original_acoustic_reading": "".join(sources[i].reading for i in asr_indices),
            "reading_source": "automatic-addition" if operation == "add" else "supplied-lyrics",
            "similarity": similarity,
        })

    def gap(first, last, source_first, source_last):
        if first < last:
            append(range(first, last), range(source_first, source_last), "retain_supplied")
            return
        if not add_missing:
            ignored_asr.extend(range(source_first, source_last))
            return
        for index in range(source_first, source_last):
            source = sources[index]
            # An extra complete occurrence uses the supplied spelling. Other
            # additions keep the recognized line intact, with distinct provenance.
            repetitions = [i for i, target in enumerate(surfaces)
                           if normalize_surface(source.text) == normalize_surface(target.text)
                           or (source.reading and target.reading
                               and normalize_surface(source.reading)
                               == normalize_surface(target.reading))]
            spellings = {supplied[i] for i in repetitions}
            if len(spellings) == 1:
                append([], [index], "repeat", text=next(iter(spellings)))
                groups[-1]["repeated_supplied_indices"] = repetitions
            else:
                append([], [index], "add", text=recognized[index].text)

    prior_input = prior_asr = 0
    for anchor in anchors:
        ii, aa = anchor["supplied_indices"], anchor["asr_indices"]
        gap(prior_input, ii[0], prior_asr, aa[0])
        append(ii, aa, "match", similarity=anchor["similarity"])
        prior_input, prior_asr = ii[-1] + 1, aa[-1] + 1
    gap(prior_input, len(supplied), prior_asr, len(recognized))
    assert [i for group in groups for i in group["supplied_indices"]] == list(range(len(supplied)))
    return {
        "schema_version": 1, "mode": "supplied-lyrics-first",
        "supplied_lines": list(supplied), "recognized_lines": [asdict(line) for line in recognized],
        "groups": groups, "unused_supplied_indices": [],
        "unmatched_supplied_indices": list(matched["unused_supplied_indices"]),
        "ignored_asr_indices": ignored_asr, "add_missing": add_missing,
        "minimum_similarity": minimum_similarity, "max_group_lines": max_group_lines,
        "similarity_is_calibrated_probability": False,
        "display_text": "\n".join(group["display_text"] for group in groups),
    }


def locate_supplied_groups(plan, recognized, duration):
    """Bound unmatched text by its neighboring anchors and the recording limits."""
    groups = [dict(group) for group in plan["groups"]]
    for index, group in enumerate(groups):
        if group["operation"] in {"match", "add", "repeat"}:
            sources = [recognized[i] for i in group["asr_indices"]]
            group.update(start_sec=sources[0].start_sec, end_sec=sources[-1].end_sec)
        else:
            left = next((g for g in reversed(groups[:index])
                         if g["operation"] in {"match", "add", "repeat"}), None)
            right = next((g for g in groups[index + 1:]
                          if g["operation"] in {"match", "add", "repeat"}), None)
            group.update(
                start_sec=recognized[left["asr_indices"][-1]].end_sec if left else 0.,
                end_sec=recognized[right["asr_indices"][0]].start_sec if right else duration,
            )
    # A missing phrase can be inside a coarse, touching Whisper interval. Align
    # it jointly with a neighbor instead of inventing a fractional timestamp.
    index = 0
    while index < len(groups):
        group = groups[index]
        if group["end_sec"] > group["start_sec"] or len(groups) == 1:
            index += 1
            continue
        start = index if index + 1 < len(groups) else index - 1
        left, right = groups[start:start + 2]
        groups[start:start + 2] = [{
            "operation": "retain_supplied", "reading_source": "supplied-lyrics",
            "supplied_indices": left["supplied_indices"] + right["supplied_indices"],
            "asr_indices": left["asr_indices"] + right["asr_indices"],
            "display_text": left["display_text"] + "\n" + right["display_text"],
            "original_text": "\n".join(filter(None, [left["original_text"], right["original_text"]])),
            "original_acoustic_reading": (left["original_acoustic_reading"]
                                           + right["original_acoustic_reading"]),
            "similarity": None, "start_sec": left["start_sec"], "end_sec": right["end_sec"],
        }]
        index = max(0, start - 1)
    return groups


def prepare_supplied_audio(path, supplied, recognized, notes, adapters, *, add_missing):
    """Prepare authoritative text and measure whether unresolved windows have support."""
    from .audio import AudioPipelineError, LyricLine, _run_adapter

    duration = max([0.] + [line.end_sec for line in recognized] + [note.end_sec for note in notes])
    if adapters.audio_duration is not None:
        duration = float(adapters.audio_duration(path))
    if not math.isfinite(duration) or duration < 0:
        raise AudioPipelineError("lyrics", "recording duration must be finite and nonnegative")
    plan = plan_supplied_lyrics(supplied, recognized, reading=adapters.lyric_reading,
                                add_missing=add_missing)
    groups = locate_supplied_groups(plan, recognized, duration)
    windows = [(g["start_sec"], g["end_sec"]) for g in groups
               if g["end_sec"] > g["start_sec"]]
    activity = (tuple(_run_adapter("vocal activity", adapters.vocal_activity, path, windows))
                if adapters.vocal_activity is not None and windows else None)
    if activity is not None and len(activity) != len(windows):
        raise AudioPipelineError("vocal activity", "one result is required per lyric window")
    by_window = dict(zip(windows, activity, strict=True)) if activity is not None else {}
    retained = []
    unobserved = []
    lines = []
    for group in groups:
        start, end = group["start_sec"], group["end_sec"]
        pitched = any(n.start_sec < end and n.end_sec > start for n in notes)
        voiced = by_window.get((start, end))
        recognized_support = bool(group["asr_indices"])
        group["support"] = {"melody": pitched, "recognition": recognized_support,
                            "vocal_activity": voiced.supported if voiced is not None else None}
        if (group["operation"] in {"add", "repeat"} and not pitched
                and voiced is not None and not voiced.supported):
            plan.setdefault("rejected_additions", []).append(group)
            continue
        missing = (end <= start or not pitched and not recognized_support
                   and voiced is not None and not voiced.supported)
        index = len(lines)
        group.update(line_indices=[index], utterance_ids=[f"u{index}"])
        if missing:
            group["alignment_status"] = ("window-unresolved" if end <= start
                                         else "no-performance-evidence")
            unobserved.append(index)
        else:
            group["alignment_status"] = "pending"
        # Keep unsupported text canonical without forcing it into silent audio.
        lines.append(LyricLine(group["display_text"],
                               None if missing else start, None if missing else end))
        retained.append(group)
    plan.update(groups=retained, display_text="\n".join(line.text for line in lines),
                unobserved_supplied_indices=[i for index in unobserved
                                             for i in retained[index]["supplied_indices"]],
                recording_duration_sec=duration)
    return tuple(lines), plan, tuple(unobserved)
