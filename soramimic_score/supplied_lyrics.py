"""Preserve supplied lyrics while using recognition as ordered location hints."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
import math
from typing import TYPE_CHECKING, Any

from .japanese import strip_ruby
from .parenthetical import matching_forms
from .surface import SurfaceLine, align_lyric_surface, normalize_surface

if TYPE_CHECKING:
    from .audio import LyricLine


_TIMESTAMP_EPSILON_SEC = 1e-6
# Bound only newly joined CTC targets; existing long windows remain unchanged.
_MAX_CTC_CONTEXT_SEC = 60.


def _has_duration(start, end):
    return end - start > _TIMESTAMP_EPSILON_SEC


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
    surfaces = []
    for text in supplied:
        base, annotation = matching_forms(text)
        surfaces.append(SurfaceLine(strip_ruby(base), convert(text),
                                    convert(annotation) if annotation != text else None))
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


def _locate_groups(plan, recognized, duration):
    """Locate raw groups before resolving touching recognition boundaries."""
    groups = [dict(group) for group in plan["groups"]]
    for index, group in enumerate(groups):
        if "phonetic_location" in group:
            start, end = group["phonetic_location"]["window_sec"]
            group.update(start_sec=start, end_sec=end)
        elif group["operation"] in {"match", "add", "repeat"}:
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
    return groups


def locate_supplied_groups(plan, recognized, duration):
    """Keep the reading-selection windows independent of CTC context expansion."""
    groups = _locate_groups(plan, recognized, duration)
    for index, group in enumerate(groups):
        group["location_group_indices"] = [index]
    # A missing phrase can be inside a coarse, touching Whisper interval. Align
    # it jointly with a neighbor instead of inventing a fractional timestamp.
    index = 0
    while index < len(groups):
        group = groups[index]
        if _has_duration(group["start_sec"], group["end_sec"]) or len(groups) == 1:
            index += 1
            continue
        start = index if index + 1 < len(groups) else index - 1
        left, right = groups[start:start + 2]
        groups[start:start + 2] = [{
            "operation": "retain_supplied", "reading_source": "supplied-lyrics",
            "location_group_indices": left["location_group_indices"] + right["location_group_indices"],
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


def prepare_supplied_audio(path, supplied, recognized, notes, adapters, *, adjust_lyrics,
                           recognition_evidence):
    """Prepare authoritative text and measure whether unresolved windows have support."""
    from .audio import AudioPipelineError, LyricLine, _run_adapter

    duration = max([0.] + [line.end_sec for line in recognized] + [note.end_sec for note in notes])
    if adapters.audio_duration is not None:
        duration = float(adapters.audio_duration(path))
    if not math.isfinite(duration) or duration < 0:
        raise AudioPipelineError("lyrics", "recording duration must be finite and nonnegative")
    plan = plan_supplied_lyrics(supplied, recognized, reading=adapters.lyric_reading,
                                add_missing=adjust_lyrics)
    from .supplied_location import propose_phonetic_locations
    context_edges = propose_phonetic_locations(
        plan, recognized, reading=adapters.lyric_reading,
        candidates=adapters.lyric_reading_candidates,
    )
    located = _locate_groups(plan, recognized, duration)
    # A zero-duration gap does not identify which neighbor contains the omitted
    # phrase. Retain both anchors for a later, joint CTC target, after readings
    # and support decisions have been fixed using the original windows.
    plan["ctc_context_edges"] = context_edges + [
        [index, neighbor]
        for index, group in enumerate(located)
        if not _has_duration(group["start_sec"], group["end_sec"])
        for neighbor in (index - 1, index + 1) if 0 <= neighbor < len(located)
    ]
    groups = locate_supplied_groups(plan, recognized, duration)
    windows = [(g["start_sec"], g["end_sec"]) for g in groups
               if _has_duration(g["start_sec"], g["end_sec"])]
    activity = (tuple(_run_adapter("vocal activity", adapters.vocal_activity, path, windows))
                if adapters.vocal_activity is not None and windows else None)
    if activity is not None and len(activity) != len(windows):
        raise AudioPipelineError("vocal activity", "one result is required per lyric window")
    by_window = dict(zip(windows, activity, strict=True)) if activity is not None else {}
    retained = []
    removed = []
    unobserved = []
    lines = []
    for group in groups:
        start, end = group["start_sec"], group["end_sec"]
        pitched = any(n.start_sec < end and n.end_sec > start for n in notes)
        voiced = by_window.get((start, end))
        # Filtered or mismatched recognition still prevents claiming that no
        # voice was heard. A missing adapter means unknown, not an empty result.
        recognized_support = bool(group["asr_indices"]) or (
            any(line.start_sec < end and line.end_sec > start for line in recognition_evidence)
            if recognition_evidence is not None else None)
        measured_activity = (voiced is not None
                             and all(math.isfinite(value) for value in (
                                 voiced.percentile_dbfs, voiced.relative_db,
                                 voiced.active_frame_ratio))
                             and 0 <= voiced.active_frame_ratio <= 1
                             and type(voiced.supported) is bool)
        # A short utterance can fall below the percentile gate while some
        # frames are still active. Any such frame is evidence to retain input.
        vocal_support = None
        if measured_activity:
            if (voiced.supported or voiced.active_frame_ratio > 0
                    or voiced.silence_confirmed is False):
                vocal_support = True
            elif voiced.silence_confirmed is True:
                vocal_support = False
        group["support"] = {"melody": pitched, "recognition": recognized_support,
                            "vocal_activity": vocal_support}
        if measured_activity:
            group["vocal_activity_measurement"] = asdict(voiced)
        if (group["operation"] in {"add", "repeat"} and not pitched
                and voiced is not None and not voiced.supported):
            plan.setdefault("rejected_additions", []).append(group)
            continue
        no_evidence = (not pitched and recognized_support is False and vocal_support is False)
        # Removal needs a real, fully measured interval inside the recording.
        # Unresolved/touching windows and an inferred recording end cannot prove
        # absence. Preserve the original input and the complete removal audit.
        if (adjust_lyrics and group["supplied_indices"] and no_evidence
                and adapters.audio_duration is not None and 0 <= start < end <= duration):
            removed.append(group | {
                "operation": "remove", "reason": "no-performance-evidence",
                "alignment_status": "not-aligned", "line_indices": [], "utterance_ids": [],
            })
            continue
        missing = not _has_duration(start, end) or no_evidence
        index = len(lines)
        group.update(line_indices=[index], utterance_ids=[f"u{index}"])
        if missing:
            group["alignment_status"] = ("window-unresolved" if not _has_duration(start, end)
                                         else "no-performance-evidence")
            unobserved.append(index)
        else:
            group["alignment_status"] = "pending"
        # Keep unsupported text canonical without forcing it into silent audio.
        lines.append(LyricLine(group["display_text"],
                               None if missing else start, None if missing else end))
        retained.append(group)
    removed_indices = [i for group in removed for i in group["supplied_indices"]]
    plan.update(groups=retained, removed_groups=removed, removed_supplied_indices=removed_indices,
                unused_supplied_indices=removed_indices,
                display_text="\n".join(line.text for line in lines),
                unobserved_supplied_indices=[i for index in unobserved
                                             for i in retained[index]["supplied_indices"]],
                recording_duration_sec=duration)
    return tuple(lines), plan, tuple(unobserved)


def select_authored_readings(lines, select):
    """Select each authored line, retaining a joint target for acoustic alignment.

    Recognition can locate several input lines in one interval. Keeping each
    selected reading explicitly avoids guessing a text-to-mora cut afterwards.
    """
    from dataclasses import replace
    from .audio import ReadingSelection, _validate_readings

    expanded = []
    owners = []
    for index, line in enumerate(lines):
        for text in line.text.split("\n"):
            expanded.append(replace(line, text=text))
            owners.append(index)
    selected = _validate_readings(expanded, select(tuple(expanded), tuple(owners)))
    output = []
    for index, line in enumerate(lines):
        parts = [reading for owner, reading in zip(owners, selected, strict=True)
                 if owner == index]
        if len(parts) == 1:
            output.append(parts[0])
            continue
        kana = "".join(part.kana for part in parts)
        detail = {"authored_line_readings": [asdict(part) for part in parts]}
        if any("parenthetical_readings" in part.detail for part in parts):
            detail.update(
                resolved_text="\n".join(part.detail.get("resolved_text", text)
                                        for part, text in zip(parts, line.text.split("\n"), strict=True)),
                parenthetical_readings=[item for part in parts
                                        for item in part.detail.get("parenthetical_readings", ())],
            )
        output.append(ReadingSelection(kana, "supplied-lines", min(p.confidence for p in parts),
                                        (kana,), detail))
    return tuple(output)


def _authored_groups(group, parts):
    """Keep each authored line's provenance when CTC joins reading windows."""
    if "authored_line_groups" in group:
        return group["authored_line_groups"]
    indices = group["supplied_indices"]
    if len(parts) > 1 and indices and len(indices) != len(parts):
        raise ValueError("authored lines do not match supplied lyric provenance")
    originals = group.get("supplied_display_text", group["display_text"]).split("\n")
    output = []
    for index, part in enumerate(parts):
        item = dict(group)
        if len(parts) > 1:
            item["supplied_indices"] = indices[index:index + 1]
            item.pop("supplied_display_text", None)
            item.pop("parenthetical_readings", None)
            if "parenthetical_readings" in part.detail:
                item.update(supplied_display_text=originals[index],
                            parenthetical_readings=part.detail["parenthetical_readings"])
        output.append(item)
    return output


def expand_alignment_windows(lines, readings, overlay, unobserved):
    """Join CTC targets around unresolved gaps without reselecting readings.

    Unsupported/removed groups are barriers. A chain of missing recognition
    anchors must not grow a newly joined target beyond the resource bound.
    Original reading windows and per-line support remain auditable separately.
    """
    from dataclasses import replace
    from .audio import ReadingSelection

    groups = overlay["groups"]
    owners = {raw: index for index, group in enumerate(groups)
              for raw in group["location_group_indices"]}
    parents = list(range(len(groups)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left, right in overlay["ctc_context_edges"]:
        if left not in owners or right not in owners:
            continue
        a, b = sorted((owners[left], owners[right]))
        if any(i in unobserved for i in range(a, b + 1)):
            continue
        a, b = root(a), root(b)
        if a == b:
            continue
        members = [i for i in range(len(groups)) if root(i) in (a, b)]
        start, end = lines[members[0]].start_sec, lines[members[-1]].end_sec
        if end - start > _MAX_CTC_CONTEXT_SEC:
            continue
        parents[b] = a
    components = []
    for index in range(len(groups)):
        if not components or root(components[-1][0]) != root(index):
            components.append([])
        components[-1].append(index)
    if all(len(ids) == 1 for ids in components):
        return lines, readings, unobserved

    target_lines, target_readings, target_groups, target_unobserved = [], [], [], []
    for target, ids in enumerate(components):
        if len(ids) == 1:
            index = ids[0]
            target_lines.append(lines[index])
            target_readings.append(readings[index])
            target_groups.append(groups[index])
            if index in unobserved:
                target_unobserved.append(target)
        else:
            parts, provenance = [], []
            for index in ids:
                reading = readings[index]
                children = reading.detail.get("authored_line_readings")
                children = [ReadingSelection(**part) for part in children] if children else [reading]
                parts.extend(children)
                provenance.extend(_authored_groups(groups[index], children))
            kana = "".join(part.kana for part in parts)
            target_lines.append(replace(lines[ids[0]],
                text="\n".join(lines[i].text for i in ids), end_sec=lines[ids[-1]].end_sec))
            target_readings.append(ReadingSelection(
                kana, "supplied-lines", min(part.confidence for part in parts), (kana,),
                {"authored_line_readings": [asdict(part) for part in parts]},
            ))
            target_groups.append({
                "operation": "retain_supplied", "reading_source": "supplied-lyrics",
                "supplied_indices": [s for i in ids for s in groups[i]["supplied_indices"]],
                "asr_indices": [s for i in ids for s in groups[i]["asr_indices"]],
                "display_text": target_lines[-1].text,
                "original_text": "\n".join(groups[i]["original_text"] for i in ids
                                             if groups[i]["original_text"]),
                "original_acoustic_reading": "".join(groups[i]["original_acoustic_reading"] for i in ids),
                "similarity": None, "start_sec": target_lines[-1].start_sec,
                "end_sec": target_lines[-1].end_sec, "alignment_status": "pending",
                "acoustic_reading": kana, "reading_candidates": [kana],
                "reading_stage_group_indices": ids, "authored_line_groups": provenance,
            })
        target_groups[-1] = target_groups[-1] | {
            "line_indices": [target], "utterance_ids": [f"u{target}"],
        }
    overlay["reading_stage_groups"] = groups
    overlay["groups"] = target_groups
    return tuple(target_lines), tuple(target_readings), tuple(target_unobserved)


def restore_authored_lines(lines, readings, moras, overlay, unobserved):
    """Restore input boundaries using exact selected mora counts and CTC times."""
    from dataclasses import replace
    from .audio import ReadingSelection
    from .japanese import kana_to_moras

    if not any("authored_line_readings" in reading.detail for reading in readings):
        return lines, readings, moras, unobserved
    restored_lines, restored_readings, restored_moras, restored_groups = [], [], [], []
    restored_unobserved = []
    for index, (line, reading, group) in enumerate(zip(lines, readings, overlay["groups"], strict=True)):
        parts = reading.detail.get("authored_line_readings")
        parts = tuple(ReadingSelection(**part) for part in parts) if parts else (reading,)
        texts = line.text.split("\n") if len(parts) > 1 else (line.text,)
        provenance = _authored_groups(group, parts)
        offset = 0
        for part_index, (text, part) in enumerate(zip(texts, parts, strict=True)):
            target = len(restored_lines)
            count = len(kana_to_moras(part.kana))
            aligned = [replace(mora, line_index=target, mora_index=mora.mora_index - offset)
                       for mora in moras if mora.line_index == index
                       and offset <= mora.mora_index < offset + count]
            offset += count
            start, end = line.start_sec, line.end_sec
            if len(parts) > 1:
                start = min((m.start_sec for m in aligned), default=None)
                end = max((m.end_sec for m in aligned), default=None)
            restored_lines.append(replace(line, text=text, start_sec=start, end_sec=end))
            restored_readings.append(part)
            restored_moras.extend(aligned)
            if index in unobserved:
                restored_unobserved.append(target)
            restored_group = provenance[part_index] | {
                "display_text": text, "acoustic_reading": part.kana,
                "reading_candidates": list(part.candidates),
                "line_indices": [target], "utterance_ids": [f"u{target}"],
                "start_sec": start if start is not None else group["start_sec"],
                "end_sec": end if end is not None else group["end_sec"],
                "alignment_status": group["alignment_status"],
                "acoustic_group_index": index, "shared_acoustic_window": len(parts) > 1,
            }
            restored_groups.append(restored_group)
    overlay["acoustic_groups"] = overlay["groups"]
    overlay["groups"] = restored_groups
    return (tuple(restored_lines), tuple(restored_readings), tuple(restored_moras),
            tuple(restored_unobserved))
