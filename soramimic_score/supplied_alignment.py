"""Verify supplied phrases before giving their CTC timing precedence over notes."""

from __future__ import annotations

from dataclasses import asdict, replace
import math
import statistics

from .ir import MINIMUM_USABLE_TIMING_CONFIDENCE
from .semantic import MIN_CTC_MEDIAN_SCORE
from .surface import normalize_surface
from .readings import _substring_distance


def _quality(moras):
    if not moras:
        return 0.0
    return math.exp(statistics.mean(math.log(max(m.confidence, 1e-30)) for m in moras))


def _has_timing(moras):
    return bool(moras) and min(m.confidence for m in moras) >= MINIMUM_USABLE_TIMING_CONFIDENCE


def _candidate_key(moras):
    return ("".join(m.kana for m in moras), moras[0].start_sec, moras[-1].end_sec) if moras else ()


def _verified(moras):
    # This is a conservative gate, not a calibrated probability of correctness.
    # One catastrophic forced token cannot establish a complete supplied phrase.
    return _has_timing(moras) and _quality(moras) >= MIN_CTC_MEDIAN_SCORE


def _similarity(kana, text):
    expected = normalize_surface(kana).replace("ー", "")
    heard = normalize_surface(text).replace("ー", "")
    # Require coverage in both directions. A matching short suffix must not
    # validate an unrelated long utterance. Kanasim tolerates nearby sounds in
    # the independent kana recognition without treating all edits equally.
    try:
        scores = [1. - _substring_distance(a, b) / _substring_distance(a, "")
                  for a, b in ((expected, heard), (heard, expected)) if a]
    except (ValueError, ZeroDivisionError):
        return 0.0
    return max(0.0, min(scores)) if len(scores) == 2 else 0.0


def _source_agreement(window, reading, overlay, adapters, cache):
    parts = []
    for source in overlay.get("recognized_lines", ()):
        if window[0] - 1e-6 <= source["start_sec"] and source["end_sec"] <= window[1] + 1e-6:
            text = source.get("text", "")
            if text not in cache:
                cache[text] = adapters.lyric_reading(text) if adapters.lyric_reading else text
            parts.append(cache[text])
    return _similarity(reading.kana, "".join(parts)) if parts else 0.0


def _recognize_candidates(path, line, candidates, source_support, reading, overlay, adapters, audit):
    from .audio import _run_adapter

    by_window = {}
    minimum = overlay.get("minimum_similarity", .72)
    def rank(candidate):
        support = source_support.get(_candidate_key(candidate), 0.)
        return (support >= minimum, support, _quality(candidate))
    for candidate in sorted(candidates, key=rank, reverse=True):
        if _has_timing(candidate):
            window = (candidate[0].start_sec, candidate[-1].end_sec)
            if window[1] - window[0] <= 24.:
                by_window.setdefault(window, candidate)
    # Bound new acoustic work even when recognition produced many small spans.
    windows = tuple(by_window)[:3]
    # CTC peaks can cut off the consonant or held vowel at a phrase edge.
    # Context is only for verification; it does not replace the measured times.
    recognition_windows = tuple((max(line.start_sec, lo - .5), min(line.end_sec, hi + .5))
                                for lo, hi in windows)
    texts = tuple(_run_adapter("supplied phrase recognition",
        adapters.repetition_evidence, path, recognition_windows)) if windows else ()
    if len(texts) != len(windows):
        raise ValueError("supplied phrase recognition needs one result per window")
    checks = []
    selected = max(candidates, key=_quality)
    accepted = []
    for window, text in zip(windows, texts, strict=True):
        similarity = _similarity("".join(m.kana for m in by_window[window]), text)
        source_similarity = source_support.get(_candidate_key(by_window[window]), 0.)
        verified = max(similarity, source_similarity) >= minimum
        checks.append({"window_sec": list(window), "text": text,
                       "similarity": similarity, "verified": verified,
                       "asr_phonetic_similarity": source_similarity,
                       "similarity_is_calibrated_probability": False})
        if verified:
            accepted.append((window[1] - window[0], -similarity, -_quality(by_window[window]),
                             window, text))
    if accepted:
        _duration, _similarity_score, _ctc_score, window, text = min(accepted)
        selected = by_window[window]
        audit["selected_recognition"] = text
    audit["independent_recognition"] = checks
    return selected, bool(accepted)


def _search_windows(line, group, overlay):
    """Bound retries by observed recognition intervals, never proportional cuts."""
    sources = overlay.get("recognized_lines", ())
    indices = group.get("asr_indices", ())
    windows = {(line.start_sec, line.end_sec)}
    for start in range(len(indices)):
        for end in range(start + 1, min(len(indices), start + overlay.get("max_group_lines", 4)) + 1):
            left, right = sources[indices[start]], sources[indices[end - 1]]
            lo = max(line.start_sec, left["start_sec"])
            hi = min(line.end_sec, right["end_sec"])
            if hi > lo and hi - lo <= 24.:
                windows.add((lo, hi))
    return tuple(sorted(windows))


def verify_supplied_alignment(path, lines, readings, moras, overlay, unobserved, adapters):
    """Relocate unmatched phrases with free audio edges; never invent timestamps.

    Matched spelling remains a location hint even with a weak CTC posterior.
    An unmatched phrase must establish its acoustic correspondence independently.
    Raw and retried observations remain in the private Score evidence, including
    rejected attempts. Custom adapters without a locator retain the same gate.
    """
    from .audio import CTCWindowCapacityError, _run_adapter, _validate_moras, _validate_readings

    updated_lines, updated_readings = list(lines), list(readings)
    missing = set(unobserved)
    source_readings = {}
    output = []
    for index, (line, reading, group) in enumerate(zip(lines, readings, overlay["groups"], strict=True)):
        original = tuple(m for m in moras if m.line_index == index)
        if index in missing:
            group["ctc_verified"] = False
            continue
        selected = original
        candidates = [original]
        source_support = {}
        if original:
            source_support[_candidate_key(original)] = _source_agreement(
                (line.start_sec, line.end_sec), reading, overlay, adapters, source_readings)
        audit = {"initial_quality": _quality(original), "attempts": []}
        unmatched = group["operation"] == "retain_supplied"
        # Shared targets already contain the neighboring supplied phrases. Do
        # not independently move a child across another authored line's timing.
        if (unmatched and not group.get("shared_acoustic_window")
                and adapters.supplied_mora_aligner is not None):
            audit["initial_moras"] = [asdict(m) for m in original]
            for lo, hi in _search_windows(line, group, overlay):
                target = replace(line, start_sec=lo, end_sec=hi)
                for kana in dict.fromkeys((reading.kana, *reading.candidates)):
                    option = replace(reading, kana=kana)
                    try:
                        candidate = _validate_moras((option,), _run_adapter(
                            "supplied mora alignment", adapters.supplied_mora_aligner,
                            path, (target,), (option,)))
                    except CTCWindowCapacityError:
                        continue
                    candidate = tuple(replace(m, line_index=index) for m in candidate)
                    candidates.append(candidate)
                    key = _candidate_key(candidate)
                    source_support[key] = max(source_support.get(key, 0.), _source_agreement(
                        (lo, hi), option, overlay, adapters, source_readings))
                    audit["attempts"].append({"method": "free-audio-edges",
                        "window_sec": [lo, hi], "reading": kana, "quality": _quality(candidate),
                        "moras": [asdict(m) for m in candidate]})
            selected = max(candidates, key=_quality)
            if selected != original:
                audit["selected"] = "free-audio-edges"
        verified = _verified(selected) or (group["operation"] == "match" and _has_timing(selected))
        if unmatched and adapters.repetition_evidence is not None:
            # A forced target's score alone cannot identify a phrase. Confirm
            # its reading using recognition which never receives that target.
            selected, verified = _recognize_candidates(
                path, line, candidates, source_support, reading, overlay, adapters, audit)
            if selected != original:
                audit["selected"] = "free-audio-edges"
        selected_kana = "".join(m.kana for m in selected)
        if verified and selected_kana != reading.kana:
            updated_readings[index] = replace(reading, kana=selected_kana,
                source="supplied-ctc-reading", detail={**reading.detail,
                    "prior_selected": reading.kana, "selected": selected_kana,
                    "reason": "localized-ctc-and-independent-recognition"})
            reading = updated_readings[index]
        if (unmatched and adapters.supplied_mora_aligner is not None
                and not group.get("shared_acoustic_window")):
            # Reading alternatives are compared against the newly located
            # phrase rather than unrelated speech in its original search range.
            if (verified and len(reading.candidates) > 1
                    and adapters.supplied_reading_selector is not None):
                narrow = replace(line, start_sec=selected[0].start_sec,
                                 end_sec=selected[-1].end_sec)
                refined, = _validate_readings((narrow,), _run_adapter(
                    "supplied readings", adapters.supplied_reading_selector,
                    path, (narrow,), (None,)))
                if refined.kana != reading.kana:
                    retry = _validate_moras((refined,), _run_adapter(
                        "supplied mora alignment", adapters.supplied_mora_aligner,
                        path, (narrow,), (refined,)))
                    audit["attempts"].append({"method": "localized-reading",
                        "quality": _quality(retry), "moras": [asdict(m) for m in retry]})
                    agreement = (_similarity(refined.kana, audit["selected_recognition"])
                                 >= overlay.get("minimum_similarity", .72)
                                 if "selected_recognition" in audit else _verified(retry))
                    if _has_timing(retry) and agreement and _quality(retry) > _quality(selected):
                        selected = tuple(replace(m, line_index=index) for m in retry)
                        updated_readings[index] = refined
                        audit["selected"] = "localized-reading"
        group["ctc_verified"] = verified
        audit["quality"] = _quality(selected)
        audit["verified"] = verified
        group["ctc_alignment"] = audit
        if unmatched and not verified and not group.get("shared_acoustic_window"):
            audit.setdefault("initial_moras", [asdict(m) for m in original])
            group["alignment_status"] = "ctc-unresolved"
            missing.add(index)
            updated_lines[index] = replace(line, start_sec=None, end_sec=None)
            continue
        if audit.get("selected"):
            updated_lines[index] = replace(line, start_sec=selected[0].start_sec,
                                           end_sec=selected[-1].end_sec)
            group.update(start_sec=selected[0].start_sec, end_sec=selected[-1].end_sec,
                         acoustic_reading=updated_readings[index].kana,
                         reading_candidates=list(updated_readings[index].candidates))
        output.extend(selected)
    overlay["unobserved_supplied_indices"] = [s for i in sorted(missing)
                                             for s in overlay["groups"][i]["supplied_indices"]]
    return tuple(updated_lines), tuple(updated_readings), tuple(output), tuple(sorted(missing))
