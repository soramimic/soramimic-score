"""Bounded phonetic proposals for gaps in supplied-lyric location hints."""

from __future__ import annotations

from functools import lru_cache

from .readings import _comparison_kana, _substring_distance


def propose_phonetic_locations(plan, recognized, *, reading, candidates):
    """Leave established anchors intact and propose only local unmatched windows.

    Dictionary alternatives locate text; they do not select its final reading.
    A weaker, distinctive overlap can propose joint CTC context with an adjacent
    anchor. It never promotes a retained lyric to a match or verifies its CTC.
    Recognition and acoustic verification downstream remain independent.
    """
    if reading is None or candidates is None:
        return []
    groups = plan["groups"]
    edges = []

    @lru_cache(maxsize=None)
    def source_reading(index):
        return _comparison_kana(reading(recognized[index].text))

    @lru_cache(maxsize=4096)
    def distance(target, evidence):
        return _substring_distance(target, evidence)

    for index, group in enumerate(groups):
        if group["operation"] != "retain_supplied":
            continue
        neighbors = [i for i in (index - 1, index + 1)
                     if 0 <= i < len(groups) and groups[i]["operation"] == "match"]
        boundary = [groups[i]["asr_indices"][-1 if i < index else 0] for i in neighbors]
        indices = [*group["asr_indices"], *boundary]
        if not indices:
            continue
        first, last = min(indices), max(indices) + 1
        # A gap with excessive recognition fragments remains unresolved rather
        # than starting an unbounded search. Each proposed phrase is shorter.
        if (last - first > 32
                or recognized[last - 1].end_sec - recognized[first].start_sec > 60.):
            continue
        options = tuple(dict.fromkeys(_comparison_kana(kana)
                                      for kana in candidates(group["display_text"])))
        options = tuple(kana for kana in options if kana and distance(kana, "") > 0)
        proposals = []
        for start in range(first, last):
            for end in range(start + 1, min(last, start + 4) + 1):
                lo, hi = recognized[start].start_sec, recognized[end - 1].end_sec
                if not 0 < hi - lo <= 24.:
                    continue
                evidence = "".join(source_reading(i) for i in range(start, end))
                if not evidence:
                    continue
                for kana in options:
                    similarity = max(0., 1. - distance(kana, evidence) / distance(kana, ""))
                    proposals.append({"similarity": similarity, "reading": kana,
                                      "asr_indices": list(range(start, end)),
                                      "window_sec": [lo, hi]})
        if not proposals:
            continue
        proposals.sort(key=lambda p: (-p["similarity"],
                                     p["window_sec"][1] - p["window_sec"][0],
                                     len(p["asr_indices"])))
        best = proposals[0]
        used = set(best["asr_indices"])
        runner = next((p for p in proposals if used.isdisjoint(p["asr_indices"])), None)
        margin = best["similarity"] - runner["similarity"] if runner else None
        shared = [i for i in neighbors if used.intersection(groups[i]["asr_indices"])]
        # Equivalent repetitions cannot supply a new anchor. Weak overlap is
        # useful only as joint acoustic context, with a distinct local winner.
        distinctive = margin is None or margin >= .15
        strong = best["similarity"] >= plan["minimum_similarity"]
        context = bool(shared) and best["similarity"] >= .5
        audit = {**best, "runner_up_similarity": runner["similarity"] if runner else None,
                 "margin": margin, "similarity_is_calibrated_probability": False,
                 "accepted": distinctive and (strong or context)}
        group["phonetic_location_proposal"] = audit
        if not audit["accepted"]:
            continue
        group["phonetic_location"] = {**audit, "prior_asr_indices": group["asr_indices"],
                                      "joint_context_group_indices": shared}
        # Keep the authored surface and the retained-input operation. The ASR
        # indices now describe the proposed window used for independent checks.
        group["asr_indices"] = best["asr_indices"]
        group["original_text"] = "\n".join(recognized[i].text for i in best["asr_indices"])
        group["original_acoustic_reading"] = "".join(source_reading(i) for i in best["asr_indices"])
        edges.extend([index, neighbor] for neighbor in shared)
    return edges
