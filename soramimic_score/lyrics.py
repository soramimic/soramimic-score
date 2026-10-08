"""Additive whole-line completion of supplied lyrics from recognition."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from .supplied_lyrics import plan_supplied_lyrics

if TYPE_CHECKING:
    from .audio import LyricLine


@dataclass(frozen=True)
class LyricAdjustment:
    lines: tuple[LyricLine, ...]
    detail: dict


def adjust_known_lyrics(
    supplied: Sequence[str],
    recognized: Sequence[LyricLine],
    *,
    reading: Callable[[str], str] | None = None,
    min_similarity: float = .65,
    max_group_lines: int = 4,
) -> LyricAdjustment:
    """Preserve every supplied line and add distinct extra recognized occurrences.

    The transcript can locate input and suggest additions, but cannot delete or
    replace it. Disagreement inside an unmatched input span is not an addition.
    The caller remains responsible for rejecting ASR hallucinations.
    """
    from .audio import LyricLine, _validate_lines

    recognized = _validate_lines(recognized, timed=True) if recognized else ()
    plan = plan_supplied_lyrics(
        supplied, recognized, reading=reading, add_missing=True,
        minimum_similarity=min_similarity, max_group_lines=max_group_lines,
    )
    output = []
    decisions = []
    for group in plan["groups"]:
        indices = group["supplied_indices"]
        texts = [supplied[i] for i in indices] if indices else [group["display_text"]]
        sources = [recognized[i] for i in group["asr_indices"]]
        timed = len(texts) == 1 and group["operation"] != "retain_supplied" and sources
        first = len(output)
        output.extend(LyricLine(text, sources[0].start_sec if timed else None,
                                sources[-1].end_sec if timed else None) for text in texts)
        decisions.append({
            "operation": "keep" if group["operation"] == "match" else group["operation"],
            "supplied_line_indices": indices,
            "recognized_line_indices": group["asr_indices"],
            "output_line_indices": list(range(first, len(output))),
            "similarity": group["similarity"],
            "repeated_supplied_indices": group.get("repeated_supplied_indices", []),
        })
    return LyricAdjustment(tuple(output), {
        "mode": "additive-audio-completion", "supplied_lines": list(supplied),
        "recognized_lines": [asdict(line) for line in recognized],
        "adjusted_lines": [asdict(line) for line in output], "decisions": decisions,
        "min_similarity": min_similarity, "max_group_lines": max_group_lines,
        "confidence_available": False,
        "limitation": "Recognition errors may cause incorrect additions; input lines are retained.",
    })
