"""Compare supplied parenthetical readings without assuming they are annotations."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
import re
import unicodedata

from .japanese import kana_to_moras, katakana, strip_ruby


_KANJI = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af々〆〇"
_RUN = re.compile(rf"[{_KANJI}ぁ-ゖァ-ヶー]+$")
_BASE = re.compile(rf"[{_KANJI}][{_KANJI}ぁ-ゖァ-ヶー]*$")
_HAS_KANJI = re.compile(rf"[{_KANJI}]")
_KANA = re.compile(r"[ァ-ヶー]+")
_READING = {"(": ")", "（": "）", "[": "]", "［": "］"}
_QUOTES = {"「": "」", "『": "』", "｢": "｣", "【": "】", "《": "》",
           "≪": "≫", "〈": "〉", "＜": "＞", "“": "”"}
_PAIRS = {**_READING, **_QUOTES, "{": "}", "｛": "｝"}
_TOKENS = re.compile(
    r"[|｜][^|｜《\r\n]+《[^《》\r\n]+》|\\.|"
    + "[" + re.escape("".join(_PAIRS) + "".join(_PAIRS.values())) + "]")


@dataclass(frozen=True)
class Parenthesis:
    start: int
    end: int
    base_start: int
    base_end: int
    run_start: int
    reading: str


def parentheses(text: str) -> tuple[Parenthesis, ...]:
    """Find adjacent kana in four bracket forms, including inside quotations.

    Literal escapes, explicit ruby, whitespace and nested reading brackets are
    preserved. Quotation marks may surround either the word or the whole line.
    Finding a candidate is not a decision to remove the parenthetical text.
    """
    if not any(c in text for c in _READING):
        return ()
    stack = []
    closed = {}
    result = []
    for token in _TOKENS.finditer(text):
        char = token.group()
        if len(char) != 1:
            continue
        if char in _PAIRS:
            stack.append((char, token.start()))
            continue
        if not stack or _PAIRS[stack[-1][0]] != char:
            continue
        opener, start = stack.pop()
        closed[token.end()] = (opener, start, token.start())
        if opener not in _READING or any(c not in _QUOTES for c, _ in stack):
            continue
        reading = katakana(unicodedata.normalize("NFKC", text[start + 1:token.start()]))
        if not _KANA.fullmatch(reading):
            continue
        base_end = start
        minimum = 0
        if start in closed and closed[start][0] in _QUOTES:
            _, left, base_end = closed[start]
            minimum = left + 1
        base = _BASE.search(text, minimum, base_end)
        run = _RUN.search(text, minimum, base_end)
        if base is None or run is None:
            continue
        result.append(Parenthesis(start, token.end(), base.start(), base_end,
                                  run.start(), reading))
    return tuple(result)


def matching_forms(text: str) -> tuple[str, str]:
    """Return surface/read-as hints for locating text, without changing the input."""
    base, annotation = text, text
    for item in reversed(parentheses(text)):
        # The location hint needs the nearby word, not the entire contiguous
        # Japanese clause. Broader scopes remain candidates in reading_options.
        start = item.base_start
        try:
            run = text[item.run_start:item.base_end]
            start = max((item.run_start + i for i in _word_starts(run)
                         if _HAS_KANJI.search(run[i:])), default=start)
        except ImportError:
            # Lightweight custom adapters need not install the audio dictionary.
            pass
        base = base[:item.start] + base[item.end:]
        annotation = (annotation[:start]
                      + f"｜{text[start:item.base_end]}《{item.reading}》"
                      + annotation[item.base_end:item.start] + annotation[item.end:])
    return base, annotation


def _word_starts(text: str) -> tuple[int, ...]:
    import MeCab
    import unidic_lite

    tagger = MeCab.Tagger(f'-d "{unidic_lite.DICDIR}"')
    node = tagger.parseToNode(text)
    cursor = 0
    starts = []
    while node:
        if node.surface:
            starts.append(cursor)
            cursor += len(node.surface)
        node = node.next
    return tuple(starts) if cursor == len(text) else ()


def _key(kana: str) -> tuple[str, ...]:
    from .readings import _comparison_kana
    return tuple(kana_to_moras(_comparison_kana(kana)))


def _edit_cost(expected, observed, *, substring=False):
    """Unit-cost mora alignment; only acoustic context may have free outer tails."""
    previous = [0] * (len(observed) + 1) if substring else list(range(len(observed) + 1))
    for i, mora in enumerate(expected, 1):
        current = [i]
        for j, evidence in enumerate(observed, 1):
            current.append(min(previous[j] + 1, current[-1] + 1,
                               previous[j - 1] + (mora != evidence)))
        previous = current
    return min(previous) if substring else previous[-1]


def _render(text, item, start, reading):
    return (text[:start] + f"｜{text[start:item.base_end]}《{reading}》"
            + text[item.base_end:item.start] + text[item.end:])


def reading_options(text: str, readings: Callable[[str], Sequence[str]], *,
                    word_starts: Callable[[str], Sequence[int]] | None = None) -> tuple[dict, ...]:
    """Generate literal, base and annotation readings, including dictionary-external kana.

    Each parenthesis is compared with the same surrounding text. Other
    annotations and dictionary readings remain context alternatives. Scope
    candidates start at morphological boundaries; no mora-length pruning.
    """
    found = parentheses(text)
    if not found:
        return ()
    word_starts = word_starts or _word_starts

    @lru_cache(maxsize=None)
    def convert(value):
        if not any(not c.isspace() and unicodedata.category(c)[0] not in "PZC"
                   for c in value):
            return ("",)
        return tuple(readings(value))

    output = []
    for item in found:
        prefixes = (*matching_forms(text[:item.run_start]), text[:item.run_start])
        suffixes = (*matching_forms(text[item.end:]), text[item.end:])
        run = text[item.run_start:item.base_end]
        starts = {item.base_start}
        starts.update(item.run_start + i for i in word_starts(run)
                      if _HAS_KANJI.search(run[i:]))
        choices = []
        seen = set()
        for start in sorted(starts, reverse=True):
            base = text[start:item.base_end]
            for mode, pronunciation in [
                ("annotation", item.reading),
                *(("base", reading) for reading in convert(base)),
                *(("literal", reading + item.reading) for reading in convert(base)),
            ]:
                # Explicit local ruby prevents the dictionary from appending the
                # supplied kana to a reading of the base during this comparison.
                for prefix, suffix in dict.fromkeys(zip(prefixes, suffixes, strict=True)):
                    left = prefix + text[item.run_start:start]
                    right = text[item.base_end:item.start] + suffix
                    for kana in convert(left + f"｜{base}《{pronunciation}》" + right):
                        key = mode, start, pronunciation, kana
                        if kana and key not in seen:
                            choices.append({"mode": mode, "start": start, "base": base,
                                            "reading": pronunciation, "kana": kana})
                            seen.add(key)
        output.append({"span": item, "choices": choices})
    return tuple(output)


def choose_reading(options: dict, transcripts: dict[str, str], *,
                   recognition_text: str | None = None) -> dict:
    """Require a unique pronunciation supported by every available audio view.

    Ordinary Whisper can distinguish one occurrence from a repeated reading.
    Re-reading the same input kanji cannot establish a dictionary-external
    pronunciation. Acoustic ties preserve the
    literal text, including repeated words and overlapping backing responses.
    """
    is_whisper = recognition_text is not None
    evidence = {}
    for view, text in transcripts.items():
        try:
            key = _key(text)
        except ValueError:
            continue
        if key:
            evidence[view] = key
    choices = options["choices"]
    rows = []
    for choice in choices:
        key = _key(choice["kana"])
        costs = {view: _edit_cost(key, observed, substring=not is_whisper)
                 for view, observed in evidence.items()}
        full_costs = {view: _edit_cost(key, observed) for view, observed in evidence.items()}
        rows.append({"mode": choice["mode"], "base": choice["base"],
                     "reading": choice["reading"], "kana": choice["kana"],
                     "start": choice["start"], "costs": costs, "full_costs": full_costs})
    decision = {"status": "unresolved", "reason": "no-evidence", "selected": None,
                "source": "whisper" if is_whisper else "kana-whisper",
                "transcripts": dict(transcripts), "candidates": rows,
                "distance": "mora-edit-global",
                "confidence_available": False}
    if not evidence or not choices:
        return decision
    if len(evidence) != len(transcripts):
        return decision | {"reason": "incomplete-evidence"}
    # Marginalize unrelated dictionary/context alternatives: disagreement about
    # another word must not masquerade as ambiguity about this parenthesis.
    # Equal performed pronunciations also cannot distinguish scopes or modes.
    keys = [_key(choice["kana"]) for choice in choices]
    parents = list(range(len(choices)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    by_action, by_sound = {}, {}
    for i, choice in enumerate(choices):
        action = (choice["mode"] == "literal", choice["start"], _key(choice["reading"]))
        # Identical full readings can assign a repetition to different
        # parentheses. Keep single and doubled performances distinct so that
        # such a tie cannot independently remove both parenthetical occurrences.
        sound = (choice["mode"] == "literal", keys[i])
        for table, key in ((by_action, action), (by_sound, sound)):
            if key in table:
                parents[root(i)] = root(table[key])
            table[key] = i
    groups = {}
    for i in range(len(choices)):
        groups.setdefault(root(i), []).append(i)
    supports = {group: {view: min(indices, key=lambda i: (
                    rows[i]["full_costs"][view], rows[i]["costs"][view]))
                       for view in evidence} for group, indices in groups.items()}
    costs = {group: {view: (rows[i]["full_costs"][view], rows[i]["costs"][view])
                    for view, i in support.items()} for group, support in supports.items()}
    totals = {group: tuple(sum(row[j] for row in views.values()) for j in (0, 1))
              for group, views in costs.items()}
    best_group = min(groups, key=lambda group: totals[group])
    rivals = [group for group in groups if group != best_group]
    if any(totals[group] == totals[best_group] for group in rivals):
        return decision | {"reason": "ambiguous-evidence"}
    if any(costs[best_group][view] >= costs[group][view]
           for group in rivals for view in evidence):
        return decision | {"reason": "conflicting-evidence"}
    best = min(groups[best_group], key=lambda i: (
        choices[i]["mode"] != "annotation", -choices[i]["start"],
        sum(rows[i]["full_costs"].values())))
    choice = choices[best]
    max_error = max(rows[i]["costs"][view] / max(1, len(keys[i]))
                    for view, i in supports[best_group].items())
    if max_error > .25:
        return decision | {"reason": "weak-evidence"}
    if is_whisper and choice["mode"] != "literal":
        normalized = unicodedata.normalize("NFKC", recognition_text)
        same_word = any(unicodedata.normalize("NFKC", row["base"]) in normalized
                        for row in choices)
        shared_reading = choice["mode"] == "annotation" and any(
            row["mode"] == "base" and row["start"] == choice["start"]
            and _key(row["reading"]) == _key(choice["reading"]) for row in choices)
        if same_word and not shared_reading:
            return decision | {"reason": "kanji-reading-not-observed"}
    return decision | {"status": "retained-sung" if choice["mode"] == "literal" else "resolved",
                       "reason": "whisper-agreement" if is_whisper else "acoustic-agreement",
                       "selected": best, "supporting_candidates": supports[best_group]}


def resolved_text(text: str, options: Sequence[dict], decisions: Sequence[dict]) -> str:
    """Apply supported choices without discarding any unresolved parenthesis."""
    for option, decision in reversed(tuple(zip(options, decisions, strict=True))):
        if decision["status"] not in {"resolved", "retained-sung"}:
            continue
        choice = option["choices"][decision["selected"]]
        item = option["span"]
        if decision["status"] == "retained-sung":
            # Keep both sung parts visible while fixing the selected base
            # pronunciation before the general dictionary selector runs.
            start = choice["start"]
            base_reading = choice["reading"][:-len(item.reading)]
            text = (text[:start] + f"｜{text[start:item.base_end]}《{base_reading}》"
                    + text[item.base_end:])
        else:
            text = _render(text, item, choice["start"], choice["reading"])
    return text


def selection_detail(text, resolved, options, decisions):
    return {"supplied_text": text, "resolved_text": resolved,
            "parenthetical_readings": [
                {"start": option["span"].start, "end": option["span"].end,
                 "supplied_reading": option["span"].reading, **decision}
                for option, decision in zip(options, decisions, strict=True)],
            "parenthetical_display_text": strip_ruby(resolved)}
