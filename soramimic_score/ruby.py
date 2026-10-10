"""Video-style display ruby on non-kana runs of the selected pronunciation."""

from __future__ import annotations

import re

from .japanese import katakana, normalize_lyric_input


_KANA = frozenset(chr(code) for code in range(0x3041, 0x3097)) | frozenset(
    chr(code) for code in range(0x30A1, 0x30FB)
) | frozenset("ーゝゞヽヾ")
_SILENT = frozenset("・･＝=゠:： 　")
_VOWEL_ROWS = {
    "ア": "アカサタナハマヤラワガザダバパァャヮ",
    "イ": "イキシチニヒミリギジヂビピィ",
    "ウ": "ウクスツヌフムユルグズヅブプヴゥュ",
    "エ": "エケセテネヘメレゲゼデベペェ",
    "オ": "オコソトノホモヨロヲゴゾドボポォョ",
}
_VOWEL = {char: vowel for vowel, row in _VOWEL_ROWS.items() for char in row}


def _hiragana(text: str) -> str:
    return "".join(chr(ord(char) - 96) if "ァ" <= char <= "ヶ" else char for char in text)


def _normalize_long_vowels(text: str) -> str:
    output = []
    for char in text:
        previous = _VOWEL.get(output[-1]) if output else None
        if output and output[-1] != "ー" and (
            (char == "ウ" and previous in ("オ", "ウ"))
            or (char == "イ" and previous in ("エ", "イ"))
            or (char in "アイウエオ" and previous == char)
        ):
            char = "ー"
        output.append(char)
    return "".join(output)


def _runs(surface: str) -> list[tuple[int, int, bool]]:
    runs = []
    for index, char in enumerate(surface):
        is_kana = char in _KANA or char in _SILENT
        if runs and runs[-1][2] == is_kana:
            runs[-1] = (runs[-1][0], index + 1, is_kana)
        else:
            runs.append((index, index + 1, is_kana))
    return runs


def _token_segments(surface: str, reading: str) -> list[dict[str, str]]:
    """Split a word by kana/non-kana runs and put ruby on the latter."""
    plain = [{"text": surface, "reading": ""}]
    bare = "".join(char for char in surface if char not in _SILENT)
    if not reading or all(char in _KANA for char in bare):
        return plain
    if _normalize_long_vowels(katakana(bare)) == _normalize_long_vowels(katakana(reading)):
        return plain
    runs = _runs(surface)
    non_kana = [run for run in runs if not run[2]]
    has_anchor = any(any(char not in _SILENT for char in surface[start:end])
                     for start, end, is_kana in runs if is_kana)
    if len(non_kana) > 1 and not has_anchor:
        return [{"text": surface, "reading": _hiragana(reading)}]
    for normalize in (False, True):
        pattern = ""
        for start, end, is_kana in runs:
            if is_kana:
                literal = katakana("".join(char for char in surface[start:end]
                                     if char not in _SILENT))
                pattern += re.escape(_normalize_long_vowels(literal) if normalize else literal)
            else:
                pattern += "(.+?)"
        target = katakana(reading)
        match = re.fullmatch(pattern, _normalize_long_vowels(target) if normalize else target)
        if match is None:
            continue
        pieces = []
        group = 0
        for start, end, is_kana in runs:
            part = surface[start:end]
            if is_kana:
                pieces.append({"text": part, "reading": ""})
                continue
            group += 1
            first, last = match.span(group)
            pronunciation = reading[first:last]
            identical = (_normalize_long_vowels(katakana(part))
                         == _normalize_long_vowels(katakana(pronunciation)))
            pieces.append({"text": part,
                           "reading": "" if identical else _hiragana(pronunciation)})
        return pieces
    return [{"text": surface, "reading": _hiragana(reading)}]


def _readings_by_kana_anchors(tokens, readings: list[str], selected: str) -> list[str] | None:
    """Locate changed words between unchanged, uniquely placed kana tokens."""
    target = katakana(selected)
    anchors = []
    for index, token in enumerate(tokens):
        surface = token["surface_form"]
        pronunciation = katakana(readings[index])
        if (surface and pronunciation and target.count(pronunciation) == 1
                and all(char in _KANA or char in _SILENT for char in surface)):
            anchors.append((index, target.index(pronunciation), len(pronunciation)))
    result = readings.copy()
    previous_token = previous_pos = 0
    for index, position, length in [*anchors, (len(tokens), len(target), 0)]:
        if position < previous_pos:
            return None
        segment = target[previous_pos:position]
        changed = [i for i in range(previous_token, index) if readings[i]]
        if len(changed) == 1:
            result[changed[0]] = segment
        elif katakana("".join(readings[previous_token:index])) != segment:
            return None
        previous_token = index + 1
        previous_pos = position + length
    return result if katakana("".join(result)) == target else None


def ruby_segments(text: str, selected_kana: str) -> list[dict[str, str]]:
    """Keep selected readings while placing partial ruby like Video subtitles."""
    if not text:
        return []
    normalized = normalize_lyric_input(text)
    if normalized != text:
        # Changed character counts need an explicit token-to-original map.
        # Preserve the original display instead of guessing ruby boundaries.
        return [{"text": text, "reading": ""}]
    try:
        from soramimic_yomi import get_tokens
        tokens = get_tokens(text, apply_rules=True)
    except Exception:
        return [{"text": text, "reading": ""}]
    readings = [token["pronunciation"] for token in tokens]
    if "".join(token["surface_form"] for token in tokens) != text:
        return [{"text": text, "reading": ""}]
    if katakana("".join(readings)) != katakana(selected_kana):
        # Keep ruby when the selected singing pronunciation changes one token.
        # The surrounding dictionary readings anchor that token unambiguously.
        for index in range(len(tokens)):
            prefix = "".join(readings[:index])
            suffix = "".join(readings[index + 1:])
            if (katakana(selected_kana).startswith(katakana(prefix))
                    and katakana(selected_kana).endswith(katakana(suffix))):
                replacement = selected_kana[len(prefix):len(selected_kana) - len(suffix) if suffix else None]
                if replacement:
                    readings[index] = replacement
                    break
        if katakana("".join(readings)) != katakana(selected_kana):
            readings = _readings_by_kana_anchors(tokens, readings, selected_kana)
            if readings is None:
                return [{"text": text, "reading": ""}]
    if katakana("".join(readings)) == katakana(selected_kana):
        return [part for token, reading in zip(tokens, readings, strict=True)
                for part in _token_segments(token["surface_form"], reading)]
    return [{"text": text, "reading": ""}]
