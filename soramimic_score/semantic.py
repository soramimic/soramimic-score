"""Conservative checks for credit-like Whisper text in singing audio."""

from __future__ import annotations

from collections.abc import Sequence
import re
import threading
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .audio import LyricLine, MelodyNote


_LABEL = r"(?:作詞|作曲|編曲|原作|監督|制作|製作|出演|翻訳|歌唱|動画制作|イラスト)"
_VALUE = r"[0-9a-zA-Zぁ-んァ-ヶ一-龯々〆ヵヶー@._・]{1,32}"
_SINGLE = re.compile(
    rf"^\s*(?:{_LABEL}|サブタイトル)(?:担当|協力|提供|制作|作成)?"
    rf"(?:\s*[:：/／|｜]\s*|\s+)(?P<value>{_VALUE})\s*$"
)
_COMPOUND = re.compile(
    rf"^\s*{_LABEL}(?:\s*[・･/&＆,，、／|｜]\s*{_LABEL})+"
    rf"\s*(?:[:：+＋]|\s)\s*(?P<value>{_VALUE})\s*$"
)
_SUBTITLES = re.compile(r"^\s*(?:字幕(?:制作|製作|提供|翻訳)?|翻訳[・/ ]?字幕)(?:\s*[:：]\s*\S+)?\s*$")
_ENGLISH = re.compile(
    rf"^\s*(?:(?:lyrics|music|arrangement|vocal)\s*[:：]\s*{_VALUE}"
    rf"|subtitles?\s+by\s+{_VALUE})\s*$", re.I,
)
_CONTEXTUAL = re.compile(
    r"^\s*(?P<label>映像|歌)(?:担当|制作|作成)?"
    r"(?:\s*[:：/／|｜]\s*|\s+)"
    r"(?P<value>[0-9a-zA-Zぁ-んァ-ヶ一-龯々〆ヵヶー"
    r"@._・+＋#＃&＆*＊\-\s]{1,48}?)\s*$"
)
_ORGANIZATION = re.compile(r"(?:研究所|スタジオ|工房|プロジェクト|チーム|制作室|映像部)$")
_SOFT_TEMPLATES = (
    ("closing-greeting", re.compile(r"(?:お疲れさま|お疲れ様|おつかれさま)(?:です|でした)?")),
    ("viewing-thanks", re.compile(
        r"(?:最後まで)?ご視聴(?:いただき)?"
        r"(?:ありがとう(?:ございます|ございました)|感謝(?:します|いたします)|ください)")),
    ("channel-registration", re.compile(
        r"チャンネル登録(?:と高評価)?(?:を)?(?:よろしく)?"
        r"(?:お願いします|お願いいたします|ありがとう(?:ございます|ございました))")),
)
_STOCK_MEDIA = re.compile(
    r"(?:🐯?soundhodori사운드호돌이サウンドゥ?ホドリ|instagramtwitterホドリ)"
)
MIN_CTC_MEDIAN_SCORE = 0.00075
_tagger = None
_tagger_lock = threading.Lock()


def normalize_recognized_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(char for char in normalized
                   if unicodedata.category(char)[0] not in {"P", "Z"})


def non_lyric_template_family(text: str) -> str | None:
    """Classify only whole-line matches, as the Video semantic gate does."""
    normalized = normalize_recognized_text(text)
    if not normalized:
        return None
    if is_credit_hallucination(text) or re.fullmatch(_LABEL, normalized):
        return "credits"
    if _STOCK_MEDIA.fullmatch(normalized):
        return "stock-media-credit"
    for family, pattern in _SOFT_TEMPLATES:
        if pattern.fullmatch(normalized):
            return family
    return None


def _confirmed_credit_value(text: str) -> str | None:
    value = unicodedata.normalize("NFKC", text).strip()
    for pattern in (_SINGLE, _COMPOUND):
        match = pattern.fullmatch(value)
        if match:
            return normalize_recognized_text(match.group("value"))
    return None


def _entity_like(value: str) -> bool:
    compact = re.sub(r"\s+", "", value)
    if not compact or len(compact) > 32:
        return False
    global _tagger
    with _tagger_lock:
        if _tagger is None:
            try:
                import MeCab
                import unidic_lite
            except ImportError:
                return bool(_ORGANIZATION.search(compact) or re.search(r"[A-Za-z]", compact))
            _tagger = MeCab.Tagger("-d " + unidic_lite.DICDIR)
        node = _tagger.parseToNode(compact)
        parts = []
        while node:
            if node.surface:
                fields = node.feature.split(",")
                parts.append((fields[0], fields[1] if len(fields) > 1 else ""))
            node = node.next
    if any(major in {"動詞", "形容詞", "助詞", "助動詞", "代名詞"}
           for major, _ in parts):
        return False
    return (any(major == "名詞" and minor == "固有名詞" for major, minor in parts)
            or bool(_ORGANIZATION.search(compact))
            or bool(re.search(r"[A-Za-z]", compact)))


def contextual_non_lyric_template_families(lines: Sequence[LyricLine]) -> tuple[str | None, ...]:
    """Use a creator-like value and neighboring confirmed credits for soft labels."""
    families = [non_lyric_template_family(line.text) for line in lines]
    candidates = []
    for line in lines:
        match = _CONTEXTUAL.fullmatch(unicodedata.normalize("NFKC", line.text))
        candidates.append((match.group("label"), match.group("value").strip()) if match else None)
    confirmed = {value for line in lines
                 if (value := _confirmed_credit_value(line.text)) is not None}
    entity_like = [candidate is not None and
                   (_entity_like(candidate[1]) or
                    normalize_recognized_text(candidate[1]) in confirmed)
                   for candidate in candidates]
    for index, candidate in enumerate(candidates):
        if candidate is not None and candidate[0] == "映像" and entity_like[index]:
            families[index] = "credits"
    changed = True
    while changed:
        changed = False
        for index, candidate in enumerate(candidates):
            if candidate is None or families[index] is not None or not entity_like[index]:
                continue
            current = lines[index]
            adjacent = False
            if index:
                previous = lines[index - 1]
                adjacent = (families[index - 1] == "credits"
                            and previous.start_sec is not None and previous.end_sec is not None
                            and current.start_sec is not None and current.end_sec is not None
                            and current.start_sec >= previous.start_sec
                            and abs(current.start_sec - previous.end_sec) <= .5)
            if index + 1 < len(lines):
                next_line = lines[index + 1]
                adjacent |= (families[index + 1] == "credits"
                             and current.start_sec is not None and current.end_sec is not None
                             and next_line.start_sec is not None and next_line.end_sec is not None
                             and next_line.start_sec >= current.start_sec
                             and abs(next_line.start_sec - current.end_sec) <= .5)
            if adjacent:
                families[index] = "credits"
                changed = True
    return tuple(families)


def is_credit_hallucination(text: str) -> bool:
    """Match a whole credit template, leaving lyrics containing role words intact."""
    value = unicodedata.normalize("NFKC", text).strip()
    return bool(_SINGLE.fullmatch(value) or _COMPOUND.fullmatch(value)
                or _SUBTITLES.fullmatch(value) or _ENGLISH.fullmatch(value))


def credit_recovery_windows(line: LyricLine, notes: Sequence[MelodyNote], *,
                            template_family: str | None = None) -> tuple[tuple[float, float], ...]:
    """Find substantial singing islands inside a credit-like ASR segment."""
    if (template_family is None and non_lyric_template_family(line.text) is None
            or line.start_sec is None or line.end_sec is None):
        return ()
    intervals = sorted(
        (max(line.start_sec, note.start_sec), min(line.end_sec, note.end_sec))
        for note in notes
        if note.end_sec > line.start_sec and note.start_sec < line.end_sec
    )
    intervals = [(start, end) for start, end in intervals if end > start]
    if not intervals:
        return ()
    islands = []
    start, end = intervals[0]
    melody_seconds = end - start
    for note_start, note_end in intervals[1:]:
        if note_start - end <= 1.0:
            melody_seconds += max(0.0, note_end - max(note_start, end))
            end = max(end, note_end)
        else:
            if end - start >= 1.5 and melody_seconds >= .5:
                islands.append((start, end))
            start, end = note_start, note_end
            melody_seconds = end - start
    if end - start >= 1.5 and melody_seconds >= .5:
        islands.append((start, end))
    return tuple(islands)


def has_melodic_support(line: LyricLine, notes: Sequence[MelodyNote]) -> bool:
    if line.start_sec is None or line.end_sec is None:
        return False
    duration = max(0.0, line.end_sec - line.start_sec)
    normalized = unicodedata.normalize("NFKC", line.text).strip()
    required = min(duration * .25, len(normalized) * .05)
    overlap = sum(max(0.0, min(line.end_sec, note.end_sec)
                      - max(line.start_sec, note.start_sec)) for note in notes)
    return required > 0 and overlap >= required
