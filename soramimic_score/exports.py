"""Interoperable exports derived only from a validated Score document."""

from __future__ import annotations

from collections import defaultdict
from xml.etree import ElementTree as ET

from .document import ScoreDocument, dumps


def _stamp(seconds: float, *, srt: bool = False) -> str:
    millis = max(0, round(seconds * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return (f"{hours:02}:{minutes:02}:{secs:02},{ms:03}" if srt
            else f"{minutes + hours * 60:02}:{secs:02}.{ms // 10:02}")


def lyric_cues(document: ScoreDocument) -> list[tuple[float, float, str]]:
    by_line: dict[str, list] = defaultdict(list)
    for slot in document.score.synthesis_plan:
        by_line[slot.utterance_id].append(slot)
    cues = []
    for line in document.score.canonical:
        slots = by_line[line.utterance_id]
        if slots:
            cues.append((min(s.start_sec for s in slots), max(s.end_sec for s in slots), line.text))
    return sorted(cues)


def export_srt(document: ScoreDocument) -> bytes:
    return ("\n\n".join(
        f"{i}\n{_stamp(start, srt=True)} --> {_stamp(end, srt=True)}\n{text}"
        for i, (start, end, text) in enumerate(lyric_cues(document), 1)
    ) + "\n").encode("utf-8")


def export_lrc(document: ScoreDocument) -> bytes:
    return ("\n".join(f"[{_stamp(start)}]{text}" for start, _, text in lyric_cues(document))
            + "\n").encode("utf-8")


def export_midi(document: ScoreDocument) -> bytes:
    """Type 0 MIDI with one lyric event per note onset, at 120 BPM."""
    def vlq(value: int) -> bytes:
        out = [value & 0x7f]
        value >>= 7
        while value:
            out.insert(0, 0x80 | (value & 0x7f))
            value >>= 7
        return bytes(out)

    ticks_per_second = 960  # 480 PPQ, 500000 microseconds per quarter.
    events: list[tuple[int, int, bytes]] = []
    for slot in document.score.synthesis_plan:
        start = round(slot.start_sec * ticks_per_second)
        end = max(start + 1, round(slot.end_sec * ticks_per_second))
        events.append((start, 2, bytes((0x90, slot.midi_pitch, 80))))
        events.append((end, 0, bytes((0x80, slot.midi_pitch, 0))))
        if "spoken" in slot.pitch_sources:
            marker = b"spoken: neutral render pitch, not measured melody"
            events.append((start, 1, b"\xff\x01" + vlq(len(marker)) + marker))
        if not slot.continuation:
            lyric = slot.kana.encode("utf-8")
            events.append((start, 1, b"\xff\x05" + vlq(len(lyric)) + lyric))
    track = bytearray(b"\x00\xff\x51\x03\x07\xa1\x20")
    previous = 0
    for tick, _, event in sorted(events):
        track += vlq(max(0, tick - previous)) + event
        previous = tick
    track += b"\x00\xff\x2f\x00"
    return (b"MThd" + (6).to_bytes(4, "big") + b"\x00\x00\x00\x01\x01\xe0"
            + b"MTrk" + len(track).to_bytes(4, "big") + track)


def export_musicxml(document: ScoreDocument) -> bytes:
    """A 60 BPM score with four-beat measures, rests, and tied crossings."""
    root = ET.Element("score-partwise", version="4.0")
    ET.SubElement(root, "work")
    parts = ET.SubElement(root, "part-list")
    score_part = ET.SubElement(parts, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Vocal"
    part = ET.SubElement(root, "part", id="P1")
    measure_number = 0
    measure = None
    tempo = None

    def next_measure():
        nonlocal measure_number, measure, tempo
        measure_number += 1
        measure = ET.SubElement(part, "measure", number=str(measure_number))
        if measure_number == 1:
            attributes = ET.SubElement(measure, "attributes")
            ET.SubElement(attributes, "divisions").text = "1000"
            signature = ET.SubElement(attributes, "time")
            ET.SubElement(signature, "beats").text = "4"
            ET.SubElement(signature, "beat-type").text = "4"
            clef = ET.SubElement(attributes, "clef")
            ET.SubElement(clef, "sign").text = "G"
            ET.SubElement(clef, "line").text = "2"
            tempo = ET.SubElement(measure, "direction")

    next_measure()
    sound = ET.SubElement(tempo, "sound")
    sound.set("tempo", "60")
    cursor = 0

    def add_note(duration: int, pitch: int | None, lyric: str = "",
                 *, tie_start: bool = False, tie_stop: bool = False,
                 spoken: bool = False) -> None:
        note = ET.SubElement(measure, "note")
        if pitch is None:
            ET.SubElement(note, "rest")
        elif spoken:
            p = ET.SubElement(note, "unpitched")
            ET.SubElement(p, "display-step").text = "C"
            ET.SubElement(p, "display-octave").text = "4"
        else:
            p = ET.SubElement(note, "pitch")
            names = ("C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B")
            ET.SubElement(p, "step").text = names[pitch % 12]
            if pitch % 12 in (1, 3, 6, 8, 10):
                ET.SubElement(p, "alter").text = "1"
            ET.SubElement(p, "octave").text = str(pitch // 12 - 1)
        ET.SubElement(note, "duration").text = str(duration)
        if tie_stop:
            ET.SubElement(note, "tie", type="stop")
        if tie_start:
            ET.SubElement(note, "tie", type="start")
        if spoken:
            ET.SubElement(note, "notehead").text = "x"
        if tie_start or tie_stop:
            notations = ET.SubElement(note, "notations")
            if tie_stop:
                ET.SubElement(notations, "tied", type="stop")
            if tie_start:
                ET.SubElement(notations, "tied", type="start")
        if lyric:
            ET.SubElement(ET.SubElement(note, "lyric"), "text").text = lyric

    def append_interval(end: int, pitch: int | None, lyric: str = "", *, spoken=False) -> None:
        nonlocal cursor
        first_piece = True
        while cursor < end:
            measure_end = ((cursor // 4000) + 1) * 4000
            piece_end = min(end, measure_end)
            more = piece_end < end
            add_note(piece_end - cursor, pitch, lyric if first_piece else "",
                     tie_start=bool(pitch is not None and more),
                     tie_stop=bool(pitch is not None and not first_piece), spoken=spoken)
            cursor = piece_end
            first_piece = False
            if cursor == measure_end:
                next_measure()

    for slot in document.score.synthesis_plan:
        start = round(slot.start_sec * 1000)
        end = max(start + 1, round(slot.end_sec * 1000))
        if start > cursor:
            append_interval(start, None)
        append_interval(end, slot.midi_pitch, "" if slot.continuation else slot.kana,
                        spoken="spoken" in slot.pitch_sources)
    if not list(measure.findall("note")) and measure_number > 1:
        part.remove(measure)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


EXPORTS = {
    "json": ("application/json; charset=utf-8", lambda d: dumps(d).encode("utf-8")),
    "mid": ("audio/midi", export_midi),
    "musicxml": ("application/vnd.recordare.musicxml+xml", export_musicxml),
    "srt": ("application/x-subrip; charset=utf-8", export_srt),
    "lrc": ("text/plain; charset=utf-8", export_lrc),
}
