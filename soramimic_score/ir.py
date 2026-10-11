"""Version 1 JSON contract for lyric, pronunciation, and alignment evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
UNIT_STATUSES = frozenset({"observed", "weak", "unobserved"})
LINK_OPERATIONS = frozenset({
    "match", "melisma", "stack", "sokuon", "note_only", "unit_only", "rest",
})
PHONEME_ROLES = frozenset({"consonant", "vowel", "nasal", "closure"})

# A nonzero posterior is not automatically usable timing evidence. CTC can
# emit a numerically positive floor for a token peak far outside the performed
# phrase. Keep that observation in the IR for provenance, but do not let a
# catastrophic near-zero peak move note ownership or synthesis boundaries.
MINIMUM_USABLE_TIMING_CONFIDENCE = 1e-5


class ValidationError(ValueError):
    """The document is not a valid Soramimic Score observation document."""


@dataclass(frozen=True)
class Boundary:
    time_sec: float
    confidence: float
    evidence_ids: tuple[str, ...]


def has_usable_timing(boundary: Boundary | None) -> bool:
    return (boundary is not None
            and boundary.confidence >= MINIMUM_USABLE_TIMING_CONFIDENCE)


@dataclass(frozen=True)
class Evidence:
    id: str
    source: str
    kind: str
    confidence: float
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Utterance:
    id: str
    surface: str
    surface_span: tuple[int, int]
    reading_candidate_ids: tuple[str, ...]
    selected_reading_id: str


@dataclass(frozen=True)
class Reading:
    id: str
    utterance_id: str
    kana: str
    source: str
    score: float
    mora_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Mora:
    id: str
    reading_id: str
    text: str
    surface_span: tuple[int, int]
    phoneme_ids: tuple[str, ...]
    singing_unit_ids: tuple[str, ...]


@dataclass(frozen=True)
class Phoneme:
    id: str
    symbol: str
    role: str
    mora_ids: tuple[str, ...]


@dataclass(frozen=True)
class VowelNucleus:
    id: str
    singing_unit_id: str
    symbol: str
    phoneme_ids: tuple[str, ...]
    start: Boundary | None
    end: Boundary | None


@dataclass(frozen=True)
class SingingUnit:
    id: str
    mora_ids: tuple[str, ...]
    phoneme_ids: tuple[str, ...]
    vowel_nucleus_ids: tuple[str, ...]
    status: str
    confidence: float
    consonant_start: Boundary | None
    end: Boundary | None
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class NoteCandidate:
    id: str
    start_sec: float
    end_sec: float
    midi_pitch: int | None
    confidence: float
    sources: tuple[str, ...]
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class LinkCost:
    """Weighted, additive components explaining one correspondence decision."""

    ctc_anchor_timing: float = 0.0
    candidate_confidence: float = 0.0
    complexity: float = 0.0

    @property
    def total(self) -> float:
        return self.ctc_anchor_timing + self.candidate_confidence + self.complexity


LINK_COST_COMPONENTS = frozenset({
    "ctc_anchor_timing", "candidate_confidence", "complexity",
})


def _link_cost(value: Mapping[str, Any]) -> LinkCost:
    if not isinstance(value, Mapping) or set(value) != LINK_COST_COMPONENTS:
        raise ValidationError(
            f"link cost components must be exactly {sorted(LINK_COST_COMPONENTS)}"
        )
    return LinkCost(**{key: float(component) for key, component in value.items()})


@dataclass(frozen=True)
class Link:
    id: str
    singing_unit_ids: tuple[str, ...]
    note_candidate_ids: tuple[str, ...]
    operation: str
    cost: float
    confidence: float
    cost_components: LinkCost
    evidence_ids: tuple[str, ...] = ()


def _boundary(value: Mapping[str, Any] | None) -> Boundary | None:
    if value is None:
        return None
    return Boundary(float(value["time_sec"]), float(value["confidence"]),
                    tuple(value["evidence_ids"]))


def _span(value: Sequence[int]) -> tuple[int, int]:
    if len(value) != 2:
        raise ValidationError("a surface span must contain exactly two offsets")
    return int(value[0]), int(value[1])


@dataclass(frozen=True)
class IntermediateRepresentation:
    schema_version: int
    canonical_text: str
    utterances: tuple[Utterance, ...] = ()
    readings: tuple[Reading, ...] = ()
    moras: tuple[Mora, ...] = ()
    phonemes: tuple[Phoneme, ...] = ()
    vowel_nuclei: tuple[VowelNucleus, ...] = ()
    singing_units: tuple[SingingUnit, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    note_candidates: tuple[NoteCandidate, ...] = ()
    links: tuple[Link, ...] = ()

    def __post_init__(self) -> None:
        self.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, value: str | bytes) -> "IntermediateRepresentation":
        try:
            raw = json.loads(value)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValidationError("invalid JSON") from exc
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "IntermediateRepresentation":
        if not isinstance(raw, Mapping):
            raise ValidationError("the document root must be an object")
        expected = {
            "schema_version", "canonical_text", "utterances", "readings", "moras",
            "phonemes", "vowel_nuclei", "singing_units", "evidence",
            "note_candidates", "links",
        }
        if set(raw) != expected:
            raise ValidationError(f"document fields must be exactly {sorted(expected)}")
        try:
            document = cls(
                schema_version=raw["schema_version"],
                canonical_text=raw["canonical_text"],
                utterances=tuple(Utterance(x["id"], x["surface"], _span(x["surface_span"]),
                                           tuple(x["reading_candidate_ids"]), x["selected_reading_id"])
                                 for x in raw["utterances"]),
                readings=tuple(Reading(x["id"], x["utterance_id"], x["kana"], x["source"],
                                       float(x["score"]), tuple(x["mora_ids"]),
                                       tuple(x.get("evidence_ids", ())))
                               for x in raw["readings"]),
                moras=tuple(Mora(x["id"], x["reading_id"], x["text"], _span(x["surface_span"]),
                                 tuple(x["phoneme_ids"]), tuple(x["singing_unit_ids"]))
                            for x in raw["moras"]),
                phonemes=tuple(Phoneme(x["id"], x["symbol"], x["role"], tuple(x["mora_ids"]))
                               for x in raw["phonemes"]),
                vowel_nuclei=tuple(VowelNucleus(x["id"], x["singing_unit_id"], x["symbol"],
                                                tuple(x["phoneme_ids"]), _boundary(x["start"]),
                                                _boundary(x["end"]))
                                    for x in raw["vowel_nuclei"]),
                singing_units=tuple(SingingUnit(x["id"], tuple(x["mora_ids"]),
                                                tuple(x["phoneme_ids"]),
                                                tuple(x["vowel_nucleus_ids"]), x["status"],
                                                float(x["confidence"]),
                                                _boundary(x["consonant_start"]), _boundary(x["end"]),
                                                tuple(x.get("evidence_ids", ())))
                                    for x in raw["singing_units"]),
                evidence=tuple(Evidence(x["id"], x["source"], x["kind"], float(x["confidence"]),
                                        x.get("detail", {})) for x in raw["evidence"]),
                note_candidates=tuple(NoteCandidate(x["id"], float(x["start_sec"]),
                                                     float(x["end_sec"]), x["midi_pitch"],
                                                     float(x["confidence"]), tuple(x["sources"]),
                                                     tuple(x.get("evidence_ids", ())))
                                      for x in raw["note_candidates"]),
                links=tuple(Link(x["id"], tuple(x["singing_unit_ids"]),
                                 tuple(x["note_candidate_ids"]), x["operation"], float(x["cost"]),
                                 float(x["confidence"]), _link_cost(x["cost_components"]),
                                 tuple(x.get("evidence_ids", ())))
                            for x in raw["links"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValidationError):
                raise
            raise ValidationError(f"invalid document value: {exc}") from exc
        return document

    def validate(self) -> None:
        from .japanese import validate_reading
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValidationError(f"unsupported schema_version: {self.schema_version!r}")
        if not isinstance(self.canonical_text, str):
            raise ValidationError("canonical_text must be a string")

        collections = (self.utterances, self.readings, self.moras, self.phonemes,
                       self.vowel_nuclei, self.singing_units, self.evidence,
                       self.note_candidates, self.links)
        all_ids = [item.id for items in collections for item in items]
        if any(not isinstance(item_id, str) or not item_id for item_id in all_ids):
            raise ValidationError("every object needs a non-empty string ID")
        if len(all_ids) != len(set(all_ids)):
            raise ValidationError("IDs must be unique across the document")

        by_id = {item.id: item for items in collections for item in items}
        evidence_ids = {item.id for item in self.evidence}

        def refs(owner: str, values: Sequence[str], expected_type: type) -> None:
            if len(values) != len(set(values)):
                raise ValidationError(f"{owner} contains duplicate references")
            for value in values:
                if not isinstance(by_id.get(value), expected_type):
                    raise ValidationError(f"{owner} references missing or wrong-type ID {value!r}")

        def confidence(owner: str, value: float) -> None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValidationError(f"{owner} confidence must be finite and in [0, 1]")

        def evidence_refs(owner: str, values: Sequence[str]) -> None:
            if any(value not in evidence_ids for value in values):
                raise ValidationError(f"{owner} references missing evidence")

        def boundary(owner: str, value: Boundary | None) -> None:
            if value is None:
                return
            if not math.isfinite(value.time_sec) or value.time_sec < 0:
                raise ValidationError(f"{owner} has an invalid time")
            confidence(owner, value.confidence)
            if not value.evidence_ids:
                raise ValidationError(f"{owner} must retain boundary evidence")
            evidence_refs(owner, value.evidence_ids)

        for item in self.evidence:
            if not item.source or not item.kind:
                raise ValidationError("evidence source and kind must be non-empty")
            confidence(item.id, item.confidence)
            try:
                json.dumps(item.detail)
            except (TypeError, ValueError) as exc:
                raise ValidationError(f"{item.id} detail must be JSON-compatible") from exc

        for item in self.utterances:
            start, end = item.surface_span
            if not 0 <= start < end <= len(self.canonical_text):
                raise ValidationError(f"{item.id} has an invalid surface span")
            if self.canonical_text[start:end] != item.surface:
                raise ValidationError(f"{item.id} surface does not match canonical_text")
            refs(item.id, item.reading_candidate_ids, Reading)
            if item.selected_reading_id not in item.reading_candidate_ids:
                raise ValidationError(f"{item.id} selected reading is not a candidate")
            for reading_id in item.reading_candidate_ids:
                if by_id[reading_id].utterance_id != item.id:
                    raise ValidationError(f"{reading_id} belongs to another utterance")

        for item in self.readings:
            refs(item.id, item.mora_ids, Mora)
            if (not item.kana or not item.source or not math.isfinite(item.score)
                    or not 0 <= item.score <= 1):
                raise ValidationError(f"{item.id} has invalid reading metadata")
            try:
                if validate_reading(item.kana).kana != item.kana:
                    raise ValueError("reading is not normalized")
            except ValueError as exc:
                raise ValidationError(f"{item.id} has invalid kana: {exc}") from exc
            evidence_refs(item.id, item.evidence_ids)
            if any(by_id[mora_id].reading_id != item.id for mora_id in item.mora_ids):
                raise ValidationError(f"{item.id} contains a mora owned by another reading")
            if "".join(by_id[mora_id].text for mora_id in item.mora_ids) != item.kana:
                raise ValidationError(f"{item.id} mora sequence does not reconstruct kana")

        for item in self.moras:
            if not isinstance(by_id.get(item.reading_id), Reading) or not item.text:
                raise ValidationError(f"{item.id} has invalid ownership or text")
            try:
                if validate_reading(item.text).moras != (item.text,):
                    raise ValueError("text must be one normalized mora")
            except ValueError as exc:
                raise ValidationError(f"{item.id} has invalid mora: {exc}") from exc
            start, end = item.surface_span
            if not 0 <= start < end <= len(self.canonical_text):
                raise ValidationError(f"{item.id} has an invalid surface span")
            refs(item.id, item.phoneme_ids, Phoneme)
            refs(item.id, item.singing_unit_ids, SingingUnit)
            if any(item.id not in by_id[p].mora_ids for p in item.phoneme_ids):
                raise ValidationError(f"{item.id} phoneme relation is not reciprocal")
            if any(item.id not in by_id[s].mora_ids for s in item.singing_unit_ids):
                raise ValidationError(f"{item.id} singing-unit relation is not reciprocal")

        for item in self.phonemes:
            if item.role not in PHONEME_ROLES or not item.symbol:
                raise ValidationError(f"{item.id} has invalid phoneme data")
            refs(item.id, item.mora_ids, Mora)
            if any(item.id not in by_id[m].phoneme_ids for m in item.mora_ids):
                raise ValidationError(f"{item.id} mora relation is not reciprocal")

        for item in self.singing_units:
            refs(item.id, item.mora_ids, Mora)
            refs(item.id, item.phoneme_ids, Phoneme)
            refs(item.id, item.vowel_nucleus_ids, VowelNucleus)
            if not item.mora_ids or item.status not in UNIT_STATUSES:
                raise ValidationError(f"{item.id} has invalid unit membership or status")
            confidence(item.id, item.confidence)
            evidence_refs(item.id, item.evidence_ids)
            boundary(f"{item.id}.consonant_start", item.consonant_start)
            boundary(f"{item.id}.end", item.end)
            if item.status == "unobserved" and (item.confidence != 0 or item.consonant_start or item.end or item.evidence_ids):
                raise ValidationError(f"{item.id} unobserved units cannot claim acoustic evidence")
            if item.status != "unobserved" and (item.consonant_start is None or item.end is None):
                raise ValidationError(f"{item.id} observed/weak units need explicit boundaries")
            if item.consonant_start and item.end and item.end.time_sec < item.consonant_start.time_sec:
                raise ValidationError(f"{item.id} boundaries are reversed")

        for item in self.vowel_nuclei:
            if not isinstance(by_id.get(item.singing_unit_id), SingingUnit) or not item.symbol:
                raise ValidationError(f"{item.id} has invalid vowel-nucleus ownership")
            refs(item.id, item.phoneme_ids, Phoneme)
            boundary(f"{item.id}.start", item.start)
            boundary(f"{item.id}.end", item.end)
            unit = by_id[item.singing_unit_id]
            if item.id not in unit.vowel_nucleus_ids:
                raise ValidationError(f"{item.id} relation is not reciprocal")
            if unit.status == "unobserved" and (item.start is not None or item.end is not None):
                raise ValidationError(f"{item.id} cannot time an unobserved nucleus")
            if (item.start is None) != (item.end is None):
                raise ValidationError(f"{item.id} must have both or neither nucleus boundary")
            if item.start and item.end and item.end.time_sec < item.start.time_sec:
                raise ValidationError(f"{item.id} nucleus boundaries are reversed")

        for item in self.note_candidates:
            if (not math.isfinite(item.start_sec + item.end_sec) or item.start_sec < 0
                    or item.end_sec <= item.start_sec or not item.sources):
                raise ValidationError(f"{item.id} has invalid note timing or provenance")
            if item.midi_pitch is not None and (type(item.midi_pitch) is not int or not 0 <= item.midi_pitch <= 127):
                raise ValidationError(f"{item.id} has invalid MIDI pitch")
            confidence(item.id, item.confidence)
            evidence_refs(item.id, item.evidence_ids)

        for item in self.links:
            refs(item.id, item.singing_unit_ids, SingingUnit)
            refs(item.id, item.note_candidate_ids, NoteCandidate)
            if item.operation not in LINK_OPERATIONS or not math.isfinite(item.cost) or item.cost < 0:
                raise ValidationError(f"{item.id} has invalid link operation or cost")
            shape = (len(item.singing_unit_ids), len(item.note_candidate_ids))
            valid_shape = {
                "match": shape == (1, 1),
                "melisma": shape[0] == 1 and shape[1] > 1,
                "stack": shape[0] > 1 and shape[1] == 1,
                "sokuon": shape[0] == 2 and shape[1] >= 1,
                "note_only": shape == (0, 1),
                "unit_only": shape == (1, 0),
                "rest": shape == (0, 1),
            }[item.operation]
            if not valid_shape:
                raise ValidationError(f"{item.id} has invalid membership for {item.operation}")
            if item.operation == "sokuon":
                left, closure = (by_id[value] for value in item.singing_unit_ids)
                left_symbols = tuple(by_id[value].symbol for value in left.phoneme_ids)
                closure_symbols = tuple(by_id[value].symbol for value in closure.phoneme_ids)
                if (not left_symbols or all(value == "q" for value in left_symbols)
                        or not closure_symbols or any(value != "q" for value in closure_symbols)):
                    raise ValidationError("sokuon must attach a q-only unit to its predecessor")
                if any(by_id[value].midi_pitch is None for value in item.note_candidate_ids):
                    raise ValidationError("sokuon must reference pitched candidates")
            if item.operation == "rest" and by_id[item.note_candidate_ids[0]].midi_pitch is not None:
                raise ValidationError(f"{item.id} rest must reference an unpitched candidate")
            if item.operation == "note_only" and by_id[item.note_candidate_ids[0]].midi_pitch is None:
                raise ValidationError(f"{item.id} note_only must reference a pitched candidate")
            confidence(item.id, item.confidence)
            evidence_refs(item.id, item.evidence_ids)
            if not isinstance(item.cost_components, LinkCost):
                raise ValidationError(f"{item.id} must have typed link cost components")
            component_values = asdict(item.cost_components)
            if (any(isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0
                    for value in component_values.values())
                    or not math.isclose(item.cost, item.cost_components.total,
                                        rel_tol=1e-9, abs_tol=1e-9)):
                raise ValidationError(f"{item.id} has invalid link cost components")
