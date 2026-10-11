"""The single versioned JSON document emitted by Soramimic Score."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from os import PathLike
from pathlib import Path
from typing import Any, Mapping

from .ir import IntermediateRepresentation, ValidationError
from .realization import Realization, compile_realization


FORMAT_NAME = "soramimic-score"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ScoreDocument:
    """Portable observations and their compiled lyric-aware score.

    ``observations`` retains raw and derived note candidates, lyric structure,
    correspondence links, confidence, and provenance. ``score`` is the
    consumer-facing canonical/performed/synthesis view derived from it.
    """

    format: str
    schema_version: int
    observations: IntermediateRepresentation
    score: Realization

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.format != FORMAT_NAME:
            raise ValidationError(f"unsupported score format: {self.format!r}")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValidationError(
                f"unsupported score schema_version: {self.schema_version!r}"
            )
        self.observations.validate()
        self.score.validate()
        if self.score.canonical_text != self.observations.canonical_text:
            raise ValidationError("score canonical_text does not match observations")
        readings = {item.id: item for item in self.observations.readings}
        expected = [(item.id, item.surface, readings[item.selected_reading_id].kana,
                     readings[item.selected_reading_id].mora_ids)
                    for item in self.observations.utterances]
        actual = [(item.utterance_id, item.text, item.kana, item.mora_ids)
                  for item in self.score.canonical]
        if actual != expected:
            raise ValidationError("score canonical readings do not match observations")
        if self.score.evidence != self.observations.evidence:
            raise ValidationError("score evidence does not match observations")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return dumps(self, indent=indent)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ScoreDocument":
        if not isinstance(raw, Mapping):
            raise ValidationError("the score document root must be an object")
        expected = {"format", "schema_version", "observations", "score"}
        if set(raw) != expected:
            raise ValidationError(
                f"score document fields must be exactly {sorted(expected)}"
            )
        try:
            return cls(
                format=raw["format"],
                schema_version=raw["schema_version"],
                observations=IntermediateRepresentation.from_dict(raw["observations"]),
                score=Realization.from_dict(raw["score"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValidationError):
                raise
            raise ValidationError(f"invalid score document value: {exc}") from exc

    @classmethod
    def from_json(cls, value: str | bytes) -> "ScoreDocument":
        return loads(value)


def compile_score(
    observations: IntermediateRepresentation,
    **stage3_options: Any,
) -> ScoreDocument:
    """Decode fresh observations and return the canonical score document."""
    from .pipeline import run_stage3_document

    run = run_stage3_document(observations, **stage3_options)
    return ScoreDocument(FORMAT_NAME, SCHEMA_VERSION, run.document, run.realization)


def from_linked_observations(
    observations: IntermediateRepresentation,
) -> ScoreDocument:
    """Compile an already linked observation document without decoding again."""
    return ScoreDocument(
        FORMAT_NAME,
        SCHEMA_VERSION,
        observations,
        compile_realization(observations),
    )


def dumps(document: ScoreDocument, *, indent: int | None = 2) -> str:
    """Serialize deterministically as UTF-8-friendly JSON with a trailing newline."""
    document.validate()
    return json.dumps(
        document.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        indent=indent,
        sort_keys=True,
    ) + "\n"


def loads(value: str | bytes) -> ScoreDocument:
    """Parse and validate a score document without silent schema migration."""
    try:
        raw = json.loads(value)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValidationError("invalid score document JSON") from exc
    return ScoreDocument.from_dict(raw)


def dump(document: ScoreDocument, path: str | PathLike[str]) -> Path:
    """Write one canonical score JSON file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(dumps(document), encoding="utf-8")
    return destination


def load(path: str | PathLike[str]) -> ScoreDocument:
    """Load one canonical score JSON file."""
    return loads(Path(path).read_bytes())
