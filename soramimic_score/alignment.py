"""Known-Japanese-lyrics reading selection and non-destructive unit alignment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .ir import (Boundary, Evidence, IntermediateRepresentation, Mora, Phoneme,
                 Reading, SingingUnit, Utterance, VowelNucleus)
from .japanese import (LyricSpan, kana_to_moras, kana_to_syllables,
                       mora_distance, mora_vowel, phonemes_for_mora)


@dataclass(frozen=True)
class ObservedSingingUnit:
    """A CTC/forced-alignment observation, independent of canonical text."""
    moras: tuple[str, ...]
    consonant_start: Boundary
    vowel_start: Boundary | None
    end: Boundary
    confidence: float
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.moras or any(not kana_to_moras(value) for value in self.moras):
            raise ValueError("an observed unit needs explicit Japanese morae")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("observed confidence must be in [0, 1]")
        if self.end.time_sec < self.consonant_start.time_sec:
            raise ValueError("observed unit boundaries are reversed")


def _sequence_cost(expected: Sequence[str], observed: Sequence[str], *,
                   deletion_cost: float = 2.0) -> float:
    previous = [float(index) * 2.0 for index in range(len(observed) + 1)]
    for left in expected:
        current = [previous[0] + deletion_cost]
        for index, right in enumerate(observed, 1):
            current.append(min(previous[index] + deletion_cost, current[index - 1] + 2.0,
                               previous[index - 1] + mora_distance(left, right)))
        previous = current
    return previous[-1]


def _select_candidates(spans: Sequence[LyricSpan],
                       observations: Sequence[ObservedSingingUnit]) -> tuple[int, ...]:
    """Jointly select ordered candidates without letting spans reuse evidence."""
    if all(len(span.reading_candidates) == 1 for span in spans):
        return (0,) * len(spans)
    observed = tuple(mora for unit in observations for value in unit.moras
                     for mora in kana_to_moras(value))
    # observed-prefix length -> (cost, selected candidate indices)
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for span in spans:
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        maximum_length = max(len(kana_to_moras(item.kana)) for item in span.reading_candidates)
        for start, (prior_cost, choices) in states.items():
            # Permit a missing span (width 0), ordinary local insertions, and up
            # to twice the longest candidate for noisy CTC around this span.
            limit = min(len(observed), start + max(4, maximum_length * 2))
            for end in range(start, limit + 1):
                acoustic = observed[start:end]
                for index, candidate in enumerate(span.reading_candidates):
                    expected = kana_to_moras(candidate.kana)
                    # Priors resolve close hypotheses but one full vowel error
                    # (cost 3) outweighs their entire [0, 1] range.
                    cost = (prior_cost + _sequence_cost(expected, acoustic, deletion_cost=0.0)
                            + (1 - candidate.score) * 0.25)
                    proposal = (cost, choices + (index,))
                    current = next_states.get(end)
                    if current is None or proposal < current:
                        next_states[end] = proposal
        states = next_states
    if not states:
        raise ValueError("could not select pronunciation candidates")
    _, (_, choices) = min((cost + (len(observed) - end) * 2.0, value)
                          for end, value in states.items())
    return choices


def _align(expected: Sequence[str], observations: Sequence[ObservedSingingUnit]) -> tuple[int | None, ...]:
    """Map each canonical mora to an observation; one observation may own many morae."""
    expanded = [tuple(m for value in unit.moras for m in kana_to_moras(value)) for unit in observations]
    if tuple(expected) == tuple(m for group in expanded for m in group):
        return tuple(index for index, group in enumerate(expanded) for _ in group)
    inf = float("inf")
    costs = [[inf] * (len(observations) + 1) for _ in range(len(expected) + 1)]
    back: list[list[tuple[int, int, int | None] | None]] = [
        [None] * (len(observations) + 1) for _ in range(len(expected) + 1)
    ]
    costs[0][0] = 0.0
    for i in range(len(expected) + 1):
        for j in range(len(observations) + 1):
            base = costs[i][j]
            if not math.isfinite(base):
                continue
            if i < len(expected) and base + 2.0 < costs[i + 1][j]:
                costs[i + 1][j] = base + 2.0
                back[i + 1][j] = (i, j, None)
            if j < len(observations) and base + 2.0 < costs[i][j + 1]:
                costs[i][j + 1] = base + 2.0
                back[i][j + 1] = (i, j, -1)
            if j < len(observations):
                width = len(expanded[j])
                if width and i + width <= len(expected):
                    match = sum(mora_distance(expected[i + k], expanded[j][k]) for k in range(width))
                    if base + match < costs[i + width][j + 1]:
                        costs[i + width][j + 1] = base + match
                        back[i + width][j + 1] = (i, j, j)
    result: list[int | None] = [None] * len(expected)
    i, j = len(expected), len(observations)
    while i or j:
        step = back[i][j]
        if step is None:
            raise RuntimeError("failed to reconstruct lyric alignment")
        old_i, old_j, observation_index = step
        if observation_index is not None and observation_index >= 0:
            for index in range(old_i, i):
                result[index] = observation_index
        i, j = old_i, old_j
    return tuple(result)


def _role(symbol: str) -> str:
    if symbol in "aiueo":
        return "vowel"
    if symbol == "N":
        return "nasal"
    if symbol == "q":
        return "closure"
    return "consonant"


def build_known_lyrics_document(canonical_text: str, spans: Sequence[LyricSpan],
                                observations: Sequence[ObservedSingingUnit] = (),
                                evidence: Sequence[Evidence] = (), *,
                                observation_span_indices: Sequence[int] | None = None,
                                ) -> IntermediateRepresentation:
    """Build schema v1 without deleting canonical units absent from the audio."""
    if tuple(span.surface_span for span in spans) != tuple(sorted(span.surface_span for span in spans)):
        raise ValueError("lyric spans must be in surface order")
    if any(left.surface_span[1] > right.surface_span[0] for left, right in zip(spans, spans[1:])):
        raise ValueError("lyric spans must not overlap")
    if observation_span_indices is not None:
        owners = tuple(observation_span_indices)
        if (len(owners) != len(observations)
                or any(type(i) is not int or not 0 <= i < len(spans) for i in owners)
                or owners != tuple(sorted(owners))):
            raise ValueError("observation owners must identify ordered lyric spans")
        owned = [tuple(i for i, owner in enumerate(owners) if owner == index)
                 for index in range(len(spans))]
    evidence_ids = {item.id for item in evidence}
    if len(evidence_ids) != len(evidence):
        raise ValueError("evidence IDs must be unique")
    for observation in observations:
        if any(item not in evidence_ids for item in observation.evidence_ids):
            raise ValueError("an observation references missing evidence")
    for span in spans:
        if not span.reading_candidates:
            raise ValueError("every lyric span needs a reading candidate")
        for candidate in span.reading_candidates:
            if (not kana_to_moras(candidate.kana) or not candidate.source
                    or not math.isfinite(candidate.score) or not 0 <= candidate.score <= 1):
                raise ValueError("reading candidates need kana, source, and a score in [0, 1]")
            if any(item not in evidence_ids for item in candidate.evidence_ids):
                raise ValueError("a reading candidate references missing evidence")
    utterances: list[Utterance] = []
    readings: list[Reading] = []
    moras: list[Mora] = []
    selected_mora_ids: list[str] = []
    selected_mora_text: list[str] = []
    selected_reading_ranges: list[tuple[int, int]] = []
    selected_candidates = (_select_candidates(spans, observations)
                           if observation_span_indices is None else tuple(
                               _select_candidates((span,), tuple(observations[i] for i in indices))[0]
                               for span, indices in zip(spans, owned, strict=True)))

    for utterance_index, span in enumerate(spans):
        if canonical_text[slice(*span.surface_span)] != span.surface:
            raise ValueError("a lyric span does not match canonical_text")
        selected = selected_candidates[utterance_index]
        selected_start = len(selected_mora_ids)
        reading_ids: list[str] = []
        for candidate_index, candidate in enumerate(span.reading_candidates):
            reading_id = f"r{utterance_index}-{candidate_index}"
            reading_ids.append(reading_id)
            candidate_mora_ids: list[str] = []
            for mora_index, text in enumerate(kana_to_moras(candidate.kana)):
                mora_id = f"m{utterance_index}-{candidate_index}-{mora_index}"
                candidate_mora_ids.append(mora_id)
                moras.append(Mora(mora_id, reading_id, text, span.surface_span, (), ()))
                if candidate_index == selected:
                    selected_mora_ids.append(mora_id)
                    selected_mora_text.append(text)
            readings.append(Reading(reading_id, f"u{utterance_index}", candidate.kana,
                                    candidate.source, candidate.score, tuple(candidate_mora_ids),
                                    candidate.evidence_ids))
        utterances.append(Utterance(f"u{utterance_index}", span.surface, span.surface_span,
                                    tuple(reading_ids), reading_ids[selected]))
        selected_reading_ranges.append((selected_start, len(selected_mora_ids)))

    if observation_span_indices is None:
        mapping = _align(selected_mora_text, observations)
    else:
        # Audio adapters already identify the source line. Do not let a silent
        # repeated line steal identical mora observations from its neighbor.
        mapping = tuple(
            indices[item] if item is not None else None
            for (start, end), indices in zip(selected_reading_ranges, owned, strict=True)
            for item in _align(selected_mora_text[start:end],
                               tuple(observations[i] for i in indices))
        )
    phonemes: list[Phoneme] = []
    nuclei: list[VowelNucleus] = []
    units: list[SingingUnit] = []
    mora_phonemes: dict[str, list[str]] = {item.id: [] for item in moras}
    mora_units: dict[str, list[str]] = {item.id: [] for item in moras}
    mora_by_id = {item.id: item for item in moras}
    # Pronunciation structure belongs to every candidate, not only the selected
    # performance path. This keeps alternatives usable by later evidence passes.
    for reading in readings:
        candidate_previous_vowel: str | None = None
        for mora_id in reading.mora_ids:
            mora_text = mora_by_id[mora_id].text
            symbols = phonemes_for_mora(mora_text, candidate_previous_vowel)
            vowel = mora_vowel(mora_text, candidate_previous_vowel)
            if vowel is not None and vowel in "aiueo":
                candidate_previous_vowel = vowel
            for symbol_index, symbol in enumerate(symbols):
                phoneme_id = f"p-{mora_id}-{symbol_index}"
                phonemes.append(Phoneme(phoneme_id, symbol, _role(symbol), (mora_id,)))
                mora_phonemes[mora_id].append(phoneme_id)
    # Singing units are lexical syllables, not whatever width an acoustic
    # adapter happened to emit. This is the structural basis of the downstream
    # one-note/one-syllable constraint.
    groups: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for reading_start, reading_end in selected_reading_ranges:
        reading_moras = selected_mora_text[reading_start:reading_end]
        mora_offset = 0
        for syllable in kana_to_syllables("".join(reading_moras)):
            syllable_moras: list[int] = []
            consumed = ""
            while mora_offset < len(reading_moras) and len(consumed) < len(syllable):
                syllable_moras.append(reading_start + mora_offset)
                consumed += reading_moras[mora_offset]
                mora_offset += 1
            if consumed != syllable:
                raise ValueError("syllable segmentation does not preserve the selected reading")
            observed_indices = tuple(dict.fromkeys(
                mapping[index] for index in syllable_moras if mapping[index] is not None
            ))
            groups.append((tuple(syllable_moras), observed_indices))
        if mora_offset != len(reading_moras):
            raise ValueError("syllable segmentation did not consume the selected reading")

    for unit_index, (indices, observation_indices) in enumerate(groups):
        unit_id = f"s{unit_index}"
        unit_mora_ids = tuple(selected_mora_ids[index] for index in indices)
        unit_phoneme_ids = [phoneme_id for mora_id in unit_mora_ids
                            for phoneme_id in mora_phonemes[mora_id]]
        vowel_phonemes: list[tuple[str, str, int]] = []
        for phoneme_id in unit_phoneme_ids:
            phoneme = next(item for item in phonemes if item.id == phoneme_id)
            if phoneme.role in {"vowel", "nasal"}:
                mora_index = next(index for index in indices
                                  if selected_mora_ids[index] in phoneme.mora_ids)
                vowel_phonemes.append((phoneme.symbol, phoneme_id, mora_index))
        for mora_id in unit_mora_ids:
            mora_units[mora_id].append(unit_id)

        if not observation_indices:
            status, unit_confidence = "unobserved", 0.0
            consonant_start = unit_end = None
            unit_evidence: tuple[str, ...] = ()
        else:
            matched = tuple(observations[index] for index in observation_indices)
            coverage = len(tuple(index for index in indices if mapping[index] is not None)) / len(indices)
            unit_confidence = min(item.confidence for item in matched) * coverage
            status = "observed" if unit_confidence >= 0.5 else "weak"
            consonant_start = min((item.consonant_start for item in matched),
                                  key=lambda value: value.time_sec)
            unit_end = max((item.end for item in matched), key=lambda value: value.time_sec)
            unit_evidence = tuple(dict.fromkeys(
                evidence_id for item in matched for evidence_id in item.evidence_ids
            ))
        nucleus_ids_list: list[str] = []
        for nucleus_index, (symbol, phoneme_id, mora_index) in enumerate(vowel_phonemes):
            nucleus_id = f"v{unit_index}-{nucleus_index}"
            nucleus_ids_list.append(nucleus_id)
            observation_index = mapping[mora_index]
            observation = (observations[observation_index]
                           if observation_index is not None else None)
            nucleus_start = observation.vowel_start if observation is not None else None
            nucleus_end = (observation.end
                           if observation is not None and nucleus_start is not None else None)
            nuclei.append(VowelNucleus(nucleus_id, unit_id, symbol, (phoneme_id,),
                                       nucleus_start, nucleus_end))
        nucleus_ids = tuple(nucleus_ids_list)
        units.append(SingingUnit(unit_id, unit_mora_ids, tuple(unit_phoneme_ids), nucleus_ids,
                                 status, unit_confidence, consonant_start, unit_end, unit_evidence))

    rebuilt_moras = tuple(Mora(item.id, item.reading_id, item.text, item.surface_span,
                               tuple(mora_phonemes[item.id]), tuple(mora_units[item.id]))
                          for item in moras)
    return IntermediateRepresentation(
        schema_version=1, canonical_text=canonical_text, utterances=tuple(utterances),
        readings=tuple(readings), moras=rebuilt_moras, phonemes=tuple(phonemes),
        vowel_nuclei=tuple(nuclei), singing_units=tuple(units), evidence=tuple(evidence),
    )
