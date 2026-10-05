"""Recover repeated pronunciations from the voice, independently of ASR words.

A repeated spectrogram alone can be accompaniment. A similar vowel string alone
can be a different sentence. Require both, including an adjacent acoustic pair,
before using a pronunciation observed elsewhere in the same recording.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
import math

from .japanese import mora_vowel
from .phonetic_fallback import PhoneticMora

VOWELS = frozenset("aiueo")
FRAME_SEC = .02


@dataclass(frozen=True)
class AcousticOccurrence:
    start_sec: float
    end_sec: float
    similarity: float
    observed_kana: str


@dataclass(frozen=True)
class AcousticRepetition:
    kana: str
    template_start_sec: float
    period_sec: float
    occurrences: tuple[AcousticOccurrence, ...]
    mora_offsets_sec: tuple[float, ...] = ()
    seed_start_sec: float | None = None


def pronunciation_agreement(first, second, period):
    """Compare independently decoded morae at corresponding acoustic phases."""
    used = set()
    matched = equal = 0
    tolerance = min(.18, period * .12)
    for time, kana in first:
        choices = [(abs(other_time - time), index, other_kana)
                   for index, (other_time, other_kana) in enumerate(second)
                   if index not in used and abs(other_time - time) < tolerance]
        if not choices:
            continue
        _, index, other = min(choices)
        used.add(index)
        matched += 1
        vowel = mora_vowel(kana)
        equal += int(vowel == mora_vowel(other) if vowel in VOWELS else kana == other)
    return equal / max(1, matched), matched / max(1, min(len(first), len(second))), matched


def consensus_pronunciation(occurrences, period):
    """Choose one observed copy by vowel agreement with the other copies.

    Splicing phase-wise votes can turn two alternative decodes of one vowel
    into two syllables. A medoid keeps observed vowel positions; consonant
    voting cannot add a syllable.
    """
    if len(occurrences) < 2:
        return ()
    def score(candidate):
        values = []
        for other in occurrences:
            agreement, _, matched = pronunciation_agreement(candidate, other, period)
            values.append(agreement * matched / max(len(candidate), len(other), 1))
        return sum(values) / len(values)
    medoid = max(occurrences, key=score)
    result = []
    for time, kana in medoid:
        vowel = mora_vowel(kana)
        alternatives = []
        for other in occurrences:
            nearby = [(abs(t - time), k) for t, k in other
                      if abs(t - time) < min(.12, period * .08)
                      and mora_vowel(k) == vowel]
            if nearby:
                alternatives.append(min(nearby)[1])
        result.append((time, Counter(alternatives).most_common(1)[0][0] if alternatives else kana))
    return tuple(result)


def acoustic_features(path):
    """Whiten the local spectral envelope so a sustained tone is not a motif."""
    import librosa
    import numpy as np
    from scipy.ndimage import uniform_filter1d

    samples, rate = librosa.load(str(path), sr=16000, mono=True)
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError("acoustic repetition needs finite audio")
    mel = librosa.feature.melspectrogram(y=samples, sr=rate, n_fft=512,
                                         hop_length=320, n_mels=64)
    features = librosa.power_to_db(mel, ref=np.max)
    features -= uniform_filter1d(features, size=101, axis=1)
    features /= np.maximum(np.linalg.norm(features, axis=0), 1e-8)
    return features, len(samples) / rate


def find_acoustic_repetitions(path, events, notes=()):
    features, duration = acoustic_features(path)
    return repetitions_from_features(features, duration, events, notes)


def repetitions_from_features(features, duration, events, notes=()):
    """Find an adjacent seed, then confirm further copies against its audio."""
    import numpy as np
    from scipy.signal import fftconvolve, find_peaks

    events = tuple(sorted(set(events), key=lambda e: (e.start_sec, e.end_sec)))
    if any(not isinstance(e, PhoneticMora) or not math.isfinite(e.start_sec + e.end_sec)
           or not 0 <= e.start_sec < e.end_sec <= duration + .05 for e in events):
        raise ValueError("acoustic repetition needs finite timed phonetic evidence")
    if any(a.end_sec > b.start_sec for a, b in zip(events, events[1:])):
        return ()
    times = [e.start_sec for e in events]

    def units(start, period):
        lo, hi = bisect_left(times, start - .06), bisect_left(times, start + period - .06)
        return tuple((e.start_sec - start, e.kana) for e in events[lo:hi])

    def repeated_note_onsets(first, second, period):
        one = [n.start_sec - first for n in notes if first - .1 <= n.start_sec < first + period - .1]
        two = [n.start_sec - second for n in notes if second - .1 <= n.start_sec < second + period - .1]
        return (min(len(one), len(two)) >= 4 and
                sum(any(abs(a - b) <= .12 for a in one) for b in two) / len(two) >= .8)

    prefixes = {}

    def similarity(start, lag):
        if lag not in prefixes:
            shifted = np.sum(features[:, :-lag] * features[:, lag:], axis=0)
            prefixes[lag] = np.r_[0., np.cumsum(shifted, dtype=np.float64)]
        a = round(start / FRAME_SEC)
        prefix = prefixes[lag]
        return float((prefix[a + lag] - prefix[a]) / lag) if 0 <= a < a + lag < len(prefix) else -1.

    seeds = []
    for index, event in enumerate(events):
        if mora_vowel(event.kana) not in VOWELS:
            continue
        start = event.start_sec
        last = bisect_left(times, start + 6.01)
        for other in events[index + 1:last]:
            period = other.start_sec - start
            if period < 1. or mora_vowel(event.kana) != mora_vowel(other.kana):
                continue
            # Token onsets can move a frame or two while the musical period
            # stays fixed. Refine the lag against audio rather than token count.
            center = round(period / FRAME_SEC)
            score, lag = max((similarity(start, lag), lag)
                             for lag in range(max(50, center - 3), min(300, center + 3) + 1))
            if score < .60:
                continue
            period = lag * FRAME_SEC
            first, second = units(start, period), units(start + period, period)
            agreement, coverage, matched = pronunciation_agreement(first, second, period)
            vowels = {mora_vowel(k) for _, k in first + second} & VOWELS
            if (min(len(first), len(second)) < 4 or len(vowels) < 2
                    or agreement < .65 or coverage < .65 or matched < 3):
                continue
            seeds.append((start, period, score))

    result = []
    occupied = []
    for start, period, seed_score in sorted(seeds, key=lambda s: (s[0], -s[2])):
        if any(a - .1 <= start <= b + .1 for a, b in occupied):
            continue
        size = round(period / FRAME_SEC)
        frame = round(start / FRAME_SEC)
        template = features[:, frame:frame + size]
        if template.shape[1] != size:
            continue
        scores = fftconvolve(features, template[:, ::-1], mode="valid", axes=1).sum(axis=0) / size
        second = features[:, frame + size:frame + 2 * size]
        if second.shape[1] == size:
            scores = np.maximum(scores, fftconvolve(
                features, second[:, ::-1], mode="valid", axes=1).sum(axis=0) / size)
        peaks, _ = find_peaks(scores, distance=max(1, round(size * .75)), height=.45)
        original = units(start, period)
        found = []
        for peak in peaks:
            onset = float(peak * FRAME_SEC)
            if onset + period > duration:
                continue
            observed = units(onset, period)
            agreement, coverage, matched = pronunciation_agreement(original, observed, period)
            phonetic_match = len(observed) >= 3 and agreement >= .4 and coverage >= .5 and matched >= 3
            if ((not phonetic_match and scores[peak] < .65)
                    or any(onset < b and onset + period > a for a, b in occupied)):
                continue
            found.append((onset, float(scores[peak]), observed))
        if not any(abs(a[0] + period - b[0]) <= .10 for a, b in zip(found, found[1:])):
            continue
        # A chorus may change its final copy. Only extend a measured adjacent
        # run when that next copy has both a local audio peak and timed vowels.
        additions = []
        for anchor in found:
            if anchor[1] < .50:
                continue
            for direction in (-1, 1):
                expected = anchor[0] + direction * period
                if any(abs(item[0] - expected) < period * .5 for item in found + additions):
                    continue
                lo = max(0, round((expected - .08) / FRAME_SEC))
                hi = min(len(scores), round((expected + .08) / FRAME_SEC) + 1)
                if hi <= lo:
                    continue
                peak = lo + int(np.argmax(scores[lo:hi]))
                onset = peak * FRAME_SEC
                observed = units(onset, period)
                agreement, coverage, matched = pronunciation_agreement(original, observed, period)
                two_previous = any(abs(other[0] + direction * period - anchor[0]) <= .1
                                   and other[1] >= .5 for other in found)
                rhythm_supported = (two_previous and scores[peak] >= .18
                                    and agreement >= .6 and coverage >= .8 and matched >= 4
                                    and repeated_note_onsets(anchor[0], onset, period))
                if (scores[peak] >= .50 or scores[peak] >= .35
                        and agreement >= .5 and coverage >= .5 and matched >= 3
                        or rhythm_supported):
                    if not any(onset < b and onset + period > a for a, b in occupied):
                        additions.append((onset, float(scores[peak]), observed))
        found = sorted(found + additions)
        # With no reliable lexical phrase, two similar sentences are not
        # enough. A family needs repeated evidence across at least four copies.
        if len(found) < 4:
            continue
        strong = [item[2] for item in found if item[1] >= .55 and len(item[2]) >= 4]
        if len(strong) < 4:
            continue
        pronunciation = consensus_pronunciation(strong, period)
        if not 5 <= len(pronunciation) <= 24:
            continue
        kana = "".join(k for _, k in pronunciation)
        copies = tuple(AcousticOccurrence(
            max(0., onset - .06), min(duration, onset + period - .06,
                                     found[index + 1][0] - .06 if index + 1 < len(found) else duration),
            max(-1., min(1., score)),
            "".join(k for _, k in observed),
        ) for index, (onset, score, observed) in enumerate(found))
        phases = tuple(time for time, _ in pronunciation)
        template_start = next(onset for onset, _, observed in found
                              if tuple(time for time, _ in observed) == phases)
        offsets = tuple(max(0., time + .06) for time in phases)
        result.append(AcousticRepetition(kana, template_start, period, copies, offsets, start))
        occupied.extend((c.start_sec, c.end_sec) for c in copies)
    return tuple(result)
