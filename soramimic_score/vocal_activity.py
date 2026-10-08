"""Vocal-stem support and fully measured silence for lyric decisions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence


@dataclass(frozen=True)
class VocalActivity:
    percentile_dbfs: float
    relative_db: float
    active_frame_ratio: float
    supported: bool
    # Unlike the song-relative support gate, this confirms every sampled frame
    # is below the absolute -70 dBFS floor. None means incomplete/unknown coverage.
    silence_confirmed: bool | None = None


def measure_vocal_activity(vocals_path: Path,
                           windows: Sequence[tuple[float, float]]) -> tuple[VocalActivity, ...]:
    """Match Video's 50 ms, 90th-percentile, song-relative stem gate."""
    import numpy as np
    import soundfile as sf

    samples, sample_rate = sf.read(vocals_path, dtype="float32", always_2d=True)
    if sample_rate <= 0 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("vocal stem must contain finite samples")
    frame_samples = max(1, round(sample_rate * .05))

    def frame_dbfs(segment):
        levels = []
        for start in range(0, len(segment), frame_samples):
            frame = segment[start:start + frame_samples]
            rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
            levels.append(max(-240., 20. * np.log10(max(rms, 1e-12))))
        return np.asarray(levels, dtype=np.float64)

    song_levels = frame_dbfs(samples)
    active = song_levels[song_levels >= -70.]
    reference = float(np.percentile(active if len(active) else song_levels, 90))
    floor = max(-70., reference - 30.)
    evidence = []
    for start_sec, end_sec in windows:
        start = max(0, min(len(samples), round(start_sec * sample_rate)))
        end = max(start, min(len(samples), round(end_sec * sample_rate)))
        levels = frame_dbfs(samples[start:end])
        percentile = float(np.percentile(levels, 90)) if len(levels) else -240.
        covered = bool(len(levels) and 0 <= start_sec < end_sec
                       <= (len(samples) + 1) / sample_rate)
        evidence.append(VocalActivity(percentile, percentile - reference,
                                      float(np.mean(levels >= floor)) if len(levels) else 0.,
                                      percentile >= floor,
                                      bool(np.all(levels < -70.)) if covered else None))
    return tuple(evidence)
