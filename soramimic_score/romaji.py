"""Optional local RomajiASR ONNX inference with acoustic context at gaps."""
from __future__ import annotations

import json
from pathlib import Path

from .phonetic_fallback import PhoneticMora, romaji_to_kana


def decode_romaji_ids(ids, reverse, blank, origin, duration):
    """Decode CTC token runs without inventing phonemes for unknown IDs."""
    runs = []
    index = 0
    while index < len(ids):
        token_id = int(ids[index])
        finish = index + 1
        while finish < len(ids) and int(ids[finish]) == token_id:
            finish += 1
        onset = origin + index * .02
        offset = min(origin + duration, origin + finish * .02)
        if token_id != blank and offset > onset:
            runs.append((reverse.get(token_id, ""), onset, offset))
        index = finish
    result = []
    index = 0
    while index < len(runs):
        token, onset, offset = runs[index]
        kana = romaji_to_kana(token)
        if kana is None and index + 1 < len(runs):
            following, _, next_end = runs[index + 1]
            if following in {"a", "i", "u", "e", "o"}:
                kana = romaji_to_kana(token + following)
                if kana is not None:
                    offset = next_end
                    index += 1
        if kana is not None:
            result.append(PhoneticMora(kana, onset, offset))
        index += 1
    return tuple(result)


def transcribe_romaji(path: Path, windows, model_dir: Path):
    import librosa
    import numpy as np
    import onnxruntime as ort

    audio, rate = librosa.load(str(path), sr=16000, mono=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("RomajiASR needs finite audio samples")
    directory = Path(model_dir)
    vocab = json.loads((directory / "phoneme_vocab.json").read_text())
    reverse = {index: token for token, index in vocab.items()}
    blank = vocab.get("<blank>", vocab.get("PAD", 0))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(directory / "model.onnx"), options,
                                   providers=["CPUExecutionProvider"])
    inputs = {item.name: item for item in session.get_inputs()}
    meta_path = directory / "model.meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get("sample_rate", rate) != rate:
        raise ValueError("RomajiASR model must use 16000 Hz")
    shape = inputs["input_values"].shape
    batch = shape[0] if isinstance(shape[0], int) else 1
    fixed = shape[1] if isinstance(shape[1], int) else None
    context = min(20., fixed / rate) if fixed else 20.
    duration = len(audio) / rate
    decoded = {}
    events = []
    for start, end in windows:
        if not 0 <= start < end <= duration + .05:
            raise ValueError("RomajiASR window is outside the audio")
        if end - start > context:
            chunks = [(a / rate, min(end, a / rate + context))
                      for a in range(int(start * rate), int(end * rate), int(context * rate))]
        else:
            chunks = [(start, end)]
        for a, b in chunks:
            first = int(max(0., min((a + b - context) / 2,
                                   max(0., duration - context))) * rate)
            last = min(len(audio), first + int(context * rate))
            key = first, last
            if key not in decoded:
                count = fixed or last - first
                values = np.zeros((batch, count), dtype=np.float32)
                values[:, :last - first] = audio[first:last]
                dtype = np.float16 if inputs["input_values"].type == "tensor(float16)" else np.float32
                feeds = {"input_values": values.astype(dtype)}
                if "attention_mask" in inputs:
                    mask = np.zeros((batch, count), dtype=np.int64)
                    mask[:, :last - first] = 1
                    feeds["attention_mask"] = mask
                output = session.run(None, feeds)[0][0]
                ids = output if output.ndim == 1 else output.argmax(axis=-1)
                # HuBERT's 20 ms output stride; padded frames are never exposed.
                decoded[key] = decode_romaji_ids(ids, reverse, blank, first / rate,
                                                 (last - first) / rate)
            events.extend(e for e in decoded[key] if a <= e.start_sec < b)
    return tuple(sorted(set(events), key=lambda e: (e.start_sec, e.end_sec)))
