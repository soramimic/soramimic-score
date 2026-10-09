"""Optional Demucs and audio reading inference, loaded only when requested."""
from __future__ import annotations

import gc
import logging
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)
DEMUCS_FILE = "955717e8-8726e21a.th"
DEMUCS_URL = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/" + DEMUCS_FILE
KANA_MODEL = "sbintuitions/kana-whisper"
KANA_REVISION = "88ecb3d79c5846cb4fcf76f4107b84c8fa2acd82"


def release_memory():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _demucs_package(config):
    import torch
    from demucs.repo import check_checksum

    checkpoint = (Path(config.demucs_checkpoint) if config.demucs_checkpoint is not None
                  else Path(torch.hub.get_dir()) / "checkpoints" / DEMUCS_FILE)
    if checkpoint.is_file():
        # Demucs checkpoints contain Python classes. Only the official, checked
        # HTDemucs artifact is accepted; do not disable weights_only globally.
        check_checksum(checkpoint, "8726e21a")
        return torch.load(checkpoint, map_location="cpu", weights_only=False)
    if config.local_files_only or config.demucs_checkpoint is not None:
        raise FileNotFoundError(f"HTDemucs checkpoint is not available locally: {checkpoint}")
    return torch.hub.load_state_dict_from_url(DEMUCS_URL, map_location="cpu", check_hash=True,
                                               weights_only=False)


def separate_vocals(path, output, config, accompaniment=None):
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.states import load_model

    logger.info("ボーカルを分離しています")
    model = load_model(_demucs_package(config)).eval()
    try:
        samples, rate = librosa.load(str(path), sr=model.samplerate, mono=False)
        if not samples.size or not np.isfinite(samples).all():
            raise ValueError("Audio must contain finite samples")
        if samples.ndim == 1:
            samples = np.tile(samples, (model.audio_channels, 1))
        elif samples.shape[0] != model.audio_channels:
            samples = np.tile(samples.mean(axis=0), (model.audio_channels, 1))
        wave = torch.from_numpy(samples)
        reference = wave.mean(dim=0)
        mean = reference.mean()
        scale = reference.std(unbiased=False).clamp(min=1e-8)
        with torch.inference_mode():
            separated = apply_model(model, ((wave - mean) / scale)[None],
                                    device=config.device, shifts=0, split=True,
                                    overlap=.25, progress=False)[0]
        vocals = (separated[model.sources.index("vocals")].cpu() * scale + mean).numpy()
        if vocals.shape != samples.shape or not np.isfinite(vocals).all():
            raise ValueError("Vocal separation must preserve audio length and finite samples")
        # Float WAV avoids clipping/renormalizing the isolated voice.
        sf.write(str(output), vocals.T, rate, subtype="FLOAT")
        if accompaniment is not None:
            sf.write(str(accompaniment), (samples - vocals).T, rate,
                     format="FLAC", subtype="PCM_16")
    finally:
        del model
        release_memory()


def transcribe_kana_views(paths, windows, config):
    """Infer the same bounded windows on mix/vocals without supplying lyric text."""
    import librosa
    import numpy as np
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    logger.info("音声から読み候補を比較しています")
    dtype = torch.float16 if config.device == "cuda" else torch.float32
    options = {"local_files_only": config.local_files_only}
    if config.kana_model == KANA_MODEL:
        options["revision"] = KANA_REVISION
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        config.kana_model, torch_dtype=dtype, use_safetensors=True, **options,
    ).eval().to(config.device)
    transcriber = None
    try:
        processor = AutoProcessor.from_pretrained(config.kana_model, **options)
        transcriber = pipeline("automatic-speech-recognition", model=model,
                               tokenizer=processor.tokenizer,
                               feature_extractor=processor.feature_extractor,
                               device=config.device, torch_dtype=dtype)
        results = {}
        for view, path in paths.items():
            samples, rate = librosa.load(str(path), sr=16000, mono=True)
            if not len(samples) or not np.isfinite(samples).all():
                raise ValueError("Audio must contain finite samples")
            transcripts = []
            for start, end in windows:
                if not 0 <= start < end <= len(samples) / rate + .01 or end - start > 24.01:
                    raise ValueError("KanaWhisper windows must be within the audio and <=24 seconds")
                segment = samples[round(start * rate):round(end * rate)]
                if not len(segment):
                    raise ValueError("Empty KanaWhisper audio window")
                with torch.inference_mode():
                    result = transcriber(segment, generate_kwargs={"language": "ja", "task": "transcribe"})
                transcripts.append(result["text"])
            results[view] = tuple(transcripts)
        return results
    finally:
        del transcriber, model
        release_memory()


def transcribe_whisper_views(paths, windows, config, *, shared=None):
    """Recognize bounded symbol context, without lyric prompts or hints."""
    import librosa
    import numpy as np
    import soundfile as sf

    logger.info("音声から記号の読みを確認しています")
    model = None
    try:
        if shared is None:
            from faster_whisper import WhisperModel
            model = WhisperModel(config.whisper_model, device=config.device,
                                 compute_type="int8" if config.device == "cpu" else "float16",
                                 local_files_only=config.local_files_only)
        results = {}
        for view, path in paths.items():
            samples, rate = librosa.load(str(path), sr=16000, mono=True)
            if not len(samples) or not np.isfinite(samples).all():
                raise ValueError("Audio must contain finite samples")
            texts = []
            for start, end in windows:
                if not 0 <= start < end <= len(samples) / rate + .01 or end - start > 24.01:
                    raise ValueError("Whisper reading windows must be within the audio and <=24 seconds")
                segment = samples[round(start * rate):round(end * rate)]
                if not len(segment):
                    raise ValueError("Empty Whisper reading window")
                if shared is not None:
                    with tempfile.TemporaryDirectory(prefix="soramimic-score-reading-") as directory:
                        excerpt = Path(directory) / "window.wav"
                        sf.write(excerpt, segment, rate, subtype="FLOAT")
                        response = shared.run("whisper", excerpt, {
                            "model_size": config.whisper_model, "device": "auto",
                            "language": "ja", "vad_filter": False,
                            "condition_on_previous_text": False, "temperature": 0.,
                        })
                    if (not isinstance(response, dict)
                            or not isinstance(response.get("lines"), list)
                            or response.get("requested_language") != "ja"
                            or response.get("requested_temperature") != 0.
                            or any(not isinstance(row, dict) or not isinstance(row.get("text"), str)
                                   for row in response["lines"])):
                        raise RuntimeError("shared Whisper reading response is invalid")
                    text = "".join(row["text"] for row in response["lines"])
                else:
                    segments, _ = model.transcribe(segment, language="ja", vad_filter=False,
                                                   condition_on_previous_text=False, temperature=0.)
                    text = "".join(segment.text for segment in segments)
                texts.append(text)
            results[view] = tuple(texts)
        return results
    finally:
        if model is not None:
            del model
            release_memory()
