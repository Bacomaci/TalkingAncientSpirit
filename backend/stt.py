"""
Speech-to-text using faster-whisper (local, runs on CPU or GPU).
"""
import io
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel


_model: WhisperModel | None = None


def get_model(model_size: str = "base") -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _model


def transcribe_audio(audio_bytes: bytes, model_size: str = "base", language: str = "hu") -> str:
    """Transcribe raw audio bytes (WAV format) to text."""
    model = get_model(model_size)

    audio_buffer = io.BytesIO(audio_bytes)
    audio_data, sample_rate = sf.read(audio_buffer, dtype="float32")

    # Whisper expects mono audio at 16kHz
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    if sample_rate != 16000:
        import resampy
        audio_data = resampy.resample(audio_data, sample_rate, 16000)

    segments, _ = model.transcribe(audio_data, beam_size=5, language=language)
    text = " ".join(segment.text for segment in segments).strip()
    return text
