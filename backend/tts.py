"""
Text-to-speech using ElevenLabs API.
Returns audio bytes (MP3) for the given text.
"""
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings


_client: ElevenLabs | None = None


def get_client(api_key: str) -> ElevenLabs:
    global _client
    if _client is None:
        _client = ElevenLabs(api_key=api_key)
    return _client


def synthesize_speech(
    text: str,
    voice_id: str,
    api_key: str,
    stability: float = 0.6,
    similarity_boost: float = 0.8,
    style: float = 0.3,
) -> bytes:
    """Returns MP3 audio bytes for the given text."""
    client = get_client(api_key)

    audio_stream = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=stability,
            similarity_boost=similarity_boost,
            style=style,
            use_speaker_boost=True,
        ),
    )

    return b"".join(audio_stream)
