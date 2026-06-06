"""
Altar client — runs on the altar PC.
Records player speech, sends to backend, plays spirit response.
"""
import os
import io
import base64
import tempfile
import threading
import time

import httpx
import sounddevice as sd
import soundfile as sf
import numpy as np
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
SAMPLE_RATE = 16000
CHANNELS = 1
# Silence detection: stop recording after this many seconds of silence
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 1.5   # seconds of silence before cutting off
MAX_RECORDING_SECONDS = 30


def record_until_silence() -> np.ndarray:
    """Records audio from the microphone, stops after silence."""
    print("  [Listening...]")
    chunks = []
    silent_chunks = 0
    chunk_size = int(SAMPLE_RATE * 0.1)  # 100ms chunks
    silence_chunks_needed = int(SILENCE_DURATION / 0.1)
    max_chunks = int(MAX_RECORDING_SECONDS / 0.1)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32") as stream:
        for _ in range(max_chunks):
            data, _ = stream.read(chunk_size)
            chunks.append(data.copy())
            rms = np.sqrt(np.mean(data ** 2))
            if rms < SILENCE_THRESHOLD:
                silent_chunks += 1
                if silent_chunks >= silence_chunks_needed and len(chunks) > silence_chunks_needed + 5:
                    break
            else:
                silent_chunks = 0

    audio = np.concatenate(chunks, axis=0)
    return audio


def play_audio_mp3(mp3_bytes: bytes):
    """Play MP3 bytes through the speaker."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(mp3_bytes)
        tmp_path = f.name

    data, samplerate = sf.read(tmp_path, dtype="float32")
    sd.play(data, samplerate)
    sd.wait()
    os.unlink(tmp_path)


def audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def play_greeting(greeting_b64: str, text: str):
    print(f"\n  [Spirit]: {text}\n")
    mp3 = base64.b64decode(greeting_b64)
    play_audio_mp3(mp3)


def run_conversation_loop():
    print("\n  [Session active. Speak to the spirit. Press Ctrl+C to stop.]\n")
    with httpx.Client(base_url=BACKEND_URL, timeout=60.0) as client:
        while True:
            audio = record_until_silence()

            if len(audio) < SAMPLE_RATE * 0.5:
                continue  # too short, ignore

            wav_bytes = audio_to_wav_bytes(audio)

            try:
                resp = client.post(
                    "/api/speak",
                    files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                )
                if resp.status_code == 400:
                    print("  [No active session]")
                    break
                resp.raise_for_status()
                data = resp.json()
                print(f"  [You said]: {data['player_said']}")
                print(f"  [Spirit]:   {data['spirit_said']}\n")
                mp3 = base64.b64decode(data["audio_base64"])
                play_audio_mp3(mp3)
            except httpx.HTTPError as e:
                print(f"  [Error communicating with backend: {e}]")
                time.sleep(1)


def wait_for_session():
    """Poll the backend until a session is started by the GM."""
    print("Waiting for GM to start a session...")
    with httpx.Client(base_url=BACKEND_URL, timeout=10.0) as client:
        while True:
            try:
                resp = client.get("/api/session/status")
                data = resp.json()
                if data["active"]:
                    print(f"Session started for player: {data['player_name']}")
                    return data
            except Exception:
                pass
            time.sleep(2)


def main():
    print("=== LARP Spirit Altar Client ===")
    print(f"Connecting to backend at {BACKEND_URL}")

    while True:
        status = wait_for_session()
        print(f"\nSpirit '{status['spirit_name']}' awakens...\n")

        # The greeting audio was already sent by the backend when the GM started the session
        # We just wait and listen
        try:
            run_conversation_loop()
        except KeyboardInterrupt:
            print("\nAlt client stopped.")
            break

        # After session ends, wait for next one
        print("\nSession ended. Waiting for next session...\n")


if __name__ == "__main__":
    main()
