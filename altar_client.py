"""
Altar client — runs on the altar PC.
Records player speech in a fixed time window, sends to backend, plays spirit response.
"""
import os
import io
import base64
import tempfile
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
# Fallback listen window if the spirit config doesn't specify one
DEFAULT_LISTEN_SECONDS = int(os.getenv("LISTEN_SECONDS", "10"))
# Drain pause after playback so the mic doesn't catch speaker echo
PLAYBACK_DRAIN_SECONDS = 0.5


def record_fixed_window(duration_seconds: int) -> np.ndarray:
    """Record a fixed-length audio window from the microphone."""
    print(f"  [Listening for {duration_seconds}s...]")
    total_samples = SAMPLE_RATE * duration_seconds
    chunk_size = int(SAMPLE_RATE * 0.1)  # 100ms chunks
    chunks = []
    peak_rms = 0.0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32") as stream:
        samples_recorded = 0
        chunk_index = 0
        while samples_recorded < total_samples:
            data, _ = stream.read(chunk_size)
            chunks.append(data.copy())
            samples_recorded += len(data)
            rms = float(np.sqrt(np.mean(data ** 2)))
            if rms > peak_rms:
                peak_rms = rms
            if chunk_index % 10 == 0:
                elapsed = samples_recorded / SAMPLE_RATE
                remaining = duration_seconds - elapsed
                bar = "#" * int(rms * 500)
                print(f"  [mic rms={rms:.4f} peak={peak_rms:.4f} remaining={remaining:.0f}s] {bar}")
            chunk_index += 1

    audio = np.concatenate(chunks, axis=0)
    print(f"  [Recording done — {len(audio)/SAMPLE_RATE:.1f}s, peak rms={peak_rms:.4f}]")
    return audio


def play_audio_mp3(mp3_bytes: bytes):
    """Play MP3 bytes through the speaker, then drain."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(mp3_bytes)
        tmp_path = f.name

    data, samplerate = sf.read(tmp_path, dtype="float32")
    os.unlink(tmp_path)

    # Prepend 50ms of silence so the audio device has time to warm up
    # and doesn't clip the first syllable
    silence = np.zeros((int(samplerate * 0.05),) + data.shape[1:], dtype="float32")
    data = np.concatenate([silence, data], axis=0)

    sd.play(data, samplerate)
    sd.wait()
    time.sleep(PLAYBACK_DRAIN_SECONDS)


def audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def run_conversation_loop(listen_seconds: int):
    print("\n  [Session active. Speak to the spirit. Press Ctrl+C to stop.]\n")
    with httpx.Client(base_url=BACKEND_URL, timeout=60.0) as client:
        while True:
            print(f"  [Your turn — you have {listen_seconds}s to speak]")
            audio = record_fixed_window(listen_seconds)
            wav_bytes = audio_to_wav_bytes(audio)
            print(f"  [Sending {len(wav_bytes)} bytes to backend...]")

            try:
                resp = client.post(
                    "/api/speak",
                    files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                )
                if resp.status_code == 400:
                    print("  [No active session]")
                    break
                if resp.status_code == 422:
                    print("  [Nothing understood — please try again]")
                    continue
                resp.raise_for_status()
                data = resp.json()
                print(f"  [You said]: {data['player_said']}")
                print(f"  [Spirit]:   {data['spirit_said']}\n")
                mp3 = base64.b64decode(data["audio_base64"])
                play_audio_mp3(mp3)
            except httpx.HTTPStatusError as e:
                print(f"  [Backend error {e.response.status_code}: {e.response.text}]")
                time.sleep(1)
            except httpx.HTTPError as e:
                print(f"  [Error communicating with backend: {e}]")
                time.sleep(1)


def wait_for_session() -> dict:
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


def fetch_and_play_greeting(client: httpx.Client):
    """Wait until greeting is ready, then play it."""
    print("  [Waiting for greeting...]")
    while True:
        try:
            resp = client.get("/api/session/greeting")
            if resp.status_code == 200:
                data = resp.json()
                print(f"\n  [Spirit]: {data['greeting']}\n")
                mp3 = base64.b64decode(data["audio_base64"])
                play_audio_mp3(mp3)
                return
        except Exception as e:
            print(f"  [Greeting fetch error: {e}]")
        time.sleep(0.5)


def main():
    print("=== LARP Spirit Altar Client ===")
    print(f"Connecting to backend at {BACKEND_URL}")

    while True:
        status = wait_for_session()
        listen_seconds = status.get("listen_seconds", DEFAULT_LISTEN_SECONDS)
        print(f"\nSpirit '{status['spirit_name']}' awakens... (listen window: {listen_seconds}s)\n")

        with httpx.Client(base_url=BACKEND_URL, timeout=60.0) as client:
            fetch_and_play_greeting(client)

        try:
            run_conversation_loop(listen_seconds)
        except KeyboardInterrupt:
            print("\nAlt client stopped.")
            break

        print("\nSession ended. Waiting for next session...\n")


if __name__ == "__main__":
    main()
