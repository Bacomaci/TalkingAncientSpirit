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

import collections
import webrtcvad

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
SAMPLE_RATE = 16000 # DO NOT CHANGE (this is one of the few numbers that whisper and WebRTC can bot handle!)
MIN_LISTEN_SECONDS = 5
MAX_LISTEN_SECONDS = 25
SILENCE_TIMEOUT = 1.5
CHANNELS = 1

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

def record_vad_window(max_duration_seconds: int, min_duration_seconds: int = 6, silence_timeout:float = 1.5) -> np.ndarray:
    """
    Record audio for at least min_duration_seconds. Recording stops automatically when 
    player stays silent for silence_timeout. Records for a maximum of max_duration_seconds.
    """
    # Initialize WebRTC VAD (with filter mode 2 or 3 recommended in a LARP setting) 
    vad = webrtcvad.Vad()
    vad.set_mode(3) 

    # VAD settings
    frame_duration_ms = 30
    chunk_size = int(SAMPLE_RATE * (frame_duration_ms / 1000.0))

    # Set sliding window ring buffer
    # Observing 10 times 30 ms windows
    padding_chunks = 10
    ring_buffer = collections.deque(maxlen=padding_chunks)
    
    chunks = []
    has_spoken = False
    silence_start_time = None
    start_time = time.time()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32") as stream:
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Cutoff time to be safe
            if elapsed_time >= max_duration_seconds:
                print("  [Max duration reached — cutting off]")
                break

            # Read audio data
            data, _ = stream.read(chunk_size)
            chunks.append(data.copy())
            
            # Compute RMS
            rms = float(np.sqrt(np.mean(data ** 2)))

            # Convert to INT16 PCM for WebRTC VAD
            data_int16 = (data * 32767).astype(np.int16)
            raw_pcm_bytes = data_int16.tobytes()

            # VAD test
            try:
                is_speech = vad.is_speech(raw_pcm_bytes, SAMPLE_RATE)
            except Exception:
                is_speech = False

            ring_buffer.append(is_speech)
            num_speech_frames = sum(1 for x in ring_buffer if x)

            # Raise speech flag if sliding window contains at least 80% speech
            if not has_spoken and num_speech_frames >= int(0.8 * padding_chunks):
                print("  [Speech detected...]")
                has_spoken = True

            # Check for silence
            if has_spoken:
                if num_speech_frames <= int(0.1 * padding_chunks):
                    if silence_start_time == None:
                        silence_start_time = time.time()
                    else:
                        current_silence_duration = time.time() - silence_start_time
                        
                        # End condition is 90% silence for silence_timeout
                        if elapsed_time >= min_duration_seconds and current_silence_duration >= silence_timeout:
                            print(f"  [Silence detected for {current_silence_duration:.1f}s — stop recording]")
                            break
                else:
                    silence_start_time = None

            # If player has not spoken
            if not has_spoken and elapsed_time >= min_duration_seconds:
                print("  [No speech detected within initial window — stopping]")
                break

            # Terminal feedback
            if len(chunks) % 5 == 0:
                bar = "#" * int(rms * 500)
                status = "SPEAKING" if has_spoken else "WAITING"
                print(f"  [{status} | elapsed={elapsed_time:.1f}s | rms={rms:.4f}] {bar}")

    audio = np.concatenate(chunks, axis=0)
    print(f"  [Recording done — total length: {len(audio)/SAMPLE_RATE:.1f}s]")
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


def run_conversation_loop():
    print("\n  [Session active. Speak to the spirit. Press Ctrl+C to stop.]\n")
    with httpx.Client(base_url=BACKEND_URL, timeout=60.0) as client:
        while True:

            print(f"  [Your turn — speak when ready (min {MIN_LISTEN_SECONDS}s, max {MAX_LISTEN_SECONDS}s)]")
            audio = record_vad_window(max_duration_seconds=MAX_LISTEN_SECONDS, min_duration_seconds=MIN_LISTEN_SECONDS, silence_timeout=SILENCE_TIMEOUT)
            
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
        print(f"\nSpirit '{status['spirit_name']}' awakens...\n")

        with httpx.Client(base_url=BACKEND_URL, timeout=60.0) as client:
            fetch_and_play_greeting(client)

        try:
            run_conversation_loop()
        except KeyboardInterrupt:
            print("\nAlt client stopped.")
            break

        print("\nSession ended. Waiting for next session...\n")


if __name__ == "__main__":
    main()
