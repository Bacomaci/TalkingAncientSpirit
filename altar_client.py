"""
Altar client — runs on the altar PC.
Records player speech in a fixed time window, sends to backend, plays spirit response.
"""
import os
import io
import base64
import tempfile
import time
import sys

import httpx
import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter
import aubio
from dotenv import load_dotenv

import collections
import webrtcvad

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
SAMPLE_RATE = 16000 # DO NOT CHANGE (this is one of the few numbers that whisper and WebRTC can bot handle!)
MIN_LISTEN_SECONDS = 7
MAX_LISTEN_SECONDS = 25
SILENCE_TIMEOUT = 1.5
CHANNELS = 1

# Idle altar settings
BUFFER_SIZE = 512
HOP_SIZE = 256

# Drain pause after playback so the mic doesn't catch speaker echo
PLAYBACK_DRAIN_SECONDS = 0.5

def record_vad_window(max_duration_seconds: int, min_duration_seconds: int = 6, silence_timeout:float = 1.5) -> tuple[np.ndarray, bool]:
    """
    Record audio for at least min_duration_seconds. Recording stops automatically when
    player stays silent for silence_timeout. Records for a maximum of max_duration_seconds.
    Returns (audio, spoke) where spoke is False if no speech was detected at all.
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
    return audio, has_spoken

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


def wake_up_altar_with_spacebar() -> bool:
    """Return True when the altar should wake up and start player identification.
    In testing mode this triggers on spacebar press; replace with real sensor logic for production.
    """
    if sys.stdin.isatty():
        try:
            import msvcrt
            if msvcrt.kbhit():
                key = msvcrt.getwch()
                return key == ' '
        except ImportError:
            import select
            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.read(1)
                return True
    return False

def butter_highpass(cutoff=800, fs=SAMPLE_RATE, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False) # type: ignore
    return b, a

HP_B, HP_A = butter_highpass(cutoff=800, fs=SAMPLE_RATE)

def apply_highpass_filter(data):
    """Filters deep sounds and human speech"""
    return lfilter(HP_B, HP_A, data)

def wake_up_altar() -> bool:
    """
    Listens to the microphone and returns True for the required tap pattern (tá-tá-ti-ti-tá).
    """

    # Target rhythm
    target_ratios = np.array([1.0, 1.0, 0.5, 0.5], dtype=np.float32)
    target_pattern = target_ratios / np.sum(target_ratios)

    # Number of knocks needed
    NUM_INTERVALS = len(target_pattern)
    REQUIRED_TAP_COUNT = NUM_INTERVALS + 1

    # Euclidean vector tolerance for rhythm matching
    TOLERANCE = 0.06
    
    # Debounce (minimal time between knocks)
    MIN_INTER_TAP_TIME = 0.12  # 120 ms
    
    # Rhythm resets at this timeout value
    TIMEOUT_SECONDS = 2.0

    BASE_THRESHOLD = 0.25      # Minimum threshold in silence
    NOISE_MULTIPLIER = 12.0    # Multiply RMS by this to get threshold

    # Initialize onset detector. "specdiff" algorithm good for sharp transients
    aubio_onset = aubio.onset("specdiff", BUFFER_SIZE, HOP_SIZE, SAMPLE_RATE) # type: ignore
    aubio_onset.set_threshold(0.3)

    noise_history = collections.deque(maxlen=60)
    tap_timestamps = collections.deque(maxlen=REQUIRED_TAP_COUNT)
    
    is_pattern_matched = False

    # Callback function (this runs at each window)
    def audio_callback(indata, frames, time_info, status):
        nonlocal is_pattern_matched, tap_timestamps
        
        if is_pattern_matched:
            return

        # Convert sounddevice chunk to numpy array for aubio
        raw_chunk = indata[:, 0].astype(np.float32)

        # Filter input noise
        filtered_chunk = apply_highpass_filter(raw_chunk)
        filtered_chunk = np.ascontiguousarray(filtered_chunk, dtype=np.float32)

        # Compute dynamic RMS
        rms_energy = float(np.sqrt(np.mean(filtered_chunk ** 2)))
        noise_history.append(rms_energy)
        avg_noise = np.mean(noise_history) if len(noise_history) > 0 else 0.001

        # Scale threshold dynamically
        dynamic_threshold = BASE_THRESHOLD + (avg_noise * NOISE_MULTIPLIER)
        aubio_onset.set_threshold(min(dynamic_threshold, 2.0))

        # Aubio detects onset
        if aubio_onset(filtered_chunk):
            now = time.time()
            
            # Clean buffer if rhythm timed out
            if len(tap_timestamps) > 0 and (now - tap_timestamps[-1]) > TIMEOUT_SECONDS:
                tap_timestamps.clear()

            # Filter debounce cases
            if len(tap_timestamps) == 0 or (now - tap_timestamps[-1]) >= MIN_INTER_TAP_TIME:
                tap_timestamps.append(now)
                print(f" [Tap detected! Number of taps: {len(tap_timestamps)}/{REQUIRED_TAP_COUNT}]")

                # When required number of taps is reached, pattern is matched
                if len(tap_timestamps) == REQUIRED_TAP_COUNT:
                    timestamps = np.array(tap_timestamps)
                    
                    # Inter-Onset Intervals
                    intervals = np.diff(timestamps)
                    total_duration = np.sum(intervals)

                    if total_duration > 0:
                        # Normalize intervals to achieve tempo independence
                        measured_pattern = intervals / total_duration

                        # Compute euclidean distance
                        euclidean_distance = np.linalg.norm(measured_pattern - target_pattern)
                        
                        print(f" [Rhythm analysis: euclidean distance = {euclidean_distance:.4f} (Tolerance: {TOLERANCE})]")

                        if euclidean_distance <= TOLERANCE:
                            print(" [Ritual succesful. Altar activated!]")
                            is_pattern_matched = True
                        else:
                            print(" [Wrong rhythm. Try again!]")
                            # Slide buffer, bin first tap
                            tap_timestamps.popleft()

    # Start microphone stream
    try:
        with sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=HOP_SIZE,
            dtype="float32",
            callback=audio_callback
        ):
            # Stream runs until pattern matching
            while not is_pattern_matched:
                time.sleep(0.01)

    except Exception as e:
        print(f" [Microphone stream error: {e}]")
        return False

    return is_pattern_matched


def identify_player(client: httpx.Client) -> str | None:
    """Listen for player introduction, send to backend for keyword matching, return player_id."""
    print("\n  [Identifying player — please speak your name and allegiance...]\n")
    audio, spoke = record_vad_window(
        max_duration_seconds=MAX_LISTEN_SECONDS,
        min_duration_seconds=10,
        silence_timeout=SILENCE_TIMEOUT,
    )
    if not spoke:
        print("  [No speech detected during identification]")
        return None
    wav_bytes = audio_to_wav_bytes(audio)
    try:
        resp = client.post("/api/identify", files={"file": ("audio.wav", wav_bytes, "audio/wav")})
        resp.raise_for_status()
        data = resp.json()
        print(f"  [Heard]: {data['transcribed']}")
        scores = data["scores"]
        if not scores or scores[0]["matches"] == 0:
            print("  [No player keywords matched]")
            return None
        best = scores[0]
        print(f"  [Identified]: {best['player_name']} ({best['matches']} keyword matches)")
        return best["player_id"]
    except Exception as e:
        print(f"  [Identification error: {e}]")
        return None


def run_conversation_loop():
    print("\n  [Session active. Speak to the spirit. Press Ctrl+C to stop.]\n")
    with httpx.Client(base_url=BACKEND_URL, timeout=60.0) as client:
        while True:

            print(f"  [Your turn — speak when ready (min {MIN_LISTEN_SECONDS}s, max {MAX_LISTEN_SECONDS}s)]")
            audio, spoke = record_vad_window(max_duration_seconds=MAX_LISTEN_SECONDS, min_duration_seconds=MIN_LISTEN_SECONDS, silence_timeout=SILENCE_TIMEOUT)

            if not spoke:
                print("  [No speech detected — player may have left. Triggering farewell...]")
                try:
                    resp = client.post("/api/session/abandon")
                    if resp.status_code == 200:
                        data = resp.json()
                        print(f"  [Spirit]: {data['spirit_said']}\n")
                        mp3 = base64.b64decode(data["audio_base64"])
                        play_audio_mp3(mp3)
                except Exception as e:
                    print(f"  [Abandon error: {e}]")
                break

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
                if data.get("session_ended"):
                    print("  [Session ended by spirit]\n")
                    break
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


def autonomous_wake_and_identify() -> str | None:
    """Idle loop: wait for wake trigger, then identify the player via voice.
    Returns a player_id on success, None if identification fails (caller should retry).
    """
    print("  [Altar idle — waiting for wake trigger (press SPACE to wake)...]")
    while not wake_up_altar():
        time.sleep(0.1)

    print("  [Altar awake!]")
    with httpx.Client(base_url=BACKEND_URL, timeout=30.0) as client:
        player_id = identify_player(client)
        if player_id is None:
            return None

        # Start the session on the backend
        try:
            resp = client.post("/api/session/start", json={"player_id": player_id})
            resp.raise_for_status()
            print(f"  [Session started for player_id={player_id}]")
            return player_id
        except Exception as e:
            print(f"  [Failed to start session: {e}]")
            return None


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

    gm_mode = "--gm" in sys.argv  # Pass --gm to use old GM-driven session start

    while True:
        if gm_mode:
            status = wait_for_session()
            print(f"\nSpirit '{status['spirit_name']}' awakens...\n")
        else:
            player_id = None
            while player_id is None:
                player_id = autonomous_wake_and_identify()
                if player_id is None:
                    print("  [Identification failed — returning to idle]\n")

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
