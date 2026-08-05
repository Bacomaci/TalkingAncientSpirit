# Talking Ancient Spirit — LARP Voice Bot

A voice-driven AI character system for LARP events. Players approach an altar and speak to the spirit of their ancestor, who answers in character via a speaker. A game master controls the session from a browser console. Designed for Hungarian-language events, but fully configurable for any language.

## How it works

1. A player places their bone amulet on the altar — the GM clicks **Szellem megidézése** in the browser console.
2. The spirit greets the player by voice (ElevenLabs TTS).
3. The altar client opens a configurable listening window — the player speaks, their speech is transcribed locally (faster-whisper), sent to Claude, and the reply is spoken back through the speaker.
4. As the story progresses, the GM advances milestones — the spirit automatically weaves the next secret into the conversation.
5. The GM can inject live context (player mood, recent events) at any time without the player knowing.

## Architecture

```
[Altar PC]                         [Any browser on the same network]
altar_client.py  <──────────>  FastAPI backend  <────  GM Console (index.html)
  microphone                         │                  Game Setup  (setup.html)
  speaker                        Claude API  (LLM / spirit personality)
                                 ElevenLabs  (TTS / spirit voice)
                                 faster-whisper  (STT / local, offline-capable)
```

One spirit per player — each player's ancestor has its own personality, backstory, and milestone secrets, all configured through the browser UI. No YAML editing required.

## Setup

### 1. Install dependencies

Imstall anaconda (the miniconda-3 version suffices). In anaconda prompt:

```bash
conda env create -f [PATH_TO_LOCAL_REPO]\environment.yml
```

> `faster-whisper` downloads the Whisper model on first run (`tiny` ~75 MB, `base` ~150 MB).
> For better accuracy use `WHISPER_MODEL=small` or `medium` in `.env`.

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Variable | Description | Where to get it |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude API key | [console.anthropic.com](https://console.anthropic.com) |
| `ELEVENLABS_API_KEY` | ElevenLabs API key | elevenlabs.io → Profile |
| `ELEVENLABS_VOICE_ID` | Fallback voice id to use for spirits | ElevenLabs → Voices → click a voice → ID in the URL |
| `HF_TOKEN` | Huggingface API token for downloading large whisper models | — |
| `WHISPER_MODEL` | STT model size: `tiny` / `base` / `small` / `medium` / `large-v2` | — |
| `WHISPER_LANGUAGE` | STT language code (e.g. `hu`, `en`, `de`) | — |
| `SPIRIT_LANGUAGE` | Language the spirit responds in (e.g. `Hungarian`) | — |
| `TTS_LANGUAGE_CODE` | TTS accent/pronunciation code (e.g. `hu`, `en`) | — |
| `LISTEN_SECONDS` | Default mic listening window in seconds (default: `10`) | — |
| `AUDIO_INPUT_DEVICE` | Default audio input device index | python: ```print sounddevices.query_devices()``` |
| `AUDIO_OUTPUT_DEVICE` | Default audio uotput device index | python: ```print sounddevices.query_devices()``` |

> **Note:** `TTS_LANGUAGE_CODE` requires the `eleven_turbo_v2_5` model (used by default). `eleven_multilingual_v2` does not support explicit language codes.

> **Note:** To suppress a harmless HuggingFace symlink warning on Windows, add `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to your `.env`.

### 3. Configure the game

Start the server (step 5), then open **`http://localhost:8000/setup.html`** in a browser. From there you can:

- Create spirits — name, personality, milestones, secrets, and per-spirit listening window duration
- Add players and assign each one a spirit

No YAML editing needed. Changes save immediately.

### 4. Activate conda

In anaconda prompt:

```bash
conda activate dradskolpa-env-311
```

### 5. Start the backend

Inside the environment (or using conda as interpreter in VS Code):

```bash
cd TalkingAncientSpirit
uvicorn run:app --host 0.0.0.0 --port 8000
```

### 6. Open the GM console

Navigate to **`http://localhost:8000`** in any browser (works on a phone or tablet on the same WiFi).

### 7. Start the altar client

Inside the environment (or using conda as interpreter in VS Code):

```bash
python altar_client.py
```

The altar client waits until the GM starts a session, then plays the greeting and opens the microphone.

## Can the GM PC and altar PC be the same machine?

Yes. Run the backend and altar client in two separate terminal windows on the same PC, and open the GM console in a browser on the same machine.

## Optional GM workflow

| Step | Action |
|---|---|
| Before the event | Open Setup page, create spirits and players |
| Player approaches altar | Select player, click **Szellem megidézése** |
| During conversation | Watch the live transcript, inject context as needed |

## Conjuring spirits

The altar is always in idle state and listening to the microphone when no spirit is conjured. When a player taps the correct sequence with their stick (tá-tá-ti-ti-tá), the altar enters player identification. The player needs to say their secret sentence out loud, identifying themselves. The altar recognizes the player and summons the spirit. In case any of the parts fail or the player abandons, the altar returns to idle.

## Spririt conversation workflow

The altar records a clip that is at least `MIN_LISTEN_SECONDS` (default = 7), at most `MAX_LISTEN_SECONDS` (default = 25) long, and normally ends when the player has not spoken for at least `SILENCE_TIMEOUT` (default = 1.5) seconds. The clip is transcribed by whisper locally. A prompt is edited afterwards. It has two components: a large static component comprising of all global knowledge and spirit knowledge, and a small dynamic component containing GM context and the last 10 exchanges. This prompt is sent to Claude, which answers as if it was the spirit. This answer is then regex formatted and sent to elevenlabs, which transforms it TTS. This audio file is then played back.

## File overview

| Path | Purpose |
|---|---|
| `backend/main.py` | FastAPI server — REST endpoints + WebSocket |
| `backend/game_state.py` | Player / spirit / milestone data model |
| `backend/llm.py` | Claude prompt construction and response |
| `backend/stt.py` | faster-whisper transcription |
| `backend/tts.py` | ElevenLabs voice synthesis |
| `altar_client.py` | Altar PC client — summoning, mic recording and audio playback |
| `config/game_config.yaml` | Game content (auto-updated by the Setup UI) |
| `frontend/static/index.html` | GM console — session control and live transcript |
| `frontend/static/setup.html` | Game setup — spirits, milestones, players |

## Tuning

- **ElevenLabs voice** — `stability`, `similarity_boost`, `style` in `backend/tts.py`
- **Whisper model size** — set `WHISPER_MODEL` in `.env`; larger = more accurate, slower
- **VAD filter** — enabled by default in `backend/stt.py`; skips silent segments for faster transcription
