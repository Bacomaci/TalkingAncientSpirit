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

```bash
pip install -r requirements.txt
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
| `ELEVENLABS_VOICE_ID` | Voice to use for the spirit | ElevenLabs → Voices → click a voice → ID in the URL |
| `WHISPER_MODEL` | STT model size: `tiny` / `base` / `small` / `medium` / `large-v2` | — |
| `WHISPER_LANGUAGE` | STT language code (e.g. `hu`, `en`, `de`) | — |
| `SPIRIT_LANGUAGE` | Language the spirit responds in (e.g. `Hungarian`) | — |
| `TTS_LANGUAGE_CODE` | TTS accent/pronunciation code (e.g. `hu`, `en`) | — |
| `LISTEN_SECONDS` | Default mic listening window in seconds (default: `10`) | — |

> **Note:** `TTS_LANGUAGE_CODE` requires the `eleven_turbo_v2_5` model (used by default). `eleven_multilingual_v2` does not support explicit language codes.

> **Note:** To suppress a harmless HuggingFace symlink warning on Windows, add `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to your `.env`.

### 3. Configure the game

Start the server (step 4), then open **`http://localhost:8000/setup.html`** in a browser. From there you can:

- Create spirits — name, personality, milestones, secrets, and per-spirit listening window duration
- Add players and assign each one a spirit

No YAML editing needed. Changes save immediately.

### 4. Start the backend

```bash
cd TalkingAncientSpirit
uvicorn run:app --host 0.0.0.0 --port 8000
```

### 5. Open the GM console

Navigate to **`http://localhost:8000`** in any browser (works on a phone or tablet on the same WiFi).

### 6. Start the altar client

```bash
python altar_client.py
```

The altar client waits until the GM starts a session, then plays the greeting and opens the microphone.

## Can the GM PC and altar PC be the same machine?

Yes. Run the backend and altar client in two separate terminal windows on the same PC, and open the GM console in a browser on the same machine.

## GM workflow

| Step | Action |
|---|---|
| Before the event | Open Setup page, create spirits and players |
| Player approaches altar | Select player, click **Szellem megidézése** |
| During conversation | Watch the live transcript, inject context as needed |
| Player reaches a story moment | Click **Következő mérföldkőre lépés** — the spirit reveals the next secret naturally |
| Player leaves | Click **Munkamenet befejezése** |

## Listening window

The altar client records a fixed-length audio window after each spirit response rather than using silence detection. The duration defaults to `LISTEN_SECONDS` (env) and can be overridden per spirit with `listen_seconds` in `game_config.yaml` or the Setup UI. Adjust it to match your players' speaking pace.

## File overview

| Path | Purpose |
|---|---|
| `backend/main.py` | FastAPI server — REST endpoints + WebSocket |
| `backend/game_state.py` | Player / spirit / milestone data model |
| `backend/llm.py` | Claude prompt construction and response |
| `backend/stt.py` | faster-whisper transcription |
| `backend/tts.py` | ElevenLabs voice synthesis |
| `altar_client.py` | Altar PC client — mic recording and audio playback |
| `config/game_config.yaml` | Game content (auto-updated by the Setup UI) |
| `frontend/static/index.html` | GM console — session control and live transcript |
| `frontend/static/setup.html` | Game setup — spirits, milestones, players |

## Tuning

- **ElevenLabs voice** — `stability`, `similarity_boost`, `style` in `backend/tts.py`
- **Listening window** — `LISTEN_SECONDS` in `.env` or `listen_seconds` per spirit in the Setup UI
- **Whisper model size** — set `WHISPER_MODEL` in `.env`; larger = more accurate, slower
- **VAD filter** — enabled by default in `backend/stt.py`; skips silent segments for faster transcription
