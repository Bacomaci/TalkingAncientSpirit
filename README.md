# Talking Ancient Spirit — LARP Voice Bot

A voice-driven AI character system for LARP events. Players approach an altar and speak to the spirit of their ancestor, who answers in character via a speaker. A game master controls the session from a browser console.

## How it works

1. A player places their bone amulet on the altar — the GM clicks **Summon Spirit** in the browser console.
2. The spirit greets the player by voice (ElevenLabs TTS).
3. The player asks questions aloud — their speech is transcribed locally (faster-whisper), sent to Claude, and the reply is spoken back through the speaker.
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

> `faster-whisper` downloads the Whisper model on first run (~150 MB for `base`).
> For better accuracy use `WHISPER_MODEL=small` or `medium` in `.env`.

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Variable | Where to get it |
|---|---|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |
| `ELEVENLABS_API_KEY` | [elevenlabs.io](https://elevenlabs.io) → Profile |
| `ELEVENLABS_VOICE_ID` | ElevenLabs → Voices → click a voice → ID in the URL |
| `WHISPER_MODEL` | `tiny` / `base` / `small` / `medium` / `large-v2` (default: `base`) |

### 3. Configure the game

Start the server (step 4), then open **`http://localhost:8000/setup.html`** in a browser. From there you can:

- Create spirits — name, personality, milestones, secrets
- Add players and assign each one a spirit

No YAML editing needed. Changes save immediately.

### 4. Start the backend

```bash
uvicorn run:app --host 0.0.0.0 --port 8000
```

### 5. Open the GM console

Navigate to **`http://localhost:8000`** in any browser (works on a phone or tablet on the same WiFi).

### 6. Start the altar client

```bash
python altar_client.py
```

The altar client waits until the GM starts a session, then begins listening through the microphone.

## Can the GM PC and altar PC be the same machine?

Yes. Run the backend and altar client in two separate terminal windows on the same PC, and open the GM console in a browser on the same machine.

## GM workflow

| Step | Action |
|---|---|
| Before the event | Open Setup page, create spirits and players |
| Player approaches altar | Select player, click **Summon Spirit** |
| During conversation | Watch the live transcript, inject context as needed |
| Player reaches a story moment | Click **Advance Milestone** — the spirit reveals the next secret naturally |
| Player leaves | Click **End Session** |

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
- **Silence detection** — `SILENCE_THRESHOLD`, `SILENCE_DURATION` in `altar_client.py`
- **Whisper model size** — set `WHISPER_MODEL` in `.env`; larger = more accurate, slower
