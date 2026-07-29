"""
FastAPI backend — REST + WebSocket server for the LARP spirit bot.
"""
import os
import base64
import datetime
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv

from .game_state import GameConfig, SessionState, _slugify
from .llm import get_spirit_response, player_wants_to_leave, MAX_EXCHANGES
from .stt import transcribe_audio
from .tts import synthesize_speech

load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_VOICE_ID = os.environ["ELEVENLABS_VOICE_ID"]
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "hu")
SPIRIT_LANGUAGE = os.getenv("SPIRIT_LANGUAGE", "Hungarian")
TTS_LANGUAGE_CODE = os.getenv("TTS_LANGUAGE_CODE", "hu")

import anthropic as _anthropic
_llm_client = _anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

game_config: GameConfig
session: SessionState

# Active WebSocket connections (GM console)
_ws_clients: list[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global game_config, session
    game_config = GameConfig("config/game_config.yaml")
    session = SessionState()
    yield


app = FastAPI(title="LARP Spirit Bot", lifespan=lifespan)

# ── REST API ──────────────────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    player_id: str

class UpdateContextRequest(BaseModel):
    gm_context: str

class AdvanceMilestoneResponse(BaseModel):
    success: bool
    new_milestone: int
    milestone_name: str

class SessionStatus(BaseModel):
    active: bool
    player_name: str | None
    spirit_name: str | None
    current_milestone: int
    milestone_name: str | None
    gm_context: str
    listen_seconds: int


@app.get("/api/players")
def list_players():
    return [{"id": p.id, "name": p.name} for p in game_config.players.values()]


@app.post("/api/session/start")
def start_session(req: StartSessionRequest):
    player = game_config.get_player(req.player_id)
    if not player:
        raise HTTPException(404, f"Player '{req.player_id}' not found")

    spirit = game_config.get_spirit_for_player(player)
    if not spirit:
        raise HTTPException(404, f"Spirit not found for player '{req.player_id}'")

    # Save any previous conversation before resetting
    if session.session_active and session.conversation_history:
        _save_conversation()

    session.player = player
    session.spirit = spirit
    session.session_active = True
    session.current_milestone = min(spirit.current_day, len(spirit.milestones) - 1)
    session.reset_conversation()

    # Generate opening greeting using the spirit
    milestone = session.current_milestone_obj()
    greeting_seed = milestone.greeting if milestone and milestone.greeting else None

    if greeting_seed:
        opening_display = greeting_seed
        opening_tts = greeting_seed
    else:
        opening_display, opening_tts = get_spirit_response(
            session,
            f"[The player {player.name} has just placed the bone on the altar and summoned you. Greet them.]",
            _llm_client,
            SPIRIT_LANGUAGE,
        )

    audio = synthesize_speech(opening_tts, ELEVENLABS_VOICE_ID, ELEVENLABS_API_KEY, TTS_LANGUAGE_CODE)
    audio_b64 = base64.b64encode(audio).decode()

    session.greeting = opening_display
    session.greeting_audio_b64 = audio_b64

    return {
        "greeting": opening_display,
        "audio_base64": audio_b64,
    }


@app.get("/api/session/greeting")
def get_greeting():
    if not session.session_active or not session.greeting:
        raise HTTPException(404, "No active greeting")
    return {
        "greeting": session.greeting,
        "audio_base64": session.greeting_audio_b64,
    }


@app.post("/api/session/end")
def end_session():
    if session.session_active and session.conversation_history:
        _save_conversation()
    session.session_active = False
    session.reset_conversation()
    return {"status": "ended"}


@app.post("/api/session/advance_milestone", response_model=AdvanceMilestoneResponse)
def advance_milestone():
    if not session.session_active:
        raise HTTPException(400, "No active session")
    success = session.advance_milestone()
    m = session.current_milestone_obj()
    return AdvanceMilestoneResponse(
        success=success,
        new_milestone=session.current_milestone,
        milestone_name=m.name if m else "unknown",
    )


@app.post("/api/session/context")
def update_context(req: UpdateContextRequest):
    session.gm_context = req.gm_context
    return {"status": "updated"}


@app.get("/api/session/status", response_model=SessionStatus)
def get_status():
    m = session.current_milestone_obj()
    return SessionStatus(
        active=session.session_active,
        player_name=session.player.name if session.player else None,
        spirit_name=session.spirit.name if session.spirit else None,
        current_milestone=session.current_milestone,
        milestone_name=m.name if m else None,
        gm_context=session.gm_context,
        listen_seconds=session.spirit.listen_seconds if session.spirit else 10,
    )


@app.post("/api/identify")
async def identify_player(file: UploadFile = File(...)):
    """Transcribe audio and return keyword match scores for all players."""
    audio_bytes = await file.read()
    try:
        text = transcribe_audio(audio_bytes, WHISPER_MODEL, WHISPER_LANGUAGE)
    except Exception as e:
        raise HTTPException(500, f"STT error: {e}") from e

    if not text:
        raise HTTPException(422, "Could not transcribe audio")

    text_lower = text.lower()
    scores = []
    for player in game_config.players.values():
        matches = sum(1 for kw in player.keywords if kw.lower() in text_lower)
        scores.append({"player_id": player.id, "player_name": player.name, "matches": matches})

    scores.sort(key=lambda x: x["matches"], reverse=True)
    return {"transcribed": text, "scores": scores}


@app.post("/api/speak")
async def speak(file: UploadFile = File(...)):
    """Receive player audio, transcribe, generate spirit reply, return audio."""
    if not session.session_active:
        raise HTTPException(400, "No active session — GM must start a session first")

    audio_bytes = await file.read()
    try:
        player_text = transcribe_audio(audio_bytes, WHISPER_MODEL, WHISPER_LANGUAGE)
    except Exception as e:
        raise HTTPException(500, f"STT error: {e}") from e

    if not player_text:
        raise HTTPException(422, "Could not transcribe audio")

    try:
        reply_display, reply_tts = get_spirit_response(session, player_text, _llm_client, SPIRIT_LANGUAGE)
    except Exception as e:
        raise HTTPException(500, f"LLM error: {e}") from e

    try:
        audio = synthesize_speech(reply_tts, session.spirit.voice_id or ELEVENLABS_VOICE_ID, ELEVENLABS_API_KEY, TTS_LANGUAGE_CODE)
    except Exception as e:
        raise HTTPException(500, f"TTS error: {e}") from e

    audio_b64 = base64.b64encode(audio).decode()

    # Determine whether the session should end after this reply
    farewell = player_wants_to_leave(player_text)
    limit_reached = session.exchange_count >= MAX_EXCHANGES
    end_session_now = farewell or limit_reached

    if end_session_now:
        _save_conversation()
        session.session_active = False
        session.reset_conversation()

    # Push transcript to GM console via WebSocket
    await _broadcast({
        "type": "transcript",
        "player_said": player_text,
        "spirit_said": reply_display,
        **({"type_hint": "farewell"} if farewell else {}),
    })

    return {
        "player_said": player_text,
        "spirit_said": reply_display,
        "audio_base64": audio_b64,
        "session_ended": end_session_now,
    }


# ── WebSocket (GM live feed) ──────────────────────────────────────────────────

@app.websocket("/ws/gm")
async def gm_websocket(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    try:
        while True:
            await ws.receive_text()  # keep connection alive
    except WebSocketDisconnect:
        _ws_clients.remove(ws)


async def _broadcast(data: dict):
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


# ── Config API ────────────────────────────────────────────────────────────────

class MilestoneIn(BaseModel):
    name: str
    greeting: str = ""
    secret: str = ""
    secret_intro: str = ""

class SpiritIn(BaseModel):
    name: str
    system_prompt: str
    milestones: list[MilestoneIn]
    voice_id: str = ""

class PlayerIn(BaseModel):
    name: str
    spirit_id: str
    gender: str = ""
    age: str = ""

class FullConfig(BaseModel):
    players: list[dict]
    spirits: dict


@app.get("/api/config")
def get_config():
    return game_config.to_dict()


@app.post("/api/config/spirits/{spirit_id}/advance_day")
def advance_day(spirit_id: str):
    """Increment the day counter for a spirit, advancing its default milestone."""
    if spirit_id not in game_config.spirits:
        raise HTTPException(404, "Spirit not found")
    spirit = game_config.spirits[spirit_id]
    max_day = len(spirit.milestones) - 1
    if spirit.current_day >= max_day:
        return {"current_day": spirit.current_day, "advanced": False, "message": "Already at final milestone"}
    spirit.current_day += 1
    game_config.save()
    return {"current_day": spirit.current_day, "advanced": True}


@app.post("/api/config/players")
def add_player(req: PlayerIn):
    player_id = _slugify(req.name)
    if player_id in game_config.players:
        raise HTTPException(409, f"Player id '{player_id}' already exists")
    if req.spirit_id not in game_config.spirits:
        raise HTTPException(404, f"Spirit '{req.spirit_id}' not found")
    from .game_state import Player
    game_config.players[player_id] = Player(id=player_id, name=req.name, spirit_id=req.spirit_id, gender=req.gender, age=req.age)
    game_config.save()
    return {"id": player_id}


@app.delete("/api/config/players/{player_id}")
def delete_player(player_id: str):
    if player_id not in game_config.players:
        raise HTTPException(404, "Player not found")
    del game_config.players[player_id]
    game_config.save()
    return {"status": "deleted"}


@app.post("/api/config/spirits")
def add_spirit(req: SpiritIn):
    from .game_state import Spirit, Milestone
    spirit_id = _slugify(req.name)
    milestones = [
        Milestone(
            id=i,
            name=m.name,
            greeting=m.greeting or None,
            secret=m.secret or None,
            secret_intro=m.secret_intro or None,
        )
        for i, m in enumerate(req.milestones)
    ]
    game_config.spirits[spirit_id] = Spirit(
        name=req.name,
        system_prompt=req.system_prompt,
        full_system_prompt=req.system_prompt,
        milestones=milestones,
        voice_id=req.voice_id,
    )
    game_config.save()
    return {"id": spirit_id}


@app.put("/api/config/spirits/{spirit_id}")
def update_spirit(spirit_id: str, req: SpiritIn):
    from .game_state import Spirit, Milestone
    if spirit_id not in game_config.spirits:
        raise HTTPException(404, "Spirit not found")
    milestones = [
        Milestone(
            id=i,
            name=m.name,
            greeting=m.greeting or None,
            secret=m.secret or None,
            secret_intro=m.secret_intro or None,
        )
        for i, m in enumerate(req.milestones)
    ]
    existing = game_config.spirits[spirit_id]
    game_config.spirits[spirit_id] = Spirit(
        name=req.name,
        system_prompt=req.system_prompt,
        full_system_prompt=req.system_prompt,
        milestones=milestones,
        voice_id=req.voice_id,
        current_day=existing.current_day,
    )
    game_config.save()
    return {"status": "updated"}


@app.delete("/api/config/spirits/{spirit_id}")
def delete_spirit(spirit_id: str):
    if spirit_id not in game_config.spirits:
        raise HTTPException(404, "Spirit not found")
    # Prevent deletion if any player references this spirit
    referencing = [p.name for p in game_config.players.values() if p.spirit_id == spirit_id]
    if referencing:
        raise HTTPException(409, f"Spirit is assigned to player(s): {', '.join(referencing)}")
    del game_config.spirits[spirit_id]
    game_config.save()
    return {"status": "deleted"}


# ── Conversation logging ──────────────────────────────────────────────────────

def _save_conversation():
    """Write the current session's conversation to a timestamped log file."""
    if not session.player or not session.spirit:
        return
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = logs_dir / f"{timestamp}_{session.player.name}.txt"
    lines = [
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Player: {session.player.name}",
        f"Spirit: {session.spirit.name}",
        f"Milestone: {session.current_milestone}",
        "",
    ]
    history = session.conversation_history
    i = 0
    while i < len(history):
        msg = history[i]
        if msg["role"] == "user" and not msg["content"].startswith("[SYSTEM NOTIFICATION"):
            lines.append(f"Player: {msg['content']}")
            if i + 1 < len(history) and history[i + 1]["role"] == "assistant":
                lines.append(f"Spirit: {history[i + 1]['content']}")
                i += 2
                continue
        i += 1
    filename.write_text("\n".join(lines), encoding="utf-8")


# ── Static frontend ───────────────────────────────────────────────────────────

app.mount("/", StaticFiles(directory="frontend/static", html=True), name="static")