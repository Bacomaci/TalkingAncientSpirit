"""
FastAPI backend — REST + WebSocket server for the LARP spirit bot.
"""
import os
import base64
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv

from .game_state import GameConfig, SessionState, _slugify
from .llm import get_spirit_response
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

    session.player = player
    session.spirit = spirit
    session.session_active = True
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
        audio = synthesize_speech(reply_tts, ELEVENLABS_VOICE_ID, ELEVENLABS_API_KEY, TTS_LANGUAGE_CODE)
    except Exception as e:
        raise HTTPException(500, f"TTS error: {e}") from e

    audio_b64 = base64.b64encode(audio).decode()

    # Push transcript to GM console via WebSocket
    await _broadcast({
        "type": "transcript",
        "player_said": player_text,
        "spirit_said": reply_display,
    })

    return {
        "player_said": player_text,
        "spirit_said": reply_display,
        "audio_base64": audio_b64,
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

class PlayerIn(BaseModel):
    name: str
    spirit_id: str

class FullConfig(BaseModel):
    players: list[dict]
    spirits: dict


@app.get("/api/config")
def get_config():
    return game_config.to_dict()


@app.post("/api/config/players")
def add_player(req: PlayerIn):
    player_id = _slugify(req.name)
    if player_id in game_config.players:
        raise HTTPException(409, f"Player id '{player_id}' already exists")
    if req.spirit_id not in game_config.spirits:
        raise HTTPException(404, f"Spirit '{req.spirit_id}' not found")
    from .game_state import Player
    game_config.players[player_id] = Player(id=player_id, name=req.name, spirit_id=req.spirit_id, gender="TODO", age="TODO")
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
    game_config.spirits[spirit_id] = Spirit(
        name=req.name,
        system_prompt=req.system_prompt,
        full_system_prompt=req.system_prompt,
        milestones=milestones,
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


# ── Static frontend ───────────────────────────────────────────────────────────

app.mount("/", StaticFiles(directory="frontend/static", html=True), name="static")