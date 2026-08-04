"""
LLM integration — builds prompts and calls Claude to generate spirit responses.
"""
import re
import anthropic
from .game_state import SessionState

MAX_EXCHANGES = 12  # fallback default, overridden per spirit via config

_FAREWELL_SENTINEL = "menj békével"

_NOTES_MAX_CHARS = 400


def summarize_session(history: list, player_name: str, existing_notes: str, client: anthropic.Anthropic) -> str:
    """
    Extract 1-2 sentences of what the spirit learned about the player this session.
    Appends to existing_notes, trimmed to _NOTES_MAX_CHARS total.
    Returns the updated notes string (empty string on failure).
    """
    if not history:
        return existing_notes

    # Build a plain transcript from history for the summarizer
    lines = []
    for msg in history:
        if msg["role"] == "user" and not msg["content"].startswith("[SYSTEM NOTIFICATION"):
            lines.append(f"Player: {msg['content']}")
        elif msg["role"] == "assistant":
            lines.append(f"Spirit: {msg['content']}")
    if not lines:
        return existing_notes

    transcript = "\n".join(lines)
    prompt = (
        f"The following is a conversation between a spirit and a player named {player_name}.\n\n"
        f"{transcript}\n\n"
        "In 1-2 sentences, summarize only concrete facts the spirit learned about this player "
        "(their background, goals, relationships, secrets revealed). "
        "Be very brief. Write in third person, past tense, in Hungarian. "
        "If nothing meaningful was learned, reply with an empty string."
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
        new_note = response.content[0].text.strip()  # type: ignore
    except Exception:
        return existing_notes

    if not new_note:
        return existing_notes

    combined = (existing_notes + "\n" + new_note).strip() if existing_notes else new_note
    # Trim from the front if over budget, keeping the most recent notes
    if len(combined) > _NOTES_MAX_CHARS:
        combined = combined[-_NOTES_MAX_CHARS:].lstrip()
    return combined


def spirit_said_farewell(reply_text: str) -> bool:
    return _FAREWELL_SENTINEL in reply_text.lower()


def reformat_stage_directions(text: str) -> str:
    """Convert *stage directions* to [stage directions] for display and strip them for TTS."""
    return re.sub(r'\*([^*]+)\*', r'[\1]', text)


def strip_stage_directions(text: str) -> str:
    """Remove [stage directions] entirely — used before sending text to TTS."""
    return re.sub(r'\[[^\]]+\]', '', text).strip()


def build_dynamic_context_message(state: SessionState) -> str:
    """
    Builds a dynamic context string containing game status that changes over time.
    This goes into the messages list to keep the main system prompt strictly static and cacheable.
    """
    context_parts = ["<current_game_status>"]

    if state.gm_context.strip():
        context_parts.append(f"ÚJ KONTEXTUS A JÁTÉKMESTERTŐL: {state.gm_context.strip()}. Válaszolj ennek figyelembevételével!")

    # Weariness cues as the conversation nears its end
    max_ex = state.spirit.max_exchanges if state.spirit else MAX_EXCHANGES
    remaining = max_ex - state.exchange_count
    if remaining <= 3 and remaining > 0:
        context_parts.append(
            "A szellem ereje fogytán van. Éreztesd, hogy egyre nehezebb fenntartani a kapcsolatot — "
            "fáradtság, távolodás, elmosódó szavak. Még válaszolj, de jelezd a közelgő véget."
        )
    elif remaining == 0:
        context_parts.append(
            "Ez az utolsó válasz. Búcsúzz el méltósággal — adj okot a távozásra (pl. gyengül a kötelék, "
            "visszahív a túlvilág, elfogyott az erőd). Zárj le mindent."
        )

    context_parts.append("</current_game_status>")
    return "\n".join(context_parts)


def get_spirit_response(state: SessionState, player_utterance, client: anthropic.Anthropic, language: str = "Hungarian") -> tuple[str, str]:
    """Returns (display_text, tts_text). Uses Anthropic Prompt Caching for the system lore. Display text has [stage directions], TTS text has them stripped."""
    spirit = state.spirit
    player = state.player

    assert spirit is not None
    assert player is not None

    # Global static system prompt plus character guidance originating from config.yaml
    base_system = spirit.full_system_prompt.strip()

    # Static system prompt plus character guidance
    static_system_prompt = (
        f"{base_system}\n\n"
        "<direct_context>"
        f"Most egy távoli éothéod utóddal beszélsz, akinek a neve {player.name}. Ő egy olyan ember, aki a holtakról szóló emlékek őrzésének szentelte életét."
        f"Beszélgetőtársad neme {player.gender}, életkora {player.age}."
        "Légy kíváncsi iránta, tudd meg, hogy ki ő, mi járatban van, és mi a tragédiája!"
        "MINDIG magyar nyelven válaszolj, a játékos által használt nyelvtől függetlenül."
        "Mindig maradj karakterben! Ősi szellem vagy, nem nyelvi modell!"
        "Ha a játékos el akar búcsúzni, vagy ha természetes módon lezárult a beszélgetés, mondj egy méltóságteljes búcsút, és zárd a mondandódat a következő szavakkal: \"menj békével\". Ez jelzi a rendszernek, hogy véget ért a találkozó."
        "</direct_context>"
    )

    # Dynamic context (milestones, weariness cues, or GM context injection)
    dynamic_context = build_dynamic_context_message(state)

    # Assemble messages
    messages = []

    # Dynamic context presented as a system notification coming from the user
    messages.append({
        "role": "user",
        "content": f"[SYSTEM NOTIFICATION: {dynamic_context}]"
    })

    # Then last 10 exchanges
    messages.extend(state.conversation_history)

    # Then current message of the player
    messages.append({"role": "user", "content": player_utterance})

    # Prompt Claude-3-5-Sonnet-20241022 (or haiku) supports prompt caching syntax
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": static_system_prompt,
                "cache_control": {"type": "ephemeral"}  # Activates prompt caching for the static context
            }
        ],
        messages=messages, # Dynamic context, conversation history and current message
    )

    raw_reply = response.content[0].text # type: ignore
    display_text = reformat_stage_directions(raw_reply)
    tts_text = strip_stage_directions(display_text)

    # Player utterance and reply get saved to history
    state.conversation_history.append({"role": "user", "content": player_utterance})
    state.conversation_history.append({"role": "assistant", "content": raw_reply})
    state.exchange_count += 1

    # Conversation history has a short length to keep token usage down
    if len(state.conversation_history) > 20:
        state.conversation_history = state.conversation_history[-20:]

    return display_text, tts_text