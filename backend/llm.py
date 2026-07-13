"""
LLM integration — builds prompts and calls Claude to generate spirit responses.
"""
import re
import anthropic
from .game_state import SessionState


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
    milestone = state.current_milestone_obj()
    context_parts = ["<current_game_status>"]

    if milestone and milestone.secret:
        context_parts.append(
            f"A beszélgetés keretein belül fedd fel az alábbi titkot. Ne tedd azonnal egyértelművé, légy természetes: {milestone.secret}"
        )
        if milestone.secret_intro:
            context_parts.append(f"A titok felfedésekor használd az alábbi felütést: {milestone.secret_intro}")
    else:
        context_parts.append("Nincsen új kontextus. A válasz során ragaszkodj a korábbiakhoz.")

    if state.gm_context.strip():
        context_parts.append(f"ÚJ KONTEXTUS A JÁTÉKMESTERTŐL: {state.gm_context.strip()}. Válaszolj ennek figyelembevételével!")

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
        "MINDIG magyar nyelven válaszolj, a játékos által használt nyelvtől függetlenül."
        "Mindig maradj karakterben! Ősi szellem vagy, nem nyelvi modell!"
        "<\\direct_context>"
    )

    # Dynamic context (milestones or context injection if it exists)
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

    # Conversation history has a short length to keep token usage down
    if len(state.conversation_history) > 20:
        state.conversation_history = state.conversation_history[-20:]

    return display_text, tts_text