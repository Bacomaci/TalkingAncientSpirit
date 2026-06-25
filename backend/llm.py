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


def build_system_prompt(state: SessionState, language: str = "Hungarian") -> str:
    spirit = state.spirit
    milestone = state.current_milestone_obj()

    parts = [spirit.system_prompt.strip()]

    parts.append(f"\nYou are speaking with {state.player.name}, your descendant.")
    parts.append(f"\nALWAYS respond in {language}, regardless of what language the player uses.")

    if milestone and milestone.secret:
        parts.append(
            f"\nMILESTONE REACHED: '{milestone.name}'. "
            f"You may now reveal this secret — but do so naturally, within the flow of conversation. "
            f"If the player has not yet asked about it, weave it in. "
            f"Secret to reveal: {milestone.secret}"
        )
        if milestone.secret_intro:
            parts.append(
                f"When introducing the secret, use this tone: {milestone.secret_intro}"
            )
    else:
        parts.append(
            "\nNo secret is ready to be revealed at this time. Respond to questions naturally "
            "but do not invent secrets or lore that hasn't been established."
        )

    if state.gm_context.strip():
        parts.append(
            f"\nGAME MASTER CONTEXT (use this to inform your responses, do not quote it directly): "
            f"{state.gm_context.strip()}"
        )

    return "\n".join(parts)


def get_spirit_response(state: SessionState, player_utterance: str, client: anthropic.Anthropic, language: str = "Hungarian") -> tuple[str, str]:
    """Returns (display_text, tts_text). Display text has [stage directions], TTS text has them stripped."""
    system_prompt = build_system_prompt(state, language)

    messages = list(state.conversation_history)
    messages.append({"role": "user", "content": player_utterance})

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=system_prompt,
        messages=messages,
    )

    raw_reply = response.content[0].text
    display_text = reformat_stage_directions(raw_reply)
    tts_text = strip_stage_directions(display_text)

    state.conversation_history.append({"role": "user", "content": player_utterance})
    state.conversation_history.append({"role": "assistant", "content": raw_reply})

    # Keep history bounded to avoid token bloat (last 10 exchanges)
    if len(state.conversation_history) > 20:
        state.conversation_history = state.conversation_history[-20:]

    return display_text, tts_text
