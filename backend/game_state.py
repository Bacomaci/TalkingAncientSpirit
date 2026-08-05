"""
Game state management — tracks active player, milestone, and spirit context.
"""
from dataclasses import dataclass, field
from typing import Optional
import yaml
from pathlib import Path
import re


@dataclass
class Milestone:
    id: int
    name: Optional[str] = None
    secret: Optional[str] = None
    secret_intro: Optional[str] = None


@dataclass
class Spirit:
    name: str
    system_prompt: str
    full_system_prompt: str  # common_lore + personality + milestones up to current_day
    milestones: list[Milestone]
    listen_seconds: int = 10
    current_day: int = 0
    voice_name: str = ""
    max_exchanges: int = 20
    player_notes: str = ""


@dataclass
class Player:
    id: str
    name: str
    spirit_id: str
    gender: str
    age: str
    keywords: list[str] = field(default_factory=list)


@dataclass
class SessionState:
    player: Optional[Player] = None
    spirit: Optional[Spirit] = None
    session_active: bool = False
    gm_context: str = ""
    conversation_history: list = field(default_factory=list)
    greeting: Optional[str] = None
    greeting_audio_b64: Optional[str] = None
    exchange_count: int = 0

    def reset_conversation(self):
        self.conversation_history = []
        self.exchange_count = 0


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", name.lower().strip())


def _build_full_system_prompt(common_lore: str, system_prompt: str, milestones: list[Milestone], current_day: int, player_notes: str = "") -> str:
    """Build the full static system prompt: common lore + personality + all milestone secrets up to current_day + player memory."""
    parts = [common_lore, "\n\n<your_personality>\n" + system_prompt + "\n</your_personality>"]
    active = [m for m in milestones if m.id <= current_day and m.secret]
    if active:
        parts.append("\n\n<accumulated_knowledge>")
        for m in active:
            if m.secret:
                parts.append(f"\n{m.secret}")
        parts.append("\n</accumulated_knowledge>")
    if player_notes:
        parts.append(f"\n\n<player_memory>\n{player_notes}\n</player_memory>")
    return "".join(parts)


class GameConfig:
    def __init__(self, config_path: str = "config/game_config.yaml"):
        self._config_path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self.players: dict[str, Player] = {}
        for p in data.get("players", []):
            self.players[p["id"]] = Player(
                id=p["id"],
                name=p["name"],
                spirit_id=p["spirit"],
                gender=p["gender"],
                age=p["age"],
                keywords=p.get("keywords", []),
            )

        self.voices: dict[str, str] = data.get("voices", {})
        self.common_lore: str = data.get("common_lore", "")

        self.spirits: dict[str, Spirit] = {}
        for sid, sdata in data.get("spirits", {}).items():
            if sdata.get("milestones"):
                milestones = [
                    Milestone(
                        id=m["id"],
                        name=m.get("name") or None,
                        secret=m.get("secret") or None,
                        secret_intro=m.get("secret_intro") or None,
                    )
                    for m in sdata.get("milestones", [])
                ]
            else:
                milestones = [Milestone(0)]
            current_day = sdata.get("current_day", 0)
            player_notes = sdata.get("player_notes", "")
            full_system_prompt = _build_full_system_prompt(
                self.common_lore, sdata["system_prompt"], milestones, current_day, player_notes
            )
            self.spirits[sid] = Spirit(
                name=sdata["name"],
                system_prompt=sdata["system_prompt"],
                full_system_prompt=full_system_prompt,
                milestones=milestones,
                listen_seconds=sdata.get("listen_seconds", 10),
                current_day=current_day,
                voice_name=sdata.get("voice_name", ""),
                max_exchanges=sdata.get("max_exchanges", 15),
                player_notes=player_notes,
            )

    def resolve_voice_id(self, spirit: "Spirit") -> str:
        return self.voices.get(spirit.voice_name, "")

    def get_player(self, player_id: str) -> Optional[Player]:
        return self.players.get(player_id)

    def get_spirit_for_player(self, player: Player) -> Optional[Spirit]:
        return self.spirits.get(player.spirit_id)

    def to_dict(self) -> dict:
        players_list = [
            {"id": p.id, "name": p.name, "spirit": p.spirit_id, "gender": p.gender, "age": p.age, "keywords": p.keywords}
            for p in self.players.values()
        ]
        spirits_dict = {}
        for sid, s in self.spirits.items():
            spirits_dict[sid] = {
                "name": s.name,
                "system_prompt": s.system_prompt,
                "listen_seconds": s.listen_seconds,
                "current_day": s.current_day,
                "voice_name": s.voice_name,
                "max_exchanges": s.max_exchanges,
                "player_notes": s.player_notes,
                "milestones": [
                    {
                        "id": m.id,
                        "name": m.name,
                        **({"secret": m.secret} if m.secret else {"secret": None}),
                        **({"secret_intro": m.secret_intro} if m.secret_intro else {}),
                    }
                    for m in s.milestones
                ],
            }
        return {"voices": self.voices, "players": players_list, "spirits": spirits_dict, "common_lore": self.common_lore}

    def save(self):
        with open(self._config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    def reload(self):
        self.__init__(self._config_path)
