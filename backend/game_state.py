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
    name: str
    secret: Optional[str]
    secret_intro: Optional[str] = None
    greeting: Optional[str] = None


@dataclass
class Spirit:
    name: str
    system_prompt: str
    full_system_prompt:str
    milestones: list[Milestone]
    listen_seconds: int = 10


@dataclass
class Player:
    id: str
    name: str
    spirit_id: str
    gender: str
    age: str


@dataclass
class SessionState:
    player: Optional[Player] = None
    spirit: Optional[Spirit] = None
    current_milestone: int = 0
    session_active: bool = False
    gm_context: str = ""          # Extra context injected by GM
    conversation_history: list = field(default_factory=list)
    greeting: Optional[str] = None
    greeting_audio_b64: Optional[str] = None

    def current_milestone_obj(self) -> Optional[Milestone]:
        if self.spirit is None:
            return None
        milestones = self.spirit.milestones
        idx = min(self.current_milestone, len(milestones) - 1)
        return milestones[idx]

    def advance_milestone(self) -> bool:
        """Returns True if there was a next milestone to advance to."""
        if self.spirit is None:
            return False
        if self.current_milestone < len(self.spirit.milestones) - 1:
            self.current_milestone += 1
            return True
        return False

    def reset_conversation(self):
        self.conversation_history = []


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", name.lower().strip())


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
                age=p["age"]
            )

        common_lore = data.get("common_lore", "")

        self.spirits: dict[str, Spirit] = {}
        for sid, sdata in data.get("spirits", {}).items():
            milestones = [
                Milestone(
                    id=m["id"],
                    name=m["name"],
                    secret=m.get("secret"),
                    secret_intro=m.get("secret_intro"),
                    greeting=m.get("greeting"),
                )
                for m in sdata.get("milestones", [])
            ]

            full_system_prompt = common_lore + "\n\n<your_personality>\n" + sdata["system_prompt"] + "\n<\\your_personality>"

            self.spirits[sid] = Spirit(
                name=sdata["name"],
                system_prompt = sdata["system_prompt"],
                full_system_prompt=full_system_prompt,
                milestones=milestones,
                listen_seconds=sdata.get("listen_seconds", 10),
            )

    def get_player(self, player_id: str) -> Optional[Player]:
        return self.players.get(player_id)

    def get_spirit_for_player(self, player: Player) -> Optional[Spirit]:
        return self.spirits.get(player.spirit_id)

    def to_dict(self) -> dict:
        """Serialize config back to the YAML-compatible dict structure."""
        players_list = [
            {"id": p.id, "name": p.name, "spirit": p.spirit_id}
            for p in self.players.values()
        ]
        spirits_dict = {}
        for sid, s in self.spirits.items():
            spirits_dict[sid] = {
                "name": s.name,
                "system_prompt": s.system_prompt,
                "listen_seconds": s.listen_seconds,
                "milestones": [
                    {
                        "id": m.id,
                        "name": m.name,
                        **({"greeting": m.greeting} if m.greeting else {}),
                        **({"secret": m.secret} if m.secret else {"secret": None}),
                        **({"secret_intro": m.secret_intro} if m.secret_intro else {}),
                    }
                    for m in s.milestones
                ],
            }
        return {"players": players_list, "spirits": spirits_dict}

    def save(self):
        with open(self._config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    def reload(self):
        self.__init__(self._config_path)
