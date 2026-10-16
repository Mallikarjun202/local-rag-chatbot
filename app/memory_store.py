from typing import Dict, List
from . import config


class MemoryStore:
    def __init__(self):
        self._sessions: Dict[str, List[Dict[str, str]]] = {}
        self._enabled: Dict[str, bool] = {}

    def is_enabled(self, session_id: str) -> bool:
        # Memory defaults to ON for a brand-new session.
        return self._enabled.get(session_id, True)

    def set_enabled(self, session_id: str, enabled: bool) -> None:
        self._enabled[session_id] = enabled

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return list(self._sessions.get(session_id, []))

    def add_message(self, session_id: str, role: str, content: str) -> None:
        history = self._sessions.setdefault(session_id, [])
        history.append({"role": role, "content": content})
        limit = int(config.MAX_MEMORY_MESSAGES)
        if len(history) > limit:
            del history[: len(history) - limit]

    def clear(self, session_id: str) -> None:
        self._sessions[session_id] = []


# Single shared instance used across the app
memory_store = MemoryStore()
