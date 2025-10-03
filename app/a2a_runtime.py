"""Runtime utilities to load the official python-a2a library with a fallback."""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Dict, Optional
from uuid import uuid4

__all__ = ["Agent", "Session", "Message", "using_official_impl"]

using_official_impl = False


try:
    _lib = import_module('python_a2a')
    if all(hasattr(_lib, name) for name in ('Agent', 'Session', 'Message')):
        Agent = _lib.Agent  # type: ignore[attr-defined]
        Session = _lib.Session  # type: ignore[attr-defined]
        Message = _lib.Message  # type: ignore[attr-defined]
        using_official_impl = True
    else:
        raise AttributeError('python_a2a missing Agent/Session/Message')
except (ModuleNotFoundError, AttributeError):
    from collections import deque

    @dataclass
    class Message:  # pragma: no cover - trivial data holder
        type: str
        to: str
        from_: str
        body: Dict[str, Any]
        id: str = field(default_factory=lambda: str(uuid4()))

    class Session:
        def __init__(self) -> None:
            self._agents: Dict[str, Agent] = {}
            self._queue: deque[Message] = deque()
            self.state: Dict[str, Any] = {}
            self._processing = False
            self._terminated = False

        def register(self, agent: "Agent") -> None:
            self._agents[agent.name] = agent
            agent._session = self

        def send(self, message: Message) -> None:
            if self._terminated:
                return
            self._queue.append(message)
            if not self._processing:
                self._drain()

        def end(self) -> None:
            self._terminated = True
            self._queue.clear()

        def _drain(self) -> None:
            self._processing = True
            try:
                while self._queue and not self._terminated:
                    msg = self._queue.popleft()
                    agent = self._agents.get(msg.to)
                    if not agent:
                        raise ValueError(f"Agent '{msg.to}' not registered")
                    agent.on_message(msg, self)
            finally:
                self._processing = False

    class Agent:
        def __init__(self, name: str) -> None:
            self.name = name
            self._session: Optional[Session] = None

        def on_message(self, message: Message, session: Session) -> None:  # pragma: no cover - interface
            raise NotImplementedError

        @property
        def session(self) -> Session:
            if not self._session:
                raise RuntimeError("Agent not bound to a session")
            return self._session
