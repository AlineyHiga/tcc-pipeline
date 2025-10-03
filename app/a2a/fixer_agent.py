"""Fixer agent: generates patches in response to fix requests."""
from __future__ import annotations

import logging

from app.a2a_runtime import Agent, Message, Session
from app.patcher import PatchApplicationError, apply_patch

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """
Você é o Fixer Agent. Recebe contexto estruturado do Requester com detalhes do bug
(mensagens Sonar, feedback do Tester, trechos de código) e deve gerar um patch
em diff unificado.
Regras:
- Utilize apenas caminhos relativos.
- Responda exclusivamente com o diff.
- O patch deve aplicar limpo com `git apply`.
"""


class FixerAgent(Agent):
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1) -> None:
        from app.llm_client import LLMClient

        super().__init__(name="Fixer")
        self.llm = LLMClient(role="fixer", temperature=temperature)

    def on_message(self, message: Message, session: Session) -> None:
        if message.type != "fix_request":
            LOGGER.debug("Fixer ignorou mensagem %s", message.type)
            return
        context = message.body.get("context", "")
        attempt = message.body.get("attempt")
        LOGGER.info("Fixer gerando patch (tentativa %s)", attempt)
        patch = self._generate_patch(context)
        session.state["patch"] = patch
        try:
            apply_patch(patch)
            session.state["fixer_summary"] = "Patch aplicado com sucesso"
        except PatchApplicationError as exc:
            LOGGER.error("Falha ao aplicar patch: %s", exc)
            session.state["fixer_summary"] = f"Falha ao aplicar patch: {exc}"
            session.send(
                Message(
                    type="fix_failed",
                    from_=self.name,
                    to="Requester",
                    body={"error": str(exc)},
                )
            )
            return
        session.send(
            Message(
                type="test_request",
                from_=self.name,
                to="Tester",
                body={"attempt": attempt},
            )
        )

    def _generate_patch(self, context: str) -> str:
        return self.llm.invoke(SYSTEM_PROMPT, context).strip()
