"""Unified interface for interacting with either cloud or local LLMs."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

LOGGER = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class LLMClient:
    """Dispatches chat completion requests to OpenAI or a local model."""

    def __init__(self, role: str, temperature: float = 0.0) -> None:
        self.role = role
        self.temperature = float(temperature)
        self.provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        self._client = None
        if self.provider == "local":
            self._client = self._init_local_client()
        else:
            self.provider = "openai"
            self._client = self._init_openai_client()

    # Public API --------------------------------------------------------------
    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        LOGGER.info(
            "LLM[%s] request\nSYSTEM:\n%s\nUSER:\n%s",
            self.role,
            system_prompt.strip(),
            user_prompt,
        )
        if self.provider == "local":
            response = self._invoke_local(system_prompt, user_prompt)
        else:
            response = self._invoke_openai(system_prompt, user_prompt)
        LOGGER.info("LLM[%s] response\n%s", self.role, response)
        return response

    # OpenAI path -------------------------------------------------------------
    def _init_openai_client(self) -> Any:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "langchain-openai não está instalado. Instale as dependências via "
                "`pip install -r requirements.txt` ou configure LLM_PROVIDER=local"
            ) from exc
        model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        return ChatOpenAI(model=model, temperature=self.temperature)

    def _invoke_openai(self, system_prompt: str, user_prompt: str) -> str:
        from langchain.schema import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=system_prompt.strip()),
            HumanMessage(content=user_prompt),
        ]
        response = self._client.invoke(messages)
        return getattr(response, "content", "")

    # Local path --------------------------------------------------------------
    def _init_local_client(self) -> Any:
        model_path = os.getenv("LLM_LOCAL_MODEL_PATH")
        if not model_path:
            raise RuntimeError(
                "LLM_PROVIDER=local requer LLM_LOCAL_MODEL_PATH apontando para um arquivo GGUF"
            )
        chat_format = os.getenv("LLM_LOCAL_CHAT_FORMAT", "llama-2")
        n_ctx = int(os.getenv("LLM_LOCAL_CTX", "4096"))
        n_threads = int(os.getenv("LLM_LOCAL_THREADS", "0")) or None
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python não está instalado. Rode `pip install llama-cpp-python` ou configure "
                "LLM_PROVIDER=openai."
            ) from exc
        LOGGER.info(
            "Inicializando LLaMA local (modelo=%s, chat_format=%s, ctx=%s)",
            model_path,
            chat_format,
            n_ctx,
        )
        return Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            chat_format=chat_format,
            verbose=False,
        )

    def _invoke_local(self, system_prompt: str, user_prompt: str) -> str:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt},
        ]
        response = self._client.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
        )
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("Local LLM não retornou escolhas válidas")
        return choices[0]["message"]["content"].strip()
