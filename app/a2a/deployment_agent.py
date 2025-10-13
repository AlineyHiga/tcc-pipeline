"""Deployment agent responsible for opening GitHub pull requests via LangChain."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

try:
    from langchain_community.tools.github.toolkit import GitHubToolkit
except ImportError:  # pragma: no cover - compatibility shim
    try:
        from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
    except ImportError as exc:  # pragma: no cover - clearer guidance
        raise RuntimeError(
            "langchain-community não possui GitHubToolkit instalado. "
            "Atualize com `pip install --upgrade langchain-community`"
        ) from exc

try:
    from langchain_community.utilities.github import GitHubAPIWrapper
except ImportError as exc:  # pragma: no cover - clearer guidance
    raise RuntimeError(
        "langchain-community não possui GitHubAPIWrapper. Atualize com "
        "`pip install --upgrade langchain-community`."
    ) from exc

from app.a2a.protocol import Issue, State
from app.llm_client import DEFAULT_OPENAI_MODEL
from app.utils import create_pull_request, ensure_git_branch, git_commit_all

LOGGER = logging.getLogger(__name__)


class DeploymentAgent:
    """Agent that prepares PR content and calls GitHub Toolkit to open the PR."""

    def __init__(
        self,
        *,
        temperature: float = 0.1,
        base_branch: Optional[str] = None,
        repo: Optional[str] = None,
        token: Optional[str] = None,
        repo_root: Optional[Union[Path, str]] = None,
    ) -> None:
        self.base_branch = base_branch or os.getenv("BASE_BRANCH", "main")
        self.repo = repo or os.getenv("GITHUB_REPOSITORY")
        self.token = token or os.getenv("GITHUB_TOKEN")
        if not self.repo:
            raise RuntimeError("GITHUB_REPOSITORY não configurado para DeploymentAgent")
        if not self.token:
            raise RuntimeError("GITHUB_TOKEN não configurado para DeploymentAgent")

        self.autobranch_prefix = os.getenv("AUTO_BRANCH_PREFIX", "autofix")
        root = repo_root or os.getenv("AUTOFIX_TARGET_ROOT") or os.getenv("A2A_REPO_ROOT") or Path.cwd()
        self.repo_root = Path(root).expanduser().resolve()
        LOGGER.debug("Deployment repo root set to %s", self.repo_root)

        # LangChain components ------------------------------------------------
        openai_api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not openai_api_key:
            if base_url:
                LOGGER.debug(
                    "OPENAI_API_KEY não definido; usando valor padrão 'ollama' para endpoint custom %s",
                    base_url,
                )
                openai_api_key = "ollama"
            else:
                raise RuntimeError(
                    "OPENAI_API_KEY não configurado. Defina a chave ou configure OPENAI_BASE_URL + "
                    "OPENAI_API_KEY para um endpoint compatível."
                )

        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
            temperature=temperature,
            api_key=openai_api_key,
            base_url=base_url,
        )

        # GitHub Toolkit -----------------------------------------------------
        self.pull_request_tool = self._setup_pull_request_tool()
        self.tools = [self.pull_request_tool]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    # Public API --------------------------------------------------------------
    def invoke(self, state: State) -> State:
        issue = state["issue"]
        branch = state.get("branch") or f"{self.autobranch_prefix}/{issue.key}"

        LOGGER.info("Deployment agent preparando branch %s", branch)
        ensure_git_branch(branch, cwd=self.repo_root)
        git_commit_all(f"fix: auto remediation for {issue.key}", cwd=self.repo_root)

        test_logs = (state.get("test_logs") or "(Testes não executados)").strip()
        if len(test_logs) > 2000:
            LOGGER.debug("Truncating tester logs for prompt (original %d chars)", len(test_logs))
            test_logs = test_logs[:2000] + "\n... (truncated)"

        summary = ""
        pr_response: Optional[object] = None

        system_message = SystemMessage(
            content=(
                "Você é o Deployment Agent. Analise o contexto e crie um pull request usando o "
                "tool `pull_requests_create`. Preencha título, corpo, head (branch atual) e base "
                "com os dados fornecidos. Execute a ferramenta exatamente uma vez e, após receber "
                "o resultado, responda com um breve resumo em português."
            )
        )
        human_message = HumanMessage(
            content=(
                f"Issue: {issue.key} ({issue.rule}, severidade {issue.severity}).\n"
                f"Mensagem: {issue.message}.\n"
                f"Branch head: {branch}. Branch base: {self.base_branch}.\n"
                f"Feedback acumulado:\n{(state.get('feedback_log') or 'Sem feedback adicional.').strip()}\n\n"
                f"Resumo do Fixer:\n{state.get('fixer_summary', '(Resumo indisponível)')}\n\n"
                f"Logs do Tester:\n{test_logs}"
            )
        )

        messages = [system_message, human_message]

        for iteration in range(4):
            response = self.llm_with_tools.invoke(messages)
            LOGGER.debug(
                "LLM tool-call iteration %d response content=%s tool_calls=%s",
                iteration + 1,
                self._coerce_content_to_text(response.content),
                getattr(response, "tool_calls", []),
            )
            messages.append(response)
            tool_calls = getattr(response, "tool_calls", [])
            if not tool_calls:
                summary = self._coerce_content_to_text(response.content)
                break

            LOGGER.debug("Tool call solicitado (%d chamadas)", len(tool_calls))
            for call in tool_calls:
                name = call.get("name")
                args = call.get("args", {})
                LOGGER.debug("Executando tool %s com args %s", name, args)
                if name != self.pull_request_tool.name:
                    tool_result = {"error": f"Ferramenta {name} indisponível"}
                else:
                    # Garante que head/base estejam definidos
                    args.setdefault("head", branch)
                    args.setdefault("base", self.base_branch)
                    tool_result = self.pull_request_tool.invoke(args)
                    pr_response = tool_result
                LOGGER.debug(
                    "Tool %s responded with %s",
                    name,
                    tool_result,
                )
                tool_message = ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False, default=str),
                    tool_call_id=call.get("id"),
                )
                messages.append(tool_message)
        else:
            summary = "O modelo não retornou resposta final após usar a ferramenta."

        pr_url = self._extract_pr_url(pr_response)
        LOGGER.debug("Final tool response payload: %s", pr_response)
        if pr_url:
            summary = summary or "PR criado com sucesso"
        else:
            summary = summary or "Falha ao criar PR"

        LOGGER.info("LLM deployment response: %s", summary)

        state.update(
            {
                "branch": branch,
                "pr_url": pr_url or "",
                "deployment_summary": summary,
                "deployment_failed": not bool(pr_url),
            }
        )
        LOGGER.info("Deployment agent finalizado: %s", summary)
        if pr_url:
            LOGGER.info("Pull request criado: %s", pr_url)
        return state

    # Helpers ----------------------------------------------------------------
    def _setup_pull_request_tool(self):
        app_id = os.getenv("GITHUB_APP_ID")
        app_key = os.getenv("GITHUB_APP_PRIVATE_KEY")
        if not app_id or not app_key:
            LOGGER.info(
                "Credenciais de GitHub App não definidas; usando fallback baseado em token pessoal."
            )
            return self._build_pat_pull_request_tool()

        try:
            wrapper = GitHubAPIWrapper(
                github_repository=self.repo,
                github_app_id=os.getenv("GITHUB_APP_ID"),
                github_app_private_key=os.getenv("GITHUB_APP_PRIVATE_KEY"),
            )
            toolkit = GitHubToolkit.from_github_api_wrapper(wrapper)
            tool = self._resolve_pull_request_tool(toolkit)
            LOGGER.info("GitHubToolkit inicializado com credenciais de GitHub App")
            return tool
        except Exception as exc:
            LOGGER.warning(
                "Não foi possível inicializar GitHubToolkit via GitHub App (%s). "
                "Usando fallback baseado em token pessoal.",
                exc,
            )
            return self._build_pat_pull_request_tool()

    @staticmethod
    def _resolve_pull_request_tool(toolkit: GitHubToolkit):
        for tool in toolkit.get_tools():
            if tool.name == "pull_requests_create":
                return tool
        raise RuntimeError("GitHubToolkit não forneceu pull_requests_create tool")

    def _build_pat_pull_request_tool(self) -> StructuredTool:
        class CreatePRInput(BaseModel):
            title: str = Field(..., description="Título conciso do pull request")
            body: str = Field("", description="Corpo em Markdown descrevendo mudanças e testes")
            head: str = Field(..., description="Branch head com os commits")
            base: str = Field(..., description="Branch base que receberá o PR")

        def _call(data: CreatePRInput) -> str:
            try:
                response = create_pull_request(
                    title=data.title,
                    body=data.body,
                    head=data.head,
                    base=data.base,
                    repository=self.repo,
                    token=self.token,
                )
                return json.dumps(response, ensure_ascii=False)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Falha ao criar PR via REST: %s", exc)
                return json.dumps({"error": str(exc)}, ensure_ascii=False)

        return StructuredTool(
            name="pull_requests_create",
            description=(
                "Cria um pull request no repositório apontado em GITHUB_REPOSITORY usando o token "
                "personal access token informado em GITHUB_TOKEN."
            ),
            args_schema=CreatePRInput,
            func=lambda **kwargs: _call(CreatePRInput(**kwargs)),
        )

    @staticmethod
    def _extract_pr_url(response: object) -> Optional[str]:
        if not response:
            return None
        if isinstance(response, str):
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                LOGGER.debug("Resposta da ferramenta não é JSON: %s", response)
                return None
        elif isinstance(response, dict):
            data = response
        else:
            try:
                data = asdict(response)  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001
                LOGGER.debug("Resposta inesperada da ferramenta: %r", response)
                return None
        return data.get("html_url") or data.get("url")

    @staticmethod
    def _coerce_content_to_text(content: object) -> str:
        if isinstance(content, str):
            return content.strip()
        try:
            return json.dumps(content, ensure_ascii=False)
        except TypeError:
            return str(content)


def deployment_node(state: State) -> State:
    agent = DeploymentAgent(repo_root=state.get("repo_root"))
    return agent.invoke(state)


def _build_sample_state() -> State:
    issue = Issue(
        key=os.getenv("DEPLOYMENT_ISSUE_KEY", "LOCAL-ISSUE"),
        rule=os.getenv("DEPLOYMENT_ISSUE_RULE", "S0000"),
        severity=os.getenv("DEPLOYMENT_ISSUE_SEVERITY", "MAJOR"),
        component=os.getenv("DEPLOYMENT_ISSUE_COMPONENT", "src/app.py"),
        message=os.getenv("DEPLOYMENT_ISSUE_MESSAGE", "Fix generated automatically."),
        line=None,
    )
    return {
        "issue": issue,
        "fixer_summary": os.getenv("DEPLOYMENT_FIXER_SUMMARY", "Auto fix patch gerado."),
        "test_logs": os.getenv("DEPLOYMENT_TEST_LOGS", "pytest -q -> OK"),
        "feedback_log": os.getenv("DEPLOYMENT_FEEDBACK", ""),
    }


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    # Load environment variables from root and pipeline .env files when running standalone
    load_dotenv()
    repo_root_env = Path(__file__).resolve().parents[2] / ".env"
    if repo_root_env.exists():
        load_dotenv(repo_root_env)
    pipeline_env = Path(__file__).resolve().parents[1] / ".env"
    if pipeline_env.exists():
        load_dotenv(pipeline_env, override=True)
    try:
        agent = DeploymentAgent()
        result_state = agent.invoke(_build_sample_state())
        if result_state.get("pr_url"):
            print("PR criado com sucesso")
        else:
            print("Falha ao criar PR")
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Erro ao executar DeploymentAgent: %s", exc)
        raise
