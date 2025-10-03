# AutoFix SonarQube + A2A + Property-Based Testing

Pipeline ponta-a-ponta inspirado no framework PGS (Property-Guided Synthesis) que integra SonarQube, agentes LLM orquestrados via protocolo **Agent2Agent (A2A)** e testes baseados em propriedades (Hypothesis) para gerar correções automáticas.

## Visão Geral

1. Executa `sonar-scanner` para análise estática inicial.
2. Coleta issues do projeto no SonarQube.
3. Para cada issue é criada uma sessão A2A em memória:
   - **Requester Agent** monta o contexto com código, issue e feedback acumulado.
   - **Fixer Agent** propõe um patch (diff) e o aplica via `git apply`.
   - **Tester Agent** roda `pytest` com Hypothesis e converte os logs em feedback semântico.
   - **Sonar Agent** reexecuta o `sonar-scanner` para validar a correção.
   - **PR Agent** (opcional) cria branch, commit e Pull Request se tudo passar.
   - Caso qualquer etapa falhe, o Requester agrega o feedback e dispara nova tentativa até `MAX_ROUNDS`.

## Estrutura

```
.
├─ app/
│  ├─ main.py                 # Orquestrador que inicia sessões A2A
│  ├─ sonarqube_client.py     # Cliente REST SonarQube
│  ├─ patcher.py              # Aplicação de patches via git
│  ├─ utils.py                # sonar-scanner, git, PR helpers
│  └─ a2a/
│     ├─ protocol.py          # Estruturas de estado e tipos de mensagem
│     ├─ requester_agent.py   # Gera contexto para o Fixer
│     ├─ fixer_agent.py       # Gera patch em diff unificado
│     ├─ tester_agent.py      # Pytest/Hypothesis + resumo LLM
│     ├─ sonar_agent.py       # Revalida com SonarQube
│     └─ pr_agent.py          # Cria branch/commit/PR
├─ app/a2a_runtime.py         # Carrega `python-a2a` ou fallback embutido
├─ src/sample_module.py
├─ tests/
│  ├─ test_properties.py
│  └─ conftest.py
├─ .github/workflows/autofix.yml
├─ requirements.txt
├─ sonar-project.properties
└─ .env.example
```

## Pré-requisitos

- Python 3.11+
- Git e `sonar-scanner` (ou Docker)
- SonarQube em execução com projeto configurado
- Biblioteca Python A2A (`pip install "python-a2a[openai]"` ou use `requirements.txt`)
- Chave de API OpenAI (`OPENAI_API_KEY`)
- (Opcional) Token GitHub com acesso ao repositório

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # e preencha os valores
```

## Execução Local

1. Suba o SonarQube (ex: `docker run -d -p 9000:9000 sonarqube:community`).
2. Configure tokens/variáveis (`SONARQUBE_URL`, `SONARQUBE_TOKEN`, `OPENAI_API_KEY`, `SONAR_PROJECT_KEY`).
3. Rode uma análise inicial: `sonar-scanner`.
4. Execute o pipeline:

```bash
python -m app.main
```

A sessão A2A gera correções, valida com `pytest` + Hypothesis, reexecuta o Sonar e opcionalmente cria um Pull Request.

> Nota: o runtime tenta usar a biblioteca oficial `python-a2a`. Se ela não estiver instalada, um fallback mínimo em `app/a2a_runtime.py` garante que o fluxo funcione para testes locais.

### Uso com LLM local

- Instale `llama-cpp-python` (já listado em `requirements.txt`).
- Baixe o modelo GGUF desejado e configure em `.env`:
  - `LLM_PROVIDER=local`
  - `LLM_LOCAL_MODEL_PATH=/caminho/para/modelo.gguf`
  - Ajuste `LLM_LOCAL_CTX`, `LLM_LOCAL_THREADS` e `LLM_LOCAL_CHAT_FORMAT` conforme necessário.
- Com esse modo, as chamadas de Requester/Fixer/Tester usam `llama-cpp-python` em vez da API da OpenAI.
- Para continuar usando OpenAI, mantenha `LLM_PROVIDER=openai` e defina `OPENAI_API_KEY`.

## GitHub Actions

Workflow `.github/workflows/autofix.yml` executa o mesmo fluxo no CI:

- Checkout do repositório
- Instala dependências
- Sonar scan prévio
- Roda `python -m app.main`

Secrets necessários: `OPENAI_API_KEY`, `SONARQUBE_URL`, `SONARQUBE_TOKEN`, `SONAR_PROJECT_KEY`.

## Testes

```bash
pytest -q
```

## Licença

MIT
