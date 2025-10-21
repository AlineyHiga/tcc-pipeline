# AutoFix SonarQube + A2A + Property-Based Testing

Pipeline ponta-a-ponta inspirado no framework PGS (Property-Guided Synthesis) que integra SonarQube, agentes LLM orquestrados via protocolo Agent2Agent (A2A) e testes baseados em propriedades (Hypothesis) para gerar correções automáticas.

## Visão Geral

1. Executa `sonar-scanner` para análise estática inicial.
2. Coleta issues do projeto no SonarQube.
3. Para cada issue é criada uma sessão A2A/LangGraph:
   - **Property Agent** seleciona o arquivo afetado e orienta a geração de propriedades Hypothesis.
   - **Tester Agent (modo propriedades)** gera e executa apenas os testes de propriedades; se falhar, a pipeline encerra antes do Requester.
   - **Requester Agent** monta o contexto com código, issue e feedback acumulado.
   - **Fixer Agent** propõe um patch (diff) e o aplica via `git apply`.
   - **Tester Agent** roda `pytest` + propriedades geradas e converte os logs em feedback semântico.
   - **Sonar Agent** reexecuta o `sonar-scanner` para validar a correção.
   - **PR Agent** cria branch, commit e Pull Request se tudo passar.
   - Caso qualquer etapa falhe, o contexto é enriquecido com o feedback e o loop continua até `MAX_ROUNDS`.

## Estrutura

```
.
├─ app/
│  ├─ main.py                 # Orquestrador (LangGraph + A2A)
│  ├─ sonarqube_client.py     # Cliente REST SonarQube
│  ├─ patcher.py              # Aplicação de patches via git
│  ├─ utils.py                # sonar-scanner, git, PR helpers
│  ├─ llm_client.py           # Abstração para OpenAI ou LLM local (llama.cpp)
│  └─ a2a/
│     ├─ protocol.py          # Estruturas de estado e tipos
│     ├─ property_agent.py    # Seleciona componentes e inicia propriedades
│     ├─ requester_agent.py   # Gera contexto para o Fixer
│     ├─ fixer_agent.py       # Gera patch em diff unificado
│     └─ tester_agent.py      # Pytest/Hypothesis + resumo LLM
├─ src/
│  ├─ __init__.py
│  └─ sample_module.py        # Código exemplo para testes de propriedades
├─ tests/
│  ├─ conftest.py             # Configuração Hypothesis e sys.path
│  └─ test_properties.py      # Testes baseados em propriedades
├─ .github/workflows/autofix.yml
├─ sonar-project.properties
├─ .env.example
└─ README.md
```

## Pré-requisitos

- Python 3.11+
- Git e `sonar-scanner` (ou Docker para fallback)
- SonarQube em execução com projeto configurado
- Biblioteca Python A2A / LangGraph (`pip install -r requirements.txt`)
- Chave de API OpenAI (`OPENAI_API_KEY`) ou modelo local GGUF (`LLM_PROVIDER=local`)
- (Opcional) Token GitHub com acesso ao repositório

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # e preencha os valores
```

## Execução Local

1. Suba o SonarQube (ex.: `docker run -d -p 9000:9000 sonarqube:community`).
2. Configure tokens/variáveis (`SONARQUBE_URL`, `SONARQUBE_TOKEN`, `OPENAI_API_KEY` ou `LLM_LOCAL_MODEL_PATH`, `SONAR_PROJECT_KEY`).
   - Para apontar a pipeline a um código hospedado fora de `tcc-pipeline`, defina `AUTOFIX_TARGET_ROOT` com o caminho absoluto do projeto (ex.: `AUTOFIX_TARGET_ROOT=/caminho/para/src/test-pipeline`). Esse valor é usado pelos agentes Fixer/Tester/Executor/Sonar e também pelo gerenciador de PRs.
3. Rode uma análise inicial: `sonar-scanner`.
4. Execute o pipeline:

```bash
cd tcc-pipeline
python -m app.main
```

A sessão A2A gera correções, valida com `pytest` + Hypothesis, reexecuta o Sonar e opcionalmente cria um Pull Request.

> Nota: o pipeline usa LangGraph para orquestrar os agentes e o parâmetro `LLM_PROVIDER` permite escolher entre OpenAI (`LLM_PROVIDER=openai`) ou um modelo local via `llama-cpp-python` (`LLM_PROVIDER=local`).

### Uso com LLM local

- Instale `llama-cpp-python` (já listado em `requirements.txt`).
- Baixe o modelo GGUF desejado e configure em `.env`:
  - `LLM_PROVIDER=local`
  - `LLM_LOCAL_MODEL_PATH=/caminho/para/modelo.gguf`
  - Ajuste `LLM_LOCAL_CTX`, `LLM_LOCAL_THREADS` e `LLM_LOCAL_CHAT_FORMAT` conforme necessário.
- Com esse modo, as chamadas Requester/Fixer/Tester usam `llama-cpp-python` em vez da API da OpenAI.

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
