# AutoFix SonarQube + A2A + Property-Based Testing

Pipeline ponta-a-ponta inspirado no framework PGS (Property-Guided Synthesis) que integra SonarQube, agentes LLM orquestrados via LangGraph e testes baseados em propriedades (Hypothesis) para gerar correções automáticas.

## Nova Arquitetura

Esta versão refatorada implementa:

- **Orquestração LangGraph**: Grafo de estados com nós especializados
- **RAG Local**: ChromaDB + sentence-transformers para contexto
- **Property-First**: Testes Hypothesis gerados ANTES de modificar código
- **Cobertura Integrada**: coverage.xml importado pelo SonarQube
- **Patches Seguros**: Allowlist de paths e validação de diffs
- **Loops de Refinamento**: Até MAX_ROUNDS para correção iterativa

## Fluxo da Pipeline

```
SONAR_INGEST → PLAN → RULE_RAG → PROP_SPEC → PROP_GEN → PROP_RUN
                                                                    ↓
PR_BUILDER ← LOT_GATE ← SONAR_RESCAN ← TESTS ← PATCH ← FIX_PLAN
```

### Etapas Principais

1. **SONAR_INGEST**: Coleta issues via API REST
2. **PLAN**: Agrupa issues em lotes (regra × diretório)
3. **RULE_RAG**: Busca contexto e exemplos de correção
4. **PROP_SPEC/PROP_GEN**: Gera testes Hypothesis **antes** de tocar no código
5. **PROP_RUN**: Executa propriedades, captura contraexemplos
6. **FIX_PLAN**: Planeja correção mínima baseada em contraexemplos
7. **PATCH**: Aplica diff unificado com validação de paths
8. **TESTS**: Executa suite completa + gera coverage.xml
9. **SONAR_RESCAN**: Re-executa scanner, valida Quality Gate
10. **LOT_GATE**: Decide continuar, próximo lote ou finalizar
11. **PR_BUILDER**: Cria Pull Request se lote aprovado

### Loops de Refinamento

- **Propriedade falha** → FIX_PLAN com contraexemplos
- **Teste quebra** → FIX_PLAN com trace de erro  
- **Sonar reprova** → RULE_RAG com contexto expandido
- **Max rounds** → Pula para próximo lote

## Estrutura

```
.
├─ app/
│  ├─ main.py                 # Orquestrador LangGraph
│  ├─ sonarqube_client.py     # Cliente REST SonarQube
│  ├─ patcher.py              # Aplicação segura de patches
│  ├─ utils.py                # Utilitários (scanner, git, masking)
│  ├─ llm_client.py           # Abstração LLM (OpenAI/local)
│  ├─ rag/
│  │  ├─ ingest.py            # Indexação ChromaDB
│  │  └─ retriever.py         # Retrieval híbrido
│  └─ a2a/
│     ├─ protocol.py          # AgentState e tipos
│     ├─ property_agent.py    # PROP_SPEC/PROP_GEN
│     ├─ tester_agent.py      # PROP_RUN/TESTS + coverage
│     ├─ fixer_agent.py       # FIX_PLAN/PATCH
│     └─ requester_agent.py   # Contexto (legacy)
├─ src/                        # Código fonte do projeto
├─ tests/                      # Testes convencionais
│  ├─ conftest.py             # Config Hypothesis + reports/
│  └─ test_pipeline_smoke.py  # Smoke tests
├─ tests_prop/                 # Testes de propriedades (gerados)
├─ reports/                    # JUnit XML, cobertura
├─ chroma_db/                  # Base vetorial local
├─ .coveragerc                 # Config de cobertura
├─ sonar-project.properties    # Config Sonar + coverage.xml
├─ .github/workflows/autofix.yml
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

### 1. Setup do Ambiente

```bash
# Clone e configure
git clone <repo>
cd tcc-pipeline
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Instale dependências
pip install -r requirements.txt

# Configure variáveis
cp .env.example .env
# Edite .env com seus tokens
```

### 2. SonarQube

```bash
# Suba SonarQube local
docker run -d -p 9000:9000 sonarqube:community

# Ou use instância existente
# Configure SONARQUBE_URL e SONARQUBE_TOKEN no .env
```

### 3. Indexação RAG (Opcional)

```bash
# Indexe o código para contexto
python -c "
from app.rag.ingest import RAGIngestor
from pathlib import Path
ingestor = RAGIngestor()
result = ingestor.ingest_directory(Path('.'))
print(f'Indexed {result[\"docs_indexed\"]} documents')
"
```

### 4. Execute a Pipeline

```bash
# Gere cobertura inicial (opcional)
pytest --cov=src --cov-report=xml:coverage.xml

# Execute o AutoFix
python -m app.main
```

### 5. Resultados

- **Propriedades**: `tests_prop/test_*_props.py`
- **Relatórios**: `reports/junit*.xml`
- **Cobertura**: `coverage.xml`
- **PRs**: Links exibidos no final

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
