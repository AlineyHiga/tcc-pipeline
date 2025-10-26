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
│  ├─ __init__.py
│  ├─ llm_client.py
│  ├─ logging_setup.py
│  ├─ main.py
│  ├─ patcher.py
│  ├─ codemod_apply.py
│  ├─ sonarqube_client.py
│  ├─ utils.py
│  ├─ a2a/
│  │  ├─ __init__.py
│  │  ├─ fixer_agent.py
│  │  ├─ property_agent.py
│  │  ├─ protocol.py
│  │  └─ tester_agent.py
│  └─ rag/
│     ├─ ingest.py
│     └─ retriever.py
├─ artifacts/
├─ chroma_db/
├─ logs/
│  └─ pipeline.jsonl
├─ reports/
├─ tests/
│  ├─ conftest.py
│  └─ test_pipeline_smoke.py
├─ tests_prop/
│  ├─ __init__.py
│  ├─ test_database_props.py
│  └─ test_utils_props.py
├─ .github/workflows/autofix.yml
├─ .coveragerc
├─ .env
├─ .env.example
├─ .gitignore
├─ LICENSE
├─ README.md
├─ requirements.txt
├─ sonar-project.properties
└─ test_pipeline.py
```

## Inventário de Arquivos

### Raiz
- `README.md`: Documentação principal do pipeline AutoFix.
- `requirements.txt`: Lista dependências de LangGraph, RAG, Hypothesis, cobertura e cliente OpenAI.
- `.env.example`: Template de variáveis sensíveis; copie para `.env` e preencha credenciais locais.
- `.env`: Configuração local (não deve ser versionada) com tokens para SonarQube, OpenAI e ajustes da pipeline.
- `.gitignore`: Mantém caches, ambientes virtuais, relatórios e artefatos fora do Git.
- `.coveragerc`: Configura coleta de cobertura usada por pytest/coverage.
- `sonar-project.properties`: Define chaves de projeto, fontes (`src`) e testes (`tests`) para o sonar-scanner.
- `test_pipeline.py`: Script interativo que instância `AutoFixPipeline`, executa o fluxo completo e exibe logs ricos.
- `LICENSE`: Licença MIT do projeto.

### Diretórios de suporte
- `artifacts/`: Saídas das execuções (prompts/diffs, issues normalizados, bundles zip). Cada run gera um UUID próprio.
- `chroma_db/`: Persistência local do índice vetorial utilizado pelo RAG (`chromadb.PersistentClient`).
- `logs/pipeline.jsonl`: Log estruturado (JSONL) gerado pelo `logging_setup` com contexto de spans.
- `reports/`: Pasta de destino para `coverage.xml`, `reports/junit*.xml` e demais relatórios de teste (criada sob demanda).
- `.github/workflows/autofix.yml`: Workflow GitHub Actions que replica o pipeline com Sonar, RAG e agentes.
- `.pytest_cache/`, `__pycache__/`, `.venv/`: Artefatos de ambiente/teste mantidos fora do controle de versão por `.gitignore`.

### app/
- `app/main.py`: Orquestra a LangGraph `StateGraph`, conecta agentes, executa nós `SONAR_INGEST` → `PR_BUILDER` e agrupa lotes.
- `app/logging_setup.py`: Configura logging estruturado, MDC (`set_ctx`), formatação JSON e gestão de artefatos.
- `app/codemod_apply.py`: Aplicador determinístico de specs JSON (LibCST) que reescreve funções com segurança.
- `app/llm_client.py`: Abstração de cliente LLM (OpenAI ou compatível) com logging de prompts/respostas e amostragem.
- `app/patcher.py`: Implementa `SafePatcher`, valida diffs (allowlist, orçamento de LOC, `git apply`) antes de mutar arquivos.
- `app/sonarqube_client.py`: Cliente REST simples para listar issues e quality gate usando `requests.Session`.
- `app/utils.py`: Utilitários (masking de segredos, `run_sonar_scanner` com fallback Docker, helpers git).
- `app/__init__.py`: Exporta símbolos principais do pacote `app`.

### app/a2a
- `app/a2a/protocol.py`: Define `AgentState` (Pydantic) e `A2AMessage`, núcleo do estado compartilhado.
- `app/a2a/property_agent.py`: Gera especificações e arquivos de testes Hypothesis usando LLM + RAG, com validações robustas de JSON.
- `app/a2a/tester_agent.py`: Executa `pytest` (isolado ou Docker), produz `junit.xml`, `coverage.xml` e extrai contraexemplos.
- `app/a2a/fixer_agent.py`: Monta planos determinísticos, gera diffs unificados e aplica heurísticas AST/LLM para correção.
- `app/a2a/__init__.py`: Facilita importação dos agentes e do estado.

### app/rag
- `app/rag/ingest.py`: Indexa diretórios (code/docs) no ChromaDB, com LangChain quando disponível.
- `app/rag/retriever.py`: Consulta híbrida do índice (`contexts`, `citations`, few-shots) para enriquecer agentes.

### Testes
- `tests/conftest.py`: Ajusta Hypothesis (perfil sem limite de deadline), garante `reports/` e injeta paths no `sys.path`.
- `tests/test_pipeline_smoke.py`: Smoke tests de LangGraph, `AgentState` e interações básicas do `PropertyAgent`.
- `tests/test_codemod_apply.py`: Garante que o spec JSON aplica transformações em AST e valida casos inválidos.

### Testes de propriedades
- `tests_prop/__init__.py`: Marca pacote de propriedades.
- `tests_prop/test_database_props.py`: Template Hypothesis para módulos de banco (`test_no_exception_on_valid_input`, etc.).
- `tests_prop/test_utils_props.py`: Template Hypothesis aplicado a utilidades, pronto para customização.

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
