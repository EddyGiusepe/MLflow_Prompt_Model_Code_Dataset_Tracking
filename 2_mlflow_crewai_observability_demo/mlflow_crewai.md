<h1 align="center"><font color="gree">MLflow + CrewAI Observability Demo</font></h1>

<font color="pink">Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro</font>

Uma equipe multi-agente da [CrewAI](https://www.crewai.com/) que analisa a **pressão financeira do lar** (inflação, taxas, moradia, salários, dívidas) em diversos países, com cada agente, chamada de ferramenta (tool) e completion do LLM são rastreados através de **[MLflow](https://mlflow.org/)**.

## <font color="red">O que ele faz</font>

Quatro agentes de IA executam-se em sequência, cada um com um papel distinto:

| # | Agent | Responsabilidade | Visibilidade do MLflow |
|---|-------|----------------|-------------------|
| 1 | **Orchestration Lead** | Escopo do plano de engajamento | Span do Agent |
| 2 | **Macro Data Specialist** | Busca a inflação do CPI no World Bank | Agent → Tool → `world_bank_stats_api` Span aninhado |
| 3 | **Research Analyst** | Escreve um breve de pesquisa estruturado | Span do Agent |
| 4 | **Portfolio Synthesist** | Produz a síntese de investimento | Agent → Tool → `numeric_report_validation` Span aninhado |

```text
Crew.kickoff (root span — atributos resumidos aqui)
├── Task 1 → Líder de Orquestração
├── Task 2 → Especialista de Dados Macroeconômicos
│   └── world_bank_stats_api          ← @mlflow.trace (métricas HTTP)
├── Task 3 → Analista de Pesquisa
└── Task 4 → Síntese de Portfólio
    └── numeric_report_validation     ← @mlflow.trace (métricas de validação)
```

## <font color="red">Features de Observabilidade</font>

- **`mlflow.crewai.autolog()`** — traces crew, task, and agent execution.
- **`mlflow.openai.autolog()`** — traces every OpenAI LLM call with token counts (`prompt_tokens`, `completion_tokens`, `total_tokens`).
- **`@mlflow.trace`** on `web_stats.fetch_world_bank_inflation_summary` — adds span attributes: `stats_api.duration_ms`, `stats_api.http_status`, `stats_api.observation_rows_in_response`, etc.
- **`@mlflow.trace`** on `report_validation.validate_report_numbers_against_sources` — adds: `validation.suspicious_count`, `validation.report_numbers_count`, etc.
- **Root span attributes** — `crew.total_duration_s`, `crew.agent_count`, per-agent `*.output_chars`, validation summary.

## <font color="red">Prerequisitos</font>

- Python **3.10+**
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- An **OpenAI API key** ([get one here](https://platform.openai.com/api-keys))

## <font color="red">Quickstart</font>

```bash
# 1. Clone and enter the repo:
git clone <repo-url> && cd mlflow-crewai-observability

# 2. Install dependencies:
uv sync

# 3. Configure your API key:
cp .env.example .env
# Edit .env and set OPENAI_API_KEY:

# 4. Run the crew:
uv run python financial_crew.py

# 5. Start the MLflow UI (in a separate terminal):
uv run mlflow server --backend-store-uri sqlite:///mlflow.db
# Open http://127.0.0.1:5000
```

## <font color="red">Configuração</font>

All settings go in `.env` (see `.env.example`):

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `OPENAI_API_KEY` | Yes | — | OpenAI Chat Completions |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Model to use |
| `OPENAI_BASE_URL` | No | — | Azure OpenAI / proxy |
| `MLFLOW_TRACKING_URI` | No | `sqlite:///mlflow.db` | MLflow backend store |
| `MLFLOW_EXPERIMENT_NAME` | No | `crewai-household-financial-pressure` | Experiment name |
| `CREW_STATE_DB` | No | `crew_state.db` | SQLite shared state path |

## <font color="red">Layout do Projeto</font>

```text
├── financial_crew.py      # Crew definition, agents, tasks, MLflow setup, main()
├── web_stats.py            # World Bank JSON client + @mlflow.trace
├── report_validation.py    # Numeric validation tool + @mlflow.trace
├── crew_state.py           # SQLite shared state (task callbacks)
├── pyproject.toml          # Dependencies (uv)
├── .env.example            # Environment template
└── .gitignore
```

## <font color="red">Como funciona</font>

1. **`main()`** carrega `.env`, inicializa o estado SQLite, configura o rastreamento MLflow e habilita o autologging `crewai` + `openai`.
2. **`run_crew_with_metrics()`** envolve `crew.kickoff()` em um span `@mlflow.trace`. Após todas as quatro tarefas completarem, ele define atributos resumidos (durações, tamanhos de saída, contagens de validação) no span raiz.
3. **Callbacks de Tarefa** persistem a saída de cada tarefa no SQLite (`crew_state.db`) para que a ferramenta de validação possa verificar a síntese contra o breve de pesquisa.
4. **`validate_report_numbers_against_sources`** executa tanto como uma ferramenta (chamada pelo agente sintetista) quanto como uma garantia de callback (chamada em `_on_synthesis_complete`), garantindo que o span de validação sempre apareça no rastreamento.

## <font color="red">Notas</font>

- A saída é **ilustrativa** — não é aconselhamento financeiro pessoal.
- A integração do MLflow com o CrewAI rastreia apenas `kickoff()` síncrono; veja [documentação do MLflow CrewAI](https://mlflow.org/docs/latest/tracing/integrations/crewai).
- A API do World Bank é pública e gratuita; não é necessária uma API key.