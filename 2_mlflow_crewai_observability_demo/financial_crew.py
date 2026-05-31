"""
CrewAI multi-agent financial-analysis crew with full MLflow observability.

Four agents run in sequence:
  1. Orchestrator — scopes the engagement charter.
  2. Macro Data Specialist — fetches World Bank CPI inflation (nested span).
  3. Research Analyst — writes the household-pressure research brief.
  4. Synthesist — produces the investment-style synthesis and calls
     the numeric-validation tool.

Every agent, task, tool call, and LLM completion is auto-traced via
``mlflow.crewai.autolog()`` + ``mlflow.openai.autolog()``.  The root
``@mlflow.trace`` span collects summary attributes so you can inspect
them in the MLflow UI without reading logs.
"""

from __future__ import annotations

import os
import time
from textwrap import dedent

import crew_state
import mlflow
from crewai import Agent, Crew, Task
from crewai.crews.crew_output import CrewOutput
from crewai.llm import LLM
from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool
from dotenv import load_dotenv
from report_validation import (
    ValidateReportNumbersTool,
    extract_numeric_tokens,
    validate_report_numbers_against_sources,
)
from web_stats import fetch_world_bank_inflation_summary

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def build_llm() -> BaseLLM:
    """Build a CrewAI-native OpenAI LLM from environment variables."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set.  Copy .env.example → .env and add your key."
        )
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
    if not model:
        raise ValueError("OPENAI_MODEL must not be empty when set.")
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    kwargs: dict = {"model": model, "api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")
    return LLM(**kwargs)

# ---------------------------------------------------------------------------
# Topic
# ---------------------------------------------------------------------------

TOPIC = dedent("""\
    Como as famílias comuns são afetadas quando a inflação, as taxas de juros
    e os custos de moradia seguem direções opostas: a capacidade de pagar
    aluguel e hipotecas, contas de alimentação e energia, salários reais,
    pagamentos de dívidas (cartões de crédito, empréstimos) e reservas de
    emergência — em economias típicas de alta renda e emergentes, e não
    apenas dentro do contexto político de um único país. O que gera essa
    pressão, quem está mais exposto e o que observar nos próximos 12 a 24 meses.""")

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class FetchWorldBankInflationTool(BaseTool):
    """CrewAI tool wrapping ``web_stats.fetch_world_bank_inflation_summary``."""

    name: str = "fetch_world_bank_inflation"
    description: str = dedent(
        """Carrega a inflação anual do IPC (% em relação ao ano anterior) da
           API JSON pública do Banco Mundial. Forneça um código de país ISO
           de 2 letras (por exemplo, EUA, GB, DE, JP). Use uma vez para
           ancorar uma análise quantitativa; cite o Banco Mundial no resumo."""
    )

    def _run(self, country_code: str = "US") -> str:
        return fetch_world_bank_inflation_summary(country_code.strip())

# ---------------------------------------------------------------------------
# Callbacks — persist task outputs + collect metrics for the root span
# ---------------------------------------------------------------------------

_crew_metrics: dict[str, object] = {}
delegation_counts: dict[str, int] = {}


# ---------------------------------------------------------------------------
# State handoff helpers
# ---------------------------------------------------------------------------

def handoff_efficiency_score(output_text: str, next_input_text: str) -> float:
    """Token overlap between agent output and the next agent's input.

    Score of 1.0 means the next agent received everything the previous one produced.
    Score close to 0.0 means most of the output was dropped before handoff.
    """
    output_tokens = set(output_text.lower().split())
    input_tokens = set(next_input_text.lower().split())
    if not output_tokens:
        return 0.0
    return len(output_tokens & input_tokens) / len(output_tokens)


@mlflow.trace(name="state_handoff", span_type="CHAIN")
def log_state_handoff(
    from_agent: str, to_agent: str, output: str, next_input: str
) -> None:
    """Log a traced span capturing the handoff between two agents."""
    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes({
            "handoff.from_agent": from_agent,
            "handoff.to_agent": to_agent,
            "handoff.output_chars": len(output),
            "handoff.input_chars": len(next_input),
            "handoff.efficiency_score": handoff_efficiency_score(output, next_input),
        })


def track_routing(step_output: object) -> None:
    """step_callback — fires after every agent step.

    Tracks how many steps each agent takes in a single run.
    A count > 3 for one agent is a signal of a potential loop.
    """
    agent_name = getattr(step_output, "agent", None) or "unknown"
    delegation_counts[agent_name] = delegation_counts.get(agent_name, 0) + 1
    more_than_3_steps = 3
    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes({
            f"routing.{agent_name}.step_count": delegation_counts[agent_name],
            "routing.total_steps": sum(delegation_counts.values()),
            # More than 3 steps for a single agent in one run is a loop signal
            "routing.loop_detected": any(v > more_than_3_steps for v in delegation_counts.values()),
        })


def _on_plan_complete(output: object) -> None:
    text = crew_state.task_output_to_text(output)
    crew_state.set_state("engagement_charter", text)
    _crew_metrics["orchestration_lead.output_chars"] = len(text)


def _on_stats_complete(output: object) -> None:
    text = crew_state.task_output_to_text(output)
    crew_state.set_state("macro_stats_snapshot", text)
    _crew_metrics["macro_data_specialist.output_chars"] = len(text)
    charter = crew_state.get_state("engagement_charter") or ""
    log_state_handoff(
        from_agent="orchestration_lead",
        to_agent="macro_data_specialist",
        output=charter,
        next_input=text,
    )


def _on_research_complete(output: object) -> None:
    text = crew_state.task_output_to_text(output)
    crew_state.set_state("research_brief", text)
    _crew_metrics["research_analyst.output_chars"] = len(text)
    stats = crew_state.get_state("macro_stats_snapshot") or ""
    log_state_handoff(
        from_agent="macro_data_specialist",
        to_agent="research_analyst",
        output=stats,
        next_input=text,
    )


def _on_synthesis_complete(output: object) -> None:
    synthesis = crew_state.task_output_to_text(output)
    crew_state.set_state("synthesis_result", synthesis)
    _crew_metrics["synthesist.output_chars"] = len(synthesis)
    research = crew_state.get_state("research_brief") or ""
    result = validate_report_numbers_against_sources(synthesis, research)
    _crew_metrics["validation.result_chars"] = len(result)
    log_state_handoff(
        from_agent="research_analyst",
        to_agent="synthesist",
        output=research,
        next_input=synthesis,
    )

# ---------------------------------------------------------------------------
# Crew assembly
# ---------------------------------------------------------------------------

def build_crew(llm: BaseLLM) -> Crew:
    """Assemble four agents and four sequential tasks."""

    orchestrator = Agent(
        role="Líder de Orquestração do Engajamento",
        goal=dedent(
            """Transforme uma questão de investimento de alto nível em um mandato
               de pesquisa conciso: escopo, principais métricas, riscos e os
               resultados exatos que os sub-agentes devem produzir."""
        ),
        backstory=dedent(
            """Você é um gerente sênior de engajamento em uma mesa de pesquisa de ações
               do lado comprador. Você nunca faz os cálculos básicos pessoalmente — você
               define o que é considerado "bom" para que os especialistas possam executar
               o trabalho sem ambiguidade."""
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    macro_data = Agent(
        role="Especialista em Macrodados",
        goal=dedent(
            """Obtenha uma série histórica oficial da inflação medida pelo Índice de
               Preços ao Consumidor (IPC) na API pública do Banco Mundial e resuma-a
               para que analistas posteriores possam citá-la."""
        ),
        backstory=dedent(
            """Você utiliza apenas a ferramenta fetch_world_bank_inflation (API JSON
               do Banco Mundial). Você seleciona um código ISO de país de duas letras
               relevante para o projeto, executa a ferramenta uma única vez e retorna
               o resultado bruto da ferramenta mais uma breve interpretação. Você não
               elabora o relatório completo da pesquisa — isso é responsabilidade de
               outro profissional."""
        ),
        tools=[FetchWorldBankInflationTool()],
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    researcher = Agent(
        role="Analista de Pesquisa em Economia Doméstica e Finanças do Consumidor",
        goal=dedent(
            """Elabore um relatório de pesquisa estruturado sobre como as condições
               macroeconômicas (inflação, taxas de juros, mercados de trabalho, mercados
               imobiliários) se refletem nos orçamentos familiares típicos: moradia, itens
               essenciais, serviço da dívida e reservas de poupança — utilizando padrões
               internacionais, e não as regras de um único mercado."""
        ),
        backstory=dedent(
            """Você traduz pesquisas macroeconômicas e de finanças pessoais para não
               especialistas. Você nomeia mecanismos (por exemplo, dívida flutuante versus
               fixa, repasse de aluguéis, crescimento real dos salários), indica onde os
               dados são específicos de cada país e separa relações estabelecidas de debates
               em aberto. A tarefa de Dados Macroeconômicos já obteve a inflação do IPC do
               Banco Mundial — integre e cite esse bloco do contexto da sua tarefa."""
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    synthesist = Agent(
        role="Sintetizador Financeiro de Portfólio",
        goal=dedent(
            """Apresente um resultado conciso no estilo de investimento: tese, cenários
               otimista/baixo/pessimista, lista de métricas a serem observadas e como
               você expressaria sua visão."""
        ),
        backstory=dedent(
            """Você é um estrategista alinhado com o gerente de projeto: transforma pesquisas
               em opiniões acionáveis, narrativas ponderadas por probabilidade e divulgações
               de risco em linguagem clara. Após a elaboração do relatório, execute a
               função `validate_report_numbers_against_sources` para anexar uma verificação
               numérica."""
        ),
        tools=[ValidateReportNumbersTool()],
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    # -- Tasks (sequential) ------------------------------------------------

    plan_task = Task(
        description=dedent(f"""\
            Topic for the engagement:
            {TOPIC}

            Produce:
            1) Problem statement in 3-5 bullets.
            2) Research questions (ordered) the analyst must answer.
            3) Required sections for the research brief (with purpose each).
            4) Explicit non-goals / out of scope.
            5) Success criteria for the final synthesis."""),
        expected_output=(
            "A structured engagement charter (markdown) with numbered sections "
            "exactly as listed; no preliminary research—planning only."
        ),
        agent=orchestrator,
        callback=_on_plan_complete,
    )

    stats_task = Task(
        description=dedent(f"""\
            Engagement topic (for country choice):
            {TOPIC}

            From the orchestration charter in context, pick one major economy.
            Call fetch_world_bank_inflation exactly once with that country's
            2-letter ISO code.

            Output:
            1) The full tool response (verbatim block).
            2) One short paragraph explaining how this CPI series fits the
               household budget story."""),
        expected_output="World Bank tool output + brief interpretation (markdown).",
        agent=macro_data,
        context=[plan_task],
        callback=_on_stats_complete,
    )

    research_task = Task(
        description=dedent("""\
            Execute the research mandate from the charter.  Write the brief on
            household financial pressure in a multi-country lens.  Cover:
            - Inflation / policy-rate transmission to mortgage, rent, credit, essentials
            - Real wages vs inflation
            - Shelter stress: renters vs owners, fixed vs floating debt
            - Debt vulnerability: unsecured balances, BNPL, stress signals
            - Savings buffers and inequality of exposure
            - Regional variation (cite concepts, not one regulator's filing)

            Weave in the World Bank CPI data from context with attribution.
            Use clear headings.  Mark estimates vs facts.  No buy/sell—that
            comes later.  Output full markdown with multiple sections."""),
        expected_output=(
            "Structured markdown research brief with quantitative placeholders "
            "where firm-specific data is required."
        ),
        agent=researcher,
        context=[plan_task, stats_task],
        callback=_on_research_complete,
    )

    synthesis_task = Task(
        description=dedent("""\
            Read the charter and research brief from task context.  Write:
            - One-paragraph thesis
            - Bull / base / bear narratives (2-3 bullets each)
            - Metrics and events to monitor (table or bullets)
            - How to express the view and what would flip it
            - Plain-language risk disclosure (not personal advice)

            After drafting, call validate_report_numbers_against_sources once
            with your full synthesis as final_report.  Append the returned
            markdown under ## Numeric validation (heuristic).

            Use normal markdown—do not emit raw JSON tool calls."""),
        expected_output=(
            "Executive markdown: thesis, scenarios, metrics, expression, risks."
        ),
        agent=synthesist,
        context=[plan_task, research_task],
        callback=_on_synthesis_complete,
    )

    return Crew(
        agents=[orchestrator, macro_data, researcher, synthesist],
        tasks=[plan_task, stats_task, research_task, synthesis_task],
        step_callback=track_routing,
        verbose=True,
    )

# ---------------------------------------------------------------------------
# Root traced wrapper — sets summary attributes on the top-level span
# ---------------------------------------------------------------------------

@mlflow.trace(name="Crew.kickoff", span_type="CHAIN")
def run_crew_with_metrics(crew: Crew) -> CrewOutput:
    """Run ``crew.kickoff()`` inside a traced span that collects summary metrics."""
    _crew_metrics.clear()
    delegation_counts.clear()  # reset between runs to avoid false loop detection
    t0 = time.perf_counter()

    result = crew.kickoff()

    duration_s = round(time.perf_counter() - t0, 3)
    span = mlflow.get_current_active_span()
    if span is not None:
        charter = crew_state.get_state("engagement_charter") or ""
        stats = crew_state.get_state("macro_stats_snapshot") or ""
        research = crew_state.get_state("research_brief") or ""
        synthesis = crew_state.get_state("synthesis_result") or ""

        span.set_attributes({
            "crew.total_duration_s": duration_s,
            "crew.task_count": len(crew.tasks),
            "crew.agent_count": len(crew.agents),
            "orchestration_lead.output_chars": len(charter),
            "macro_data_specialist.output_chars": len(stats),
            "research_analyst.output_chars": len(research),
            "synthesist.output_chars": len(synthesis),
            "validation.report_number_count": len(extract_numeric_tokens(synthesis)),
            "validation.source_number_count": len(extract_numeric_tokens(research)),
            **_crew_metrics,
        })

    return result

# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _task_raw(crew_output: CrewOutput, index: int) -> str:
    tasks = crew_output.tasks_output
    if index < len(tasks) and tasks[index].raw:
        return str(tasks[index].raw).strip()
    return ""


def print_run_results(crew_output: CrewOutput | object) -> None:
    """Pretty-print crew deliverables to stdout."""
    w = 78
    bar = "=" * w
    if not isinstance(crew_output, CrewOutput):
        print(f"\n{bar}\n{str(crew_output).strip()}\n{bar}\n")
        return

    print(f"\n{bar}")
    print(" CREW OUTPUT ".center(w))
    print(bar)

    labels = [
        "1. Engagement charter",
        "2. Macro stats (World Bank)",
        "3. Research brief",
        "4. Synthesis",
    ]
    for i, label in enumerate(labels):
        body = _task_raw(crew_output, i)
        print(f"\n--- {label} ---\n")
        print(body or "(no text captured — see crew logs above)")
    print()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    crew_state.init_db()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    experiment = os.environ.get(
        "MLFLOW_EXPERIMENT_NAME", "crewai-household-financial-pressure"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    mlflow.crewai.autolog()
    mlflow.openai.autolog()

    crew = build_crew(build_llm())
    result = run_crew_with_metrics(crew)
    print_run_results(result)


if __name__ == "__main__":
    main()
