import json
import os
import sys
import time
from typing import TypedDict

import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from tools import get_database_schema

sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = "data/sales.db"
MAX_STEPS = 8

GROQ_MODEL = "openai/gpt-oss-20b"
# OpenRouter fallback: a current :free general-purpose instruct model.
# meta-llama/llama-3.3-70b-instruct:free is no longer listed on OpenRouter, so we
# use Google's Gemma 4 31B Instruct (free) as an equivalent general-purpose model.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "google/gemma-4-31b-it:free"

# Cerebras: OpenAI-compatible endpoint with higher free-tier daily token quota.
# Uses gpt-oss-120b, a current model on Cerebras's free-tier catalog.
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
CEREBRAS_MODEL = "gpt-oss-120b"

# Default provider order (Cerebras -> Groq -> OpenRouter).
DEFAULT_PROVIDER_ORDER = "cerebras,groq,openrouter"

# Legacy env var name kept for backward compatibility.
LLM_PRIMARY_PROVIDER_DEFAULT = "openrouter"


class InvestigationState(TypedDict, total=False):
    question: str
    schema: str
    plan: list[str]
    plan_index: int
    steps: int
    queries: list[str]
    findings: list[dict]
    observations: list[str]
    charts: list[dict]
    current_query: str
    current_result: str
    final_answer: dict


def get_groq_llm():
    load_dotenv()
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )


def get_openrouter_llm():
    """Build an OpenRouter chat client (OpenAI-compatible), or None if no key."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
    )


def get_cerebras_llm():
    """Build a Cerebras chat client (OpenAI-compatible), or None if no key."""
    load_dotenv()
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        return None
    return ChatOpenAI(
        model=CEREBRAS_MODEL,
        api_key=api_key,
        base_url=CEREBRAS_BASE_URL,
        temperature=0,
    )


def get_provider_llm(name: str):
    """Return the chat client for a provider name ("cerebras" | "groq" | "openrouter")."""
    if name == "cerebras":
        return get_cerebras_llm()
    if name == "groq":
        return get_groq_llm()
    if name == "openrouter":
        return get_openrouter_llm()
    raise ValueError(
        f"Unknown LLM provider: {name!r}. Expected 'cerebras', 'groq', or 'openrouter'."
    )


def get_provider_order() -> list[str]:
    """Provider names in the order they should be tried.

    Priority:
    1. LLM_PROVIDER_ORDER env var (comma-separated, e.g. "cerebras,groq,openrouter").
    2. Legacy LLM_PRIMARY_PROVIDER env var — expands to a full ordered list with
       the other two providers appended as fallbacks.
    3. DEFAULT_PROVIDER_ORDER ("cerebras,groq,openrouter").

    Every entry must be one of "cerebras", "groq", or "openrouter".  A clear
    error is raised listing any invalid entries.
    """
    VALID_PROVIDERS = {"cerebras", "groq", "openrouter"}

    load_dotenv()

    raw_order = os.getenv("LLM_PROVIDER_ORDER", "").strip()

    if not raw_order:
        # Fall back to legacy LLM_PRIMARY_PROVIDER for backward compatibility.
        primary = os.getenv("LLM_PRIMARY_PROVIDER", "").strip().lower()
        if primary:
            rest = [p for p in ("cerebras", "groq", "openrouter") if p != primary]
            raw_order = f"{primary},{','.join(rest)}"
        else:
            raw_order = DEFAULT_PROVIDER_ORDER

    order = [p.strip().lower() for p in raw_order.split(",") if p.strip()]

    invalid = [p for p in order if p not in VALID_PROVIDERS]
    if invalid:
        raise ValueError(
            f"Invalid provider(s) in LLM_PROVIDER_ORDER: {invalid!r}. "
            f"Accepted values: {sorted(VALID_PROVIDERS)}."
        )

    return order


def is_rate_limit_error(err) -> bool:
    """True if an exception is a provider rate-limit/quota error.

    OpenRouter can also surface these as 429 (insufficient_quota) or 413; Groq
    uses "tokens per minute" / "tokens per day".
    """
    text = str(err).lower()
    markers = (
        "429",
        "413",
        "rate limit",
        "rate_limit",
        "tokens per minute",
        "tokens per day",
        "insufficient_quota",
        "quota",
    )
    return any(m in text for m in markers)


def _invoke_llm(llm, prompt: str, max_retries: int = 3) -> str:
    """Call a single client with retry/backoff, returning trimmed text.

    Raises the underlying error if retries are exhausted.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            content = llm.invoke(prompt).content
            if content and content.strip():
                return content.strip()
            last_err = RuntimeError("Empty LLM response")
        except Exception as e:  # noqa: BLE001 - retry transient API errors
            last_err = e
            if is_rate_limit_error(e):
                time.sleep(2 ** attempt)  # 1s, 2s, 4s
                continue
            raise
        time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM call failed after retries: {last_err}")


def invoke_with_fallback(prompt: str) -> str:
    """Run an LLM call trying providers in configurable order.

    Order is controlled by LLM_PROVIDER_ORDER (default "cerebras,groq,openrouter").
    Falls back to legacy LLM_PRIMARY_PROVIDER if LLM_PROVIDER_ORDER is unset.
    On a rate-limit/quota error the call moves to the next provider. Real
    (non-rate-limit) failures are propagated normally.

    The provider that served each call is printed so it is visible per step.
    """
    order = get_provider_order()

    # If the primary provider has no key configured, fail fast with a clear
    # message instead of silently skipping it.
    first_llm = get_provider_llm(order[0])
    if first_llm is None:
        raise RuntimeError(
            f"Primary provider '{order[0]}' has no API key configured. "
            f"Set the corresponding key in .env or adjust LLM_PROVIDER_ORDER."
        )

    errors = []
    for name in order:
        llm = get_provider_llm(name)
        if llm is None:
            # Fallback provider unavailable (e.g. openrouter with no key, when groq is primary).
            errors.append(f"{name}: no API key configured")
            continue

        try:
            result = _invoke_llm(llm, prompt)
            model = {"groq": GROQ_MODEL, "openrouter": OPENROUTER_MODEL, "cerebras": CEREBRAS_MODEL}
            print(f"[llm:{name}] step served by {name} "
                  f"({model.get(name, name)}).", flush=True)
            return result
        except Exception as e:  # noqa: BLE001
            if is_rate_limit_error(e):
                errors.append(f"{name}: {e}")
                print(f"[llm:{name}] rate-limited/unavailable, trying next provider: {e}", flush=True)
                continue
            raise

    raise RuntimeError(
        "All LLM providers were unavailable or rate-limited: "
        + ("; ".join(errors) if errors else "no providers attempted")
    )


# ---------------------------------------------------------------------------
# Node: understand/question
# ---------------------------------------------------------------------------
def understand_question(state: InvestigationState) -> dict:
    schema = state.get("schema") or get_database_schema.invoke({})

    prompt = f"""You are a data analyst building an investigation plan for a SQLite sales database.

SCHEMA:
{schema}

USER QUESTION:
{state['question']}

Break the question into an ordered list of SQL analysis steps that will fully answer it.
The FIRST step MUST be simple and establish the headline number(s): e.g. for a revenue
question, "total revenue by month for the relevant period". Do NOT combine many dimensions
in step one. Subsequent steps should drill into plausible drivers one dimension at a time
(region, then product within the worst region, then month). Keep each step focused on ONE
breakdown dimension so results stay readable.

Return ONLY a JSON array of strings, each string a short description of one SQL step to run.
Example: ["total revenue by month for July and August 2026", "revenue by region for July and August 2026", "revenue by product in the worst region for July and August 2026"]
Do not include any text outside the JSON array."""

    resp = invoke_with_fallback(prompt)
    content = resp.strip()
    # Defensive: extract first [...] block
    start = content.find("[")
    end = content.rfind("]")
    if start == -1 or end == -1:
        plan = [f"Answer: {state['question']}"]
    else:
        try:
            plan = json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            plan = [f"Answer: {state['question']}"]
    if not isinstance(plan, list) or not plan:
        plan = [f"Answer: {state['question']}"]

    return {
        "schema": schema,
        "plan": plan,
        "plan_index": 0,
        "steps": 0,
        "queries": [],
        "findings": [],
        "observations": [],
        "charts": [],
    }


# ---------------------------------------------------------------------------
# Node: generate SQL for the current investigation step
# ---------------------------------------------------------------------------
def generate_sql(state: InvestigationState) -> dict:
    step = state["plan"][state["plan_index"]]
    previous = state.get("findings", [])

    prompt = f"""You are writing a single READ-ONLY SQL query against a SQLite sales database.

SCHEMA:
{state['schema']}

USER QUESTION:
{state['question']}

CURRENT INVESTIGATION STEP:
{step}

PREVIOUS FINDINGS (use to inform the query if relevant):
{json.dumps(previous, default=str)[:2000]}

Write ONE SQL SELECT (or WITH) statement that computes the result for this step.
Use clear column aliases. Use strftime('%Y-%m', sales.date) for monthly grouping, and
sales.region, products.name for breakdowns. Sales table has columns: date, customer_id,
product_id, quantity, unit_price, discount, region, sales_rep.
Revenue per row = quantity * unit_price * (1 - discount).

Return ONLY the SQL statement. No markdown, no explanation."""

    resp = invoke_with_fallback(prompt)
    sql = resp.strip()
    if sql.startswith("```"):
        sql = sql.strip("`")
        sql = sql.replace("sql", "", 1).strip() if sql.lower().startswith("sql") else sql
    return {"current_query": sql}


# ---------------------------------------------------------------------------
# Node: validate SQL (read-only check) + execute
# ---------------------------------------------------------------------------
def query_to_dataframe(query: str):
    """Run a validated, read-only query and return (text_result, dataframe | None)."""
    forbidden = ["insert", "update", "delete", "drop", "alter", "truncate", "create"]
    lowered = query.strip().lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return "Error: Only SELECT or WITH statements are allowed.", None
    if any(word in lowered for word in forbidden):
        return "Error: Query contains a forbidden keyword. Only read-only SELECT queries are allowed.", None
    import sqlite3

    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:  # noqa: BLE001
        return f"Error executing query: {e}", None
    return df.to_string(index=False), df


def execute_sql_node(state: InvestigationState) -> dict:
    query = state["current_query"]
    result, df = query_to_dataframe(query)
    queries = list(state.get("queries", []))
    queries.append(query)
    updates = {"queries": queries}

    record = {
        "query": query,
        "step": state["plan"][state["plan_index"]],
    }
    if result.startswith("Error"):
        record["result"] = result
        record["rows"] = []
    else:
        record["result"] = result
        record["rows"] = df.to_dict(orient="records") if df is not None else []
        record["columns"] = list(df.columns) if df is not None else []

    updates["current_result"] = result
    updates["findings"] = state.get("findings", []) + [record]
    return updates


# ---------------------------------------------------------------------------
# Node: analyze result -> decide whether to continue investigating
# Routes back to generate_sql (continue) or to generate_chart (done)
# ---------------------------------------------------------------------------
def analyze_result(state: InvestigationState) -> dict:
    step = state["plan"][state["plan_index"]]
    result = state["current_result"]

    # Cap the result so we stay well under the free-tier token budget.
    result_preview = result[:1200] + ("\n...[result truncated]" if len(result) > 1200 else "")

    prompt = f"""You are a data analyst. Interpret the result of one SQL investigation step.

USER QUESTION:
{state['question']}

STEP JUST RAN:
{step}

SQL QUERY:
{state['current_query']}

QUERY RESULT (may be truncated):
{result_preview}

Write a concise observation (2-4 sentences) explaining what this result shows and whether it
points toward an explanation for the user's question. Mention specific numbers from the result.
If the step errors, note the error and what a reasonable next step is.

Return ONLY the observation text."""

    resp = invoke_with_fallback(prompt)
    observation = resp.strip()

    observations = list(state.get("observations", []))
    observations.append(f"STEP: {step}\n{observation}")

    plan_index = state["plan_index"] + 1
    steps = state["steps"] + 1

    return {
        "observations": observations,
        "plan_index": plan_index,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Router: keep investigating while steps remain and budget is not exhausted
# ---------------------------------------------------------------------------
def should_continue(state: InvestigationState) -> str:
    if state["steps"] >= MAX_STEPS:
        return "finalize"
    if state["plan_index"] >= len(state["plan"]):
        return "finalize"
    return "investigate"


# ---------------------------------------------------------------------------
# Node: generate chart specs from the findings
# ---------------------------------------------------------------------------
def generate_charts(state: InvestigationState) -> dict:
    charts = []
    for i, f in enumerate(state.get("findings", [])):
        if f.get("result", "").startswith("Error") or not f.get("rows"):
            continue
        spec = build_chart_from_finding(f)
        if spec:
            charts.append(spec)
        else:
            print(f"Skipping chart for finding {i}: no suitable label column found", flush=True)
    return {"charts": charts}


def build_chart_from_finding(finding: dict) -> dict | None:
    """Build a Plotly chart spec from a structured finding (real query rows).

    Prefers charts that tell the story: monthly trend, revenue by region, and
    revenue by product. Falls back to the first label/numeric columns.
    """
    rows = finding["rows"]
    columns = finding.get("columns", [])
    if not rows:
        return None

    df = pd.DataFrame(rows)

    def is_numeric(col):
        return pd.api.types.is_numeric_dtype(df[col])

    numeric_cols = [c for c in columns if c in df.columns and is_numeric(c)]
    non_numeric = [c for c in columns if c in df.columns and not is_numeric(c)]
    if not numeric_cols:
        return None

    query = finding.get("query", "").lower()
    label_col = None

    # Prefer a descriptive label column based on what the query grouped by.
    for candidate in ("region", "product_name", "name", "category", "industry", "customer_size", "month", "sales_rep", "discount_tier"):
        if candidate in non_numeric:
            label_col = candidate
            break
    if label_col is None and non_numeric:
        label_col = non_numeric[0]

    # A chart needs a label/x-axis column. If the finding has no non-numeric
    # column at all (e.g. a single numeric aggregate with no label dimension),
    # there is nothing to plot labels on, so skip generating a chart for it.
    if label_col is None:
        return None

    y_col = numeric_cols[0]

    # When more than one label column exists (e.g. month + region), combine them
    # into a single descriptive label so the chart is not misleadingly collapsed.
    label_cols = [c for c in non_numeric if c in df.columns]
    combined_labels = None
    if len(label_cols) >= 2:
        combined_labels = df[label_cols].astype(str).agg(" | ".join, axis=1).tolist()

    # Chart type: line for time trends, bar otherwise.
    chart_type = "line" if ("month" in query or "strftime" in query or "date" in query) else "bar"

    return {
        "title": f"{y_col} by {label_col}",
        "type": chart_type,
        "x": combined_labels if combined_labels else df[label_col].astype(str).tolist(),
        "y": [float(v) for v in df[y_col].tolist()],
        "y_label": y_col,
        "x_label": " | ".join(label_cols) if combined_labels else label_col,
        "subtitle": finding.get("step", ""),
        "query": finding.get("query", ""),
    }


# ---------------------------------------------------------------------------
# Node: generate final structured answer
# ---------------------------------------------------------------------------
def generate_answer(state: InvestigationState) -> dict:
    prompt = f"""You are a data analyst writing the final answer to a business user.

USER QUESTION:
{state['question']}

INVESTIGATION FINDINGS (query + result + observation):
{json.dumps(state.get('findings', []), default=str)[:6000]}

OBSERVATIONS:
{json.dumps(state.get('observations', []), default=str)[:3000]}

Write a structured final answer as a JSON object with these exact keys:
- "executive_summary": 2-3 sentence overview of what happened and why.
- "main_cause": single most important driver of the situation, with the specific numbers.
- "key_drivers": array of strings, each a driver with the numbers/evidence behind it.
- "evidence": array of strings citing the actual query results (specific figures, regions, products, months).
- "recommended_investigation": array of strings, follow-up questions or deeper analyses.
- "chart_captions": array of strings, one short caption per chart in order (if any).

Use ONLY numbers that appear in the findings. Never invent data.
Return ONLY the JSON object, no markdown."""

    resp = invoke_with_fallback(prompt)
    content = resp.strip()
    start = content.find("{")
    end = content.rfind("}")
    answer = {}
    if start != -1 and end != -1:
        try:
            answer = json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            answer = {}

    if not answer or not answer.get("executive_summary"):
        # Fallback: assemble a minimal, honest answer from the evidence we have.
        evidence = []
        for f in state.get("findings", []):
            if not f.get("result", "").startswith("Error"):
                evidence.append(f"{f['step']}: query returned -> {f['result'][:300]}")
        answer = {
            "executive_summary": "Investigation completed. See evidence below for the query results.",
            "main_cause": "Could not be summarized automatically; review the evidence.",
            "key_drivers": [],
            "evidence": evidence or [],
            "recommended_investigation": [],
            "chart_captions": [],
        }

    # ensure keys exist
    for k in ["executive_summary", "main_cause", "key_drivers", "evidence", "recommended_investigation", "chart_captions"]:
        answer.setdefault(k, "" if k in ("executive_summary", "main_cause") else [])
    return {"final_answer": answer}


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_graph():
    g = StateGraph(InvestigationState)

    g.add_node("understand", understand_question)
    g.add_node("generate_sql", generate_sql)
    g.add_node("execute", execute_sql_node)
    g.add_node("analyze", analyze_result)
    g.add_node("charts", generate_charts)
    g.add_node("answer", generate_answer)

    g.set_entry_point("understand")

    g.add_edge("understand", "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_edge("execute", "analyze")
    g.add_conditional_edges(
        "analyze",
        should_continue,
        {"investigate": "generate_sql", "finalize": "charts"},
    )
    g.add_edge("charts", "answer")
    g.add_edge("answer", END)

    return g.compile()


def run_investigation(question: str) -> InvestigationState:
    graph = build_graph()
    result = graph.invoke(
        {"question": question},
        config={"recursion_limit": 40},
    )
    return result
