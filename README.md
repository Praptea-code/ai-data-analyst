# ai-data-analyst

An autonomous **LangChain + LangGraph** agent that answers natural-language questions about a sales database. It writes read-only SQL, investigates results across multiple dimensions, builds charts, and explains its findings with concrete evidence — all through an interactive Streamlit UI or a CLI.

## How it works

The agent runs a LangGraph investigation loop:

```
understand question → generate SQL → validate (read-only) → execute → analyze result
        ↓ (need more investigation? loop back, up to MAX_STEPS=8)
generate chart specs → generate final answer → END
```

- **State** tracks the question, schema, investigation plan, queries run, results, observations, charts, and the structured final answer.
- Every SQL statement is strictly **read-only** — INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE/CREATE are rejected.
- Charts are built only from **real query results**, never invented numbers.
- Free-tier Groq rate limits are handled with automatic retry/backoff.

## Stack

- **Python**, **LangChain**, **LangGraph**, **LangSmith** (opt-in, free)
- **SQLite** (local data), **Groq** `openai/gpt-oss-20b` via `langchain-groq`
- **Streamlit**, **Plotly**, **pandas**

## Setup

```bash
# 1. Clone
git clone https://github.com/<you>/ai-data-analyst.git
cd ai-data-analyst

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure secrets
cp .env.example .env
# Edit .env and set GROQ_API_KEY=...  (https://console.groq.com)
# Optionally set OPENROUTER_API_KEY=...  (https://openrouter.ai/keys)

# 5. Build the example database (seeded with a deliberate August 2026 anomaly
#    in West-region Product A sales)
python database.py
```

## LLM providers: configurable order, OpenRouter-first by default

The investigation loop uses two LLM providers — **Groq** (`openai/gpt-oss-20b`) and
**OpenRouter** (free `:free` models) — and can switch between them on rate-limit /
tokens-per-minute / tokens-per-day / quota errors.

The provider order is controlled by `LLM_PRIMARY_PROVIDER` in `.env`:

- `LLM_PRIMARY_PROVIDER=openrouter` (default) — tries OpenRouter first, falls back to Groq.
- `LLM_PRIMARY_PROVIDER=groq` — the faster option; tries Groq first, falls back to OpenRouter.

It is currently recommended to leave it as `openrouter` while Groq's free tier is under
heavy testing load, but it can be flipped back to `groq` anytime without code changes.

- Add `OPENROUTER_API_KEY` to `.env` to enable OpenRouter (required when it is the primary
  provider; optional but recommended otherwise).
- OpenRouter's free, no-cost `:free` models are used. The default model is
  `google/gemma-4-31b-it:free` (the previous `meta-llama/llama-3.3-70b-instruct:free` is no
  longer listed on OpenRouter); it is configurable via `OPENROUTER_MODEL` in `graph.py`.
- Each LLM step logs which provider served it (e.g. `[llm:groq]` or `[llm:openrouter]`).
- All model calls still go through the same read-only SQL pipeline; only the LLM provider differs.

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Type a question, e.g. **"Why did revenue fall in August 2026?"**, and the app shows:

- **Executive Summary**
- **Main Cause**
- **Key Drivers**
- **Evidence with Charts** (interactive Plotly charts + evidence statements)
- **Recommended Investigation**
- a collapsible **raw trace** (plan, SQL queries, observations)

Run the CLI (prints the same structured answer + trace to the terminal):

```bash
python agent.py "Why did revenue fall in August 2026?"
python agent.py --json "What was total revenue in July 2026?"
```

## LangSmith tracing (optional, free)

Set these in your `.env` (see `.env.example`) to inspect every run's trace — question, tool/SQL calls, results, decisions, and final answer — in the LangSmith UI:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=ai-data-analyst
```

Without these values, tracing is simply disabled and everything still works. You can obtain a key at https://smith.langchain.com.

## Project layout

```
app.py          Streamlit front end
agent.py        CLI runner for the investigation graph
graph.py        LangGraph investigation loop (state + nodes + routing)
charts.py       Plotly chart renderer for investigation chart specs
tools.py        read-only SQL tools (get_database_schema, execute_sql)
database.py     generates the demo SQLite sales database
data/sales.db   SQLite database (generated)
```

## Database demo seed

`database.py` creates a fake sales database (customers, products, sales; Jan–Aug 2026) and deliberately plants an anomaly: **in August 2026, Product A sales are suppressed in the West region**, which is what makes "Why did revenue fall in August 2026?" an interesting investigation.
