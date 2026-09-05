# AI Data Analyst

An autonomous **LangChain + LangGraph** agent that answers natural-language questions about a sales database — it writes read-only SQL, drills into results across multiple dimensions, runs statistical forecasting and anomaly detection, builds charts only from real query results, and explains its findings with concrete evidence.

It ships with **three interfaces**:

1. **Web dashboard** (Next.js + Tailwind, Plotly) — the primary UI, talking to the FastAPI backend.
2. **REST API** (FastAPI on `:8000`) — exposes the agent for any client.
3. **Streamlit app + CLI** — lightweight self-contained alternatives running the agent directly.

---

## How it works

The agent runs a LangGraph investigation loop:

```
understand question → generate SQL → validate (read-only) → execute → analyze result
        ↓ (need more investigation? loop back, up to MAX_STEPS=8)
generate chart specs → run forecast/anomaly tools if requested → final structured answer → END
```

- **State** (`InvestigationState`) tracks the question, schema, investigation plan, queries run, results, observations, findings, charts, and optional forecast/anomaly flags.
- Every SQL statement is strictly **read-only** — only `SELECT` / `WITH` are allowed, and `INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE/CREATE` are rejected outright.
- Results are truncated per step to stay well under the free-tier token budget (`[llm:...]` and `...[result truncated]` markers logged).
- **Charts are never invented** — they are built only from real query result rows (`build_chart_from_finding`).
- Free-tier rate limits are handled with an automatic retry/backoff + multi-provider fallback chain.

### Forecasting & anomaly detection

The `analyze_result` step watches each observation (and the original question) for keywords:

- `forecast` / `predict future` / `future revenue` / `future trend` → the loop runs one extra historical monthly-revenue query and calls **`forecast_revenue()`** (Prophet if installed, otherwise exponential smoothing fallback).
- `anomaly` / `anomalies` / `spike` / `drop` / `unusual` / `outlier` / `pattern` → the loop runs the same historical query and calls **`detect_anomalies()`** (Z-score method, default `sensitivity=2.0` ≈ 95% confidence).

Results are appended to `findings` with the flags `forecast_done` / `anomaly_done`, so each runs automatically once per investigation.

### LLM providers: 3-tier fallback chain (Cerebras -> Groq -> OpenRouter)

Every LLM step can use three providers and switches between them on rate-limit / tokens-per-minute / tokens-per-day / quota errors:

| Provider | Model | Endpoint | When used |
| --- | --- | --- | --- |
| **Cerebras** (primary) | `gpt-oss-120b` | OpenAI-compatible | First choice — highest free-tier daily token quota |
| **Groq** | `openai/gpt-oss-20b` | native | First fallback |
| **OpenRouter** | `google/gemma-4-31b-it:free` | OpenAI-compatible | Last resort, no-cost `:free` models |

Provider order is controlled by `LLM_PROVIDER_ORDER` in `.env` (comma-separated, default `cerebras,groq,openrouter`). The legacy `LLM_PRIMARY_PROVIDER` config still works for backward compatibility. A clear error is raised if an invalid provider name is configured, and a helpful error tells you if the primary provider has no API key set. Each step logs which provider served it (e.g. `[llm:cerebras]`).

Every provider still runs the exact same read-only SQL pipeline — only the model differs.

---

## Stack

**Backend**

- **Python 3**, **LangChain**, **LangGraph**, **LangSmith** (opt-in tracing)
- **FastAPI** + **Uvicorn** (REST API), **SQLite** (local sales data)
- **pandas**, **numpy**, **scipy**, **Prophet** (optional, better forecasts)
- **Streamlit**, **Plotly** (direct UI / CLI clients)

**Frontend**

- **Next.js 16** (App Router, TypeScript) + **Tailwind CSS 4**
- **react-plotly.js** + Plotly.js — interactive charts
- **jspdf** + **html2canvas** — PDF export of results
- **date-fns** — relative timestamps in history
- **lucide-react** — icons
- **axios** — API calls to the FastAPI backend

---

## Repositories

| Part | Location |
| --- | --- |
| **Backend** (this repo) | `ai-data-analyst/` — agent, tools, FastAPI API, Streamlit + CLI |
| **Frontend** | `ai-data-analyst-ui/` (sibling) — Next.js dashboard |

---

## Backend setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure secrets
cp .env.example .env
# Edit .env and set:
#   GROQ_API_KEY=...       https://console.groq.com
#   CEREBRAS_API_KEY=...   https://cloud.cerebras.ai (primary provider)
#   OPENROUTER_API_KEY=... https://openrouter.ai/keys (optional fallback)

# 4. Build the example SQLite database (Jan–Aug 2026 sales, with a deliberate
#    August anomaly in West-region Product A sales)
python database.py
```

### Run the REST API (powers the web dashboard)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# or: python api.py
```

### Run the Streamlit app (standalone, no API needed)

```bash
streamlit run app.py
```

### Run the CLI

```bash
python agent.py "Why did revenue fall in August 2026?"
python agent.py --json "What was total revenue in July 2026?"
```

---

## API reference

Swagger UI is available at `http://localhost:8000/docs`.

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /api/analyze`

Body:

```json
{ "question": "Why did revenue fall in August 2026?" }
```

Response — the full investigation state:

```jsonc
{
  "question": "Why did revenue fall in August 2026?",
  "plan": ["total revenue by month ...", "...", "..."],            // ordered investigation steps
  "queries": ["SELECT ...", "..."],                                // every SQL run (read-only)
  "observations": ["STEP: ...\n...", "..."],                       // per-step analyst commentary
  "findings": [                                                     // query + result + rows + columns
    { "step": "...", "query": "SELECT ...", "result": "...", "rows": [...], "columns": [...] }
  ],
  "final_answer": {                                                 // structured answer
    "executive_summary": "...",
    "main_cause": "...",
    "key_drivers": ["..."],
    "evidence": ["..."],
    "recommended_investigation": ["..."],
    "chart_captions": ["..."]
  },
  "charts": [                                                       // chart specs from real rows
    { "title": "...", "type": "bar|line", "x": [...], "y": [...], "x_label": "...", "y_label": "...",
      "subtitle": "...", "query": "..." }
  ]
}
```

Forecast / anomaly results appear inside `findings` with a `forecast` or `anomalies` payload when the question requests them.

---

## Frontend setup

```bash
cd ai-data-analyst-ui

npm install

# Point the dashboard at your local FastAPI backend
# (Create .env.local with:)
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

npm run dev        # http://localhost:3000
```

The dashboard captures screenshot-quality PDFs client-side, so run it on a desktop browser.

---

## Frontend features

- **Ask a question** — type any business question; `Ctrl+Enter` runs it; clickable example questions are provided when the input is empty.
- **Result tabs** — `Answer`, `Plan`, `Queries`, `Observations`, `Charts`.
- **Structured answer cards** — Executive Summary, Main Cause, Key Drivers, Evidence (with raw query results), Recommended Investigation — color-coded with left-border accents.
- **Interactive charts** — Plotly charts rendered client-side with a responsive layout; every chart is derived from real query results. A `spec.data` / `spec.layout` pass-through is supported for richer chart specs.
- **Export** — one click turns any investigation into **JSON**, **CSV**, or **PDF** (timestamps in filenames, e.g. `investigation_2026-09-05T10-30-00.pdf`).
- **Copy / share** — copy the full results to the clipboard with a toast confirmation.
- **History page** (`/history`) — every investigation is saved to `localStorage` (`investigationHistory`, capped at the latest 50). The page shows a searchable, clearable list on the left and the full result (tabs + charts + export) on the right.
- **Quick stats** — total investigations, average runtime, and success rate from past sessions.
- **Animated loading** — a 4-step progress sequence (understanding → generating SQL → executing → analyzing) with a progress bar while the agent works.
- **Keyboard shortcuts** — `Ctrl+Enter` analyze, `?` help modal, `H` jump to history, `Esc` close modals.
- **Sidebar navigation** — Dashboard ↔ History with active-page highlighting and a user profile footer.

---

## LangSmith tracing (optional, free)

Set these in your `.env` (see `.env.example`) to inspect every run's trace — question, tool/SQL calls, results, decisions, and final answer — in the LangSmith UI:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=ai-data-analyst
```

Without these values, tracing is simply disabled and everything still works. You can obtain a key at https://smith.langchain.com.

---

## Demo database seed

`database.py` creates a fake sales database (`customers`, `products`, `sales`) covering Jan–Aug 2026 and deliberately plants an **anomaly: in August 2026, Product A sales are suppressed in the West region** — which is what makes *"Why did revenue fall in August 2026?"* an interesting investigation, and what lets the forecasting / anomaly tools demonstrate their value.

---

## Project layout

```
ai-data-analyst/                # backend repo (this repo)
├── api.py                      # FastAPI REST API (POST /api/analyze, GET /health)
├── agent.py                    # CLI runner for the investigation graph
├── app.py                      # Streamlit front end
├── graph.py                    # LangGraph investigation loop (state + nodes + routing + LLM fallback)
├── charts.py                   # Plotly chart renderer for investigation chart specs
├── tools.py                    # read-only SQL tools (get_database_schema, execute_sql)
├── tools_forecasting.py        # forecast_revenue tool (Prophet or exponential smoothing)
├── tools_anomaly.py            # detect_anomalies tool (Z-score method)
├── database.py                 # generates the demo SQLite sales database
├── .env.example                # template for API keys / LLM provider order / tracing
├── data/sales.db               # SQLite database (generated)
└── graph_diagram.png           # auto-generated graph visualization

ai-data-analyst-ui/             # frontend repo (sibling)
├── src/app/page.tsx            # main dashboard
├── src/app/history/page.tsx    # history page (query history + saved investigations)
├── src/hooks/useInvestigation.ts # Investigation type + API hook (POST /api/analyze)
└── .env.local                  # NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Roadmap / what's left

- **(Optional)** commit/ignore strategy for `graph_diagram.png` (currently auto-regenerated on import).
- Automated tests (backend pytest for `run_investigation`, frontend component tests).
- CI (lint + typecheck + build) on push.
- Optional auth / server-side sync instead of `localStorage` history.
- Deployment guide for the FastAPI backend and Next.js frontend.