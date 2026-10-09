# ProfitMind

A natural-language retail pricing-insights agent built on the [Dominick's Finer Foods](https://www.chicagobooth.edu/research/kilts/datasets/dominicks) scanner dataset (Kilts Center, University of Chicago). Ask questions about price elasticity, demand forecasting, and margin optimization in plain English — the agent translates them into SQL queries and causal/ML analyses over Databricks, then returns a structured answer with charts.

---

## What it does

| Question type | Example |
|---|---|
| Data exploration | *"Which categories drive the most revenue?"* |
| Price elasticity | *"Which beer items are most inelastic?"* |
| Causal estimation | *"What is the causal price elasticity for Soft Drinks?"* |
| Demand forecasting | *"Forecast demand for item 1234567 over the next 12 weeks"* |
| What-if analysis | *"What happens to Budweiser sales if we raise the price by 10%?"* |
| Visualization | *"Show me the revenue trend for Cereals by month"* |

Every run logs a structured trace (`workflows/traces/<timestamp>.json`) capturing all tool calls, inputs, outputs, durations, and the final answer.

---

## Data

**Source:** Dominick's Finer Foods weekly store-level scanner data, 1989–1997.  
**Storage:** Databricks on GCP, Delta Lake, Unity Catalog.  
**Primary table:** `dominicks.gold.movement` — ~393K rows, one per store × item × week.

| Column | Type | Description |
|---|---|---|
| `category` | string | Product category (Beer, Cheeses, Soft Drinks, …) |
| `store` | int | Store number |
| `week` | int | Dominick's week number |
| `week_start` / `week_end` | date | Thursday–Wednesday weekly window |
| `move` | int | Units sold |
| `qty` | int | Bundle size (`price` applies to `qty` units) |
| `price` | double | Shelf price for `qty` units |
| `unit_price` | double | `price / qty` |
| `gross_revenue` | decimal | `price × move / qty` |
| `net_revenue` | decimal | `gross_revenue × margin_pct / 100` (gross profit) |
| `margin_percentage` | double | Gross margin % |
| `nitem` | string | Numeric item code (UPC stub) — primary item key |

---

## Architecture

```
User question
     │
     ▼
┌─────────────────────────────────────────────┐
│  LangGraph agent  (Claude Sonnet via Anthropic API)  │
│  System prompt + ontology context                    │
└──────────────┬──────────────────────────────┘
               │ tool calls
       ┌───────┴────────┐
       │   Middleware   │  validated_tool decorator
       │  ① deterministic validation (category names, bounds)
       │  ② input normalization (canonical casing)
       │  ③ Haiku semantic check (nitem ≠ description)
       └───────┬────────┘
               │
    ┌──────────┼──────────────┬──────────────┐
    ▼          ▼              ▼              ▼
Data        Elasticity    Forecasting   Visualization
Retrieval   tools         tools         tools
(SQL →      (OLS /        (statsforecast (matplotlib
 Databricks) EconML /      / mlforecast)  → PNG files)
             DoubleML /
             DoWhy)
               │
               ▼
    workflows/traces/<timestamp>.json
```

### Tool categories

**Data retrieval** (`agent/tools/data_retrieval.py`) — Databricks SQL over the gold table:
- `get_category_overview` — row counts, item/store/week counts, date range, revenue
- `get_revenue_by_category` — ranked revenue with share %
- `get_revenue_timeseries` — weekly or monthly revenue over time
- `get_top_items` — top N items by revenue within a category
- `get_price_distribution` — unit_price quartiles, mean, coefficient of variation
- `get_store_revenue` — per-store revenue with cumulative concentration
- `get_item_data` — raw store-week rows for a specific item or category

**Elasticity** (`agent/tools/elasticity.py`):
- `compute_log_log_elasticity` — OLS proxy: `cov(log_price, log_units) / var(log_price)`
- `classify_elasticity` — labels items elastic / inelastic / unit-elastic (|corr| ≥ 0.3 filter)
- `estimate_causal_elasticity` — EconML `CausalForestDML` for per-item heterogeneous effects
- `estimate_avg_elasticity` — DoubleML `PLR` for category-level average with 95% CI
- `run_causal_graph` — DoWhy causal model with random-common-cause refutation

**Forecasting** (`agent/tools/forecasting.py`):
- `forecast_demand` — statsforecast `AutoARIMA` + `AutoETS`
- `forecast_demand_ml` — mlforecast LightGBM with price as exogenous feature
- `what_if_price_change` — side-by-side baseline vs. scenario forecasts with revenue delta

**Visualization** (`agent/tools/visualization.py`) — all return PNG file paths:
- `plot_bar`, `plot_timeseries`, `plot_scatter`, `plot_distribution`

### Input validation middleware

All tools are decorated with `@validated_tool` (`agent/middleware.py`) instead of bare `@tool`. Before any tool runs:

1. **Deterministic checks** — validates category names against the known list (with a "did you mean?" hint on near-misses), enforces enum values (`granularity`), and bounds-checks numeric params (`n`, `h_weeks`, `limit`, `pct_change`, `price_scenario`). Zero latency.
2. **Normalization** — silently fixes category casing (`'beer'` → `'Beer'`).
3. **Haiku semantic check** — fires only when `nitem` contains letters or spaces, catching cases where the model passes a brand name instead of a numeric UPC code. Redirects to `get_top_items` with an explanation. Fails open on API errors.

---

## Setup

### Prerequisites

- Python 3.10+
- A Databricks workspace with the `dominicks` catalog (see [Data pipeline](#data-pipeline))
- An Anthropic API key

### Install

```bash
git clone https://github.com/<you>/profitmind-demo
cd profitmind-demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Configure

Create a `.env` file in the repo root:

```
anthropic_key=sk-ant-...
DATABRICKS_HOST=8259557357780062.2.gcp.databricks.com
DATABRICKS_TOKEN=<your-personal-access-token>
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/<your-warehouse-id>
```

The Databricks host is pre-filled. To find your HTTP path: Databricks workspace → SQL Warehouses → your warehouse → Connection details.

### Run

```bash
python -m agent.agent "<your question>"
```

Output streams token-by-token to stdout. Tool calls are announced to stderr as they fire. Every run saves a trace to `workflows/traces/<timestamp>.json`.

---

## Demo

Three prompts that show the range of what the agent can do.

### 1. Capability discovery

```
python -m agent.agent "look at your list of tools that you can call. what types of questions can you answer?"
```

The agent inspects its own tool definitions and explains what it can do — price distribution, elasticity estimation, causal inference, demand forecasting, what-if scenarios, and visualization — grounded in what the tools actually support rather than a generic description.

---

### 2. Price variation analysis

```
python -m agent.agent "How much do prices vary for Cheeses across stores and weeks?"
```

Calls `get_price_distribution` and `get_item_data`, then visualizes the spread. Sample output:

| Statistic | Value |
|---|---|
| Min unit price | $0.05 |
| Median | $2.25 |
| Mean | $2.41 |
| Max | $111.33 |
| **Coefficient of Variation** | **0.47** |

CV of 0.47 across 564 SKUs and 93 stores — driven by promotional pricing, bundle sizes, and store-level variation. High enough CV to make elasticity estimation reliable.

---

### 3. Elasticity classification

```
python -m agent.agent "Label all Beer items as elastic or inelastic."
```

Calls `classify_elasticity` (OLS log-log, |corr| ≥ 0.3 filter, ≥ 52 weeks), then plots the distribution. Sample output across 265 qualifying Beer items:

| Class | Count | Share |
|---|---|---|
| Elastic (e < −1) | 253 | 95.5% |
| Other (positive elasticity) | 12 | 4.5% |
| Inelastic (−1 ≤ e < 0) | 0 | 0% |

Beer is overwhelmingly price-elastic. The least sensitive items — Labatt's Blue (−1.04), Budweiser (−1.44) — sit just past the elastic threshold. Twelve premium/import items (Corona, Anchor Steam, Zima) show anomalous positive elasticity, consistent with Veblen-good or promotion-confound effects.

---

## Repo structure

```
profitmind-demo/
├── .env                          # credentials (not committed)
├── agent/
│   ├── config.py                 # .env loading, Databricks connection factory
│   ├── ontology.py               # column semantics, category list, elasticity constants
│   ├── middleware.py             # @validated_tool: deterministic + Haiku validation
│   ├── prompts.py                # system prompt referencing ontology
│   ├── agent.py                  # LangGraph agent, TraceLogger, CLI entrypoint
│   └── tools/
│       ├── data_retrieval.py     # 7 SQL-backed tools
│       ├── elasticity.py         # 5 elasticity tools (OLS, EconML, DoubleML, DoWhy)
│       ├── forecasting.py        # 3 forecasting tools (statsforecast, mlforecast)
│       └── visualization.py      # 4 plot tools → PNG files
├── databricks/                   # Databricks notebooks (bronze → silver → gold pipeline)
├── data/                         # local CSVs (gitignored)
├── scratchwork/                  # EDA notebooks, upload scripts
├── workflows/
│   ├── traces/                   # JSON run logs (one file per agent invocation)
│   └── viz/                      # PNG chart outputs
└── requirements.txt
```

---

## Trace format

Every agent run produces a JSON log:

```json
{
  "question": "Which beer items are most inelastic?",
  "started_at": "2026-10-09T13:57:26+00:00",
  "tool_calls": [
    {
      "tool": "compute_log_log_elasticity",
      "output_preview": "[{\"category\":\"Beer\",\"nitem\":\"...",
      "duration_ms": 3241
    }
  ],
  "final_answer": "..."
}
```

---

## Data pipeline

The Databricks notebooks in `databricks/` built the gold table from raw CSVs:

1. **Raw → Bronze** — CSV upload to GCS, ingested as Delta tables under `dominicks.bronze`
2. **Bronze → Silver** — type casting, `ok=1` filter, UPC description join, week calendar merge
3. **Silver → Gold** — column cleanup, revenue/margin columns added (`gross_revenue`, `net_revenue`, `total_calculated_cost`), clustered by `(category, store, week)`

SQL conventions used throughout: `try_divide()` for all divisions (Databricks serverless ANSI mode), `ROUND(..., 2)` not truncation, backtick-quoted `` `case` `` (reserved word).

---

## Key modeling notes

- **Elasticity formula:** `cov(log(bundle_price), log(move/qty)) / var(log(bundle_price))` — covariance/variance form, not `corr()`, to avoid division by zero on constant-price series
- **Minimum history:** 52 weeks required for elasticity estimation
- **Classification filter:** |corr| ≥ 0.3 required for elastic/inelastic labeling (weak correlation = unreliable estimate)
- **`move` vs. revenue:** `move` = units sold, not dollar movement. `gross_revenue = price × move / qty`
- **Causal controls:** week and store are treated as confounders in all causal models

---

## Dependencies

| Library | Purpose |
|---|---|
| `anthropic` + `langchain-anthropic` | LLM (Claude Sonnet), tool-calling agent |
| `databricks-sql-connector` | Query gold table from local Python |
| `econml` | Heterogeneous causal elasticity (CausalForestDML) |
| `doubleml` | Average elasticity with CI (DoubleML PLR) |
| `dowhy` | Causal graph + refutation |
| `statsforecast` | Statistical demand forecasting (AutoARIMA, AutoETS) |
| `mlforecast` + `lightgbm` | ML demand forecasting with price as exogenous |
| `matplotlib` | Visualization tool outputs |
| `pandas` | DataFrame interchange between tools |
