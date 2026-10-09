# ProfitMind Demo — CLAUDE.md

## Project Overview

ProfitMind is a natural-language retail pricing-insights agent built on the Dominick's Finer Foods scanner dataset (Kilts Center, UChicago). The agent answers price elasticity, demand forecasting, and margin optimization questions via LangChain tool-calling over Databricks SQL.

**Interview demo project** — the goal is a working agent with logged workflow traces, not production polish.

## Repo Structure

```
profitmind_demo/
├── .env                         # DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_HTTP_PATH, OPENAI_API_KEY
├── databricks/                  # Databricks notebook source files (bronze/silver/gold pipeline, EDA)
├── data/                        # local parquet/csv (gitignored)
├── scratchwork/                 # upload scripts, one-off analysis
├── agent/
│   ├── __init__.py
│   ├── config.py                # loads .env, Databricks SQL connection factory
│   ├── ontology.py              # column semantics, derived concepts, domain constants
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── data_retrieval.py    # SQL-based tools (query gold table, aggregations, filters)
│   │   ├── elasticity.py        # EconML / DoubleML / DoWhy causal estimation tools
│   │   ├── forecasting.py       # statsforecast / mlforecast demand forecasting tools
│   │   └── visualization.py    # matplotlib/plotly rendering (takes DataFrames, returns file paths)
│   ├── agent.py                 # main LangChain agent with tool bindings
│   └── prompts.py               # system prompt, tool descriptions for the LLM
├── workflows/
│   └── traces/                  # JSON logs: function calls, chat histories, tool I/O per run
├── requirements.txt
└── README.md
```

## Data Architecture

- **Source:** Dominick's Finer Foods weekly store-level scanner data (1989-1997)
- **Storage:** Databricks on GCP, Delta Lake, Unity Catalog
- **Catalog:** `dominicks` with schemas `raw`, `bronze`, `silver`, `gold`, `meta`
- **Primary table:** `dominicks.gold.movement` — 393K rows, 19 columns, row-for-row copy of silver

### Gold Table Schema

| Column | Type | Description |
|---|---|---|
| category | string | product category (beer, cheese, etc.) |
| store | int | store number |
| week | int | Dominick's week number |
| week_start | date | Thursday start of week |
| week_end | date | Wednesday end of week |
| move | int | units sold in store-week |
| qty | int | bundle size (units per price point) |
| price | double | shelf price for qty units |
| unit_price | double | price / qty |
| gross_revenue | decimal(18,2) | price * move / qty |
| net_revenue | decimal(18,2) | gross_revenue * margin_pct / 100 (gross profit) |
| margin_percentage | double | gross margin % from Dominick's |
| total_calculated_cost | decimal(18,2) | gross_revenue - net_revenue |
| cost_percentage | double | 100 - margin_percentage |
| com_code | string | commodity sub-code within category |
| descrip | string | item description text |
| size | string | package size label |
| case | string | case-pack configuration |
| nitem | string | numeric item code (UPC stub) — primary item identifier |

### Dominick's Conventions

- `move` = units sold (not dollar movement)
- `qty` = bundle size — `price` applies to `qty` units, so unit_price = price / qty
- `margin_percentage` = gross margin percent (renamed from `profit` in raw data)
- `ok = 1` in raw data = usable row (already filtered in silver)
- Weekly data: each row is one store × one item × one week

## Connection

Uses `databricks-sql-connector`. Connection config via `.env`:

```
DATABRICKS_HOST=8259557357780062.2.gcp.databricks.com
DATABRICKS_TOKEN=<personal-access-token>
DATABRICKS_HTTP_PATH=<sql-warehouse-http-path>
OPENAI_API_KEY=<key>
```

## Agent Architecture

### Ontology Layer (`agent/ontology.py`)

Defines column semantics and derived concepts the agent references:
- **Identifiers:** category, store, nitem, com_code
- **Time:** week, week_start, week_end
- **Quantity:** move (units), qty (bundle)
- **Price:** price (bundle), unit_price (per unit)
- **Revenue:** gross_revenue, net_revenue
- **Cost/Margin:** margin_percentage, total_calculated_cost, cost_percentage
- **Derived concepts:** log_price, log_units, revenue_share, price_cv, elasticity, corr_price_qty

### Tool Categories

**Data Retrieval** (`agent/tools/data_retrieval.py`) — PySpark SQL against Databricks:
- `get_category_overview(category?)` → row counts, distinct items/stores/weeks
- `get_revenue_by_category()` → ranked revenue totals
- `get_revenue_timeseries(category?, granularity)` → monthly/weekly revenue
- `get_top_items(category, n)` → top items by revenue with % share
- `get_price_distribution(category?)` → quartiles, mean, CV
- `get_store_revenue()` → per-store revenue with cumulative concentration
- `get_item_data(nitem?, category?)` → raw rows for a specific item or filter

**Elasticity** (`agent/tools/elasticity.py`):
- `compute_log_log_elasticity(category?)` → OLS slope proxy per item (PySpark SQL, same as EDA block)
- `classify_elasticity(category?)` → elastic/inelastic/unit labels per item
- `estimate_causal_elasticity(category)` → EconML CausalForestDML heterogeneous treatment effect
- `estimate_avg_elasticity(category)` → DoubleML PLR for average treatment effect with CI
- `run_causal_graph(category)` → DoWhy causal model with refutation

**Forecasting** (`agent/tools/forecasting.py`):
- `forecast_demand(nitem, h_weeks)` → statsforecast AutoARIMA/ETS per item
- `forecast_demand_ml(nitem, h_weeks, price_scenario?)` → mlforecast with price as exogenous
- `what_if_price_change(nitem, pct_change, h_weeks)` → forecast under a price scenario

**Visualization** (`agent/tools/visualization.py`):
- `plot_bar(data, x, y, title)` → horizontal bar chart
- `plot_timeseries(data, x, y, group?, title)` → line chart
- `plot_scatter(data, x, y, color?, title)` → scatter plot
- `plot_distribution(data, column, title)` → histogram/box
- All return file paths to saved PNGs

### LangChain Agent (`agent/agent.py`)

- Uses `ChatOpenAI` (or Claude via `ChatAnthropic`) with tool-calling
- Tools bound via `@tool` decorator with docstrings as function definitions
- Data retrieval and visualization are separate tool calls (agent decides when to viz)
- System prompt references ontology for column semantics

### Workflow Logging

Every agent run logs to `workflows/traces/<timestamp>.json`:
```json
{
  "question": "user question",
  "messages": [...],
  "tool_calls": [{"tool": "name", "input": {...}, "output": {...}, "duration_ms": 123}],
  "final_answer": "...",
  "tokens_used": 1234
}
```

## Libraries

| Library | Use | Key API |
|---|---|---|
| databricks-sql-connector | Query gold table from local Python | `databricks.sql.connect()` |
| langchain + langchain-openai | Agent orchestration, tool-calling | `ChatOpenAI`, `@tool`, `AgentExecutor` |
| econml | Heterogeneous causal elasticity | `CausalForestDML.fit(Y, T, X=X, W=W)` |
| doubleml | Average elasticity with CI | `DoubleMLPLR(data, ml_l, ml_m).fit()` |
| dowhy | Causal graph + refutation | `CausalModel(data, treatment, outcome, graph)` |
| statsforecast | Statistical demand forecasting | `StatsForecast(models, freq).forecast(df, h)` |
| mlforecast | ML demand forecasting with price scenarios | `MLForecast(models, freq, lags).fit(df).predict(h, X_df)` |
| matplotlib | Visualization tool output | `plt.savefig()` |
| pandas | DataFrame interchange between tools | — |
| lightgbm | Nuisance models in DML, forecast models | `LGBMRegressor()` |

## Coding Conventions

- **Databricks SQL:** all divisions use `try_divide()` (serverless ANSI mode)
- **Rounding:** to the nearest cent (`ROUND(..., 2)`), not truncation
- **Column quoting:** backtick `case` (reserved word)
- **Item identifier:** always use `nitem` as the primary item key
- **Elasticity formula:** `cov(log_price [NOT log unit price], log_(move/qty)) / var(log_price)` — never use `corr()` alone (divides by zero on constant series)
- **Filtering:** items need ≥52 weeks of data for elasticity; |corr| ≥ 0.3 for classification
- **Tool separation:** data retrieval returns DataFrames, visualization takes DataFrames — never mix
- **Logging:** every tool call logs input/output/duration to the trace file

## GCP / Databricks Details

- **GCS bucket:** `gs://profitmind-systemdesign-dominicks`
- **Region:** `us-east1`
- **Workspace:** `https://8259557357780062.2.gcp.databricks.com`
- **Managed table paths:** UUID-based in GCS (`uc_managed/__unitystorage/.../tables/<uuid>`), use `DESCRIBE DETAIL` to map
- **Git folder:** Databricks Git integration syncs `databricks/` folder with GitHub repo

## Quick Start

```bash
cd profitmind_demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in credentials
python -m agent.agent "Which beer items are most inelastic?"
```

You also have anthropic_key in the .env file