import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import agent.config  # triggers .env load and sets ANTHROPIC_API_KEY

from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

from agent.prompts import SYSTEM_PROMPT
from agent.tools.data_retrieval import (
    get_category_overview, get_revenue_by_category, get_revenue_timeseries,
    get_top_items, get_price_distribution, get_store_revenue, get_item_data,
)
from agent.tools.elasticity import (
    compute_log_log_elasticity, classify_elasticity,
    estimate_causal_elasticity, estimate_avg_elasticity, run_causal_graph,
)
from agent.tools.forecasting import forecast_demand, forecast_demand_ml, what_if_price_change
from agent.tools.visualization import plot_bar, plot_timeseries, plot_scatter, plot_distribution

_TRACES_DIR = Path(__file__).resolve().parent.parent / "workflows" / "traces"
_TRACES_DIR.mkdir(parents=True, exist_ok=True)

ALL_TOOLS = [
    get_category_overview, get_revenue_by_category, get_revenue_timeseries,
    get_top_items, get_price_distribution, get_store_revenue, get_item_data,
    compute_log_log_elasticity, classify_elasticity,
    estimate_causal_elasticity, estimate_avg_elasticity, run_causal_graph,
    forecast_demand, forecast_demand_ml, what_if_price_change,
    plot_bar, plot_timeseries, plot_scatter, plot_distribution,
]


class TraceLogger(BaseCallbackHandler):
    def __init__(self):
        self.tool_calls: list[dict] = []
        self._start_times: dict[str, tuple] = {}

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        run_id = str(kwargs.get("run_id", ""))
        name = serialized.get("name", "unknown")
        self._start_times[run_id] = (time.time(), name)

    def on_tool_end(self, output: str, **kwargs):
        run_id = str(kwargs.get("run_id", ""))
        entry = self._start_times.pop(run_id, (time.time(), "unknown"))
        t, name = entry
        self.tool_calls.append({
            "tool": name,
            "output_preview": str(output)[:500],
            "duration_ms": round((time.time() - t) * 1000),
        })

    def on_tool_error(self, error: Exception, **kwargs):
        run_id = str(kwargs.get("run_id", ""))
        entry = self._start_times.pop(run_id, (time.time(), "unknown"))
        _, name = entry
        self.tool_calls.append({
            "tool": name,
            "error": str(error),
            "duration_ms": -1,
        })


def run_agent(question: str) -> str:
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    agent = create_agent(llm, ALL_TOOLS, system_prompt=SYSTEM_PROMPT)
    logger = TraceLogger()

    started = datetime.now(timezone.utc).isoformat()
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"callbacks": [logger], "recursion_limit": 30},
    )
    final_answer = result["messages"][-1].content

    trace = {
        "question": question,
        "started_at": started,
        "tool_calls": logger.tool_calls,
        "final_answer": final_answer,
    }
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace_path = _TRACES_DIR / f"{ts}.json"
    trace_path.write_text(json.dumps(trace, indent=2, default=str))
    print(f"\n[trace saved -> {trace_path}]", file=sys.stderr)
    return final_answer


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are the top revenue categories?"
    answer = run_agent(question)
    print("\n" + "=" * 60)
    print(answer)
