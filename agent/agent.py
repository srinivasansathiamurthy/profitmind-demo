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
        print(f"\n  → {name}...", file=sys.stderr, flush=True)

    def on_tool_end(self, output: str, **kwargs):
        run_id = str(kwargs.get("run_id", ""))
        entry = self._start_times.pop(run_id, (time.time(), "unknown"))
        t, name = entry
        ms = round((time.time() - t) * 1000)
        print(f"     ✓ {ms}ms", file=sys.stderr, flush=True)
        self.tool_calls.append({
            "tool": name,
            "output_preview": str(output)[:500],
            "duration_ms": ms,
        })

    def on_tool_error(self, error: Exception, **kwargs):
        run_id = str(kwargs.get("run_id", ""))
        entry = self._start_times.pop(run_id, (time.time(), "unknown"))
        _, name = entry
        print(f"     ✗ error: {error}", file=sys.stderr, flush=True)
        self.tool_calls.append({
            "tool": name,
            "error": str(error),
            "duration_ms": -1,
        })


def _stream_text(chunk) -> str:
    """Extract text from an AIMessageChunk content (str or list of content blocks)."""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") if isinstance(part, dict) else str(part)
        for part in content
        if not isinstance(part, dict) or part.get("type") == "text"
    )


def run_agent(question: str) -> str:
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    agent = create_agent(llm, ALL_TOOLS, system_prompt=SYSTEM_PROMPT)
    logger = TraceLogger()

    started = datetime.now(timezone.utc).isoformat()
    final_answer_parts: list[str] = []
    _announced_tool_ids: set[str] = set()  # track tool calls already announced

    for chunk, metadata in agent.stream(
        {"messages": [HumanMessage(content=question)]},
        config={"callbacks": [logger], "recursion_limit": 30},
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") == "model":
            # Announce each tool call the moment its name arrives in the stream,
            # before the (potentially large) args finish generating.
            for tc in getattr(chunk, "tool_call_chunks", []):
                call_id = tc.get("id") or ""
                name = tc.get("name") or ""
                if call_id and name and call_id not in _announced_tool_ids:
                    _announced_tool_ids.add(call_id)
                    print(f"\n  → {name} (preparing...)", file=sys.stderr, flush=True)

            text = _stream_text(chunk)
            if text:
                print(text, end="", flush=True)
                final_answer_parts.append(text)

    print()  # newline after streamed output
    final_answer = "".join(final_answer_parts)

    trace = {
        "question": question,
        "started_at": started,
        "tool_calls": logger.tool_calls,
        "final_answer": final_answer,
    }
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace_path = _TRACES_DIR / f"{ts}.json"
    trace_path.write_text(json.dumps(trace, indent=2, default=str))
    print(f"[trace saved -> {trace_path}]", file=sys.stderr)
    return final_answer


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are the top revenue categories?"
    run_agent(question)
