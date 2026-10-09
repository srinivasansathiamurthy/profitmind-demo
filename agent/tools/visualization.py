import json
from pathlib import Path
from langchain_core.tools import tool
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_VIZ_DIR = Path(__file__).resolve().parent.parent.parent / "workflows" / "viz"
_VIZ_DIR.mkdir(parents=True, exist_ok=True)


_MAX_VIZ_ROWS = 100  # prevent the model from embedding huge JSONs in tool calls

def _load_df(data_json: str) -> pd.DataFrame:
    records = json.loads(data_json)
    if len(records) > _MAX_VIZ_ROWS:
        records = records[:_MAX_VIZ_ROWS]
    return pd.DataFrame(records)


def _save_fig(fig, name: str) -> str:
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:60]
    path = _VIZ_DIR / f"{safe_name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)


@tool
def plot_bar(data_json: str, x: str, y: str, title: str = "") -> str:
    """
    Create a horizontal bar chart. Returns file path to saved PNG.
    data_json: JSON array of records. x: label column. y: value column. title: chart title.
    """
    df = _load_df(data_json)
    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.35)))
    df_sorted = df.sort_values(y, ascending=True)
    ax.barh(df_sorted[x].astype(str), df_sorted[y])
    ax.set_xlabel(y)
    ax.set_title(title or f"{y} by {x}")
    fig.tight_layout()
    name = title.lower().replace(" ", "_") if title else f"bar_{x}_{y}"
    return _save_fig(fig, name)


@tool
def plot_timeseries(data_json: str, x: str, y: str, group: str = "", title: str = "") -> str:
    """
    Create a line chart. Returns file path to saved PNG.
    data_json: JSON array of records. x: date column. y: value column. group: optional grouping column.
    """
    df = _load_df(data_json)
    df[x] = pd.to_datetime(df[x], errors="coerce")
    fig, ax = plt.subplots(figsize=(14, 5))
    if group and group in df.columns:
        for grp, gdf in df.groupby(group):
            ax.plot(gdf[x], gdf[y], label=str(grp), lw=1)
        ax.legend(fontsize=7, ncol=3)
    else:
        ax.plot(df[x], df[y], lw=1.2)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or f"{y} over {x}")
    fig.tight_layout()
    name = title.lower().replace(" ", "_") if title else f"ts_{y}"
    return _save_fig(fig, name)


@tool
def plot_scatter(data_json: str, x: str, y: str, color: str = "", title: str = "") -> str:
    """
    Create a scatter plot. Returns file path to saved PNG.
    data_json: JSON array of records. x, y: column names. color: optional color column.
    """
    df = _load_df(data_json)
    fig, ax = plt.subplots(figsize=(9, 6))
    if color and color in df.columns:
        for g, sub in df.groupby(color):
            ax.scatter(sub[x], sub[y], label=str(g), alpha=0.6, s=20)
        ax.legend(fontsize=7)
    else:
        ax.scatter(df[x], df[y], alpha=0.5, s=20)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or f"{y} vs {x}")
    fig.tight_layout()
    name = title.lower().replace(" ", "_") if title else f"scatter_{x}_{y}"
    return _save_fig(fig, name)


@tool
def plot_distribution(data_json: str, column: str, title: str = "") -> str:
    """
    Create a histogram with box plot for a column. Returns file path to saved PNG.
    data_json: JSON array of records. column: numeric column name.
    """
    df = _load_df(data_json)
    vals = pd.to_numeric(df[column], errors="coerce").dropna()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": [3, 1]})
    ax1.hist(vals, bins=50, edgecolor="white", linewidth=0.3)
    ax1.set_xlabel(column)
    ax1.set_ylabel("count")
    ax1.set_title(title or f"Distribution of {column}")
    ax2.boxplot(vals, vert=False, widths=0.6)
    ax2.set_xlabel(column)
    fig.tight_layout()
    name = title.lower().replace(" ", "_") if title else f"dist_{column}"
    return _save_fig(fig, name)
