from typing import Annotated

from agent.config import run_query, GOLD_TABLE
from agent.middleware import validated_tool
from agent.ontology import CATEGORIES, MIN_WEEKS_FOR_ELASTICITY

_CATEGORY_DESC = (
    "Product category name, e.g. 'Beer', 'Cheeses', 'Soft Drinks'. "
    f"Valid values: {', '.join(CATEGORIES)}. "
    "Leave empty ('') to include all categories."
)
_NITEM_DESC = (
    "Numeric item code (UPC stub), the primary item identifier, e.g. '1234567890'. "
    "Use get_top_items or get_item_data first to look up nitem codes — "
    "do NOT pass item descriptions or brand names here."
)


def _to_json(df) -> str:
    return df.to_json(orient="records", date_format="iso")


@validated_tool
def get_category_overview(
    category: Annotated[str, _CATEGORY_DESC] = "",
) -> str:
    """
    Return row counts, distinct item/store/week counts, date range, and total gross revenue
    for one category or all categories (leave category empty).
    Use this as a first-pass overview before drilling into specific analyses.
    """
    where = f"WHERE category = '{category}'" if category else ""
    sql = f"""
    SELECT
        category,
        COUNT(*) AS rows,
        COUNT(DISTINCT nitem) AS distinct_items,
        COUNT(DISTINCT store) AS distinct_stores,
        COUNT(DISTINCT week) AS distinct_weeks,
        MIN(week_start) AS first_week,
        MAX(week_end) AS last_week,
        CAST(ROUND(SUM(gross_revenue), 2) AS DOUBLE) AS total_gross_revenue
    FROM {GOLD_TABLE}
    {where}
    GROUP BY category
    ORDER BY total_gross_revenue DESC
    """
    return _to_json(run_query(sql))


@validated_tool
def get_revenue_by_category() -> str:
    """
    Return total gross revenue, net revenue (gross profit), revenue share %, item count,
    and store count for every category, ranked by revenue descending.
    Use to compare categories or identify the highest-value segments.
    """
    sql = f"""
    SELECT
        category,
        CAST(ROUND(SUM(gross_revenue), 2) AS DOUBLE) AS total_gross_revenue,
        CAST(ROUND(SUM(net_revenue), 2) AS DOUBLE) AS total_net_revenue,
        CAST(ROUND(100 * SUM(gross_revenue) / SUM(SUM(gross_revenue)) OVER (), 2) AS DOUBLE) AS revenue_share_pct,
        COUNT(DISTINCT nitem) AS n_items,
        COUNT(DISTINCT store) AS n_stores
    FROM {GOLD_TABLE}
    GROUP BY category
    ORDER BY total_gross_revenue DESC
    """
    return _to_json(run_query(sql))


@validated_tool
def get_revenue_timeseries(
    category: Annotated[str, _CATEGORY_DESC] = "",
    granularity: Annotated[
        str,
        "Time aggregation level. 'weekly' returns one row per week_start date; "
        "'monthly' groups by calendar month via DATE_TRUNC. Default 'monthly'.",
    ] = "monthly",
) -> str:
    """
    Return a revenue time series (gross_revenue, net_revenue, total_units) over time.
    Pass category to scope to one segment; leave empty for all categories combined.
    Pass the result to plot_timeseries to visualize trends.
    """
    where = f"AND category = '{category}'" if category else ""
    if granularity == "weekly":
        time_expr = "week_start"
        group_by = "week_start"
    else:
        time_expr = "DATE_TRUNC('MONTH', week_start)"
        group_by = "DATE_TRUNC('MONTH', week_start)"
    cat_col = "category," if not category else ""
    cat_group = ", category" if not category else ""
    sql = f"""
    SELECT
        {time_expr} AS period,
        {cat_col}
        CAST(ROUND(SUM(gross_revenue), 2) AS DOUBLE) AS gross_revenue,
        CAST(ROUND(SUM(net_revenue), 2) AS DOUBLE) AS net_revenue,
        SUM(move) AS total_units
    FROM {GOLD_TABLE}
    WHERE 1=1 {where}
    GROUP BY {group_by}{cat_group}
    ORDER BY period
    """
    return _to_json(run_query(sql))


@validated_tool
def get_top_items(
    category: Annotated[str, _CATEGORY_DESC],
    n: Annotated[int, "Number of top items to return, ranked by gross revenue. Range 1–200."] = 10,
) -> str:
    """
    Return the top N items in a category by total gross revenue, including nitem code,
    item description, total units sold, and revenue share %.
    Use this to identify which items drive the most value, or to look up nitem codes
    before calling elasticity or forecasting tools.
    """
    sql = f"""
    WITH item_rev AS (
        SELECT
            nitem,
            MIN(descrip) AS descrip,
            CAST(ROUND(SUM(gross_revenue), 2) AS DOUBLE) AS gross_revenue,
            SUM(move) AS total_units
        FROM {GOLD_TABLE}
        WHERE category = '{category}'
        GROUP BY nitem
    ),
    total AS (SELECT SUM(gross_revenue) AS cat_rev FROM item_rev)
    SELECT
        i.nitem,
        i.descrip,
        i.gross_revenue,
        i.total_units,
        CAST(ROUND(100 * try_divide(i.gross_revenue, t.cat_rev), 2) AS DOUBLE) AS revenue_share_pct
    FROM item_rev i CROSS JOIN total t
    ORDER BY i.gross_revenue DESC
    LIMIT {int(n)}
    """
    return _to_json(run_query(sql))


@validated_tool
def get_price_distribution(
    category: Annotated[str, _CATEGORY_DESC] = "",
) -> str:
    """
    Return unit_price distribution statistics (min, p25, median, p75, max, mean, CV)
    for one category or all categories.
    price_cv (coefficient of variation = stddev/mean) indicates how much prices vary —
    high CV means prices fluctuate a lot, which is a prerequisite for elasticity estimation.
    """
    where = f"WHERE category = '{category}'" if category else ""
    cat_col = "category," if not category else ""
    group_order = "GROUP BY category ORDER BY mean_price DESC" if not category else ""
    sql = f"""
    SELECT
        {cat_col}
        CAST(ROUND(MIN(unit_price), 4) AS DOUBLE) AS min_price,
        CAST(ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY unit_price), 4) AS DOUBLE) AS p25,
        CAST(ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY unit_price), 4) AS DOUBLE) AS median_price,
        CAST(ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY unit_price), 4) AS DOUBLE) AS p75,
        CAST(ROUND(MAX(unit_price), 4) AS DOUBLE) AS max_price,
        CAST(ROUND(AVG(unit_price), 4) AS DOUBLE) AS mean_price,
        CAST(ROUND(try_divide(STDDEV(unit_price), AVG(unit_price)), 4) AS DOUBLE) AS price_cv
    FROM {GOLD_TABLE}
    {where}
    {group_order}
    """
    return _to_json(run_query(sql))


@validated_tool
def get_store_revenue() -> str:
    """
    Return per-store total gross revenue, total units sold, rank, and cumulative revenue
    share % (useful for identifying store concentration — e.g. top 10 stores = X% of sales).
    """
    sql = f"""
    WITH store_rev AS (
        SELECT
            store,
            CAST(ROUND(SUM(gross_revenue), 2) AS DOUBLE) AS gross_revenue,
            SUM(move) AS total_units
        FROM {GOLD_TABLE}
        GROUP BY store
    ),
    ranked AS (
        SELECT *,
            ROW_NUMBER() OVER (ORDER BY gross_revenue DESC) AS rank,
            CAST(ROUND(100 * SUM(gross_revenue) OVER (
                ORDER BY gross_revenue DESC
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) / SUM(gross_revenue) OVER (), 2) AS DOUBLE) AS cum_share_pct
        FROM store_rev
    )
    SELECT * FROM ranked ORDER BY rank
    """
    return _to_json(run_query(sql))


@validated_tool
def get_item_data(
    nitem: Annotated[str, _NITEM_DESC] = "",
    category: Annotated[str, _CATEGORY_DESC] = "",
    limit: Annotated[
        int,
        "Maximum number of rows to return (1–5000). Each row is one store × week observation.",
    ] = 500,
) -> str:
    """
    Return raw store-week rows for a specific item (filter by nitem) or a whole category.
    Columns: category, store, week, week_start, nitem, descrip, move (units sold),
    qty (bundle size), price (bundle price), unit_price, gross_revenue, net_revenue,
    margin_percentage. At least one of nitem or category should be provided.
    """
    filters = []
    if nitem:
        filters.append(f"nitem = '{nitem}'")
    if category:
        filters.append(f"category = '{category}'")
    where = "WHERE " + " AND ".join(filters) if filters else ""
    sql = f"""
    SELECT category, store, week, week_start, nitem, descrip,
           move, qty, price, unit_price,
           CAST(gross_revenue AS DOUBLE) AS gross_revenue,
           CAST(net_revenue AS DOUBLE) AS net_revenue,
           margin_percentage
    FROM {GOLD_TABLE}
    {where}
    ORDER BY store, week
    LIMIT {int(limit)}
    """
    return _to_json(run_query(sql))
