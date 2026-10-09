import json
from typing import Annotated

import pandas as pd
from agent.config import run_query, GOLD_TABLE
from agent.middleware import validated_tool
from agent.ontology import CATEGORIES, MIN_WEEKS_FOR_ELASTICITY, MIN_CORR_FOR_CLASSIFICATION

_CATEGORY_DESC = (
    "Product category name, e.g. 'Beer', 'Cheeses', 'Soft Drinks'. "
    f"Valid values: {', '.join(CATEGORIES)}."
)
_CATEGORY_OPT_DESC = _CATEGORY_DESC + " Leave empty ('') to run across all categories."
_NITEM_DESC = (
    "Numeric item code (UPC stub), e.g. '1234567890'. "
    "Use get_top_items to look up nitem codes — do NOT pass brand names or descriptions."
)


def _to_json(df) -> str:
    return df.to_json(orient="records", date_format="iso")


@validated_tool
def compute_log_log_elasticity(
    category: Annotated[str, _CATEGORY_OPT_DESC] = "",
) -> str:
    """
    Compute a per-item OLS log-log price elasticity proxy using the formula:
        elasticity = cov(log(bundle_price), log(move/qty)) / var(log(bundle_price))

    A negative elasticity means demand falls when price rises (normal).
    Values < -1 = elastic (price-sensitive); -1 to 0 = inelastic (price-insensitive).

    Only items with >= 52 weeks of data are included. Returns nitem, descrip, category,
    n_weeks, elasticity, and corr_price_units (correlation as a signal quality indicator).
    Use classify_elasticity to bucket results into elastic/inelastic labels.
    """
    where = f"AND category = '{category}'" if category else ""
    sql = f"""
    WITH weekly AS (
        SELECT
            category, nitem, week,
            try_divide(SUM(gross_revenue), SUM(move)) AS avg_price,
            try_divide(SUM(move), MAX(qty)) AS units
        FROM {GOLD_TABLE}
        WHERE move > 0 AND qty > 0 AND price > 0 {where}
        GROUP BY category, nitem, week
    ),
    log_weekly AS (
        SELECT category, nitem, week,
               LN(avg_price) AS lp,
               LN(units) AS lu
        FROM weekly
        WHERE avg_price > 0 AND units > 0
    ),
    stats AS (
        SELECT
            category, nitem,
            COUNT(*) AS n_weeks,
            try_divide(covar_samp(lp, lu), var_samp(lp)) AS elasticity,
            try_divide(covar_samp(lp, lu), stddev_samp(lp) * stddev_samp(lu)) AS corr_price_units,
            avg(lp) AS mean_log_price,
            avg(lu) AS mean_log_units
        FROM log_weekly
        GROUP BY category, nitem
        HAVING n_weeks >= {MIN_WEEKS_FOR_ELASTICITY}
            AND elasticity IS NOT NULL
            AND corr_price_units IS NOT NULL
    )
    SELECT s.*, g.descrip
    FROM stats s
    LEFT JOIN (SELECT nitem, MIN(descrip) AS descrip FROM {GOLD_TABLE} GROUP BY nitem) g
      ON s.nitem = g.nitem
    ORDER BY category, elasticity
    """
    return _to_json(run_query(sql))


@validated_tool
def classify_elasticity(
    category: Annotated[str, _CATEGORY_OPT_DESC] = "",
) -> str:
    """
    Classify items as 'elastic' (elasticity < -1), 'inelastic' (-1 <= e < 0),
    or 'unit-elastic' (|e + 1| < 0.05).

    Stricter than compute_log_log_elasticity: only items with |corr| >= 0.3 are
    included (weak correlation means price variation isn't driving demand and the
    estimate is unreliable). Returns elasticity_class label alongside the raw values.
    """
    where = f"AND category = '{category}'" if category else ""
    sql = f"""
    WITH weekly AS (
        SELECT category, nitem, week,
               LN(try_divide(SUM(gross_revenue), SUM(move))) AS lp,
               LN(try_divide(SUM(move), MAX(qty))) AS lu
        FROM {GOLD_TABLE}
        WHERE move > 0 AND qty > 0 AND price > 0 {where}
        GROUP BY category, nitem, week
    ),
    stats AS (
        SELECT category, nitem,
               COUNT(*) AS n_weeks,
               try_divide(covar_samp(lp, lu), var_samp(lp)) AS elasticity,
               try_divide(covar_samp(lp, lu), stddev_samp(lp) * stddev_samp(lu)) AS corr
        FROM weekly
        WHERE lp IS NOT NULL AND lu IS NOT NULL
        GROUP BY category, nitem
        HAVING n_weeks >= {MIN_WEEKS_FOR_ELASTICITY}
            AND corr IS NOT NULL
            AND ABS(corr) >= {MIN_CORR_FOR_CLASSIFICATION}
    ),
    classified AS (
        SELECT *,
            CASE
                WHEN elasticity < -1          THEN 'elastic'
                WHEN elasticity >= -1
                 AND elasticity < 0           THEN 'inelastic'
                WHEN ABS(elasticity + 1) < 0.05 THEN 'unit-elastic'
                ELSE 'other'
            END AS elasticity_class
        FROM stats
    )
    SELECT c.*, g.descrip
    FROM classified c
    LEFT JOIN (SELECT nitem, MIN(descrip) AS descrip FROM {GOLD_TABLE} GROUP BY nitem) g
      ON c.nitem = g.nitem
    ORDER BY category, elasticity
    """
    return _to_json(run_query(sql))


@validated_tool
def estimate_causal_elasticity(
    category: Annotated[
        str,
        _CATEGORY_DESC + " Required — causal estimation is too slow to run across all categories at once.",
    ],
) -> str:
    """
    Estimate heterogeneous per-item price elasticity using EconML CausalForestDML.

    Models the causal effect of log(bundle_price) on log(move/qty), controlling for
    week and store as confounders (W) and using nitem dummies as effect moderators (X).
    Returns a per-item CATE (conditional average treatment effect) — the causal elasticity
    for each item, accounting for store and time variation that OLS cannot remove.

    Slower than compute_log_log_elasticity. Use when you need statistically rigorous
    heterogeneous effects rather than a quick correlation-based proxy.
    """
    try:
        from econml.dml import CausalForestDML
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        return json.dumps({"error": "econml not installed — run: pip install econml"})

    sql = f"""
    SELECT nitem, week, store,
           LN(try_divide(SUM(gross_revenue), SUM(move))) AS log_price,
           LN(try_divide(SUM(move), MAX(qty))) AS log_units
    FROM {GOLD_TABLE}
    WHERE category = '{category}' AND move > 0 AND qty > 0 AND price > 0
    GROUP BY nitem, week, store
    HAVING log_price IS NOT NULL AND log_units IS NOT NULL
    """
    df = run_query(sql)
    if df.empty:
        return json.dumps({"error": f"No data for category '{category}'"})

    df = df.dropna(subset=["log_price", "log_units"])
    Y = df["log_units"].values
    T = df["log_price"].values
    X = pd.get_dummies(df[["nitem"]], drop_first=True).values.astype(float)
    W = df[["week", "store"]].fillna(0).values.astype(float)

    est = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=50, max_depth=3),
        model_t=GradientBoostingRegressor(n_estimators=50, max_depth=3),
        n_estimators=100,
        random_state=42,
    )
    est.fit(Y, T, X=X, W=W)
    cate = est.effect(X)

    df = df.copy()
    df["cate_elasticity"] = cate
    result = (
        df.groupby("nitem")["cate_elasticity"]
        .mean()
        .reset_index()
        .rename(columns={"cate_elasticity": "causal_elasticity"})
    )
    result["category"] = category
    return _to_json(result)


@validated_tool
def estimate_avg_elasticity(
    category: Annotated[
        str,
        _CATEGORY_DESC + " Required — DoubleML runs per-category.",
    ],
) -> str:
    """
    Estimate the average price elasticity with a 95% confidence interval using
    DoubleML Partially Linear Regression (PLR).

    Orthogonalizes both log(price) and log(units) against week and store controls
    using LightGBM nuisance models, then estimates the causal slope on the residuals.
    Returns avg_elasticity, ci_lower, ci_upper, and p_value.

    Use after compute_log_log_elasticity to add statistical rigour and a CI to the
    category-level average. Not item-level — use estimate_causal_elasticity for that.
    """
    try:
        import doubleml as dml
        from lightgbm import LGBMRegressor
    except ImportError:
        return json.dumps({"error": "doubleml/lightgbm not installed — run: pip install doubleml lightgbm"})

    sql = f"""
    SELECT
        LN(try_divide(SUM(gross_revenue), SUM(move))) AS log_price,
        LN(try_divide(SUM(move), MAX(qty))) AS log_units,
        week, store
    FROM {GOLD_TABLE}
    WHERE category = '{category}' AND move > 0 AND qty > 0 AND price > 0
    GROUP BY week, store
    HAVING log_price IS NOT NULL AND log_units IS NOT NULL
    """
    df = run_query(sql).dropna()
    if len(df) < 50:
        return json.dumps({"error": f"Insufficient data for '{category}' (got {len(df)} rows, need >= 50)"})

    data = dml.DoubleMLData(df, y_col="log_units", d_cols="log_price", x_cols=["week", "store"])
    model = dml.DoubleMLPLR(
        data,
        LGBMRegressor(n_estimators=50, verbose=-1),
        LGBMRegressor(n_estimators=50, verbose=-1),
    )
    model.fit()
    s = model.summary
    return json.dumps({
        "category": category,
        "avg_elasticity": float(s["coef"].iloc[0]),
        "ci_lower": float(s["2.5 %"].iloc[0]),
        "ci_upper": float(s["97.5 %"].iloc[0]),
        "p_value": float(s["P>|t|"].iloc[0]),
    })


@validated_tool
def run_causal_graph(
    category: Annotated[
        str,
        _CATEGORY_DESC + " Required — DoWhy builds a per-category causal model.",
    ],
) -> str:
    """
    Estimate the causal effect of log(bundle_price) on log(move/qty) using DoWhy
    with a hand-specified DAG: week -> price, week -> units, store -> price, store -> units.

    Runs a backdoor linear regression estimator and a random-common-cause refutation test.
    Returns causal_estimate, refutation_new_effect, and refutation_p_value.
    A large change in effect or low p-value in the refutation suggests the estimate
    may not be robust.
    """
    try:
        from dowhy import CausalModel
    except ImportError:
        return json.dumps({"error": "dowhy not installed — run: pip install dowhy"})

    sql = f"""
    SELECT
        LN(try_divide(SUM(gross_revenue), SUM(move))) AS log_price,
        LN(try_divide(SUM(move), MAX(qty))) AS log_units,
        week, store
    FROM {GOLD_TABLE}
    WHERE category = '{category}' AND move > 0 AND qty > 0 AND price > 0
    GROUP BY week, store
    HAVING log_price IS NOT NULL AND log_units IS NOT NULL
    """
    df = run_query(sql).dropna()
    if len(df) < 50:
        return json.dumps({"error": f"Insufficient data for '{category}'"})

    causal_graph = """
    digraph {
        log_price -> log_units;
        week -> log_price; week -> log_units;
        store -> log_price; store -> log_units;
    }
    """
    model = CausalModel(data=df, treatment="log_price", outcome="log_units", graph=causal_graph)
    identified = model.identify_effect()
    estimate = model.estimate_effect(identified, method_name="backdoor.linear_regression")
    refute = model.refute_estimate(identified, estimate, method_name="random_common_cause")
    return json.dumps({
        "category": category,
        "causal_estimate": float(estimate.value),
        "refutation_new_effect": float(refute.new_effect),
        "refutation_p_value": float(refute.refutation_result.get("p_value", -1)),
    })
