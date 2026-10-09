import json
from typing import Annotated

import pandas as pd
from agent.config import run_query, GOLD_TABLE
from agent.middleware import validated_tool

_NITEM_DESC = (
    "Numeric item code (UPC stub), e.g. '1234567890'. The primary item identifier. "
    "Use get_top_items or get_item_data to look up nitem codes first — "
    "do NOT pass brand names or item descriptions here."
)
_H_WEEKS_DESC = (
    "Forecast horizon in weeks (1–260). The dataset ends ~1997 so forecasts are "
    "in-sample projections. 12 weeks (~1 quarter) is a good default."
)


def _to_json(df) -> str:
    return df.to_json(orient="records", date_format="iso")


def _fetch_item_series(nitem: str) -> pd.DataFrame:
    """Weekly aggregate series for an item: unique_id, ds (week_start), y (units), price."""
    sql = f"""
    SELECT
        '{nitem}' AS unique_id,
        week_start AS ds,
        SUM(try_divide(move, qty)) AS y,
        AVG(unit_price) AS price
    FROM {GOLD_TABLE}
    WHERE nitem = '{nitem}' AND move > 0 AND qty > 0
    GROUP BY week_start
    ORDER BY week_start
    """
    df = run_query(sql)
    df["ds"] = pd.to_datetime(df["ds"])
    return df.dropna(subset=["y"])


@validated_tool
def forecast_demand(
    nitem: Annotated[str, _NITEM_DESC],
    h_weeks: Annotated[int, _H_WEEKS_DESC] = 12,
) -> str:
    """
    Forecast weekly demand (units = move/qty) for a single item using statsforecast
    AutoARIMA and AutoETS. Returns h_weeks rows of forecasts from both models.

    Requires >= 52 weeks of history for the item. Use what_if_price_change if you
    want to model how a price change would affect demand — this tool ignores price.
    """
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA, AutoETS
    except ImportError:
        return json.dumps({"error": "statsforecast not installed — run: pip install statsforecast"})

    df = _fetch_item_series(nitem)
    if len(df) < 52:
        return json.dumps({"error": f"nitem='{nitem}' has only {len(df)} weeks of history (need >= 52)"})

    sf = StatsForecast(models=[AutoARIMA(), AutoETS()], freq="W-THU")
    forecast = sf.forecast(df=df[["unique_id", "ds", "y"]], h=int(h_weeks))
    return _to_json(forecast.reset_index())


@validated_tool
def forecast_demand_ml(
    nitem: Annotated[str, _NITEM_DESC],
    h_weeks: Annotated[int, _H_WEEKS_DESC] = 12,
    price_scenario: Annotated[
        float,
        "Fixed unit_price to hold constant across the forecast horizon (>= 0). "
        "0 means use the last observed price. Use this to model a specific price point. "
        "For a % change scenario use what_if_price_change instead.",
    ] = 0.0,
) -> str:
    """
    Forecast weekly demand using mlforecast (LightGBM) with price as an exogenous feature.
    Lags: 1, 2, 4, 8, 12, 52 weeks. Date features: week-of-year, month.

    Unlike forecast_demand, this model treats price as a driver of demand, so changing
    price_scenario will change the forecast. Requires >= 52 weeks of history.
    """
    try:
        from mlforecast import MLForecast
        from lightgbm import LGBMRegressor
    except ImportError:
        return json.dumps({"error": "mlforecast/lightgbm not installed — run: pip install mlforecast lightgbm"})

    df = _fetch_item_series(nitem)
    if len(df) < 52:
        return json.dumps({"error": f"nitem='{nitem}' has only {len(df)} weeks of history (need >= 52)"})

    mlf = MLForecast(
        models=[LGBMRegressor(n_estimators=100, verbose=-1)],
        freq="W-THU",
        lags=[1, 2, 4, 8, 12, 52],
        date_features=["week", "month"],
    )
    mlf.fit(df[["unique_id", "ds", "y", "price"]], static_features=[])

    future_price = price_scenario if price_scenario > 0 else float(df["price"].dropna().iloc[-1])
    future = mlf.make_future_dataframe(h=int(h_weeks))
    future["price"] = future_price
    return _to_json(mlf.predict(h=int(h_weeks), X_df=future))


@validated_tool
def what_if_price_change(
    nitem: Annotated[str, _NITEM_DESC],
    pct_change: Annotated[
        float,
        "Percentage price change to apply, e.g. 10 for a +10% increase, -5 for a 5% cut. "
        "Range -90 to 500. Applied to the last observed unit_price as the baseline.",
    ],
    h_weeks: Annotated[int, _H_WEEKS_DESC] = 12,
) -> str:
    """
    Compare demand forecasts under a price change scenario vs the baseline (current price).

    Trains an mlforecast LightGBM model on historical data (price as exogenous), then
    predicts twice: once at current price (baseline) and once at current_price * (1 + pct_change/100).

    Returns side-by-side: baseline_units, scenario_units, unit_delta (demand change),
    revenue_delta (revenue change), and both price levels. Use to quantify the trade-off
    between price and volume for margin optimization decisions.
    """
    try:
        from mlforecast import MLForecast
        from lightgbm import LGBMRegressor
    except ImportError:
        return json.dumps({"error": "mlforecast/lightgbm not installed — run: pip install mlforecast lightgbm"})

    df = _fetch_item_series(nitem)
    if len(df) < 52:
        return json.dumps({"error": f"nitem='{nitem}' has only {len(df)} weeks of history (need >= 52)"})

    current_price = float(df["price"].dropna().iloc[-1])
    new_price = current_price * (1 + pct_change / 100)

    mlf = MLForecast(
        models=[LGBMRegressor(n_estimators=100, verbose=-1)],
        freq="W-THU",
        lags=[1, 2, 4, 8, 12, 52],
        date_features=["week", "month"],
    )
    mlf.fit(df[["unique_id", "ds", "y", "price"]], static_features=[])

    future_base = mlf.make_future_dataframe(h=int(h_weeks))
    future_base["price"] = current_price
    base = mlf.predict(h=int(h_weeks), X_df=future_base).rename(
        columns={"LGBMRegressor": "baseline_units"}
    )

    future_new = mlf.make_future_dataframe(h=int(h_weeks))
    future_new["price"] = new_price
    scenario = mlf.predict(h=int(h_weeks), X_df=future_new).rename(
        columns={"LGBMRegressor": "scenario_units"}
    )

    result = base.merge(scenario, on=["unique_id", "ds"])
    result["price_change_pct"] = pct_change
    result["baseline_price"] = current_price
    result["scenario_price"] = new_price
    result["unit_delta"] = result["scenario_units"] - result["baseline_units"]
    result["revenue_delta"] = (result["scenario_units"] * new_price) - (result["baseline_units"] * current_price)
    return _to_json(result)
