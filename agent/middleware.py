import functools
import json

from langchain_core.tools import tool as _lc_tool
from agent.ontology import CATEGORIES

_CATEGORY_MAP = {c.lower(): c for c in CATEGORIES}


def _normalize_category(val: str) -> str | None:
    """Return canonical category name, '' (all categories), or None (unrecognized)."""
    if not val:
        return ""
    lower = val.strip().lower()
    if lower in _CATEGORY_MAP:
        return _CATEGORY_MAP[lower]
    matches = [c for k, c in _CATEGORY_MAP.items() if lower in k or k in lower]
    return matches[0] if len(matches) == 1 else None


def _deterministic_errors(tool_name: str, params: dict) -> list[str]:
    errors = []

    if params.get("category"):
        canon = _normalize_category(params["category"])
        if canon is None:
            close = [c for c in CATEGORIES if params["category"].lower() in c.lower()]
            hint = f" Did you mean: {close[0]!r}?" if close else f" Valid: {CATEGORIES[:4]}..."
            errors.append(f"Unknown category {params['category']!r}.{hint}")

    if "granularity" in params and params["granularity"] not in ("weekly", "monthly"):
        errors.append(f"granularity must be 'weekly' or 'monthly', got {params['granularity']!r}")

    for param, lo, hi in [("n", 1, 200), ("h_weeks", 1, 260), ("limit", 1, 5000)]:
        if param in params:
            v = params[param]
            if not (isinstance(v, int) and lo <= v <= hi):
                errors.append(f"{param} must be an integer {lo}–{hi}, got {v!r}")

    if "pct_change" in params:
        v = params["pct_change"]
        if not (isinstance(v, (int, float)) and -90 <= v <= 500):
            errors.append(f"pct_change must be -90 to 500, got {v!r}")

    if "price_scenario" in params:
        v = params["price_scenario"]
        if not (isinstance(v, (int, float)) and v >= 0):
            errors.append(f"price_scenario must be >= 0, got {v!r}")

    return errors


def _normalize_inputs(params: dict) -> dict:
    if params.get("category"):
        canon = _normalize_category(params["category"])
        if canon is not None:
            params = {**params, "category": canon}
    return params


def _nitem_looks_like_description(nitem: str) -> bool:
    """True if nitem contains letters or spaces — likely a description, not a UPC stub."""
    return bool(nitem) and (any(c.isalpha() for c in nitem) or " " in nitem)


def _haiku_nitem_check(tool_name: str, nitem: str) -> tuple[bool, str]:
    """
    Haiku sanity check: did the caller pass an item description instead of a numeric nitem code?
    Returns (is_valid, message). Fails open on API errors — never blocks the tool.
    """
    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            messages=[{
                "role": "user",
                "content": (
                    f"In the Dominick's grocery dataset, 'nitem' is a numeric UPC stub "
                    f"like '1234567890'. The caller passed nitem={nitem!r} to '{tool_name}'.\n"
                    "Is this a valid numeric nitem code, or did they pass an item description by mistake?\n"
                    "Reply exactly VALID or INVALID:<reason under 12 words>."
                ),
            }],
        )
        text = resp.content[0].text.strip()
        return text.startswith("VALID"), text
    except Exception as exc:
        # Fail open — log but don't block
        return True, f"haiku_check_skipped({exc})"


def validated_tool(func):
    """
    Drop-in replacement for @tool with three-stage input middleware:

    1. Deterministic validation — category names, enum values, numeric bounds.
       Fast, no API call. Returns a validation_error JSON on failure.

    2. Normalization — canonicalizes category casing so 'beer' -> 'Beer'.

    3. Haiku semantic check — only fires when nitem looks like a description
       (contains letters/spaces). Catches the model passing 'Budweiser Beer'
       where a UPC stub like '1234567' is required. Fails open on API errors.
    """
    @functools.wraps(func)
    def wrapper(**kwargs):
        errors = _deterministic_errors(func.__name__, kwargs)
        if errors:
            return json.dumps({"validation_error": "; ".join(errors)})

        kwargs = _normalize_inputs(kwargs)

        nitem = str(kwargs.get("nitem", "")).strip()
        if _nitem_looks_like_description(nitem):
            ok, msg = _haiku_nitem_check(func.__name__, nitem)
            if not ok:
                return json.dumps({
                    "validation_error": (
                        f"nitem appears to be an item description, not a numeric code. "
                        f"Use get_top_items or get_item_data to look up the nitem code first. "
                        f"Haiku check: {msg}"
                    )
                })

        return func(**kwargs)

    return _lc_tool(wrapper)
