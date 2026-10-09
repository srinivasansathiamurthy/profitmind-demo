IDENTIFIERS = ["category", "store", "nitem", "com_code"]
TIME_COLS = ["week", "week_start", "week_end"]
QUANTITY_COLS = ["move", "qty"]
PRICE_COLS = ["price", "unit_price"]
REVENUE_COLS = ["gross_revenue", "net_revenue"]
COST_COLS = ["margin_percentage", "total_calculated_cost", "cost_percentage"]

DERIVED_CONCEPTS = {
    "log_price": "LN(price)",
    "log_units": "LN(try_divide(move, qty))",
    "revenue_share": "gross_revenue / SUM(gross_revenue) OVER (PARTITION BY category)",
    "price_cv": "STDDEV(unit_price) / AVG(unit_price)",
    "elasticity": "COVAR_SAMP(log_price, log_units) / VAR_SAMP(log_price)",
    "corr_price_qty": "try_divide(COVAR_SAMP(log_price, log_units), STDDEV_SAMP(log_price) * STDDEV_SAMP(log_units))",
}

COLUMN_DESCRIPTIONS = {
    "category": "product category (beer, cheese, etc.)",
    "store": "store number",
    "week": "Dominick's week number",
    "week_start": "Thursday start of week",
    "week_end": "Wednesday end of week",
    "move": "units sold in store-week",
    "qty": "bundle size — price applies to qty units",
    "price": "shelf price for qty units (bundle price)",
    "unit_price": "price per single unit = price / qty",
    "gross_revenue": "price * move / qty (total revenue)",
    "net_revenue": "gross_revenue * margin_pct / 100 (gross profit)",
    "margin_percentage": "gross margin percent",
    "total_calculated_cost": "gross_revenue - net_revenue",
    "cost_percentage": "100 - margin_percentage",
    "com_code": "commodity sub-code within category",
    "descrip": "item description text",
    "size": "package size label",
    "case": "case-pack configuration",
    "nitem": "numeric item code (UPC stub) — primary item identifier",
}

# Human-readable display names (used in system prompt)
CATEGORIES = [
    "Analgesics", "Bath Soap", "Bathroom Tissues", "Beer", "Bottled Juices",
    "Canned Soup", "Canned Tuna", "Cereals", "Cheeses", "Cigarettes",
    "Cookies", "Crackers", "Dish Detergent", "Fabric Softeners",
    "Front-end-candies", "Frozen Dinners", "Frozen Entrees", "Frozen Juices",
    "Grooming Products", "Laundry Detergents", "Oatmeal", "Paper Towels",
    "Shampoos", "Snack Crackers", "Soaps", "Soft Drinks", "Toothbrushes",
    "Toothpastes",
]

# Actual values stored in dominicks.gold.movement — lowercase snake_case
DB_CATEGORIES = [c.lower().replace(" ", "_").replace("-", "_") for c in CATEGORIES]

# Display name → DB value, e.g. "Soft Drinks" → "soft_drinks"
DISPLAY_TO_DB: dict[str, str] = dict(zip(CATEGORIES, DB_CATEGORIES))

MIN_WEEKS_FOR_ELASTICITY = 52
MIN_CORR_FOR_CLASSIFICATION = 0.3
