import os
from pathlib import Path
from dotenv import load_dotenv
import databricks.sql

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

# Map python-style .env key to standard env var
if not os.getenv("ANTHROPIC_API_KEY"):
    val = os.getenv("anthropic_key", "")
    if val:
        os.environ["ANTHROPIC_API_KEY"] = val

ANTHROPIC_API_KEY_ENV = os.environ.get("ANTHROPIC_API_KEY", "")
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "8259557357780062.2.gcp.databricks.com")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH", "")
CATALOG = "dominicks"
GOLD_TABLE = "dominicks.gold.movement"


def get_connection():
    if not DATABRICKS_TOKEN:
        raise RuntimeError("DATABRICKS_TOKEN not set in .env")
    if not DATABRICKS_HTTP_PATH:
        raise RuntimeError("DATABRICKS_HTTP_PATH not set in .env")
    return databricks.sql.connect(
        server_hostname=DATABRICKS_HOST,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN,
    )


def run_query(sql: str) -> "pd.DataFrame":
    import pandas as pd
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description]
    return pd.DataFrame(rows, columns=cols)
