"""
Thin Supabase persistence layer.

Reads SUPABASE_URL and SUPABASE_KEY from os.environ so it works in both
dashboard_app.py (Streamlit Cloud secrets injected at startup) and
predict.py (CLI or called from within the dashboard process).

Table layout — single table model_config:
    key         TEXT PRIMARY KEY
    value       JSONB

Bets:
    key = 'bets'       → value = [list of bet dicts]

Injury report:
    key = 'injury_report'  → value = {"fetched_at": ISO-8601, "data": [...]}
"""

import os
import logging
from datetime import datetime, timezone

log = logging.getLogger("supabase_store")

_TABLE = "model_config"
INJURY_TTL_HOURS = 2.0


def _client():
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception as e:
        log.debug(f"Supabase client init failed: {e}")
        return None


def save(key: str, value) -> bool:
    sb = _client()
    if not sb:
        return False
    try:
        sb.table(_TABLE).upsert({"key": key, "value": value}).execute()
        log.debug(f"Supabase saved: {key}")
        return True
    except Exception as e:
        log.warning(f"Supabase save({key}) failed: {e}")
        return False


def load(key: str, default=None):
    sb = _client()
    if not sb:
        return default
    try:
        res = sb.table(_TABLE).select("value").eq("key", key).execute()
        if res.data:
            return res.data[0]["value"]
    except Exception as e:
        log.warning(f"Supabase load({key}) failed: {e}")
    return default


# ── Injury report helpers ─────────────────────────────────────────────────────

def save_injury_report(df) -> bool:
    """Persist an injury report DataFrame to Supabase with a fetch timestamp."""
    try:
        payload = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "data": df.to_dict(orient="records"),
        }
        return save("injury_report", payload)
    except Exception as e:
        log.warning(f"Supabase save_injury_report failed: {e}")
        return False


def load_injury_report(ttl_hours: float = INJURY_TTL_HOURS):
    """
    Load injury report from Supabase.

    Returns (DataFrame, is_fresh: bool).
    is_fresh=True  → data is within TTL, use it directly
    is_fresh=False → data is stale or absent, caller should re-fetch from ESPN
    """
    import pandas as pd

    cached = load("injury_report")
    if not cached:
        return None, False

    try:
        fetched_at = datetime.fromisoformat(cached["fetched_at"])
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        age_h = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600
        df = pd.DataFrame(cached.get("data", []))
        is_fresh = age_h < ttl_hours
        return df, is_fresh
    except Exception as e:
        log.warning(f"Supabase load_injury_report parse failed: {e}")
        return None, False
