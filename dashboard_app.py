"""
WNBA Model Dashboard
Run: python -m streamlit run dashboard_app.py
"""
import os, sys, json
from pathlib import Path
from datetime import date, datetime

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Inject Streamlit secrets into os.environ so backend modules that read
# os.environ at import time (odds_scraper, supabase_store, etc.) pick them up.
for _secret_key in ["ODDS_API_KEY", "SUPABASE_URL", "SUPABASE_KEY"]:
    if _secret_key not in os.environ:
        _val = st.secrets.get(_secret_key, "")
        if _val:
            os.environ[_secret_key] = _val

st.set_page_config(
    page_title="WNBA Model Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
(ROOT / "logs").mkdir(exist_ok=True)

# ── Authentication ────────────────────────────────────────────────────────────
def _check_password():
    pwd = st.secrets.get("password", "")
    if not pwd:
        st.error("Dashboard password not configured. Set `password` in Streamlit secrets.")
        st.stop()
    def _submit():
        st.session_state["_auth"] = (st.session_state.get("_pw") == pwd)
    if not st.session_state.get("_auth"):
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.markdown("## 🏀 WNBA Model")
            st.text_input("Password", type="password", key="_pw", on_change=_submit)
            if st.session_state.get("_auth") is False:
                st.error("Incorrect password")
        st.stop()

_check_password()

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Tighten top padding */
.block-container { padding-top: 1.2rem !important; }

/* Game cards */
.game-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px 20px 13px;
    margin-bottom: 6px;
}

/* Flagged bet cards */
.bet-card {
    background: linear-gradient(135deg, rgba(0,200,150,0.08) 0%, rgba(0,200,150,0.02) 100%);
    border: 1px solid rgba(0,200,150,0.28);
    border-left: 4px solid #00c896;
    border-radius: 12px;
    padding: 18px 22px 15px;
    margin-bottom: 6px;
}

/* Subtle divider replacement */
hr { border-color: rgba(255,255,255,0.06) !important; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def american_to_decimal(ml):
    try:
        ml = float(ml)
        return ml / 100 + 1 if ml > 0 else 100 / abs(ml) + 1
    except (ValueError, TypeError):
        return 1.909

def fmt_ml(ml):
    try:
        ml = int(float(ml))
        return f"+{ml}" if ml > 0 else str(ml)
    except (ValueError, TypeError):
        return str(ml)

def fmt_myt(commence_str: str) -> str:
    """Convert a UTC ISO-8601 commence_time string to Malaysia Time (UTC+8)."""
    from datetime import timezone, timedelta
    MYT = timezone(timedelta(hours=8))
    try:
        dt = datetime.fromisoformat(commence_str.replace("Z", "+00:00"))
        return dt.astimezone(MYT).strftime("%I:%M %p MYT")
    except Exception:
        return ""

# ── Storage ───────────────────────────────────────────────────────────────────
TRACKER_FILE = ROOT / "logs" / "bet_tracker.json"

from src.supabase_store import (
    save as _sb_save, load as _sb_load,
    save_injury_report as _sb_save_injuries,
    load_injury_report as _sb_load_injuries,
)

def load_bets():
    # Try Supabase first (persists across Streamlit Cloud redeploys)
    bets = _sb_load("bets")
    if bets is not None:
        return bets
    # JSON fallback (local dev or if Supabase not yet configured)
    if TRACKER_FILE.exists():
        try:
            with open(TRACKER_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            import shutil
            shutil.copy(TRACKER_FILE, str(TRACKER_FILE) + ".bak")
    return []

def save_bets(bets):
    # Persist to Supabase when configured
    _sb_save("bets", bets)
    # Always write JSON backup (atomic write to avoid partial files)
    tmp = str(TRACKER_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(bets, f, indent=2, default=str)
    os.replace(tmp, TRACKER_FILE)

def find_bet(bets, date, home, away, bet_team):
    for i, b in enumerate(bets):
        if (b.get("date") == date and b.get("home_team") == home
                and b.get("away_team") == away and b.get("bet_team") == bet_team):
            return i
    return -1

# ── Backtest data ─────────────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"

CLOSING_BOOKS = {
    "pinnacle":      "Pinnacle",
    "draftkings":    "DraftKings",
    "fanduel":       "FanDuel",
    "betfair_ex_eu": "Betfair EU",
    "unibet_uk":     "Unibet UK",
    "betsson":       "Betsson",
    "nordicbet":     "NordicBet",
}

BACKTEST_FILES = {
    "2022": LOG_DIR / "backtest_real_2022.csv",
    "2023": LOG_DIR / "backtest_real_2023.csv",
    "2024": LOG_DIR / "backtest_real_2024.csv",
    "2025": LOG_DIR / "backtest_real_2025.csv",
}
CLEAN_SEASONS = ["2024", "2025"]
VALID_SEASONS = ["2023"]

@st.cache_data(ttl=3600)
def load_all_backtest():
    dfs = []
    for season, path in BACKTEST_FILES.items():
        if path.exists():
            df = pd.read_csv(path)
            df["season"] = season
            if "moneyline_home" in df.columns and "home_odds" not in df.columns:
                df = df.rename(columns={"moneyline_home": "home_odds",
                                        "moneyline_away": "away_odds"})
            if "home_won" not in df.columns and "home_actually_won" in df.columns:
                df["home_won"] = df["home_actually_won"]
            if "home_odds" in df.columns:
                df["home_odds"] = pd.to_numeric(df["home_odds"], errors="coerce")
                df["away_odds"] = pd.to_numeric(df["away_odds"], errors="coerce")
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    for col in ["model_prob_home", "home_odds", "away_odds",
                "home_won", "result_correct", "pnl_per_unit", "bet_odds"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    return combined

@st.cache_data(ttl=3600)
def load_closing_book(book_key: str) -> pd.DataFrame:
    dfs = []
    for season in ["2022", "2023", "2024", "2025"]:
        path = LOG_DIR / f"backtest_{book_key}_{season}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["season"] = season
        df = df.rename(columns={
            "p_home_win": "model_prob_home",
            "home_ml":    "home_odds",
            "away_ml":    "away_odds",
        })
        for col in ["model_prob_home", "home_odds", "away_odds", "home_won"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

_REQUIRED_PRED_COLS = ["home_team", "away_team", "p_home_win", "recommendation"]

@st.cache_data(ttl=300)
def load_predictions_for_date(target_date: str) -> pd.DataFrame:
    f = LOG_DIR / f"predictions_{target_date}.csv"
    if not f.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(f)
    except Exception:
        return pd.DataFrame()
    missing = [c for c in _REQUIRED_PRED_COLS if c not in df.columns]
    if missing:
        return pd.DataFrame()
    return df

@st.cache_data(ttl=300)
def load_raw_odds_for_date(target_date: str) -> list:
    try:
        from src.odds_scraper import get_todays_odds
        return get_todays_odds(target_date=target_date)
    except Exception:
        return []

@st.cache_data(ttl=1800)
def load_live_injury_report() -> pd.DataFrame:
    _empty = pd.DataFrame(columns=["team", "player", "status", "reason", "TEAM_ABBREVIATION"])
    # Check Supabase cache first (survives redeploys, shared across sessions)
    try:
        cached_df, is_fresh = _sb_load_injuries()
        if is_fresh and cached_df is not None and not cached_df.empty:
            if "TEAM_ABBREVIATION" not in cached_df.columns and "team" in cached_df.columns:
                from src.scraper import WNBA_TEAMS
                cached_df["TEAM_ABBREVIATION"] = cached_df["team"].map(WNBA_TEAMS)
            return cached_df
    except Exception:
        pass
    # Supabase stale or unavailable — fetch fresh from ESPN and save
    try:
        from src.scraper import fetch_injury_report
        df = fetch_injury_report()
        if not df.empty:
            _sb_save_injuries(df)
        return df if not df.empty else _empty
    except Exception:
        return _empty

# ── Filter engine ─────────────────────────────────────────────────────────────
def implied_prob(ml):
    ml = float(ml)
    return 100/(ml+100) if ml > 0 else abs(ml)/(abs(ml)+100)

def remove_vig(h, a):
    t = h+a; return h/t, a/t

def apply_filters(df, min_edge, max_edge, min_odds, max_odds, underdogs_only, seasons, away_only=True):
    if df.empty: return pd.DataFrame()
    rows = df[df["season"].isin(seasons)].copy() if seasons else df.copy()
    if rows.empty: return pd.DataFrame()

    if "home_odds" not in rows.columns and "moneyline_home" in rows.columns:
        rows = rows.rename(columns={"moneyline_home": "home_odds",
                                    "moneyline_away": "away_odds"})
    if "home_won" not in rows.columns and "home_actually_won" in rows.columns:
        rows = rows.rename(columns={"home_actually_won": "home_won"})

    results = []
    for _, row in rows.iterrows():
        try:
            p_home = float(row.get("model_prob_home", 0))
            ho     = float(row.get("home_odds", 0))
            ao     = float(row.get("away_odds", 0))
            hw     = float(row.get("home_won", np.nan))
        except: continue
        if pd.isna(p_home) or pd.isna(ho) or pd.isna(ao) or pd.isna(hw): continue

        rh = implied_prob(ho); ra = implied_prob(ao)
        fh, fa = remove_vig(rh, ra)
        eh = (p_home - fh) * 100
        ea = (1-p_home - fa) * 100

        if eh >= ea and eh >= min_edge:
            bet_odds, bet_edge, is_home = ho, eh, True
        elif ea >= min_edge:
            bet_odds, bet_edge, is_home = ao, ea, False
        else: continue

        if underdogs_only and bet_odds < 0:         continue
        if bet_odds > max_odds:                      continue
        if abs(min_odds) > 0 and bet_odds > 0 and bet_odds <= abs(min_odds): continue
        if bet_edge > max_edge:                      continue
        if away_only and is_home:                    continue

        won = (is_home == (hw == 1))
        dec = american_to_decimal(bet_odds)
        pnl = (dec-1) if won else -1.0
        results.append({"season": row["season"], "is_home": is_home,
                         "bet_odds": bet_odds, "bet_edge": round(bet_edge,2),
                         "won": won, "pnl": pnl})

    return pd.DataFrame(results)

def annotate_all_games(df, min_edge, max_edge, min_odds, max_odds, underdogs_only, seasons, away_only=True):
    """Return all games in selected seasons with edge/bet columns appended."""
    if df.empty:
        return pd.DataFrame()
    rows = df[df["season"].isin(seasons)].copy() if seasons else df.copy()
    if rows.empty:
        return pd.DataFrame()

    if "home_odds" not in rows.columns and "moneyline_home" in rows.columns:
        rows = rows.rename(columns={"moneyline_home": "home_odds", "moneyline_away": "away_odds"})
    if "home_won" not in rows.columns and "home_actually_won" in rows.columns:
        rows = rows.rename(columns={"home_actually_won": "home_won"})

    records = []
    for _, row in rows.iterrows():
        try:
            p_home = float(row.get("model_prob_home", 0))
            ho     = float(row.get("home_odds", 0))
            ao     = float(row.get("away_odds", 0))
            hw     = float(row.get("home_won", np.nan))
        except:
            continue
        if pd.isna(p_home) or pd.isna(ho) or pd.isna(ao) or pd.isna(hw):
            continue

        rh = implied_prob(ho); ra = implied_prob(ao)
        fh, fa = remove_vig(rh, ra)
        eh = (p_home - fh) * 100
        ea = (1 - p_home - fa) * 100

        # determine which side to bet (model's preferred side)
        if eh >= ea:
            pref_side, pref_edge, pref_odds, is_home = "HOME", eh, ho, True
        else:
            pref_side, pref_edge, pref_odds, is_home = "AWAY", ea, ao, False

        qualifies = (
            pref_edge >= min_edge
            and pref_edge <= max_edge
            and not (underdogs_only and pref_odds < 0)
            and pref_odds <= max_odds
            and not (abs(min_odds) > 0 and pref_odds > 0 and pref_odds <= abs(min_odds))
            and not (away_only and is_home)
        )

        won = pnl = None
        if qualifies:
            won = bool(is_home == (hw == 1))
            dec = american_to_decimal(pref_odds)
            pnl = round((dec - 1) if won else -1.0, 4)

        records.append({
            "game_date":        row.get("game_date", ""),
            "home_team":        row.get("home_team", ""),
            "away_team":        row.get("away_team", ""),
            "season":           row["season"],
            "model_prob_home":  round(p_home, 4),
            "home_odds":        ho,
            "away_odds":        ao,
            "home_edge_pct":    round(eh, 2),
            "away_edge_pct":    round(ea, 2),
            "pref_side":        pref_side,
            "pref_edge_pct":    round(pref_edge, 2),
            "home_won":         int(hw),
            "qualifies":        qualifies,
            "bet_odds":         pref_odds if qualifies else None,
            "won":              won,
            "pnl":              pnl,
        })

    return pd.DataFrame(records)

def summarise(df):
    if df.empty: return {"bets":0,"roi":0,"wr":0,"pnl":0,"be":0}
    wins = df["won"].sum(); total = len(df); pnl = df["pnl"].sum()
    avg_dec = df["bet_odds"].apply(american_to_decimal).mean()
    return {"bets":total,"roi":round(pnl/total*100,2),
            "wr":round(wins/total*100,1),"pnl":round(pnl,2),
            "be":round(1/avg_dec*100,1)}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🏀 WNBA Model")
st.sidebar.caption(f"{date.today().strftime('%b %d, %Y')}  |  {date.today().year} Season")
st.sidebar.caption("v2.0 · May 2026")

# Supabase connection indicator
try:
    from src.supabase_store import _client as _sb_client
    _sb_ok = _sb_client() is not None
except Exception:
    _sb_ok = False
st.sidebar.caption("🟢 Supabase connected" if _sb_ok else "🔴 Supabase offline (local JSON)")

page = st.sidebar.radio("Navigation", [
    "🏀 Today's Predictions",
    "🔬 Filter Playground",
    "📋 Bet Tracker",
    "📈 Performance",
    "ℹ️ About",
], label_visibility="collapsed")

# Season-to-date sidebar stats
try:
    _sb = load_bets()
    if _sb:
        _sdf = pd.DataFrame(_sb)
        _dec = _sdf[_sdf["result"].isin(["win","loss"])]
        if len(_dec):
            _w = (_dec["result"]=="win").sum(); _t = len(_dec)
            _pnl = pd.to_numeric(_dec["pnl"], errors="coerce").sum()
            st.sidebar.divider()
            st.sidebar.caption("📊 Season-to-date")
            c1, c2 = st.sidebar.columns(2)
            c1.metric("Record", f"{_w}W-{_t-_w}L")
            c2.metric("ROI", f"{_pnl/_t*100:+.1f}%")
            st.sidebar.metric("P&L", f"{_pnl:+.2f}u", delta=f"{(len(_sdf)-_t)} pending")
except Exception:
    pass

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: TODAY'S PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏀 Today's Predictions":
    from datetime import timedelta, timezone as _tz

    # WNBA games tip off 7 PM+ ET = next calendar day in MYT (UTC+8).
    # date_opts holds the US/schedule dates used for prediction files.
    # day_names shows the MYT date the user will actually watch the game.
    _MYT      = _tz(timedelta(hours=8))
    today_myt = datetime.now(_MYT).date()
    today_us  = today_myt - timedelta(days=1)   # US game date ≡ MYT date − 1

    date_opts = [(today_us  + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(8)]
    myt_dates = [(today_myt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(8)]
    day_names = (["Today", "Tomorrow"]
                 + [(today_myt + timedelta(days=i)).strftime("%a %b %d") for i in range(2, 8)])

    # ── Header: title + date selector + refresh button in one row ────────────
    st.title("🏀 WNBA Predictions")

    _radio_col, _btn_col = st.columns([5, 1])
    with _radio_col:
        selected_date = st.radio(
            "date_range", options=date_opts,
            format_func=lambda d: day_names[date_opts.index(d)],
            horizontal=True, label_visibility="collapsed",
        )
    with _btn_col:
        st.markdown('<div style="margin-top:4px"></div>', unsafe_allow_html=True)
        _refresh_clicked = st.button("🔄 Refresh", use_container_width=True)

    # Display MYT date in caption (US date + 1)
    _myt_display = (date.fromisoformat(selected_date) + timedelta(days=1)).strftime("%A, %B %d, %Y")
    _pred_path   = LOG_DIR / f"predictions_{selected_date}.csv"
    _last_ts     = st.session_state.get(f"refresh_ts_{selected_date}")
    if _last_ts:
        _age_s   = (datetime.now() - _last_ts).total_seconds()
        _age_str = f"{int(_age_s/60)}m ago" if _age_s < 3600 else f"{int(_age_s/3600)}h ago"
        st.caption(f"{_myt_display} MYT  ·  🕐 Updated {_age_str}")
    else:
        st.caption(_myt_display + " MYT" + ("  ·  🕐 From cache" if _pred_path.exists() else ""))

    # Auto-refresh once per session per date, but only if the CSV is stale/missing
    _session_key = f"predicted_{selected_date}"
    if _session_key not in st.session_state:
        st.session_state[_session_key] = False
    _csv_age_s = (
        (datetime.now() - datetime.fromtimestamp(_pred_path.stat().st_mtime)).total_seconds()
        if _pred_path.exists() else float("inf")
    )
    if not st.session_state[_session_key] and _csv_age_s > 900:
        with st.spinner("Fetching latest predictions and odds..."):
            try:
                from src.predict import predict_today, get_current_season
                predict_today(target_date=selected_date,
                              season=get_current_season(selected_date))
                load_predictions_for_date.clear()
            except Exception:
                pass
        st.session_state[_session_key] = True
        st.session_state[f"refresh_ts_{selected_date}"] = datetime.now()
    elif not st.session_state[_session_key]:
        st.session_state[_session_key] = True

    if _refresh_clicked:
        st.session_state[_session_key] = False
        try:
            from src.predict import predict_today, get_current_season
            result = predict_today(target_date=selected_date,
                                   season=get_current_season(selected_date))
            load_predictions_for_date.clear()
            st.session_state[f"refresh_ts_{selected_date}"] = datetime.now()
            st.success(f"Updated — {len(result)} games." if result is not None and not result.empty
                       else "No games found.")
        except Exception as e:
            st.error(str(e))

    preds      = load_predictions_for_date(selected_date)
    raw_odds   = load_raw_odds_for_date(selected_date)
    inj_report = load_live_injury_report()
    has_inj    = not inj_report.empty and "TEAM_ABBREVIATION" in inj_report.columns

    def _team_injuries(team):
        if not has_inj:
            return []
        rows = inj_report[inj_report["TEAM_ABBREVIATION"] == team]
        return [(r["player"], r.get("status", "")) for _, r in rows.iterrows()
                if pd.notna(r["player"])]

    def _pinnacle(home, away):
        for g in raw_odds:
            if (g.get("home_team","").upper() == home.upper() and
                    g.get("away_team","").upper() == away.upper()):
                return g.get("bookmakers", {}).get("pinnacle") or {}
        return {}

    def _game_time_myt(home, away) -> str:
        for g in raw_odds:
            if (g.get("home_team","").upper() == home.upper() and
                    g.get("away_team","").upper() == away.upper()):
                return fmt_myt(g.get("commence_time",""))
        return ""

    if preds.empty:
        if raw_odds:
            st.info("No model predictions yet — showing live odds. Click **Refresh / Generate** to run the model.")
            st.subheader("📊 Odds Board")
            for g in raw_odds:
                ht = g.get("home_team","?"); at = g.get("away_team","?")
                pinn = g.get("bookmakers",{}).get("pinnacle",{})
                commence = g.get("commence_time","")
                c1, c2 = st.columns([3, 3])
                with c1:
                    st.markdown(f"**{ht}** vs {at}")
                    if commence:
                        myt = fmt_myt(commence)
                        st.caption(myt if myt else commence[:16])
                    for tm, players in [(ht, _team_injuries(ht)), (at, _team_injuries(at))]:
                        if players:
                            names = ", ".join(f"{p} ({s})" for p, s in players)
                            st.markdown(f"<span style='color:#f5a623;font-size:12px'>⚠️ {tm}: {names}</span>",
                                        unsafe_allow_html=True)
                with c2:
                    if pinn:
                        st.markdown(f"**Pinnacle:** {ht} {fmt_ml(pinn.get('home',0))} / {at} {fmt_ml(pinn.get('away',0))}")
                    else:
                        st.caption("No Pinnacle line")
                st.divider()
        else:
            st.info("No predictions yet. Click **Refresh / Generate** or run: `python predict.py`")
            if selected_date == today_us.strftime("%Y-%m-%d"):
                with st.expander("📋 Enter odds manually"):
                    st.code('python predict.py --odds "LAS:-150,IND:+130;NYL:-200,CON:+170"')
    else:
        # Warn if model failed — all predictions stuck at 50%
        if "p_home_win" in preds.columns:
            _probs = preds["p_home_win"].dropna()
            if len(_probs) > 0 and (_probs == 0.5).all():
                st.warning("⚠️ Model failed to load — all predictions defaulted to 50%. Check that model files are present and sklearn version matches.", icon="⚠️")

        try:
            from config.settings import MIN_EDGE_PCT, BET_MAX_ODDS, BET_MIN_ODDS
            live_min_edge = st.session_state.get("saved_min_edge", int(MIN_EDGE_PCT))
            live_min_odds = st.session_state.get("saved_min_odds", int(abs(BET_MIN_ODDS))) + 1
            live_max_odds = st.session_state.get("saved_max_odds", int(BET_MAX_ODDS))
        except (ImportError, Exception):
            live_min_edge, live_min_odds, live_max_odds = 15, 141, 500

        if "has_edge" in preds.columns:
            flagged = preds[preds["has_edge"] == True]
        else:
            flagged = preds[preds["recommendation"].astype(str).str.contains("BET", na=False) &
                           ~preds["recommendation"].astype(str).str.contains("NO BET", na=False)]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Games", len(preds))
        m2.metric("Flagged Bets", len(flagged), delta="⭐ BET" if len(flagged) > 0 else None)
        m3.metric("Min Edge", f"{live_min_edge}%")
        m4.metric("Odds Range", f"+{live_min_odds}–+{live_max_odds}")
        st.divider()

        # ── Flagged bets ──────────────────────────────────────────────────────
        if len(flagged) > 0:
            st.subheader("⭐ Flagged Bets")
            _all_bets = load_bets()
            for _, row in flagged.iterrows():
                rec      = str(row.get("recommendation",""))
                bet_team = rec.split("BET ")[1].split(" ")[0] if "BET " in rec else "?"
                odds_str = rec.split("(")[1].split(")")[0] if "(" in rec else "?"
                edge_str = rec.split("Edge: ")[1].split("%")[0] if "Edge:" in rec else "?"
                ev_str   = rec.split("EV: ")[1] if "EV:" in rec else "—"
                kelly    = float(row.get("kelly_units", 0) or 0)
                home     = row.get("home_team","?")
                away     = row.get("away_team","?")
                home_imp = float(row.get("home_injury_impact", 0) or 0)
                away_imp = float(row.get("away_injury_impact", 0) or 0)

                existing_idx   = find_bet(_all_bets, selected_date, home, away, bet_team)
                already_logged = existing_idx >= 0

                _gt_bet  = _game_time_myt(home, away)
                _elo_bet = float(row.get("elo_diff", 0) or 0)
                _sub     = " · ".join(filter(None, [_gt_bet, f"{home} vs {away}"]))
                st.markdown(f"""<div class="bet-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start">
    <div>
      <div style="color:#777;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.6px;margin-bottom:6px">{_sub}</div>
      <div style="font-size:22px;font-weight:800;color:#00c896">🎯 BET {bet_team}</div>
      <div style="font-size:20px;font-weight:700;color:#eee;margin-top:3px">{odds_str}</div>
    </div>
    <div style="text-align:right">
      <div style="font-size:34px;font-weight:900;color:#00c896;line-height:1">{edge_str}%</div>
      <div style="font-size:10px;font-weight:700;color:#00c896;letter-spacing:1px;text-transform:uppercase">EDGE</div>
    </div>
  </div>
  <div style="display:flex;gap:24px;margin-top:14px;padding-top:12px;border-top:1px solid rgba(0,200,150,0.15)">
    <div><div style="color:#666;font-size:10px;text-transform:uppercase;letter-spacing:.5px">EV</div><div style="font-weight:700;font-size:15px;color:#eee">{ev_str}</div></div>
    <div><div style="color:#666;font-size:10px;text-transform:uppercase;letter-spacing:.5px">Kelly</div><div style="font-weight:700;font-size:15px;color:#eee">{kelly:.1f}%</div></div>
    <div><div style="color:#666;font-size:10px;text-transform:uppercase;letter-spacing:.5px">Elo diff</div><div style="font-weight:700;font-size:15px;color:#eee">{_elo_bet:+.0f}</div></div>
  </div>
</div>""", unsafe_allow_html=True)

                with st.container():
                    lc1, lc2, lc3 = st.columns([2, 2, 1])
                    with lc1:
                        default_units = float(_all_bets[existing_idx].get("units", 1.0)) if already_logged else 1.0
                        log_units = st.number_input(
                            "Units", min_value=0.1, max_value=100.0,
                            value=default_units, step=0.5,
                            key=f"units_{home}_{away}",
                        )
                    with lc2:
                        st.markdown('<div style="margin-top:26px"></div>', unsafe_allow_html=True)
                        if already_logged:
                            if st.button("🔄 Update", key=f"upd_{home}_{away}", use_container_width=True):
                                _all_bets[existing_idx]["units"] = log_units
                                save_bets(_all_bets)
                                st.rerun()
                            st.caption("✅ Already logged")
                        else:
                            if st.button(f"✅ Log {bet_team}", key=f"log_{home}_{away}", type="primary", use_container_width=True):
                                _all_bets.append({
                                    "date": selected_date, "home_team": home, "away_team": away,
                                    "bet_team": bet_team, "bet_odds": odds_str, "edge_pct": edge_str,
                                    "result": "pending", "units": log_units, "pnl": None,
                                })
                                save_bets(_all_bets)
                                st.rerun()
                    with lc3:
                        if already_logged:
                            st.markdown('<div style="margin-top:26px"></div>', unsafe_allow_html=True)
                            if st.button("🗑️", key=f"unlog_{home}_{away}", use_container_width=True):
                                _all_bets.pop(existing_idx)
                                save_bets(_all_bets)
                                st.rerun()
                    st.markdown('<div style="margin-bottom:6px"></div>', unsafe_allow_html=True)

        # ── All games ─────────────────────────────────────────────────────────
        st.subheader("All Games")
        for _, row in preds.iterrows():
            home    = row.get("home_team","?")
            away    = row.get("away_team","?")
            p_home  = float(row.get("p_home_win", row.get("model_prob_home", 0.5)))
            rec     = str(row.get("recommendation",""))
            is_bet  = "BET" in rec and "NO BET" not in rec
            elo     = float(row.get("elo_diff", 0) or 0)
            b2b_h   = " [B2B]" if row.get("home_b2b") else ""
            b2b_a   = " [B2B]" if row.get("away_b2b") else ""
            edge_h  = row.get("edge_home_pct")
            edge_a  = row.get("edge_away_pct")
            home_imp = float(row.get("home_injury_impact", 0) or 0)
            away_imp = float(row.get("away_injury_impact", 0) or 0)

            pinn   = _pinnacle(home, away)
            pinn_h = pinn.get("home") if pinn else row.get("home_ml")
            pinn_a = pinn.get("away") if pinn else row.get("away_ml")

            _gt      = _game_time_myt(home, away)
            _h_b2b_s = '<span style="background:rgba(245,166,35,.15);color:#f5a623;font-size:10px;font-weight:700;padding:1px 6px;border-radius:4px;margin-left:5px;vertical-align:middle">B2B</span>' if row.get("home_b2b") else ""
            _a_b2b_s = '<span style="background:rgba(245,166,35,.15);color:#f5a623;font-size:10px;font-weight:700;padding:1px 6px;border-radius:4px;margin-left:5px;vertical-align:middle">B2B</span>' if row.get("away_b2b") else ""
            _time_s  = f'<span style="color:#666;font-size:12px">{_gt}</span>' if _gt else ""
            _border  = "border-left:3px solid #00c896;" if is_bet else ""

            if is_bet:
                _bet_part = rec.split("BET")[1].split("(")[0].strip() if "BET" in rec else ""
                _rec_s = f'<span style="background:rgba(0,200,150,.15);color:#00c896;font-size:11px;font-weight:700;padding:2px 10px;border-radius:20px;margin-left:8px">⭐ BET {_bet_part}</span>'
            else:
                _rec_s = ""

            # Odds row
            _odds_row = ""
            try:
                if pinn_h is not None and pinn_a is not None:
                    hs = fmt_ml(pinn_h); as_ = fmt_ml(pinn_a)
                    if edge_h is not None and str(edge_h) != "nan":
                        eh = float(edge_h); ea = float(edge_a)
                        hc = "#00c896" if eh > 3 else ("#ff4b4b" if eh < -1 else "#888")
                        ac = "#00c896" if ea > 3 else ("#ff4b4b" if ea < -1 else "#888")
                        _odds_row = (
                            f'<div style="margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.05);'
                            f'display:flex;align-items:center;gap:16px;font-size:12px">'
                            f'<span style="color:#444;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.5px">Pinnacle</span>'
                            f'<span><b style="color:#ddd">{home}</b> <b>{hs}</b> <span style="color:{hc}">({eh:+.1f}%)</span></span>'
                            f'<span><b style="color:#ddd">{away}</b> <b>{as_}</b> <span style="color:{ac}">({ea:+.1f}%)</span></span>'
                            f'</div>'
                        )
                    else:
                        _odds_row = (
                            f'<div style="margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.05);'
                            f'display:flex;align-items:center;gap:16px;font-size:12px">'
                            f'<span style="color:#444;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.5px">Pinnacle</span>'
                            f'<span><b style="color:#ddd">{home}</b> <b>{hs}</b></span>'
                            f'<span><b style="color:#ddd">{away}</b> <b>{as_}</b></span>'
                            f'</div>'
                        )
            except Exception:
                pass

            st.markdown(f"""<div class="game-card" style="{_border}">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
    <div style="font-size:16px;font-weight:700">
      {home}{_h_b2b_s}
      <span style="color:#444;margin:0 8px;font-weight:400">vs</span>
      {away}{_a_b2b_s}
    </div>
    <div style="display:flex;align-items:center;gap:8px">{_time_s}{_rec_s}</div>
  </div>
  <div style="background:rgba(255,255,255,0.07);border-radius:4px;height:6px;overflow:hidden">
    <div style="width:{p_home*100:.1f}%;height:100%;border-radius:4px;background:linear-gradient(90deg,#ff6b35,#f5a623)"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:12px;margin-top:5px">
    <span style="font-weight:600;color:#ddd">{home} {p_home*100:.0f}%</span>
    <span style="color:#444">Elo {elo:+.0f}</span>
    <span style="font-weight:600;color:#ddd">{(1-p_home)*100:.0f}% {away}</span>
  </div>
  {_odds_row}
</div>""", unsafe_allow_html=True)

            # ── Model breakdown ───────────────────────────────────────────────
            with st.expander("📊 Breakdown"):
                h_streak = int(row.get("home_streak", 0) or 0)
                a_streak = int(row.get("away_streak", 0) or 0)
                _h_str   = f"W{h_streak}" if h_streak > 0 else (f"L{abs(h_streak)}" if h_streak < 0 else "—")
                _a_str   = f"W{a_streak}" if a_streak > 0 else (f"L{abs(a_streak)}" if a_streak < 0 else "—")
                _s_color = lambda s: "#00c896" if s.startswith("W") else ("#ff4b4b" if s.startswith("L") else "#888")

                def _inj_str(val):
                    s = str(val) if val is not None else ""
                    return "" if s.lower() in ("none", "nan", "") else s
                _h_inj = _inj_str(row.get("home_injuries"))
                _a_inj = _inj_str(row.get("away_injuries"))

                def _inj_pills(inj_str):
                    if not inj_str:
                        return '<span style="color:#555;font-size:12px">None</span>'
                    pills = []
                    for p in inj_str.split(", "):
                        color = "#ff4b4b" if "Out" in p else "#f5a623"
                        pills.append(f'<span style="background:rgba(255,255,255,0.07);color:{color};font-size:11px;padding:2px 8px;border-radius:10px;margin:2px 2px 2px 0;display:inline-block">{p}</span>')
                    return "".join(pills)

                _h_b2b_warn = '<span style="color:#f5a623;font-size:11px"> ⚠️ B2B</span>' if row.get("home_b2b") else ""
                _a_b2b_warn = '<span style="color:#f5a623;font-size:11px"> ⚠️ B2B</span>' if row.get("away_b2b") else ""

                st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:4px 0">
  <div>
    <div style="color:#555;font-size:10px;text-transform:uppercase;letter-spacing:.6px;margin-bottom:8px">Home — {home}{_h_b2b_warn}</div>
    <div style="display:flex;gap:20px;margin-bottom:10px">
      <div><div style="color:#666;font-size:10px;text-transform:uppercase">Streak</div>
           <div style="font-weight:700;color:{_s_color(_h_str)};font-size:15px">{_h_str}</div></div>
      <div><div style="color:#666;font-size:10px;text-transform:uppercase">Elo</div>
           <div style="font-weight:700;font-size:15px">{elo:+.0f}</div></div>
      <div><div style="color:#666;font-size:10px;text-transform:uppercase">Inj. impact</div>
           <div style="font-weight:700;font-size:15px;color:{"#ff4b4b" if home_imp > 0.1 else "#eee"}">{home_imp:.0%}</div></div>
    </div>
    <div style="color:#666;font-size:10px;text-transform:uppercase;margin-bottom:4px">Injuries</div>
    <div>{_inj_pills(_h_inj)}</div>
  </div>
  <div>
    <div style="color:#555;font-size:10px;text-transform:uppercase;letter-spacing:.6px;margin-bottom:8px">Away — {away}{_a_b2b_warn}</div>
    <div style="display:flex;gap:20px;margin-bottom:10px">
      <div><div style="color:#666;font-size:10px;text-transform:uppercase">Streak</div>
           <div style="font-weight:700;color:{_s_color(_a_str)};font-size:15px">{_a_str}</div></div>
      <div><div style="color:#666;font-size:10px;text-transform:uppercase">Elo</div>
           <div style="font-weight:700;font-size:15px">{-elo:+.0f}</div></div>
      <div><div style="color:#666;font-size:10px;text-transform:uppercase">Inj. impact</div>
           <div style="font-weight:700;font-size:15px;color:{"#ff4b4b" if away_imp > 0.1 else "#eee"}">{away_imp:.0%}</div></div>
    </div>
    <div style="color:#666;font-size:10px;text-transform:uppercase;margin-bottom:4px">Injuries</div>
    <div>{_inj_pills(_a_inj)}</div>
  </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: FILTER PLAYGROUND
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔬 Filter Playground":
    st.title("🔬 Filter Playground")
    st.caption("Adjust filters and instantly see the impact on backtest performance.")

    # ── Data source selector ─────────────────────────────────────────────────
    src_labels = list(CLOSING_BOOKS.values())
    src_keys   = list(CLOSING_BOOKS.keys())
    chosen_book_key = src_keys[src_labels.index(
        st.selectbox("Book", src_labels, index=src_labels.index("Pinnacle"))
    )]

    raw = load_closing_book(chosen_book_key)
    if raw.empty:
        st.warning(f"No {CLOSING_BOOKS[chosen_book_key]} closing data found. Run: `python backtest_closing.py --all --edge 1`")
        st.stop()

    available_seasons = sorted(raw["season"].unique().tolist())

    # Only show validation + clean seasons (hide training seasons)
    SHOW_SEASONS = CLEAN_SEASONS + VALID_SEASONS
    raw = raw[raw["season"].isin(SHOW_SEASONS)]
    available_seasons = sorted(raw["season"].unique().tolist())

    # Load saved settings
    try:
        from config.settings import MIN_EDGE_PCT, BET_MAX_ODDS, BET_MIN_ODDS, BET_MAX_EDGE
        cfg_min_edge = int(MIN_EDGE_PCT); cfg_max_edge = int(BET_MAX_EDGE)
        cfg_min_odds = int(abs(BET_MIN_ODDS)); cfg_max_odds = int(BET_MAX_ODDS)
    except:
        cfg_min_edge, cfg_max_edge = 15, 30
        cfg_min_odds, cfg_max_odds = 140, 500

    saved_min_edge = st.session_state.get("saved_min_edge", cfg_min_edge)
    saved_max_edge = st.session_state.get("saved_max_edge", cfg_max_edge)
    saved_min_odds = st.session_state.get("saved_min_odds", cfg_min_odds)
    saved_max_odds = st.session_state.get("saved_max_odds", cfg_max_odds)

    st.subheader("📐 Filter Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Edge**")
        min_edge = st.slider("Min edge %", 4, 30, saved_min_edge, step=1)
        max_edge = st.slider("Max edge %", 15, 60, saved_max_edge, step=1)
    with col2:
        st.markdown("**Odds range**")
        min_odds_abs = st.slider("Min odds", 100, 200, saved_min_odds, step=5)
        max_odds     = st.slider("Max odds", 200, 1000, saved_max_odds, step=25)
        try:
            from config.settings import BET_UNDERDOGS_ONLY as _default_underdogs
        except:
            _default_underdogs = False
        underdogs_only = st.checkbox("Underdogs only", value=_default_underdogs)
        try:
            from config.settings import BET_AWAY_ONLY as _default_away
        except:
            _default_away = False
        away_only = st.checkbox("Away bets only", value=_default_away,
                                help="WNBA: home underdogs consistently lose. Away only = +38% ROI on clean seasons.")
    with col3:
        st.markdown("**Seasons**")
        if st.button("Select all seasons", key="sel_all"):
            st.session_state["sel_seasons"] = available_seasons
        selected_seasons = st.multiselect(
            "Include seasons", options=available_seasons,
            default=st.session_state.get("sel_seasons", available_seasons),
            key="sel_seasons_widget",
        )
        if not selected_seasons:
            selected_seasons = available_seasons
        show_clean_only = st.checkbox("Highlight clean seasons", value=True)

    min_odds_filter = -min_odds_abs

    filtered = apply_filters(raw, min_edge, max_edge, min_odds_filter,
                             max_odds, underdogs_only, selected_seasons, away_only=away_only)
    try:
        from config.settings import MIN_EDGE_PCT, BET_MAX_ODDS, BET_MIN_ODDS, BET_MAX_EDGE, BET_AWAY_ONLY, BET_UNDERDOGS_ONLY
        current = apply_filters(raw, MIN_EDGE_PCT, BET_MAX_EDGE, BET_MIN_ODDS,
                                BET_MAX_ODDS, BET_UNDERDOGS_ONLY, selected_seasons, away_only=BET_AWAY_ONLY)
    except:
        current = apply_filters(raw, 15, 55, 130, 250, True, selected_seasons, away_only=True)

    st.divider()
    st.subheader("📊 Results")

    new_s  = summarise(filtered)
    curr_s = summarise(current)

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Total bets",    new_s["bets"],
              delta=f"{new_s['bets']-curr_s['bets']:+d} vs current")
    k2.metric("Bets / season", f"{new_s['bets']/max(len(selected_seasons),1):.0f}",
              delta=f"{(new_s['bets']-curr_s['bets'])/max(len(selected_seasons),1):+.0f}")
    k3.metric("Win rate",      f"{new_s['wr']:.1f}%",
              delta=f"{new_s['wr']-curr_s['wr']:+.1f}%")
    k4.metric("ROI",           f"{new_s['roi']:+.1f}%",
              delta=f"{new_s['roi']-curr_s['roi']:+.1f}%")
    k5.metric("Total P&L",     f"{new_s['pnl']:+.1f}u",
              delta=f"{new_s['pnl']-curr_s['pnl']:+.1f}u")

    if new_s["bets"] == 0:
        st.warning("No bets match these filters.")
        st.stop()

    # Per-season breakdown
    st.subheader("Per-Season Breakdown")
    season_rows = []
    for s in selected_seasons:
        s_df = filtered[filtered["season"]==s] if not filtered.empty else pd.DataFrame()
        if s_df.empty:
            season_rows.append({"Season":s,"Bets":0,"Win%":"—","ROI":"—","P&L":"—",
                                 "Status":"no bets"})
            continue
        sm = summarise(s_df)
        status = "✅ clean" if s in CLEAN_SEASONS else "🔵 validation" if s in VALID_SEASONS else "⚠️ training"
        season_rows.append({"Season":s,"Bets":sm["bets"],"Win%":f"{sm['wr']:.1f}%",
                             "ROI":f"{sm['roi']:+.1f}%","P&L":f"{sm['pnl']:+.1f}u",
                             "Status":status})
    st.dataframe(pd.DataFrame(season_rows), use_container_width=True, hide_index=True)

    # Chart
    plot_data = [r for r in season_rows if r["ROI"] != "—"]
    if plot_data:
        seasons_p = [r["Season"] for r in plot_data]
        pnls_p    = [float(r["P&L"].replace("u","").replace("+","")) for r in plot_data]
        colors    = ["#ff6b35" if s in CLEAN_SEASONS else "#888780" for s in seasons_p]
        fig = go.Figure(go.Bar(
            x=seasons_p, y=pnls_p,
            marker_color=[c if p >= 0 else "#A32D2D" for c,p in zip(colors,pnls_p)],
            text=[f"{p:+.1f}u" for p in pnls_p], textposition="outside",
        ))
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(title="Total P&L by season (units)", height=300,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    # Clean seasons info
    clean_df = filtered[filtered["season"].isin(CLEAN_SEASONS)] if not filtered.empty else pd.DataFrame()
    if not clean_df.empty and show_clean_only:
        clean_s = summarise(clean_df)
        st.info(f"🎯 **Clean out-of-sample only:** {clean_s['bets']} bets | "
                f"WR {clean_s['wr']:.1f}% vs BE {clean_s['be']:.1f}% | "
                f"ROI **{clean_s['roi']:+.1f}%**")

    # Download all games (with bet qualification flag) as CSV
    st.divider()
    all_games_df = annotate_all_games(raw, min_edge, max_edge, min_odds_filter,
                                      max_odds, underdogs_only, selected_seasons, away_only=away_only)
    if not all_games_df.empty:
        all_games_df = all_games_df.sort_values(
            ["season", "game_date"] if "game_date" in all_games_df.columns else ["season"]
        )
    st.download_button(
        label="⬇️ Download all games (CSV)",
        data=all_games_df.to_csv(index=False),
        file_name=f"wnba_all_games_{chosen_book_key}.csv",
        mime="text/csv",
        help="All games in the selected seasons — 'qualifies' column marks which ones pass your filters.",
    )

    # Save button
    st.divider()
    if st.button("💾 Apply these filters as live config", type="primary"):
        st.session_state["saved_min_edge"]  = min_edge
        st.session_state["saved_max_edge"]  = max_edge
        st.session_state["saved_min_odds"]  = min_odds_abs
        st.session_state["saved_max_odds"]  = max_odds
        st.session_state["saved_away_only"] = away_only
        try:
            import re
            cfg_path = ROOT / "config" / "settings.py"
            with open(cfg_path) as f: cfg = f.read()
            cfg = re.sub(r"MIN_EDGE_PCT\s*=\s*[\d.]+",  f"MIN_EDGE_PCT       = {float(min_edge)}", cfg)
            cfg = re.sub(r"BET_MAX_ODDS\s*=\s*[\d.]+",  f"BET_MAX_ODDS       = {int(max_odds)}", cfg)
            cfg = re.sub(r"BET_MIN_ODDS\s*=\s*[\d.]+", f"BET_MIN_ODDS       = {int(min_odds_abs)}", cfg)
            cfg = re.sub(r"BET_MAX_EDGE\s*=\s*[\d.]+",  f"BET_MAX_EDGE       = {float(max_edge)}", cfg)
            cfg = re.sub(r"BET_UNDERDOGS_ONLY\s*=.*", f"BET_UNDERDOGS_ONLY = {underdogs_only}", cfg)
            cfg = re.sub(r"BET_AWAY_ONLY\s*=.*",     f"BET_AWAY_ONLY      = {away_only}", cfg)
            with open(cfg_path, "w") as f: f.write(cfg)
        except: pass
        st.success(f"✅ Saved! Min edge: {min_edge}%, Odds: +{min_odds_abs+1}–+{max_odds}")
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: BET TRACKER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📋 Bet Tracker":
    st.title("📋 Bet Tracker")
    bets = load_bets()

    with st.expander("➕ Log a new bet", expanded=len(bets)==0):
        c1,c2,c3 = st.columns(3)
        with c1:
            b_date     = st.date_input("Date", value=date.today())
            b_home     = st.text_input("Home team", placeholder="LAS").upper()
            b_away     = st.text_input("Away team", placeholder="IND").upper()
        with c2:
            b_bet_team = st.text_input("Bet on", placeholder="IND").upper()
            b_odds     = st.number_input("Odds (American)", value=155, step=5)
            b_edge     = st.number_input("Edge %", value=20.0, step=0.5)
        with c3:
            b_units    = st.number_input("Units staked", value=1.0, step=0.5)
            b_result   = st.selectbox("Result", ["pending","win","loss"])

        if st.button("Log Bet", type="primary"):
            dec = american_to_decimal(b_odds)
            pnl = None
            if b_result == "win":  pnl = round((dec-1)*b_units, 3)
            if b_result == "loss": pnl = -b_units
            bets.append({
                "date": b_date.isoformat(), "home_team": b_home,
                "away_team": b_away, "bet_team": b_bet_team,
                "bet_odds": fmt_ml(b_odds), "edge_pct": b_edge,
                "result": b_result, "units": b_units, "pnl": pnl,
            })
            save_bets(bets)
            st.success("Logged!")
            st.rerun()

    # Auto-reconcile pending bets against game log results
    if bets and any(b.get("result") == "pending" for b in bets):
        if st.button("🔄 Auto-reconcile pending bets"):
            from src.scraper import fetch_season_game_log
            updated = 0
            all_bets = load_bets()
            for b in all_bets:
                if b.get("result") != "pending":
                    continue
                try:
                    bet_dt = pd.to_datetime(b["date"])
                    season = str(bet_dt.year) if bet_dt.month >= 5 else str(bet_dt.year - 1)
                    logs = fetch_season_game_log(season)
                    if logs.empty:
                        continue
                    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"]).dt.strftime("%Y-%m-%d")
                    game_day = logs[logs["GAME_DATE"] == b["date"]]
                    team_row = game_day[game_day["TEAM_ABBREVIATION"] == b["bet_team"]]
                    if team_row.empty:
                        continue
                    result = "win" if team_row.iloc[0]["WL"] == "W" else "loss"
                    dec    = american_to_decimal(b["bet_odds"])
                    units  = float(b["units"])
                    b["result"] = result
                    b["pnl"]    = round((dec - 1) * units, 3) if result == "win" else -units
                    updated += 1
                except Exception:
                    continue
            save_bets(all_bets)
            if updated:
                st.success(f"Reconciled {updated} bet(s).")
                st.rerun()
            else:
                st.info("No pending bets could be reconciled — results may not be posted yet.")

    if not bets:
        st.info("No bets yet.")
    else:
        df = pd.DataFrame(bets)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date", ascending=False)
        decided = df[df["result"].isin(["win","loss"])]

        if len(decided):
            w = (decided["result"]=="win").sum(); t = len(decided)
            pnl = decided["pnl"].sum()
            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("Total", len(df))
            m2.metric("Decided", t)
            m3.metric("Win Rate", f"{w/t:.1%}")
            m4.metric("P&L", f"{pnl:+.2f}u")
            m5.metric("ROI", f"{pnl/t*100:+.1f}%")
            st.divider()

        pending = df[df["result"]=="pending"]
        if len(pending):
            st.subheader(f"⏳ Pending ({len(pending)})")
            for i, (idx, row) in enumerate(pending.iterrows()):
                c1,c2,c3,c4 = st.columns([4,2,2,1])
                with c1:
                    st.write(f"**{row['bet_team']} ({row['bet_odds']})** — "
                             f"{row['home_team']} vs {row['away_team']} · "
                             f"{row['date'].strftime('%b %d')}")
                with c2:
                    new_result = st.selectbox("", ["pending","win","loss"],
                                              key=f"r_{i}",
                                              index=["pending","win","loss"].index(row["result"]))
                with c3:
                    if st.button("Update", key=f"u_{i}") and new_result != "pending":
                        all_bets = load_bets()
                        for b in all_bets:
                            if (b["date"] == row["date"].date().isoformat() and
                                b["bet_team"] == row["bet_team"] and
                                b["result"] == "pending"):
                                dec = american_to_decimal(b["bet_odds"])
                                b["result"] = new_result
                                b["pnl"] = round((dec-1)*float(b["units"]),3) if new_result=="win" else -float(b["units"])
                                break
                        save_bets(all_bets)
                        st.rerun()
                with c4:
                    if st.button("🗑️", key=f"del_{i}", help="Delete this bet"):
                        all_bets = load_bets()
                        all_bets = [b for b in all_bets if not (
                            b.get("date") == row["date"].date().isoformat() and
                            b.get("home_team") == row["home_team"] and
                            b.get("away_team") == row["away_team"] and
                            b.get("bet_team") == row["bet_team"]
                        )]
                        save_bets(all_bets)
                        st.rerun()
            st.divider()

        display = df[["date","home_team","away_team","bet_team","bet_odds",
                       "edge_pct","units","result","pnl"]].copy()
        display["date"] = display["date"].dt.strftime("%b %d")
        display.columns = ["Date","Home","Away","Bet","Odds","Edge%","Units","Result","P&L"]
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.divider()
        dl_bets = df[["date","home_team","away_team","bet_team","bet_odds",
                       "edge_pct","units","result","pnl"]].copy()
        dl_bets["date"] = dl_bets["date"].dt.strftime("%Y-%m-%d")
        st.download_button(
            label="⬇️ Download bet log (CSV)",
            data=dl_bets.to_csv(index=False),
            file_name="wnba_bet_log.csv",
            mime="text/csv",
        )

        with st.expander("🗑️ Delete a bet"):
            labels = [
                f"{row['date'].strftime('%b %d')} — {row['bet_team']} ({row['bet_odds']}) "
                f"{row['home_team']} vs {row['away_team']} [{row['result']}]"
                for _, row in df.iterrows()
            ]
            to_delete = st.selectbox("Select bet to delete", labels, key="del_select")
            if st.button("Delete selected bet", type="primary", key="del_confirmed"):
                del_idx = labels.index(to_delete)
                del_row = df.iloc[del_idx]
                all_bets = load_bets()
                all_bets = [b for b in all_bets if not (
                    b.get("date") == del_row["date"].date().isoformat() and
                    b.get("home_team") == del_row["home_team"] and
                    b.get("away_team") == del_row["away_team"] and
                    b.get("bet_team") == del_row["bet_team"]
                )]
                save_bets(all_bets)
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Performance":
    st.title("📈 Live Performance")
    bets = load_bets()
    if not bets:
        st.info("No bets tracked yet.")
        st.stop()

    df = pd.DataFrame(bets)
    df["date"] = pd.to_datetime(df["date"])
    df["pnl"]  = pd.to_numeric(df["pnl"], errors="coerce")
    decided = df[df["result"].isin(["win","loss"])].sort_values("date")

    if decided.empty:
        st.info("No decided bets yet.")
        st.stop()

    wins = (decided["result"]=="win").sum(); total = len(decided)
    pnl  = decided["pnl"].sum(); roi = pnl/total*100

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Bets", total)
    k2.metric("Win Rate", f"{wins/total:.1%}")
    k3.metric("P&L", f"{pnl:+.2f}u")
    k4.metric("ROI", f"{roi:+.1f}%")
    k5.metric("Pending", (df["result"]=="pending").sum())

    decided["cum_pnl"] = decided["pnl"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=decided["date"], y=decided["cum_pnl"],
        mode="lines+markers", line=dict(color="#ff6b35", width=2),
        marker=dict(size=8,
            color=decided["result"].map({"win":"#ff6b35","loss":"#A32D2D"})),
        hovertemplate="%{x|%b %d}<br>%{y:+.2f}u<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(title="Cumulative P&L", height=350,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    decided["month"] = decided["date"].dt.to_period("M").astype(str)
    monthly = decided.groupby("month").agg(
        bets=("pnl","count"), pnl=("pnl","sum"),
        wins=("result", lambda x: (x=="win").sum())
    ).reset_index()
    monthly["roi"] = monthly["pnl"]/monthly["bets"]*100

    c1,c2 = st.columns(2)
    with c1:
        fig2 = px.bar(monthly, x="month", y="pnl",
                      color="pnl", color_continuous_scale=["#A32D2D","#BA7517","#ff6b35"],
                      title="Monthly P&L")
        fig2.update_layout(height=260, coloraxis_showscale=False,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        fig3 = px.bar(monthly, x="month", y="roi",
                      color="roi", color_continuous_scale=["#A32D2D","#BA7517","#ff6b35"],
                      title="Monthly ROI %")
        fig3.update_layout(height=260, coloraxis_showscale=False,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: ABOUT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ About":
    st.title("ℹ️ About This Model")

    st.markdown("""
    ### What this is
    A machine-learning ensemble that predicts the probability of the home team winning each WNBA
    game, then compares that probability against Pinnacle's vig-removed line to identify value bets.

    ### How predictions are made
    The model blends four signals:

    | Signal | Weight | Description |
    |--------|--------|-------------|
    | Logistic Regression | 25% | Linear baseline on rolling team stats |
    | XGBoost | 35% | Gradient-boosted trees, Optuna-tuned |
    | LightGBM | 35% | Gradient-boosted trees, Optuna-tuned |
    | Elo | 5% | Head-to-head team strength rating |

    Rolling features include: points scored/allowed, plus-minus, pace, win streak, rest days,
    back-to-back flags, and live injury impact (minute-weighted ESPN data).

    ### Bet filter criteria
    A bet is flagged only when **all** of the following are met:
    - Model edge vs. Pinnacle ≥ **15%**
    - Odds between **+120** and **+325** (away underdogs only)
    - Edge cap of **60%** (extreme outliers excluded)

    These filters were optimised on clean out-of-sample seasons (2024–2025).

    ### Data sources
    - **Game logs & schedule:** stats.wnba.com (official WNBA stats API)
    - **Injury reports:** ESPN
    - **Odds:** The Odds API (Pinnacle sharp line + major US books)

    ### Model performance (backtest, 2024–2025)
    Backtested against Pinnacle closing lines — the sharpest available market.
    See the **Filter Playground** tab for full season-by-season breakdown.

    ---
    ### ⚠️ Disclaimer
    This tool is provided for **informational and entertainment purposes only**.
    It does not constitute financial, investment, or gambling advice.
    Past model performance does not guarantee future results.
    Sports betting involves significant risk — only bet what you can afford to lose.
    The operator of this tool accepts no liability for any losses incurred.
    Please comply with the gambling laws applicable in your jurisdiction.
    """)

    st.divider()
    st.caption("Model v2.0 · Trained May 2026 · WNBA seasons 2015–2025")