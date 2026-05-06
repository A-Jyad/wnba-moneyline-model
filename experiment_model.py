"""
experiment_model.py — Model improvement experiments (read-only, no production changes).

Experiments:
  1. Ensemble weight optimisation   (scipy.optimize)
  2. XGB + LGB hyperparameter tuning (Optuna)
  3. Feature selection              (XGB+LGB importance, optional SHAP)
  4. Defensive rating feature       (DEF_RTG_roll10 from opponent PTS allowed)

Usage:
  python experiment_model.py                      # all four experiments
  python experiment_model.py --exp 1 3            # specific subset
  python experiment_model.py --trials 60          # more Optuna trials (exp 2)
  python experiment_model.py --top 15 25 35       # feature counts (exp 3)
  python experiment_model.py --exp 3 --use-tuned  # exp 3 with Optuna params
  python experiment_model.py --exp 2 3 --use-tuned  # tune then select (combined)
"""
import sys, argparse, warnings, logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss, accuracy_score

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)   # suppress src/ INFO spam

# ── Tuned hyperparameters from last Optuna run ────────────────────────────────
# These are updated automatically when exp 2 runs in the same session.
# If running exp 3 standalone with --use-tuned, these constants are used.
# Update manually after each new Optuna run.
TUNED_XGB_PARAMS = {
    "n_estimators": 420, "max_depth": 2, "learning_rate": 0.012689963970428347,
    "subsample": 0.5200124713778093, "colsample_bytree": 0.4787843586012409,
    "min_child_weight": 11, "reg_alpha": 0.010210643059632651,
    "reg_lambda": 0.012055221510750417,
}
TUNED_LGB_PARAMS = {
    "n_estimators": 504, "max_depth": 2, "learning_rate": 0.04048453066116998,
    "subsample": 0.97691511499413, "colsample_bytree": 0.5359101510033306,
    "min_child_samples": 41, "reg_alpha": 0.0005417857894104932,
    "reg_lambda": 0.003731304786469815,
}

# ── Shared helpers ─────────────────────────────────────────────────────────────

def load_and_split():
    from src.model import load_feature_matrix
    from src.features import get_feature_columns
    from config.settings import VALID_SEASON, TEST_SEASON, TEST_SEASON_2

    df = load_feature_matrix()
    feat_cols = get_feature_columns(df)
    held_out = [VALID_SEASON, TEST_SEASON, TEST_SEASON_2]

    def _split(mask):
        sub = df[mask].dropna(subset=feat_cols + ["HOME_WIN"]).copy()
        X   = sub[feat_cols].values
        y   = sub["HOME_WIN"].values
        elo = (1 / (1 + 10 ** (-sub["DIFF_ELO_PRE"].values / 400))
               if "DIFF_ELO_PRE" in sub.columns else np.full(len(sub), 0.5))
        return X, y, elo, sub

    X_tr, y_tr, _,      _       = _split(~df["SEASON"].isin(held_out))
    X_va, y_va, elo_va, _       = _split(df["SEASON"] == VALID_SEASON)
    X_te, y_te, elo_te, meta_te = _split(df["SEASON"] == TEST_SEASON)

    print(f"Train: {len(y_tr):,}  Valid: {len(y_va):,}  Test: {len(y_te):,}  Features: {len(feat_cols)}")
    return df, feat_cols, X_tr, y_tr, X_va, y_va, X_te, y_te, elo_va, elo_te


def train_base_models(X_tr, y_tr, X_va, y_va):
    from src.model import train_logistic, train_xgboost, train_lightgbm
    print("Training base models (LR + XGB + LGB)…")
    lr  = train_logistic(X_tr, y_tr, X_va, y_va)
    xgb = train_xgboost(X_tr, y_tr, X_va, y_va)
    lgb = train_lightgbm(X_tr, y_tr, X_va, y_va)
    return lr, xgb, lgb


def _probs(model, X):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.predict_proba(X)[:, 1]


def _blend(p_lr, p_xgb, p_lgb, p_elo, w_lr, w_xgb, w_lgb, w_elo=0.05):
    total = w_lr + w_xgb + w_lgb + w_elo
    return (p_lr*w_lr + p_xgb*w_xgb + p_lgb*w_lgb + p_elo*w_elo) / total


def _metrics(y, p):
    return {
        "brier":   round(brier_score_loss(y, p), 4),
        "auc":     round(roc_auc_score(y, p), 4),
        "acc":     round(accuracy_score(y, (p >= 0.5).astype(int)), 4),
        "logloss": round(log_loss(y, p), 4),
    }


def _fmt(label, m, ref=None):
    s = f"  {label:<34s} Brier={m['brier']:.4f}  AUC={m['auc']:.4f}  Acc={m['acc']:.4f}"
    if ref:
        db = m["brier"] - ref["brier"]
        da = m["auc"]   - ref["auc"]
        s += f"  dBrier={db:+.4f}  dAUC={da:+.4f}"
    return s


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Ensemble weight optimisation
# ══════════════════════════════════════════════════════════════════════════════

def exp1_weights(models, elo_va, elo_te, y_va, y_te):
    from scipy.optimize import minimize
    from config.settings import ENSEMBLE_WEIGHTS

    print("\n" + "="*70)
    print("EXPERIMENT 1 — Ensemble Weight Optimisation")
    print("="*70)

    lr, xgb, lgb = models
    p_lr_va  = _probs(lr,  X_va);  p_lr_te  = _probs(lr,  X_te)
    p_xgb_va = _probs(xgb, X_va);  p_xgb_te = _probs(xgb, X_te)
    p_lgb_va = _probs(lgb, X_va);  p_lgb_te = _probs(lgb, X_te)

    w = ENSEMBLE_WEIGHTS
    p_curr_va = _blend(p_lr_va, p_xgb_va, p_lgb_va, elo_va, w["lr"], w["xgb"], w["lgb"], w["elo"])
    p_curr_te = _blend(p_lr_te, p_xgb_te, p_lgb_te, elo_te, w["lr"], w["xgb"], w["lgb"], w["elo"])
    curr_va = _metrics(y_va, p_curr_va)
    curr_te = _metrics(y_te, p_curr_te)

    print(f"\nCurrent weights  LR={w['lr']}  XGB={w['xgb']}  LGB={w['lgb']}  Elo={w['elo']}")
    print(_fmt("valid (current)", curr_va))
    print(_fmt("test  (current)", curr_te))

    # Optimise 4-weight blend on validation Brier
    def objective(weights):
        wl, wx, wg, we = weights
        p = _blend(p_lr_va, p_xgb_va, p_lgb_va, elo_va, wl, wx, wg, we)
        return brier_score_loss(y_va, p)

    constraints = [{"type": "eq", "fun": lambda w: sum(w) - 1}]
    bounds = [(0, 1)] * 4
    starts = [
        [0.25, 0.35, 0.35, 0.05],
        [0.33, 0.33, 0.33, 0.01],
        [0.10, 0.45, 0.45, 0.00],
        [0.50, 0.25, 0.25, 0.00],
        [0.20, 0.40, 0.35, 0.05],
    ]

    best = None
    for s in starts:
        res = minimize(objective, s, method="SLSQP", bounds=bounds, constraints=constraints)
        if best is None or res.fun < best.fun:
            best = res

    wl, wx, wg, we = best.x
    p_opt_va = _blend(p_lr_va, p_xgb_va, p_lgb_va, elo_va, wl, wx, wg, we)
    p_opt_te = _blend(p_lr_te, p_xgb_te, p_lgb_te, elo_te, wl, wx, wg, we)
    opt_va = _metrics(y_va, p_opt_va)
    opt_te = _metrics(y_te, p_opt_te)

    print(f"\nOptimised weights  LR={wl:.3f}  XGB={wx:.3f}  LGB={wg:.3f}  Elo={we:.3f}")
    print(_fmt("valid (optimised)", opt_va, curr_va))
    print(_fmt("test  (optimised)", opt_te, curr_te))

    return {"lr": round(wl, 3), "xgb": round(wx, 3), "lgb": round(wg, 3), "elo": round(we, 3)}


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Hyperparameter tuning (Optuna)
# ══════════════════════════════════════════════════════════════════════════════

def exp2_hparams(X_tr, y_tr, X_va, y_va, X_te, y_te, elo_va, elo_te, n_trials=40):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("\nEXPERIMENT 2 — skipped (pip install optuna)")
        return

    import xgboost as xgb_lib
    import lightgbm as lgb_lib
    from src.model import _calibrate, train_logistic
    from config.settings import ENSEMBLE_WEIGHTS

    print("\n" + "="*70)
    print(f"EXPERIMENT 2 — Hyperparameter Tuning  ({n_trials} Optuna trials each)")
    print("="*70)

    # Baseline ensemble on test for comparison (LR uses defaults, already trained above)
    lr_base = train_logistic(X_tr, y_tr, X_va, y_va)

    def _base_xgb():
        m = xgb_lib.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03,
                                    subsample=0.8, colsample_bytree=0.8,
                                    eval_metric="logloss", random_state=42, n_jobs=-1)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return _calibrate(m, X_va, y_va, method="isotonic")

    def _base_lgb():
        m = lgb_lib.LGBMClassifier(n_estimators=400, max_depth=4, learning_rate=0.03,
                                     subsample=0.8, colsample_bytree=0.8,
                                     random_state=42, n_jobs=-1, verbose=-1)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb_lib.early_stopping(50, verbose=False), lgb_lib.log_evaluation(-1)])
        return _calibrate(m, X_va, y_va, method="isotonic")

    print("\nBaseline (current params)…")
    xgb_base = _base_xgb();  lgb_base = _base_lgb()
    w = ENSEMBLE_WEIGHTS
    p_base_te = _blend(_probs(lr_base, X_te), _probs(xgb_base, X_te),
                        _probs(lgb_base, X_te), elo_te, w["lr"], w["xgb"], w["lgb"], w["elo"])
    base_te = _metrics(y_te, p_base_te)
    print(_fmt("test  (baseline)", base_te))

    # ── XGBoost search ────────────────────────────────────────────────────────
    print(f"\nOptimising XGBoost ({n_trials} trials)…")

    def xgb_obj(trial):
        p = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 700),
            "max_depth":        trial.suggest_int("max_depth", 2, 6),
            "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "eval_metric": "logloss", "random_state": 42, "n_jobs": -1,
        }
        m = xgb_lib.XGBClassifier(**p)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        cal = _calibrate(m, X_va, y_va, method="isotonic")
        return brier_score_loss(y_va, _probs(cal, X_va))

    xgb_study = optuna.create_study(direction="minimize")
    xgb_study.optimize(xgb_obj, n_trials=n_trials, show_progress_bar=True)
    best_xgb_p = xgb_study.best_params

    xgb_tuned_m = xgb_lib.XGBClassifier(**{**best_xgb_p,
                                            "eval_metric": "logloss", "random_state": 42, "n_jobs": -1})
    xgb_tuned_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb_tuned = _calibrate(xgb_tuned_m, X_va, y_va, method="isotonic")

    # ── LightGBM search ───────────────────────────────────────────────────────
    print(f"\nOptimising LightGBM ({n_trials} trials)…")

    def lgb_obj(trial):
        p = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 700),
            "max_depth":         trial.suggest_int("max_depth", 2, 6),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": 42, "n_jobs": -1, "verbose": -1,
        }
        m = lgb_lib.LGBMClassifier(**p)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb_lib.early_stopping(50, verbose=False), lgb_lib.log_evaluation(-1)])
        cal = _calibrate(m, X_va, y_va, method="isotonic")
        return brier_score_loss(y_va, _probs(cal, X_va))

    lgb_study = optuna.create_study(direction="minimize")
    lgb_study.optimize(lgb_obj, n_trials=n_trials, show_progress_bar=True)
    best_lgb_p = lgb_study.best_params

    lgb_tuned_m = lgb_lib.LGBMClassifier(**{**best_lgb_p, "random_state": 42, "n_jobs": -1, "verbose": -1})
    lgb_tuned_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                    callbacks=[lgb_lib.early_stopping(50, verbose=False), lgb_lib.log_evaluation(-1)])
    lgb_tuned = _calibrate(lgb_tuned_m, X_va, y_va, method="isotonic")

    # ── Combined result ───────────────────────────────────────────────────────
    p_tuned_te = _blend(_probs(lr_base, X_te), _probs(xgb_tuned, X_te),
                         _probs(lgb_tuned, X_te), elo_te, w["lr"], w["xgb"], w["lgb"], w["elo"])
    tuned_te = _metrics(y_te, p_tuned_te)

    print(f"\nBest XGB params: {best_xgb_p}")
    print(f"Best LGB params: {best_lgb_p}")
    print(_fmt("test  (tuned ensemble)", tuned_te, base_te))

    return {"xgb": best_xgb_p, "lgb": best_lgb_p}


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — Feature selection
# ══════════════════════════════════════════════════════════════════════════════

def exp3_features(models, feat_cols, X_tr, y_tr, X_va, y_va, X_te, y_te,
                  elo_va, elo_te, top_n_list=(15, 25, 35),
                  xgb_params=None, lgb_params=None):
    import xgboost as xgb_lib
    import lightgbm as lgb_lib
    from src.model import train_logistic, _calibrate
    from config.settings import ENSEMBLE_WEIGHTS

    use_tuned = xgb_params is not None and lgb_params is not None

    print("\n" + "="*70)
    print("EXPERIMENT 3 — Feature Selection" + (" [with TUNED params]" if use_tuned else " [with DEFAULT params]"))
    print("="*70)

    # ── Helpers to train XGB/LGB with any param dict ──────────────────────────
    def _train_xgb(Xtr, ytr, Xva, yva, params):
        m = xgb_lib.XGBClassifier(**{**params, "eval_metric": "logloss",
                                      "random_state": 42, "n_jobs": -1})
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return _calibrate(m, Xva, yva, method="isotonic")

    def _train_lgb(Xtr, ytr, Xva, yva, params):
        m = lgb_lib.LGBMClassifier(**{**params, "random_state": 42,
                                       "n_jobs": -1, "verbose": -1})
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb_lib.early_stopping(50, verbose=False),
                         lgb_lib.log_evaluation(-1)])
        return _calibrate(m, Xva, yva, method="isotonic")

    from config.settings import XGB_PARAMS, LGB_PARAMS
    _xgb_p = xgb_params if use_tuned else {k: v for k, v in XGB_PARAMS.items() if k != "use_label_encoder"}
    _lgb_p = lgb_params if use_tuned else LGB_PARAMS

    if use_tuned:
        print(f"\n  XGB: {xgb_params}")
        print(f"  LGB: {lgb_params}")
    else:
        print("\n  Using default params. Run with --use-tuned to apply Optuna params.")

    # ── Feature importance ranking (always uses the default-trained base models) ──
    def _get_base(cal):
        if hasattr(cal, "estimator"):  return cal.estimator
        if hasattr(cal, "base"):       return cal.base
        return cal

    _, xgb, lgb = models
    imp_xgb = np.asarray(_get_base(xgb).feature_importances_, dtype=float)
    imp_lgb = np.asarray(_get_base(lgb).feature_importances_, dtype=float)
    imp_xgb /= imp_xgb.sum(); imp_lgb /= imp_lgb.sum()
    ranked = sorted(zip(feat_cols, (imp_xgb + imp_lgb) / 2), key=lambda x: -x[1])

    print(f"\nTop 20 features (XGB+LGB blended importance):")
    for i, (feat, score) in enumerate(ranked[:20], 1):
        print(f"  {i:2d}. {feat:<50s} {score*100:.2f}%")

    try:
        import shap
        print("\nComputing SHAP values (XGB)…")
        shap_vals  = np.abs(shap.TreeExplainer(_get_base(xgb)).shap_values(X_va)).mean(axis=0)
        shap_vals /= shap_vals.sum()
        ranked     = sorted(zip(feat_cols, shap_vals), key=lambda x: -x[1])
        print("Top 20 features by SHAP:")
        for i, (feat, score) in enumerate(ranked[:20], 1):
            print(f"  {i:2d}. {feat:<50s} {score*100:.2f}%")
    except ImportError:
        print("\n  SHAP not installed (pip install shap) — using blended importance.")

    # ── Baseline: all features, with whichever params are active ─────────────
    w = ENSEMBLE_WEIGHTS
    print(f"\nTraining baseline (all {len(feat_cols)} features)…")
    lr_b  = train_logistic(X_tr, y_tr, X_va, y_va)
    xgb_b = _train_xgb(X_tr, y_tr, X_va, y_va, _xgb_p)
    lgb_b = _train_lgb(X_tr, y_tr, X_va, y_va, _lgb_p)
    p_base_va = _blend(_probs(lr_b, X_va), _probs(xgb_b, X_va), _probs(lgb_b, X_va),
                        elo_va, w["lr"], w["xgb"], w["lgb"], w["elo"])
    p_base_te = _blend(_probs(lr_b, X_te), _probs(xgb_b, X_te), _probs(lgb_b, X_te),
                        elo_te, w["lr"], w["xgb"], w["lgb"], w["elo"])
    base_va = _metrics(y_va, p_base_va)
    base_te = _metrics(y_te, p_base_te)

    print(f"\n{'Features':<10s} {'Brier-valid':>12s} {'Brier-test':>11s} {'AUC-test':>10s}  {'dBrier-test':>12s}  {'dAUC-test':>10s}")
    print(f"  {'all ('+str(len(feat_cols))+')':<8s}  {base_va['brier']:>10.4f}  {base_te['brier']:>10.4f}  {base_te['auc']:>10.4f}")

    best_result = None
    for top_n in top_n_list:
        top_feats = [f for f, _ in ranked[:top_n]]
        idx = [feat_cols.index(f) for f in top_feats]
        Xtr_s = X_tr[:, idx]; Xva_s = X_va[:, idx]; Xte_s = X_te[:, idx]

        lr_s  = train_logistic(Xtr_s, y_tr, Xva_s, y_va)
        xgb_s = _train_xgb(Xtr_s, y_tr, Xva_s, y_va, _xgb_p)
        lgb_s = _train_lgb(Xtr_s, y_tr, Xva_s, y_va, _lgb_p)

        p_va = _blend(_probs(lr_s, Xva_s), _probs(xgb_s, Xva_s), _probs(lgb_s, Xva_s),
                       elo_va, w["lr"], w["xgb"], w["lgb"], w["elo"])
        p_te = _blend(_probs(lr_s, Xte_s), _probs(xgb_s, Xte_s), _probs(lgb_s, Xte_s),
                       elo_te, w["lr"], w["xgb"], w["lgb"], w["elo"])
        m_va = _metrics(y_va, p_va); m_te = _metrics(y_te, p_te)
        db = m_te["brier"] - base_te["brier"]
        da = m_te["auc"]   - base_te["auc"]

        print(f"  top-{top_n:<5d}  {m_va['brier']:>10.4f}  {m_te['brier']:>10.4f}  {m_te['auc']:>10.4f}  {db:>+12.4f}  {da:>+10.4f}")

        if best_result is None or m_te["brier"] < best_result[1]["brier"]:
            best_result = (top_n, m_te, [f for f, _ in ranked[:top_n]])

    if best_result:
        print(f"\nBest subset: top-{best_result[0]} features  "
              f"(dBrier test = {best_result[1]['brier']-base_te['brier']:+.4f})")

    return best_result


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4 — Defensive rating feature
# ══════════════════════════════════════════════════════════════════════════════

def exp4_def_rtg(df, feat_cols, X_tr, y_tr, X_va, y_va, X_te, y_te, elo_va, elo_te):
    from src.model import train_logistic, train_xgboost, train_lightgbm, split_data
    from config.settings import RAW_DIR, VALID_SEASON, TEST_SEASON, TEST_SEASON_2, ENSEMBLE_WEIGHTS

    print("\n" + "="*70)
    print("EXPERIMENT 4 — Defensive Rating Feature (DEF_RTG_roll10)")
    print("="*70)

    raw_path = RAW_DIR / "all_game_logs.parquet"
    if not raw_path.exists():
        print(f"  Raw game logs not found at {raw_path}. Run --fetch first. Skipping.")
        return

    raw = pd.read_parquet(raw_path)
    raw.columns = [c.upper() for c in raw.columns]
    raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"])
    raw = raw.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)

    # PTS_ALLOWED = opponent's PTS in the same game
    pts_by_game_team = raw.groupby(["GAME_ID", "TEAM_ABBREVIATION"])["PTS"].first().to_dict()

    def _opp_team(matchup, my_team):
        m = str(matchup)
        if "vs." in m: return m.split("vs.")[-1].strip().split()[0]
        if "@" in m:
            parts = m.split("@")
            return parts[-1].strip().split()[0] if parts[0].strip().split()[0] == my_team \
                   else parts[0].strip().split()[0]
        return None

    raw["_OPP"] = raw.apply(lambda r: _opp_team(r.get("MATCHUP", ""), r["TEAM_ABBREVIATION"]), axis=1)
    raw["PTS_ALLOWED"] = raw.apply(
        lambda r: pts_by_game_team.get((r["GAME_ID"], r["_OPP"]), np.nan), axis=1
    )

    # Pace (possessions proxy)
    if all(c in raw.columns for c in ["FGA", "FTA", "TOV", "OREB"]):
        raw["PACE"] = (raw["FGA"] + 0.4*raw["FTA"] + raw["TOV"] - raw["OREB"]).replace(0, np.nan)
    elif "PACE" not in raw.columns:
        print("  Cannot compute PACE — skipping exp 4.")
        return

    raw["DEF_RTG"] = (raw["PTS_ALLOWED"] / raw["PACE"] * 100).fillna(0)

    grp = raw.groupby("TEAM_ABBREVIATION")
    raw["DEF_RTG_roll10"] = grp["DEF_RTG"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()
    )
    raw["DEF_RTG_ewm"] = grp["DEF_RTG"].transform(
        lambda x: x.shift(1).ewm(span=10, min_periods=1).mean()
    )

    def_map = raw.set_index(["GAME_ID", "TEAM_ABBREVIATION"])[["DEF_RTG_roll10", "DEF_RTG_ewm"]]

    df_exp = df.copy()
    for side, col in [("HOME", "HOME_TEAM_ABBREVIATION"), ("AWAY", "AWAY_TEAM_ABBREVIATION")]:
        side_def = (
            df_exp[["GAME_ID", col]]
            .merge(def_map.reset_index().rename(columns={"TEAM_ABBREVIATION": col}),
                   on=["GAME_ID", col], how="left")
        )
        df_exp[f"{side}_DEF_RTG_roll10"] = side_def["DEF_RTG_roll10"].values
        df_exp[f"{side}_DEF_RTG_ewm"]    = side_def["DEF_RTG_ewm"].values

    df_exp["DIFF_DEF_RTG_roll10"] = df_exp["HOME_DEF_RTG_roll10"] - df_exp["AWAY_DEF_RTG_roll10"]

    new_feats = [c for c in ["HOME_DEF_RTG_roll10", "HOME_DEF_RTG_ewm",
                              "AWAY_DEF_RTG_roll10", "AWAY_DEF_RTG_ewm",
                              "DIFF_DEF_RTG_roll10"] if c in df_exp.columns]
    feat_cols_exp = feat_cols + new_feats
    held_out = [VALID_SEASON, TEST_SEASON, TEST_SEASON_2]

    def _split_exp(mask):
        sub = df_exp[mask].dropna(subset=feat_cols_exp + ["HOME_WIN"]).copy()
        return sub[feat_cols_exp].values, sub["HOME_WIN"].values

    Xtr_e, ytr_e = _split_exp(~df_exp["SEASON"].isin(held_out))
    Xva_e, yva_e = _split_exp(df_exp["SEASON"] == VALID_SEASON)
    Xte_e, yte_e = _split_exp(df_exp["SEASON"] == TEST_SEASON)

    print(f"\nNew features added: {new_feats}")
    print(f"NaN coverage: {df_exp[new_feats].notna().mean().round(2).to_dict()}")

    w = ENSEMBLE_WEIGHTS
    lr_e  = train_logistic(Xtr_e, ytr_e, Xva_e, yva_e)
    xgb_e = train_xgboost(Xtr_e, ytr_e, Xva_e, yva_e)
    lgb_e = train_lightgbm(Xtr_e, ytr_e, Xva_e, yva_e)

    p_va_e = _blend(_probs(lr_e, Xva_e), _probs(xgb_e, Xva_e), _probs(lgb_e, Xva_e),
                     elo_va, w["lr"], w["xgb"], w["lgb"], w["elo"])
    p_te_e = _blend(_probs(lr_e, Xte_e), _probs(xgb_e, Xte_e), _probs(lgb_e, Xte_e),
                     elo_te, w["lr"], w["xgb"], w["lgb"], w["elo"])

    # Baseline: same models trained without DEF_RTG features
    lr_b  = train_logistic(X_tr, y_tr, X_va, y_va)
    xgb_b = train_xgboost(X_tr, y_tr, X_va, y_va)
    lgb_b = train_lightgbm(X_tr, y_tr, X_va, y_va)
    p_va_b = _blend(_probs(lr_b, X_va), _probs(xgb_b, X_va), _probs(lgb_b, X_va),
                     elo_va, w["lr"], w["xgb"], w["lgb"], w["elo"])
    p_te_b = _blend(_probs(lr_b, X_te), _probs(xgb_b, X_te), _probs(lgb_b, X_te),
                     elo_te, w["lr"], w["xgb"], w["lgb"], w["elo"])

    base_va = _metrics(y_va, p_va_b); base_te = _metrics(y_te, p_te_b)
    exp_va  = _metrics(yva_e, p_va_e); exp_te  = _metrics(yte_e, p_te_e)

    print(_fmt("valid (no DEF_RTG)",  base_va))
    print(_fmt("valid (+ DEF_RTG)",   exp_va,  base_va))
    print(_fmt("test  (no DEF_RTG)",  base_te))
    print(_fmt("test  (+ DEF_RTG)",   exp_te,  base_te))


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WNBA model improvement experiments")
    parser.add_argument("--exp",     nargs="+", type=int, default=[1, 2, 3, 4],
                        help="Which experiments to run (e.g. --exp 1 3)")
    parser.add_argument("--trials",  type=int,  default=40,
                        help="Optuna trials per model for exp 2 (default 40)")
    parser.add_argument("--top",       nargs="+", type=int, default=[15, 25, 35],
                        help="Feature counts to try in exp 3 (default 15 25 35)")
    parser.add_argument("--use-tuned", action="store_true",
                        help="Exp 3: use Optuna-tuned params (from exp 2 run or TUNED_*_PARAMS constants)")
    args = parser.parse_args()

    print("Loading data and splitting…")
    df, feat_cols, X_tr, y_tr, X_va, y_va, X_te, y_te, elo_va, elo_te = load_and_split()

    # Train base models once — reused by exp 1 and exp 3
    need_base_models = bool(set(args.exp) & {1, 3})
    models = train_base_models(X_tr, y_tr, X_va, y_va) if need_base_models else None

    results = {}

    if 1 in args.exp:
        results["exp1"] = exp1_weights(models, elo_va, elo_te, y_va, y_te)

    if 2 in args.exp:
        results["exp2"] = exp2_hparams(X_tr, y_tr, X_va, y_va, X_te, y_te,
                                        elo_va, elo_te, n_trials=args.trials)

    if 3 in args.exp:
        if args.use_tuned:
            # Prefer live exp 2 results from this session; fall back to module constants
            xgb_p = results["exp2"]["xgb"] if "exp2" in results else TUNED_XGB_PARAMS
            lgb_p = results["exp2"]["lgb"] if "exp2" in results else TUNED_LGB_PARAMS
            print(f"\n--use-tuned: sourcing params from "
                  f"{'exp 2 (this session)' if 'exp2' in results else 'TUNED_*_PARAMS constants'}")
        else:
            xgb_p = lgb_p = None
        results["exp3"] = exp3_features(models, feat_cols, X_tr, y_tr, X_va, y_va,
                                         X_te, y_te, elo_va, elo_te, top_n_list=args.top,
                                         xgb_params=xgb_p, lgb_params=lgb_p)

    if 4 in args.exp:
        results["exp4"] = exp4_def_rtg(df, feat_cols,
                                        X_tr, y_tr, X_va, y_va, X_te, y_te,
                                        elo_va, elo_te)

    print("\n" + "="*70)
    print("Done. Copy any promising params into config/settings.py and retrain.")
    print("="*70)
