"""
WNBA ensemble model — identical architecture to NBA model.
LR 25% + XGBoost 35% + LightGBM 35% + Elo 5%
"""
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

from config.settings import (
    MODEL_DIR, PROC_DIR, VALID_SEASON, TEST_SEASON, TEST_SEASON_2,
    ENSEMBLE_WEIGHTS
)

log = logging.getLogger("model")


def split_data(df: pd.DataFrame, feat_cols: list):
    """Time-based split — exclude both test seasons from training."""
    try:
        held_out = [VALID_SEASON, TEST_SEASON, TEST_SEASON_2]
    except Exception:
        held_out = [VALID_SEASON, TEST_SEASON]

    train_mask = ~df["SEASON"].isin(held_out)
    valid_mask =  df["SEASON"] == VALID_SEASON
    test_mask  =  df["SEASON"] == TEST_SEASON

    def get_xy(mask):
        sub = df[mask].dropna(subset=feat_cols + ["HOME_WIN"])
        return sub[feat_cols].fillna(0).values, sub["HOME_WIN"].values, sub

    X_tr, y_tr, meta_tr = get_xy(train_mask)
    X_va, y_va, meta_va = get_xy(valid_mask)
    X_te, y_te, meta_te = get_xy(test_mask)

    log.info(f"Train: {len(X_tr):,} | Valid: {len(X_va):,} | Test: {len(X_te):,}")
    return X_tr, y_tr, X_va, y_va, X_te, y_te, meta_te


class WNBAEnsemble:
    def __init__(self):
        self.scaler    = StandardScaler()
        self.lr        = None
        self.xgb_model = None
        self.lgb_model = None
        self.feat_cols = None
        self.weights   = ENSEMBLE_WEIGHTS

    def fit(self, X_tr, y_tr, X_va, y_va, feat_cols: list):
        self.feat_cols = feat_cols

        X_tr_s = self.scaler.fit_transform(X_tr)
        X_va_s = self.scaler.transform(X_va)

        # Logistic Regression
        log.info("Training Logistic Regression...")
        lr_base = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.lr = CalibratedClassifierCV(lr_base, cv=3, method="isotonic")
        self.lr.fit(X_tr_s, y_tr)
        lr_auc = roc_auc_score(y_va, self.lr.predict_proba(X_va_s)[:, 1])
        log.info(f"  LR  — AUC: {lr_auc:.4f}")

        # XGBoost
        log.info("Training XGBoost...")
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="logloss", verbosity=0,
        )
        self.xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        xgb_auc = roc_auc_score(y_va, self.xgb_model.predict_proba(X_va)[:, 1])
        log.info(f"  XGB — AUC: {xgb_auc:.4f}")

        # LightGBM
        log.info("Training LightGBM...")
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=-1,
        )
        self.lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            feature_name=feat_cols,
        )
        lgb_auc = roc_auc_score(y_va, self.lgb_model.predict_proba(X_va)[:, 1])
        log.info(f"  LGB — AUC: {lgb_auc:.4f}")

        return self

    def predict_proba(self, X: np.ndarray, elo_probs: np.ndarray = None) -> np.ndarray:
        X_s = self.scaler.transform(X)
        p_lr  = self.lr.predict_proba(X_s)[:, 1]
        p_xgb = self.xgb_model.predict_proba(X)[:, 1]
        p_lgb = self.lgb_model.predict_proba(X)[:, 1]

        w = self.weights
        if elo_probs is not None:
            blend = (w["lr"] * p_lr + w["xgb"] * p_xgb +
                     w["lgb"] * p_lgb + w["elo"] * elo_probs)
        else:
            scale = 1 - w["elo"]
            blend = (w["lr"] / scale * p_lr + w["xgb"] / scale * p_xgb +
                     w["lgb"] / scale * p_lgb)

        return blend

    def evaluate(self, X, y, elo_probs=None, label="Test"):
        probs = self.predict_proba(X, elo_probs)
        metrics = {
            "n_games":    len(y),
            "accuracy":   round(accuracy_score(y, (probs > 0.5).astype(int)), 4),
            "brier_score": round(brier_score_loss(y, probs), 4),
            "roc_auc":    round(roc_auc_score(y, probs), 4),
            "log_loss":   round(log_loss(y, probs), 3),
        }
        log.info(f"Evaluation metrics ({label}): {metrics}")
        return metrics

    def save(self):
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(self.scaler,     MODEL_DIR / "model_scaler.pkl")
        joblib.dump(self.lr,         MODEL_DIR / "model_lr.pkl")
        joblib.dump(self.xgb_model,  MODEL_DIR / "model_xgb.pkl")
        joblib.dump(self.lgb_model,  MODEL_DIR / "model_lgb.pkl")
        joblib.dump(self.weights,    MODEL_DIR / "weights.pkl")
        joblib.dump(self.feat_cols,  MODEL_DIR / "feat_cols.pkl")
        log.info(f"Models saved to {MODEL_DIR}")

    def load(self):
        self.scaler     = joblib.load(MODEL_DIR / "model_scaler.pkl")
        self.lr         = joblib.load(MODEL_DIR / "model_lr.pkl")
        self.xgb_model  = joblib.load(MODEL_DIR / "model_xgb.pkl")
        self.lgb_model  = joblib.load(MODEL_DIR / "model_lgb.pkl")
        self.weights    = joblib.load(MODEL_DIR / "weights.pkl")
        self.feat_cols  = joblib.load(MODEL_DIR / "feat_cols.pkl")
        log.info("Models loaded.")
        return self
