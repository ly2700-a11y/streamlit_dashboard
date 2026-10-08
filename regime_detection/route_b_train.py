"""
route_b_train.py
================
Route B: Train A2C with regime-augmented 12-dim state.

Flowchart:
  REGIME DETECTION (per asset class)
    Load 50 contracts → Build 60×9 state matrix → FFT (180-dim) →
    GMM (n=3, training period only) → soft probs [p0, p1, p2]

  TRAINING
    Original 9-dim state  +  Regime 3-dim [p0, p1, p2]
    ──────────────────────────────────────────────────
    Concatenate → 12-dim augmented state
    Train single A2C (LSTM, n_features=12)

  TEST
    Apply trained GMM to test period → soft probs
    Augment state to 12-dim
    Single A2C inference → output action directly

Periods
-------
  Period 1: Train 2005-01-01 ~ 2010-12-31  |  Test 2011-01-01 ~ 2015-12-31
  Period 2: Train 2010-01-01 ~ 2015-12-31  |  Test 2016-01-01 ~ 2019-12-31

Asset classes (mirrors original A2C models):
  'Commodity', 'Equity Index', 'Fixed Income', 'Forex', 'All'
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── project root on sys.path ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "drl_models" / "a2c"))

from data_loader import load_clc_full
from config import ASSET_CLASSES
from drl_models.a2c.loaddata import (
    load_paper_rad_data,
    build_paper_features,
    make_state_tensor_single,
)
from drl_models.a2c.a2c_model import (
    PaperTradingEnv,
    PaperA2CTrainer,
    build_envs_from_state_dict,
    DEVICE,
)
from regime_detection.timeseries_fft_regime import (
    detect_regimes_for_asset_class_timeseries,
    predict_regime_soft_probs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
PERIODS = {
    "period_1": {
        "train": ("2005-01-01", "2010-12-31"),
        "test":  ("2011-01-01", "2015-12-31"),
        "data_start": "2004-01-01",   # extra warmup for MACD/RSI
    },
    "period_2": {
        "train": ("2010-01-01", "2015-12-31"),
        "test":  ("2016-01-01", "2019-12-31"),
        "data_start": "2009-01-01",
    },
}

# 9-dim feature columns (must match drl_models/a2c/loaddata.py build_paper_features)
FEATURE_COLS_9 = [
    "close_norm",
    "ret_1m", "ret_2m", "ret_3m", "ret_1y",
    "macd_8_24", "macd_16_48", "macd_32_96",
    "rsi_30",
]
# 3 regime soft-prob columns appended for Route B
REGIME_COLS = ["regime_prob_0", "regime_prob_1", "regime_prob_2"]
FEATURE_COLS_12 = FEATURE_COLS_9 + REGIME_COLS

N_UPDATES  = 2000   # training steps (reduce to 500 for a quick sanity check)
SIGMA_TARGET = 0.15 / np.sqrt(252)   # daily vol target ≈ 15% annual

OUTPUT_DIR = PROJECT_ROOT / "drl_models" / "a2c"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_clc_data_for_regime(tickers: List[str], data_start: str) -> Dict:
    """Load CLC data via data_loader (uppercase columns) for regime detection."""
    clc_data = {}
    for t in tickers:
        try:
            df = load_clc_full(t, start_date=data_start)
            if df is not None:
                clc_data[t] = df
        except Exception as e:
            logger.debug(f"  skip {t}: {e}")
    return clc_data


def _load_rad_data_for_a2c(tickers: List[str], data_start: str, data_end: str) -> Dict:
    """
    Load RAD data via loaddata.load_paper_rad_data (lowercase columns) and
    build paper features. Returns feature_dict: {ticker: feature_df}.
    """
    root_path = str(PROJECT_ROOT)
    data_dict, _, missing = load_paper_rad_data(
        root_path, start=data_start, end=data_end
    )
    if missing:
        logger.warning(f"  Missing tickers: {missing}")

    # restrict to the requested tickers
    data_dict = {t: v for t, v in data_dict.items() if t in tickers}
    feature_dict, _ = build_paper_features(data_dict, dropna=False)
    return feature_dict


def _merge_regime_probs(
    feature_df: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join regime soft probs onto feature_df by date.
    Missing dates (regime_df doesn't cover every feature row) are
    forward-filled, then backward-filled as fallback.
    """
    # Normalise date columns to date-only (no time component)
    fd = feature_df.copy()
    rd = regime_df[["date", "regime_prob_0", "regime_prob_1", "regime_prob_2"]].copy()

    fd["_date_key"] = pd.to_datetime(fd["date"]).dt.normalize()
    rd["_date_key"] = pd.to_datetime(rd["date"]).dt.normalize()

    merged = fd.merge(rd, on="_date_key", how="left", suffixes=("", "_r"))
    merged = merged.drop(columns=["_date_key", "date_r"], errors="ignore")

    # fill gaps
    for col in REGIME_COLS:
        merged[col] = merged[col].ffill().bfill().fillna(1.0 / 3)

    return merged.reset_index(drop=True)


def _build_augmented_state_dict(
    feature_dict: Dict[str, pd.DataFrame],
    regime_df: pd.DataFrame,
    train_start: str,
    train_end:   str,
    test_start:  str,
    test_end:    str,
) -> Tuple[Dict, Dict]:
    """
    Build (train_state_dict, test_state_dict) where each entry has
    X of shape (n_samples, 60, 12) and aligned_df with trade columns.

    Returns
    -------
    train_state_dict, test_state_dict : Dict[ticker, {"X": ..., "aligned_df": ...}]
    """
    train_s = pd.Timestamp(train_start)
    train_e = pd.Timestamp(train_end)
    test_s  = pd.Timestamp(test_start)
    test_e  = pd.Timestamp(test_end)

    train_state_dict = {}
    test_state_dict  = {}

    for ticker, feat_df in feature_dict.items():
        try:
            merged_df = _merge_regime_probs(feat_df, regime_df)

            # ---- training slice ----
            tr_df = merged_df[
                (merged_df["date"] >= train_s) & (merged_df["date"] <= train_e)
            ].reset_index(drop=True)

            X_tr, dates_tr, aligned_tr = make_state_tensor_single(
                tr_df, FEATURE_COLS_12, window=60,
                return_dates=True, return_current_row=True,
            )

            if len(X_tr) > 0:
                # aligned_tr comes from loaddata (lowercase columns)
                train_state_dict[ticker] = {
                    "X": X_tr,
                    "aligned_df": aligned_tr[["date", "close", "ret_1d", "ewm_vol_60"]],
                }

            # ---- test slice ----
            te_df = merged_df[
                (merged_df["date"] >= test_s) & (merged_df["date"] <= test_e)
            ].reset_index(drop=True)

            X_te, dates_te, aligned_te = make_state_tensor_single(
                te_df, FEATURE_COLS_12, window=60,
                return_dates=True, return_current_row=True,
            )

            if len(X_te) > 0:
                test_state_dict[ticker] = {
                    "X": X_te,
                    "aligned_df": aligned_te[["date", "close", "ret_1d", "ewm_vol_60"]],
                }

        except Exception as e:
            logger.warning(f"  state build failed for {ticker}: {e}")

    return train_state_dict, test_state_dict


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_route_b(
    asset_class: str,
    tickers: List[str],
    period_name: str,
    n_updates: int = N_UPDATES,
    sigma_target: float = SIGMA_TARGET,
    output_dir: Path = OUTPUT_DIR,
    n_updates_rollout: int = 32,
):
    """
    Full Route B pipeline for one asset class and one period.

    Parameters
    ----------
    asset_class  : e.g. 'Commodity', 'All'
    tickers      : list of tickers in this class
    period_name  : 'period_1' or 'period_2'
    """
    cfg = PERIODS[period_name]
    train_start, train_end = cfg["train"]
    test_start,  test_end  = cfg["test"]
    data_start             = cfg["data_start"]

    logger.info("=" * 70)
    logger.info(f"Route B  |  {asset_class}  |  {period_name}")
    logger.info(f"  Train: {train_start} ~ {train_end}")
    logger.info(f"  Test : {test_start} ~ {test_end}")
    logger.info("=" * 70)

    # ── Phase 1: fit GMM on training period ──────────────────────────────────
    logger.info("\n[Phase 1] Regime detection (fit on training data)...")
    clc_data = _load_clc_data_for_regime(tickers, data_start)

    train_regime_result = detect_regimes_for_asset_class_timeseries(
        clc_data=clc_data,
        asset_class_tickers=tickers,
        asset_class_name=asset_class,
        n_regimes=3,
        date_range=(train_start, train_end),
    )

    if train_regime_result is None:
        logger.error("  Phase 1 failed – no regime result. Aborting.")
        return None

    trained_gmm    = train_regime_result["gmm_model"]
    trained_scaler = train_regime_result["fft_scaler"]
    silhouette     = train_regime_result["silhouette_score"]
    logger.info(f"  Silhouette score (train): {silhouette:.4f}")

    # ── Phase 2: predict soft probs for full period (train + test) ───────────
    logger.info("\n[Phase 2] Predicting soft probs for full period...")
    full_start = data_start   # use full loaded range
    full_end   = test_end

    full_regime_df = predict_regime_soft_probs(
        clc_data=clc_data,
        asset_class_tickers=tickers,
        asset_class_name=asset_class,
        trained_gmm=trained_gmm,
        trained_scaler=trained_scaler,
        date_range=(full_start, full_end),
    )

    if full_regime_df is None:
        logger.error("  Phase 2 failed – no soft probs. Aborting.")
        return None

    # ── Phase 3: build 12-dim augmented state ────────────────────────────────
    logger.info("\n[Phase 3] Building augmented 12-dim state tensors...")
    feature_dict = _load_rad_data_for_a2c(tickers, data_start, test_end)

    train_state_dict, test_state_dict = _build_augmented_state_dict(
        feature_dict=feature_dict,
        regime_df=full_regime_df,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )

    logger.info(
        f"  Train state: {len(train_state_dict)} tickers | "
        f"Test state: {len(test_state_dict)} tickers"
    )

    if not train_state_dict:
        logger.error("  No training states. Aborting.")
        return None

    # ── Phase 4: train A2C (n_features=12) ───────────────────────────────────
    logger.info("\n[Phase 4] Training A2C (n_features=12)...")
    train_envs = build_envs_from_state_dict(
        state_dict=train_state_dict,
        tickers=list(train_state_dict.keys()),
        sigma_target=sigma_target,
    )
    logger.info(f"  Training envs: {len(train_envs)}")

    checkpoint_name = (
        f"a2c_rb_{asset_class.replace(' ', '_')}_{period_name}.pt"
    )
    checkpoint_path = output_dir / checkpoint_name

    trainer = PaperA2CTrainer(n_features=12, device=DEVICE)
    trainer.fit(
        envs=train_envs,
        n_updates=n_updates,
        rollout_steps=n_updates_rollout,
        log_every=100,
        checkpoint_path=str(checkpoint_path),
        checkpoint_every=200,
    )
    trainer.save_checkpoint(str(checkpoint_path))
    logger.info(f"  Checkpoint saved: {checkpoint_path}")

    # ── Phase 5: evaluate on test period ─────────────────────────────────────
    logger.info("\n[Phase 5] Evaluating on test period...")
    results = _evaluate(
        actor=trainer.actor,
        test_state_dict=test_state_dict,
        sigma_target=sigma_target,
    )

    logger.info(f"\n  Sharpe  (equal-weight portfolio): {results['portfolio_sharpe']:.4f}")
    logger.info(f"  Calmar  : {results['portfolio_calmar']:.4f}")
    logger.info(f"  Ann.Ret : {results['portfolio_ann_ret']:.4f}")
    logger.info(f"  Ann.Std : {results['portfolio_ann_std']:.4f}")

    return {
        "asset_class": asset_class,
        "period": period_name,
        "train_regime_result": train_regime_result,
        "checkpoint": str(checkpoint_path),
        "metrics": results,
    }


def _evaluate(actor, test_state_dict: Dict, sigma_target: float) -> Dict:
    """
    Run deterministic inference on test environments and compute portfolio metrics.
    """
    import torch

    actor.eval()
    daily_pnl_by_ticker: Dict[str, pd.Series] = {}

    for ticker, sd in test_state_dict.items():
        X        = sd["X"]          # (n, 60, 12)
        al_df    = sd["aligned_df"] # date, close, ret_1d, ewm_vol_60
        al_df    = al_df.reset_index(drop=True)

        if len(X) == 0:
            continue

        pnl_rows = []
        prev_action = 0.0

        for i in range(len(X) - 1):
            row_t   = al_df.iloc[i]
            row_tp1 = al_df.iloc[i + 1]

            sigma_t = row_t["ewm_vol_60"]
            if pd.isna(sigma_t) or sigma_t == 0:
                pnl_rows.append((row_tp1["date"], np.nan))
                continue

            state = torch.tensor(X[i:i+1], dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                action = actor.deterministic_action(state).item()

            action = float(np.clip(action, -1.0, 1.0))
            scaled_pos = sigma_target / sigma_t * action

            prev_scaled = 0.0
            if i > 0:
                prev_sigma = al_df.iloc[i - 1]["ewm_vol_60"]
                if not pd.isna(prev_sigma) and prev_sigma > 0:
                    prev_scaled = sigma_target / prev_sigma * prev_action

            r_tp1   = row_tp1["ret_1d"]
            price_t = row_t["close"]

            if pd.isna(r_tp1) or pd.isna(price_t):
                pnl_rows.append((row_tp1["date"], np.nan))
            else:
                from drl_models.a2c.a2c_model import BP, MU
                gross = MU * scaled_pos * r_tp1
                cost  = MU * BP * price_t * abs(scaled_pos - prev_scaled)
                pnl_rows.append((row_tp1["date"], gross - cost))

            prev_action = action

        if pnl_rows:
            dates, pnls = zip(*pnl_rows)
            daily_pnl_by_ticker[ticker] = pd.Series(
                pnls, index=pd.DatetimeIndex(dates), name=ticker
            )

    if not daily_pnl_by_ticker:
        return {}

    pnl_df = pd.DataFrame(daily_pnl_by_ticker).dropna(how="all")
    portfolio = pnl_df.mean(axis=1).dropna()

    ann = np.sqrt(252)
    mean_ret = portfolio.mean() * 252
    std_ret  = portfolio.std() * ann
    sharpe   = mean_ret / std_ret if std_ret > 0 else 0.0

    cum = portfolio.cumsum()
    rolling_max = cum.cummax()
    drawdown    = cum - rolling_max
    mdd         = drawdown.min()
    calmar      = mean_ret / abs(mdd) if mdd < 0 else 0.0

    return {
        "portfolio_ann_ret": mean_ret,
        "portfolio_ann_std": std_ret,
        "portfolio_sharpe":  sharpe,
        "portfolio_calmar":  calmar,
        "portfolio_mdd":     mdd,
        "n_tickers":         len(daily_pnl_by_ticker),
        "portfolio_pnl":     portfolio,   # pd.Series(daily net PnL, DatetimeIndex)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Train Route B for all asset classes and both periods.
    Edit N_UPDATES at the top of the file to control training length.
    """
    # Build "All" ticker list
    all_tickers = []
    for v in ASSET_CLASSES.values():
        all_tickers.extend(v)

    run_targets = [
        # (asset_class_label, tickers_list)
        ("Commodity",    ASSET_CLASSES["Commodity"]),
        ("Equity Index", ASSET_CLASSES["Equity Index"]),
        ("Fixed Income", ASSET_CLASSES["Fixed Income"]),
        ("Forex",        ASSET_CLASSES["Forex"]),
        ("All",          all_tickers),
    ]

    all_results = {}

    for period_name in ["period_1", "period_2"]:
        all_results[period_name] = {}
        for asset_class, tickers in run_targets:
            result = run_route_b(
                asset_class=asset_class,
                tickers=tickers,
                period_name=period_name,
                n_updates=N_UPDATES,
            )
            if result is not None:
                all_results[period_name][asset_class] = result

    # Summary table
    print("\n" + "=" * 70)
    print("Route B – Summary")
    print("=" * 70)
    print(f"{'Period':<12} {'Asset Class':<16} {'Sharpe':>8} {'Calmar':>8} {'Ann.Ret':>9}")
    print("-" * 60)
    for period_name, ac_dict in all_results.items():
        for asset_class, res in ac_dict.items():
            m = res["metrics"]
            print(
                f"{period_name:<12} {asset_class:<16} "
                f"{m.get('portfolio_sharpe', float('nan')):>8.3f} "
                f"{m.get('portfolio_calmar', float('nan')):>8.3f} "
                f"{m.get('portfolio_ann_ret', float('nan')):>9.4f}"
            )

    return all_results


if __name__ == "__main__":
    main()
