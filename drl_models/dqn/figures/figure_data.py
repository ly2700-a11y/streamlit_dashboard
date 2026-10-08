"""Shared data loading helpers for DQN figures."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vol_scaling import get_portfolio_bridge

ASSET_PATH_MAP = {
    'Commodity': 'Commodity',
    'Equity Index': 'Equity_Index',
    'Fixed Income': 'Fixed_Income',
    'Forex': 'Forex',
}


def get_ensemble_npz_path(asset_name, bp=None):
    """Return path to top5_ensemble_R.npz for an asset, optionally BP-filtered."""
    slug = ASSET_PATH_MAP[asset_name]
    if bp is not None:
        bp_label = f"bp{int(bp * 10000)}"
        return Path(f"drl/dqn/reports/ensemble_table2_bp/{slug}/{bp_label}/top5_ensemble_R.npz")
    return Path(f"drl/dqn/reports/ensemble_table2/{slug}/top5_ensemble_R.npz")


def sorted_return_series(dates, returns) -> pd.Series:
    """Build a dated return series and enforce chronological order."""
    values = np.asarray(returns, dtype=float)
    index = pd.to_datetime(dates)
    if len(index) != len(values):
        raise ValueError(f"dates/returns length mismatch: {len(index)} != {len(values)}")

    series = pd.Series(values, index=index).sort_index()
    if series.index.has_duplicates:
        series = series.groupby(level=0).mean()
    return series


def scale_return_series(series: pd.Series, port_vol_target: float) -> pd.Series:
    """Apply the same post-hoc portfolio volatility bridge while preserving dates."""
    scaler = get_portfolio_bridge("constant_posthoc", port_vol_target)
    scaled = scaler(series.to_numpy(dtype=float))
    return pd.Series(scaled, index=series.index)


def load_scaled_ensemble_series(npz_path: Path, port_vol_target: float) -> pd.Series:
    """Load saved DQN ensemble returns as a sorted, scaled dated Series."""
    if not npz_path.exists():
        raise ValueError(f"DQN ensemble data not found at {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    series = sorted_return_series(data["dates"], data["portfolio_returns"])
    return scale_return_series(series, port_vol_target)
