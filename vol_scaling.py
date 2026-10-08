"""
vol_scaling.py — Per-contract and portfolio-level volatility scaling
Reference: Zhang et al. 2019 Eq 4, 13; [17] Harvey; [27] Lim; [37] Moskowitz

Framework: additive profits r_t = p_t - p_{t-1}
σ_t = EWMA(60) std of daily price diffs (same units as r_t)
σ_tgt = 0.10 (in price-diff units, constant across all contracts)
σ_tgt/σ_t is dimensionless → normalises each contract to same daily vol

The specific value of σ_tgt doesn't affect Sharpe, Sortino, % +ve, Ave P/L.
It only scales E(R), std(R), DD, MDD proportionally.
"""
import numpy as np
import pandas as pd
from config import EWMA_SPAN, PORT_TGT_STD, TRADING_DAYS


def compute_ewma_vol(values, span=EWMA_SPAN):
    """
    σ_t = EWMA(span) std of daily values.
    Works with additive price diffs (units: price/day).
    Returns array of same length. NaN/zero filled with first valid value.
    """
    vol = pd.Series(values).ewm(span=span, adjust=False).std().values
    first_valid = None
    for v in vol:
        if not np.isnan(v) and v > 0:
            first_valid = v
            break
    if first_valid is None:
        first_valid = 1e-6
    vol = np.nan_to_num(vol, nan=first_valid, posinf=first_valid, neginf=first_valid)
    vol[vol == 0] = first_valid
    return vol


def scale_per_contract(price_diffs, sigma_tgt_daily, span=EWMA_SPAN, max_leverage=None):
    """
    Per-contract volatility scaling (Formula 4 inner scaling).

    For ADDITIVE price diffs:
      σ_t = EWMA(60) std of (p_t - p_{t-1})  [price units/day]
      scaling = σ_tgt_daily / σ_t             [dimensionless]

    After scaling, each contract's daily vol ≈ σ_tgt_daily.
    σ_tgt_daily is the same for all contracts → normalises to same scale.

    Args:
        price_diffs:     daily price differences r_t = p_t - p_{t-1}
        sigma_tgt_daily: target daily volatility (same units as r_t)
        span:            EWMA span
        max_leverage:    optional cap on |scaling| to prevent extreme leverage

    Returns:
        scaling factor array (same length as input)
    """
    vol = compute_ewma_vol(price_diffs, span)
    scaling = sigma_tgt_daily / vol
    if max_leverage is not None:
        scaling = np.clip(scaling, -max_leverage, max_leverage)
    return scaling


def scale_portfolio(returns, target_std=PORT_TGT_STD):
    """
    Portfolio-level volatility scaling (Eq 13).
    Scale portfolio returns so annualised std = target_std.
    """
    current_std = np.std(returns) * np.sqrt(TRADING_DAYS)
    if current_std > 0:
        return returns * (target_std / current_std)
    return returns
