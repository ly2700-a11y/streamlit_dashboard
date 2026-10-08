"""
data_loader.py — CLC data loading utilities
"""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


CSV_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI']
V2_CONTRACTS = ['ZH', 'ZU', 'US', 'ZN']
PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_data_dir(data_dir):
    path = Path(data_dir)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


@lru_cache(maxsize=None)
def _read_clc_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path, header=None, names=CSV_COLUMNS)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    return df.sort_values('Date').reset_index(drop=True)


def _clean_price_frame(df, start_date='2009-01-01'):
    if df is None:
        return None
    out = df[df['Close'].notna() & np.isfinite(df['Close']) & (df['Close'] > 0)].copy()
    if start_date:
        out = out[out['Date'] >= start_date]
    out = out.sort_values('Date').reset_index(drop=True)
    if len(out) < 500:
        return None
    return out


@lru_cache(maxsize=None)
def _generate_non_fwd_anchored(ticker, data_dir='data/CLC', anchor_date='2011-01-01'):
    """Generate an additive forward-adjusted NON path anchored at the test start.

    Construction:
      NON_FWD_ANCHORED[t] = NON[t] + (REV[t] - NON[t]) - (REV[t0] - NON[t0])
                          = REV[t] - adj[t0]

    where t0 is the first date >= anchor_date in the merged NON∩REV window.

    Properties:
      - at the anchor date, the series equals NON exactly
      - additive day-to-day moves match REV day-to-day moves
      - adjustment changes propagate forward, not backward
      - prices stay on a more realistic level near the test start than raw REV
    """
    data_dir = _resolve_data_dir(data_dir)
    non = _read_clc_csv(data_dir / f'{ticker}_NON.CSV')
    rev = _read_clc_csv(data_dir / f'{ticker}_REV.CSV')
    if non is None or rev is None:
        return None

    merged = non[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI']].merge(
        rev[['Date', 'Close']],
        on='Date',
        how='inner',
        suffixes=('', '_rev'),
    )
    merged = merged.sort_values('Date').reset_index(drop=True)

    p_non = pd.to_numeric(merged['Close'], errors='coerce').values
    p_rev = pd.to_numeric(merged['Close_rev'], errors='coerce').values

    valid = np.isfinite(p_non) & np.isfinite(p_rev) & (p_non > 0)
    merged = merged[valid].reset_index(drop=True)
    p_non = p_non[valid]
    p_rev = p_rev[valid]
    if len(merged) == 0:
        return None

    anchor_mask = merged['Date'] >= pd.Timestamp(anchor_date)
    if not anchor_mask.any():
        return None
    anchor_idx = int(anchor_mask.idxmax())

    adj = p_rev - p_non
    anchor_adj = adj[anchor_idx]
    out = merged[['Date', 'Open', 'High', 'Low', 'Volume', 'OI']].copy()
    out['Close'] = p_non + adj - anchor_adj
    out = out[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI']]
    return out


@lru_cache(maxsize=None)
def _generate_rad_regen(ticker, data_dir='data/CLC'):
    """Regenerate ratio-adjusted close series from NON + REV adjustment shifts."""
    data_dir = _resolve_data_dir(data_dir)
    non = _read_clc_csv(data_dir / f'{ticker}_NON.CSV')
    rev = _read_clc_csv(data_dir / f'{ticker}_REV.CSV')
    if non is None or rev is None:
        return None

    merged = non[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI']].merge(
        rev[['Date', 'Close']],
        on='Date',
        how='inner',
        suffixes=('', '_rev'),
    )
    merged = merged.sort_values('Date').reset_index(drop=True)
    p_non = pd.to_numeric(merged['Close'], errors='coerce').values
    p_rev = pd.to_numeric(merged['Close_rev'], errors='coerce').values
    valid = np.isfinite(p_non) & np.isfinite(p_rev) & (p_non > 0)
    merged = merged[valid].reset_index(drop=True)
    p_non = p_non[valid]
    p_rev = p_rev[valid]

    adj = p_rev - p_non
    adj_diff = np.diff(adj)
    roll_idx = np.where(np.abs(adj_diff) > 1e-6)[0]
    cum_ratio = np.ones(len(p_non))
    for idx in roll_idx:
        new_price = p_non[idx + 1]
        if abs(new_price) > 1e-12:
            ratio = p_non[idx] / new_price
            cum_ratio[idx + 1:] *= ratio

    out = merged[['Date', 'Open', 'High', 'Low', 'Volume', 'OI']].copy()
    out['Close'] = p_non * cum_ratio
    out = out[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI']]
    return out


@lru_cache(maxsize=None)
def load_clc_full(ticker, data_dir='data/CLC', start_date='2009-01-01', source='RAD', anchor_date='2011-01-01'):
    """
    Load CLC price data from start_date onwards.

    Supported sources:
      - RAD: current repo baseline, with RAD_v2 fallback for known damaged files
      - REV: vendor back-adjusted continuous series
      - NON: vendor non-adjusted continuous series
      - RAD_REGEN: regenerate ratio-adjusted series from NON + REV adjustment shifts
      - NON_FWD_ANCHORED: additive forward-adjusted NON anchored to NON at anchor_date

    Default 2009-01-01 gives enough warmup before the 2011 test window.
    """
    data_dir = _resolve_data_dir(data_dir)
    source = source.upper()
    if source == 'RAD':
        if ticker in V2_CONTRACTS:
            path = data_dir / f'{ticker}_RAD_v2.CSV'
        else:
            path = data_dir / f'{ticker}_RAD.CSV'
        df = _read_clc_csv(path)
    elif source == 'REV':
        df = _read_clc_csv(data_dir / f'{ticker}_REV.CSV')
    elif source == 'NON':
        df = _read_clc_csv(data_dir / f'{ticker}_NON.CSV')
    elif source == 'RAD_REGEN':
        df = _generate_rad_regen(ticker, data_dir=data_dir)
    elif source == 'NON_FWD_ANCHORED':
        df = _generate_non_fwd_anchored(ticker, data_dir=data_dir, anchor_date=anchor_date)
    else:
        raise ValueError(f'Unknown source: {source}')

    return _clean_price_frame(df, start_date=start_date)


def get_price_diffs(prices):
    """
    Additive profits: r_t = p_t - p_{t-1}
    Paper Section 3.2: "additive profits of holding a single contract"
    Used for: trade return computation (Formula 4)
    """
    diff = np.zeros(len(prices))
    diff[1:] = prices[1:] - prices[:-1]
    return diff


def get_pct_returns(prices):
    """
    Percentage returns: r_t = p_t / p_{t-1} - 1
    Used for: signal computation (Sign(R) cumulative return)
    """
    ret = np.zeros(len(prices))
    ret[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
    return ret


def extract_test_period(df, test_start='2011-01-01', test_end='2019-12-31'):
    """
    Get index boundaries and dates for the test period.
    Returns (start_idx, end_idx, date_series) or (None, None, None).
    """
    mask_start = df['Date'] >= test_start
    mask_end = df['Date'] <= test_end
    if not mask_start.any() or not mask_end.any():
        return None, None, None
    t0 = mask_start.idxmax()
    t1 = len(df) - 1 - mask_end[::-1].values.argmax()
    return t0, t1, df['Date']
