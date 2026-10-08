"""
repro_analysis.py

Shared analysis utilities for paper-vs-ours reproduction reporting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from baseline_run import (
    DEFAULT_SIGMA_TGT,
    apply_portfolio_vol_scaling,
    compute_contract_returns,
    load_contracts,
)
from config import BP, SIGN_LOOKBACK
from config import (
    ASSET_CLASSES,
    EXCLUDED_CONTRACTS,
    PAPER_TABLE2,
    PAPER_TABLE3,
    PORT_TGT_STD,
    SOURCE_OVERRIDES,
    TRADING_DAYS,
)


ASSETS = ['Commodity', 'Equity Index', 'Fixed Income', 'Forex', 'All']
ALL_METRICS = ['E(R)', 'std(R)', 'DD', 'Sharpe', 'Sortino',
               'MDD', 'Calmar', '% +ve', 'Ave P/L']
LANE_A_METRICS = ['E(R)', 'Sharpe', 'DD', 'Sortino']
LANE_B_METRICS = ['std(R)', '% +ve', 'Ave P/L']
LANE_C_METRICS = ['MDD', 'Calmar']

EXCLUSION_PRESETS = {
    'none': [],
    'memory_5': ['LB', 'JO', 'ZO', 'CC', 'FB'],
    'memory_5_plus_us': ['LB', 'JO', 'ZO', 'CC', 'FB', 'US'],
    'memory_5_plus_us_zh': ['LB', 'JO', 'ZO', 'CC', 'FB', 'US', 'ZH'],
    'current_config': list(EXCLUDED_CONTRACTS),
}


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    series_kind: str  # additive_portfolio or nav_portfolio
    dd_mode: str      # subset or full
    mdd_mode: str     # additive or nav


METRIC_DEFINITIONS = {
    'additive_subset': MetricDefinition(
        name='additive_subset',
        series_kind='additive_portfolio',
        dd_mode='subset',
        mdd_mode='additive',
    ),
    'nav_subset': MetricDefinition(
        name='nav_subset',
        series_kind='nav_portfolio',
        dd_mode='subset',
        mdd_mode='nav',
    ),
    'nav_full': MetricDefinition(
        name='nav_full',
        series_kind='nav_portfolio',
        dd_mode='full',
        mdd_mode='nav',
    ),
}


def pct_err(ours: float, paper: float) -> float:
    if abs(paper) < 1e-12:
        return math.inf if abs(ours) > 1e-12 else 0.0
    return abs((ours - paper) / abs(paper)) * 100.0


def abs_gap(ours: float, paper: float) -> float:
    return abs(ours - paper)


def load_asset_contracts(asset: str, test_start='2011-01-01', test_end='2019-12-31',
                         excluded_contracts=None, source_overrides=None):
    if excluded_contracts is None:
        excluded_contracts = EXCLUDED_CONTRACTS
    if source_overrides is None:
        source_overrides = SOURCE_OVERRIDES
    if asset == 'All':
        raw = []
        for name in ['Commodity', 'Equity Index', 'Fixed Income', 'Forex']:
            raw.extend(load_contracts(
                name,
                test_start=test_start,
                test_end=test_end,
                excluded_contracts=excluded_contracts,
                source_overrides=source_overrides,
            ))
        return raw
    return load_contracts(
        asset,
        test_start=test_start,
        test_end=test_end,
        excluded_contracts=excluded_contracts,
        source_overrides=source_overrides,
    )


def contract_additive_series(rd, strat='Long', sigma_tgt=DEFAULT_SIGMA_TGT):
    rt = compute_contract_returns(rd, strat, sigma_tgt)
    start, t1, dates = rd['start'], rd['t1'], rd['dates']
    slc = rt[start:t1]
    return pd.Series(slc[:len(dates)], index=dates[:len(slc)])


def contract_additive_components(rd, strat='Long', sigma_tgt=DEFAULT_SIGMA_TGT):
    """Return aligned additive trade, signal, and tc series for one contract."""
    rt, sigma, prices = rd['rt'], rd['sigma'], rd['prices']
    n = len(rt)

    if strat == 'Long':
        pos = np.ones(n)
    else:
        raise ValueError('Attribution helper currently supports Long only')

    trade = np.zeros(n)
    signal = np.zeros(n)
    tc = np.zeros(n)
    for t in range(1, n):
        if sigma[t - 1] > 0 and (t < 2 or sigma[t - 2] > 0):
            a_prev = pos[t - 1]
            a_prev2 = pos[t - 2] if t >= 2 else 1.0
            sp = a_prev * sigma_tgt / sigma[t - 1]
            spp = a_prev2 * sigma_tgt / sigma[t - 2] if t >= 2 else 0.0
            signal[t] = sp * rt[t]
            tc[t] = BP * prices[t - 1] * abs(sp - spp)
            trade[t] = signal[t] - tc[t]

    start, t1, dates = rd['start'], rd['t1'], rd['dates']
    idx = dates[:len(trade[start:t1])]
    return {
        'trade': pd.Series(trade[start:t1][:len(idx)], index=idx),
        'signal': pd.Series(signal[start:t1][:len(idx)], index=idx),
        'tc': pd.Series(tc[start:t1][:len(idx)], index=idx),
    }


def contract_nav_return_series(rd, strat='Long', sigma_tgt=DEFAULT_SIGMA_TGT):
    rt_add = compute_contract_returns(rd, strat, sigma_tgt)
    prices = rd['prices']
    r_nav = np.zeros(len(rt_add))
    valid = prices[:-1] > 0
    r_nav[1:][valid] = rt_add[1:][valid] / prices[:-1][valid]
    start, t1, dates = rd['start'], rd['t1'], rd['dates']
    slc = r_nav[start:t1]
    return pd.Series(slc[:len(dates)], index=dates[:len(slc)])


def portfolio_series(raw_data, series_kind='additive_portfolio', aggregation_mode='variable_n',
                     strat='Long', sigma_tgt=DEFAULT_SIGMA_TGT):
    series = []
    for rd in raw_data:
        if series_kind == 'additive_portfolio':
            s = contract_additive_series(rd, strat=strat, sigma_tgt=sigma_tgt)
        elif series_kind == 'nav_portfolio':
            s = contract_nav_return_series(rd, strat=strat, sigma_tgt=sigma_tgt)
        else:
            raise ValueError(f'Unknown series_kind: {series_kind}')
        series.append(s)
    df_all = pd.DataFrame(series)
    if aggregation_mode == 'dropna':
        port = df_all.T.dropna().mean(axis=1)
    elif aggregation_mode == 'variable_n':
        port = df_all.T.mean(axis=1)
    else:
        raise ValueError(f'Unknown aggregation_mode: {aggregation_mode}')
    return port


def downside_deviation(series, mode='subset', mar=0.0):
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0

    if mode == 'subset':
        neg = arr[arr < mar]
        return float(np.std(neg, ddof=0) * math.sqrt(TRADING_DAYS)) if len(neg) > 1 else 0.0

    if mode == 'full':
        downside = np.minimum(0.0, arr - mar)
        return float(np.sqrt(np.mean(downside ** 2)) * math.sqrt(TRADING_DAYS))

    raise ValueError(f'Unknown DD mode: {mode}')


def max_drawdown_additive(series, w0=1.0):
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    wealth = w0 + np.cumsum(arr)
    if len(wealth) == 0:
        return 0.0
    peak = np.maximum.accumulate(wealth)
    drawdown = (peak - wealth) / peak
    return float(np.nanmax(drawdown))


def max_drawdown_nav(returns):
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    nav = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(nav)
    drawdown = (peak - nav) / peak
    return float(np.nanmax(drawdown))


def compute_metrics_from_series(series, dd_mode='subset', mdd_mode='additive',
                                n_contracts=None):
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {name: 0.0 for name in ALL_METRICS}

    er = float(np.mean(arr) * TRADING_DAYS)
    vol = float(np.std(arr, ddof=0) * math.sqrt(TRADING_DAYS))
    dd = downside_deviation(arr, mode=dd_mode, mar=0.0)
    sharpe = er / vol if vol > 0 else 0.0
    sortino = er / dd if dd > 0 else 0.0

    pos = arr[arr > 0]
    neg = arr[arr < 0]
    pct_pos = len(pos) / len(arr) if len(arr) else 0.0
    avg_pl = (pos.mean() / abs(neg.mean())) if len(pos) > 0 and len(neg) > 0 else 0.0

    if mdd_mode == 'additive':
        if n_contracts is None:
            w0 = 1.0
        else:
            w0 = float(n_contracts)
        mdd = max_drawdown_additive(arr, w0=w0)
    elif mdd_mode == 'nav':
        mdd = max_drawdown_nav(arr)
    else:
        raise ValueError(f'Unknown MDD mode: {mdd_mode}')
    calmar = er / mdd if mdd > 0 else 0.0

    return {
        'E(R)': round(er, 3),
        'std(R)': round(vol, 3),
        'DD': round(dd, 3),
        'Sharpe': round(sharpe, 3),
        'Sortino': round(sortino, 3),
        'MDD': round(mdd, 3),
        'Calmar': round(calmar, 3),
        '% +ve': round(pct_pos, 3),
        'Ave P/L': round(avg_pl, 3),
    }


def evaluate_table(asset, paper_table, metric_def: MetricDefinition,
                   excluded_contracts=None, sigma_tgt=DEFAULT_SIGMA_TGT,
                   aggregation_mode='variable_n', port_scaler=None,
                   source_overrides=None,
                   test_start='2011-01-01', test_end='2019-12-31'):
    raw = load_asset_contracts(
        asset,
        test_start=test_start,
        test_end=test_end,
        excluded_contracts=excluded_contracts,
        source_overrides=source_overrides,
    )
    if not raw:
        return None

    port = portfolio_series(
        raw,
        series_kind=metric_def.series_kind,
        aggregation_mode=aggregation_mode,
        sigma_tgt=sigma_tgt,
    )
    if port_scaler is not None:
        port = pd.Series(port_scaler(port.values), index=port.index)

    metrics = compute_metrics_from_series(
        port.values,
        dd_mode=metric_def.dd_mode,
        mdd_mode=metric_def.mdd_mode,
        n_contracts=len(raw),
    )
    paper = paper_table[asset]['Long']
    percent_errors = {name: pct_err(metrics[name], paper[name]) for name in ALL_METRICS}
    absolute_gaps = {name: abs_gap(metrics[name], paper[name]) for name in ALL_METRICS}
    calmar_internal_gap = abs(metrics['Calmar'] - (metrics['E(R)'] / metrics['MDD'])) if metrics['MDD'] else math.inf
    paper_calmar_internal_gap = abs(paper['Calmar'] - (paper['E(R)'] / paper['MDD'])) if paper['MDD'] else math.inf

    return {
        'contracts': len(raw),
        'series': port,
        'metrics': metrics,
        'paper': paper,
        'percent_errors': percent_errors,
        'absolute_gaps': absolute_gaps,
        'calmar_internal_gap': calmar_internal_gap,
        'paper_calmar_internal_gap': paper_calmar_internal_gap,
    }


def score_table3_scenario(asset_results):
    lane_a_pass_10 = 0
    lane_a_pass_15 = 0
    lane_b_pass_10 = 0
    lane_b_pass_15 = 0
    lane_a_errs = []
    lane_b_errs = []
    all_abs = {}

    for asset, result in asset_results.items():
        errs = result['percent_errors']
        if asset != 'All':
            for name in LANE_A_METRICS:
                lane_a_errs.append(errs[name])
                lane_a_pass_10 += int(errs[name] < 10)
                lane_a_pass_15 += int(errs[name] < 15)
            for name in LANE_B_METRICS:
                lane_b_errs.append(errs[name])
                lane_b_pass_10 += int(errs[name] < 10)
                lane_b_pass_15 += int(errs[name] < 15)
        else:
            all_abs['E(R)'] = result['absolute_gaps']['E(R)']
            all_abs['Sharpe'] = result['absolute_gaps']['Sharpe']

    return {
        'lane_a_pass_10': lane_a_pass_10,
        'lane_a_pass_15': lane_a_pass_15,
        'lane_b_pass_10': lane_b_pass_10,
        'lane_b_pass_15': lane_b_pass_15,
        'lane_a_mae': float(np.mean(lane_a_errs)) if lane_a_errs else math.inf,
        'lane_a_median': float(np.median(lane_a_errs)) if lane_a_errs else math.inf,
        'lane_b_mae': float(np.mean(lane_b_errs)) if lane_b_errs else math.inf,
        'all_abs_er': all_abs.get('E(R)', math.inf),
        'all_abs_sharpe': all_abs.get('Sharpe', math.inf),
    }


def table3_sort_key(score):
    return (
        -score['lane_a_pass_15'],
        -score['lane_a_pass_10'],
        -score['lane_b_pass_15'],
        -score['lane_b_pass_10'],
        score['lane_a_mae'],
        score['lane_a_median'],
        score['all_abs_er'],
        score['all_abs_sharpe'],
    )


def constant_posthoc_scaler(target_std=PORT_TGT_STD):
    def scale(values):
        return apply_portfolio_vol_scaling(values, target_std)
    return scale


def ewma_portfolio_scaler(target_std=PORT_TGT_STD, span=60):
    target_daily = target_std / math.sqrt(TRADING_DAYS)

    def scale(values):
        ser = pd.Series(values)
        vol = ser.ewm(span=span, adjust=False).std().shift(1)
        first_valid = vol[vol > 0].iloc[0] if (vol > 0).any() else target_daily
        vol = vol.fillna(first_valid)
        vol = vol.replace(0, first_valid)
        k = target_daily / vol
        return (ser * k).values
    return scale


def rolling_portfolio_scaler(target_std=PORT_TGT_STD, window=252):
    target_daily = target_std / math.sqrt(TRADING_DAYS)

    def scale(values):
        ser = pd.Series(values)
        vol = ser.rolling(window=window, min_periods=20).std().shift(1)
        if (vol > 0).any():
            first_valid = vol[vol > 0].iloc[0]
        else:
            first_valid = target_daily
        vol = vol.fillna(first_valid)
        vol = vol.replace(0, first_valid)
        k = target_daily / vol
        return (ser * k).values
    return scale


TABLE2_BRIDGES = {
    'constant_posthoc': constant_posthoc_scaler(),
    'ewma60_lagged': ewma_portfolio_scaler(span=60),
    'rolling252_lagged': rolling_portfolio_scaler(window=252),
}


def yearly_summary(series):
    if len(series) == 0:
        return {}
    ser = pd.Series(series.values, index=pd.to_datetime(series.index))
    grouped = ser.groupby(ser.index.year)
    return {int(year): round(vals.sum(), 4) for year, vals in grouped}


def realized_er_contributions(raw_data, sigma_tgt=DEFAULT_SIGMA_TGT, strat='Long'):
    """Decompose portfolio E(R) into per-contract realized contributions.

    For variable-N aggregation:
      R_port,t = (1 / N_t) * Σ_i R_i,t
    so:
      E(R_port) = 252 * mean_t[(1 / N_t) * Σ_i R_i,t]

    This function returns annualized realized contributions for trade, signal,
    and transaction-cost drag using that exact identity.
    """
    trade_series = {}
    signal_series = {}
    tc_series = {}
    for rd in raw_data:
        comps = contract_additive_components(rd, strat=strat, sigma_tgt=sigma_tgt)
        tk = rd['tk']
        trade_series[tk] = comps['trade']
        signal_series[tk] = comps['signal']
        tc_series[tk] = comps['tc']

    trade_df = pd.DataFrame(trade_series)
    signal_df = pd.DataFrame(signal_series)
    tc_df = pd.DataFrame(tc_series)
    availability = trade_df.notna().sum(axis=1).replace(0, np.nan)

    rows = []
    for tk in trade_df.columns:
        trade_contrib = (trade_df[tk] / availability).fillna(0.0)
        signal_contrib = (signal_df[tk] / availability).fillna(0.0)
        tc_contrib = (tc_df[tk] / availability).fillna(0.0)
        rows.append({
            'ticker': tk,
            'er_contrib': float(trade_contrib.mean() * TRADING_DAYS),
            'signal_contrib': float(signal_contrib.mean() * TRADING_DAYS),
            'tc_contrib': float(tc_contrib.mean() * TRADING_DAYS),
            'n_obs': int(trade_df[tk].notna().sum()),
        })
    return rows
