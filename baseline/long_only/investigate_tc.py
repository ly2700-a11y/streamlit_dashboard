#!/usr/bin/env python3
"""
investigate_tc.py — Investigate why Fixed Income has high TC

Diagnoses:
1. σ_t distribution per contract (why UB/FB/DT have high TC)
2. DA price level issue
3. Compare RAD_v2 vs vendor RAD where applicable

Usage:
  cd IEOR4733_Project && python tests/investigate_tc.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / 'data' / 'CLC'

from data_loader import load_clc_full
from config import ASSET_CLASSES, BP, TRADING_DAYS, EXCLUDED_CONTRACTS

TEST_START = '2011-01-01'
TEST_END = '2019-12-31'
SIGMA_TGT = 0.063
EWMA_SPAN = 60


def diagnose_contract(ticker):
    """Deep diagnosis of one contract's σ_t and TC behavior."""
    rad_df = load_clc_full(ticker)
    if rad_df is None:
        return None
    
    prices = rad_df['Close'].values.astype(float)
    rt = np.zeros(len(prices))
    rt[1:] = prices[1:] - prices[:-1]
    
    # σ_t with EWMA(60, adjust=False)
    sigma = pd.Series(rt).ewm(span=EWMA_SPAN, adjust=False).std().values
    
    # Test period
    test_mask = (rad_df['Date'] >= TEST_START) & (rad_df['Date'] <= TEST_END)
    t0 = test_mask.idxmax()
    t1 = len(rad_df) - 1 - test_mask[::-1].values.argmax()
    
    test_prices = prices[t0:t1]
    test_rt = rt[t0:t1]
    test_sigma = sigma[t0:t1]
    
    # Compute daily TC = bp * price * |sp_change|
    daily_tc = np.zeros(t1 - t0)
    daily_ret = np.zeros(t1 - t0)
    sp_arr = np.zeros(t1 - t0)
    
    for i in range(t1 - t0):
        t = t0 + i
        if t >= 2 and sigma[t-1] > 0 and sigma[t-2] > 0:
            sp = SIGMA_TGT / sigma[t-1]
            spp = SIGMA_TGT / sigma[t-2]
            daily_tc[i] = BP * prices[t-1] * abs(sp - spp)
            daily_ret[i] = sp * rt[t]
            sp_arr[i] = sp
        elif t >= 1 and sigma[t-1] > 0:
            sp = SIGMA_TGT / sigma[t-1]
            daily_ret[i] = sp * rt[t]
            sp_arr[i] = sp
    
    # Filter out zeros (warmup)
    valid = sp_arr > 0
    daily_tc = daily_tc[valid]
    daily_ret = daily_ret[valid]
    sp_arr = sp_arr[valid]
    test_sigma_v = test_sigma[valid]
    test_rt_v = test_rt[valid]
    test_prices_v = test_prices[valid]
    
    if len(daily_tc) == 0:
        return None
    
    return {
        'ticker': ticker,
        'n_valid': len(daily_tc),
        # Price stats
        'price_mean': np.mean(test_prices_v),
        'price_min': np.min(test_prices_v),
        'price_max': np.max(test_prices_v),
        # σ_t stats
        'sigma_mean': np.mean(test_sigma_v),
        'sigma_median': np.median(test_sigma_v),
        'sigma_min': np.min(test_sigma_v[test_sigma_v > 0]),
        'sigma_max': np.max(test_sigma_v),
        'sigma_p10': np.percentile(test_sigma_v[test_sigma_v > 0], 10),
        'sigma_p90': np.percentile(test_sigma_v[test_sigma_v > 0], 90),
        # Position scaling (σ_tgt/σ)
        'sp_mean': np.mean(sp_arr),
        'sp_median': np.median(sp_arr),
        'sp_max': np.max(sp_arr),
        'sp_min': np.min(sp_arr),
        # Daily TC
        'tc_mean': np.mean(daily_tc) * TRADING_DAYS,
        'tc_median': np.median(daily_tc) * TRADING_DAYS,
        'tc_max': np.max(daily_tc),
        'tc_sum': np.sum(daily_tc),
        # Daily return (scaled)
        'ret_mean': np.mean(daily_ret) * TRADING_DAYS,
        # E(R) = ret - tc
        'er': np.mean(daily_ret - daily_tc) * TRADING_DAYS,
        # σ of σ_t changes (driver of TC)
        'sigma_diff_std': np.std(np.diff(test_sigma_v)),
    }


def main():
    # All non-excluded tickers
    all_tickers = []
    for ac, tickers in ASSET_CLASSES.items():
        for tk in tickers:
            if tk not in EXCLUDED_CONTRACTS:
                all_tickers.append(tk)
    
    print('=' * 140)
    print('TC & σ_t DIAGNOSTIC')
    print(f'σ_tgt={SIGMA_TGT}, bp={BP}, EWMA span={EWMA_SPAN}')
    print('=' * 140)
    
    results = []
    for tk in all_tickers:
        r = diagnose_contract(tk)
        if r:
            results.append(r)
    
    # Sort by TC descending
    results.sort(key=lambda x: x['tc_mean'], reverse=True)
    
    print(f"\n{'TK':>4} | {'Price':>8} {'σ_mean':>8} {'σ_min':>8} {'σ_max':>8} {'σ_range':>8} | {'sp_mean':>7} {'sp_max':>7} | {'Ret':>7} {'TC':>7} {'E(R)':>7} | {'TC/Ret':>7} | Notes")
    print('-' * 140)
    
    for r in results:
        ac = 'Unknown'
        for name, tickers in ASSET_CLASSES.items():
            if r['ticker'] in tickers:
                ac = name
                break
        
        sigma_range = r['sigma_max'] / r['sigma_min'] if r['sigma_min'] > 0 else float('inf')
        tc_ret_ratio = r['tc_mean'] / abs(r['ret_mean']) if abs(r['ret_mean']) > 0.001 else float('inf')
        
        notes = []
        if r['sp_max'] > 5:
            notes.append(f'HIGH_LEV(sp_max={r["sp_max"]:.1f})')
        if sigma_range > 10:
            notes.append(f'VOL_RANGE({sigma_range:.0f}×)')
        if tc_ret_ratio > 0.5:
            notes.append(f'TC_DOMINANT({tc_ret_ratio:.0%})')
        if r['price_mean'] > 100:
            notes.append('HIGH_PRICE')
        if r['price_mean'] < 1:
            notes.append('LOW_PRICE')
            
        note_str = ' '.join(notes) if notes else ''
        
        print(f"{r['ticker']:>4} | {r['price_mean']:>8.2f} {r['sigma_mean']:>8.4f} {r['sigma_min']:>8.4f} {r['sigma_max']:>8.4f} {sigma_range:>8.1f} | {r['sp_mean']:>7.2f} {r['sp_max']:>7.2f} | {r['ret_mean']:>+7.3f} {r['tc_mean']:>7.3f} {r['er']:>+7.3f} | {tc_ret_ratio:>7.1%} | {note_str}")
    
    # Detailed analysis for top TC contracts
    print(f"\n{'=' * 140}")
    print('DETAILED: Top TC contracts')
    print(f"{'=' * 140}")
    
    for r in results[:8]:
        print(f"\n  {r['ticker']}:")
        print(f"    Price: mean={r['price_mean']:.2f}, range=[{r['price_min']:.2f}, {r['price_max']:.2f}]")
        print(f"    σ_t:   mean={r['sigma_mean']:.4f}, range=[{r['sigma_min']:.4f}, {r['sigma_max']:.4f}]")
        print(f"    Scale: mean={r['sp_mean']:.2f}, max={r['sp_max']:.2f}")
        print(f"    Daily: ret={r['ret_mean']:+.3f}/yr, TC={r['tc_mean']:.3f}/yr, E(R)={r['er']:+.3f}/yr")
        print(f"    σ_t diff std: {r['sigma_diff_std']:.6f}")


if __name__ == '__main__':
    main()