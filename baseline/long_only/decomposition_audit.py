#!/usr/bin/env python3
"""
decomposition_audit.py — Per-contract E(R) decomposition & p0 normalization test

For each active contract:
1. E(R) decomposition: signal return vs TC drag
2. % +ve and Ave P/L (with and without p0 normalization)
3. σ_t/price ratio (diagnoses FI TC problem)

Usage:
  cd IEOR4733_Project && python tests/decomposition_audit.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_loader import load_clc_full
from config import ASSET_CLASSES, BP, TRADING_DAYS, EXCLUDED_CONTRACTS

TEST_START = '2011-01-01'
TEST_END = '2019-12-31'
SIGMA_TGT = 0.063
EWMA_SPAN = 60
T = TRADING_DAYS


def decompose_contract(ticker, use_p0_norm=False):
    """Full per-contract decomposition."""
    df = load_clc_full(ticker)
    if df is None:
        return None
    
    prices = df['Close'].values.astype(float)
    if len(prices) < 500:
        return None
    
    # p0 normalization
    if use_p0_norm:
        p0 = prices[0]
        if p0 > 0:
            prices = prices / p0
    
    # Additive returns
    rt = np.zeros(len(prices))
    rt[1:] = prices[1:] - prices[:-1]
    
    # σ_t
    sigma = pd.Series(rt).ewm(span=EWMA_SPAN, adjust=False).std().values
    
    # Test period
    mask_s = df['Date'] >= TEST_START
    mask_e = df['Date'] <= TEST_END
    if not mask_s.any() or not mask_e.any():
        return None
    t0 = mask_s.idxmax()
    t1 = len(df) - 1 - mask_e[::-1].values.argmax()
    start = max(t0, 252)  # SIGN_LOOKBACK warmup
    
    # Compute R_t with decomposition
    Rt = np.zeros(len(rt))
    signal_sum = 0.0
    tc_sum = 0.0
    count = 0
    
    pos_R = []
    neg_R = []
    
    for t in range(1, len(rt)):
        if sigma[t-1] > 0 and (t < 2 or sigma[t-2] > 0):
            sp = SIGMA_TGT / sigma[t-1]
            spp = SIGMA_TGT / sigma[t-2] if t >= 2 and sigma[t-2] > 0 else 0.0
            signal = sp * rt[t]
            tc = BP * prices[t-1] * abs(sp - spp)
            Rt[t] = signal - tc
            
            if t >= start and t < t1:
                signal_sum += signal
                tc_sum += tc
                count += 1
                if Rt[t] > 0:
                    pos_R.append(Rt[t])
                elif Rt[t] < 0:
                    neg_R.append(Rt[t])
    
    test_Rt = Rt[start:t1]
    if len(test_Rt) == 0 or count == 0:
        return None
    
    er = np.mean(test_Rt) * T
    avg_signal = signal_sum / count * T
    avg_tc = tc_sum / count * T
    
    # % +ve and Ave P/L
    pos_arr = np.array(pos_R)
    neg_arr = np.array(neg_R)
    pct_pos = len(pos_arr) / count if count > 0 else 0
    avg_pl = (pos_arr.mean() / abs(neg_arr.mean())) if len(pos_arr) > 0 and len(neg_arr) > 0 else 0
    
    # σ_t / price ratio (annualized) — key diagnostic
    test_sigma = sigma[start:t1]
    test_prices_slice = prices[start:t1]
    valid = test_sigma > 0
    if valid.any():
        sigma_price_ratio = np.median(test_sigma[valid] / test_prices_slice[valid]) * np.sqrt(T)
    else:
        sigma_price_ratio = 0
    
    # Asset class
    ac = 'Unknown'
    for name, tickers in ASSET_CLASSES.items():
        if ticker in tickers:
            ac = name
            break
    
    return {
        'ticker': ticker, 'ac': ac,
        'n': count,
        'price_mean': np.mean(test_prices_slice),
        'sigma_mean': np.mean(test_sigma[valid]) if valid.any() else 0,
        'sigma_price_ratio': sigma_price_ratio,
        'er': er,
        'signal': avg_signal,
        'tc': avg_tc,
        'tc_pct_signal': abs(avg_tc / avg_signal) * 100 if avg_signal != 0 else 0,
        'pct_pos': pct_pos,
        'avg_pl': avg_pl,
        'mean_pos': pos_arr.mean() * T if len(pos_arr) > 0 else 0,
        'mean_neg': neg_arr.mean() * T if len(neg_arr) > 0 else 0,
    }


def main():
    # All active tickers
    all_tickers = []
    for ac, tickers in ASSET_CLASSES.items():
        for tk in tickers:
            if tk not in EXCLUDED_CONTRACTS:
                all_tickers.append((tk, ac))
    
    print('=' * 140)
    print('PER-CONTRACT E(R) DECOMPOSITION (without p0 normalization)')
    print(f'σ_tgt={SIGMA_TGT}, bp={BP}, EWMA={EWMA_SPAN}, test={TEST_START} to {TEST_END}')
    print('=' * 140)
    
    results_raw = {}
    results_norm = {}
    
    for tk, ac in all_tickers:
        r = decompose_contract(tk, use_p0_norm=False)
        if r:
            results_raw[tk] = r
        r2 = decompose_contract(tk, use_p0_norm=True)
        if r2:
            results_norm[tk] = r2
    
    # Print by asset class
    for ac in ['Fixed Income', 'Commodity', 'Equity Index', 'Forex']:
        ac_results = {tk: r for tk, r in results_raw.items() if r['ac'] == ac}
        if not ac_results:
            continue
        
        print(f"\n{'─' * 140}")
        print(f"  {ac} ({len(ac_results)} contracts)")
        print(f"{'─' * 140}")
        print(f"{'TK':>4} | {'Price':>8} {'σ_mean':>8} {'σ/P(ann)':>8} | {'E(R)':>8} {'Signal':>8} {'TC':>8} {'TC%Sig':>7} | {'%+ve':>6} {'P/L':>6} {'mean+':>8} {'mean-':>8} | Status")
        print('-' * 140)
        
        for tk in sorted(ac_results.keys()):
            r = ac_results[tk]
            issues = []
            if r['tc_pct_signal'] > 50:
                issues.append('🚨TC>50%')
            elif r['tc_pct_signal'] > 30:
                issues.append('⚠️TC')
            if abs(r['pct_pos'] - 0.5) > 0.05:
                issues.append('⚠️%+ve')
            status = ' '.join(issues) if issues else '✅'
            
            print(f"{r['ticker']:>4} | {r['price_mean']:>8.2f} {r['sigma_mean']:>8.4f} {r['sigma_price_ratio']:>8.4f} | {r['er']:>+8.3f} {r['signal']:>+8.3f} {r['tc']:>8.3f} {r['tc_pct_signal']:>6.1f}% | {r['pct_pos']:>6.4f} {r['avg_pl']:>6.3f} {r['mean_pos']:>+8.4f} {r['mean_neg']:>+8.4f} | {status}")
    
    # p0 normalization comparison
    print(f"\n{'=' * 140}")
    print('p0 NORMALIZATION COMPARISON (raw vs normalized)')
    print(f"{'=' * 140}")
    print(f"{'TK':>4} {'AC':>12} | {'E(R)raw':>8} {'E(R)norm':>9} {'ΔE(R)':>8} | {'P/Lraw':>7} {'P/Lnorm':>8} {'ΔP/L':>6} | {'TCraw':>7} {'TCnorm':>8} {'ΔTC':>6}")
    print('-' * 130)
    
    for ac in ['Fixed Income', 'Commodity', 'Equity Index', 'Forex']:
        ac_tks = [(tk, r) for tk, r in results_raw.items() if r['ac'] == ac]
        for tk, r_raw in sorted(ac_tks):
            r_norm = results_norm.get(tk)
            if r_norm:
                d_er = r_norm['er'] - r_raw['er']
                d_pl = r_norm['avg_pl'] - r_raw['avg_pl']
                d_tc = r_norm['tc'] - r_raw['tc']
                print(f"{tk:>4} {ac:>12} | {r_raw['er']:>+8.3f} {r_norm['er']:>+9.3f} {d_er:>+8.3f} | {r_raw['avg_pl']:>7.3f} {r_norm['avg_pl']:>8.3f} {d_pl:>+6.3f} | {r_raw['tc']:>7.3f} {r_norm['tc']:>8.3f} {d_tc:>+6.3f}")
    
    # Portfolio-level comparison
    print(f"\n{'=' * 140}")
    print('PORTFOLIO-LEVEL: raw vs p0-normalized (Long strategy, equal-weight)')
    print(f"{'=' * 140}")
    for ac in ['Fixed Income', 'Commodity', 'Equity Index', 'Forex']:
        ac_raw = [r for r in results_raw.values() if r['ac'] == ac]
        ac_norm = [r for r in results_norm.values() if r['ac'] == ac]
        if not ac_raw:
            continue
        
        # Equal-weight portfolio average
        raw_er = np.mean([r['er'] for r in ac_raw])
        raw_signal = np.mean([r['signal'] for r in ac_raw])
        raw_tc = np.mean([r['tc'] for r in ac_raw])
        raw_pct = np.mean([r['pct_pos'] for r in ac_raw])
        raw_pl = np.mean([r['avg_pl'] for r in ac_raw])
        
        norm_er = np.mean([r['er'] for r in ac_norm])
        norm_signal = np.mean([r['signal'] for r in ac_norm])
        norm_tc = np.mean([r['tc'] for r in ac_norm])
        norm_pct = np.mean([r['pct_pos'] for r in ac_norm])
        norm_pl = np.mean([r['avg_pl'] for r in ac_norm])
        
        print(f"\n  {ac} ({len(ac_raw)} contracts):")
        print(f"    Raw:   E(R)={raw_er:+.3f}  Signal={raw_signal:+.3f}  TC={raw_tc:.3f}  %+ve={raw_pct:.4f}  P/L={raw_pl:.3f}")
        print(f"    Norm:  E(R)={norm_er:+.3f}  Signal={norm_signal:+.3f}  TC={norm_tc:.3f}  %+ve={norm_pct:.4f}  P/L={norm_pl:.3f}")
        print(f"    Delta: E(R)={norm_er-raw_er:+.3f}  Signal={norm_signal-raw_signal:+.3f}  TC={norm_tc-raw_tc:+.3f}  %+ve={norm_pct-raw_pct:+.4f}  P/L={norm_pl-raw_pl:+.3f}")


if __name__ == '__main__':
    main()