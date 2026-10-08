#!/usr/bin/env python3
"""
compare_data_sources.py — Compare 3 approaches for price data:

A) REV returns:  r_t = REV[t] - REV[t-1]  (back-adjusted, non-roll days == NON)
B) Regenerated RAD from NON+REV: detect rolls from adj change, forward-accumulate ratio
C) Vendor RAD (current approach)

For each approach, run Eq 4 (Long, Table 3) and compare to paper values.

Key insight from cross-validation:
- REV: 50/50 pass non-roll-day consistency with NON (r_REV == r_NON)
- Vendor RAD: only 3/50 pass pct_ret consistency with NON
- So vendor RAD is broken for 47/50 contracts
"""
import numpy as np
import pandas as pd
from config import ASSET_CLASSES, BP, TRADING_DAYS, PAPER_TABLE3, METRIC_NAMES
from metrics import compute_metrics

EWMA_SPAN = 60
SIGMA_TGT = 0.063
T = TRADING_DAYS

# ─── Data Loading ────────────────────────────────────────────────

def load_raw(ticker):
    """Load NON, REV, RAD raw data, return merged DataFrame."""
    non = pd.read_csv(f'data/CLC/{ticker}_NON.CSV', header=None,
                      names=['Date','O','H','L','C','V','OI'])
    rev = pd.read_csv(f'data/CLC/{ticker}_REV.CSV', header=None,
                      names=['Date','O','H','L','C','V','OI'])
    rad = pd.read_csv(f'data/CLC/{ticker}_RAD.CSV', header=None,
                      names=['Date','O','H','L','C','V','OI'])
    
    for df in [non, rev, rad]:
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    
    m = non[['Date','C']].merge(rev[['Date','C']], on='Date', suffixes=('_non','_rev'))
    m = m.merge(rad[['Date','C']], on='Date')
    m.columns = ['Date','C_non','C_rev','C_rad']
    m = m.sort_values('Date').reset_index(drop=True)
    return m


def get_returns_approach_A(m, start='2009-01-01', end='2019-12-31'):
    """Approach A: Use REV prices directly.
    REV = NON + cum_adj. On non-roll days r_REV == r_NON exactly.
    Returns: prices array, test period indices.
    """
    mask = (m['Date'] >= start) & (m['Date'] <= end)
    sub = m[mask].reset_index(drop=True)
    prices = pd.to_numeric(sub['C_rev'], errors='coerce').values
    
    # Need warmup from 2009, test from 2011
    return sub, prices


def get_returns_approach_B(m, start='2009-01-01', end='2019-12-31'):
    """Approach B: Regenerate RAD from NON + REV.
    
    Algorithm:
    1. Detect rolls: where (REV - NON) diff != 0
    2. At each roll: ratio = NON[t-1] / NON[t]  (price gap due to contract switch)
    3. Forward-accumulate ratios
    4. RAD_regen[t] = NON[t] * cum_ratio[t]
    
    This ensures: pct_ret(RAD_regen) == pct_ret(NON) on all days (by construction).
    Absolute returns: r_RAD_regen = cum_ratio * r_NON
    """
    mask = (m['Date'] >= start) & (m['Date'] <= end)
    sub = m[mask].reset_index(drop=True)
    
    p_non = pd.to_numeric(sub['C_non'], errors='coerce').values
    p_rev = pd.to_numeric(sub['C_rev'], errors='coerce').values
    
    # Detect rolls from adj changes
    adj = p_rev - p_non
    adj_diff = np.diff(adj)
    roll_idx = np.where(np.abs(adj_diff) > 1e-6)[0]
    
    # Build cumulative ratio (forward from t=0)
    cum_ratio = np.ones(len(p_non))
    for ri in roll_idx:
        # At roll point ri -> ri+1: NON jumps from p_non[ri] to p_non[ri+1]
        # ratio adjusts for this jump
        if abs(p_non[ri + 1]) > 1e-10:
            ratio = p_non[ri] / p_non[ri + 1]
        else:
            ratio = 1.0
        cum_ratio[ri + 1:] *= ratio
    
    # RAD_regen = NON * cum_ratio
    prices = p_non * cum_ratio
    
    return sub, prices


def get_returns_approach_C(m, start='2009-01-01', end='2019-12-31'):
    """Approach C: Use vendor RAD as-is (current approach)."""
    mask = (m['Date'] >= start) & (m['Date'] <= end)
    sub = m[mask].reset_index(drop=True)
    prices = pd.to_numeric(sub['C_rad'], errors='coerce').values
    
    # Filter out zero/NaN
    valid = np.isfinite(prices) & (prices > 0)
    sub = sub[valid].reset_index(drop=True)
    prices = prices[valid]
    
    return sub, prices


# ─── Backtest Engine ──────────────────────────────────────────────

def run_backtest(prices, dates, test_start='2011-01-01', test_end='2019-12-31'):
    """Run Eq 4 Long strategy, return portfolio returns in test period."""
    n = len(prices)
    
    # Additive returns
    rt = np.zeros(n)
    rt[1:] = prices[1:] - prices[:-1]
    
    # EWMA std
    sigma = pd.Series(rt).ewm(span=EWMA_SPAN, adjust=False).std().values
    
    # Test period
    mask_s = dates >= test_start
    mask_e = dates <= test_end
    if not mask_s.any() or not mask_e.any():
        return None
    
    t0 = mask_s.idxmax()
    t1 = len(dates) - 1 - mask_e[::-1].values.argmax()
    t0 = max(t0, 252)  # warmup
    
    # Eq 4: R_t = (σ_tgt / σ_{t-1}) * r_t - bp * p_{t-1} * |Δscaled_pos|
    Rt = np.zeros(n)
    for t in range(1, n):
        if sigma[t-1] > 0 and (t < 2 or sigma[t-2] > 0):
            sp = SIGMA_TGT / sigma[t-1]
            spp = SIGMA_TGT / sigma[t-2] if t >= 2 else 0.0
            Rt[t] = sp * rt[t] - BP * prices[t-1] * abs(sp - spp)
    
    return Rt[t0:t1]


# ─── Main ─────────────────────────────────────────────────────────

def run_all_approaches():
    approaches = {
        'A_REV': get_returns_approach_A,
        'B_RAD_regen': get_returns_approach_B,
        'C_Vendor_RAD': get_returns_approach_C,
    }
    
    for ac_name in ['Commodity', 'Equity Index', 'Fixed Income', 'Forex']:
        tickers = ASSET_CLASSES[ac_name]
        print(f"\n{'='*80}")
        print(f"  {ac_name} ({len(tickers)} contracts)")
        print(f"{'='*80}")
        
        paper = PAPER_TABLE3[ac_name]['Long']
        core = ['E(R)','std(R)','Sharpe','% +ve','Ave P/L']
        
        for appr_name, appr_fn in approaches.items():
            port_returns = []
            n_loaded = 0
            
            for tk in tickers:
                try:
                    m = load_raw(tk)
                    sub, prices = appr_fn(m)
                    
                    if len(prices) < 500:
                        continue
                    if np.sum(np.isfinite(prices) & (prices > 0)) < 500:
                        continue
                    
                    dates = sub['Date'].values
                    test_idx = (dates >= np.datetime64('2011-01-01')) & (dates <= np.datetime64('2019-12-31'))
                    if test_idx.sum() < 100:
                        continue
                    
                    Rt = run_backtest(prices, sub['Date'], '2011-01-01', '2019-12-31')
                    if Rt is not None and len(Rt) > 100:
                        n_loaded += 1
                        # Align to dates
                        t0_idx = sub[sub['Date'] >= '2011-01-01'].index[0]
                        t0_idx = max(t0_idx, 252)
                        t1_idx = sub[sub['Date'] <= '2019-12-31'].index[-1]
                        test_dates = sub['Date'].iloc[t0_idx:t1_idx].values[:len(Rt)]
                        port_returns.append(pd.Series(Rt[:len(test_dates)], index=test_dates))
                except Exception as e:
                    pass
            
            if not port_returns:
                print(f"  {appr_name}: NO DATA")
                continue
            
            # Equal-weight portfolio
            df = pd.DataFrame(port_returns)
            R_port = df.T.mean(axis=1).values
            
            m_all = compute_metrics(R_port, n_loaded)
            m_dict = dict(zip(METRIC_NAMES, m_all))
            
            # Compare to paper
            errs = {}
            for k in core:
                pv = paper[k]
                mv = m_dict[k]
                if abs(pv) > 1e-6:
                    errs[k] = abs((mv - pv) / pv) * 100
                else:
                    errs[k] = 0
            
            n10 = sum(1 for e in errs.values() if e < 10)
            n15 = sum(1 for e in errs.values() if e < 15)
            
            print(f"\n  {appr_name} ({n_loaded} contracts, {len(R_port)} days)")
            print(f"  {'Metric':<10} {'Ours':>8} {'Paper':>8} {'%Err':>8}")
            print(f"  {'-'*36}")
            for k in core:
                print(f"  {k:<10} {m_dict[k]:>+8.3f} {paper[k]:>+8.3f} {errs[k]:>7.1f}%")
            print(f"  ≤10%: {n10}/{len(core)}  ≤15%: {n15}/{len(core)}")

if __name__ == '__main__':
    run_all_approaches()
