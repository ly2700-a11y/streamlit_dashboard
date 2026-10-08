#!/usr/bin/env python3
"""Yahoo Finance vs CLC NON/REV: correlation analysis, roll-day vs non-roll-day"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np

yf_map = {'CC':'CC=F','LB':'LBS=F','JO':'OJ=F','ZO':'ZO=F','ZH':'HO=F'}

for tk in ['CC','LB','JO','ZO','ZH']:
    # Load CLC NON and REV
    non = pd.read_csv(f'data/CLC/{tk}_NON.CSV', header=None, names=['D','O','H','L','C','V','OI'])
    rev = pd.read_csv(f'data/CLC/{tk}_REV.CSV', header=None, names=['D','O','H','L','C','V','OI'])
    non['D'] = pd.to_datetime(non['D'], format='%m/%d/%Y')
    rev['D'] = pd.to_datetime(rev['D'], format='%m/%d/%Y')
    m = non[['D','C']].merge(rev[['D','C']], on='D', suffixes=('_n','_r'))
    m = m.sort_values('D').reset_index(drop=True)

    # Load Yahoo
    df_yf = pd.read_csv(f'data/yahoo/{tk}_yahoo.csv', header=[0,1], index_col=0, parse_dates=True)
    df_yf.columns = [c[0] for c in df_yf.columns]
    df_yf = df_yf.reset_index()
    df_yf.columns = ['Date','Close','High','Low','Open','Volume']
    df_yf['Date'] = pd.to_datetime(df_yf['Date'])
    df_yf['Close'] = pd.to_numeric(df_yf['Close'], errors='coerce')
    df_yf = df_yf.sort_values('Date').reset_index(drop=True)

    # Merge all three on Date, test period only
    ts = pd.Timestamp('2011-01-01'); te = pd.Timestamp('2019-12-31')
    m = m[(m['D']>=ts)&(m['D']<=te)].reset_index(drop=True)
    m.columns = ['Date','C_non','C_rev']
    m['C_non'] = m['C_non'].astype(float)
    m['C_rev'] = m['C_rev'].astype(float)

    merged = m.merge(df_yf[['Date','Close']], on='Date', how='inner')
    merged = merged.dropna()
    merged = merged[merged['C_non'] > 0]
    p_n = merged['C_non'].values
    p_r = merged['C_rev'].values
    p_y = merged['Close'].values
    dates = merged['Date'].values
    n = len(merged)

    # Detect rolls
    adj = p_r - p_n; adj_diff = np.diff(adj)
    roll_idx = set(np.where(np.abs(adj_diff) > 1e-6)[0])

    # Returns
    ret_n = np.diff(p_n)
    ret_r = np.diff(p_r)
    ret_y = np.diff(p_y)

    # Pct returns
    pct_n = np.diff(p_n)/p_n[:-1]
    pct_r = np.diff(p_r)/p_r[:-1]
    pct_y = np.diff(p_y)/p_y[:-1]

    # Split: non-roll vs roll days
    nr_mask = np.array([i not in roll_idx for i in range(n-1)])
    r_mask = ~nr_mask

    v_all = np.isfinite(pct_n) & np.isfinite(pct_r) & np.isfinite(pct_y)
    v_nr = nr_mask & v_all
    v_r = r_mask & v_all

    print(f"\n{'='*70}")
    print(f"{tk}: {n} days, {r_mask.sum()} roll days, {nr_mask.sum()} non-roll days")
    print(f"{'='*70}")
    print(f"  Price: NON=[{p_n.min():.1f},{p_n.max():.1f}] REV=[{p_r.min():.1f},{p_r.max():.1f}] YF=[{p_y.min():.1f},{p_y.max():.1f}]")
    print(f"  YF/NON ratio: {p_y[0]/p_n[0]:.4f}")
    print(f"  YF/REV ratio: {p_y[0]/p_r[0]:.4f}")

    print(f"\n  === All days pct_ret correlation ===")
    for pair, a, b in [('YF vs NON', pct_y, pct_n), ('YF vs REV', pct_y, pct_r), ('REV vs NON', pct_r, pct_n)]:
        c = np.corrcoef(a[v_all], b[v_all])[0,1]
        print(f"    {pair}: {c:.6f}")

    print(f"\n  === Non-roll days pct_ret correlation ===")
    for pair, a, b in [('YF vs NON', pct_y, pct_n), ('YF vs REV', pct_y, pct_r), ('REV vs NON', pct_r, pct_n)]:
        c = np.corrcoef(a[v_nr], b[v_nr])[0,1]
        exact = np.sum(np.abs(a[v_nr]-b[v_nr])<1e-10)
        print(f"    {pair}: {c:.10f}  exact={exact}/{v_nr.sum()}")

    print(f"\n  === Roll days pct_ret correlation ===")
    for pair, a, b in [('YF vs NON', pct_y, pct_n), ('YF vs REV', pct_y, pct_r), ('REV vs NON', pct_r, pct_n)]:
        if v_r.sum() > 5:
            c = np.corrcoef(a[v_r], b[v_r])[0,1]
        else:
            c = float('nan')
        print(f"    {pair}: {c:.6f}")

    # Level correlation
    print(f"\n  === Level correlation ===")
    print(f"    YF vs NON: {np.corrcoef(p_y, p_n)[0,1]:.6f}")
    print(f"    YF vs REV: {np.corrcoef(p_y, p_r)[0,1]:.6f}")

    # Is YF closer to NON or REV in level?
    err_non = np.mean(np.abs(p_y - p_n)) / np.mean(p_n) * 100
    err_rev = np.mean(np.abs(p_y - p_r)) / np.mean(np.abs(p_r)) * 100 if np.mean(np.abs(p_r)) > 0 else 999
    print(f"    Mean |YF-NON|/mean(NON): {err_non:.1f}%")
    print(f"    Mean |YF-REV|/mean(|REV|): {err_rev:.1f}%")
