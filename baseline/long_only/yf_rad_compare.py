#!/usr/bin/env python3
"""Use Yahoo Finance as NON, apply Method C RAD generation, compare per-contract E(R)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np
from data_loader import load_clc_full

EWMA=60; ST=0.063; B=0.002

yf_map = {'CC':'CC=F','LB':'LBS=F','JO':'OJ=F','ZO':'ZO=F','ZH':'HO=F'}

def load_yf(tk):
    df = pd.read_csv(f'data/yahoo/{tk}_yahoo.csv', header=[0,1], index_col=0, parse_dates=True)
    df.columns = [c[0] for c in df.columns]
    p = pd.to_numeric(df['Close'], errors='coerce').values.flatten()
    d = df.index.values
    mask = (d >= np.datetime64('2009-01-01')) & (d <= np.datetime64('2019-12-31'))
    p = p[mask]; d = d[mask]
    valid = np.isfinite(p) & (p > 0)
    return d[valid], p[valid]

def backtest(prices, dates):
    n = len(prices)
    rt = np.zeros(n); rt[1:] = prices[1:] - prices[:-1]
    sig = pd.Series(rt).ewm(span=EWMA, adjust=False).std().values
    ms = dates >= np.datetime64('2011-01-01'); me = dates <= np.datetime64('2019-12-31')
    if not ms.any() or not me.any(): return None, None, None
    t0 = max(np.argmax(ms), 252); t1 = n-1-np.argmax(me[::-1])
    Rt = np.zeros(n)
    for t in range(1,n):
        if sig[t-1]>0 and (t<2 or sig[t-2]>0):
            sp=ST/sig[t-1]; spp=ST/sig[t-2] if t>=2 else 0
            Rt[t]=sp*rt[t]-B*abs(prices[t-1])*abs(sp-spp)
    slc = Rt[t0:t1]
    return np.mean(slc)*252, np.std(slc)*np.sqrt(252), np.mean(sig[t0:t1])

print(f"{'TK':4s} | {'CLC_NON':>9s} {'σ':>6s} | {'CLC_REV':>9s} {'σ':>6s} | {'CLC_RAD':>9s} {'σ':>6s} | {'YF_NON':>9s} {'σ':>6s} | {'YF_RAD':>9s} {'σ':>6s}")
print("-"*100)

for tk in ['CC','LB','JO','ZO','ZH']:
    # CLC sources
    non = pd.read_csv(f'data/CLC/{tk}_NON.CSV', header=None, names=['D','O','H','L','C','V','OI'])
    rev = pd.read_csv(f'data/CLC/{tk}_REV.CSV', header=None, names=['D','O','H','L','C','V','OI'])
    non['D'] = pd.to_datetime(non['D'], format='%m/%d/%Y')
    rev['D'] = pd.to_datetime(rev['D'], format='%m/%d/%Y')
    m = non[['D','C']].merge(rev[['D','C']], on='D', suffixes=('_n','_r'))
    m = m.sort_values('D').reset_index(drop=True)
    p_n = m['C_n'].astype(float).values; p_r = m['C_r'].astype(float).values; d_c = m['D'].values
    mask = d_c >= np.datetime64('2009-01-01')
    p_n=p_n[mask]; p_r=p_r[mask]; d_c=d_c[mask]

    # CLC Method C RAD from CLC NON+REV
    adj = p_r - p_n; adj_diff = np.diff(adj)
    rolls = np.where(np.abs(adj_diff) > 1e-6)[0]
    cum = np.ones(len(p_n))
    for j in rolls:
        prev = p_n[j]; new_c = prev - adj_diff[j]
        if abs(new_c) > 1e-10: cum[j+1:] *= prev / new_c
    p_rad_c = p_n * cum

    # CLC vendor RAD
    df_clc = load_clc_full(tk)
    p_vrad = df_clc['Close'].astype(float).values; d_vrad = df_clc['Date'].values

    # Yahoo NON
    d_yf, p_yf = load_yf(tk)

    # Yahoo RAD: use same roll dates and adj_changes from CLC REV-NON
    # but apply to YF prices
    # Align CLC rolls to YF dates
    cum_yf = np.ones(len(p_yf))
    d_clc_masked = d_c
    yf_date_set = {pd.Timestamp(d): i for i, d in enumerate(d_yf)}
    
    for j in rolls:
        roll_date = pd.Timestamp(d_clc_masked[j])
        # Find this date in YF
        if roll_date in yf_date_set:
            yj = yf_date_set[roll_date]
        else:
            # Try nearest business day
            continue
        
        prev_yf = p_yf[yj]
        # Use the same ratio as CLC (ratio = prev/(prev-adj_change))
        ratio = p_n[j] / (p_n[j] - adj_diff[j])
        if yj+1 < len(cum_yf):
            cum_yf[yj+1:] *= ratio
    
    p_yf_rad = p_yf * cum_yf

    # Backtest all 5
    er_n_c, _, sig_n = backtest(p_n[p_n>0], d_c[p_n>0])
    er_r_c, _, sig_r = backtest(p_r[np.isfinite(p_r)&(p_r!=0)], d_c[np.isfinite(p_r)&(p_r!=0)])
    er_vrad, _, sig_v = backtest(p_vrad, d_vrad)
    er_yf_n, _, sig_yn = backtest(p_yf, d_yf)
    er_yf_rad, _, sig_yr = backtest(p_yf_rad[p_yf_rad>0], d_yf[p_yf_rad>0])

    print(f"{tk:4s} | {er_n_c:+9.4f} {sig_n:6.2f} | {er_r_c:+9.4f} {sig_r:6.2f} | {er_vrad:+9.4f} {sig_v:6.2f} | {er_yf_n:+9.4f} {sig_yn:6.2f} | {er_yf_rad:+9.4f} {sig_yr:6.2f}")
