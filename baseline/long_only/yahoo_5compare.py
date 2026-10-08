#!/usr/bin/env python3
"""5 contracts × 5 data sources: NON, REV, CLC RAD, Method C RAD, Yahoo Finance"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd, numpy as np
from data_loader import load_clc_full

EWMA=60; ST=0.063; B=0.002
yf_map = {'CC':'CC=F','LB':'LBS=F','JO':'OJ=F','ZO':'ZO=F','ZH':'HO=F'}

def gen_rad_C(tk):
    non = pd.read_csv(f'data/CLC/{tk}_NON.CSV', header=None, names=['D','O','H','L','C','V','OI'])
    rev = pd.read_csv(f'data/CLC/{tk}_REV.CSV', header=None, names=['D','O','H','L','C','V','OI'])
    non['D'] = pd.to_datetime(non['D'], format='%m/%d/%Y')
    rev['D'] = pd.to_datetime(rev['D'], format='%m/%d/%Y')
    m = non[['D','C']].merge(rev[['D','C']], on='D', suffixes=('_n','_r'))
    m = m.sort_values('D').reset_index(drop=True)
    p_n = m['C_n'].astype(float).values
    p_r = m['C_r'].astype(float).values
    adj = p_r - p_n; adj_diff = np.diff(adj)
    rolls = np.where(np.abs(adj_diff) > 1e-6)[0]
    cum = np.ones(len(p_n))
    for j in rolls:
        prev = p_n[j]; new_c = prev - adj_diff[j]
        if abs(new_c) > 1e-10: cum[j+1:] *= prev / new_c
    return m['D'].values, p_n, p_r, p_n * cum

def load_yf(tk):
    df = pd.read_csv(f'data/yahoo/{tk}_yahoo.csv', header=[0,1], index_col=0, parse_dates=True)
    df.columns = [c[0] for c in df.columns]
    p = pd.to_numeric(df['Close'], errors='coerce').values
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

sources = ['NON', 'REV', 'CLC_RAD', 'MC_RAD', 'Yahoo']

print(f"{'TK':4s} | ", end="")
for s in sources:
    print(f"{s+' E(R)':>10s} {s+' σ':>8s} | ", end="")
print()
print("-" * 140)

for tk in ['CC','LB','JO','ZO','ZH']:
    # Load all sources
    dates_all, p_non, p_rev, p_mc = gen_rad_C(tk)
    mask = dates_all >= np.datetime64('2009-01-01')
    p_non = p_non[mask]; p_rev = p_rev[mask]; p_mc = p_mc[mask]; dates_all = dates_all[mask]

    df_clc = load_clc_full(tk)
    p_vrad = df_clc['Close'].astype(float).values; d_vrad = df_clc['Date'].values

    d_yf, p_yf = load_yf(tk)

    results = {}
    for name, p, d in [
        ('NON', p_non[p_non>0], dates_all[p_non>0]),
        ('REV', p_rev[np.isfinite(p_rev)&(p_rev!=0)], dates_all[np.isfinite(p_rev)&(p_rev!=0)]),
        ('CLC_RAD', p_vrad, d_vrad),
        ('MC_RAD', p_mc[p_mc>0], dates_all[p_mc>0]),
        ('Yahoo', p_yf, d_yf),
    ]:
        er, vol, sig = backtest(p, d)
        results[name] = (er, sig)

    print(f"{tk:4s} | ", end="")
    for s in sources:
        er, sig = results.get(s, (None, None))
        if er is not None:
            print(f"{er:+10.4f} {sig:8.2f} | ", end="")
        else:
            print(f"{'N/A':>10s} {'N/A':>8s} | ", end="")
    print()

# Also show price levels
print(f"\n{'TK':4s} | ", end="")
for s in ['NON','REV','CLC_RAD','MC_RAD','Yahoo']:
    print(f"{s+' price':>12s} | ", end="")
print()
print("-" * 85)
for tk in ['CC','LB','JO','ZO','ZH']:
    dates_all, p_non, p_rev, p_mc = gen_rad_C(tk)
    mask = dates_all >= np.datetime64('2011-01-01')
    p_non_t = p_non[mask]; p_rev_t = p_rev[mask]; p_mc_t = p_mc[mask]
    
    df_clc = load_clc_full(tk)
    p_vrad = df_clc['Close'].astype(float).values; d_vrad = df_clc['Date'].values
    vm = d_vrad >= np.datetime64('2011-01-01'); p_vrad_t = p_vrad[vm]
    
    d_yf, p_yf = load_yf(tk)
    ym = d_yf >= np.datetime64('2011-01-01'); p_yf_t = p_yf[ym]

    def rng(p):
        v = p[np.isfinite(p)&(p>0)]
        return f"[{v.min():.1f},{v.max():.1f}]" if len(v)>0 else "N/A"

    print(f"{tk:4s} | {rng(p_non_t):12s} | {rng(p_rev_t):12s} | {rng(p_vrad_t):12s} | {rng(p_mc_t):12s} | {rng(p_yf_t):12s} |")
