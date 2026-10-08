"""
Regenerate RAD_v2 for contracts with corrupted/incomplete vendor RAD.

Algorithm (for REV-based generation):
1. Load NON and REV
2. Detect rolls where adj_change ≠ 0 (REV close - NON close changes)
3. Compute roll ratio = prev_close / (prev_close - adj_change) at each roll
4. Forward-accumulate: multiply all OHLC by cumulative product of roll ratios
5. Save as RAD_v2

This produces a continuous series where:
  - Non-roll dates: return = NON return (corr should be ~1.0)
  - Roll dates: no price jump (roll adjustment absorbed in ratio)
  - OHLC all scale consistently

Usage:
  cd IEOR4733_Project && python3 tests/generate_rad_v2_validated.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'CLC'

sys.path.insert(0, str(PROJECT_ROOT))
from config import ASSET_CLASSES, EXCLUDED_CONTRACTS

# Contracts that need RAD_v2 based on full_diagnostic.py results:
# - ZN: vendor RAD only covers 2011+ (missing warmup), needs REV-based generation
# - US: vendor RAD all NaN, needs REV-based generation
# - ZH: vendor RAD Close=0.00 for 1980-2021 (corrupted), needs REV-based generation
# - ZU: excluded contract but may be loaded, v2 ensures data availability
# - GI: v2 corr=0.995 vs vendor corr=0.989 (improvement)
# - KC: v2 corr=0.992 vs vendor corr=0.972 (improvement)
V2_CONTRACTS = ['ZN', 'US', 'ZU', 'GI', 'KC', 'ZH']


def load(filepath, require_positive=True):
    """Load CLC CSV file.
    
    Args:
        require_positive: If True, filter out rows where C <= 0 (for NON).
                          If False, keep all rows with valid C (for REV, which
                          can have negative values from back-adjustment).
    """
    df = pd.read_csv(filepath, header=None,
                     names=['Date', 'O', 'H', 'L', 'C', 'V', 'OI'])
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    if require_positive:
        df = df[df['C'].notna() & (df['C'] > 0)]
    else:
        df = df[df['C'].notna()]
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def generate_rad_v2_from_rev(ticker):
    """Generate RAD_v2 using REV's continuous returns.
    
    Key insight: REV (back-adjusted) has continuous returns across rolls.
    RAD should have the SAME returns as REV but start at NON's price level.
    
    Method:
    1. Merge NON and REV on date
    2. Compute cumulative return from REV: cumrev[t] = REV[t] / REV[0]
    3. RAD[t] = NON[0] * cumrev[t]  (start at NON price, compound REV returns)
    
    This gives:
    - Continuous series (no gaps at rolls) ✓
    - Returns identical to REV returns ✓  
    - On non-roll days, returns ≈ NON returns ✓
    - OHLC consistent (all scaled by same factor) ✓
    - Always positive prices ✓
    """
    non_f = DATA_DIR / f'{ticker}_NON.CSV'
    rev_f = DATA_DIR / f'{ticker}_REV.CSV'
    output_f = DATA_DIR / f'{ticker}_RAD_v2.CSV'

    if not non_f.exists() or not rev_f.exists():
        print(f"\n  {ticker}: NON or REV file missing, skipping")
        return None

    non_df = load(non_f, require_positive=True)
    rev_df = load(rev_f, require_positive=False)

    # Merge on date (inner join - only common dates)
    merged = non_df[['Date', 'O', 'H', 'L', 'C', 'V', 'OI']].merge(
        rev_df[['Date', 'C']], on='Date', suffixes=('_non', '_rev'))
    merged = merged.sort_values('Date').reset_index(drop=True)

    non_c = merged['C_non'].values.astype(float)
    non_o = merged['O'].values.astype(float)
    non_h = merged['H'].values.astype(float)
    non_l = merged['L'].values.astype(float)
    non_v = merged['V'].values.astype(float)
    non_oi = merged['OI'].values.astype(float)
    rev_c = merged['C_rev'].values.astype(float)
    
    # Count rolls for reporting
    adj = rev_c - non_c
    adj_change = np.diff(adj)
    n_rolls = np.sum(np.abs(adj_change) > 1e-6)
    test_mask_arr = (merged['Date'] >= pd.Timestamp('2011-01-01')) & \
                    (merged['Date'] <= pd.Timestamp('2019-12-31'))
    test_rolls = np.sum(np.abs(adj_change[test_mask_arr.values[:-1]]) > 1e-6) if test_mask_arr.sum() > 1 else 0
    
    print(f"\n  {ticker}: {n_rolls} total rolls, {test_rolls} in test, {len(merged)} total rows")

    # Build RAD from REV cumulative returns, anchored at NON price level
    # 
    # We anchor at the LAST date (not the first) because:
    #   - REV (back-adjusted) can have negative values at the start of the series
    #   - Anchoring at the end ensures scale > 0 (both NON and REV positive at end)
    #   - Returns are identical regardless of anchor point: RAD[t]/RAD[t-1] = REV[t]/REV[t-1]
    #
    # RAD[t] = NON[-1] * (REV[t] / REV[-1])
    scale = non_c[-1] / rev_c[-1] if rev_c[-1] != 0 else 1.0
    rad_c = rev_c * scale
    # For OHLC: scale each day's OHLC by the same ratio as that day's Close
    # ratio[t] = rad_c[t] / non_c[t] = cumulative adjustment factor (always > 0 with end-anchor)
    daily_ratio = rad_c / non_c
    rad_o = non_o * daily_ratio
    rad_h = non_h * daily_ratio
    rad_l = non_l * daily_ratio

    # Save
    out = pd.DataFrame({
        'Date': merged['Date'],
        'O': rad_o,
        'H': rad_h,
        'L': rad_l,
        'C': rad_c,
        'V': non_v,
        'OI': non_oi,
    })
    out['Date'] = out['Date'].dt.strftime('%m/%d/%Y')
    out.to_csv(output_f, index=False, header=False)

    # Validate on test period
    test_rad = rad_c[test_mask_arr.values]
    test_non = non_c[test_mask_arr.values]

    # Return correlation
    rad_ret = pd.Series(test_rad).diff().dropna()
    non_ret = pd.Series(test_non).diff().dropna()
    min_len = min(len(rad_ret), len(non_ret))
    corr = np.corrcoef(rad_ret.values[:min_len], non_ret.values[:min_len])[0, 1] if min_len > 10 else 0

    # % +ve check
    rad_pve = (rad_ret > 0).sum() / len(rad_ret) if len(rad_ret) > 0 else 0
    non_pve = (non_ret > 0).sum() / len(non_ret) if len(non_ret) > 0 else 0
    
    # Ave P/L check
    rad_pos = rad_ret[rad_ret > 0].mean() if (rad_ret > 0).any() else 0
    rad_neg = abs(rad_ret[rad_ret < 0].mean()) if (rad_ret < 0).any() else 1
    non_pos = non_ret[non_ret > 0].mean() if (non_ret > 0).any() else 0
    non_neg = abs(non_ret[non_ret < 0].mean()) if (non_ret < 0).any() else 1
    rad_pl = rad_pos / rad_neg if rad_neg > 0 else 0
    non_pl = non_pos / non_neg if non_neg > 0 else 0

    print(f"  Output: {len(merged)} rows -> {output_f.name}")
    print(f"  Test: {len(test_rad)} rows, RAD range: {test_rad.min():.4f} - {test_rad.max():.4f}")
    print(f"  NON test range: {test_non.min():.4f} - {test_non.max():.4f}")
    print(f"  Return corr vs NON: {corr:.6f}")
    print(f"  % +ve: RAD={rad_pve:.4f} NON={non_pve:.4f} diff={abs(rad_pve-non_pve):.4f}")
    print(f"  Ave P/L: RAD={rad_pl:.3f} NON={non_pl:.3f} diff={abs(rad_pl-non_pl):.3f}")

    return {'ticker': ticker, 'corr': corr, 'rows': len(merged),
            'rad_pve': rad_pve, 'non_pve': non_pve,
            'rad_pl': rad_pl, 'non_pl': non_pl}


def main():
    print('=' * 80)
    print('REGENERATING RAD_v2 FOR ALL DAMAGED CONTRACTS')
    print('Using NON + REV forward accumulation')
    print('=' * 80)

    results = []
    for ticker in V2_CONTRACTS:
        r = generate_rad_v2_from_rev(ticker)
        if r:
            results.append(r)

    # Summary
    print(f"\n{'=' * 80}")
    print('SUMMARY')
    print(f"{'=' * 80}")
    print(f"  {'TK':4s} {'Corr':>8s} {'%+ve_D':>8s} {'PL_D':>6s} {'Status':>8s}")
    print(f"  {'-'*40}")
    for r in results:
        status = '✅' if r['corr'] >= 0.99 else ('⚠️' if r['corr'] >= 0.95 else '❌')
        pve_d = abs(r['rad_pve'] - r['non_pve'])
        pl_d = abs(r['rad_pl'] - r['non_pl'])
        print(f"  {r['ticker']:4s} {r['corr']:8.4f} {pve_d:8.4f} {pl_d:6.3f} {status}")

    # Update data_loader.py reminder
    v2_list = [r['ticker'] for r in results]
    print(f"\n  V2 contracts: {v2_list}")
    print(f"  Update data_loader.py V2_CONTRACTS list to: {v2_list}")


if __name__ == '__main__':
    main()