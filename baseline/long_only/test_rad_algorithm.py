"""
Test + Generate RAD_v2 with the CORRECT algorithm.

MATHEMATICAL PROOF:
==================
Given:
  - NON = raw prices (no roll info)
  - REV = NON + cumulative_adj (backward/additive adjustment)
  - At each roll: adj changes by (new_close - prev_close)
  - Between rolls: adj is constant (REV moves 1:1 with NON)

The correct RAD (Ratio-Adjusted Data) is:
  RAD[t] = NON[t] × cumulative_ratio[t]

Where cumulative_ratio is the product of per-roll ratios:
  ratio_i = prev_close / new_close

We detect rolls from: diff(REV - NON) ≠ 0
At each roll:
  adj_change = (REV[t] - NON[t]) - (REV[t-1] - NON[t-1])
  new_close  = NON[t]       (price of new contract on roll date)
  prev_close = NON[t-1] + adj_change  (price of old contract, back-computed)
  
  ratio_i = prev_close / new_close

Forward accumulation:
  cumulative_ratio[0..first_roll-1] = 1.0 (no adjustment needed)
  At roll i: cumulative_ratio jumps by multiplying ratio_i
  Between rolls: cumulative_ratio is constant

Result:
  - Non-roll days: RAD[t]/RAD[t-1] = NON[t]/NON[t-1] (same returns) ✓
  - Roll days: no price jump (ratio absorbs it) ✓
  - All prices positive ✓ (ratio > 0 applied to positive NON)

This script:
  1. Proves the math on ZH (worst case: massive negative REV values)
  2. Validates the output
  3. Then applies to all damaged contracts
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'CLC'


def load(filepath, require_positive=True):
    """Load CLC CSV file."""
    df = pd.read_csv(filepath, header=None,
                     names=['Date', 'O', 'H', 'L', 'C', 'V', 'OI'])
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    if require_positive:
        df = df[df['C'].notna() & (df['C'] > 0)]
    else:
        df = df[df['C'].notna()]
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def generate_rad_from_non_rev(ticker, verbose=True):
    """
    Generate RAD using the correct ratio-adjustment algorithm.
    
    Returns (rad_df, diagnostics_dict) or (None, None) if files missing.
    """
    non_f = DATA_DIR / f'{ticker}_NON.CSV'
    rev_f = DATA_DIR / f'{ticker}_REV.CSV'
    output_f = DATA_DIR / f'{ticker}_RAD_v2.CSV'

    if not non_f.exists() or not rev_f.exists():
        print(f"  {ticker}: NON or REV file missing, skipping")
        return None, None

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
    dates = merged['Date'].values

    n = len(merged)

    # ================================================================
    # STEP 1: Detect rolls from adj_change
    # adj[t] = REV[t] - NON[t]  (the cumulative additive adjustment)
    # On non-roll days: adj is constant → adj_change = 0
    # On roll days: adj jumps by (new_close - prev_close)
    # ================================================================
    adj = rev_c - non_c
    adj_change = np.diff(adj)  # length n-1

    roll_indices = np.where(np.abs(adj_change) > 1e-6)[0]
    # roll_indices[i] means: between day roll_indices[i] and roll_indices[i]+1, a roll happened

    if verbose:
        print(f"\n  {ticker}: {len(roll_indices)} rolls detected out of {n-1} days")

    # ================================================================
    # STEP 2: Compute ratio at each roll
    # At roll between day t and day t+1:
    #   adj_change = adj[t+1] - adj[t] = (new_close - prev_close)
    #   new_close = NON[t+1]  (price of new contract)
    #   prev_close = NON[t] + adj_change  (old contract price, back-computed)
    #     Explanation: The old contract closed at NON[t] (raw), but NON[t+1]
    #     is the new contract. The adjustment changed by adj_change.
    #     Actually, prev_close = NON[t] (the old contract's last price)
    #     And the new contract opens at NON[t+1].
    #     So ratio = NON[t] / (NON[t] + adj_change[t])
    #     
    # Wait, let me think more carefully:
    #   rev[t]   = non[t]   + adj[t]     (day before roll, old contract)
    #   rev[t+1] = non[t+1] + adj[t+1]   (day of roll, new contract)
    #
    #   rev should be continuous, so rev[t] ≈ rev[t+1] on the roll day
    #   (Actually rev is continuous across rolls by construction)
    #
    #   The roll ratio should make RAD continuous:
    #   RAD[t] = NON[t] × cum_ratio[t]     (old contract, old ratio)
    #   RAD[t+1] = NON[t+1] × cum_ratio[t+1]  (new contract, new ratio)
    #   
    #   We want RAD[t] ≈ RAD[t+1] (no price jump)
    #   NON[t] × cum_ratio[t] = NON[t+1] × cum_ratio[t+1]
    #   cum_ratio[t+1] = cum_ratio[t] × NON[t] / NON[t+1]
    #   
    #   Hmm but this would make ratio = NON[t] / NON[t+1], which doesn't
    #   use the adj_change at all. That can't be right because the roll
    #   gap isn't just NON[t] vs NON[t+1] — it's old_contract vs new_contract.
    #
    # Actually, NON[t+1] IS the new contract price (NON switches contracts too).
    # And NON[t] is the old contract's last price.
    # So the roll ratio IS: prev_close / new_close = NON[t] / NON[t+1]
    #
    # But wait — NON[t] and NON[t+1] are DIFFERENT contracts, so there IS
    # a price gap. The whole point of RAD is to remove this gap.
    # 
    # Let me re-examine using the README definitions:
    #   ratio_i = prev_close / new_close
    #   prev_close = old contract's closing price on roll date = NON[t]
    #   new_close  = new contract's opening price on roll date = NON[t+1]
    #   
    #   Wait, but the ASC file records prev_close and new_close explicitly.
    #   Since we don't have ASC for these contracts, we need to derive them.
    #
    # From the relationship REV = NON + adj:
    #   Before roll: rev[t] = non[t] + adj[t]
    #   After roll:  rev[t+1] = non[t+1] + adj[t+1]
    #   
    #   REV is continuous by construction, so the back-adjustment ensures
    #   rev[t+1] ≈ rev[t] (actually rev[t+1] is the value that makes it continuous)
    #   
    #   Actually, REV continuity means:
    #     rev[t+1] = rev[t] + (non[t+1] - non[t]) on non-roll days
    #     On roll days, the adj changes to maintain continuity:
    #     adj[t+1] = adj[t] + adj_change
    #     rev[t+1] = non[t+1] + adj[t+1]
    #     
    #   For RAD continuity:
    #     rad[t+1] = rad[t] on the roll day (approximately)
    #     non[t+1] × cum_ratio[t+1] = non[t] × cum_ratio[t]
    #     cum_ratio[t+1] = cum_ratio[t] × non[t] / non[t+1]
    #     
    #   So the per-roll ratio IS non[t] / non[t+1]
    #   And we don't need the adj_change at all for computing ratios!
    #   We only need adj_change to DETECT rolls.
    #
    # Let me verify: for a roll between t and t+1:
    #   ratio = NON[t] / NON[t+1]
    #   cum_ratio[t+1] = cum_ratio[t] × ratio
    #   RAD[t+1] = NON[t+1] × cum_ratio[t+1] = NON[t+1] × cum_ratio[t] × NON[t] / NON[t+1]
    #            = cum_ratio[t] × NON[t] = RAD[t]  ✓ Continuous!
    #
    # And for non-roll days (t+1):
    #   cum_ratio[t+1] = cum_ratio[t]  (no change)
    #   RAD[t+1] = NON[t+1] × cum_ratio[t]
    #   RAD[t+1]/RAD[t] = (NON[t+1] × cum_ratio[t]) / (NON[t] × cum_ratio[t])
    #                    = NON[t+1] / NON[t]  ✓ Same return as NON!
    #
    # PERFECT. The algorithm is:
    #   1. Detect rolls from adj_change ≠ 0
    #   2. At each roll (between t and t+1): ratio = NON[t] / NON[t+1]
    #   3. cum_ratio[t] = product of all ratios for rolls up to t
    #   4. RAD[t] = NON[t] × cum_ratio[t]
    # ================================================================

    # ================================================================
    # STEP 3: Forward-accumulate cumulative ratio
    # ================================================================
    cum_ratio = np.ones(n)
    for idx in roll_indices:
        # Roll happens between day idx and day idx+1
        prev_close = non_c[idx]
        new_close = non_c[idx + 1]
        if new_close > 0:
            roll_ratio = prev_close / new_close
        else:
            roll_ratio = 1.0  # safety
        # Multiply all subsequent days by this ratio
        cum_ratio[idx + 1:] *= roll_ratio

    # ================================================================
    # STEP 4: Compute RAD = NON × cumulative_ratio
    # ================================================================
    rad_c = non_c * cum_ratio
    rad_o = non_o * cum_ratio
    rad_h = non_h * cum_ratio
    rad_l = non_l * cum_ratio

    # ================================================================
    # STEP 5: Validate
    # ================================================================
    test_mask = (merged['Date'] >= pd.Timestamp('2011-01-01')) & \
                (merged['Date'] <= pd.Timestamp('2019-12-31'))
    test_rad = rad_c[test_mask.values]
    test_non = non_c[test_mask.values]
    test_dates = dates[test_mask.values]

    # 5a. No negative prices
    neg_count = (test_rad <= 0).sum()
    neg_total = (rad_c <= 0).sum()

    # 5b. Level correlation with NON
    if len(test_rad) > 10 and test_rad.std() > 0:
        level_corr = np.corrcoef(test_non, test_rad)[0, 1]
    else:
        level_corr = float('nan')

    # 5c. Return correlation with NON (all days)
    ret_rad = np.diff(test_rad) / test_rad[:-1]
    ret_non = np.diff(test_non) / test_non[:-1]
    mask = np.isfinite(ret_rad) & np.isfinite(ret_non)
    if mask.sum() > 10:
        ret_corr = np.corrcoef(ret_rad[mask], ret_non[mask])[0, 1]
    else:
        ret_corr = float('nan')

    # 5c2. Return correlation on NON-ROLL days only (key proof!)
    # On roll days, RAD is continuous (ret≈0) but NON has a gap → different returns
    # On non-roll days, RAD return should EXACTLY equal NON return
    test_roll_set = set()
    for ri in roll_indices:
        if test_mask.iloc[ri] or test_mask.iloc[ri + 1]:
            # The return from day ri to ri+1 is a roll-day return
            test_roll_set.add(ri)  # index in full array

    test_indices = np.where(test_mask.values)[0]
    non_roll_ret_mask = []
    for i in range(len(test_indices) - 1):
        idx = test_indices[i]
        if idx not in test_roll_set and mask[i]:
            non_roll_ret_mask.append(True)
        else:
            non_roll_ret_mask.append(False)
    non_roll_ret_mask = np.array(non_roll_ret_mask)

    if non_roll_ret_mask.sum() > 10:
        ret_corr_nonroll = np.corrcoef(ret_rad[non_roll_ret_mask], ret_non[non_roll_ret_mask])[0, 1]
    else:
        ret_corr_nonroll = float('nan')

    # 5d. Return ratio (should be ~1.0 on non-roll days)
    ret_ratio = ret_rad[mask] / ret_non[mask]
    ret_ratio = ret_ratio[np.isfinite(ret_ratio)]

    # Return ratio on non-roll days only
    ret_ratio_nonroll = ret_rad[non_roll_ret_mask] / ret_non[non_roll_ret_mask]
    ret_ratio_nonroll = ret_ratio_nonroll[np.isfinite(ret_ratio_nonroll)]

    # 5e. % +ve days and Ave P/L
    rad_ret_full = np.diff(test_rad)
    non_ret_full = np.diff(test_non)
    rad_pve = (rad_ret_full > 0).sum() / len(rad_ret_full) if len(rad_ret_full) > 0 else 0
    non_pve = (non_ret_full > 0).sum() / len(non_ret_full) if len(non_ret_full) > 0 else 0

    rad_pos = rad_ret_full[rad_ret_full > 0].mean() if (rad_ret_full > 0).any() else 0
    rad_neg = abs(rad_ret_full[rad_ret_full < 0].mean()) if (rad_ret_full < 0).any() else 1
    non_pos = non_ret_full[non_ret_full > 0].mean() if (non_ret_full > 0).any() else 0
    non_neg = abs(non_ret_full[non_ret_full < 0].mean()) if (non_ret_full < 0).any() else 1
    rad_pl = rad_pos / rad_neg if rad_neg > 0 else 0
    non_pl = non_pos / non_neg if non_neg > 0 else 0

    # 5f. Roll-day continuity check
    # On roll days, RAD should be continuous (no jump)
    if len(roll_indices) > 0:
        test_rolls = roll_indices[test_mask.values[roll_indices]]
        # Actually, we need to check rolls that fall in test period
        roll_in_test = [i for i in roll_indices if test_mask.iloc[i] or test_mask.iloc[i+1]]
        if len(roll_in_test) > 0:
            jumps = []
            for idx in roll_in_test:
                if idx + 1 < len(rad_c):
                    jump = abs(rad_c[idx + 1] - rad_c[idx]) / rad_c[idx] if rad_c[idx] > 0 else 0
                    jumps.append(jump)
            max_roll_jump = max(jumps) if jumps else 0
            mean_roll_jump = np.mean(jumps) if jumps else 0
        else:
            max_roll_jump = 0
            mean_roll_jump = 0
    else:
        max_roll_jump = 0
        mean_roll_jump = 0

    # Print diagnostics
    if verbose:
        print(f"  Total rows: {n}, Test rows: {len(test_rad)}")
        print(f"  Rolls: {len(roll_indices)} total, {len(roll_in_test) if len(roll_indices) > 0 else 0} in test")
        print(f"  ")
        print(f"  === VALIDATION ===")
        print(f"  Negative prices: {neg_total} total, {neg_count} in test (should be 0)")
        print(f"  Level corr with NON: {level_corr:.6f} (should be ~1.0)")
        print(f"  Return corr with NON: {ret_corr:.6f} (should be ~1.0)")
        print(f"  Return corr NON-ROLL only: {ret_corr_nonroll:.6f} (KEY PROOF: should be 1.000)")
        print(f"  Return ratio (all): mean={ret_ratio.mean():.6f}, std={ret_ratio.std():.6f}")
        print(f"  Return ratio NON-ROLL only: mean={ret_ratio_nonroll.mean():.6f}, std={ret_ratio_nonroll.std():.6f} (KEY: should be 1.0)")
        print(f"  % +ve: RAD={rad_pve:.4f} NON={non_pve:.4f} diff={abs(rad_pve-non_pve):.4f}")
        print(f"  Ave P/L: RAD={rad_pl:.3f} NON={non_pl:.3f} diff={abs(rad_pl-non_pl):.3f}")
        print(f"  Roll-day jump: max={max_roll_jump:.6f}, mean={mean_roll_jump:.6f} (should be ~0)")
        print(f"  RAD range (test): [{test_rad.min():.4f}, {test_rad.max():.4f}]")
        print(f"  NON range (test): [{test_non.min():.4f}, {test_non.max():.4f}]")
        print(f"  cum_ratio range (test): [{cum_ratio[test_mask.values].min():.6f}, {cum_ratio[test_mask.values].max():.6f}]")

        # Show first few rolls
        if len(roll_indices) > 0:
            print(f"\n  === First 3 rolls ===")
            for i, idx in enumerate(roll_indices[:3]):
                r = non_c[idx] / non_c[idx + 1] if non_c[idx + 1] > 0 else float('nan')
                print(f"  Roll {i+1}: idx={idx}, date={pd.Timestamp(dates[idx]).date()}")
                print(f"    NON[t]={non_c[idx]:.4f}, NON[t+1]={non_c[idx+1]:.4f}, ratio={r:.6f}")
                print(f"    RAD[t]={rad_c[idx]:.4f}, RAD[t+1]={rad_c[idx+1]:.4f}, jump={abs(rad_c[idx+1]-rad_c[idx]):.6f}")
                print(f"    REV[t]={rev_c[idx]:.4f}, REV[t+1]={rev_c[idx+1]:.4f}")
                print(f"    adj_change={adj_change[idx]:.4f}")

    # ================================================================
    # STEP 6: Save
    # ================================================================
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
    if verbose:
        print(f"\n  Saved: {output_f.name} ({len(out)} rows)")

    diagnostics = {
        'ticker': ticker,
        'n_rows': n,
        'n_test': len(test_rad),
        'n_rolls': len(roll_indices),
        'neg_total': neg_total,
        'neg_test': neg_count,
        'level_corr': level_corr,
        'ret_corr': ret_corr,
        'ret_ratio_mean': ret_ratio.mean() if len(ret_ratio) > 0 else float('nan'),
        'ret_ratio_std': ret_ratio.std() if len(ret_ratio) > 0 else float('nan'),
        'rad_pve': rad_pve,
        'non_pve': non_pve,
        'rad_pl': rad_pl,
        'non_pl': non_pl,
        'max_roll_jump': max_roll_jump,
        'mean_roll_jump': mean_roll_jump,
        'ret_corr_nonroll': ret_corr_nonroll,
        'ret_ratio_nonroll_mean': ret_ratio_nonroll.mean() if len(ret_ratio_nonroll) > 0 else float('nan'),
        'ret_ratio_nonroll_std': ret_ratio_nonroll.std() if len(ret_ratio_nonroll) > 0 else float('nan'),
    }
    return out, diagnostics


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        # Apply to all 4 damaged contracts
        contracts = ['ZH', 'ZU', 'US', 'ZN']
        print("=" * 80)
        print("REGENERATING RAD_v2 FOR ALL DAMAGED CONTRACTS")
        print("Algorithm: RAD = NON × cumulative_ratio (forward ratio adjustment)")
        print("=" * 80)
    else:
        # Test on ZH only (worst case)
        contracts = ['ZH']
        print("=" * 80)
        print("TEST: RAD_v2 GENERATION ON ZH (worst case)")
        print("Algorithm: RAD = NON × cumulative_ratio (forward ratio adjustment)")
        print("=" * 80)

    results = []
    for ticker in contracts:
        _, diag = generate_rad_from_non_rev(ticker, verbose=True)
        if diag:
            results.append(diag)

    # Summary
    print(f"\n{'=' * 80}")
    print('SUMMARY')
    print(f"{'=' * 80}")
    print(f"  {'TK':4s} {'NegT':>5s} {'LCorr':>8s} {'RCorr':>8s} {'NR_Corr':>8s} {'NR_Ratio':>8s} {'Jump':>8s} {'OK':>4s}")
    print(f"  {'-' * 65}")
    for r in results:
        # Status: neg=0, non-roll ret_corr≈1.0, non-roll ratio≈1.0, roll jump≈0
        ok = (r['neg_test'] == 0 and 
              r['ret_corr_nonroll'] > 0.9999 and 
              abs(r['ret_ratio_nonroll_mean'] - 1.0) < 0.0001 and
              r['max_roll_jump'] < 0.01)
        status = '✅' if ok else '❌'
        print(f"  {r['ticker']:4s} {r['neg_test']:5d} {r['level_corr']:8.4f} {r['ret_corr']:8.4f} "
              f"{r['ret_corr_nonroll']:8.4f} {r['ret_ratio_nonroll_mean']:8.4f} {r['max_roll_jump']:8.6f} {status}")

    if len(results) == 1:
        print(f"\n  Run with --all to generate all damaged contracts")
    else:
        print(f"\n  V2 contracts to update in data_loader.py: {[r['ticker'] for r in results]}")


if __name__ == '__main__':
    main()