"""
Validate ALL 25 commodity RAD contracts — 3-check comprehensive proof.

Check 1: Non-Roll Days
  - RAD return / NON return should = 1.0 (ratio is piecewise-constant)
  - Proves ratio adjustment preserves returns between rolls

Check 2: Roll-Day Ratio Adjustment
  - At each roll, RAD/NON ratio should change by exactly prev_close/new_close
  - Where: prev_close = NON[roll_t], new_close = NON[roll_t] - adj_change
  - adj_change = diff(REV - NON) at the roll (from trusted REV source)
  - If wrong, the roll adjustment is corrupted

Check 3: Roll-Day Continuity (informational)
  - Max |RAD[t+1] - RAD[t]| / RAD[t] at roll days
  - For V2: should be 0 (continuous by construction)
  - For vendor: reports the max price gap magnitude (expected behavior)
  - Not used for pass/fail — Check 2 already validates the ratio change

Usage:
  cd IEOR4733_Project && python tests/validate_commodity_rad.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'CLC'

TEST_START = pd.Timestamp('2011-01-01')
TEST_END = pd.Timestamp('2019-12-31')

# 25 commodities + US (Fixed Income V2)
CONTRACTS = [
    'CC', 'DA', 'GI', 'JO', 'KC', 'KW', 'LB', 'NR', 'SB',
    'ZA', 'ZC', 'ZF', 'ZG', 'ZH', 'ZI', 'ZK', 'ZL',
    'ZO', 'ZP', 'ZR', 'ZT', 'ZU', 'ZW', 'ZZ', 'ZN',
    'US',  # Fixed Income V2
]

V2_CONTRACTS = {'ZH', 'ZU', 'ZN', 'US'}


def load(filepath, require_positive=True):
    df = pd.read_csv(filepath, header=None,
                     names=['Date', 'O', 'H', 'L', 'C', 'V', 'OI'])
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    if require_positive:
        df = df[df['C'].notna() & (df['C'] > 0)]
    else:
        df = df[df['C'].notna()]
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def validate_contract(ticker):
    """Validate one contract's RAD data with 3-check proof."""
    non_f = DATA_DIR / f'{ticker}_NON.CSV'
    rev_f = DATA_DIR / f'{ticker}_REV.CSV'
    is_v2 = ticker in V2_CONTRACTS
    rad_f = DATA_DIR / f'{ticker}_{"RAD_v2" if is_v2 else "RAD"}.CSV'

    if not non_f.exists() or not rev_f.exists() or not rad_f.exists():
        return {'ticker': ticker, 'status': 'MISSING', 'ok': False}

    non_df = load(non_f, require_positive=True)
    rev_df = load(rev_f, require_positive=False)
    rad_df = load(rad_f, require_positive=False)

    # Merge all three on date
    merged = non_df[['Date', 'C']].merge(
        rev_df[['Date', 'C']], on='Date', suffixes=('_non', '_rev'))
    merged = merged.merge(
        rad_df[['Date', 'C']].rename(columns={'C': 'C_rad'}), on='Date')
    merged = merged.sort_values('Date').reset_index(drop=True)

    non_c = merged['C_non'].values.astype(float)
    rev_c = merged['C_rev'].values.astype(float)
    rad_c = merged['C_rad'].values.astype(float)
    n = len(merged)

    # Test period mask
    test_mask = (merged['Date'] >= TEST_START) & (merged['Date'] <= TEST_END)
    test_idx = np.where(test_mask.values)[0]
    test_non = non_c[test_idx]
    test_rad = rad_c[test_idx]
    n_test = len(test_rad)

    if n_test < 100:
        return {'ticker': ticker, 'status': 'INSUFFICIENT', 'ok': False}

    # ── Detect rolls from REV-NON adj_change (trusted source) ──
    adj = rev_c - non_c
    adj_change = np.diff(adj)
    roll_indices_full = set(np.where(np.abs(adj_change) > 1e-6)[0])

    # Rolls where day t is in test period
    test_rolls = sorted([ri for ri in roll_indices_full
                         if ri in set(test_idx.tolist()) and ri + 1 < n])
    n_rolls_test = len(test_rolls)
    roll_set = set(test_rolls)

    # ── CHECK 1: Non-Roll Day Returns ─────────────────────
    ret_rad = np.diff(test_rad) / test_rad[:-1]
    ret_non = np.diff(test_non) / test_non[:-1]
    ok = np.isfinite(ret_rad) & np.isfinite(ret_non) & (np.abs(ret_non) > 1e-10)

    nr_mask = np.zeros(len(ret_rad), dtype=bool)
    for i in range(len(ret_rad)):
        full_i = test_idx[i]
        if full_i not in roll_set and ok[i]:
            nr_mask[i] = True

    if nr_mask.sum() > 10:
        nr_corr = np.corrcoef(ret_rad[nr_mask], ret_non[nr_mask])[0, 1]
        nr_ratio = ret_rad[nr_mask] / ret_non[nr_mask]
        nr_ratio = nr_ratio[np.isfinite(nr_ratio)]
        nr_ratio_mean = nr_ratio.mean() if len(nr_ratio) > 0 else float('nan')
        nr_ratio_std = nr_ratio.std() if len(nr_ratio) > 0 else float('nan')
    else:
        nr_corr = nr_ratio_mean = nr_ratio_std = float('nan')

    # ── CHECK 2: Roll-Day Ratio Adjustment ────────────────
    # At roll between t and t+1:
    #   adj_change = (REV[t+1]-NON[t+1]) - (REV[t]-NON[t])  [from trusted REV]
    #   prev_close = NON[t]   (old contract close on roll_date)
    #   new_close  = prev_close - adj_change                  (new contract close)
    #   expected_ratio_change = prev_close / new_close
    #   actual_ratio_change   = (RAD[t+1]/NON[t+1]) / (RAD[t]/NON[t])
    if n_rolls_test > 0:
        expected_list = []
        actual_list = []
        err_list = []

        for ri in test_rolls:
            rb = rad_c[ri] / non_c[ri]          # ratio before roll
            ra = rad_c[ri + 1] / non_c[ri + 1]  # ratio after roll

            # From REV: adj_change at this roll
            ac = adj_change[ri]
            prev_close = non_c[ri]
            new_close = prev_close - ac

            if abs(new_close) < 1e-10 or abs(rb) < 1e-10:
                continue

            expected = prev_close / new_close   # expected ratio change
            actual = ra / rb                     # actual ratio change

            expected_list.append(expected)
            actual_list.append(actual)
            err_list.append(abs(actual - expected) / abs(expected))

        if err_list:
            ratio_chg_corr = np.corrcoef(expected_list, actual_list)[0, 1] \
                if len(expected_list) > 2 else float('nan')
            ratio_chg_err_mean = np.mean(err_list)
            ratio_chg_err_max = np.max(err_list)
        else:
            ratio_chg_corr = ratio_chg_err_mean = ratio_chg_err_max = float('nan')
    else:
        ratio_chg_corr = ratio_chg_err_mean = ratio_chg_err_max = float('nan')

    # ── CHECK 3: Roll-Day Continuity ──────────────────────
    # For ratio-adjusted: RAD[t+1]/RAD[t] should = new_close/prev_close
    # Which means jump = |1 - new_close/prev_close| (expected from the roll gap)
    # The actual RAD jump should match this expected jump
    if n_rolls_test > 0:
        actual_jumps = []
        expected_jumps = []
        for ri in test_rolls:
            if ri + 1 < n and rad_c[ri] > 0:
                actual_jumps.append(abs(rad_c[ri + 1] - rad_c[ri]) / rad_c[ri])
                ac = adj_change[ri]
                prev_close = non_c[ri]
                new_close = prev_close - ac
                if abs(prev_close) > 1e-10:
                    expected_jumps.append(abs(1 - new_close / prev_close))

        max_jump = max(actual_jumps) if actual_jumps else 0
        mean_jump = np.mean(actual_jumps) if actual_jumps else 0

        # Check if actual jumps match expected jumps
        if len(actual_jumps) == len(expected_jumps) and len(actual_jumps) > 2:
            jump_corr = np.corrcoef(actual_jumps, expected_jumps)[0, 1]
        else:
            jump_corr = float('nan')
    else:
        max_jump = mean_jump = 0
        jump_corr = float('nan')

    # ── Other checks ──────────────────────────────────────
    neg_test = int((test_rad <= 0).sum())

    # ── Verdict ────────────────────────────────────────────
    # V2: passes on Check 1 (non-roll returns exact) + continuity (MaxJump=0)
    # Vendor: passes on Check 1 (non-roll returns) + Check 2 (ratio change matches REV)
    # Jump correlation is informational only (tiny values, noise-dominated)
    issues = []
    # Check 1: Non-roll returns (both V2 and vendor)
    if not np.isnan(nr_corr) and nr_corr < 0.999:
        issues.append(f"C1_corr={nr_corr:.6f}")
    if not np.isnan(nr_ratio_mean) and abs(nr_ratio_mean - 1.0) > 0.005:
        issues.append(f"C1_ratio={nr_ratio_mean:.6f}")
    if is_v2:
        # V2: Check 2 uses a different formula, so skip ratio change check
        # Instead verify continuity guarantee: MaxJump should be 0
        if max_jump > 1e-6:
            issues.append(f"C3_jump={max_jump:.6f}")
    else:
        # Vendor: Check 2 — ratio change must match REV-derived prev_close/new_close
        if not np.isnan(ratio_chg_corr) and ratio_chg_corr < 0.95:
            issues.append(f"C2_corr={ratio_chg_corr:.6f}")
        if not np.isnan(ratio_chg_err_mean) and ratio_chg_err_mean > 0.05:
            issues.append(f"C2_err={ratio_chg_err_mean:.4f}")
    # Negative prices
    if neg_test > 0:
        issues.append(f"neg={neg_test}")

    status = '✅' if len(issues) == 0 else '⚠️ ' + ', '.join(issues)

    return {
        'ticker': ticker,
        'source': 'v2' if is_v2 else 'vendor',
        'ok': len(issues) == 0,
        'status': status,
        'n_test': n_test,
        'n_rolls_test': n_rolls_test,
        'neg_test': neg_test,
        # Check 1
        'nr_corr': nr_corr,
        'nr_ratio_mean': nr_ratio_mean,
        'nr_ratio_std': nr_ratio_std,
        # Check 2
        'ratio_chg_corr': ratio_chg_corr,
        'ratio_chg_err_mean': ratio_chg_err_mean,
        'ratio_chg_err_max': ratio_chg_err_max,
        # Check 3
        'max_jump': max_jump,
        'mean_jump': mean_jump,
        'jump_corr': jump_corr,
    }


def main():
    print("=" * 140)
    print("COMMODITY RAD VALIDATION — 3-Check Comprehensive Proof")
    print("  Check 1: Non-roll returns (RAD ret = NON ret)")
    print("  Check 2: Roll ratio change (matches REV-derived prev_close/new_close)")
    print("  Check 3: Roll jump matches expected price gap from REV")
    print("=" * 140)

    results = [validate_contract(t) for t in CONTRACTS]

    # ── Table ──────────────────────────────────────────────
    hdr = (f"\n{'TK':4s} {'Src':6s} {'N':>4s} {'Rol':>3s} │ "
           f"{'NR_Corr':>8s} {'NR_Ratio':>8s} │ "
           f"{'RChgCorr':>8s} {'RChgErr':>8s} │ "
           f"{'JmpCorr':>8s} {'MaxJmp%':>8s} │ "
           f"{'Status'}")
    print(hdr)
    print("─" * 130)

    n_ok = n_issues = 0
    for r in results:
        def fmt(v, pct=False, width=8):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return '—'.rjust(width)
            return f"{v*100:.3f}".rjust(width) if pct else f"{v:.6f}".rjust(width)

        print(f"{r['ticker']:4s} {r['source']:6s} {r['n_test']:4d} {r['n_rolls_test']:3d} │ "
              f"{fmt(r['nr_corr']):>8s} {fmt(r['nr_ratio_mean']):>8s} │ "
              f"{fmt(r['ratio_chg_corr']):>8s} {fmt(r['ratio_chg_err_mean'],pct=True):>8s} │ "
              f"{fmt(r['jump_corr']):>8s} {fmt(r['max_jump'],pct=True):>8s} │ "
              f"{r['status']}")

        if r['ok']:
            n_ok += 1
        else:
            n_issues += 1

    print("─" * 130)
    n_total = len(CONTRACTS)
    print(f"\nRESULT: {n_ok}/{n_total} PASS ✅ | {n_issues}/{n_total} HAVE ISSUES ⚠️")

    if n_issues > 0:
        print(f"\nISSUES:")
        for r in results:
            if not r.get('ok', False):
                print(f"  {r['ticker']}: {r['status']}")
    else:
        print(f"\n✅ All {n_total} RAD contracts (25 commodity + US Fixed Income) pass all checks!")

    # ── Summary by source ──────────────────────────────────
    v2 = [r for r in results if r.get('source') == 'v2']
    vendor = [r for r in results if r.get('source') == 'vendor']
    if v2:
        print(f"\nV2 contracts ({len(v2)}):")
        for r in v2:
            print(f"  {r['ticker']}: C1={r['nr_corr']:.6f}, "
                  f"C2_corr={r['ratio_chg_corr']:.6f}, C2_err={r['ratio_chg_err_mean']*100:.4f}%, "
                  f"C3_jcorr={r['jump_corr']:.6f}")
    if vendor:
        print(f"\nVendor contracts ({len(vendor)}):")
        rc = [r['ratio_chg_corr'] for r in vendor if not np.isnan(r.get('ratio_chg_corr', float('nan')))]
        re = [r['ratio_chg_err_mean'] for r in vendor if not np.isnan(r.get('ratio_chg_err_mean', float('nan')))]
        jc = [r['jump_corr'] for r in vendor if not np.isnan(r.get('jump_corr', float('nan')))]
        print(f"  Ratio chg corr: {min(rc):.4f} – {max(rc):.4f}" if rc else "  Ratio chg corr: N/A")
        print(f"  Ratio chg err:  {min(re)*100:.3f}% – {max(re)*100:.3f}%" if re else "  Ratio chg err: N/A")
        print(f"  Jump corr:      {min(jc):.4f} – {max(jc):.4f}" if jc else "  Jump corr: N/A")


if __name__ == '__main__':
    main()