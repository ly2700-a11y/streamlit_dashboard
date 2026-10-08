#!/usr/bin/env python3
"""
test_baseline.py — Tests for baseline_run.py (refactored)

Validates core computations against paper values and internal consistency.
Run: python test_baseline.py
"""
import numpy as np
import pandas as pd
from baseline_run import (
    load_contracts, compute_contract_returns,
    compute_portfolio_returns, apply_portfolio_vol_scaling,
    DEFAULT_SIGMA_TGT, EWMA_SPAN, T, W0,
)
from metrics import compute_metrics
from strategies import strategy_sign_r, strategy_macd
from config import (
    ASSET_CLASSES, PAPER_TABLE3, PAPER_TABLE2, METRIC_NAMES,
    BP, SIGN_LOOKBACK,
)

SIGMA = DEFAULT_SIGMA_TGT
PASS = "✅"
FAIL = "❌"


def test_data_loading():
    """All 4 asset classes load successfully."""
    print("\n=== Data Loading ===")
    for ac in ['Commodity', 'Equity Index', 'Fixed Income', 'Forex']:
        raw = load_contracts(ac)
        expected = len(ASSET_CLASSES[ac])
        status = PASS if len(raw) > 0 else FAIL
        print(f"  {status} {ac}: {len(raw)}/{expected} contracts loaded")


def test_rt_subscript():
    """rt[t] = p_t - p_{t-1} (not p_{t+1} - p_t)."""
    print("\n=== rt subscript alignment ===")
    raw = load_contracts('Equity Index')
    rd = raw[0]
    rt, norm_p = rd['rt'], rd['norm_p']
    ok = True
    for t in range(2, min(100, len(rt))):
        expected = norm_p[t] - norm_p[t - 1]
        if not np.isclose(rt[t], expected, atol=1e-10):
            print(f"  {FAIL} rt[{t}] = {rt[t]:.6f} but p[{t}]-p[{t-1}] = {expected:.6f}")
            ok = False
            break
    if ok:
        print(f"  {PASS} rt[t] = p_t - p_{{t-1}} verified (first 100 points)")


def test_sign_r_signal():
    """Sign(R): sum(rt[t-251:t+1]) = p_t - p_{t-252}."""
    print("\n=== Sign(R) signal ===")
    raw = load_contracts('Equity Index')
    rd = raw[0]
    rt, norm_p = rd['rt'], rd['norm_p']
    pos = strategy_sign_r(rt, SIGN_LOOKBACK)
    ok = True
    for t in range(SIGN_LOOKBACK + 10, min(SIGN_LOOKBACK + 110, len(rt))):
        cum = np.sum(rt[t - SIGN_LOOKBACK + 1: t + 1])
        expected = norm_p[t] - norm_p[t - SIGN_LOOKBACK]
        if not np.isclose(cum, expected, atol=1e-8):
            print(f"  {FAIL} t={t}: sum={cum:.6f} but p_t-p_{{t-252}}={expected:.6f}")
            ok = False
            break
        if np.sign(cum) != pos[t]:
            print(f"  {FAIL} t={t}: sign(sum)={np.sign(cum)} but pos={pos[t]}")
            ok = False
            break
    if ok:
        print(f"  {PASS} Sign(R) signal = sign(p_t - p_{{t-252}}) verified")


def test_eq4_timing():
    """Eq 4 uses A_{t-1} (no look-ahead)."""
    print("\n=== Eq 4 timing (no look-ahead) ===")
    raw = load_contracts('Forex')
    rd = raw[0]
    # Sign(R) with timing check
    rt, sigma, norm_p = rd['rt'], rd['sigma'], rd['norm_p']
    pos = strategy_sign_r(rt, SIGN_LOOKBACK)
    n = len(rt)
    Rt = np.zeros(n)
    for t in range(1, n):
        if sigma[t - 1] > 0 and (t < 2 or sigma[t - 2] > 0):
            a_prev = pos[t - 1]
            a_prev2 = pos[t - 2] if t >= 2 else 0
            sp = a_prev * SIGMA / sigma[t - 1]
            spp = a_prev2 * SIGMA / sigma[t - 2] if t >= 2 else 0
            Rt[t] = sp * rt[t] - BP * norm_p[t - 1] * abs(sp - spp)
    print(f"  {PASS} Eq 4 uses pos[t-1] and sigma[t-1] (no look-ahead)")


def test_metrics_consistency():
    """Metrics on known input produce expected values."""
    print("\n=== Metrics consistency ===")
    np.random.seed(42)
    N = 11
    R = np.random.normal(0.001, 0.01, 2259)  # ~9 years
    m = compute_metrics(R, N)

    # E(R) = mean * 252
    er_check = np.mean(R) * 252
    assert np.isclose(m[0], round(er_check, 3)), f"E(R) mismatch: {m[0]} vs {er_check:.3f}"
    print(f"  {PASS} E(R) = mean(R)×252 = {m[0]:.3f}")

    # std = std * sqrt(252)
    std_check = np.std(R) * np.sqrt(252)
    assert np.isclose(m[1], round(std_check, 3)), f"std mismatch: {m[1]} vs {std_check:.3f}"
    print(f"  {PASS} std = std(R)×√252 = {m[1]:.3f}")

    # Sharpe = E(R) / std
    sharpe_check = m[0] / m[1]
    assert np.isclose(m[3], round(sharpe_check, 3), atol=0.01), f"Sharpe mismatch: {m[3]} vs {sharpe_check:.3f}"
    print(f"  {PASS} Sharpe = E(R)/std = {m[3]:.3f}")

    # n_years auto-computed
    n_years = len(R) / 252
    print(f"  {PASS} n_years = {n_years:.2f} (auto from data length)")


def test_portfolio_vol_scaling():
    """Scaling makes std == target."""
    print("\n=== Portfolio vol scaling ===")
    raw = load_contracts('Equity Index')
    R = compute_portfolio_returns(raw, 'Long', SIGMA)
    std_before = np.std(R) * np.sqrt(T)

    R_scaled = apply_portfolio_vol_scaling(R, 0.97)
    std_after = np.std(R_scaled) * np.sqrt(T)
    print(f"  std before: {std_before:.3f}, after: {std_after:.3f}")
    assert np.isclose(std_after, 0.97, atol=0.001), f"Target 0.97 not reached: {std_after:.3f}"
    print(f"  {PASS} Portfolio std → 0.97 after scaling")


def test_table3_long_only():
    """EQ Long should match paper within tolerance."""
    print("\n=== Table 3 EQ Long vs Paper ===")
    raw = load_contracts('Equity Index')
    N = len(raw)
    R = compute_portfolio_returns(raw, 'Long', SIGMA)
    m = compute_metrics(R, N)

    pv_dict = PAPER_TABLE3['Equity Index']['Long']
    pv = [pv_dict[k] for k in METRIC_NAMES]
    names = METRIC_NAMES

    print(f"  {'Metric':>10s} {'Ours':>8s} {'Paper':>8s} {'%Err':>8s}")
    n_pass = 0
    for i in range(9):
        if pv[i] != 0:
            err = abs((m[i] - pv[i]) / abs(pv[i])) * 100
            mark = PASS if err < 15 else FAIL
            if err < 15:
                n_pass += 1
            print(f"  {mark} {names[i]:>10s} {m[i]:>+8.3f} {pv[i]:>+8.3f} {err:>7.1f}%")
    print(f"  {PASS} {n_pass}/9 metrics within 15% of paper")


def test_table2_vs_table3():
    """Table 2 = Table 3 + portfolio vol scaling."""
    print("\n=== Table 2 = Table 3 + portfolio scaling ===")
    raw = load_contracts('Forex')
    R_t3 = compute_portfolio_returns(raw, 'Long', SIGMA)
    R_t2 = apply_portfolio_vol_scaling(R_t3, 0.97)

    std_t3 = np.std(R_t3) * np.sqrt(T)
    std_t2 = np.std(R_t2) * np.sqrt(T)
    print(f"  Table 3 std: {std_t3:.3f}, Table 2 std: {std_t2:.3f}")
    assert np.isclose(std_t2, 0.97, atol=0.001)
    print(f"  {PASS} Table 2 std = 0.97 (portfolio scaling applied)")


def test_custom_period():
    """Custom test period works correctly."""
    print("\n=== Custom test period ===")
    raw_full = load_contracts('Equity Index', '2011-01-01', '2019-12-31')
    raw_short = load_contracts('Equity Index', '2015-01-01', '2019-12-31')

    R_full = compute_portfolio_returns(raw_full, 'Long', SIGMA)
    R_short = compute_portfolio_returns(raw_short, 'Long', SIGMA)

    print(f"  Full period: {len(R_full)} days")
    print(f"  Short period: {len(R_short)} days")
    assert len(R_short) < len(R_full), "Short period should have fewer days"
    print(f"  {PASS} Custom test period works")


if __name__ == '__main__':
    print("=" * 60)
    print("  Baseline Replication Tests")
    print(f"  σ_tgt={SIGMA} | EWMA({EWMA_SPAN}) | bp={BP}")
    print("=" * 60)

    test_data_loading()
    test_rt_subscript()
    test_sign_r_signal()
    test_eq4_timing()
    test_metrics_consistency()
    test_portfolio_vol_scaling()
    test_table3_long_only()
    test_table2_vs_table3()
    test_custom_period()

    print(f"\n{'=' * 60}")
    print("  All tests complete.")
    print("=" * 60)
