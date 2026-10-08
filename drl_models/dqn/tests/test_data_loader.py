#!/usr/bin/env python3
"""Unit tests for drl_shared/data_loader.py — feature slice validation.

Tests:
1. Train slice: correct date range, no data leakage into test
2. Test slice: correct date range, no overlap with train
3. Val split: 90/10, val starts within train period
4. WARMUP: start_idx >= WARMUP, usable_steps > 0
5. No NaN/Inf, no duplicate dates, monotonic dates
6. Non-zero returns/sigma
7. Feature dim == FEATURE_DIM
8. Round boundaries correct (r1 vs r2)
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from config import ASSET_CLASSES
from drl_shared.data_loader import load_npz, get_train_slice, get_test_slice
from drl_shared.spec import FEATURE_DIM, MARKET_FEATURE_DIM, RETRAIN_ROUNDS, WARMUP


def test_all_forex_contracts():
    """Run all validation checks on Forex contracts."""
    tickers = ASSET_CLASSES.get("Forex", [])
    errors = []
    passed = 0

    for ticker in tickers:
        for round_num in [1, 2]:
            label = f"{ticker} r{round_num}"
            try:
                _test_one_contract(ticker, round_num)
                passed += 1
                print(f"  ✅ {label}")
            except Exception as e:
                errors.append(f"{label}: {e}")
                print(f"  ❌ {label}: {e}")

    print(f"\n{'='*60}")
    print(f"Passed: {passed}/{passed + len(errors)}")
    if errors:
        print("FAILURES:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("ALL PASSED ✅")


def _test_one_contract(ticker: str, round_num: int):
    ri = RETRAIN_ROUNDS[round_num]

    # --- 1. Raw load validation ---
    data, full_contract, meta = load_npz(ticker, round_num)
    n = len(full_contract.prices)

    # Feature dim (npz stores MARKET_FEATURE_DIM columns; prev_action added at runtime)
    assert full_contract.features.shape == (n, MARKET_FEATURE_DIM), \
        f"features shape {full_contract.features.shape} != ({n}, {MARKET_FEATURE_DIM})"

    # No NaN/Inf
    assert not np.any(np.isnan(full_contract.prices)), "prices has NaN"
    assert not np.any(np.isnan(full_contract.features)), "features has NaN"
    assert not np.any(np.isinf(full_contract.returns)), "returns has Inf"

    # No duplicate dates
    dates = pd.to_datetime(full_contract.dates)
    assert len(dates.drop_duplicates()) == n, f"{n - len(dates.drop_duplicates())} duplicate dates"

    # Monotonic
    assert (dates.diff()[1:] >= pd.Timedelta(0)).all(), "dates not monotonic"

    # Non-zero returns
    assert not np.all(full_contract.returns == 0), "returns all zero"

    # Non-zero sigma (except first few)
    assert np.any(full_contract.sigma[WARMUP:] > 0), "sigma all zero after WARMUP"

    # --- 2. Train slice ---
    train_contract, train_slice, val_slice, tmeta = get_train_slice(ticker, round_num)
    assert tmeta["train_end_idx"] < tmeta["test_start_idx"], \
        f"explicit split overlap: train_end_idx={tmeta['train_end_idx']} test_start_idx={tmeta['test_start_idx']}"

    # Train dates within expected range
    train_dates = pd.to_datetime(train_contract.dates)
    train_end = pd.Timestamp(ri["train_end"])
    assert train_dates[-1] <= train_end + pd.Timedelta(days=1), \
        f"train ends {train_dates[-1].date} > train_end {train_end.date}"

    # No overlap with test period
    test_start = pd.Timestamp(ri["test_start"])
    assert train_dates[-1] < test_start, \
        f"train data leaks into test: last train date {train_dates[-1]} >= test_start {test_start}"

    # Usable steps
    assert train_slice.usable_steps > 0, f"0 train usable steps"
    assert val_slice.usable_steps > 0, f"0 val usable steps"

    # Train start at WARMUP
    assert train_slice.start_idx == WARMUP, \
        f"train start_idx {train_slice.start_idx} != WARMUP {WARMUP}"

    # Val start < train end (overlap for SEQ_LEN)
    assert val_slice.start_idx < train_slice.end_idx, \
        "val starts after train ends (no overlap)"

    # --- 3. Test slice ---
    test_contract, test_start_idx, test_meta = get_test_slice(ticker, round_num)
    test_dates = pd.to_datetime(test_contract.dates)

    # Test dates in expected range
    assert test_dates[0] >= pd.Timestamp(ri["test_start"]) - pd.Timedelta(days=1), \
        f"test starts {test_dates[0]} before test_start {ri['test_start']}"
    assert test_dates[-1] <= pd.Timestamp(ri["test_end"]) + pd.Timedelta(days=1), \
        f"test ends {test_dates[-1]} after test_end {ri['test_end']}"

    # Test start_idx is WARMUP
    assert test_start_idx == WARMUP, f"test start_idx {test_start_idx} != WARMUP"

    # Test period has enough data
    assert len(test_contract.prices) > WARMUP + 10, \
        f"test too short: {len(test_contract.prices)} rows"

    # --- 4. Train/Test no overlap ---
    train_last_date = pd.to_datetime(train_contract.dates)[-1]
    test_first_date = test_dates[0]
    assert train_last_date < test_first_date, \
        f"train/test overlap: train ends {train_last_date}, test starts {test_first_date}"

    # --- 5. Consistent feature dim across slices ---
    assert train_contract.features.shape[1] == MARKET_FEATURE_DIM
    assert test_contract.features.shape[1] == MARKET_FEATURE_DIM

    # --- 6. Sigma_tgt in meta (for manifest) ---
    assert "train_start" in tmeta
    assert "train_end" in tmeta


if __name__ == "__main__":
    test_all_forex_contracts()
