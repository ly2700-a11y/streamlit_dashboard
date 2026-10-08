#!/usr/bin/env python3
"""Unit tests for DQN walk-forward training pipeline.

Ensures critical imports, env construction, sanity checks, and health
monitors work correctly so training won't silently fail or produce garbage.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drl_models.dqn.model import DQNAgent
from drl_shared.spec import FEATURE_DIM, MARKET_FEATURE_DIM, SEQ_LEN, WARMUP
from drl_shared.state_space import ContractArrays, ContractEnv


def test_warmup_import():
    """WARMUP must be importable from drl_shared.state_space (used in val_envs)."""
    from drl_shared.state_space import WARMUP
    from drl_shared.spec import WARMUP as SPEC_WARMUP
    assert WARMUP == SPEC_WARMUP, f"WARMUP mismatch: state_space={WARMUP}, spec={SPEC_WARMUP}"
    print(f"✅ WARMUP import: {WARMUP}")


def test_seq_len_import():
    """SEQ_LEN must be importable (used in val_envs start_idx calculation)."""
    from drl_shared.spec import SEQ_LEN
    assert SEQ_LEN == 60, f"SEQ_LEN should be 60, got {SEQ_LEN}"
    print("✅ SEQ_LEN import: 60")


def test_validation_split_import():
    """VALIDATION_SPLIT and EARLY_STOPPING_PATIENCE must be importable from spec."""
    from drl_models.dqn.spec import VALIDATION_SPLIT, EARLY_STOPPING_PATIENCE
    assert 0 < VALIDATION_SPLIT < 1, f"VALIDATION_SPLIT out of range: {VALIDATION_SPLIT}"
    assert EARLY_STOPPING_PATIENCE > 0, f"EARLY_STOPPING_PATIENCE must be positive: {EARLY_STOPPING_PATIENCE}"
    print(f"✅ VALIDATION_SPLIT={VALIDATION_SPLIT}, EARLY_STOPPING_PATIENCE={EARLY_STOPPING_PATIENCE}")


def test_constant_epsilon_schedule():
    """epsilon_for_step should remain constant at 0.3."""
    agent = DQNAgent(device="cpu")
    for step in (0, 1, 100, 50000, 250000):
        eps = agent.epsilon_for_step(step)
        assert eps == 0.3, f"Expected constant epsilon 0.3 at step {step}, got {eps}"
    print("✅ epsilon schedule: constant 0.3")


def test_val_env_construction():
    """val_envs construction must succeed: ContractEnv with start_idx > WARMUP."""
    from drl_models.dqn.spec import VALIDATION_SPLIT, SEQ_LEN
    from drl_shared.spec import SIGMA_TGT_DEFAULT
    from drl_shared.state_space import WARMUP

    n = 2000
    prices = np.cumsum(np.random.randn(n)) + 100.0
    returns = np.diff(prices, prepend=prices[0])
    sigma = np.abs(returns) + 1e-6
    features = np.random.randn(n, MARKET_FEATURE_DIM).astype(np.float32)
    dates = np.arange(n)

    contract = ContractArrays(
        ticker="TEST",
        prices=prices,
        returns=returns,
        sigma=sigma,
        features=features,
        dates=dates,
        source="synthetic",
    )

    split_idx = int(n * (1 - VALIDATION_SPLIT))
    start_idx = max(WARMUP, split_idx - SEQ_LEN)

    env = ContractEnv(contract, sigma_tgt=SIGMA_TGT_DEFAULT, start_idx=start_idx)

    assert env.start_idx >= WARMUP, f"start_idx {env.start_idx} < WARMUP {WARMUP}"
    assert env.start_idx < n, f"start_idx {env.start_idx} >= data length {n}"
    print(f"✅ val_env construction OK: n={n}, split={split_idx}, start_idx={start_idx}")


def test_val_env_start_idx_bounds():
    """Edge case: very short data should still produce valid start_idx."""
    from drl_models.dqn.spec import VALIDATION_SPLIT, SEQ_LEN
    from drl_shared.spec import WARMUP

    n = 500
    split_idx = int(n * (1 - VALIDATION_SPLIT))
    start_idx = max(WARMUP, split_idx - SEQ_LEN)
    expected = max(WARMUP, split_idx - SEQ_LEN)
    assert start_idx == expected, f"Expected {expected}, got {start_idx}"

    n = 300
    split_idx = int(n * (1 - VALIDATION_SPLIT))
    start_idx = max(WARMUP, split_idx - SEQ_LEN)
    expected = max(WARMUP, split_idx - SEQ_LEN)
    assert start_idx == expected, f"Expected {expected}, got {start_idx}"

    print("✅ val_env start_idx bounds correct")


def test_train_envs_no_warmup_needed():
    """train_envs use max_idx=split_idx, no WARMUP dependency."""
    from drl_models.dqn.spec import VALIDATION_SPLIT
    from drl_shared.spec import SIGMA_TGT_DEFAULT, WARMUP

    n = 2000
    prices = np.cumsum(np.random.randn(n)) + 100.0
    returns = np.diff(prices, prepend=prices[0])
    sigma = np.abs(returns) + 1e-6
    features = np.random.randn(n, MARKET_FEATURE_DIM).astype(np.float32)
    dates = np.arange(n)

    contract = ContractArrays(
        ticker="TEST",
        prices=prices,
        returns=returns,
        sigma=sigma,
        features=features,
        dates=dates,
        source="synthetic",
    )

    split_idx = int(n * (1 - VALIDATION_SPLIT))
    env = ContractEnv(contract, sigma_tgt=SIGMA_TGT_DEFAULT, max_idx=split_idx)

    assert env.start_idx >= WARMUP
    assert env.idx >= WARMUP
    print(f"✅ train_env construction OK: max_idx={split_idx}")


def test_all_critical_imports_in_training_module():
    """Verify train_dqn_walkforward.py has all constants it uses at runtime."""
    required = {
        "WARMUP": ("drl_shared.spec", WARMUP),
        "ContractArrays": ("drl_shared.state_space", None),
        "ContractEnv": ("drl_shared.state_space", None),
        "VALIDATION_SPLIT": ("drl.dqn.spec", 0.1),
        "EARLY_STOPPING_PATIENCE": ("drl.dqn.spec", 20),
    }

    missing = []
    for name, (source, expected) in required.items():
        parts = source.rsplit(".", 1)
        src_mod = __import__(source, fromlist=[parts[-1]])
        val = getattr(src_mod, name, None)
        if val is None:
            missing.append(f"{name} not found in {source}")
        elif expected is not None and val != expected:
            missing.append(f"{name}: expected {expected}, got {val}")

    from drl_models.dqn.spec import SEQ_LEN
    if SEQ_LEN != 60:
        missing.append(f"SEQ_LEN: expected 60, got {SEQ_LEN}")

    if missing:
        print("❌ Missing/mismatched imports:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    print("✅ All critical imports verified")


def test_sanity_check_clean_data():
    """_sanity_check_contract returns no warnings for good data."""
    from drl_models.dqn.train.train_dqn_walkforward import _sanity_check_contract

    n = 500
    contract = ContractArrays(
        ticker="GOOD",
        prices=np.cumsum(np.random.randn(n)) + 100.0,
        returns=np.random.randn(n) * 0.01,
        sigma=np.abs(np.random.randn(n)) * 0.01 + 1e-3,
        features=np.random.randn(n, MARKET_FEATURE_DIM).astype(np.float32),
        dates=np.arange(n),
        source="test",
    )
    warns = _sanity_check_contract(contract, "GOOD")
    assert not warns, f"Good data should have no warnings, got: {warns}"
    print("✅ sanity_check: clean data → no warnings")


def test_sanity_check_nan_prices():
    """_sanity_check_contract detects NaN in prices."""
    from drl_models.dqn.train.train_dqn_walkforward import _sanity_check_contract

    n = 500
    bad_prices = np.cumsum(np.random.randn(n)) + 100.0
    bad_prices[100] = np.nan
    contract = ContractArrays(
        ticker="NAN",
        prices=bad_prices,
        returns=np.diff(bad_prices, prepend=bad_prices[0]),
        sigma=np.abs(np.random.randn(n)) * 0.01 + 1e-3,
        features=np.random.randn(n, MARKET_FEATURE_DIM).astype(np.float32),
        dates=np.arange(n),
        source="test",
    )
    warns = _sanity_check_contract(contract, "NAN")
    assert any("NaN" in w for w in warns), f"Should detect NaN, got: {warns}"
    print("✅ sanity_check: NaN in prices → detected")


def test_sanity_check_short_data():
    """_sanity_check_contract detects data shorter than WARMUP."""
    from drl_models.dqn.train.train_dqn_walkforward import _sanity_check_contract

    contract = ContractArrays(
        ticker="SHORT",
        prices=np.ones(20),
        returns=np.zeros(20),
        sigma=np.ones(20) * 0.01,
        features=np.random.randn(20, MARKET_FEATURE_DIM).astype(np.float32),
        dates=np.arange(20),
        source="test",
    )
    warns = _sanity_check_contract(contract, "SHORT")
    assert any("WARMUP" in w for w in warns), f"Should detect short data, got: {warns}"
    print("✅ sanity_check: short data → detected")


def test_sanity_check_wrong_feature_dim():
    """_sanity_check_contract detects wrong feature dimensions."""
    from drl_models.dqn.train.train_dqn_walkforward import _sanity_check_contract

    n = 500
    contract = ContractArrays(
        ticker="WRONG",
        prices=np.cumsum(np.random.randn(n)) + 100.0,
        returns=np.random.randn(n) * 0.01,
        sigma=np.abs(np.random.randn(n)) * 0.01 + 1e-3,
        features=np.random.randn(n, 10).astype(np.float32),  # wrong dim
        dates=np.arange(n),
        source="test",
    )
    warns = _sanity_check_contract(contract, "WRONG")
    assert any("cols" in w for w in warns), f"Should detect wrong feature dim, got: {warns}"
    print("✅ sanity_check: wrong feature dim → detected")


def test_sanity_check_near_zero_sigma():
    """_sanity_check_contract detects near-zero sigma values."""
    from drl_models.dqn.train.train_dqn_walkforward import _sanity_check_contract

    n = 500
    contract = ContractArrays(
        ticker="LOWSIG",
        prices=np.ones(n) * 100.0,
        returns=np.random.randn(n) * 1e-10,
        sigma=np.ones(n) * 1e-12,  # near-zero sigma
        features=np.random.randn(n, MARKET_FEATURE_DIM).astype(np.float32),
        dates=np.arange(n),
        source="test",
    )
    warns = _sanity_check_contract(contract, "LOWSIG")
    assert any("near-zero sigma" in w for w in warns), f"Should detect near-zero sigma, got: {warns}"
    print("✅ sanity_check: near-zero sigma → detected")


def test_sanity_check_non_monotonic_dates():
    """_sanity_check_contract detects non-monotonic dates."""
    from drl_models.dqn.train.train_dqn_walkforward import _sanity_check_contract

    n = 500
    dates = np.arange(n, dtype=float)
    dates[100], dates[101] = dates[101], dates[100]  # swap
    contract = ContractArrays(
        ticker="UNSORT",
        prices=np.cumsum(np.random.randn(n)) + 100.0,
        returns=np.random.randn(n) * 0.01,
        sigma=np.abs(np.random.randn(n)) * 0.01 + 1e-3,
        features=np.random.randn(n, MARKET_FEATURE_DIM).astype(np.float32),
        dates=dates,
        source="test",
    )
    warns = _sanity_check_contract(contract, "UNSORT")
    assert any("non-monotonic" in w for w in warns), f"Should detect unsorted dates, got: {warns}"
    print("✅ sanity_check: non-monotonic dates → detected")


def test_preflight_passes_good_env():
    """_preflight_check_envs returns no errors for valid envs."""
    from drl_models.dqn.train.train_dqn_walkforward import _preflight_check_envs
    from drl_models.dqn.logging_utils import RunLogger

    n = 500
    contract = ContractArrays(
        ticker="OK",
        prices=np.cumsum(np.random.randn(n)) + 100.0,
        returns=np.random.randn(n) * 0.01,
        sigma=np.abs(np.random.randn(n)) * 0.01 + 1e-3,
        features=np.random.randn(n, MARKET_FEATURE_DIM).astype(np.float32),
        dates=np.arange(n),
        source="test",
    )
    env = ContractEnv(contract, sigma_tgt=0.0063, max_idx=450)

    with tempfile.TemporaryDirectory() as td:
        logger = RunLogger("test", "Test", 1, run_id="test", base_dir=Path(td))
        errors = _preflight_check_envs({"OK": env}, {}, {"OK": contract}, logger)
        assert not errors, f"Good env should pass, got: {errors}"
    print("✅ preflight: good env → no errors")


def test_preflight_catches_broken_env():
    """_preflight_check_envs catches envs with 0 usable steps."""
    from drl_models.dqn.train.train_dqn_walkforward import _preflight_check_envs
    from drl_models.dqn.logging_utils import RunLogger

    n = 260  # barely over WARMUP
    contract = ContractArrays(
        ticker="TINY",
        prices=np.cumsum(np.random.randn(n)) + 100.0,
        returns=np.random.randn(n) * 0.01,
        sigma=np.abs(np.random.randn(n)) * 0.01 + 1e-3,
        features=np.random.randn(n, MARKET_FEATURE_DIM).astype(np.float32),
        dates=np.arange(n),
        source="test",
    )
    # start_idx=252, max_idx=252 → 0 usable steps
    env = ContractEnv(contract, sigma_tgt=0.0063, start_idx=252, max_idx=252)

    with tempfile.TemporaryDirectory() as td:
        logger = RunLogger("test", "Test", 1, run_id="test", base_dir=Path(td))
        errors = _preflight_check_envs({"TINY": env}, {}, {"TINY": contract}, logger)
        assert any("0 usable steps" in e for e in errors), f"Should detect 0 steps, got: {errors}"
    print("✅ preflight: 0 usable steps → detected")


def test_training_episode_runs_to_env_termination():
    """_run_training_episode should not truncate before env termination."""
    from drl_models.dqn.train.train_dqn_walkforward import _run_training_episode

    class FakeEnv:
        def __init__(self, total_steps: int):
            self.total_steps = total_steps
            self.steps = 0

        def reset(self):
            self.steps = 0
            return np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32)

        def step(self, action_id):
            _ = action_id
            self.steps += 1
            done = self.steps >= self.total_steps
            return np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32), 0.1, done

    class FakeAgent:
        def epsilon_for_step(self, step):
            _ = step
            return 0.3

        def act(self, state, eps):
            _ = state, eps
            return 1

        def push(self, s, a, r, s2, d):
            _ = s, a, r, s2, d

        def learn(self):
            return 0.0

    env = FakeEnv(total_steps=1705)
    reward, steps, losses, global_step = _run_training_episode(env, FakeAgent(), 0)
    assert steps == 1705, f"Expected full env termination at 1705, got {steps}"
    assert global_step == 1705, f"Expected global_step to advance fully, got {global_step}"
    assert reward > 0
    assert losses == []
    print("✅ training episode: runs to env termination")


def test_validation_reward_runs_full_span():
    """_validation_reward should evaluate the full validation env span."""
    from drl_models.dqn.train.train_dqn_walkforward import _validation_reward

    class FakeEnv:
        def __init__(self, total_steps: int):
            self.total_steps = total_steps
            self.steps = 0

        def reset(self):
            self.steps = 0
            return np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32)

        def step(self, action_id):
            _ = action_id
            self.steps += 1
            done = self.steps >= self.total_steps
            return np.zeros((SEQ_LEN, FEATURE_DIM), dtype=np.float32), 1.0, done

    class DummyNet:
        def eval(self):
            return None

        def train(self):
            return None

    class FakeAgent:
        def __init__(self):
            self.q_net = DummyNet()

        def act(self, state, eps):
            _ = state, eps
            return 1

    envs = {"A": FakeEnv(1601), "B": FakeEnv(1601)}
    reward = _validation_reward(envs, FakeAgent())
    assert reward == 1601.0, f"Expected full-span validation reward 1601.0, got {reward}"
    print("✅ validation reward: runs full span")


def test_training_health_nan_loss():
    """_check_training_health detects NaN loss."""
    from drl_models.dqn.train.train_dqn_walkforward import _check_training_health
    from drl_models.dqn.logging_utils import RunLogger

    with tempfile.TemporaryDirectory() as td:
        logger = RunLogger("test", "Test", 1, run_id="test", base_dir=Path(td))
        agent = DQNAgent(device="cpu")

        warns = _check_training_health(1, [1.0, 2.0], [float('nan'), 0.5], agent, 0, logger)
        assert any("NaN" in w for w in warns), f"Should detect NaN loss, got: {warns}"
        print("✅ training_health: NaN loss → detected")


def test_training_health_extreme_reward():
    """_check_training_health detects extreme rewards."""
    from drl_models.dqn.train.train_dqn_walkforward import _check_training_health
    from drl_models.dqn.logging_utils import RunLogger

    with tempfile.TemporaryDirectory() as td:
        logger = RunLogger("test", "Test", 1, run_id="test", base_dir=Path(td))
        agent = DQNAgent(device="cpu")

        warns = _check_training_health(2, [1e8], [0.1], agent, 0, logger)
        assert any("extreme reward" in w for w in warns), f"Should detect extreme reward, got: {warns}"
        print("✅ training_health: extreme reward → detected")


def test_training_health_normal():
    """_check_training_health returns no warnings for normal training."""
    from drl_models.dqn.train.train_dqn_walkforward import _check_training_health
    from drl_models.dqn.logging_utils import RunLogger

    with tempfile.TemporaryDirectory() as td:
        logger = RunLogger("test", "Test", 1, run_id="test", base_dir=Path(td))
        agent = DQNAgent(device="cpu")

        warns = _check_training_health(3, [1.0, -1.0], [0.01, 0.02], agent, 1000, logger)
        assert not warns, f"Normal training should have no warnings, got: {warns}"
        print("✅ training_health: normal training → no warnings")


def test_training_health_buffer_full_cycle1():
    """_check_training_health warns when replay buffer fills in cycle 1."""
    from drl_models.dqn.train.train_dqn_walkforward import _check_training_health
    from drl_models.dqn.logging_utils import RunLogger
    from drl_models.dqn.spec import MEMORY_SIZE

    with tempfile.TemporaryDirectory() as td:
        logger = RunLogger("test", "Test", 1, run_id="test", base_dir=Path(td))
        agent = DQNAgent(device="cpu")
        # Fill replay buffer
        for _ in range(MEMORY_SIZE + 1):
            agent.push(np.random.randn(SEQ_LEN, FEATURE_DIM), 0, 1.0, np.random.randn(SEQ_LEN, FEATURE_DIM), 0.0)

        warns = _check_training_health(1, [1.0], [0.1], agent, 100, logger)
        assert any("replay buffer full" in w for w in warns), f"Should detect full buffer in cycle 1, got: {warns}"
        print("✅ training_health: buffer full in cycle 1 → detected")


def main():
    tests = [
        # Import checks
        test_warmup_import,
        test_seq_len_import,
        test_validation_split_import,
        # Env construction
        test_val_env_construction,
        test_val_env_start_idx_bounds,
        test_train_envs_no_warmup_needed,
        test_all_critical_imports_in_training_module,
        # Sanity checks
        test_sanity_check_clean_data,
        test_sanity_check_nan_prices,
        test_sanity_check_short_data,
        test_sanity_check_wrong_feature_dim,
        test_sanity_check_near_zero_sigma,
        test_sanity_check_non_monotonic_dates,
        # Preflight
        test_preflight_passes_good_env,
        test_preflight_catches_broken_env,
        # Training health
        test_training_health_nan_loss,
        test_training_health_extreme_reward,
        test_training_health_normal,
        test_training_health_buffer_full_cycle1,
    ]

    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            failed.append((test.__name__, e))
            print(f"❌ {test.__name__}: {e}")

    print(f"\n{'='*50}")
    if failed:
        print(f"FAILED: {len(failed)}/{len(tests)} tests")
        for name, err in failed:
            print(f"  {name}: {err}")
        sys.exit(1)
    else:
        print(f"PASSED: {len(tests)}/{len(tests)} tests")


if __name__ == "__main__":
    main()
