#!/usr/bin/env python3
"""Asset-class walk-forward DQN training."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import ASSET_CLASSES, BP
from drl_models.dqn.logging_utils import RunLogger, make_run_id
from drl_models.dqn.model import DQNAgent
from drl_models.dqn.spec import (
    BATCH_SIZE,
    DISCRETE_ACTION_DIM,
    EARLY_STOPPING_PATIENCE,
    EPISODES,
    EPS_SCHEDULE,
    FEATURE_DIM,
    MAX_STEPS_PER_EP,
    MEMORY_RATIO,
    MEMORY_SIZE_MIN,
    MODEL_ROOT,
    RETRAIN_ROUNDS,
    UPDATE_FREQ,
    VALIDATION_SPLIT,
    asset_slug,
    checkpoint_metadata,
    checkpoint_path_for_bundle,
    contract_data_path,
    model_bundle_root,
    round_name,
)
from drl_shared.spec import (
    SEQ_LEN,
    SIGMA_TGT_DEFAULT,
    asset_index_path,
    current_source_policy,
    feature_spec,
    universe_tickers,
)

from drl_shared.state_space import (
    WARMUP,
    ContractArrays,
    ContractEnv,
    action_id_to_position,
    compute_eq4_reward,
    get_feature_window,
)

def _npz_scalar(data, key: str, default=None):
    if key not in data:
        return default
    value = data[key]
    if getattr(value, "shape", None) == ():
        return value.item()
    return value


def _derive_round_split_indices_from_dates(dates: np.ndarray, round_info: dict[str, str]) -> dict[str, int]:
    dt = np.asarray(dates)
    train_start = np.datetime64(round_info["train_start"])
    train_end = np.datetime64(round_info["train_end"])
    test_start = np.datetime64(round_info["test_start"])
    test_end = np.datetime64(round_info["test_end"])
    train_start_idx_arr = np.where(dt >= train_start)[0]
    train_idx = np.where(dt <= train_end)[0]
    test_idx = np.where((dt >= test_start) & (dt <= test_end))[0]
    if len(train_start_idx_arr) == 0:
        raise ValueError(f"No rows found on/after train_start={round_info['train_start']}")
    if len(train_idx) == 0:
        raise ValueError(f"No rows found on/before train_end={round_info['train_end']}")
    train_start_idx = int(train_start_idx_arr[0])
    train_end_idx = int(train_idx[-1])
    if len(test_idx) == 0:
        test_start_idx = len(dt)
        test_end_idx = len(dt) - 1
    else:
        test_start_idx = int(test_idx[0])
        test_end_idx = int(test_idx[-1])
    return {
        "train_start_idx": train_start_idx,
        "train_end_idx": train_end_idx,
        "test_start_idx": test_start_idx,
        "test_end_idx": test_end_idx,
    }


def parse_rounds(value: str | int | None) -> list[int]:
    if value is None or value == "" or str(value).lower() == "both":
        return sorted(RETRAIN_ROUNDS)
    round_num = int(value)
    if round_num not in RETRAIN_ROUNDS:
        raise ValueError(f"Unknown retrain round: {value}")
    return [round_num]


def load_contract_round(ticker: str, round_num: int) -> tuple[ContractArrays, dict]:
    ticker = ticker.upper()
    path = contract_data_path(round_num, ticker)
    if not path.exists():
        raise FileNotFoundError(
            f"No prepared shared feature data found at {path}. "
            "Run python drl_shared/prepare_features.py first."
        )
    data = np.load(path, allow_pickle=True)
    expected = feature_spec()
    actual_state = _npz_scalar(data, "state_spec_version")
    if actual_state != expected["state_spec_version"]:
        raise ValueError(
            f"Feature spec mismatch for {ticker} r{round_num}: "
            f"{actual_state!r} != {expected['state_spec_version']!r}. "
            "Regenerate features with drl_shared/prepare_features.py."
        )
    meta = {
        "feature_artifact_path": str(path),
        "feature_spec": json.loads(str(_npz_scalar(data, "feature_spec", "{}"))),
        "state_spec_version": actual_state,
        "feature_line": str(_npz_scalar(data, "feature_line", expected["feature_line"])),
        "preset": _npz_scalar(data, "preset", None),
        "source_overrides": json.loads(str(_npz_scalar(data, "source_overrides", "{}"))),
        "excluded_contracts": json.loads(str(_npz_scalar(data, "excluded_contracts", "[]"))),
        "train_start": str(_npz_scalar(data, "train_start", "")),
        "train_end": str(_npz_scalar(data, "train_end", "")),
        "test_start": str(_npz_scalar(data, "test_start", "")),
        "test_end": str(_npz_scalar(data, "test_end", "")),
        "train_start_idx": _npz_scalar(data, "train_start_idx", None),
        "train_end_idx": _npz_scalar(data, "train_end_idx", None),
        "test_start_idx": _npz_scalar(data, "test_start_idx", None),
        "test_end_idx": _npz_scalar(data, "test_end_idx", None),
    }
    contract = ContractArrays(
        ticker=ticker,
        prices=data["prices"],
        returns=data["returns"],
        sigma=data["sigma"],
        features=data["features"],
        dates=data["dates"],
        source=str(_npz_scalar(data, "source", "")),
    )
    if any(meta[key] is None for key in ("train_start_idx", "train_end_idx", "test_start_idx", "test_end_idx")):
        split_meta = _derive_round_split_indices_from_dates(contract.dates, RETRAIN_ROUNDS[round_num])
        meta.update(split_meta)
    else:
        for key in ("train_start_idx", "train_end_idx", "test_start_idx", "test_end_idx"):
            meta[key] = int(meta[key])
    return contract, meta


def _slice_train_contract(contract: ContractArrays, meta: dict) -> tuple[ContractArrays, int, int, int]:
    train_start_idx = int(meta["train_start_idx"])
    train_end_idx = int(meta["train_end_idx"])
    if train_start_idx < 0:
        raise ValueError(f"Expected non-negative train_start_idx, got {train_start_idx}")
    if train_end_idx < WARMUP:
        raise ValueError(f"Train period too short after warmup: train_end_idx={train_end_idx}, WARMUP={WARMUP}")

    train_contract = ContractArrays(
        ticker=contract.ticker,
        prices=contract.prices[:train_end_idx + 1],
        returns=contract.returns[:train_end_idx + 1],
        sigma=contract.sigma[:train_end_idx + 1],
        features=contract.features[:train_end_idx + 1],
        dates=contract.dates[:train_end_idx + 1],
        source=contract.source,
    )
    n_train_period = train_end_idx - train_start_idx + 1
    split_idx = train_start_idx + int(n_train_period * (1 - VALIDATION_SPLIT))
    train_env_start = max(WARMUP, train_start_idx)
    val_start = max(train_env_start, split_idx - SEQ_LEN)
    if split_idx <= train_env_start:
        raise ValueError(
            f"Train split too short after warmup: n_train_period={n_train_period}, "
            f"split_idx={split_idx}, train_env_start={train_env_start}"
        )
    return train_contract, train_env_start, split_idx, val_start


def _asset_tickers_from_index(asset_name: str, round_num: int) -> list[str]:
    index_path = asset_index_path(asset_name, round_num)
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return [str(t).upper() for t in payload.get("member_tickers", [])]
    policy = current_source_policy()
    excluded = set(policy["excluded_contracts"])
    return [t for t in universe_tickers(asset_name) if t.upper() not in excluded]


def _sanity_check_contract(contract: ContractArrays, ticker: str) -> list[str]:
    """Comprehensive data sanity checks. Returns list of warnings (empty = all OK)."""
    warnings = []
    n = len(contract.prices)
    from drl_models.dqn.spec import FEATURE_DIM as FDIM
    from drl_shared.spec import MARKET_FEATURE_DIM

    # --- Length checks ---
    if n <= WARMUP:
        warnings.append(f"data length {n} <= WARMUP {WARMUP}, env will have no usable steps")
    for attr in ("prices", "returns", "sigma", "features", "dates"):
        arr = getattr(contract, attr)
        if len(arr) != n:
            warnings.append(f"{attr} length {len(arr)} != prices length {n}")

    # --- Shape check ---
    if contract.features.ndim != 2:
        warnings.append(f"features has {contract.features.ndim} dims, expected 2 (n, {MARKET_FEATURE_DIM})")
    elif contract.features.shape[1] != MARKET_FEATURE_DIM:
        warnings.append(f"features has {contract.features.shape[1]} cols, expected {MARKET_FEATURE_DIM}")

    # --- NaN / Inf checks ---
    for attr in ("prices", "returns", "sigma"):
        arr = getattr(contract, attr)
        if np.any(np.isnan(arr)):
            warnings.append(f"{attr} has {np.isnan(arr).sum()} NaN values")
        if np.any(np.isinf(arr)):
            warnings.append(f"{attr} has {np.isinf(arr).sum()} Inf values")

    if np.any(np.isnan(contract.features)):
        warnings.append(f"features has {np.isnan(contract.features).sum()} NaN values")
    if np.any(np.isinf(contract.features)):
        warnings.append(f"features has {np.isinf(contract.features).sum()} Inf values")

    # --- Sigma health (critical for reward scaling) ---
    if np.all(contract.sigma == 0):
        warnings.append("sigma is all zeros — division by zero in reward scaling")
    else:
        valid_sigma = contract.sigma[contract.sigma > 0]
        if len(valid_sigma) > 0:
            s_min, s_max, s_mean = valid_sigma.min(), valid_sigma.max(), valid_sigma.mean()
            if s_min < 1e-8:
                warnings.append(f"sigma min={s_min:.2e} — near-zero sigma will blow up reward scaling")
            if s_max / max(s_mean, 1e-10) > 100:
                warnings.append(f"sigma range extreme: min={s_min:.4f} max={s_max:.4f} mean={s_mean:.4f}")

    # --- Returns distribution ---
    if np.all(contract.returns == 0):
        warnings.append("returns are all zeros")
    else:
        ret_abs_max = np.max(np.abs(contract.returns))
        ret_std = np.std(contract.returns)
        if ret_std > 0 and ret_abs_max / ret_std > 100:
            warnings.append(f"returns have extreme outliers: max_abs={ret_abs_max:.4f} std={ret_std:.6f} ratio={ret_abs_max/ret_std:.0f}")

    # --- Date ordering ---
    if n > 1:
        dates = contract.dates
        # Handle both numeric and string dates
        try:
            if np.issubdtype(dates.dtype, np.number):
                not_mono = np.sum(np.diff(dates) <= 0)
            else:
                not_mono = np.sum(np.array(dates[:-1] >= dates[1:]))
            if not_mono > 0:
                warnings.append(f"{not_mono} non-monotonic date transitions")
        except TypeError:
            pass  # non-comparable date format, skip

    # --- Duplicate dates ---
    if len(np.unique(contract.dates)) != n:
        dupes = n - len(np.unique(contract.dates))
        warnings.append(f"{dupes} duplicate dates")

    return warnings


def _preflight_check_envs(
    train_envs: dict[str, ContractEnv],
    val_envs: dict[str, ContractEnv],
    contracts: dict[str, ContractArrays],
    logger: RunLogger,
) -> list[str]:
    """Check envs can produce valid states/rewards before training starts."""
    errors = []

    for ticker, env in train_envs.items():
        usable = env.max_idx - env.start_idx
        if usable <= 0:
            errors.append(f"{ticker} train_env: 0 usable steps (start={env.start_idx}, max={env.max_idx})")

        # Verify first feature window is valid
        try:
            state = env.reset()
            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                errors.append(f"{ticker} train_env: initial state has NaN/Inf")
            if state.shape != (SEQ_LEN, FEATURE_DIM):
                errors.append(f"{ticker} train_env: state shape {state.shape} != ({SEQ_LEN}, {FEATURE_DIM})")
        except Exception as e:
            errors.append(f"{ticker} train_env: reset() failed — {e}")

        # Verify first step produces valid reward
        try:
            _, reward, done = env.step(1)  # long position
            if np.isnan(reward) or np.isinf(reward):
                errors.append(f"{ticker} train_env: first step reward is NaN/Inf ({reward})")
            if abs(reward) > 1e6:
                errors.append(f"{ticker} train_env: first step reward extreme ({reward:.2f})")
        except Exception as e:
            errors.append(f"{ticker} train_env: step() failed — {e}")

    for ticker, env in val_envs.items():
        usable = env.max_idx - env.start_idx
        if usable <= 0:
            errors.append(f"{ticker} val_env: 0 usable steps (start={env.start_idx}, max={env.max_idx})")

    if errors:
        logger.log("--- Preflight FAILED ---")
        for e in errors:
            logger.log(f"  ❌ {e}")
    else:
        logger.log("--- Preflight: all envs OK (reset + step + reward) ---")

    return errors


def _check_agent_health(agent: DQNAgent, logger: RunLogger) -> None:
    """Log agent architecture info after construction."""
    import torch
    n_params = sum(p.numel() for p in agent.q_net.parameters())
    trainable = sum(p.numel() for p in agent.q_net.parameters() if p.requires_grad)
    logger.log(f"Agent: {agent.q_net.__class__.__name__} | params={n_params:,} (trainable={trainable:,})")
    logger.log(f"Device: {agent.device} | CUDA available: {torch.cuda.is_available()}")
    if agent.device == "cuda" and torch.cuda.is_available():
        logger.log(f"GPU: {torch.cuda.get_device_name(0)} | mem={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")


def _check_training_health(
    cycle: int,
    cycle_rewards: list[float],
    cycle_losses: list[float],
    agent: DQNAgent,
    global_step: int,
    logger: RunLogger,
) -> list[str]:
    """Per-cycle training health checks. Returns warnings."""
    import torch
    warnings = []

    # --- Reward checks ---
    if cycle_rewards:
        r_max = max(abs(r) for r in cycle_rewards)
        if r_max > 1e6:
            warnings.append(f"cycle {cycle}: extreme reward |max|={r_max:.0f}")
        if all(r == 0.0 for r in cycle_rewards):
            warnings.append(f"cycle {cycle}: all rewards are exactly 0")

    # --- Loss checks ---
    if cycle_losses:
        if any(np.isnan(l) for l in cycle_losses):
            warnings.append(f"cycle {cycle}: NaN loss detected — training may be diverging")
        if any(l > 1e4 for l in cycle_losses):
            warnings.append(f"cycle {cycle}: extreme loss max={max(cycle_losses):.0f}")

    # --- Q-value health (sample check) ---
    try:
        # Use a dummy state to check Q-value range
        with torch.no_grad():
            dummy = torch.randn(1, SEQ_LEN, FEATURE_DIM).to(agent.device)
            q_vals = agent.q_net(dummy)
            q_min, q_max = float(q_vals.min()), float(q_vals.max())
            if np.isnan(q_min) or np.isnan(q_max):
                warnings.append(f"cycle {cycle}: Q-values are NaN")
            elif abs(q_max) > 1e6 or abs(q_min) > 1e6:
                warnings.append(f"cycle {cycle}: Q-values extreme [{q_min:.0f}, {q_max:.0f}]")
    except Exception:
        pass  # non-critical, don't crash training

    # --- Epsilon sanity ---
    eps = agent.epsilon_for_step(global_step)
    if eps < 0.01 and cycle <= 2:
        warnings.append(f"cycle {cycle}: epsilon already at {eps:.4f} (step={global_step}) — was this resumed?")

    # --- Replay buffer ---
    buf_size = len(agent.replay)
    buf_cap = agent.replay.capacity
    if buf_size >= buf_cap and cycle <= 1:
        warnings.append(f"cycle {cycle}: replay buffer full ({buf_size}/{buf_cap}) in first cycle")

    if warnings:
        for w in warnings:
            logger.log(f"  ⚠️  {w}")

    return warnings


def _validate_feature_policy(feature_meta: dict, ticker: str):
    expected_policy = current_source_policy()
    resolved_preset = feature_meta.get("preset") or expected_policy["preset"]
    if resolved_preset != expected_policy["preset"]:
        raise ValueError(
            f"Mainline DQN requires preset={expected_policy['preset']!r}, got {resolved_preset!r}"
        )
    if feature_meta.get("source_overrides") != expected_policy["source_overrides"]:
        raise ValueError(f"Feature artifact source_overrides do not match structural_38 for {ticker}")
    if sorted(feature_meta.get("excluded_contracts", [])) != sorted(expected_policy["excluded_contracts"]):
        raise ValueError(f"Feature artifact excluded_contracts do not match structural_38 for {ticker}")

def _run_training_episode(env: ContractEnv, agent: DQNAgent, global_step: int) -> tuple[float, int, list[float], int]:
    """Run a full sequential episode for one contract (legacy, kept for reference)."""
    state = env.reset()
    total_reward = 0.0
    losses: list[float] = []
    done = False
    steps = 0
    while not done and steps < MAX_STEPS_PER_EP:
        eps = agent.epsilon_for_step(global_step)
        action_id = agent.act(state, eps)
        next_state, reward, done = env.step(action_id)
        agent.push(state, action_id, reward, next_state, float(done))
        loss = agent.learn()
        if loss > 0:
            losses.append(loss)
        state = next_state
        total_reward += reward
        steps += 1
        global_step += 1
    return float(total_reward), steps, losses, global_step


def _run_interleaved_cycle(
    train_envs: dict[str, ContractEnv],
    agent: DQNAgent,
    global_step: int,
    contract_stats: dict,
) -> tuple[dict[str, float], dict[str, int], list[float], int]:
    """Run one cycle with step-interleaved training across all contracts.

    Instead of running full episodes sequentially (all steps of contract A,
    then all of B, etc.), this interleaves one step at a time across contracts.
    This ensures the replay buffer always contains a balanced mix of experience.

    Returns:
        per_contract_rewards: {ticker: total_reward}
        per_contract_steps: {ticker: step_count}
        all_losses: list of all loss values
        global_step: updated global step counter
    """
    buffer_fill_threshold = agent.replay.capacity  # fill buffer to capacity before learning

    # Reset all envs and initialize tracking
    active_states: dict[str, np.ndarray] = {}
    active_tickers: list[str] = []
    per_contract_rewards: dict[str, float] = {}
    per_contract_steps: dict[str, int] = {}

    for ticker, env in train_envs.items():
        active_states[ticker] = env.reset()
        active_tickers.append(ticker)
        per_contract_rewards[ticker] = 0.0
        per_contract_steps[ticker] = 0

    all_losses: list[float] = []
    transition_count = 0  # count transitions for UPDATE_FREQ

    while active_tickers:
        # Batched action selection across all active contracts
        tickers_now = list(active_tickers)
        n_active = len(tickers_now)

        if global_step < buffer_fill_threshold:
            # Phase 1: pure random
            actions = np.random.randint(DISCRETE_ACTION_DIM, size=n_active).tolist()
        else:
            eps = agent.epsilon_for_step(global_step)
            states = np.stack([active_states[t] for t in tickers_now])
            actions = agent.act_batch(states, eps)

        # Step each env independently
        to_remove = []
        for i, ticker in enumerate(tickers_now):
            action_id = actions[i]
            state = active_states[ticker]

            next_state, reward, done = train_envs[ticker].step(action_id)
            agent.push(state, action_id, reward, next_state, float(done))

            active_states[ticker] = next_state
            per_contract_rewards[ticker] += reward
            per_contract_steps[ticker] += 1
            global_step += 1
            transition_count += 1

            # Only learn after buffer is filled
            if global_step >= buffer_fill_threshold and transition_count % UPDATE_FREQ == 0:
                loss = agent.learn()
                if loss > 0:
                    all_losses.append(loss)

            if done:
                to_remove.append(ticker)

        for t in to_remove:
            active_tickers.remove(t)

    return per_contract_rewards, per_contract_steps, all_losses, global_step


def _is_degenerate(val_action_counts: dict[int, int], threshold: float = 0.95) -> bool:
    total = sum(val_action_counts.values())
    if total == 0:
        return True
    return max(val_action_counts.values()) / total >= threshold


def _validation_reward(envs: dict[str, ContractEnv], agent: DQNAgent) -> tuple[float, dict[int, int]]:
    """Run one full validation episode per contract with greedy policy, return avg reward and action counts."""
    agent.q_net.eval()  # eval mode: disable dropout
    rewards = []
    action_counts = {0: 0, 1: 0, 2: 0}
    for ticker, env in envs.items():
        state = env.reset()
        total = 0.0
        done = False
        steps = 0
        while not done and steps < MAX_STEPS_PER_EP:
            action_id = agent.act(state, 0.0)  # greedy
            action_counts[action_id] += 1
            state, reward, done = env.step(action_id)
            total += reward
            steps += 1
        rewards.append(total)
    agent.q_net.train()  # back to train mode
    return float(np.mean(rewards)), action_counts


def train_asset_round(
    asset_name: str,
    round_num: int,
    episodes: int = EPISODES,
    device: str | None = None,
    seed: int | None = None,
    gamma: float | None = None,
    sigma_tgt: float = SIGMA_TGT_DEFAULT,
    bp: float = BP,
    resume: bool = False,
    tickers_override: list[str] | None = None,
) -> tuple[Path, Path]:
    t0_start = time.time()

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if gamma is not None:
        import drl_models.dqn.spec as _spec
        _spec.GAMMA = gamma

    tickers = tickers_override if tickers_override else _asset_tickers_from_index(asset_name, round_num)
    if not tickers:
        raise ValueError(f"No eligible contracts found for {asset_name} {round_name(round_num)}")

    contracts: dict[str, ContractArrays] = {}
    train_envs: dict[str, ContractEnv] = {}
    val_envs: dict[str, ContractEnv] = {}
    feature_meta_by_ticker: dict[str, dict] = {}
    skipped: list[str] = []
    sanity_warnings: dict[str, list[str]] = {}
    env_log_lines: list[str] = []  # collect before logger exists

    for ticker in tickers:
        try:
            full_contract, feature_meta = load_contract_round(ticker, round_num)
            _validate_feature_policy(feature_meta, ticker)
            contract, train_env_start, split_idx, val_start = _slice_train_contract(full_contract, feature_meta)

            # Feature sanity check
            warns = _sanity_check_contract(contract, ticker)
            if warns:
                sanity_warnings[ticker] = warns

            contracts[ticker] = contract
            feature_meta_by_ticker[ticker] = feature_meta
            n = len(contract.prices)

            # Train env
            train_envs[ticker] = ContractEnv(contract, sigma_tgt=sigma_tgt, bp=bp, start_idx=train_env_start, max_idx=split_idx)
            env_log_lines.append(
                f"  {ticker}: train_dates=[{contract.dates[int(feature_meta['train_start_idx'])]}..{contract.dates[-1]}] "
                f"train_env=[{train_env_start}..{split_idx}] ({split_idx - train_env_start} steps)"
            )

            # Val env
            val_envs[ticker] = ContractEnv(contract, sigma_tgt=sigma_tgt, bp=bp, start_idx=val_start, max_idx=n)
            env_log_lines.append(
                f"  {ticker}: val_dates=[{contract.dates[val_start]}..{contract.dates[-1]}] "
                f"val_env=[{val_start}..{n}] ({n - val_start} steps) "
                f"test_dates=[{feature_meta['test_start']}..{feature_meta['test_end']}] "
                f"test_idx=[{feature_meta['test_start_idx']}..{feature_meta['test_end_idx']}]"
            )
        except Exception as exc:
            skipped.append(f"{ticker}: {exc}")
            import traceback
            env_log_lines.append(f"  {ticker}: FAILED — {exc}")
            env_log_lines.append(f"    {traceback.format_exc().strip().splitlines()[-1]}")

    if not contracts:
        raise ValueError(f"No loadable contracts for {asset_name} {round_name(round_num)}")

    n_contracts = len(contracts)
    steps_per_contract = sum(env.max_idx - env.start_idx for env in train_envs.values())
    total_expected_steps = episodes * steps_per_contract
    buffer_sizes = {
        'Forex': 5000,
        'Equity Index': 5000,
        'Commodity': 5000,
        'Fixed Income': 3000,
    }
    memory_size = buffer_sizes.get(asset_name, 5000)
    agent = DQNAgent(device=device, total_steps=total_expected_steps, memory_size=memory_size)
    
    if resume:
        # Find latest checkpoint for this asset+round
        # Prefer latest_checkpoint.pt (full training state), fall back to checkpoint.pt (best)
        asset_model_dir = Path(MODEL_ROOT) / asset_slug(asset_name) / round_name(round_num)
        if asset_model_dir.exists():
            existing_bundles = sorted(
                [d for d in asset_model_dir.iterdir() if d.is_dir() and (
                    (d / "latest_checkpoint.pt").exists() or (d / "checkpoint.pt").exists()
                )],
                key=lambda d: d.name, reverse=True,
            )
            for prev_bundle in existing_bundles:
                latest = prev_bundle / "latest_checkpoint.pt"
                best = prev_bundle / "checkpoint.pt"
                ckpt_path = latest if latest.exists() else best
                agent.load(ckpt_path, resume=True)
                print(f"Resumed from {ckpt_path} (train_steps={agent.train_steps}, replay={len(agent.replay)})")
                break
    run_id = make_run_id(bp=bp)
    if seed is not None:
        run_id = f"{run_id}_s{seed}"
    bundle_dir = model_bundle_root(round_num, asset_name, run_id=run_id)
    logger = RunLogger("dqn", asset_name, round_num, run_id=run_id, base_dir=bundle_dir)
    round_info = RETRAIN_ROUNDS[round_num]
    f_spec = feature_spec()
    metadata = checkpoint_metadata(
        round_num,
        asset_name,
        algorithm="dqn",
        extra={
            "run_id": logger.run_id,
            "cycles": episodes,
            "episodes_per_cycle": len(contracts),
            "total_planned_contract_episodes": episodes * len(contracts),
            "device": agent.device,
            "log_dir": str(logger.dir),
            "bundle_dir": str(bundle_dir),
            "seed": seed,
            "sigma_tgt": sigma_tgt,
            "bp": bp,
            "member_tickers": list(contracts),
            "loaded_contracts": list(contracts),
            "skipped_contracts": skipped,
            "asset_class_count": len(contracts),
            "feature_artifact_paths": {
                ticker: meta["feature_artifact_path"] for ticker, meta in feature_meta_by_ticker.items()
            },
            "contract_round_splits": {
                ticker: {
                    "train_start_idx": int(meta["train_start_idx"]),
                    "train_end_idx": int(meta["train_end_idx"]),
                    "test_start_idx": int(meta["test_start_idx"]),
                    "test_end_idx": int(meta["test_end_idx"]),
                    "train_start": meta["train_start"],
                    "train_end": meta["train_end"],
                    "test_start": meta["test_start"],
                    "test_end": meta["test_end"],
                }
                for ticker, meta in feature_meta_by_ticker.items()
            },
            "state_spec_version": f_spec["state_spec_version"],
        },
    )
    logger.write_json("manifest.json", metadata)
    logger.write_json("train_config.json", metadata["hyperparameters"] | {
        "cycles": episodes,
        "seed": seed,
        "eps_total_steps": agent.total_steps,
        "eps_schedule": [list(s) for s in EPS_SCHEDULE],
        "memory_size": agent.replay.capacity,
        "memory_ratio": MEMORY_RATIO,
        "bp": bp,
    })
    logger.write_json("feature_spec.json", metadata["feature_spec"])

    logger.log(f"{'=' * 70}")
    logger.log(f"DQN Asset-Class Training — {asset_name} — {round_name(round_num)}")
    logger.log(f"Train: {round_info['train_start']} ~ {round_info['train_end']}")
    logger.log(f"Test : {round_info['test_start']} ~ {round_info['test_end']}")
    logger.log(f"Contracts: {len(contracts)}/{len(tickers)} loaded | Device: {agent.device}")
    if skipped:
        logger.log(f"Skipped: {skipped}")
    if val_envs:
        logger.log(f"Val envs: {len(val_envs)}/{len(contracts)} contracts")
    else:
        logger.log("❌ Val envs: 0 — no validation environments constructed. Aborting.")
        raise RuntimeError(
            f"All {len(contracts)} contracts failed to construct val_envs. "
            f"Check logs for errors (likely missing import like WARMUP/SEQ_LEN). "
            f"Skipped: {skipped}"
        )
    logger.log(f"Model bundle: {bundle_dir}")
    logger.log(f"State spec: {metadata['state_spec_version']}")
    logger.log(
        "DQN stabilizers [49]/[18]/[50]: fixed Q-targets, Double DQN, "
        "Dueling DQN; target hard-copy every 1000 learn steps; dropout 0.2"
    )
    logger.log(
        f"Validation: {VALIDATION_SPLIT*100:.0f}% split, early stopping patience={EARLY_STOPPING_PATIENCE}, "
        f"epsilon: {EPS_SCHEDULE[0][1]:.2f}→{EPS_SCHEDULE[-1][1]:.3f} over {agent.total_steps:,} steps, "
        f"epsilon_val=0.0"
    )

    # Log env construction details
    logger.log("--- Env construction ---")
    for line in env_log_lines:
        logger.log(line)
    if sanity_warnings:
        logger.log("--- Data sanity warnings ---")
        for ticker, warns in sanity_warnings.items():
            for w in warns:
                logger.log(f"  ⚠️  {ticker}: {w}")
    else:
        logger.log("--- Data sanity: all contracts OK ---")

    # Agent architecture & device info
    _check_agent_health(agent, logger)

    # Preflight: verify envs produce valid states and rewards
    preflight_errors = _preflight_check_envs(train_envs, val_envs, contracts, logger)
    if preflight_errors:
        logger.log(f"⚠️  Preflight errors: {len(preflight_errors)} — training may fail or produce garbage results")

    logger.log(f"{'=' * 70}")

    # --- Phase timing bookkeeping ---
    t_phase = time.time()
    t_data_load = t_phase - t0_start
    t_val_total = 0.0

    t0 = time.time()
    report_interval = max(1, episodes // 10)
    global_step = agent.train_steps if resume else 0
    global_episode = 0
    start_cycle = 1
    episode_rows: list[dict] = []
    validation_rows: list[dict] = []
    contract_stats = defaultdict(lambda: {
        "episodes_seen": 0,
        "transitions_added": 0,
        "reward_sum": 0.0,
        "loss_sum": 0.0,
        "loss_count": 0,
        "last_reward": 0.0,
        "last_loss": 0.0,
    })

    ordered_tickers = list(contracts)

    best_val_reward = float("-inf")
    _last_val_reward = 0.0
    patience_counter = 0
    best_bundle_dir = bundle_dir  # track best checkpoint dir
    early_stopped = False

    try:
        from tqdm import tqdm
        cycle_iterator = tqdm(range(1, episodes + 1), desc="Training Cycles", unit="cycle", dynamic_ncols=True)
        _tqdm_available = True
    except ImportError:
        cycle_iterator = range(1, episodes + 1)
        _tqdm_available = False
        print("Hint: pip install tqdm  for progress bar with live metrics")
        
    for cycle in cycle_iterator:
        # --- Interleaved training: step all contracts round-robin ---
        per_contract_rewards, per_contract_steps, cycle_losses, global_step = _run_interleaved_cycle(
            train_envs, agent, global_step, contract_stats,
        )
        cycle_rewards = list(per_contract_rewards.values())
        global_episode += len(per_contract_rewards)

        for ticker in per_contract_rewards:
            reward = per_contract_rewards[ticker]
            steps = per_contract_steps[ticker]
            mean_loss = float(np.mean(cycle_losses)) if cycle_losses else 0.0
            last_loss = float(cycle_losses[-1]) if cycle_losses else 0.0

            stats = contract_stats[ticker]
            stats["episodes_seen"] += 1
            stats["transitions_added"] += steps
            stats["reward_sum"] += reward
            stats["loss_sum"] += sum(cycle_losses) / max(1, len(per_contract_rewards))  # distribute evenly
            stats["loss_count"] += max(1, len(cycle_losses) // max(1, len(per_contract_rewards)))
            stats["last_reward"] = reward
            stats["last_loss"] = last_loss

            row = {
                "episode": global_episode,
                "cycle": cycle,
                "round": round_num,
                "ticker": ticker,
                "reward": round(reward, 6),
                "steps": steps,
                "epsilon_end": round(agent.epsilon_for_step(global_step), 6),
                "mean_loss": round(mean_loss, 6),
                "last_loss": round(last_loss, 6),
                "learn_steps": agent.train_steps,
                "target_updates": agent.target_updates,
                "replay_size": len(agent.replay),
            }
            episode_rows.append(row)

        # Calculate early to log for this cycle
        t_val0 = time.time()
        val_reward, val_action_counts = _validation_reward(val_envs, agent) if val_envs else (0.0, {0:0, 1:0, 2:0})
        t_val_total += time.time() - t_val0
        _last_val_reward = val_reward

        # Update tqdm postfix with live metrics every cycle
        if _tqdm_available:
            elapsed = time.time() - t0
            avg_reward = float(np.mean(cycle_rewards)) if cycle_rewards else 0.0
            avg_loss = float(np.mean(cycle_losses)) if cycle_losses else 0.0
            cycle_iterator.set_postfix({
                "rew": f"{avg_reward:+.4f}",
                "val": f"{val_reward:+.4f}",
                "loss": f"{avg_loss:.4f}",
                "eps": f"{agent.epsilon_for_step(global_step):.3f}",
                "buf": f"{len(agent.replay)}",
                "t": f"{elapsed:.0f}s",
            }, refresh=False)

        # Log every cycle for full diagnostic trail
        elapsed = time.time() - t0
        avg_loss = float(np.mean(cycle_losses)) if cycle_losses else 0.0
        eta = (elapsed / cycle) * max(0, episodes - cycle) if cycle > 0 else 0.0
        total_val_actions = sum(val_action_counts.values()) or 1
        action_dist_str = f"[L:{val_action_counts.get(2, 0)/total_val_actions*100:.0f}% S:{val_action_counts.get(0, 0)/total_val_actions*100:.0f}% F:{val_action_counts.get(1, 0)/total_val_actions*100:.0f}%]"

        logger.log(
            f"cycle {cycle}/{episodes} reward_avg={np.mean(cycle_rewards):+.4f} "
            f"val_reward={_last_val_reward:+.4f} val_actions={action_dist_str} "
            f"loss={avg_loss:.6f} "
            f"epsilon={agent.epsilon_for_step(global_step):.4f} replay={len(agent.replay)} "
            f"target_updates={agent.target_updates} patience={patience_counter}/{EARLY_STOPPING_PATIENCE} "
            f"elapsed={elapsed:.0f}s eta={eta:.0f}s"
        )
        # Training health check
        _check_training_health(cycle, cycle_rewards, cycle_losses, agent, global_step, logger)
        validation_rows.append({
            "cycle": cycle,
            "round": round_num,
            "validation_reward_mean": round(val_reward, 6),
            "best_validation_reward": round(max(best_val_reward, val_reward), 6) if best_val_reward != float("-inf") else round(val_reward, 6),
            "patience_counter": patience_counter,
            "contract_count": len(val_envs),
        })
        # Flush diagnostics every 10 cycles so data survives crashes
        if cycle % 10 == 0:
            logger.write_csv("episode_metrics.csv", episode_rows)
            logger.write_csv("validation_metrics.csv", validation_rows)
        degenerate = _is_degenerate(val_action_counts)
        if degenerate:
            patience_counter += 1
        elif val_reward > best_val_reward:
            best_val_reward = val_reward
            patience_counter = 0
            best_ckpt = checkpoint_path_for_bundle(bundle_dir)
            agent.save(best_ckpt, metadata=metadata, include_training_state=True)
            best_bundle_dir = bundle_dir
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE and cycle >= 30:
                logger.log(f"Early stopping at cycle {cycle}: val_reward={val_reward:+.4f} best={best_val_reward:+.4f}")
                early_stopped = True
                break

        # Save latest checkpoint every reporting interval for resume capability
        if cycle % report_interval == 0:
            latest_ckpt = bundle_dir / "latest_checkpoint.pt"
            agent.save(latest_ckpt, metadata=metadata, include_training_state=True)

    contract_rows = []
    for ticker, stats in sorted(contract_stats.items()):
        loss_count = max(1, stats["loss_count"])
        episodes_seen = max(1, stats["episodes_seen"])
        contract_rows.append({
            "ticker": ticker,
            "round": round_num,
            "episodes_seen": stats["episodes_seen"],
            "transitions_added": stats["transitions_added"],
            "avg_reward": round(stats["reward_sum"] / episodes_seen, 6),
            "avg_loss": round(stats["loss_sum"] / loss_count, 6),
            "last_reward": round(stats["last_reward"], 6),
            "last_loss": round(stats["last_loss"], 6),
        })

    metadata["completed_cycles"] = cycle if early_stopped else episodes
    metadata["early_stopped"] = early_stopped
    metadata["best_val_reward"] = best_val_reward
    metadata["learn_steps"] = agent.train_steps
    metadata["target_updates"] = agent.target_updates
    out_path = checkpoint_path_for_bundle(best_bundle_dir)
    if best_bundle_dir == bundle_dir:
        # Already saved as best during training
        pass
    else:
        agent.save(out_path, metadata=metadata, include_training_state=True)
    logger.write_csv("episode_metrics.csv", episode_rows)
    logger.write_csv("contract_metrics.csv", contract_rows)
    logger.write_csv("validation_metrics.csv", validation_rows)
    logger.write_json("checkpoint_metadata.json", metadata)
    logger.write_json("manifest.json", metadata)
    logger.log(f"Saved checkpoint: {out_path}")
    logger.log(f"Saved logs: {logger.dir}")

    # --- Timing Summary ---
    t_total = time.time() - t0_start
    completed_cycles = metadata["completed_cycles"]
    n_contracts = len(contracts)
    transitions = sum(s["transitions_added"] for s in contract_stats.values())
    learn_steps = agent.train_steps
    t_train_loop = t_total - t_data_load - t_val_total

    logger.log("")
    logger.log(f"{'=' * 70}")
    logger.log("Training Timing Summary")
    logger.log(f"{'=' * 70}")
    logger.log(f"  Data loading + env construction:  {t_data_load:6.1f}s")
    logger.log(f"  Preflight checks + agent init:      (included above)")
    logger.log(f"  Training loop ({completed_cycles} cycles):            {t_train_loop:6.1f}s")
    logger.log(f"    avg per cycle:                    {t_train_loop/max(1,completed_cycles):6.1f}s")
    logger.log(f"    avg per transition:               {t_train_loop/max(1,transitions)*1000:6.1f}ms")
    logger.log(f"  Validation (total):                 {t_val_total:6.1f}s")
    logger.log(f"    avg per cycle:                    {t_val_total/max(1,completed_cycles):6.1f}s")
    logger.log(f"  Total wall time:                    {t_total:6.1f}s")
    logger.log(f"  Device: {agent.device:<6s} | Params: {sum(p.numel() for p in agent.q_net.parameters()):,}")
    logger.log(f"  Transitions: {transitions:,} | Learn steps: {learn_steps:,}")
    logger.log(f"  Contracts: {n_contracts} | Stopped: {'early' if early_stopped else 'complete'}")
    logger.log(f"{'=' * 70}")
    return out_path, logger.dir


def train_contract_round(*args, **kwargs):
    raise RuntimeError(
        "Mainline DQN training is asset-class based. "
        "Use train_asset_round(asset_name, round_num, ...) or the CLI with --asset."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="Forex", help='Asset universe or "All"')
    parser.add_argument("--round", default="both", help='1, 2, or "both" (default)')
    parser.add_argument("--episodes", type=int, default=EPISODES, help="Training cycles; each cycle visits every contract once.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None, help="Override GAMMA (default: use spec.py value)")
    parser.add_argument("--sigma-tgt", type=float, default=SIGMA_TGT_DEFAULT)
    parser.add_argument("--tc-bp", type=float, default=BP, help="Transaction cost BP rate (default: BP from config)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint for each round")
    parser.add_argument("--tickers", nargs="+", default=None, help="Specific tickers to train (e.g. --tickers AN BN). Overrides asset index.")
    args = parser.parse_args()

    rounds = parse_rounds(args.round)
    assets = list(ASSET_CLASSES) if args.asset in (None, "", "All") else [args.asset]
    for round_num in rounds:
        for asset_name in assets:
            train_asset_round(
                asset_name,
                round_num,
                episodes=args.episodes,
                device=args.device,
                seed=args.seed,
                gamma=args.gamma,
                sigma_tgt=args.sigma_tgt,
                bp=args.tc_bp,
                resume=args.resume,
                tickers_override=args.tickers,
            )


if __name__ == "__main__":
    main()
