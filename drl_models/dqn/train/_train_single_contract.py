#!/usr/bin/env python3
"""Train a single contract DQN model with multi-seed support.

Runs N seeds, evaluates each on validation set, picks the best.
Saves: models/<TICKER>/r<round>/per_<timestamp>_s<seed>/checkpoint.pt
       models/<TICKER>/r<round>/best_seed.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from config import BP
from drl_models.dqn.model import DQNAgent
from drl_models.dqn.spec import (
    BATCH_SIZE,
    DISCRETE_ACTION_DIM,
    DROPOUT,
    DQN_SPEC_VERSION,
    EPS_DECAY_STEPS,
    EPS_END,
    EPS_START,
    FEATURE_DIM,
    GAMMA,
    LEAKY_RELU_SLOPE,
    LR,
    LSTM_HIDDEN_SIZES,
    MAX_STEPS_PER_EP,
    MEMORY_SIZE,
    MODEL_ROOT,
    SEQ_LEN,
    TAU,
    contract_data_path,
)
from drl_shared.spec import (
    SIGMA_TGT_DEFAULT,
    SEQ_LEN as SHARED_SEQ_LEN,
    current_source_policy,
    feature_spec,
    universe_tickers,
    ticker_asset_class,
)
from drl_shared.state_space import (
    WARMUP,
    ContractArrays,
    ContractEnv,
    action_id_to_position,
    compute_eq4_reward,
    get_feature_window,
)

SEEDS = 5


def load_contract(ticker: str, round_num: int) -> tuple[ContractArrays, dict]:
    """Load train-period data using shared data_loader (date-validated)."""
    from drl_shared.data_loader import get_train_slice
    train_contract, train_slice, val_slice, meta = get_train_slice(ticker, round_num)
    # Store split info in meta for the training loop
    meta["split_idx"] = train_slice.end_idx
    meta["val_start"] = val_slice.start_idx
    return train_contract, meta


import sys as _sys
def _log(msg: str):
    print(msg, flush=True)


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _greedy_val_reward(val_env: ContractEnv, agent: DQNAgent) -> float:
    """Run one greedy episode on validation env, return total reward."""
    state = val_env.reset()
    total = 0.0
    done = False
    steps = 0
    while not done and steps < MAX_STEPS_PER_EP:
        action_id = agent.act(state, eps=0.0)
        state, reward, done = val_env.step(action_id)
        total += reward
        steps += 1
    return total


def _train_one_seed(
    ticker: str, round_num: int, seed: int, episodes: int,
    train_env: ContractEnv, val_env: ContractEnv,
    device: str, sigma_tgt: float, early_stop_patience: int,
) -> tuple[DQNAgent, dict]:
    """Train a single seed and return (agent, stats)."""
    _set_seed(seed)
    agent = DQNAgent(device=device)
    global_step = 0
    rewards_log = []
    losses_log = []
    best_val_reward = None
    best_agent_state = None
    no_improve_count = 0

    for ep in range(episodes):
        state = train_env.reset()
        total_reward = 0.0
        ep_losses = []
        done = False
        steps = 0
        while not done and steps < MAX_STEPS_PER_EP:
            eps = agent.epsilon_for_step(global_step)
            action_id = agent.act(state, eps)
            next_state, reward, done = train_env.step(action_id)
            agent.push(state, action_id, reward, next_state, float(done))
            loss = agent.learn()
            if loss > 0:
                ep_losses.append(loss)
            state = next_state
            total_reward += reward
            steps += 1
            global_step += 1

        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        rewards_log.append(total_reward)
        losses_log.append(avg_loss)

        # Validate every episode
        val_reward = _greedy_val_reward(val_env, agent)
        if best_val_reward is None or val_reward > best_val_reward:
            best_val_reward = val_reward
            no_improve_count = 0
            import torch
            best_agent_state = {k: v.cpu().clone() for k, v in agent.q_net.state_dict().items()}
        else:
            no_improve_count += 1

        if early_stop_patience > 0 and no_improve_count >= early_stop_patience:
            if best_agent_state is not None:
                agent.q_net.load_state_dict(best_agent_state)
            break

        if (ep + 1) % 50 == 0:
            _log(f"    s{seed} ep{ep+1}/{episodes} r={total_reward:.4f} loss={avg_loss:.4f} val={val_reward:.4f} best_val={best_val_reward:.4f}")

    # Use best val weights (not last episode weights)
    if best_agent_state is not None:
        agent.q_net.load_state_dict(best_agent_state)

    final_val = _greedy_val_reward(val_env, agent)
    stats = {
        "seed": seed,
        "episodes_trained": len(rewards_log),
        "final_val_reward": final_val,
        "best_val_reward": best_val_reward,
        "final_avg_reward_10": float(np.mean(rewards_log[-10:])) if rewards_log else 0.0,
    }
    return agent, stats


def train_multi_seed(
    ticker: str, round_num: int, episodes: int, device: str,
    sigma_tgt: float, n_seeds: int = SEEDS, early_stop_patience: int = 20,
) -> Path:
    """Train n_seeds models, pick best by validation reward, save all + best_seed.json."""
    _log(f"  {ticker} r{round_num} START seeds={n_seeds} episodes={episodes} sigma_tgt={sigma_tgt} device={device}")
    contract, meta = load_contract(ticker, round_num)
    n = len(contract.prices)
    split_idx = meta["split_idx"]
    val_start = meta["val_start"]
    _log(f"  {ticker} r{round_num} data: {n} steps, train_dates={meta.get('train_dates','?')}")
    _log(f"  {ticker} r{round_num} split: train [0, {split_idx}), val [{val_start}, {n})")
    train_env = ContractEnv(contract, sigma_tgt=sigma_tgt, max_idx=split_idx - 1)
    val_env = ContractEnv(contract, sigma_tgt=sigma_tgt, max_idx=n - 1, start_idx=val_start)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    asset_class = ticker_asset_class(ticker)
    all_stats = []

    for seed in range(n_seeds):
        agent, stats = _train_one_seed(
            ticker, round_num, seed, episodes, train_env, val_env,
            device, sigma_tgt, early_stop_patience,
        )

        # Save this seed's model
        run_id = f"per_{timestamp}_s{seed}"
        save_dir = MODEL_ROOT / ticker / f"r{round_num}" / run_id
        save_dir.mkdir(parents=True, exist_ok=True)

        agent.save(save_dir / "checkpoint.pt", include_training_state=False, metadata={
            "ticker": ticker,
            "round": round_num,
            "seed": seed,
            "episodes": episodes,
            "device": device,
            "sigma_tgt": sigma_tgt,
            "training_mode": "per_contract_multiseed",
            "asset_class": asset_class,
            "val_reward": stats["final_val_reward"],
        })

        manifest = {
            "ticker": ticker, "round": round_num, "seed": seed,
            "episodes": stats["episodes_trained"],
            "sigma_tgt": sigma_tgt, "asset_class": asset_class,
            "val_reward": stats["final_val_reward"],
            "learning_curve": {},  # omit for space; stats captured above
        }
        with open(save_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        all_stats.append({**stats, "model_dir": str(save_dir)})
        _log(f"  {ticker} r{round_num} s{seed}: val={stats['final_val_reward']:.4f} → {save_dir.name}")

    # Pick best seed
    best = max(all_stats, key=lambda s: s["final_val_reward"])
    best_info = {
        "ticker": ticker,
        "round": round_num,
        "timestamp": timestamp,
        "n_seeds": n_seeds,
        "best_seed": best["seed"],
        "best_val_reward": best["final_val_reward"],
        "best_model_dir": best["model_dir"],
        "all_seeds": [
            {"seed": s["seed"], "val_reward": s["final_val_reward"], "dir": s["model_dir"]}
            for s in all_stats
        ],
    }
    best_path = MODEL_ROOT / ticker / f"r{round_num}" / "best_seed.json"
    with open(best_path, "w") as f:
        json.dump(best_info, f, indent=2)

    _log(f"  {ticker} r{round_num} BEST: seed={best['seed']} val={best['final_val_reward']:.4f}")
    return Path(best["model_dir"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=SEEDS, help=f"Number of seeds (default {SEEDS})")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sigma-tgt", type=float, default=0.058)
    parser.add_argument("--early-stop", type=int, default=20, help="Early stopping patience (0=disabled)")
    args = parser.parse_args()
    train_multi_seed(
        args.ticker.upper(), args.round, args.episodes, args.device,
        args.sigma_tgt, n_seeds=args.seeds, early_stop_patience=args.early_stop,
    )


if __name__ == "__main__":
    main()
