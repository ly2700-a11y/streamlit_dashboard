#!/usr/bin/env python3
"""Multi-seed asset-class walkforward DQN training + ensemble backtest.

Trains N independent walkforward agents (each sees all contracts),
picks top-3 by val reward, runs ensemble backtest on test period.

Usage:
    python3 drl/dqn/train/train_walkforward_multiseed.py --asset Forex --seeds 5 --episodes 200 --device cuda
"""
from __future__ import annotations
import argparse, json, sys, time, os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from drl_models.dqn.train.train_dqn_walkforward import train_asset_round
from drl_models.dqn.spec import MODEL_ROOT
from drl_models.dqn.model import DQNAgent
from drl_shared.data_loader import get_test_slice
from drl_shared.spec import RETRAIN_ROUNDS, universe_tickers, current_source_policy
from drl_shared.state_space import ContractEnv


def compute_mdd(r):
    cum = np.cumsum(r)
    return float(np.min(cum - np.maximum.accumulate(cum)))


def train_seeds_parallel(asset_name: str, round_num: int, n_seeds: int, episodes: int, device: str, sigma_tgt: float, max_parallel: int = 4):
    """Train n_seeds independent walkforward agents in parallel."""
    import subprocess
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _train_one_seed(seed):
        t0 = time.time()
        import random
        random.seed(seed)
        np.random.seed(seed)
        ckpt_path, log_dir = train_asset_round(
            asset_name=asset_name,
            round_num=round_num,
            episodes=episodes,
            device=device,
            seed=seed,
            sigma_tgt=sigma_tgt,
        )
        elapsed = time.time() - t0
        manifest_path = Path(log_dir) / "manifest.json"
        val_reward = 0
        if manifest_path.exists():
            manifest = json.load(open(manifest_path))
            val_reward = manifest.get("best_val_reward", 0)
        print(f"  Seed {seed}: val={val_reward:.4f}, elapsed={elapsed:.0f}s", flush=True)
        return {"seed": seed, "val_reward": val_reward, "checkpoint": str(ckpt_path), "log_dir": str(log_dir), "elapsed": elapsed}

    results = []
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futs = {ex.submit(_train_one_seed, s): s for s in range(n_seeds)}
        for f in as_completed(futs):
            results.append(f.result())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = MODEL_ROOT / f"{asset_name}_r{round_num}_walkforward_seeds_{timestamp}.json"
    summary = {
        "asset": asset_name, "round": round_num, "n_seeds": n_seeds,
        "episodes": episodes, "sigma_tgt": sigma_tgt, "timestamp": timestamp,
        "seeds": sorted(results, key=lambda x: x["val_reward"], reverse=True),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSeed summary: {summary_path}")
    return results


def backtest_ensemble(asset_name: str, round_num: int, seed_results: list, mode: str = "top3"):
    """Backtest ensemble of top seeds vs Long on test period."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    sorted_seeds = sorted(seed_results, key=lambda x: x["val_reward"], reverse=True)
    if mode == "top3":
        selected = sorted_seeds[:3]
        weights = [1/3, 1/3, 1/3]
    elif mode == "best":
        selected = sorted_seeds[:1]
        weights = [1.0]
    elif mode == "weighted":
        selected = sorted_seeds[:3]
        weights = [0.6, 0.3, 0.1]
    else:
        selected = sorted_seeds[:3]
        weights = [1/3, 1/3, 1/3]

    tickers = universe_tickers(asset_name)
    policy = current_source_policy()
    excluded = set(policy.get("excluded_contracts", []))
    tickers = [t for t in tickers if t.upper() not in excluded]
    sigma_tgt = seed_results[0].get("sigma_tgt", 0.058) if seed_results else 0.058

    # Load agents once
    agents = []
    for s, w in zip(selected, weights):
        ckpt = Path(s["checkpoint"])
        if not ckpt.exists():
            continue
        ag = DQNAgent(device="cuda")
        ag.load(ckpt)
        ag.q_net.eval()
        agents.append((ag, w))

    if not agents:
        print("No valid checkpoints for backtest")
        return

    def run_one(ticker):
        try:
            test_contract, start_idx, meta = get_test_slice(ticker, round_num)
        except Exception:
            return {"ticker": ticker, "error": "no test data"}

        # DQN ensemble
        env = ContractEnv(test_contract, sigma_tgt=sigma_tgt, start_idx=start_idx)
        state = env.reset()
        dr = []
        done = False
        while not done:
            state_t = torch.from_numpy(state).unsqueeze(0).float().cuda()
            q_weighted = None
            for ag, w in agents:
                with torch.no_grad():
                    q = ag.q_net(state_t)
                q_weighted = w * q if q_weighted is None else q_weighted + w * q
            action_id = q_weighted.argmax().item()
            state, reward, done = env.step(action_id)
            dr.append(reward)

        # Long
        env2 = ContractEnv(test_contract, sigma_tgt=sigma_tgt, start_idx=start_idx)
        env2.reset()
        lr = []
        done = False
        while not done:
            _, reward, done = env2.step(2)
            lr.append(reward)

        dr, lr = np.array(dr), np.array(lr)
        return {
            "ticker": ticker,
            "dqn_cum": float(dr.sum()), "long_cum": float(lr.sum()),
            "dqn_sharpe": float(dr.mean()/(dr.std()+1e-10)),
            "long_sharpe": float(lr.mean()/(lr.std()+1e-10)),
            "win": "DQN" if dr.sum() > lr.sum() else "LONG",
        }

    label = f"{asset_name} r{round_num} ({mode}, {len(agents)} agents)"
    print(f"\n--- Backtest: {label} ---")
    print(f"{'Ticker':>6} {'DQN_cum':>9} {'Long_cum':>9} {'DQN_Shp':>8} {'Long_Shp':>8} {'Win':>6}")
    print("-" * 55)

    results = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_one, t): t for t in tickers}
        for f in as_completed(futs):
            res = f.result()
            results.append(res)
            if "error" in res:
                print(f"{res['ticker']:>6}  ERROR: {res['error']}")
            else:
                print(f"{res['ticker']:>6} {res['dqn_cum']:>+9.3f} {res['long_cum']:>+9.3f} "
                      f"{res['dqn_sharpe']:>+8.3f} {res['long_sharpe']:>+8.3f} {res['win']:>6}")

    ok = [r for r in results if "error" not in r]
    wins = sum(1 for r in ok if r["win"] == "DQN")
    print(f"\nDQN beats Long: {wins}/{len(ok)}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="Forex")
    parser.add_argument("--round", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sigma-tgt", type=float, default=0.058)
    parser.add_argument("--skip-train", action="store_true", help="Skip training, only backtest")
    parser.add_argument("--parallel", type=int, default=4, help="Max parallel seeds")
    args = parser.parse_args()

    rounds = args.round if args.round else [1, 2]

    for round_num in rounds:
        print(f"\n{'#'*70}")
        print(f"# {args.asset} — Round {round_num}")
        print(f"{'#'*70}")

        if not args.skip_train:
            seed_results = train_seeds_parallel(
                args.asset, round_num, args.seeds, args.episodes, args.device, args.sigma_tgt, args.parallel
            )
        else:
            # Load from latest summary
            summaries = sorted(MODEL_ROOT.glob(f"{args.asset}_r{round_num}_walkforward_seeds_*.json"))
            if not summaries:
                print(f"No seed summary for {args.asset} r{round_num}")
                continue
            summary = json.load(open(summaries[-1]))
            seed_results = summary["seeds"]

        # Backtest: best seed
        print(f"\n=== Best Seed ===")
        backtest_ensemble(args.asset, round_num, seed_results, mode="best")

        # Backtest: top-3 equal
        print(f"\n=== Top-3 Equal Weight ===")
        backtest_ensemble(args.asset, round_num, seed_results, mode="top3")

        # Backtest: 60/30/10
        print(f"\n=== Top-3 Weighted (60/30/10) ===")
        backtest_ensemble(args.asset, round_num, seed_results, mode="weighted")


if __name__ == "__main__":
    main()
