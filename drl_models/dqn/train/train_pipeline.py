#!/usr/bin/env python3
"""Unified DQN training pipeline.

- Reads training_config.yaml as single source of truth
- Detects completed/failed/pending tasks
- Supports resume from interruption
- Records full manifest per run for reproducibility
- Backtest code reads config + manifest to ensure consistency

Usage:
  python drl/dqn/train/train_pipeline.py                    # all contracts
  python drl/dqn/train/train_pipeline.py --asset Forex       # single asset class
  python drl/dqn/train/train_pipeline.py --tickers AN BN CN  # specific tickers
  python drl/dqn/train/train_pipeline.py --status             # show progress
  python drl/dqn/train/train_pipeline.py --force              # retrain even if completed
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from config import ASSET_CLASSES
from drl_shared.spec import current_source_policy

CONFIG_PATH = Path(__file__).parent / "training_config.yaml"
MODEL_ROOT = REPO / "drl" / "dqn" / "models"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_tickers(asset: str | None = None, config: dict | None = None) -> list[str]:
    policy = current_source_policy()
    excluded = set(policy["excluded_contracts"])
    cfg = config or load_config()
    if asset and asset in cfg["asset_classes"]:
        return [t for t in cfg["asset_classes"][asset] if t not in excluded]
    all_tickers = []
    for asset_name, tickers in cfg["asset_classes"].items():
        all_tickers.extend(t for t in tickers if t not in excluded)
    return sorted(set(all_tickers))


def task_status(ticker: str, round_num: int) -> str:
    """Check if a training task is completed.

    A task is completed if:
    - best_seed.json exists
    - The manifest in best_seed.json has sigma_tgt matching config
    - All n_seeds checkpoints exist
    """
    best_path = MODEL_ROOT / ticker / f"r{round_num}" / "best_seed.json"
    if not best_path.exists():
        return "pending"

    try:
        with open(best_path) as f:
            info = json.load(f)
    except (json.JSONDecodeError, OSError):
        return "corrupt"

    # Check config version consistency
    config = load_config()
    expected_sigma = config["training"]["sigma_tgt"]
    expected_seeds = config["training"]["n_seeds"]
    expected_preset = config["feature"]["preset"]

    # Validate key parameters match current config
    model_dir = info.get("best_model_dir", "")
    manifest_path = Path(model_dir) / "manifest.json" if model_dir else None
    if manifest_path and manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        meta_sigma = manifest.get("sigma_tgt")
        meta_preset = manifest.get("preset")
        if meta_sigma != expected_sigma:
            return "stale"  # trained with different sigma_tgt
        if meta_preset != expected_preset:
            return "stale"  # trained with different feature preset

    # Check all seeds completed
    actual_seeds = len(info.get("all_seeds", []))
    if actual_seeds < expected_seeds:
        return "incomplete"

    return "completed"


def show_status(args):
    """Print training status for all tasks."""
    config = load_config()
    tickers = get_tickers(args.asset, config)
    rounds = [1, 2]

    status_counts = {"completed": 0, "stale": 0, "incomplete": 0, "corrupt": 0, "pending": 0}
    tasks = []

    for ticker in tickers:
        for rnd in rounds:
            s = task_status(ticker, rnd)
            status_counts[s] += 1
            tasks.append((ticker, rnd, s))

    total = len(tasks)
    print(f"DQN Training Status — {total} tasks")
    print(f"Config: preset={config['feature']['preset']} sigma_tgt={config['training']['sigma_tgt']} "
          f"seeds={config['training']['n_seeds']} early_stop={config['training']['early_stopping_patience']}")
    print(f"{'='*50}")

    by_status = {}
    for ticker, rnd, s in tasks:
        by_status.setdefault(s, []).append(f"{ticker} r{rnd}")

    for s in ["completed", "stale", "incomplete", "corrupt", "pending"]:
        count = status_counts[s]
        if count > 0:
            icon = {"completed": "✅", "stale": "⚠️ ", "incomplete": "🔶", "corrupt": "❌", "pending": "⬜"}[s]
            print(f"  {icon} {s}: {count}")
            if s != "completed" and count <= 10:
                for t in by_status[s]:
                    print(f"      {t}")

    print(f"\n  Progress: {status_counts['completed']}/{total} completed")


def run_training(args):
    """Execute training pipeline."""
    config = load_config()
    tickers = get_tickers(args.asset, config)

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]

    rounds = [1, 2]
    parallel = args.parallel or config["execution"]["parallel"]
    device = args.device or config["execution"]["device"]
    episodes = args.episodes or config["hyperparameters"]["episodes"]
    seeds = args.seeds or config["training"]["n_seeds"]
    early_stop = config["training"]["early_stopping_patience"]
    sigma_tgt = config["training"]["sigma_tgt"]

    # Build task queue, skip completed unless --force
    tasks = []
    for ticker in tickers:
        for rnd in rounds:
            s = task_status(ticker, rnd)
            if s == "completed" and not args.force:
                continue
            if s == "stale" and not args.force:
                print(f"⚠️  {ticker} r{rnd} is STALE (config mismatch) — use --force to retrain")
                continue
            tasks.append((ticker, rnd))

    if not tasks:
        print("All tasks completed. Nothing to do.")
        return

    print(f"DQN Training Pipeline")
    print(f"{'='*60}")
    print(f"Config: {CONFIG_PATH}")
    print(f"  preset={config['feature']['preset']} sigma_tgt={sigma_tgt} "
          f"seeds={seeds} early_stop={early_stop}")
    print(f"Tasks: {len(tasks)} | parallel={parallel} | device={device}")
    print(f"{'='*60}")

    running: list[tuple[subprocess.Popen, str, int, str]] = []
    remaining = list(tasks)
    completed = 0
    failed = 0
    t0 = time.time()

    while remaining or running:
        while len(running) < parallel and remaining:
            ticker, rnd = remaining.pop(0)
            log_file = f"/tmp/dqn_pipeline_{ticker}_r{rnd}.log"
            cmd = [
                sys.executable,
                str(REPO / "drl" / "dqn" / "train" / "_train_single_contract.py"),
                "--ticker", ticker,
                "--round", str(rnd),
                "--episodes", str(episodes),
                "--seeds", str(seeds),
                "--device", device,
                "--sigma-tgt", str(sigma_tgt),
                "--early-stop", str(early_stop),
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=open(log_file, "w"),
                stderr=subprocess.STDOUT,
                cwd=str(REPO),
            )
            running.append((proc, ticker, rnd, log_file))
            elapsed = time.time() - t0
            print(f"[{time.strftime('%H:%M:%S')}] ({elapsed:.0f}s) START {ticker} r{rnd} "
                  f"(PID {proc.pid}) | queue={len(remaining)} running={len(running)}")

        if not running:
            break

        time.sleep(10)

        still_running = []
        for proc, ticker, rnd, log in running:
            ret = proc.poll()
            if ret is not None:
                elapsed = time.time() - t0
                if ret == 0:
                    completed += 1
                    print(f"[{time.strftime('%H:%M:%S')}] ({elapsed:.0f}s) ✅ {ticker} r{rnd} "
                          f"| done={completed} fail={failed} left={len(remaining)+len(running)-1}")
                else:
                    failed += 1
                    print(f"[{time.strftime('%H:%M:%S')}] ({elapsed:.0f}s) ❌ {ticker} r{rnd} (exit {ret})")
                    try:
                        with open(log) as f:
                            lines = f.readlines()
                        for line in lines[-3:]:
                            print(f"    {line.rstrip()}")
                    except Exception:
                        pass
            else:
                still_running.append((proc, ticker, rnd, log))

        running = still_running

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Completed: {completed} | Failed: {failed}")

    # Save run summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": str(CONFIG_PATH),
        "preset": config["feature"]["preset"],
        "sigma_tgt": sigma_tgt,
        "seeds": seeds,
        "early_stopping_patience": early_stop,
        "episodes": episodes,
        "tasks_total": len(tasks),
        "tasks_completed": completed,
        "tasks_failed": failed,
        "elapsed_seconds": round(elapsed),
    }
    summary_path = REPO / "drl" / "dqn" / f"training_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Run summary: {summary_path}")

    if failed:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Unified DQN Training Pipeline")
    parser.add_argument("--status", action="store_true", help="Show training progress")
    parser.add_argument("--asset", default=None, help="Asset class filter")
    parser.add_argument("--tickers", nargs="*", default=None, help="Specific tickers")
    parser.add_argument("--parallel", type=int, default=None, help="Override parallel workers")
    parser.add_argument("--device", default=None, help="Override device")
    parser.add_argument("--episodes", type=int, default=None, help="Override episodes")
    parser.add_argument("--seeds", type=int, default=None, help="Override seeds")
    parser.add_argument("--force", action="store_true", help="Retrain even if completed")
    args = parser.parse_args()

    if args.status:
        show_status(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
