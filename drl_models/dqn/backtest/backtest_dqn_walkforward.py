#!/usr/bin/env python3
"""Table 3 strategy runner via the unified baseline backtest stack."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drl_models.dqn.backtest.engine import canonical_strategy_name, paper_reference, portfolio_metrics


def _sigma_from_bundle(bundle: str | None) -> float | None:
    if not bundle:
        return None
    manifest = Path(bundle) / "manifest.json"
    if not manifest.exists():
        return None
    with manifest.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    value = payload.get("sigma_tgt")
    return float(value) if value is not None else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="Long", choices=["Long", "Sign(R)", "MACD", "DQN"])
    parser.add_argument("--asset", default="Forex")
    parser.add_argument("--round", type=int, default=None, help="Only for DQN. If omitted, stitch all rounds.")
    parser.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path for DQN.")
    parser.add_argument("--checkpoint-bundle", default=None, help="Optional explicit bundle directory for DQN.")
    parser.add_argument("--run-id", default="latest")
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto", help="Torch device for DQN inference.")
    parser.add_argument("--progress", action="store_true", help="Show per-contract DQN inference progress bars with ETA.")
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    strategy = canonical_strategy_name(args.strategy)
    effective_sigma = args.sigma
    if effective_sigma is None and strategy == "DQN":
        effective_sigma = _sigma_from_bundle(args.checkpoint_bundle)
    if effective_sigma is None:
        effective_sigma = 0.058
    metrics = portfolio_metrics(
        args.asset,
        strategy,
        round_num=args.round,
        checkpoint=args.checkpoint,
        checkpoint_bundle=args.checkpoint_bundle,
        run_id=args.run_id,
        device=args.device,
        progress=args.progress,
        batch_size=args.batch_size,
        sigma_tgt=effective_sigma,
    )
    paper = paper_reference(args.asset, strategy)

    print(f"\n{'=' * 70}")
    print(f"Table 3 Backtest — {strategy} — {args.asset}")
    print(f"sigma_tgt={effective_sigma}")
    if strategy == "DQN":
        print(f"DQN device: {args.device}")
        if args.round is not None:
            print(f"DQN round override: r{args.round}")
        else:
            print("DQN using stitched round schedule (r1 + r2)")
    print(f"{'=' * 70}")
    for metric, ours in metrics.items():
        if paper is not None and metric in paper:
            target = paper[metric]
            err = abs((ours - target) / abs(target)) * 100 if target != 0 else 0.0
            print(f"  {metric:8s}: {ours:+.3f} vs {target:+.3f} err={err:.1f}%")
        else:
            print(f"  {metric:8s}: {ours:+.3f}")
