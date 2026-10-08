#!/usr/bin/env python3
"""Minimal test: resume training produces identical weights as continuous training.

Agent A: seed → train N → save (including RNG state) → load resume → train N
Agent B: seed → train 2N (continuous)
=> weights must be bitwise identical
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drl_models.dqn.model import DQNAgent


SEED = 42
STATE_DIM = 7
SEQ_LEN = 30


def _init_seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def _train_steps(agent: DQNAgent, n: int):
    """Push random transitions and learn. RNG drives everything."""
    for _ in range(n):
        s = torch.randn(SEQ_LEN, STATE_DIM)
        a = random.randint(0, 2)
        r = float(np.random.randn())
        s2 = torch.randn(SEQ_LEN, STATE_DIM)
        d = float(np.random.random() < 0.05)
        agent.push(s, a, r, s2, d)
        agent.learn()


def _get_weights(agent: DQNAgent) -> dict[str, np.ndarray]:
    return {name: param.data.numpy().copy() for name, param in agent.q_net.named_parameters()}


def main():
    tmp = Path(__file__).parent / "_test_tmp"
    tmp.mkdir(exist_ok=True)
    ckpt = tmp / "checkpoint.pt"

    N = 80  # must be > BATCH_SIZE=64

    # --- Agent A: train N → save → resume → train N ---
    _init_seed()
    agent_a = DQNAgent(device="cpu")
    _train_steps(agent_a, N)
    agent_a.save(ckpt, include_training_state=True)

    agent_a2 = DQNAgent(device="cpu")
    agent_a2.load(ckpt, resume=True)
    _train_steps(agent_a2, N)
    weights_a = _get_weights(agent_a2)

    # --- Agent B: train 2N from scratch (same seed) ---
    _init_seed()
    agent_b = DQNAgent(device="cpu")
    _train_steps(agent_b, 2 * N)
    weights_b = _get_weights(agent_b)

    # --- Compare ---
    mismatches = []
    for name in weights_a:
        if not np.array_equal(weights_a[name], weights_b[name]):
            diff = np.abs(weights_a[name] - weights_b[name]).max()
            mismatches.append(f"{name}: max_diff={diff:.8e}")

    if mismatches:
        print("❌ Resume weights differ from continuous training:")
        for m in mismatches:
            print(f"  {m}")
        sys.exit(1)

    print(f"✅ Resume test passed: {len(weights_a)} weight tensors bitwise identical")
    print(f"   A: train {N} → save → resume → train {N}")
    print(f"   B: train {2*N} continuous")
    print(f"   train_steps: A={agent_a2.train_steps}, B={agent_b.train_steps}")

    # Cleanup
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
