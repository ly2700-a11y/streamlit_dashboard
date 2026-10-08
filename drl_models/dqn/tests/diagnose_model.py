#!/usr/bin/env python3
"""
Diagnostic script for inspecting the health and output of a trained DQN model.
Usage: python diagnose_model.py /path/to/checkpoint.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drl_models.dqn.model import DQNAgent
from drl_shared.spec import FEATURE_DIM, SEQ_LEN


def main():
    parser = argparse.ArgumentParser(description="Diagnose DQN Model Health")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint.pt")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    print(f"==================================================")
    print(f"🩺 DQN Model Diagnostic Report")
    print(f"==================================================")
    print(f"Loading model: {ckpt_path.name}")
    
    agent = DQNAgent(device="cpu")
    try:
        agent.load(ckpt_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
        
    print(f"✅ Model loaded successfully on {agent.device}.")
    
    # 1. Weight Integrity Check
    print(f"\n--- 1. Weight Integrity Check ---")
    weights = list(agent.q_net.parameters())
    all_zero = True
    has_nan = False
    for w in weights:
        w_data = w.detach().numpy()
        if not np.all(w_data == 0):
            all_zero = False
        if np.isnan(w_data).any() or np.isinf(w_data).any():
            has_nan = True
            
    if has_nan:
        print("❌ CRITICAL: Model contains NaN or Inf weights. Gradient explosion occurred.")
    elif all_zero:
        print("❌ CRITICAL: Model weights are entirely zero. Training failed completely.")
    else:
        print("✅ Weights contain valid finite values.")

    # 2. Q-Value Variance Check on Random Noise
    print(f"\n--- 2. Output Variance Check (Random Noise) ---")
    print("Feeding 1000 random sequences to the network...")
    # Generate random standard normal noise representing features
    dummy_states = np.random.randn(1000, SEQ_LEN, FEATURE_DIM).astype(np.float32)
    
    agent.q_net.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(dummy_states).to(agent.device)
        q_values = agent.q_net(tensor)
        actions = q_values.argmax(dim=1).numpy()
        q_numpy = q_values.numpy()

    action_counts = np.bincount(actions, minlength=3)
    dist_str = f"Short(0): {action_counts[0]/10:.1f}%, Flat(1): {action_counts[1]/10:.1f}%, Long(2): {action_counts[2]/10:.1f}%"
    print(f"Action Distribution: {dist_str}")
    
    if max(action_counts) > 950:
        print("⚠️ WARNING: Action Collapse detected! The model is outputting the same action for >95% of random inputs.")
        print("   This suggests precision loss or gradient explosion during training.")
    else:
        print("✅ Action Distribution looks diverse.")

    print(f"\nQ-Value Statistics:")
    print(f"  Mean: {q_numpy.mean():.4f}")
    print(f"  Std Dev across all states: {q_numpy.std():.4f}")
    q_std_per_state = q_numpy.std(axis=1)
    print(f"  Avg Std Dev between actions (Action Separability): {q_std_per_state.mean():.6f}")
    
    if q_std_per_state.mean() < 1e-4:
        print("❌ CRITICAL: Q-values are identical across actions (Separability < 1e-4).")
        print("   The network cannot distinguish between Long and Short. The argmax is blindly picking the first index.")
    else:
        print("✅ Q-values have adequate separation.")
        
    print(f"\n==================================================")
    print(f"Diagnosis Complete.")

if __name__ == "__main__":
    main()
