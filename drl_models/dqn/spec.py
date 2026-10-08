"""Active DQN spec on top of the shared structural-38 DRL state-space."""
from __future__ import annotations

import json
from pathlib import Path

from config import BP, EWMA_SPAN, MACD_VOL_WINDOW
from drl_shared.spec import (
    ACTIVE_FEATURE_LINE,
    asset_slug,
    CONTINUOUS_ACTION_RANGE,
    DISCRETE_ACTION_VALUES,
    FEATURE_DIM,
    HORIZONS,
    MACD_PAIRS_ACTIVE,
    MARKET_FEATURE_DIM,
    RETRAIN_ROUNDS,
    RSI_WINDOW,
    SEQ_LEN,
    SIGMA_TGT_DEFAULT,
    current_source_policy,
    feature_data_path,
    feature_spec,
    round_name,
    ticker_asset_class,
    ticker_slug,
    universe_tickers,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DQN_ROOT = REPO_ROOT / "drl" / "dqn"
MODEL_ROOT = DQN_ROOT / "models"
LOG_ROOT = REPO_ROOT / "logs" / "rl"

DQN_SPEC_VERSION = "asset_class_structural38_mainline"
ACTIVE_MODEL_LINE = ACTIVE_FEATURE_LINE
TRAINING_MODE = "asset_class_shared"

DISCRETE_ACTION_DIM = len(DISCRETE_ACTION_VALUES)

# Paper Table 1 hyperparameters
LR = 0.0001
GAMMA = 0.6  # selected via 3-gamma comparison (0.5/0.6/0.7) on Forex r1+r2
BATCH_SIZE = 64
MEMORY_SIZE = 5000
TAU = 1000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY_STEPS = "dynamic"  # legacy; superseded by EPS_SCHEDULE + total_steps percentage
EPISODES = 100
MAX_STEPS_PER_EP = 5_000  # non-binding safety guardrail for finite datasets
SIGMA_TGT = SIGMA_TGT_DEFAULT
LEAKY_RELU_SLOPE = 0.01
LSTM_HIDDEN_SIZES = (64, 32)
DROPOUT = 0.2
VALIDATION_SPLIT = 0.1
EARLY_STOPPING_PATIENCE = 20
UPDATE_FREQ = 4  # learn every N transitions in interleaved training
WARMUP = SEQ_LEN

# ── Memory (percentage-based) ──
MEMORY_RATIO = 0.5             # buffer = max(3000, 50% of 1 episode steps)
MEMORY_SIZE_MIN = 5000         # floor (paper default)
REPLAY_MODE = "uniform"        # "uniform" (paper default) | "action_balanced" | "reward_stratified"
REPLAY_DIAG_INTERVAL = 5       # log replay diagnostics every N cycles

# ── Epsilon decay (3-phase, cycle-based) ──
# Phase 1: ε=1.0 for first 5000 transitions (fill replay buffer with random experience)
# Phase 2: ε=0.3→0.1 over cycles 1-20 (structured exploration)
# Phase 3: ε=0.1→0.01 over cycles 20-50 (fine-tuning)
# Phase 4: ε=0.01 fixed for cycles 50-100 (exploitation)
#
# Schedule is defined as (fraction_of_total_cycles, epsilon_value) pairs.
# With EPISODES=100:
#   Phase 1 is handled in training loop (first 5000 transitions)
#   Phase 2: 0%-20% of cycles, ε 0.3→0.1
#   Phase 3: 20%-50% of cycles, ε 0.1→0.01
#   Phase 4: 50%-100% of cycles, ε 0.01 fixed
EPS_BUFFER_FILL = 5000          # pure random transitions before learning starts
EPS_SCHEDULE = [
    (0.00, 0.300),             # cycle 0:   ε = 0.30 (after buffer fill)
    (0.20, 0.100),             # cycle 20:  ε decayed to 0.10
    (0.50, 0.030),             # cycle 50:  ε decayed to 0.03
    (0.90, 0.010),             # cycle 90:  ε decayed to 0.01
    (1.00, 0.010),             # cycle 100: ε stays at 0.01
]

# ── Stability enhancements ──
GRAD_CLIP_MAX_NORM = 1.0       # gradient L2 norm clipping (0 = disabled)
USE_HUBER_LOSS = False         # MSE loss (paper default); set True for smooth_l1_loss
HUBER_DELTA = 1.0              # Huber loss delta (threshold between L1/L2)

# ── Locked seeds for reproducibility ──
# These 5 seeds are fixed. All experiments, ablations, and reports use LOCKED_SEEDS.
LOCKED_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

# ── Transaction cost BP levels for experiment grid ──
TC_BP_LEVELS = [0.0001, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0045]
TC_BP_PHASE1 = [0.0001, 0.0020, 0.0045, 0.0010, 0.0030]  # 1, 20(恢复), 45, 10, 30 bps
TC_BP_PHASE2 = [0.0005, 0.0015, 0.0025, 0.0035, 0.0040]  # 5, 15, 25, 35, 40 bps


def contract_data_path(round_num: int, ticker: str) -> Path:
    return feature_data_path(round_num, ticker)


def model_bundle_root(round_num: int, asset_name: str, run_id: str) -> Path:
    return MODEL_ROOT / asset_slug(asset_name) / round_name(round_num) / run_id


def legacy_contract_model_bundle_root(round_num: int, ticker: str, run_id: str) -> Path:
    return MODEL_ROOT / ticker_slug(ticker) / round_name(round_num) / run_id


def checkpoint_path_for_bundle(bundle_dir: str | Path) -> Path:
    return Path(bundle_dir) / "checkpoint.pt"


def _valid_bundle_dirs(round_num: int, asset_name: str) -> list[Path]:
    root = MODEL_ROOT / asset_slug(asset_name) / round_name(round_num)
    if not root.exists():
        return []
    bundles = []
    for child in root.iterdir():
        if child.is_dir() and (child / "checkpoint.pt").exists():
            bundles.append(child)
    return sorted(bundles, key=lambda p: p.name)


def resolve_model_bundle(round_num: int, asset_name: str, run_id: str = "latest") -> Path:
    if run_id == "latest":
        bundles = _valid_bundle_dirs(round_num, asset_name)
        if not bundles:
            return MODEL_ROOT / asset_slug(asset_name) / round_name(round_num) / "latest"
        return bundles[-1]
    return model_bundle_root(round_num, asset_name, run_id=run_id)


def resolve_checkpoint_path(
    round_num: int,
    asset_name: str,
    run_id: str = "latest",
    checkpoint_bundle: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> Path:
    if checkpoint:
        return Path(checkpoint)
    if checkpoint_bundle:
        return checkpoint_path_for_bundle(checkpoint_bundle)
    return checkpoint_path_for_bundle(resolve_model_bundle(round_num, asset_name, run_id=run_id))


def load_manifest(bundle_dir: str | Path) -> dict | None:
    path = Path(bundle_dir) / "manifest.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def maybe_load_manifest_for_checkpoint(checkpoint: str | Path) -> dict | None:
    path = Path(checkpoint)
    manifest = path.parent / "manifest.json"
    if manifest.exists():
        with manifest.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def run_log_dir(algorithm: str, asset_or_ticker: str, round_num: int, run_id: str) -> Path:
    try:
        slug = asset_slug(asset_or_ticker)
    except ValueError:
        slug = ticker_slug(asset_or_ticker)
    return LOG_ROOT / algorithm.lower() / slug / round_name(round_num) / run_id


def checkpoint_metadata(
    round_num: int,
    asset_or_ticker: str,
    algorithm: str = "dqn",
    extra: dict | None = None,
) -> dict:
    f_spec = feature_spec()
    policy = current_source_policy()
    if asset_or_ticker in universe_tickers("All"):
        ticker = asset_or_ticker.upper()
        asset_class = ticker_asset_class(ticker)
    else:
        ticker = None
        asset_class = asset_or_ticker
    meta = {
        "feature_line": ACTIVE_FEATURE_LINE,
        "spec_version": DQN_SPEC_VERSION,
        "state_spec_version": f_spec["state_spec_version"],
        "training_mode": TRAINING_MODE,
        "algorithm": algorithm.lower(),
        "ticker": ticker,
        "asset_class": asset_class,
        "round": round_num,
        "round_info": RETRAIN_ROUNDS[round_num],
        "feature_dim": FEATURE_DIM,
        "market_feature_dim": MARKET_FEATURE_DIM,
        "seq_len": SEQ_LEN,
        "feature_spec": f_spec,
        "feature_artifact_path": None,
        "preset": policy["preset"],
        "source_overrides": policy["source_overrides"],
        "excluded_contracts": policy["excluded_contracts"],
        "discrete_action_values": list(DISCRETE_ACTION_VALUES),
        "continuous_action_range": list(CONTINUOUS_ACTION_RANGE),
        "sigma_tgt": SIGMA_TGT,
        "bp": BP,
        "ewma_span": EWMA_SPAN,
        "return_horizons": list(HORIZONS),
        "macd_pairs": [list(p) for p in MACD_PAIRS_ACTIVE],
        "macd_vol_window": MACD_VOL_WINDOW,
        "rsi_window": RSI_WINDOW,
        "hyperparameters": {
            "lr": LR,
            "gamma": GAMMA,
            "batch_size": BATCH_SIZE,
            "memory_size_default": MEMORY_SIZE,
            "memory_size_note": f"actual value: max({MEMORY_SIZE_MIN}, total_expected_steps * {MEMORY_RATIO})",
            "memory_ratio": MEMORY_RATIO,
            "tau": TAU,
            "epsilon_mode": "3_stage_percentage",
            "eps_schedule": [list(s) for s in EPS_SCHEDULE],
            "eps_start": EPS_START,
            "eps_end": EPS_END,
            "eps_decay_steps": EPS_DECAY_STEPS,
            "episodes": EPISODES,
            "max_steps_per_ep": MAX_STEPS_PER_EP,
            "lstm_hidden_sizes": list(LSTM_HIDDEN_SIZES),
            "leaky_relu_slope": LEAKY_RELU_SLOPE,
            "dropout": DROPOUT,
            "validation_split": VALIDATION_SPLIT,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
            "use_huber_loss": USE_HUBER_LOSS,
            "huber_delta": HUBER_DELTA,
        },
        "reward_spec": {
            "name": "eq4_additive_price_difference",
            "return_definition": "r_t = p_t - p_{t-1}",
            "position_scale": "sigma_tgt / sigma_{t-1}",
            "transaction_cost": "bp * p_{t-1} * abs(scaled_position_t - scaled_position_{t-1})",
        },
        "architecture": {
            "family": "lstm_dqn",
            "lstm_hidden_sizes": list(LSTM_HIDDEN_SIZES),
            "head": "dueling_value_advantage",
            "fixed_q_targets": True,
            "target_network_update": f"hard_copy_every_{TAU}_learn_steps",
            "target_update_tau": TAU,
            "double_dqn": True,
            "dueling_dqn": True,
            "paper_reference_ids": [49, 18, 50],
        },
    }
    if extra:
        bp_value = extra.get("bp")
        if bp_value is not None:
            meta["reward_spec"]["bp"] = bp_value
            meta["reward_spec"]["bp_bps"] = int(bp_value * 10000)
        meta.update(extra)
    return meta
