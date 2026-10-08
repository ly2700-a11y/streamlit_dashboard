"""Active asset-class-shared DQN model and agent."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# macOS/miniforge can load duplicate OpenMP runtimes before torch import. This
# keeps local inference usable; GPU servers can override the env var normally.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
    torch = None
    nn = None
    F = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

import drl_models.dqn.spec as _spec
from drl_models.dqn.spec import (
    BATCH_SIZE,
    DQN_SPEC_VERSION,
    DISCRETE_ACTION_DIM,
    DROPOUT,
    EPS_SCHEDULE,
    EPS_START,
    FEATURE_DIM,
    GRAD_CLIP_MAX_NORM,
    HUBER_DELTA,
    LEAKY_RELU_SLOPE,
    LR,
    LSTM_HIDDEN_SIZES,
    MEMORY_SIZE,
    SEQ_LEN,
    TAU,
    USE_HUBER_LOSS,
)

def resolve_device(device: str | None = None) -> str:
    if torch is None:
        return "cpu"
    requested = (device or os.environ.get("DRL_TORCH_DEVICE") or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA device, but torch.cuda.is_available() is False.")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise ValueError("Requested MPS device, but torch.backends.mps.is_available() is False.")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported torch device: {device}")
    return requested


DEVICE = resolve_device()


class DuelingDQNLSTM(nn.Module if nn is not None else object):
    def __init__(self):
        if _TORCH_IMPORT_ERROR is not None:
            raise RuntimeError("PyTorch is required for DQN model usage but is not installed in this environment.") from _TORCH_IMPORT_ERROR
        super().__init__()
        h1, h2 = LSTM_HIDDEN_SIZES
        self.lstm1 = nn.LSTM(FEATURE_DIM, h1, batch_first=True)
        self.lstm2 = nn.LSTM(h1, h2, batch_first=True)
        self.dropout = nn.Dropout(p=DROPOUT)
        self.value = nn.Linear(h2, 1)
        self.advantage = nn.Linear(h2, DISCRETE_ACTION_DIM)

        for name, param in self.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(param, gain=nn.init.calculate_gain("tanh"))
            elif "weight" in name:
                nn.init.orthogonal_(param, gain=0.1)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = F.leaky_relu(out, LEAKY_RELU_SLOPE)
        out, _ = self.lstm2(out)
        out = F.leaky_relu(out, LEAKY_RELU_SLOPE)
        out = self.dropout(out)
        last = out[:, -1, :]
        value = self.value(last)
        advantage = self.advantage(last)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class FCHeadDQNLSTM(nn.Module if nn is not None else object):
    """Single-FC-head LSTM DQN used by the retained Forex checkpoints."""

    def __init__(self):
        if _TORCH_IMPORT_ERROR is not None:
            raise RuntimeError("PyTorch is required for DQN model usage but is not installed in this environment.") from _TORCH_IMPORT_ERROR
        super().__init__()
        h1, h2 = LSTM_HIDDEN_SIZES
        self.lstm1 = nn.LSTM(FEATURE_DIM, h1, batch_first=True)
        self.lstm2 = nn.LSTM(h1, h2, batch_first=True)
        self.fc = nn.Linear(h2, DISCRETE_ACTION_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = F.leaky_relu(out, LEAKY_RELU_SLOPE)
        out, _ = self.lstm2(out)
        out = F.leaky_relu(out, LEAKY_RELU_SLOPE)
        return self.fc(out[:, -1, :])


class ReplayBuffer:
    def __init__(self, capacity: int = MEMORY_SIZE):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, s, a, r, s2, d):
        item = (np.asarray(s, dtype=np.float32), int(a), float(r), np.asarray(s2, dtype=np.float32), float(d))
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int = BATCH_SIZE):
        batch = random.sample(self.buffer, batch_size)
        return self._to_arrays(batch)

    def _to_arrays(self, batch):
        states = np.stack([x[0] for x in batch])
        actions = np.array([x[1] for x in batch], dtype=np.int64)
        rewards = np.array([x[2] for x in batch], dtype=np.float32)
        next_states = np.stack([x[3] for x in batch])
        dones = np.array([x[4] for x in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def action_distribution(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for t in self.buffer:
            counts[t[1]] = counts.get(t[1], 0) + 1
        return counts

    def reward_distribution(self) -> dict:
        rewards = np.array([t[2] for t in self.buffer])
        if len(rewards) == 0:
            return {"mean": 0, "std": 0, "near_zero_pct": 0, "high_abs_pct": 0}
        return {
            "mean": float(rewards.mean()),
            "std": float(rewards.std()),
            "near_zero_pct": float((np.abs(rewards) < 0.01).mean() * 100),
            "high_abs_pct": float((np.abs(rewards) > 0.5).mean() * 100),
        }

    def turnover_rate(self) -> float:
        if len(self.buffer) < 2:
            return 0.0
        actions = [t[1] for t in self.buffer]
        changes = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i - 1])
        return changes / (len(actions) - 1)

    def __len__(self):
        return len(self.buffer)


class StratifiedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int = MEMORY_SIZE, mode: str = "uniform"):
        super().__init__(capacity)
        if mode not in ("uniform", "action_balanced", "reward_stratified"):
            raise ValueError(f"Unknown replay mode: {mode}")
        self.mode = mode

    def sample(self, batch_size: int = BATCH_SIZE):
        if self.mode == "uniform" or len(self.buffer) < batch_size:
            return super().sample(batch_size)
        if self.mode == "action_balanced":
            return self._sample_action_balanced(batch_size)
        return self._sample_reward_stratified(batch_size)

    def _sample_action_balanced(self, batch_size: int):
        per_action = batch_size // (DISCRETE_ACTION_DIM + 1)
        batch = []
        for action_id in range(DISCRETE_ACTION_DIM):
            pool = [t for t in self.buffer if t[1] == action_id]
            if pool:
                n = min(per_action, len(pool))
                batch.extend(random.sample(pool, n))
        remaining = batch_size - len(batch)
        if remaining > 0:
            sorted_by_abs_r = sorted(self.buffer, key=lambda t: -abs(t[2]))
            high_reward = sorted_by_abs_r[:max(remaining * 3, 100)]
            n = min(remaining, len(high_reward))
            batch.extend(random.sample(high_reward, n))
        random.shuffle(batch)
        return self._to_arrays(batch[:batch_size])

    def _sample_reward_stratified(self, batch_size: int):
        rewards = np.array([t[2] for t in self.buffer])
        abs_r = np.abs(rewards)
        n_bins = 4
        edges = np.percentile(abs_r, np.linspace(0, 100, n_bins + 1))
        per_bin = batch_size // n_bins
        batch = []
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            if i == n_bins - 1:
                pool_idx = np.where((abs_r >= lo) & (abs_r <= hi))[0]
            else:
                pool_idx = np.where((abs_r >= lo) & (abs_r < hi))[0]
            if len(pool_idx) == 0:
                continue
            n = min(per_bin, len(pool_idx))
            chosen = random.sample(list(pool_idx), n)
            batch.extend(self.buffer[j] for j in chosen)
        remaining = batch_size - len(batch)
        if remaining > 0:
            batch.extend(random.sample(self.buffer, min(remaining, len(self.buffer))))
        random.shuffle(batch)
        return self._to_arrays(batch[:batch_size])


class DQNAgent:
    def __init__(
        self,
        device: str | None = None,
        total_steps: int = 100000,
        memory_size: int = MEMORY_SIZE,
        clip_grad_norm: float = GRAD_CLIP_MAX_NORM,
        use_huber: bool = USE_HUBER_LOSS,
        replay_mode: str = "uniform",
    ):
        if _TORCH_IMPORT_ERROR is not None:
            raise RuntimeError("PyTorch is required for DQN training/inference but is not installed in this environment.") from _TORCH_IMPORT_ERROR
        self.device = resolve_device(device)
        self.total_steps = max(1, int(total_steps))
        self.clip_grad_norm = float(clip_grad_norm)
        self.use_huber = bool(use_huber)
        self.q_net = DuelingDQNLSTM().to(self.device)
        self.target = DuelingDQNLSTM().to(self.device)
        self.target.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LR)
        self.replay = StratifiedReplayBuffer(memory_size, mode=replay_mode)
        self.train_steps = 0
        self.target_updates = 0

    def epsilon_for_step(self, step: int) -> float:
        """Look up epsilon from EPS_SCHEDULE based on fraction of total steps."""
        frac = min(1.0, step / self.total_steps)
        # Interpolate linearly between schedule waypoints
        prev_frac, prev_eps = 0.0, EPS_SCHEDULE[0][1]
        for sched_frac, sched_eps in EPS_SCHEDULE:
            if frac <= sched_frac:
                # Linear interpolation between prev and current waypoint
                if sched_frac == prev_frac:
                    return sched_eps
                alpha = (frac - prev_frac) / (sched_frac - prev_frac)
                return prev_eps + alpha * (sched_eps - prev_eps)
            prev_frac, prev_eps = sched_frac, sched_eps
        return prev_eps  # past end of schedule

    def act(self, state: np.ndarray, eps: float) -> int:
        if np.random.random() < eps:
            return np.random.randint(DISCRETE_ACTION_DIM)
        self.q_net.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
            return int(self.q_net(tensor).argmax().item())

    def act_batch(self, states: np.ndarray, eps: float) -> list[int]:
        """Batched action selection: one forward pass for N states.

        Each state independently gets epsilon-greedy exploration.
        """
        n = len(states)
        # Determine which indices explore vs exploit
        explore = np.random.random(n) < eps
        actions = np.random.randint(DISCRETE_ACTION_DIM, size=n)

        if not explore.all():
            exploit_idx = np.where(~explore)[0]
            self.q_net.eval()
            with torch.no_grad():
                tensor = torch.from_numpy(
                    np.asarray(states, dtype=np.float32
                )[exploit_idx]).to(self.device)
                q_vals = self.q_net(tensor).argmax(dim=1).detach().cpu().numpy()
            actions[exploit_idx] = q_vals

        return actions.tolist()

    def predict_action_id(self, state: np.ndarray) -> int:
        return int(self.predict_action_ids(np.asarray(state, dtype=np.float32)[None, ...])[0])

    def predict_action_ids(self, states: np.ndarray) -> np.ndarray:
        self.q_net.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(self.device)
            return self.q_net(tensor).argmax(dim=1).detach().cpu().numpy().astype(np.int64)

    def push(self, s, a, r, s2, d):
        self.replay.push(s, a, r, s2, d)

    def learn(self) -> float:
        if len(self.replay) < BATCH_SIZE:
            return 0.0

        self.q_net.train()
        states, actions, rewards, next_states, dones = self.replay.sample(BATCH_SIZE)
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)

        next_actions = self.q_net(next_states).argmax(1).detach()
        next_q = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1).detach()
        target_q = rewards + (1.0 - dones) * _spec.GAMMA * next_q

        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        if self.use_huber:
            loss = F.smooth_l1_loss(current_q, target_q, beta=HUBER_DELTA)
        else:
            loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % TAU == 0:
            self.target.load_state_dict(self.q_net.state_dict())
            self.target_updates += 1
        return float(loss.item())

    def save(self, path: str | Path, metadata: dict | None = None, include_training_state: bool = False):
        payload = {
            "spec_version": DQN_SPEC_VERSION,
            "metadata": metadata or {},
            "q_net": self.q_net.state_dict(),
            "target_net": self.target.state_dict(),
        }
        if include_training_state:
            payload["training_state"] = {
                "optimizer": self.optimizer.state_dict(),
                "train_steps": self.train_steps,
                "target_updates": self.target_updates,
                "replay": self.replay.buffer.copy(),
                "replay_position": self.replay.position,
                "torch_rng": torch.random.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
            }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load(self, path: str | Path, resume: bool = False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if "q_net" in checkpoint and "target_net" in checkpoint:
            q_state = checkpoint["q_net"]
            target_state = checkpoint["target_net"]
        elif "q" in checkpoint and "t" in checkpoint:
            q_state = checkpoint["q"]
            target_state = checkpoint["t"]
        else:
            raise ValueError("Checkpoint is not a valid DQN checkpoint with q_net/target_net.")
        if "fc.weight" in q_state:
            self.q_net = FCHeadDQNLSTM().to(self.device)
            self.target = FCHeadDQNLSTM().to(self.device)
            self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LR)
        self.q_net.load_state_dict(q_state)
        self.target.load_state_dict(target_state)
        if resume and "training_state" in checkpoint:
            ts = checkpoint["training_state"]
            self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LR)
            self.optimizer.load_state_dict(ts["optimizer"])
            self.train_steps = ts["train_steps"]
            self.target_updates = ts["target_updates"]
            self.replay.buffer = ts["replay"].copy()
            self.replay.position = ts["replay_position"]
            if "torch_rng" in ts:
                torch.random.set_rng_state(ts["torch_rng"])
            if "numpy_rng" in ts:
                np.random.set_state(ts["numpy_rng"])
            if "python_rng" in ts:
                random.setstate(ts["python_rng"])
        return checkpoint.get("metadata", {})


SharedDQNAgent = DQNAgent
