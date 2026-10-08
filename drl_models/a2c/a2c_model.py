"""
a2c_model.py
============
A2C model core extracted from `project code.ipynb`.

Exports
-------
- StackedLSTMBackbone
- ActorContinuous
- CriticValue
- PaperTradingEnv
- collect_rollout
- compute_returns_and_advantages
- PaperA2CTrainer
- build_envs_from_state_dict
- load_actor_from_checkpoint   ← convenience helper for regime_detection tests

Usage from regime_detection
---------------------------
    from drl_models.a2c.a2c_model import load_actor_from_checkpoint

    actor = load_actor_from_checkpoint(
        path='drl_models/a2c/a2c_All_period1.pt',
        n_features=9,          # 9-dim state; use 12 if regime probs are appended
    )
    state_tensor = ...         # shape (batch, 60, n_features)
    action = actor.deterministic_action(state_tensor)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ============================================================
# Paper-aligned defaults (only what is explicitly stated)
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LR_ACTOR = 1e-4       # paper Table 1
LR_CRITIC = 1e-3      # paper Table 1
GAMMA = 0.3           # paper Table 1
BATCH_SIZE = 128      # paper Table 1
BP = 0.0020           # paper Table 1
MU = 1.0              # paper Eq.4 text
WINDOW = 60           # past 60 observations in state
HIDDEN_SIZES = [64, 32]   # paper Section 4.3
LEAKY_RELU_SLOPE = 0.01


# ============================================================
# 1) Two-layer LSTM backbone
#    Paper says:
#      - LSTM for actor and critic
#      - two-layer LSTM
#      - 64 and 32 units
#      - Leaky-ReLU activation
# ============================================================
class StackedLSTMBackbone(nn.Module):
    def __init__(self, input_size, hidden_sizes=(64, 32)):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_sizes[0],
            num_layers=1,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_sizes[0],
            hidden_size=hidden_sizes[1],
            num_layers=1,
            batch_first=True,
        )
        self.act = nn.LeakyReLU(LEAKY_RELU_SLOPE)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out1, _ = self.lstm1(x)
        out1 = self.act(out1)

        out2, _ = self.lstm2(out1)
        out2 = self.act(out2)

        # IMPLEMENTATION ASSUMPTION:
        # paper does not specify pooling/readout method from LSTM.
        # Standard choice: use last time-step hidden representation.
        return out2[:, -1, :]   # (batch, 32)


# ============================================================
# 2) Actor: continuous action in [-1, 1]
#    Paper states A2C uses continuous action space [−1,1],
#    but does not specify the exact policy distribution.
# ============================================================
class ActorContinuous(nn.Module):
    def __init__(self, input_size, hidden_sizes=(64, 32)):
        super().__init__()
        self.backbone = StackedLSTMBackbone(input_size, hidden_sizes)
        self.mu_head = nn.Linear(hidden_sizes[-1], 1)

        # IMPLEMENTATION ASSUMPTION:
        # paper does not specify how to parameterize continuous action policy.
        # We use a Gaussian policy with a trainable global log_std.
        self.log_std = nn.Parameter(torch.tensor(-0.5, dtype=torch.float32))

    def forward(self, x):
        h = self.backbone(x)
        mu = torch.tanh(self.mu_head(h))   # action mean constrained to [-1,1]
        std = torch.exp(self.log_std).clamp(min=1e-4, max=2.0)
        return mu, std

    def sample_action(self, x):
        mu, std = self.forward(x)
        dist = Normal(mu, std)

        z = dist.rsample()
        action = torch.tanh(z)

        # tanh squash correction
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def deterministic_action(self, x):
        mu, _ = self.forward(x)
        return torch.tanh(mu)


# ============================================================
# 3) Critic: state value V(s)
# ============================================================
class CriticValue(nn.Module):
    def __init__(self, input_size, hidden_sizes=(64, 32)):
        super().__init__()
        self.backbone = StackedLSTMBackbone(input_size, hidden_sizes)
        self.value_head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x):
        h = self.backbone(x)
        v = self.value_head(h)
        return v.squeeze(-1)


# ============================================================
# 4) Paper-style trading environment
#
#    Expected inputs:
#      X: shape (n_samples, 60, n_features)
#      aligned_df: DataFrame aligned with X, containing at least:
#        date, close, ret_1d, ewm_vol_60
#
#    Reward logic follows the paper's Eq.(4), with timing written as:
#      choose A_t from state at time t
#      receive reward on t -> t+1
#
#    sigma_target is left as an input argument because the paper
#    includes sigma_tgt in Eq.(4) but does not fix one universal value.
# ============================================================
class PaperTradingEnv:
    def __init__(
        self,
        X,
        aligned_df,
        sigma_target,
        bp=BP,
        mu=MU,
    ):
        self.X = np.asarray(X, dtype=np.float32)
        self.df = aligned_df.reset_index(drop=True).copy()
        self.sigma_target = sigma_target
        self.bp = bp
        self.mu = mu

        required_cols = ["date", "close", "ret_1d", "ewm_vol_60"]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"aligned_df missing required columns: {missing}")

        if len(self.X) != len(self.df):
            raise ValueError("X and aligned_df must have the same number of samples.")

        self.n = len(self.df)
        self.reset()

    def reset(self):
        self.t = 0
        self.prev_action = 0.0
        return self.X[self.t]

    def step(self, action):
        action = float(np.clip(action, -1.0, 1.0))

        if self.t >= self.n - 1:
            return self.X[self.t], 0.0, True, {}

        row_t = self.df.iloc[self.t]
        row_tp1 = self.df.iloc[self.t + 1]

        sigma_t = row_t["ewm_vol_60"]
        sigma_tm1 = self.df.iloc[self.t - 1]["ewm_vol_60"] if self.t - 1 >= 0 else np.nan
        price_t = row_t["close"]
        r_tp1 = row_tp1["ret_1d"]

        if pd.isna(sigma_t) or sigma_t == 0 or pd.isna(price_t) or pd.isna(r_tp1):
            reward = np.nan
        else:
            scaled_pos_t = self.sigma_target / sigma_t * action

            if self.t == 0 or pd.isna(sigma_tm1) or sigma_tm1 == 0:
                scaled_pos_tm1 = 0.0
            else:
                scaled_pos_tm1 = self.sigma_target / sigma_tm1 * self.prev_action

            gross = self.mu * scaled_pos_t * r_tp1
            cost = self.mu * self.bp * price_t * abs(scaled_pos_t - scaled_pos_tm1)
            reward = gross - cost

        self.prev_action = action
        self.t += 1
        done = self.t >= self.n - 1

        next_state = self.X[self.t]
        info = {
            "date": row_tp1["date"],
            "reward": reward,
        }
        return next_state, float(0.0 if pd.isna(reward) else reward), done, info


# ============================================================
# 5) Collect synchronous rollouts across environments
# ============================================================
def collect_rollout(envs, actor, critic, rollout_steps, device=DEVICE):
    states = []
    actions = []
    rewards = []
    dones = []
    values = []

    if not hasattr(collect_rollout, "_current_states") or len(collect_rollout._current_states) != len(envs):
        collect_rollout._current_states = [env.reset() for env in envs]

    current_states = collect_rollout._current_states

    for _ in range(rollout_steps):
        state_batch = torch.tensor(np.stack(current_states), dtype=torch.float32, device=device)

        with torch.no_grad():
            action_batch, _, _ = actor.sample_action(state_batch)
            value_batch = critic(state_batch)

        next_states = []
        reward_batch = []
        done_batch = []

        for i, env in enumerate(envs):
            a = action_batch[i].item()
            ns, r, d, _ = env.step(a)

            if d:
                ns = env.reset()

            next_states.append(ns)
            reward_batch.append(r)
            done_batch.append(float(d))

        states.append(state_batch)
        actions.append(action_batch.squeeze(-1))
        rewards.append(torch.tensor(reward_batch, dtype=torch.float32, device=device))
        dones.append(torch.tensor(done_batch, dtype=torch.float32, device=device))
        values.append(value_batch)

        current_states = next_states

    collect_rollout._current_states = current_states

    last_state_batch = torch.tensor(np.stack(current_states), dtype=torch.float32, device=device)
    with torch.no_grad():
        last_values = critic(last_state_batch)

    batch = {
        "states": torch.stack(states),       # (T, N, 60, F)
        "actions": torch.stack(actions),     # (T, N)
        "rewards": torch.stack(rewards),     # (T, N)
        "dones": torch.stack(dones),         # (T, N)
        "values": torch.stack(values),       # (T, N)
        "last_values": last_values,          # (N,)
    }
    return batch


# ============================================================
# 6) Returns / advantages
# ============================================================
def compute_returns_and_advantages(rewards, dones, values, last_values, gamma=GAMMA):
    T, N = rewards.shape
    returns = torch.zeros_like(rewards)
    running = last_values

    for t in reversed(range(T)):
        running = rewards[t] + gamma * (1.0 - dones[t]) * running
        returns[t] = running

    advantages = returns - values
    return returns, advantages


# ============================================================
# 7) Main A2C trainer
#
#    Checkpoint / resume is added as an engineering feature.
#    It does not change the paper's model or hyperparameters.
# ============================================================
class PaperA2CTrainer:
    def __init__(
        self,
        n_features,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        gamma=GAMMA,
        entropy_coef=1e-3,    # IMPLEMENTATION ASSUMPTION: not specified in paper
        max_grad_norm=1.0,    # IMPLEMENTATION ASSUMPTION: not specified in paper
        device=DEVICE,
    ):
        self.device = device
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.actor = ActorContinuous(n_features, HIDDEN_SIZES).to(device)
        self.critic = CriticValue(n_features, HIDDEN_SIZES).to(device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.train_log = []

    def update(self, batch):
        T, N, W, feat_dim = batch["states"].shape
        states_flat = batch["states"].reshape(T * N, W, feat_dim)

        returns, advantages = compute_returns_and_advantages(
            batch["rewards"],
            batch["dones"],
            batch["values"],
            batch["last_values"],
            gamma=self.gamma,
        )

        returns_flat = returns.reshape(T * N)
        advantages_flat = advantages.reshape(T * N)
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        # ---- Critic ----
        values_pred = self.critic(states_flat)
        critic_loss = F.mse_loss(values_pred, returns_flat)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optim.step()

        # ---- Actor ----
        mu, std = self.actor(states_flat)
        dist = Normal(mu, std)

        actions_flat = batch["actions"].reshape(T * N).unsqueeze(-1).clamp(-0.999, 0.999)
        z = 0.5 * torch.log((1 + actions_flat) / (1 - actions_flat))
        log_probs = dist.log_prob(z) - torch.log(1 - actions_flat.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1).mean()
        actor_loss = -(log_probs * advantages_flat.detach()).mean() - self.entropy_coef * entropy

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optim.step()

        out = {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(entropy.item()),
            "avg_return": float(returns_flat.mean().item()),
            "avg_reward": float(batch["rewards"].mean().item()),
        }
        self.train_log.append(out)
        return out

    def save_checkpoint(self, path, extra_state=None):
        ckpt = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optim_state_dict": self.actor_optim.state_dict(),
            "critic_optim_state_dict": self.critic_optim.state_dict(),
            "train_log": self.train_log,
            "gamma": self.gamma,
            "entropy_coef": self.entropy_coef,
            "max_grad_norm": self.max_grad_norm,
        }
        if extra_state is not None:
            ckpt["extra_state"] = extra_state

        torch.save(ckpt, path)

    def load_checkpoint(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device)

        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.actor_optim.load_state_dict(ckpt["actor_optim_state_dict"])
        self.critic_optim.load_state_dict(ckpt["critic_optim_state_dict"])
        self.train_log = ckpt.get("train_log", [])

        return ckpt

    def fit(
        self,
        envs,
        n_updates=1000,
        rollout_steps=32,
        log_every=50,
        checkpoint_path=None,
        checkpoint_every=50,
        start_update=0,
    ):
        # avoid rollout state leaking across asset classes / separate runs
        if hasattr(collect_rollout, "_current_states"):
            del collect_rollout._current_states

        for update_idx in range(start_update + 1, start_update + n_updates + 1):
            batch = collect_rollout(
                envs=envs,
                actor=self.actor,
                critic=self.critic,
                rollout_steps=rollout_steps,
                device=self.device,
            )

            metrics = self.update(batch)

            if update_idx % log_every == 0:
                print(
                    f"[{update_idx:4d}] "
                    f"actor_loss={metrics['actor_loss']:.4f} "
                    f"critic_loss={metrics['critic_loss']:.4f} "
                    f"avg_reward={metrics['avg_reward']:.6f} "
                    f"entropy={metrics['entropy']:.4f}"
                )

            if checkpoint_path is not None and update_idx % checkpoint_every == 0:
                self.save_checkpoint(
                    checkpoint_path,
                    extra_state={
                        "update_idx": update_idx,
                        "rollout_steps": rollout_steps,
                    }
                )
                print(f"checkpoint saved to {checkpoint_path}")

        return self


# ============================================================
# 8) Build envs from state_dict
# ============================================================
def build_envs_from_state_dict(
    state_dict,
    tickers=None,
    sigma_target=None,
    bp=BP,
    mu=MU,
):
    if sigma_target is None:
        raise ValueError("sigma_target must be provided explicitly because the paper does not fix a universal value.")

    envs = []
    selected = tickers if tickers is not None else list(state_dict.keys())

    for ticker in selected:
        if ticker not in state_dict:
            continue

        X = state_dict[ticker]["X"]
        aligned_df = state_dict[ticker]["aligned_df"].copy()

        if not isinstance(aligned_df, pd.DataFrame):
            continue
        if len(aligned_df) == 0 or len(X) == 0:
            continue
        if not all(c in aligned_df.columns for c in ["close", "ret_1d", "ewm_vol_60"]):
            continue

        usable = aligned_df[["close", "ret_1d", "ewm_vol_60"]].notna().all(axis=1).values
        X = X[usable]
        aligned_df = aligned_df.loc[usable].reset_index(drop=True)

        if len(X) > 2:
            envs.append(
                PaperTradingEnv(
                    X=X,
                    aligned_df=aligned_df,
                    sigma_target=sigma_target,
                    bp=bp,
                    mu=mu,
                )
            )
    return envs


# ============================================================
# 9) Convenience: load a pre-trained actor from a .pt checkpoint
#
#    The .pt files saved by PaperA2CTrainer.save_checkpoint()
#    contain "actor_state_dict" and "critic_state_dict".
#
#    Parameters
#    ----------
#    path       : path to the .pt file, e.g. '../rl_models/a2c_All_period1.pt'
#    n_features : number of input features per time-step.
#                 Use 9 for the original state; use 12 if you appended
#                 the 3-dim regime soft-probabilities (Route B).
#    device     : 'cpu' or 'cuda'
#
#    Returns
#    -------
#    actor : ActorContinuous, in eval() mode, ready for inference
# ============================================================
def load_actor_from_checkpoint(path, n_features=9, device=None):
    if device is None:
        device = DEVICE
    ckpt = torch.load(path, map_location=device)
    actor = ActorContinuous(n_features, HIDDEN_SIZES).to(device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    return actor


def load_critic_from_checkpoint(path, n_features=9, device=None):
    if device is None:
        device = DEVICE
    ckpt = torch.load(path, map_location=device)
    critic = CriticValue(n_features, HIDDEN_SIZES).to(device)
    critic.load_state_dict(ckpt["critic_state_dict"])
    critic.eval()
    return critic
