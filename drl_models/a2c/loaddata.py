import os
import pandas as pd


PAPER_50 = [
    # Commodities
    "CC","DA","GI","JO","KC","KW","LB","NR","SB",
    "ZA","ZC","ZF","ZG","ZH","ZI","ZK","ZL","ZN",
    "ZO","ZP","ZR","ZT","ZU","ZW","ZZ",
    # Equity Index
    "CA","EN","ER","ES","LX","MD","SC","SP","XU","XX","YM",
    # Fixed Income
    "DT","FB","TY","UB","US",
    # FX
    "AN","BN","CN","DX","FN","JN","MP","NK","SN"
]
# -----------------------------
# Find CLCDATA floder
# -----------------------------
def find_clcdata(root_path):
    """Construct path to data/CLC from project root."""
    clc_path = os.path.join(root_path, 'data', 'CLC')
    if not os.path.isdir(clc_path):
        raise FileNotFoundError(f"CLC folder not found at {clc_path}")
    return clc_path
# -----------------------------
# Read sing RAD data
# -----------------------------
def read_rad_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    df.columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest"
    ]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").drop_duplicates("date")

    return df
# -----------------------------
# Main function
# -----------------------------
def load_paper_rad_data(root_path, start=None, end=None):

    clc_path = find_clcdata(root_path)

    data_dict = {}
    missing = []

    
    file_map = {}
    for root, _, files in os.walk(clc_path):
        for f in files:
            file_map[f.upper()] = os.path.join(root, f)

    for ticker in PAPER_50:
        fname = f"{ticker}_RAD.CSV".upper()

        if fname not in file_map:
            missing.append(ticker)
            continue

        df = read_rad_csv(file_map[fname])

        if start:
            df = df[df["date"] >= pd.to_datetime(start)]
        if end:
            df = df[df["date"] <= pd.to_datetime(end)]

        df["ticker"] = ticker

        data_dict[ticker] = df.reset_index(drop=True)


    panel = pd.concat(data_dict.values(), ignore_index=True)

    return data_dict, panel, missing


import numpy as np
import pandas as pd


def build_paper_features(data_dict, dropna=False):
    """
    Build paper-style features from load_paper_rad_data output.

    Parameters
    ----------
    data_dict : dict
        {ticker: DataFrame}
        Each DataFrame should contain at least:
        ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest']
        and optionally ['ticker'].

    dropna : bool
        If True, drop rows with missing core feature values.

    Returns
    -------
    feature_dict : dict
        {ticker: feature_df}
    feature_panel : DataFrame
        concatenated long-format feature panel
    """

    feature_dict = {}

    for ticker, df in data_dict.items():
        x = df.copy()

        required_cols = ["date", "open", "high", "low", "close", "volume", "open_interest"]
        missing_cols = [c for c in required_cols if c not in x.columns]
        if missing_cols:
            raise ValueError(f"{ticker} missing columns: {missing_cols}")

        x = x.sort_values("date").reset_index(drop=True)


        x["date"] = pd.to_datetime(x["date"], errors="coerce")
        for col in ["open", "high", "low", "close", "volume", "open_interest"]:
            x[col] = pd.to_numeric(x[col], errors="coerce")


        if "ticker" not in x.columns:
            x["ticker"] = ticker

        x["ret_1d"] = x["close"].diff()

        # 60-day EWMA volatility on daily additive returns
        x["ewm_vol_60"] = x["ret_1d"].ewm(
            span=60,
            adjust=False,
            min_periods=60
        ).std()

        x["ewm_vol_60"] = x["ewm_vol_60"].replace(0, np.nan)

        # -------------------------
        # 3) normalized close price series
        # 60-day rolling z-score
        # -------------------------
        close_mean_60 = x["close"].rolling(window=60, min_periods=60).mean()
        close_std_60 = x["close"].rolling(window=60, min_periods=60).std()
        close_std_60 = close_std_60.replace(0, np.nan)

        x["close_norm"] = (x["close"] - close_mean_60) / close_std_60

        # -------------------------
        # 4) return features
        #  1m, 2m, 3m, 1y
        #  sigma_t * sqrt(h) standardization
        # -------------------------
        horizons = {
            "ret_1m": 21,
            "ret_2m": 42,
            "ret_3m": 63,
            "ret_1y": 252
        }

        for feat, h in horizons.items():
            raw_col = f"{feat}_raw"
            x[raw_col] = x["close"] - x["close"].shift(h)
            x[feat] = x[raw_col] / (x["ewm_vol_60"] * np.sqrt(h))

        # -------------------------
        # 5) MACD features
        # paper:
        # q_t = (EMA(S) - EMA(L)) / std(p_{t-63:t})
        # MACD_t = q_t / std(q_{t-252:t})
        # -------------------------
        price_std_63 = x["close"].rolling(window=63, min_periods=63).std()
        price_std_63 = price_std_63.replace(0, np.nan)

        macd_pairs = [(8, 24), (16, 48), (32, 96)]
        macd_cols = []

        for short_span, long_span in macd_pairs:
            ema_s = x["close"].ewm(span=short_span, adjust=False, min_periods=short_span).mean()
            ema_l = x["close"].ewm(span=long_span, adjust=False, min_periods=long_span).mean()

            q_col = f"q_{short_span}_{long_span}"
            macd_col = f"macd_{short_span}_{long_span}"

            x[q_col] = (ema_s - ema_l) / price_std_63

            q_std_252 = x[q_col].rolling(window=252, min_periods=252).std()
            q_std_252 = q_std_252.replace(0, np.nan)

            x[macd_col] = x[q_col] / q_std_252
            macd_cols.append(macd_col)

        x["macd_avg"] = x[macd_cols].mean(axis=1)

        # -------------------------
        # 6) Wilder RSI(30)
        # -------------------------
        delta = x["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1/30, adjust=False, min_periods=30).mean()
        avg_loss = loss.ewm(alpha=1/30, adjust=False, min_periods=30).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        x["rsi_30"] = 100 - (100 / (1 + rs))


        x.loc[(avg_loss == 0) & (avg_gain > 0), "rsi_30"] = 100
        x.loc[(avg_loss == 0) & (avg_gain == 0), "rsi_30"] = 50


        keep_cols = [
            "date", "ticker",
            "open", "high", "low", "close", "volume", "open_interest",
            "ret_1d", "ewm_vol_60", "close_norm",
            "ret_1m_raw", "ret_2m_raw", "ret_3m_raw", "ret_1y_raw",
            "ret_1m", "ret_2m", "ret_3m", "ret_1y",
            "macd_8_24", "macd_16_48", "macd_32_96", "macd_avg",
            "rsi_30"
        ]

        x = x[keep_cols].copy()

        core_cols = [
            "ewm_vol_60", "close_norm",
            "ret_1m", "ret_2m", "ret_3m", "ret_1y",
            "macd_8_24", "macd_16_48", "macd_32_96",
            "rsi_30"
        ]

        if dropna:
            x = x.dropna(subset=core_cols).reset_index(drop=True)

        feature_dict[ticker] = x

    feature_panel = pd.concat(feature_dict.values(), ignore_index=True)
    feature_panel = feature_panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    return feature_dict, feature_panel

import numpy as np
import pandas as pd


def make_state_tensor_single(
    df,
    feature_cols,
    window=60,
    return_dates=True,
    return_current_row=False
):
    """
    Convert one ticker's daily feature table into rolling state tensors.

    Each sample at time t uses the previous `window` observations:
        X_t = features[t-window : t]

    So the aligned current row is row t.
    """
    x = df.copy()
    x = x.sort_values("date").reset_index(drop=True)


    x = x.dropna(subset=feature_cols).reset_index(drop=True)

    X = []
    dates = []
    rows = []

    for i in range(window, len(x)):
        state = x.loc[i - window:i - 1, feature_cols].to_numpy(dtype=np.float32)
        X.append(state)
        dates.append(x.loc[i, "date"])
        rows.append(x.loc[i].to_dict())

    if len(X) == 0:
        X = np.empty((0, window, len(feature_cols)), dtype=np.float32)
    else:
        X = np.stack(X).astype(np.float32)

    outputs = [X]

    if return_dates:
        outputs.append(np.array(dates))

    if return_current_row:
        aligned_df = pd.DataFrame(rows).reset_index(drop=True)
        outputs.append(aligned_df)

    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)


def make_state_tensors_all(feature_dict, feature_cols, window=60):
    """
    Build rolling state tensors for all tickers.
    """
    state_dict = {}

    for ticker, df in feature_dict.items():
        X, dates, aligned_df = make_state_tensor_single(
            df,
            feature_cols=feature_cols,
            window=window,
            return_dates=True,
            return_current_row=True
        )

        state_dict[ticker] = {
            "X": X,
            "dates": dates,
            "aligned_df": aligned_df
        }

    return state_dict

def build_baselines(df):

    out = df.copy()

    # --- Long Only ---
    out["long_only"] = 1.0

    # --- Sign(R) ---
    ret_1y = out["close"] - out["close"].shift(252)
    out["sign"] = np.sign(ret_1y)

    # --- MACD ---
    macd = out["macd_avg"]
    out["macd_signal"] = macd * np.exp(-macd**2 / 4) / 0.89


    out["long_only"] = out["long_only"].shift(1)
    out["sign"] = out["sign"].shift(1)
    out["macd_signal"] = out["macd_signal"].shift(1)

    return out

import numpy as np
import pandas as pd


def compute_pnl(
    df,
    signal_col,
    price_col="close",
    ret_col="ret_1d",
    vol_col="ewm_vol_60",
    sigma_target=0.064, 
    bp=0.0020,
    mu=1.0,
    dropna=False
):
    """
    Compute paper-style daily PnL for one strategy on one contract.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least:
        [price_col, ret_col, vol_col, signal_col]
    signal_col : str
        Column name of strategy position A_t in [-1, 1]
    price_col : str
        Usually 'close'
    ret_col : str
        Usually 'ret_1d' = p_t - p_{t-1}
    vol_col : str
        Usually 'ewm_vol_60'
    sigma_target : float
        sigma_tgt in the paper
    bp : float
        Cost rate. Paper Table 1 uses 0.0020 in training. :contentReference[oaicite:1]{index=1}
    mu : float
        Per-contract scaling factor. Paper sets mu = 1. :contentReference[oaicite:2]{index=2}
    dropna : bool
        If True, drop rows where key outputs are NaN

    Returns
    -------
    out : pd.DataFrame
        Original df plus:
        signal_prev
        vol_prev
        vol_prev2
        scaled_pos_prev
        scaled_pos_prev2
        gross_pnl
        trading_cost
        net_pnl
    """
    out = df.copy()

    # A_{t-1}, sigma_{t-1}, sigma_{t-2}, p_{t-1}
    out["signal_prev"] = out[signal_col].shift(1)
    out["signal_prev2"] = out[signal_col].shift(2)

    out["vol_prev"] = out[vol_col].shift(1)
    out["vol_prev2"] = out[vol_col].shift(2)

    out["price_prev"] = out[price_col].shift(1)


    out["vol_prev"] = out["vol_prev"].replace(0, np.nan)
    out["vol_prev2"] = out["vol_prev2"].replace(0, np.nan)

    # sigma_target / sigma * A
    out["scaled_pos_prev"] = sigma_target / out["vol_prev"] * out["signal_prev"]
    out["scaled_pos_prev2"] = sigma_target / out["vol_prev2"] * out["signal_prev2"]

    # gross pnl = mu * scaled_pos_{t-1} * r_t
    out["gross_pnl"] = mu * out["scaled_pos_prev"] * out[ret_col]

    # trading cost = mu * bp * p_{t-1} * |scaled_pos_{t-1} - scaled_pos_{t-2}|
    out["trading_cost"] = (
        mu
        * bp
        * out["price_prev"]
        * (out["scaled_pos_prev"] - out["scaled_pos_prev2"]).abs()
    )

    # net pnl
    out["net_pnl"] = out["gross_pnl"] - out["trading_cost"]

    if dropna:
        out = out.dropna(
            subset=[
                "scaled_pos_prev",
                "scaled_pos_prev2",
                "gross_pnl",
                "trading_cost",
                "net_pnl",
            ]
        ).reset_index(drop=True)

    return out

def build_portfolio_pnl(pnl_by_ticker, pnl_col="net_pnl"):
    """
    pnl_by_ticker: dict[ticker] = DataFrame returned by compute_pnl
    """
    frames = []
    for ticker, df in pnl_by_ticker.items():
        tmp = df[["date", pnl_col]].copy()
        tmp["ticker"] = ticker
        frames.append(tmp)

    panel = pd.concat(frames, ignore_index=True)

    port = (
        panel.groupby("date", as_index=False)[pnl_col]
        .mean()
        .rename(columns={pnl_col: "portfolio_pnl"})
    )

    return panel, port

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


