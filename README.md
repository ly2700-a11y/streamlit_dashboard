# IEOR 4733 — Deep Reinforcement Learning for Trading

Reproduction of **Zhang, Zohren, Roberts (2019)** — *Deep Reinforcement Learning for Trading* — extended with an A2C regime-detection branch and an interactive Streamlit dashboard.

Paper: [arXiv 1911.10107](https://arxiv.org/pdf/1911.10107)

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r streamlit_requirements.txt

# 3. Launch the dashboard
streamlit run src/app/main.py --server.port 8501
```

Open **http://localhost:8501** in your browser.

Alternatively use the helper script:

```bash
chmod +x run_app.sh
./run_app.sh
```

---

## Platform Overview

The Streamlit dashboard provides four tabs.

### Tab 1 — Strategy Comparison

Displays cumulative trade-return curves for all six strategies across five asset classes (Commodity, Equity Index, Fixed Income, Forex, All).

- Strategies shown: **Long Only**, **Sign(R)**, **MACD**, **A2C**, **A2C + Regime (B)**, **DQN (Paper)**
- Pre-computed results are loaded from CSV files — no live computation at render time
- Use the **sidebar** to filter by date range (default 2011–2019) and to show or hide individual strategies
- Each asset class is shown on its own sub-tab; all curves are colour-coded consistently

### Tab 2 — Performance Metrics

Tabular summary of key metrics for every strategy × asset class combination within the selected date range.

| Column | Description |
|---|---|
| Asset Class | Commodity / Equity Index / Fixed Income / Forex |
| Strategy | One of the six strategies |
| Total Return (%) | Cumulative return over the period |
| Ann. Return (%) | Annualised return (CAGR) |
| Ann. Volatility (%) | Annualised standard deviation of daily returns |
| Sharpe Ratio | Ann. Return ÷ Ann. Volatility |
| Max Drawdown (%) | Largest peak-to-trough decline |

A **Download Metrics CSV** button exports the current table.

### Tab 3 — Risk Analysis

Side-by-side horizontal bar charts comparing all selected strategies:

- **Max Drawdown Comparison** — sorted ascending by drawdown depth
- **Volatility Comparison** — sorted ascending by annualised volatility

Both charts respect the sidebar date range and strategy selection.

### Tab 4 — Data Pipeline

Reference information about the underlying data:

- Data cleaning workflow (raw RAD → validated → feature-engineered → strategy-ready)
- Supported file formats (CLCData RAD CSV)
- Summary metrics: data period, asset classes, approximate trading-day count
- Data source locations within the repository

---

## Sidebar Controls

| Control | Effect |
|---|---|
| **Start Date / End Date** | Filters all charts and metric calculations to the selected window |
| **Select Strategies** | Show or hide individual strategies across all tabs |

---

## Repository Structure

```
IEOR4733_Project/
├── src/app/                  # Streamlit application
├── data/                     # Raw and pre-processed market data
├── reproduction_of_figures/  # Notebook + pre-computed strategy CSVs
├── regime_detection/         # FFT + HMM regime model (Route B)
├── drl_models/               # Trained DRL model weights and code
├── rl_models/                # A2C results (legacy path)
├── docs/                     # Audit reports and analysis documents
├── config/                   # Contract lists and roll-mapping config
├── references/               # Literature notes and reference list
├── tests/                    # Ad-hoc analysis and bridge scripts
├── tests_MACD/               # MACD-specific test scripts
├── tests_Signr/              # Sign(R)-specific test scripts
└── final_requirement/        # Submission artefacts
```

### `src/app/`

The Streamlit application.

| File | Purpose |
|---|---|
| `main.py` | Entry point — all UI layout, data loading, and chart rendering |
| `main_backup_v1.py` | Earlier version kept for reference |
| `main_complex.py` | Experimental richer version (not currently served) |

### `data/`

All market data used by the backtests.

| Path | Contents |
|---|---|
| `data/CLC/` | Raw CLCData RAD CSV files — one file per futures contract |
| `data/yahoo/` | Yahoo Finance price data used for equity-index fallback |
| `data/index_data.csv` | Pre-merged index-level daily prices |
| `data/risk_free_rate.csv` | Daily risk-free rate series |

### `reproduction_of_figures/`

Everything needed to reproduce the paper's strategy-comparison plots.

| Path | Contents |
|---|---|
| `strategies_comparison.ipynb` | Master notebook — runs all six strategies and exports results |
| `baseline_results/` | Pre-computed CSVs from the notebook (pure RAD, no overrides); **these are the files read by the Streamlit app** |
| `a2c_csv_data/` | Older A2C result files with long-override adjustments (cross-checks only) |
| `*.png` | Strategy comparison figures at various stages of development |

### `regime_detection/`

Frequency-based regime detection (Route B) layered on top of A2C.

| File | Purpose |
|---|---|
| `timeseries_fft_regime.py` | Core FFT-based regime classifier |
| `route_b_train.py` | Training script for the regime-conditioned A2C model |
| `route_b_experiment.ipynb` | Exploratory notebook for regime analysis |
| `reeval_route_b_final.py` | Final evaluation — generates `results/` CSVs used by the app |
| `results/` | Per-asset per-period cumulative return CSVs for the Route B model |

### `drl_models/`

Trained model weights and code for the deep RL strategies.

| Path | Contents |
|---|---|
| `drl_models/a2c/` | A2C model definition (`a2c_model.py`), pre-trained `.pt` weights (Original / Extension Reward variants), data loader, and per-asset result CSVs |
| `drl_models/dqn/` | DQN implementation, trained weights, and figure/data output |

### `rl_models/`

Legacy path retained for backward compatibility.

| File | Purpose |
|---|---|
| `a2c_results_wide.csv` | Wide-format A2C results across all asset classes and periods |

### `config/`

Static configuration shared across scripts.

| File | Purpose |
|---|---|
| `contract_months.json` | Delivery months for each futures contract |
| `current_roll_mapping.txt` | Active roll-date mapping used by the data loader |

### `docs/`

Accumulated audit reports produced during development. These document metric alignment, data-source investigations, and iteration histories (e.g. `calmar_alignment_iteration.md`, `table3_sweep_report.md`). Useful for understanding why specific design choices were made.

### `references/`

| File | Contents |
|---|---|
| `ref_list.txt` | Full bibliography |
| `DRL_journal.txt` | Reading notes on DRL literature |

### `tests/`, `tests_MACD/`, `tests_Signr/`

One-off analysis scripts, bridge comparisons, and strategy-specific sweep scripts. Not part of the main app but useful for reproducing specific numbers in the audit reports.

---

## Key Top-Level Files

| File | Purpose |
|---|---|
| `baseline_run.py` | Core backtesting engine — loads contracts, computes returns, and reports Table 3 metrics for Long Only / Sign(R) / MACD. Entry point for all command-line reproductions. |
| `config.py` | Central configuration: asset-class groupings, excluded contracts, source overrides |
| `data_loader.py` | `load_clc_full()` — reads CLCData RAD CSVs, validates prices, aligns dates |
| `indicators.py` | Technical indicator calculations (MACD, EWM, etc.) |
| `metrics.py` | `compute_metrics()`, `max_drawdown_from_path()`, `cagr_from_path()` |
| `strategies.py` | Position-sizing logic for Long Only, Sign(R), and MACD |
| `vol_scaling.py` | Volatility-targeting and position scaling utilities |
| `generate_table3_comparison.py` | Quick console comparison of all strategies in Table 3 format |
| `generate_table3_markdown.py` | Writes a Markdown file of the same table for reporting |
| `generate_results_tables.py` | Exports full results tables used by the Streamlit app |
| `repro_analysis.py` | Diagnostic script comparing reproduced vs paper metrics |
| `test_baseline.py` | Unit tests for the baseline pipeline |
| `verify_setup.py` | Checks that all data files and dependencies are present |
| `run_app.sh` | One-command launcher for the Streamlit app |
| `streamlit_requirements.txt` | Python dependencies for the app |
| `PROJECT_MEMORY.md` | Running development log — read this first when resuming work |

---

## Command-Line Reproductions

```bash
# Table 3 — live baseline (all 50 contracts)
python baseline_run.py --table 3 --all-metrics --sigma 0.058

# Generate Table 3 Markdown report
python generate_table3_markdown.py

# Reproduce regime detection results
python regime_detection/reeval_route_b_final.py

# Run unit tests
python test_baseline.py
```

---

## Paper Results vs Reproduction

| Version | Contracts ≤ 15 % / 45 | Notes |
|---|---|---|
| Paper target | 45/45 | Ground truth |
| Live baseline | ~31/45 | All contracts, same-rule, pure RAD |
| Clean same-rule max | ~34/45 | Best achievable under same-rule doctrine |
| Experimental upper bound | ~41/45 | Excludes 5 equity contracts; uses legacy source for Equity |
