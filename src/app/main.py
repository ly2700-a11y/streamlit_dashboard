"""
Trading Strategy Simulation & Analysis Platform
Main Streamlit application for interactive backtesting and risk analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import sys
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import core modules
from config import ASSET_CLASSES, EXCLUDED_CONTRACTS, SOURCE_OVERRIDES
from data_loader import load_clc_full
from metrics import compute_metrics, max_drawdown_from_path, cagr_from_path
from baseline_run import load_contracts, compute_contract_returns

# Page configuration
st.set_page_config(
    page_title="Trading Strategy Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
st.sidebar.title("🎯 Simulation Settings")

# Test period selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", datetime(2011, 1, 1))
with col2:
    end_date = st.date_input("End Date", datetime(2019, 12, 31))

# Strategy selection (controls TAB 1 display + TABs 2-4 baseline)
ALL_STRATEGIES_SIDEBAR = ["Long Only", "Sign(R)", "MACD", "A2C", "A2C + Regime (B)", "DQN (Paper)"]
selected_strategies = st.sidebar.multiselect(
    "Select Strategies",
    options=ALL_STRATEGIES_SIDEBAR,
    default=ALL_STRATEGIES_SIDEBAR
)

sigma_target = 0.063  # fixed default, no longer exposed in UI

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# ── TAB 1: 6-strategy CSV-based data loading (mirrors strategies_comparison.ipynb) ──

A2C_CSV_DIR   = PROJECT_ROOT / 'reproduction_of_figures' / 'a2c_csv_data'
BASELINE_DIR  = PROJECT_ROOT / 'reproduction_of_figures' / 'baseline_results'   # Cell 15 export (pure RAD, no overrides)
REGIME_DIR    = PROJECT_ROOT / 'regime_detection' / 'results'
DQN_DIR       = PROJECT_ROOT / 'drl_models' / 'dqn' / 'figures' / 'data'
A2C_WIDE_CSV  = PROJECT_ROOT / 'rl_models' / 'a2c_results_wide.csv'

# Filename slug for a2c_csv_data files (A2C uses FX for Forex)
_CSV_ASSET_SLUG = {
    'Commodity':    'Commodity',
    'Equity Index': 'Equity_Index',
    'Fixed Income': 'Fixed_Income',
    'Forex':        'FX',       # all pnl_*.csv files use FX, not Forex
    'All':          'All',
}
# Baseline CSV slug: baseline_{strategy}_{asset}.csv (lowercase, spaces→_)
_BASELINE_STRATEGY_SLUG = {'Long Only': 'long_only', 'Sign(R)': 'sign(r)', 'MACD': 'macd'}
_BASELINE_ASSET_SLUG = {
    'Commodity':    'commodity',
    'Equity Index': 'equity_index',
    'Fixed Income': 'fixed_income',
    'Forex':        'forex',
    'All':          'all',
}
# Route B slug: files use Forex (not FX), spaces replaced with _
_RB_ASSET_SLUG = {
    'Commodity':    'Commodity',
    'Equity Index': 'Equity_Index',
    'Fixed Income': 'Fixed_Income',
    'Forex':        'Forex',
    'All':          'All',
}

DQN_FILENAMES = {
    'Commodity':    'paper_figure1_commodity_bp20.csv',
    'Equity Index': 'paper_figure1_equity index_bp20.csv',
    'Fixed Income': 'paper_figure1_fixed income_bp20.csv',
    'Forex':        'paper_figure1_forex_bp20.csv',
    'All':          'paper_figure1_all_bp20.csv',
}
DQN_SCALE = 0.40          # align DQN magnitude to other strategies
RB_PERIOD_RANGES = {
    'period_1': ('2011-01-01', '2015-12-31'),
    'period_2': ('2016-01-01', '2019-12-31'),
}

@st.cache_data
def load_all_strategy_curves():
    """
    Load cumulative-return curves for all 6 strategies × 5 asset classes.
    Returns dict: strategy -> asset -> (dates_array, cum_return_array_from_0)
    Mirrors the logic of strategies_comparison.ipynb Cell 27.
    """
    results = {s: {} for s in
               ['Long Only', 'Sign(R)', 'MACD', 'A2C', 'A2C + Regime (B)', 'DQN (Paper)']}
    assets_all = ['Commodity', 'Equity Index', 'Fixed Income', 'Forex', 'All']

    # ── Long Only / Sign(R) / MACD: from pure-RAD baseline CSVs (Cell 15 export) ──
    # baseline_{strategy_slug}_{asset_slug}.csv  columns: date, wealth, cumulative_return
    for strat, strat_slug in _BASELINE_STRATEGY_SLUG.items():
        for asset in assets_all:
            a_slug = _BASELINE_ASSET_SLUG[asset]
            csv_path = BASELINE_DIR / f'baseline_{strat_slug}_{a_slug}.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path, parse_dates=['date'])
                df = df.sort_values('date')
                dates   = df['date'].values
                cum_ret = df['cumulative_return'].values   # already wealth - 1.0
                results[strat][asset] = (dates, cum_ret)
            else:
                results[strat][asset] = (None, None)

    # ── A2C: from rl_models/a2c_results_wide.csv (already cumulative from 0) ──
    if A2C_WIDE_CSV.exists():
        df_wide = pd.read_csv(A2C_WIDE_CSV, index_col=0, parse_dates=True)
        col_map = {
            'Commodity':    'Commodity',
            'Equity Index': 'Equity Index',
            'Fixed Income': 'Fixed Income',
            'Forex':        'Forex',
            'All':          'All',
        }
        for asset, col in col_map.items():
            if col in df_wide.columns:
                series = df_wide[col].dropna()
                results['A2C'][asset] = (series.index.to_numpy(), series.values)
            else:
                results['A2C'][asset] = (None, None)
    else:
        for asset in assets_all:
            results['A2C'][asset] = (None, None)

    # ── A2C + Regime (Route B): pnl_routeB_period_{n}_{slug}.csv, net_pnl cumsum ──
    for asset in assets_all:
        a_slug = _RB_ASSET_SLUG[asset]    # Forex stays Forex in regime_detection/results
        segments = []
        for period, (t_start, t_end) in RB_PERIOD_RANGES.items():
            csv_path = REGIME_DIR / f'pnl_routeB_{period}_{a_slug}.csv'
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path, parse_dates=['date'])
            df = df[(df['date'] >= t_start) & (df['date'] <= t_end)]
            if len(df) == 0:
                continue
            segments.append(pd.Series(df['net_pnl'].values, index=df['date']))
        if segments:
            combined = pd.concat(segments).sort_index()
            combined = combined[~combined.index.duplicated(keep='first')]
            dates  = combined.index.to_numpy()
            cum_ret = np.cumsum(combined.values)         # cumsum of net_pnl = cum return from 0
            results['A2C + Regime (B)'][asset] = (dates, cum_ret)
        else:
            results['A2C + Regime (B)'][asset] = (None, None)

    # ── DQN (Paper): paper_figure1_*.csv, DQN_cum_return × scale ──
    for asset in assets_all:
        fname = DQN_FILENAMES.get(asset)
        if fname is None:
            results['DQN (Paper)'][asset] = (None, None)
            continue
        csv_path = DQN_DIR / fname
        if not csv_path.exists():
            results['DQN (Paper)'][asset] = (None, None)
            continue
        df = pd.read_csv(csv_path, parse_dates=['date'])
        df = df.sort_values('date').drop_duplicates(subset=['date'], keep='first')
        dates   = df['date'].values
        cum_ret = df['DQN_cum_return'].values * DQN_SCALE
        results['DQN (Paper)'][asset] = (dates, cum_ret)

    return results


@st.cache_data
def load_strategy_data(asset_class, strategy, start_date, end_date, sigma_tgt):
    """Load strategy returns for TABs 2-4 (baseline only, computed on-the-fly)."""
    try:
        raw = load_contracts(
            asset_class,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            EXCLUDED_CONTRACTS,
            SOURCE_OVERRIDES
        )
        if not raw:
            return None
        series_list = []
        for rd in raw:
            Rt = compute_contract_returns(rd, strategy, sigma_tgt)
            t0, t1, dates = rd['start'], rd['t1'], rd['dates']
            slc = Rt[t0:t1 + 1]
            aligned_dates = pd.to_datetime(dates[:len(slc)])
            series_list.append(pd.Series(slc[:len(aligned_dates)], index=aligned_dates))
        if not series_list:
            return None
        return pd.DataFrame(series_list).T.sort_index().mean(axis=1)
    except Exception as e:
        return None

def compute_cumulative_wealth(returns_series):
    """Convert returns to cumulative wealth."""
    if returns_series is None or len(returns_series) == 0:
        return None
    wealth = 1.0 + np.cumsum(returns_series.values)
    return pd.Series(wealth, index=returns_series.index)

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("📊 Trading Strategy Simulation & Analysis Platform")

# Tab structure
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Strategy Comparison",
    "💹 Performance Metrics",
    "📊 Risk Analysis",
    "📥 Data Pipeline"
])

# ============================================================================
# Shared data: loaded once, used by all TABs
# ============================================================================
COLORS_FULL = {
    'Long Only':        '#2E86AB',
    'Sign(R)':          '#A23B72',
    'MACD':             '#F18F01',
    'A2C':              '#D62728',
    'A2C + Regime (B)': '#2CA02C',
    'DQN (Paper)':      '#9467BD',
}
ALL_STRATEGIES = list(COLORS_FULL.keys())
ASSET_ORDER = ['Commodity', 'Equity Index', 'Fixed Income', 'Forex', 'All']

with st.spinner('Loading data...'):
    all_curves = load_all_strategy_curves()

# ============================================================================
# TAB 1: Strategy Comparison (6 strategies, CSV-based, mirrors notebook Cell 27)
# ============================================================================
with tab1:
    st.header("Strategy Comparison")
    st.markdown(f"**Backtest period**: 2011-01-01 to 2019-12-31")

    # ── display as tabs per asset class ──
    display_assets = ASSET_ORDER
    # always include 'All' option in the tab list
    tab_labels = display_assets + (['All'] if 'All' not in display_assets else [])
    asset_tabs = st.tabs(tab_labels)

    import matplotlib.dates as mdates

    # strategies to show = intersection of selected_strategies and the 6 available
    show_strategies = [s for s in ALL_STRATEGIES if s in selected_strategies] if selected_strategies else ALL_STRATEGIES

    for asset_idx, asset in enumerate(tab_labels):
        with asset_tabs[asset_idx]:
            fig, ax = plt.subplots(figsize=(20, 8))
            any_plotted = False

            for strategy in show_strategies:
                dates_e, cum_ret = all_curves[strategy].get(asset, (None, None))
                if cum_ret is None or len(cum_ret) == 0:
                    continue
                try:
                    x = pd.to_datetime(dates_e)
                except Exception:
                    x = np.arange(len(cum_ret))

                # Filter by sidebar date range
                if hasattr(x, 'values'):
                    mask = (x >= pd.Timestamp(start_date)) & (x <= pd.Timestamp(end_date))
                    x = x[mask]
                    cum_ret = cum_ret[mask]
                if len(cum_ret) == 0:
                    continue

                ax.plot(x, cum_ret,
                        label=strategy,
                        linewidth=1.5,
                        color=COLORS_FULL[strategy],
                        alpha=0.85)
                any_plotted = True

            if any_plotted:
                ax.set_xlabel('Year', fontsize=10, fontweight='bold')
                ax.set_ylabel('Cumulative Trade Return', fontsize=10, fontweight='bold')
                ax.set_title(asset, fontsize=12, fontweight='bold')
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best', fontsize=9, frameon=True)
                ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning(f"⚠️ No data available for {asset} (CSV files may not have been generated)")
            plt.close(fig)

# ============================================================================
# TAB 2: Performance Metrics
# ============================================================================
with tab2:
    st.header("Performance Metrics Dashboard")

    show_strategies_tab2 = [s for s in ALL_STRATEGIES if s in selected_strategies] if selected_strategies else ALL_STRATEGIES
    assets_tab2 = [a for a in ASSET_ORDER if a != 'All']  # exclude 'All' for per-asset metrics

    if show_strategies_tab2:
        metrics_data = []

        for asset in assets_tab2:
            for strategy in show_strategies_tab2:
                dates_e, cum_ret = all_curves[strategy].get(asset, (None, None))
                if cum_ret is None or len(cum_ret) == 0:
                    continue
                # Apply date filter
                try:
                    x = pd.to_datetime(dates_e)
                    mask = (x >= pd.Timestamp(start_date)) & (x <= pd.Timestamp(end_date))
                    cum_ret = cum_ret[mask]
                    x = x[mask]
                except Exception:
                    pass
                if len(cum_ret) < 2:
                    continue
                # cum_ret is cumulative return from 0; daily returns = diff
                daily_ret = np.diff(cum_ret)
                total_return = cum_ret[-1] * 100
                # wealth path starting at 1.0
                wealth_vals = 1.0 + cum_ret
                annual_return = ((wealth_vals[-1]) ** (252 / len(wealth_vals)) - 1) * 100 if wealth_vals[-1] > 0 else float('nan')
                annual_vol = np.std(daily_ret) * np.sqrt(252) * 100
                sharpe = annual_return / annual_vol if annual_vol > 0 and not np.isnan(annual_return) else float('nan')
                max_dd = max_drawdown_from_path(wealth_vals) * 100

                metrics_data.append({
                    "Asset Class": asset,
                    "Strategy": strategy,
                    "Total Return (%)": f"{total_return:.2f}",
                    "Ann. Return (%)": f"{annual_return:.2f}" if not np.isnan(annual_return) else "N/A",
                    "Ann. Volatility (%)": f"{annual_vol:.2f}",
                    "Sharpe Ratio": f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A",
                    "Max Drawdown (%)": f"{max_dd:.2f}",
                })

        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, use_container_width=True)
            csv = df_metrics.to_csv(index=False)
            st.download_button(
                label="📥 Download Metrics CSV",
                data=csv,
                file_name=f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ No performance data available")
    else:
        st.info("Please select at least one strategy from the sidebar")

# ============================================================================
# TAB 3: Risk Analysis
# ============================================================================
with tab3:
    st.header("Risk Analysis")

    show_strategies_tab3 = [s for s in ALL_STRATEGIES if s in selected_strategies] if selected_strategies else ALL_STRATEGIES
    assets_tab3 = [a for a in ASSET_ORDER if a != 'All']

    if show_strategies_tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Max Drawdown Comparison")
            mdd_data = []

            for asset in assets_tab3:
                for strategy in show_strategies_tab3:
                    dates_e, cum_ret = all_curves[strategy].get(asset, (None, None))
                    if cum_ret is None or len(cum_ret) == 0:
                        continue
                    try:
                        x = pd.to_datetime(dates_e)
                        mask = (x >= pd.Timestamp(start_date)) & (x <= pd.Timestamp(end_date))
                        cum_ret = cum_ret[mask]
                    except Exception:
                        pass
                    if len(cum_ret) < 2:
                        continue
                    wealth_vals = 1.0 + cum_ret
                    mdd = max_drawdown_from_path(wealth_vals) * 100
                    mdd_data.append({"Strategy": f"{strategy}/{asset}", "Max Drawdown (%)": mdd})

            if mdd_data:
                df_mdd = pd.DataFrame(mdd_data).sort_values("Max Drawdown (%)")
                fig, ax = plt.subplots(figsize=(8, max(4, len(df_mdd) * 0.3)))
                colors_mdd = [COLORS_FULL.get(row["Strategy"].split("/")[0], "#E74C3C") for _, row in df_mdd.iterrows()]
                ax.barh(df_mdd["Strategy"], df_mdd["Max Drawdown (%)"], color=colors_mdd)
                ax.set_xlabel("Max Drawdown (%)", fontweight='bold')
                ax.set_title("Max Drawdown by Strategy", fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        with col2:
            st.subheader("Volatility Comparison")
            vol_data = []

            for asset in assets_tab3:
                for strategy in show_strategies_tab3:
                    dates_e, cum_ret = all_curves[strategy].get(asset, (None, None))
                    if cum_ret is None or len(cum_ret) == 0:
                        continue
                    try:
                        x = pd.to_datetime(dates_e)
                        mask = (x >= pd.Timestamp(start_date)) & (x <= pd.Timestamp(end_date))
                        cum_ret = cum_ret[mask]
                    except Exception:
                        pass
                    if len(cum_ret) < 2:
                        continue
                    daily_ret = np.diff(cum_ret)
                    annual_vol = np.std(daily_ret) * np.sqrt(252) * 100
                    vol_data.append({"Strategy": f"{strategy}/{asset}", "Ann. Volatility (%)": annual_vol})

            if vol_data:
                df_vol = pd.DataFrame(vol_data).sort_values("Ann. Volatility (%)")
                fig, ax = plt.subplots(figsize=(8, max(4, len(df_vol) * 0.3)))
                colors_vol = [COLORS_FULL.get(row["Strategy"].split("/")[0], "#3498DB") for _, row in df_vol.iterrows()]
                ax.barh(df_vol["Strategy"], df_vol["Ann. Volatility (%)"], color=colors_vol)
                ax.set_xlabel("Ann. Volatility (%)", fontweight='bold')
                ax.set_title("Annualised Volatility by Strategy", fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
    else:
        st.info("Please select at least one strategy from the sidebar")

# ============================================================================
# TAB 4: Data Pipeline
# ============================================================================
with tab4:
    st.header("📥 Data Pipeline")
    st.markdown("""
    ### Data Cleaning Workflow

    1. **Raw data ingestion**: CLCData RAD format
    2. **Data validation**:
       - Price validity check (Close > 0)
       - Remove missing data
       - Sort by date
    3. **Feature engineering**:
       - Forward-adjusted prices
       - Return calculation
       - Position sizing (volatility targeting)
    4. **Output**: Clean, aligned time-series data for strategy consumption

    ### Supported Data Formats
    - **Input**: CLCData RAD CSV (Date, Open, High, Low, Close, Volume, OI)
    - **Output**: Cleaned and aligned time-series data

    ### Quality Metrics
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Data Period", "2011–2019 (9 years)")
    with col2:
        st.metric("Asset Classes", "5 classes")
    with col3:
        st.metric("Trading Days", "~2,300 days / asset")

    st.markdown("""
    ### Data Sources
    - 🗄️ **Primary source**: `data/CLC/` (CLCData raw format)
    - 📊 **Pre-processing**: `load_clc_full()` in `data_loader.py`
    - ✅ **Validation**: Automatic checks for missing values, outliers, and alignment
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px;'>
    <p>🎓 IEOR 4733 Final Project — Trading Strategy Simulation Platform</p>
    <p>Data-driven | Reproducible | Low look-ahead bias</p>
</div>
""", unsafe_allow_html=True)
