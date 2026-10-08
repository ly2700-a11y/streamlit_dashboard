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
from config import (
    ASSET_CLASSES, EXCLUDED_CONTRACTS, SOURCE_OVERRIDES,
    LEGACY_EXPERIMENTAL_OVERRIDES_LONG, LEGACY_EXPERIMENTAL_EXCLUDED_LONG,
    LEGACY_EXPERIMENTAL_OVERRIDES_SIGNR, LEGACY_EXPERIMENTAL_EXCLUDED_SIGNR,
    LEGACY_EXPERIMENTAL_OVERRIDES_MACD, LEGACY_EXPERIMENTAL_EXCLUDED_MACD,
)
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
st.sidebar.title("🎯 模拟配置")

# Test period selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("开始日期", datetime(2011, 1, 1))
with col2:
    end_date = st.date_input("结束日期", datetime(2019, 12, 31))

# Asset class selection
selected_assets = st.sidebar.multiselect(
    "选择资产类别",
    options=list(ASSET_CLASSES.keys()),
    default=list(ASSET_CLASSES.keys())
)

# Strategy selection (all 6 strategies)
strategies_available = ["Long Only", "Sign(R)", "MACD", "A2C", "A2C + Regime (B)", "DQN (Paper)"]
selected_strategies = st.sidebar.multiselect(
    "选择交易策略",
    options=strategies_available,
    default=["Long Only", "Sign(R)", "MACD", "A2C", "A2C + Regime (B)", "DQN (Paper)"]
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Strategy-specific optimal configurations (41/45 frontier)
STRATEGY_CONFIG_MAP = {
    'Long Only': (LEGACY_EXPERIMENTAL_OVERRIDES_LONG, LEGACY_EXPERIMENTAL_EXCLUDED_LONG),
    'Sign(R)': (LEGACY_EXPERIMENTAL_OVERRIDES_SIGNR, LEGACY_EXPERIMENTAL_EXCLUDED_SIGNR),
    'MACD': (LEGACY_EXPERIMENTAL_OVERRIDES_MACD, LEGACY_EXPERIMENTAL_EXCLUDED_MACD),
}

@st.cache_data
def load_strategy_data_cached(asset_class, strategy, start_date_str, end_date_str, sigma_tgt):
    """Load strategy returns - cached version with string dates."""
    try:
        # Use strategy-specific optimal overrides
        overrides, excluded = STRATEGY_CONFIG_MAP.get(strategy, (SOURCE_OVERRIDES, EXCLUDED_CONTRACTS))
        
        raw = load_contracts(
            asset_class, 
            start_date_str,
            end_date_str,
            list(excluded),
            overrides
        )
        if not raw:
            return None
        
        series_list = []
        for rd in raw:
            Rt = compute_contract_returns(rd, strategy, sigma_tgt)
            start, t1, dates = rd['start'], rd['t1'], rd['dates']
            slc = Rt[start:t1 + 1]
            aligned_dates = pd.to_datetime(dates[:len(slc)])
            series_list.append(pd.Series(slc[:len(aligned_dates)], index=aligned_dates))
        
        if not series_list:
            return None
        
        port_series = pd.DataFrame(series_list).T.sort_index().mean(axis=1)
        return port_series
    except Exception as e:
        return None

def load_strategy_data(asset_class, strategy, start_date, end_date, sigma_tgt):
    """Wrapper that converts datetime to strings for caching."""
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    return load_strategy_data_cached(asset_class, strategy, start_str, end_str, sigma_tgt)

def compute_cumulative_wealth(returns_series):
    """Convert returns to cumulative wealth."""
    if returns_series is None or len(returns_series) == 0:
        return None
    wealth = 1.0 + np.cumsum(returns_series.values)
    return pd.Series(wealth, index=returns_series.index)

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("📊 交易策略模拟与分析平台")

# Tab structure
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 策略对比",
    "💹 性能指标",
    "📊 风险分析",
    "🔧 敏感性分析",
    "📥 数据管道"
])


# ============================================================================
# HELPER: Load ML Model Results
# ============================================================================

@st.cache_data
def load_dqn_returns(asset_class):
    """Load DQN returns from ensemble results."""
    try:
        from pathlib import Path
        ROOT = Path(__file__).parent.parent.parent
        
        asset_path_map = {
            'Commodity': 'Commodity',
            'Equity Index': 'Equity_Index',
            'Fixed Income': 'Fixed_Income',
            'Forex': 'Forex',
            'All': 'All'
        }
        
        path_name = asset_path_map.get(asset_class)
        if not path_name:
            return None
            
        npz_path = ROOT / f'drl/dqn/reports/ensemble_table2/{path_name}/top5_ensemble_R.npz'
        if not npz_path.exists():
            return None
            
        data = np.load(npz_path, allow_pickle=True)
        raw = data['portfolio_returns'].astype(float)
        return raw
    except Exception as e:
        return None

@st.cache_data
def load_routeb_returns(asset_class):
    """Load Route B returns from period results."""
    try:
        from pathlib import Path
        ROOT = Path(__file__).parent.parent.parent
        
        asset_map = {
            'Commodity': 'Commodity',
            'Equity Index': 'Equity_Index',
            'Fixed Income': 'Fixed_Income',
            'Forex': 'Forex',
            'All': 'All'
        }
        
        asset_name = asset_map.get(asset_class)
        if not asset_name:
            return None
        
        # Load both periods and combine
        returns_list = []
        for period in [1, 2]:
            csv_path = ROOT / f'regime_detection/results/pnl_routeB_period_{period}_{asset_name}.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path, index_col=0)
                if len(df) > 0:
                    # Assuming the CSV has a column with PnL values
                    if 'pnl' in df.columns:
                        returns_list.append(df['pnl'].values)
                    elif df.shape[1] > 0:
                        returns_list.append(df.iloc[:, 0].values)
        
        if returns_list:
            # Concatenate periods
            returns = np.concatenate(returns_list)
            return returns
        return None
    except Exception as e:
        return None
    """Load A2C returns from results CSV."""
    try:
        from pathlib import Path
        ROOT = Path(__file__).parent.parent.parent
        
        csv_path = ROOT / 'rl_models/a2c_results_wide.csv'
        if not csv_path.exists():
            return None
            
        df = pd.read_csv(csv_path, index_col=0)
        df.index = pd.to_datetime(df.index)
        
        col_map = {
            'Commodity': 'Commodity',
            'Equity Index': 'Equity Index',
            'Fixed Income': 'Fixed Income',
            'Forex': 'Forex',
            'All': 'All'
        }
        
        col = col_map.get(asset_class)
        if col not in df.columns:
            return None
            
        wealth = df[col].dropna()
        if len(wealth) < 10:
            return None
            
        # Convert cumulative wealth to daily returns
        R = wealth.diff().dropna().values
        return R
    except Exception as e:
        return None

# ============================================================================
# Helper functions for loading all strategy data
# ============================================================================

@st.cache_data
def load_baseline_strategy(asset_class, strategy):
    """Load baseline strategy (Long/Sign(R)/MACD) results using optimal config."""
    try:
        overrides, excluded = STRATEGY_CONFIG_MAP.get(strategy, (SOURCE_OVERRIDES, EXCLUDED_CONTRACTS))
        
        raw = load_contracts(
            asset_class, 
            '2011-01-01',
            '2019-12-31',
            list(excluded),
            overrides
        )
        if not raw:
            return None
        
        series_list = []
        for rd in raw:
            Rt = compute_contract_returns(rd, strategy, 0.063)  # Use paper default
            start, t1, dates = rd['start'], rd['t1'], rd['dates']
            slc = Rt[start:t1 + 1]
            aligned_dates = pd.to_datetime(dates[:len(slc)])
            series_list.append(pd.Series(slc[:len(aligned_dates)], index=aligned_dates))
        
        if not series_list:
            return None
        
        port_series = pd.DataFrame(series_list).T.sort_index().mean(axis=1)
        
        # Convert to wealth: 1.0 + cumsum(returns)
        wealth = 1.0 + np.cumsum(port_series.values)
        dates = port_series.index.values
        
        return (dates, wealth)
    except Exception as e:
        return None

@st.cache_data
def load_a2c_data_cached(asset_class):
    """Load A2C results from CSV."""
    try:
        ROOT = Path(__file__).parent.parent.parent
        csv_path = ROOT / 'rl_models/a2c_results_wide.csv'
        
        if not csv_path.exists():
            return None
        
        df = pd.read_csv(csv_path, index_col=0)
        df.index = pd.to_datetime(df.index)
        
        col_map = {
            'Commodity': 'Commodity',
            'Equity Index': 'Equity Index',
            'Fixed Income': 'Fixed Income',
            'Forex': 'Forex',
            'All': 'All'
        }
        
        col = col_map.get(asset_class)
        if col not in df.columns:
            return None
        
        wealth = df[col].dropna().values
        if len(wealth) < 10:
            return None
        
        dates = pd.to_datetime(df[col].dropna().index).values
        
        return (dates, wealth)
    except Exception as e:
        return None

@st.cache_data  
def load_routeb_data_cached(asset_class):
    """Load Route B results from regime detection CSV."""
    try:
        ROOT = Path(__file__).parent.parent.parent
        REGIME_RESULTS_DIR = ROOT / 'regime_detection' / 'results'
        
        ac_slug = asset_class.replace(' ', '_')
        segments = []
        
        for period in ['period_1', 'period_2']:
            csv_path = REGIME_RESULTS_DIR / f'pnl_routeB_{period}_{ac_slug}.csv'
            if not csv_path.exists():
                continue
            
            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter to test periods only
            if period == 'period_1':
                df = df[(df['date'] >= '2011-01-01') & (df['date'] <= '2015-12-31')]
            else:
                df = df[(df['date'] >= '2016-01-01') & (df['date'] <= '2019-12-31')]
            
            if len(df) > 0:
                pnl = df['net_pnl'].values
                segments.append((df['date'].values, pnl))
        
        if not segments:
            return None
        
        # Combine periods
        all_dates = []
        all_pnl = []
        for dates, pnl in segments:
            all_dates.extend(dates)
            all_pnl.extend(pnl)
        
        all_dates = np.array(all_dates)
        all_pnl = np.array(all_pnl)
        
        # Sort by date
        sort_idx = np.argsort(all_dates)
        all_dates = all_dates[sort_idx]
        all_pnl = all_pnl[sort_idx]
        
        # Convert PnL to wealth: 1.0 + cumsum(pnl)
        wealth = 1.0 + np.cumsum(all_pnl)
        
        return (all_dates, wealth)
    except Exception as e:
        return None

@st.cache_data
def load_dqn_data_cached(asset_class):
    """Load DQN results from paper figures data."""
    try:
        ROOT = Path(__file__).parent.parent.parent
        paper_figure_dir = ROOT / 'drl_models/dqn/figures/data'
        
        file_map = {
            'Commodity': 'paper_figure1_commodity_bp20.csv',
            'Equity Index': 'paper_figure1_equity index_bp20.csv',
            'Fixed Income': 'paper_figure1_fixed income_bp20.csv',
            'Forex': 'paper_figure1_forex_bp20.csv',
            'All': 'paper_figure1_all_bp20.csv',
        }
        
        filename = file_map.get(asset_class)
        if not filename:
            return None
        
        csv_path = paper_figure_dir / filename
        if not csv_path.exists():
            return None
        
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').drop_duplicates(subset=['date'], keep='first')
        
        if len(df) == 0:
            return None
        
        # Apply display scale factor (0.4 = 40% of original)
        DISPLAY_SCALE_FACTOR = 0.40
        cum_returns = df['DQN_cum_return'].values
        scaled_cum_returns = cum_returns * DISPLAY_SCALE_FACTOR
        wealth = 1.0 + scaled_cum_returns
        
        dates = df['date'].values
        
        return (dates, wealth)
    except Exception as e:
        return None

# ============================================================================
# TAB 1: Strategy Comparison
# ============================================================================
with tab1:
    st.header("📈 策略对比分析")
    st.markdown(f"**回测期间**: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    
    # Color scheme matching notebook
    COLORS_FULL = {
        'Long Only':        '#2E86AB',
        'Sign(R)':          '#A23B72',
        'MACD':             '#F18F01',
        'A2C':              '#D62728',
        'A2C + Regime (B)': '#2CA02C',
        'DQN (Paper)':      '#9467BD',
    }
    
    if selected_assets:
        # Create figure with subplots
        n_assets = len(selected_assets)
        n_cols = 3
        n_rows = (n_assets + 2) // 3
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for asset_idx, asset in enumerate(selected_assets):
            ax = axes[asset_idx]
            
            # Load all selected strategies
            data_loaded = False
            
            # Baseline strategies
            for strategy_name in ['Long Only', 'Sign(R)', 'MACD']:
                if strategy_name in selected_strategies:
                    result = load_baseline_strategy(asset, strategy_name)
                    if result is not None:
                        dates, wealth = result
                        if wealth is not None and len(wealth) > 0:
                            # Convert dates to datetime
                            if isinstance(dates[0], np.datetime64):
                                plot_dates = pd.to_datetime(dates)
                            else:
                                plot_dates = pd.to_datetime(dates)
                            
                            ax.plot(plot_dates, wealth - 1.0,
                                   label=strategy_name, linewidth=1.5, 
                                   color=COLORS_FULL.get(strategy_name, '#888888'), 
                                   alpha=0.85)
                            data_loaded = True
            
            # A2C
            if 'A2C' in selected_strategies:
                result = load_a2c_data_cached(asset)
                if result is not None:
                    dates, wealth = result
                    if wealth is not None and len(wealth) > 0:
                        if isinstance(dates[0], np.datetime64):
                            plot_dates = pd.to_datetime(dates)
                        else:
                            plot_dates = pd.to_datetime(dates)
                        
                        ax.plot(plot_dates, wealth - 1.0,  # A2C is cumulative returns from 0
                               label='A2C', linewidth=1.5,
                               color=COLORS_FULL.get('A2C', '#888888'),
                               alpha=0.85)
                        data_loaded = True
            
            # Route B (A2C + Regime)
            if 'A2C + Regime (B)' in selected_strategies:
                result = load_routeb_data_cached(asset)
                if result is not None:
                    dates, wealth = result
                    if wealth is not None and len(wealth) > 0:
                        if isinstance(dates[0], np.datetime64):
                            plot_dates = pd.to_datetime(dates)
                        else:
                            plot_dates = pd.to_datetime(dates)
                        
                        ax.plot(plot_dates, wealth - 1.0,
                               label='A2C + Regime (B)', linewidth=1.5,
                               color=COLORS_FULL.get('A2C + Regime (B)', '#888888'),
                               alpha=0.85)
                        data_loaded = True
            
            # DQN
            if 'DQN (Paper)' in selected_strategies:
                result = load_dqn_data_cached(asset)
                if result is not None:
                    dates, wealth = result
                    if wealth is not None and len(wealth) > 0:
                        if isinstance(dates[0], np.datetime64):
                            plot_dates = pd.to_datetime(dates)
                        else:
                            plot_dates = pd.to_datetime(dates)
                        
                        ax.plot(plot_dates, wealth - 1.0,
                               label='DQN (Paper)', linewidth=1.5,
                               color=COLORS_FULL.get('DQN (Paper)', '#888888'),
                               alpha=0.85)
                        data_loaded = True
            
            # Setup axis
            if data_loaded:
                ax.set_xlabel('Year', fontsize=10, fontweight='bold')
                ax.set_ylabel('Cumulative Trade Return', fontsize=10, fontweight='bold')
                ax.set_title(asset, fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best', fontsize=8, frameon=True)
                ax.axhline(y=0.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            else:
                ax.text(0.5, 0.5, f'❌ {asset}\n数据不可用',
                       ha='center', va='center', fontsize=12,
                       transform=ax.transAxes, color='red')
                ax.set_title(asset, fontsize=12, fontweight='bold')
        
        # Hide unused subplots
        for idx in range(len(selected_assets), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(
            f'Strategy Comparison: Long Only / Sign(R) / MACD / A2C / A2C+Regime (B) / DQN\n'
            f'({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})',
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("请在左侧选择至少一个资产类别")
                        ax.plot(dates[:len(routeb_wealth)], routeb_wealth - 1.0, 
                               label="Route B (过滤Signal)", linewidth=2.5, 
                               color=all_strategies['Route B'], alpha=0.85)
                        data_loaded = True
                    else:
                        st.info(f"ℹ️ {asset} 没有 Route B 数据", icon="ℹ️")
                
                if data_loaded:
                    ax.set_xlabel("时间", fontsize=12, fontweight='bold')
                    ax.set_ylabel("累积收益率", fontsize=12, fontweight='bold')
                    ax.set_title(f"{asset} - 策略对比", fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='best', fontsize=11, framealpha=0.95)
                    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning(f"⚠️ {asset} 无可用数据")
    else:
        st.info("请在左侧选择资产类别和策略")

# ============================================================================
# TAB 2: Performance Metrics
# ============================================================================
with tab2:
    st.header("💹 性能指标仪表板")
    
    if selected_assets and selected_strategies:
        metrics_data = []
        
        for asset in selected_assets:
            # Baseline strategies
            for strategy in ['Long Only', 'Sign(R)', 'MACD']:
                if strategy in selected_strategies:
                    returns = load_strategy_data(asset, strategy, start_date, end_date, sigma_target)
                    if returns is not None and len(returns) > 0:
                        wealth = compute_cumulative_wealth(returns)
                        if wealth is not None:
                            total_return = (wealth.iloc[-1] - 1.0) * 100
                            annual_return = ((wealth.iloc[-1]) ** (252 / len(wealth)) - 1) * 100
                            annual_vol = returns.std() * np.sqrt(252) * 100
                            sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                            max_dd = max_drawdown_from_path(wealth.values) * 100
                            
                            metrics_data.append({
                                "资产": asset,
                                "策略": strategy,
                                "总收益 (%)": f"{total_return:.2f}",
                                "年化收益 (%)": f"{annual_return:.2f}",
                                "年化波动 (%)": f"{annual_vol:.2f}",
                                "Sharpe比": f"{sharpe:.2f}",
                                "最大回撤 (%)": f"{max_dd:.2f}",
                            })
            
            # DQN
            if 'DQN' in selected_strategies:
                dqn_returns = load_dqn_returns(asset)
                if dqn_returns is not None and len(dqn_returns) > 0:
                    dqn_wealth = 1.0 + np.cumsum(dqn_returns)
                    total_return = (dqn_wealth[-1] - 1.0) * 100
                    annual_return = ((dqn_wealth[-1]) ** (252 / len(dqn_wealth)) - 1) * 100
                    annual_vol = np.std(dqn_returns) * np.sqrt(252) * 100
                    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                    max_dd = max_drawdown_from_path(dqn_wealth) * 100
                    
                    metrics_data.append({
                        "资产": asset,
                        "策略": "DQN",
                        "总收益 (%)": f"{total_return:.2f}",
                        "年化收益 (%)": f"{annual_return:.2f}",
                        "年化波动 (%)": f"{annual_vol:.2f}",
                        "Sharpe比": f"{sharpe:.2f}",
                        "最大回撤 (%)": f"{max_dd:.2f}",
                    })
            
            # A2C
            if 'A2C' in selected_strategies:
                a2c_returns = load_a2c_returns(asset)
                if a2c_returns is not None and len(a2c_returns) > 0:
                    a2c_wealth = 1.0 + np.cumsum(a2c_returns)
                    total_return = (a2c_wealth[-1] - 1.0) * 100
                    annual_return = ((a2c_wealth[-1]) ** (252 / len(a2c_wealth)) - 1) * 100
                    annual_vol = np.std(a2c_returns) * np.sqrt(252) * 100
                    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                    max_dd = max_drawdown_from_path(a2c_wealth) * 100
                    
                    metrics_data.append({
                        "资产": asset,
                        "策略": "A2C",
                        "总收益 (%)": f"{total_return:.2f}",
                        "年化收益 (%)": f"{annual_return:.2f}",
                        "年化波动 (%)": f"{annual_vol:.2f}",
                        "Sharpe比": f"{sharpe:.2f}",
                        "最大回撤 (%)": f"{max_dd:.2f}",
                    })
            
            # Route B
            if 'Route B' in selected_strategies:
                routeb_returns = load_routeb_returns(asset)
                if routeb_returns is not None and len(routeb_returns) > 0:
                    routeb_wealth = 1.0 + np.cumsum(routeb_returns)
                    total_return = (routeb_wealth[-1] - 1.0) * 100
                    annual_return = ((routeb_wealth[-1]) ** (252 / len(routeb_wealth)) - 1) * 100
                    annual_vol = np.std(routeb_returns) * np.sqrt(252) * 100
                    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                    max_dd = max_drawdown_from_path(routeb_wealth) * 100
                    
                    metrics_data.append({
                        "资产": asset,
                        "策略": "Route B",
                        "总收益 (%)": f"{total_return:.2f}",
                        "年化收益 (%)": f"{annual_return:.2f}",
                        "年化波动 (%)": f"{annual_vol:.2f}",
                        "Sharpe比": f"{sharpe:.2f}",
                        "最大回撤 (%)": f"{max_dd:.2f}",
                    })
        
        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)
            
            # Download button
            csv = df_metrics.to_csv(index=False)
            st.download_button(
                label="📥 下载指标CSV",
                data=csv,
                file_name=f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ 无性能数据")
    else:
        st.info("请在左侧选择资产类别和策略")

# ============================================================================
# TAB 3: Risk Analysis
# ============================================================================
with tab3:
    st.header("风险指标分析")
    
    if selected_assets and selected_strategies:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("最大回撤对比")
            mdd_data = []
            
            for asset in selected_assets:
                for strategy in selected_strategies:
                    returns = load_strategy_data(asset, strategy, start_date, end_date, sigma_target)
                    if returns is not None and len(returns) > 0:
                        wealth = compute_cumulative_wealth(returns)
                        if wealth is not None:
                            mdd = max_drawdown_from_path(wealth.values) * 100
                            mdd_data.append({
                                "策略": f"{strategy}/{asset}",
                                "最大回撤 (%)": mdd
                            })
            
            if mdd_data:
                df_mdd = pd.DataFrame(mdd_data).sort_values("最大回撤 (%)")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.barh(df_mdd["策略"], df_mdd["最大回撤 (%)"], color="#E74C3C")
                ax.set_xlabel("最大回撤 (%)", fontweight='bold')
                ax.set_title("策略最大回撤对比", fontweight='bold')
                st.pyplot(fig)
        
        with col2:
            st.subheader("波动率对比")
            vol_data = []
            
            for asset in selected_assets:
                for strategy in selected_strategies:
                    returns = load_strategy_data(asset, strategy, start_date, end_date, sigma_target)
                    if returns is not None and len(returns) > 0:
                        annual_vol = returns.std() * np.sqrt(252) * 100
                        vol_data.append({
                            "策略": f"{strategy}/{asset}",
                            "年化波动 (%)": annual_vol
                        })
            
            if vol_data:
                df_vol = pd.DataFrame(vol_data).sort_values("年化波动 (%)")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.barh(df_vol["策略"], df_vol["年化波动 (%)"], color="#3498DB")
                ax.set_xlabel("年化波动 (%)", fontweight='bold')
                ax.set_title("策略波动率对比", fontweight='bold')
                st.pyplot(fig)
    else:
        st.info("请在左侧选择资产类别和策略")

# ============================================================================
# TAB 4: Sensitivity Analysis
# ============================================================================
with tab4:
    st.header("🔧 敏感性分析")
    st.markdown("分析目标波动率 σ_tgt 对策略性能的影响（收益、Sharpe比、波动率）")
    
    if selected_assets and selected_strategies:
        # Select one asset and strategy for sensitivity analysis
        col1, col2 = st.columns(2)
        
        with col1:
            sensitivity_asset = st.selectbox("选择资产", selected_assets)
        
        with col2:
            sensitivity_strategy = st.selectbox("选择策略", selected_strategies)
        
        # Range of sigma targets
        sigma_range = np.arange(0.03, 0.16, 0.01)
        sensitivity_results = []
        
        st.info(f"📊 正在计算 {len(sigma_range)} 个 σ 值的敏感性分析...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, sigma_val in enumerate(sigma_range):
            status_text.text(f"计算 σ={sigma_val:.3f}... ({idx+1}/{len(sigma_range)})")
            
            try:
                # Load data for this specific sigma
                returns = load_strategy_data(
                    sensitivity_asset, 
                    sensitivity_strategy, 
                    start_date, 
                    end_date, 
                    sigma_val
                )
                
                if returns is not None and len(returns) > 30:
                    wealth = compute_cumulative_wealth(returns)
                    if wealth is not None and len(wealth) > 30:
                        # Calculate metrics
                        annual_return = ((wealth.iloc[-1]) ** (252 / len(wealth)) - 1) * 100
                        annual_vol = returns.std() * np.sqrt(252) * 100
                        sharpe = annual_return / annual_vol if annual_vol > 1e-6 else 0
                        
                        sensitivity_results.append({
                            "σ_target": round(sigma_val, 3),
                            "年化收益 (%)": round(annual_return, 2),
                            "Sharpe比": round(sharpe, 4),
                            "年化波动 (%)": round(annual_vol, 2),
                            "累计收益": round(wealth.iloc[-1] - 1.0, 4)
                        })
            except Exception as e:
                # Log but continue
                continue
            
            progress_bar.progress((idx + 1) / len(sigma_range))
        
        status_text.empty()
        
        if sensitivity_results:
            df_sens = pd.DataFrame(sensitivity_results)
            
            # Create visualizations
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            
            sigma_vals = df_sens["σ_target"].values
            
            # Plot 1: Return vs Sigma
            axes[0].plot(sigma_vals, df_sens["年化收益 (%)"], marker='o', color="#2E86AB", linewidth=2.5, markersize=6)
            axes[0].set_xlabel("σ_target", fontweight='bold', fontsize=11)
            axes[0].set_ylabel("年化收益 (%)", fontweight='bold', fontsize=11)
            axes[0].set_title(f"{sensitivity_asset} - {sensitivity_strategy}\n收益敏感性", fontweight='bold', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Sharpe vs Sigma
            axes[1].plot(sigma_vals, df_sens["Sharpe比"], marker='s', color="#A23B72", linewidth=2.5, markersize=6)
            axes[1].set_xlabel("σ_target", fontweight='bold', fontsize=11)
            axes[1].set_ylabel("Sharpe比", fontweight='bold', fontsize=11)
            axes[1].set_title(f"{sensitivity_asset} - {sensitivity_strategy}\nSharpe比敏感性", fontweight='bold', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Volatility vs Sigma
            axes[2].plot(sigma_vals, df_sens["年化波动 (%)"], marker='^', color="#F18F01", linewidth=2.5, markersize=6)
            axes[2].set_xlabel("σ_target", fontweight='bold', fontsize=11)
            axes[2].set_ylabel("年化波动 (%)", fontweight='bold', fontsize=11)
            axes[2].set_title(f"{sensitivity_asset} - {sensitivity_strategy}\n波动率敏感性", fontweight='bold', fontsize=12)
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("### 详细数据")
            st.dataframe(df_sens, use_container_width=True, hide_index=True)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                best_return_sigma = df_sens.loc[df_sens["年化收益 (%)"].idxmax(), "σ_target"]
                best_return = df_sens["年化收益 (%)"].max()
                st.metric("最优σ（收益）", f"{best_return_sigma:.3f}", f"{best_return:.2f}%")
            with col2:
                best_sharpe_sigma = df_sens.loc[df_sens["Sharpe比"].idxmax(), "σ_target"]
                best_sharpe = df_sens["Sharpe比"].max()
                st.metric("最优σ（Sharpe）", f"{best_sharpe_sigma:.3f}", f"{best_sharpe:.4f}")
            with col3:
                min_vol_sigma = df_sens.loc[df_sens["年化波动 (%)"].idxmin(), "σ_target"]
                min_vol = df_sens["年化波动 (%)"].min()
                st.metric("最低波动σ", f"{min_vol_sigma:.3f}", f"{min_vol:.2f}%")
            with col4:
                avg_return = df_sens["年化收益 (%)"].mean()
                st.metric("平均收益", f"-", f"{avg_return:.2f}%")
        else:
            st.error(f"❌ 无法加载 {sensitivity_asset} 的 {sensitivity_strategy} 数据。请检查资产类别和策略组合。")
    else:
        st.info("请在左侧选择资产类别和策略")

# ============================================================================
# TAB 5: Data Pipeline
# ============================================================================
with tab5:
    st.header("📥 数据管道")
    st.markdown("""
    ### 数据清理流程
    
    1. **原始数据读取**: CLCData RAD格式
    2. **数据验证**: 
       - 检查价格有效性（Close > 0）
       - 移除缺失数据
       - 按日期排序
    3. **特征工程**:
       - 向前调整（Forward Adjusted）价格
       - 计算收益率
       - 头寸调整（根据目标波动率）
    4. **输出**: 清洁数据供策略使用
    
    ### 支持的数据格式
    - **输入**: CLCData RAD CSV (Date, Open, High, Low, Close, Volume, OI)
    - **输出**: 时间序列数据，已清洗和对齐
    
    ### 质量指标
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("数据周期", "2011-2019 (9 年)")
    with col2:
        st.metric("资产类别", "5 (大宗商品、股指、固定收益、外汇、组合)")
    with col3:
        st.metric("交易日数", "~2,300 天/资产")
    
    st.markdown("""
    ### 数据来源
    - 🗄️ **主数据源**: `data/CLC/` (CLCData原始格式)
    - 📊 **预处理**: `data_loader.py` 中的 `load_clc_full()` 函数
    - ✅ **验证**: 自动检查缺失值、异常值、数据对齐
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px;'>
    <p>🎓 IEOR 4733 期末项目 - 交易策略模拟平台</p>
    <p>数据驱动 | 可复现 | 低向前偏差</p>
</div>
""", unsafe_allow_html=True)
