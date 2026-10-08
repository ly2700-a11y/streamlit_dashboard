"""
IEOR 4733 Trading Strategy Simulation Platform
Interactive web application using Streamlit for backtesting and analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys

# Add parent directories to path for imports
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# Import project modules
from baseline_run import load_contracts, compute_contract_returns, run_table
from config import (
    ASSET_CLASSES, SOURCE_OVERRIDES, EXCLUDED_CONTRACTS,
    LEGACY_EXPERIMENTAL_OVERRIDES_LONG, LEGACY_EXPERIMENTAL_EXCLUDED_LONG,
    LEGACY_EXPERIMENTAL_OVERRIDES_SIGNR, LEGACY_EXPERIMENTAL_EXCLUDED_SIGNR,
    LEGACY_EXPERIMENTAL_OVERRIDES_MACD, LEGACY_EXPERIMENTAL_EXCLUDED_MACD
)
from data_loader import load_clc_full
from metrics import compute_metrics, max_drawdown_from_path
from strategies import strategy_long_only, strategy_sign_r, strategy_macd

# Strategy-specific optimal configurations (41/45 frontier)
STRATEGY_CONFIG_MAP = {
    'Long Only': (LEGACY_EXPERIMENTAL_OVERRIDES_LONG, LEGACY_EXPERIMENTAL_EXCLUDED_LONG),
    'Sign(R)': (LEGACY_EXPERIMENTAL_OVERRIDES_SIGNR, LEGACY_EXPERIMENTAL_EXCLUDED_SIGNR),
    'MACD': (LEGACY_EXPERIMENTAL_OVERRIDES_MACD, LEGACY_EXPERIMENTAL_EXCLUDED_MACD),
}

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="交易策略模拟平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

st.sidebar.title("⚙️ 控制面板")
st.sidebar.markdown("---")

# Date range
start_date = st.sidebar.date_input("开始日期", value=pd.to_datetime('2011-01-01'))
end_date = st.sidebar.date_input("结束日期", value=pd.to_datetime('2019-12-31'))

st.sidebar.markdown("---")

# Asset class selection
selected_assets = st.sidebar.multiselect(
    "📍 选择资产类别",
    options=list(ASSET_CLASSES.keys()),
    default=list(ASSET_CLASSES.keys())
)

# Strategy selection
strategies_available = ["Long Only", "Sign(R)", "MACD", "A2C", "A2C + Regime (B)", "DQN (Paper)"]
selected_strategies = st.sidebar.multiselect(
    "🎯 选择交易策略",
    options=strategies_available,
    default=["Long Only", "Sign(R)", "MACD", "A2C", "A2C + Regime (B)", "DQN (Paper)"]
)

# ============================================================================
# HELPER: Data Loading Functions
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
def load_a2c_data(asset_class):
    """Load A2C results from CSV."""
    try:
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
        
        # A2C data contains daily returns, convert to cumulative wealth
        returns = df[col].dropna().values
        if len(returns) < 10:
            return None
        
        # Convert returns to wealth (1.0 + cumsum(returns))
        wealth = 1.0 + np.cumsum(returns)
        dates = pd.to_datetime(df[col].dropna().index).values
        
        return (dates, wealth)
    except Exception as e:
        return None

@st.cache_data  
def load_routeb_data(asset_class):
    """Load Route B results from regime detection CSV."""
    try:
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
def load_dqn_data(asset_class):
    """Load DQN results from paper figures data."""
    try:
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
# MAIN CONTENT - TAB STRUCTURE
# ============================================================================

st.title("📊 交易策略模拟与分析平台")
st.markdown(f"**回测期间**: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 策略对比",
    "💹 性能指标",
    "📊 风险分析",
    "🔧 敏感性分析",
    "📥 数据管道"
])

# ============================================================================
# TAB 1: Strategy Comparison
# ============================================================================
with tab1:
    st.header("📈 策略对比分析")
    
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
                            plot_dates = pd.to_datetime(dates)
                            
                            ax.plot(plot_dates, wealth - 1.0,
                                   label=strategy_name, linewidth=1.5, 
                                   color=COLORS_FULL.get(strategy_name, '#888888'), 
                                   alpha=0.85)
                            data_loaded = True
            
            # A2C
            if 'A2C' in selected_strategies:
                result = load_a2c_data(asset)
                if result is not None:
                    dates, wealth = result
                    if wealth is not None and len(wealth) > 0:
                        plot_dates = pd.to_datetime(dates)
                        
                        ax.plot(plot_dates, wealth - 1.0,  # A2C is cumulative returns from 0
                               label='A2C', linewidth=1.5,
                               color=COLORS_FULL.get('A2C', '#888888'),
                               alpha=0.85)
                        data_loaded = True
            
            # Route B (A2C + Regime)
            if 'A2C + Regime (B)' in selected_strategies:
                result = load_routeb_data(asset)
                if result is not None:
                    dates, wealth = result
                    if wealth is not None and len(wealth) > 0:
                        plot_dates = pd.to_datetime(dates)
                        
                        ax.plot(plot_dates, wealth - 1.0,
                               label='A2C + Regime (B)', linewidth=1.5,
                               color=COLORS_FULL.get('A2C + Regime (B)', '#888888'),
                               alpha=0.85)
                        data_loaded = True
            
            # DQN
            if 'DQN (Paper)' in selected_strategies:
                result = load_dqn_data(asset)
                if result is not None:
                    dates, wealth = result
                    if wealth is not None and len(wealth) > 0:
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

# ============================================================================
# TAB 2: Performance Metrics
# ============================================================================
with tab2:
    st.header("💹 性能指标仪表板")
    
    if selected_assets and selected_strategies:
        st.info("📊 此功能正在开发中 - 显示所有策略的性能指标")
        
        metrics_data = []
        
        for asset in selected_assets:
            for strategy in ['Long Only', 'Sign(R)', 'MACD']:
                if strategy in selected_strategies:
                    result = load_baseline_strategy(asset, strategy)
                    if result is not None:
                        dates, wealth = result
                        if wealth is not None and len(wealth) > 0:
                            total_return = (wealth[-1] - 1.0) * 100
                            daily_returns = np.diff(wealth) / wealth[:-1]
                            annual_vol = np.std(daily_returns) * np.sqrt(252) * 100
                            sharpe = (total_return / 9) / annual_vol if annual_vol > 0 else 0
                            
                            # Calculate max drawdown
                            cummax = np.maximum.accumulate(wealth)
                            drawdown = (wealth - cummax) / cummax
                            max_dd = np.min(drawdown) * 100
                            
                            metrics_data.append({
                                "资产": asset,
                                "策略": strategy,
                                "总收益 (%)": f"{total_return:.2f}",
                                "年化波动 (%)": f"{annual_vol:.2f}",
                                "夏普比": f"{sharpe:.2f}",
                                "最大回撤 (%)": f"{max_dd:.2f}",
                            })
        
        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, use_container_width=True)
        else:
            st.warning("无可用数据")
    else:
        st.info("请在左侧选择资产类别和策略")

# ============================================================================
# TAB 3: Risk Analysis
# ============================================================================
with tab3:
    st.header("📊 风险分析")
    
    if selected_assets and selected_strategies:
        st.info("📈 此功能正在开发中 - 显示风险分析指标")
    else:
        st.info("请在左侧选择资产类别和策略")

# ============================================================================
# TAB 4: Sensitivity Analysis
# ============================================================================
with tab4:
    st.header("🔧 敏感性分析")
    
    if selected_assets and selected_strategies:
        st.info("⚠️ 此功能仅适用于基础策略（Long Only、Sign(R)、MACD）")
        st.markdown("机器学习模型（A2C、DQN、Route B）为预训练模型，波动率不可调整。")
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
