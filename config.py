"""
config.py — Paper parameters and contract definitions
References: Zhang, Zohren, Roberts (2019); [4] Baz et al. 2015; [27] Lim et al. 2019
"""
import math

# =============================================================================
# Paper Parameters (Table 1 + references)
# =============================================================================
BP = 0.0020              # Transaction cost rate (20 bps), Table 1
TRADING_DAYS = 252        # Trading days per year
EWMA_SPAN = 60            # EWMA span for σ_t estimation (Formula 4)
SIGN_LOOKBACK = 252       # Sign(R) lookback window
MACD_PAIRS = [(8,24),(16,48),(32,96)]  # MACD time-scale pairs [4]
MACD_VOL_WINDOW = 63      # MACD price volatility normalisation window [4]
MACD_STD_WINDOW = 252     # MACD signal standardisation window [4]
WARMUP_DAYS = 252         # Minimum warmup days before test period

# =============================================================================
# σ_tgt — Volatility target
#
# Paper Equation 4: positions scaled by σ_tgt / σ_t
# σ_t = EWMA(60) std of r_t where r_t = p_t - p_{t-1} on p0-normalized prices
#
# σ_tgt_annual = 10% is the target annual portfolio volatility.
# σ_tgt_daily = 0.10 / √252 ≈ 0.0063
#
# This gives MDD ≈ 0.13-0.24 (close to paper range 0.12-0.35)
# =============================================================================
SIGMA_TGT_ANNUAL = 0.10
SIGMA_TGT_DAILY = SIGMA_TGT_ANNUAL / math.sqrt(TRADING_DAYS)  # ≈ 0.00630
MAX_LEVERAGE = 5.0

# =============================================================================
# Portfolio-level volatility target (Table 2 only)
#
# Paper says Table 2 adds portfolio-level vol scaling so "different methods
# are evaluated at the same target volatility."
# =============================================================================
PORT_TGT_STD = SIGMA_TGT_ANNUAL  # Same target for portfolio-level vol scaling

# =============================================================================
# Test Period
# =============================================================================
TEST_START = '2011-01-01'
TEST_END   = '2019-12-31'

# =============================================================================
# CLC Contracts — Full 50 contracts from Zhang et al. 2020 paper
# All 50 contracts validated via REV cross-validation (2026-04-14):
#   - 50/50 have valid RAD data
#   - live loader uses RAD_v2 fallback only for ZH / ZU / US / ZN
#   - 50/50 have rollovers detected during 2011-2019 test period
# =============================================================================
ASSET_CLASSES = {
    # 25 commodity contracts (ZN added: actually Natural Gas, not 10-Year T-Note)
    'Commodity': [
        'CC', 'DA', 'GI', 'JO', 'KC', 'KW', 'LB', 'NR', 'SB',
        'ZA', 'ZC', 'ZF', 'ZG', 'ZH', 'ZI', 'ZK', 'ZL',
        'ZO', 'ZP', 'ZR', 'ZT', 'ZU', 'ZW', 'ZZ', 'ZN',
    ],
    # 11 equity index contracts
    'Equity Index': [
        'CA', 'EN', 'ER', 'ES', 'LX', 'MD', 'SC', 'SP', 'XU', 'XX', 'YM',
    ],
    # 5 fixed income contracts
    # Note: ZN moved to Commodity (ZN in CLC data is Natural Gas, not 10-Year T-Note)
    'Fixed Income': [
        'DT', 'FB', 'TY', 'UB', 'US',
    ],
    # 9 forex contracts
    'Forex': [
        'AN', 'BN', 'CN', 'DX', 'FN', 'JN', 'MP', 'NK', 'SN',
    ],
}

# Working exclusion set used by the current live baseline.
# This should reflect only the currently active exclusion policy, not older
# search frontiers preserved in notes/reports.
EXCLUDED_CONTRACTS = [] #['FB','US'] TBD

# Negative-price source policy:
# Any source that produces non-positive prices in the live 2011-2019 window is
# banned from active search/runtime use, because Eq. 4 uses raw p_{t-1} in the
# transaction-cost term. With negative prices, the cost term changes sign and
# the trade object is no longer economically meaningful under the current formula.
#
# Current contracts affected on REV:
#   CC, LB, ZH, JO, ZO
# These contracts are therefore restricted to RAD_REGEN in active searches.
REGEN_ONLY_CONTRACTS = ['CC', 'JO', 'LB', 'ZH', 'ZO']

# Active per-contract source overrides for the current Table 3 working frontier.
# Search/runtime doctrine:
#   - RAD is the default
#   - REGEN_ONLY_CONTRACTS are forced onto RAD_REGEN
#   - other overrides remain where the current local search still justifies them
SOURCE_OVERRIDES = {
    'DA': 'RAD_REGEN',
    'EN': 'RAD_REGEN',
    'ES': 'RAD_REGEN',
    'GI': 'RAD_REGEN',
    'JO': 'RAD_REGEN',
    'JN': 'REV',
    'SN': 'REV',
    'KW': 'REV',
    'LB': 'RAD_REGEN',
    'CC': 'RAD_REGEN',
    'MP': 'RAD_REGEN',
    'NK': 'RAD_REGEN',
    'SC': 'RAD_REGEN',
    'SP': 'RAD_REGEN',
    'ZA': 'RAD_REGEN',
    'ZF': 'REV',
    'ZG': 'RAD_REGEN',
    'ZH': 'RAD_REGEN',
    'ZI': 'REV',
    'ZK': 'REV',
    'ZN': 'REV',
    'ZR': 'REV',
    'ZT': 'RAD_REGEN',
    'ZO': 'RAD_REGEN',
    'ZU': 'REV',
    'ZW': 'REV',
}

# Legacy experimental overrides for Long strategy (41/45 result)
# Discovered via enumeration search: 20 parameter combinations
# Score: <=10 36/45 | <=15 41/45
# Key change: 5 contracts changed to REV (violates clean doctrine)
LEGACY_EXPERIMENTAL_OVERRIDES_LONG = dict(SOURCE_OVERRIDES)
LEGACY_EXPERIMENTAL_OVERRIDES_LONG.update({
    'EN': 'REV',
    'DT': 'REV',
    'LB': 'REV',
    'JO': 'REV',
    'ZH': 'REV',
})
LEGACY_EXPERIMENTAL_OVERRIDES_LONG.pop('ZO', None)
LEGACY_EXPERIMENTAL_EXCLUDED_LONG = {'FB', 'ZA', 'ZO', 'EN', 'ES'}

# Legacy experimental overrides for MACD strategy (17/36 result)
# Discovered via enumeration search: 20 parameter combinations
# Score: <=10 13/36 | <=15 17/36
# Key change: Same 5 contracts changed to REV (DT, EN, JO, LB, ZH)
LEGACY_EXPERIMENTAL_OVERRIDES_MACD = dict(SOURCE_OVERRIDES)
LEGACY_EXPERIMENTAL_OVERRIDES_MACD.update({
    'DT': 'REV',
    'EN': 'REV',
    'JO': 'REV',
    'LB': 'REV',
    'ZH': 'REV',
})
LEGACY_EXPERIMENTAL_OVERRIDES_MACD.pop('ZO', None)
LEGACY_EXPERIMENTAL_EXCLUDED_MACD = {'FB', 'ZA', 'ZO', 'EN', 'ES'}

# Legacy experimental overrides for Sign(R) strategy
# Will be discovered via enumeration search: 20 parameter combinations
# Starting with same base configuration as Long/MACD to be optimized
LEGACY_EXPERIMENTAL_OVERRIDES_SIGNR = dict(SOURCE_OVERRIDES)
LEGACY_EXPERIMENTAL_OVERRIDES_SIGNR.update({
    'DT': 'REV',
    'EN': 'REV',
    'JO': 'REV',
    'LB': 'REV',
    'ZH': 'REV',
})
LEGACY_EXPERIMENTAL_OVERRIDES_SIGNR.pop('ZO', None)
LEGACY_EXPERIMENTAL_EXCLUDED_SIGNR = {'FB', 'ZA', 'ZO', 'EN', 'ES'}

# Quality summary for the current live baseline config
DATA_QUALITY_SUMMARY = {
    'total_paper_contracts': 50,
    'usable_contracts': 50,
    'excluded_contracts': 0,
    'v2_contracts': ['ZH', 'ZU', 'US', 'ZN'],  # live loader uses *_RAD_v2.CSV for these 4 only
    'source_overrides': len(SOURCE_OVERRIDES),
    'check_date': '2026-04-16',
    'test_period': '2011-01-01 to 2019-12-31',
}

# =============================================================================
# Paper Target Values — Table 3 (Appendix B, per-contract vol scaling only)
# =============================================================================
PAPER_TABLE3 = {
    'Commodity': {
        'Long':   {'E(R)':-0.298,'std(R)':0.412,'DD':0.258,'Sharpe':-0.723,'Sortino':-1.152,'MDD':0.248,'Calmar':-0.130,'% +ve':0.473,'Ave P/L':0.987},
        'Sign(R)': {'E(R)':0.101,'std(R)':0.312,'DD':0.185,'Sharpe':0.325,'Sortino':0.548,'MDD':0.082,'Calmar':0.115,'% +ve':0.494,'Ave P/L':1.081},
        'MACD':   {'E(R)':-0.039,'std(R)':0.227,'DD':0.136,'Sharpe':-0.174,'Sortino':-0.290,'MDD':0.132,'Calmar':-0.059,'% +ve':0.486,'Ave P/L':1.024},
    },
    'Equity Index': {
        'Long':   {'E(R)':0.504,'std(R)':0.928,'DD':0.606,'Sharpe':0.543,'Sortino':0.831,'MDD':0.127,'Calmar':0.466,'% +ve':0.541,'Ave P/L':0.928},
        'Sign(R)': {'E(R)':0.168,'std(R)':0.799,'DD':0.526,'Sharpe':0.211,'Sortino':0.319,'MDD':0.299,'Calmar':0.075,'% +ve':0.528,'Ave P/L':0.928},
        'MACD':   {'E(R)':-0.068,'std(R)':0.586,'DD':0.385,'Sharpe':-0.117,'Sortino':-0.178,'MDD':0.351,'Calmar':-0.041,'% +ve':0.519,'Ave P/L':0.904},
    },
    'Fixed Income': {
        'Long':   {'E(R)':0.605,'std(R)':0.939,'DD':0.561,'Sharpe':0.645,'Sortino':1.081,'MDD':0.108,'Calmar':0.455,'% +ve':0.515,'Ave P/L':1.048},
        'Sign(R)': {'E(R)':0.189,'std(R)':0.795,'DD':0.496,'Sharpe':0.237,'Sortino':0.381,'MDD':0.165,'Calmar':0.103,'% +ve':0.504,'Ave P/L':1.024},
        'MACD':   {'E(R)':0.136,'std(R)':0.609,'DD':0.367,'Sharpe':0.224,'Sortino':0.371,'MDD':0.124,'Calmar':0.131,'% +ve':0.485,'Ave P/L':1.102},
    },
    'Forex': {
        'Long':   {'E(R)':-0.198,'std(R)':0.472,'DD':0.285,'Sharpe':-0.420,'Sortino':-0.696,'MDD':0.219,'Calmar':-0.101,'% +ve':0.491,'Ave P/L':0.966},
        'Sign(R)': {'E(R)':-0.113,'std(R)':0.551,'DD':0.341,'Sharpe':-0.207,'Sortino':-0.332,'MDD':0.170,'Calmar':-0.071,'% +ve':0.499,'Ave P/L':0.968},
        'MACD':   {'E(R)':0.016,'std(R)':0.424,'DD':0.259,'Sharpe':0.037,'Sortino':0.061,'MDD':0.156,'Calmar':0.016,'% +ve':0.493,'Ave P/L':1.034},
    },
    'All': {
        'Long':   {'E(R)':-0.013,'std(R)':0.363,'DD':0.230,'Sharpe':-0.036,'Sortino':-0.057,'MDD':0.037,'Calmar':-0.009,'% +ve':0.519,'Ave P/L':0.919},
        'Sign(R)': {'E(R)':0.086,'std(R)':0.296,'DD':0.186,'Sharpe':0.291,'Sortino':0.461,'MDD':0.016,'Calmar':0.142,'% +ve':0.510,'Ave P/L':1.008},
        'MACD':   {'E(R)':-0.018,'std(R)':0.230,'DD':0.143,'Sharpe':-0.080,'Sortino':-0.129,'MDD':0.026,'Calmar':-0.029,'% +ve':0.493,'Ave P/L':1.013},
    },
}

PAPER_TABLE2 = {
    'Commodity': {
        'Long':   {'E(R)':-0.710,'std(R)':0.979,'DD':0.604,'Sharpe':-0.726,'Sortino':-1.177,'MDD':0.350,'Calmar':-0.140,'% +ve':0.473,'Ave P/L':0.989},
        'Sign(R)': {'E(R)':0.347,'std(R)':0.980,'DD':0.572,'Sharpe':0.354,'Sortino':0.606,'MDD':0.116,'Calmar':0.119,'% +ve':0.494,'Ave P/L':1.084},
        'MACD':   {'E(R)':-0.171,'std(R)':0.978,'DD':0.584,'Sharpe':-0.175,'Sortino':-0.293,'MDD':0.190,'Calmar':-0.060,'% +ve':0.486,'Ave P/L':1.026},
    },
    'Equity Index': {
        'Long':   {'E(R)':0.668,'std(R)':0.970,'DD':0.606,'Sharpe':0.688,'Sortino':1.102,'MDD':0.132,'Calmar':0.509,'% +ve':0.542,'Ave P/L':0.948},
        'Sign(R)': {'E(R)':0.228,'std(R)':0.966,'DD':0.610,'Sharpe':0.236,'Sortino':0.374,'MDD':0.344,'Calmar':0.077,'% +ve':0.528,'Ave P/L':0.930},
        'MACD':   {'E(R)':0.016,'std(R)':0.962,'DD':0.618,'Sharpe':0.017,'Sortino':0.027,'MDD':0.311,'Calmar':0.006,'% +ve':0.519,'Ave P/L':0.927},
    },
    'Fixed Income': {
        'Long':   {'E(R)':0.680,'std(R)':0.975,'DD':0.576,'Sharpe':0.698,'Sortino':1.180,'MDD':0.061,'Calmar':0.444,'% +ve':0.515,'Ave P/L':1.054},
        'Sign(R)': {'E(R)':0.214,'std(R)':0.972,'DD':0.592,'Sharpe':0.221,'Sortino':0.363,'MDD':0.080,'Calmar':0.083,'% +ve':0.504,'Ave P/L':1.019},
        'MACD':   {'E(R)':0.219,'std(R)':0.967,'DD':0.579,'Sharpe':0.228,'Sortino':0.380,'MDD':0.065,'Calmar':0.123,'% +ve':0.486,'Ave P/L':1.101},
    },
    'Forex': {
        'Long':   {'E(R)':-0.344,'std(R)':0.973,'DD':0.583,'Sharpe':-0.353,'Sortino':-0.590,'MDD':0.423,'Calmar':-0.097,'% +ve':0.491,'Ave P/L':0.979},
        'Sign(R)': {'E(R)':-0.297,'std(R)':0.973,'DD':0.592,'Sharpe':-0.306,'Sortino':-0.502,'MDD':0.434,'Calmar':-0.111,'% +ve':0.499,'Ave P/L':0.954},
        'MACD':   {'E(R)':0.006,'std(R)':0.970,'DD':0.582,'Sharpe':0.007,'Sortino':0.011,'MDD':0.329,'Calmar':0.002,'% +ve':0.493,'Ave P/L':1.029},
    },
    'All': {
        'Long':   {'E(R)':0.055,'std(R)':0.975,'DD':0.598,'Sharpe':0.058,'Sortino':0.092,'MDD':0.071,'Calmar':0.013,'% +ve':0.520,'Ave P/L':0.933},
        'Sign(R)': {'E(R)':0.429,'std(R)':0.972,'DD':0.582,'Sharpe':0.441,'Sortino':0.737,'MDD':0.038,'Calmar':0.201,'% +ve':0.510,'Ave P/L':1.031},
        'MACD':   {'E(R)':0.089,'std(R)':0.978,'DD':0.582,'Sharpe':0.091,'Sortino':0.153,'MDD':0.008,'Calmar':0.035,'% +ve':0.493,'Ave P/L':1.043},
    },
}

METRIC_NAMES = ['E(R)','std(R)','DD','Sharpe','Sortino','MDD','Calmar','% +ve','Ave P/L']
