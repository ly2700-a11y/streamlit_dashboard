#!/usr/bin/env python3
"""Probe whether Yahoo-based ES/EN RAD_REGEN paths help when putting Equity back.

This script focuses on the historical high-score legacy experimental family:
  - exclusions base: FB, ZA, ZO
  - Equity reporting override: risk_price_non
  - numerator: annual_mean_sleeve
  - all_mode: contract_equal_path

It compares:
  1. legacy upper bound with EN/ES excluded
  2. put EN/ES back with current CLC sources
  3. put EN/ES back with Yahoo-based NON / RAD_REGEN replacements

Yahoo mappings are inferred/verified from prior local comparison:
  - ES -> ES=F
  - EN -> NQ=F
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TESTS_DIR = ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import frontier_40plus_enumeration as fe  # noqa: E402
from baseline_run import EWMA_SPAN, SIGN_LOOKBACK, strategy_macd  # noqa: E402
from config import SOURCE_OVERRIDES  # noqa: E402


YF_MAP = {"ES": "ES=F", "EN": "NQ=F"}
YF_MODES = ("YF_NON", "YF_RAD_REGEN")


def load_yf_cached(ticker: str):
    path = ROOT / "data" / "yahoo" / f"{ticker}_yahoo.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Yahoo cache for {ticker}: {path}")
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    df.columns = [c[0] for c in df.columns]
    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(df.index),
            "Close": pd.to_numeric(df["Close"], errors="coerce").values.flatten(),
        }
    )
    out = out[(out["Date"] >= "2009-01-01") & (out["Date"] <= "2019-12-31")]
    out = out[np.isfinite(out["Close"]) & (out["Close"] > 0)].sort_values("Date").reset_index(drop=True)
    return out


def load_clc_non_rev(ticker: str):
    non = pd.read_csv(ROOT / "data" / "CLC" / f"{ticker}_NON.CSV", header=None,
                      names=["Date", "Open", "High", "Low", "Close", "Volume", "OI"])
    rev = pd.read_csv(ROOT / "data" / "CLC" / f"{ticker}_REV.CSV", header=None,
                      names=["Date", "Open", "High", "Low", "Close", "Volume", "OI"])
    non["Date"] = pd.to_datetime(non["Date"], format="%m/%d/%Y")
    rev["Date"] = pd.to_datetime(rev["Date"], format="%m/%d/%Y")
    merged = non[["Date", "Close"]].merge(
        rev[["Date", "Close"]],
        on="Date",
        how="inner",
        suffixes=("_non", "_rev"),
    )
    merged = merged.sort_values("Date").reset_index(drop=True)
    p_non = pd.to_numeric(merged["Close_non"], errors="coerce").values
    p_rev = pd.to_numeric(merged["Close_rev"], errors="coerce").values
    valid = np.isfinite(p_non) & np.isfinite(p_rev) & (p_non > 0)
    merged = merged[valid].reset_index(drop=True)
    merged["Close_non"] = p_non[valid]
    merged["Close_rev"] = p_rev[valid]
    merged = merged[(merged["Date"] >= "2009-01-01") & (merged["Date"] <= "2019-12-31")].reset_index(drop=True)
    return merged


def build_yf_non(ticker: str):
    df = load_yf_cached(ticker)
    return df["Date"].to_numpy(), df["Close"].to_numpy(dtype=float)


def build_yf_rad_regen(ticker: str):
    dates_yf, prices_yf = build_yf_non(ticker)
    merged = load_clc_non_rev(ticker)
    p_non = merged["Close_non"].to_numpy(dtype=float)
    adj = merged["Close_rev"].to_numpy(dtype=float) - p_non
    adj_diff = np.diff(adj)
    roll_idx = np.where(np.abs(adj_diff) > 1e-6)[0]

    cum_ratio = np.ones(len(prices_yf))
    yf_map = {pd.Timestamp(d): i for i, d in enumerate(dates_yf)}
    clc_dates = merged["Date"].to_numpy()
    for idx in roll_idx:
        roll_date = pd.Timestamp(clc_dates[idx])
        yj = yf_map.get(roll_date)
        if yj is None or yj + 1 >= len(cum_ratio):
            continue
        new_price = p_non[idx + 1]
        if abs(new_price) <= 1e-12:
            continue
        ratio = p_non[idx] / new_price
        cum_ratio[yj + 1:] *= ratio

    p_rad = prices_yf * cum_ratio
    valid = np.isfinite(p_rad) & (p_rad > 0)
    return dates_yf[valid], p_rad[valid]


def prepare_custom_contract(ticker: str, dates, prices, source_label: str):
    rt = np.zeros(len(prices))
    rt[1:] = prices[1:] - prices[:-1]
    sigma = pd.Series(rt).ewm(span=EWMA_SPAN, adjust=False).std().values

    dates_s = pd.Series(pd.to_datetime(dates))
    mask_s = dates_s >= "2011-01-01"
    mask_e = dates_s <= "2019-12-31"
    if not mask_s.any() or not mask_e.any():
        return None
    t0 = int(mask_s.idxmax())
    t1 = int(len(dates_s) - 1 - mask_e[::-1].values.argmax())
    start = max(t0, SIGN_LOOKBACK)
    return {
        "tk": ticker,
        "rt": rt,
        "sigma": sigma,
        "prices": np.asarray(prices, dtype=float),
        "start": start,
        "t1": t1,
        "dates": dates_s.iloc[start:t1 + 1].to_numpy(),
        "source": source_label,
        "macd_pos": strategy_macd(np.asarray(prices, dtype=float)),
    }


def custom_prepared(mode_map: dict[str, str]):
    out = {}
    for ticker, mode in mode_map.items():
        if mode == "YF_NON":
            dates, prices = build_yf_non(ticker)
        elif mode == "YF_RAD_REGEN":
            dates, prices = build_yf_rad_regen(ticker)
        else:
            raise ValueError(mode)
        out[ticker] = prepare_custom_contract(ticker, dates, prices, mode)
    return out


def scenario_with_custom_sources(label: str, excluded: set[str], mode_map: dict[str, str] | None):
    original_load_raw = fe.load_raw
    prepared = custom_prepared(mode_map or {}) if mode_map else {}

    def patched_load_raw(asset, overrides, excluded_list):
        if asset == "All":
            raw = []
            for asset_name in fe.ASSETS4:
                raw.extend(patched_load_raw(asset_name, overrides, excluded_list))
            return raw
        raw = fe.load_contracts(asset, excluded_contracts=excluded_list, source_overrides=overrides)
        if asset == "Equity Index" and prepared:
            by_tk = {rd["tk"]: rd for rd in raw}
            for ticker, rd in prepared.items():
                if ticker in excluded_list:
                    continue
                by_tk[ticker] = rd
            ordered = []
            for tk in fe.ASSET_CLASSES[asset]:
                if tk in excluded_list:
                    continue
                if tk in by_tk:
                    ordered.append(by_tk[tk])
            return ordered
        return raw

    try:
        fe.load_raw = patched_load_raw
        fe.evaluate_scenario.cache_clear()
        row = fe.scenario(
            label=label,
            family="equity_yf_probe",
            overrides=fe.LEGACY_EXPERIMENTAL_OVERRIDES,
            excluded=excluded,
            asset_capital_overrides={"Equity Index": "risk_price_non"},
            numerator_mode="annual_mean_sleeve",
            asset_path_mode="contract_equal_path",
            all_mode="contract_equal_path",
            same_rule=False,
            asset_specific=True,
            structural_heavy=True,
            experimental=True,
        )
        return row
    finally:
        fe.load_raw = original_load_raw
        fe.evaluate_scenario.cache_clear()


def summarize(row):
    s = row["summary"]
    return {
        "score10": s["score10"],
        "score15": s["score15"],
        "equity_misses": s["results"]["Equity Index"]["misses15"],
        "all_misses": s["results"]["All"]["misses15"],
        "equity_er": s["results"]["Equity Index"]["metrics"]["E(R)"],
        "equity_mdd": s["results"]["Equity Index"]["metrics"]["MDD"],
        "equity_cal": s["results"]["Equity Index"]["metrics"]["Calmar"],
        "all_er": s["results"]["All"]["metrics"]["E(R)"],
        "all_mdd": s["results"]["All"]["metrics"]["MDD"],
        "all_cal": s["results"]["All"]["metrics"]["Calmar"],
    }


def main():
    # Verify current mappings are still sensible against current CLC NON.
    lines = ["# ES/EN Yahoo RAD_REGEN Probe", "", "## Mapping", ""]
    for ticker, sym in YF_MAP.items():
        lines.append(f"- `{ticker}` uses Yahoo `{sym}`")
    lines.append("")

    scenarios = [
        ("legacy upper bound (EN/ES excluded)", {"FB", "ZA", "ZO", "EN", "ES"}, None),
        ("put back EN+ES with current CLC sources", {"FB", "ZA", "ZO"}, None),
        ("put back EN+ES with YF_NON on both", {"FB", "ZA", "ZO"}, {"EN": "YF_NON", "ES": "YF_NON"}),
        ("put back EN+ES with YF_RAD_REGEN on both", {"FB", "ZA", "ZO"}, {"EN": "YF_RAD_REGEN", "ES": "YF_RAD_REGEN"}),
        ("put back EN only via YF_RAD_REGEN", {"FB", "ZA", "ZO"}, {"EN": "YF_RAD_REGEN"}),
        ("put back ES only via YF_RAD_REGEN", {"FB", "ZA", "ZO"}, {"ES": "YF_RAD_REGEN"}),
    ]

    rows = []
    for label, excluded, mode_map in scenarios:
        row = scenario_with_custom_sources(label, excluded, mode_map)
        info = summarize(row)
        rows.append((label, info))

    lines.extend(
        [
            "## Score Summary",
            "",
            "| Scenario | <=10 | <=15 | Equity misses | All misses |",
            "| --- | ---: | ---: | --- | --- |",
        ]
    )
    for label, info in rows:
        lines.append(
            f"| {label} | {info['score10']} | {info['score15']} | "
            f"{', '.join(info['equity_misses']) or 'none'} | {', '.join(info['all_misses']) or 'none'} |"
        )

    lines.extend(
        [
            "",
            "## Key Metrics",
            "",
            "| Scenario | Equity E(R) | Equity MDD | Equity Calmar | All E(R) | All MDD | All Calmar |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, info in rows:
        lines.append(
            f"| {label} | {info['equity_er']:+.3f} | {info['equity_mdd']:.3f} | {info['equity_cal']:+.3f} | "
            f"{info['all_er']:+.3f} | {info['all_mdd']:.3f} | {info['all_cal']:+.3f} |"
        )

    text = "\n".join(lines) + "\n"
    out = ROOT / "docs" / "equity_yf_rad_regen_probe.md"
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
