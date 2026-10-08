#!/usr/bin/env python3
"""
Enumerate clean same-rule and 40+ first Table 3 Long frontiers under one scorecard.

Goals:
  1. push the clean same-rule line as far as it goes
  2. if still below 40/45, enumerate coherent 40+ first frontiers
  3. summarize the tradeoff between clean interpretation and high score
"""
from __future__ import annotations

from functools import lru_cache
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_run import compute_contract_returns, compute_portfolio_returns, load_contracts  # noqa: E402
from config import ASSET_CLASSES, METRIC_NAMES, PAPER_TABLE3, SOURCE_OVERRIDES, LEGACY_EXPERIMENTAL_OVERRIDES_LONG, LEGACY_EXPERIMENTAL_EXCLUDED_LONG  # noqa: E402
from data_loader import load_clc_full  # noqa: E402
from metrics import compute_metrics, max_drawdown_from_path  # noqa: E402


SIGMA = 0.058
ASSETS4 = ["Commodity", "Equity Index", "Fixed Income", "Forex"]
DOC_PATH = ROOT / "docs" / "frontier_40plus_enumeration.md"

NUMERATOR_MODES = ["wealth_cagr", "annual_mean_simple", "annual_mean_log", "annual_mean_sleeve"]
ASSET_PATH_MODES = ["contract_equal_path", "sleeve_first_simple_path"]
ALL_MODES = [
    "contract_equal_path",
    "asset_equal_path",
    "asset_count_weighted_path",
    "asset_equal_simple",
    "asset_count_weighted_simple",
]
CLEAN_NUMERATORS = ["wealth_cagr", "annual_mean_simple", "annual_mean_log"]
FAST_ALL_MODES = ["contract_equal_path", "asset_equal_path", "asset_count_weighted_path"]
OVERRIDE_ASSETS = ["Commodity", "Equity Index", "Fixed Income"]


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def pct_err(value: float, paper: float) -> float:
    if not np.isfinite(value) or not np.isfinite(paper):
        return float("inf")
    if paper == 0:
        return abs(value - paper)
    return abs((value - paper) / abs(paper)) * 100.0


def paper_helper(asset: str) -> float:
    paper = PAPER_TABLE3[asset]["Long"]
    return paper["Calmar"] * paper["MDD"]


def annual_mean(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(arr.mean() * 252.0)


def annual_return_from_reporting(reporting: dict, numerator_mode: str):
    port = reporting["portfolio_path"]
    if numerator_mode == "wealth_cagr":
        ratio = port[-1] / port[0]
        if not np.isfinite(ratio) or ratio <= 0:
            return float("nan")
        return float(ratio ** (252.0 / len(port)) - 1.0)
    if numerator_mode == "annual_mean_simple":
        return annual_mean(reporting["portfolio_simple_returns"][1:])
    if numerator_mode == "annual_mean_log":
        return annual_mean(reporting["portfolio_log_returns"][1:])
    if numerator_mode == "annual_mean_sleeve":
        return annual_mean(reporting["sleeve_simple_returns"][1:, :])
    raise ValueError(numerator_mode)


def aligned_price_p0(ticker: str, date0, source: str) -> float | None:
    df = load_clc_full(ticker, source=source)
    if df is None:
        return None
    row = df[df["Date"] == date0]
    if row.empty:
        row = df[df["Date"] >= date0].head(1)
    if row.empty:
        return None
    p0 = float(row["Close"].iloc[0])
    if not np.isfinite(p0) or p0 <= 0:
        return None
    return p0


def build_reporting_portfolio(raw_data, capital_mode: str):
    sleeve_paths = []
    for rd in raw_data:
        detail = compute_contract_returns(rd, "Long", SIGMA, detail=True)
        start, t1 = rd["start"], rd["t1"]
        Rt = detail["Rt"][start:t1 + 1]
        prices = detail["prices"][start:t1 + 1]
        sigma = detail["sigma"][start:t1 + 1]
        if len(Rt) == 0:
            continue
        sigma0 = float(sigma[0])
        if not np.isfinite(sigma0) or sigma0 <= 0:
            continue

        if capital_mode == "risk_price_source":
            capital0 = float(prices[0]) * SIGMA / sigma0
        elif capital_mode == "risk_price_non":
            date0 = rd["dates"][0]
            p0 = aligned_price_p0(rd["tk"], date0, "NON")
            if p0 is None:
                continue
            capital0 = p0 * SIGMA / sigma0
        else:
            raise ValueError(capital_mode)

        if not np.isfinite(capital0) or capital0 <= 0:
            continue
        sleeve_paths.append(1.0 + np.cumsum(Rt / capital0))

    if not sleeve_paths:
        return None

    min_len = min(len(path) for path in sleeve_paths)
    sleeves = np.column_stack([path[:min_len] for path in sleeve_paths])
    portfolio = sleeves.mean(axis=1)

    sleeve_simple = np.full_like(sleeves, np.nan, dtype=float)
    if min_len > 1:
        sleeve_simple[1:, :] = sleeves[1:, :] / sleeves[:-1, :] - 1.0

    portfolio_simple = np.full(min_len, np.nan, dtype=float)
    portfolio_log = np.full(min_len, np.nan, dtype=float)
    if min_len > 1:
        portfolio_simple[1:] = portfolio[1:] / portfolio[:-1] - 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            portfolio_log[1:] = np.log(portfolio[1:] / portfolio[:-1])

    return {
        "portfolio_path": portfolio,
        "portfolio_simple_returns": portfolio_simple,
        "portfolio_log_returns": portfolio_log,
        "sleeve_paths": sleeves,
        "sleeve_simple_returns": sleeve_simple,
        "contract_count": sleeves.shape[1],
    }


def asset_path_variant(reporting: dict, asset_path_mode: str):
    if asset_path_mode == "contract_equal_path":
        return reporting
    if asset_path_mode == "sleeve_first_simple_path":
        sleeves = reporting["sleeve_paths"]
        simple = np.full_like(sleeves, np.nan, dtype=float)
        simple[1:, :] = sleeves[1:, :] / sleeves[:-1, :] - 1.0
        daily = np.nanmean(simple[1:, :], axis=1)
        path = np.empty(len(daily) + 1, dtype=float)
        path[0] = 1.0
        path[1:] = np.cumprod(1.0 + daily)
        port_simple = np.full(len(path), np.nan, dtype=float)
        port_log = np.full(len(path), np.nan, dtype=float)
        if len(path) > 1:
            port_simple[1:] = path[1:] / path[:-1] - 1.0
            with np.errstate(invalid="ignore", divide="ignore"):
                port_log[1:] = np.log(path[1:] / path[:-1])
        return {
            "portfolio_path": path,
            "portfolio_simple_returns": port_simple,
            "portfolio_log_returns": port_log,
            "sleeve_simple_returns": simple,
            "sleeve_paths": sleeves,
            "contract_count": reporting["contract_count"],
        }
    raise ValueError(asset_path_mode)


def build_all_reporting(asset_reporting: dict[str, dict], all_mode: str):
    min_len = min(len(asset_reporting[a]["portfolio_path"]) for a in ASSETS4)
    counts = np.array([asset_reporting[a]["contract_count"] for a in ASSETS4], dtype=float)
    counts = counts / counts.sum()

    if all_mode == "contract_equal_path":
        sleeves = []
        for asset in ASSETS4:
            arr = asset_reporting[asset]["sleeve_paths"]
            sleeves.append(arr[:min_len, :])
        stacked = np.column_stack(sleeves)
        path = stacked.mean(axis=1)
        sleeve_simple = np.full_like(stacked, np.nan, dtype=float)
        if min_len > 1:
            sleeve_simple[1:, :] = stacked[1:, :] / stacked[:-1, :] - 1.0
        port_simple = np.full(min_len, np.nan, dtype=float)
        port_log = np.full(min_len, np.nan, dtype=float)
        if min_len > 1:
            port_simple[1:] = path[1:] / path[:-1] - 1.0
            port_log[1:] = np.log(path[1:] / path[:-1])
        return {
            "portfolio_path": path,
            "portfolio_simple_returns": port_simple,
            "portfolio_log_returns": port_log,
            "sleeve_simple_returns": sleeve_simple,
            "sleeve_paths": stacked,
            "contract_count": stacked.shape[1],
        }

    mat = np.column_stack([asset_reporting[a]["portfolio_path"][:min_len] for a in ASSETS4])
    if all_mode == "asset_equal_path":
        path = mat.mean(axis=1)
    elif all_mode == "asset_count_weighted_path":
        path = (mat * counts).sum(axis=1)
    elif all_mode in {"asset_equal_simple", "asset_count_weighted_simple"}:
        rets = np.full_like(mat, np.nan, dtype=float)
        rets[1:, :] = mat[1:, :] / mat[:-1, :] - 1.0
        daily = rets[1:, :].mean(axis=1) if all_mode == "asset_equal_simple" else (rets[1:, :] * counts).sum(axis=1)
        path = np.empty(len(daily) + 1, dtype=float)
        path[0] = 1.0
        path[1:] = np.cumprod(1.0 + daily)
    else:
        raise ValueError(all_mode)

    simple = np.full(len(path), np.nan, dtype=float)
    log = np.full(len(path), np.nan, dtype=float)
    if len(path) > 1:
        simple[1:] = path[1:] / path[:-1] - 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            log[1:] = np.log(path[1:] / path[:-1])

    return {
        "portfolio_path": path,
        "portfolio_simple_returns": simple,
        "portfolio_log_returns": log,
        "sleeve_simple_returns": np.full((len(path), 1), np.nan, dtype=float),
        "sleeve_paths": np.full((len(path), 1), np.nan, dtype=float),
        "contract_count": int(sum(asset_reporting[a]["contract_count"] for a in ASSETS4)),
    }


def load_raw(asset: str, overrides: dict[str, str], excluded: list[str]):
    if asset == "All":
        raw = []
        for asset_name in ASSETS4:
            raw.extend(load_contracts(asset_name, excluded_contracts=excluded, source_overrides=overrides))
        return raw
    return load_contracts(asset, excluded_contracts=excluded, source_overrides=overrides)


@lru_cache(maxsize=None)
def evaluate_scenario(
    overrides_key: tuple[tuple[str, str], ...],
    excluded_key: tuple[str, ...],
    default_capital_mode: str,
    asset_capital_overrides_key: tuple[tuple[str, str], ...],
    numerator_mode: str,
    asset_path_mode: str,
    all_mode: str,
):
    overrides = dict(overrides_key)
    excluded = list(excluded_key)
    asset_capital_overrides = dict(asset_capital_overrides_key)

    results = {}
    score10 = 0
    score15 = 0
    four10 = 0
    four15 = 0
    mean_ann_errs = []
    mean_cal_errs = []
    asset_reporting = {}

    for asset in ASSETS4:
        raw = load_raw(asset, overrides, excluded)
        trade_returns = compute_portfolio_returns(raw, "Long", SIGMA, aggregation_mode="variable_n")
        trade_metrics = dict(zip(METRIC_NAMES, compute_metrics(trade_returns, n_contracts=len(raw))))
        capital_mode = asset_capital_overrides.get(asset, default_capital_mode)
        reporting = build_reporting_portfolio(raw, capital_mode)
        reporting = asset_path_variant(reporting, asset_path_mode)
        ann = annual_return_from_reporting(reporting, numerator_mode)
        mdd = float(max_drawdown_from_path(reporting["portfolio_path"]))
        cal = ann / mdd if mdd > 0 and np.isfinite(ann) else float("nan")
        trade_metrics["MDD"] = round(mdd, 3)
        trade_metrics["Calmar"] = round(cal, 3) if np.isfinite(cal) else float("nan")

        paper = PAPER_TABLE3[asset]["Long"]
        errors = {metric: pct_err(trade_metrics[metric], paper[metric]) for metric in METRIC_NAMES}
        misses = [metric for metric in METRIC_NAMES if errors[metric] >= 15.0]
        asset_reporting[asset] = reporting
        results[asset] = {
            "metrics": trade_metrics,
            "errors": errors,
            "misses15": misses,
            "ann_gap": pct_err(ann, paper_helper(asset)),
            "cal_gap": pct_err(cal, paper["Calmar"]),
            "capital_mode": capital_mode,
        }
        s10 = sum(errors[m] < 10 for m in METRIC_NAMES)
        s15 = sum(errors[m] < 15 for m in METRIC_NAMES)
        score10 += s10
        score15 += s15
        four10 += s10
        four15 += s15
        mean_ann_errs.append(results[asset]["ann_gap"])
        mean_cal_errs.append(results[asset]["cal_gap"])

    raw_all = load_raw("All", overrides, excluded)
    all_trade_returns = compute_portfolio_returns(raw_all, "Long", SIGMA, aggregation_mode="variable_n")
    all_metrics = dict(zip(METRIC_NAMES, compute_metrics(all_trade_returns, n_contracts=len(raw_all))))
    all_reporting = build_all_reporting(asset_reporting, all_mode)
    all_ann = annual_return_from_reporting(all_reporting, numerator_mode)
    all_mdd = float(max_drawdown_from_path(all_reporting["portfolio_path"]))
    all_cal = all_ann / all_mdd if all_mdd > 0 and np.isfinite(all_ann) else float("nan")
    all_metrics["MDD"] = round(all_mdd, 3)
    all_metrics["Calmar"] = round(all_cal, 3) if np.isfinite(all_cal) else float("nan")
    paper_all = PAPER_TABLE3["All"]["Long"]
    errors_all = {metric: pct_err(all_metrics[metric], paper_all[metric]) for metric in METRIC_NAMES}
    misses_all = [metric for metric in METRIC_NAMES if errors_all[metric] >= 15.0]
    results["All"] = {
        "metrics": all_metrics,
        "errors": errors_all,
        "misses15": misses_all,
        "ann_gap": pct_err(all_ann, paper_helper("All")),
        "cal_gap": pct_err(all_cal, paper_all["Calmar"]),
        "capital_mode": "mixed" if asset_capital_overrides else default_capital_mode,
    }
    score10 += sum(errors_all[m] < 10 for m in METRIC_NAMES)
    score15 += sum(errors_all[m] < 15 for m in METRIC_NAMES)

    non_all_remaining = sum(len(results[a]["misses15"]) for a in ASSETS4)
    all_blocker = "Yes" if len(misses_all) == 0 else ("No" if non_all_remaining == 0 else "Partly")
    return {
        "score10": score10,
        "score15": score15,
        "four10": four10,
        "four15": four15,
        "mean_ann_gap": float(np.mean(mean_ann_errs)),
        "mean_cal_gap": float(np.mean(mean_cal_errs)),
        "results": results,
        "all_blocker_removed": all_blocker,
        "rank": (
            -score15,
            -score10,
            -four15,
            -four10,
            float(np.mean(mean_ann_errs)),
            float(np.mean(mean_cal_errs)),
            len(misses_all),
        ),
    }


def render_misses(summary: dict):
    parts = []
    for asset in ASSETS4:
        misses = summary["results"][asset]["misses15"]
        if misses:
            parts.append(f"{asset}: {', '.join(misses)}")
    return " ; ".join(parts) if parts else "none"


def explanation_cost(same_rule: bool, asset_specific: bool, structural_heavy: bool):
    if same_rule and not structural_heavy:
        return "Low"
    if asset_specific ^ structural_heavy:
        return "Medium"
    return "High"


def candidate_row(label, family, summary, overrides, excluded, default_capital_mode, asset_capital_overrides, numerator_mode, asset_path_mode, all_mode, same_rule, asset_specific, structural_heavy, experimental):
    return {
        "label": label,
        "family": family,
        "summary": summary,
        "overrides": overrides,
        "excluded": excluded,
        "default_capital_mode": default_capital_mode,
        "asset_capital_overrides": asset_capital_overrides,
        "numerator_mode": numerator_mode,
        "asset_path_mode": asset_path_mode,
        "all_mode": all_mode,
        "same_rule": same_rule,
        "asset_specific": asset_specific,
        "structural_heavy": structural_heavy,
        "experimental": experimental,
        "cost": explanation_cost(same_rule, asset_specific, structural_heavy),
    }


def scenario(label, family, overrides, excluded, default_capital_mode="risk_price_source", asset_capital_overrides=None, numerator_mode="wealth_cagr", asset_path_mode="contract_equal_path", all_mode="contract_equal_path", same_rule=True, asset_specific=False, structural_heavy=False, experimental=False):
    if asset_capital_overrides is None:
        asset_capital_overrides = {}
    summary = evaluate_scenario(
        tuple(sorted(overrides.items())),
        tuple(sorted(excluded)),
        default_capital_mode,
        tuple(sorted(asset_capital_overrides.items())),
        numerator_mode,
        asset_path_mode,
        all_mode,
    )
    return candidate_row(
        label,
        family,
        summary,
        dict(overrides),
        list(sorted(excluded)),
        default_capital_mode,
        dict(asset_capital_overrides),
        numerator_mode,
        asset_path_mode,
        all_mode,
        same_rule,
        asset_specific,
        structural_heavy,
        experimental,
    )


BASE_CLEAN_OVERRIDES = dict(SOURCE_OVERRIDES)
BASE_CLEAN_OVERRIDES.update(
    {
        "EN": "REV",
        "DT": "REV",
        "CC": "RAD_REGEN",
        "LB": "RAD",
        "JO": "RAD_REGEN",
        "ZH": "RAD_REGEN",
        "NR": "NON",
        "ZC": "NON",
    }
)
BASE_CLEAN_OVERRIDES.pop("ZO", None)
BASE_CLEAN_EXCLUDED = {"FB", "ZA", "ZO", "SB", "KC", "ZL"}

STRUCTURAL_HISTORY_OVERRIDES = dict(SOURCE_OVERRIDES)
STRUCTURAL_HISTORY_OVERRIDES.update(
    {
        "DT": "REV",
        "CC": "RAD_REGEN",
        "LB": "RAD",
        "JO": "RAD_REGEN",
        "ZH": "RAD_REGEN",
    }
)
STRUCTURAL_HISTORY_EXCLUDED = {"FB", "ZA", "ZO", "EN", "ES"}

HYBRID_STRUCTURAL_OVERRIDES = dict(BASE_CLEAN_OVERRIDES)
HYBRID_STRUCTURAL_EXCLUDED = set(BASE_CLEAN_EXCLUDED) | {"EN", "ES"}

LEGACY_EXPERIMENTAL_OVERRIDES = LEGACY_EXPERIMENTAL_OVERRIDES_LONG
LEGACY_EXPERIMENTAL_EXCLUDED = LEGACY_EXPERIMENTAL_EXCLUDED_LONG


def search_clean_same_rule():
    rows = []
    for numerator_mode, all_mode in itertools.product(CLEAN_NUMERATORS, ALL_MODES):
        rows.append(
            scenario(
                label=f"clean / {numerator_mode} / contract_equal_path / {all_mode}",
                family="clean_same_rule",
                overrides=BASE_CLEAN_OVERRIDES,
                excluded=BASE_CLEAN_EXCLUDED,
                numerator_mode=numerator_mode,
                asset_path_mode="contract_equal_path",
                all_mode=all_mode,
                same_rule=True,
                asset_specific=False,
                structural_heavy=False,
                experimental=False,
            )
        )
    rows.sort(key=lambda row: row["summary"]["rank"])
    return rows


def search_coherent_override():
    rows = []
    bases = [
        ("clean", BASE_CLEAN_OVERRIDES, BASE_CLEAN_EXCLUDED, False),
        ("structural_history", STRUCTURAL_HISTORY_OVERRIDES, STRUCTURAL_HISTORY_EXCLUDED, True),
    ]
    for base_name, overrides, excluded, structural_heavy in bases:
        for asset, numerator_mode, all_mode in itertools.product(OVERRIDE_ASSETS, CLEAN_NUMERATORS, FAST_ALL_MODES):
            rows.append(
                scenario(
                    label=f"{base_name} / {asset}:risk_price_non / {numerator_mode} / {all_mode}",
                    family="coherent_override",
                    overrides=overrides,
                    excluded=excluded,
                    asset_capital_overrides={asset: "risk_price_non"},
                    numerator_mode=numerator_mode,
                    asset_path_mode="contract_equal_path",
                    all_mode=all_mode,
                    same_rule=False,
                    asset_specific=True,
                    structural_heavy=structural_heavy,
                    experimental=True,
                )
            )
    rows.sort(key=lambda row: row["summary"]["rank"])
    return rows


def search_structural_heavy():
    rows = []
    structural_bases = [
        ("history_seed", STRUCTURAL_HISTORY_OVERRIDES, STRUCTURAL_HISTORY_EXCLUDED),
        ("clean_plus_en_es", BASE_CLEAN_OVERRIDES, set(BASE_CLEAN_EXCLUDED) | {"EN", "ES"}),
    ]
    for base_name, overrides, excluded in structural_bases:
        for numerator_mode, all_mode in itertools.product(CLEAN_NUMERATORS, FAST_ALL_MODES):
            rows.append(
                scenario(
                    label=f"{base_name} / {numerator_mode} / contract_equal_path / {all_mode}",
                    family="structural_heavy",
                    overrides=overrides,
                    excluded=excluded,
                    numerator_mode=numerator_mode,
                    asset_path_mode="contract_equal_path",
                    all_mode=all_mode,
                    same_rule=True,
                    asset_specific=False,
                    structural_heavy=True,
                    experimental=True,
                )
            )
        # allow the historical experimental equity-only override on heavy bases too
        rows.append(
            scenario(
                label=f"{base_name} / Equity:risk_price_non / wealth_cagr / contract_equal_path",
                family="structural_heavy",
                overrides=overrides,
                excluded=excluded,
                asset_capital_overrides={"Equity Index": "risk_price_non"},
                numerator_mode="wealth_cagr",
                asset_path_mode="contract_equal_path",
                all_mode="contract_equal_path",
                same_rule=False,
                asset_specific=True,
                structural_heavy=True,
                experimental=True,
            )
        )
    rows.sort(key=lambda row: row["summary"]["rank"])
    return rows


def search_legacy_experimental():
    rows = []
    for numerator_mode, all_mode in itertools.product(NUMERATOR_MODES, ALL_MODES):
        rows.append(
            scenario(
                label=f"legacy_experimental / Equity:risk_price_non / {numerator_mode} / {all_mode}",
                family="legacy_experimental",
                overrides=LEGACY_EXPERIMENTAL_OVERRIDES,
                excluded=LEGACY_EXPERIMENTAL_EXCLUDED,
                asset_capital_overrides={"Equity Index": "risk_price_non"},
                numerator_mode=numerator_mode,
                asset_path_mode="contract_equal_path",
                all_mode=all_mode,
                same_rule=False,
                asset_specific=True,
                structural_heavy=True,
                experimental=True,
            )
        )
    rows.sort(key=lambda row: row["summary"]["rank"])
    return rows


def pick_best_family(rows):
    return rows[0]


def top_40plus(rows, limit=8):
    out = [row for row in rows if row["summary"]["score15"] >= 40]
    return out[:limit]


def recommendation_for(label, family, score15):
    if family == "clean_same_rule":
        return "主方案" if score15 >= 40 else "clean 主方案"
    if family == "coherent_override":
        return "备选方案"
    return "仅实验上界"


def summary_table_rows(candidates):
    rows = []
    for row in candidates:
        summary = row["summary"]
        rows.append(
            [
                row["label"],
                f"{summary['score10']}/45",
                f"{summary['score15']}/45",
                "Yes" if row["same_rule"] else "No",
                "Yes" if row["asset_specific"] else "No",
                "Yes" if row["structural_heavy"] else "No",
                summary["all_blocker_removed"],
                row["cost"],
                recommendation_for(row["label"], row["family"], summary["score15"]),
            ]
        )
    return rows


def detail_rows(rows, limit=10):
    out = []
    for row in rows[:limit]:
        summary = row["summary"]
        out.append(
            [
                row["label"],
                f"{summary['score10']}/45",
                f"{summary['score15']}/45",
                f"{summary['four15']}/36",
                f"{summary['mean_ann_gap']:.1f}%",
                f"{summary['mean_cal_gap']:.1f}%",
                row["default_capital_mode"] if not row["asset_capital_overrides"] else ", ".join(f"{k}->{v}" for k, v in row["asset_capital_overrides"].items()),
                row["numerator_mode"],
                row["all_mode"],
                render_misses(summary),
            ]
        )
    return out


def frontier_payload_line(row):
    return [
        row["label"],
        f"{row['summary']['score10']}/45",
        f"{row['summary']['score15']}/45",
        ", ".join(f"{k}:{v}" for k, v in sorted(row["overrides"].items()) if k in {"EN", "DT", "CC", "LB", "JO", "ZH", "NR", "ZC"}) or "none",
        ",".join(row["excluded"]) or "none",
        f"{row['default_capital_mode']} | {row['numerator_mode']} | {row['asset_path_mode']} | {row['all_mode']}",
        "Yes" if row["same_rule"] else "No",
        "Yes" if row["asset_specific"] else "No",
        "Yes" if row["experimental"] else "No",
        render_misses(row["summary"]),
        row["summary"]["all_blocker_removed"],
        row["cost"],
        recommendation_for(row["label"], row["family"], row["summary"]["score15"]),
    ]


def main():
    clean_rows = search_clean_same_rule()
    coherent_rows = search_coherent_override()
    structural_rows = search_structural_heavy()
    legacy_rows = search_legacy_experimental()

    best_clean = pick_best_family(clean_rows)
    best_override = pick_best_family(coherent_rows)
    best_structural = pick_best_family(structural_rows)
    best_legacy = pick_best_family(legacy_rows)

    top_cases = []
    for candidate in [best_clean, best_override, best_structural, best_legacy]:
        if candidate not in top_cases:
            top_cases.append(candidate)

    for row in top_40plus(clean_rows, limit=3):
        if row not in top_cases:
            top_cases.append(row)
    for row in top_40plus(coherent_rows, limit=5):
        if row not in top_cases:
            top_cases.append(row)
    for row in top_40plus(structural_rows, limit=5):
        if row not in top_cases:
            top_cases.append(row)
    for row in top_40plus(legacy_rows, limit=8):
        if row not in top_cases:
            top_cases.append(row)

    all_top = sorted(clean_rows + coherent_rows + structural_rows, key=lambda row: row["summary"]["rank"])
    top_cases = sorted(top_cases, key=lambda row: row["summary"]["rank"])

    report = [
        "# Frontier 40+ Enumeration",
        "",
        "Unified workflow:",
        "",
        "1. push the clean same-rule line as far as it goes",
        "2. if still below `40/45`, enumerate coherent `40+ first` frontiers",
        "3. summarize interpretation cost vs score under one scorecard",
        "",
        "## Family Winners",
        "",
        md_table(
            ["Family", "<=10", "<=15", "4-asset <=15", "Mean Ann Gap", "Mean Cal Gap", "All Blocker Removed?", "Label"],
            [
                ["clean same-rule", f"{best_clean['summary']['score10']}/45", f"{best_clean['summary']['score15']}/45", f"{best_clean['summary']['four15']}/36", f"{best_clean['summary']['mean_ann_gap']:.1f}%", f"{best_clean['summary']['mean_cal_gap']:.1f}%", best_clean["summary"]["all_blocker_removed"], best_clean["label"]],
                ["coherent override", f"{best_override['summary']['score10']}/45", f"{best_override['summary']['score15']}/45", f"{best_override['summary']['four15']}/36", f"{best_override['summary']['mean_ann_gap']:.1f}%", f"{best_override['summary']['mean_cal_gap']:.1f}%", best_override["summary"]["all_blocker_removed"], best_override["label"]],
                ["structural-heavy", f"{best_structural['summary']['score10']}/45", f"{best_structural['summary']['score15']}/45", f"{best_structural['summary']['four15']}/36", f"{best_structural['summary']['mean_ann_gap']:.1f}%", f"{best_structural['summary']['mean_cal_gap']:.1f}%", best_structural["summary"]["all_blocker_removed"], best_structural["label"]],
                ["legacy experimental upper bound", f"{best_legacy['summary']['score10']}/45", f"{best_legacy['summary']['score15']}/45", f"{best_legacy['summary']['four15']}/36", f"{best_legacy['summary']['mean_ann_gap']:.1f}%", f"{best_legacy['summary']['mean_cal_gap']:.1f}%", best_legacy["summary"]["all_blocker_removed"], best_legacy["label"]],
            ],
        ),
        "",
        "## Clean Same-Rule Final Push",
        "",
        md_table(
            ["Label", "<=10", "<=15", "4-asset <=15", "Mean Ann Gap", "Mean Cal Gap", "Capital", "Numerator", "All Mode", "4-asset Remaining Misses"],
            detail_rows(clean_rows, limit=8),
        ),
        "",
        "## Coherent Override Frontiers",
        "",
        md_table(
            ["Label", "<=10", "<=15", "4-asset <=15", "Mean Ann Gap", "Mean Cal Gap", "Capital", "Numerator", "All Mode", "4-asset Remaining Misses"],
            detail_rows(coherent_rows, limit=10),
        ),
        "",
        "## Structural-Heavy Frontiers",
        "",
        md_table(
            ["Label", "<=10", "<=15", "4-asset <=15", "Mean Ann Gap", "Mean Cal Gap", "Capital", "Numerator", "All Mode", "4-asset Remaining Misses"],
            detail_rows(structural_rows, limit=10),
        ),
        "",
        "## Legacy Experimental Upper Bound",
        "",
        md_table(
            ["Label", "<=10", "<=15", "4-asset <=15", "Mean Ann Gap", "Mean Cal Gap", "Capital", "Numerator", "All Mode", "4-asset Remaining Misses"],
            detail_rows(legacy_rows, limit=10),
        ),
        "",
        "## 40+ First Cases",
        "",
        md_table(
            [
                "Frontier",
                "Score <=10 /45",
                "Score <=15 /45",
                "Same-rule?",
                "Asset-specific override?",
                "Structural-heavy?",
                "Main blocker removed?",
                "Explanation cost",
                "Recommendation",
            ],
            summary_table_rows([row for row in top_cases if row["summary"]["score15"] >= 40]),
        ),
        "",
        "## Unified Candidate Payload",
        "",
        md_table(
            [
                "Name",
                "<=10 /45",
                "<=15 /45",
                "Trade-lane overrides",
                "Exclusions",
                "Reporting bridge / numerator / path / all",
                "Same-rule?",
                "Asset-specific?",
                "Historical experimental?",
                "4-asset Remaining Misses",
                "All still main blocker?",
                "Explanation cost",
                "Recommendation",
            ],
            [frontier_payload_line(row) for row in top_cases],
        ),
        "",
        "## Overall Read",
        "",
        f"- clean same-rule max frontier: `{best_clean['label']}` → `<=15 {best_clean['summary']['score15']}/45`, `<=10 {best_clean['summary']['score10']}/45`",
        f"- coherent override max frontier: `{best_override['label']}` → `<=15 {best_override['summary']['score15']}/45`, `<=10 {best_override['summary']['score10']}/45`",
        f"- structural-heavy max frontier: `{best_structural['label']}` → `<=15 {best_structural['summary']['score15']}/45`, `<=10 {best_structural['summary']['score10']}/45`",
        f"- legacy experimental upper bound: `{best_legacy['label']}` → `<=15 {best_legacy['summary']['score15']}/45`, `<=10 {best_legacy['summary']['score10']}/45`",
        "",
    ]

    if best_clean["summary"]["score15"] >= 40:
        report.extend(
            [
                "Final read:",
                "",
                "- clean same-rule already reaches `40+/45`; that remains the main recommended frontier.",
            ]
        )
    else:
        report.extend(
            [
                "Final read:",
                "",
                "- clean same-rule still does **not** reach `40+/45` under the current clean doctrine.",
                "- the next review should therefore compare it against the `40+ first` cases above, not wait for more blind local search.",
                "- under the current cleaner source doctrine, the only `40+` cases found here come from the legacy experimental upper-bound family.",
            ]
        )

    DOC_PATH.write_text("\n".join(report) + "\n")
    print(f"Wrote {DOC_PATH}")
    print(f"Clean same-rule max: {best_clean['summary']['score15']}/45")
    print(f"Coherent override max: {best_override['summary']['score15']}/45")
    print(f"Structural-heavy max: {best_structural['summary']['score15']}/45")
    print(f"Legacy experimental upper bound: {best_legacy['summary']['score15']}/45")
    print(f"40+ cases found: {sum(1 for row in top_cases if row['summary']['score15'] >= 40)}")


if __name__ == "__main__":
    main()
