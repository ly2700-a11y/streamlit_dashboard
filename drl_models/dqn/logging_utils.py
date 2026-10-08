"""Helpers for per-run RL log directories and artifacts."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from drl_models.dqn.spec import run_log_dir


def make_run_id(bp: float | None = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    if bp is not None:
        prefix = f"bp{int(bp * 10000)}_"
        return f"{prefix}{timestamp}"
    return timestamp


@dataclass
class RunLogger:
    algorithm: str
    ticker: str
    round_num: int
    run_id: str = field(default_factory=make_run_id)
    base_dir: Path | None = None

    def __post_init__(self):
        self.dir = self.base_dir or run_log_dir(self.algorithm, self.ticker, self.round_num, self.run_id)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "train.log"

    def log(self, message: str):
        print(message)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(message + "\n")

    def write_json(self, filename: str, payload: dict):
        path = self.dir / filename
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        return path

    def write_csv(self, filename: str, rows: list[dict]):
        path = self.dir / filename
        if not rows:
            with path.open("w", encoding="utf-8", newline="") as fh:
                fh.write("")
            return path
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return path
