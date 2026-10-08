#!/usr/bin/env python3
"""
Exhibit 5 Orchestration Script — loops over all (asset, round, BP, seed)
combinations and launches training in controlled batches.

Usage:
  python train_exhibit5.py --phase 1          # TC_BP_PHASE1: 4 BPs × 4 assets × 2 rounds × 10 seeds = 320 jobs
  python train_exhibit5.py --phase 2          # TC_BP_PHASE2: 5 BPs × 4 assets × 2 rounds × 10 seeds = 400 jobs
  python train_exhibit5.py --dry-run          # Print job count without launching
  python train_exhibit5.py --resume           # Only launch pending/failed jobs
  python train_exhibit5.py --status           # Print summary from training_jobs.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import ASSET_CLASSES
from drl_models.dqn.spec import (
    LOCKED_SEEDS,
    MODEL_ROOT,
    TC_BP_PHASE1,
    TC_BP_PHASE2,
    asset_slug,
    round_name,
)
from drl_shared.spec import RETRAIN_ROUNDS

TRAINING_JOBS_PATH = REPO_ROOT / "ensemble_table2_bp" / "training_jobs.json"
LOG_DIR = REPO_ROOT / "drl" / "dqn" / "logs"
TRAIN_SCRIPT = REPO_ROOT / "drl" / "dqn" / "train" / "train_dqn_walkforward.py"

BATCH_SIZE = 4
SLEEP_BETWEEN_BATCHES = 60


def _bp_to_bps(bp: float) -> int:
    return int(round(bp * 10000))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _job_key(asset: str, round_num: int, bp: float, seed: int) -> str:
    bps = _bp_to_bps(bp)
    return f"{asset_slug(asset)}_r{round_num}_bp{bps}_s{seed}"


def _log_path(asset: str, round_num: int, bp: float, seed: int) -> Path:
    bps = _bp_to_bps(bp)
    asset_s = asset_slug(asset)
    return LOG_DIR / f"train_bp{bps}_{asset_s}_r{round_num}_s{seed}.log"


def _model_dir(asset: str, round_num: int, bp: float, seed: int) -> Path:
    bps = _bp_to_bps(bp)
    asset_s = asset_slug(asset)
    rn = round_name(round_num)
    return MODEL_ROOT / asset_s / rn / f"bp{bps}_{asset_s}_r{round_num}_s{seed}"


def _checkpoint_exists(asset: str, round_num: int, bp: float, seed: int) -> bool:
    bps = _bp_to_bps(bp)
    asset_s = asset_slug(asset)
    rn = round_name(round_num)
    parent = MODEL_ROOT / asset_s / rn
    if not parent.exists():
        return False
    bp_prefix = f"bp{bps}_"
    seed_suffix = f"_s{seed}"
    for d in parent.iterdir():
        if d.is_dir() and d.name.startswith(bp_prefix) and d.name.endswith(seed_suffix):
            if (d / "checkpoint.pt").exists():
                return True
    return False


def _log_has_error(log_file: Path) -> bool:
    if not log_file.exists():
        return False
    content = log_file.read_text(errors="ignore")
    return "ERROR" in content or "Traceback" in content


def load_jobs() -> dict:
    if TRAINING_JOBS_PATH.exists():
        with open(TRAINING_JOBS_PATH, "r") as f:
            return json.load(f)
    return {}


def save_jobs(jobs: dict) -> None:
    _ensure_dir(TRAINING_JOBS_PATH.parent)
    with open(TRAINING_JOBS_PATH, "w") as f:
        json.dump(jobs, f, indent=2)


def make_job(asset: str, round_num: int, bp: float, seed: int) -> dict:
    bps = _bp_to_bps(bp)
    return {
        "asset": asset,
        "round": round_num,
        "bp": bp,
        "bp_bps": bps,
        "seed": seed,
        "status": "pending",
        "pid": None,
        "log": str(_log_path(asset, round_num, bp, seed)),
        "dir": str(_model_dir(asset, round_num, bp, seed)),
        "started_at": None,
        "finished_at": None,
    }


def launch_batch(jobs_batch: list[dict]) -> list[tuple[dict, subprocess.Popen, Path]]:
    launched = []
    for job in jobs_batch:
        asset = job["asset"]
        round_num = job["round"]
        bp = job["bp"]
        seed = job["seed"]
        log_file = Path(job["log"])
        _ensure_dir(log_file.parent)

        cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "--tc-bp", str(bp),
            "--asset", asset,
            "--round", str(round_num),
            "--seed", str(seed),
            "--device", "cuda",
        ]

        log_fh = open(log_file, "w")
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
        )

        job["status"] = "running"
        job["pid"] = proc.pid
        job["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        launched.append((job, proc, log_fh))

    return launched


def wait_batch(launched: list[tuple[dict, subprocess.Popen, Path]]) -> None:
    for job, proc, log_fh in launched:
        proc.wait()
        log_fh.close()

        finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        job["finished_at"] = finished_at

        if proc.returncode != 0:
            job["status"] = "failed"
            job["error"] = f"exit_code={proc.returncode}"
        elif _checkpoint_exists(job["asset"], job["round"], job["bp"], job["seed"]):
            job["status"] = "complete"
        elif _log_has_error(Path(job["log"])):
            job["status"] = "failed"
        else:
            job["status"] = "complete"


def generate_job_list(phase: int) -> list[tuple]:
    if phase == 1:
        bp_levels = TC_BP_PHASE1
    elif phase == 2:
        bp_levels = TC_BP_PHASE2
    else:
        raise ValueError(f"Unknown phase: {phase}")

    jobs = []
    for bp in bp_levels:
        for asset in ASSET_CLASSES:
            for round_num in RETRAIN_ROUNDS:
                for seed in LOCKED_SEEDS:
                    jobs.append((asset, round_num, bp, seed))
    return jobs


def check_existing_status(jobs: dict) -> dict:
    for key, job in jobs.items():
        if job.get("status") in ("complete", "failed"):
            continue
        if job.get("status") == "running" and job.get("pid") is not None:
            pid = job["pid"]
            try:
                os.kill(pid, 0)
            except OSError:
                if _checkpoint_exists(job["asset"], job["round"], job["bp"], job["seed"]):
                    job["status"] = "complete"
                elif _log_has_error(Path(job["log"])):
                    job["status"] = "failed"
                else:
                    job["status"] = "failed"
            continue
        if _checkpoint_exists(job["asset"], job["round"], job["bp"], job["seed"]):
            job["status"] = "complete"
            continue
        if _log_has_error(Path(job["log"])):
            job["status"] = "failed"
    return jobs


def print_status_summary(jobs: dict) -> None:
    if not jobs:
        print("No jobs found in training_jobs.json")
        return
    counts = {"pending": 0, "running": 0, "complete": 0, "failed": 0}
    for job in jobs.values():
        status = job.get("status", "unknown")
        counts[status] = counts.get(status, 0)
    total = len(jobs)
    parts = [f"{counts.get(s, 0)} {s}" for s in ("complete", "running", "failed", "pending")]
    print(f"{total} jobs: {', '.join(parts)}")


def main():
    parser = argparse.ArgumentParser(description="Exhibit 5 orchestration script")
    parser.add_argument("--phase", type=int, choices=[1, 2], help="Phase to run (1 or 2)")
    parser.add_argument("--dry-run", action="store_true", help="Print job count without launching")
    parser.add_argument("--resume", action="store_true", help="Only launch pending/failed jobs")
    parser.add_argument("--status", action="store_true", help="Print status summary from JSON")
    args = parser.parse_args()

    if args.status:
        jobs = load_jobs()
        check_existing_status(jobs)
        print_status_summary(jobs)
        return

    if args.phase is None:
        parser.print_help()
        return

    _ensure_dir(LOG_DIR)

    job_list = generate_job_list(args.phase)
    total_jobs = len(job_list)
    print(f"Phase {args.phase}: {total_jobs} jobs to process")

    existing_jobs = load_jobs()

    if args.resume:
        print("Resume mode: checking existing job statuses...")
        existing_jobs = check_existing_status(existing_jobs)
        save_jobs(existing_jobs)

        to_launch = []
        for asset, round_num, bp, seed in job_list:
            key = _job_key(asset, round_num, bp, seed)
            if key in existing_jobs:
                status = existing_jobs[key].get("status", "pending")
                if status in ("pending", "failed"):
                    to_launch.append((asset, round_num, bp, seed))
            else:
                to_launch.append((asset, round_num, bp, seed))
        print(f"Resume: {len(to_launch)} jobs to launch (pending/failed or new)")
        job_list = to_launch

    if args.dry_run:
        print("\nJob details:")
        for i, (asset, round_num, bp, seed) in enumerate(job_list[:5]):
            bps = _bp_to_bps(bp)
            log_f = _log_path(asset, round_num, bp, seed)
            print(f"  [{i+1}] {asset} r{round_num} bp{bps} seed{seed} -> {log_f}")
        if len(job_list) > 5:
            print(f"  ... and {len(job_list) - 5} more")
        print(f"\nTotal: {len(job_list)} jobs")
        return

    print(f"Launching in batches of {BATCH_SIZE} (sleep {SLEEP_BETWEEN_BATCHES}s between batches)...")

    for batch_start in range(0, len(job_list), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(job_list))
        batch_keys = job_list[batch_start:batch_end]

        jobs_batch = []
        skipped = 0
        for asset, round_num, bp, seed in batch_keys:
            key = _job_key(asset, round_num, bp, seed)
            if _checkpoint_exists(asset, round_num, bp, seed):
                if key in existing_jobs:
                    existing_jobs[key]["status"] = "complete"
                skipped += 1
                continue
            if key in existing_jobs and existing_jobs[key].get("status") == "complete":
                continue
            if key in existing_jobs:
                job_data = existing_jobs[key]
            else:
                job_data = make_job(asset, round_num, bp, seed)
                existing_jobs[key] = job_data
            jobs_batch.append(job_data)

        if skipped:
            print(f"[batch {batch_start}-{batch_end}] skipping {skipped} already-complete jobs")

        if not jobs_batch:
            print(f"[batch {batch_start}-{batch_end}] all complete, skipping")
            continue

        print(f"[batch {batch_start}-{batch_end}] launching {len(jobs_batch)} jobs...")
        launched = launch_batch(jobs_batch)
        wait_batch(launched)

        for job in jobs_batch:
            print(f"  {job['asset']}_r{job['round']}_bp{job['bp_bps']}_s{job['seed']}: {job['status']}")

        save_jobs(existing_jobs)
        print(f"  batch complete, sleeping {SLEEP_BETWEEN_BATCHES}s...")
        if batch_end < len(job_list):
            time.sleep(SLEEP_BETWEEN_BATCHES)

    save_jobs(existing_jobs)
    print(f"\nDone. {len(job_list)} jobs processed.")
    print(f"Job status saved to: {TRAINING_JOBS_PATH}")


if __name__ == "__main__":
    main()