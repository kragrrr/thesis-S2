#!/usr/bin/env python3
"""Master script — run the full pipeline end-to-end.

  python run_all.py                        # everything
  python run_all.py --skip supcon          # YOLO only
  python run_all.py --skip yolo            # SupCon only
  python run_all.py --only eval            # evaluation + export only
  python run_all.py --from-step 3          # resume from step 3
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.utils import load_config, banner, timestamp


STEPS = [
    ("01", "01_download_data",  "Download datasets",         {"yolo", "supcon", "data"}),
    ("02", "02_prepare_data",   "Prepare datasets",          {"yolo", "supcon", "data"}),
    ("03", "03_train_yolo",     "Train YOLO pipeline",       {"yolo", "train"}),
    ("04", "04_eval_yolo",      "Evaluate YOLO pipeline",    {"yolo", "eval"}),
    ("05", "05_train_supcon",   "Train SupCon encoder",      {"supcon", "train"}),
    ("06", "06_eval_supcon",    "Evaluate SupCon encoder",   {"supcon", "eval"}),
    ("07", "07_export_results", "Export results",            {"export"}),
]


def run_step(module_name: str, cfg_path: str) -> None:
    mod = importlib.import_module(module_name)
    orig_argv = sys.argv
    sys.argv = [module_name, "--config", cfg_path]
    try:
        mod.main()
    finally:
        sys.argv = orig_argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Skip steps matching these tags (yolo, supcon, data, train, eval, export).")
    parser.add_argument("--only", nargs="*", default=[],
                        help="Run only steps matching these tags.")
    parser.add_argument("--from-step", type=int, default=1,
                        help="Start from this step number (1-7).")
    args = parser.parse_args()

    skip_tags = set(args.skip)
    only_tags = set(args.only)

    banner(f"PV Anomaly Detection — Full Pipeline  ({timestamp()})")

    t0 = time.time()
    for step_num, module_name, description, tags in STEPS:
        num = int(step_num)
        if num < args.from_step:
            print(f"  [{step_num}] {description} — skipped (before --from-step)")
            continue
        if skip_tags and tags & skip_tags:
            print(f"  [{step_num}] {description} — skipped (--skip {skip_tags & tags})")
            continue
        if only_tags and not (tags & only_tags):
            print(f"  [{step_num}] {description} — skipped (not in --only)")
            continue

        banner(f"[{step_num}] {description}")
        step_t0 = time.time()
        try:
            run_step(module_name, args.config)
        except Exception as e:
            print(f"\n  ✗ Step {step_num} failed: {e}")
            print("    Fix the issue and re-run with --from-step", num)
            sys.exit(1)
        elapsed = time.time() - step_t0
        print(f"  [{step_num}] done in {elapsed/60:.1f} min")

    total = time.time() - t0
    banner(f"Pipeline complete — {total/60:.1f} min total")


if __name__ == "__main__":
    main()
