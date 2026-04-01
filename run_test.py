#!/usr/bin/env python3
"""Smoke test — generate tiny synthetic data, then run every pipeline step.

This verifies the full codebase works end-to-end on CPU/MPS with minimal
data (takes ~2-5 minutes).  No real downloads are needed.

Run:  python run_test.py
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

CFG_PATH = str(SCRIPT_DIR / "config_test.yaml")
from lib.utils import load_config, CLASS_NAMES_12, banner

N_IMAGES = 120  # tiny dataset (10 per class)


def make_synthetic_raptor(data_root: Path) -> None:
    """Create a fake Raptor Maps dataset with random grayscale images."""
    banner("Generating synthetic Raptor Maps dataset")
    base = data_root / "raptor_raw" / "InfraredSolarModules" / "InfraredSolarModules"
    img_dir = base / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)
    metadata = {}
    for i in range(N_IMAGES):
        cls_name = CLASS_NAMES_12[i % 12]
        fname = f"{i}.jpg"
        img = Image.fromarray(rng.randint(0, 255, (40, 24), dtype=np.uint8), mode="L")
        img.save(img_dir / fname)
        metadata[str(i)] = {
            "image_filepath": f"images/{fname}",
            "anomaly_class": cls_name,
        }

    with open(base / "module_metadata.json", "w") as f:
        json.dump(metadata, f)

    # mark as "downloaded"
    (data_root / "raptor_raw" / ".download_complete").touch()
    print(f"  ✓ {N_IMAGES} synthetic images in {img_dir}")


def make_synthetic_zenodo(data_root: Path) -> None:
    """Create a fake Zenodo UAV dataset with panels."""
    banner("Generating synthetic Zenodo UAV dataset")
    base = data_root / "zenodo_raw"

    rng = np.random.RandomState(99)
    for split, n_frames in [("train", 4), ("val", 2), ("test", 1)]:
        img_dir = base / split / "images"
        lbl_dir = base / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for j in range(n_frames):
            fname = f"frame_{split}_{j:03d}"
            img = Image.fromarray(rng.randint(0, 255, (512, 640), dtype=np.uint8), mode="L")
            img.save(img_dir / f"{fname}.jpg")
            # 3-5 fake panel boxes per frame
            n_boxes = rng.randint(3, 6)
            lines = []
            for _ in range(n_boxes):
                xc = rng.uniform(0.1, 0.9)
                yc = rng.uniform(0.1, 0.9)
                w = rng.uniform(0.05, 0.15)
                h = rng.uniform(0.03, 0.10)
                lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            (lbl_dir / f"{fname}.txt").write_text("\n".join(lines))

    (base / ".download_complete").touch()
    print(f"  ✓ Synthetic UAV frames in {base}")


def run_step(module_name: str, description: str) -> bool:
    """Import and run a pipeline step, return True on success."""
    import importlib
    banner(f"TEST: {description}")
    t0 = time.time()

    orig_argv = sys.argv
    sys.argv = [module_name, "--config", CFG_PATH]
    try:
        mod = importlib.import_module(module_name)
        mod.main()
        elapsed = time.time() - t0
        print(f"  ✓ {description} — passed ({elapsed:.1f}s)")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ✗ {description} — FAILED ({elapsed:.1f}s)")
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        sys.argv = orig_argv


def main() -> None:
    banner("SMOKE TEST — full pipeline on synthetic data")
    cfg = load_config(CFG_PATH)
    data_root = Path(cfg["paths"]["data_root"])
    output_root = Path(cfg["paths"]["output_root"])

    # clean previous test run
    for d in [data_root, output_root]:
        if d.exists():
            shutil.rmtree(d)

    # generate synthetic data
    make_synthetic_raptor(data_root)
    make_synthetic_zenodo(data_root)

    results = []

    # Step 02: prepare data (skip 01 since we made synthetic data)
    results.append(("02_prepare_data", run_step("02_prepare_data", "Prepare data")))

    # Step 03: train YOLO (stage1 + stage2 only, stage0 disabled)
    results.append(("03_train_yolo", run_step("03_train_yolo", "Train YOLO")))

    # Step 04: eval YOLO
    results.append(("04_eval_yolo", run_step("04_eval_yolo", "Evaluate YOLO")))

    # Step 05: train SupCon
    results.append(("05_train_supcon", run_step("05_train_supcon", "Train SupCon")))

    # Step 06: eval SupCon
    results.append(("06_eval_supcon", run_step("06_eval_supcon", "Evaluate SupCon")))

    # Step 07: export
    results.append(("07_export_results", run_step("07_export_results", "Export results")))

    # Summary
    banner("TEST RESULTS")
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n  All steps passed! Pipeline is working correctly.")
        # show what was produced
        print("\n  Generated outputs:")
        for p in sorted(output_root.rglob("*")):
            if p.is_file():
                size = p.stat().st_size
                unit = "KB" if size > 1024 else "B"
                val = size / 1024 if size > 1024 else size
                rel = p.relative_to(output_root)
                print(f"    {rel}  ({val:.0f} {unit})")
    else:
        print("\n  Some steps failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
