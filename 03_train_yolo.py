#!/usr/bin/env python3
"""Step 3 — Train the YOLO 3-stage pipeline.

Stage 0  Panel Detector      (yolo detection  on Zenodo UAV frames)
Stage 1  Binary Sorter        (yolo-cls: Healthy vs Defective)
Stage 2  Defect Diagnostician (yolo-cls: 11 anomaly classes)

Run:  python 03_train_yolo.py [--config config.yaml] [--stage 0|1|2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.utils import (
    banner,
    get_data_dir,
    get_output_dir,
    load_config,
    safe_num_workers,
    seed_everything,
    yolo_amp_enabled,
    yolo_device,
    yolo_stage_amp,
)


def train_stage0(cfg: dict) -> Path | None:
    """Train the panel detector on Zenodo UAV thermal frames."""
    from ultralytics import YOLO

    s0 = cfg["yolo"]["stage0"]
    if not s0.get("enabled", True):
        print("  Stage 0 disabled in config — skipping.")
        return None

    banner("YOLO Stage 0 — Panel Detector")

    data_yaml = get_data_dir(cfg, "yolo_det_uav") / "data.yaml"
    if not data_yaml.exists():
        print(f"  ⚠  {data_yaml} not found. Run 02_prepare_data.py first.")
        return None

    out_dir = get_output_dir(cfg, "yolo")
    model = YOLO(s0["model"])

    aug = s0.get("augment", {})

    results = model.train(
        data=str(data_yaml),
        epochs=s0["epochs"],
        imgsz=s0["imgsz"],
        batch=s0["batch"],
        patience=s0["patience"],
        optimizer=s0["optimizer"],
        lr0=s0["lr0"],
        lrf=s0["lrf"],
        device=yolo_device(cfg),
        workers=safe_num_workers(cfg),
        amp=yolo_stage_amp(cfg, s0),
        seed=cfg.get("seed", 42),
        deterministic=True,
        project=str(out_dir),
        name="stage0_detector",
        exist_ok=True,
        # augmentation
        mosaic=aug.get("mosaic", 1.0),
        flipud=aug.get("flipud", 0.0),
        fliplr=aug.get("fliplr", 0.5),
        degrees=aug.get("degrees", 0.0),
        hsv_h=aug.get("hsv_h", 0.015),
        hsv_s=aug.get("hsv_s", 0.7),
        hsv_v=aug.get("hsv_v", 0.4),
    )

    best_pt = out_dir / "stage0_detector" / "weights" / "best.pt"
    print(f"  ✓ Stage 0 best weights: {best_pt}")
    return best_pt


def train_stage1(cfg: dict) -> Path | None:
    """Train the binary sorter (Healthy / Defective)."""
    from ultralytics import YOLO

    s1 = cfg["yolo"]["stage1"]
    if not s1.get("enabled", True):
        print("  Stage 1 disabled in config — skipping.")
        return None

    banner("YOLO Stage 1 — Binary Sorter (Healthy / Defective)")

    data_dir = get_data_dir(cfg, "yolo_cls_binary")
    if not (data_dir / "train").exists():
        print(f"  ⚠  {data_dir}/train not found. Run 02_prepare_data.py first.")
        return None

    out_dir = get_output_dir(cfg, "yolo")
    model = YOLO(s1["model"])

    aug = s1.get("augment", {})

    results = model.train(
        data=str(data_dir),
        epochs=s1["epochs"],
        imgsz=s1["imgsz"],
        batch=s1["batch"],
        patience=s1["patience"],
        optimizer=s1["optimizer"],
        lr0=s1["lr0"],
        weight_decay=s1.get("weight_decay", 5e-4),
        device=yolo_device(cfg),
        workers=safe_num_workers(cfg),
        amp=yolo_amp_enabled(cfg),
        seed=cfg.get("seed", 42),
        deterministic=True,
        pretrained=True,
        project=str(out_dir),
        name="stage1_sorter",
        exist_ok=True,
        # augmentation
        auto_augment=aug.get("auto_augment", "randaugment"),
        fliplr=aug.get("fliplr", 0.5),
        flipud=aug.get("flipud", 0.0),
        degrees=aug.get("degrees", 0.0),
        hsv_h=aug.get("hsv_h", 0.015),
        hsv_s=aug.get("hsv_s", 0.7),
        hsv_v=aug.get("hsv_v", 0.4),
        mosaic=aug.get("mosaic", 1.0),
        mixup=aug.get("mixup", 0.0),
        erasing=aug.get("erasing", 0.0),
        scale=aug.get("scale", 0.5),
        translate=aug.get("translate", 0.1),
    )

    best_pt = out_dir / "stage1_sorter" / "weights" / "best.pt"
    print(f"  ✓ Stage 1 best weights: {best_pt}")
    return best_pt


def train_stage2(cfg: dict) -> Path | None:
    """Train the 11-class defect diagnostician."""
    from ultralytics import YOLO

    s2 = cfg["yolo"]["stage2"]
    if not s2.get("enabled", True):
        print("  Stage 2 disabled in config — skipping.")
        return None

    banner("YOLO Stage 2 — Defect Diagnostician (11 classes)")

    data_dir = get_data_dir(cfg, "yolo_cls_defects")
    if not (data_dir / "train").exists():
        print(f"  ⚠  {data_dir}/train not found. Run 02_prepare_data.py first.")
        return None

    out_dir = get_output_dir(cfg, "yolo")
    model = YOLO(s2["model"])

    aug = s2.get("augment", {})

    results = model.train(
        data=str(data_dir),
        epochs=s2["epochs"],
        imgsz=s2["imgsz"],
        batch=s2["batch"],
        patience=s2["patience"],
        optimizer=s2["optimizer"],
        lr0=s2["lr0"],
        weight_decay=s2.get("weight_decay", 5e-4),
        device=yolo_device(cfg),
        workers=safe_num_workers(cfg),
        amp=yolo_amp_enabled(cfg),
        seed=cfg.get("seed", 42),
        deterministic=True,
        pretrained=True,
        project=str(out_dir),
        name="stage2_diagnostician",
        exist_ok=True,
        auto_augment=aug.get("auto_augment", "randaugment"),
        fliplr=aug.get("fliplr", 0.5),
        flipud=aug.get("flipud", 0.0),
        degrees=aug.get("degrees", 0.0),
        hsv_h=aug.get("hsv_h", 0.015),
        hsv_s=aug.get("hsv_s", 0.7),
        hsv_v=aug.get("hsv_v", 0.4),
        mosaic=aug.get("mosaic", 1.0),
        mixup=aug.get("mixup", 0.0),
        erasing=aug.get("erasing", 0.0),
        scale=aug.get("scale", 0.5),
        translate=aug.get("translate", 0.1),
    )

    best_pt = out_dir / "stage2_diagnostician" / "weights" / "best.pt"
    print(f"  ✓ Stage 2 best weights: {best_pt}")
    return best_pt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument(
        "--stage", type=int, default=None,
        help="Train only this stage (0, 1, or 2). Default: train all enabled.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))

    stages = {0: train_stage0, 1: train_stage1, 2: train_stage2}

    if args.stage is not None:
        if args.stage not in stages:
            print(f"Invalid stage {args.stage}. Choose 0, 1, or 2.")
            sys.exit(1)
        stages[args.stage](cfg)
    else:
        for fn in stages.values():
            fn(cfg)

    banner("YOLO training complete")


if __name__ == "__main__":
    main()
