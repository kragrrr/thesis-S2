#!/usr/bin/env python3
"""Step 4 — Evaluate YOLO stages and run the full 3-stage pipeline on UAV images.

Generates confusion matrices, per-class metrics, and annotated visualisations.

Run:  python 04_eval_yolo.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.utils import (
    CLASS_NAMES_12,
    DEFECT_CLASSES,
    SEVERE_DEFECTS,
    MILD_DEFECTS,
    load_config,
    get_data_dir,
    get_output_dir,
    banner,
)


# ── helpers ─────────────────────────────────────────────────

def _load_yolo(weights_path: Path):
    from ultralytics import YOLO
    if not weights_path.exists():
        print(f"  ⚠  Weights not found: {weights_path}")
        return None
    return YOLO(str(weights_path))


def _top1(result: Any) -> tuple[str, float]:
    probs = result.probs
    idx = int(probs.top1)
    return str(result.names[idx]), float(probs.top1conf)


def _save_confusion_matrix(
    y_true, y_pred, labels, title, save_path, cfg,
):
    pcfg = cfg.get("plots", {})
    fig, ax = plt.subplots(figsize=tuple(pcfg.get("figsize", [10, 8])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels,
                cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(save_path, dpi=pcfg.get("dpi", 150))
    plt.close(fig)
    print(f"  ✓ {save_path.name}")


# ── per-stage validation ────────────────────────────────────

def eval_stage(cfg: dict, stage: int) -> None:
    name_map = {0: "stage0_detector", 1: "stage1_sorter", 2: "stage2_diagnostician"}
    data_map = {0: "yolo_det_uav", 1: "yolo_cls_binary", 2: "yolo_cls_defects"}
    label_map = {
        1: ["Defective", "Healthy"],
        2: DEFECT_CLASSES,
    }

    stage_name = name_map[stage]
    out_dir = get_output_dir(cfg, "yolo")
    weights = out_dir / stage_name / "weights" / "best.pt"
    model = _load_yolo(weights)
    if model is None:
        return

    banner(f"Evaluating YOLO {stage_name}")

    eval_dir = get_output_dir(cfg, "yolo", "evaluation")

    if stage == 0:
        data_yaml = get_data_dir(cfg, data_map[stage]) / "data.yaml"
        metrics = model.val(data=str(data_yaml), device=cfg.get("device", "0"))
        summary = {
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }
        (eval_dir / "stage0_metrics.json").write_text(json.dumps(summary, indent=2))
        print(f"  mAP50={summary['mAP50']:.4f}  mAP50-95={summary['mAP50-95']:.4f}")
        return

    data_dir = get_data_dir(cfg, data_map[stage])
    val_dir = data_dir / "val"
    if not val_dir.exists():
        print(f"  ⚠  Val dir not found: {val_dir}")
        return

    labels = label_map[stage]
    y_true, y_pred, confs = [], [], []

    for cls_dir in sorted(val_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        true_label = cls_dir.name
        for img_path in cls_dir.glob("*.jpg"):
            results = model.predict(source=str(img_path), verbose=False)
            pred_label, conf = _top1(results[0])
            y_true.append(true_label)
            y_pred.append(pred_label)
            confs.append(conf)

    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    report_path = eval_dir / f"{stage_name}_classification_report.txt"
    report_path.write_text(report)
    print(report)

    _save_confusion_matrix(
        y_true, y_pred, labels,
        title=f"{stage_name} Confusion Matrix",
        save_path=eval_dir / f"{stage_name}_confusion_matrix.png",
        cfg=cfg,
    )

    report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    accuracy = report_dict.get("accuracy", 0)
    print(f"  Top-1 Accuracy: {accuracy:.4f}")
    (eval_dir / f"{stage_name}_metrics.json").write_text(
        json.dumps({"accuracy": accuracy, "report": report_dict}, indent=2)
    )


# ── full 3-stage pipeline on UAV crops ──────────────────────

def run_pipeline(cfg: dict) -> None:
    banner("Running full 3-stage pipeline on UAV panel crops")

    out_dir = get_output_dir(cfg, "yolo")
    eval_dir = get_output_dir(cfg, "yolo", "evaluation")

    s1_weights = out_dir / "stage1_sorter" / "weights" / "best.pt"
    s2_weights = out_dir / "stage2_diagnostician" / "weights" / "best.pt"

    s1_model = _load_yolo(s1_weights)
    s2_model = _load_yolo(s2_weights)
    if s1_model is None or s2_model is None:
        print("  ⚠  Need both Stage 1 and Stage 2 weights for pipeline.")
        return

    crops_dir = get_data_dir(cfg, "uav_crops")
    crop_files = sorted(crops_dir.rglob("*.jpg"))
    if not crop_files:
        print(f"  ⚠  No crops found in {crops_dir}. Run 02_prepare_data.py first.")
        return

    ar_thresh = cfg["data_prep"]["zenodo_uav"]["slice_ar_threshold"]
    results_rows = []

    for crop_path in crop_files:
        img = Image.open(crop_path).convert("RGB")
        w, h = img.size
        chunks = [img]
        if h > 0 and w / h > ar_thresh:
            n = max(1, round(w / h))
            cw = w // n
            chunks = [img.crop((i * cw, 0, w if i == n - 1 else (i + 1) * cw, h)) for i in range(n)]

        s1_res = s1_model.predict(source=chunks, verbose=False)

        defective_chunks = []
        for chunk, r1 in zip(chunks, s1_res):
            label, conf = _top1(r1)
            norm = label.strip().lower().replace("_", "-")
            if norm not in {"healthy", "no-anomaly"}:
                defective_chunks.append(chunk)

        if not defective_chunks:
            results_rows.append({
                "crop": crop_path.name, "stage1": "Healthy", "stage2": "",
                "s1_conf": float(s1_res[0].probs.top1conf), "s2_conf": "",
            })
            continue

        s2_res = s2_model.predict(source=defective_chunks, verbose=False)
        best_label, best_conf = max(
            [_top1(r) for r in s2_res], key=lambda x: x[1],
        )
        results_rows.append({
            "crop": crop_path.name, "stage1": "Defective", "stage2": best_label,
            "s1_conf": float(s1_res[0].probs.top1conf), "s2_conf": best_conf,
        })

    csv_path = eval_dir / "pipeline_predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["crop", "stage1", "stage2", "s1_conf", "s2_conf"])
        writer.writeheader()
        writer.writerows(results_rows)

    healthy = sum(1 for r in results_rows if r["stage1"] == "Healthy")
    defective = len(results_rows) - healthy
    print(f"  Panels: {len(results_rows)}  Healthy: {healthy}  Defective: {defective}")

    if defective:
        defect_counts = Counter(r["stage2"] for r in results_rows if r["stage2"])
        pcfg = cfg.get("plots", {})
        fig, ax = plt.subplots(figsize=tuple(pcfg.get("figsize", [10, 8])))
        names = list(defect_counts.keys())
        vals = [defect_counts[n] for n in names]
        ax.barh(names, vals, color="coral")
        ax.set_xlabel("Count")
        ax.set_title("Predicted Defect Distribution (UAV)")
        plt.tight_layout()
        fig.savefig(eval_dir / "pipeline_defect_distribution.png", dpi=pcfg.get("dpi", 150))
        plt.close(fig)

    print(f"  ✓ Pipeline results: {csv_path}")


# ── main ────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--stage", type=int, default=None,
                        help="Evaluate only this stage (0, 1, or 2).")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Skip the full pipeline run on UAV crops.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.stage is not None:
        eval_stage(cfg, args.stage)
    else:
        for s in [0, 1, 2]:
            eval_stage(cfg, s)

    if not args.skip_pipeline:
        run_pipeline(cfg)

    banner("YOLO evaluation complete")


if __name__ == "__main__":
    main()
