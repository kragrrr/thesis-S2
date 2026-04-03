#!/usr/bin/env python3
"""Step 4 — Evaluate YOLO stages and run the full 3-stage pipeline on UAV images.

Generates confusion matrices, per-class metrics, and annotated visualisations.

Run:  python 04_eval_yolo.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import random
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import classification_report, confusion_matrix

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.utils import (
    CLASS_NAMES_12,
    DEFECT_CLASSES,
    MILD_DEFECTS,
    SEVERE_DEFECTS,
    banner,
    get_data_dir,
    get_output_dir,
    load_config,
    yolo_device,
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


def _resolve_overlay_font(size: int) -> ImageFont.ImageFont:
    """TTF first (Windows Fonts dir, then DejaVu); avoid load_default for sizing bugs."""
    candidates = [
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "arial.ttf",
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "calibri.ttf",
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    ]
    for p in candidates:
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size=size)
            except OSError:
                continue
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _preview_upscale_for_display(img: Image.Image) -> Image.Image:
    """Upscale tiny or extreme-aspect crops so JPG previews + captions are legible in the gallery."""
    out = img.convert("RGB").copy()
    w, h = out.size
    if w < 1 or h < 1:
        return out

    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS  # type: ignore[attr-defined]

    short = min(w, h)
    long_ = max(w, h)
    ar = long_ / max(short, 1)

    scale = 1.0
    # Tiny IR chips (e.g. 64 px)
    if short < 160:
        scale = max(scale, min(6.0, 160.0 / short))
    # Panoramic strips (many panels in one crop): avoid paper-thin thumbnails
    if ar > 5.0 and short < 200:
        scale = max(scale, min(4.0, 200.0 / short))

    if scale > 1.01:
        out = out.resize((max(1, int(w * scale)), max(1, int(h * scale))), resample)
        w, h = out.size

    # Cap megapixel so files stay reasonable
    max_side = 2048
    if max(w, h) > max_side:
        r = max_side / max(w, h)
        out = out.resize((max(1, int(w * r)), max(1, int(h * r))), resample)

    return out


def _pipeline_row_to_overlay(img: Image.Image, row: dict) -> Image.Image:
    """Return a copy of ``img`` with a small fixed-height caption band at the bottom."""
    out = _preview_upscale_for_display(img)
    draw = ImageDraw.Draw(out)
    w, h = out.size

    def _cf(x: Any) -> str:
        if x == "" or x is None:
            return "—"
        return f"{float(x):.3f}"

    s2 = (row.get("stage2") or "—") or "—"
    if len(s2) > 36:
        s2 = s2[:33] + "…"
    line1 = f"S1: {row['stage1']}  ({_cf(row.get('s1_conf'))})"
    line2 = f"S2: {s2}  ({_cf(row.get('s2_conf'))})"

    # Fixed band height (never derived from textbbox — avoids huge overlays with default font).
    band_h = min(96, max(26, int(h * 0.14)), max(8, h - 4))
    bar_top = max(0, h - band_h)

    fs = max(11, min(14, band_h // 5))
    font = _resolve_overlay_font(fs)
    v_pad = max(3, (band_h - 2 * fs - 4) // 2)
    y1 = bar_top + v_pad
    y2 = y1 + fs + 2

    draw.rectangle([0, bar_top, w, h], fill=(22, 22, 26))
    draw.text((6, y1), line1, font=font, fill=(248, 248, 248))
    draw.text((6, y2), line2, font=font, fill=(248, 248, 248))
    return out


def _preview_indices(n_total: int, max_save: int, seed: int) -> set[int]:
    if n_total == 0:
        return set()
    if max_save <= 0 or max_save >= n_total:
        return set(range(n_total))
    rng = random.Random(seed)
    return set(rng.sample(range(n_total), max_save))


def _write_pipeline_gallery(previews_root: Path) -> bool:
    """Write ``gallery.html`` under ``previews_root``. Returns True if file was written."""
    jpgs = sorted(previews_root.rglob("*.jpg"))
    if not jpgs:
        return False

    now = datetime.now(timezone.utc)
    gen_iso = now.strftime("%Y-%m-%d %H:%M:%S UTC")
    build_id = now.strftime("%Y%m%d%H%M%S")

    items: list[tuple[str, str]] = []
    for p in jpgs:
        rel = p.relative_to(previews_root)
        items.append((rel.as_posix(), html.escape(rel.as_posix())))

    body_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta http-equiv='Cache-Control' content='no-cache, no-store, must-revalidate'>",
        "<meta http-equiv='Pragma' content='no-cache'>",
        f"<title>Pipeline previews ({gen_iso})</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee;}",
        "h1{font-size:1.2rem;} .sub{color:#888;font-size:0.9rem;margin-bottom:1rem;}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px;}",
        "figure{margin:0;background:#222;border-radius:8px;overflow:hidden;display:flex;flex-direction:column;}",
        ".imgwrap{background:#0a0a0a;min-height:200px;display:flex;align-items:center;justify-content:center;padding:10px;box-sizing:border-box;}",
        ".imgwrap img{max-width:100%;max-height:min(420px,55vh);width:auto;height:auto;object-fit:contain;vertical-align:middle;image-rendering:auto;}",
        "figcaption{padding:8px;font-size:11px;word-break:break-all;color:#aaa;}",
        "footer{margin-top:2rem;padding-top:1rem;border-top:1px solid #333;color:#666;font-size:12px;}",
        "</style>",
        "</head>",
        "<body>",
        f"<!-- gallery template v4 generated={gen_iso} -->",
        "<h1>Panel pipeline previews</h1>",
        "<p class='sub'>Wide UAV panel strips use a letterboxed preview so you see the full crop, not a zoomed strip. "
        "Stage 1 = binary; Stage 2 = defect class. Re-run <code>python 04_eval_yolo.py</code> to regenerate JPGs after pull.</p>",
        "<div class='grid'>",
    ]
    for rel_posix, esc in items:
        src = html.escape(rel_posix, quote=True)
        body_parts.append(
            f"<figure><div class='imgwrap'><img src='{src}' loading='lazy' alt=''></div>"
            f"<figcaption>{esc}</figcaption></figure>"
        )
    body_parts.extend([
        "</div>",
        f"<footer>Generated {html.escape(gen_iso)} — {len(items)} images — build {html.escape(build_id)}</footer>",
        "</body></html>",
    ])
    gallery_path = previews_root / "gallery.html"
    gallery_path.write_text("\n".join(body_parts), encoding="utf-8")
    return True


def refresh_pipeline_gallery(cfg: dict) -> None:
    """Rewrite ``gallery.html`` if preview JPEGs exist (updates template without re-running inference)."""
    pe = cfg.get("yolo", {}).get("pipeline_eval", {})
    if not pe.get("write_gallery_html", True):
        return
    preview_root = get_output_dir(cfg, "yolo", "evaluation") / "pipeline_previews"
    if not preview_root.is_dir():
        return
    if not list(preview_root.rglob("*.jpg")):
        return
    if _write_pipeline_gallery(preview_root):
        print(f"  ✓ Refreshed gallery: {preview_root / 'gallery.html'}")


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
        metrics = model.val(data=str(data_yaml), device=yolo_device(cfg))
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
    dev = yolo_device(cfg)

    for cls_dir in sorted(val_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        true_label = cls_dir.name
        for img_path in cls_dir.glob("*.jpg"):
            results = model.predict(source=str(img_path), verbose=False, device=dev)
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

def run_pipeline(cfg: dict, preview_max_override: int | None = None) -> None:
    banner("Running full 3-stage pipeline on UAV panel crops")

    out_dir = get_output_dir(cfg, "yolo")
    eval_dir = get_output_dir(cfg, "yolo", "evaluation")
    dev = yolo_device(cfg)

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

    pe = cfg.get("yolo", {}).get("pipeline_eval", {})
    save_previews = bool(pe.get("save_preview_images", True))
    max_preview = preview_max_override if preview_max_override is not None else int(
        pe.get("preview_max_images", 600)
    )
    preview_root = eval_dir / "pipeline_previews"
    save_idx: set[int] = set()
    if save_previews:
        if preview_root.exists():
            shutil.rmtree(preview_root)
        preview_root.mkdir(parents=True, exist_ok=True)
        save_idx = _preview_indices(len(crop_files), max_preview, cfg.get("seed", 42))

    ar_thresh = cfg["data_prep"]["zenodo_uav"]["slice_ar_threshold"]
    results_rows = []

    for idx, crop_path in enumerate(crop_files):
        img = Image.open(crop_path).convert("RGB")
        w, h = img.size
        chunks = [img]
        if h > 0 and w / h > ar_thresh:
            n = max(1, round(w / h))
            cw = w // n
            chunks = [img.crop((i * cw, 0, w if i == n - 1 else (i + 1) * cw, h)) for i in range(n)]

        s1_res = s1_model.predict(source=chunks, verbose=False, device=dev)

        defective_chunks = []
        for chunk, r1 in zip(chunks, s1_res):
            label, conf = _top1(r1)
            norm = label.strip().lower().replace("_", "-")
            if norm not in {"healthy", "no-anomaly"}:
                defective_chunks.append(chunk)

        if not defective_chunks:
            row = {
                "crop": crop_path.name, "stage1": "Healthy", "stage2": "",
                "s1_conf": float(s1_res[0].probs.top1conf), "s2_conf": "",
            }
            results_rows.append(row)
        else:
            s2_res = s2_model.predict(source=defective_chunks, verbose=False, device=dev)
            best_label, best_conf = max(
                [_top1(r) for r in s2_res], key=lambda x: x[1],
            )
            row = {
                "crop": crop_path.name, "stage1": "Defective", "stage2": best_label,
                "s1_conf": float(s1_res[0].probs.top1conf), "s2_conf": float(best_conf),
            }
            results_rows.append(row)

        if save_previews and idx in save_idx:
            try:
                rel = crop_path.relative_to(crops_dir)
            except ValueError:
                rel = Path(crop_path.name)
            out_sub = preview_root / rel.parent
            out_sub.mkdir(parents=True, exist_ok=True)
            out_name = f"{crop_path.stem}_pipeline{crop_path.suffix}"
            overlay = _pipeline_row_to_overlay(img, row)
            overlay.save(out_sub / out_name, quality=92)

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

    if save_previews:
        n_saved = len(save_idx)
        print(f"  ✓ Preview images: {preview_root}  ({n_saved} files)")

    print(f"  ✓ Pipeline results: {csv_path}")


# ── main ────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--stage", type=int, default=None,
                        help="Evaluate only this stage (0, 1, or 2).")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Skip the full pipeline run on UAV crops.")
    parser.add_argument(
        "--pipeline-preview-max", type=int, default=None,
        help="Override yolo.pipeline_eval.preview_max_images (0 = save every crop).",
    )
    parser.add_argument(
        "--gallery-only",
        action="store_true",
        help="Only rewrite outputs/.../pipeline_previews/gallery.html from existing JPGs (fast).",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.gallery_only:
        refresh_pipeline_gallery(cfg)
        print("  Done (--gallery-only).")
        return

    if args.stage is not None:
        eval_stage(cfg, args.stage)
    else:
        for s in [0, 1, 2]:
            eval_stage(cfg, s)

    if not args.skip_pipeline:
        run_pipeline(cfg, preview_max_override=args.pipeline_preview_max)

    refresh_pipeline_gallery(cfg)

    banner("YOLO evaluation complete")


if __name__ == "__main__":
    main()
