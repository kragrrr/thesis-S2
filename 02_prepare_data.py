#!/usr/bin/env python3
"""Step 2 — Prepare datasets for training.

Creates the following directory layouts under data/:

  yolo_cls_binary/   train/{Healthy,Defective}  val/{…}    (Stage 1)
  yolo_cls_defects/  train/{11 classes}          val/{…}    (Stage 2)
  yolo_det_uav/      images/{train,val,test}  labels/{…}   (Stage 0)
  raptor_all/        12-class flat layout for SupCon (resize or raw — no zero-pad by default)
  uav_crops/         panel crops from Zenodo frames

Run:  python 02_prepare_data.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.utils import (
    CLASS_NAMES_12,
    DEFECT_CLASSES,
    banner,
    get_data_dir,
    get_raptor_clone_root,
    load_config,
    resolve_raptor_source_dir,
)


# ── helpers ─────────────────────────────────────────────────

def pad_to_square(img: np.ndarray, target: int, mode: str = "zero") -> np.ndarray:
    h, w = img.shape[:2]
    if mode == "resize":
        return cv2.resize(img, (target, target), interpolation=cv2.INTER_LINEAR)
    pad_h = (target - h) // 2
    pad_w = (target - w) // 2
    return cv2.copyMakeBorder(
        img, pad_h, target - h - pad_h, pad_w, target - w - pad_w,
        cv2.BORDER_CONSTANT, value=[0, 0, 0],
    )


def _raptor_preprocess(img: np.ndarray, out_h: int, out_w: int, mode: str) -> np.ndarray:
    """Prepare Raptor IR module crops for YOLO-cls / raptor_all.

    * ``resize`` — scale to ``(out_w, out_h)`` (no letterboxing / zero borders).
    * ``none`` — save as read from disk (native resolution; YOLO resizes at train time).
    * ``zero`` — legacy: pad with zeros to a square of side ``max(out_h, out_w)``.
    """
    m = (mode or "resize").strip().lower()
    if m == "none":
        return img
    if m == "resize":
        return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    if m == "zero":
        return pad_to_square(img, max(out_h, out_w), mode="zero")
    raise ValueError(
        f"Unknown data_prep.raptor.padding_mode: {mode!r} "
        '(expected "resize", "none", or "zero")'
    )


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


# ── Raptor Maps preparation ────────────────────────────────

def prepare_raptor(cfg: dict) -> None:
    banner("Preparing Raptor Maps dataset")

    raptor_raw = resolve_raptor_source_dir(cfg)
    if raptor_raw is None:
        print("  ⚠  Raptor dataset not found.")
        print(f"     Expected images/ + module_metadata.json under {get_raptor_clone_root(cfg)}")
        print("     Run 01_download_data.py first.")
        return

    images_dir = raptor_raw / "images"
    meta_path = raptor_raw / "module_metadata.json"

    with open(meta_path) as f:
        meta = json.load(f)

    prep = cfg["data_prep"]["raptor"]
    h, w = prep["image_size"]
    pad_mode = prep["padding_mode"]
    split_ratio = prep["train_split"]
    seed = cfg.get("seed", 42)

    entries = []
    for entry in meta.values():
        fname = Path(entry["image_filepath"]).name
        img_path = images_dir / fname
        if not img_path.exists():
            continue
        entries.append((img_path, entry["anomaly_class"]))

    print(f"  {len(entries)} images found")

    indices = list(range(len(entries)))
    labels_for_split = [e[1] for e in entries]
    train_idx, val_idx = train_test_split(
        indices, train_size=split_ratio, stratify=labels_for_split, random_state=seed,
    )

    # ── 12-class flat (for SupCon) ──
    all12_dir = get_data_dir(cfg, "raptor_all")
    _write_cls_split(entries, train_idx, val_idx, all12_dir, h, w, pad_mode, class_map=None)
    print(f"  ✓ raptor_all (12-class): {all12_dir}")

    # ── Binary: Healthy / Defective (Stage 1) ──
    binary_map = {}
    for name in CLASS_NAMES_12:
        binary_map[name] = "Healthy" if name == "No-Anomaly" else "Defective"
    binary_dir = get_data_dir(cfg, "yolo_cls_binary")
    _write_cls_split(entries, train_idx, val_idx, binary_dir, h, w, pad_mode, binary_map)
    print(f"  ✓ yolo_cls_binary (Stage 1): {binary_dir}")

    # ── 11-class defects only (Stage 2) ──
    defect_entries = [(p, c) for p, c in entries if c != "No-Anomaly"]
    d_indices = list(range(len(defect_entries)))
    d_labels = [e[1] for e in defect_entries]
    d_train, d_val = train_test_split(
        d_indices, train_size=split_ratio, stratify=d_labels, random_state=seed,
    )
    defect_dir = get_data_dir(cfg, "yolo_cls_defects")
    _write_cls_split(defect_entries, d_train, d_val, defect_dir, h, w, pad_mode, class_map=None)
    print(f"  ✓ yolo_cls_defects (Stage 2): {defect_dir}")

    _print_class_dist(entries, "Full dataset")


def _write_cls_split(
    entries, train_idx, val_idx, out_dir, h, w, pad_mode, class_map,
):
    for split_name, idxs in [("train", train_idx), ("val", val_idx)]:
        for i in tqdm(idxs, desc=f"  {split_name}", leave=False):
            img_path, cls_name = entries[i]
            target_cls = class_map[cls_name] if class_map else cls_name
            dest_dir = out_dir / split_name / target_cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = _raptor_preprocess(img, h, w, pad_mode)
            cv2.imwrite(str(dest_dir / img_path.name), img)


def _print_class_dist(entries, title):
    counts = Counter(e[1] for e in entries)
    total = sum(counts.values())
    print(f"\n  {title} distribution ({total} images):")
    for name in CLASS_NAMES_12:
        c = counts.get(name, 0)
        bar = "█" * int(40 * c / max(counts.values()))
        print(f"    {name:20s}  {c:>6d}  ({100*c/total:5.1f}%)  {bar}")


# ── Zenodo UAV preparation ──────────────────────────────────

def prepare_zenodo(cfg: dict) -> None:
    banner("Preparing Zenodo UAV dataset")

    data_root = get_data_dir(cfg)
    zenodo_raw = data_root / "zenodo_raw"

    # Try to find the dataset root (may be nested under long folder names)
    candidates = [
        zenodo_raw,
        *sorted(zenodo_raw.glob("*")),
        *sorted(zenodo_raw.glob("*/*")),
    ]
    dataset_root = None
    for c in candidates:
        if (c / "train" / "images").is_dir():
            dataset_root = c
            break

    # Deeper layouts: …/Thermal …/train/images (zip added one extra folder level)
    if dataset_root is None:
        for train_imgs in zenodo_raw.rglob("train/images"):
            if not train_imgs.is_dir():
                continue
            cand = train_imgs.parent.parent
            if (cand / "train" / "labels").is_dir():
                dataset_root = cand
                break

    if dataset_root is None:
        print("  ⚠  Zenodo UAV dataset not found.")
        print("     Expected structure: zenodo_raw/.../train/images/ + labels/")
        print("     Run 01_download_data.py or place the dataset manually.")
        return

    # ── Copy into standardised detection layout (Stage 0) ──
    det_dir = get_data_dir(cfg, "yolo_det_uav")
    data_yaml = det_dir / "data.yaml"

    for split in ["train", "val", "test"]:
        src_imgs = dataset_root / split / "images"
        src_lbls = dataset_root / split / "labels"
        if not src_imgs.exists():
            continue
        dst_imgs = det_dir / "images" / split
        dst_lbls = det_dir / "labels" / split
        dst_imgs.mkdir(parents=True, exist_ok=True)
        dst_lbls.mkdir(parents=True, exist_ok=True)

        for img in src_imgs.iterdir():
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                shutil.copy2(img, dst_imgs / img.name)
        for lbl in src_lbls.iterdir():
            if lbl.suffix == ".txt":
                shutil.copy2(lbl, dst_lbls / lbl.name)

        n_img = len(list(dst_imgs.iterdir()))
        print(f"  {split}: {n_img} images")

    data_yaml.write_text(
        f"path: {det_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names: ['Solar_Panel']\n"
    )
    print(f"  ✓ Detection layout: {det_dir}")
    print(f"  ✓ data.yaml: {data_yaml}")

    # ── Crop panels for downstream classification (Stages 1+2 / SupCon) ──
    _crop_panels(cfg, dataset_root)


def _crop_panels(cfg: dict, dataset_root: Path) -> None:
    """Crop individual panels using YOLO bounding-box labels."""
    prep = cfg["data_prep"]["zenodo_uav"]
    padding = prep["crop_padding"]
    crops_dir = get_data_dir(cfg, "uav_crops")
    manifest = crops_dir / "manifest.csv"

    print("  Cropping panels from UAV frames …")
    rows = []
    for split in ["train", "val", "test"]:
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"
        if not img_dir.exists():
            continue
        split_out = crops_dir / split
        split_out.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ih, iw = img.shape[:2]

            for i, line in enumerate(lbl_path.read_text().splitlines()):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, xc, yc, bw, bh = map(float, parts)
                x1 = int((xc - bw / 2) * iw)
                y1 = int((yc - bh / 2) * ih)
                x2 = int((xc + bw / 2) * iw)
                y2 = int((yc + bh / 2) * ih)
                px = int((x2 - x1) * padding)
                py = int((y2 - y1) * padding)
                x1, y1 = clamp(x1 - px, 0, iw), clamp(y1 - py, 0, ih)
                x2, y2 = clamp(x2 + px, 0, iw), clamp(y2 + py, 0, ih)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img[y1:y2, x1:x2]
                crop_name = f"{img_path.stem}__box{i:04d}.jpg"
                cv2.imwrite(str(split_out / crop_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append({
                    "split": split, "source": img_path.name,
                    "crop": crop_name, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })

    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "source", "crop", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  ✓ {len(rows)} panel crops → {crops_dir}")


# ── main ────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)

    prepare_raptor(cfg)
    prepare_zenodo(cfg)

    print("\n" + "=" * 60)
    print("  Data preparation complete.  Ready for training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
