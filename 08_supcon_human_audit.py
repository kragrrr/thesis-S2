#!/usr/bin/env python3
"""Export SupCon + k-NN predictions for human audit (CSV + HTML gallery).

Uses the same train/val split and checkpoint as ``06_eval_supcon.py``.

  Source validation: true vs predicted label (spot-check metadata and model).
  UAV crops (optional): predicted class + scores only (no ground truth in dataset).

Run:
  python 08_supcon_human_audit.py [--config config.yaml] [--max-samples 200]
  python 08_supcon_human_audit.py --uav --max-samples 150
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.supcon_dataset import RaptorDataset, UAVPanelDataset
from lib.supcon_model import ResNet34Encoder
from lib.utils import (
    CLASS_NAMES_12,
    banner,
    get_data_dir,
    get_device,
    get_output_dir,
    load_config,
    resolve_raptor_source_dir,
    safe_num_workers,
    seed_everything,
)


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    all_emb, all_extra = [], []
    for batch in tqdm(loader, desc="  Embeddings", leave=False):
        imgs, extra = batch
        imgs = imgs.to(device, non_blocking=True)
        emb = model(imgs).cpu().numpy()
        all_emb.append(emb)
        if isinstance(extra, torch.Tensor):
            all_extra.extend(extra.tolist())
        else:
            all_extra.extend(extra)
    return np.concatenate(all_emb, axis=0), all_extra


def _write_html(
    out_path: Path,
    title: str,
    rows: list[tuple[str, str, str]],
) -> None:
    """rows: (thumb_relpath, caption_line1, caption_line2)"""
    parts = [
        "<!DOCTYPE html>",
        "<html lang='en'><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:24px;background:#fafafa;}",
        "h1{font-size:1.05rem;}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px;}",
        "figure{margin:0;background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px;}",
        "img{width:100%;height:96px;object-fit:contain;background:#111;}",
        "figcaption{font-size:11px;margin-top:6px;word-break:break-word;color:#333;}",
        ".err{border-color:#c0392b;}",
        "</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        "<p>Open this file in a browser. Compare <b>true</b> (metadata) vs <b>pred</b> (k-NN).</p>",
        "<div class='grid'>",
    ]
    for rel, line1, line2 in rows:
        err_cls = " err" if "✗" in line1 or "mismatch" in line1.lower() else ""
        parts.append(
            f"<figure class='{err_cls.strip()}'>"
            f"<img src='{html.escape(rel, quote=True)}' alt=''>"
            f"<figcaption>{html.escape(line1)}<br>{html.escape(line2)}</figcaption>"
            f"</figure>"
        )
    parts.append("</div></body></html>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def audit_source(
    cfg: dict,
    model: ResNet34Encoder,
    device: torch.device,
    audit_dir: Path,
    max_samples: int,
    seed: int,
) -> None:
    sc = cfg["supcon"]
    raptor_dir = resolve_raptor_source_dir(cfg)
    if raptor_dir is None:
        print("  ⚠  Raptor dataset not found.")
        return
    images_dir = raptor_dir / "images"
    meta_path = raptor_dir / "module_metadata.json"
    do_std = sc.get("standardize", False)
    full_ds = RaptorDataset(images_dir, meta_path, standardize=do_std)
    all_labels = full_ds.labels
    n = len(full_ds)
    ds_mean = ds_std = None
    if do_std:
        ds_mean, ds_std = full_ds.compute_stats()

    train_idx, val_idx = train_test_split(
        list(range(n)), train_size=0.8, stratify=all_labels,
        random_state=cfg.get("seed", 42),
    )
    nw = safe_num_workers(cfg)
    bs = sc["batch_size"]
    pin = device.type == "cuda"

    train_ds = RaptorDataset(
        images_dir, meta_path, indices=train_idx,
        standardize=do_std, mean=ds_mean, std=ds_std,
    )
    val_ds = RaptorDataset(
        images_dir, meta_path, indices=val_idx,
        standardize=do_std, mean=ds_mean, std=ds_std,
    )
    train_loader = DataLoader(train_ds, batch_size=bs, num_workers=nw, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=bs, num_workers=nw, pin_memory=pin)

    train_emb, train_labels = extract_embeddings(model, train_loader, device)
    val_emb, _ = extract_embeddings(model, val_loader, device)
    train_labels = np.array(train_labels)
    val_labels = np.array([val_ds.samples[i][1] for i in range(len(val_ds))])
    val_paths = [val_ds.samples[i][0] for i in range(len(val_ds))]

    knn_cfg = sc["knn"]
    best_k = knn_cfg.get("best_k", 100)
    metric = knn_cfg.get("metric", "cosine")
    knn = KNeighborsClassifier(n_neighbors=best_k, metric=metric, n_jobs=-1)
    knn.fit(train_emb, train_labels)
    preds = knn.predict(val_emb)
    dists, _ = knn.kneighbors(val_emb)
    mean_dist = dists.mean(axis=1)

    normal_idx = CLASS_NAMES_12.index("No-Anomaly")
    knn_bin = KNeighborsClassifier(n_neighbors=best_k, metric=metric, n_jobs=-1)
    knn_bin.fit(train_emb, (train_labels != normal_idx).astype(int))
    bin_prob = knn_bin.predict_proba(val_emb)[:, 1]

    wrong = np.where(preds != val_labels)[0]
    right = np.where(preds == val_labels)[0]
    rng = np.random.default_rng(seed)
    n_err = min(len(wrong), max(1, max_samples * 2 // 3))
    n_ok = max(0, max_samples - n_err)
    pick_wrong = rng.choice(wrong, size=min(n_err, len(wrong)), replace=False) if len(wrong) else np.array([], dtype=int)
    if len(right) and n_ok > 0:
        pick_right = rng.choice(right, size=min(n_ok, len(right)), replace=False)
    else:
        pick_right = np.array([], dtype=int)
    order = np.concatenate([pick_wrong, pick_right])

    thumb_dir = audit_dir / "audit_source_thumbs"
    if thumb_dir.exists():
        shutil.rmtree(thumb_dir)
    thumb_dir.mkdir(parents=True)

    csv_path = audit_dir / "audit_source_val.csv"
    html_rows: list[tuple[str, str, str]] = []

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["path", "true_class", "pred_class", "correct",
             "binary_prob_anomaly", "mean_knn_dist"],
        )
        for i, vi in enumerate(order):
            p = val_paths[vi]
            t = int(val_labels[vi])
            pr = int(preds[vi])
            ok = t == pr
            name_t = CLASS_NAMES_12[t]
            name_p = CLASS_NAMES_12[pr]
            rel = f"audit_source_thumbs/{i:04d}.jpg"
            dst = audit_dir / rel
            shutil.copy2(p, dst)
            w.writerow(
                [str(p), name_t, name_p, ok, f"{bin_prob[vi]:.6f}", f"{mean_dist[vi]:.6f}"],
            )
            mark = "✓" if ok else "✗"
            html_rows.append(
                (
                    rel,
                    f"{mark} true={name_t} → pred={name_p}",
                    f"P(anom)={bin_prob[vi]:.3f}  d̄={mean_dist[vi]:.3f}",
                )
            )

    _write_html(
        audit_dir / "audit_source_val.html",
        "SupCon human audit — source validation (metadata vs k-NN)",
        html_rows,
    )
    print(f"  ✓ {audit_dir / 'audit_source_val.csv'}")
    print(f"  ✓ {audit_dir / 'audit_source_val.html'}")


def audit_uav(
    cfg: dict,
    model: ResNet34Encoder,
    device: torch.device,
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    audit_dir: Path,
    max_samples: int,
    seed: int,
) -> None:
    sc = cfg["supcon"]
    uav_crops_dir = get_data_dir(cfg, "uav_crops")
    if not uav_crops_dir.exists() or not any(uav_crops_dir.rglob("*.jpg")):
        print("  ⚠  No UAV crops under data path; skip UAV audit.")
        return

    do_std = sc.get("standardize", False)
    raptor_dir = resolve_raptor_source_dir(cfg)
    if raptor_dir is None:
        print("  ⚠  Need Raptor path for standardisation stats.")
        return
    meta_path = raptor_dir / "module_metadata.json"
    images_dir = raptor_dir / "images"
    ref = RaptorDataset(images_dir, meta_path, standardize=do_std)
    ds_mean = ds_std = None
    if do_std:
        ds_mean, ds_std = ref.compute_stats()

    uav_ds = UAVPanelDataset(
        uav_crops_dir, standardize=do_std, mean=ds_mean, std=ds_std,
    )
    nw = safe_num_workers(cfg)
    bs = sc["batch_size"]
    pin = device.type == "cuda"
    uav_loader = DataLoader(uav_ds, batch_size=bs, num_workers=nw, pin_memory=pin)
    uav_emb, uav_paths = extract_embeddings(model, uav_loader, device)

    knn_cfg = sc["knn"]
    best_k = knn_cfg.get("best_k", 100)
    metric = knn_cfg.get("metric", "cosine")
    knn = KNeighborsClassifier(n_neighbors=best_k, metric=metric, n_jobs=-1)
    knn.fit(train_emb, train_labels)
    preds = knn.predict(uav_emb)
    dists, _ = knn.kneighbors(uav_emb)
    mean_dist = dists.mean(axis=1)

    normal_idx = CLASS_NAMES_12.index("No-Anomaly")
    knn_bin = KNeighborsClassifier(n_neighbors=best_k, metric=metric, n_jobs=-1)
    knn_bin.fit(train_emb, (train_labels != normal_idx).astype(int))
    bin_prob = knn_bin.predict_proba(uav_emb)[:, 1]

    rng = np.random.default_rng(seed)
    n = len(uav_paths)
    idx = rng.choice(n, size=min(max_samples, n), replace=False)

    thumb_dir = audit_dir / "audit_uav_thumbs"
    if thumb_dir.exists():
        shutil.rmtree(thumb_dir)
    thumb_dir.mkdir(parents=True)

    csv_uav = audit_dir / "audit_uav.csv"
    html_rows: list[tuple[str, str, str]] = []

    with csv_uav.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "pred_class", "binary_prob_anomaly", "mean_knn_dist"])
        for i, ui in enumerate(idx):
            p = Path(uav_paths[ui])
            pr = int(preds[ui])
            name_p = CLASS_NAMES_12[pr]
            rel = f"audit_uav_thumbs/{i:04d}.jpg"
            dst = audit_dir / rel
            shutil.copy2(p, dst)
            w.writerow(
                [str(p), name_p, f"{bin_prob[ui]:.6f}", f"{mean_dist[ui]:.6f}"],
            )
            html_rows.append(
                (
                    rel,
                    f"pred={name_p} (no GT in UAV set)",
                    f"P(anom)={bin_prob[ui]:.3f}  d̄={mean_dist[ui]:.3f}",
                )
            )
    _write_html(
        audit_dir / "audit_uav.html",
        "SupCon human audit — Zenodo UAV (predictions only)",
        html_rows,
    )
    print(f"  ✓ {audit_dir / 'audit_uav.csv'}")
    print(f"  ✓ {audit_dir / 'audit_uav.html'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--max-samples", type=int, default=160, help="Max images in each gallery.")
    parser.add_argument("--uav", action="store_true", help="Also build UAV audit gallery.")
    parser.add_argument("--uav-only", action="store_true", help="Only UAV (requires source pass for k-NN).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for sampling (default: config seed).")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    seed_everything(seed)

    banner("SupCon human audit export")
    sc = cfg["supcon"]
    device = get_device(cfg)
    ckpt_dir = get_output_dir(cfg, "supcon", "checkpoints")
    best_ckpt = ckpt_dir / "best_encoder.pth"
    if not best_ckpt.exists():
        print("  ⚠  Train first: 05_train_supcon.py")
        return

    model = ResNet34Encoder(
        embed_dim=sc["embed_dim"],
        proj_hidden=sc.get("proj_hidden", 512),
        pretrained=False,
    ).to(device)
    state = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    audit_dir = get_output_dir(cfg, "supcon", "audit")

    train_emb = train_labels = None
    if not args.uav_only:
        audit_source(cfg, model, device, audit_dir, args.max_samples, seed)

    if args.uav or args.uav_only:
        # Need train embeddings for k-NN fit
        raptor_dir = resolve_raptor_source_dir(cfg)
        if raptor_dir is None:
            print("  ⚠  Raptor dataset required for UAV audit.")
            return
        images_dir = raptor_dir / "images"
        meta_path = raptor_dir / "module_metadata.json"
        do_std = sc.get("standardize", False)
        full_ds = RaptorDataset(images_dir, meta_path, standardize=do_std)
        all_labels = full_ds.labels
        n = len(full_ds)
        ds_mean = ds_std = None
        if do_std:
            ds_mean, ds_std = full_ds.compute_stats()
        train_idx, _ = train_test_split(
            list(range(n)), train_size=0.8, stratify=all_labels,
            random_state=cfg.get("seed", 42),
        )
        train_ds = RaptorDataset(
            images_dir, meta_path, indices=train_idx,
            standardize=do_std, mean=ds_mean, std=ds_std,
        )
        nw = safe_num_workers(cfg)
        bs = sc["batch_size"]
        pin = device.type == "cuda"
        train_loader = DataLoader(
            train_ds, batch_size=bs, num_workers=nw, pin_memory=pin,
        )
        train_emb, train_labels = extract_embeddings(model, train_loader, device)
        train_labels = np.array(train_labels)
        audit_uav(cfg, model, device, train_emb, train_labels, audit_dir, args.max_samples, seed)

    meta = {
        "checkpoint_epoch": int(state["epoch"]) + 1,
        "val_loss": float(state["val_loss"]),
        "max_samples": args.max_samples,
        "seed": seed,
    }
    (audit_dir / "audit_run.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\n  Done. Open HTML files under:\n    {audit_dir}")
    banner("Audit export complete")


if __name__ == "__main__":
    main()
