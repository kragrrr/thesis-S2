#!/usr/bin/env python3
"""Step 6 — Evaluate the SupCon encoder.

* k-NN classification on source hold-out
* Binary AUROC (normal vs anomaly)
* Cross-domain predictions on UAV crops
* t-SNE embedding visualisation
* Confusion matrix & per-class report

Run:  python 06_eval_supcon.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
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


# ── embedding extraction ────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    model: ResNet34Encoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, list]:
    """Return (N, D) embedding array and list of labels/paths."""
    model.eval()
    all_emb, all_lbl = [], []
    for batch in tqdm(loader, desc="  Extracting embeddings", leave=False):
        imgs, labels = batch
        imgs = imgs.to(device, non_blocking=True)
        emb = model(imgs).cpu().numpy()
        all_emb.append(emb)
        if isinstance(labels, torch.Tensor):
            all_lbl.extend(labels.tolist())
        else:
            all_lbl.extend(labels)
    return np.concatenate(all_emb, axis=0), all_lbl


# ── k-NN evaluation ─────────────────────────────────────────

def knn_evaluate(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    k_values: list[int],
    metric: str = "cosine",
) -> dict:
    results = {}
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric, n_jobs=-1)
        knn.fit(train_emb, train_labels)
        preds = knn.predict(test_emb)
        acc = (preds == test_labels).mean()
        results[k] = {"accuracy": float(acc), "predictions": preds}
        print(f"    k={k:>4d}  accuracy={acc:.4f}")
    return results


# ── main evaluation pipeline ────────────────────────────────

def evaluate(cfg: dict) -> None:
    banner("SupCon Evaluation")

    sc = cfg["supcon"]
    device = get_device(cfg)
    seed_everything(cfg.get("seed", 42))

    ckpt_dir = get_output_dir(cfg, "supcon", "checkpoints")
    eval_dir = get_output_dir(cfg, "supcon", "evaluation")
    plot_dir = get_output_dir(cfg, "supcon", "plots")

    best_ckpt = ckpt_dir / "best_encoder.pth"
    if not best_ckpt.exists():
        print("  ⚠  No checkpoint found. Run 05_train_supcon.py first.")
        return

    model = ResNet34Encoder(
        embed_dim=sc["embed_dim"],
        proj_hidden=sc.get("proj_hidden", 512),
        pretrained=False,
    ).to(device)

    state = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"  Loaded checkpoint from epoch {state['epoch'] + 1}  "
          f"(val_loss={state['val_loss']:.4f})")

    # ── source dataset ──
    raptor_dir = resolve_raptor_source_dir(cfg)
    if raptor_dir is None:
        print("  ⚠  Raptor dataset not found. Run 01_download_data.py first.")
        return
    images_dir = raptor_dir / "images"
    meta_path = raptor_dir / "module_metadata.json"

    do_std = sc.get("standardize", False)
    full_ds = RaptorDataset(images_dir, meta_path, standardize=do_std)
    all_labels = full_ds.labels
    n = len(full_ds)

    ds_mean, ds_std = None, None
    if do_std:
        ds_mean, ds_std = full_ds.compute_stats()
        print(f"  Dataset mean={ds_mean:.4f}  std={ds_std:.4f}")

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
    val_emb, val_labels = extract_embeddings(model, val_loader, device)

    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)

    # save embeddings for later reuse
    np.savez_compressed(
        eval_dir / "source_embeddings.npz",
        train_emb=train_emb, train_labels=train_labels,
        val_emb=val_emb, val_labels=val_labels,
    )

    # ── k-NN sweep ──
    banner("k-NN Classification (Source Hold-Out)")
    knn_cfg = sc["knn"]
    knn_results = knn_evaluate(
        train_emb, train_labels, val_emb, val_labels,
        k_values=knn_cfg["k_values"],
        metric=knn_cfg.get("metric", "cosine"),
    )

    best_k = knn_cfg.get("best_k", 100)
    best_preds = knn_results[best_k]["predictions"]

    # ── classification report ──
    report = classification_report(
        val_labels, best_preds,
        labels=list(range(12)),
        target_names=CLASS_NAMES_12,
        zero_division=0,
    )
    (eval_dir / "classification_report.txt").write_text(report)
    print(f"\n{report}")

    # ── confusion matrix ──
    pcfg = cfg.get("plots", {})
    cm = confusion_matrix(val_labels, best_preds, labels=list(range(12)))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES_12,
                yticklabels=CLASS_NAMES_12, cmap="Blues", ax=ax)
    ax.set_title(f"SupCon + k-NN (k={best_k}) Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(plot_dir / "confusion_matrix.png", dpi=pcfg.get("dpi", 150))
    plt.close(fig)

    # ── binary AUROC with decision-threshold δ (Bommes §IV-B) ──
    normal_idx = CLASS_NAMES_12.index("No-Anomaly")
    binary_true = (val_labels != normal_idx).astype(int)  # 1 = anomaly
    knn = KNeighborsClassifier(n_neighbors=best_k, metric=knn_cfg.get("metric", "cosine"), n_jobs=-1)
    knn.fit(train_emb, (train_labels != normal_idx).astype(int))
    binary_probs = knn.predict_proba(val_emb)[:, 1]
    auroc = roc_auc_score(binary_true, binary_probs)
    print(f"  Binary AUROC (normal vs anomaly): {auroc:.4f}")

    delta = knn_cfg.get("decision_threshold", 0.1)
    binary_preds_delta = (binary_probs >= delta).astype(int)
    tp = ((binary_preds_delta == 1) & (binary_true == 1)).sum()
    tn = ((binary_preds_delta == 0) & (binary_true == 0)).sum()
    tpr = tp / max(binary_true.sum(), 1)
    tnr = tn / max((1 - binary_true).sum(), 1)
    print(f"  At δ={delta}: TPR={tpr:.4f}  TNR={tnr:.4f}  "
          f"G-Mean={np.sqrt(tpr * tnr):.4f}")

    # ── k accuracy plot ──
    ks = sorted(knn_results.keys())
    accs = [knn_results[k]["accuracy"] for k in ks]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, accs, "o-", color="teal")
    ax.set_xlabel("k")
    ax.set_ylabel("12-class Accuracy")
    ax.set_title("k-NN Accuracy vs k")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(plot_dir / "knn_k_sweep.png", dpi=pcfg.get("dpi", 150))
    plt.close(fig)

    # ── t-SNE visualisation (source) ──
    banner("t-SNE Embedding Visualisation")
    max_pts = 3000
    if len(val_emb) > max_pts:
        idx = np.random.choice(len(val_emb), max_pts, replace=False)
        tsne_emb = val_emb[idx]
        tsne_lbl = val_labels[idx]
    else:
        tsne_emb = val_emb
        tsne_lbl = val_labels

    perp = min(30, max(2, len(tsne_emb) - 1))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=cfg.get("seed", 42))
    coords = tsne.fit_transform(tsne_emb)

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.cm.get_cmap("tab20", 12)
    for i, name in enumerate(CLASS_NAMES_12):
        mask = tsne_lbl == i
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.6,
                   label=name, color=cmap(i))
    ax.legend(fontsize=8, markerscale=3, loc="best")
    ax.set_title("t-SNE of SupCon Embeddings (Source Val)")
    plt.tight_layout()
    fig.savefig(plot_dir / "tsne_source.png", dpi=pcfg.get("dpi", 150))
    plt.close(fig)

    # ── anomaly score histogram ──
    dists, _ = knn.kneighbors(val_emb)
    mean_dists = dists.mean(axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(mean_dists[binary_true == 0], bins=50, alpha=0.6, label="Normal", color="green")
    ax.hist(mean_dists[binary_true == 1], bins=50, alpha=0.6, label="Anomaly", color="red")
    ax.axvline(knn_cfg.get("anomaly_threshold", 0.5), color="black", ls="--", label="Threshold")
    ax.set_xlabel("Mean k-NN Distance")
    ax.set_title("Anomaly Score Distribution")
    ax.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "anomaly_histogram.png", dpi=pcfg.get("dpi", 150))
    plt.close(fig)

    # ── cross-domain: UAV crops ──
    uav_crops_dir = get_data_dir(cfg, "uav_crops")
    if uav_crops_dir.exists() and any(uav_crops_dir.rglob("*.jpg")):
        banner("Cross-Domain Analysis (Zenodo UAV)")
        uav_ds = UAVPanelDataset(
            uav_crops_dir, standardize=do_std, mean=ds_mean, std=ds_std,
        )
        uav_loader = DataLoader(uav_ds, batch_size=bs, num_workers=nw, pin_memory=pin)
        uav_emb, uav_paths = extract_embeddings(model, uav_loader, device)

        knn_12 = KNeighborsClassifier(n_neighbors=best_k, metric=knn_cfg.get("metric", "cosine"), n_jobs=-1)
        knn_12.fit(train_emb, train_labels)
        uav_preds = knn_12.predict(uav_emb)
        uav_dists, _ = knn_12.kneighbors(uav_emb)
        uav_mean_dist = uav_dists.mean(axis=1)

        pred_counts = Counter(uav_preds)
        normal_count = pred_counts.get(normal_idx, 0)
        total = len(uav_preds)
        print(f"  Total patches : {total}")
        print(f"  Normal        : {normal_count} ({100*normal_count/total:.1f}%)")
        print(f"  Anomalous     : {total - normal_count} ({100*(total-normal_count)/total:.1f}%)")

        uav_summary = {
            "total": total,
            "normal": int(normal_count),
            "anomalous": int(total - normal_count),
            "class_distribution": {
                CLASS_NAMES_12[i]: int(c) for i, c in sorted(pred_counts.items())
            },
        }
        (eval_dir / "uav_predictions.json").write_text(json.dumps(uav_summary, indent=2))

        # domain-shift PCA
        from sklearn.decomposition import PCA
        combined = np.concatenate([val_emb[:max_pts], uav_emb[:max_pts]])
        domain_labels = (["Source"] * min(len(val_emb), max_pts) +
                         ["Target"] * min(len(uav_emb), max_pts))
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(combined)

        fig, ax = plt.subplots(figsize=(10, 8))
        for domain, color in [("Source", "blue"), ("Target", "orange")]:
            mask = [d == domain for d in domain_labels]
            ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1],
                       s=5, alpha=0.4, label=domain, color=color)
        ax.legend()
        ax.set_title("PCA — Source vs Target Domain Embeddings")
        plt.tight_layout()
        fig.savefig(plot_dir / "pca_domain_shift.png", dpi=pcfg.get("dpi", 150))
        plt.close(fig)
        print(f"  ✓ Domain shift PCA → {plot_dir / 'pca_domain_shift.png'}")

    # ── summary JSON ──
    summary = {
        "best_k": best_k,
        "decision_threshold_delta": delta,
        "12_class_accuracy": float(knn_results[best_k]["accuracy"]),
        "binary_auroc": float(auroc),
        "tpr_at_delta": float(tpr),
        "tnr_at_delta": float(tnr),
        "k_sweep": {str(k): v["accuracy"] for k, v in knn_results.items()},
    }
    (eval_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n  ✓ All evaluation artefacts saved to:")
    print(f"    {eval_dir}")
    print(f"    {plot_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate(cfg)
    banner("SupCon evaluation complete")


if __name__ == "__main__":
    main()
