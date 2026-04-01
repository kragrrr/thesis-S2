#!/usr/bin/env python3
"""Step 5 — Train the SupCon encoder on the Raptor Maps source domain.

Produces:
  outputs/supcon/checkpoints/best_encoder.pth
  outputs/supcon/checkpoints/final_encoder.pth
  outputs/supcon/plots/loss_curve.png

Run:  python 05_train_supcon.py [--config config.yaml] [--resume PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.supcon_dataset import RaptorDataset
from lib.supcon_model import AnomalyContrastiveLoss, ResNet34Encoder
from lib.utils import (
    CLASS_NAMES_12,
    get_data_dir,
    get_output_dir,
    load_config,
    seed_everything,
    get_device,
    banner,
    safe_num_workers,
)


# ── augmentation as a torch transform ───────────────────────

class SupConAugment(nn.Module):
    """Augmentation for single-channel 64×64 tensors.

    Follows Bommes et al. (arXiv:2112.02922 §IV-C.3):
      - random horizontal & vertical flip
      - random rotation by multiples of 90°
    Plus v2 additions:
      - random erasing
      - Gaussian blur
    """

    def __init__(self, cfg_aug: dict):
        super().__init__()
        self.hflip = cfg_aug.get("horizontal_flip", True)
        self.vflip = cfg_aug.get("vertical_flip", True)
        self.rot90 = cfg_aug.get("random_rotation_90", True)
        self.erasing_p = cfg_aug.get("random_erasing", 0.0)
        self.blur_k = cfg_aug.get("gaussian_blur_kernel", 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hflip and torch.rand(1).item() < 0.5:
            x = x.flip(-1)
        if self.vflip and torch.rand(1).item() < 0.5:
            x = x.flip(-2)
        if self.rot90:
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                x = torch.rot90(x, k, dims=(-2, -1))
        if self.erasing_p > 0 and torch.rand(1).item() < self.erasing_p:
            _, h, w = x.shape
            eh, ew = int(h * 0.3), int(w * 0.3)
            y0 = torch.randint(0, h - eh + 1, (1,)).item()
            x0 = torch.randint(0, w - ew + 1, (1,)).item()
            x[:, y0:y0 + eh, x0:x0 + ew] = 0.0
        return x


# ── training loop ───────────────────────────────────────────

def train(cfg: dict, resume_path: str | None = None) -> None:
    banner("SupCon Training — ResNet-34 Encoder")

    sc = cfg["supcon"]
    device = get_device(cfg)
    seed_everything(cfg.get("seed", 42))

    # ── dataset ──
    raptor_dir = get_data_dir(cfg) / "raptor_raw" / "InfraredSolarModules" / "InfraredSolarModules"
    images_dir = raptor_dir / "images"
    meta_path = raptor_dir / "module_metadata.json"

    if not meta_path.exists():
        print("  ⚠  Raptor dataset not found. Run 01 + 02 first.")
        return

    augment = SupConAugment(sc.get("augment", {}))
    do_std = sc.get("standardize", False)

    full_ds = RaptorDataset(images_dir, meta_path, standardize=do_std)
    all_labels = full_ds.labels
    n = len(full_ds)

    # compute per-dataset stats before splitting (Bommes §IV-C.2)
    ds_mean, ds_std = None, None
    if do_std:
        ds_mean, ds_std = full_ds.compute_stats()
        print(f"  Dataset mean={ds_mean:.4f}  std={ds_std:.4f}")

    train_idx, val_idx = train_test_split(
        list(range(n)), train_size=0.8, stratify=all_labels,
        random_state=cfg.get("seed", 42),
    )

    train_ds = RaptorDataset(
        images_dir, meta_path, transform=augment, indices=train_idx,
        standardize=do_std, mean=ds_mean, std=ds_std,
    )
    val_ds = RaptorDataset(
        images_dir, meta_path, transform=None, indices=val_idx,
        standardize=do_std, mean=ds_mean, std=ds_std,
    )

    # weighted sampling for class imbalance
    if sc.get("weighted_sampling", True):
        train_labels = train_ds.labels
        counts = np.bincount(train_labels, minlength=12).astype(float)
        weights = 1.0 / np.maximum(counts, 1.0)
        sample_weights = [weights[l] for l in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        shuffle = False
    else:
        sampler = None
        shuffle = True

    nw = safe_num_workers(cfg)
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=sc["batch_size"], sampler=sampler,
        shuffle=shuffle, num_workers=nw, pin_memory=pin, drop_last=True,
        persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=sc["batch_size"], shuffle=False,
        num_workers=nw, pin_memory=pin,
        persistent_workers=nw > 0,
    )

    # ── model ──
    model = ResNet34Encoder(
        embed_dim=sc["embed_dim"],
        proj_hidden=sc.get("proj_hidden", 512),
        pretrained=sc.get("pretrained", True),
    ).to(device)

    normal_label = CLASS_NAMES_12.index("No-Anomaly")  # 11
    criterion = AnomalyContrastiveLoss(
        temperature=sc["temperature"], normal_label=normal_label,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=sc["lr"],
        momentum=sc["momentum"],
        weight_decay=sc["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=sc["epochs"])

    start_epoch = 0
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"  Resumed from epoch {start_epoch}")

    ckpt_dir = get_output_dir(cfg, "supcon", "checkpoints")
    plot_dir = get_output_dir(cfg, "supcon", "plots")

    use_amp = cfg.get("amp", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_loss = float("inf")
    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(start_epoch, sc["epochs"]):
        # ── train ──
        model.train()
        running_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{sc['epochs']}", leave=False)
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = torch.tensor(labels, device=device) if not isinstance(labels, torch.Tensor) else labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                embeddings = model(imgs)
                loss = criterion(embeddings, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss = running_loss / max(n_batches, 1)

        # ── val ──
        model.eval()
        val_running = 0.0
        val_n = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = torch.tensor(labels, device=device) if not isinstance(labels, torch.Tensor) else labels.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    embeddings = model(imgs)
                    loss = criterion(embeddings, labels)
                val_running += loss.item()
                val_n += 1
        val_loss = val_running / max(val_n, 1)

        lr_now = scheduler.get_last_lr()[0]
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr_now)

        print(f"  Epoch {epoch+1:>3d}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  lr={lr_now:.6f}")

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(state, ckpt_dir / "last_encoder.pth")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(state, ckpt_dir / "best_encoder.pth")

    # ── save final ──
    torch.save(model.state_dict(), ckpt_dir / "final_encoder_weights.pth")

    # ── loss curve ──
    pcfg = cfg.get("plots", {})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["epoch"], history["train_loss"], label="Train")
    ax1.plot(history["epoch"], history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("SupCon Loss Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["epoch"], history["lr"], color="orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("LR Schedule")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(plot_dir / "loss_curve.png", dpi=pcfg.get("dpi", 150))
    plt.close(fig)

    (get_output_dir(cfg, "supcon") / "training_history.json").write_text(
        json.dumps(history, indent=2)
    )

    print(f"\n  ✓ Best val loss: {best_loss:.4f}")
    print(f"  ✓ Checkpoints : {ckpt_dir}")
    print(f"  ✓ Loss curve  : {plot_dir / 'loss_curve.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, resume_path=args.resume)
    banner("SupCon training complete")


if __name__ == "__main__":
    main()
