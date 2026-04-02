#!/usr/bin/env python3
"""Step 7 — Package all models, plots, and metrics into a portable archive.

Creates:
  outputs/export/pv_anomaly_<timestamp>.tar.gz

The archive contains everything needed to continue building the O&M
system on another machine: trained weights, evaluation plots, metrics,
config snapshot, and a manifest.

Run:  python 07_export_results.py [--config config.yaml] [--tag my_run]
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.utils import load_config, get_output_dir, timestamp, banner


EXPORT_GLOBS = [
    # YOLO weights
    "yolo/*/weights/best.pt",
    "yolo/*/weights/last.pt",
    # YOLO training artefacts
    "yolo/*/args.yaml",
    "yolo/*/results.csv",
    "yolo/*/results.png",
    "yolo/*/*.png",
    # YOLO evaluation
    "yolo/evaluation/*.json",
    "yolo/evaluation/*.txt",
    "yolo/evaluation/*.csv",
    "yolo/evaluation/*.png",
    "yolo/evaluation/pipeline_previews/gallery.html",
    # SupCon checkpoints
    "supcon/checkpoints/*.pth",
    # SupCon plots
    "supcon/plots/*.png",
    # SupCon evaluation
    "supcon/evaluation/*.json",
    "supcon/evaluation/*.txt",
    "supcon/evaluation/*.npz",
    # SupCon training history
    "supcon/training_history.json",
]


def export(cfg: dict, tag: str | None = None) -> None:
    banner("Exporting results")

    output_root = get_output_dir(cfg)
    export_dir = get_output_dir(cfg, "export")

    ts = timestamp()
    name = f"pv_anomaly_{tag}_{ts}" if tag else f"pv_anomaly_{ts}"
    archive_path = export_dir / f"{name}.tar.gz"

    # collect files
    files: list[Path] = []
    for pattern in EXPORT_GLOBS:
        files.extend(output_root.glob(pattern))

    # add config snapshot
    config_path = SCRIPT_DIR / "config.yaml"
    if config_path.exists():
        files.append(config_path)

    files = sorted(set(files))
    prev_root = output_root / "yolo" / "evaluation" / "pipeline_previews"
    if prev_root.is_dir():
        files.extend(sorted(prev_root.rglob("*.jpg")))

    files = sorted(set(files))
    if not files:
        print("  ⚠  No output files found. Run training + evaluation first.")
        return

    # build manifest
    manifest = []
    for f in files:
        try:
            rel = f.relative_to(output_root)
        except ValueError:
            rel = f.name
        manifest.append({
            "path": str(rel),
            "size_mb": round(f.stat().st_size / 1e6, 2),
        })

    manifest_path = export_dir / f"{name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    files.append(manifest_path)

    # create tarball
    total_mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"  Packing {len(files)} files ({total_mb:.1f} MB) …")

    with tarfile.open(archive_path, "w:gz") as tar:
        for f in files:
            try:
                arcname = str(f.relative_to(output_root))
            except ValueError:
                arcname = f.name
            tar.add(f, arcname=arcname)

    print(f"\n  ✓ Archive: {archive_path}")
    print(f"    Size:    {archive_path.stat().st_size / 1e6:.1f} MB")
    print(f"    Files:   {len(files)}")
    print(f"\n  Transfer this file to your home machine and extract with:")
    print(f"    tar xzf {archive_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--tag", default=None,
                        help="Optional tag to include in archive filename (e.g. 'rtx4090_run1').")
    args = parser.parse_args()
    cfg = load_config(args.config)
    export(cfg, tag=args.tag)


if __name__ == "__main__":
    main()
