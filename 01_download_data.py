#!/usr/bin/env python3
"""Step 1 — Download source datasets.

* Raptor Maps InfraredSolarModules (GitHub)
* Zenodo UAV Thermal PV Panel Detection Dataset

Run:  python 01_download_data.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.utils import load_config, get_data_dir, banner


def clone_raptor(cfg: dict) -> Path:
    """Clone the Raptor Maps InfraredSolarModules repo."""
    banner("Downloading Raptor Maps dataset (GitHub)")
    data_dir = get_data_dir(cfg, "raptor_raw")
    marker = data_dir / ".download_complete"
    if marker.exists():
        print(f"  Already downloaded → {data_dir}")
        return data_dir

    git_url = cfg["datasets"]["raptor"]["git_url"]
    clone_target = data_dir / "InfraredSolarModules"

    if clone_target.exists():
        shutil.rmtree(clone_target)

    print(f"  git clone {git_url}")
    subprocess.run(
        ["git", "clone", "--depth", "1", git_url, str(clone_target)],
        check=True,
    )

    nested = clone_target / "InfraredSolarModules"
    if not nested.exists():
        raise FileNotFoundError(
            f"Expected nested dir {nested}. Repo layout may have changed."
        )

    zip_candidates = list(nested.glob("*.zip"))
    if zip_candidates:
        print(f"  Extracting {zip_candidates[0].name} …")
        with zipfile.ZipFile(zip_candidates[0]) as zf:
            zf.extractall(nested)

    images_dir = nested / "images"
    meta_path = nested / "module_metadata.json"
    if not images_dir.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Could not locate images/ or module_metadata.json inside the repo."
        )

    n_images = len(list(images_dir.glob("*.jpg")))
    print(f"  ✓ {n_images} images, metadata at {meta_path.relative_to(data_dir)}")
    marker.touch()
    return data_dir


def _download_file(url: str, dest: Path, desc: str = "") -> None:
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc or dest.name,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))


def download_zenodo(cfg: dict) -> Path:
    """Download the Zenodo UAV thermal PV panel detection dataset."""
    banner("Downloading Zenodo UAV dataset")
    data_dir = get_data_dir(cfg, "zenodo_raw")
    marker = data_dir / ".download_complete"
    if marker.exists():
        print(f"  Already downloaded → {data_dir}")
        return data_dir

    ds_cfg = cfg["datasets"]["zenodo"]
    direct_url = ds_cfg.get("direct_url")
    record_id = ds_cfg.get("record_id")

    if direct_url:
        file_urls = [direct_url]
    elif record_id:
        print(f"  Querying Zenodo API for record {record_id} …")
        api_url = f"https://zenodo.org/api/records/{record_id}"
        resp = requests.get(api_url, timeout=30)
        resp.raise_for_status()
        record = resp.json()
        file_urls = [f["links"]["self"] for f in record.get("files", [])]
        if not file_urls:
            print("  ⚠  No files found via API. Please set datasets.zenodo.direct_url in config.yaml")
            print("     or manually download and place the dataset in:", data_dir)
            marker.touch()
            return data_dir
    else:
        print("  ⚠  No Zenodo URL configured. Set record_id or direct_url in config.yaml.")
        print("     Place the dataset manually in:", data_dir)
        marker.touch()
        return data_dir

    for url in file_urls:
        fname = url.split("/")[-1].split("?")[0]
        dest = data_dir / fname
        if dest.exists():
            print(f"  {fname} already exists, skipping")
            continue
        print(f"  Downloading {fname} …")
        _download_file(url, dest, desc=fname)

        if dest.suffix == ".zip":
            print(f"  Extracting {fname} …")
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(data_dir)

    n_files = sum(1 for _ in data_dir.rglob("*.jpg"))
    print(f"  ✓ {n_files} images found under {data_dir}")
    marker.touch()
    return data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)

    raptor_dir = clone_raptor(cfg)
    zenodo_dir = download_zenodo(cfg)

    print("\n" + "=" * 60)
    print("  Download complete.")
    print(f"  Raptor : {raptor_dir}")
    print(f"  Zenodo : {zenodo_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
