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
from urllib.parse import quote

import requests
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from lib.utils import (
    banner,
    find_raptor_dataset_dir,
    get_data_dir,
    load_config,
)


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

    content = find_raptor_dataset_dir(clone_target)
    if content is None:
        zip_candidates = sorted(clone_target.glob("*.zip"))
        if not zip_candidates:
            zip_candidates = sorted(clone_target.rglob("*.zip"))
        if zip_candidates:
            # Prefer the large dataset archive over any tiny auxiliary zips
            zip_path = max(zip_candidates, key=lambda p: p.stat().st_size)
            print(f"  Extracting {zip_path.relative_to(clone_target)} …")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(clone_target)
            content = find_raptor_dataset_dir(clone_target)

    if content is None:
        raise FileNotFoundError(
            f"Could not find images/ and module_metadata.json under {clone_target}. "
            "The GitHub repo may have changed again — check RaptorMaps/InfraredSolarModules."
        )

    images_dir = content / "images"
    meta_path = content / "module_metadata.json"

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

    file_jobs: list[tuple[str, str]] = []

    if direct_url:
        du = direct_url.strip()
        fname = du.split("/")[-1].split("?")[0] or "dataset.bin"
        file_jobs = [(du, fname)]
    elif record_id:
        print(f"  Querying Zenodo API for record {record_id} …")
        api_url = f"https://zenodo.org/api/records/{record_id}"
        resp = requests.get(api_url, timeout=30)
        resp.raise_for_status()
        record = resp.json()
        file_entries = list(record.get("files") or [])
        # Zenodo RDM may omit embedded files — follow collection link
        if not file_entries and record.get("links", {}).get("files"):
            files_resp = requests.get(record["links"]["files"], timeout=30)
            files_resp.raise_for_status()
            payload = files_resp.json()
            file_entries = payload.get("entries") or payload.get("hits", {}).get("hits") or []

        # links["self"] is …/filename.zip/content — basename would be "content", so use "key"
        for fmeta in file_entries:
            link = fmeta.get("links") or {}
            url = link.get("self")
            if not url:
                continue
            name = fmeta.get("key")
            if not name and url.rstrip("/").endswith("/content"):
                parts = url.rstrip("/").split("/")
                name = parts[-2] if len(parts) >= 2 else "download.bin"
            if not name:
                name = url.split("/")[-1].split("?")[0]
            file_jobs.append((url, name))

        if not file_jobs:
            print("  ⚠  No files found via API. Please set datasets.zenodo.direct_url in config.yaml")
            print("     or manually download and place the dataset in:", data_dir)
            marker.touch()
            return data_dir
    else:
        print("  ⚠  No Zenodo URL configured. Set record_id or direct_url in config.yaml.")
        print("     Place the dataset manually in:", data_dir)
        marker.touch()
        return data_dir

    for url, fname in file_jobs:
        dest = data_dir / fname
        if dest.exists():
            print(f"  {fname} already exists, skipping")
            continue
        print(f"  Downloading {fname} …")
        # Zenodo self links may contain spaces; ensure path is request-safe
        download_url = url
        if "zenodo.org" in url and "/files/" in url and url.rstrip("/").endswith("/content"):
            prefix, _, rest = url.partition("/files/")
            seg, _, _ = rest.partition("/content")
            encoded = quote(seg, safe="/")
            download_url = f"{prefix}/files/{encoded}/content"
        _download_file(download_url, dest, desc=fname)

        if dest.suffix.lower() == ".zip":
            print(f"  Extracting {fname} …")
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(data_dir)

    n_files = sum(1 for _ in data_dir.rglob("*.jpg"))
    print(f"  ✓ {n_files} images found under {data_dir}")
    if n_files == 0:
        print(
            "  ⚠  No images extracted. If an earlier run saved a junk file named "
            "'content' here, delete it, remove zenodo_raw/.download_complete, "
            "and run 01_download_data.py again."
        )
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
