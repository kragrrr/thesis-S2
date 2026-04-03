"""Shared utilities for the PV anomaly-detection pipeline."""

from __future__ import annotations

import os
import platform
import random
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import yaml

IS_WINDOWS = platform.system() == "Windows"

# Ensure UTF-8 output on Windows (avoids UnicodeEncodeError for ✓/✗/⚙/⚠ symbols)
if IS_WINDOWS:
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8")
            except Exception:
                pass

# ── Canonical class vocabulary ──────────────────────────────

CLASS_NAMES_12 = [
    "Cell", "Cell-Multi", "Cracking", "Hot-Spot",
    "Hot-Spot-Multi", "Shadowing", "Diode", "Diode-Multi",
    "Vegetation", "Soiling", "Offline-Module", "No-Anomaly",
]

DEFECT_CLASSES = CLASS_NAMES_12[:11]
HEALTHY_CLASSES = ["No-Anomaly"]

SEVERE_DEFECTS = {
    "Hot-Spot", "Hot-Spot-Multi", "Diode", "Diode-Multi",
    "Cracking", "Offline-Module",
}
MILD_DEFECTS = {"Cell", "Cell-Multi", "Shadowing", "Soiling", "Vegetation"}

SCRIPT_DIR = Path(__file__).resolve().parent.parent  # …/scripts/


# ── Safe Unicode printing (Windows cmd.exe fallback) ────────

_UNICODE_MAP = {"✓": "[OK]", "✗": "[FAIL]", "⚙": "[*]", "⚠": "[!]", "→": "->"}

def safe_print(*args, **kwargs) -> None:
    """Print that gracefully degrades Unicode symbols on Windows cmd.exe."""
    if IS_WINDOWS:
        try:
            "✓".encode(sys.stdout.encoding or "utf-8")
        except (UnicodeEncodeError, LookupError):
            args = tuple(
                _replace_unicode(str(a)) for a in args
            )
    print(*args, **kwargs)


def _replace_unicode(text: str) -> str:
    for sym, repl in _UNICODE_MAP.items():
        text = text.replace(sym, repl)
    return text


# ── Multiprocessing helpers ──────────────────────────────────

def safe_num_workers(cfg: dict) -> int:
    """Return num_workers capped for the current OS.

    Windows uses 'spawn' multiprocessing which is slower and more fragile
    than Unix 'fork'. Cap to 0-4 workers to avoid BrokenPipeError / hangs.
    """
    nw = cfg.get("num_workers", 4)
    if IS_WINDOWS:
        nw = min(nw, 4)
    return nw


# ── Config helpers ──────────────────────────────────────────

def load_config(path: str | Path | None = None) -> dict:
    if path is None:
        path = SCRIPT_DIR / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(cfg_path: str, base: Path | None = None) -> Path:
    """Turn a possibly-relative config path into an absolute one."""
    p = Path(cfg_path)
    if p.is_absolute():
        return p
    return (base or SCRIPT_DIR) / p


def get_data_dir(cfg: dict, *parts: str) -> Path:
    d = resolve_path(cfg["paths"]["data_root"]) / Path(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_output_dir(cfg: dict, *parts: str) -> Path:
    d = resolve_path(cfg["paths"]["output_root"]) / Path(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_raptor_clone_root(cfg: dict) -> Path:
    """Directory where `git clone` of InfraredSolarModules lands."""
    return get_data_dir(cfg, "raptor_raw", "InfraredSolarModules")


def find_raptor_dataset_dir(clone_root: Path) -> Path | None:
    """Return the folder that contains ``images/`` and ``module_metadata.json``.

    The upstream GitHub repo layout has changed over time (nested folder vs.
    zip-at-root only). We pick the shallowest matching path under ``clone_root``.
    """
    if not clone_root.is_dir():
        return None
    best: Path | None = None
    best_depth = 10**9
    for meta in clone_root.rglob("module_metadata.json"):
        root = meta.parent
        if (root / "images").is_dir():
            depth = len(meta.relative_to(clone_root).parts)
            if depth < best_depth:
                best_depth = depth
                best = root
    return best


def resolve_raptor_source_dir(cfg: dict) -> Path | None:
    """Resolved Raptor Maps data root, or ``None`` if not prepared yet."""
    return find_raptor_dataset_dir(get_raptor_clone_root(cfg))


# ── Reproducibility ─────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Device helper ───────────────────────────────────────────


def _cuda_has_usable_gpu() -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)


def get_device(cfg: dict) -> torch.device:
    dev = str(cfg.get("device", "0"))
    if dev == "cpu" or not _cuda_has_usable_gpu():
        print("⚙  Using CPU")
        return torch.device("cpu")
    idx = int(dev)
    if idx < 0 or idx >= torch.cuda.device_count():
        print(f"⚙  Using CPU (GPU index {idx} not available)")
        return torch.device("cpu")
    props = torch.cuda.get_device_properties(idx)
    print(f"⚙  GPU {idx}: {props.name}  ({props.total_memory / 1e9:.1f} GB)")
    return torch.device(f"cuda:{idx}")


_YOLO_CPU_FALLBACK_PRINTED = False


def yolo_device(cfg: dict) -> int | str:
    """``device=`` value for Ultralytics. Uses CPU if config asks for a GPU id but CUDA is missing."""
    global _YOLO_CPU_FALLBACK_PRINTED
    raw = cfg.get("device", "0")
    dev_s = str(raw).strip().lower()
    if dev_s == "cpu":
        return "cpu"
    if not _cuda_has_usable_gpu():
        if not _YOLO_CPU_FALLBACK_PRINTED:
            print(
                "⚙  Config requests a CUDA device but none is usable — using CPU for YOLO.\n"
                "   (CUDA build with no visible GPU is common on CPU-only cloud nodes.)\n"
                "   For a local GPU: install NVIDIA drivers and a matching CUDA PyTorch wheel, e.g.\n"
                "   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
                "   python -c \"import torch; print(torch.__version__, torch.cuda.device_count())\""
            )
            _YOLO_CPU_FALLBACK_PRINTED = True
        return "cpu"
    if dev_s.startswith("cuda"):
        return raw if isinstance(raw, str) else dev_s
    try:
        idx = int(raw)
    except (TypeError, ValueError):
        return str(raw)
    if idx < 0 or idx >= torch.cuda.device_count():
        if not _YOLO_CPU_FALLBACK_PRINTED:
            print(
                f"⚙  Config device index {idx} is out of range "
                f"(0–{torch.cuda.device_count() - 1}) — using CPU for YOLO."
            )
            _YOLO_CPU_FALLBACK_PRINTED = True
        return "cpu"
    return idx


def yolo_amp_enabled(cfg: dict) -> bool:
    """Mixed precision only when CUDA is available (Ultralytics AMP is GPU-oriented)."""
    return bool(cfg.get("amp", True)) and torch.cuda.is_available()


def yolo_stage_amp(cfg: dict, stage_cfg: dict) -> bool:
    """Per-stage AMP: ``stage_cfg['amp']`` overrides global ``cfg['amp']`` when set."""
    if not torch.cuda.is_available():
        return False
    if "amp" in stage_cfg:
        return bool(stage_cfg["amp"])
    return bool(cfg.get("amp", True))


# ── Timestamp tag (used by export) ──────────────────────────

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ── Pretty banner ───────────────────────────────────────────

def banner(text: str) -> None:
    rule = "=" * 60
    print(f"\n{rule}\n  {text}\n{rule}\n")
