# PV Module Anomaly Detection — Reproducible Training Pipeline

End-to-end scripts for training and evaluating two approaches to infrared
PV module anomaly detection:

| Approach | Architecture | Task |
|----------|-------------|------|
| **YOLO 3-Stage** | YOLO detection + classification | Panel detection → binary triage → 11-class diagnosis |
| **SupCon + k-NN** | ResNet-34 encoder, contrastive loss | Domain-adaptive classification via learned embeddings |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run everything (download → prepare → train → evaluate → export)
python run_all.py

# Or run YOLO only / SupCon only
python run_all.py --skip supcon
python run_all.py --skip yolo
```

## Windows Setup (RTX 4090)

> The scripts are fully cross-platform. No path changes needed.

**Prerequisites:**

1. **Python 3.9+** — download from [python.org](https://www.python.org/downloads/) (check "Add to PATH" during install)
2. **Git** — install [Git for Windows](https://git-scm.com/download/win) (needed by `01_download_data.py`)
3. **CUDA Toolkit 12.x** — download from [NVIDIA](https://developer.nvidia.com/cuda-downloads)
4. **PyTorch with CUDA** — install *before* `requirements.txt`:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**Windows-specific notes:**

- Use `python` instead of `python3` in all commands
- `num_workers` is automatically capped to 4 on Windows (multiprocessing uses `spawn` instead of `fork`)
- `pin_memory` is only enabled when CUDA is detected
- Unicode symbols (✓/✗/⚙) display correctly in Windows Terminal; `cmd.exe` output is auto-reconfigured to UTF-8

**Smoke test:**

```powershell
cd scripts
python run_test.py
```

## Step-by-Step Usage

Each step is a standalone script that reads `config.yaml`:

| # | Script | What it does |
|---|--------|-------------|
| 01 | `01_download_data.py` | Clones Raptor Maps repo, downloads Zenodo UAV dataset |
| 02 | `02_prepare_data.py` | Organises images into YOLO-cls / detection layouts, crops UAV panels |
| 03 | `03_train_yolo.py` | Trains Stage 0 (detector), Stage 1 (binary), Stage 2 (11-class) |
| 04 | `04_eval_yolo.py` | Validates each stage, runs full pipeline on UAV crops |
| 05 | `05_train_supcon.py` | Trains ResNet-34 with anomaly-contrastive loss |
| 06 | `06_eval_supcon.py` | k-NN evaluation, AUROC, t-SNE, cross-domain analysis |
| 07 | `07_export_results.py` | Packages models + plots into a portable `.tar.gz` |

```bash
# Run individual steps
python 01_download_data.py
python 02_prepare_data.py
python 03_train_yolo.py --stage 1          # train only Stage 1
python 04_eval_yolo.py --stage 2           # evaluate only Stage 2
python 05_train_supcon.py --resume outputs/supcon/checkpoints/last_encoder.pth
python 07_export_results.py --tag rtx4090_run1
```

## Configuration

All hyperparameters live in **`config.yaml`**. Key tuning knobs:

### Hardware (RTX 4090 defaults)

```yaml
device: "0"          # GPU id
num_workers: 12      # DataLoader workers
amp: true            # mixed precision — saves VRAM, speeds up training
```

### YOLO Batch Sizes

The 64 px classification images are tiny, so you can use very large batches:

| Stage | Image Size | Default Batch | VRAM ~Usage |
|-------|-----------|---------------|-------------|
| Stage 0 (detection) | 640 px | 32 | ~12 GB |
| Stage 1 (binary cls) | 64 px | 2048 | ~8 GB |
| Stage 2 (defect cls) | 64 px | 2048 | ~8 GB |

### SupCon

```yaml
supcon:
  batch_size: 512    # increase to 1024 if VRAM allows
  epochs: 200        # loss was still decreasing at 100; 200-300 recommended
  temperature: 0.07  # sharper gradients → tighter clusters
  lr: 0.06
  weighted_sampling: true  # critical for the 588:1 class imbalance
```

## Output Structure

```
outputs/
├── yolo/
│   ├── stage0_detector/    weights/ plots/ results.csv args.yaml
│   ├── stage1_sorter/      weights/ plots/ results.csv args.yaml
│   ├── stage2_diagnostician/
│   └── evaluation/         metrics, confusion matrices, pipeline CSV
├── supcon/
│   ├── checkpoints/        best_encoder.pth, last_encoder.pth
│   ├── plots/              loss_curve, t-SNE, confusion matrix, PCA
│   ├── evaluation/         k-NN results, AUROC, classification report
│   └── training_history.json
└── export/
    └── pv_anomaly_<timestamp>.tar.gz   ← portable archive
```

## Exporting for the Home Machine

After training, package everything for transfer:

```bash
python 07_export_results.py --tag rtx4090_final

# The archive contains all weights, plots, metrics, and a manifest.
# Transfer and extract:
scp outputs/export/pv_anomaly_rtx4090_final_*.tar.gz user@home:~/
ssh user@home "cd ~ && tar xzf pv_anomaly_rtx4090_final_*.tar.gz"
```

## Datasets

| Dataset | Source | Role |
|---------|--------|------|
| **Raptor Maps InfraredSolarModules** | [GitHub](https://github.com/RaptorMaps/InfraredSolarModules) | 20k pre-cropped IR modules (12 classes) — classifier training |
| **Zenodo UAV Thermal PV** | Zenodo record `16420123` | Full-frame UAV thermal imagery — panel detection + cross-domain eval |

### Class Vocabulary (12 classes)

| ID | Class | Severity |
|----|-------|----------|
| 0 | Cell | Mild |
| 1 | Cell-Multi | Mild |
| 2 | Cracking | Severe |
| 3 | Hot-Spot | Severe |
| 4 | Hot-Spot-Multi | Severe |
| 5 | Shadowing | Mild |
| 6 | Diode | Severe |
| 7 | Diode-Multi | Severe |
| 8 | Vegetation | Mild |
| 9 | Soiling | Mild |
| 10 | Offline-Module | Severe |
| 11 | No-Anomaly | Healthy |

## Resuming Interrupted Training

**YOLO**: Ultralytics auto-saves `last.pt`; re-run the same `03_train_yolo.py`
command and it will pick up from the latest checkpoint (set `exist_ok: True`).

**SupCon**: Pass `--resume`:

```bash
python 05_train_supcon.py --resume outputs/supcon/checkpoints/last_encoder.pth
```

**Full pipeline**: Use `--from-step`:

```bash
python run_all.py --from-step 5   # resume from SupCon training
```
