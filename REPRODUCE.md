# Reproducing AdaSemSeg

This document provides step-by-step commands to reproduce every result reported in the paper.

## 0. Setup

```bash
conda env create -f environment.yml
conda activate adasemseg

# Download datasets and trained weights
python scripts/download_assets.py --all

# Point the repo to the downloaded data (optional; defaults to ./data)
export ADASEMSEG_DATA_ROOT=/path/to/Repo/data
```

Expected data layout after extraction:

```
data/
├── F3/
│   ├── train/
│   │   ├── train_seismic.npy
│   │   └── train_labels.npy
│   ├── test/
│   │   ├── test1_seismic.npy
│   │   ├── test1_labels.npy
│   │   ├── test2_seismic.npy
│   │   └── test2_labels.npy
│   └── split_train_val_test_f3.json
├── Parihaka/
│   ├── parihaka_facies_train_images.npy
│   ├── parihaka_facies_train_labels.npy
│   └── split_train_val_test_parihaka.json
└── Penobscot/
    ├── seismic.npy
    ├── seismic_labels.npy
    └── split_train_val_test_penobscot.json
```

## 0.1 Train / val / test splits

The `split_train_val_test_*.json` files encode the exact slices used for training, validation, and testing. All competing methods and the SimCLR pretraining draw patches/volumes from these same files.

| Dataset | Train region | Val region | Test region |
|---|---|---|---|
| **F3 inline** | inlines 0–400, xlines 0–410 | inlines 401–410 | inlines 411–460 (test1) |
| **F3 crossline** | inlines 0–400, xlines 0–410 | xlines 411–420 | xlines 421–471 (test2) |
| **Parihaka inline** | inlines 10–401, xlines 0–691 | inlines 0–10 | inlines 0–10 (val_1) and inlines 0–401 / xlines 691–701 (val_2) are reused as the test set |
| **Parihaka crossline** | inlines 0–401, xlines 0–691 | xlines 691–701 | same val regions used for test |
| **Penobscot inline** | inlines 0–530, xlines 1–662 | inlines 530–540 | inlines 540–590 (test1) |
| **Penobscot crossline** | inlines 0–530, xlines 1–662 | xlines 662–682 | xlines 682–782 (test2) |

Support-set sizes (1, 5, 10, 20, 50) are specified inside each split JSON (e.g. `train_inline_5`).

## 1. SimCLR pretraining (image-encoder initialization)

The SimCLR checkpoint used in all downstream evaluations is downloaded from Zenodo in step 0 above (`python scripts/download_assets.py --weights`) into `checkpoints/simclr/simclr_resnet50_epoch10.pth.tar`.

To retrain from scratch, see [`docs/running.md`](docs/running.md#simclr-pretraining-image-encoder-initialization).

## 2. AdaSemSeg training (leave-one-out)

See [`docs/running.md`](docs/running.md#adasemseg) for per-method commands.

Dataset selection is controlled via the `--classes` flag or by editing the `classes` list in `methods/adasemseg/config/local_config.py`.

## 3. Evaluation

All published checkpoints are indexed in `checkpoints/scenarios.json`. Use the `--scenario` flag to evaluate a specific paper result, or pass `--checkpoint` explicitly.

### AdaSemSeg

AdaSemSeg supports two support-sampling strategies that correspond to the two
evaluation modes in Table I:

- **K-shot random support**: for F3 and Penobscot results.
- **Nearest-slice support**: for Parihaka results (and the nearest-slice ablation
  in Table I).

#### F3 and Penobscot (5-shot random support)
```bash
# AdaSemSeg SimCLR 5-shot on F3
python scripts/evaluate_adasemseg.py --scenario simclr_5-shot_sampling_f3 --device cuda:0

# AdaSemSeg SimCLR 5-shot on Penobscot
python scripts/evaluate_adasemseg.py --scenario simclr_5-shot_sampling_penobscot --device cuda:0
```

#### Parihaka (nearest-slice support)
```bash
python scripts/evaluate_adasemseg.py --scenario simclr_5-shot_nearest_slice_parihaka --device cuda:0
```

The script writes metrics to `./evaluation_results/metrics.json` (both averaged
across target datasets and per-dataset).

### ProtoSemSeg
ProtoSemSeg evaluation code is included; its trained checkpoints are downloaded from Zenodo into `checkpoints/protosemseg/<dataset>/<shots>-shot/` (see the README's "Model weights" section). To evaluate ProtoSemSeg, provide `--checkpoint_dir` directly:

```bash
python scripts/evaluate_protosemseg.py \
    --checkpoint_dir <path_to_protosemseg_checkpoint_dir> \
    --run_id 1 --shots 5 --eval_mode test \
    --source_class f3_facies_data_inline --device cuda:0
```

### Baselines
```bash
python scripts/evaluate_baseline.py \
    --checkpoint_dir methods/baselines/logs/checkpoints/Run_1 \
    --eval_mode test \
    --source_class f3_facies_data_inline --device cuda:0
```

## 4. Paper tables and figures

| Paper item | Command |
|------------|---------|
| Table I | `python scripts/reproduce_table1.py --device cuda:0` |
| Table II | `python scripts/reproduce_table2.py --device cuda:0` |
| Table III | `python scripts/reproduce_table3.py --device cuda:0` |
| Table IV | `python scripts/reproduce_table4.py --shots 5 --device cuda:0` |
| Table V | `python scripts/reproduce_table5.py --device cuda:0` |
| Table VI | `python scripts/reproduce_table6.py --checkpoint <CKPT> --device cuda:0` |
| Table VII | `python scripts/reproduce_table7.py --checkpoint <CKPT> --device cuda:0` |
| Figures | `python scripts/reproduce_figures.py --checkpoint <CKPT> --target {f3,parihaka,penobscot} --device cuda:0` |

Add `--run_commands` to Tables II/III to execute the printed commands. Tables II/III also require the source/target classes in each method's `config/local_config.py` to be set correctly (the script prints the expected list).

## 5. Metrics

All metrics are computed per pixel and per class where applicable:

- **Pixel Accuracy (PA)**: fraction of correctly classified pixels.
- **Class Accuracy**: per-class accuracy.
- **Mean Class Accuracy (MCA)**: average of per-class accuracies.
- **Intersection over Union (IoU)**: per-class IoU.
- **Frequency-weighted IoU (FwIoU)**: IoU weighted by class frequency.
- **F1 score**: per-class F1.
- **Frequency-weighted F1 (FwF1)**: F1 weighted by class frequency.

See `scripts/metrics.py` for implementations.

## 6. Notes

- All SimCLR-init experiments use the same checkpoint: `checkpoints/simclr/simclr_resnet50_epoch10.pth.tar`.
- Dataset selection is controlled via the `--classes` flag or each method's `config/local_config.py`. The central `configs/datasets.yaml` is also used by the unified evaluation scripts.
- Unified evaluation wrappers are `scripts/evaluate_adasemseg.py` and `scripts/evaluate_protosemseg.py`; the corresponding scenario-to-checkpoint map lives in `checkpoints/scenarios.json`.
- Model weights are large (~300 MB each) and are **not** committed to this repository. Download them from Zenodo with `python scripts/download_assets.py --weights` (see the README's "Download datasets and weights" section).
