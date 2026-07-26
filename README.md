# AdaSemSeg

**AdaSemSeg: An Adaptive Few-shot Semantic Segmentation of Seismic Facies**

This repository contains a complete, reproducible implementation of the paper *AdaSemSeg: An Adaptive Few-shot Semantic Segmentation of Seismic Facies* by Surojit Saha and Ross Whitaker.

The repository includes:
- The proposed **AdaSemSeg** method
- All baselines: **Baseline-1**, **Baseline-2**, and **transfer learning**
- The competing **ProtoSemSeg** prototype-based few-shot method
- **SimCLR** self-supervised pretraining for image-encoder initialization
- All ablation studies: initialization, data augmentation, architecture
- Trained model weights and evaluation datasets (hosted externally via IEEE DataPort)
- Unified evaluation scripts and metric definitions

## Quick links

- Paper: [LaTeX source + figures in `docs/figures`]
- Method summaries: [`docs/html_summaries/`](docs/html_summaries/)
- Method descriptions: [`docs/methods/`](docs/methods/)
- Running commands: [`docs/running.md`](docs/running.md)
- Reproduction guide: [`REPRODUCE.md`](REPRODUCE.md)
- Data and weights: see [IEEE DataPort placeholder — update after upload]
- Papers with Code: [placeholder — update after submission]

## Installation

```bash
# Create a conda environment
conda env create -f environment.yml
conda activate adasemseg

# Or use pip
pip install -r requirements.txt
```

## Download datasets and weights

Large datasets and trained weights are hosted externally via **IEEE DataPort**. Run:

```bash
# After filling in the real URLs in scripts/download_assets.py
python scripts/download_assets.py --all
```

The model weights shipped in the repository are tracked with **Git LFS**. After cloning, run:

```bash
git lfs pull
```

By default the code looks for data under `./data`:

```
data/
├── F3/
│   ├── train/...
│   ├── test/...
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

To use a different location, set:

```bash
export ADASEMSEG_DATA_ROOT=/path/to/data
```

On Windows PowerShell:
```powershell
$env:ADASEMSEG_DATA_ROOT = "C:\path\to\data"
```

## Repository structure

```
AdaSemSeg/
├── methods/
│   ├── adasemseg/       # Proposed method
│   ├── baselines/       # Baseline-1/2 and transfer learning
│   └── protosemseg/     # Prototype-based few-shot method
├── pretraining/
│   └── simclr/          # SimCLR image-encoder initialization
├── configs/
│   ├── datasets.yaml    # Unified dataset paths
│   └── common.yaml      # Shared hyperparameters
├── scripts/
│   ├── download_assets.py
│   ├── evaluate_adasemseg.py
│   ├── evaluate_protosemseg.py
│   └── evaluate_baseline.py
├── checkpoints/
│   ├── adasemseg/       # AdaSemSeg paper checkpoints
│   ├── protosemseg/     # ProtoSemSeg paper checkpoints
│   ├── simclr/          # SimCLR weights used for image-encoder init
│   ├── scenarios.json   # Scenario-to-checkpoint map
│   └── checkpoints_index.json
├── data/
│   └── splits/          # Train/val/test split files
├── docs/
│   ├── figures/         # Paper figures
│   ├── html_summaries/  # Per-method HTML summaries
│   ├── methods/         # Markdown method descriptions
│   └── running.md       # Centralized training/evaluation commands
├── README.md
├── REPRODUCE.md
├── requirements.txt
└── environment.yml
```

## Quick evaluation

Evaluate a published AdaSemSeg checkpoint using its paper scenario from `checkpoints/scenarios.json`:

```bash
python scripts/evaluate_adasemseg.py --scenario simclr_5-shot_sampling_f3 --device cuda:0
```

ProtoSemSeg evaluation code is included, but its trained checkpoints are not shipped in this repository push (they will be available via IEEE DataPort). See [`REPRODUCE.md`](REPRODUCE.md) for details.

See [`REPRODUCE.md`](REPRODUCE.md) and [`docs/running.md`](docs/running.md) for the full command reference.

## Citation

If you use this code, please cite:

```bibtex
@article{saha2025adasemseg,
  title={AdaSemSeg: An Adaptive Few-shot Semantic Segmentation of Seismic Facies},
  author={Saha, Surojit and Whitaker, Ross},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025}
}
```

## License

[Add your license here]
