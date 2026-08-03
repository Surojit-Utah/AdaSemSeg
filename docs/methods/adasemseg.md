# AdaSemSeg

## Overview
AdaSemSeg turns multi-class seismic facies segmentation into multiple binary tasks using a shared model. It uses a ResNet50 image encoder, a U-Net-like mask encoder, Gaussian process (GP) regression in latent space, and a decoder that fuses GP outputs with skip connections.

## Paper sections
- Section IV — The Proposed Method: AdaSemSeg
- Section V-B — Experimental Results
- Appendix A/B — Architecture and training details, Appendix C — Additional results (class-imbalance sensitivity, qualitative predictions)

## Key design points
- Multi-class labels are decomposed into class-wise binary masks.
- The image encoder, mask encoder, and decoder are shared across classes.
- Meta-training uses only source datasets; the target dataset is never fine-tuned.
- Image encoder is initialized with SimCLR-trained weights.

## Code location
- `methods/adasemseg/Main.py` — entry point (`--train` and `--test`)
- `methods/adasemseg/evaluate.py` — unified evaluator (also used by `scripts/evaluate_adasemseg.py`)
- `methods/adasemseg/models/DGP_resnet_unet.py` — model (ResNet50 encoder, the architecture used throughout the paper)
- `methods/adasemseg/models/DGP_unet.py` — alternate plain-UNet encoder variant (no ImageNet/SimCLR pretraining path; select with `--img_enc_type unet`). Ported from the original research workspace for completeness; we don't have confirmed provenance tying it to a specific published table/figure, so treat it as available-but-unverified rather than paper-reproducing.
- `methods/adasemseg/predict/DGP_evaluator.py` — qualitative figure-assembly helper (`show_combined_model_pred_images`), used by `scripts/assemble_figures.py`
- `methods/adasemseg/qol/erase_data.py` — deletes a run's logs/checkpoints/exp_spec (e.g. to redo a run from scratch)
- `methods/adasemseg/data/Datasets.py` — task construction
- `methods/adasemseg/train/DGP_trainer.py` — training loop
- `methods/adasemseg/kernels/gp_kernels.py` — GP kernels

## Training
```bash
cd methods/adasemseg
python Main.py --run_id 1 --shots 5 --train \
    --img_enc_checkpoint ../../checkpoints/simclr/simclr_resnet50_epoch10.pth.tar \
    --device cuda:0
```

Switch source/target datasets by editing `methods/adasemseg/config/local_config.py`, passing `--classes`, or by using wrapper scripts.

Pass `--img_enc_type unet` (both `Main.py` and `evaluate.py`/`scripts/evaluate_adasemseg.py`) to use the alternate `DGP_unet` encoder instead of the default ResNet50; `--img_enc_checkpoint` is ignored in that mode since there's no pretraining path for it.

## Training configuration (paper Appendix B, "Training the AdaSemSeg")
- **GP regression:** configuration follows DGPNet (`kernels/gp_kernels.py`, cited in-code).
- **Training data:** 8,100 / 7,664 / 8,206 patches extracted from Penobscot / F3 / Parihaka respectively (leave-one-out: two datasets are source/meta-train, the third is the unseen target).
- **Augmentations** (`data/Datasets.py`'s `aug_dict`, `data/transform.py`): RandomRotate in [-20°, 20°], RandomHorizontalFlip, GaussianBlur with σ ∈ [0.1, 2.0], GaussNoise with variance ∈ [1e-4, 5e-2]. The same ranges are used in `methods/protosemseg/` and `methods/baselines/` (including `Transfer_learning/`) for a fair comparison, per the paper's "Training the Baselines and Competing methods" section.
- **Optimizer:** AdamW, lr = 5e-5, weight decay = 1e-3.
- **LR scheduler:** `ReduceLROnPlateau`, factor = 0.25, patience = 5 epochs (on validation loss).
- **Image encoder init:** SimCLR-trained weights (see [`docs/methods/simclr.md`](simclr.md)).

## Evaluation and figures
`scripts/evaluate_adasemseg.py --scenario <name>` runs the unified evaluator against a `checkpoints/scenarios.json` entry. Passing `--save_predictions` to `methods/adasemseg/evaluate.py` (used internally by `scripts/reproduce_figures.py`) saves per-patch predictions, which `scripts/assemble_figures.py` turns into paper-style qualitative panel figures.
