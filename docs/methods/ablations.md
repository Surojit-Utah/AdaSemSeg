# Ablation Studies

## Initialization ablation (Table IV)
Compares a randomly initialized image encoder against the SimCLR-initialized image encoder for AdaSemSeg.

- **Random init:** Use `methods/adasemseg/Main.py` with `--random_init` (omit `--img_enc_checkpoint`).
- **SimCLR init:** Use `methods/adasemseg/Main.py` with `--img_enc_checkpoint checkpoints/simclr/simclr_resnet50_epoch10.pth.tar`.

`scripts/reproduce_table4.py` retrains both variants from scratch rather than relying on a pre-shipped checkpoint. No pre-trained random-init checkpoint is currently shipped or verified against the paper's Table IV numbers.

## Data augmentation ablation (Table V)
Trains AdaSemSeg with individual augmentation techniques:
- None
- RandomRotate
- RandomHorizontalFlip
- GaussianBlur
- GaussNoise
- All combined

## Architecture ablation (Section V-B6)
Baseline-2 (encoder-decoder without mask encoder or GP regression) serves as the architecture ablation.

## Class imbalance sensitivity (Table VII)
Per-class IoU and F1 scores are reported to analyze sensitivity to under-represented classes.
