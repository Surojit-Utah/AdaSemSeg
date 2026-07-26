# Ablation Studies

## Initialization ablation (Table 5)
Compares a randomly initialized image encoder against the SimCLR-initialized image encoder for AdaSemSeg.

- **Random init:** Use `Random_init/Main.py` without `--img_enc_checkpoint`.
- **SimCLR init:** Use `methods/adasemseg/Main.py` with `--img_enc_checkpoint checkpoints/simclr/simclr_resnet50_run2_epoch10.pth.tar`.

## Data augmentation ablation (Table 6)
Trains AdaSemSeg with individual augmentation techniques:
- None
- RandomRotate
- RandomHorizontalFlip
- GaussianBlur
- GaussNoise
- All combined

## Architecture ablation (Section 5.2.6)
Baseline-2 (encoder-decoder without mask encoder or GP regression) serves as the architecture ablation.

## Class imbalance sensitivity (Table 8)
Per-class IoU and F1 scores are reported to analyze sensitivity to under-represented classes.
