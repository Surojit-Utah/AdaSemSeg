# AdaSemSeg

## Overview
AdaSemSeg turns multi-class seismic facies segmentation into multiple binary tasks using a shared model. It uses a ResNet50 image encoder, a U-Net-like mask encoder, Gaussian process (GP) regression in latent space, and a decoder that fuses GP outputs with skip connections.

## Paper sections
- Section 4 — The Proposed Method: AdaSemSeg
- Section 5.2 — Experimental Results
- Appendix — Architecture and training details

## Key design points
- Multi-class labels are decomposed into class-wise binary masks.
- The image encoder, mask encoder, and decoder are shared across classes.
- Meta-training uses only source datasets; the target dataset is never fine-tuned.
- Image encoder is initialized with SimCLR-trained weights.

## Code location
- `methods/adasemseg/Main.py` — entry point
- `methods/adasemseg/models/DGP_resnet_unet.py` — model
- `methods/adasemseg/data/Datasets.py` — task construction
- `methods/adasemseg/train/DGP_trainer.py` — training loop
- `methods/adasemseg/kernels/gp_kernels.py` — GP kernels

## Training
```bash
cd methods/adasemseg
python Main.py --run_id 1 --shots 5 --train \
    --img_enc_checkpoint ../../checkpoints/simclr/simclr_resnet50_run2_epoch10.pth.tar \
    --device cuda:0
```

Switch source/target datasets by editing `methods/adasemseg/config/local_config.py` or by using wrapper scripts.

## Known gaps
- The `--test` argument in `Main.py` is currently unused; unified evaluation is under `scripts/evaluate_adasemseg.py` (stub) and the legacy `Evaluation/` folders.
