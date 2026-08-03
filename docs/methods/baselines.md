# Baselines and Transfer Learning

## Overview
This folder implements the baseline methods and transfer-learning experiments used for comparison in the paper.

## Methods

### Baseline-1
AdaSemSeg architecture trained **only on the target dataset**. This shows the performance ceiling when the model is allowed to see target data, unlike the few-shot AdaSemSeg which is trained only on source data. There is no dedicated script for this in `methods/baselines/` — it's `methods/adasemseg/Main.py` invoked with `--classes` restricted to the target dataset's own classes (no leave-one-out). See [`docs/running.md`](../running.md#baselines).

### Baseline-2
A standard encoder-decoder segmentation network (ResNet-UNet or UNet) trained **only on the target dataset**. This is the conventional supervised segmentation baseline. Implemented in this folder.

### Transfer Learning
Baseline-2 is first pretrained on patches from the **source datasets**, then fine-tuned on a handful of annotated slices from the **target dataset**. This represents the classic transfer-learning approach.

## Code location
- `methods/adasemseg/Main.py` — Baseline-1 training entry point (`--classes` restricted to target dataset)
- `methods/baselines/Main.py` — Baseline-2 training entry point (always builds the plain ResNet-UNet/UNet architecture, regardless of flags)
- `methods/baselines/Transfer_learning/Main.py` — transfer-learning entry point
- `methods/baselines/Evaluation/Main.py` — Baseline-2 evaluation entry point
- `methods/baselines/Transfer_learning/Evaluation/Main.py` — transfer-learning evaluation entry point
- `methods/baselines/models/DGP_resnet_unet.py` — ResNet-UNet
- `methods/baselines/models/DGP_unet.py` — UNet

## Commands
See [`docs/running.md`](../running.md#baselines) for all commands.
