# Baselines and Transfer Learning

## Overview
This folder implements the baseline methods and transfer-learning experiments used for comparison in the paper.

## Methods

### Baseline-1
AdaSemSeg architecture trained **only on the target dataset**. This shows the performance ceiling when the model is allowed to see target data, unlike the few-shot AdaSemSeg which is trained only on source data.

### Baseline-2
A standard encoder-decoder segmentation network (ResNet-UNet or UNet) trained **only on the target dataset**. This is the conventional supervised segmentation baseline.

### Transfer Learning
Baseline-2 is first pretrained on patches from the **source datasets**, then fine-tuned on a handful of annotated slices from the **target dataset**. This represents the classic transfer-learning approach.

## Code location
- `methods/baselines/Main.py` — training entry point for Baseline-1 and Baseline-2
- `methods/baselines/Transfer_learning/Main.py` — transfer-learning entry point
- `methods/baselines/Evaluation/Main.py` — evaluation entry point
- `methods/baselines/models/DGP_resnet_unet.py` — ResNet-UNet
- `methods/baselines/models/DGP_unet.py` — UNet

## Commands
See [`docs/running.md`](../running.md#baselines) for all commands.
