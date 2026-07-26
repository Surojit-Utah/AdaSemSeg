# ProtoSemSeg

## Overview
ProtoSemSeg is the prototype-based few-shot semantic segmentation competing method. It uses a ResNet50 encoder plus a U-Net-like decoder. For each class, foreground and background prototypes are computed from support masks via masked average pooling. Query features are compared to prototypes using cosine similarity, and the resulting similarity maps are decoded into binary masks per class. The final multi-class mask is assembled by argmax over class-wise predictions.

## Code location
- `methods/protosemseg/Main.py` — training entry point
- `methods/protosemseg/train/DGP_trainer.py` — training loop
- `methods/protosemseg/predict/DGP_evaluator.py` — inference
- `methods/protosemseg/models/` — model implementation

## Commands
See [`docs/running.md`](../running.md#protosemseg) for all commands.

## Known issues
- The evaluation script prompts interactively for `source_class`. This should be replaced by a CLI flag for automated reproduction.
