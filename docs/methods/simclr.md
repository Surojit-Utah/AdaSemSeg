# SimCLR Pretraining (Image-Encoder Initialization)

SimCLR is not a competing segmentation method. It is the self-supervised pretraining technique used to initialize the ResNet50 image encoder of AdaSemSeg, Baseline-2, and ProtoSemSeg.

## Paper context
SimCLR is trained on unlabeled seismic patches and only the backbone weights (not the projection head) are reused for downstream segmentation methods. The paper reports top-1 accuracy of 93.75% and top-5 accuracy of 98.44% for the trained encoder. An ablation compares random encoder initialization vs SimCLR initialization on the Parihaka dataset under 1-shot and 5-shot settings.

## Training details from paper
- **Data:** 35,648 patches from F3, Penobscot, and Parihaka (all unlabeled).
- **Augmentations** for positive/negative pairs:
  - Random rotation in [-20°, 20°]
  - Random horizontal flip
  - Gaussian blur with σ ∈ [0.1, 2.0]
  - Gaussian noise with variance ∈ [1e-4, 5e-2]
  - Random crop with resize
  - Brightness jitter ∈ [0.5, 1.5]
  - Contrast jitter ∈ [0.0, 2.0]
- **Model:** ResNet50 backbone + 2-layer projection head (output dim = 128).
- **Loss:** InfoNCE / NT-Xent with temperature τ = 0.07.
- **Optimizer:** Adam, lr = 3e-4, weight decay = 1e-4.
- **Batch size:** 32.
- **Epochs:** 10.
- **Reported accuracy:** top-1 = 93.75%, top-5 = 98.44%.

## Code location
- `pretraining/simclr/run.py` — main entry point
- `pretraining/simclr/simclr.py` — training loop and InfoNCE loss
- `pretraining/simclr/models/resnet_simclr.py` — model
- `pretraining/simclr/data/Datasets.py` — dataset and augmentations

## Reproducing
```bash
cd pretraining/simclr
python run.py --run_id 1 --epochs 10 --batch-size 32 --arch resnet50 --device cuda:0
cd ../..
```

## Trained weights used in evaluations
All downstream evaluations use:
```
checkpoints/simclr/simclr_resnet50_run2_epoch10.pth.tar
```
This corresponds to `Contrastive_learning/SimCLR_debug/logs/checkpoints/Run_2/checkpoint_0010.pth.tar` from the original research folder.

**Note:** Run_1 in the original logs matches the paper's reported top-1/top-5 values more closely, but Run_2 is the checkpoint actually referenced by all downstream `runfile.txt` and experiment spec files.
