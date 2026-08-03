# Running AdaSemSeg Experiments

All experiments assume you have downloaded the datasets and weights and set `ADASEMSEG_DATA_ROOT`. See [`REPRODUCE.md`](../REPRODUCE.md) for setup instructions.

## SimCLR pretraining (image-encoder initialization)

SimCLR is not a competing segmentation method; it is the self-supervised technique used to initialize the image encoder of AdaSemSeg, Baseline-2, and ProtoSemSeg.

```bash
cd pretraining/simclr
python run.py --run_id 1 --epochs 10 --batch-size 32 --arch resnet50 --device cuda:0
cd ../..
```

The SimCLR checkpoint used in all paper evaluations is downloaded from Zenodo
(`python scripts/download_assets.py --weights`, see [`REPRODUCE.md`](../REPRODUCE.md))
into:
```
checkpoints/simclr/simclr_resnet50_epoch10.pth.tar
```

## AdaSemSeg

### 1-shot training
```bash
cd methods/adasemseg
python Main.py --run_id 1 --shots 1 --train \
    --img_enc_checkpoint ../../checkpoints/simclr/simclr_resnet50_epoch10.pth.tar \
    --device cuda:0
```

### 5-shot training
```bash
python Main.py --run_id 1 --shots 5 --train \
    --img_enc_checkpoint ../../checkpoints/simclr/simclr_resnet50_epoch10.pth.tar \
    --device cuda:0
```

### Ablation flags
- Random initialization (Table IV): add `--random_init`
- Single augmentation (Table V): add `--augmentation RandomRotate` (also accepts `none`, `HFlip`, `GaussianBlur`, `GaussNoise`, `all`)
- Alternate encoder architecture: add `--img_enc_type unet` to use the plain UNet encoder (`models/DGP_unet.py`) instead of the default ResNet50; no `--img_enc_checkpoint` applies in this mode. Ported from the original research workspace; not tied to a specific verified paper table.

Example:
```bash
python Main.py --run_id 1 --shots 1 --train --random_init --device cuda:0
python Main.py --run_id 1 --shots 1 --train --augmentation HFlip --device cuda:0
```

### Evaluation (test split)

The simplest way to evaluate a published checkpoint is to use its scenario key from `checkpoints/scenarios.json`:

```bash
python scripts/evaluate_adasemseg.py --scenario simclr_5-shot_sampling_f3 --device cuda:0
```

#### K-shot random support (F3 and Penobscot in Table I)
```bash
python scripts/evaluate_adasemseg.py --scenario simclr_5-shot_sampling_f3 --device cuda:0
python scripts/evaluate_adasemseg.py --scenario simclr_5-shot_sampling_penobscot --device cuda:0
```

#### Nearest-slice support (Parihaka in Table I)
```bash
python scripts/evaluate_adasemseg.py --scenario simclr_5-shot_nearest_slice_parihaka --device cuda:0
```

You can also evaluate directly from `methods/adasemseg`:
```bash
cd methods/adasemseg
python evaluate.py --checkpoint checkpoints/bestmodel.pth.tar --shots 5 --use_nearest_slice
```

Metrics are written to `./evaluation_results/metrics.json`.

### Switching source/target datasets
Pass `--classes` to `Main.py` and `evaluate.py`:

```bash
# Parihaka as target (train on F3 + Penobscot)
python Main.py --train --classes f3_facies_data_inline f3_facies_data_crossline \
    penobscot_facies_data_inline penobscot_facies_data_crossline --device cuda:0

# F3 as target (train on Parihaka + Penobscot)
python Main.py --train --classes parihaka_facies_data_inline parihaka_facies_data_crossline \
    penobscot_facies_data_inline penobscot_facies_data_crossline --device cuda:0

# Penobscot as target (train on F3 + Parihaka)
python Main.py --train --classes f3_facies_data_inline f3_facies_data_crossline \
    parihaka_facies_data_inline parihaka_facies_data_crossline --device cuda:0
```

Alternatively, edit the `classes` list in `methods/adasemseg/config/local_config.py`.

## Baselines

### Baseline-1 (AdaSemSeg trained only on target)
Baseline-1 is *not* a separate architecture -- per the paper it's "the AdaSemSeg
trained only on the target dataset" (no leave-one-out, no other source datasets).
There's no dedicated script for it: use `methods/adasemseg/Main.py` itself, with
`--classes` restricted to the target dataset's own classes instead of a source
pair. `methods/baselines/Main.py` (below) always builds the Baseline-2
architecture regardless of flags -- it does not implement Baseline-1.
```bash
cd methods/adasemseg
python Main.py --run_id 1 --shots 5 --train --freeze_bn \
    --classes parihaka_facies_data_inline parihaka_facies_data_crossline \
    --device cuda:0
```
Evaluate the same way as any AdaSemSeg checkpoint, via `scripts/evaluate_adasemseg.py --checkpoint <CKPT> --classes parihaka_facies_data_inline parihaka_facies_data_crossline`.

### Baseline-2 (ResNet-UNet)
```bash
cd methods/baselines
python Main.py --run_id 1 --train --img_enc_type resnet --freeze_bn --device cuda:0
```
Evaluate with:
```bash
python scripts/evaluate_baseline.py --checkpoint_dir methods/baselines/logs/checkpoints/Run_1 \
    --source_class f3_facies_data_inline --eval_mode test --device cuda:0
```

### Transfer learning (source pretrain → target fine-tune)
Two steps: first train Baseline-2 on the source datasets (as above, with
`--classes` set to the two source datasets), then fine-tune that checkpoint on
a handful of target slices:
```bash
cd methods/baselines
python Transfer_learning\Main.py --run_id 1 --train --train_indices 5 \
    --checkpoint <source_baseline2_ckpt> --img_enc_type resnet --freeze_bn --device cuda:0
```
Evaluate with:
```bash
python scripts/evaluate_baseline.py --transfer_learning --train_indices 5 \
    --checkpoint_dir methods/baselines/Transfer_learning/logs/checkpoints/Train_slices_5/Run_1 \
    --source_class parihaka_facies_data_inline --eval_mode test --device cuda:0
```

## ProtoSemSeg

### Training
```bash
cd methods/protosemseg
python Main.py --run_id 1 --shots 5 --train \
    --img_enc_checkpoint ../../checkpoints/simclr/simclr_resnet50_epoch10.pth.tar \
    --device cuda:0
```

### Evaluation
```bash
python Evaluation_sampling\Main.py --visualize 1 --run_id 1 --shots 5 \
    --eval_mode val --checkpoint_dir <CKPT_DIR> --best_model --device cuda:0
```

Or use the unified wrapper with a scenario key (requires the corresponding checkpoint directory; ProtoSemSeg checkpoints are downloaded from Zenodo into `checkpoints/protosemseg/<dataset>/<shots>-shot/`, see the README's "Model weights" section):
```bash
python scripts/evaluate_protosemseg.py --scenario simclr_5-shot_sampling_f3 --device cuda:0
```

## Evaluation notes
- Class labels start at index 0 during training (required for cross-entropy loss).
- During evaluation, class labels start at index 1 to match the original implementations.
- For Penobscot, the region above the top horizon is indicated by -1 in training and 0 in evaluation.

## Reproduction scripts

For paper-specific reproduction, use the wrapper scripts:

```bash
# Table I: main AdaSemSeg results
python scripts/reproduce_table1.py --device cuda:0

# Table II: AdaSemSeg vs baselines trained on target
python scripts/reproduce_table2.py --device cuda:0

# Table III: AdaSemSeg vs ProtoSemSeg vs transfer learning
python scripts/reproduce_table3.py --device cuda:0

# Table IV: random init vs SimCLR init on Parihaka
python scripts/reproduce_table4.py --shots 5 --device cuda:0

# Table V: augmentation ablation on Parihaka (1-shot)
python scripts/reproduce_table5.py --device cuda:0

# Table VI: inference-time comparison (requires trained checkpoints)
python scripts/reproduce_table6.py --checkpoint <CKPT> --device cuda:0

# Table VII: class-wise IoU/F1 sensitivity
python scripts/reproduce_table7.py --checkpoint <CKPT> --device cuda:0

# Figures 12/17/18 (F3/Parihaka/Penobscot prediction patches; Parihaka and Penobscot figures are in Appendix C)
python scripts/reproduce_figures.py --checkpoint <CKPT> --target f3 --device cuda:0
python scripts/reproduce_figures.py --checkpoint <CKPT> --target parihaka --device cuda:0
python scripts/reproduce_figures.py --checkpoint <CKPT> --target penobscot --device cuda:0

# Assemble the saved patches into paper-style qualitative panel figures
python scripts/assemble_figures.py --patches_dir <output_dir>/prediction_patches
```

Add `--run_commands` to Tables 2/3 to execute the printed commands.
Add `--skip_training` to Tables 1/5/6 to evaluate existing checkpoints only.
Prediction patches from `reproduce_figures.py` are written to `<output_dir>/prediction_patches/`.
