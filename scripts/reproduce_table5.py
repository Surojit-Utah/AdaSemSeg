#!/usr/bin/env python
"""
Reproduce Table V: Data-augmentation ablation on Parihaka (1-shot).

Trains AdaSemSeg on Penobscot + F3 using a single augmentation strategy at a
time (or none/all) and evaluates on Parihaka inline/crossline with nearest-slice
support selection.
"""
import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, cwd=REPO_ROOT):
    print("\n" + "=" * 70)
    print("Running:", " ".join(cmd))
    print("=" * 70)
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Reproduce Table V")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training and only run evaluation (expects checkpoints)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--data_root", default=os.environ.get("ADASEMSEG_DATA_ROOT", "./data"))
    args = parser.parse_args()

    os.environ.setdefault("ADASEMSEG_DATA_ROOT", args.data_root)

    method_dir = os.path.join(REPO_ROOT, "methods", "adasemseg")
    simclr_ckpt = os.path.join(REPO_ROOT, "checkpoints", "simclr", "simclr_resnet50_epoch10.pth.tar")

    source_classes = [
        "f3_facies_data_inline",
        "f3_facies_data_crossline",
        "penobscot_facies_data_inline",
        "penobscot_facies_data_crossline",
    ]

    augmentations = ["none", "RandomRotate", "HFlip", "GaussianBlur", "GaussNoise", "all"]

    for aug in augmentations:
        run_id = 1
        train_dir = os.path.join(method_dir, "logs", "table5", aug)
        os.makedirs(train_dir, exist_ok=True)

        if not args.skip_training:
            run([
                sys.executable, "Main.py",
                "--run_id", str(run_id),
                "--shots", "1",
                "--train",
                "--classes",
            ] + source_classes + [
                "--augmentation", aug,
                "--img_enc_checkpoint", simclr_ckpt,
                "--device", args.device,
            ], cwd=method_dir)

        ckpt = os.path.join(train_dir, "checkpoints", "1-shot", f"Run_{run_id}", "bestmodel.pth.tar")
        if not os.path.exists(ckpt):
            print(f"WARNING: checkpoint {ckpt} not found; skipping {aug} evaluation.")
            continue

        out_dir = os.path.join(train_dir, "eval")
        run([
            sys.executable, "evaluate.py",
            "--checkpoint", ckpt,
            "--img_enc_checkpoint", simclr_ckpt,
            "--shots", "1",
            "--use_nearest_slice",
            "--classes", "parihaka_facies_data_inline", "parihaka_facies_data_crossline",
            "--output_dir", out_dir,
            "--device", args.device,
        ], cwd=method_dir)

    print("\nTable V reproduction complete. See evaluation outputs under methods/adasemseg/logs/table5/")


if __name__ == "__main__":
    main()
