#!/usr/bin/env python
"""
Reproduce Table 5: Effect of image-encoder initialization on Parihaka.

Trains AdaSemSeg on F3 + Penobscot (source datasets) and evaluates on
Parihaka inline/crossline using both random and SimCLR initialization.
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
    parser = argparse.ArgumentParser(description="Reproduce Table 5")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shots", type=int, default=5, choices=[1, 5])
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training and only run evaluation (expects checkpoints)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--data_root", default=os.environ.get("ADASEMSEG_DATA_ROOT", "./data"))
    args = parser.parse_args()

    os.environ.setdefault("ADASEMSEG_DATA_ROOT", args.data_root)

    method_dir = os.path.join(REPO_ROOT, "methods", "adasemseg")
    simclr_ckpt = os.path.join(REPO_ROOT, "checkpoints", "simclr", "simclr_resnet50_epoch10.pth.tar")

    # Source datasets for Table 5 (Parihaka is the held-out target)
    source_classes = [
        "f3_facies_data_inline",
        "f3_facies_data_crossline",
        "penobscot_facies_data_inline",
        "penobscot_facies_data_crossline",
    ]

    # 1) SimCLR initialization
    simclr_train_dir = os.path.join(method_dir, "logs", "table5", f"simclr_{args.shots}shot")
    os.makedirs(simclr_train_dir, exist_ok=True)
    if not args.skip_training:
        run([
            sys.executable, "Main.py",
            "--run_id", "1",
            "--shots", str(args.shots),
            "--train",
            "--classes",
        ] + source_classes + [
            "--img_enc_checkpoint", simclr_ckpt,
            "--device", args.device,
        ], cwd=method_dir)
    simclr_ckpt_path = os.path.join(simclr_train_dir, "checkpoints", f"{args.shots}-shot", "Run_1", "bestmodel.pth.tar")

    # 2) Random initialization
    random_train_dir = os.path.join(method_dir, "logs", "table5", f"random_{args.shots}shot")
    os.makedirs(random_train_dir, exist_ok=True)
    if not args.skip_training:
        run([
            sys.executable, "Main.py",
            "--run_id", "1",
            "--shots", str(args.shots),
            "--train",
            "--classes",
        ] + source_classes + [
            "--random_init",
            "--device", args.device,
        ], cwd=method_dir)
    random_ckpt_path = os.path.join(random_train_dir, "checkpoints", f"{args.shots}-shot", "Run_1", "bestmodel.pth.tar")

    # 3) Evaluate both on Parihaka (nearest-slice support)
    for name, ckpt, init_flag in [
        ("simclr", simclr_ckpt_path, False),
        ("random", random_ckpt_path, True),
    ]:
        if not os.path.exists(ckpt):
            print(f"WARNING: checkpoint {ckpt} not found; skipping {name} evaluation.")
            continue
        out_dir = os.path.join(method_dir, "logs", "table5", f"{name}_{args.shots}shot_eval")
        cmd = [
            sys.executable, "evaluate.py",
            "--checkpoint", ckpt,
            "--shots", str(args.shots),
            "--use_nearest_slice",
            "--classes", "parihaka_facies_data_inline", "parihaka_facies_data_crossline",
            "--output_dir", out_dir,
            "--device", args.device,
        ]
        if not init_flag:
            cmd.extend(["--img_enc_checkpoint", simclr_ckpt])
        else:
            cmd.append("--random_init")
        run(cmd, cwd=method_dir)

    print("\nTable 5 reproduction complete. See evaluation outputs under methods/adasemseg/logs/table5/")


if __name__ == "__main__":
    main()
