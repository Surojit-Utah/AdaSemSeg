#!/usr/bin/env python
"""
Reproduce Table 1: AdaSemSeg main results on F3, Parihaka, and Penobscot.

For each held-out target dataset, trains on the remaining two source datasets
and evaluates on the target. F3 and Penobscot use 5-shot random support;
Parihaka uses nearest-slice support.
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
    parser = argparse.ArgumentParser(description="Reproduce Table 1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training and only run evaluation (expects checkpoints)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--data_root", default=os.environ.get("ADASEMSEG_DATA_ROOT", "./data"))
    args = parser.parse_args()

    os.environ.setdefault("ADASEMSEG_DATA_ROOT", args.data_root)

    method_dir = os.path.join(REPO_ROOT, "methods", "adasemseg")
    simclr_ckpt = os.path.join(REPO_ROOT, "checkpoints", "simclr", "simclr_resnet50_epoch10.pth.tar")

    experiments = [
        {
            "name": "f3_target",
            "sources": ["parihaka_facies_data_inline", "parihaka_facies_data_crossline",
                        "penobscot_facies_data_inline", "penobscot_facies_data_crossline"],
            "targets": ["f3_facies_data_inline", "f3_facies_data_crossline"],
            "shots": 5,
            "use_nearest_slice": False,
        },
        {
            "name": "parihaka_target",
            "sources": ["f3_facies_data_inline", "f3_facies_data_crossline",
                        "penobscot_facies_data_inline", "penobscot_facies_data_crossline"],
            "targets": ["parihaka_facies_data_inline", "parihaka_facies_data_crossline"],
            "shots": 5,
            "use_nearest_slice": True,
        },
        {
            "name": "penobscot_target",
            "sources": ["f3_facies_data_inline", "f3_facies_data_crossline",
                        "parihaka_facies_data_inline", "parihaka_facies_data_crossline"],
            "targets": ["penobscot_facies_data_inline", "penobscot_facies_data_crossline"],
            "shots": 5,
            "use_nearest_slice": False,
        },
    ]

    for exp in experiments:
        print(f"\n=== Experiment: {exp['name']} ===")
        print(f"Sources : {exp['sources']}")
        print(f"Targets : {exp['targets']}")

        run_id = 1
        train_dir = os.path.join(method_dir, "logs", "table1", exp["name"])
        os.makedirs(train_dir, exist_ok=True)

        if not args.skip_training:
            run([
                sys.executable, "Main.py",
                "--run_id", str(run_id),
                "--shots", str(exp["shots"]),
                "--train",
                "--classes",
            ] + exp["sources"] + [
                "--img_enc_checkpoint", simclr_ckpt,
                "--device", args.device,
            ], cwd=method_dir)

        ckpt = os.path.join(train_dir, "checkpoints", f"{exp['shots']}-shot", f"Run_{run_id}", "bestmodel.pth.tar")
        if not os.path.exists(ckpt):
            print(f"WARNING: checkpoint {ckpt} not found; skipping {exp['name']} evaluation.")
            continue

        out_dir = os.path.join(train_dir, "eval")
        cmd = [
            sys.executable, "evaluate.py",
            "--checkpoint", ckpt,
            "--img_enc_checkpoint", simclr_ckpt,
            "--shots", str(exp["shots"]),
            "--classes"] + exp["targets"] + [
            "--output_dir", out_dir,
            "--device", args.device,
        ]
        if exp["use_nearest_slice"]:
            cmd.append("--use_nearest_slice")
        run(cmd, cwd=method_dir)

    print("\nTable 1 reproduction complete. See evaluation outputs under methods/adasemseg/logs/table1/")


if __name__ == "__main__":
    main()
