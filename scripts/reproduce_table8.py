#!/usr/bin/env python
"""
Reproduce Table 8: Class-wise IoU/F1 sensitivity analysis for AdaSemSeg.

Runs the unified AdaSemSeg evaluator and prints per-class IoU and F1 from the
metrics.json output.
"""
import argparse
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Reproduce Table 8")
    parser.add_argument("--checkpoint", required=True, help="Trained AdaSemSeg checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shots", type=int, default=5, choices=[1, 5])
    parser.add_argument("--use_nearest_slice", action="store_true")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Class names to evaluate (default: all)")
    parser.add_argument("--output_dir", default="./evaluation_results")
    parser.add_argument("--data_root", default=os.environ.get("ADASEMSEG_DATA_ROOT", "./data"))
    args = parser.parse_args()

    os.environ.setdefault("ADASEMSEG_DATA_ROOT", args.data_root)

    cmd = [
        sys.executable, os.path.join(REPO_ROOT, "methods", "adasemseg", "evaluate.py"),
        "--checkpoint", args.checkpoint,
        "--shots", str(args.shots),
        "--output_dir", args.output_dir,
        "--device", args.device,
    ]
    if args.use_nearest_slice:
        cmd.append("--use_nearest_slice")
    if args.classes:
        cmd.extend(["--classes"] + args.classes)

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "r") as f:
        data = json.load(f)

    print("\nClass-wise IoU / F1 sensitivity")
    print("=" * 70)
    for class_name, metrics in data["per_class"].items():
        print(f"\n{class_name}")
        ious = metrics.get("IoU_per_class", [])
        f1s = metrics.get("F1_per_class", [])
        print(f"  IoU per class: {[round(x, 3) if x is not None else 'NA' for x in ious]}")
        print(f"  F1  per class: {[round(x, 3) if x is not None else 'NA' for x in f1s]}")
    print(f"\nFull metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
