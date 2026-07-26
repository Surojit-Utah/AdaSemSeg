#!/usr/bin/env python
"""
Unified evaluation script for Baseline-2 and transfer-learning models.
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_SCRIPT = os.path.join(REPO_ROOT, "methods", "baselines", "Evaluation", "Main.py")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained baseline model")
    parser.add_argument("--checkpoint_dir", required=True, help="Directory containing bestmodel.pth.tar")
    parser.add_argument("--source_class", required=True,
                        help="Class/dataset to evaluate, e.g. f3_facies_data_inline")
    parser.add_argument("--eval_mode", default="test", choices=["test", "val"], help="Evaluation split")
    parser.add_argument("--run_id", type=int, default=1, help="Run ID")
    parser.add_argument("--best_model", action="store_true", default=True,
                        help="Use bestmodel.pth.tar")
    parser.add_argument("--device", default="cuda:0", help="Device string")
    parser.add_argument("--data_root", default=os.environ.get("ADASEMSEG_DATA_ROOT", "./data"), help="Data root")
    args = parser.parse_args()

    os.environ.setdefault("ADASEMSEG_DATA_ROOT", args.data_root)

    cmd = [
        sys.executable, EVAL_SCRIPT,
        "--visualize", "0",
        "--source_class", args.source_class,
        "--eval_mode", args.eval_mode,
        "--run_id", str(args.run_id),
        "--checkpoint_dir", args.checkpoint_dir,
        "--device", args.device,
    ]
    if args.best_model:
        cmd.append("--best_model")

    print("Running:", " ".join(cmd))
    sys.exit(subprocess.run(cmd, cwd=os.path.join(REPO_ROOT, "methods", "baselines")).returncode)


if __name__ == "__main__":
    main()
