#!/usr/bin/env python
"""
Unified evaluation script for Baseline-2 and transfer-learning models.
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_EVAL_SCRIPT = os.path.join(REPO_ROOT, "methods", "baselines", "Evaluation", "Main.py")
TRANSFER_LEARNING_EVAL_SCRIPT = os.path.join(
    REPO_ROOT, "methods", "baselines", "Transfer_learning", "Evaluation", "Main.py")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Baseline-2 or transfer-learning model")
    parser.add_argument("--checkpoint_dir", required=True, help="Directory containing bestmodel.pth.tar")
    parser.add_argument("--source_class", required=True,
                        help="Class/dataset to evaluate, e.g. f3_facies_data_inline")
    parser.add_argument("--eval_mode", default="test", choices=["test", "val"], help="Evaluation split")
    parser.add_argument("--run_id", type=int, default=1, help="Run ID")
    parser.add_argument("--best_model", action="store_true", default=True,
                        help="Use bestmodel.pth.tar")
    parser.add_argument("--device", default="cuda:0", help="Device string")
    parser.add_argument("--data_root", default=os.environ.get("ADASEMSEG_DATA_ROOT", "./data"), help="Data root")
    parser.add_argument("--transfer_learning", action="store_true",
                        help="Evaluate a fine-tuned transfer-learning checkpoint instead of Baseline-2")
    parser.add_argument("--train_indices", type=str, default=None,
                        help="Required with --transfer_learning: number of target slices the checkpoint "
                             "was fine-tuned on (matches the --train_indices used at training time)")
    args = parser.parse_args()

    os.environ.setdefault("ADASEMSEG_DATA_ROOT", args.data_root)

    if args.transfer_learning:
        if not args.train_indices:
            parser.error("--train_indices is required with --transfer_learning")
        eval_script = TRANSFER_LEARNING_EVAL_SCRIPT
        cwd = os.path.join(REPO_ROOT, "methods", "baselines", "Transfer_learning")
    else:
        eval_script = BASELINE_EVAL_SCRIPT
        cwd = os.path.join(REPO_ROOT, "methods", "baselines")

    cmd = [
        sys.executable, eval_script,
        "--visualize", "0",
        "--source_class", args.source_class,
        "--eval_mode", args.eval_mode,
        "--run_id", str(args.run_id),
        "--checkpoint_dir", args.checkpoint_dir,
        "--device", args.device,
    ]
    if args.transfer_learning:
        cmd.extend(["--train_indices", args.train_indices])
    if args.best_model:
        cmd.append("--best_model")

    print("Running:", " ".join(cmd))
    sys.exit(subprocess.run(cmd, cwd=cwd).returncode)


if __name__ == "__main__":
    main()
