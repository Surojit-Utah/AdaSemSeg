#!/usr/bin/env python
"""
Reproduce Table VII: class-wise IoU/F1 sensitivity to under-represented classes,
comparing AdaSemSeg against ProtoSemSeg (paper Appendix C, "Sensitivity to class
imbalance across datasets").

Paper setting: AdaSemSeg trained in the 1-shot setup on source datasets,
evaluated on unseen Parihaka/Penobscot target datasets. Classes making up
<=5% of a target dataset's labeled pixels are the "under-represented" ones
the paper highlights.

Runs the unified AdaSemSeg evaluator to get per-class IoU/F1/class-distribution.
If --protosemseg_checkpoint_dir is given, also runs the ProtoSemSeg evaluator
for the same classes and merges its per-class IoU/F1 into the same table.
"""
import argparse
import json
import os
import pickle
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNDER_REPRESENTED_THRESHOLD = 0.05


def run_adasemseg(args):
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

    with open(os.path.join(args.output_dir, "metrics.json")) as f:
        return json.load(f)["per_class"]


def run_protosemseg(args, class_name):
    """Run ProtoSemSeg's evaluator for one class and load its pickled per-class metrics."""
    cmd = [
        sys.executable, os.path.join(REPO_ROOT, "scripts", "evaluate_protosemseg.py"),
        "--checkpoint_dir", args.protosemseg_checkpoint_dir,
        "--run_id", str(args.protosemseg_run_id),
        "--shots", str(args.shots),
        "--eval_mode", "test",
        "--source_class", class_name,
        "--device", args.device,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    # methods/protosemseg/Evaluation_sampling/Main.py writes results under
    # <cwd>/logs/results/<class>/<shots>-shot/Run_<id>_best_model/test/
    results_dir = os.path.join(
        REPO_ROOT, "logs", "results", class_name, f"{args.shots}-shot",
        f"Run_{args.protosemseg_run_id}_best_model", "test",
    )
    pickle_path = os.path.join(results_dir, "save_metric_scorers_dict.pickle")
    if not os.path.isfile(pickle_path):
        print(f"WARNING: expected ProtoSemSeg results at {pickle_path}, not found. Skipping.")
        return None
    with open(pickle_path, "rb") as f:
        scores = pickle.load(f)

    # Keys are legacy-formatted with a trailing ": " (see
    # methods/protosemseg/Evaluation_sampling/predict/Metric_scores.py).
    classwise_iou = scores.get("classwise_IoU: ", {})
    f1_per_class = scores.get("f1_score: ", [])
    return {
        "IoU_per_class": [classwise_iou.get(i) for i in range(len(f1_per_class))],
        "F1_per_class": list(f1_per_class),
    }


def format_row(label, values, flagged_indices):
    cells = []
    for i, v in enumerate(values):
        if v is None:
            cells.append("NA")
        else:
            marker = "*" if i in flagged_indices else ""
            cells.append(f"{v:.2f}{marker}")
    return f"  {label:<12} " + " ".join(f"{c:>7}" for c in cells)


def main():
    parser = argparse.ArgumentParser(description="Reproduce Table VII")
    parser.add_argument("--checkpoint", required=True, help="Trained AdaSemSeg checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shots", type=int, default=1, choices=[1, 5],
                        help="Paper's Table VII uses 1-shot")
    parser.add_argument("--use_nearest_slice", action="store_true",
                        help="Use for Parihaka, matching the paper's Table I finding")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Class names to evaluate (default: all)")
    parser.add_argument("--output_dir", default="./evaluation_results")
    parser.add_argument("--data_root", default=os.environ.get("ADASEMSEG_DATA_ROOT", "./data"))
    parser.add_argument("--protosemseg_checkpoint_dir", default=None,
                        help="If given, also evaluate ProtoSemSeg on the same classes and compare")
    parser.add_argument("--protosemseg_run_id", type=int, default=1)
    args = parser.parse_args()

    os.environ.setdefault("ADASEMSEG_DATA_ROOT", args.data_root)

    adasemseg_per_class = run_adasemseg(args)

    print("\nClass-wise IoU / F1 sensitivity (paper Table VII style)")
    print("=" * 78)
    for class_name, metrics in adasemseg_per_class.items():
        dist = metrics.get("Class_distribution", [])
        ious = metrics.get("IoU_per_class", [])
        f1s = metrics.get("F1_per_class", [])

        # Index 0 is background; the paper's class columns start at 1.
        # Classes entirely absent (distribution == 0, i.e. not present in this
        # target/direction at all) are NOT flagged as "under-represented" -- the
        # paper leaves those as plain NA in IoU/F1 rather than highlighting them;
        # "under-represented" means genuinely rare but present (0 < dist <= 5%).
        facies_dist = dist[1:]
        flagged = {i for i, v in enumerate(facies_dist) if v is not None and 0 < v <= UNDER_REPRESENTED_THRESHOLD}

        print(f"\n{class_name}")
        print(format_row("distribution", facies_dist, flagged))
        print(format_row("AdaSemSeg IoU", ious[1:], flagged))
        print(format_row("AdaSemSeg F1", f1s[1:], flagged))

        if args.protosemseg_checkpoint_dir:
            proto = run_protosemseg(args, class_name)
            if proto:
                print(format_row("ProtoSemSeg IoU", proto["IoU_per_class"], flagged))
                print(format_row("ProtoSemSeg F1", proto["F1_per_class"], flagged))

        if flagged:
            print(f"  (* = under-represented class, <= {UNDER_REPRESENTED_THRESHOLD:.0%} of labeled pixels)")

    print(f"\nFull AdaSemSeg metrics written to {os.path.join(args.output_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
