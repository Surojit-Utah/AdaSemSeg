#!/usr/bin/env python
"""
Unified evaluation script for AdaSemSeg.

This is a thin wrapper around methods/adasemseg/evaluate.py so users can run
AdaSemSeg evaluation from the repository root.
"""

import argparse
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_SCRIPT = os.path.join(REPO_ROOT, "methods", "adasemseg", "evaluate.py")
SCENARIO_FILE = os.path.join(REPO_ROOT, "checkpoints", "scenarios.json")


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def load_scenario(method, scenario_key):
    if not os.path.isfile(SCENARIO_FILE):
        raise FileNotFoundError(f"Scenario file not found: {SCENARIO_FILE}")
    with open(SCENARIO_FILE) as f:
        scenarios = json.load(f)
    if method not in scenarios:
        raise KeyError(f"Unknown method '{method}'. Available: {list(scenarios.keys())}")
    method_scenarios = scenarios[method]

    init = None
    for k in method_scenarios:
        if scenario_key.startswith(k + "_"):
            init = k
            break
    if init is None:
        raise KeyError(f"Scenario '{scenario_key}' does not start with a known init. "
                       f"Available: {list(method_scenarios.keys())}")

    remainder = scenario_key[len(init) + 1:]
    setting = None
    dataset = None
    for k in sorted(method_scenarios[init].keys(), key=len, reverse=True):
        if remainder.startswith(k + "_"):
            setting = k
            dataset = remainder[len(k) + 1:]
            break
    if setting is None or dataset is None:
        raise KeyError(f"Cannot parse setting/dataset from '{scenario_key}'. "
                       f"Settings: {list(method_scenarios[init].keys())}")
    if dataset not in method_scenarios[init][setting]:
        raise KeyError(f"Unknown dataset '{dataset}' for {method}/{init}/{setting}. "
                       f"Available: {list(method_scenarios[init][setting].keys())}")
    return method_scenarios[init][setting][dataset]


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained AdaSemSeg model")
    parser.add_argument("--checkpoint", default=None, help="Path to bestmodel.pth.tar")
    parser.add_argument("--scenario", default=None,
                        help="Scenario key from checkpoints/scenarios.json, e.g. simclr_5-shot_sampling_f3")
    parser.add_argument("--img_enc_checkpoint",
                        default=os.path.join("checkpoints", "simclr", "simclr_resnet50_run2_epoch10.pth.tar"),
                        help="SimCLR backbone checkpoint")
    parser.add_argument("--random_init", action="store_true",
                        help="Randomly initialize the image encoder (Table 5 ablation)")
    parser.add_argument("--shots", type=int, default=None, choices=[1, 5], help="Number of support examples")
    parser.add_argument("--use_nearest_slice", action="store_true", help="Use nearest support slice")
    parser.add_argument("--eval_mode", default="test", choices=["val", "test"], help="Evaluate on val or test split")
    parser.add_argument("--classes", nargs="+", default=None, help="Class names to evaluate (default: all)")
    parser.add_argument("--device", default="cuda:0", help="Device string")
    parser.add_argument("--data_root", default=os.environ.get("ADASEMSEG_DATA_ROOT", "./data"), help="Data root")
    parser.add_argument("--output_dir", default="./evaluation_results", help="Directory to save metrics.json")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    os.environ.setdefault("ADASEMSEG_DATA_ROOT", args.data_root)

    if args.scenario:
        cfg = load_scenario("adasemseg", args.scenario)
        args.checkpoint = resolve_path(cfg["checkpoint"])
        if args.shots is None:
            args.shots = cfg["shots"]
        if args.classes is None:
            args.classes = cfg["classes"]
        if cfg.get("use_nearest_slice"):
            args.use_nearest_slice = True
        if cfg.get("random_init"):
            args.random_init = True
        print(f"Loaded scenario '{args.scenario}': checkpoint={args.checkpoint}, "
              f"shots={args.shots}, classes={args.classes}")

    if not args.checkpoint:
        parser.error("Either --checkpoint or --scenario is required")
    if args.shots is None:
        args.shots = 5

    cmd = [
        sys.executable, EVAL_SCRIPT,
        "--checkpoint", args.checkpoint,
        "--img_enc_checkpoint", resolve_path(args.img_enc_checkpoint),
        "--shots", str(args.shots),
        "--eval_mode", args.eval_mode,
        "--device", args.device,
        "--data_root", args.data_root,
        "--output_dir", args.output_dir,
        "--seed", str(args.seed),
    ]
    if args.use_nearest_slice:
        cmd.append("--use_nearest_slice")
    if args.random_init:
        cmd.append("--random_init")
    if args.classes:
        cmd.extend(["--classes"] + args.classes)

    print("Running:", " ".join(cmd))
    sys.exit(subprocess.run(cmd, cwd=REPO_ROOT).returncode)


if __name__ == "__main__":
    main()
