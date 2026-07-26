#!/usr/bin/env python
"""
Unified evaluation script for ProtoSemSeg.

This is a thin wrapper around methods/protosemseg/Evaluation_sampling/Main.py.
It supports loading a paper scenario from checkpoints/scenarios.json so users
can evaluate a published checkpoint without manually locating it.
"""

import argparse
import json
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_SCRIPT = os.path.join(REPO_ROOT, "methods", "protosemseg", "Evaluation_sampling", "Main.py")
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


def extract_run_id(checkpoint_path):
    match = re.search(r"Run_(\d+)", checkpoint_path)
    if match:
        return int(match.group(1))
    return 1


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained ProtoSemSeg model")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Directory containing bestmodel.pth.tar")
    parser.add_argument("--scenario", default=None,
                        help="Scenario key from checkpoints/scenarios.json, e.g. simclr_1-shot_sampling_f3")
    parser.add_argument("--run_id", type=int, default=None, help="Run id for output naming")
    parser.add_argument("--shots", type=int, default=None, choices=[1, 5], help="Number of support examples")
    parser.add_argument("--eval_mode", default="test", choices=["val", "test"], help="Evaluate on val or test split")
    parser.add_argument("--source_class", type=str, default=None, help="Class name to evaluate")
    parser.add_argument("--device", default="cuda:0", help="Device string")
    parser.add_argument("--data_root", default=os.environ.get("ADASEMSEG_DATA_ROOT", "./data"), help="Data root")
    parser.add_argument("--visualize", type=int, default=0, help="Set to 1 to save visualizations")
    args = parser.parse_args()

    os.environ.setdefault("ADASEMSEG_DATA_ROOT", args.data_root)

    scenario_cfg = None
    if args.scenario:
        scenario_cfg = load_scenario("protosemseg", args.scenario)
        ckpt_path = resolve_path(scenario_cfg["checkpoint"])
        args.checkpoint_dir = os.path.dirname(ckpt_path)
        if args.run_id is None:
            args.run_id = extract_run_id(ckpt_path)
        if args.shots is None:
            args.shots = scenario_cfg["shots"]
        classes = scenario_cfg["classes"]
        print(f"Loaded scenario '{args.scenario}': checkpoint_dir={args.checkpoint_dir}, "
              f"run_id={args.run_id}, shots={args.shots}, classes={classes}")
    else:
        if args.checkpoint_dir is None:
            parser.error("Either --checkpoint_dir or --scenario is required")
        if args.run_id is None:
            args.run_id = 1
        if args.shots is None:
            args.shots = 5
        classes = [args.source_class] if args.source_class else []

    if not os.path.isdir(args.checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint_dir}")

    if args.source_class:
        classes = [args.source_class]
    if not classes:
        raise ValueError("No classes to evaluate. Provide --source_class or --scenario.")

    errors = []
    for class_name in classes:
        cmd = [
            sys.executable, EVAL_SCRIPT,
            "--visualize", str(args.visualize),
            "--run_id", str(args.run_id),
            "--shots", str(args.shots),
            "--eval_mode", args.eval_mode,
            "--source_class", class_name,
            "--best_model",
            "--checkpoint_dir", args.checkpoint_dir,
            "--device", args.device,
        ]
        print("Running:", " ".join(cmd))
        ret = subprocess.run(cmd, cwd=REPO_ROOT).returncode
        if ret != 0:
            errors.append((class_name, ret))

    if errors:
        print("Errors during evaluation:", errors, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
