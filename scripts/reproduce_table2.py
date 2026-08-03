#!/usr/bin/env python
"""
Reproduce Table II: AdaSemSeg vs baselines trained only on the target dataset.

Table II compares three methods:
  - AdaSemSeg (few-shot, trained on source datasets, evaluated on target)
  - Baseline-1 (AdaSemSeg architecture, trained on the target dataset's own
    support slices instead of the two source datasets -- there's no separate
    script for this, it's methods/adasemseg/Main.py with --classes restricted
    to the target)
  - Baseline-2 (regular ResNet-UNet segmentation, methods/baselines/Main.py,
    trained on the target dataset's own support slices)

Because each method has a different training/evaluation protocol, this script
prints the exact commands to run. It can also execute them when
`--run_commands` is set.

Note on checkpoint paths: Main.py (both methods/adasemseg/ and
methods/baselines/) always writes checkpoints under
<cwd>/logs/checkpoints/[<shots>-shot/]Run_<run_id>/bestmodel.pth.tar -- there
is no flag to redirect this. Every training invocation below therefore gets
its own --run_id (tracked per method directory, since that's what scopes the
output path), otherwise later runs would silently overwrite earlier ones'
checkpoints at the same path.
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
    parser = argparse.ArgumentParser(description="Reproduce Table II")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run_commands", action="store_true",
                        help="Actually execute the commands (default: print only)")
    parser.add_argument("--shots", type=int, default=5, choices=[1, 5],
                        help="Support-set size used for evaluation")
    parser.add_argument("--data_root", default=os.environ.get("ADASEMSEG_DATA_ROOT", "./data"))
    args = parser.parse_args()

    os.environ.setdefault("ADASEMSEG_DATA_ROOT", args.data_root)

    targets = {
        "f3": ["f3_facies_data_inline", "f3_facies_data_crossline"],
        "parihaka": ["parihaka_facies_data_inline", "parihaka_facies_data_crossline"],
        "penobscot": ["penobscot_facies_data_inline", "penobscot_facies_data_crossline"],
    }

    simclr_ckpt = os.path.join(REPO_ROOT, "checkpoints", "simclr", "simclr_resnet50_epoch10.pth.tar")
    adasemseg_dir = os.path.join(REPO_ROOT, "methods", "adasemseg")
    baseline_dir = os.path.join(REPO_ROOT, "methods", "baselines")

    print("Table II reproduction commands")
    print("=" * 70)
    print("For each target dataset T, the required steps are:")
    print("  1. AdaSemSeg: train on the two source datasets, evaluate on T inline+crossline.")
    print("  2. Baseline-1: train methods/adasemseg/Main.py on T's own inline+crossline (no leave-one-out).")
    print("  3. Baseline-2: train ResNet-UNet (methods/baselines/Main.py) on T inline and crossline separately.")
    print("=" * 70)

    commands = []
    # Run-id counters, scoped per method directory (that's what scopes the
    # checkpoint output path -- see module docstring).
    adasemseg_run_id = 0
    baseline_run_id = 0

    for target_name, target_classes in targets.items():
        # --- AdaSemSeg (leave-one-out) ---
        all_datasets = ["f3", "parihaka", "penobscot"]
        source_names = [d for d in all_datasets if d != target_name]
        source_classes = []
        for s in source_names:
            source_classes.extend(targets[s])

        adasemseg_run_id += 1
        run_id = adasemseg_run_id
        adasemseg_train = [
            sys.executable, "Main.py", "--run_id", str(run_id), "--shots", str(args.shots), "--train",
            "--classes",
        ] + source_classes + [
            "--img_enc_checkpoint", simclr_ckpt, "--device", args.device,
        ]
        adasemseg_ckpt = os.path.join(adasemseg_dir, "logs", "checkpoints",
                                       f"{args.shots}-shot", f"Run_{run_id}", "bestmodel.pth.tar")
        adasemseg_eval = [
            sys.executable, "evaluate.py", "--checkpoint", adasemseg_ckpt,
            "--img_enc_checkpoint", simclr_ckpt,
            "--shots", str(args.shots),
            "--classes",
        ] + target_classes + [
            "--output_dir", os.path.join(adasemseg_dir, "evaluation_results", "table2", f"{target_name}_adasemseg"),
            "--device", args.device,
        ]
        if target_name == "parihaka":
            adasemseg_eval.append("--use_nearest_slice")
        commands.append((f"AdaSemSeg ({target_name})", adasemseg_dir, adasemseg_train, None))
        commands.append((f"AdaSemSeg eval ({target_name})", adasemseg_dir, adasemseg_eval, None))

        for class_name in target_classes:
            # --- Baseline-1: methods/adasemseg/Main.py, target-only classes ---
            adasemseg_run_id += 1
            b1_run_id = adasemseg_run_id
            baseline1_train = [
                sys.executable, "Main.py", "--run_id", str(b1_run_id), "--shots", str(args.shots),
                "--train", "--freeze_bn",
                "--classes", class_name,
                "--img_enc_checkpoint", simclr_ckpt, "--device", args.device,
            ]
            b1_ckpt = os.path.join(adasemseg_dir, "logs", "checkpoints",
                                    f"{args.shots}-shot", f"Run_{b1_run_id}", "bestmodel.pth.tar")
            baseline1_eval = [
                sys.executable, "evaluate.py", "--checkpoint", b1_ckpt,
                "--img_enc_checkpoint", simclr_ckpt,
                "--shots", str(args.shots),
                "--classes", class_name,
                "--output_dir", os.path.join(adasemseg_dir, "evaluation_results", "table2",
                                              f"{target_name}_baseline1_{class_name}"),
                "--device", args.device,
            ]
            commands.append((f"Baseline-1 ({target_name} {class_name})", adasemseg_dir, baseline1_train, None))
            commands.append((f"Baseline-1 eval ({target_name} {class_name})", adasemseg_dir, baseline1_eval, None))

            # --- Baseline-2: methods/baselines/Main.py, target-only classes ---
            baseline_run_id += 1
            b2_run_id = baseline_run_id
            baseline2_train = [
                sys.executable, "Main.py", "--run_id", str(b2_run_id), "--train", "--img_enc_type", "resnet",
                "--freeze_bn", "--classes", class_name,
                "--device", args.device,
            ]
            b2_ckpt_dir = os.path.join(baseline_dir, "logs", "checkpoints", f"Run_{b2_run_id}")
            baseline2_eval = [
                sys.executable, os.path.join("Evaluation", "Main.py"),
                "--visualize", "0", "--run_id", str(b2_run_id), "--eval_mode", "test",
                "--img_enc_type", "resnet",
                "--source_class", class_name, "--best_model",
                "--checkpoint_dir", b2_ckpt_dir,
                "--device", args.device,
            ]
            commands.append((f"Baseline-2 ({target_name} {class_name})", baseline_dir, baseline2_train, None))
            commands.append((f"Baseline-2 eval ({target_name} {class_name})", baseline_dir, baseline2_eval, None))

    for label, cwd, cmd, note in commands:
        print(f"\n[{label}]")
        if note:
            print(f"  Prerequisite: {note}")
        print(f"  cd {cwd}")
        print(f"  {' '.join(cmd)}")
        if args.run_commands:
            run(cmd, cwd=cwd)

    print("\nTable II reproduction instructions printed.")


if __name__ == "__main__":
    main()
