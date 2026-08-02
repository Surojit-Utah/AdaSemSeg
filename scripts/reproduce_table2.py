#!/usr/bin/env python
"""
Reproduce Table 2: AdaSemSeg vs baselines trained only on the target dataset.

Table 2 compares three methods:
  - AdaSemSeg (few-shot, trained on source datasets, evaluated on target)
  - Baseline-1 (AdaSemSeg architecture trained on target support slices)
  - Baseline-2 (regular ResNet-UNet segmentation trained on target support slices)

Because each method has a different training/evaluation protocol, this script
prints the exact commands to run. It can also execute them when
`--run_commands` is set.
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
    parser = argparse.ArgumentParser(description="Reproduce Table 2")
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

    print("Table 2 reproduction commands")
    print("=" * 70)
    print("For each target dataset T, the required steps are:")
    print("  1. AdaSemSeg: train on the two source datasets, evaluate on T inline+crossline.")
    print("  2. Baseline-1: train on T inline and crossline separately, evaluate each.")
    print("  3. Baseline-2: train ResNet-UNet on T inline and crossline separately, evaluate each.")
    print("=" * 70)

    commands = []
    for target_name, target_classes in targets.items():
        # AdaSemSeg sources = the other two datasets
        all_datasets = ["f3", "parihaka", "penobscot"]
        source_names = [d for d in all_datasets if d != target_name]
        source_classes = []
        for s in source_names:
            source_classes.extend(targets[s])

        adasemseg_train = [
            sys.executable, "Main.py", "--run_id", "1", "--shots", str(args.shots), "--train",
            "--classes",
        ] + source_classes + [
            "--img_enc_checkpoint", simclr_ckpt, "--device", args.device,
        ]
        adasemseg_eval = [
            sys.executable, "evaluate.py", "--checkpoint",
            os.path.join(adasemseg_dir, "logs", f"table2_{target_name}", "checkpoints",
                         f"{args.shots}-shot", "Run_1", "bestmodel.pth.tar"),
            "--img_enc_checkpoint", simclr_ckpt,
            "--shots", str(args.shots),
            "--classes",
        ] + target_classes + ["--output_dir", os.path.join(adasemseg_dir, "logs", f"table2_{target_name}", "eval"),
                              "--device", args.device]
        if target_name == "parihaka":
            adasemseg_eval.append("--use_nearest_slice")

        for class_name in target_classes:
            baseline1_train = [
                sys.executable, "Main.py", "--run_id", "1", "--train", "--freeze_bn",
                "--classes", class_name,
                "--device", args.device,
            ]
            baseline1_eval = [
                sys.executable, os.path.join("Evaluation", "Main.py"),
                "--visualize", "0", "--run_id", "1", "--eval_mode", "test",
                "--source_class", class_name, "--best_model",
                "--checkpoint_dir", os.path.join(baseline_dir, "logs", f"table2_{target_name}_baseline1",
                                                 "checkpoints", class_name, "Run_1"),
                "--device", args.device,
            ]
            commands.append((f"Baseline-1 ({target_name} {class_name})", baseline_dir, baseline1_train, None))
            commands.append((f"Baseline-1 eval ({target_name} {class_name})", baseline_dir, baseline1_eval, None))

            baseline2_train = [
                sys.executable, "Main.py", "--run_id", "1", "--train", "--img_enc_type", "resnet",
                "--freeze_bn", "--classes", class_name,
                "--device", args.device,
            ]
            baseline2_eval = [
                sys.executable, os.path.join("Evaluation", "Main.py"),
                "--visualize", "0", "--run_id", "1", "--eval_mode", "test",
                "--img_enc_type", "resnet",
                "--source_class", class_name, "--best_model",
                "--checkpoint_dir", os.path.join(baseline_dir, "logs", f"table2_{target_name}_baseline2",
                                                 "checkpoints", class_name, "Run_1"),
                "--device", args.device,
            ]
            commands.append((f"Baseline-2 ({target_name} {class_name})", baseline_dir, baseline2_train, None))
            commands.append((f"Baseline-2 eval ({target_name} {class_name})", baseline_dir, baseline2_eval, None))

        commands.append((f"AdaSemSeg ({target_name})", adasemseg_dir, adasemseg_train, None))
        commands.append((f"AdaSemSeg eval ({target_name})", adasemseg_dir, adasemseg_eval, None))

    for label, cwd, cmd, note in commands:
        print(f"\n[{label}]")
        if note:
            print(f"  Prerequisite: {note}")
        print(f"  cd {cwd}")
        print(f"  {' '.join(cmd)}")
        if args.run_commands:
            run(cmd, cwd=cwd)

    print("\nTable 2 reproduction instructions printed.")


if __name__ == "__main__":
    main()
