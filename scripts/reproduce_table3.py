#!/usr/bin/env python
"""
Reproduce Table III: AdaSemSeg vs ProtoSemSeg vs transfer learning (1/5-shot).

For each held-out target dataset, this script prints (and optionally runs) the
training and evaluation commands for:
  - AdaSemSeg (trained on source datasets, evaluated on target)
  - ProtoSemSeg (trained on source datasets, evaluated on target)
  - Transfer learning (Baseline-2 trained on sources and fine-tuned on target)
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
    parser = argparse.ArgumentParser(description="Reproduce Table III")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run_commands", action="store_true",
                        help="Actually execute the commands (default: print only)")
    parser.add_argument("--shots", type=int, default=5, choices=[1, 5])
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
    proto_dir = os.path.join(REPO_ROOT, "methods", "protosemseg")
    baseline_dir = os.path.join(REPO_ROOT, "methods", "baselines")

    print("Table III reproduction commands")
    print("=" * 70)

    commands = []
    for target_name, target_classes in targets.items():
        all_datasets = ["f3", "parihaka", "penobscot"]
        source_names = [d for d in all_datasets if d != target_name]
        source_classes = []
        for s in source_names:
            source_classes.extend(targets[s])

        # AdaSemSeg
        adasemseg_train = [
            sys.executable, "Main.py", "--run_id", "1", "--shots", str(args.shots), "--train",
            "--classes",
        ] + source_classes + [
            "--img_enc_checkpoint", simclr_ckpt, "--device", args.device,
        ]
        adasemseg_eval = [
            sys.executable, "evaluate.py",
            "--checkpoint", os.path.join(adasemseg_dir, "logs", f"table3_{target_name}", "checkpoints",
                                         f"{args.shots}-shot", "Run_1", "bestmodel.pth.tar"),
            "--img_enc_checkpoint", simclr_ckpt,
            "--shots", str(args.shots),
            "--classes",
        ] + target_classes + ["--output_dir", os.path.join(adasemseg_dir, "logs", f"table3_{target_name}", "eval"),
                              "--device", args.device]
        if target_name == "parihaka":
            adasemseg_eval.append("--use_nearest_slice")

        # ProtoSemSeg
        proto_train = [
            sys.executable, "Main.py", "--run_id", "1", "--shots", str(args.shots), "--train",
            "--classes",
        ] + source_classes + [
            "--img_enc_checkpoint", simclr_ckpt, "--device", args.device,
        ]

        # Transfer learning: train Baseline-2 on sources, then fine-tune on target slices
        transfer_source_train = [
            sys.executable, os.path.join("Transfer_learning", "Main.py"),
            "--run_id", "1", "--train", "--img_enc_type", "resnet",
            "--freeze_bn", "--classes",
        ] + source_classes + ["--device", args.device]
        transfer_finetune = [
            sys.executable, os.path.join("Transfer_learning", "Main.py"),
            "--run_id", "1", "--train", "--train_indices", str(args.shots),
            "--checkpoint", os.path.join(baseline_dir, "logs", f"table3_{target_name}_transfer_source",
                                         "checkpoints", "Run_1", "bestmodel.pth.tar"),
            "--img_enc_type", "resnet", "--freeze_bn", "--classes",
        ] + target_classes + ["--device", args.device]

        commands.append((f"AdaSemSeg ({target_name})", adasemseg_dir, adasemseg_train, None))
        commands.append((f"AdaSemSeg eval ({target_name})", adasemseg_dir, adasemseg_eval, None))
        commands.append((f"ProtoSemSeg ({target_name})", proto_dir, proto_train, None))
        for class_name in target_classes:
            proto_eval = [
                sys.executable, os.path.join("Evaluation_sampling", "Main.py"),
                "--visualize", "0", "--run_id", "1", "--shots", str(args.shots),
                "--eval_mode", "test", "--best_model",
                "--source_class", class_name,
                "--checkpoint_dir", os.path.join(proto_dir, "logs", f"table3_{target_name}", "checkpoints",
                                                 f"{args.shots}-shot", "Run_1"),
                "--device", args.device,
            ]
            commands.append((f"ProtoSemSeg eval ({target_name} {class_name})", proto_dir, proto_eval, None))
        commands.append((f"Transfer source train ({target_name})", baseline_dir, transfer_source_train, None))
        commands.append((f"Transfer fine-tune ({target_name})", baseline_dir, transfer_finetune, None))

    for label, cwd, cmd, note in commands:
        print(f"\n[{label}]")
        if note:
            print(f"  Prerequisite: {note}")
        print(f"  cd {cwd}")
        print(f"  {' '.join(cmd)}")
        if args.run_commands:
            run(cmd, cwd=cwd)

    print("\nTable III reproduction instructions printed.")


if __name__ == "__main__":
    main()
