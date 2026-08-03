#!/usr/bin/env python
"""
Reproduce Table III: AdaSemSeg vs ProtoSemSeg vs transfer learning (1/5-shot).

For each held-out target dataset, this script prints (and optionally runs) the
training and evaluation commands for:
  - AdaSemSeg (trained on source datasets, evaluated on target)
  - ProtoSemSeg (trained on source datasets, evaluated on target)
  - Transfer learning (Baseline-2 trained on sources and fine-tuned on target)

Note on checkpoint paths: Main.py (in methods/adasemseg/, methods/protosemseg/,
and methods/baselines/) always writes checkpoints under
<cwd>/logs/checkpoints/[<shots>-shot/]Run_<run_id>/bestmodel.pth.tar -- there is
no flag to redirect this. Every training invocation below gets its own
--run_id (tracked per method directory, since that's what scopes the output
path), otherwise later runs would silently overwrite earlier ones' checkpoints
at the same path.
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
    # Run-id counters, scoped per method directory (that's what scopes the
    # checkpoint output path -- see module docstring).
    adasemseg_run_id = 0
    proto_run_id = 0
    baseline_run_id = 0

    for target_name, target_classes in targets.items():
        all_datasets = ["f3", "parihaka", "penobscot"]
        source_names = [d for d in all_datasets if d != target_name]
        source_classes = []
        for s in source_names:
            source_classes.extend(targets[s])

        # --- AdaSemSeg ---
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
            sys.executable, "evaluate.py",
            "--checkpoint", adasemseg_ckpt,
            "--img_enc_checkpoint", simclr_ckpt,
            "--shots", str(args.shots),
            "--classes",
        ] + target_classes + [
            "--output_dir", os.path.join(adasemseg_dir, "evaluation_results", "table3", f"{target_name}_adasemseg"),
            "--device", args.device,
        ]
        if target_name == "parihaka":
            adasemseg_eval.append("--use_nearest_slice")

        # --- ProtoSemSeg ---
        proto_run_id += 1
        p_run_id = proto_run_id
        proto_train = [
            sys.executable, "Main.py", "--run_id", str(p_run_id), "--shots", str(args.shots), "--train",
            "--classes",
        ] + source_classes + [
            "--img_enc_checkpoint", simclr_ckpt, "--device", args.device,
        ]
        proto_ckpt_dir = os.path.join(proto_dir, "logs", "checkpoints", f"{args.shots}-shot", f"Run_{p_run_id}")

        # --- Transfer learning: train Baseline-2 on sources, then fine-tune on target slices ---
        baseline_run_id += 1
        source_run_id = baseline_run_id
        transfer_source_train = [
            sys.executable, os.path.join("Transfer_learning", "Main.py"),
            "--run_id", str(source_run_id), "--train", "--train_indices", "all", "--img_enc_type", "resnet",
            "--freeze_bn", "--classes",
        ] + source_classes + ["--device", args.device]
        transfer_source_ckpt = os.path.join(baseline_dir, "logs", "checkpoints",
                                             f"Run_{source_run_id}", "bestmodel.pth.tar")

        baseline_run_id += 1
        finetune_run_id = baseline_run_id
        transfer_finetune = [
            sys.executable, os.path.join("Transfer_learning", "Main.py"),
            "--run_id", str(finetune_run_id), "--train", "--train_indices", str(args.shots),
            "--checkpoint", transfer_source_ckpt,
            "--img_enc_type", "resnet", "--freeze_bn", "--classes",
        ] + target_classes + ["--device", args.device]
        transfer_ckpt_dir = os.path.join(baseline_dir, "Transfer_learning", "logs", "checkpoints",
                                          f"Train_slices_{args.shots}", f"Run_{finetune_run_id}")

        commands.append((f"AdaSemSeg ({target_name})", adasemseg_dir, adasemseg_train, None))
        commands.append((f"AdaSemSeg eval ({target_name})", adasemseg_dir, adasemseg_eval, None))
        commands.append((f"ProtoSemSeg ({target_name})", proto_dir, proto_train, None))
        for class_name in target_classes:
            proto_eval = [
                sys.executable, os.path.join("Evaluation_sampling", "Main.py"),
                "--visualize", "0", "--run_id", str(p_run_id), "--shots", str(args.shots),
                "--eval_mode", "test", "--best_model",
                "--source_class", class_name,
                "--checkpoint_dir", proto_ckpt_dir,
                "--device", args.device,
            ]
            commands.append((f"ProtoSemSeg eval ({target_name} {class_name})", proto_dir, proto_eval, None))
        commands.append((f"Transfer source train ({target_name})", baseline_dir, transfer_source_train, None))
        commands.append((f"Transfer fine-tune ({target_name})", baseline_dir, transfer_finetune,
                         f"Requires transfer source checkpoint at {transfer_source_ckpt}"))
        for class_name in target_classes:
            transfer_eval = [
                sys.executable, os.path.join("scripts", "evaluate_baseline.py"),
                "--transfer_learning", "--train_indices", str(args.shots),
                "--checkpoint_dir", transfer_ckpt_dir,
                "--source_class", class_name, "--eval_mode", "test",
                "--device", args.device,
            ]
            commands.append((f"Transfer eval ({target_name} {class_name})", REPO_ROOT, transfer_eval, None))

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
