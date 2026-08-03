#!/usr/bin/env python
"""
Reproduce paper prediction figures (F3, Parihaka, Penobscot).

Runs the unified AdaSemSeg evaluator with --save_predictions so that predicted
patches, ground-truth patches, and metadata are written to disk. Assemble these
into the paper-style qualitative panels with scripts/assemble_figures.py.
"""
import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Reproduce paper figures")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--target", required=True,
                        choices=["f3", "parihaka", "penobscot"],
                        help="Target dataset for which to generate predictions")
    parser.add_argument("--output_dir", default="./figure_predictions")
    parser.add_argument("--data_root", default=os.environ.get("ADASEMSEG_DATA_ROOT", "./data"))
    args = parser.parse_args()

    os.environ.setdefault("ADASEMSEG_DATA_ROOT", args.data_root)

    target_classes = {
        "f3": ["f3_facies_data_inline", "f3_facies_data_crossline"],
        "parihaka": ["parihaka_facies_data_inline", "parihaka_facies_data_crossline"],
        "penobscot": ["penobscot_facies_data_inline", "penobscot_facies_data_crossline"],
    }

    simclr_ckpt = os.path.join(REPO_ROOT, "checkpoints", "simclr", "simclr_resnet50_epoch10.pth.tar")

    cmd = [
        sys.executable, os.path.join(REPO_ROOT, "methods", "adasemseg", "evaluate.py"),
        "--checkpoint", args.checkpoint,
        "--img_enc_checkpoint", simclr_ckpt,
        "--shots", str(args.shots),
        "--classes",
    ] + target_classes[args.target] + [
        "--output_dir", args.output_dir,
        "--device", args.device,
        "--save_predictions",
    ]
    if args.target == "parihaka":
        cmd.append("--use_nearest_slice")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    patches_dir = os.path.join(args.output_dir, 'prediction_patches')
    print(f"\nPrediction patches saved to {patches_dir}")
    print(f"Assemble into paper-style panel figures with:\n"
          f"  python scripts/assemble_figures.py --patches_dir {patches_dir}")


if __name__ == "__main__":
    main()
