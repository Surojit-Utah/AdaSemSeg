#!/usr/bin/env python
"""
Assemble the qualitative prediction figures (paper style: Fig. 12/17/18) from the
raw prediction patches saved by `methods/adasemseg/evaluate.py --save_predictions`
(see scripts/reproduce_figures.py, which produces those patches).

Uses methods/adasemseg/predict/DGP_evaluator.py's show_combined_model_pred_images(),
ported from the original AdaSemSeg research workspace, to build a 3-row panel per
batch of patches: query image / ground truth / prediction.
"""
import argparse
import glob
import os
import sys

import matplotlib
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "methods", "adasemseg"))

from predict.DGP_evaluator import show_combined_model_pred_images


def to_rgb_grayscale(patch):
    """(H, W) float in [0, 1] -> (H, W, 3) uint8."""
    img = np.clip(patch * 255.0, 0, 255).astype(np.uint8)
    return np.stack([img, img, img], axis=-1)


def label_to_rgb(label, num_classes, cmap_name="tab10"):
    """(H, W) integer class labels -> (H, W, 3) uint8 using a fixed colormap."""
    cmap = matplotlib.colormaps[cmap_name].resampled(max(num_classes, 1))
    normalized = label.astype(np.float32) / max(num_classes - 1, 1)
    rgb = cmap(normalized)[:, :, :3]
    return (rgb * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Assemble qualitative prediction figures from saved patches")
    parser.add_argument("--patches_dir", required=True,
                        help="Directory of .npz patches, e.g. <output_dir>/prediction_patches from reproduce_figures.py")
    parser.add_argument("--output_dir", default=None,
                        help="Where to write assembled panel PNGs (default: <patches_dir>/../assembled_figures)")
    parser.add_argument("--batch_size", type=int, default=5, help="Patches per panel figure")
    parser.add_argument("--max_batches", type=int, default=None, help="Limit number of panels produced (default: all)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(os.path.dirname(os.path.normpath(args.patches_dir)), "assembled_figures")
    os.makedirs(output_dir, exist_ok=True)

    patch_files = sorted(glob.glob(os.path.join(args.patches_dir, "*.npz")))
    if not patch_files:
        raise FileNotFoundError(f"No .npz patches found under {args.patches_dir}")
    print(f"Found {len(patch_files)} patches in {args.patches_dir}")

    num_classes = 1
    for f in patch_files:
        with np.load(f) as d:
            num_classes = max(num_classes, int(d["gt"].max()), int(d["pred"].max()))

    batch_idx = 0
    for start in range(0, len(patch_files), args.batch_size):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        batch_files = patch_files[start:start + args.batch_size]

        input_images, gt_images, pred_images = [], [], []
        for f in batch_files:
            with np.load(f) as d:
                input_images.append(to_rgb_grayscale(d["query_image"]))
                gt_images.append(label_to_rgb(d["gt"], num_classes))
                pred_images.append(label_to_rgb(d["pred"], num_classes))

        input_images = np.stack(input_images)
        gt_images = np.stack(gt_images)
        pred_images = np.stack(pred_images)

        out_path = os.path.join(output_dir, f"Pred_panel_{batch_idx + 1}.png")
        show_combined_model_pred_images(input_images, gt_images, pred_images, out_path)
        print(f"Wrote {out_path}")
        batch_idx += 1

    print(f"\nAssembled {batch_idx} panel figure(s) in {output_dir}")


if __name__ == "__main__":
    main()
