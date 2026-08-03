#!/usr/bin/env python
"""
Reproduce Table VI: Average inference time (seconds) on GPU and CPU.

Times the evaluation of a trained model on a single query patch over a small
number of warm-up and measured iterations. Timing must be performed on the
actual hardware where the paper experiments were run; this script provides a
standardized measurement harness.
"""
import argparse
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "methods", "adasemseg"))
from evaluate import build_model, load_trained_model
from data.TestDataset import AdaSemSegTestDataset
from scripts.config_loader import make_eval_data_info_catalogue


def main():
    parser = argparse.ArgumentParser(description="Reproduce Table VI")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--img_enc_checkpoint",
                        default=os.path.join("checkpoints", "simclr", "simclr_resnet50_epoch10.pth.tar"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--use_nearest_slice", action="store_true")
    parser.add_argument("--warm_up", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--data_root", default=os.environ.get("ADASEMSEG_DATA_ROOT", "./data"))
    args = parser.parse_args()

    os.environ["ADASEMSEG_DATA_ROOT"] = args.data_root
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(args.img_enc_checkpoint, device)
    model = load_trained_model(args.checkpoint, model, device)
    model.eval()

    data_info = make_eval_data_info_catalogue()
    classes = args.classes if args.classes else list(data_info.keys())
    test_dataset = AdaSemSegTestDataset(
        classes=classes,
        data_info=data_info,
        patch_size=256,
        k_shot=args.shots,
        use_nearest_slice=args.use_nearest_slice,
        eval_mode='test',
        batch_size=1,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    times = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            if batch_idx >= args.warm_up + args.iterations:
                break
            data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            support_images = data['support_images']
            support_split_masks = data['support_split_masks']
            query_image = data['query_image']

            start = time.perf_counter()
            class_name = data['data_class'][0]
            class_label = test_dataset.class_labels[class_name][0]
            online_models = model.learn(support_images, support_split_masks[:, :, class_label - 1])
            _ = torch.squeeze(model(query_image, online_models), axis=1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            if batch_idx >= args.warm_up:
                times.append(elapsed)

    if times:
        print(f"Mean inference time over {len(times)} iterations: {np.mean(times):.4f}s")
        print(f"Std: {np.std(times):.4f}s")
    else:
        print("No iterations measured.")


if __name__ == "__main__":
    main()
