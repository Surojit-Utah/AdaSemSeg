"""
Evaluate a trained AdaSemSeg checkpoint on the test splits.

Supports both K-shot random support and nearest-slice support selection.
"""
import argparse
import os
import sys
import json
import numpy as np
from collections import OrderedDict
from itertools import islice
import torch
import torch.nn as nn
from torchvision import models

sys.path.insert(0, os.path.dirname(__file__))

from data.TestDataset import AdaSemSegTestDataset
from models import DGP_resnet_unet
from kernels.gp_kernels import RBF
from scripts.config_loader import make_eval_data_info_catalogue


def seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(img_enc_checkpoint, device, freeze_bn=False):
    """Build AdaSemSeg model with ResNet50 encoder and load SimCLR weights if provided."""
    covar_size = 5
    covariance_output_mode = 'concatenate variance'
    depth_image_encoder = 512

    resnet = models.resnet50(pretrained=False)
    resnet.fc = nn.Identity()

    if img_enc_checkpoint:
        checkpoint_dict = torch.load(img_enc_checkpoint, map_location=device)
        trained_model_param = checkpoint_dict['state_dict']
        sliced = islice(trained_model_param.items(), len(trained_model_param.keys()) - 4)
        trained_model_param = OrderedDict(sliced)
        trained_model_param = OrderedDict([
            (k.replace('backbone.', ''), v) for k, v in trained_model_param.items()
        ])
        resnet.load_state_dict(trained_model_param, strict=True)
        print("Loaded SimCLR backbone.")

    img_encoder_obj = DGP_resnet_unet.Image_Encoder(resnet, freeze_bn)
    mask_encoder_obj = DGP_resnet_unet.Mask_Encoder()
    dgp_model = DGP_resnet_unet.DGPModel(
        kernel=RBF(length=(1 / (depth_image_encoder ** 0.25))),
        covariance_output_mode=covariance_output_mode,
        covar_size=covar_size
    )
    fss_decoder_obj = DGP_resnet_unet.FSS_Decoder(covar_size=covar_size)
    fss_learner_obj = DGP_resnet_unet.FSSLearner(
        image_encoder=img_encoder_obj,
        anno_encoder=mask_encoder_obj,
        dgp_model=dgp_model,
        upsampler=fss_decoder_obj
    )
    fss_learner_obj.to(device)
    return fss_learner_obj


def load_trained_model(checkpoint_path, model, device):
    """Load full model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model


def evaluate(model, test_loader, device, num_classes, class_indices, class_weights,
             save_predictions=False, save_dir=None):
    """Run inference and collect predictions and ground truths."""
    model.eval()
    predict_act = torch.nn.Softmax2d()

    if save_predictions and save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Accumulate per-sample results keyed by (class, slice, row, col)
    results = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            class_name = data['data_class'][0]

            support_images = data['support_images']
            support_split_masks = data['support_split_masks']
            query_image = data['query_image']
            query_segmentation = data['query_segmentation']

            pred_per_class_softmax = []
            for class_index, class_label in enumerate(class_indices[class_name]):
                online_models = model.learn(support_images, support_split_masks[:, :, class_label - 1])
                query_pred = torch.squeeze(model(query_image, online_models), axis=1)
                output_segs = predict_act(query_pred)
                output_segs = output_segs[:, 1, :, :]
                pred_per_class_softmax.append(output_segs)

            pred_per_class_softmax = torch.stack(pred_per_class_softmax, dim=1)
            pred_summary = torch.argmax(pred_per_class_softmax, dim=1) + 1

            pred_np = pred_summary.cpu().numpy()
            gt_np = query_segmentation.cpu().numpy()

            # Save individual prediction patches for figure generation
            if save_predictions and save_dir:
                b = pred_np.shape[0]
                for i in range(b):
                    out_name = f"{class_name}_s{data['query_slice_index'][i]}_r{data['query_row'][i]}_c{data['query_col'][i]}.npz"
                    np.savez(os.path.join(save_dir, out_name),
                             pred=pred_np[i, 0],
                             gt=gt_np[i, 0],
                             class_name=class_name,
                             slice_index=int(data['query_slice_index'][i]),
                             row=int(data['query_row'][i]),
                             col=int(data['query_col'][i]))

            # Move to CPU and store
            results.append({
                'class_name': class_name,
                'query_slice_index': data['query_slice_index'].cpu().numpy(),
                'query_row': data['query_row'].cpu().numpy(),
                'query_col': data['query_col'].cpu().numpy(),
                'pred': pred_np,
                'gt': gt_np,
            })

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} batches")

    return results


def aggregate_metrics(results, class_indices, num_classes):
    """Aggregate patch predictions and compute metrics per class/slice.

    Metrics are computed independently for each dataset/class and then averaged,
    matching the paper's per-dataset reporting style. Per-class IoU and F1 are
    also reported for Table 8-style sensitivity analysis.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
    from metrics import (pixel_accuracy, mean_class_accuracy, frequency_weighted_iou,
                         frequency_weighted_f1, intersection_over_union, f1_score)

    per_class = {c: {'pred': [], 'gt': []} for c in class_indices.keys()}
    for r in results:
        per_class[r['class_name']]['pred'].append(r['pred'].ravel())
        per_class[r['class_name']]['gt'].append(r['gt'].ravel())

    metrics_by_class = {}
    for class_name, accum in per_class.items():
        if not accum['pred']:
            continue
        pred = np.concatenate(accum['pred'])
        gt = np.concatenate(accum['gt'])
        n_cls = num_classes[class_name] + 1  # include background
        pa = pixel_accuracy(pred, gt)
        mca = mean_class_accuracy(pred, gt, n_cls)
        fwiou = frequency_weighted_iou(pred, gt, n_cls)
        fwf1 = frequency_weighted_f1(pred, gt, n_cls)
        iou_per_class = intersection_over_union(pred, gt, n_cls)
        f1_per_class = f1_score(pred, gt, n_cls)
        metrics_by_class[class_name] = {
            'PA': float(pa),
            'MCA': float(mca),
            'FwIoU': float(fwiou),
            'FwF1': float(fwf1),
            'IoU_per_class': [float(x) if not np.isnan(x) else None for x in iou_per_class],
            'F1_per_class': [float(x) if not np.isnan(x) else None for x in f1_per_class],
        }

    # Average across evaluated classes/datasets
    avg_metrics = {}
    for key in ['PA', 'MCA', 'FwIoU', 'FwF1']:
        values = [m[key] for m in metrics_by_class.values()]
        avg_metrics[key] = float(np.mean(values)) if values else 0.0

    return avg_metrics, metrics_by_class


def run_evaluation(checkpoint, img_enc_checkpoint=None, classes=None, shots=5,
                   use_nearest_slice=False, eval_mode='test', data_root='./data',
                   output_dir='./evaluation_results', device='cuda:0',
                   batch_size=1, patch_size=256, seed=0, save_predictions=False):
    """Programmatic entry point used by both the CLI and Main.py."""
    seed_all(seed)
    os.environ["ADASEMSEG_DATA_ROOT"] = data_root

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_info = make_eval_data_info_catalogue()
    if classes is None:
        classes = list(data_info.keys())

    print("Evaluating classes:", classes)

    # Build model
    model = build_model(img_enc_checkpoint, device)
    model = load_trained_model(checkpoint, model, device)

    # Build test dataset
    test_dataset = AdaSemSegTestDataset(
        classes=classes,
        data_info=data_info,
        patch_size=patch_size,
        k_shot=shots,
        use_nearest_slice=use_nearest_slice,
        eval_mode=eval_mode,
        batch_size=batch_size,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    class_indices = {c: test_dataset.class_labels[c] for c in classes}
    num_classes = {c: test_dataset.class_label_count[c] for c in classes}

    pred_save_dir = os.path.join(output_dir, "prediction_patches") if save_predictions else None
    results = evaluate(model, test_loader, device, num_classes, class_indices,
                       class_weights={},
                       save_predictions=save_predictions,
                       save_dir=pred_save_dir)

    avg_metrics, metrics_by_class = aggregate_metrics(results, class_indices, num_classes)

    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, "metrics.json")
    output = {
        "average": avg_metrics,
        "per_class": metrics_by_class,
        "config": {
            "shots": shots,
            "use_nearest_slice": use_nearest_slice,
            "eval_mode": eval_mode,
            "checkpoint": checkpoint,
            "classes": classes,
        },
    }
    with open(result_path, "w") as f:
        json.dump(output, f, indent=2)
    print("Average metrics:", avg_metrics)
    print("Per-class metrics:", metrics_by_class)
    print(f"Saved results to {result_path}")
    if save_predictions:
        print(f"Saved prediction patches to {pred_save_dir}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained AdaSemSeg")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (bestmodel.pth.tar)")
    parser.add_argument("--img_enc_checkpoint",
                        default="checkpoints/simclr/simclr_resnet50_run2_epoch10.pth.tar",
                        help="SimCLR backbone checkpoint (use with --random_init to omit loading)")
    parser.add_argument("--random_init", action="store_true",
                        help="Initialize the image encoder randomly (Table 5 random-init ablation)")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Class names to evaluate (default: all target classes)")
    parser.add_argument("--shots", type=int, default=5, choices=[1, 5])
    parser.add_argument("--use_nearest_slice", action="store_true")
    parser.add_argument("--eval_mode", default="test", choices=["val", "test"])
    parser.add_argument("--data_root", default=os.environ.get("ADASEMSEG_DATA_ROOT", "./data"))
    parser.add_argument("--output_dir", default="./evaluation_results")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save individual prediction patches for figure generation")
    args = parser.parse_args()

    img_enc_checkpoint = None if args.random_init else args.img_enc_checkpoint
    run_evaluation(
        checkpoint=args.checkpoint,
        img_enc_checkpoint=img_enc_checkpoint,
        classes=args.classes,
        shots=args.shots,
        use_nearest_slice=args.use_nearest_slice,
        eval_mode=args.eval_mode,
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        seed=args.seed,
        save_predictions=args.save_predictions,
    )


if __name__ == "__main__":
    main()
