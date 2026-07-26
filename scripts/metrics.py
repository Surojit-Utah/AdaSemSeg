"""
Standard segmentation metrics used in the AdaSemSeg paper.

All functions accept integer label maps (pred, target) with shape (H, W) or
batched shape (N, H, W). Background / ignored pixels can be excluded via the
`ignore_index` argument.
"""

import numpy as np


def _bin_count(pred, target, num_classes, ignore_index=-1):
    """Compute pixel-wise confusion matrix entries per class."""
    mask = (target != ignore_index)
    pred = pred[mask]
    target = target[mask]
    cm = np.bincount(num_classes * target + pred, minlength=num_classes ** 2)
    return cm.reshape(num_classes, num_classes)


def pixel_accuracy(pred, target, ignore_index=-1):
    """Pixel Accuracy (PA): fraction of correctly classified pixels."""
    mask = target != ignore_index
    return np.mean(pred[mask] == target[mask])


def class_accuracy(pred, target, num_classes, ignore_index=-1):
    """Per-class accuracy."""
    cm = _bin_count(pred, target, num_classes, ignore_index)
    tp = np.diag(cm)
    total_per_class = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = tp / total_per_class
    acc[total_per_class == 0] = np.nan
    return acc


def mean_class_accuracy(pred, target, num_classes, ignore_index=-1):
    """Mean Class Accuracy (MCA)."""
    acc = class_accuracy(pred, target, num_classes, ignore_index)
    return np.nanmean(acc)


def intersection_over_union(pred, target, num_classes, ignore_index=-1):
    """Per-class Intersection over Union (IoU)."""
    cm = _bin_count(pred, target, num_classes, ignore_index)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / denom
    iou[denom == 0] = np.nan
    return iou


def frequency_weighted_iou(pred, target, num_classes, ignore_index=-1):
    """Frequency-weighted IoU (FwIoU)."""
    cm = _bin_count(pred, target, num_classes, ignore_index)
    iou = intersection_over_union(pred, target, num_classes, ignore_index)
    total = cm.sum()
    freq = cm.sum(axis=1) / total
    return np.nansum(freq * iou)


def f1_score(pred, target, num_classes, ignore_index=-1):
    """Per-class F1 score."""
    cm = _bin_count(pred, target, num_classes, ignore_index)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.zeros(num_classes, dtype=np.float64)
    recall = np.zeros(num_classes, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
    denom = precision + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * precision * recall / denom
    f1[denom == 0] = np.nan
    return f1


def frequency_weighted_f1(pred, target, num_classes, ignore_index=-1):
    """Frequency-weighted F1 score (FwF1)."""
    cm = _bin_count(pred, target, num_classes, ignore_index)
    f1 = f1_score(pred, target, num_classes, ignore_index)
    total = cm.sum()
    freq = cm.sum(axis=1) / total
    return np.nansum(freq * f1)


def compute_all_metrics(pred, target, num_classes, ignore_index=-1):
    """Return a dictionary of all metrics."""
    iou = intersection_over_union(pred, target, num_classes, ignore_index)
    f1 = f1_score(pred, target, num_classes, ignore_index)
    class_acc = class_accuracy(pred, target, num_classes, ignore_index)
    return {
        "PA": pixel_accuracy(pred, target, ignore_index),
        "MCA": mean_class_accuracy(pred, target, num_classes, ignore_index),
        "FwIoU": frequency_weighted_iou(pred, target, num_classes, ignore_index),
        "FwF1": frequency_weighted_f1(pred, target, num_classes, ignore_index),
        "IoU_per_class": iou,
        "F1_per_class": f1,
        "Class_accuracy": class_acc,
    }
