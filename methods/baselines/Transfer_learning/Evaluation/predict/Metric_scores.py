import numpy as np

'''
Using the Confusion matrix computed using a unique mechanism:
Adapted from https://github.com/yalaudah/facies_classification_benchmark/blob/main/core/metrics.py
https://github.com/meetps/pytorch-semseg/blob/master/ptsemseg/metrics.py
https://github.com/wkentaro/pytorch-fcn/blob/main/torchfcn/utils.py

Using the TP, FP, TN, FN:
https://github.com/hsiangyuzhao/Segmentation-Metrics-PyTorch#pixel-accuracy
'''
class runningScore:

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        # Masking helps in taking care of the 0th class above the horizon
        # as the index for that class would be -1
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        # For this evaluation technique the class labels starts at 0
        label_trues, label_preds = label_trues - 1, label_preds - 1
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.around(np.diag(hist).sum() / hist.sum(), 2)
        acc_cls = np.around(np.diag(hist) / hist.sum(axis=1), 2)
        mean_acc_cls = np.around(np.nanmean(acc_cls), 2)
        iu = np.around(np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)), 2)
        mean_iu = np.around(np.nanmean(iu[iu>0.0]), 2)
        freq = np.around(hist.sum(axis=1) / hist.sum(), 2) # fraction of the pixels that come from each class
        fw_iou = np.around((freq[freq > 0] * iu[freq > 0]).sum(), 2)
        f1_score = np.around((2*np.diag(hist))/(hist.sum(axis=1) + hist.sum(axis=0)), 2)
        mean_f1_score = np.around(np.nanmean(f1_score[f1_score>0.0]), 2)
        fw_f1_score = np.around((freq[freq > 0] * f1_score[freq > 0]).sum(), 2)
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {'Pixel Acc: ': acc,
                'Class Accuracy: ': acc_cls.tolist(),
                'Mean Class Acc: ': mean_acc_cls,
                'Frequency: ': freq.tolist(),
                'Mean IoU: ': mean_iu,
                'Freq Weighted IoU: ': fw_iou,
                'mean_f1_score: ': mean_f1_score,
                'Freq Weighted F1 score: ': fw_f1_score,
                'f1_score: ': f1_score,
                # 'confusion_matrix: ': self.confusion_matrix,
                'classwise_IoU: ': cls_iu}

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))