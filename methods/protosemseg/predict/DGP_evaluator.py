"""Performance measure code (get_intersection_and_union) adapted from https://github.com/dvlab-research/PFENet.git
"""
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import numpy as np
import torch
import sys
sys.path.append('..')
from utils.recursive_functions import recursive_to
from utils.general_util import revert_normalization, COLOR_RED, COLOR_WHITE


def show_combined_model_pred_images(input_images, anno_input_images, pred_input_images, query_pred_batch_path):
    num_images = input_images.shape[0]
    fig = plt.figure()
    gs = fig.add_gridspec(3, num_images)
    gs.update(wspace=0.05)

    for img_index in range(num_images):
        img = input_images[img_index, :, :, :]
        # Clipping the Range [0, 255]
        img = (img * 255.0).astype(np.uint8)
        img = np.clip(img, 0, 255)

        anno_img = anno_input_images[img_index, :, :, :]
        anno_img = anno_img.astype(np.uint8)

        pred_img = pred_input_images[img_index, :, :, :]
        pred_img = pred_img.astype(np.uint8)

        ax = fig.add_subplot(gs[0, img_index])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.axis('off')
        plt.imshow(img)

        ax = fig.add_subplot(gs[1, img_index])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.axis('off')
        plt.imshow(anno_img)

        ax = fig.add_subplot(gs[2, img_index])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.axis('off')
        plt.imshow(pred_img)

    plt.savefig(query_pred_batch_path)
    plt.close(plt.gcf())

    return


class FSSEvaluator:
    def __init__(self, data_visualization_path, device):
        self._visualization_path = data_visualization_path
        self._device = device
        self._issued_warnings = set()

        self._test_mode_path = os.path.join(self._visualization_path, 'Mode_test')
        if not os.path.isdir(self._test_mode_path):
            os.makedirs(self._test_mode_path, exist_ok=True)


    def _get_visualization(self, images, segmentations):
        background = (segmentations == 0).cpu().detach().float()
        target = (segmentations == 1).cpu().detach().float()
        visualization = (background * images
                         + target * (0.5*images + 0.5*COLOR_RED))
        visualization = (visualization * 255).byte()
        return visualization

    def _plot_data(self, data, model_output, episode):

        query_pred_path = self._test_mode_path

        support_images = revert_normalization(data['support_images'].cpu().detach())
        _, S, _, _, _ = support_images.size()
        support_images_np = support_images.numpy().transpose(0, 1, 3, 4, 2)

        support_anno_vis = self._get_visualization(support_images, data['support_segmentations'])
        support_anno_vis_np = support_anno_vis.numpy().transpose(0, 1, 3, 4, 2)

        query_images = revert_normalization(data['query_image'].cpu().detach())
        B, Q, _, H, W = query_images.size()
        query_images_np = query_images.numpy().transpose(0, 1, 3, 4, 2)

        query_anno_vis = self._get_visualization(query_images, data['query_segmentation'])
        query_anno_vis_np = query_anno_vis.numpy().transpose(0, 1, 3, 4, 2)

        query_pred_vis = self._get_visualization(query_images, model_output)
        query_pred_vis_np = query_pred_vis.numpy().transpose(0, 1, 3, 4, 2)

        input_images = np.zeros((B, (S + Q), H, W, 3))
        input_images[:, :S, :, :, :] = support_images_np
        input_images[:, S:, :, :, :] = query_images_np

        anno_input_images = np.zeros((B, (S + Q), H, W, 3))
        anno_input_images[:, :S, :, :, :] = support_anno_vis_np
        anno_input_images[:, S:, :, :, :] = query_anno_vis_np

        pred_input_images = np.zeros((B, (S + Q), H, W, 3))
        pred_input_images[:, :S, :, :, :] = support_anno_vis_np
        pred_input_images[:, S:, :, :, :] = query_pred_vis_np

        for b in range(B):
            query_pred_batch_path = os.path.join(query_pred_path, 'Pred_Image_' + str(episode + 1) + '.png')
            show_combined_model_pred_images(input_images[b], anno_input_images[b], pred_input_images[b],
                                            query_pred_batch_path)

    def _get_iou(self, segs, segannos):
        """Note that this is an IoU-measure used only during training. It is not the same as the
        IoU reported during evaluation and more reminiscent of the IoU-measure used in the Video
        Object Segmentation problem.
        Args:
            segs (LongTensor(B, Q, H, W))
            segannos (LongTensor(B, Q, H, W))
            classes (LongTensor(B, Q, C))
        Returns:
            iou (Tensor(B, Q, C))
        """

        B, Q, C, H, W = segs.size()

        iou = torch.zeros((B, Q, C), device=segs.device)
        segs = segs.clone()

        mask_pred = (segs == 1)
        mask_anno = (segannos == 1)
        intersection = (mask_pred * mask_anno).sum(dim=(2, 3, 4))
        union = mask_pred.sum(dim=(2, 3, 4)) + mask_anno.sum(dim=(2, 3, 4)) - intersection
        iou[:, :, C-1] = (intersection + 1e-5) / (union + 1e-5)

        return {'val': iou.mean(), 'N': B * Q * C}

    def _cell_seg_iou(self, preds, annos):
        """
        Args:
            preds (LongTensor(B, Q, H, W))
            annos (LongTensor(B, Q, H, W))
        Returns:
            average iou
        """
        iou = 0
        labels_tens = annos.type(torch.BoolTensor)
        if preds.shape[0] > 1:
            intersection_tens = (preds.squeeze().type(torch.BoolTensor) & labels_tens.squeeze()).float().sum((1, 2))
            union_tens = (preds.squeeze().type(torch.BoolTensor) | labels_tens.squeeze()).float().sum((1, 2))
            iou = torch.mean((intersection_tens + 1e-05) / (union_tens + 1e-05))
        else:
            intersection_tens = (preds.squeeze().type(torch.BoolTensor) & labels_tens.squeeze()).float().sum()
            union_tens = (preds.squeeze().type(torch.BoolTensor) | labels_tens.squeeze()).float().sum()
            iou = torch.mean((intersection_tens + 1e-05) / (union_tens + 1e-05))

        return iou

    def _evaluate_episode(self, model, data, episode, visualize=False):
        data = recursive_to(data, self._device)

        support_images = data['support_images']
        support_segmentations = data['support_segmentations']
        query_image = data['query_image']
        query_segmentation = data['query_segmentation']

        online_models = model.learn(support_images, support_segmentations)

        model_out = model(query_image, online_models)
        output_segs = (torch.sigmoid(model_out) > 0.7).float()

        if visualize and self._visualization_path is not None:
            self._plot_data(data, output_segs, episode)

        train_iou = self._get_iou(output_segs, data['query_segmentation'])['val']
        cell_seg_iou = self._cell_seg_iou(output_segs, data['query_segmentation'])

        print("Train IoU     : " + str(train_iou.data.cpu().numpy()))
        print("Cell seg IoU  : " + str(cell_seg_iou.data.cpu().numpy()))

        return (train_iou, cell_seg_iou)

    def evaluate(self, model, dataloader):

        train_iou_list = []
        cell_seg_iou_list = []
        with torch.no_grad():
            for episode, data in enumerate(dataloader):
                train_iou, cell_seg_iou = self._evaluate_episode(model, data, episode, visualize=True)
                train_iou_list.append(train_iou)
                cell_seg_iou_list.append(cell_seg_iou)
            method1_avg = np.mean(np.array(train_iou_list))
            method2_avg = np.mean(np.array(cell_seg_iou_list))
            print("Average IoU method 1 : " + str(method1_avg))
            print("Average IoU method 2 : " + str(method2_avg))

        return method1_avg, method2_avg