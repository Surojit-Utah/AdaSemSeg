"""Performance measure code (get_intersection_and_union) adapted from https://github.com/dvlab-research/PFENet.git
"""
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import numpy as np
import time
import imageio
import torch
import torch.nn.functional as F
from utils.recursive_functions import recursive_to
from utils.general_util import revert_normalization, COLOR_RED, COLOR_WHITE, show_support_images, show_query_pred_image
from predict.Metric_scores import runningScore
from sklearn.metrics import classification_report


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
    def __init__(self, num_classes, data_visualization_path, device):
        self._visualization_path = data_visualization_path
        self._device = device
        self._issued_warnings = set()
        self._seg_threshold = 0.5

        self.runningScore_obj = runningScore(num_classes)

    def _get_visualization_summary(self, class_indices, images, segmentations_masks):

        visualization = None
        bit_length = 3
        pred_over_all_classes = torch.zeros_like(images).to(torch.bool)
        for class_index in class_indices:
            target = (segmentations_masks == class_index).cpu().detach().float()
            target_bool = (segmentations_masks == class_index).cpu().detach().bool()
            pred_over_all_classes = torch.logical_or(pred_over_all_classes, target_bool)

            # Select the RED channel from the RGB and concat the same after unsqueezing it
            # This helps us in visualizing the predicted output
            single_channel_img = images[:, :, 0, :, :]
            single_channel_img = torch.unsqueeze(single_channel_img, dim=2)
            images = torch.cat((single_channel_img, single_channel_img, single_channel_img), dim=2)

            class_index_binary = np.binary_repr(class_index, width=bit_length)
            class_index_binary = torch.tensor(np.array([int(d) for d in class_index_binary])).view(3, 1, 1)

            if visualization is None:
                visualization = target*(0.5 * images + 0.5 * class_index_binary)
            else:
                visualization += target*(0.5*images + 0.5*class_index_binary)

        background = (pred_over_all_classes == 0).cpu().detach().float()
        visualization += background*images

        visualization = (visualization * 255).byte()

        return visualization

    def _get_iou(self, segs, segannos):
        """Note that this is an IoU-measure used only during training. It is not the same as the
        IoU reported during evaluation and more reminiscent of the IoU-measure used in the Video
        Object Segmentation problem.
        Args:
            segs (LongTensor(B, H, W))
            segannos (LongTensor(B, H, W))
        Returns:
            iou (Tensor) averaged over the batch size
        """

        segs = segs.clone()
        mask_pred = (segs == 1)
        mask_anno = (segannos == 1)
        intersection = (mask_pred * mask_anno).sum(dim=(1, 2))
        union = mask_pred.sum(dim=(1, 2)) + mask_anno.sum(dim=(1, 2)) - intersection
        iou = torch.div((intersection.to(torch.float32)), (union.to(torch.float32) + 1e-5)).mean()

        return iou

    def _get_iou_summary(self, class_indices, segs, segannos):

        segs = segs.clone()
        segannos = torch.squeeze(segannos, 1)
        segannos = torch.squeeze(segannos, 1)
        iou_array = np.zeros(len(class_indices))
        for class_index, class_label in enumerate(class_indices):
            mask_pred = (segs == class_label)
            mask_anno = (segannos == class_label)
            intersection = (mask_pred * mask_anno).sum(dim=(1, 2))
            union = mask_pred.sum(dim=(1, 2)) + mask_anno.sum(dim=(1, 2)) - intersection
            iou = torch.div((intersection.to(torch.float32)), (union.to(torch.float32) + 1e-5)).mean()
            iou_array[class_index] = iou.detach().cpu().numpy()

        return iou_array


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

    def _vis_query_example(self, vis_data, model, class_indices, iter_num):

        print("Inside vis_query_example....")
        self._support_path = os.path.join(self._visualization_path, str(iter_num))
        if not os.path.isdir(self._support_path):
            os.makedirs(self._support_path, exist_ok=True)

        self._query_path = os.path.join(self._visualization_path, str(iter_num), 'Query_pred')
        if not os.path.isdir(self._query_path):
            os.makedirs(self._query_path, exist_ok=True)

        # Visualize support images
        B, _, _, _, _ = vis_data['support_images'].size()
        support_images = vis_data['support_images'].cpu().detach()
        support_images_np = support_images.numpy().transpose(0, 1, 3, 4, 2)
        support_anno_vis = self._get_visualization_summary(class_indices, support_images,
                                                           vis_data['support_segmentations'])
        support_anno_vis_np = support_anno_vis.numpy().transpose(0, 1, 3, 4, 2)
        for b in range(B):
            support_images_path = os.path.join(self._support_path, 'Support_images_batch_' + str(b + 1) + '.png')
            show_support_images(support_images_np[b], support_anno_vis_np[b], support_images_path)

        # Visualize query images
        with torch.no_grad():
            query_image, query_segmentation, pred_summary = self._evaluate_episode(model, vis_data, 0, class_indices,
                                                                                   visualize=True)
            # Input image
            query_image = query_image.cpu().detach()
            query_image_np = query_image.numpy().transpose(0, 1, 3, 4, 2)
            # Segmentation Label
            query_anno_vis = self._get_visualization_summary(class_indices, query_image, query_segmentation)
            query_anno_vis_np = query_anno_vis.numpy().transpose(0, 1, 3, 4, 2)
            # Prediction for IoU score
            query_pred_vis = self._get_visualization_summary(class_indices, query_image, pred_summary)
            query_pred_vis_np = query_pred_vis.numpy().transpose(0, 1, 3, 4, 2)

            B, _, _, _, _ = query_image.size()
            for b in range(B):
                query_pred_path = os.path.join(self._query_path, 'Pred_Image_batch_' + str(b + 1) + '.png')
                show_query_pred_image(query_image_np[b].squeeze(axis=0), query_anno_vis_np[b].squeeze(axis=0),
                                      query_pred_vis_np[b].squeeze(axis=0), query_pred_path)


    def _evaluate_episode(self, model, data, episode, class_indices, visualize=False):

        cosine_scaler = 20
        predict_act = torch.nn.Softmax2d()

        data = recursive_to(data, self._device)

        support_images = data['support_images']
        support_segmentations = data['support_segmentations']
        support_split_masks = data['support_split_masks']

        query_image = data['query_image']
        query_segmentation = data['query_segmentation']
        query_split_masks = data['query_split_masks']

        # encoded support images used for estimating the prototypes
        S_B, S_S, S_C, S_H, S_W = support_images.size()
        encoder_input = support_images.view(S_B * S_S, S_C, S_H, S_W)
        support_features = model.image_encoder(encoder_input)
        encoded_support_images_prototype = support_features['f32']
        _, S_C, S_H, S_W = encoded_support_images_prototype.size()

        # encoded query image to be compared with the prototypes along the channels
        Q_B, Q_S, Q_C, Q_H, Q_W = query_image.size()
        encoder_input = query_image.view(Q_B * Q_S, Q_C, Q_H, Q_W)
        query_features = model.image_encoder(encoder_input)
        encoded_query_image_prototype = query_features['f32']

        # class_indices = np.arange(num_classes).tolist() #[0, 1, 2, 3, 4, 5]
        iou_per_class = np.zeros(len(class_indices))
        pred_array_size = (support_split_masks.shape[0], len(class_indices),
                           support_split_masks.shape[4], support_split_masks.shape[5])
        pred_per_class = torch.zeros(pred_array_size).to(self._device)
        for class_index, class_label in enumerate(class_indices):
            cur_fg_mask = support_split_masks[:, :, class_label - 1]
            cur_bg_mask = 1 - support_split_masks[:, :, class_label - 1]

            # Binary masks reduced to the size of the encoded feature maps
            F_B, F_S, _, F_H, F_W = cur_fg_mask.size()
            cur_fg_mask = cur_fg_mask.view(F_B * F_S, 1, F_H, F_W)
            cur_bg_mask = cur_bg_mask.view(F_B * F_S, 1, F_H, F_W)
            encoded_support_fg_mask_prototype = torch.nn.Upsample(size=(S_H, S_W), mode='bilinear', align_corners=True)(cur_fg_mask)
            encoded_support_bg_mask_prototype = torch.nn.Upsample(size=(S_H, S_W), mode='bilinear', align_corners=True)(cur_bg_mask)

            # Foreground prototype
            fg_masked_fts = torch.sum(encoded_support_images_prototype * encoded_support_fg_mask_prototype, dim=(2, 3)) \
                            / (encoded_support_fg_mask_prototype.sum(dim=(2, 3)) + 1e-5)  # S_B*S_S x C
            fg_masked_fts = torch.mean(fg_masked_fts, dim=0)  # masked average pooling over examples in a support batch

            # Background prototype
            bg_masked_fts = torch.sum(encoded_support_images_prototype * encoded_support_bg_mask_prototype, dim=(2, 3)) \
                            / (encoded_support_bg_mask_prototype.sum(dim=(2, 3)) + 1e-5)  # S_B*S_S x C
            bg_masked_fts = torch.mean(bg_masked_fts, dim=0)  # masked average pooling over examples in a support batch

            # Similarity scores
            fg_similarity_score = F.cosine_similarity(encoded_query_image_prototype,
                                                      fg_masked_fts[None, ..., None, None], dim=1) * cosine_scaler
            bg_similarity_score = F.cosine_similarity(encoded_query_image_prototype,
                                                      bg_masked_fts[None, ..., None, None], dim=1) * cosine_scaler

            # Softmax on the similarity scores
            fg_similarity_score = torch.unsqueeze(fg_similarity_score, dim=1)
            bg_similarity_score = torch.unsqueeze(bg_similarity_score, dim=1)
            similarity_score = torch.cat([fg_similarity_score, bg_similarity_score], dim=1)
            similarity_score = predict_act(similarity_score)

            # query prediction
            query_features['f32'] = similarity_score  # fed as input to the decoder
            query_pred = model.image_decoder(query_features)

            GT_B, GT_M, _, GT_H, GT_W = query_split_masks[:, :, class_index].size()
            groundtruth = query_split_masks[:, :, class_index].view(GT_B * GT_M, GT_H, GT_W).long()

            output_segs = predict_act(query_pred)
            output_segs = output_segs[:, 1, :, :]

            pred_per_class[0, class_index, :, :] = output_segs

            output_segs_threshold = (output_segs > self._seg_threshold).type(torch.uint8)
            iou_score = self._get_iou(output_segs_threshold.to(torch.int64),
                                      groundtruth.to(torch.int64))
            iou_per_class[class_index] = iou_score.detach().cpu().numpy()

        pred_summary = torch.argmax(pred_per_class, axis=1) + 1
        if visualize:
            return query_image, query_segmentation, pred_summary

        iou_summary = self._get_iou_summary(class_indices, pred_summary.to(torch.int64),
                                                  query_segmentation.to(torch.int64))

        # Metric evaluation for each sample
        gt_np = query_segmentation.detach().cpu().numpy()
        pred_np = pred_summary.to(torch.int64).detach().cpu().numpy()
        self.runningScore_obj.update(gt_np, pred_np)

        print("Episode       : " + str(episode + 1))
        print("Classwise IoU score  : " + str(iou_per_class))
        print("Average IoU score    : " + str(np.mean(iou_per_class)))
        print(iou_summary)
        print(np.mean(iou_summary))

        return iou_summary, gt_np, pred_np


    def evaluate(self, model, dataloader, visualize=False):

        class_labels = np.unique(dataloader.train_label_vol).astype(np.uint8)
        class_labels = (class_labels[class_labels > 0]).tolist()
        num_classes = len(class_labels)

        num_query_examples = len(dataloader.query_slice_indices)
        iou_score_by_class = np.zeros((num_query_examples, num_classes))
        train_iou_list = []
        tricky_examples = []
        if 'f3' in dataloader.class_name:
            image_height = 256
        else:
            image_height = dataloader.train_data_vol.shape[-1]
        image_width = dataloader.query_slice_width[1] - dataloader.query_slice_width[0]
        image_height = (image_height//32)*32
        image_width = (image_width//32)*32
        gt_labels = np.zeros((num_query_examples, image_height, image_width), dtype=np.int8)
        pred_labels = np.zeros((num_query_examples, image_height, image_width), dtype=np.int8)
        iter_per_epoch = num_query_examples//dataloader.batch_size
        with torch.no_grad():
            total_time_taken = 0
            for iter_num in range(iter_per_epoch):

                # Gets a minibatch from the train or val data loader
                data = dataloader.get_minibatch()

                if visualize:
                    self._vis_query_example(data, model, class_labels, iter_num+1)

                start_time = time.time()
                class_iou_scores, gt_np, pred_np = self._evaluate_episode(model, data, iter_num, class_labels, visualize=False)
                gt_labels[iter_num] = gt_np
                pred_labels[iter_num] = pred_np
                gpu_mem = torch.cuda.max_memory_allocated() // 1000 ** 2
                time_taken = time.time() - start_time
                total_time_taken += time_taken
                print("GPU memory used in inference : " + str(gpu_mem))
                print("Inference time               : " + str(time.time()-start_time))

                if np.mean(class_iou_scores) < 1e-04:
                    print("Something wrong with example : " + str(iter_num + 1))
                    tricky_examples.append(iter_num + 1)
                    continue
                iou_score_by_class[iter_num] = class_iou_scores
                train_iou_list.append(np.mean(class_iou_scores))
            avg_time_taken = total_time_taken/iter_per_epoch
            print(total_time_taken, iter_per_epoch, avg_time_taken)
            print("Average time taken : " + str(avg_time_taken))

            iou_list = []
            for entry in train_iou_list:
                iou_list.append(float(entry))

            method1_avg = np.mean(np.array(train_iou_list))
            print("Average IoU method 1 : " + str(method1_avg))


        # Metric evaluation
        metric_scores_dict = self.runningScore_obj.get_scores()
        print(metric_scores_dict)

        # This is another approach to compute the F1 scores of the classes
        # This was used to validate the F1 scores computed using the confusion matrix in *Metric_scores.py*
        # target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5']
        # print(classification_report(y_true=gt_labels.flatten()-1, y_pred=pred_labels.flatten()-1, labels=[0, 1, 2, 3, 4, 5], target_names=target_names))

        # save the predictions on the query images as numpy volume
        direction = 'inline' if dataloader.data_axis==0 else 'crossline'
        save_pred_np_path = os.path.join(self._visualization_path, 'Query_pred_' + direction + '.npy')
        np.save(save_pred_np_path, pred_labels)

        eval_time_stats = {
        'avg_time_taken': avg_time_taken,
        'num_samples': len(iou_list),
        'image_height': image_height,
        'image_width': image_width
        }

        return iou_score_by_class, iou_list, tricky_examples, metric_scores_dict, eval_time_stats