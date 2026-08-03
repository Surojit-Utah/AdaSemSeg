"""
Qualitative figure-assembly helper, ported from the original AdaSemSeg research
workspace (SimCLR_init_dataset_v1). Combines a query image with its ground-truth
and predicted segmentation into the 3-row panel style used for the paper's
qualitative figures (e.g. Fig. 12/17/18).

See scripts/assemble_figures.py for how this is used together with
methods/adasemseg/evaluate.py's --save_predictions output.
"""
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import numpy as np


def show_combined_model_pred_images(input_images, anno_input_images, pred_input_images, query_pred_batch_path):
    num_images = input_images.shape[0]
    fig = plt.figure()
    gs = fig.add_gridspec(3, num_images)
    gs.update(wspace=0.05)

    for img_index in range(num_images):
        img = input_images[img_index, :, :, :]
        # Clipping the Range [0, 255]
        img = (img * 255.0).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
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
