import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import numpy as np
COLOR_RED = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)
COLOR_MAGENTA = torch.tensor([1.0, 0.0, 1.0]).view(3, 1, 1)
COLOR_WHITE = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
        plt.imshow(img, vmin=0, vmax=255)

        ax = fig.add_subplot(gs[1, img_index])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.axis('off')
        plt.imshow(anno_img, vmin=0, vmax=255)

        ax = fig.add_subplot(gs[2, img_index])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.axis('off')
        plt.imshow(pred_img, vmin=0, vmax=255)

    plt.savefig(query_pred_batch_path)
    plt.close(plt.gcf())

    return


def show_combined_images(input_images, anno_images, save_img_path):

    num_images = input_images.shape[0]
    fig = plt.figure()
    gs = fig.add_gridspec(2, num_images)
    gs.update(wspace=0.05)

    for img_index in range(num_images):

        img = input_images[img_index, :, :, :]
        anno_img = anno_images[img_index, :, :, :]

        # Clipping the Range [0, 255]
        img = (img * 255.0).astype(np.uint8)
        img = np.clip(img, 0, 255)
        anno_img = anno_img.astype(np.uint8)

        ax = fig.add_subplot(gs[0, img_index])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.axis('off')
        plt.imshow(img, vmin=0, vmax=255)

        ax = fig.add_subplot(gs[1, img_index])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.axis('off')
        plt.imshow(anno_img, vmin=0, vmax=255)

    plt.savefig(save_img_path)
    plt.close(plt.gcf())

    return


def revert_normalization(sample):
    """
    sample (Tensor): of size (nsamples,nchannels,height,width)
    """
    # Imagenet mean and std
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    mean_tensor = torch.Tensor(mean).view(3,1,1).to(sample.device)
    std_tensor = torch.Tensor(std).view(3,1,1).to(sample.device)
    non_normalized_sample = sample*std_tensor + mean_tensor
    return non_normalized_sample