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


def show_query_pred_image(input_image, anno_input_image, pred_input_image, query_pred_path):

    num_images = 1
    height = input_image.shape[0]
    width = input_image.shape[1]
    wspace = 0.05
    hspace = 0.05

    fig = plt.figure()
    fig_dpi = fig.get_dpi()
    h_inches = ((num_images*height)//fig_dpi)+1
    w_inches = ((3*width)//fig_dpi)+1
    fig.set_size_inches(w_inches, h_inches)
    gs = fig.add_gridspec(num_images, 3)
    gs.update(wspace=wspace)
    gs.update(hspace=hspace)

    # Clipping the Range [0, 255]
    input_image = (input_image * 255.0).astype(np.uint8)
    input_image = np.clip(input_image, 0, 255)
    single_channel = np.expand_dims(input_image[:, :, 0], axis=2)
    input_image = np.concatenate((single_channel, single_channel, single_channel), axis=2)
    anno_input_image = anno_input_image.astype(np.uint8)
    pred_input_image = pred_input_image.astype(np.uint8)

    # Debug
    # img_path = os.path.join(os.path.split(query_pred_path)[0], 'query_image.png')
    # imageio.imwrite(img_path, input_image)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.axis('off')
    plt.imshow(input_image)

    ax = fig.add_subplot(gs[0, 1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.axis('off')
    plt.imshow(anno_input_image)

    ax = fig.add_subplot(gs[0, 2])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.axis('off')
    plt.imshow(pred_input_image)

    plt.tight_layout()
    plt.savefig(query_pred_path)
    plt.close(plt.gcf())

    return


def show_combined_model_pred_images(input_images, anno_input_images, pred_input_images, query_pred_batch_path):

    num_images = input_images.shape[0]
    fig = plt.figure(figsize=(num_images, 3))
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
    fig = plt.figure(figsize=(num_images, 2))
    gs = fig.add_gridspec(2, num_images)
    gs.update(wspace=0.05)

    for img_index in range(num_images):

        img = input_images[img_index, :, :, :]
        anno_img = anno_images[img_index, :, :, :]

        # Clipping the Range [0, 255]
        img = (img * 255.0).astype(np.uint8)
        img = np.clip(img, 0, 255)
        anno_img = anno_img.astype(np.uint8)
        anno_img = np.clip(anno_img, 0, 255)

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