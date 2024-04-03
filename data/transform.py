import math
import random
from typing import Sequence
import warnings
import torch
import numpy as np
from PIL import Image, ImageFilter
import torchvision as tv
from torchvision.transforms.functional import InterpolationMode,rotate,_interpolation_modes_from_int
import torchvision.transforms.functional as F


class Compose:
    # Composes segtransforms: segtransform.Compose([segtransform.RandScale([0.5, 2.0]), segtransform.ToTensor()])
    def __init__(self, segtransform):
        self.segtransform = segtransform

    def __call__(self, image, label, split_mask_list):
        for t in self.segtransform:
            image, label, split_mask_list = t(image, label, split_mask_list)
        return image, label, split_mask_list


class Contrast:
    def __init__(self, min_contrast=0.0, max_contrast=2.0, p=0.5):
        self._min_contrast = min_contrast
        self._max_contrast = max_contrast
        self.p = p

    def __call__(self, image, label):
        if np.random.random() < self.p:
            contrast_val = random.uniform(self._min_contrast, self._max_contrast)
            # tv.transforms.ColorJitter does not works
            image = tv.transforms.functional.adjust_contrast(image, contrast_val)

        return image, label


class Brightness:
    def __init__(self, min_brightness=0.5, max_brightness=1.5, p=0.5):
        self._min_brightness = min_brightness
        self._max_brightness = max_brightness
        self.p = p

    def __call__(self, image, label):
        if np.random.random() < self.p:
            brightness_val = random.uniform(self._min_brightness, self._max_brightness)
            # tv.transforms.ColorJitter does not works
            image = tv.transforms.functional.adjust_brightness(image, brightness_val)

        return image, label


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.], p=0.5):
        self.sigma = sigma
        self.p = p
    def __call__(self, image, label, split_mask_list):
        if np.random.random() < self.p:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        return image, label, split_mask_list


class GaussNoise:
    """Gaussian Noise to be applied to images that have been scaled to fit in the range 0-1"""

    def __init__(self, var_limit=(1e-4, 1e-2), p=0.5):
        self.var_limit = np.log(var_limit)
        self.p = p

    def __call__(self, image, label, split_mask_list):
        sigma = np.exp(np.random.uniform(*self.var_limit)) ** 0.5
        image = tv.transforms.ToTensor()(image)
        noise = np.random.normal(0, sigma, size=image.shape).astype(np.float32)
        image = image + torch.from_numpy(noise)
        image = torch.clamp(image, 0, 1)

        return image, label, split_mask_list


class RandomRotate:
    def __init__(self, rot_range=30, fill=0, p=0.5):
        self.rot_range = np.linspace(-rot_range, rot_range)
        self.input_fill = fill
        self.label_fill = fill
        self.p = p

    def __call__(self, image, label, split_mask_list):
        if np.random.random() < self.p:
            angle = random.choice(self.rot_range)
            image = tv.transforms.functional.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, expand=True, fill=self.input_fill)
            label = tv.transforms.functional.rotate(label, angle, interpolation=InterpolationMode.NEAREST, expand=True, fill=self.label_fill)
            trans_split_mask_list = []
            for split_mask in split_mask_list:
                split_mask_label = tv.transforms.functional.rotate(split_mask, angle, interpolation=InterpolationMode.NEAREST, expand=True, fill=self.label_fill)
                trans_split_mask_list.append(split_mask_label)
            return image, label, trans_split_mask_list
        else:
            return image, label, split_mask_list


class Resize:
    # p = 1.0 is performed to ensure the image sizes are similar, typically after rotation
    def __init__(self, size, p=1.0):
        self.im_resize = tv.transforms.Resize(size)
        self.label_resize = tv.transforms.Resize(size, interpolation=InterpolationMode.NEAREST)
        self.p = p

    def __call__(self, image, label, split_mask_list):
        if np.random.random() < self.p:
            image = self.im_resize(image)
            label = self.label_resize(label)
            trans_split_mask_list = []
            for split_mask in split_mask_list:
                split_mask_label = self.label_resize(split_mask)
                trans_split_mask_list.append(split_mask_label)

            return image, label, trans_split_mask_list
        else:
            return image, label, split_mask_list

class Scale:
    def __init__(self, final_image_size=128, upsample_scaling_factor=[1.3, 1.5], downsample_scaling_factor=[0.8, 0.9], p=1.0):
        self._final_image_size = final_image_size
        self._adjust_size = ['upsample', 'downsample']
        self._upsample_scaling_factor = upsample_scaling_factor
        self._downsample_scaling_factor = downsample_scaling_factor
        self.p = p

    def __call__(self, image, label):
        # Apply the transformation
        if np.random.random() < self.p:
            width, height = image.size
            adjustment = self._adjust_size[np.random.randint(len(self._adjust_size), size=1)[0]]
            # print(adjustment)
            if adjustment=='upsample':
                scaling_factor = self._upsample_scaling_factor[np.random.randint(len(self._upsample_scaling_factor), size=1)[0]]
                # print(scaling_factor)
                new_width, new_height = int(scaling_factor*width), int(scaling_factor*height)
                resized_image = tv.transforms.Resize((new_height, new_width))(image)
                label_resize = tv.transforms.Resize((new_height, new_width), interpolation=InterpolationMode.NEAREST)(label)
                output_image = tv.transforms.CenterCrop(self._final_image_size)(resized_image)
                output_label = tv.transforms.CenterCrop(self._final_image_size)(label_resize)
            if adjustment=='downsample':
                scaling_factor = self._downsample_scaling_factor[np.random.randint(len(self._downsample_scaling_factor), size=1)[0]]
                # print(scaling_factor)
                new_width, new_height = int(scaling_factor*width), int(scaling_factor*height)
                resized_image = tv.transforms.Resize((new_height, new_width))(image)
                label_resize = tv.transforms.Resize((new_height, new_width), interpolation=InterpolationMode.NEAREST)(label)
                output_image = resized_image
                output_label = label_resize
        # Skip the transformation
        else:
            output_image, output_label = image, label

        return output_image, output_label


class ToTensor:
    def __init__(self) -> None:
        self.totensor = tv.transforms.ToTensor()

    # Jan 18, 2022: totensor() is also called for the label data
    def __call__(self, image, label, split_mask_list):
        if not torch.is_tensor(image):
            image = self.totensor(image)
        if not torch.is_tensor(label):
            arr_label = np.expand_dims(np.array(label), axis=0)
            label = torch.from_numpy(arr_label).type(torch.long)
        trans_split_mask = torch.zeros(len(split_mask_list), 1, label.shape[1], label.shape[2])
        for split_mask_index, split_mask in enumerate(split_mask_list):
            if not torch.is_tensor(split_mask):
                split_mask_label = self.totensor(split_mask).type(torch.long)
                trans_split_mask[split_mask_index] = split_mask_label

        return image, label, trans_split_mask


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label, split_mask_list):
        if random.random() < self.p:
            image = F.hflip(image)
            label = F.hflip(label)
            trans_split_mask_list = []
            for split_mask in split_mask_list:
                split_mask_label = F.hflip(split_mask)
                trans_split_mask_list.append(split_mask_label)

            return image, label, trans_split_mask_list
        else:
            return image, label, split_mask_list


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = F.vflip(image)
            label = F.vflip(label)
        return image, label


class RandomRotate90:
    def __call__(self, image, label):
        # Jan 18, 2022: Added 2 to the list
        rot = random.choice([0, 1, 2, 3])
        image, label = rotate(image,90*rot,expand=True), label.rot90(rot,[-2,-1])
        return image, label


class Normalize:
    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        if self.std is None:
            image.sub_(self.mean)
        else:
            image.sub_(self.mean).div_(self.std)

        return image, label
    

def _setup_size(size, error_msg):
    if isinstance(size, int):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


def convert_color_factory(src, dst):
    import cv2
    code = getattr(cv2, f'COLOR_{src.upper()}2{dst.upper()}')

    def convert_color(img):
        out_img = cv2.cvtColor(img, code)
        return out_img

    convert_color.__doc__ = f"""Convert a {src.upper()} image to {dst.upper()}
        image.
    Args:
        img (ndarray or str): The input image.
    Returns:
        ndarray: The converted {dst.upper()} image.
    """

    return convert_color


bgr2rgb = convert_color_factory('bgr', 'rgb')

rgb2bgr = convert_color_factory('rgb', 'bgr')

bgr2hsv = convert_color_factory('bgr', 'hsv')

hsv2bgr = convert_color_factory('hsv', 'bgr')

bgr2hls = convert_color_factory('bgr', 'hls')

hls2bgr = convert_color_factory('hls', 'bgr')