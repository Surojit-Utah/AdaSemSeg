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

    def __call__(self, image):
        for t in self.segtransform:
            image = t(image)
        return image


class Contrast:
    def __init__(self, min_contrast=0.0, max_contrast=2.0, p=0.5):
        self._min_contrast = min_contrast
        self._max_contrast = max_contrast
        self.p = p

    def __call__(self, image):
        if np.random.random() < self.p:
            contrast_val = random.uniform(self._min_contrast, self._max_contrast)
            # tv.transforms.ColorJitter does not works
            image = tv.transforms.functional.adjust_contrast(image, contrast_val)

        return image


class Brightness:
    def __init__(self, min_brightness=0.5, max_brightness=1.5, p=0.5):
        self._min_brightness = min_brightness
        self._max_brightness = max_brightness
        self.p = p

    def __call__(self, image):
        if np.random.random() < self.p:
            brightness_val = random.uniform(self._min_brightness, self._max_brightness)
            # tv.transforms.ColorJitter does not works
            image = tv.transforms.functional.adjust_brightness(image, brightness_val)

        return image


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.], p=0.5):
        self.sigma = sigma
        self.p = p
    def __call__(self, image):
        if np.random.random() < self.p:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        return image


class GaussNoise:
    """Gaussian Noise to be applied to images that have been scaled to fit in the range 0-1"""

    def __init__(self, var_limit=(1e-4, 5e-2), p=0.5):
        self.var_limit = np.log(var_limit)
        self.p = p

    def __call__(self, image):
        sigma = np.exp(np.random.uniform(*self.var_limit)) ** 0.5
        image = tv.transforms.ToTensor()(image)
        noise = np.random.normal(0, sigma, size=image.shape).astype(np.float32)
        image = image + torch.from_numpy(noise)
        image = torch.clamp(image, 0, 1)

        return image


class RandomRotate:
    def __init__(self, rot_range=20, fill=0, p=0.5):
        self.rot_range = np.linspace(-rot_range, rot_range)
        self.input_fill = fill
        self.label_fill = fill
        self.p = p

    def __call__(self, image):
        if np.random.random() < self.p:
            angle = random.choice(self.rot_range)
            image = tv.transforms.functional.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, expand=True, fill=self.input_fill)
            return image
        else:
            return image


class Resize:
    # p = 1.0 is performed to ensure the image sizes are similar, typically after rotation
    def __init__(self, size, p=1.0):
        self.im_resize = tv.transforms.Resize(size)
        self.label_resize = tv.transforms.Resize(size, interpolation=InterpolationMode.NEAREST)
        self.p = p

    def __call__(self, image):
        if np.random.random() < self.p:
            image = self.im_resize(image)
            return image
        else:
            return image

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

    def __call__(self, image):
        if not torch.is_tensor(image):
            image = self.totensor(image)

        return image


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            image = F.hflip(image)

        return image


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