"""
Test dataset for AdaSemSeg evaluation.

Supports both K-shot random support selection and nearest-slice support selection
(required for reproducing Table 1 and the Parihaka results in the paper).
"""
import sys
sys.path.append('..')
from PIL import Image
from data.transform import Compose, Resize, ToTensor
from torch.utils.data import DataLoader, Dataset
import os
import torch
import gc
import numpy as np
import json
import copy


class AdaSemSegTestDataset(Dataset):
    """Patch-based test dataset for AdaSemSeg.

    Loads train volumes for support slices and test volumes for query slices.
    For F3 the test volume is separate (test1/test2). For Parihaka and Penobscot
    the test region is taken from the same volume using the split JSON.
    """

    def __init__(self, classes, data_info, patch_size=256, k_shot=5,
                 use_nearest_slice=False, eval_mode='test', batch_size=1,
                 patch_overlap=0.5, debug=False):

        self.classes = classes
        self.data_info = data_info
        self.patch_size = patch_size
        self.k_shot = k_shot
        self.use_nearest_slice = use_nearest_slice
        self.eval_mode = eval_mode
        self.batch_size = batch_size
        self.patch_overlap = patch_overlap
        self.debug = debug

        self.train_data_vol = dict()
        self.train_label_vol = dict()
        self.test_data_vol = dict()
        self.test_label_vol = dict()
        self.class_labels = dict()
        self.class_label_count = dict()
        self.support_slice_indices = dict()
        self.query_slice_indices = dict()
        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        query_class, query_example, support_examples = self.sample_episode(idx)
        query_img, query_mask, query_split_masks, support_imgs, support_masks, support_split_masks = \
            self.load_frame(query_class, query_example, support_examples)

        support_set = []
        for s_i, s_cm, s_sm in zip(support_imgs, support_masks, support_split_masks):
            cur_trans = Compose([Resize(self.patch_size), ToTensor()])
            support_set.append(cur_trans(s_i, s_cm, s_sm))

        support_imgs = torch.stack([i for i, _, _ in support_set])
        support_masks = torch.stack([m for _, m, _ in support_set])
        support_split_masks = torch.stack([s_m for _, _, s_m in support_set])

        query_transform = Compose([Resize(self.patch_size), ToTensor()])
        query_img, query_mask, query_split_masks = query_transform(
            query_img[0], query_mask[0], query_split_masks[0])
        query_img = torch.unsqueeze(query_img, dim=0)
        query_mask = torch.unsqueeze(query_mask, dim=0)
        query_split_masks = torch.unsqueeze(query_split_masks, dim=0)

        return {
            'data_class': query_class,
            'query_image': query_img,
            'query_segmentation': query_mask,
            'query_split_masks': query_split_masks,
            'support_images': support_imgs,
            'support_segmentations': support_masks,
            'support_split_masks': support_split_masks,
            'query_slice_index': self.img_metadata[idx][1],
            'query_row': self.img_metadata[idx][2],
            'query_col': self.img_metadata[idx][3],
        }

    def get_data_stat(self, class_name, data_axis, train_slices, data_split_filepath):
        data_split_fptr = open(data_split_filepath)
        data_split_dict = json.load(data_split_fptr)

        direction = 'inline' if data_axis == 0 else 'xline'

        # Support (train) slice indices
        if train_slices == 'all':
            train_min_index_key_name = 'train_' + direction + '_min'
            train_max_index_key_name = 'train_' + direction + '_max'
            train_min_index = data_split_dict[train_min_index_key_name]
            train_max_index = data_split_dict[train_max_index_key_name]
            support_indices = np.arange(train_min_index, train_max_index).tolist()
        else:
            train_key_name = 'train_' + direction + '_' + train_slices
            support_indices = data_split_dict[train_key_name]

        # Query (test) slice indices
        if self.eval_mode in ('val', 'test'):
            if 'f3' in class_name:
                # F3 uses separate test volumes; all slices in the test volume are evaluated
                query_indices = np.arange(self.test_data_vol[class_name].shape[data_axis]).tolist()
            else:
                eval_min_key_name = self.eval_mode + '_' + str(data_axis + 1) + '_' + direction + '_min'
                eval_max_key_name = self.eval_mode + '_' + str(data_axis + 1) + '_' + direction + '_max'
                eval_min_index = data_split_dict[eval_min_key_name]
                eval_max_index = data_split_dict[eval_max_key_name]
                query_indices = np.arange(eval_min_index, eval_max_index).tolist()
        else:
            raise ValueError(f"Unsupported eval_mode: {self.eval_mode}")

        return support_indices, query_indices

    def build_img_metadata(self):
        img_metadata = []
        for class_name in self.classes:
            train_datapath = self.data_info[class_name]['train_data_dir']
            train_data_vol_name = self.data_info[class_name]['train_data_vol_name']
            train_label_vol_name = self.data_info[class_name]['train_label_vol_name']
            test_datapath = self.data_info[class_name]['test_data_dir']
            test_data_vol_name = self.data_info[class_name]['test_data_vol_name']
            test_label_vol_name = self.data_info[class_name]['test_label_vol_name']
            data_split_filepath = self.data_info[class_name]['train_val_test_split']
            train_slices = self.data_info[class_name]['train_indices']
            data_axis = self.data_info[class_name]['axis']

            # Load train volume (for support)
            train_data = np.load(os.path.join(train_datapath, train_data_vol_name))
            min_intensity, max_intensity = np.min(train_data), np.max(train_data)
            self.train_data_vol[class_name] = (((train_data - min_intensity) / (max_intensity - min_intensity)) * 255).astype(np.uint8)

            if 'penobscot' in class_name:
                self.train_label_vol[class_name] = np.load(os.path.join(train_datapath, train_label_vol_name)).astype(np.uint8)
            else:
                self.train_label_vol[class_name] = np.load(os.path.join(train_datapath, train_label_vol_name)).astype(np.uint8) + 1

            # Load test volume (for query)
            test_data = np.load(os.path.join(test_datapath, test_data_vol_name))
            min_intensity, max_intensity = np.min(test_data), np.max(test_data)
            self.test_data_vol[class_name] = (((test_data - min_intensity) / (max_intensity - min_intensity)) * 255).astype(np.uint8)

            if 'penobscot' in class_name:
                self.test_label_vol[class_name] = np.load(os.path.join(test_datapath, test_label_vol_name)).astype(np.uint8)
            else:
                self.test_label_vol[class_name] = np.load(os.path.join(test_datapath, test_label_vol_name)).astype(np.uint8) + 1

            class_labels = np.unique(self.train_label_vol[class_name])
            self.class_labels[class_name] = (class_labels[class_labels > 0]).tolist()
            self.class_label_count[class_name] = len(self.class_labels[class_name])

            support_indices, query_indices = self.get_data_stat(class_name, data_axis, train_slices, data_split_filepath)
            self.support_slice_indices[class_name] = support_indices
            self.query_slice_indices[class_name] = query_indices

            # Build patch grid for each query slice
            test_vol = self.test_data_vol[class_name]
            if data_axis == 0:
                slice_shape = (test_vol.shape[1], test_vol.shape[2])
            else:
                slice_shape = (test_vol.shape[0], test_vol.shape[2])

            image_height, image_width = slice_shape
            min_row = self.patch_size // 2
            max_row = image_height - (self.patch_size // 2)
            min_col = self.patch_size // 2
            max_col = image_width - (self.patch_size // 2)
            row_sep = max(4, int(self.patch_size * self.patch_overlap))
            col_sep = max(4, int(self.patch_size * self.patch_overlap))

            valid_row_indices = np.arange(min_row, max_row, row_sep).tolist()
            valid_col_indices = np.arange(min_col, max_col, col_sep).tolist()

            for query_slice in query_indices:
                for row_index in valid_row_indices:
                    for col_index in valid_col_indices:
                        img_metadata.append([class_name, query_slice, row_index, col_index])

            print(f'Total (test) patches for class {class_name}: {len(img_metadata)}')

        return img_metadata

    def sample_episode(self, idx):
        query_class, query_slice, row_index, col_index = self.img_metadata[idx]
        query_example = [[query_slice, row_index, col_index]]
        support_examples = []
        for _ in range(self.k_shot):
            support_examples.append([query_slice, row_index, col_index])

        support_indices = np.array(self.support_slice_indices[query_class])
        if self.use_nearest_slice:
            distances = np.abs(support_indices - query_slice)
            nearest_order = np.argsort(distances)
            support_sample_index = support_indices[nearest_order[:self.k_shot]].tolist()
        elif len(support_indices) < self.k_shot:
            support_sample_index = np.random.choice(support_indices, self.k_shot, replace=True).tolist()
        else:
            support_sample_index = np.random.choice(support_indices, self.k_shot, replace=False).tolist()

        for set_support_sample_index in range(self.k_shot):
            support_examples[set_support_sample_index][0] = support_sample_index[set_support_sample_index]

        return query_class, query_example, support_examples

    def load_frame(self, query_class, query_example, support_examples):
        query_img, query_mask, query_split_masks = self.read_data(query_class, query_example, mode='query')
        support_imgs, support_masks, support_split_masks = self.read_data(query_class, support_examples, mode='support')
        gc.collect()
        return query_img, query_mask, query_split_masks, support_imgs, support_masks, support_split_masks

    def read_data(self, data_class, img_info, mode='query'):
        data_axis = self.data_info[data_class]['axis']
        data_vol = self.test_data_vol[data_class] if mode == 'query' else self.train_data_vol[data_class]
        label_vol = self.test_label_vol[data_class] if mode == 'query' else self.train_label_vol[data_class]

        mask_list = []
        image_list = []
        all_split_mask_list = []

        for info in img_info:
            img_index = info[0]
            row_index = info[1]
            col_index = info[2]

            if data_axis == 0:
                cur_img = copy.deepcopy(data_vol[img_index].T)
                label_data = copy.deepcopy(label_vol[img_index].T)
            else:
                cur_img = copy.deepcopy(data_vol[:, img_index, :].T)
                label_data = copy.deepcopy(label_vol[:, img_index, :].T)

            if data_vol.shape[-1] < self.patch_size:
                # F3 case
                img_patch = np.ones((self.patch_size, self.patch_size)) * (np.max(data_vol) / 2)
                data_start_index = (self.patch_size - data_vol.shape[-1]) // 2 + 1
                data_end_index = data_start_index + data_vol.shape[-1]
                img_patch[data_start_index:data_end_index] = copy.deepcopy(
                    cur_img[:, (col_index - self.patch_size // 2):(col_index + self.patch_size // 2)])

                label_patch = np.zeros((self.patch_size, self.patch_size))
                label_patch[data_start_index:data_end_index] = copy.deepcopy(
                    label_data[:, (col_index - self.patch_size // 2):(col_index + self.patch_size // 2)])
            else:
                img_patch = copy.deepcopy(cur_img[
                    (row_index - self.patch_size // 2):(row_index + self.patch_size // 2),
                    (col_index - self.patch_size // 2):(col_index + self.patch_size // 2)])
                label_patch = copy.deepcopy(label_data[
                    (row_index - self.patch_size // 2):(row_index + self.patch_size // 2),
                    (col_index - self.patch_size // 2):(col_index + self.patch_size // 2)])

            image_list.append(Image.fromarray(img_patch).convert("RGB"))
            mask_list.append(Image.fromarray(label_patch).convert("L"))

            split_mask_list = []
            label_patch_one_hot = torch.nn.functional.one_hot(
                torch.from_numpy(label_patch).to(torch.int64), num_classes=self.class_label_count[data_class] + 1).numpy()
            for mask_index in self.class_labels[data_class]:
                pil_split_mask = Image.fromarray((label_patch_one_hot[:, :, mask_index] * 255).astype(np.uint8))
                split_mask_list.append(pil_split_mask.convert("L"))
            all_split_mask_list.append(split_mask_list)

        return image_list, mask_list, all_split_mask_list
