import sys
sys.path.append('..')
from PIL import Image
from data.transform import Compose, Normalize, Resize, ToTensor, RandomHorizontalFlip
from data.transform import RandomRotate, GaussianBlur, GaussNoise, Scale
from torch.utils.data import DataLoader, Dataset, RandomSampler
import os
import torch
import gc
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import random
import copy
from tqdm import tqdm
import time
import json
from config.local_config import create_config


class Seismic_Segmentation_Task():

    def __init__(self, classes, data_info, patch_size, k_shot=None, train_batch_size=10, val_batch_size=10, debug=False):

        self.traindataset = CustomDataset(classes, data_info, patch_size, train_batch_size, k_shot=k_shot, mode='train', debug=debug)
        self.train_batch_size = train_batch_size
        train_per_class_sample_indices = self.traindataset.per_class_sample_indices
        self.train_batch_sampler_obj = BatchSampler(train_per_class_sample_indices, batch_size=self.train_batch_size)
        self.train_loader = DataLoader(self.traindataset, batch_sampler=self.train_batch_sampler_obj, num_workers=1)

        self.validdataset = CustomDataset(classes, data_info, patch_size, val_batch_size, k_shot=k_shot, mode='val', debug=debug)
        self.val_batch_size = val_batch_size
        val_per_class_sample_indices = self.validdataset.per_class_sample_indices
        self.val_batch_sampler_obj = BatchSampler(val_per_class_sample_indices, batch_size=self.val_batch_size)
        self.val_loader = DataLoader(self.validdataset, batch_sampler=self.val_batch_sampler_obj, num_workers=1)


class CustomDataset(Dataset):

    def __init__(self, classes, data_info, patch_size, batch_size, k_shot=None, mode=None, debug=False):

        # importance sampling parameters
        self.train_cushion_sample_count = 10
        self.train_import_sample_count = 40
        self.train_skip_image_indices = 5

        self.classes = classes
        self.data_info = data_info
        self.patch_size = patch_size
        self.k_shot = k_shot
        self.batch_size = batch_size
        self.mode = mode
        self.transform = None
        self.debug = debug

        self.data_vol = dict()
        self.label_vol = dict()
        self.sel_mask_vol = dict()
        self.class_labels = dict()
        self.class_label_count = dict()
        self.slice_indices = dict()
        self.eval_support_slice_indices = dict()
        self.img_metadata = self.build_img_metadata()

        keep_entries = (len(self.img_metadata)//self.batch_size)*self.batch_size
        self.img_metadata = self.img_metadata[:keep_entries]

        self.img_metadata_classwise, self.per_class_sample_indices = self.build_img_metadata_classwise()

        # Min and Max range for the each class in *self.img_metadata*
        # This is required for the importance sampling
        self.per_class_sample_min_max_indices = dict()
        class_index = 0
        for class_name in self.classes:
            class_indices = self.per_class_sample_indices[class_index]
            self.per_class_sample_min_max_indices[class_name] = [min(class_indices), max(class_indices)]
            class_index += 1
        print(self.per_class_sample_min_max_indices)

        self.aug_dict = {1: RandomRotate(p=0.5),
                         2: RandomHorizontalFlip(p=0.5),
                         3: GaussianBlur(p=0.5),
                         4: GaussNoise(p=0.5)}

        self.total_aug_count = len(self.aug_dict.keys())

        # Debug messages
        self.support_aug_str = None
        self.query_aug_str = None
        self.support_train_aug = None
        self.support_val_aug = None
        self.query_train_aug = None
        self.query_val_aug = None


    def __len__(self):
        return len(self.img_metadata)


    def __getitem__(self, idx):

        query_class, query_example, support_examples = self.sample_episode(idx)

        # Load the query and support images as PIL images
        query_img, query_mask, query_split_masks, support_imgs, support_masks, support_split_masks = self.load_frame(query_class, query_example, support_examples)

        support_set = []
        # Apply the randomly selected augmentations on the support and query images
        if self.mode == 'train':
            # For debugging
            if self.support_train_aug is None:
                self.support_train_aug = "Random augmentations to be applied on the support images...."

            if self.query_train_aug is None:
                self.query_train_aug = "Random augmentation to be applied on the query image...."

            query_aug_index = query_example[0][-1]
            support_aug_index_list = []
            for support_example in support_examples:
                support_aug_index_list.append(support_example[-1])

            self.support_aug_str = support_aug_index_list
            self.query_aug_str = query_aug_index
            for s_i, s_cm, s_sm, s_aug in zip(support_imgs, support_masks, support_split_masks, support_aug_index_list):
                cur_trans = Compose([self.aug_dict[query_aug_index],
                                     Resize(self.patch_size),
                                     ToTensor()
                                     ])
                support_set.append(cur_trans(s_i, s_cm, s_sm))

            query_transform = Compose([self.aug_dict[query_aug_index],
                                       Resize(self.patch_size),
                                       ToTensor()
                                       ])

        # No augmentations on the validation data set
        else:
            # For debugging
            if self.support_val_aug is None:
                self.support_val_aug = "No augmentations to be applied on the support images...."

            if self.query_val_aug is None:
                self.query_val_aug = "No augmentations to be applied on the query image...."

            for s_i, s_cm, s_sm in zip(support_imgs, support_masks, support_split_masks):
                cur_trans = Compose([Resize(self.patch_size),
                                     ToTensor()
                                     ])
                support_set.append(cur_trans(s_i, s_cm, s_sm))

            query_transform = Compose([Resize(self.patch_size),
                                       ToTensor()
                                       ])
        support_imgs, support_masks, support_split_masks = torch.stack([i for i, _, _ in support_set]), \
                                                           torch.stack([m for _, m, _ in support_set]), \
                                                           torch.stack([s_m for _, _, s_m in support_set])

        query_img, query_mask, query_split_masks = query_transform(query_img[0], query_mask[0], query_split_masks[0])
        query_img = torch.unsqueeze(query_img, dim=0)
        query_mask = torch.unsqueeze(query_mask, dim=0)
        query_split_masks = torch.unsqueeze(query_split_masks, dim=0)

        if self.debug:

            # Augmentations based on training or validation stage
            if self.mode=='train':
                print(self.support_train_aug)
                print(self.support_aug_str)
                print(self.query_train_aug)
                print(self.query_aug_str)
            else:
                print(self.support_val_aug)
                print(self.support_aug_str)
                print(self.query_val_aug)
                print(self.query_aug_str)

            # Shape of the support minibatch
            print("Shape of a support example in a minibatch")
            print(support_imgs.shape)
            print(support_masks.shape)
            print(support_split_masks.shape)

            print("Shape of a query example in a minibatch")
            print(query_class)
            print(query_img.shape)
            print(query_mask.shape)
            print(query_split_masks.shape)

            self.debug = False

        output = {'data_class': query_class, 'query_image': query_img, 'query_segmentation': query_mask, 'query_split_masks': query_split_masks,
                  'support_images': support_imgs, 'support_segmentations': support_masks, 'support_split_masks': support_split_masks}

        return output


    def get_data_stat(self, data_axis, train_slices, data_split_filepath):
        data_split_fptr = open(data_split_filepath)
        data_split_dict = json.load(data_split_fptr)

        direction = 'inline' if data_axis==0 else 'xline'
        ortho_direction = 'xline' if data_axis==0 else 'inline'
        image_width_stat = None
        if self.mode=='train':
            # Get the slice indices
            if train_slices=='all':
                train_min_index_key_name = 'train_' + direction + '_min'
                train_max_index_key_name = 'train_' + direction + '_max'
                train_min_index = data_split_dict[train_min_index_key_name]
                train_max_index = data_split_dict[train_max_index_key_name]
                selected_indices = np.arange(train_min_index, train_max_index).tolist()
            else:
                train_key_name = 'train_' + direction + '_' + train_slices
                selected_indices = data_split_dict[train_key_name]

            # Get the valid width of the image
            train_min_col_index_key_name = 'train_' + ortho_direction + '_min'
            train_max_col_index_key_name = 'train_' + ortho_direction + '_max'
            train_min_col_index = data_split_dict[train_min_col_index_key_name]
            train_max_col_index = data_split_dict[train_max_col_index_key_name]
            image_width_stat = [train_min_col_index, train_max_col_index]

            return selected_indices, image_width_stat

        elif self.mode=='val':
            # Get the train slice indices
            # to be used as support images for evaluating the validation examples (query images)
            if train_slices=='all':
                train_min_index_key_name = 'train_' + direction + '_min'
                train_max_index_key_name = 'train_' + direction + '_max'
                train_min_index = data_split_dict[train_min_index_key_name]
                train_max_index = data_split_dict[train_max_index_key_name]
                train_selected_indices = np.arange(train_min_index, train_max_index).tolist()
            else:
                train_key_name = 'train_' + direction + '_' + train_slices
                train_selected_indices = data_split_dict[train_key_name]

            if direction=='inline':
                # Get the slice indices
                val_min_key_name = 'val_1_' + direction + '_min'
                val_max_key_name = 'val_1_' + direction + '_max'
                val_min_index = data_split_dict[val_min_key_name]
                val_max_index = data_split_dict[val_max_key_name]
                selected_indices = np.arange(val_min_index, val_max_index).tolist()

                # Get the valid width of the image
                val_min_col_index_key_name = 'val_1_' + ortho_direction + '_min'
                val_max_col_index_key_name = 'val_1_' + ortho_direction + '_max'
                val_min_col_index = data_split_dict[val_min_col_index_key_name]
                val_max_col_index = data_split_dict[val_max_col_index_key_name]
                image_width_stat = [val_min_col_index, val_max_col_index]

            elif direction=='xline':
                # Get the slice indices
                val_min_key_name = 'val_2_' + direction + '_min'
                val_max_key_name = 'val_2_' + direction + '_max'
                val_min_index = data_split_dict[val_min_key_name]
                val_max_index = data_split_dict[val_max_key_name]
                selected_indices = np.arange(val_min_index, val_max_index).tolist()

                # Get the valid width of the image
                val_min_col_index_key_name = 'val_2_' + ortho_direction + '_min'
                val_max_col_index_key_name = 'val_2_' + ortho_direction + '_max'
                val_min_col_index = data_split_dict[val_min_col_index_key_name]
                val_max_col_index = data_split_dict[val_max_col_index_key_name]
                image_width_stat = [val_min_col_index, val_max_col_index]

            return train_selected_indices, selected_indices, image_width_stat


    def build_img_metadata(self):

        img_metadata = []
        existing_entry = 0
        for class_name in self.classes:

            datapath = self.data_info[class_name]['data_dir']
            data_vol_name  = self.data_info[class_name]['data_vol_name']
            label_vol_name = self.data_info[class_name]['label_vol_name']
            patch_overlap = self.data_info[class_name]['patch_overlap']
            data_split_filepath = self.data_info[class_name]['train_val_test_split']
            train_slices = self.data_info[class_name]['train_indices']
            data_axis = self.data_info[class_name]['axis']

            ##########################################################################
            # Read the JSON file to get the following:
            # 1. slices indices for the train and validation data
            # 2. Start and end index along the column. Used to compute the image width
            ##########################################################################
            if self.mode == 'train':
                labeled_indices, image_col_indices = self.get_data_stat(data_axis, train_slices, data_split_filepath)
                self.slice_indices[class_name] = labeled_indices
            elif self.mode == 'val':
                train_labeled_indices, labeled_indices, image_col_indices = self.get_data_stat(data_axis, train_slices, data_split_filepath)
                self.eval_support_slice_indices[class_name] = train_labeled_indices
                self.slice_indices[class_name] = labeled_indices

            #####################################################################
            # Load the data and label along with the selection mask (if provided)
            #####################################################################
            self.data_vol[class_name] = np.load(os.path.join(datapath, data_vol_name))
            # Normalizing the input data
            min_intensity, max_intensity = np.min(self.data_vol[class_name]), np.max(self.data_vol[class_name])
            self.data_vol[class_name] = (((self.data_vol[class_name] - min_intensity) / (max_intensity - min_intensity)) * 255).astype(np.uint8)

            # Class index starts at 1
            # This ignores the class label *0* for the Penobscot data set
            if 'penobscot' in class_name:
                self.label_vol[class_name] = np.load(os.path.join(datapath, label_vol_name)).astype(np.uint8)
            else:
                self.label_vol[class_name] = np.load(os.path.join(datapath, label_vol_name)).astype(np.uint8) + 1

            # Ignore the class index *0* for the Penobscot data set
            cur_class_labels = np.unique(self.label_vol[class_name])
            self.class_labels[class_name] = (cur_class_labels[cur_class_labels > 0]).tolist()
            self.class_label_count[class_name] = len(self.class_labels[class_name])

            ##########################################
            # Valid row and column indices for a slice
            ##########################################
            # Row indices
            # This is related to the height of the data volume (z-axis)
            image_height = int(self.data_vol[class_name].shape[-1])
            min_row_index = self.patch_size//2

            # Applicable for the F3 dataset
            # Where the height (255) is less than the patch size (256)
            if image_height < self.patch_size:
                max_row_index = min_row_index + 1
                row_min_separation = 1 if patch_overlap != -1 else -1
            else:
                max_row_index = image_height - (self.patch_size//2)
                row_min_separation = max(4, int(self.patch_size*patch_overlap)) if patch_overlap != -1 else -1

            # Column indices
            # Width of the images are bigger than the patch size for all the three data sets
            image_width = image_col_indices[1] - image_col_indices[0]
            min_col_index = self.patch_size//2
            max_col_index = image_width - (self.patch_size//2)
            col_min_separation = max(4, int(self.patch_size*patch_overlap)) if patch_overlap != -1 else -1

            valid_row_indices = None
            valid_col_indices = None
            if row_min_separation != -1:
                valid_row_indices = np.arange(min_row_index, max_row_index, row_min_separation).tolist()
            if col_min_separation != 1:
                valid_col_indices = np.arange(min_col_index, max_col_index, col_min_separation).tolist()

            if self.mode=='train' and self.debug:
                print("Processing class : " + str(class_name))
                print("Image height                         : " + str(image_height))
                print("Image width                          : " + str(image_width))
                print("Patch size                           : " + str(self.patch_size))
                print("Separation between patches (row)     : " + str(row_min_separation))
                print("Separation between patches (col)     : " + str(col_min_separation))
                if row_min_separation!= -1 and col_min_separation!= -1:
                    print("Labeled indices              : " + str(labeled_indices))
                    print("Number of labeled indices    : " + str(len(labeled_indices)))
                    print("Shape of the data volume     : " + str(self.data_vol[class_name].shape))
                    print("Valid row indices            : " + str(valid_row_indices))
                    print("Number of valid row indices  : " + str(len(valid_row_indices)))
                    print("Valid col indices            : " + str(valid_col_indices))
                    print("Number of valid col indices  : " + str(len(valid_col_indices))+"\n")

            if self.mode in ['train', 'val']:
                for image_index in tqdm(labeled_indices):
                    if valid_row_indices is not None and valid_col_indices is not None:
                        for row_index in valid_row_indices:
                            for col_index in valid_col_indices:
                                img_metadata.append([class_name, image_index, row_index, col_index])
            else:
                raise Exception('Undefined mode %s: ' % self.mode)

            print('Total (%s) images for class %s are : %d' % (self.mode, class_name, len(img_metadata)-existing_entry))
            existing_entry = len(img_metadata)

        print('Total (%s) images are : %d' %(self.mode, len(img_metadata)))

        return img_metadata


    def build_img_metadata_classwise(self):
        class_name_to_index = {}
        img_metadata_classwise = {}
        class_index = 0
        for class_name in self.classes:
            img_metadata_classwise[class_name] = []
            class_name_to_index[class_name] = class_index
            class_index += 1

        per_class_sample_indices = [[] for _ in range(len(self.classes))]
        for i, (class_name, image_index, row_index, col_index) in enumerate(self.img_metadata):
            img_metadata_classwise[class_name].append([image_index, row_index, col_index])
            per_class_sample_indices[class_name_to_index[class_name]].append(i)

        print(f"Loaded (class_idx, num_samples): {[(c, len(lst)) for c, lst in img_metadata_classwise.items()]}")

        return img_metadata_classwise, per_class_sample_indices


    def sample_episode(self, idx):

        query_example = []
        support_examples = []

        # Sampled query image data
        # The use of the selection mask is taken care in the meta data building process
        patch_info_list = self.img_metadata[idx]
        # get the class label
        query_class = patch_info_list[0]
        # get the image index, row and column index
        patch_info_list = patch_info_list[1:]
        # get the image index
        query_image_index = patch_info_list[0]

        # Copy the query patch as support patches
        # The image index will be replaced with the new image indices
        # We fix the row and column index for the support images as the query image
        for _ in range(self.k_shot):
            support_examples.append(copy.deepcopy(patch_info_list))

        # Add augmentation to the query image
        if self.mode == 'train':
            query_aug_index = random.choice(range(1, self.total_aug_count + 1))
            patch_info_list.append(query_aug_index)
        query_example.append(patch_info_list)

        ####################################################################################
        # Sample support images data
        # For the validation data, training data is used as support images
        # For the training data we ignore the adjacent indices as support images to present a harder regression task
        #####################################################################################
        if self.mode=='train':
            min_slice_query_class = self.slice_indices[query_class][0]
            max_slice_query_class = self.slice_indices[query_class][-1]

            # left indices
            min_valid_left_index = max(query_image_index - self.train_cushion_sample_count - self.train_import_sample_count, min_slice_query_class)
            max_valid_left_index = max(query_image_index - self.train_cushion_sample_count, min_slice_query_class)
            valid_left_indices = np.arange(min_valid_left_index, max_valid_left_index + 1, self.train_skip_image_indices)

            # right indices
            min_valid_right_index = min(query_image_index + self.train_cushion_sample_count, max_slice_query_class)
            max_valid_right_index = min(query_image_index + self.train_cushion_sample_count + self.train_import_sample_count, max_slice_query_class)
            valid_right_indices = np.arange(min_valid_right_index, max_valid_right_index + 1, self.train_skip_image_indices)

            # combining the left and right indices
            valid_indices = np.append(valid_left_indices, valid_right_indices).tolist()

        if self.mode=='val':
            valid_indices = self.eval_support_slice_indices[query_class]

        ##############################################
        # Select support images from the valid indices
        ##############################################
        if len(valid_indices) < self.k_shot:
            support_sample_index = np.random.choice(valid_indices, self.k_shot, replace=True).tolist()
        else:
            support_sample_index = np.random.choice(valid_indices, self.k_shot, replace=False).tolist()
        for set_support_sample_index in range(self.k_shot):
            support_patch_info_list = support_examples[set_support_sample_index]
            support_patch_info_list[0] = support_sample_index[set_support_sample_index]
            if self.mode == 'train':
                support_aug_index = random.choice(range(1, self.total_aug_count + 1))
                support_patch_info_list.append(support_aug_index)
            support_examples[set_support_sample_index] = support_patch_info_list

        # Check if the support example belong to the query class or not
        if self.debug:
            print("Query class              : " + query_class)
            print("Query index              : " + str(query_image_index))
            print("Valid support indices    : " + str(valid_indices))
            print("Query sample             : " + str(query_example))
            print("Support samples          : " + str(support_examples))

        return query_class, query_example, support_examples


    def load_frame(self, query_class, query_example, support_examples):

        query_img, query_mask, query_split_masks = self.read_data(query_class, query_example)
        support_imgs, support_masks, support_split_masks = self.read_data(query_class, support_examples)

        # deallocate the memory
        gc.collect()

        return query_img, query_mask, query_split_masks, support_imgs, support_masks, support_split_masks


    def read_data(self, data_class, img_info):
        r"""Return segmentation mask in PIL Image"""

        data_axis = self.data_info[data_class]['axis']
        all_split_mask_list = []    # binary masks for the sematic labels
        mask_list = []              # labels, applicable for both binary and multi-class data
        image_list = []             # input images
        for index in range(len(img_info)):
            img_index = img_info[index][0]
            row_index = img_info[index][1]
            col_index = img_info[index][2]

            ###############################################################
            # Load input images based on the direction, inline or crossline
            ###############################################################
            try:
                if data_axis==0:
                    cur_img = copy.deepcopy(self.data_vol[data_class][img_index].T)
                elif data_axis==1:
                    cur_img = copy.deepcopy(self.data_vol[data_class][:, img_index, :].T)
            except Exception as e:
                print("The error is: ", e)
                print("Data class       : " + str(data_class))
                print("Data axis        : " + str(data_axis))
                print("The image index  : " + str(img_index))
                print(self.data_vol[data_class].shape)

            # F3 facies
            if self.data_vol[data_class].shape[-1] < self.patch_size:
                base_intensity = np.max(self.data_vol[data_class])/2
                img_patch = np.ones((self.patch_size, self.patch_size)) * base_intensity
                # Offset the start index by 1 to take care of close calls for volumes like facies
                data_start_index = (self.patch_size - self.data_vol[data_class].shape[-1])//2 + 1
                data_end_index = data_start_index + self.data_vol[data_class].shape[-1]
                try:
                    img_patch[data_start_index:data_end_index] = copy.deepcopy(cur_img[:, (col_index-self.patch_size//2):(col_index+self.patch_size//2)])
                except Exception as e:
                    print("The error is: ", e)
                    print("Data class : " + str(data_class))
                    print("The column range")
                    print(col_index-self.patch_size//2, col_index+self.patch_size//2)
                    print(cur_img.shape)

            # Penobscot and Parihaka
            else:
                try:
                    img_patch = copy.deepcopy(cur_img[(row_index-self.patch_size//2):(row_index+self.patch_size//2), (col_index-self.patch_size//2):(col_index+self.patch_size//2)])
                except Exception as e:
                    print("The error is: ", e)
                    print("Data class : " + str(data_class))
                    print("The column range")
                    print(col_index-self.patch_size//2, col_index+self.patch_size//2)
                    print("The row range")
                    print(row_index-self.patch_size//2, row_index+self.patch_size//2)
                    print(cur_img.shape)

            pil_image = Image.fromarray(img_patch)
            rgb_image = pil_image.convert("RGB")
            image_list.append(rgb_image)

            #############################################################################
            # Load corresponding label images based on the direction, inline or crossline
            #############################################################################
            try:
                if data_axis==0:
                    label_data = copy.deepcopy(self.label_vol[data_class][img_index].T)
                elif data_axis==1:
                    label_data = copy.deepcopy(self.label_vol[data_class][:, img_index, :].T)
            except Exception as e:
                print("The error is: ", e)
                print("Data class       : " + str(data_class))
                print("Data axis        : " + str(data_axis))
                print("The image index  : " + str(img_index))
                print(self.label_vol[data_class].shape)

            # F3 facies
            if self.data_vol[data_class].shape[-1] < self.patch_size:
                label_patch = np.zeros((self.patch_size, self.patch_size))
                # Offset the start index by 1 to take care of close calls for volumes like facies
                data_start_index = (self.patch_size - self.data_vol[data_class].shape[-1]) // 2 + 1
                data_end_index = data_start_index + self.data_vol[data_class].shape[-1]
                try:
                    label_patch[data_start_index:data_end_index] = copy.deepcopy(label_data[:, (col_index - self.patch_size // 2):(col_index + self.patch_size // 2)])
                except Exception as e:
                    print("The error is: ", e)
                    print("Data class : " + str(data_class))
                    print("The column range")
                    print(col_index-self.patch_size//2, col_index+self.patch_size//2)
                    print(label_data.shape)

            # Penobscot and Parihaka
            else:
                try:
                    label_patch = copy.deepcopy(label_data[(row_index-self.patch_size//2):(row_index+self.patch_size//2), (col_index-self.patch_size//2):(col_index+self.patch_size//2)])
                except Exception as e:
                    print("The error is: ", e)
                    print("Data class : " + str(data_class))
                    print("The column range")
                    print(col_index-self.patch_size//2, col_index+self.patch_size//2)
                    print("The row range")
                    print(row_index-self.patch_size//2, row_index+self.patch_size//2)
                    print(label_data.shape)

            pil_mask = Image.fromarray(label_patch)
            rgb_mask = pil_mask.convert("L")
            mask_list.append(rgb_mask)

            # delete the image and label data
            del cur_img, label_data


            ##################################################
            # Generate the binary masks for the sematic labels
            ##################################################
            split_mask_list = []
            if 'facies' in data_class:
                label_patch_one_hot = torch.nn.functional.one_hot(torch.from_numpy(label_patch).to(torch.int64), num_classes=self.class_label_count[data_class]+1).numpy()
                for mask_index in self.class_labels[data_class]:
                    pil_split_mask = Image.fromarray((label_patch_one_hot[:, :, mask_index] * 255).astype(np.uint8))
                    rgb_split_mask = pil_split_mask.convert("L")
                    split_mask_list.append(rgb_split_mask)
            all_split_mask_list.append(split_mask_list)

        return image_list, mask_list, all_split_mask_list


'''
https://stackoverflow.com/questions/74252067/efficiently-sample-batches-from-only-one-class-at-each-iteration-with-pytorch
Baseline: https://stackoverflow.com/questions/66065272/customizing-the-batch-with-specific-elements

Related stuff:
https://discuss.pytorch.org/t/load-the-same-number-of-data-per-class/65198
'''
class BatchSampler:
    def __init__(self, per_class_sample_indices, batch_size):
        # classes is a list of lists where each sublist refers to a class and contains
        # the sample ids that belond to this class
        self.per_class_sample_indices = per_class_sample_indices
        self.n_batches = sum([len(x) for x in per_class_sample_indices]) // batch_size
        self.min_class_size = min([len(x) for x in per_class_sample_indices])
        self.batch_size = batch_size
        self.class_range = list(range(len(self.per_class_sample_indices)))
        random.shuffle(self.class_range)

    def __iter__(self):
        for j in range(self.n_batches):
            if j < len(self.class_range):
                batch_class = self.class_range[j]
            else:
                batch_class = random.choice(self.class_range)
            if self.batch_size <= len(self.per_class_sample_indices[batch_class]):
                batch = np.random.choice(self.per_class_sample_indices[batch_class], self.batch_size, replace=False)
            else:
                batch = self.per_class_sample_indices[batch_class]
            yield batch


def revert_normalization(sample):
    """
    sample (Tensor): of size (nsamples,nchannels,height,width)
    """
    mean = [0.5]
    std = [0.5]
    mean_tensor = torch.Tensor(mean)
    std_tensor = torch.Tensor(std)
    non_normalized_sample = sample*std_tensor + mean_tensor
    return non_normalized_sample


def show_combined_images(input_images, anno_images, num_classes, save_img_path):

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
        anno_img = (anno_img * (255//num_classes)).astype(np.uint8)
        anno_img = np.clip(anno_img, 0, 255)

        ax = fig.add_subplot(gs[0, img_index])
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth(1)
        # ax.spines['bottom'].set_color('0.0')
        # ax.spines['top'].set_color('0.0')
        # ax.spines['right'].set_color('0.0')
        # ax.spines['left'].set_color('0.0')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.axis('off')
        plt.imshow(img, vmin=0, vmax=255)

        ax = fig.add_subplot(gs[1, img_index])
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth(1)
        # ax.spines['bottom'].set_color('0.0')
        # ax.spines['top'].set_color('0.0')
        # ax.spines['right'].set_color('0.0')
        # ax.spines['left'].set_color('0.0')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.axis('off')
        plt.imshow(anno_img, vmin=0, vmax=255)

    plt.savefig(save_img_path)
    plt.close(plt.gcf())

    return


if __name__ == '__main__':

    config = create_config()
    classes = config['classes']
    data_info = config['data_info']
    set_debug = True

    seismic_seg_data_loader = Seismic_Segmentation_Task(classes, data_info, patch_size=256, k_shot=5, train_batch_size=5, val_batch_size=5, debug=set_debug)
    train_loader = seismic_seg_data_loader.train_loader
    val_loader = seismic_seg_data_loader.val_loader

    print("Number of training examples     : " + str(len(seismic_seg_data_loader.traindataset)))
    print("Number of validation examples   : " + str(len(seismic_seg_data_loader.validdataset)) + "\n")

    print("Testing the training data loader....")
    for i, data in enumerate(train_loader):
        if i == 0:
            _iter_path = os.path.join('Train', 'Iter_' + str(i + 1))
            if not os.path.isdir(_iter_path):
                os.makedirs(_iter_path, exist_ok=True)

            if set_debug:
                print("\nShape of the training minibatch....")
                print(data['query_image'].cpu().detach().shape)
                print(data['query_segmentation'].cpu().detach().shape)
                print(data['support_images'].cpu().detach().shape)
                print(data['support_segmentations'].cpu().detach().shape)
                input()

            support_images = data['support_images'].cpu().detach()
            support_images_np = support_images.numpy().transpose(0, 1, 3, 4, 2)

            support_anno = data['support_segmentations'].cpu().detach()
            B, N, C, H, W = support_anno.size()
            support_anno_np = support_anno.numpy().transpose(0, 1, 3, 4, 2)
            support_split_anno = data['support_split_masks'].cpu().detach()
            support_split_anno_np = support_split_anno.numpy().transpose(0, 1, 2, 4, 5, 3)

            data_class = data['data_class'][0]
            num_labels = seismic_seg_data_loader.traindataset.class_label_count[data_class]
            for b in range(B):
                support_batch_path = os.path.join(_iter_path, 'Support_Batch_' + str(b + 1) + '.png')
                show_combined_images(support_images_np[b], support_anno_np[b], num_labels, support_batch_path)

                for class_index in range(support_split_anno_np.shape[2]):
                    _split_mask_path = os.path.join(_iter_path, 'Mask_' + str(class_index + 1))
                    if not os.path.isdir(_split_mask_path):
                        os.makedirs(_split_mask_path, exist_ok=True)

                    support_split_mask_path = os.path.join(_split_mask_path, 'Support_Batch_' + str(b + 1) + '.png')
                    show_combined_images(support_images_np[b], support_split_anno_np[b, :, class_index], 2, support_split_mask_path)

            query_images = data['query_image'].cpu().detach()
            query_images_np = query_images.numpy().transpose(0, 1, 3, 4, 2)
            query_anno = data['query_segmentation'].cpu().detach()
            B, N, C, H, W = query_anno.size()
            query_anno_np = query_anno.numpy().transpose(0, 1, 3, 4, 2)
            query_split_anno = data['query_split_masks'].cpu().detach()
            query_split_anno_np = query_split_anno.numpy().transpose(0, 1, 2, 4, 5, 3)
            for b in range(B):
                query_batch_path = os.path.join(_iter_path, 'Query_Batch_' + str(b + 1) + '.png')
                show_combined_images(query_images_np[b], query_anno_np[b], num_labels, query_batch_path)

                for class_index in range(query_split_anno_np.shape[2]):
                    _split_mask_path = os.path.join(_iter_path, 'Mask_' + str(class_index + 1))
                    if not os.path.isdir(_split_mask_path):
                        os.makedirs(_split_mask_path, exist_ok=True)

                    query_split_mask_path = os.path.join(_split_mask_path, 'Query_Batch_' + str(b + 1) + '.png')
                    show_combined_images(query_images_np[b], query_split_anno_np[b, :, class_index], 2, query_split_mask_path)
        else:
            print("Done with training data loader...." + "\n\n")
            break

    print("Testing the validation data loader....")
    for i, data in enumerate(val_loader):
        if i == 0:
            _iter_path = os.path.join('Val', 'Iter_' + str(i + 1))
            if not os.path.isdir(_iter_path):
                os.makedirs(_iter_path, exist_ok=True)

            if set_debug:
                print("\nShape of the validation minibatch....")
                print(data['query_image'].cpu().detach().shape)
                print(data['query_segmentation'].cpu().detach().shape)
                print(data['support_images'].cpu().detach().shape)
                print(data['support_segmentations'].cpu().detach().shape)

            support_images = data['support_images'].cpu().detach()
            support_images_np = support_images.numpy().transpose(0, 1, 3, 4, 2)

            support_anno = data['support_segmentations'].cpu().detach()
            B, N, C, H, W = support_anno.size()
            support_anno_np = support_anno.numpy().transpose(0, 1, 3, 4, 2)
            support_split_anno = data['support_split_masks'].cpu().detach()
            support_split_anno_np = support_split_anno.numpy().transpose(0, 1, 2, 4, 5, 3)

            data_class = data['data_class'][0]
            num_labels = seismic_seg_data_loader.traindataset.class_label_count[data_class]
            for b in range(B):
                support_batch_path = os.path.join(_iter_path, 'Support_Batch_' + str(b + 1) + '.png')
                show_combined_images(support_images_np[b], support_anno_np[b], num_labels, support_batch_path)

                for class_index in range(support_split_anno_np.shape[2]):
                    _split_mask_path = os.path.join(_iter_path, 'Mask_' + str(class_index + 1))
                    if not os.path.isdir(_split_mask_path):
                        os.makedirs(_split_mask_path, exist_ok=True)

                    support_split_mask_path = os.path.join(_split_mask_path, 'Support_Batch_' + str(b + 1) + '.png')
                    show_combined_images(support_images_np[b], support_split_anno_np[b, :, class_index], 2, support_split_mask_path)

            query_images = data['query_image'].cpu().detach()
            query_images_np = query_images.numpy().transpose(0, 1, 3, 4, 2)
            query_anno = data['query_segmentation'].cpu().detach()
            B, N, C, H, W = query_anno.size()
            query_anno_np = query_anno.numpy().transpose(0, 1, 3, 4, 2)
            query_split_anno = data['query_split_masks'].cpu().detach()
            query_split_anno_np = query_split_anno.numpy().transpose(0, 1, 2, 4, 5, 3)
            for b in range(B):
                query_batch_path = os.path.join(_iter_path, 'Query_Batch_' + str(b + 1) + '.png')
                show_combined_images(query_images_np[b], query_anno_np[b], num_labels, query_batch_path)

                for class_index in range(query_split_anno_np.shape[2]):
                    _split_mask_path = os.path.join(_iter_path, 'Mask_' + str(class_index + 1))
                    if not os.path.isdir(_split_mask_path):
                        os.makedirs(_split_mask_path, exist_ok=True)

                    query_split_mask_path = os.path.join(_split_mask_path, 'Query_Batch_' + str(b + 1) + '.png')
                    show_combined_images(query_images_np[b], query_split_anno_np[b, :, class_index], 2, query_split_mask_path)

        else:
            print("Done with test data loader....")
            break