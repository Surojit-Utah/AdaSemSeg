import sys
sys.path.append('..')
from PIL import Image
from data.transform import Compose, Normalize, Resize, ToTensor
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import copy
import json
from config.local_config import create_config


class Dataset_Loader():

    def __init__(self, class_name, data_info, eval_mode='val', batch_size=None, k_shot=None, debug=False):

        self.class_labels = dict()
        self.class_label_count = dict()

        print("Class name : " + str(class_name))
        self.class_name = class_name
        self.data_info = data_info
        self.eval_mode = eval_mode

        # train data details
        train_datapath = self.data_info[self.class_name]['train_data_dir']
        train_data_vol_name = self.data_info[self.class_name]['train_data_vol_name']
        train_label_data_vol_name = self.data_info[self.class_name]['train_label_vol_name']
        data_split_filepath = self.data_info[class_name]['train_val_test_split']
        num_train_slices = self.data_info[class_name]['train_indices']
        self.data_axis = self.data_info[class_name]['axis']
        self.data_axis_complement = int(not bool(int(self.data_axis)))

        # test data details
        test_datapath = self.data_info[self.class_name]['test_data_dir']
        test_data_vol_name = self.data_info[self.class_name]['test_data_vol_name']
        test_label_data_vol_name = self.data_info[self.class_name]['test_label_vol_name']

        # train data volume
        self.train_data_vol = np.load(os.path.join(train_datapath, train_data_vol_name))
        self.image_height, self.image_width = self.train_data_vol.shape[1], self.train_data_vol.shape[2]
        # Normalizing the input data
        min_intensity, max_intensity = np.min(self.train_data_vol), np.max(self.train_data_vol)
        self.train_data_vol = (((self.train_data_vol - min_intensity) / (max_intensity - min_intensity)) * 255).astype(np.uint8)

        # test data volume
        self.test_data_vol = np.load(os.path.join(test_datapath, test_data_vol_name))
        # Normalizing the input data
        min_intensity, max_intensity = np.min(self.test_data_vol), np.max(self.test_data_vol)
        self.test_data_vol = (((self.test_data_vol - min_intensity) / (max_intensity - min_intensity)) * 255).astype(np.uint8)

        # Class index starts at 1
        # This ignores the class label *0* for the Penobscot data set
        if 'penobscot' in class_name:
            self.train_label_vol = np.load(os.path.join(train_datapath, train_label_data_vol_name)).astype(np.uint8)
            self.test_label_vol = np.load(os.path.join(test_datapath, test_label_data_vol_name)).astype(np.uint8)
        else:
            self.train_label_vol = np.load(os.path.join(train_datapath, train_label_data_vol_name)).astype(np.uint8) + 1
            self.test_label_vol = np.load(os.path.join(test_datapath, test_label_data_vol_name)).astype(np.uint8) + 1

        if 'f3' in class_name:
            self.support_slice_indices, self.support_slice_width, self.query_slice_indices, self.query_slice_width = self.get_data_stat_f3(self.data_axis, num_train_slices, data_split_filepath)
        else:
            self.support_slice_indices, self.support_slice_width, self.query_slice_indices, self.query_slice_width = self.get_data_stat(self.data_axis, num_train_slices, data_split_filepath)
        self.vis_selected_indices = self.query_slice_indices
        self.start_query_slice_index = 0
        self.start_vis_slice_index = 0

        self.batch_size = batch_size
        self.k_shot = k_shot

        # Ignore the class index *0* for the Penobscot data set
        class_labels = np.unique(self.train_label_vol)
        self.class_labels[class_name] = (class_labels[class_labels > 0]).tolist()
        self.class_label_count[class_name] = len(self.class_labels[class_name])

        self.debug = debug


    def get_data_stat(self, data_axis, train_slices, data_split_filepath):
        data_split_fptr = open(data_split_filepath)
        data_split_dict = json.load(data_split_fptr)

        direction = 'inline' if data_axis==0 else 'xline'
        ortho_direction = 'xline' if data_axis==0 else 'inline'
        if train_slices == 'all':
            train_min_index_key_name = 'train_' + direction + '_min'
            train_max_index_key_name = 'train_' + direction + '_max'
            train_min_index = data_split_dict[train_min_index_key_name]
            train_max_index = data_split_dict[train_max_index_key_name]
            train_selected_indices = np.arange(train_min_index, train_max_index)
        else:
            train_key_name = 'train_' + direction + '_' + train_slices
            train_selected_indices = data_split_dict[train_key_name]

        # Get the valid width of the image
        train_min_col_index_key_name = 'train_' + ortho_direction + '_min'
        train_max_col_index_key_name = 'train_' + ortho_direction + '_max'
        train_min_col_index = data_split_dict[train_min_col_index_key_name]
        train_max_col_index = data_split_dict[train_max_col_index_key_name]
        train_image_width_stat = [train_min_col_index, train_max_col_index]

        # get val or test slice indices
        if self.eval_mode=='val' or self.eval_mode=='test':
            if direction == 'inline':
                # Get the slice indices
                eval_min_key_name = self.eval_mode + '_' + str(self.data_axis + 1) + '_' + direction + '_min'
                eval_max_key_name = self.eval_mode + '_' + str(self.data_axis + 1) + '_' + direction + '_max'
                eval_min_index = data_split_dict[eval_min_key_name]
                eval_max_index = data_split_dict[eval_max_key_name]
                eval_selected_indices = np.arange(eval_min_index, eval_max_index)

                # Get the valid width of the image
                eval_image_width_stat = [train_min_col_index, train_max_col_index]

            elif direction == 'xline':
                # Get the slice indices
                eval_min_key_name = self.eval_mode + '_' + str(self.data_axis + 1) + '_' + direction + '_min'
                eval_max_key_name = self.eval_mode + '_' + str(self.data_axis + 1) + '_' + direction + '_max'
                eval_min_index = data_split_dict[eval_min_key_name]
                eval_max_index = data_split_dict[eval_max_key_name]
                eval_selected_indices = np.arange(eval_min_index, eval_max_index)

                # Get the valid width of the image
                eval_image_width_stat = [train_min_col_index, train_max_col_index]

        return train_selected_indices, train_image_width_stat, eval_selected_indices, eval_image_width_stat


    def get_data_stat_f3(self, data_axis, train_slices, data_split_filepath):

        data_split_fptr = open(data_split_filepath)
        data_split_dict = json.load(data_split_fptr)

        direction = 'inline' if data_axis==0 else 'xline'
        ortho_direction = 'xline' if data_axis==0 else 'inline'
        if train_slices == 'all':
            train_min_index_key_name = 'train_' + direction + '_min'
            train_max_index_key_name = 'train_' + direction + '_max'
            train_min_index = data_split_dict[train_min_index_key_name]
            train_max_index = data_split_dict[train_max_index_key_name]
            train_selected_indices = np.arange(train_min_index, train_max_index)
        else:
            train_key_name = 'train_' + direction + '_' + train_slices
            train_selected_indices = data_split_dict[train_key_name]

        # Get the valid width of the image
        train_min_col_index_key_name = 'train_' + ortho_direction + '_min'
        train_max_col_index_key_name = 'train_' + ortho_direction + '_max'
        train_min_col_index = data_split_dict[train_min_col_index_key_name]
        train_max_col_index = data_split_dict[train_max_col_index_key_name]
        train_image_width_stat = [train_min_col_index, train_max_col_index]

        # get val or test slice indices
        if self.eval_mode=='test':
            if direction == 'inline':
                # Get the slice indices
                eval_min_index = 0
                eval_max_index = self.test_data_vol.shape[0]
                eval_selected_indices = np.arange(eval_min_index, eval_max_index)

                # Get the valid width of the image
                eval_image_width_stat = [train_min_col_index, train_max_col_index]

            elif direction == 'xline':
                # Get the slice indices
                # Get the slice indices
                eval_min_index = 0
                eval_max_index = self.test_data_vol.shape[1]
                eval_selected_indices = np.arange(eval_min_index, eval_max_index)

                # Get the valid width of the image
                # 200 is added such that the support and query images are of same size
                test_min_col_index = train_min_col_index+200
                test_max_col_index = self.test_data_vol.shape[0]
                eval_image_width_stat = [test_min_col_index, test_max_col_index]

        return train_selected_indices, train_image_width_stat, eval_selected_indices, eval_image_width_stat


    def get_minibatch(self):

        support_set_minibatch = []
        query_set_minibatch = []

        for index in range(self.batch_size):
            query_selected_index = [self.query_slice_indices[self.start_query_slice_index+index]]
            query_class_labels = self.get_class_labels(query_selected_index[0], train_volume=False)
            print("Query image index : " + str(query_selected_index[0]))

            ############################
            # Sample support images data
            ############################
            support_selected_indices = []
            while True:  # keep sampling support set if query == support
                support_sample_index = np.random.choice(self.support_slice_indices, 1)[0]
                support_class_labels = self.get_class_labels(support_sample_index, train_volume=True)

                # check the presence of class labels in the query image
                support_query_intersection = list(set(query_class_labels).intersection(set(support_class_labels)))
                if (len(support_query_intersection) == len(query_class_labels)) and (support_sample_index not in support_selected_indices):
                    support_selected_indices.append(support_sample_index)
                    if len(support_selected_indices) == self.k_shot:
                        break

            # Load the query and support images as PIL images
            query_img, query_label, query_split_mask, support_imgs, support_labels, support_split_masks = self.load_frame(support_selected_indices, query_selected_index)

            # Support image
            support_set = []
            for s_i, s_cm, s_sm in zip(support_imgs, support_labels, support_split_masks):
                cur_trans = Compose([ToTensor()])
                support_set.append(cur_trans(s_i, s_cm, s_sm))

            support_imgs, support_labels, support_split_masks = torch.stack([i for i, _, _ in support_set]), \
                                                          torch.stack([m for _, m, _ in support_set]), \
                                                          torch.stack([l for _, _, l in support_set])

            support_set_minibatch.append((support_imgs, support_labels, support_split_masks))

            # Query image
            query_transform = Compose([ToTensor()])
            query_img, query_label, query_split_masks = query_transform(query_img[0], query_label[0], query_split_mask[0])
            query_img = torch.unsqueeze(query_img, dim=0)
            query_label = torch.unsqueeze(query_label, dim=0)
            query_split_masks = torch.unsqueeze(query_split_masks, dim=0)
            query_set_minibatch.append((query_img, query_label, query_split_masks))

        self.start_query_slice_index += self.batch_size

        support_imgs_minibatch, support_masks_minibatch, support_split_masks_minibatch = torch.stack([i for i, _, _ in support_set_minibatch]), \
                                                                                    torch.stack([m for _, m, _ in support_set_minibatch]), \
                                                                                    torch.stack([l for _, _, l in support_set_minibatch])

        query_imgs_minibatch, query_masks_minibatch, query_split_masks_minibatch = torch.stack([i for i, _, _ in query_set_minibatch]), \
                                                                              torch.stack([m for _, m, _ in query_set_minibatch]), \
                                                                              torch.stack([l for _, _, l in query_set_minibatch])

        output = {'query_image': query_imgs_minibatch, 'query_segmentation': query_masks_minibatch, 'query_split_masks': query_split_masks_minibatch,
                  'support_images': support_imgs_minibatch, 'support_segmentations': support_masks_minibatch, 'support_split_masks': support_split_masks_minibatch}

        if self.debug:

            # Shape of the support minibatch
            print("Shape of support sets....")
            print(support_imgs_minibatch.shape)
            print(support_masks_minibatch.shape)
            print(support_split_masks_minibatch.shape)

            # Shape of the query minibatch
            print("Shape of query set....")
            print(query_imgs_minibatch.shape)
            print(query_masks_minibatch.shape)
            print(query_split_masks_minibatch.shape)

            self.debug = False

        return output


    def get_vis_minibatch(self):

        support_set_minibatch = []
        query_set_minibatch = []

        for index in range(self.batch_size):

            support_selected_indices = np.random.choice(self.support_slice_indices, self.k_shot)
            query_selected_index = [self.vis_selected_indices[self.start_vis_slice_index+index]]
            print("Query image index : " + str(query_selected_index[0]))

            # Load the query and support images as PIL images
            query_img, query_label, query_split_mask, support_imgs, support_labels, support_split_masks = self.load_frame(support_selected_indices, query_selected_index)

            # Support image
            support_set = []
            for s_i, s_cm, s_sm in zip(support_imgs, support_labels, support_split_masks):
                cur_trans = Compose([ToTensor()])
                support_set.append(cur_trans(s_i, s_cm, s_sm))

            support_imgs, support_labels, support_split_masks = torch.stack([i for i, _, _ in support_set]), \
                                                          torch.stack([m for _, m, _ in support_set]), \
                                                          torch.stack([l for _, _, l in support_set])

            support_set_minibatch.append((support_imgs, support_labels, support_split_masks))

            # Query image
            query_transform = Compose([ToTensor()])
            query_img, query_label, query_split_masks = query_transform(query_img[0], query_label[0], query_split_mask[0])
            query_img = torch.unsqueeze(query_img, dim=0)
            query_label = torch.unsqueeze(query_label, dim=0)
            query_split_masks = torch.unsqueeze(query_split_masks, dim=0)
            query_set_minibatch.append((query_img, query_label, query_split_masks))

        self.start_vis_slice_index += self.batch_size

        support_imgs_minibatch, support_masks_minibatch, support_split_masks_minibatch = torch.stack([i for i, _, _ in support_set_minibatch]), \
                                                                                    torch.stack([m for _, m, _ in support_set_minibatch]), \
                                                                                    torch.stack([l for _, _, l in support_set_minibatch])

        query_imgs_minibatch, query_masks_minibatch, query_split_masks_minibatch = torch.stack([i for i, _, _ in query_set_minibatch]), \
                                                                              torch.stack([m for _, m, _ in query_set_minibatch]), \
                                                                              torch.stack([l for _, _, l in query_set_minibatch])

        output = {'query_image': query_imgs_minibatch, 'query_segmentation': query_masks_minibatch, 'query_split_masks': query_split_masks_minibatch,
                  'support_images': support_imgs_minibatch, 'support_segmentations': support_masks_minibatch, 'support_split_masks': support_split_masks_minibatch}

        return output


    def load_frame(self, support_selected_indices, query_selected_index):

        support_imgs, support_labels, support_split_mask_list, query_img, query_label, query_split_mask_list = self.read_data(query_selected_index, support_selected_indices)

        return query_img, query_label, query_split_mask_list, support_imgs, support_labels, support_split_mask_list


    def read_data(self, query_selected_index, support_selected_indices):
        r"""Return segmentation mask in PIL Image"""

        support_image_list = []
        support_label_list = []
        support_split_mask_list = []
        start_image_width_index = self.support_slice_width[0]
        end_image_width_index = self.support_slice_width[1]
        for index in range(len(support_selected_indices)):
            img_index = support_selected_indices[index]

            # Get the input image and its label
            if self.data_axis==0:
                ori_cur_img = copy.deepcopy(self.train_data_vol[img_index, start_image_width_index:end_image_width_index, :].T)
                ori_label_data = copy.deepcopy(self.train_label_vol[img_index, start_image_width_index:end_image_width_index, :].T)
            else:
                ori_cur_img = copy.deepcopy(self.train_data_vol[start_image_width_index:end_image_width_index, img_index, :].T)
                ori_label_data = copy.deepcopy(self.train_label_vol[start_image_width_index:end_image_width_index, img_index, :].T)

            # Adjust the image height for F3 facies dataset
            if 'f3_facies' in self.class_name:
                # Adjust image height for F3 facies, such that we do not loose mich information from the input image
                base_intensity = np.max(ori_cur_img)/2
                cur_img = np.ones((256, ori_cur_img.shape[-1]))*base_intensity
                cur_img[1:, :] = ori_cur_img
                # label
                label_data = np.zeros((256, ori_label_data.shape[1]))
                label_data[1:, :] = ori_label_data
            else:
                # image and its label
                cur_img = ori_cur_img
                label_data = ori_label_data

            # Adjust the image height and width for model prediction
            image_height = int((cur_img.shape[0]//32)*32)
            image_width = int((cur_img.shape[1]//32)*32)
            crop_start_row = int((cur_img.shape[0] - image_height)/2)
            crop_start_col = int((cur_img.shape[1] - image_width)/2)

            cur_img = cur_img[crop_start_row:(crop_start_row+image_height), crop_start_col:(crop_start_col+image_width)]
            pil_image = Image.fromarray(cur_img)
            rgb_image = pil_image.convert("RGB")
            support_image_list.append(rgb_image)

            label_data = label_data[crop_start_row:(crop_start_row+image_height), crop_start_col:(crop_start_col+image_width)]
            pil_mask = Image.fromarray(label_data)
            rgb_mask = pil_mask.convert("L")
            support_label_list.append(rgb_mask)

            # Generate the binary masks for the sematic labels
            split_mask_list = []
            label_one_hot = torch.nn.functional.one_hot(torch.from_numpy(label_data).to(torch.int64),
                                                              num_classes=self.class_label_count[self.class_name]+1).numpy()
            for mask_index in self.class_labels[self.class_name]:
                pil_split_mask = Image.fromarray((label_one_hot[:, :, mask_index] * 255).astype(np.uint8))
                rgb_split_mask = pil_split_mask.convert("L")
                split_mask_list.append(rgb_split_mask)
            support_split_mask_list.append(split_mask_list)


        query_image_list = []
        query_label_list = []
        query_split_mask_list = []
        start_image_width_index = self.support_slice_width[0]
        end_image_width_index = self.support_slice_width[1]
        for index in range(len(query_selected_index)):
            img_index = query_selected_index[index]

            # Get the input image and its label
            if self.data_axis==0:
                ori_cur_img = copy.deepcopy(self.test_data_vol[img_index, start_image_width_index:end_image_width_index, :].T)
                ori_label_data = copy.deepcopy(self.test_label_vol[img_index, start_image_width_index:end_image_width_index, :].T)
            else:
                ori_cur_img = copy.deepcopy(self.test_data_vol[start_image_width_index:end_image_width_index, img_index, :].T)
                ori_label_data = copy.deepcopy(self.test_label_vol[start_image_width_index:end_image_width_index, img_index, :].T)


            # Adjust the image height for F3 facies dataset
            if 'f3_facies' in self.class_name:
                base_intensity = np.max(ori_cur_img)/2
                cur_img = np.ones((256, ori_cur_img.shape[1]))*base_intensity
                cur_img[1:, :] = ori_cur_img
                # label
                label_data = np.zeros((256, ori_label_data.shape[1]))
                label_data[1:, :] = ori_label_data
            else:
                # image and its label
                cur_img = ori_cur_img
                label_data = ori_label_data

            # Adjust the image height and width for model prediction
            image_height = int((cur_img.shape[0]//32)*32)
            image_width = int((cur_img.shape[1]//32)*32)
            crop_start_row = int((cur_img.shape[0] - image_height)/2)
            crop_start_col = int((cur_img.shape[1] - image_width)/2)

            cur_img = cur_img[crop_start_row:(crop_start_row+image_height), crop_start_col:(crop_start_col+image_width)]
            pil_image = Image.fromarray(cur_img)
            rgb_image = pil_image.convert("RGB")
            query_image_list.append(rgb_image)

            label_data = label_data[crop_start_row:(crop_start_row+image_height), crop_start_col:(crop_start_col+image_width)]
            pil_mask = Image.fromarray(label_data)
            rgb_mask = pil_mask.convert("L")
            query_label_list.append(rgb_mask)

            # Generate the binary masks for the sematic labels
            split_mask_list = []
            label_one_hot = torch.nn.functional.one_hot(torch.from_numpy(label_data).to(torch.int64),
                                                              num_classes=self.class_label_count[self.class_name]+1).numpy()

            for mask_index in self.class_labels[self.class_name]:
                pil_split_mask = Image.fromarray((label_one_hot[:, :, mask_index] * 255).astype(np.uint8))
                rgb_split_mask = pil_split_mask.convert("L")
                split_mask_list.append(rgb_split_mask)
            query_split_mask_list.append(split_mask_list)

        return support_image_list, support_label_list, support_split_mask_list, query_image_list, query_label_list, query_split_mask_list


    def get_class_labels(self, img_index, train_volume=False):
        r"""Return segmentation mask in PIL Image"""

        start_image_width_index = self.support_slice_width[0]
        end_image_width_index = self.support_slice_width[1]
        if train_volume:
            if self.data_axis == 0:
                label_data = copy.deepcopy(self.train_label_vol[img_index, start_image_width_index:end_image_width_index, :].T)
            else:
                label_data = copy.deepcopy(self.train_label_vol[start_image_width_index:end_image_width_index, img_index, :].T)
        else:
            # Get the input image and its label
            if self.data_axis == 0:
                label_data = copy.deepcopy(self.test_label_vol[img_index, start_image_width_index:end_image_width_index, :].T)
            else:
                label_data = copy.deepcopy(self.test_label_vol[start_image_width_index:end_image_width_index, img_index, :].T)

        # image to be processed by the model
        image_height = int((label_data.shape[0] // 32) * 32)
        image_width = int((label_data.shape[1] // 32) * 32)
        crop_start_row = int((label_data.shape[0] - image_height)/2)
        crop_start_col = int((label_data.shape[1] - image_width)/2)
        label_data = label_data[crop_start_row:(crop_start_row + image_height), crop_start_col:(crop_start_col + image_width)]

        class_labels = np.unique(label_data)
        class_labels = (class_labels[class_labels > 0]).tolist()

        del label_data

        return class_labels


def show_combined_images(input_images, anno_images, save_img_path):

    num_images = input_images.shape[0]
    fig = plt.figure(figsize=(num_images*50, 2*50))
    fig_width, fig_height = fig.get_size_inches()
    gs = fig.add_gridspec(num_images, 2)

    total_image_height = num_images*input_images.shape[1]
    total_image_width = 2*input_images.shape[2]
    image_shape_array = np.array([total_image_width, total_image_height])
    shape_index = np.argmax(image_shape_array)
    # If width is bigger, map it to the image width size
    if shape_index==0:
        # print("Width is bigger....")
        set_fig_width = fig_width
        set_fig_height = (fig_width/total_image_width)*total_image_height
        fig.set_figwidth(set_fig_width)
        fig.set_figheight(set_fig_height)
    else:
        # print("Height is bigger....")
        set_fig_height = fig_height
        set_fig_width = (fig_height/total_image_height)*total_image_width
        fig.set_figwidth(set_fig_width)
        fig.set_figheight(set_fig_height)

    for img_index in range(num_images):

        img = input_images[img_index, :, :, :]
        anno_img = anno_images[img_index, :, :, :]

        # Clipping the Range [0, 255]
        img = (img * 255.0).astype(np.uint8)
        anno_img = (anno_img * (255//6)).astype(np.uint8)
        anno_img = np.clip(anno_img, 0, 255)

        ax = fig.add_subplot(gs[img_index, 0])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.axis('off')
        plt.imshow(img, vmin=0, vmax=255)

        ax = fig.add_subplot(gs[img_index, 1])
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
    class_name = config['classes'][1]
    data_mode = 'test' #'val'
    data_info = config['data_info']

    test_batch_size = 1
    data_visualize = True
    test_loader = Dataset_Loader(class_name, data_info, eval_mode=data_mode, batch_size=1, k_shot=5, debug=True)
    test_minibatch = test_loader.get_minibatch()

    print("Testing the training data loader....")
    print(test_minibatch['query_image'].cpu().detach().shape)
    print(test_minibatch['query_segmentation'].cpu().detach().shape)
    print(test_minibatch['query_split_masks'].cpu().detach().shape)
    print(test_minibatch['support_images'].cpu().detach().shape)
    print(test_minibatch['support_segmentations'].cpu().detach().shape)
    print(test_minibatch['support_split_masks'].cpu().detach().shape)

    if data_visualize:
        print("Visualizing the data for a minibatch....")
        _iter_path = os.path.join(class_name, data_mode)
        if not os.path.isdir(_iter_path):
            os.makedirs(_iter_path, exist_ok=True)

        support_images = test_minibatch['support_images'].cpu().detach()
        support_images_np = support_images.numpy().transpose(0, 1, 3, 4, 2)

        support_anno = test_minibatch['support_segmentations'].cpu().detach()
        B, N, C, H, W = support_anno.size()
        support_anno_np = support_anno.numpy().transpose(0, 1, 3, 4, 2)

        support_split_anno = test_minibatch['support_split_masks'].cpu().detach()
        support_split_anno_np = support_split_anno.numpy().transpose(0, 1, 2, 4, 5, 3)

        for b in range(B):
            support_batch_path = os.path.join(_iter_path, 'Support_Batch_' + str(b + 1) + '.png')
            show_combined_images(support_images_np[b], support_anno_np[b]+1, support_batch_path)

            for class_index in range(support_split_anno_np.shape[2]):
                _split_mask_path = os.path.join(_iter_path, 'Mask_' + str(class_index + 1))
                if not os.path.isdir(_split_mask_path):
                    os.makedirs(_split_mask_path, exist_ok=True)

                support_split_mask_path = os.path.join(_split_mask_path, 'Support_Batch_' + str(b + 1) + '.png')
                show_combined_images(support_images_np[b], support_split_anno_np[b, :, class_index]*(class_index+1), support_split_mask_path)

        query_images = test_minibatch['query_image'].cpu().detach()
        query_images_np = query_images.numpy().transpose(0, 1, 3, 4, 2)
        query_anno = test_minibatch['query_segmentation'].cpu().detach()
        B, N, C, H, W = query_anno.size()
        query_anno_np = query_anno.numpy().transpose(0, 1, 3, 4, 2)
        query_split_anno = test_minibatch['query_split_masks'].cpu().detach()
        query_split_anno_np = query_split_anno.numpy().transpose(0, 1, 2, 4, 5, 3)
        for b in range(B):
            query_batch_path = os.path.join(_iter_path, 'Query_Batch_' + str(b + 1) + '.png')
            show_combined_images(query_images_np[b], query_anno_np[b]+1, query_batch_path)

            for class_index in range(query_split_anno_np.shape[2]):
                _split_mask_path = os.path.join(_iter_path, 'Mask_' + str(class_index + 1))
                if not os.path.isdir(_split_mask_path):
                    os.makedirs(_split_mask_path, exist_ok=True)

                query_split_mask_path = os.path.join(_split_mask_path, 'Query_Batch_' + str(b + 1) + '.png')
                show_combined_images(query_images_np[b], query_split_anno_np[b, :, class_index]*(class_index+1), query_split_mask_path)

    print("Done with the testing of the training data loader....")