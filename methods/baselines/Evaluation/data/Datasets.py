import sys
sys.path.append('..')
from PIL import Image
from data.transform import Compose, ToTensor
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

    def __init__(self, class_name, data_info, eval_mode='val', batch_size=None, debug=False):

        self.class_labels = dict()
        self.class_label_count = dict()

        print("Class name : " + str(class_name))
        self.class_name = class_name
        self.data_info = data_info
        self.eval_mode = eval_mode

        train_datapath = self.data_info[self.class_name]['train_data_dir']
        train_label_data_vol_name = self.data_info[self.class_name]['train_label_vol_name']
        data_split_filepath = self.data_info[class_name]['train_val_test_split']
        self.data_axis = self.data_info[class_name]['axis']
        self.data_axis_complement = int(not bool(int(self.data_axis)))

        # test data details
        test_datapath = self.data_info[self.class_name]['test_data_dir']
        test_data_vol_name = self.data_info[self.class_name]['test_data_vol_name']
        test_label_data_vol_name = self.data_info[self.class_name]['test_label_vol_name']

        # test data volume
        self.test_data_vol = np.load(os.path.join(test_datapath, test_data_vol_name))
        # Normalizing the input data
        min_intensity, max_intensity = np.min(self.test_data_vol), np.max(self.test_data_vol)
        self.test_data_vol = (((self.test_data_vol - min_intensity) / (max_intensity - min_intensity)) * 255).astype(np.uint8)

        # Class index starts at 1
        # This ignores the class label *0* for the Penobscot data set
        if 'penobscot' in class_name:
            # Train labels are used to compute the number of labels in the dataset
            self.train_label_vol = np.load(os.path.join(train_datapath, train_label_data_vol_name)).astype(np.uint8)
            self.test_label_vol = np.load(os.path.join(test_datapath, test_label_data_vol_name)).astype(np.uint8)
        else:
            # Train labels are used to compute the number of labels in the dataset
            self.train_label_vol = np.load(os.path.join(train_datapath, train_label_data_vol_name)).astype(np.uint8) + 1
            self.test_label_vol = np.load(os.path.join(test_datapath, test_label_data_vol_name)).astype(np.uint8) + 1

        if 'f3' in class_name:
            self.query_slice_indices, self.query_slice_width = self.get_data_stat_f3(data_split_filepath)
        else:
            self.query_slice_indices, self.query_slice_width = self.get_data_stat(data_split_filepath)
        self.vis_selected_indices = self.query_slice_indices
        self.start_query_slice_index = 0
        self.start_vis_slice_index = 0

        self.batch_size = batch_size

        # Ignore the class index *0* for the Penobscot data set
        class_labels = np.unique(self.train_label_vol)
        self.class_labels[class_name] = (class_labels[class_labels > 0]).tolist()
        self.class_label_count[class_name] = len(self.class_labels[class_name])

        self.debug = debug


    def get_data_stat(self, data_split_filepath):
        data_split_fptr = open(data_split_filepath)
        data_split_dict = json.load(data_split_fptr)

        direction = 'inline' if self.data_axis==0 else 'xline'
        ortho_direction = 'xline' if self.data_axis==0 else 'inline'
        # Get the valid width of the image
        train_min_col_index_key_name = 'train_' + ortho_direction + '_min'
        train_max_col_index_key_name = 'train_' + ortho_direction + '_max'
        train_min_col_index = data_split_dict[train_min_col_index_key_name]
        train_max_col_index = data_split_dict[train_max_col_index_key_name]

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

        return eval_selected_indices, eval_image_width_stat


    def get_data_stat_f3(self, data_split_filepath):

        data_split_fptr = open(data_split_filepath)
        data_split_dict = json.load(data_split_fptr)

        direction = 'inline' if self.data_axis==0 else 'xline'
        ortho_direction = 'xline' if self.data_axis==0 else 'inline'
        # Get the valid width of the image
        train_min_col_index_key_name = 'train_' + ortho_direction + '_min'
        train_max_col_index_key_name = 'train_' + ortho_direction + '_max'
        train_min_col_index = data_split_dict[train_min_col_index_key_name]
        train_max_col_index = data_split_dict[train_max_col_index_key_name]

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

                # This section is ignored for the baseline
                # As the width of the support (train) images are used even in the evaluation
                # Get the valid width of the image
                # 200 is added such that the support and query images are of same size
                # test_min_col_index = train_min_col_index+200
                # test_max_col_index = self.test_data_vol.shape[0]
                # eval_image_width_stat = [test_min_col_index, test_max_col_index]

                eval_image_width_stat = [train_min_col_index, train_max_col_index]

        return eval_selected_indices, eval_image_width_stat


    def get_minibatch(self):

        query_set_minibatch = []

        for index in range(self.batch_size):
            query_selected_index = [self.query_slice_indices[self.start_query_slice_index+index]]
            print("Query image index : " + str(query_selected_index[0]))

            # Load the query and support images as PIL images
            query_img, query_label = self.load_frame(query_selected_index)

            # Query image
            query_transform = Compose([ToTensor()])
            query_img, query_label = query_transform(query_img[0], query_label[0])
            query_set_minibatch.append((query_img, query_label))

        self.start_query_slice_index += self.batch_size

        query_imgs_minibatch, query_masks_minibatch = torch.stack([i for i, _ in query_set_minibatch]), \
                                                    torch.stack([m for _, m in query_set_minibatch])

        output = {'query_image': query_imgs_minibatch, 'query_segmentation': query_masks_minibatch}

        if self.debug:

            # Shape of the query minibatch
            print("Shape of query set....")
            print(query_imgs_minibatch.shape)
            print(query_masks_minibatch.shape)

            self.debug = False

        return output


    def load_frame(self, query_selected_index):

        query_img, query_label = self.read_data(query_selected_index)

        return query_img, query_label


    def read_data(self, query_selected_index):
        r"""Return segmentation mask in PIL Image"""

        query_image_list = []
        query_label_list = []
        start_image_width_index = self.query_slice_width[0]
        end_image_width_index = self.query_slice_width[1]
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

        return query_image_list, query_label_list


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

        img = input_images[img_index]
        anno_img = anno_images[img_index]

        # Clipping the Range [0, 255]
        img = (img * 255.0).astype(np.uint8)
        anno_img = (anno_img * (255//7)).astype(np.uint8)
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
    test_loader = Dataset_Loader(class_name, data_info, eval_mode=data_mode, batch_size=1, debug=True)
    test_minibatch = test_loader.get_minibatch()

    print("Testing the training data loader....")
    print(test_minibatch['query_image'].cpu().detach().shape)
    print(test_minibatch['query_segmentation'].cpu().detach().shape)

    if data_visualize:
        print("Visualizing the data for a minibatch....")
        _iter_path = os.path.join(class_name, data_mode)
        if not os.path.isdir(_iter_path):
            os.makedirs(_iter_path, exist_ok=True)

        query_images = test_minibatch['query_image'].cpu().detach()
        query_images_np = query_images.numpy().transpose(0, 2, 3, 1)
        query_anno = test_minibatch['query_segmentation'].cpu().detach()
        B, C, H, W = query_anno.size()
        query_anno_np = query_anno.numpy().transpose(0, 2, 3, 1)
        query_anno_np_scaled = query_anno_np*(255 // 6)
        query_batch_path = os.path.join(_iter_path, 'Query_Batch.png')
        show_combined_images(query_images_np, query_anno_np, query_batch_path)

    print("Done with the testing of the training data loader....")