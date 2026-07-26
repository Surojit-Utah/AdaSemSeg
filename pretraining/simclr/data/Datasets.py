import sys
sys.path.append('..')
from PIL import Image
from data.transform import Compose, Normalize, Resize, ToTensor, RandomHorizontalFlip, Contrast, Brightness
from torchvision import transforms
from data.transform import RandomRotate, GaussianBlur, GaussNoise, Scale
from torch.utils.data import DataLoader, Dataset, RandomSampler, WeightedRandomSampler
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import random
import copy
import json
from config.local_config import create_config
from collections import defaultdict


class SimCLR_dataset():

    def __init__(self, classes, data_info, patch_size, train_batch_size=10, val_batch_size=10, debug=False):

        self.traindataset = CustomDataset(classes, data_info, patch_size, train_batch_size, mode='train', debug=debug)
        training_sampler = WeightedRandomSampler(self.traindataset.sample_weights, len(self.traindataset.sample_weights))
        self.train_loader = DataLoader(self.traindataset, sampler=training_sampler, batch_size=1, num_workers=1)

        self.validdataset = CustomDataset(classes, data_info, patch_size, val_batch_size, mode='val', debug=debug)
        validation_sampler = WeightedRandomSampler(self.validdataset.sample_weights, len(self.validdataset.sample_weights))
        self.val_loader = DataLoader(self.validdataset, sampler=validation_sampler, batch_size=1, num_workers=1)


class CustomDataset(Dataset):

    def __init__(self, classes, data_info, patch_size, batch_size, mode=None, debug=False):

        self.classes = classes
        self.data_info = data_info
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.mode = mode
        self.transform = None
        self.debug = debug

        self.data_vol = dict()
        self.img_metadata, self.class_metadata = self.build_img_metadata()

        keep_entries = (len(self.img_metadata)//self.batch_size)*self.batch_size
        self.img_metadata = self.img_metadata[:keep_entries]

        self.img_metadata_classwise, self.per_class_sample_indices = self.build_img_metadata_classwise()

        #################################################################
        # Get the class weights for all the samples in the image metadata
        # This is required fo sampling unrepresented class (F3 facies)
        # Will be used by the *WeightedRandomSampler*
        #################################################################
        self.class_based_sample_count = defaultdict(int)
        total_sample_count = 0
        for class_name in self.classes:
            image_sample_indices = self.img_metadata_classwise[class_name]
            self.class_based_sample_count[class_name] = len(image_sample_indices)
            total_sample_count += len(image_sample_indices)

        self.class_weights = dict()
        for class_name in self.classes:
            class_distribution = self.class_based_sample_count[class_name]
            self.class_weights[class_name] = np.round(total_sample_count / (class_distribution * len(self.classes)), 3)
        print(f"Class weights in the {self.mode} mode")
        print(self.class_weights)

        self.sample_weights = np.zeros(len(self.img_metadata))
        for i, (class_name, _, _, _) in enumerate(self.img_metadata):
            self.sample_weights[i] = self.class_weights[class_name]
        self.sample_weights = torch.from_numpy(self.sample_weights)
        self.sample_weights = self.sample_weights.double()

        ####################################################################
        # Min and Max range for the each class in *self.img_metadata*
        # This is required for finding negative examples in the anchor class
        ####################################################################
        self.per_class_sample_min_max_indices = dict()
        class_index = 0
        for class_name in self.classes:
            class_indices = self.per_class_sample_indices[class_index]
            self.per_class_sample_min_max_indices[class_name] = [min(class_indices), max(class_indices)]
            class_index += 1

        self.aug_dict = {1: RandomRotate(p=0.8),
                         2: RandomHorizontalFlip(p=0.8),
                         3: GaussianBlur(p=0.8),
                         4: GaussNoise(p=0.8),
                         5: transforms.RandomResizedCrop(size=self.patch_size),
                         6: Brightness(p=0.8),
                         7: Contrast(p=0.8)}

        self.total_aug_count = len(self.aug_dict.keys())
        self.num_views = 2

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

        minibatch_sample_indices = self.sample_episode(idx)

        class_names = []
        for batch_index in range(self.batch_size):
            img_metadata_id = minibatch_sample_indices[batch_index]
            sample_class_name = self.img_metadata[img_metadata_id][0]
            class_names.extend([sample_class_name])

        # Load the query and support images as PIL images
        minibatch_data = self.read_data(minibatch_sample_indices)

        # get the augmented examples with multiview property
        # two views
        multiview_aug_imgs = [[], []]
        aug_list = []
        for img_index, img in enumerate(minibatch_data):
            trans = []
            for view_cnt in range(self.num_views):
                aug_index = random.choice(range(1, self.total_aug_count + 1))
                offset = len(minibatch_data)*view_cnt
                aug_list_index = img_index + offset
                aug_list.insert(aug_list_index, aug_index)
                cur_trans = Compose([self.aug_dict[aug_index],
                                     Resize(self.patch_size),
                                     ToTensor()
                                     ])
                trans.append(cur_trans)
            multiview_aug_imgs[0].append(trans[0](img))
            multiview_aug_imgs[1].append(trans[1](img))

        # Stack the samples as a 4D tensor
        multiview_minibatch = [torch.stack([i for i in multiview_aug_imgs[view_index]]) for view_index in range(self.num_views)]

        if self.debug:
            print("Shape of a minibatch")
            print(multiview_minibatch[0].shape)
            print(multiview_minibatch[1].shape)
            self.debug = False

        output = {'data': multiview_minibatch, 'image_aug': aug_list, 'class_names': class_names}

        return output


    def get_data_stat(self, data_axis, data_split_filepath):
        data_split_fptr = open(data_split_filepath)
        data_split_dict = json.load(data_split_fptr)

        direction = 'inline' if data_axis==0 else 'xline'
        ortho_direction = 'xline' if data_axis==0 else 'inline'
        image_width_stat = None
        if self.mode=='train':
            train_min_index_key_name = 'train_' + direction + '_min'
            train_max_index_key_name = 'train_' + direction + '_max'
            train_min_index = data_split_dict[train_min_index_key_name]
            train_max_index = data_split_dict[train_max_index_key_name]
            selected_indices = np.arange(train_min_index, train_max_index).tolist()

            # Get the valid width of the image
            train_min_col_index_key_name = 'train_' + ortho_direction + '_min'
            train_max_col_index_key_name = 'train_' + ortho_direction + '_max'
            train_min_col_index = data_split_dict[train_min_col_index_key_name]
            train_max_col_index = data_split_dict[train_max_col_index_key_name]
            image_width_stat = [train_min_col_index, train_max_col_index]

        elif self.mode=='val':
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

        return selected_indices, image_width_stat


    def build_img_metadata(self):

        img_metadata = []
        class_metadata = dict()
        existing_entry = 0
        for class_name in self.classes:

            datapath = self.data_info[class_name]['data_dir']
            data_vol_name  = self.data_info[class_name]['data_vol_name']
            patch_overlap = self.data_info[class_name]['patch_overlap']
            data_split_filepath = self.data_info[class_name]['train_val_test_split']
            data_axis = self.data_info[class_name]['axis']

            ##########################################################################
            # Read the JSON file to get the following:
            # 1. slices indices for the train and validation data
            # 2. Start and end index along the column. Used to compute the image width
            ##########################################################################
            if self.mode in ['train', 'val']:
                labeled_indices, image_col_indices = self.get_data_stat(data_axis, data_split_filepath)

            ############################
            # Load the data
            # Normalizing the input data
            ############################
            self.data_vol[class_name] = np.load(os.path.join(datapath, data_vol_name))
            min_intensity, max_intensity = np.min(self.data_vol[class_name]), np.max(self.data_vol[class_name])
            self.data_vol[class_name] = (((self.data_vol[class_name] - min_intensity) / (max_intensity - min_intensity)) * 255).astype(np.uint8)

            ##########################################
            # Valid row and column indices for a slice
            # Row represents the depth of the volume (z-axis)
            # Column represents the spread for a slice
            ##########################################
            # Row indices
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

            if self.mode in ['train', 'val'] and self.debug:
                print("Processing class : " + str(class_name))
                print("Image height                         : " + str(image_height))
                print("Image width                          : " + str(image_width))
                print("Patch size                           : " + str(self.patch_size))
                print("Separation between patches (row)     : " + str(row_min_separation))
                print("Separation between patches (col)     : " + str(col_min_separation))
                if row_min_separation!= -1 and col_min_separation!= -1:
                    # print("Labeled indices              : " + str(labeled_indices))
                    print("Number of labeled indices    : " + str(len(labeled_indices)))
                    print("Shape of the data volume     : " + str(self.data_vol[class_name].shape))
                    print("Valid row indices            : " + str(valid_row_indices))
                    print("Number of valid row indices  : " + str(len(valid_row_indices)))
                    print("Valid col indices            : " + str(valid_col_indices))
                    print("Number of valid col indices  : " + str(len(valid_col_indices))+"\n")

            if self.mode in ['train', 'val']:
                for row_index in valid_row_indices:
                    for col_index in valid_col_indices:
                        for image_index in labeled_indices:
                                img_metadata.append([class_name, image_index, row_index, col_index])
            else:
                raise Exception('Undefined mode %s: ' % self.mode)

            print('Total (%s) images for class %s are : %d' % (self.mode, class_name, len(img_metadata)-existing_entry))
            existing_entry = len(img_metadata)

            ##########################################################
            # Get the metadata of the class
            # To be used in sampling the positive and negative samples
            ##########################################################
            if self.mode in ['train', 'val']:
                class_metadata[class_name] = [valid_row_indices, valid_col_indices, labeled_indices]

        print('Total (%s) images are : %d' %(self.mode, len(img_metadata)))

        return img_metadata, class_metadata


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

        #############################################################
        # Construct a minibatch using the anchor and negative samples
        #############################################################
        # samples per class including the anchor class
        samples_per_class = self.batch_size // len(self.classes)
        minibatch_sample_indices = [idx]
        count_samples = 1
        for class_index, class_name in enumerate(self.classes):

            # for the last class add the sample remaining number of examples in the minibatch
            if class_index == len(self.classes)-1:
                samples_per_class = self.batch_size - count_samples

            index_range = self.per_class_sample_min_max_indices[class_name]
            total_class_indices = np.arange(index_range[0], index_range[1]).tolist()
            if len(total_class_indices) < samples_per_class:
                negative_sample_indices = np.random.choice(total_class_indices, samples_per_class,
                                                           replace=True).tolist()
            else:
                negative_sample_indices = np.random.choice(total_class_indices, samples_per_class,
                                                           replace=False).tolist()

            # negative examples obtained so far
            count_samples += samples_per_class
            # add the negative samples to the list
            minibatch_sample_indices.extend(negative_sample_indices)

        if self.debug:
            #######################################
            # Classwise distribution in a minibatch
            #######################################
            classwise_samples = defaultdict(int)
            for batch_index in range(self.batch_size):
                img_metadata_id = minibatch_sample_indices[batch_index]
                class_name = self.img_metadata[img_metadata_id][0]
                classwise_samples[class_name] += 1
            print('Distribution in a minibatch....')
            print(f"Loaded (class_idx, num_samples): {[(class_name, sample_count) for class_name, sample_count in classwise_samples.items()]}")

        return minibatch_sample_indices


    def read_data(self, minibatch_sample_indices):
        r"""Return segmentation mask in PIL Image"""

        # input images
        image_list = []
        for index in minibatch_sample_indices:
            class_name = self.img_metadata[index][0]
            img_index = self.img_metadata[index][1]
            row_index = self.img_metadata[index][2]
            col_index = self.img_metadata[index][3]

            ###############################################################
            # Load input images based on the direction, inline or crossline
            ###############################################################
            data_axis = self.data_info[class_name]['axis']
            if data_axis==0:
                cur_img = copy.deepcopy(self.data_vol[class_name][img_index].T)
            elif data_axis==1:
                cur_img = copy.deepcopy(self.data_vol[class_name][:, img_index, :].T)

            # F3 facies
            if self.data_vol[class_name].shape[-1] < self.patch_size:
                base_intensity = np.max(self.data_vol[class_name])/2
                img_patch = np.ones((self.patch_size, self.patch_size)) * base_intensity
                # Offset the start index by 1 to take care of close calls for volumes like facies
                data_start_index = (self.patch_size - self.data_vol[class_name].shape[-1])//2 + 1
                data_end_index = data_start_index + self.data_vol[class_name].shape[-1]
                img_patch[data_start_index:data_end_index] = copy.deepcopy(cur_img[:, (col_index-self.patch_size//2):(col_index+self.patch_size//2)])
            # Penobscot and Parihaka
            else:
                img_patch = copy.deepcopy(cur_img[(row_index-self.patch_size//2):(row_index+self.patch_size//2), (col_index-self.patch_size//2):(col_index+self.patch_size//2)])
            pil_image = Image.fromarray(img_patch)
            rgb_image = pil_image.convert("RGB")
            image_list.append(rgb_image)

        return image_list


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


def show_combined_images(minibatch_data, data_aug, class_names, save_img_path):

    aug_dict = {1: 'RandomRotate',
                2: 'RandomHorizontalFlip',
                3: 'GaussianBlur',
                4: 'GaussNoise',
                5: 'RandomResizedCrop',
                6: 'Brightness',
                7: 'Contrast'}

    num_images = minibatch_data.shape[0]//2
    fig = plt.figure(figsize=(2*50, num_images*50))
    fig_width, fig_height = fig.get_size_inches()
    gs = fig.add_gridspec(2, num_images)

    total_image_height = 2*minibatch_data.shape[2]
    total_image_width = num_images*minibatch_data.shape[1]
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

    for minibatch_index in range(num_images):
        img_1 = minibatch_data[minibatch_index, :, :, :]
        ax1 = fig.add_subplot(gs[0, minibatch_index])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        aug_index = data_aug[minibatch_index].cpu().detach().numpy()[0]
        ax1.set_title(class_names[minibatch_index][0], fontweight="bold", size=30)
        ax1.set_xlabel(aug_dict[aug_index], fontweight="bold", size=30)
        plt.imshow(img_1, vmin=0, vmax=255)
        img_2 = minibatch_data[minibatch_index+num_images, :, :, :]
        ax2 = fig.add_subplot(gs[1, minibatch_index])
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_aspect('equal')
        aug_index = data_aug[minibatch_index+num_images].cpu().detach().numpy()[0]
        ax2.set_xlabel(aug_dict[aug_index], fontweight="bold", size=30)
        plt.imshow(img_2, vmin=0, vmax=255)

    plt.savefig(save_img_path)
    plt.close(plt.gcf())

    return


if __name__ == '__main__':

    config = create_config()
    classes = config['classes']
    data_info = config['data_info']
    set_debug = True
    train_batch_size = val_batch_size = 10

    seismic_seg_data_loader = SimCLR_dataset(classes, data_info, patch_size=256, train_batch_size=train_batch_size, val_batch_size=val_batch_size, debug=set_debug)
    train_loader = seismic_seg_data_loader.train_loader
    print("Number of training examples     : " + str(len(seismic_seg_data_loader.traindataset)))

    print("Testing the training data loader....")
    for i, train_minibatch in enumerate(train_loader):
        data = train_minibatch['data']
        data_aug = train_minibatch['image_aug']
        class_names = train_minibatch['class_names']

        if i < 10:
            if set_debug:
                print("\nShape of the training minibatch....")
                for view_index in range(seismic_seg_data_loader.traindataset.num_views):
                    print(torch.squeeze(data[view_index], dim=0).shape)
                input()

            _iter_path = os.path.join('Train', 'Iter_' + str(i + 1))
            if not os.path.isdir(_iter_path):
                os.makedirs(_iter_path, exist_ok=True)

            data = torch.cat((torch.squeeze(data[0], dim=0), torch.squeeze(data[1], dim=0)), dim=0)
            print(data.shape)
            multiview_input_images = data.cpu().detach()
            multiview_input_images_np = multiview_input_images.numpy().transpose(0, 2, 3, 1)

            # visualize the train data
            support_batch_path = os.path.join(_iter_path, 'Train_aug_data.png')
            show_combined_images(multiview_input_images_np, data_aug, class_names, support_batch_path)

        else:
            print("Done with training data loader...." + "\n\n")
            break

    val_loader = seismic_seg_data_loader.val_loader
    print("Number of validation examples   : " + str(len(seismic_seg_data_loader.validdataset)) + "\n")
    print("Testing the validation data loader....")
    for i, val_minibatch in enumerate(val_loader):
        data = val_minibatch['data']
        data_aug = val_minibatch['image_aug']
        class_names = val_minibatch['class_names']
        if i < 10:
            if set_debug:
                print("\nShape of the validation minibatch....")
                for view_index in range(seismic_seg_data_loader.validdataset.num_views):
                    print(torch.squeeze(data[view_index], dim=0).shape)
                input()

            _iter_path = os.path.join('Val', 'Iter_' + str(i + 1))
            if not os.path.isdir(_iter_path):
                os.makedirs(_iter_path, exist_ok=True)

            data = torch.cat((torch.squeeze(data[0], dim=0), torch.squeeze(data[1], dim=0)), dim=0)
            print(data.shape)
            multiview_input_images = data.cpu().detach()
            multiview_input_images_np = multiview_input_images.numpy().transpose(0, 2, 3, 1)

            # visualize the validation data
            support_batch_path = os.path.join(_iter_path, 'Val_aug_data.png')
            show_combined_images(multiview_input_images_np, data_aug, class_names, support_batch_path)

        else:
            print("Done with test data loader....")
            break