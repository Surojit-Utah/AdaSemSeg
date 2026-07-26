import glob
import os
import psutil
import gc
from itertools import islice
from collections import OrderedDict
from datetime import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import numpy as np
import torch
import torch.nn as nn
import time
import sys
sys.path.append(('..', '..'))
from utils.recursive_functions import recursive_detach, recursive_to
from utils.general_util import show_combined_model_pred_images, show_combined_images


class Deep_GP_Trainer:

    def __init__(self, tensorboard_writer, model, loss, optimizer, dataset_n_loader,
                 checkpoint_path, data_visualization_path, device,
                 checkpoint_epochs, print_interval, visualization_epochs, save_name, lr_sched=None, seg_threshold=0.5):

        self._model = model
        self._loss = loss

        self._optimizer = optimizer
        self._lr_sched = lr_sched

        self.num_train_batches = dataset_n_loader.train_batch_sampler_obj.n_batches
        self.num_val_batches = dataset_n_loader.val_batch_sampler_obj.n_batches

        self._train_loader = dataset_n_loader.train_loader
        self._val_loader = dataset_n_loader.val_loader

        self._checkpoint_path = checkpoint_path
        self._data_visualization_path = data_visualization_path
        self._save_name = save_name

        self._train_mode_path = ''
        self._val_mode_path = ''
        self._train_epoch_path = ''
        self._val_epoch_path = ''

        self._device = device

        self._checkpoint_epochs = checkpoint_epochs
        self._print_interval = print_interval
        self._visualization_epochs = visualization_epochs

        self._epoch = 0
        self._seg_threshold = seg_threshold
        self._summary_writer = tensorboard_writer

        self.best_val_loss = 1e10
        self.best_val_epoch = 0

        self._query_image = None
        self._query_segmentation = None

        self._num_classes = dataset_n_loader.traindataset.class_label_count
        self.class_indices = dataset_n_loader.traindataset.class_labels
        self.class_weights = dict()

        #########
        # CE loss
        #########
        # Class weights used in the CE loss
        self.class_weights['parihaka'] = [1.11, 0.395, 4.032, 0.579, 16.748, 1.878]
        self.class_weights['penobscot'] = [1.314, 1.055, 5.368, 4.278, 0.457, 0.838, 0.402]
        self.class_weights['f3'] = [0.593, 1.402, 0.343, 2.509, 5.083, 11.05]

        # target class ignored in the CE loss for the Penobscot dataset representing regions above the horizon
        self.ignore_index = -1

        # Processing of the predictions for visualization
        self.predict_act = torch.nn.Softmax2d()


    def _plot_data(self, mode, class_name, data, output_segs, num_classes, i):

        if mode=='train':
            query_pred_path = os.path.join(self._train_epoch_path, class_name, 'Model_pred_' + str(i + 1))
        if 'val' in mode:
            query_pred_path = os.path.join(self._val_epoch_path, class_name, 'Model_pred_' + str(i + 1))

        if not os.path.isdir(query_pred_path):
            os.makedirs(query_pred_path, exist_ok=True)

        query_images_np = data['query_image'].cpu().detach().numpy().transpose(0, 2, 3, 1)
        query_anno_vis_np = ((data['query_segmentation']).cpu().detach().numpy().transpose(0, 2, 3, 1) + 1) * (255//num_classes)

        output_segs = torch.unsqueeze(output_segs, dim=1)
        query_pred_vis_np = (output_segs.cpu().detach().numpy().transpose(0, 2, 3, 1) + 1) * (255//num_classes)

        query_pred_batch_path = os.path.join(query_pred_path, 'prediction.png')
        show_combined_model_pred_images(query_images_np, query_anno_vis_np, query_pred_vis_np, query_pred_batch_path)


    def _get_iou(self, segs, segannos):
        """Note that this is an IoU-measure used only during training. It is not the same as the
        IoU reported during evaluation and more reminiscent of the IoU-measure used in the Video
        Object Segmentation problem.
        Args:
            segs (LongTensor(B, H, W))
            segannos (LongTensor(B, H, W))
        Returns:
            iou (Tensor) averaged over the batch size
        """

        segs = segs.clone()
        mask_pred = (segs == 1)
        mask_anno = (segannos == 1)
        intersection = (mask_pred * mask_anno).sum(dim=(1, 2))
        union = mask_pred.sum(dim=(1, 2)) + mask_anno.sum(dim=(1, 2)) - intersection
        iou = torch.div((intersection.to(torch.float32)), (union.to(torch.float32) + 1e-5)).mean()

        del segs, mask_pred, mask_anno, intersection, union

        return iou


    def _get_iou_summary(self, class_indices, segs, segannos):

        segs = segs.clone()

        iou_array = np.zeros(len(class_indices))
        # class indices starts at 0
        for class_index in class_indices:
            mask_pred = (segs == class_index)
            mask_anno = (segannos == class_index)
            intersection = (mask_pred * mask_anno).sum(dim=(1, 2))
            union = mask_pred.sum(dim=(1, 2)) + mask_anno.sum(dim=(1, 2)) - intersection
            iou = torch.div((intersection.to(torch.float32)), (union.to(torch.float32) + 1e-5)).mean()

            # adjusted the class index
            iou_array[class_index] = iou.detach().cpu().numpy()

        return iou_array


    def train(self, max_epochs):

        # Save the training input
        self._train_mode_path = os.path.join(self._data_visualization_path, 'Mode_train')
        if not os.path.isdir(self._train_mode_path):
            os.makedirs(self._train_mode_path, exist_ok=True)

        # Save the validation input
        self._val_mode_path = os.path.join(self._data_visualization_path, 'Mode_val')
        if not os.path.isdir(self._val_mode_path):
            os.makedirs(self._val_mode_path, exist_ok=True)

        print(f"Training epochs {self._epoch + 1} to {max_epochs}. Moving model to {self._device}.")
        self._model.to(self._device)
        for epoch in range(self._epoch + 1, max_epochs + 1):

            if epoch in self._visualization_epochs:
                self._train_epoch_path = os.path.join(self._train_mode_path, 'Epoch_' + str(epoch))
                if not os.path.isdir(self._train_epoch_path):
                    os.makedirs(self._train_epoch_path, exist_ok=True)

                self._val_epoch_path = os.path.join(self._val_mode_path, 'Epoch_' + str(epoch))
                if not os.path.isdir(self._val_epoch_path):
                    os.makedirs(self._val_epoch_path, exist_ok=True)

            # Training for an epoch
            self._epoch = epoch
            print(f"Starting epoch {epoch}")
            self._train_epoch()

            if self._epoch in self._checkpoint_epochs:
                print("Saving Checkpoint....")
                self.save_checkpoint()

        print(f"Finished training!")
        return self.best_val_epoch, self.best_val_loss.cpu().detach().numpy()

    # Training the model parameters for an epoch
    def _train_epoch(self):
        """Do one epoch of training and validation."""

        # Meta-learning stage
        # Samples from the training set are used for training the model parameters
        self._model.train(True)
        self._run_epoch(mode='train', data_loader=self._train_loader)

        # Applied on the validation class
        self._model.train(False)
        with torch.no_grad():
            self._run_epoch(mode='val', data_loader=self._val_loader)


    def _run_epoch(self, mode, data_loader):

        avg_WCE_loss = 0
        time_between_logs = 0
        start_time = time.time()
        print_classes = []
        for iter_num, data in enumerate(data_loader):

            self._query_class = class_name = data['data_class'][0]

            # Added by SSaha on Dec 11, 2021
            if mode == 'train':

                if iter_num == 0 and self._epoch in self._visualization_epochs:

                    _iter_path = os.path.join(self._train_epoch_path, 'Iter_' + str(iter_num + 1))
                    if not os.path.isdir(_iter_path):
                        os.makedirs(_iter_path, exist_ok=True)

                    # Save the query images in the training set
                    query_images = data['query_image'].cpu().detach()
                    query_images_np = query_images.numpy().transpose(0, 2, 3, 1)
                    query_anno_np = (data['query_segmentation'].cpu().detach().numpy().transpose(0, 2, 3, 1)+1)*(255//self._num_classes[class_name])
                    query_batch_path = os.path.join(_iter_path, 'Query_Iter_' + str(iter_num + 1) + '.png')
                    show_combined_images(query_images_np, query_anno_np, query_batch_path)

                    del query_images_np, query_anno_np

            else:
                if iter_num == 0 and self._epoch in self._visualization_epochs:

                    _iter_path = os.path.join(self._val_epoch_path, 'Iter_' + str(iter_num + 1))
                    if not os.path.isdir(_iter_path):
                        os.makedirs(_iter_path, exist_ok=True)

                    # Save the query images in the validation set
                    query_images = data['query_image'].cpu().detach()
                    query_images_np = query_images.numpy().transpose(0, 2, 3, 1)
                    query_anno_np = (data['query_segmentation'].cpu().detach().numpy().transpose(0, 2, 3, 1)+1)*(255//self._num_classes[class_name])
                    query_batch_path = os.path.join(_iter_path, 'Query_Iter_' + str(iter_num + 1) + '.png')
                    show_combined_images(query_images_np, query_anno_np, query_batch_path)

                    del query_images_np, query_anno_np

            # Migrates tensor in the data dictionary to the GPU
            data = recursive_to(data, self._device)

            self._query_image = data['query_image']
            self._query_segmentation = data['query_segmentation']

            # Done for the CrossEntropy Loss Pytorch
            groundtruth = torch.squeeze(self._query_segmentation, dim=1)

            # Prediction of logits by the model
            # This requires the class type and the input
            self._model.class_name = class_name
            query_pred = self._model(self._query_image)        # Shape : [B, C, H, W], where C varies from one dataset to the other

            # weighted CE loss with ignore_index=-1 for the Penobscot dataset
            ce_weight = self.class_weights[class_name.split("_")[0]]

            if 'penobscot' in class_name:
                WCE_loss = nn.CrossEntropyLoss(weight=torch.tensor(ce_weight).to(self._device), ignore_index=self.ignore_index)(query_pred, groundtruth)
            else:
                WCE_loss = nn.CrossEntropyLoss(weight=torch.tensor(ce_weight).to(self._device))(query_pred, groundtruth)
            avg_WCE_loss += WCE_loss

            # Softmax on the predicted logits to get pixel-wise class predictions
            softmax_query_pred = self.predict_act(query_pred)
            softmax_query_pred = torch.argmax(softmax_query_pred, axis=1)

            # Get IoU scores using the predicted classes and the ground truth
            iou_summary = self._get_iou_summary(self.class_indices[class_name], softmax_query_pred.to(torch.int64), groundtruth.to(torch.int64))
            iou_sum = np.nanmean(iou_summary[iou_summary > 0.0])

            # Optimization of the model paramters
            self._optimizer.zero_grad()
            if mode == 'train':
                WCE_loss.backward()
                self._optimizer.step()

            # Visualization of the input and the predicted output
            # visualize_this_iteration = (iter_num == 0 and self._epoch in self._visualization_epochs)
            # if visualize_this_iteration:
            if class_name not in print_classes and self._epoch in self._visualization_epochs:
                print(f"Visualization at split {mode}, epoch {self._epoch}, and in-epoch-iteration {iter_num + 1}.")
                self._plot_data(mode, class_name, data, softmax_query_pred, self._num_classes[class_name], iter_num)
                print_classes.append(class_name)

            # Printing the loss
            if (iter_num) % self._print_interval == 0:
                if iter_num > 0:
                    mean_iter_time = time_between_logs/self._print_interval
                    time_between_logs = 0
                else:
                    time_between_logs += (time.time() - start_time)
                    mean_iter_time = time_between_logs

                loss_str = WCE_loss
                now = datetime.now()
                # RAM used in GB
                cpu_mem = psutil.Process(os.getpid()).memory_info().rss//(1024**3)
                # GPU memory used in MB
                gpu_mem = torch.cuda.max_memory_allocated()//(1024**2)
                print(f"{now} [{mode}: {self._epoch}, {iter_num + 1:4d}], Class: {class_name}, Num_classes: {self._num_classes[class_name]}, "
                      f"Class_indices: {self.class_indices[class_name]}, Loss: {loss_str}, IoU: {iou_summary}, "
                      f"Pred_Class_Argmax: {np.unique(softmax_query_pred.cpu().detach().numpy())}, GT_Class: {np.unique(self._query_segmentation.cpu().detach().numpy())}"
                      f"CPU MEM: {cpu_mem} GB, GPU MEM:{gpu_mem} MB,"
                      f"Iter time: {mean_iter_time},")
                torch.cuda.reset_peak_memory_stats(self._device)

            # Writing the losses to the tensorboard
            if mode == 'train':
                global_iter = (self.num_train_batches * (self._epoch - 1)) + iter_num
                self._summary_writer.add_scalar('train/CE_loss', WCE_loss, global_iter)
                self._summary_writer.add_scalar('train/IoU_Sum', iou_sum, global_iter)
            else:
                global_iter = (self.num_val_batches * (self._epoch - 1)) + iter_num
                self._summary_writer.add_scalar('val/CE_loss', WCE_loss, global_iter)
                self._summary_writer.add_scalar('val/IoU_Sum', iou_sum, global_iter)

            time_between_logs += (time.time() - start_time)
            start_time = time.time()

            # Deallocate memory
            del data
            gc.collect()
            torch.cuda.empty_cache()


        avg_WCE_loss = avg_WCE_loss/(iter_num+1)
        if mode == 'val':
            # Saving the model with the best validation loss
            if avg_WCE_loss < self.best_val_loss:
                self.best_val_loss = avg_WCE_loss
                self.best_val_epoch = self._epoch
                print("Saving the best model parameters obtained so far....")
                self.save_checkpoint(is_best=True)
            # Adjusting the learning rate based on the validation loss
            self._lr_sched.step(avg_WCE_loss)

        # Tracking the learning rate change
        cur_learning_rate = self._optimizer.param_groups[0]['lr']
        self._summary_writer.add_scalar('LR adjustment', cur_learning_rate, self._epoch)

        print(f"Number of minibatches used in {mode} mode, is {iter_num + 1}.")
        print(f"[{mode}: {self._epoch}] Loss: {avg_WCE_loss}")

        return


    def save_checkpoint(self, is_best=False):
        """Saves a checkpoint of the network and other variables."""
        state = {
            'epoch': self._epoch,
            'net_type': type(self._model).__name__,
            'net': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict()
        }
        if is_best == True:
            best_model_file_path = '{}/bestmodel.pth.tar'.format(self._checkpoint_path)
            print(best_model_file_path)
            torch.save(state, best_model_file_path)
        else:
            file_path = '{}/{}_ep{:04d}.pth.tar'.format(self._checkpoint_path, self._save_name, self._epoch)
            print(file_path)
            torch.save(state, file_path)


    def load_checkpoint(self, checkpoint=None):
        """Loads a network checkpoint file.
        """
        if checkpoint is None: # Load most recent checkpoint
            checkpoint_list = sorted(glob.glob('{}/{}_ep*.pth.tar'.format(self._checkpoint_path, self._save_name)))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int): # Checkpoint is the epoch number
            checkpoint_path = '{}/{}_ep{:04d}.pth.tar'.format(self._checkpoint_path, self._save_name, checkpoint)
        elif isinstance(checkpoint, str): # checkpoint is the epoch file path
            checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError
        if not os.path.isfile(checkpoint_path):
            print(f"WARNING: Attempted to load checkpoint at epoch {checkpoint}, but it does not"
                  + " exist. Continuing without loading. If runfile is correctly set up, there will"
                  + " be an upcoming training stage that will begin from scratch.")
            return

        checkpoint_dict = torch.load(checkpoint_path)
        assert type(self._model).__name__ == checkpoint_dict['net_type'], 'Network is not of correct type'

        # Remove the FC layers of the projection head (2 FC layers with weights and biases for three different datasets)
        trained_model_param = checkpoint_dict['net']
        sliced = islice(trained_model_param.items(), len(trained_model_param.keys()) - 6)
        trained_model_param = OrderedDict(sliced)
        image_encoder_param = OrderedDict([(k.replace('image_encoder.', ''), v) for k, v in trained_model_param.items() if 'image_encoder' in k])
        self._model.image_encoder.load_state_dict(image_encoder_param, strict=True)
        print("Loaded the encoder parameters....")
        shared_decoder_param = OrderedDict([(k.replace('shared_decoder.', ''), v) for k, v in trained_model_param.items() if 'shared_decoder' in k])
        self._model.shared_decoder.load_state_dict(shared_decoder_param, strict=True)
        print("Loaded the decoder parameters....")

        print("Loaded: {}".format(checkpoint_path))