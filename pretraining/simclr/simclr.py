import logging
import os
import glob
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import accuracy #save_config_file, save_checkpoint
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        # logging paths
        self.checkpoint_path = kwargs['checkpoint_path']
        self.data_visualization_path = kwargs['data_visualization_path']
        self.train_data_visualization_path = os.path.join(self.data_visualization_path, 'Train')
        os.makedirs(self.train_data_visualization_path, exist_ok=True)
        self.val_data_visualization_path = os.path.join(self.data_visualization_path, 'Val')
        os.makedirs(self.val_data_visualization_path, exist_ok=True)

        self.tb_dir = kwargs['tb_dir']
        self.checkpoint_epochs = kwargs['checkpoint_epochs']
        self.viz_images = 5

        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter(self.tb_dir)
        logging.basicConfig(filename=os.path.join(self.tb_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

        if self.args.load_checkpoint:
            self.load_checkpoint(self.args.checkpoint)

        # track the number of epochs the model has been trained
        self._epoch = 0


    def save_checkpoint(self, is_best=False):
        """Saves a checkpoint of the network and other variables."""
        state = {
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if is_best == True:
            best_model_file_path = '{}/bestmodel.pth.tar'.format(self.checkpoint_path)
            torch.save(state, best_model_file_path)
            logging.info(f"Model checkpoint and metadata has been saved at {best_model_file_path}.")
        else:
            # save model checkpoints
            file_path = '{}/checkpoint_{:04d}.pth.tar'.format(self.checkpoint_path, self._epoch)
            torch.save(state, file_path)
            logging.info(f"Model checkpoint and metadata has been saved at {file_path}.")


    def load_checkpoint(self, checkpoint=None):
        """Loads a network checkpoint file.
        """
        if checkpoint is None:  # Load most recent checkpoint
            checkpoint_list = sorted(glob.glob('{}/checkpoint_*.pth.tar'.format(self.checkpoint_path)))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int):  # Checkpoint is the epoch number
            checkpoint_path = '{}/checkpoint_{:04d}.pth.tar'.format(self.checkpoint_path, checkpoint)
        elif isinstance(checkpoint, str):  # checkpoint is the epoch file path
            checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError
        if not os.path.isfile(checkpoint_path):
            print(f"WARNING: Attempted to load checkpoint at epoch {checkpoint}, but it does not"
                  + " exist. Continuing without loading. If runfile is correctly set up, there will"
                  + " be an upcoming training stage that will begin from scratch.")
            return
        checkpoint_dict = torch.load(checkpoint_path)
        # print(type(self.model).__name__)
        # print(checkpoint_dict['arch'])
        # assert type(self.model).__name__ == checkpoint_dict['arch'], 'Network is not of correct type'
        self.epoch = checkpoint_dict['epoch']
        self.model.load_state_dict(checkpoint_dict['state_dict'])
        self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
        print("Loaded: {}".format(checkpoint_path))


    def show_combined_images(self, train_minibatch, logits, save_img_path):

        aug_dict = {1: 'RandomRotate',
                    2: 'RandomHorizontalFlip',
                    3: 'GaussianBlur',
                    4: 'GaussNoise',
                    5: 'RandomResizedCrop',
                    6: 'Brightness',
                    7: 'Contrast'}

        data = train_minibatch['data']
        data_aug = train_minibatch['image_aug']
        class_names = train_minibatch['class_names']
        data = torch.cat((torch.squeeze(data[0], dim=0), torch.squeeze(data[1], dim=0)), dim=0)
        minibatch_data = data.detach().cpu()
        minibatch_data = minibatch_data.numpy().transpose(0, 2, 3, 1)
        class_names.extend(class_names)
        topmax = 5

        fig = plt.figure(figsize=(self.viz_images*50, (topmax+1)*50))
        fig_width, fig_height = fig.get_size_inches()
        gs = fig.add_gridspec(self.viz_images, (topmax+1))

        total_image_height = self.viz_images * minibatch_data.shape[2]
        total_image_width = (topmax+1) * minibatch_data.shape[1]
        image_shape_array = np.array([total_image_width, total_image_height])
        shape_index = np.argmax(image_shape_array)
        # If width is bigger, map it to the image width size
        if shape_index == 0:
            # print("Width is bigger....")
            set_fig_width = fig_width
            set_fig_height = (fig_width / total_image_width) * total_image_height
            fig.set_figwidth(set_fig_width)
            fig.set_figheight(set_fig_height)
        else:
            # print("Height is bigger....")
            set_fig_height = fig_height
            set_fig_width = (fig_height / total_image_height) * total_image_width
            fig.set_figwidth(set_fig_width)
            fig.set_figheight(set_fig_height)

        # select anchor samples only along the forward positive sample (i, j)
        # and ignore (j, i) indices
        selected_image_samples = np.random.choice(self.args.batch_size, self.viz_images, replace=False)

        # Top 5 samples closest to each anchor sample
        topk_distances, logit_indices = logits.topk(topmax, 1, True, True)
        topk_distances = topk_distances.detach().cpu().numpy()
        logit_indices = logit_indices.detach().cpu().numpy()

        # Outer loop for the anchor image
        for selected_index, anchor_image_index in enumerate(selected_image_samples):
            ax1 = fig.add_subplot(gs[selected_index, 0])
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect('equal')
            ax1.set_title(class_names[anchor_image_index][0], fontweight="bold", size=100)
            aug_index = data_aug[anchor_image_index].cpu().detach().numpy()[0]
            ax1.set_xlabel('Minibatch index : ' + str(anchor_image_index) + '\n' + aug_dict[aug_index], fontweight="bold", size=100)
            plt.imshow(minibatch_data[anchor_image_index], vmin=0, vmax=255)
            # Inner loop for the nearest matching image
            anchor_image_topk_distances = topk_distances[anchor_image_index]
            for list_index, logit_index in enumerate(logit_indices[anchor_image_index]):
                if logit_index==0:
                    nearest_image_index = self.args.batch_size + anchor_image_index
                elif logit_index > 0 and logit_index <= anchor_image_index:
                    nearest_image_index = logit_index-1
                elif logit_index >= self.args.batch_size + anchor_image_index:
                    nearest_image_index = logit_index+1
                # within [anchor_image_index, anchor_image_index+self.args.batch_size]
                # no changes in index
                else:
                    nearest_image_index = logit_index
                ax2 = fig.add_subplot(gs[selected_index, list_index+1])
                ax2.set_xticklabels([])
                ax2.set_yticklabels([])
                ax2.set_aspect('equal')
                ax2.set_title(class_names[nearest_image_index][0], fontweight="bold", size=100)
                aug_index = data_aug[nearest_image_index].cpu().detach().numpy()[0]
                ax2.set_xlabel('Minibatch index : '+str(nearest_image_index) + '\n' + aug_dict[aug_index], fontweight="bold", size=100)
                ax2.set_ylabel('Similarity : '+str(anchor_image_topk_distances[list_index]), fontweight="bold", size=100)
                plt.imshow(minibatch_data[nearest_image_index], vmin=0, vmax=255)

        plt.savefig(save_img_path)
        plt.close(plt.gcf())

        return


    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels


    def train(self, seismic_seg_data_loader):

        train_loader = seismic_seg_data_loader.train_loader
        val_loader = seismic_seg_data_loader.val_loader

        scaler = GradScaler(enabled=self.args.fp16_precision)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {not self.args.disable_cuda}.")
        logging.info(f"GPU index: {self.args.gpu_index}.")

        for epoch_counter in range(self.args.epochs):
            for train_minibatch in tqdm(train_loader):

                data = train_minibatch['data']
                images = torch.cat((torch.squeeze(data[0], dim=0), torch.squeeze(data[1], dim=0)), dim=0)
                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                # train loss and accuracy
                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                # visualization of the training data
                if n_iter % self.args.viz_every_n_steps == 0:
                    # Train data visualization
                    save_img_path = os.path.join(self.train_data_visualization_path, 'top5_'+str(n_iter)+'.png')
                    self.show_combined_images(train_minibatch, logits, save_img_path)

                # validation loss and accuracy
                if n_iter % self.args.log_val_every_n_steps == 0:
                    with torch.no_grad():
                        for val_minibatch in tqdm(val_loader):
                            val_data = val_minibatch['data']
                            val_images = torch.cat((torch.squeeze(val_data[0], dim=0), torch.squeeze(val_data[1], dim=0)), dim=0)
                            val_images = val_images.to(self.args.device)
                            val_features = self.model(val_images)
                            val_logits, val_labels = self.info_nce_loss(val_features)
                            val_loss = self.criterion(val_logits, val_labels)
                            val_top1, val_top5 = accuracy(val_logits, val_labels, topk=(1, 5))
                            self.writer.add_scalar('val_loss', val_loss, global_step=n_iter)
                            self.writer.add_scalar('acc/val_top1', val_top1[0], global_step=n_iter)
                            self.writer.add_scalar('acc/val_top5', val_top5[0], global_step=n_iter)

                    # Validation data visualization
                    save_img_path = os.path.join(self.val_data_visualization_path, 'top5_'+str(n_iter)+'.png')
                    self.show_combined_images(val_minibatch, val_logits, save_img_path)

                n_iter += 1

            self._epoch += 1
            if self._epoch in self.checkpoint_epochs:
                print("Saving Checkpoint....")
                self.save_checkpoint()

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        self.save_checkpoint()
        logging.info(f"Model checkpoint and metadata has been saved at {self.tb_dir}.")