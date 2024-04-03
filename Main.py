import os
import argparse
import random
import numpy as np
from itertools import islice
from collections import OrderedDict
import torch
torch.set_printoptions(edgeitems=4, linewidth=117)
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from config.local_config import create_config
from train.DGP_trainer import Deep_GP_Trainer
from models import DGP_resnet_unet
from kernels.gp_kernels import RBF, LinearKernel
from data.Datasets import Seismic_Segmentation_Task


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_init(to_device):
    cuda_avail = torch.cuda.is_available()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cpu")
    if cuda_avail and 'cuda' in to_device:
        device = torch.device(to_device)
        torch.cuda.set_device(device)

    return cuda_avail, device


def test(model, device, save_dir, test_data_loader):

    evaluator = DGP_Evaluator(
        data_visualization_path=save_dir,
        device=device,
    )
    return evaluator.evaluate(model, test_data_loader)


def main():

    parser = argparse.ArgumentParser(description="Experiment runfile, you run experiments from this file")
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--test_num_support", type=int, default=5)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--freeze_bn", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--img_enc_type", type=str, default='resnet', help="Could be UNet (input unet) or ResNet50 (input resnet)")
    parser.add_argument("--img_enc_checkpoint")
    parser.add_argument("--checkpoint")
    parser.add_argument("--image_net_stat", type=bool, default=False)
    parser.add_argument("-d", "--device", dest="device", help="Device to run on, the cpu or gpu.",
                        type=str, default="cuda:0")
    parser.add_argument("--restart", action="store_true", default=False)
    args = parser.parse_args()

    print("Started script: {}, with pytorch {}".format(os.path.basename(__file__), torch.__version__))
    print(args)
    print(f"Seeding with seed: {args.seed}")
    seed_all(args.seed)
    cuda_avail, device = torch_init(args.device)
    print("pytorch using device", device)
    print(args.image_net_stat)

    # Dataset details
    k_shot = args.shots

    # Gaussian process parameters
    covar_size = 5
    covariance_output_mode = 'concatenate variance'
    depth_image_encoder = 512

    # Freeze batch normalization layers of the encoder, UNet or ResNet50
    freeze_bn = args.freeze_bn

    img_enc_type = args.img_enc_type
    loaded_encoder_chkpnt = False
    if img_enc_type=='resnet':
        resnet = models.resnet50(pretrained=args.image_net_stat)
        resnet.fc = nn.Identity()

        if args.img_enc_checkpoint:
            # LANDMASS statistics path
            # checkpoint_path = '../Classification/checkpoints/Run_2/classification_0_ep0025.pth.tar'
            checkpoint_dict = torch.load(args.img_enc_checkpoint, map_location=args.device)
            trained_model_param = checkpoint_dict['state_dict']

            # Remove the FC layers of the projection head (2 FC layers with weights and biases)
            sliced = islice(trained_model_param.items(), len(trained_model_param.keys()) - 4)
            trained_model_param = OrderedDict(sliced)

            # replace the prefix 'backbone.' with None
            trained_model_param = OrderedDict([(k.replace('backbone.', ''), v) for k, v in trained_model_param.items()])

            # load the model parameters with strict checking
            resnet.load_state_dict(trained_model_param, strict=True)
            print("Loaded the SimCLR backbone....")

        img_encoder_obj = DGP_resnet_unet.Image_Encoder(resnet, freeze_bn)
        mask_encoder_obj = DGP_resnet_unet.Mask_Encoder()
        dgp_model = DGP_resnet_unet.DGPModel(kernel=RBF(length=(1/(depth_image_encoder**0.25))),
                             covariance_output_mode=covariance_output_mode, covar_size=covar_size)
        fss_decoder_obj = DGP_resnet_unet.FSS_Decoder(covar_size=covar_size)
        fss_learner_obj = DGP_resnet_unet.FSSLearner(image_encoder=img_encoder_obj, anno_encoder=mask_encoder_obj, dgp_model=dgp_model, upsampler=fss_decoder_obj)
        print("Done with the model initialization of the Resnet_UNet....")
        loaded_encoder_chkpnt = True
    fss_learner_obj.to(device)

    config = create_config()
    # Training of the model parameters
    if args.train:

        # Added the run_id
        run_id = args.run_id
        config['exp_spec'] = os.path.join(config['exp_spec'], str(k_shot)+'-shot', 'Run_' + str(run_id))
        os.makedirs(config['exp_spec'], exist_ok=True)
        config['tb_dir'] = os.path.join(config['tb_dir'], str(k_shot)+'-shot', 'Run_' + str(run_id))
        os.makedirs(config['tb_dir'], exist_ok=True)
        config['visualization_path'] = os.path.join(config['visualization_path'], str(k_shot)+'-shot', 'Run_' + str(run_id))
        os.makedirs(config['visualization_path'], exist_ok=True)
        config['checkpoint_path'] = os.path.join(config['checkpoint_path'], str(k_shot)+'-shot', 'Run_' + str(run_id))
        os.makedirs(config['checkpoint_path'], exist_ok=True)

        # Tensorboard writer
        td_dir = config['tb_dir']
        tb_writer = SummaryWriter(td_dir)

        fptr_path = os.path.join(config['exp_spec'], 'Experiment_spec_run_id_'+str(run_id)+'.txt')
        exp_spec_fptr = open(fptr_path, 'w')
        exp_spec_fptr.write('Details about experiment run_id : ' + str(run_id) + '\n')
        exp_spec_fptr.write('Shots                           : ' + str(k_shot) + '\n')

        if args.image_net_stat and not args.img_enc_checkpoint:
            pretrained_stat_str = 'Image encoder initialized with Image net statistics'
        elif not args.image_net_stat and args.img_enc_checkpoint:
            pretrained_stat_str = 'Image encoder initialized with SimCLR statistics'
        else:
            pretrained_stat_str = 'Image encoder initialized randomly'

        exp_spec_fptr.write('Pretrained statistics           : ' + pretrained_stat_str + '\n')
        exp_spec_fptr.write('Freeze batch normalization stat : ' + str(freeze_bn) + '\n\n')

        if loaded_encoder_chkpnt:
            exp_spec_fptr.write("\nCheckpoint used for loading ONLY the encoder model parameters.... \n")
            exp_spec_fptr.write('Checkpoint path                 : ' + str(args.img_enc_checkpoint) + '\n\n')

        if args.checkpoint:
            exp_spec_fptr.write('\nCheckpoint used for loading all the model parameters.... \n')
            exp_spec_fptr.write('Checkpoint path                 : ' + str(args.checkpoint) + '\n\n')


        params_image_encoder = [param for param in img_encoder_obj.parameters()]
        params_mask_encoder = [param for param in mask_encoder_obj.parameters()]
        params_gp = {param for param in dgp_model.parameters()}
        params_image_decoder = [param for param in fss_decoder_obj.parameters()]
        other_params = set(params_mask_encoder + params_image_decoder) - params_gp
        trainable_params = [{'params': [param for param in other_params if param.requires_grad]},
                      {'params': [param for param in params_image_encoder if param.requires_grad], 'lr': 1e-06}] #2.5e-07 For Run ID: 1, 3 in 5-shot from epoch 11



        # Optimization hyper-parameters
        num_epochs = 10
        train_batch_size = 2
        val_batch_size = 8
        patch_size = args.patch_size

        # Optimizer parameters
        model_lr = 5e-5 #1.25e-05 For Run ID: 1, 3 in 5-shot from epoch 11
        model_weight_decay = 1e-3
        model_optimizer = optim.AdamW(params=trainable_params, lr=model_lr,
                                     weight_decay=model_weight_decay)

        # LR schedular parameters
        lr_decay_factor = 0.25
        adjust_lr_epoch = 5
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(model_optimizer,
                                                         mode='min',
                                                         factor=lr_decay_factor,
                                                         patience=adjust_lr_epoch
                                                         )

        # Data loader for training and validation data
        classes = config['classes']
        data_info = config['data_info']

        seismic_seg_data_loader = Seismic_Segmentation_Task(classes, data_info, patch_size=patch_size, k_shot=k_shot, train_batch_size=train_batch_size,
                                                            val_batch_size=val_batch_size, debug=False)

        epoch_interval = 5
        vis_epochs = [1] + np.arange(epoch_interval, num_epochs+epoch_interval, epoch_interval).tolist()
        dgp_fss_trainer = Deep_GP_Trainer(tensorboard_writer=tb_writer, model=fss_learner_obj, loss='weighted_bce',
                                          optimizer=model_optimizer, dataset_n_loader=seismic_seg_data_loader, checkpoint_path=config['checkpoint_path'],
                                          data_visualization_path=config['visualization_path'], device=device, checkpoint_epochs=[num_epochs//2, num_epochs], print_interval=20,
                                          visualization_epochs=vis_epochs, #[num_epochs//2, num_epochs]
                                          save_name=f"{os.path.splitext(os.path.basename(__file__))[0]}_{args.seed}", lr_sched=lr_scheduler)

        if args.checkpoint:
            dgp_fss_trainer.load_checkpoint(args.device, args.checkpoint)
            for param_group in dgp_fss_trainer._optimizer.param_groups:
                print(param_group['lr'])
            print(dgp_fss_trainer._epoch)

        exp_spec_fptr.write('Meta-train classes              : ' + str(classes) + '\n')
        exp_spec_fptr.write('Training and validation data' + '\n')
        exp_spec_fptr.write("Number of training examples     : " + str(len(seismic_seg_data_loader.traindataset)) + '\n')
        exp_spec_fptr.write("Number of validation examples   : " + str(len(seismic_seg_data_loader.validdataset)) + '\n')
        exp_spec_fptr.write("Initial seed                    : " + str(args.seed) + '\n\n')

        exp_spec_fptr.write('Details about the training patches' + '\n')
        for class_name in classes:
            exp_spec_fptr.write('Class name                     : ' + str(class_name) + '\n')
            exp_spec_fptr.write('Patch overlap                  : ' + str(data_info[class_name]['patch_overlap']) + '\n')
            exp_spec_fptr.write('Number of train slices         : ' + data_info[class_name]['train_indices'] + '\n\n')
        exp_spec_fptr.write('\n')

        exp_spec_fptr.write('Distribution of patches' + '\n')
        exp_spec_fptr.write(f"(class_idx, num_samples): {[(c, len(lst)) for c, lst in seismic_seg_data_loader.traindataset.img_metadata_classwise.items()]}")
        exp_spec_fptr.write("\n")
        exp_spec_fptr.write(f"(class_idx, num_samples): {[(c, len(lst)) for c, lst in seismic_seg_data_loader.validdataset.img_metadata_classwise.items()]}")
        exp_spec_fptr.write('\n')

        exp_spec_fptr.write('Class weights for the loss computation' + '\n')
        exp_spec_fptr.write('Number of classes               : ' + str(len(dgp_fss_trainer.class_weights)) + '\n')
        exp_spec_fptr.write('Class weights for loss          : ' + str(dgp_fss_trainer.class_weights) + '\n')

        exp_spec_fptr.write('Training and validation parameters' + '\n')
        exp_spec_fptr.write('Epochs                          : ' + str(num_epochs) + '\n')
        exp_spec_fptr.write('Patch size                      : ' + str(patch_size) + '\n')
        exp_spec_fptr.write('Train batch size                : ' + str(train_batch_size) + '\n')
        exp_spec_fptr.write('Val batch size                  : ' + str(val_batch_size) + '\n')

        exp_spec_fptr.write('Optimizer parameters' + '\n')
        exp_spec_fptr.write('Learning rate                   : ' + str(model_lr) + '\n')
        exp_spec_fptr.write('Weight decay                    : ' + str(model_weight_decay) + '\n\n')

        exp_spec_fptr.write('Scheduler parameters' + '\n')
        exp_spec_fptr.write('Learning rate decay             : ' + str(lr_decay_factor) + '\n')
        exp_spec_fptr.write('Patience (epoch)                : ' + str(adjust_lr_epoch) + '\n\n')
        exp_spec_fptr.flush()

        if not args.restart:
            dgp_fss_trainer.load_checkpoint(args.device)
            for param_group in dgp_fss_trainer._optimizer.param_groups:
                print(param_group['lr'])

        best_val_epoch, best_val_loss = dgp_fss_trainer.train(num_epochs)
        exp_spec_fptr.write('Best model parameters are obtained for epoch ' + str(best_val_epoch) + ' with validation loss of ' + str(best_val_loss) + '\n')
        exp_spec_fptr.flush()
        exp_spec_fptr.close()


# Run main
main()