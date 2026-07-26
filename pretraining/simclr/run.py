import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
from data.Datasets import SimCLR_dataset
from config.local_config import create_config
import os
import numpy as np


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--log-val-every-n-steps', default=1000, type=int,
                    help='Log val results every n steps')
parser.add_argument('--viz_every_n_steps', default=1000, type=int,
                    help='Visaulize top5 every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument("--patch_size", type=int, default=256)
parser.add_argument("--run_id", type=int, required=True)
parser.add_argument("--load_checkpoint", action="store_true", default=False)
parser.add_argument("--checkpoint")


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    config = create_config()
    classes = config['classes']
    data_info = config['data_info']
    train_batch_size = val_batch_size = args.batch_size

    ############################
    # Create the log directories
    ############################
    run_id = args.run_id
    config['exp_spec'] = os.path.join(config['exp_spec'], 'Run_' + str(run_id))
    os.makedirs(config['exp_spec'], exist_ok=True)
    config['tb_dir'] = os.path.join(config['tb_dir'], 'Run_' + str(run_id))
    os.makedirs(config['tb_dir'], exist_ok=True)
    config['visualization_path'] = os.path.join(config['visualization_path'], 'Run_' + str(run_id))
    os.makedirs(config['visualization_path'], exist_ok=True)
    config['checkpoint_path'] = os.path.join(config['checkpoint_path'], 'Run_' + str(run_id))
    os.makedirs(config['checkpoint_path'], exist_ok=True)

    # dataset and dataloader
    seismic_seg_data_loader = SimCLR_dataset(classes, data_info, patch_size=256, train_batch_size=train_batch_size, val_batch_size=val_batch_size, debug=False)
    train_loader = seismic_seg_data_loader.train_loader

    # model
    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

    # optimizer and schedular
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    ##########################
    # Experiment specification
    ##########################
    fptr_path = os.path.join(config['exp_spec'], 'Experiment_spec_run_id_' + str(run_id) + '.txt')
    exp_spec_fptr = open(fptr_path, 'w')
    exp_spec_fptr.write('Details about experiment run_id    : ' + str(run_id) + '\n')
    exp_spec_fptr.write('Classes                            : ' + str(classes) + '\n')
    exp_spec_fptr.write('Training and validation data' + '\n')
    exp_spec_fptr.write("Number of training examples        : " + str(len(seismic_seg_data_loader.traindataset)) + '\n')
    exp_spec_fptr.write("Number of validation examples      : " + str(len(seismic_seg_data_loader.validdataset)) + '\n')

    exp_spec_fptr.write('Details about the training patches' + '\n')
    for class_name in classes:
        exp_spec_fptr.write('Class name                     : ' + str(class_name) + '\n')
        exp_spec_fptr.write('Patch overlap                  : ' + str(data_info[class_name]['patch_overlap']) + '\n')
        exp_spec_fptr.write('Number of train slices         : ' + data_info[class_name]['train_indices'] + '\n')
        exp_spec_fptr.write('Class weight                   : ' + str(seismic_seg_data_loader.traindataset.class_weights[class_name]) + '\n\n')
    exp_spec_fptr.write('\n')

    exp_spec_fptr.write('Distribution of patches' + '\n')
    exp_spec_fptr.write(f"(class_idx, num_samples): {[(c, len(lst)) for c, lst in seismic_seg_data_loader.traindataset.img_metadata_classwise.items()]}")
    exp_spec_fptr.write("\n")
    exp_spec_fptr.write(f"(class_idx, num_samples): {[(c, len(lst)) for c, lst in seismic_seg_data_loader.validdataset.img_metadata_classwise.items()]}")
    exp_spec_fptr.write('\n')

    exp_spec_fptr.write('Training and validation parameters' + '\n')
    exp_spec_fptr.write('Epochs                             : ' + str(args.epochs) + '\n')
    exp_spec_fptr.write('Patch size                         : ' + str(args.patch_size) + '\n')
    exp_spec_fptr.write('Batch size                         : ' + str(train_batch_size) + '\n')
    exp_spec_fptr.write('Temperature                        : ' + str(args.temperature) + '\n\n')

    exp_spec_fptr.write('Optimizer parameters' + '\n')
    exp_spec_fptr.write('Learning rate                      : ' + str(args.lr) + '\n')
    exp_spec_fptr.write('Weight decay                       : ' + str(args.weight_decay) + '\n\n')

    exp_spec_fptr.flush()
    exp_spec_fptr.close()

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args,
                        checkpoint_path=config['checkpoint_path'],
                        data_visualization_path=config['visualization_path'],
                        tb_dir=config['tb_dir'],
                        checkpoint_epochs=np.arange(1, args.epochs, 5))
        simclr.train(seismic_seg_data_loader)


if __name__ == "__main__":
    main()
