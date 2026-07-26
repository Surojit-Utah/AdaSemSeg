import os
import argparse
import random
import numpy as np
import pickle
import torch
import torch.nn as nn
from torchvision import models
from config.local_config import create_config
from models import DGP_unet, DGP_resnet_unet
from data.Datasets import Dataset_Loader
from predict.DGP_evaluator import FSSEvaluator


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


def test(model, device, save_dir, test_data_loader, visualize=False):
    num_classes = test_data_loader.class_label_count[test_data_loader.class_name]

    evaluator = FSSEvaluator(
        num_classes,
        data_visualization_path=save_dir,
        device=device,
    )
    return evaluator.evaluate(model, test_data_loader, visualize=visualize)


def main():

    parser = argparse.ArgumentParser(description="Experiment runfile, you run experiments from this file")
    parser.add_argument("--visualize", type=int, required=True)
    parser.add_argument("--img_enc_type", type=str, default='resnet', help="Could be UNet (input unet) or ResNet50 (input resnet)")
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--best_model", action="store_true", default=False)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--train_indices", type=str, required=True)
    parser.add_argument("--eval_mode", type=str, default='test', required=True)
    parser.add_argument("-d", "--device", dest="device", help="Device to run on, the cpu or gpu.",
                        type=str, default="cuda:0")
    args = parser.parse_args()
    cuda_avail, device = torch_init(args.device)

    # exp config
    config = create_config()
    data_info = config['data_info']

    # Dataset details
    datasets = config['classes']
    print("Select from classes : " + str(datasets))
    source_class = None
    while(1):
        source_class = input("Enter your source_class : ")
        if source_class in config['data_info'].keys():
            break
    batch_size = 1
    best_model = args.best_model
    run_id = args.run_id
    eval_mode = args.eval_mode

    freeze_bn = True
    img_enc_type = args.img_enc_type
    if img_enc_type=='resnet':
        resnet = models.resnet50()
        img_encoder_obj = DGP_resnet_unet.Image_Encoder(resnet, freeze_bn)
        img_decoder_obj = DGP_resnet_unet.Image_Decoder()
        seg_model_obj = DGP_resnet_unet.Segmentation_Network(image_encoder=img_encoder_obj, upsampler=img_decoder_obj)
        print("Done with the model initialization....")
    else:
        img_encoder_obj = DGP_unet.Image_Encoder(features=32)
        img_decoder_obj = DGP_unet.Image_Decoder()
        seg_model_obj = DGP_unet.FSSLearner(image_encoder=img_encoder_obj, upsampler=img_decoder_obj)
        print("Done with the model initialization....")

    loaded_FSS_chkpoint = False
    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
        if best_model:
            exp_spec_path = os.path.join(config['exp_spec'], source_class, 'Train_slices_' + args.train_indices,
                                         'Run_' + str(run_id) + '_best_model', eval_mode)
            os.makedirs(exp_spec_path, exist_ok=True)
            visualization_path = os.path.join(config['visualization_path'], source_class, 'Train_slices_' + args.train_indices,
                                              'Run_' + str(run_id) + '_best_model', eval_mode)
            os.makedirs(visualization_path, exist_ok=True)
            results_path = os.path.join(config['results_path'], source_class, 'Train_slices_' + args.train_indices,
                                        'Run_' + str(run_id) + '_best_model', eval_mode)
            os.makedirs(results_path, exist_ok=True)

            # Loading model parameters from checkpoint path
            checkpoint_path = os.path.join(checkpoint_dir, 'bestmodel.pth.tar')
        else:
            epoch = args.epoch
            model_save_path = '{}_ep{:04d}.pth.tar'.format('Main_0', epoch)
            exp_spec_path = os.path.join(config['exp_spec'], source_class, 'Train_slices_' + args.train_indices,
                                         'Run_' + str(run_id) + '_trained_model', eval_mode)
            os.makedirs(exp_spec_path, exist_ok=True)
            visualization_path = os.path.join(config['visualization_path'], source_class, 'Train_slices_' + args.train_indices,
                                              'Run_' + str(run_id) + '_trained_model', eval_mode)
            os.makedirs(visualization_path, exist_ok=True)
            results_path = os.path.join(config['results_path'], source_class, 'Train_slices_' + args.train_indices,
                                        'Run_' + str(run_id) + '_trained_model', eval_mode)
            os.makedirs(results_path, exist_ok=True)

            # Loading model parameters from checkpoint path
            checkpoint_path = os.path.join(checkpoint_dir, model_save_path)

        checkpoint = torch.load(checkpoint_path, map_location=torch.device(args.device))
        assert type(seg_model_obj).__name__ == checkpoint['net_type']
        seg_model_obj.load_state_dict(checkpoint['net'])
        print("Loaded the model parameters....")
        seg_model_obj.to(device)
        loaded_FSS_chkpoint = True

    fptr_path = os.path.join(exp_spec_path, 'Experiment_spec_run_id_' + str(run_id) + '.txt')
    exp_spec_fptr = open(fptr_path, 'w')
    exp_spec_fptr.write('Details about experiment run_id : ' + str(run_id) + '\n')
    exp_spec_fptr.write('Image encoder type              : ' + str(img_enc_type) + '\n')
    exp_spec_fptr.write('Source class                    : ' + source_class + '\n')
    exp_spec_fptr.write('Number of training slices       : ' + args.train_indices + '\n')

    if loaded_FSS_chkpoint:
        exp_spec_fptr.write('\nCheckpoint used for loading all the model parameters.... \n')
        exp_spec_fptr.write('Checkpoint path                 : ' + str(checkpoint_path) + '\n\n')

    exp_spec_fptr.flush()
    exp_spec_fptr.close()

    test_loader = Dataset_Loader(source_class, data_info, eval_mode=eval_mode, batch_size=batch_size)

    # Calling the model evaluator
    visualize_model_pred = bool(args.visualize)
    cur_save_dir = visualization_path
    iou_score_by_class, iou_list, tricky_examples, metric_scores_dict, eval_time_stats = test(seg_model_obj, device, cur_save_dir, test_loader, visualize_model_pred)

    ###############################################
    # Save the IoU scores and related metric scores
    ###############################################
    save_dict_path = os.path.join(results_path, 'save_iou_dict.pickle')
    iou_dict = dict()
    iou_dict['iou_scores'] = np.array(iou_list)
    iou_dict['iou_scores_by_class'] = iou_score_by_class
    with open(save_dict_path, 'wb') as handle:
        pickle.dump(iou_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    save_dict_path = os.path.join(results_path, 'save_metric_scorers_dict.pickle')
    with open(save_dict_path, 'wb') as handle:
        pickle.dump(metric_scores_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    ########################################
    # Save the IoU scores computed per class
    ########################################
    method1_avg = np.mean(np.array(iou_list))
    method1_std_dev = np.std(np.array(iou_list))
    file_path = os.path.join(results_path, "Summary.txt")
    sel_summary_fptr = open(file_path, 'w')

    for class_index in range(iou_score_by_class.shape[1]):
        sel_summary_fptr.write('Number of training slices : ' + args.train_indices + '\n')
        sel_summary_fptr.write('IoU scores for class      : ' + str(class_index+1) + '\n')
        class_iou_scores = iou_score_by_class[:, class_index]
        mean_iou_score = np.mean(class_iou_scores)
        stddev_iou_score = np.std(class_iou_scores)
        sel_summary_fptr.write(str(np.round(mean_iou_score, 3)) + ' \u00B1 ' + str(np.round(stddev_iou_score, 3)) + '\n')
    sel_summary_fptr.write('\n\n')

    sel_summary_fptr.write('IoU scores      : ' + str(iou_list) + '\n')
    sel_summary_fptr.write('Average IoU using Method1 is ' + str(np.round(method1_avg, 3)) +
                           ' with std-dev ' + str(np.round(method1_std_dev, 3)) + '\n\n')
    sel_summary_fptr.write('Tricky examples \n')
    for example in tricky_examples:
        sel_summary_fptr.write(str(example) + ', ')
    sel_summary_fptr.write('\n')
    sel_summary_fptr.write(f'Average Evaluation Time for {eval_time_stats["num_samples"]} samples is {eval_time_stats["avg_time_taken"]}\n')
    sel_summary_fptr.write(f'Resolution: {eval_time_stats["image_height"]} X {eval_time_stats["image_width"]}\n')
    sel_summary_fptr.flush()
    sel_summary_fptr.close()


    ########################
    # Save the Metric scores
    ########################
    file_path = os.path.join(results_path, "Metric_scores_summary.txt")
    score_summary_fptr = open(file_path, 'w')
    for key, value in metric_scores_dict.items():
        if key=='confusion_matrix':
            continue
        elif key == 'classwise_IoU':
            score_summary_fptr.write('\n')
            for class_index, iou_score in value.items():
                score_summary_fptr.write('IoU scores for class      : ' + str(class_index+1) + ' is ' + str(iou_score) + '\n')
            score_summary_fptr.write('\n')
        else:
            score_summary_fptr.write(key + str(value) + '\n')
    score_summary_fptr.flush()
    score_summary_fptr.close()


# Run main
main()