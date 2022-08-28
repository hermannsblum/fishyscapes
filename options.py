import argparse
from optparse import check_choice
from re import T
import torch
import os
import time

def get_calculate_and_inference_parser():
    parser = argparse.ArgumentParser(description='Parameters in calculating statistics and inference on anormaly detection')
    parser.add_argument("--cal", default=False, action="store_true")
    parser.add_argument("--inf", default=False, action="store_true")
    parser.add_argument("--draw", default="False", action="store_true")
    parser.add_argument("--min", default=-1, type=int)
    parser.add_argument("--max", type=int, default=1e8)
    parser.add_argument("--prev_log", type=str, default='')
    parser.add_argument("--cmd_append", type=str, default='')
    parser.add_argument("--disable_mp", default=False, action="store_true")
    parser.add_argument("--tag_if_none", type=str, default='')
    parser.add_argument("--dirname", default='', type=str)
    parser.add_argument('--inf_temp', nargs='+', type=float)
    parser.add_argument('--spec', type=int, default=-1)
    parser.add_argument('--inf_dilation_rate', type=int, nargs='+')
    return parser


def get_train_parser():
    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus',
                        help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                        and deepWV3Plus (backbone: WideResNet38).')
    parser.add_argument('--dataset', nargs='*', type=str, default=['cityscapes'],
                        help='a list of datasets; cityscapes')
    parser.add_argument('--image_uniform_sampling', action='store_true', default=False,
                        help='uniformly sample images across the multiple source domains')
    parser.add_argument('--val_dataset', nargs='*', type=str, default=['cityscapes'],
                        help='validation dataset list')
    parser.add_argument('--val_interval', type=int, default=1, help='validation interval')
    parser.add_argument('--cv', type=int, default=0,
                        help='cross-validation split id to use. Default # of splits set to 3 in config')
    parser.add_argument('--class_uniform_pct', type=float, default=0,
                        help='What fraction of images is uniformly sampled')
    parser.add_argument('--class_uniform_tile', type=int, default=1024,
                        help='tile size for class uniform sampling')
    parser.add_argument('--coarse_boost_classes', type=str, default=None,
                        help='use coarse annotations to boost fine data with specific classes')

    parser.add_argument('--img_wt_loss', action='store_true', default=False,
                        help='per-image class-weighted loss')
    parser.add_argument('--cls_wt_loss', action='store_true', default=False,
                        help='class-weighted loss')
    parser.add_argument('--batch_weighting', action='store_true', default=False,
                        help='Batch weighting for class (use nll class weighting using batch stats')

    parser.add_argument('--jointwtborder', action='store_true', default=False,
                        help='Enable boundary label relaxation')
    parser.add_argument('--strict_bdr_cls', type=str, default='',
                        help='Enable boundary label relaxation for specific classes')
    parser.add_argument('--rlx_off_iter', type=int, default=-1,
                        help='Turn off border relaxation after specific epoch count')
    parser.add_argument('--rescale', type=float, default=1.0,
                        help='Warm Restarts new learning rate ratio compared to original lr')
    parser.add_argument('--repoly', type=float, default=1.5,
                        help='Warm Restart new poly exp')

    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Use Nvidia Apex AMP')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='parameter used by apex library')

    parser.add_argument('--sgd', action='store_true', default=True)
    parser.add_argument('--adam', action='store_true', default=False)
    parser.add_argument('--amsgrad', action='store_true', default=False)
    parser.add_argument('--freeze_trunk', action='store_true', default=False)

    parser.add_argument('--hardnm', default=0, type=int,
                        help='0 means no aug, 1 means hard negative mining iter 1,' +
                        '2 means hard negative mining iter 2')

    parser.add_argument('--trunk', type=str, default='resnet101',
                        help='trunk model, can be: resnet101 (default), resnet50')
    parser.add_argument('--max_epoch', type=int, default=180)
    parser.add_argument('--max_iter', type=int, default=30000)
    parser.add_argument('--max_cu_epoch', type=int, default=100000,
                        help='Class Uniform Max Epochs')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--crop_nopad', action='store_true', default=False)
    parser.add_argument('--rrotate', type=int,
                        default=0, help='degree of random roate')
    parser.add_argument('--color_aug', type=float,
                        default=0.0, help='level of color augmentation')
    parser.add_argument('--gblur', action='store_true', default=False,
                        help='Use Guassian Blur Augmentation')
    parser.add_argument('--bblur', action='store_true', default=False,
                        help='Use Bilateral Blur Augmentation')
    parser.add_argument('--lr_schedule', type=str, default='poly',
                        help='name of lr schedule: poly')
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='polynomial LR exponent')
    parser.add_argument('--bs_mult', type=int, default=2,
                        help='Batch size for training per gpu')
    parser.add_argument('--bs_mult_val', type=int, default=1,
                        help='Batch size for Validation per gpu')
    parser.add_argument('--crop_size', type=int, default=720,
                        help='training crop size')
    parser.add_argument('--pre_size', type=int, default=None,
                        help='resize image shorter edge to this before augmentation')
    parser.add_argument('--scale_min', type=float, default=0.5,
                        help='dynamically scale training images down to this size')
    parser.add_argument('--scale_max', type=float, default=2.0,
                        help='dynamically scale training images up to this size')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--snapshot', type=str, default=None)
    parser.add_argument('--restore_optimizer', action='store_true', default=False)

    parser.add_argument('--city_mode', type=str, default='train',
                        help='experiment directory date name')
    parser.add_argument('--date', type=str, default='default',
                        help='experiment directory date name')
    parser.add_argument('--exp', type=str, default='default',
                        help='experiment directory name')
    parser.add_argument('--tb_tag', type=str, default='',
                        help='add tag to tb dir')
    parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                        help='Save Checkpoint Point')
    parser.add_argument('--tb_path', type=str, default='logs/tb',
                        help='Save Tensorboard Path')
    parser.add_argument('--syncbn', action='store_true', default=True,
                        help='Use Synchronized BN')
    parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                        help='Dump Augmentated Images for sanity check')
    parser.add_argument('--test_mode', action='store_true', default=False,
                        help='Minimum testing to verify nothing failed, ' +
                        'Runs code for 1 epoch of train and val')
    parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                        help='Weight Scaling for the losses')
    parser.add_argument('--maxSkip', type=int, default=0,
                        help='Skip x number of  frames of video augmented dataset')
    parser.add_argument('--scf', action='store_true', default=False,
                        help='scale correction factor')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--backbone_lr', type=float, default=-1.0,
                        help='different learning rate on backbone network')

    parser.add_argument('--pooling', type=str, default='mean',
                        help='pooling methods, average is better than max')

    # Anomaly score mode - msp, max_logit, standardized_max_logit
    parser.add_argument('--score_mode', type=str, default='msp',
                        help='score mode for anomaly [msp, max_logit, standardized_max_logit]')

    # Boundary suppression configs
    parser.add_argument('--enable_boundary_suppression', type=bool, default=False,
                        help='enable boundary suppression')
    parser.add_argument('--boundary_width', type=int, default=0,
                        help='initial boundary suppression width')
    parser.add_argument('--boundary_iteration', type=int, default=0,
                        help='the number of boundary iterations')

    # Dilated smoothing configs
    parser.add_argument('--enable_dilated_smoothing', type=bool, default=False,
                        help='enable dilated smoothing')
    parser.add_argument('--smoothing_kernel_size', type=int, default=0,
                        help='kernel size of dilated smoothing')
    parser.add_argument('--smoothing_kernel_dilation', type=int, default=0,
                        help='kernel dilation rate of dilated smoothing')


    parser.add_argument('--skip_train', action='store_true', default=False)

    parser.add_argument('--T', type=float, default=0.1, 
                        help='temperature of log_softmax for language embedding')
    parser.add_argument('--tag', type=str, default='')      

    parser.add_argument('--context_optimize', type=str, default='none',
                        help="in ['none', 'coop', 'cocoop']")

    parser.add_argument('--manual_init', type=str, default='')

    parser.add_argument('--class_token_position', type=str, default='end')

    parser.add_argument('--n_ctx', default=4, type=int)

    parser.add_argument('--CSC', default=False, type=bool)

    parser.add_argument('--logit_type', type=str, default='others_logsm', help="['others_logsm', 'simple_prod']")

    parser.add_argument('--prompt_lr', type=float, default=-1.0)

    parser.add_argument('--pt_only', default=False, action="store_true")

    parser.add_argument('--pixelwise_prompt', default=False, action="store_true")

    parser.add_argument('--anoramly_co', type=float, default=1.0)

    parser.add_argument('--disable_le', default=False, action="store_true")

    parser.add_argument('--normalize', type=lambda x: x.lower() == "true", default=False)

    parser.add_argument('--orth_feat', default=False, action='store_true')

    parser.add_argument('--tau', default=1, type=float)

    parser.add_argument('--lang_aux', default=False, action="store_true")

    parser.add_argument('--normalize_feat', default=False, action='store_true')

    parser.add_argument('--warmup_iter', type=int, default=-1)

    parser.add_argument('--temp', type=str, default='fixed')

    parser.add_argument('--enable_main_out_temp', default=False, action='store_true')

    parser.add_argument('--stru', default=False, action='store_true')

    parser.add_argument('--freeze_backbone', default=False, action='store_true')

    parser.add_argument('--freeze_backbone_and_neck', default=False, action='store_true')

    parser.add_argument('--freeze_decoder', default=False, action='store_true')

    parser.add_argument('--set_seed', default=0, type=int)

    parser.add_argument('--post_prompt', default=False, action="store_true")

    parser.add_argument('--initial_size', default=1e-5, type=float)

    parser.add_argument('--force_init_decoder', default=False, action='store_true')

    return parser

def get_anormaly_parser():
    # Argument Parser
    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--arch', type=str, default='network.deepv3.DeepR101V3PlusD_OS8',
                        help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                        and deepWV3Plus (backbone: WideResNet38).')
    parser.add_argument('--dataset', type=str, default='cityscapes',
                        help='possible datasets for statistics; cityscapes')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Use Nvidia Apex AMP')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='parameter used by apex library')
    parser.add_argument('--trunk', type=str, default='resnet101',
                        help='trunk model, can be: resnet101 (default), resnet50')
    parser.add_argument('--bs_mult', type=int, default=2,
                        help='Batch size for training per gpu')
    parser.add_argument('--bs_mult_val', type=int, default=4,
                        help='Batch size for Validation per gpu')
    parser.add_argument('--class_uniform_pct', type=float, default=0,
                        help='What fraction of images is uniformly sampled')
    parser.add_argument('--class_uniform_tile', type=int, default=1024,
                        help='tile size for class uniform sampling')
    parser.add_argument('--batch_weighting', action='store_true', default=False,
                        help='Batch weighting for class (use nll class weighting using batch stats')
    parser.add_argument('--jointwtborder', action='store_true', default=False,
                        help='Enable boundary label relaxation')

    parser.add_argument('--snapshot', type=str, default='')
    parser.add_argument('--restore_optimizer', action='store_true', default=False)

    parser.add_argument('--date', type=str, default='default',
                        help='experiment directory date name')
    parser.add_argument('--exp', type=str, default='default',
                        help='experiment directory name')
    parser.add_argument('--tb_tag', type=str, default='',
                        help='add tag to tb dir')
    parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                        help='Save Checkpoint Point')
    parser.add_argument('--tb_path', type=str, default='logs/tb',
                        help='Save Tensorboard Path')
    parser.add_argument('--syncbn', action='store_true', default=True,
                        help='Use Synchronized BN')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--backbone_lr', type=float, default=0.0,
                        help='different learning rate on backbone network')
    parser.add_argument('--pooling', type=str, default='mean',
                        help='pooling methods, average is better than max')

    # parser.add_argument('--ood_dataset_path', type=str,
    #                     default='/home/nas1_userB/dataset/ood_segmentation/fishyscapes',
    #                     help='OoD dataset path')

    # Anomaly score mode - msp, max_logit, standardized_max_logit
    parser.add_argument('--score_mode', type=str, default='all_mix',
                        help='score mode for anomaly [msp, max_logit, standardized_max_logit]')

    # Boundary suppression configs
    parser.add_argument('--enable_boundary_suppression', type=lambda x: x.lower() == 'true', default=True,
                        help='enable boundary suppression')
    parser.add_argument('--boundary_width', type=int, default=4,
                        help='initial boundary suppression width')
    parser.add_argument('--boundary_iteration', type=int, default=4,
                        help='the number of boundary iterations')

    # Dilated smoothing configs
    parser.add_argument('--enable_dilated_smoothing', type=lambda x: x.lower() == 'true', default=True,
                        help='enable dilated smoothing')
    parser.add_argument('--smoothing_kernel_size', type=int, default=7,
                        help='kernel size of dilated smoothing')
    parser.add_argument('--smoothing_kernel_dilation', type=int, default=6,
                        help='kernel dilation rate of dilated smoothing')

    # FS LostAndFound data structure cannot be transformed to the desired structure
    # Therefore, when using it, extract images and masks and store then into lists,
    # and substitute images and masks in the code with those in the lists.
    parser.add_argument('--fs_lost_and_found', type=bool, default=True)

    
    ####### BEGIN Parameters to be adjusted
    parser.add_argument('--threshold', type=float, default=1e9,
                        help='threshold for anormaly')

    parser.add_argument('--th_type', type=str, default='tpr')

    parser.add_argument('--save_at', type=str,default='./results')

    parser.add_argument('--tag', type=str,default='')

    parser.add_argument('--ths', nargs='+', type=float)

    parser.add_argument('--curve_ckpt', action='store_true', default=False)

    parser.add_argument('--plot_only', action='store_true', default=False)

    parser.add_argument('--manual_init', type=str, default='')

    parser.add_argument('--class_token_position', type=str, default='end')

    parser.add_argument('--n_ctx', default=4, type=int)

    parser.add_argument('--CSC', default=False, type=bool)

    parser.add_argument('--logit_type', type=str, default='others_logsm', help="['others_logsm', 'simple_prod']")

    parser.add_argument('--prompt_lr', type=float, default=-1)

    parser.add_argument('--context_optimize', type=str, default='none',
                        help="in ['none', 'coop', 'cocoop']")

    parser.add_argument('--pt_only', default=False, action="store_true")

    parser.add_argument("--save_npz", default=False, action="store_true")

    parser.add_argument("--mode", type=str, default="file")

    parser.add_argument('--disable_le', default=False, action="store_true")

    parser.add_argument('--T', type=float, default=0.1, 
                        help='temperature of log_softmax for language embedding')

    parser.add_argument('--normalize', type=lambda x: x.lower() == "true", default=False)

    parser.add_argument('--orth_feat', default=False, action='store_true')

    parser.add_argument('--tau', default=1, type=float)

    parser.add_argument('--lang_aux', default=False, action="store_true")

    parser.add_argument('--normalize_feat', default=False, action='store_true')

    parser.add_argument('--temp', type=str, default='fixed')

    parser.add_argument('--top_k', type=int, default=2)

    parser.add_argument('--tag_suffix', default='', type=str)

    parser.add_argument('--enable_main_out_temp', action='store_true', default=False)

    parser.add_argument('--inf_temp', type=float, default=1)

    parser.add_argument('--stru', default=False, action='store_true')

    parser.add_argument('--post_prompt', default=False, action="store_true")

    parser.add_argument('--initial_size', default=1e-5, type=float)

    return parser


def init_nvidia(args):
    if 'WORLD_SIZE' in os.environ:
        # args.apex = int(os.environ['WORLD_SIZE']) > 1
        args.world_size = int(os.environ['WORLD_SIZE'])
        print("Total world size: ", int(os.environ['WORLD_SIZE']))

    torch.cuda.set_device(args.local_rank)
    print('My Rank:', args.local_rank)
    # Initialize distributed communication
    args.dist_url = args.dist_url + str(8000 + (int(time.time()%1000))//10)

    torch.distributed.init_process_group(backend='nccl',
                                        init_method=args.dist_url,
                                        world_size=1,
                                        rank=args.local_rank)

    
    return args