"""
Evaluation Scripts
"""
from __future__ import absolute_import
from __future__ import division
from collections import namedtuple, OrderedDict

from kornia import save_pointcloud_ply
from network import mynn
import argparse
import logging
import os
import torch
import re
import numpy as np

from config import cfg, assert_and_infer_cfg
import network
# from ood_metrics import fpr_at_95_tpr
from tqdm import tqdm

from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import torchvision.transforms as standard_transforms
import math
import sys
import matplotlib.pyplot as plt
import options

from IPython import embed

import shlex

def load_stats():
    class_mean = np.load(f'stats/sub1_mean.npy', allow_pickle=True)
    class_var = np.load(f'stats/sub1_var.npy', allow_pickle=True)
    print(class_mean)
    print(class_var)
    return class_mean, class_var

def preprocess_image(x, mean_std):
    x = Image.fromarray(x)
    x = standard_transforms.ToTensor()(x)
    x = standard_transforms.Normalize(*mean_std)(x)

    x = x.cuda()

    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    return x

def fpr_at_95_tpr(roc_tuple):
    fpr, tpr, _ = roc_tuple
    
    if all(tpr < 0.95):
        # No threshold allows TPR >= 0.95
        return 0
    elif all(tpr >= 0.95):    
        # All thresholds allow TPR >= 0.95, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tpr) if x>=0.95]
        return min(map(lambda idx: fpr[idx], idxs))
    else:
        # Linear interp between values to get FPR at TPR == 0.95
        return np.interp(0.95, tpr, fpr)

def get_and_save_roc_curve(args, preds, labels):
    _path = os.path.join(args.save_at, "roc.npz")
    if args.curve_ckpt:
        print("Load ROC from", _path)
        roc_tuple = np.load(_path)['curve']
    else:
        roc_tuple = roc_curve(preds, labels)
        if args.save_npz:
            np.savez(_path ,curve=roc_tuple, dtype=object)
    return roc_tuple


def get_and_save_prc_curve(args, preds, labels):
    _path = os.path.join(args.save_at, "prc.npz")
    if args.curve_ckpt:
        print("Load PRC from", _path)
        prc_tuple = np.load(_path, allow_pickle=True)['curve']
    else:
        prc_tuple = precision_recall_curve(preds, labels)
        if args.save_npz:
            np.savez(_path ,curve=prc_tuple, dtype=object)
    return prc_tuple

def metrics(args, anomaly_score_list, ood_gts_list):

    os.makedirs(args.save_at, exist_ok=True)

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    # drop void pixels
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = -1 * anomaly_scores[ood_mask]
    ind_out = -1 * anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    print('Measuring metrics...')

    fpr, tpr, ths = get_and_save_roc_curve(args, val_label, val_out)

    roc_auc = auc(fpr, tpr)
    precision, recall, prc_ths = get_and_save_prc_curve(args, val_label, val_out)
    prc_auc = average_precision_score(val_label, val_out)
    fpr_tpr95 = fpr_at_95_tpr((fpr, tpr, ths))
    
    print(f'AUROC score: {roc_auc}')
    print(f'AUPRC score: {prc_auc}')
    print(f'FPR@TPR95: {fpr_tpr95}')

    return fpr, tpr, ths 
    
def forgiving_state_restore(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    
    if list(net_state_dict.keys())[0].startswith("module."):
        filtered_state_dict = loaded_dict
    else:
        pattern = re.compile(r"^module\.")
        filtered_state_dict = {pattern.sub("", k): v for k, v in loaded_dict.items()} 
        

    for k in net_state_dict:
        if k in filtered_state_dict and net_state_dict[k].size() == filtered_state_dict[k].size():
            new_loaded_dict[k] = filtered_state_dict[k]
        else:
            print("Skipped loading parameter", k)
            # logging.info("Skipped loading parameter %s", k)
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net
    
def get_score(net, image):
    """
    Get Anormaly Score for Image
    :param image: numpy.ndarray It should be in the shape of (H, W, 3), entry in [0, 255]

    :return: torch.Tensor The anormaly_score
    """
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    with torch.no_grad():
        img_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75]
        image2 = preprocess_image(image, mean_std)
        anomaly_score_accu = torch.zeros((1, *image.shape[:2]), device="cuda")
        for r in img_ratios:
            image_in = torch.nn.functional.interpolate(
                image2, 
                scale_factor=r, 
                align_corners=True,
                mode='bilinear'
            )
            main_out, anomaly_score = net(image_in)
            del main_out
            anomaly_score_accu += torch.nn.functional.interpolate(
                anomaly_score.unsqueeze(1),
                size=anomaly_score_accu.shape[-2:],
                align_corners=True,
                mode="bilinear"
            ).squeeze(1)
            del anomaly_score
        anomaly_score_accu /= len(img_ratios)

    return anomaly_score_accu.cpu().squeeze()

def iter_over_FS_LAF(net):
    #############
    # FS LAF
    #############

    ds = np.load('/home/DISCOVER_summer2022/liumd/le106/_data.npz')
    image_list = ds['arr_0']
    mask_list = ds['arr_1']
    
    anomaly_score_list = []
    ood_gts_list = []

    # Iterate over all images
    data_len = len(image_list)
    # data_len = 5
    for i in tqdm(range(data_len), desc="Evaluating"):
        # get the ith mask from the mask list
        mask = mask_list[i]

        # get the ith image from the image list
        image = image_list[i]
        image = image.astype('uint8')

        anomaly_score = get_score(net, image)

        anomaly_score_list.append(anomaly_score.cpu().unsqueeze(0).numpy())
        ood_gts_list.append(np.expand_dims(mask, 0))

    return anomaly_score_list, ood_gts_list, None

def get_net():
    # Build the network

    cmd = "--exp r101_os8_base_60K --logit_type others_logsm --context_optimize none --n_ctx 0 --normalize True --T 0.07 --tau 0.8  --inf_temp 1.0 --enable_boundary_suppression False  --smoothing_kernel_dilation 4 "
    ckpt_path = "./net.pth"
    print("Loading Checkpoint ")
    sd = torch.load(ckpt_path)
    parser = options.get_anormaly_parser()
    args = parser.parse_args(shlex.split(cmd))

    print("Building the network")
    net = network.get_net(args, criterion=None, criterion_aux=None)

    print("Loading parameters")
    net = forgiving_state_restore(net, sd)
    
    class_mean, class_var = load_stats()
    class_mean, class_var = class_mean.item(), class_var.item()
    net.set_statistics(mean=class_mean, var=class_var)

    torch.cuda.empty_cache()
    net.eval()

    return net, args

if __name__ == "__main__":
    nw, ar = get_net()
    ar.tag = 'results_test_inf'
    ar.save_at = 'results_test_inf'
    ar.save_npz = False

    as_list, ood_list, _ = iter_over_FS_LAF(nw)
    fp, tp, th = metrics(ar, as_list, ood_list)
    th *= -1