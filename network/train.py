"""
training code
"""
from __future__ import absolute_import
from __future__ import division
import argparse
from email.policy import default
import imp
import logging
import os
from matplotlib.pyplot import get
import torch

from config import cfg, assert_and_infer_cfg
from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist
import datasets
import loss
import network
import optimizer_2
import time
import torchvision.utils as vutils
import torch.nn.functional as F
from network.mynn import freeze_weights, unfreeze_weights
import numpy as np
import random

from IPython import embed

from labels import get_label_matrix

import options

# Argument Parser

parser = options.get_train_parser()
args = parser.parse_args()
args.ckpt = f"c_{args.tag}"
args.tb_path = f"c_{args.tag}"


# Enable CUDNN Benchmarking optimization
#torch.backends.cudnn.benchmark = True
random_seed = cfg.RANDOM_SEED  #304
print("RANDOM_SEED", random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

args.world_size = 1

# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2


if 'WORLD_SIZE' in os.environ:
    # args.apex = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

# args.world_size = 3
torch.cuda.set_device(args.local_rank)
print('My Rank:', args.local_rank)
# Initialize distributed communication
args.dist_url = args.dist_url + str(8000 + (int(time.time()%1000))//10)

torch.distributed.init_process_group(backend='nccl',
                                     init_method=args.dist_url,
                                     world_size=args.world_size,
                                     rank=args.local_rank)

def main():

    """
    Main Function
    """
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    writer = prep_experiment(args, parser)

    train_loader, val_loaders, train_obj, extra_val_loaders = datasets.setup_loaders(args)

    # embed()

    criterion, criterion_val = loss.get_loss(args)
    criterion_aux = loss.get_loss_aux(args)
    net = network.get_net(args, criterion, criterion_aux)

    optim, scheduler = optimizer_2.get_optimizer(args, net)

    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = network.warp_network_in_dataparallel(net, args.local_rank)
    epoch = 0
    i = 0

    if args.snapshot:
        epoch, mean_iu, _ = optimizer_2.load_weights(net, optim, scheduler,
                            args.snapshot, args.restore_optimizer)
        epoch += 1
        if args.restore_optimizer is True:
            iter_per_epoch = len(train_loader)
            i = iter_per_epoch * epoch
        else:
            epoch = 0

    print("#### iteration", i)
    torch.cuda.empty_cache()
    # Main Loop
    # for epoch in range(args.start_epoch, args.max_epoch):
    if not args.skip_train:
        # from IPython import embed
        # embed()
        while i < args.max_iter:
            # Update EPOCH CTR
            cfg.immutable(False)
            cfg.ITER = i
            cfg.immutable(True)

            i = train(train_loader, net, optim, epoch, writer, scheduler, args.max_iter)
            train_loader.sampler.set_epoch(epoch + 1)
            if i % args.val_interval == 0:
                for dataset, val_loader in val_loaders.items():
                    validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i)
            else:
                if args.local_rank == 0:
                    print("Saving pth file...")
                    evaluate_eval(args, net, optim, scheduler, None, None, [],
                                writer, epoch, "None", None, i, save_pth=True)

            if args.class_uniform_pct:
                if epoch >= args.max_cu_epoch:
                    train_obj.build_epoch(cut=True)
                    # if args.apex:
                    train_loader.sampler.set_num_samples()
                else:
                    train_obj.build_epoch()
            epoch += 1

    # Validation after epochs

    assert len(val_loaders) == 1

    for dataset, val_loader in val_loaders.items():
        _, best_snaphot = validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i)
    # best_snaphot = "/DATA2/gaoha/liumd/le/c_log_softmax_with_others/ckpt/0630_2359/r101_os8_base_60K/07_03_22/best_cityscapes_epoch_162_mean-iu_0.79037.pth"
    
    #Evaluate Anormaly Detection Performance
    if args.local_rank == 0:
        print("best_snapshot is", best_snaphot)
        with open("./scripts/calc_and_inf_r101_os8.sh", "r") as f:
            sh = f.read() \
                .replace("$1", f"'{best_snaphot}'") \
                .replace('$2', f"'{args.tag}'") \
                .replace("$3", "'0.80 0.90 0.95'")
        with open("./scripts/check.sh", "w") as f:
            f.write(sh)
        print("checker modified")

    # for dataset, val_loader in extra_val_loaders.items():
    #     print("Extra validating... This won't save pth file")
    #     validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, save_pth=False)
    



def train(train_loader, net, optim, curr_epoch, writer, scheduler, max_iter):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return:
    """
    net.train()

    train_total_loss = AverageMeter()
    time_meter = AverageMeter()

    curr_iter = curr_epoch * len(train_loader)
    print("current iteration is", curr_iter, " | current epoch is", curr_epoch)

    # # load clip model
    # embeds = get_label_matrix()

    for i, data in enumerate(train_loader):
        if curr_iter >= max_iter:
            break
        inputs, seg_gts, ood_gts, _, aux_gts = data

        B, C, H, W = inputs.shape
        num_domains = 1
        inputs = [inputs]
        seg_gts = [seg_gts]
        ood_gts = [ood_gts]
        aux_gts = [aux_gts]

        batch_pixel_size = C * H * W

        for di, ingredients in enumerate(zip(inputs, seg_gts, ood_gts, aux_gts)):
            input, seg_gt, ood_gt, aux_gt = ingredients

            start_ts = time.time()

            img_gt = None

            input, seg_gt, ood_gt = input.cuda(), seg_gt.cuda(), ood_gt.cuda()

            optim.zero_grad()

            outputs_index = 0
            # print(net.module.T.item())
            from IPython import embed
            # embed()
            outputs = net(input, seg_gts=seg_gt, ood_gts=ood_gt, aux_gts=aux_gt)
            main_loss, aux_loss, anomaly_score = outputs
            total_loss = main_loss + (0.4 * aux_loss)

            log_total_loss = total_loss.clone().detach_()
            torch.distributed.all_reduce(log_total_loss, torch.distributed.ReduceOp.SUM)
            log_total_loss = log_total_loss / args.world_size
            train_total_loss.update(log_total_loss.item(), batch_pixel_size)

            total_loss.backward()
            optim.step()

            time_meter.update(time.time() - start_ts)

            del total_loss, log_total_loss

            if args.local_rank == 0:
                if i % 20 == 19:
                    from IPython import embed
                    # embed()
                    if type(net.module.T) == float or type(net.module.T) == int:
                        T = net.module.T
                    else:
                        T = net.module.T.item()

                    lrstrs = [
                        "PG(Len:{}, lr:{:.6e})".format(
                            len(p['params']),
                            p["lr"]
                        )
                        for p in optim.param_groups
                    ]
                    # embed()
                    msg = '[epoch {}], [iter {} / {} : {}], [total loss {:0.6f}], [seg loss {:0.6f}], [T {:0.6f}], [lr {}], [time {:0.4f}]'.format(
                        curr_epoch, i + 1, len(train_loader), curr_iter, train_total_loss.avg,
                        main_loss.item(), T, '|'.join(lrstrs),
                        time_meter.avg / args.train_batch_size)

                    logging.info(msg)

                    # Log tensorboard metrics for each iteration of the training phase
                    writer.add_scalar('loss/train_loss', (train_total_loss.avg),
                                    curr_iter)
                    writer.add_scalar('loss/main_loss', (main_loss.item()),
                                    curr_iter)
                    writer.add_scalar('T', (T),
                                    curr_iter)
                    train_total_loss.reset()
                    time_meter.reset()

        curr_iter += 1
        scheduler.step()

        if i > 5 and args.test_mode:
            return curr_iter

    return curr_iter

def validate(val_loader, dataset, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, save_pth=True):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    dataset: dataset name (str)
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    net.eval()
    val_loss = AverageMeter()
    iou_acc = 0
    error_acc = 0
    dump_images = []

    # embeds = get_label_matrix()
  
    for val_idx, data in enumerate(val_loader):
        # input        = torch.Size([1, 3, 713, 713])
        # gt_image           = torch.Size([1, 713, 713])
        inputs, seg_gts, ood_gts, img_names, _ = data

        assert len(inputs.size()) == 4 and len(seg_gts.size()) == 3
        assert inputs.size()[2:] == seg_gts.size()[1:]

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)

        inputs = inputs.cuda()
        seg_gts_cuda = seg_gts.cuda()

        with torch.no_grad():
            main_out, anomaly_score = net(inputs)

        del inputs

        assert main_out.size()[2:] == seg_gts.size()[1:]
        assert main_out.size()[1] == datasets.num_classes

        main_loss = criterion(main_out, seg_gts_cuda)


        val_loss.update(main_loss.item(), batch_pixel_size)

        del seg_gts_cuda

        # Collect data from different GPU to a single GPU since
        # encoding.parallel.criterionparallel function calculates distributed loss
        # functions
        predictions = main_out.data.max(1)[1].cpu()

        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
        if val_idx > 10 and args.test_mode:
            break

        # Image Dumps
        if val_idx < 10:
            dump_images.append([seg_gts, predictions, img_names])

        iou_acc += fast_hist(predictions.numpy().flatten(), seg_gts.numpy().flatten(),
                             datasets.num_classes)
        del main_out, val_idx, data

    iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
    torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
    iou_acc = iou_acc_tensor.cpu().numpy()

    best_path = ''

    if args.local_rank == 0:
        best_path = evaluate_eval(args, net, optim, scheduler, val_loss, iou_acc, dump_images,
                    writer, curr_epoch, dataset, None, curr_iter, save_pth=save_pth)

    return val_loss.avg, best_path

if __name__ == '__main__':
    main()

