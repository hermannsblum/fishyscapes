import json
import os
import sys
from PIL import Image
import wandb
import numpy as np
import time
from distutils import spawn

from utils import run, calculate_metrics_perpixAP, list_img_from_dir, MeanIoU


def main():
    pr_id = json.loads(sys.argv[1])
    print(f'pr_id: {pr_id}', flush=True)
    dataset = sys.argv[2]
    print(f'{dataset=}', flush=True)
    assert dataset in ['cityscapes_validation']
    

    with open('settings.json', 'r') as f:
        settings = json.load(f)

    # make directories and copy data
    img_path = os.path.join(os.environ['TMPDIR'], 'inputs')
    run(['mkdir', img_path])
    run(['cp', f'/cluster/work/riner/users/fishyscapes/{dataset}/test_images.zip', os.environ['TMPDIR']])
    run(['unzip', os.path.join(os.environ['TMPDIR'], 'test_images.zip'), '-d', img_path])
    out_path = os.path.join(os.environ['TMPDIR'], 'outputs')
    run(['mkdir', out_path])
    simg_path = os.path.join(os.environ['TMPDIR'], 'image.simg')
    run(['cp', os.path.join('/cluster', 'scratch', 'blumh', f'fishyscapes_pr_{pr_id}'), simg_path])

    cmd = [
        'singularity', 'run', '--nv', '--writable-tmpfs',
        '--bind', f"{out_path}:/output,"
                  f"{img_path}:/input",
        simg_path
    ]
    try:
        start = time.time()
        run(cmd)
        end = time.time()
    except AssertionError:
        raise UserWarning("Execution of submitted container failed. Please take a look at the logs and resubmit a new container.")

    # get evaluation labels
    label_path = os.path.join(os.environ['TMPDIR'], 'labels')
    run(['mkdir', label_path])
    run(['cp', f'/cluster/work/riner/users/fishyscapes/{dataset}/test_labels.zip', os.environ['TMPDIR']])
    run(['unzip', os.path.join(os.environ['TMPDIR'], 'test_labels.zip'), '-d', label_path])

    # evaluate outputs
    labels = list_img_from_dir(label_path, '_labels.png')
    preds = list_img_from_dir(out_path, '_segmentation.npy')
    label_mapping = np.array([-1, -1, -1, -1, -1, -1, -1, 0, 1, -1, -1, 2, 3, 4, -1, -1, -1, 5, -1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1, 16, 17, 18])
    miou = MeanIoU(num_labels=19, ignore_index=-1)

    for i in range(len(labels)):
        label = label_mapping[np.asarray(Image.open(labels[i]))]
        pred = np.load(preds[i])
        print(f"{label.shape=}, {pred.shape=}")
        miou.update(pred, label)
    ret = {'miou': miou.compute() }
    ret['inference_time'] = end - start
    print(ret, flush=True)
    wandb.init(project='fishyscapes', 
               name=f"{pr_id}-{dataset}",
               #mode='offline',
               config=dict(pr=pr_id, dataset=dataset))
    wandb.log(ret)
    wandb.finish()


if __name__ == '__main__':
    main()
