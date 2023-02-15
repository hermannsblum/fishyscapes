import json
import os
import sys
from PIL import Image
import wandb
import numpy as np

from utils import run, calculate_metrics_perpixAP, list_img_from_dir


def main():
    pr_id = json.loads(sys.argv[1])
    print(f'pr_id: {pr_id}', flush=True)
    dataset = 'lostandfound_fishyscapes'
    

    with open('settings.json', 'r') as f:
        settings = json.load(f)

    # make directories and copy data
    img_path = os.path.join(os.environ['TMPDIR'], 'inputs')
    run(['mkdir', img_path])
    run(['cp', '/cluster/work/riner/users/fishyscapes/lostandfound_fishyscapes/test_images.zip', os.environ['TMPDIR']])
    run(['unzip', os.path.join(os.environ['TMPDIR'], 'test_images.zip'), '-d', img_path])
    out_path = os.path.join(os.environ['TMPDIR'], 'outputs')
    run(['mkdir', out_path])
    simg_path = os.path.join(os.environ['TMPDIR'], 'image.simg')
    run(['cp', os.path.join('/cluster', 'scratch', 'blumh', f'fishyscapes_pr_{pr_id}'), simg_path])

    cmd = [
        'singularity', 'run', '--nv', '--pwd', settings['run']['pwd'],
        '--bind', f"{out_path}:{settings['run']['pred_path']},"
                  f"{img_path}:{settings['run']['rgb_path']}",
        simg_path
    ]
    try:
        run(cmd)
    except AssertionError:
        raise UserWarning("Execution of submitted container failed. Please take a look at the logs and resubmit a new container.")

    # get evaluation labels
    label_path = os.path.join(os.environ['TMPDIR'], 'labels')
    run(['mkdir', label_path])
    run(['cp', '/cluster/work/riner/users/fishyscapes/lostandfound_fishyscapes/test_labels.zip', os.environ['TMPDIR']])
    run(['unzip', os.path.join(os.environ['TMPDIR'], 'test_labels.zip'), '-d', label_path])

    # evaluate outputs
    labels = list_img_from_dir(label_path, '_labels.png')
    labels = [np.asarray(Image.open(p)) for p in labels]
    scores = list_img_from_dir(out_path, '.npy')
    scores = [np.load(p) for p in scores]

    ret = calculate_metrics_perpixAP(labels, scores)
    print(ret)
    wandb.init(project='fishyscapes', name=f"{pr_id}-{dataset}", config=dict(pr=pr_id, dataset=dataset))
    wandb.log(ret)
    wandb.finish()


if __name__ == '__main__':
    main()
