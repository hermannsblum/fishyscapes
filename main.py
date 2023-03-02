import json
import os
import sys

from utils import run


def main():
    pr_id = json.loads(sys.argv[1])['event']['number']
    print(f'pr_id: {pr_id}')

    with open('settings.json', 'r') as f:
        settings = json.load(f)

    try:
        run(['cp', os.path.join('/submissions', f'fishyscapes_pr_{pr_id}'), os.path.join('/tmp', f'fishyscapes_pr_{pr_id}.simg')])
    except AssertionError:
        raise UserWarning("Failed to copy singularity container. Have you uploaded a container following the website instructions?")

    run(['mkdir', '-p', settings['tmp_pred_path']])
    run(['rm', '-rf', os.path.join(settings['tmp_pred_path'], '*')])
    run(['ls', '-al', os.path.join(settings['tmp_pred_path'])])
    exit(0)
    cmd = [
        'singularity', 'run', '--nv',
        '--bind', f"{settings['tmp_pred_path']}:{settings['run']['pred_path']},"
                  f"{settings['val_rgb_path']}:{settings['run']['rgb_path']}",
        os.path.join('/tmp', f'fishyscapes_pr_{pr_id}.simg')
    ]
    try:
        run(cmd)
    except AssertionError:
        raise UserWarning("Execution of submitted container failed. Please take a look at the logs and resubmit a new container.")


if __name__ == '__main__':
    main()
