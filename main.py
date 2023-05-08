import json
import os
import sys

from utils import run


def main():
    pr_id = json.loads(sys.argv[1])['event']['number']
    print(f'pr_id: {pr_id}')

    with open('settings.json', 'r') as f:
        settings = json.load(f)
    with open('validation_performance.json', 'r') as f:
        settings.update(json.load(f))

    if settings.get('download_url'):
        # download image from set url instead of upload form
        run(['wget', settings['download_url'], '-O', f'/submissions/fishyscapes_pr_{pr_id}', '-o', '/tmp/wget_output.log'])

    try:
        run(['cp', os.path.join('/submissions', f'fishyscapes_pr_{pr_id}'), os.path.join('/tmp', f'fishyscapes_pr_{pr_id}.simg')])
    except AssertionError:
        raise UserWarning("Failed to copy singularity container. Have you uploaded a container following the website instructions?")

    run(['mkdir', '-p', settings['tmp_pred_path']])
    run(['chmod', '777', settings['tmp_pred_path']])
    run(' '.join(['rm', '-rf', os.path.join(settings['tmp_pred_path'], '*')]), shell=True)
    cmd = [
        'singularity', 'exec', '--nv', '--no-privs',
        '--bind', f"{settings['tmp_pred_path']}:/output,"
                  f"{settings['val_rgb_path']}:/input",
        os.path.join('/tmp', f'fishyscapes_pr_{pr_id}.simg'),
        'bash -c "whoami && ls -al /home/user && groups"'
    ]
    run(['runuser', '-l', 'blumh', '-c', ' '.join(cmd)])
    cmd = [
        'singularity', 'run', '--nv', '-u',
        '--bind', f"{settings['tmp_pred_path']}:/output,"
                  f"{settings['val_rgb_path']}:/input",
        os.path.join('/tmp', f'fishyscapes_pr_{pr_id}.simg')
    ]
    try:
        run(['runuser', '-l', 'blumh', '-c', ' '.join(cmd)])
    except AssertionError:
        raise UserWarning("Execution of submitted container failed. Please take a look at the logs and resubmit a new container.")
        
    run(['ls', settings['tmp_pred_path']])


if __name__ == '__main__':
    main()
