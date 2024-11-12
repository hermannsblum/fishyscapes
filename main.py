import json
import os
import sys

from utils import run


def main():
    pr_id = int(sys.argv[1])
    print(f'pr_id: {pr_id}')

    with open('settings.json', 'r') as f:
        settings = json.load(f)
    with open('validation_performance.json', 'r') as f:
        settings.update(json.load(f))

    if settings.get('download_url'):
        # download image from set url instead of upload form
        run(['wget', settings['download_url'], '-P', '/tmp', '-o', '/tmp/wget_output.log'])

    downloaded_file_path = '/tmp/cad_4.0.simg'

    if not os.path.exists(downloaded_file_path):
        raise UserWarning("Container file not found. Please check the download URL and try again.")

    run(['mkdir', '-p', settings['tmp_pred_path']])
    run(['chmod', '777', settings['tmp_pred_path']])
    run(' '.join(['rm', '-rf', os.path.join(settings['tmp_pred_path'], '*')]), shell=True)
    cmd = [
        'CUDA_LAUNCH_BLOCKING=1', 'sudo', 'singularity', 'run', '--nv',
        '--bind', f"{settings['tmp_pred_path']}:/output,"
                  f"{settings['val_rgb_path']}:/input",
        downloaded_file_path
    ]
    try:
        run(cmd)
    except AssertionError:
        raise UserWarning("Execution of submitted container failed. Please take a look at the logs and resubmit a new container.")
        
    run(['ls', settings['tmp_pred_path']])


if __name__ == '__main__':
    main()

