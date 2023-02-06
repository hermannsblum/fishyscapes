import json
import os
import sys

from utils import run


def main():
    print(json.loads(sys.argv[1])['event'])
    with open('settings.json', 'r') as f:
        settings = json.load(f)

    run(['mkdir', '-p', settings['tmp_pred_path']])
    run(['rm', '-rf', os.path.join(settings['tmp_pred_path'], '*')])
    cmd = [
        'singularity', 'run', '--nv', '--pwd', settings['run']['pwd'],
        '--bind', f"{settings['tmp_pred_path']}:{settings['run']['pred_path']},"
                  f"{settings['val_rgb_path']}:{settings['run']['rgb_path']},"
                  f"demo.py:/workspace/synboost/main.py",
        settings['sif_path']
    ]
    run(cmd)


if __name__ == '__main__':
    main()
