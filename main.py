import json
import os

from utils import run


def main():
    with open('settings.json', 'r') as f:
        settings = json.load(f)

    run(['mkdir', '-p', settings['tmp_results_path']])
    run(['rm', '-rf', os.path.join(settings['tmp_results_path'], '*')])
    cmd = [
        'singularity', 'run', '--nv', '--pwd', settings['run']['pwd'],
        '--bind', f"{settings['tmp_results_path']}:{settings['run']['results_path']}",
        settings['sif_path'],
    ]
    run(cmd, '/submitted_containers')


if __name__ == '__main__':
    main()
