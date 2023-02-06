import sys

from utils import run


def main():
    print(f'>>> sys.version\n{sys.version}')
    print(f'>>> sys.argv\n{sys.argv}')
    run(['pwd'])
    run(['cat', '/proc/cpuinfo'])
    run(['free'])
    run(['nvidia-smi', '-L'])
    run(['nvidia-smi'])
    run(['singularity', '--version'])


if __name__ == '__main__':
    main()
