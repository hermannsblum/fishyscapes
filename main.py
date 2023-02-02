import os
import subprocess
import sys


def run(cmd: list[str], cwd: str = None, env: dict[str, str] = None):
    print(f'>>> {cwd} $ {" ".join(cmd)}')
    p = subprocess.Popen(cmd, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=cwd, env=env, universal_newlines=True, bufsize=1)
    os.set_blocking(p.stdout.fileno(), False)
    os.set_blocking(p.stderr.fileno(), False)
    lines = []
    lines_out = []
    lines_err = []
    while True:
        rc = p.poll()
        while line_out := p.stdout.readline():
            lines.append(line_out)
            lines_out.append(line_out)
            print(line_out, end='')
        while line_err := p.stderr.readline():
            lines.append(line_err)
            lines_err.append(line_err)
            print(line_err, end='')
        if rc is not None:
            break
    return rc, ''.join(lines), ''.join(lines_out), ''.join(lines_err)


def main():
    print(sys.version)
    run(['mkdir', '-p', '/tmp/results'])
    run(['rm', '-rf', '/tmp/results/*'])
    cmd = [
        'singularity', 'run', '--nv', '--pwd', '/workspace/synboost',
        '--bind', '/tmp/results:/workspace/synboost/results',
        'synboost_1.0.sif',
    ]
    rc, *_ = run(cmd, '/submitted_containers')
    print(rc)
    pass


if __name__ == '__main__':
    main()
