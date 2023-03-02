import os
import subprocess


def run(cmd, cwd=None, env=None, shell=False):
    print(f'>>> {cwd} $ {" ".join(cmd)}')
    p = subprocess.Popen(cmd, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         shell=shell, cwd=cwd, env=env, universal_newlines=True, bufsize=1)
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
    print('EXIT CODE:', rc)
    assert rc == 0
    return rc, ''.join(lines), ''.join(lines_out), ''.join(lines_err)
