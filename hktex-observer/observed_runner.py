"""Compose the verified V4 runner with process-local recording hooks."""
import argparse
import os
from pathlib import Path
import sys


def main():
    if '--version' in sys.argv:
        print('HKTex observed runner 1.0.0')
        return
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--record-every', type=int, default=250)
    p.add_argument('--eval-points', type=int, default=8192)
    opts, rest = p.parse_known_args()
    if opts.record_every < 1 or opts.eval_points < 64:
        raise SystemExit('record-every >= 1 and eval-points >= 64 required')
    sys.argv = [sys.argv[0], *rest]
    import checked_runner_v4
    args, _ = checked_runner_v4.parse_args()
    if args.check_cli:
        checked_runner_v4.main()
        return
    sys.path.insert(0, args.repo)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.gpu)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    from recording import install
    with install(args.repo, opts.record_every, opts.eval_points):
        checked_runner_v4.main()


if __name__ == '__main__':
    main()
