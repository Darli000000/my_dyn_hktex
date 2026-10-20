"""Extract one frame from the newest completed comparison video; no retraining."""
import argparse
from pathlib import Path
import imageio.v2 as imageio

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', default='apple-baseline')
parser.add_argument('--repo', default='/root/hktex-work/hktex')
args = parser.parse_args()
root = Path(args.repo) / 'outputs' / args.experiment
videos = list(root.glob('*/ckpts/gt_vs_out.mp4'))
if not videos:
    raise SystemExit(f'No saved comparison video found below {root}')
video = max(videos, key=lambda p: p.stat().st_mtime)
target = video.with_name('preview_first_frame.png')
with imageio.get_reader(str(video)) as reader:
    imageio.imwrite(str(target), reader.get_data(0))
print('Comparison video:', video)
print('Single-frame preview:', target)
print('Left to right: GT | HKTex output | kernel rings (when available).')
