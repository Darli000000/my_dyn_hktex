"""Tiny CPU integration test for common-target checks and weighted metrics."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from PIL import Image


def main():
    with tempfile.TemporaryDirectory(prefix='hktex-route-comparison-test-') as tmp:
        root=Path(tmp)
        for label,mse in [('uv',.01),('ray',.02),('image',.03)]:
            d=root/label; d.mkdir(); (d/'final').mkdir();run=root/(label+'-run');run.mkdir()
            protocol=dict(version=1,dataset_manifest_sha256='fixture',views=['view_000','view_002'],
                size=8,spp=4,tile_size=8,seed=21931,point_batch=512,precision='highest',allow_tf32=False,
                denoise=False,metric='fixture-mse',renderer='fixture-renderer',gt='fixture-gt',
                run=str(run),field_adapter='fixture-identity')
            (d/'common_protocol.json').write_text(json.dumps(protocol))
            (d/'completed.json').write_text(json.dumps([dict(directory='final')]))
            rows=[]
            for i in range(2):
                np.savez_compressed(d/'final'/f'gt_{i:03d}.npz',linear_rgb=np.zeros((8,8,3),np.float32))
                Image.new('RGB',(8,8),'black').save(d/'final'/f'gt_{i:03d}_raw.png')
                Image.new('RGB',(8,8),'gray').save(d/'final'/f'prediction_{i:03d}_raw.png')
                rows.append(dict(view=protocol['views'][i],raw_mse=mse*(i+1),valid_pixels=10*(i+1)))
            (d/'final'/'image_errors.json').write_text(json.dumps(rows))
        command=[sys.executable,str(Path(__file__).with_name('compare.py')),'--evaluations',
                 *[f'{label}={root/label}' for label in ['uv','ray','image']],'--output',str(root/'comparison')]
        subprocess.run(command,check=True,capture_output=True,text=True)
        result=json.loads((root/'comparison/comparison.json').read_text())
        assert abs(result['results'][0]['foreground_mse']-(.01*10+.02*20)/30)<1e-12
        assert abs(result['results'][0]['mean_view_mse']-.015)<1e-12
        with Image.open(root/'comparison/comparison_raw.png') as im: assert im.size==(32,96)
        protocol=json.loads((root/'ray/common_protocol.json').read_text());protocol['seed']+=1
        (root/'ray/common_protocol.json').write_text(json.dumps(protocol))
        command[-1]=str(root/'must-fail')
        failed=subprocess.run(command,capture_output=True,text=True)
        assert failed.returncode!=0 and 'Evaluation settings differ' in failed.stderr
        protocol['seed']-=1
        (root/'ray/common_protocol.json').write_text(json.dumps(protocol))
        np.savez_compressed(root/'ray/final/gt_000.npz',linear_rgb=np.ones((8,8,3),np.float32))
        command[-1]=str(root/'different-gt-must-fail')
        failed=subprocess.run(command,capture_output=True,text=True)
        assert failed.returncode!=0 and 'AssertionError' in failed.stderr
    print('PASS: weighted MSE, per-view MSE, contact sheet, mismatched settings/GT rejection')


if __name__=='__main__':main()
