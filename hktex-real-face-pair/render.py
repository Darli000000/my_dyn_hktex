#!/usr/bin/env python3
"""Render final/all saved exact checkpoints, including ADC before/after states."""
import argparse
import json
from pathlib import Path
import sys
import traceback
from common import write_json,sha
from experiment import environment,context,cached_views,make
from engine import restore
from runtime import predict,metrics
from previews import preview

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair',required=True);p.add_argument('--arm',choices=['baseline','adc'],required=True)
    p.add_argument('--which',choices=['final','all'],default='final');p.add_argument('--output',required=True)
    p.add_argument('--repo',default=str(Path(__file__).resolve().parent.parent));p.add_argument('--gpu',default='0')
    a=p.parse_args();out=Path(a.output).resolve()
    if out.exists():p.error('Use a new render output directory')
    out.mkdir(parents=True);report=dict(status='running',args=vars(a),frames=[])
    try:
        report['environment']=environment(a)
        import torch
        pair,c,dataset,m,g=context(a);views=cached_views(pair,m);arm=pair/a.arm
        paths=[arm/f'step_{c["steps"]:05d}'/'checkpoint.json'] if a.which=='final' else list(arm.glob('step_*/checkpoint.json'))+list(arm.glob('adc_snapshots/*/checkpoint.json'))
        paths.sort(key=lambda p:(json.loads(p.read_text())['step'],{'before_adc':0,'after_adc':1,'evaluation':2}[json.loads(p.read_text())['phase']]))
        if not paths:raise ValueError('No checkpoints')
        model=None;loaded_n=None;model_number=0
        for path in paths:
            meta=json.loads(path.read_text());state_file=path.parent/'state.pt'
            if meta['state_sha256']!=sha(state_file):raise ValueError('Checkpoint was modified')
            if meta['manifest_sha256']!=sha(dataset/'manifest.json'):raise ValueError('Checkpoint/dataset mismatch')
            state=torch.load(state_file,map_location='cpu',weights_only=True)
            if state['kernels']!=loaded_n:
                if model is not None:del model;torch.cuda.empty_cache()
                construction=out/f'_model_{model_number:03d}';construction.mkdir();model_number+=1
                model,_,_=make(a,pair,c,g,construction,state_file);loaded_n=state['kernels']
            else:restore(model,state)
            name=str(path.parent.relative_to(arm)).replace('/','_');folder=out/name;folder.mkdir()
            record=dict(checkpoint=str(path.parent),step=meta['step'],phase=meta['phase'],kernels=loaded_n,views={})
            for v in views:
                if v['row']['camera_index'] in ([0,7,15] if meta['step']==c['steps'] else [0,7]):
                    pred=predict(model,v['points'],v['faces']);preview(v,pred,folder)
                    record['views'][v['row']['id']]=metrics(pred,v['rgb'],v['regions'])
            write_json(folder/'metrics.json',record);report['frames'].append(record);write_json(out/'report.json',report)
            print('[RENDERED]',name,flush=True)
        report['status']='completed';write_json(out/'report.json',report)
    except BaseException as e:
        report.update(status='failed',error=str(e),traceback=traceback.format_exc());write_json(out/'report.json',report);raise
if __name__=='__main__':main()
