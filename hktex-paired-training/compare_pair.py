#!/usr/bin/env python3
"""Check pairing and summarize common metrics. No statistical winner declaration."""
import argparse
import hashlib
import json
from pathlib import Path


def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def compare(pair,stage):
    runs=[pair/stage/a for a in ['linear','loggamma']]
    reports=[read(r/'report.json') for r in runs];metrics=[rows(r/'metrics.jsonl') for r in runs]
    end=int(stage[1:])
    if any(r['status'] not in ['complete','stage_complete'] or r['completed_updates']!=end for r in reports):
        raise ValueError('Both arms must finish this stage before proceeding')
    settings=[r['settings'] for r in reports];meta=[s['paired_loss'] for s in settings]
    if [m['arm'] for m in meta]!=['linear','loggamma']:raise ValueError('Wrong arm assignment')
    if meta[0]['initial_sha256']!=meta[1]['initial_sha256'] or meta[0]['code_hash']!=meta[1]['code_hash']:
        raise ValueError('Initial state or launcher code differs')
    h=hashlib.sha256((pair/'initial.pt').read_bytes()).hexdigest()
    if h!=meta[0]['initial_sha256']:raise ValueError('Shared initial state changed')
    for key in ['seed','steps','batch_pixels','aa','transport','sources','adc','manifest_sha256','source_fingerprints']:
        if settings[0][key]!=settings[1][key]:raise ValueError('Unequal setting '+key)
    if settings[0]['adc'] or settings[0]['sources']!=5000 or settings[0]['steps']!=5000:
        raise ValueError('Expected fixed 5000 kernels and 5000-step scheduler')
    signatures=[[(r['update'],r['batch_signature']) for r in m] for m in metrics]
    if signatures[0]!=signatures[1]:raise ValueError('Pixel/view/render-seed stream differs')
    start=reports[0]['start_update']
    if reports[1]['start_update']!=start or [r['update'] for r in metrics[0]]!=list(range(start+1,end+1)):
        raise ValueError('Missing or duplicate updates')
    if any(r['kernels']!=5000 for m in metrics for r in m):raise ValueError('Kernel count changed')
    result=dict(pairing_passed=True,stage=stage,start=start,end=end,matched_updates=len(metrics[0]),
        initial_sha256=h,scope='Common linear metrics; transformed objectives are not comparable across arms',arms={})
    for arm,run,report,m in zip(['linear','loggamma'],runs,reports,metrics):
        tail=m[-min(100,len(m)):]
        field=rows(run/'field_metrics.jsonl')[-1]
        if field['update']!=end:raise ValueError('Surface probe is not from the stage endpoint')
        result['arms'][arm]=dict(eval_mse=report['final_eval']['mse'],
            last_100_or_fewer_linear_l1=sum(r['loss_l1'] for r in tail)/len(tail),
            last_100_or_fewer_batch_mse=sum(r['batch_mse'] for r in tail)/len(tail),
            field=field)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pair',required=True);p.add_argument('--stage',required=True)
    a=p.parse_args();pair=Path(a.pair);result=compare(pair,a.stage)
    out=pair/a.stage/'comparison.json';out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));print('[COMPARISON]',out)


if __name__=='__main__':main()
