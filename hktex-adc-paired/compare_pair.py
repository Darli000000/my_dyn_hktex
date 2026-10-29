#!/usr/bin/env python3
"""Validate ADC execution and pairing; activity is not evidence of improved quality."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from adc_support import POLICY,scheduled_decisions


def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def compare(pair,until):
    if until not in (2000,5000):raise ValueError('Invalid stage endpoint')
    result=dict(until=until,functional_passed=True,quality_improvement_established=False,arms={})
    streams=[];identities=[];settings_all=[]
    actual_sha=hashlib.sha256((pair/'initial.pt').read_bytes()).hexdigest()
    stages=[('s02000',0,2000)]+([('s05000',2000,5000)] if until==5000 else [])
    for arm in ['linear','loggamma']:
        metrics=[];actions=[];decisions=[]
        for stage,begin,end in stages:
            d=pair/stage/arm;r=read(d/'report.json');s=r['settings'];meta=s['paired_loss'];settings_all.append(s)
            expected_status='complete' if end==5000 else 'stage_complete'
            if r['status']!=expected_status or r['start_update']!=begin or r['completed_updates']!=end:
                raise ValueError('Incomplete stage '+str(d))
            if meta['arm']!=arm or meta['initial_sha256']!=actual_sha or s['adc_full']!=POLICY:
                raise ValueError('Wrong arm/initialization/policy')
            identities.append((meta['initial_sha256'],meta['code_hash']))
            m=rows(d/'metrics.jsonl')
            if [x['update'] for x in m]!=list(range(begin+1,end+1)):raise ValueError('Missing updates')
            for x in m:
                if not POLICY['min_kernels']<=x['kernels']<=POLICY['max_kernels']:raise ValueError('Kernel budget')
                if not all(math.isfinite(v) for v in [x['loss_l1'],x['loss_objective'],x['batch_mse'],x['peak_cuda_gb'],*x['gradients'].values()]):
                    raise ValueError('Nonfinite metrics')
            if begin and not read(d/'resume_repair.json')['passed']:raise ValueError('Resume reference checks failed')
            metrics+=m;actions+=rows(d/'adc_actions.jsonl');decisions+=rows(d/'adc_decisions.jsonl')
        for x in actions:
            if x['status']!='complete' or not x['state_checks_passed']:raise ValueError('Failed mutation')
            if x['origin']!='natural':raise ValueError('Forced probes forbidden in full experiment')
            if x['operation'] not in ('clone','split','prune') or x['selected']<0:raise ValueError('Invalid operation')
            kind='prune' if x['operation']=='prune' else 'densify'
            if (x['update'],kind) not in scheduled_decisions(until):raise ValueError('Mutation outside schedule')
            delta=x['selected']*(-1 if x['operation']=='prune' else 1)
            if x['after']-x['before']!=delta:raise ValueError('Wrong structural count change')
        count=POLICY['initial_kernels'];by_update={}
        for x in actions:by_update.setdefault(x['update'],[]).append(x)
        for row in metrics:
            for action in by_update.get(row['update'],[]):
                if action['before']!=count:raise ValueError('Broken mutation count history')
                count=action['after']
            if row['kernels']!=count:raise ValueError('Metrics disagree with mutation history')
        if [(x['update'],x['kind']) for x in decisions]!=scheduled_decisions(until):
            raise ValueError('Natural controller decisions missing or mis-timed')
        for x in decisions:
            if x['kind']=='densify' and x['threshold']!=POLICY['error_threshold']:raise ValueError('Densification threshold changed')
            if x['kind']=='prune' and (x['min_kernels']!=POLICY['min_kernels'] or
                    x['selection_threshold']!=POLICY['selection_threshold'] or x['contrib_threshold']!=POLICY['contrib_threshold']):
                raise ValueError('Pruning policy changed')
        counts={k:sum(x['selected'] for x in actions if x['operation']==k) for k in ['clone','split','prune']}
        active=counts['clone']+counts['split']>0 and counts['prune']>0
        if not all(math.isfinite(v) for v in r['final_eval']['mse'].values()):raise ValueError('Nonfinite evaluation')
        result['arms'][arm]=dict(final_kernels=count,peak_torch_cuda_gb=max(x['peak_cuda_gb'] for x in metrics),
            natural_selected=counts,natural_growth_and_pruning_observed=active,decisions=decisions,
            final_eval=r['final_eval'],resume_passed=True if until==5000 else None,
            kernel_min=min(x['kernels'] for x in metrics),kernel_max=max(x['kernels'] for x in metrics))
        streams.append([(x['update'],x['batch_signature']) for x in metrics])
    if len(set(identities))!=1 or streams[0]!=streams[1]:raise ValueError('A/B initial state, code or input stream mismatch')
    for key in ['seed','steps','sources','batch_pixels','aa','transport','adc','eval_pixels','manifest_sha256','source_fingerprints']:
        if any(s[key]!=settings_all[0][key] for s in settings_all):raise ValueError('Unequal setting '+key)
    result['paired_inputs_passed']=True
    result['adc_activity_passed']=all(x['natural_growth_and_pruning_observed'] for x in result['arms'].values())
    result['status']='passed' if result['adc_activity_passed'] else 'functional_passed_natural_activity_inconclusive'
    result['next']='Inspect validation images and ADC decisions; activity alone does not establish quality improvement.'
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',required=True);p.add_argument('--until',type=int,choices=[2000,5000],required=True)
    p.add_argument('--require-active',action='store_true');a=p.parse_args();root=Path(a.pair)
    result=compare(root,a.until);out=root/f'comparison_{a.until:05d}.json'
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print('[COMPARISON]',out)
    if a.require_active and not result['adc_activity_passed']:
        raise SystemExit('Natural ADC activity is insufficient. Inspect comparison before changing policy in a NEW pair.')
