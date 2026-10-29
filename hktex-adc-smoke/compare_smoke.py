#!/usr/bin/env python3
import argparse,hashlib,json,math
from pathlib import Path
from adc_support import POLICY


def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def compare(pair):
    result=dict(functional_passed=True,quality_comparison=False,arms={})
    streams=[];identities=[];settings_all=[]
    actual_sha=hashlib.sha256((pair/'initial.pt').read_bytes()).hexdigest()
    for arm in ['linear','loggamma']:
        metrics=[];actions=[];decisions=[]
        for stage,begin,end in [('s00250',0,250),('s00450',250,450)]:
            d=pair/stage/arm;r=read(d/'report.json');s=r['settings'];meta=s['paired_loss'];settings_all.append(s)
            if r['status'] not in ['complete','stage_complete'] or r['start_update']!=begin or r['completed_updates']!=end:
                raise ValueError('Incomplete stage '+str(d))
            if meta['arm']!=arm or meta['initial_sha256']!=actual_sha or s['adc_smoke']!=POLICY:
                raise ValueError('Wrong arm/initialization/policy')
            identities.append((meta['initial_sha256'],meta['code_hash']))
            m=rows(d/'metrics.jsonl')
            if [x['update'] for x in m]!=list(range(begin+1,end+1)):raise ValueError('Missing updates')
            for x in m:
                if not POLICY['min_kernels']<=x['kernels']<=POLICY['max_kernels']:raise ValueError('Kernel budget')
                if not all(math.isfinite(v) for v in [x['loss_l1'],x['loss_objective'],x['batch_mse'],*x['gradients'].values()]):
                    raise ValueError('Nonfinite metrics')
            if begin and not read(d/'resume_repair.json')['passed']:raise ValueError('Resume reference checks failed')
            metrics+=m;actions+=rows(d/'adc_actions.jsonl');decisions+=rows(d/'adc_decisions.jsonl')
        if any(x['status']!='complete' or not x['state_checks_passed'] for x in actions):raise ValueError('Failed mutation')
        for x in actions:
            delta=x['selected']*(-1 if x['operation']=='prune' else 1)
            if x['after']-x['before']!=delta:raise ValueError('Wrong structural count change')
        count=POLICY['initial_kernels']
        by_update={}
        for x in actions:by_update.setdefault(x['update'],[]).append(x)
        for row in metrics:
            for action in by_update.get(row['update'],[]):
                if action['before']!=count:raise ValueError('Broken mutation count history')
                count=action['after']
            if row['kernels']!=count:raise ValueError('Metrics disagree with mutation history')
        forced=[x for x in actions if x['origin']=='forced-smoke-probe']
        if [(x['update'],x['operation'],x['selected']) for x in forced]!=[(10,'clone',1),(20,'split',1),(30,'prune',1)]:
            raise ValueError('Not all explicit mutation probes ran')
        if [(x['update'],x['kind']) for x in decisions]!=[(200,'densify'),(400,'prune'),(400,'densify')]:
            raise ValueError('Natural controller decisions missing or mis-timed')
        natural=[x for x in actions if x['origin']=='natural']
        counts={k:sum(x['selected'] for x in natural if x['operation']==k) for k in ['clone','split','prune']}
        active=counts['clone']+counts['split']>0 and counts['prune']>0
        result['arms'][arm]=dict(final_kernels=metrics[-1]['kernels'],forced_probes_passed=True,
            natural_selected=counts,natural_growth_and_pruning_observed=active,decisions=decisions,
            final_eval=r['final_eval'],resume_passed=True)
        streams.append([(x['update'],x['batch_signature']) for x in metrics])
    if len(set(identities))!=1 or streams[0]!=streams[1]:raise ValueError('A/B initial state, code or input stream mismatch')
    for key in ['seed','steps','sources','batch_pixels','aa','transport','adc','manifest_sha256','source_fingerprints']:
        if any(s[key]!=settings_all[0][key] for s in settings_all):raise ValueError('Unequal setting '+key)
    result['paired_inputs_passed']=True
    result['status']='passed' if all(x['natural_growth_and_pruning_observed'] for x in result['arms'].values()) else 'functional_passed_natural_activity_inconclusive'
    result['next']='Review decisions/thresholds and mutation checks before building a fresh full ADC pair. Never resume smoke as full.'
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',required=True);a=p.parse_args();root=Path(a.pair)
    r=compare(root);(root/'smoke_comparison.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
