#!/usr/bin/env python3
"""Sequential shared-initialization baseline/ADC experiment. No core-file edits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from common import REGIONS,verify_dataset,write_json,sha
from policy import validate_config,event_steps

HERE=Path(__file__).resolve().parent

def package_hashes():return {p.name:sha(p) for p in sorted(HERE.glob('*.py'))}

def log(report,out,stage):
    report['stage']=stage;write_json(out/'report.json',report);print('[STAGE]',stage,flush=True)

def environment(a):
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu;os.environ['HKTEX_STABLE_KNN']='1';os.environ.setdefault('MPLBACKEND','Agg')
    repo=Path(a.repo).resolve();sys.path.insert(0,str(repo))
    import torch
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')
    if not torch.cuda.is_available():raise RuntimeError('Run in the working HKTex CUDA environment')
    torch.cuda.set_device(0);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    from stable_search import install
    backend=install()
    return dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),mitsuba=mi.__version__,knn_backend=backend)

def context(a,setup=False):
    pair=Path(a.pair).resolve()
    if setup:
        c=validate_config(json.loads(Path(a.config).read_text()));dataset=Path(a.dataset).resolve()
    else:
        protocol=json.loads((pair/'protocol.json').read_text());c=validate_config(protocol['configuration']);dataset=Path(protocol['dataset'])
        if protocol['package_sha256']!=package_hashes():raise ValueError('Package code changed during pair; start a new experiment')
        if protocol['initial_state_sha256']!=sha(pair/'initial_state.pt'):raise ValueError('Shared initialization changed')
    m=verify_dataset(dataset)
    if not setup and sha(dataset/'manifest.json')!=protocol['manifest_sha256']:raise ValueError('Dataset manifest changed')
    with np.load(dataset/m['mesh'],allow_pickle=False) as z:g={k:z[k].copy() for k in z.files}
    return pair,c,dataset,m,g

def cached_views(pair,m):
    index=json.loads((pair/'ray_cache/index.json').read_text());rows=[]
    for row in m['views']:
        file=pair/'ray_cache'/f'{row["id"]}.npz'
        if sha(file)!=index[row['id']]:raise ValueError('Changed first-hit cache')
        with np.load(file,allow_pickle=False) as z:rows.append(dict(row=row,**{k:z[k].copy() for k in z.files}))
    return rows

def make(a,pair,c,g,out,state_file=None):
    import torch
    from engine import seed_all,restore
    from runtime import make_model
    seed_all(c['seed']);state=torch.load(state_file or pair/'initial_state.pt',map_location='cpu',weights_only=True)
    model,mesh,source=make_model(Path(a.repo).resolve(),Path(a.repo).resolve()/'outputs/real-face-spectral-cache',g,out,n_sources=state['kernels'])
    restore(model,state)
    # Detect a modified core/config between arms instead of silently comparing them.
    protocol=json.loads((pair/'protocol.json').read_text())
    if json.loads((out/'source_fingerprints.json').read_text())!=protocol['core_source_sha256']:raise ValueError('HKTex core changed since setup')
    if sha(Path(a.repo)/'configs/multiview_hktex_knn_ray_small.yaml')!=protocol['source_config_sha256']:raise ValueError('Base network config changed')
    return model,mesh,source

def setup(a,out,report):
    import torch
    from runtime import load_views,make_model,pool_views
    from checks import preflight
    from engine import seed_all,raw_state,digest
    pair,c,dataset,m,g=context(a,True);seed_all(c['seed'])
    if (pair/'protocol.json').exists():raise ValueError('Use a new pair directory')
    pre=dict(status='running',environment=report['environment'])
    def save(stage):pre['stage']=stage;write_json(out/'preflight.json',pre);log(report,out,stage)
    try:
        views=load_views(dataset,m,g,pre,save)
        model,mesh,_=make_model(Path(a.repo).resolve(),Path(a.repo).resolve()/'outputs/real-face-spectral-cache',g,out)
        preflight(model,mesh,g,pool_views(views,'train'),pre,save)
        state=raw_state(model);torch.save(state,pair/'initial_state.pt')
        folder=pair/'ray_cache';folder.mkdir();hashes={}
        for v in views:
            file=folder/f'{v["row"]["id"]}.npz';np.savez_compressed(file,**{k:x for k,x in v.items() if k!='row'});hashes[v['row']['id']]=sha(file)
        write_json(folder/'index.json',hashes)
        protocol=dict(schema='real-face-adc-pair-v1',dataset=str(dataset),manifest_sha256=sha(dataset/'manifest.json'),
            configuration=c,schedule=event_steps(c),package_sha256=package_hashes(),
            initial_state_sha256=sha(pair/'initial_state.pt'),initial_parameter_sha256=digest(state),
            core_source_sha256=json.loads((out/'source_fingerprints.json').read_text()),
            source_config_sha256=sha(Path(a.repo)/'configs/multiview_hktex_knn_ray_small.yaml'),environment=report['environment'],
            contract=dict(loss='importance-corrected linear RGB pixel MSE',geometry_fixed=True,cameras_fixed=True,
                          sampling='512/256/128/128, both arms share seed and sampling stream',
                          adc_stats='MAE, positive-weight hit count and raw contribution, importance corrected to a 1024-uniform-pixel-equivalent batch',
                          adc_mechanics='upstream clone/split/prune, AFTER optimizer step; component floors and prune cap added',
                          split_radius='median unique mesh edge length times configured factor',
                          learning_rates='original small-ray Adam x0.3, geo 0.001; both cosine decay to 10%',
                          initialization='shared exact float32 state; fresh 5000-step runs, NOT resuming the pilot',
                          evaluation='train fixed 4096-point weighted probe; full val camera 7; test camera 15 only final',
                          appearance='observed fixed-lighting RGB, not albedo; ROI only; geometry used all VHAP cameras'))
        write_json(pair/'protocol.json',protocol)
    except BaseException as e:
        pre.update(status='failed',error=str(e),traceback=traceback.format_exc());write_json(out/'preflight.json',pre);raise
    report.update(status='passed',initial_parameter_sha256=protocol['initial_parameter_sha256']);log(report,out,'setup_complete')

def smoke(a,out,report):
    from runtime import pool_views,sample_stratified
    from engine import optimizers,update
    from adc import WeightedADC
    from checks import preflight
    pair,c,dataset,m,g=context(a);views=cached_views(pair,m);train=pool_views(views,'train')
    model,mesh,source=make(a,pair,c,g,out);opts=optimizers(model,source,c)
    dc=WeightedADC({},mesh=mesh,model=model.model,optimizers=opts,network=model,geometry=g,policy=c['adc'])
    rng=np.random.default_rng(c['seed']);report['mechanics_only']=True;report['events']=[]
    for step in range(1,5):
        batch,w=sample_stratified(train,rng);update(model,opts,g,batch,w,step,c)
    # Exercise every mutation on a disposable copy. Do not loosen production thresholds.
    n=model.model.N_sources;components=dc.component_ids();head=int(g['face_components'][g['face_regions']==1][0])
    headid=int(np.flatnonzero(components==head)[0]);mask=np.zeros(n,bool);mask[headid]=True
    dc.mutate('clone',mask);report['events'].append(dict(kind='forced_clone',kernels=model.model.N_sources))
    comps=dc.component_ids();mask=np.zeros(model.model.N_sources,bool)
    for label in [3,4]:
        ci=int(g['face_components'][g['face_regions']==label][0]);mask[np.flatnonzero(comps==ci)[0]]=True
    dc.mutate('split',mask);report['events'].append(dict(kind='forced_split_both_eyes',kernels=model.model.N_sources))
    # Restore total count to 5000; test prune reindexing without deleting an eye.
    mask=np.zeros(model.model.N_sources,bool);mask[np.flatnonzero(dc.component_ids()==head)[-3:]]=True
    dc.mutate('prune',mask);report['events'].append(dict(kind='forced_prune',kernels=model.model.N_sources))
    for step in range(5,9):
        batch,w=sample_stratified(train,rng);update(model,opts,g,batch,w,step,c)
    report['optimizer_audit']=dc.optimizer_audit();report['components_after']=dc.component_counts()
    pre=dict(status='running')
    def save(stage):pre['stage']=stage;write_json(out/'post_mutation_preflight.json',pre);log(report,out,stage)
    try:preflight(model,mesh,g,train,pre,save)
    except BaseException as e:
        pre.update(status='failed',error=str(e),traceback=traceback.format_exc());write_json(out/'post_mutation_preflight.json',pre);raise
    report.update(status='passed',note='Forced mutation mechanics passed; both full arms restart from the untouched shared initialization.')
    log(report,out,'adc_mechanics_smoke_passed')

def train(a,out,report):
    import torch
    from engine import optimizers,update,checkpoint,Evaluator,raw_state,digest
    from runtime import sample_stratified
    from adc import WeightedADC
    pair,c,dataset,m,g=context(a)
    if json.loads((pair/'smoke/report.json').read_text())['status']!='passed':raise ValueError('ADC smoke has not passed')
    model,mesh,source=make(a,pair,c,g,out);views=cached_views(pair,m);ev=Evaluator(views,c['seed'])
    opts=optimizers(model,source,c);dc=None
    if a.arm=='adc':dc=WeightedADC({},mesh=mesh,model=model.model,optimizers=opts,network=model,geometry=g,policy=c['adc'])
    protocol=json.loads((pair/'protocol.json').read_text());mh=protocol['manifest_sha256']
    report.update(arm=a.arm,configuration=c,initial_parameter_sha256=digest(raw_state(model)),manifest_sha256=mh,
                  evaluations=[],adc_events=[],total_kernel_face_changes=0,max_kernel_displacement=0.)
    if report['initial_parameter_sha256']!=protocol['initial_parameter_sha256']:raise ValueError('Shared initialization mismatch')
    write_json(out/'settings.json',dict(configuration=c,learning_rates={group['name']:group['initial_lr'] for opt in opts for group in opt.param_groups},adc_enabled=dc is not None,
        component_floors=dc.floors if dc else None,split_radius=dc.radius if dc else None,knn_backend=report['environment']['knn_backend']))
    def evaluate(step):
        folder=checkpoint(model,opts,out/f'step_{step:05d}',step,'evaluation',mh,dc)
        record=ev.run(model,folder,step,step==c['steps']);report['evaluations'].append(record)
        log(report,out,'evaluated_'+str(step))
    evaluate(0);rng=np.random.default_rng(c['seed']);sampling=hashlib.sha256();started=time.monotonic()
    for step in range(1,c['steps']+1):
        batch,w=sample_stratified(ev.train,rng)
        for k in ['points','faces','rgb','regions']:sampling.update(batch[k].tobytes())
        sampling.update(w.tobytes())
        rec=update(model,opts,g,batch,w,step,c,dc)
        report['total_kernel_face_changes']+=rec['face_changes'];report['max_kernel_displacement']=max(report['max_kernel_displacement'],rec['max_position_move'])
        if step==c['adc']['start']-1:
            state=raw_state(model)
            if a.arm=='baseline':torch.save(state,pair/'shared_prefix.pt')
            else:
                ref=torch.load(pair/'shared_prefix.pt',map_location='cpu',weights_only=True)
                error=max(float((v-ref['parameters'][k]).abs().max()) for k,v in state['parameters'].items())
                same_faces=torch.equal(state['face_ids'],ref['face_ids'])
                report['prefix_check']=dict(step=step,max_parameter_difference=error,identical_face_ids=same_faces,
                    matches_within_1e_minus6_tolerance=bool(same_faces and error<=1e-6),
                    note='Audit only: CUDA backward reductions may differ across independent runs. Exact initialization and sampling are separately enforced.')
        if dc is not None and dc.due(step):
            snapshots=out/'adc_snapshots';snapshots.mkdir(exist_ok=True)
            checkpoint(model,opts,snapshots/f'u{step:05d}_before',step,'before_adc',mh,dc)
            event=dc.apply(step);report['adc_events'].append(event)
            with open(out/'adc_events.jsonl','a') as f:f.write(json.dumps(event,allow_nan=False)+'\n')
            checkpoint(model,opts,snapshots/f'u{step:05d}_after',step,'after_adc',mh,dc)
            log(report,out,f'adc_{step}: {event["kernels_before"]} -> {event["kernels_after"]}')
        if step==1 or step%c['log_every']==0 or step==c['steps']:
            rec.update(seconds=time.monotonic()-started,kernels=model.model.N_sources)
            with open(out/'metrics.jsonl','a') as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
            report['last_step']=step;log(report,out,f'train_{step} loss={rec["loss"]:.6g}')
        if step==300 or step%c['evaluate_every']==0 or step==c['steps']:evaluate(step)
    comp=np.bincount(g['face_components'][model.model._kernel_face_ids.detach().cpu().numpy()],minlength=int(g['face_components'].max())+1).tolist()
    report.update(status='completed',final_kernels=model.model.N_sources,final_component_counts=comp,
                  sampling_sha256=sampling.hexdigest(),training_and_intermediate_eval_seconds=time.monotonic()-started)
    log(report,out,'training_complete')

def compare(a,out,report):
    from engine import combine_previews
    pair,c,_,_,_=context(a)
    arms={key:json.loads((pair/key/'report.json').read_text()) for key in ['baseline','adc']}
    if any(v['status']!='completed' for v in arms.values()):raise ValueError('Both full arms must complete')
    checks={key:arms['baseline'][key]==arms['adc'][key] for key in ['initial_parameter_sha256','manifest_sha256','sampling_sha256']}
    if not all(checks.values()):raise ValueError('Pair controls mismatch: '+str(checks))
    final={key:v['evaluations'][-1] for key,v in arms.items()}
    report.update(status='completed',controls=checks,prefix_check=arms['adc'].get('prefix_check'),results=final,kernels={key:v['final_kernels'] for key,v in arms.items()},
                  final_component_counts={key:v['final_component_counts'] for key,v in arms.items()},
                  adc_totals={k:sum(e[k] for e in arms['adc']['adc_events']) for k in ['pruned','cloned','split_parents']},
                  metric_scope='Linear RGB, shared valid ROI; test is appearance-held-out only, geometry used all cameras.')
    report['adc_minus_baseline']={split:{metric:final['adc'][split]['all'][metric]-final['baseline'][split]['all'][metric] for metric in ['mse','psnr']} for split in ['val','test']}
    write_json(pair/'comparison.json',report)
    combine_previews(pair/'baseline'/f'step_{c["steps"]:05d}',pair/'adc'/f'step_{c["steps"]:05d}',pair/'comparison.png')
    log(report,out,'pair_complete')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['setup','smoke','train','compare'])
    p.add_argument('--pair',required=True);p.add_argument('--repo',default=str(HERE.parent));p.add_argument('--gpu',default='0')
    p.add_argument('--dataset');p.add_argument('--config',default=str(HERE/'config.json'));p.add_argument('--arm',choices=['baseline','adc'])
    a=p.parse_args()
    if a.mode=='setup' and not a.dataset:p.error('setup requires --dataset')
    if a.mode=='train' and not a.arm:p.error('train requires --arm')
    pair=Path(a.pair).resolve();out=pair/(a.arm if a.mode=='train' else a.mode)
    if out.exists():p.error('Stage output exists; use a new pair directory')
    out.mkdir(parents=True);report=dict(status='running',mode=a.mode,args=vars(a))
    try:
        if a.mode!='compare':report['environment']=environment(a)
        globals()[a.mode](a,out,report)
    except BaseException as e:
        report.update(status='failed',error=str(e),traceback=traceback.format_exc());write_json(out/'report.json',report);raise
    print('[SUCCESS]',out,flush=True)
if __name__=='__main__':main()
