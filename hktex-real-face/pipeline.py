#!/usr/bin/env python3
"""Eye-inclusive preflight followed automatically by a 300-step surface RGB pilot."""
import argparse
import json
import os
import random
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from common import REGIONS,verify_dataset,write_json,sha
from runtime import tensor,load_views,make_model,pool_views,sample_stratified,predict,metrics
from checks import preflight,surface_state
from previews import preview

def run(a,out,report):
    def stage(name):
        report['stage']=name;write_json(out/'report.json',report);print('[STAGE]',name,flush=True)
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu;os.environ.setdefault('MPLBACKEND','Agg')
    repo=Path(a.repo).resolve();sys.path.insert(0,str(repo))
    import torch
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')
    if not torch.cuda.is_available():raise RuntimeError('Use the HKTex CUDA environment')
    torch.cuda.set_device(0);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    from hktex.modules.geodesic_opt import GeodesicOpt
    dataset=Path(a.dataset).resolve();m=verify_dataset(dataset)
    with np.load(dataset/m['mesh'],allow_pickle=False) as z:g={k:z[k].copy() for k in z.files}
    report['environment']=dict(torch=str(torch.__version__),cuda=torch.version.cuda,mitsuba=mi.__version__,gpu=torch.cuda.get_device_name(0))
    report['contract']=dict(loss='linear-RGB pixel MSE',sources=5000,adc=False,geometry_fixed=True,cameras_fixed=True,
        lighting='baked observed appearance, no relighting or additional BSDF',mask_review_required=True,
        supervised_regions=REGIONS,tracer=dict(eps=1e-6,avoid_holes=False,max_iterations=256),
        initialization='mean RGB 0.35; residual N(0,0.003); original surface FPS for all 5000 kernel centres',
        sampling='512 base, 256 eye-surround, 128 left, 128 right; pixel-count importance correction',
        manifest_sha256=sha(dataset/'manifest.json'))
    pre=dict(status='running',environment=report['environment']);write_json(out/'preflight.json',pre)
    def save_check(name):
        pre['stage']=name;write_json(out/'preflight.json',pre);stage('check: '+name)
    try:
        views=load_views(dataset,m,g,pre,save_check);train=pool_views(views,'train')
        save_check('spectral_model_initialization')
        model,mesh,optimizer_cfg=make_model(repo,Path(a.cache_root).resolve(),g,out)
        preflight(model,mesh,g,train,pre,save_check)
    except BaseException as e:
        pre.update(status='failed',error=str(e),traceback=traceback.format_exc());write_json(out/'preflight.json',pre);raise
    # Only preflight-passed packages enter this short pilot. No silent config fallback.
    stage('short_training_setup')
    settings=dict(steps=a.steps,seed=a.seed,evaluate_every=100,geo_lr=.001,adam={},schedule='constant during short pilot')
    groups=[]
    for opt in optimizer_cfg:
        if opt.name=='Adam':
            for name,param in opt.params.items():
                key=name.split('.')[-1];lr=float(param.lr)*.3
                groups.append(dict(params=[getattr(model.model,key)],lr=lr));settings['adam'][key]=lr
    adam=torch.optim.Adam(groups)
    geo=GeodesicOpt([dict(params=[model.model._kernel_locations],face_ids=[model.model._kernel_face_ids])],tracer=model.tracer,lr=settings['geo_lr'])
    write_json(out/'settings.json',settings)
    rng=np.random.default_rng(a.seed);probe_rng=np.random.default_rng(10000+a.seed)
    probe,probe_weights=sample_stratified(train,probe_rng,(2048,1024,512,512))
    report['training_region_pixels']={name:int((train['regions']==k).sum()) for k,name in REGIONS.items()}
    report['evaluations']=[];report['total_kernel_face_changes']=0;report['max_kernel_displacement']=0.
    def evaluate(step,final=False):
        folder=out/f'step_{step:05d}';folder.mkdir()
        model.reset();model.model.save_torch(folder/'field.pt',compressed=False)
        write_json(folder/'checkpoint.json',dict(step=step,network='../network.yaml',dataset=str(dataset),manifest_sha256=sha(dataset/'manifest.json'),format='HKTex state_dict, uncompressed float32',sources=5000))
        pred=predict(model,probe['points'],probe['faces']);record=dict(step=step)
        record['train_probe']=metrics(pred,probe['rgb'],probe['regions'])
        weighted=float((((pred.astype(np.float64)-probe['rgb'])**2).mean(1)*probe_weights).sum())
        record['train_probe']['all'].update(mse=weighted,psnr=float(-10*np.log10(max(weighted,1e-12))))
        record['train_probe']['sampling']='fixed region-stratified 4096-point probe, all-MSE importance corrected'
        for split in ['val']+(['test'] if final else []):
            pooled=pool_views(views,split);pred=predict(model,pooled['points'],pooled['faces'])
            record[split]=metrics(pred,pooled['rgb'],pooled['regions'])
        for view in views:
            ci=view['row']['camera_index']
            if ci in [0,7] or (final and ci==15):
                pred=predict(model,view['points'],view['faces']);preview(view,pred,folder/'previews')
        write_json(folder/'metrics.json',record);report['evaluations'].append(record)
        stage('evaluated_step_'+str(step))
    evaluate(0)
    started=time.monotonic();diag=float(np.linalg.norm(np.ptp(g['vertices'],axis=0)))
    for step in range(1,a.steps+1):
        batch,weight=sample_stratified(train,rng)
        pts=tensor(batch['points']);fi=tensor(batch['faces'],torch.int32);target=tensor(batch['rgb']);w=tensor(weight)
        model.reset();model.zero_grad(set_to_none=True)
        pred=model(pts,face_ids=fi);err=(pred-target).square().mean(1);loss=(err*w).sum()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward()
        for name,param in model.model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():raise FloatingPointError('Nonfinite gradient '+name)
        loc=model.model._kernel_locations;fid=model.model._kernel_face_ids
        if loc.grad is None:raise ValueError('Missing position gradient')
        previous=loc.detach().clone();previous_fid=fid.detach().clone()
        requested=loc.grad.detach().norm(dim=1)*settings['geo_lr']
        adam.step();geo.step();model.post_optimizer_step();model.reset()
        with torch.no_grad():
            displacement=(loc-previous).norm(dim=1)
            if not torch.isfinite(loc).all() or (fid<0).any() or (fid>=len(g['faces'])).any():raise ValueError('Invalid position update')
            labels=tensor(g['face_components'],torch.long)
            if not torch.equal(labels[fid.long()],labels[previous_fid.long()]):raise ValueError('Position update crossed component')
            if bool((displacement>requested+diag*2e-5).any()):raise ValueError('Position update exceeded requested travel')
            changed=int((fid!=previous_fid).sum());max_move=float(displacement.max())
        report['total_kernel_face_changes']+=changed;report['max_kernel_displacement']=max(report['max_kernel_displacement'],max_move)
        if step%10==0 or step==1 or step==a.steps:
            surface_state(mesh,g,loc,fid,previous_fid)
            rec=dict(step=step,loss=float(loss.detach()),seconds=time.monotonic()-started,kernels=model.model.N_sources,
                region_batch_mse={name:float(err[tensor(batch['regions']==k,torch.bool)].detach().mean()) for k,name in REGIONS.items()},
                max_position_move=max_move,face_changes=changed,requested_above_eps=int((requested>1e-6).sum()),
                moved_above_numerical_floor=int((displacement>diag*2e-7).sum()))
            with open(out/'metrics.jsonl','a') as stream:stream.write(json.dumps(rec,allow_nan=False)+'\n')
            report['last_step']=step;stage(f'train_{step:05d} loss={rec["loss"]:.6g}')
        if step%100==0 or step==a.steps:evaluate(step,final=step==a.steps)
    torch.save(dict(adam=adam.state_dict(),geodesic=geo.state_dict(),step=a.steps),out/'optimizer_final.pt')
    report.update(status='short_training_completed',full_training_ready=False,roi_visual_review_required=True,
        note='Review regional validation errors and eye crops; test split evaluated only at final step. No ADC or geometry optimization.',
        next='Inspect report/previews before any 5000-step run.')
    write_json(out/'report.json',report)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--cache-root');p.add_argument('--gpu',default='0')
    p.add_argument('--steps',type=int,default=300);p.add_argument('--seed',type=int,default=0)
    a=p.parse_args()
    if not 1<=a.steps<=1000:p.error('This entry is a short pilot only: steps 1..1000')
    if a.cache_root is None:a.cache_root=str(Path(a.repo)/'outputs/real-face-spectral-cache')
    out=Path(a.output).resolve()
    if out.exists():p.error('Use a new output directory')
    out.mkdir(parents=True);report=dict(status='running',version='real-face-eyes-v1',args=vars(a))
    try:run(a,out,report)
    except BaseException as e:
        report.update(status='failed',error=str(e),traceback=traceback.format_exc());write_json(out/'report.json',report);raise
    print('[COMPLETE]',out,flush=True)
if __name__=='__main__':main()
