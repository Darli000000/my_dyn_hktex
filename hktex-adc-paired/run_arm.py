#!/usr/bin/env python3
"""Full paired ADC: shared initialization, natural mutations, 2000/5000 stages."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from losses import ARMS,objective,objective_name,validate_arm,dr_transform
from adc_support import configure_adc, install_runtime, run_adc, validate_state, accumulation_active


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def make_pair_identity(cfg,settings):
    c=copy.deepcopy(cfg)
    # These two flags describe the deliberate loss difference; all else is paired.
    c.pop('use_linear_gamma_space',None);c.pop('use_hdr',None)
    c['network']['model']['n_sources']=5000  # active count can change on resume
    return dict(config=c,manifest=settings['manifest_sha256'],mesh=settings['mesh_sha256'],
                sources=settings['source_fingerprints'],seed=settings['seed'],steps=settings['steps'],
                batch=settings['batch_pixels'],aa=settings['aa'],transport=settings['transport'])


def install(module,trainer_module,a,code_hash):
    import torch
    import numpy as np
    import mitsuba as mi
    import drjit as dr
    from omegaconf import OmegaConf
    from observer_common import PARAMS,sample_surface
    from offline_common import write_json
    from image_trainer import color3,cpu_state
    original_cfg=module.make_config
    original_setup=trainer_module.ExternalImageTrainer.setup_images
    cls=trainer_module.ExternalImageTrainer
    module.PROFILES['baseline'].update(steps=5000,batch_pixels=256,aa=4,transport=4,sources=5000,
        eval_pixels=1024,eval_every=250,save_every=250,preview_size=0,adc=True)

    def config(repo,data,s,cache_root,oc,resume_cfg=None):
        if resume_cfg is not None:
            if 'paired_loss' not in s:raise ValueError('Resume requires a paired-training checkpoint')
            validate_arm(s['paired_loss'],a.arm,code_hash)
            if s['paired_loss']['pair_root']!=str(Path(a.pair).resolve()):raise ValueError('Pair directory changed')
        cfg,renderer,cache=original_cfg(repo,data,s,cache_root,oc,resume_cfg)
        if s['steps']!=5000 or s['sources']!=5000 or not s['adc']:
            raise ValueError('Expected full ADC with 5000 initial kernels and 5000-step scheduler')
        configure_adc(cfg,s)
        cfg.use_linear_gamma_space=a.arm=='loggamma';cfg.use_hdr=a.arm=='linear'
        s['paired_loss']=dict(version='adc-full-1.0.0',arm=a.arm,code_hash=code_hash,
            pair_root=str(Path(a.pair).resolve()),objective=objective_name(a.arm),
            initialization='shared initial.pt; no UV initialization',scheduler_horizon=5000,
            scope='Two losses under the same natural ADC policy; no forced probes')
        return cfg,renderer,cache
    module.make_config=config

    def setup(t,data,out,s):
        import fcntl
        original_setup(t,data,out,s)
        pair=Path(a.pair).resolve();pair.mkdir(parents=True,exist_ok=True)
        bundle=pair/'initial.pt'
        identity=make_pair_identity(s['trainer_config'],s)
        identity_sha=canonical_hash(identity)
        with open(pair/'initial.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not bundle.exists():
                if a.resume:raise ValueError('Shared initialization missing; do not recreate on resume')
                initial=dict(identity=identity_sha,
                    parameters={k:cpu_state(getattr(t.model.model,k)) for k in PARAMS},
                    numpy_rng=t.rng.bit_generator.state,sampler=data.sampler_state(),
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
                tmp=pair/'initial.tmp.pt';torch.save(initial,tmp);tmp.replace(bundle)
                write_json(pair/'initial_identity.json',identity)
            initial=torch.load(bundle,map_location='cpu',weights_only=True)
            if initial['identity']!=identity_sha:raise ValueError('A/B configuration or dataset differs')
            if not a.resume:
                with torch.no_grad():
                    for k in PARAMS:
                        p=getattr(t.model.model,k);b=initial['parameters'][k].to(p.device,p.dtype)
                        if p.shape!=b.shape:raise ValueError('Initial parameter shape differs: '+k)
                        p.copy_(b)
            if not a.resume:
                t.rng.bit_generator.state=initial['numpy_rng'];data.load_sampler_state(initial['sampler'])
                torch.set_rng_state(initial['torch_rng']);torch.cuda.set_rng_state(initial['cuda_rng'])
        s['paired_loss']['initial_sha256']=sha(bundle)
        s['colour_space']='saved linear RGB; objective: '+objective_name(a.arm)
        write_json(Path(out)/'settings.json',s)
        # Fail before any optimizer step if the stabilized loss is nonfinite.
        x=mi.Float([0.,1e-8,1e-5,.01,.3,1.,1.1,-.001]);dr.enable_grad(x)
        val=dr.mean(dr_transform(x,dr));dr.backward(val)
        if not bool(dr.all(dr.isfinite(val))) or not bool(dr.all(dr.isfinite(dr.grad(x)))):
            raise FloatingPointError('log/gamma numerical preflight failed')
        write_json(Path(out)/'loss_preflight.json',dict(passed=True,
            inputs=[0.,1e-8,1e-5,.01,.3,1.,1.1,-.001],gradient=np.asarray(dr.grad(x)).tolist(),
            initial_sha256=s['paired_loss']['initial_sha256']))
        t.model.reset()
        install_runtime(t)
    cls.setup_images=setup

    original_restore=cls.restore
    def restore(t,state):
        from resume_support import rebind_geodesic_faces
        if state['update']!=2000:raise ValueError('Full continuation requires the completed 2000-update checkpoint')
        original_restore(t,state)
        repair=rebind_geodesic_faces(t)
        for dc in t.density_controllers:dc.refresh_state()
        validate_state(t)
        write_json(t.out/'resume_repair.json',dict(update=t.update,groups=repair,passed=True))
        print('[ADC RESUME]',t.update,'model, optimizer references and accumulators verified',flush=True)
    cls.restore=restore

    def step(t,r,ids,target,seed):
        if not t.has_density_controllers:raise RuntimeError('ADC missing')
        for opt in t.optimizers:opt.zero_grad(set_to_none=True)
        before=t.model.model._kernel_colours.detach().clone() if t.update==0 else None
        pred=t.prediction(r,ids,t.settings['aa'],t.settings['transport'],seed,True)
        target_dr=color3(target)
        loss=objective(pred,target_dr,a.arm,mi,dr)
        if not bool(dr.all(dr.isfinite(loss))):raise FloatingPointError('Nonfinite objective')
        colours=np.asarray(pred).T.copy()
        if not np.isfinite(colours).all():raise FloatingPointError('Nonfinite rendered RGB')
        if t.observer is not None:t.observer.sample(r,ids,target,colours,seed)
        # ADC always uses raw linear radiance, independently of the loss arm.
        x=(ids % r['width']).astype(np.float32)+.5
        y=(ids // r['width']).astype(np.float32)+.5
        ray,_=t.sensor(r).sample_ray(0.,.5,mi.Point2f(mi.Float(x/r['width']),mi.Float(y/r['height'])),mi.Point2f(.5))
        hit=t.scene.ray_intersect(ray)
        if not bool(dr.all(hit.is_valid())):raise RuntimeError('ADC centre hit invalid')
        contributions,idx,weights,_=t._ray_topk_from_first_hits(hit.p.torch().T.contiguous(),hit.prim_index.torch().to(torch.int32))
        for dc in t.density_controllers:
            if not accumulation_active(dc,t.update+1):continue
            dc.pre_backward_step(step=t.update+1,rendered_colours=torch.as_tensor(colours,device=t.device),
                gt_colours=torch.as_tensor(target,device=t.device),kernel_contributions=contributions,
                topk_kernel_idxs=idx,topk_kernel_contribs=weights)
        dr.backward(loss);norms=t._check_gradients()
        run_adc(t)
        with torch.no_grad():t.model.reset();t.model.prepare_kernels()
        for opt in t.optimizers:opt.step()
        t._post_optimizer_step()
        for scheduler in t.schedulers:scheduler.step()
        if before is not None:
            t.first_update_delta=float((before-t.model.model._kernel_colours).abs().max())
            if t.first_update_delta<=0:raise RuntimeError('Colour parameters did not update')
        for name,p in t.model.model.named_parameters():
            if not bool(torch.isfinite(p).all()):raise FloatingPointError('Nonfinite parameter '+name)
        t.model.reset()
        return dict(update=t.update+1,view=r['id'],split=r['split'],
            loss_l1=float(np.abs(colours-target).mean()),loss_objective=float(loss[0]),
            objective=objective_name(a.arm),batch_mse=float(((colours-target)**2).mean()),
            kernels=t.model.model.N_sources,gradients=norms,
            batch_signature=hashlib.sha256(r['id'].encode()+np.asarray(ids,dtype=np.int64).tobytes()+str(seed).encode()).hexdigest(),
            peak_cuda_gb=torch.cuda.max_memory_allocated()/1024**3)
    cls.step_images=step

    def field_probe(t):
        with torch.no_grad():
            if not hasattr(t,'_pair_probe'):
                fid,bary,pos=sample_surface(t.mesh.verts.cpu().numpy(),t.mesh.faces.cpu().numpy(),2048,19019)
                t._pair_probe=(torch.as_tensor(fid,device=t.device,dtype=torch.int32),
                               torch.as_tensor(pos,device=t.device))
            fid,pos=t._pair_probe;m=t.model.model
            t.model.reset();t.model.prepare_kernels()
            info=m.prepare_points(t.mesh,t.model.eigalbo_interp,face_ids=fid,pts=pos)
            colour,_,idx,w=m.diffuse_heat_kernels(t.model.eigalbo_interp,info)
            idx=idx.squeeze(-1).T.long();w=w.squeeze(-1).T
            pre=m._mean_colour+(w[...,None]*m.kernel_colours[idx]).sum(1)/w.sum(1,keepdim=True).clamp_min(1)
            torch.testing.assert_close(pre.clamp(0,1),colour,atol=2e-6,rtol=1e-5)
            row=dict(update=t.update,kernels=m.N_sources,clipped_fraction=float(((pre<0)|(pre>1)).any(1).float().mean()),
                below_zero_rgb=(pre<0).float().mean(0).cpu().tolist(),preclamp_min=float(pre.min()),preclamp_max=float(pre.max()))
            t.model.reset()
        with open(t.out/'field_metrics.jsonl','a') as f:f.write(json.dumps(row)+'\n')

    def train(t):
        from image_observer import ImageRecorder
        if not t.update<a.until<=5000:raise ValueError('Endpoint must be after current update and <=5000')
        t.observer=ImageRecorder(t,t.observer_resume)
        t.geometry_check();field_probe(t)
        report=dict(status='running',start_update=t.update,stage_until=a.until,scheduler_horizon=5000,
                    settings=t.settings,initial_eval=t.evaluate(),observer_directory=str(t.observer.root))
        t.save_checkpoint();write_json(t.out/'report.json',report)
        try:
            while t.update<a.until:
                r,ids,target=t.dataset.train_batch(t.rng,t.settings['batch_pixels'])
                seed=int(t.rng.integers(1,2**30));row=t.step_images(r,ids,target,seed);t.update+=1
                t.observer.after_update(row)
                with open(t.out/'metrics.jsonl','a') as f:f.write(json.dumps(row)+'\n')
                if t.update%10==0 or t.update==1:
                    print('[PAIR]',a.arm,t.update,'objective',row['loss_objective'],'linear L1',row['loss_l1'],flush=True)
                if t.update%250==0:field_probe(t);t.save_checkpoint();t.evaluate()
            if t.update%250:field_probe(t)
            validate_state(t)
            # "final" refers to this stage, so the existing viewer can render it.
            t.observer.snapshot('final','paired stage endpoint; global scheduler unchanged')
            t.observer.finish('complete' if t.update==5000 else 'stage_complete')
            t.save_checkpoint()
            final=t.evaluate(final=t.update==5000)
            report.update(status='complete' if t.update==5000 else 'stage_complete',final_eval=final,
                note='Natural ADC only; compare both losses and inspect held-out images. No forced structural probes.')
        except BaseException as exc:
            report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            t.observer.finish('failed',report['error']);raise
        finally:
            report.update(completed_updates=t.update,elapsed_seconds=time.monotonic()-t.started,
                gradient_checks=t.gradient_checks,first_update_colour_max_delta=t.first_update_delta,
                density_events=t.events,dataset_reads=sorted(t.dataset.read_paths))
            write_json(t.out/'report.json',report)
    cls.train_images=train


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--pair',required=True)
    p.add_argument('--output',required=True);p.add_argument('--until',type=int,required=True)
    p.add_argument('--resume',help='Checkpoint of SAME arm from the previous stage')
    p.add_argument('--repo',default='/root/hktex-work/hktex')
    p.add_argument('--dataset',default='/root/hktex-work/data/offline_multiview/smpl-male-full-20261025-003839-35340')
    p.add_argument('--gpu',default='0');a=p.parse_args()
    if a.until not in (2000,5000):p.error('Stages must end at 2000 or 5000')
    if (a.until==5000)!=bool(a.resume):p.error('2000 starts fresh; 5000 resumes its own 2000 checkpoint')
    repo=Path(a.repo).resolve();here=Path(__file__).parent
    expected=json.loads((here/'expected_sources.json').read_text())
    bad=[n for n,h in expected['sha256'].items() if not (repo/n).is_file() or sha(repo/n)!=h]
    if bad:p.error('Source differs from inspected main: '+', '.join(bad))
    code_hash=canonical_hash({f:sha(here/f) for f in ['run_arm.py','losses.py','expected_sources.json','adc_support.py','resume_support.py']})
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu;os.environ.setdefault('MPLBACKEND','Agg')
    directory=repo/'hktex-image-training-observed'
    sys.path.insert(0,str(repo));sys.path.insert(0,str(directory))
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.cuda.set_device(0);torch.set_float32_matmul_precision('highest')
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')
    import image_trainer
    spec=importlib.util.spec_from_file_location('paired_training_entry',directory/'train_images.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    install(module,image_trainer,a,code_hash)
    args=[str(directory/'train_images.py'),'resume' if a.resume else 'train',
          '--profile','baseline','--repo',str(repo),'--dataset',a.dataset,'--output',a.output,
          '--observer-interval','200','--gpu',a.gpu]
    if a.resume:args+=['--checkpoint',a.resume]
    sys.argv=args;module.main()


if __name__=='__main__':main()
