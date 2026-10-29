#!/usr/bin/env python3
"""Read-only, scoped radiance and colour-gradient checks for image trainer v1.1."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
import numpy as np
from numerics import compare_radiance, compare_gradient, patches, pixel_ids, combined_status


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def write(path,value):
    p=Path(path); tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)


class Suite:
    def __init__(self,a,out):
        self.a,self.out=a,out
        self.report=dict(version='1.0.0', status='running', args=vars(a), checks=[],
            scope='Static image pipeline; sampled patches and colour directions only. Not full geometry-gradient validation.')
        self.started=time.monotonic()
        self.temp=None

    def record(self,name,result):
        self.report['checks'].append(dict(name=name,**result))
        write(self.out/'report.json',self.report)
        print('[CHECK]',name,result['status'],flush=True)

    def boot(self):
        a=self.a;repo=Path(a.repo).resolve();run=Path(a.run).resolve()
        expected=json.loads((Path(__file__).parent/'expected_sources.json').read_text())
        mismatch=[n for n,h in expected['sha256'].items()
                  if not (repo/n).is_file() or digest(repo/n)!=h]
        if mismatch: raise RuntimeError('Source differs from inspected main: '+', '.join(mismatch))
        self.report['inspected_commit']=expected['commit']
        os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
        os.environ.setdefault('MPLBACKEND','Agg')
        sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'hktex-image-training-observed'))
        global torch,mi,dr,rendering,sensor_dict
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
        torch.cuda.set_device(0);torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False
        import mitsuba as mi
        mi.set_variant('cuda_ad_rgb')
        import drjit as dr
        import trimesh
        from omegaconf import OmegaConf
        from types import SimpleNamespace
        import hktex.rendering as rendering
        from image_dataset import ImageDataset,check_contract
        from image_trainer import ExternalImageTrainer
        from offline_common import sensor_dict
        from observer_common import PARAMS
        self.param_names=PARAMS
        cp=run/'checkpoint.pt'
        self.report['checkpoint_sha256']=digest(cp)
        state=torch.load(cp,map_location='cpu',weights_only=True)
        settings=state['settings'];self.settings=copy.deepcopy(settings)
        dataset=Path(a.dataset or settings['dataset']).resolve()
        checks=check_contract(dataset,expected_size=2048)
        if checks['manifest_sha256']!=settings['manifest_sha256'] or checks['mesh_sha256']!=settings['mesh_sha256']:
            raise RuntimeError('Dataset identity differs from checkpoint')
        for n,h in settings['source_fingerprints'].items():
            if digest(repo/n)!=h: raise RuntimeError('Checkpoint source fingerprint differs: '+n)
        self.data=ImageDataset(dataset)
        self.report.update(dataset=str(dataset),dataset_manifest_sha256=checks['manifest_sha256'],
            checkpoint_update=state['update'],gpu=torch.cuda.get_device_name(0),
            torch_version=str(torch.__version__),mitsuba_version=mi.__version__,
            diagnostic_reads_eval_gt=False, training_or_optimizer_steps=0)
        write(self.out/'dataset_check.json',checks)
        cfg=OmegaConf.create(copy.deepcopy(settings['trainer_config']))
        cfg.optimizers=[];cfg.density_controllers=[]
        cfg.network.model.n_sources=len(state['texture']['_kernel_colours'])
        ec=cfg.network.eigen_albo
        cache=Path(a.cache or (str(ec.mesh_path).rsplit('.',1)[0]+'_'+ec.get('precomputed_name','eigen_albo')+'_'+ec.get('local_frames','principal_curvatures')+'.pt')).resolve()
        ch=digest(cache);self.report['cache_sha256']=ch
        manifest=run/'observer/manifest.json'
        if manifest.is_file():
            old=json.loads(manifest.read_text())
            if ch!=old['cache_sha256']: raise RuntimeError('Eigen cache differs from observer digest')
            self.record('cache_identity',dict(status='passed',historical_digest=True))
        else:
            self.record('cache_identity',dict(status='inconclusive',reason='No historical observer cache digest'))
        self.temp=tempfile.TemporaryDirectory(prefix='hktex-validation-')
        ec.mesh_path=str(Path(self.temp.name)/'mesh.obj');ec.precomputed_name='audit'
        ec.use_precomputed=True;ec.error_if_not_precomputed=True
        (Path(self.temp.name)/f'mesh_audit_{ec.local_frames}.pt').symlink_to(cache)
        with np.load(dataset/self.data.manifest['train_mesh'],allow_pickle=False) as z:
            v,f,n=[z[k].copy() for k in ['vertices','faces','normals']]
        self.tri=trimesh.Trimesh(vertices=v,faces=f,vertex_normals=n,process=False)
        if not np.array_equal(self.tri.faces,f) or not np.array_equal(self.tri.vertices,v):
            raise RuntimeError('Mesh changed on import')
        rc=OmegaConf.create(json.loads((dataset/self.data.manifest['renderer_config']).read_text()))
        self.trainer=ExternalImageTrainer(cfg,SimpleNamespace(mesh=self.tri,saved_normals=n),renderer_cfg=rc)
        self.trainer.setup_images(self.data,self.out,settings)
        m=self.trainer.model.model
        with torch.no_grad():
            for k in PARAMS:
                p=getattr(m,k);q=state['texture'][k].to(p.device,p.dtype)
                if p.shape!=q.shape: raise RuntimeError('Checkpoint shape mismatch: '+k)
                p.copy_(q)
        self.original={k:getattr(m,k).detach().clone() for k in PARAMS}
        for p in m.parameters():p.requires_grad_(False)
        m._kernel_colours.requires_grad_(True);m._mean_colour.requires_grad_(True)
        self.trainer.model.reset();self.trainer.geometry_check()
        self.record('training_geometry',dict(status='passed',file='geometry_check.json'))
        # Resolve the learned mesh by its attached Python object, not scene shape ordering.
        self.mi_mesh,self.texture=self.trainer.renderer.mesh_to_mitsuba(self.tri,self.trainer.model,
            bsdf_additional_settings=dict(specular=0.,metallic=0.,roughness=1.))
        p=mi.traverse(self.mi_mesh);p['vertex_normals']=n.astype(np.float32).ravel();p.update()
        self.pred_scene=self.trainer.scene
        views=self.data.views
        if a.views:
            requested=a.views.split(',');lookup={r['id']:r for r in views}
            self.views=[lookup[x] for x in requested]
            if len(set(requested))!=len(requested):raise ValueError('Duplicate views')
        else:
            self.views=[self.data.by_split[s][0] for s in ['train','val','test']]
        self.record('source_and_checkpoint',dict(status='passed',kernels=m.N_sources,views=[r['id'] for r in self.views]))

    def restore_colours(self,neutral=False):
        m=self.trainer.model.model
        with torch.no_grad():
            for k in self.param_names:getattr(m,k).copy_(self.original[k])
            if neutral:
                m._mean_colour.copy_(torch.tensor([[.3,.4,.5]],device='cuda'))
                m._kernel_colours.zero_()
        self.trainer.model.inference_mode(False);self.trainer.model.reset()

    def scene(self,mesh,r,patch=None,spp=1):
        spec=self.trainer.renderer.configure_scene()
        spec['integrator']=dict(type='path',max_depth=3,hide_emitters=True)
        sd=sensor_dict(mi,r,spp);sd['film']['pixel_format']='rgb'
        if patch:
            sd['film'].update(crop_offset_x=patch['x'],crop_offset_y=patch['y'],
                              crop_width=patch['size'],crop_height=patch['size'])
        else:
            # RBIntegrator needs a dummy film, while rays use a separate full sensor.
            sd['film'].update(width=1,height=1)
        spec['camera']=sd;spec['mesh']=mesh
        return mi.load_dict(spec)

    def gt_mesh(self):
        p=self.data.root/'eval_gt/reference.npz'
        self.report['diagnostic_reads_eval_gt']=True
        self.report['eval_gt_sha256']=digest(p)
        with np.load(p,allow_pickle=False) as z: g={k:z[k].copy() for k in z.files}
        tex=g['texture_linear']
        if not np.isfinite(tex).all() or tex.min()<0 or tex.max()>1:
            raise ValueError('Invalid saved linear GT texture')
        bsdf=mi.load_dict(dict(type='principled',specular=0.,metallic=0.,roughness=1.,
            base_color=dict(type='bitmap',bitmap=mi.Bitmap(tex),raw=True)))
        props=mi.Properties();props['mesh_bsdf']=bsdf
        mesh=mi.Mesh('diagnostic_gt',vertex_count=len(g['vertices']),face_count=len(g['faces']),
                     has_vertex_texcoords=True,has_vertex_normals=True,props=props)
        uv=g['uv'].copy();uv[:,1]=1-uv[:,1]
        params=mi.traverse(mesh)
        for key,val in [('vertex_positions',g['vertices']),('faces',g['faces']),('vertex_normals',g['normals']),('vertex_texcoords',uv)]:
            params[key]=np.ascontiguousarray(val).ravel()
        params.update();return mesh

    def raw_rays(self,scene,record,ids,seed,spp):
        # Use the exact ExternalImageTrainer ray construction and reduction.
        ray=self.trainer.rays(record,ids,4,spp//4,np.random.default_rng(seed))
        L=rendering.render_ray(scene,ray=ray,integrator=self.trainer.integrator,spp=spp,seed=seed+1)
        return np.asarray(rendering.integrate_ray_samples(L,spp)).T.copy()

    def radiance(self):
        from PIL import Image
        self.restore_colours();gt=self.gt_mesh();a=self.a
        with torch.no_grad(),dr.suspend_grad():
            for vi,r in enumerate(self.views):
                mask=np.asarray(Image.open(self.data.root/r['mask']))>0
                specs=patches(mask,self.data.image(r),a.patch_size,a.seed+vi)
                for pi,patch in enumerate(specs):
                    key=f'{r["id"]}_{patch["label"]}';ids=pixel_ids(r,patch)
                    target=self.data.target(r,ids)
                    gscene=self.scene(gt,r,patch,a.spp);grays=self.scene(gt,r)
                    pscene=self.scene(self.mi_mesh,r,patch,a.spp)
                    # Independently check GT hit geometry/normals against training mesh.
                    ray=self.trainer.rays(r,ids,1,1,np.random.default_rng(a.seed))
                    gh,ph=grays.ray_intersect(ray),self.pred_scene.ray_intersect(ray)
                    poserr=float(np.max(np.abs(np.asarray(gh.p)-np.asarray(ph.p))))
                    nerr=float(np.max(np.abs(np.asarray(gh.sh_frame.n)-np.asarray(ph.sh_frame.n))))
                    valid=bool(dr.all(gh.is_valid() & ph.is_valid()))
                    self.record(key+'_gt_geometry',dict(status='passed' if valid and poserr<2e-5 and nerr<2e-4 else 'failed',position_max=poserr,normal_max=nerr))
                    arr={k:[] for k in ['gt_native','gt_ray','prediction_native','prediction_ray']}
                    for j in range(a.repeats):
                        seed=a.seed+vi*100000+pi*10000+j*100
                        print('[RADIANCE]',key,j+1,'/',a.repeats,flush=True)
                        arr['gt_native'].append(np.asarray(mi.render(gscene,seed=seed,spp=a.spp)).reshape(-1,3).copy())
                        arr['gt_ray'].append(self.raw_rays(grays,r,ids,seed+10,a.spp))
                        self.trainer.model.reset();self.trainer.model.inference_mode(True)
                        self.trainer.model.prepare_kernels()
                        arr['prediction_native'].append(np.asarray(mi.render(pscene,seed=seed+20,spp=a.spp)).reshape(-1,3).copy())
                        self.trainer.model.inference_mode(False);self.trainer.model.reset()
                        arr['prediction_ray'].append(self.trainer.predict_array(r,ids,4,a.spp//4,seed+30))
                    arr={k:np.stack(v) for k,v in arr.items()}
                    np.savez_compressed(self.out/f'{key}.npz',pixel_ids=ids,target=target,**arr)
                    self.record(key+'_saved_vs_online',dict(**compare_radiance(arr['gt_native'],target,saved_spp=r['spp'],render_spp=a.spp),patch=patch))
                    self.record(key+'_gt_ray_vs_native',compare_radiance(arr['gt_ray'],arr['gt_native']))
                    self.record(key+'_prediction_ray_vs_native',compare_radiance(arr['prediction_ray'],arr['prediction_native']))
        self.restore_colours()

    def gradient_case(self,name,parameters,directions,evaluate,expect_zero=False):
        """Linear radiance functional: independent forward seed cannot bias its adjoint.

        FD primal uses the backward replay seed (NOT the forward seed). Matching
        seeds is essential for useful comparisons with the custom PRB operator.
        """
        originals=[p.detach().clone() for p in parameters]
        eps=[.001,.003,.01];ads=[];fds=[]
        try:
            for i in range(self.a.grad_repeats):
                primal=self.a.seed+700000+i*1000;backward=primal+137
                for p in parameters:p.grad=None
                val=evaluate(True,primal,backward)
                dr.backward(val)
                ad=sum(float((p.grad*d).sum().item()) if p.grad is not None else 0.
                       for p,d in zip(parameters,directions))
                if not np.isfinite(ad):raise FloatingPointError('Nonfinite AD')
                ads.append(ad);row=[]
                for e in eps:
                    vals=[]
                    for sign in [1,-1]:
                        with torch.no_grad():
                            for p,b,d in zip(parameters,originals,directions):p.copy_(b+sign*e*d)
                        with torch.no_grad(),dr.suspend_grad():
                            vals.append(float(evaluate(False,backward,0)[0]))
                    row.append((vals[0]-vals[1])/(2*e))
                fds.append(row)
                with torch.no_grad():
                    for p,b in zip(parameters,originals):p.copy_(b)
                print('[GRADIENT]',name,i+1,'AD',ad,'FD',row,flush=True)
        finally:
            with torch.no_grad():
                for p,b in zip(parameters,originals):p.copy_(b)
            for p in parameters:p.grad=None
        self.record(name,compare_gradient(ads,fds,eps,expect_zero))

    def control_gradient(self):
        from hktex.rendering.torch_texture_renderer import TorchTextureNetwork
        from hktex.utils import TraversableDict
        class Constant(torch.nn.Module):
            def __init__(self):
                super().__init__();self.rgb=torch.nn.Parameter(torch.tensor([.3,.4,.5],device='cuda'))
            def requires_face_ids(self):return False
            def forward(self,pts,**kwargs):return self.rgb.expand(len(pts),3)+pts*0
        net=Constant();wrapper=TorchTextureNetwork(net)
        tex=mi.load_dict({'type':'torch_texture'});tex.network=wrapper
        scene=mi.load_dict(dict(type='scene',integrator=dict(type='path',max_depth=2),
            sensor=dict(type='perspective',to_world=mi.ScalarTransform4f().look_at(origin=[0,0,3],target=[0,0,0],up=[0,1,0]),
                film=dict(type='hdrfilm',width=1,height=1,rfilter=dict(type='box')),sampler=dict(type='independent',sample_count=1)),
            emitter=dict(type='constant',radiance=1.),
            plane=dict(type='rectangle',bsdf=dict(type='diffuse',reflectance=tex))))
        params=mi.traverse(TraversableDict({'texture':[tex,mi.ParamFlags.Differentiable]}))
        dr.enable_grad(params['texture.grad_activator'])
        integrator=mi.load_dict(dict(type='rb_ray',integrator=dict(type='prb',max_depth=2)))
        n=self.a.grad_pixels*self.a.grad_spp
        ray=mi.RayDifferential3f(o=mi.Point3f(dr.zeros(mi.Float,n),dr.zeros(mi.Float,n),dr.full(mi.Float,3,n)),d=mi.Vector3f(0,0,-1))
        def evaluate(grad,seed,seed_grad):
            L=rendering.render_ray(scene,params=params if grad else None,ray=ray,
                integrator=integrator,spp=self.a.grad_spp,seed=seed,seed_grad=seed_grad)
            return dr.mean(dr.dot(L,mi.Color3f(.2,.3,.5)))
        for c in range(3):
            d=torch.zeros_like(net.rgb);d[c]=1
            self.gradient_case('bridge_diffuse_rgb_'+str(c),[net.rgb],[d],evaluate)
        # Hard clipping is in the original TorchTextureNetwork wrapper.
        with torch.no_grad():net.rgb[0]=-.2
        d=torch.zeros_like(net.rgb);d[0]=1
        self.gradient_case('bridge_expected_clamp_zero',[net.rgb],[d],evaluate,expect_zero=True)

    def gradients(self):
        self.control_gradient()
        t=self.trainer;m=t.model.model;r=self.views[0]
        ids=self.data.fixed_pixels(r,self.a.grad_pixels,self.a.seed)
        self.restore_colours()
        # First-hit information is used only to choose strongly observed colour directions.
        pos=mi.Point2f(mi.Float((ids%r['width']+.5)/r['width']),mi.Float((ids//r['width']+.5)/r['height']))
        ray,_=t.sensor(r).sample_ray(0.,.5,pos,mi.Point2f(.5));hit=t.scene.ray_intersect(ray)
        if not bool(dr.all(hit.is_valid())):raise ValueError('Gradient pixels miss mesh')
        pts=hit.p.torch().T.contiguous();fids=hit.prim_index.torch().int()
        with torch.no_grad():
            t.model.reset();_,idx,weights,_=t._ray_topk_from_first_hits(pts,fids)
            score=torch.zeros(m.N_sources,device='cuda')
            score.scatter_add_(0,idx.long().reshape(-1),weights.reshape(-1))
            selected=score.topk(min(8,len(score))).indices
        params=[m._mean_colour,m._kernel_colours]
        dirs=[]
        for label,c,sparse in [('mean_red',0,False),('kernel_red',0,True),('kernel_blue',2,True)]:
            dm,dc=torch.zeros_like(params[0]),torch.zeros_like(params[1])
            if sparse:dc[selected,c]=1
            else:dm[0,c]=1
            dirs.append((label,[dm,dc]))
        self.report['gradient_probe']=dict(view=r['id'],pixel_ids=ids.tolist(),kernel_rows=selected.cpu().tolist(),
            objective='mean(0.2 R + 0.3 G + 0.5 B), no L1/log; isolates render derivative',
            explanation='Original render_ray uses different forward/backward seeds; FD uses backward seed.')
        for neutral in [True,False]:
            mode='neutral_field' if neutral else 'checkpoint_field'
            self.restore_colours(neutral)
            with torch.no_grad():
                direct=t.model(pts,face_ids=fids)
                self.report[mode+'_probe_clipped_fraction']=float(((direct<=0)|(direct>=1)).any(-1).float().mean().item())
            for label,direction in dirs:
                # Deterministic PyTorch-only field test, then complete PRB bridge.
                originals=[p.detach().clone() for p in params]
                for p in params:p.grad=None
                t.model.reset();out=t.model(pts,face_ids=fids)
                obj=(out*torch.tensor([.2,.3,.5],device='cuda')).sum(-1).mean();obj.backward()
                ad=sum(float((p.grad*d).sum()) if p.grad is not None else 0. for p,d in zip(params,direction))
                fd=[]
                try:
                    for e in [.001,.003,.01]:
                        values=[]
                        for sign in [1,-1]:
                            with torch.no_grad():
                                for p,b,d in zip(params,originals,direction):p.copy_(b+sign*e*d)
                                t.model.reset();v=t.model(pts,face_ids=fids)
                                values.append(float((v*torch.tensor([.2,.3,.5],device='cuda')).sum(-1).mean()))
                        fd.append((values[0]-values[1])/(2*e))
                finally:
                    with torch.no_grad():
                        for p,b in zip(params,originals):p.copy_(b)
                self.record(mode+'_torch_'+label,compare_gradient([ad]*3,[fd]*3,[.001,.003,.01]))
                def evaluate(grad,seed,seed_grad):
                    t.model.inference_mode(False);t.model.reset()
                    # Fixed primary rays across +/- perturbations; random transport replay uses seed_grad.
                    rr=t.rays(r,ids,1,self.a.grad_spp,np.random.default_rng(self.a.seed+91))
                    L=rendering.render_ray(t.scene,params=t.grad_params if grad else None,ray=rr,
                        integrator=t.integrator,spp=self.a.grad_spp,seed=seed,seed_grad=seed_grad)
                    return dr.mean(dr.dot(L,mi.Color3f(.2,.3,.5)))
                self.gradient_case(mode+'_prb_'+label,params,direction,evaluate)
        self.restore_colours()

    def finish(self):
        if hasattr(self,'original'):
            self.restore_colours()
            for k,b in self.original.items():
                if not torch.equal(getattr(self.trainer.model.model,k),b):raise RuntimeError('In-memory restoration failed: '+k)
            if digest(Path(self.a.run)/'checkpoint.pt')!=self.report['checkpoint_sha256']:
                raise RuntimeError('Input checkpoint changed during audit')
            self.report['checkpoint_unchanged']=True
        self.report['status']=combined_status(r['status'] for r in self.report['checks'])
        self.report['ready_for_controlled_5000_comparison']=self.a.stage=='all' and self.report['status']=='passed'
        self.report['elapsed_seconds']=time.monotonic()-self.started
        write(self.out/'report.json',self.report)
        if self.temp:self.temp.cleanup()
        return self.report['status']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default='/root/hktex-work/hktex')
    p.add_argument('--run',required=True,help='Input image-training run with checkpoint.pt and observer/')
    p.add_argument('--dataset');p.add_argument('--cache');p.add_argument('--output',required=True)
    p.add_argument('--gpu',default='0');p.add_argument('--stage',choices=['all','radiance','gradients'],default='all')
    p.add_argument('--views',help='Default: first train, val and test view')
    for key,value in [('spp',64),('repeats',4),('patch-size',8),('grad-spp',64),('grad-pixels',16),('grad-repeats',3),('seed',27191)]:
        p.add_argument('--'+key,type=int,default=value)
    a=p.parse_args()
    if a.spp<4 or a.spp%4 or min(a.grad_spp,a.grad_pixels,a.patch_size)<1 or min(a.repeats,a.grad_repeats)<3:
        p.error('spp must be a multiple of 4; repeats >=3; other sizes positive')
    out=Path(a.output).resolve()
    if out.exists():p.error('--output must be a NEW directory')
    out.mkdir(parents=True);s=Suite(a,out)
    try:
        s.boot()
        for name,fn in [('radiance',s.radiance),('gradients',s.gradients)]:
            if a.stage not in ['all',name]:continue
            try:fn()
            except Exception as exc:
                s.record(name+'_exception',dict(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()))
                # Still collect the independent stage; never train automatically.
        status=s.finish()
    except Exception as exc:
        s.record('startup_or_finalization',dict(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()))
        s.report['status']='failed';s.report['ready_for_controlled_5000_comparison']=False
        write(out/'report.json',s.report);status='failed'
        if s.temp:s.temp.cleanup()
    print('[RESULT]',status,out/'report.json',flush=True)
    raise SystemExit({'passed':0,'failed':2,'inconclusive':3}[status])


if __name__=='__main__':main()
