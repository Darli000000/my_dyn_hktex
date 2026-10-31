#!/usr/bin/env python3
"""CUDA compatibility diagnostics using the repository's actual HKTex model and tracer.
No real-image optimization, ADC, geometry repair, or synthetic-light rendering.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np
from PIL import Image
from common import verify_dataset, rays, project, sha, write_json


def run(a,out,report):
    def stage(name):
        report['stage']=name;write_json(out/'report.json',report);print('[CHECK]',name,flush=True)
    stage('dataset_integrity')
    root=Path(a.dataset).resolve();m=verify_dataset(root);repo=Path(a.repo).resolve()
    with np.load(root/m['mesh'],allow_pickle=False) as data:g={k:data[k].copy() for k in data.files}
    v,f=g['vertices'],g['faces'];diag=float(np.linalg.norm(np.ptp(v,axis=0)))
    tolerance=max(diag*2e-5,1e-7)
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu;os.environ.setdefault('MPLBACKEND','Agg')
    sys.path.insert(0,str(repo))
    stage('cuda_imports')
    import torch
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')
    import drjit as dr
    if not torch.cuda.is_available():raise RuntimeError('Run in the HKTex CUDA environment, not VHAP')
    torch.cuda.set_device(0);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    report['environment']=dict(torch=str(torch.__version__),cuda=torch.version.cuda,mitsuba=mi.__version__,gpu=torch.cuda.get_device_name(0))
    rng=np.random.default_rng(a.seed)
    stage('mitsuba_camera_and_first_hits')
    shape=mi.Mesh('vhap',vertex_count=len(v),face_count=len(f),has_vertex_normals=True)
    params=mi.traverse(shape);params['vertex_positions']=v.ravel();params['faces']=f.astype(np.uint32).ravel();params['vertex_normals']=g['vertex_normals'].ravel();params.update()
    if not np.array_equal(np.asarray(params['faces']).reshape(-1,3),f):raise ValueError('Mitsuba reordered faces')
    scene=mi.load_dict({'type':'scene','mesh':shape})
    report['views']=[];probe_pts=[];probe_fids=[]
    for r in m['views']:
        cam=r['camera'];candidate=np.asarray(Image.open(root/r['mask']))>0
        with np.load(root/r['geometry'],allow_pickle=False) as geom:
            fid=geom['face_ids'];bary=geom['barycentric'];depth=geom['depth_cv']
        interior=(fid>=0)&(bary.min(-1)>1e-4)
        ids0=np.flatnonzero(candidate&interior);ids1=np.flatnonzero(interior)
        take=lambda ids,n:rng.choice(ids,min(n,len(ids)),replace=False)
        ids=np.unique(np.r_[take(ids0,a.samples),take(ids1,a.samples//2)])
        if not len(ids):raise ValueError('No interior mesh pixels for '+r['id'])
        origin,direction=rays(cam,ids)
        ray=mi.Ray3f(mi.Point3f(*[mi.Float(origin[:,i]) for i in range(3)]),mi.Vector3f(*[mi.Float(direction[:,i]) for i in range(3)]))
        hit=scene.ray_intersect(ray)
        if not bool(dr.all(hit.is_valid())):raise ValueError('CPU hit misses Mitsuba '+r['id'])
        xyz=np.asarray(hit.p).T.copy();got=np.asarray(hit.prim_index).astype(np.int64)
        expected=fid.ravel()[ids];bc=bary.reshape(-1,3)[ids];ref=(v[f[expected]]*bc[:,:,None]).sum(1)
        px,z=project(xyz,cam);y,x=np.divmod(ids,cam['w'])
        err=float(abs(px-np.c_[x+.5,y+.5]).max());xyz_err=float(np.linalg.norm(xyz-ref,axis=1).max());zerr=float(abs(z-depth.ravel()[ids]).max())
        wrong=int((got!=expected).sum())
        if err>.02 or xyz_err>tolerance or zerr>tolerance or wrong:raise ValueError(f'First hit mismatch {r["id"]}: pixel={err}, xyz={xyz_err}, depth={zerr}, wrong_faces={wrong}')
        # Negative rays: well outside mesh silhouette, avoiding shared-edge ambiguity.
        from PIL import ImageFilter
        dilated=np.asarray(Image.fromarray(((fid>=0)*255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5)))>0
        misses=take(np.flatnonzero(~dilated),64);false_hits=0
        if len(misses):
            oo,dd=rays(cam,misses)
            rr=mi.Ray3f(mi.Point3f(*[mi.Float(oo[:,i]) for i in range(3)]),mi.Vector3f(*[mi.Float(dd[:,i]) for i in range(3)]))
            false_hits=int(np.asarray(scene.ray_test(rr)).sum())
            if false_hits:raise ValueError('Expected-background rays hit mesh')
        report['views'].append(dict(id=r['id'],samples=len(ids),projection_max_px=err,position_max_abs=xyz_err,depth_max_abs=zerr,wrong_faces=wrong,background_false_hits=false_hits))
        if r['split']=='train':
            keep=candidate.ravel()[ids];probe_pts.extend(xyz[keep].tolist());probe_fids.extend(got[keep].tolist())
    if not probe_pts:raise ValueError('No training-view probe points')
    stage('hktex_mesh_and_geodesic_tracer')
    import hktex
    from hktex.modules.mesh import Mesh
    from hktex.modules.tracer import GPUGeodesicTracer
    from omegaconf import OmegaConf
    mesh=Mesh(v,f,g['face_normals'],g['vertex_normals'],uv=None,device='cuda:0')
    if not np.array_equal(mesh.faces.cpu().numpy(),f):raise ValueError('HKTex reordered faces')
    tracer=GPUGeodesicTracer(dict(avoid_holes=True,max_iterations=256),mesh)
    tensor=lambda x,dtype=torch.float32:torch.as_tensor(x,dtype=dtype,device='cuda:0')
    def check_trace(fid,bc,direction,label,expect_move):
        pos=(v[f[fid]]*bc[:,:,None]).sum(1);p=tensor(pos);fi=tensor(fid,torch.int32)
        q,qi=tracer.trace(p,fi,tensor(direction),bary_coords=tensor(bc))
        qn=q.cpu().numpy();qf=qi.cpu().numpy().astype(int)
        if not np.isfinite(qn).all() or (qf<0).any() or (qf>=len(f)).any():raise ValueError('Invalid trace '+label)
        if not np.array_equal(g['face_components'][qf],g['face_components'][fid]):raise ValueError('Trace jumped connected component')
        qb=mesh.cartesian_to_barycentric(q,mesh.get_face_vertices(qi)).cpu().numpy()
        recon=(v[f[qf]]*qb[:,:,None]).sum(1)
        residual=float(np.linalg.norm(recon-qn,axis=1).max());displacement=np.linalg.norm(qn-pos,axis=1)
        if not np.isfinite(qb).all() or qb.min() < -1e-3 or abs(qb.sum(1)-1).max()>1e-3 or residual>tolerance:raise ValueError('Trace left surface '+label)
        if expect_move and float(displacement.max())<diag*1e-8:raise ValueError('Nonzero tracer request did not move')
        if label=='zero' and float(displacement.max())>tolerance:raise ValueError('Zero trace changed position')
        return dict(case=label,count=len(fid),component_preserved=True,min_barycentric=float(qb.min()),max_surface_residual=residual,max_displacement=float(displacement.max()))
    choose=[]
    for ci in np.unique(g['face_components']):
        pool=np.flatnonzero(g['face_components']==ci);choose.extend(take(pool,16).tolist())
    choose=np.array(choose);tri=v[f[choose]];bc=np.full((len(choose),3),1/3)
    direction=(tri[:,1]-tri[:,0])*.005
    report['traces']=[check_trace(choose,bc,np.zeros_like(direction),'zero',False),check_trace(choose,bc,direction,'small_tangent_each_component',True)]
    bd=g['boundary_edges'];bd=bd[take(np.arange(len(bd)),64)]
    if len(bd):
        fi=bd[:,2];bc=np.zeros((len(bd),3));direction=[]
        for j,(aa,bb,ff) in enumerate(bd):
            loc=[int(np.flatnonzero(f[ff]==aa)[0]),int(np.flatnonzero(f[ff]==bb)[0])];other=({0,1,2}-set(loc)).pop()
            bc[j,loc]=.49;bc[j,other]=.02
            mid=(v[aa]+v[bb])/2;direction.append((mid-v[f[ff,other]])*.1)
        report['traces'].append(check_trace(fi,bc,np.array(direction),'outward_across_open_boundary',False))
    stage('spectral_precompute_and_model_initialization')
    cfg0=OmegaConf.load(repo/'configs/multiview_hktex_knn_ray_small.yaml');cfg0.data.mesh_path='unused-real-image-reference'
    cfg=OmegaConf.create(OmegaConf.to_container(cfg0.trainer.network,resolve=True))
    cfg.model.n_sources=a.sources;cfg.point_batching=256
    cfg.eigen_albo.use_precomputed=True;cfg.eigen_albo.error_if_not_precomputed=False;cfg.tracer.max_iterations=256
    # No reductions of eigencount or anisotropy grid: exercise the intended configuration.
    config_identity=OmegaConf.to_container(cfg,resolve=True);config_identity['eigen_albo'].pop('mesh_path',None)
    fingerprints={str(p.relative_to(repo)):sha(p) for p in sorted((repo/'hktex').rglob('*.py'))}
    identity=dict(mesh_sha256=sha(root/'mesh.npz'),network=config_identity,sources=fingerprints,environment=report['environment'])
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    cache=Path(a.cache_root).resolve()/key;cache.mkdir(parents=True,exist_ok=True)
    cfg.eigen_albo.mesh_path=str(cache/'geometry.npz');write_json(cache/'identity.json',identity)
    report['cache']=str(cache);report['network_config']=OmegaConf.to_container(cfg,resolve=True)
    write_json(out/'source_fingerprints.json',fingerprints);OmegaConf.save(cfg,out/'network.yaml')
    import fcntl
    with open(cache/'build.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        model=hktex.find('modules.heat-kernel-model-knn')(cfg,mesh=mesh)
    eig=model.eigalbo_interp
    for name in ['_iso_eigen_val','_iso_eigen_vec','_eigen_val','_eigen_vec','_mass','_local_direction','_iso_embeddings']:
        x=getattr(eig,name)
        if not bool(torch.isfinite(x).all()):raise FloatingPointError('Nonfinite '+name)
    ev=eig._iso_eigen_val.detach().cpu().numpy()
    report['spectrum']=dict(first_eigenvalues=ev[:12].tolist(),components=int(g['face_components'].max()+1),
        note='KNN implementation drops only one eigenmode; disconnected meshes need cross-component influence inspection.')
    with torch.no_grad():
        model.model._mean_colour.fill_(.35);model.model._kernel_colours.normal_(0,.003)
    pick=take(np.arange(len(probe_pts)),min(a.samples,256));pts=tensor(np.array(probe_pts)[pick]);fids=tensor(np.array(probe_fids)[pick],torch.int32)
    stage('surface_query_and_cross_component_influence')
    with torch.no_grad():
        model.reset();model.prepare_kernels();info=model.model.prepare_points(mesh,eig,fids,pts=pts)
        colour,_,idx,weight=model.model.diffuse_heat_kernels(eig,info)
        weights=weight.squeeze(-1).T;ki=idx.squeeze(-1).T.long()
        labels=tensor(g['face_components'],torch.long)
        kc=labels[model.model._kernel_face_ids.long()];pc=labels[fids.long()]
        wrong=kc[ki]!=pc[:,None];mass=weights.abs();den=mass.sum()
        contamination=float((mass*wrong).sum()/den.clamp_min(1e-30))
        report['kernel_components']=np.bincount(kc.cpu().numpy(),minlength=int(labels.max())+1).tolist()
        report['cross_component_weight_fraction']=contamination
        report['surface_query']=dict(rgb_min=float(colour.min()),rgb_max=float(colour.max()),mean_weight_sum=float(mass.sum(1).mean()),points=len(pts))
        if not torch.isfinite(colour).all() or not torch.isfinite(weights).all():raise FloatingPointError('Nonfinite surface query')
    stage('backward_and_optimizer_checks')
    model.reset();model.zero_grad(set_to_none=True)
    pred=model(pts,face_ids=fids)
    # Known synthetic diagnostic target, not a fit to held-out or real-image labels.
    target=torch.tensor([.2,.45,.6],device=pts.device).expand_as(pred)
    loss=((pred-target)**2).mean();loss.backward()
    norms={}
    for name,param in model.model.named_parameters():
        grad=param.grad
        norms[name]=None if grad is None else float(grad.norm())
        if grad is not None and not torch.isfinite(grad).all():raise FloatingPointError('Nonfinite gradient '+name)
    report['gradients']=norms
    if not norms.get('_kernel_colours') or not norms.get('_mean_colour'):raise RuntimeError('No kernel-colour/mean gradient')
    # Central finite difference on largest kernel-colour gradient, away from RGB clipping.
    p=model.model._kernel_colours;flat=int(p.grad.abs().argmax());analytical=float(p.grad.flatten()[flat]);eps=1e-3
    saved=p.detach().clone()
    def evaluate():
        model.reset()
        with torch.no_grad():return float(((model(pts,face_ids=fids)-target)**2).mean())
    with torch.no_grad():p.flatten()[flat]+=eps
    plus=evaluate()
    with torch.no_grad():p.copy_(saved);p.flatten()[flat]-=eps
    minus=evaluate()
    with torch.no_grad():p.copy_(saved)
    numerical=(plus-minus)/(2*eps);error=abs(numerical-analytical)
    report['finite_difference']=dict(analytical=analytical,numerical=numerical,absolute_error=error,epsilon=eps)
    if error>max(2e-5,.03*abs(analytical)):raise RuntimeError('Kernel colour gradient disagrees with finite difference')
    # One colour-only descent step; retain model state only inside this diagnostic process.
    opt=torch.optim.Adam([model.model._kernel_colours,model.model._mean_colour],lr=1e-3)
    opt.step();after=evaluate();report['colour_step']=dict(before=float(loss),after=after,max_colour_change=float((p-saved).abs().max()))
    if after>=float(loss) or report['colour_step']['max_colour_change']<=0:raise RuntimeError('Known-target colour step did not reduce loss')
    # Actual repository position optimizer, using freshly prepared barycentric cache.
    from hktex.modules.geodesic_opt import GeodesicOpt
    model.reset();model.zero_grad(set_to_none=True)
    pred=model(pts,face_ids=fids);((pred-target)**2).mean().backward()
    loc=model.model._kernel_locations;loc_before=loc.detach().clone();fid_before=model.model._kernel_face_ids.detach().clone()
    if loc.grad is None or not torch.isfinite(loc.grad).all():raise RuntimeError('Missing/invalid location gradient')
    geo=GeodesicOpt([{'params':[loc],'face_ids':[model.model._kernel_face_ids]}],tracer=model.tracer,lr=1e-3)
    geo.step();model.post_optimizer_step()
    newfid=model.model._kernel_face_ids
    if not torch.isfinite(loc).all() or (newfid<0).any() or (newfid>=len(f)).any():raise RuntimeError('Invalid location optimizer result')
    labels=tensor(g['face_components'],torch.long)
    if not torch.equal(labels[newfid.long()],labels[fid_before.long()]):raise RuntimeError('Position optimizer crossed a component')
    bc=mesh.cartesian_to_barycentric(loc,mesh.get_face_vertices(newfid))
    recon=(mesh.get_vertex_locations(mesh.get_face_vertices(newfid))*bc[:,:,None]).sum(1)
    residual=float((recon-loc).norm(dim=1).max())
    if not torch.isfinite(bc).all() or float(bc.min())<-.001 or residual>tolerance:raise RuntimeError('Position optimizer left surface')
    report['position_step']=dict(max_displacement=float((loc-loc_before).norm(dim=1).max()),min_barycentric=float(bc.min()),max_surface_residual=residual,step_size=1e-3)
    if not np.isfinite(evaluate()):raise RuntimeError('Nonfinite prediction after position update')
    warnings=[]
    if not m['checks']['has_region_labels']:warnings.append('No anatomical ROI: geometry-only package')
    if contamination>1e-3:warnings.append('Non-negligible cross-component kernel influence; inspect spectral behavior before training')
    if not norms.get('_kernel_locations'):warnings.append('Location gradient is zero on probe points')
    report.update(status='inconclusive' if warnings else 'passed',warnings=warnings,
        tested_scope='pixel centres -> Mitsuba first hits -> original HKTex field -> Torch gradients/optimizers; no PRB/Dr.Jit differentiation bridge',
        roi_visual_review_required=True,full_training_validated=False,mesh_modified=False,
        peak_cuda_gb=torch.cuda.max_memory_allocated()/1024**3)
    write_json(out/'report.json',report)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cache-root',default='/root/hktex-work/hktex/outputs/vhap-check-cache');p.add_argument('--gpu',default='0');p.add_argument('--sources',type=int,default=5000);p.add_argument('--samples',type=int,default=256);p.add_argument('--seed',type=int,default=0)
    a=p.parse_args();out=Path(a.output).resolve()
    if out.exists():p.error('Output must be new')
    if a.sources<32 or a.samples<16:p.error('sources >=32, samples >=16 required')
    out.mkdir(parents=True);report=dict(status='running',args=vars(a))
    try:run(a,out,report)
    except BaseException as e:
        report.update(status='failed',error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc());write_json(out/'report.json',report);raise
    print('[RESULT]',report['status'],out/'report.json',flush=True)
if __name__=='__main__':main()
