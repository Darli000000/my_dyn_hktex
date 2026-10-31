"""Fixed-geometry, fixed-lighting surface RGB runtime. No path-tracing/albedo claim."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from common import REGIONS, verify_dataset, rays, sha, write_json

def tensor(x,dtype=None):
    import torch
    return torch.as_tensor(x,dtype=dtype or torch.float32,device='cuda:0')

def load_views(root,m,g,report,save):
    import mitsuba as mi
    v,f=g['vertices'],g['faces'];fn=g['face_normals']
    shape=mi.Mesh('vhap',vertex_count=len(v),face_count=len(f),has_vertex_normals=True)
    params=mi.traverse(shape);params['vertex_positions']=v.ravel();params['faces']=f.astype(np.uint32).ravel();params['vertex_normals']=g['vertex_normals'].ravel();params.update()
    if not np.array_equal(np.asarray(params['faces']).reshape(-1,3),f):raise ValueError('Mitsuba changed face order')
    scene=mi.load_dict({'type':'scene','mesh':shape});views=[];report['views']=[]
    for row in m['views']:
        labels=np.asarray(Image.open(root/row['regions']));mask=np.asarray(Image.open(root/row['mask']))>0
        if not np.array_equal(mask,labels>0):raise ValueError('Mask/region mismatch')
        ids=np.flatnonzero(mask);rgb=np.load(root/row['rgb'],allow_pickle=False).reshape(-1,3)[ids]
        with np.load(root/row['geometry'],allow_pickle=False) as z:ref=z['face_ids'].ravel()[ids];bc=z['barycentric'].reshape(-1,3)[ids]
        points=np.empty((len(ids),3),np.float32);fids=np.empty(len(ids),np.int32)
        rec=dict(id=row['id'],pixels=len(ids),misses=0,backfaces=0,disallowed=0,wrong_interior_faces=0,position_error_max=0.)
        for start in range(0,len(ids),8192):
            sl=slice(start,start+8192);o,d=rays(row['camera'],ids[sl])
            ray=mi.Ray3f(mi.Point3f(*[mi.Float(o[:,j]) for j in range(3)]),mi.Vector3f(*[mi.Float(d[:,j]) for j in range(3)]))
            hit=scene.ray_intersect(ray);valid=np.asarray(hit.is_valid())
            if not valid.all():raise ValueError('Missing ROI first hit: '+row['id'])
            fi=np.asarray(hit.prim_index).astype(np.int32);xyz=np.asarray(hit.p).T.copy()
            cos=-(fn[fi]*d).sum(1)
            rec['backfaces']+=int((cos<m['checks']['min_cosine']-1e-5).sum())
            rec['disallowed']+=int((g['face_regions'][fi]!=labels.ravel()[ids[sl]]).sum())
            rec['wrong_interior_faces']+=int(((fi!=ref[sl])&(bc[sl].min(1)>1e-4)).sum())
            expected=(v[f[ref[sl]]]*bc[sl,:,None]).sum(1)
            rec['position_error_max']=max(rec['position_error_max'],float(np.linalg.norm(xyz-expected,axis=1).max()))
            points[sl]=xyz;fids[sl]=fi
        rec['regions']={name:int((labels.ravel()[ids]==k).sum()) for k,name in REGIONS.items()}
        report['views'].append(rec);save('ray_check_'+row['id'])
        if rec['backfaces'] or rec['disallowed'] or rec['wrong_interior_faces'] or rec['position_error_max']>max(float(np.linalg.norm(np.ptp(v,axis=0)))*2e-5,1e-7):
            raise ValueError('ROI first-hit checks failed: '+row['id'])
        views.append(dict(row=row,ids=ids,points=points,faces=fids,rgb=rgb,regions=labels.ravel()[ids]))
    return views

def make_model(repo,cache_root,g,out):
    import torch
    import hktex
    from hktex.modules.mesh import Mesh
    from hktex.modules.tracer import GPUGeodesicTracer
    from omegaconf import OmegaConf
    if 'eps' not in GPUGeodesicTracer.Config.__dataclass_fields__:
        raise RuntimeError('Apply hktex-tracer-eps.patch before running; patched tracer config is required')
    source=OmegaConf.load(repo/'configs/multiview_hktex_knn_ray_small.yaml');source.data.mesh_path='unused-real-face'
    cfg=OmegaConf.create(OmegaConf.to_container(source.trainer.network,resolve=True))
    cfg.model.n_sources=5000;cfg.point_batching=512
    cfg.tracer.eps=1e-6;cfg.tracer.avoid_holes=False;cfg.tracer.max_iterations=256
    cfg.eigen_albo.use_precomputed=True;cfg.eigen_albo.error_if_not_precomputed=False
    fingerprints={str(p.relative_to(repo)):sha(p) for p in sorted((repo/'hktex').rglob('*.py'))}
    eigen=OmegaConf.to_container(cfg.eigen_albo,resolve=True);eigen.pop('mesh_path',None)
    geomhash=hashlib.sha256(g['vertices'].tobytes()+g['faces'].tobytes()+g['vertex_normals'].tobytes()).hexdigest()
    identity=dict(geometry_sha256=geomhash,eigen=eigen,source=fingerprints,torch=str(torch.__version__))
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    cache=cache_root/key;cache.mkdir(parents=True,exist_ok=True);write_json(cache/'identity.json',identity)
    cfg.eigen_albo.mesh_path=str(cache/'geometry.npz');OmegaConf.save(cfg,out/'network.yaml');write_json(out/'source_fingerprints.json',fingerprints)
    mesh=Mesh(g['vertices'],g['faces'],g['face_normals'],g['vertex_normals'],uv=None,device='cuda:0')
    import fcntl
    with open(cache/'build.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);model=hktex.find('modules.heat-kernel-model-knn')(cfg,mesh=mesh)
    with torch.no_grad():
        model.model._mean_colour.fill_(.35);model.model._kernel_colours.normal_(0,.003)
    return model,mesh,source.trainer.optimizers

def pool_views(views,split):
    rows=[v for v in views if v['row']['split']==split]
    if not rows:raise ValueError('Missing split: '+split)
    return {k:np.concatenate([v[k] for v in rows]) for k in ['points','faces','rgb','regions']}

def sample_stratified(pool,rng,quotas=(512,256,128,128)):
    """Oversample small regions for coverage; region-count weights preserve pixel MSE."""
    picks=[];weights=[]
    for k,n in zip(REGIONS,quotas):
        ids=np.flatnonzero(pool['regions']==k)
        if not len(ids):raise ValueError('No training pixels for '+REGIONS[k])
        picks.append(rng.choice(ids,n,replace=True));weights.extend([len(ids)/len(pool['regions'])/n]*n)
    ids=np.concatenate(picks)
    return {k:pool[k][ids] for k in pool},np.array(weights,np.float32)

def predict(model,points,faces):
    import torch
    outputs=[]
    with torch.no_grad():
        model.reset();model.prepare_kernels();old=model.inference_mode(True)
        try:
            for start in range(0,len(points),512):
                p=model(tensor(points[start:start+512]),face_ids=tensor(faces[start:start+512],torch.int32))
                if not torch.isfinite(p).all():raise FloatingPointError('Nonfinite prediction')
                outputs.append(p.detach().cpu().numpy())
        finally:model.inference_mode(old);model.reset()
    return np.concatenate(outputs) if outputs else np.empty((0,3),np.float32)

def metrics(pred,target,labels):
    errors=((pred.astype(np.float64)-target)**2).mean(1)
    def stat(mask):
        n=int(mask.sum())
        if not n:return dict(pixels=0,mse=None,psnr=None)
        mse=float(errors[mask].mean())
        return dict(pixels=n,mse=mse,psnr=float(-10*np.log10(max(mse,1e-12))))
    return dict(all=stat(np.ones(len(labels),bool)),regions={name:stat(labels==k) for k,name in REGIONS.items()})
