#!/usr/bin/env python3
"""Targeted follow-up: ROI backfaces, analytic in-face moves, boundary endpoints.
Reuses a stage12 dataset. No training, eigen precompute, geometry edits or package changes.
"""
import argparse
import inspect
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np
from PIL import Image
from common import verify_dataset, rays, sha, write_json

def motion_metrics(start, direction, end, same_face, diag):
    if not np.isfinite(end).all():
        return dict(count=len(start),all_analytic_endpoints_pass=False,error='Nonfinite tracer endpoints')
    requested=np.linalg.norm(direction,axis=1)
    actual=np.linalg.norm(end-start,axis=1)
    error=np.linalg.norm(end-(start+direction),axis=1)
    limit=np.maximum(diag*2e-7,requested*.02)
    good=(error<=limit)&same_face&np.isfinite(end).all(1)
    ratio=np.divide(actual,requested,out=np.zeros_like(actual),where=requested>0)
    return dict(count=len(start),bad_endpoints=int((~good).sum()),
        requested_min=float(requested.min()),requested_median=float(np.median(requested)),requested_max=float(requested.max()),
        actual_max=float(actual.max()),median_actual_over_requested=float(np.median(ratio)),
        max_endpoint_error=float(error.max()),all_analytic_endpoints_pass=bool(good.all()))

def run(a,out,r):
    def save(stage):
        r['stage']=stage;write_json(out/'report.json',r);print('[CHECK]',stage,flush=True)
    save('data_integrity')
    root=Path(a.dataset).resolve();m=verify_dataset(root)
    with np.load(root/m['mesh'],allow_pickle=False) as z:g={k:z[k].copy() for k in z.files}
    v,f,fn=g['vertices'],g['faces'],g['face_normals'];diag=float(np.linalg.norm(np.ptp(v,axis=0)))
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    sys.path.insert(0,str(Path(a.repo).resolve()))
    import torch
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')
    import hktex
    import digeo
    import digeo.ops
    from hktex.modules.mesh import Mesh
    from hktex.modules.tracer import GPUGeodesicTracer
    try:version=importlib.metadata.version('digeo')
    except importlib.metadata.PackageNotFoundError:version='unknown'
    source=inspect.getsourcefile(digeo.ops.trace_geodesics)
    r['environment']=dict(torch=str(torch.__version__),digeo=version,
        digeo_signature=str(inspect.signature(digeo.ops.trace_geodesics)),
        digeo_source_sha256=sha(source) if source else None,
        hktex_tracer_sha256=sha(Path(a.repo)/'hktex/modules/tracer.py'))
    if source:(out/'installed_digeo_geodesic.py.txt').write_text(Path(source).read_text())
    issues=[]
    save('all_candidate_pixels_mitsuba_first_hits')
    shape=mi.Mesh('vhap',vertex_count=len(v),face_count=len(f),has_vertex_normals=True)
    p=mi.traverse(shape);p['vertex_positions']=v.ravel();p['faces']=f.astype(np.uint32).ravel();p['vertex_normals']=g['vertex_normals'].ravel();p.update()
    scene=mi.load_dict({'type':'scene','mesh':shape});r['views']=[]
    for row in m['views']:
        mask=np.asarray(Image.open(root/row['mask']))>0;ids=np.flatnonzero(mask)
        with np.load(root/row['geometry'],allow_pickle=False) as z:
            ref_fid=z['face_ids'].ravel()[ids];bc=z['barycentric'].reshape(-1,3)[ids]
        counters=dict(view=row['id'],pixels=len(ids),misses=0,backfaces=0,below_min_cosine=0,
            disallowed_faces=0,interior_face_mismatch=0,min_front_cosine=1.)
        for start in range(0,len(ids),8192):
            sl=slice(start,start+8192);o,d=rays(row['camera'],ids[sl])
            ray=mi.Ray3f(mi.Point3f(*[mi.Float(o[:,i]) for i in range(3)]),mi.Vector3f(*[mi.Float(d[:,i]) for i in range(3)]))
            hit=scene.ray_intersect(ray);valid=np.asarray(hit.is_valid());fi=np.asarray(hit.prim_index).astype(np.int64)
            counters['misses']+=int((~valid).sum());fi=fi[valid]
            cos=-(fn[fi]*d[valid]).sum(1)
            if len(cos):counters['min_front_cosine']=min(counters['min_front_cosine'],float(cos.min()))
            counters['backfaces']+=int((cos<=0).sum())
            counters['below_min_cosine']+=int((cos<m['checks']['min_cosine']-1e-5).sum())
            counters['disallowed_faces']+=int((~g['allowed_faces'][fi]).sum())
            interior=bc[sl][valid].min(1)>1e-4
            counters['interior_face_mismatch']+=int(((fi!=ref_fid[sl][valid])&interior).sum())
        if any(counters[k] for k in ['misses','backfaces','below_min_cosine','disallowed_faces','interior_face_mismatch']):
            issues.append('ROI first-hit failure: '+row['id'])
        r['views'].append(counters);save('all_candidate_pixels: '+row['id'])
    save('tracer_analytic_endpoints')
    tensor=lambda x,dtype=torch.float32:torch.as_tensor(x,dtype=dtype,device='cuda:0')
    mesh=Mesh(v,f,fn,g['vertex_normals'],uv=None,device='cuda:0')
    tracer=GPUGeodesicTracer(dict(avoid_holes=True,max_iterations=256),mesh)
    rng=np.random.default_rng(0);chosen=[]
    for ci in np.unique(g['face_components']):
        pool=np.flatnonzero(g['face_components']==ci);chosen.extend(rng.choice(pool,min(16,len(pool)),replace=False))
    chosen=np.array(chosen);tri=v[f[chosen]];bc=np.full((len(chosen),3),1/3,np.float32)
    pos=(tri*bc[:,:,None]).sum(1);fi=tensor(chosen,torch.int32)
    def raw_trace(face,bary,direction,eps,avoid):
        points=digeo.MeshPointBatch(tensor(face,torch.int32),tensor(bary[:,1:]))
        endpoint,_=digeo.ops.trace_geodesics(tracer._digeo_mesh,points,tensor(direction),
            gradient='none',use_python=False,max_steps=256,save_parallel_transport=False,
            save_end_direction=False,debug=False,print_warnings=True,eps=eps,avoid_holes=avoid)
        return endpoint.interpolate(tracer._digeo_mesh).cpu().numpy(),endpoint.faces.cpu().numpy().astype(int)
    r['motion_cases']=[]
    for fraction in [0.,.005,.05,.2]:
        direction=(tri[:,1]-tri[:,0])*fraction
        # Even fraction .2 stays inside its initial face: bary=(1/3-f,1/3+f,1/3).
        q,qi=tracer.trace(tensor(pos),fi,tensor(direction),bary_coords=tensor(bc))
        case=motion_metrics(pos,direction,q.cpu().numpy(),qi.cpu().numpy()==chosen,diag)
        case.update(route='hktex_wrapper_default',edge_fraction=fraction);r['motion_cases'].append(case)
        if not case['all_analytic_endpoints_pass']:issues.append('Default tracer endpoint mismatch at fraction '+str(fraction))
        if fraction:
            for eps in [1e-4,1e-6,1e-8]:
                try:
                    q,qi=raw_trace(chosen,bc,direction,eps,True)
                    case=motion_metrics(pos,direction,q,qi==chosen,diag)
                except Exception as e:case=dict(error=f'{type(e).__name__}: {e}')
                case.update(route='direct_digeo_diagnostic_only',eps=eps,edge_fraction=fraction);r['motion_cases'].append(case)
        save('tracer_fraction: '+str(fraction))
    save('all_open_boundary_edges')
    bd=g['boundary_edges'];face=bd[:,2];bc=np.zeros((len(bd),3),np.float32);directions=[]
    for j,(aa,bb,ff) in enumerate(bd):
        loc=[int(np.flatnonzero(f[ff]==aa)[0]),int(np.flatnonzero(f[ff]==bb)[0])];other=({0,1,2}-set(loc)).pop()
        bc[j,loc]=.49;bc[j,other]=.02
        directions.append(((v[aa]+v[bb])/2-v[f[ff,other]])*.1)
    directions=np.array(directions);start=(v[f[face]]*bc[:,:,None]).sum(1);r['boundary_cases']=[]
    for eps,avoid in [(-1.,True),(-1.,False),(1e-6,True),(1e-6,False)]:
        try:
            q,qi=raw_trace(face,bc,directions,eps,avoid)
            if not np.isfinite(q).all():raise ValueError('Nonfinite boundary endpoints')
            valid=np.isfinite(q).all(1)&(qi>=0)&(qi<len(f));safe=np.clip(qi,0,len(f)-1)
            bary=mesh.cartesian_to_barycentric(tensor(q),mesh.get_face_vertices(tensor(safe,torch.int32))).cpu().numpy()
            residual=np.linalg.norm((v[f[safe]]*bary[:,:,None]).sum(1)-q,axis=1)
            onmesh=valid&np.isfinite(bary).all(1)&(bary.min(1)>=-.001)&(abs(bary.sum(1)-1)<.001)&(residual<=diag*2e-5)
            same=g['face_components'][safe]==g['face_components'][face]
            excess=np.linalg.norm(q-start,axis=1)-np.linalg.norm(directions,axis=1)
            case=dict(count=len(face),invalid_endpoints=int((~onmesh).sum()),component_jumps=int((~same).sum()),
                displacement_exceeds_requested_count=int((excess>diag*2e-5).sum()),max_displacement=float(np.linalg.norm(q-start,axis=1).max()))
            # Retain per-edge endpoints; passing only establishes endpoint safety, not stop-vs-slide semantics.
            np.savez_compressed(out/f'boundary_eps{eps}_avoid{avoid}.npz',edges=bd,start=start,direction=directions,end=q,end_face=qi)
            if eps==-1 and avoid and any(case[k] for k in ['invalid_endpoints','component_jumps','displacement_exceeds_requested_count']):
                issues.append('Default boundary handling needs inspection')
        except Exception as e:
            case=dict(error=f'{type(e).__name__}: {e}')
            if eps==-1 and avoid:issues.append('Default boundary trace failed')
        case.update(eps=eps,avoid_holes=avoid);r['boundary_cases'].append(case);save('boundary_case_complete')
    r.update(status='inconclusive' if issues else 'passed',issues=issues,
        scope='All current mask pixel centres; in-face analytic moves; all open-edge endpoint probes. No diffusion solver or full training test.',
        parameter_changes_applied=False,
        note='Epsilon/avoid_holes sweeps are diagnostics only; do not adopt a setting solely because a small-motion test passes.')
    write_json(out/'report.json',r)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--gpu',default='0')
    a=p.parse_args();out=Path(a.output).resolve()
    if out.exists():p.error('Choose a new output directory')
    out.mkdir(parents=True);r=dict(status='running',args=vars(a))
    try:run(a,out,r)
    except BaseException as e:
        r.update(status='failed',error=str(e),traceback=traceback.format_exc());write_json(out/'report.json',r);raise
    print('[RESULT]',r['status'],out/'report.json',flush=True)
if __name__=='__main__':main()
