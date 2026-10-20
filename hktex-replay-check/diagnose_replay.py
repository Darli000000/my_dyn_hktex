"""Three read-only replays of ONE saved state. No training or source modification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime
import numpy as np
from recording import PARAMS, cpu, json_write, npz_write, sha256
from replay import stage_values, reconstruct_colour
from compare import compare


def fingerprint(value):
    arr=cpu(value)
    return dict(shape=list(arr.shape),dtype=str(arr.dtype),sha256=hashlib.sha256(arr.tobytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',required=True)
    p.add_argument('--repo',default='/root/hktex-work/hktex')
    p.add_argument('--cache')
    p.add_argument('--gpu',default='0')
    p.add_argument('--chunk',type=int,default=64)
    p.add_argument('--output')
    a=p.parse_args()
    if a.chunk<1:p.error('chunk must be positive; retain 64 to match the previous replay')
    root=Path(a.recording).resolve();repo=Path(a.repo).resolve()
    manifest=json.loads((root/'manifest.json').read_text());index=json.loads((root/'snapshots.json').read_text())
    final=next(row for row in reversed(index) if row['phase']=='final')
    periodic=next(row for row in reversed(index) if row['phase']=='post_update' and row['completed_updates']==final['completed_updates'])
    state=dict(np.load(root/'snapshots'/final['file'],allow_pickle=False))
    prior=dict(np.load(root/'snapshots'/periodic['file'],allow_pickle=False))
    if set(state)!=set(prior) or not all(state[k].dtype==prior[k].dtype and state[k].shape==prior[k].shape and state[k].tobytes()==prior[k].tobytes() for k in state):
        raise SystemExit('Snapshots are not byte-identical arrays; inspect before doing a same-state replay test.')
    old=[]
    for row in [periodic,final]:
        old.append(dict(np.load(root/'analysis'/(Path(row['file']).stem+'_fields.npz'),allow_pickle=False)))
    cache=Path(a.cache or manifest['cache_path']).resolve()
    if sha256(cache)!=manifest['cache_sha256']:raise SystemExit('Eigen cache hash mismatch')
    differences=[name for name,digest in manifest['source_hashes'].items() if not (repo/name).is_file() or sha256(repo/name)!=digest]
    if differences:raise SystemExit('Source changed since recording: '+', '.join(differences[:12]))
    out=Path(a.output).resolve() if a.output else root/('repeatability-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    out.mkdir(parents=True,exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    os.environ.setdefault('MPLBACKEND','Agg');sys.path.insert(0,str(repo))
    import torch
    from omegaconf import OmegaConf
    from hktex.modules.mesh import Mesh
    from hktex.modules.eigen_albo_knn import EigenAlboInterpolationKNN
    from hktex.modules.heat_kernel_texture_knn import HeatKernelTextureKNN
    if not torch.cuda.is_available():raise SystemExit('Use the existing server HKTex CUDA environment')
    torch.set_float32_matmul_precision(manifest['matmul_precision']);torch.backends.cuda.matmul.allow_tf32=manifest['allow_tf32']
    surface=dict(np.load(root/'surface.npz',allow_pickle=False))
    mesh=Mesh(surface['vertices'],surface['faces'],surface['fnorms'],surface['vnorms'],device='cuda')
    mc=OmegaConf.create(manifest['model_config']);mc.n_sources=1
    model=HeatKernelTextureKNN(mc,mesh)
    ec=OmegaConf.create(manifest['eigen_config'])
    with tempfile.TemporaryDirectory(prefix='hktex-check-') as td:
        ec.mesh_path=str(Path(td)/'mesh.obj');ec.precomputed_name='check';ec.use_precomputed=True;ec.error_if_not_precomputed=True
        (Path(td)/f'mesh_check_{ec.local_frames}.pt').symlink_to(cache)
        eig=EigenAlboInterpolationKNN(ec,mesh)
    faces=torch.as_tensor(surface['face_ids'],device='cuda',dtype=torch.long)
    bary=torch.as_tensor(surface['bary'],device='cuda',dtype=torch.float32)
    old_change=np.abs(old[0]['predicted']-old[1]['predicted']).max(1)
    audit_points=np.unique(np.r_[np.argsort(-old_change)[:16],np.arange(8)])
    hashes=[];runs=[]
    def build():
        for key in PARAMS:
            t=torch.as_tensor(state[key],device='cuda')
            setattr(model,key,torch.nn.Buffer(t) if key=='_kernel_face_ids' else torch.nn.Parameter(t,requires_grad=False))
        model.N_sources=len(state['stable_ids']);eig.reset();model.prepare_kernels(mesh,eig,save_barycentric=False)
        return dict(database=fingerprint(eig.faiss_index._db),grid_indices=fingerprint(eig.grid[0]),grid_weights=fingerprint(eig.grid[1]),
            source_phi=fingerprint(eig.knn_heat._cache.evecs_src),source_self_heat=fingerprint(eig.knn_heat._cache.src_self_heat))
    with torch.no_grad():
        hashes.append(build())
        embeddings=cpu(eig.faiss_index._db)
        npz_write(out/'source_embeddings.npz',embedding=embeddings,stable_ids=state['stable_ids'])
        for run_name,rebuild in [('A_first',False),('B_same_index',False),('C_rebuilt_index',True)]:
            if rebuild:hashes.append(build())
            pieces={};probes={}
            def append(dst,key,value):dst.setdefault(key,[]).append(cpu(value))
            for start in range(0,len(faces),a.chunk):
                end=min(start+a.chunk,len(faces))
                pts=model.prepare_points(mesh,eig,faces[start:end],barys=bary[start:end])
                values=stage_values(model,eig,dict(evals=pts['albo_evals'],evecs=pts['albo_evecs'],indices=pts['indices'],weights=pts['weights'],distances=pts['distances']))
                pred,_,selected,contrib=model.diffuse_heat_kernels(eig,pts);pred=model(pred)
                reconstructed,_,_,_=reconstruct_colour(values['alpha'],model.kernel_colours[pts['indices']],model._mean_colour,model.cfg.knn_inner_k)
                append(pieces,'predicted',pred);append(pieces,'indices',pts['indices'])
                append(pieces,'reconstructed',reconstructed)
                append(pieces,'selected',selected.squeeze(-1).transpose(0,1));append(pieces,'contrib',contrib.squeeze(-1).transpose(0,1))
                for key in ['raw','normalized','weighted','alpha','distance_squared','distance_weight','self_heat']:append(pieces,key,values[key])
                inside=audit_points[(audit_points>=start)&(audit_points<end)]
                if len(inside):
                    ix=torch.as_tensor(inside-start,device='cuda');probes.setdefault('point_ids',[]).append(inside)
                    for key in ['evals','phi_query','phi_source']:append(probes,key,values[key][ix])
                    append(probes,'indices',pts['indices'][ix]);append(probes,'predicted',pred[ix])
                if start==0 or end==len(faces) or end%1024==0:print('[CHECK]',run_name,end,'/',len(faces),flush=True)
            arrays={k:np.concatenate(v) for k,v in pieces.items()};probe_arrays={k:np.concatenate(v) for k,v in probes.items()}
            if not all(np.isfinite(v).all() for v in arrays.values()):raise RuntimeError('Non-finite capture')
            ix=arrays['indices']
            if ix.min()<0 or ix.max()>=model.N_sources or np.any(np.diff(np.sort(ix,axis=1),axis=1)==0):
                raise RuntimeError('Invalid or duplicate candidate indices in a KNN row')
            npz_write(out/(run_name+'.npz'),**arrays);npz_write(out/(run_name+'_probes.npz'),**probe_arrays);runs.append(arrays)
    summary=dict(snapshot=final['file'],snapshots_identical=True,chunk=a.chunk,cache_fingerprints=hashes,
        original_outputs=compare(old[0],old[1]),same_index_repeat=compare(runs[0],runs[1],embeddings),
        rebuilt_index_repeat=compare(runs[0],runs[2],embeddings if hashes[0]['database']==hashes[1]['database'] else None),
        run_A_vs_old_periodic=compare(runs[0],old[0]),run_A_vs_old_final=compare(runs[0],old[1]),
        within_run_decomposition_error=[float(np.abs(r['predicted']-r['reconstructed']).max()) for r in runs],
        note='A/B reuse one prepared index; C reloads identical model arrays and rebuilds. Captures add CPU synchronization; failure to reproduce does not prove the old run was stable.')
    json_write(out/'summary.json',summary)
    print(json.dumps(summary,indent=2));print('[CHECK] Output:',out,flush=True)


if __name__=='__main__':main()
