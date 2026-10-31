#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys
import numpy as np
from common import read_obj, face_regions, region_counts, topology, sha, write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--mesh',required=True);p.add_argument('--output',required=True)
    src=p.add_mutually_exclusive_group(required=True);src.add_argument('--source-regions');src.add_argument('--vhap')
    a=p.parse_args();mesh=Path(a.mesh).resolve();out=Path(a.output).resolve()
    if out.exists() or out.suffix!='.npz':p.error('Use a new .npz output')
    v,f=read_obj(mesh)
    if a.source_regions:
        with np.load(a.source_regions,allow_pickle=False) as z:
            if int(z['vertex_count'])!=len(v) or not np.array_equal(z['faces'],f):raise ValueError('Source label topology mismatch')
            groups={k[7:]:z[k].copy() for k in z.files if k.startswith('vertex_') and k!='vertex_count'}
        provenance=dict(source_regions_sha256=sha(a.source_regions))
    else:
        root=Path(a.vhap).resolve();sys.path.insert(0,str(root));os.chdir(root)
        from vhap.model.flame import FlameHead
        model=FlameHead(300,100,add_teeth=True)
        if len(model.v_template)!=len(v) or not np.array_equal(model.faces.detach().cpu().numpy(),f):raise ValueError('VHAP topology mismatch')
        groups={k:x.detach().cpu().numpy() for k,x in model.mask.v.named_buffers()}
        provenance=dict(flame_source_sha256=sha(root/'vhap/model/flame.py'))
    labels=face_regions(f,len(v),groups);_,_,_,components,_=topology(v,f)
    mapped={k:np.unique(components[labels==k]).tolist() for k in range(1,5)}
    if any(len(c)!=1 for c in mapped.values()) or mapped[1]!=mapped[2] or len({mapped[k][0] for k in [1,3,4]})!=3:
        raise ValueError('Expected head/eye-surround together and two separate eye components: '+str(mapped))
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,faces=f,vertex_count=len(v),face_regions=labels,allowed_faces=labels>0)
    write_json(out.with_suffix('.json'),dict(region_faces=region_counts(labels),component_mapping=mapped,mesh_sha256=sha(mesh),**provenance))
    print('[REGIONS]',region_counts(labels),flush=True)
if __name__=='__main__':main()
