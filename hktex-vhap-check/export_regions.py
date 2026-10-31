#!/usr/bin/env python3
"""Run in VHAP environment. Export face-region labels without changing geometry."""
import argparse
import os
from pathlib import Path
import sys
import numpy as np
from common import read_obj, sha, write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vhap',required=True);p.add_argument('--mesh',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.vhap).resolve();out=Path(a.output).resolve();mesh=Path(a.mesh).resolve()
    if out.suffix!='.npz':p.error('Output must end in .npz')
    if out.exists():p.error('Output already exists; choose a new path')
    v,f=read_obj(mesh)
    sys.path.insert(0,str(root));os.chdir(root)
    from vhap.model.flame import FlameHead
    # Matches this run and the official mesh exporter. Exact topology is checked below.
    model=FlameHead(300,100,add_teeth=True)
    ref=model.faces.detach().cpu().numpy()
    if len(v)!=len(model.v_template) or not np.array_equal(ref,f):
        raise ValueError('VHAP topology differs from exported mesh; stop, do not guess vertex mappings')
    groups={k:x.detach().cpu().numpy() for k,x in model.mask.v.named_buffers()}
    include=['face'];exclude=['neck','scalp','boundary','eye_region','left_eyeball','right_eyeball','lips','lip_inside','teeth']
    missing=set(include+exclude)-groups.keys()
    if missing:raise ValueError('Missing region labels: '+str(sorted(missing)))
    selected=np.zeros(len(v),bool);selected[groups['face']]=True
    for k in exclude:selected[groups[k]]=False
    allowed=selected[f].all(1)
    if not allowed.any():raise ValueError('Empty face ROI')
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,faces=f,vertex_count=np.array(len(v)),allowed_faces=allowed,
                        **{'vertex_'+k:groups[k] for k in include+exclude})
    write_json(out.with_suffix('.json'),dict(mesh_sha256=sha(mesh),faces=len(f),allowed_faces=int(allowed.sum()),
        include=include,exclude=exclude,rule='all triangle vertices in face minus exclusions',
        disclaimer='Conservative anatomical candidate only; photograph occlusion still requires visual review',
        flame_source_sha256=sha(root/'vhap/model/flame.py')))
    print('[REGIONS]',out,'allowed faces:',int(allowed.sum()))
if __name__=='__main__':main()
