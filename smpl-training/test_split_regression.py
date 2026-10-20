"""CPU regression using the real density-controller methods; no FAISS/DiGeo needed."""
import argparse
import ast
from abc import abstractmethod
from dataclasses import dataclass
import math
from pathlib import Path
import torch
from apply_split_fix import OLD, NEW, patched_text

class BaseObject:
    @dataclass
    class Config: pass
    def configure(self): pass
class GeodesicOpt: pass

def controller_class(source):
    tree=ast.parse(source)
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='BaseDensityController')
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module=ast.fix_missing_locations(ast.Module(body=[future,cls],type_ignores=[]))
    ns=dict(torch=torch,math=math,dataclass=dataclass,abstractmethod=abstractmethod,
            BaseObject=BaseObject,GeodesicOpt=GeodesicOpt,
            rotate_on_plane=lambda direction,normal,angle:direction)
    exec(compile(module,'density-controller-under-test','exec'),ns)
    return ns['BaseDensityController']

class Mesh:
    def __init__(self):
        self.verts=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.], [3.,0.,0.],[4.,0.,0.],[3.,1.,0.]])
        self.faces=torch.tensor([[0,1,2],[3,4,5]])
        self.fnorms=torch.tensor([[0.,0.,1.],[0.,0.,1.]])
    def get_face_vertices(self,ids):return self.faces[ids.long()]
    def cartesian_to_barycentric(self,pts,ids):
        # Both right triangles have unit X/Y edges.
        d=pts-self.verts[ids[:,0]]
        return torch.stack([1-d[:,0]-d[:,1],d[:,0],d[:,1]],dim=1)

class Model(torch.nn.Module):
    def __init__(self,mesh):
        super().__init__();self.N_sources=3
        faces=torch.tensor([0,1,0])
        bary=torch.tensor([[.8,.1,.1],[.2,.3,.5],[.3,.3,.4]])
        self._kernel_locations=torch.nn.Parameter((mesh.verts[mesh.faces[faces]]*bary[:,:,None]).sum(1))
        self._thresholds=torch.nn.Parameter(torch.full((3,),.5))
        self._angles=torch.nn.Parameter(torch.zeros(3))
        self._kernel_colours=torch.nn.Parameter(torch.zeros(3,3))
        self._mean_colour=torch.nn.Parameter(torch.zeros(1,3))
        self.register_buffer('_kernel_face_ids',faces)
        self.register_buffer('_dummy',torch.tensor(0.))
        self.register_buffer('_error_accumulator',torch.ones(3,1))
    @property
    def kernel_locations(self):return self._kernel_locations
    @property
    def thresholds(self):return self._thresholds
    @property
    def angles(self):return self._angles

class Tracer:
    def __init__(self,mesh):self.mesh=mesh;self.calls=0
    def trace(self,coords,faces,direction,*,bary_coords):
        reconstructed=(self.mesh.verts[self.mesh.faces[faces]]*bary_coords[:,:,None]).sum(1)
        error=float((reconstructed-coords).norm(dim=1).max())
        if error>1e-5:raise AssertionError(f'STALE_GEOMETRY: position reconstruction error {error}')
        self.calls+=1
        return coords+direction,faces.clone()

def run_case(source,mode):
    mesh=Mesh();model=Model(mesh);controller=controller_class(source)()
    opt=torch.optim.Adam([{'params':[p],'name':'model.'+n} for n,p in model.named_parameters()],lr=0.)
    sum(p.sum() for p in model.parameters()).backward();opt.step();opt.zero_grad()
    controller.configure(mesh,model,[opt])
    kernel_info={'vert_idx':mesh.faces[model._kernel_face_ids].clone(),
                 'barycentric_coords':mesh.cartesian_to_barycentric(model.kernel_locations,mesh.faces[model._kernel_face_ids]).clone()}
    if mode in ('prune','prune_clone','prune_clone_poison'):
        controller._remove_kernels(torch.tensor([True,False,False]))
    if mode in ('prune_clone','prune_clone_poison','clone'):
        mask=torch.zeros(model.N_sources,dtype=torch.bool);mask[-1]=True;controller.clone(mask)
    if mode=='prune_clone_poison':
        kernel_info['barycentric_coords'].fill_(float('nan'));kernel_info['vert_idx'].fill_(-99)
    tracer=Tracer(mesh)
    class Interp:
        def barycentric_local_directions_gaussians(self,*,barycentric_coords,vert_idx):
            assert torch.equal(vert_idx,mesh.faces[model._kernel_face_ids[:1]]), 'STALE_VERTEX_IDS: cached vertices belong to a different current face'
            assert torch.isfinite(barycentric_coords).all()
            return torch.tensor([[1.,0.,0.]])
    before=model.N_sources
    selected=torch.zeros(before,dtype=torch.bool);selected[0]=True
    controller.split(selected,Interp(),kernel_info,tracer,.01)
    assert tracer.calls==1 and model.N_sources==before+1
    for name,p in model.named_parameters():
        if name!='_mean_colour':assert p.shape[0]==model.N_sources
        for key in ('exp_avg','exp_avg_sq'):assert opt.state[p][key].shape==p.shape
    assert len(model._kernel_face_ids)==model.N_sources

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='/root/hktex-work/hktex');args=p.parse_args()
    source=(Path(args.repo)/'hktex/density_controllers/base.py').read_text()
    original=source.replace(NEW,OLD,1) if NEW in source else source
    fixed=patched_text(original)
    assert patched_text(fixed)==fixed
    # Old code takes a pre-pruning vertex triple for a different current face.
    try:run_case(original,'prune')
    except AssertionError as exc:
        assert str(exc).startswith(('STALE_VERTEX_IDS:', 'STALE_GEOMETRY:')), str(exc)
        print('PASS: reproduced original bug:', exc)
    else:raise AssertionError('Original code unexpectedly passed the regression')
    for mode in ('plain','clone','prune','prune_clone','prune_clone_poison'):
        run_case(fixed,mode);print('PASS:',mode)
    print('All CPU regressions passed. GPU tracing and full training still require the server run.')

if __name__=='__main__':main()
