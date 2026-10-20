"""CPU verification: real source ADC methods, recording transparency, and math decomposition."""
import argparse
import ast
from abc import abstractmethod
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import types
from unittest.mock import patch
import numpy as np
import torch
import trimesh
from PIL import Image
from omegaconf import OmegaConf
from recording import Lineage, sample_surface, install
from replay import reconstruct_colour, stage_values


def source_math_test(repo):
    """Exercise the repository's heat implementation and KNN colour reduction on CPU."""
    def extracted(path, ns, names=None):
        tree=ast.parse(path.read_text())
        body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and (names is None or n.name in names)]
        module=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*body],type_ignores=[]))
        exec(compile(module,str(path),'exec'),ns)
    ops=dict(torch=torch)
    extracted(repo/'hktex/knn_heat/knn_heat/_ops/heat.py',ops)
    ns=dict(torch=torch,dataclass=dataclass,_ops=types.SimpleNamespace(**ops))
    extracted(repo/'hktex/knn_heat/knn_heat/knn_heat.py',ns)
    heat=ns['SpectralKnnHeat'](types.SimpleNamespace(compile_heat=False,eps=1e-8))
    tree=ast.parse((repo/'hktex/modules/heat_kernel_texture_knn.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HeatKernelTextureKNN')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='diffuse_heat_kernels')
    module=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[]))
    reduction=dict(torch=torch);exec(compile(module,'actual_knn_colour','exec'),reduction)
    torch.manual_seed(74)
    ev=torch.rand(7,16)*80;src=torch.randn(7,16);t=torch.tensor(.06245)
    heat.build(ev,src,t)
    idx=torch.stack([torch.randperm(7)[:5] for _ in range(4)])
    query=torch.randn(4,5,16);weights=torch.rand(4,5)
    cfg=types.SimpleNamespace(power_diffused_diracs=1,knn_inner_k=3)
    model=types.SimpleNamespace(cfg=cfg,diff_times=t,kernel_colours=torch.rand(7,3)*2-1,
        _mean_colour=torch.rand(1,3),thresholds=torch.rand(7),sharpnesses=torch.rand(7)*10)
    model.kernel_filter_func=lambda x,epsilon,sharpness:torch.sigmoid((x-epsilon)*sharpness)
    eig=types.SimpleNamespace(knn_heat=heat,diffuse_heat=lambda e,v,i,weights:heat.query(i,e,v,weights_post_diff=weights))
    for w in (weights,None):
        info=dict(evals=ev[idx],evecs=query,indices=idx,weights=w,distances=torch.rand(4,5))
        stages=stage_values(model,eig,info)
        pts=dict(albo_evals=ev[idx],albo_evecs=query,indices=idx,weights=w,distances=info['distances'])
        actual,*_=reduction['diffuse_heat_kernels'](model,eig,pts)
        rebuilt,*_=reconstruct_colour(stages['alpha'],model.kernel_colours[idx],model._mean_colour,3)
        torch.testing.assert_close(actual,rebuilt)
        expected=(torch.exp(-ev[idx]*t)*query*src[idx]).sum(-1)
        torch.testing.assert_close(stages['raw'],expected)
        _,norm=heat.query(idx,ev[idx],query,weights_post_diff=w)
        torch.testing.assert_close(stages['weighted'],norm)
    print('PASS: actual source spectral heat, self-normalization and KNN colour reduction (CPU, eager)')


class BaseObject:
    @dataclass
    class Config: pass
    def configure(self,*a,**kw): pass


def actual_controller(source):
    cls=next(x for x in ast.parse(source).body if isinstance(x,ast.ClassDef) and x.name=='BaseDensityController')
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module=ast.fix_missing_locations(ast.Module(body=[future,cls],type_ignores=[]))
    ns=dict(torch=torch,math=math,dataclass=dataclass,abstractmethod=abstractmethod,
        BaseObject=BaseObject,GeodesicOpt=type('GeodesicOpt',(),{}),rotate_on_plane=lambda d,n,a:d)
    exec(compile(module,'actual_density_controller','exec'),ns)
    return ns['BaseDensityController']


class Mesh:
    def __init__(self):
        self.verts=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        self.faces=torch.tensor([[0,1,2]])
        self.fnorms=torch.tensor([[0.,0.,1.]])
        self.vnorms=self.fnorms.repeat(3,1)
    def get_face_vertices(self,f):return self.faces[f.long()]
    def cartesian_to_barycentric(self,p,ids):return torch.stack([1-p[:,0]-p[:,1],p[:,0],p[:,1]],dim=1)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.N_sources=3
        self.cfg=OmegaConf.create(dict(n_sources=3,out_net=False,normalize_colours=False,mass_type='one'))
        for key,shape in [('_kernel_locations',(3,3)),('_kernel_colours',(3,3)),('_mean_colour',(1,3)),
                          ('_angles',(3,)),('_anisotropies',(3,)),('_thresholds',(3,)),('_sharpnesses',(3,))]:
            setattr(self,key,torch.nn.Parameter(torch.rand(shape)*.2+.1))
        with torch.no_grad():self._kernel_locations[:,2]=0
        self.register_buffer('_kernel_face_ids',torch.zeros(3,dtype=torch.long))
        self.register_buffer('_dummy',torch.tensor(0.))
        self.register_buffer('_error_accumulator',torch.ones(3,1))
        self.register_buffer('_hit_accumulator',torch.ones(3,1))
        self.register_buffer('_contrib_accumulator',torch.ones(3,1))
    @property
    def kernel_locations(self):return self._kernel_locations
    @property
    def thresholds(self):return self._thresholds
    @property
    def angles(self):return self._angles
    def post_optimizer_step(self):
        with torch.no_grad():self._thresholds.clamp_(.01,.99)


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='/root/hktex-work/hktex');a=p.parse_args()
    repo=Path(a.repo).resolve();C=actual_controller((repo/'hktex/density_controllers/base.py').read_text())
    class Densifier(C):
        def _densify(self):
            mask=torch.zeros(self._model.N_sources,dtype=torch.bool);mask[0]=True;self.clone(mask)
    class Pruner(C):
        def prune_redundant_kernels(self):
            mask=torch.zeros(self._model.N_sources,dtype=torch.bool);mask[0]=True;self._remove_kernels(mask)
    class Trainer:
        def __init__(self,cache):
            self.mesh=Mesh();self.model=Model()
            tm=trimesh.Trimesh(self.mesh.verts.numpy(),self.mesh.faces.numpy(),process=False)
            tm.visual=trimesh.visual.TextureVisuals(uv=np.array([[0,0],[1,0],[0,1]]),image=Image.new('RGB',(4,4),(50,100,150)))
            self.datamodule=types.SimpleNamespace(mesh=tm)
            self.eigalbo_interp=types.SimpleNamespace(cfg=OmegaConf.create(dict(mesh_path=str(cache/'mesh.obj'),precomputed_name='eigen',local_frames='test')))
            self.optimizers=[torch.optim.Adam([{'params':[v],'name':'model.'+k} for k,v in self.model.named_parameters()],lr=.002)]
            self.dc=Densifier();self.pc=Pruner()
            for dc in [self.dc,self.pc]:
                dc.configure(self.mesh,self.model,self.optimizers)
                dc.cfg=OmegaConf.create(dict(error_threshold=.1,max_kernels=20,max_densify_ratio=.4,accumulation_interval=2))
        def compute_loss(self,colours,gt_colours,data=None):
            e=((colours-gt_colours)**2).sum(1);return e.mean(),e
        def optimise(self,n_iter=6,debug_log_dir=None):
            for i in range(n_iter):
                x=torch.rand(4,3)
                pred=x*sum(v.sum() for v in self.model.parameters())
                loss,_=self.compute_loss(pred,torch.zeros_like(pred),data={'pos':x,'face_id':torch.zeros(4,dtype=torch.long)})
                loss.backward()
                self.dc.refresh_state();self.pc.refresh_state()
                if i==1:self.dc._densify()
                if i==2:
                    mask=torch.zeros(self.model.N_sources,dtype=torch.bool);mask[1]=True
                    class Interp:
                        def barycentric_local_directions_gaussians(self,**kw):return torch.tensor([[1.,0.,0.]])
                    class Tracer:
                        def trace(self,pos,fids,vec,**kw):return pos+vec,fids
                    ids=self.mesh.get_face_vertices(self.model._kernel_face_ids)
                    self.dc.split(mask,Interp(),dict(vert_idx=ids,barycentric_coords=self.mesh.cartesian_to_barycentric(self.model.kernel_locations,ids)),Tracer(),.01)
                if i==3:self.pc.prune_redundant_kernels()
                for opt in self.optimizers:opt.step();opt.zero_grad()
                self.model.post_optimizer_step()
            return self.model
    modules={}
    for name in ['hktex','hktex.utils','hktex.trainers','hktex.trainers.base','hktex.modules',
                 'hktex.modules.heat_kernel_texture','hktex.density_controllers','hktex.density_controllers.base',
                 'hktex.density_controllers.error_densification','hktex.density_controllers.importance_pruning']:
        modules[name]=types.ModuleType(name);modules[name].__path__=[]
    modules['hktex'].utils=modules['hktex.utils']
    modules['hktex.trainers.base'].BaseTrainer=Trainer
    modules['hktex.modules.heat_kernel_texture'].HeatKernelTexture=Model
    modules['hktex.density_controllers.base'].BaseDensityController=C
    modules['hktex.density_controllers.error_densification'].ErrorDensificationController=Densifier
    modules['hktex.density_controllers.importance_pruning'].ImportancePruningController=Pruner
    with tempfile.TemporaryDirectory() as td,patch.dict(sys.modules,modules):
        td=Path(td);(td/'mesh_eigen_test.pt').write_bytes(b'CPU fixture cache')
        def run(observed):
            torch.manual_seed(51);np.random.seed(52);random.seed(53)
            trainer=Trainer(td)
            cfg=OmegaConf.create(dict(trial_dir=str(td/('on' if observed else 'off'))))
            modules['hktex.utils'].load_config=lambda:cfg
            old_optimise=Trainer.optimise
            if observed:
                with install(repo,interval=2,eval_points=64):
                    modules['hktex.utils'].load_config();trainer.optimise()
                assert Trainer.optimise is old_optimise
            else:trainer.optimise()
            return ({k:v.detach().clone() for k,v in trainer.model.state_dict().items()},
                    [v.detach().clone() for p in trainer.model.parameters() for k,v in trainer.optimizers[0].state[p].items() if torch.is_tensor(v)],
                    torch.rand(8),np.random.random(8),random.random())
        off,on=run(False),run(True)
        assert all(torch.equal(off[0][k],on[0][k]) for k in off[0])
        assert all(torch.equal(a,b) for a,b in zip(off[1],on[1]))
        assert torch.equal(off[2],on[2]) and np.array_equal(off[3],on[3]) and off[4]==on[4]
        directory=td/'on/observer';events=json.loads((directory/'events.json').read_text())
        assert [e['kind'] for e in events]==['clone','split','prune']
        with np.load(directory/'snapshots'/events[1]['after'],allow_pickle=False) as s:
            assert s['stable_ids'].tolist()==[0,2,3,4,5]
            assert s['root_ids'].tolist()==[0,2,0,1,1]
        assert json.loads((directory/'manifest.json').read_text())['completed_updates']==6
        print('PASS: on/off model parameters, Adam state and all RNG streams are identical; hooks restored')
        print('PASS: actual clone/split/prune array order, stable IDs and parent lineage')
    alpha=torch.tensor([[.1,.2,.3],[1.,.8,.7]])
    colours=torch.tensor([[[1.,-1.,.2],[.2,.4,-.5],[.5,.5,.5]]]).expand(2,-1,-1)
    mean=torch.tensor([[.5,.5,.5]])
    result,idx,contrib,pre=reconstruct_colour(alpha,colours,mean,2)
    for i in range(2):
        ix=alpha[i].argsort(descending=True)[:2]
        expected=(mean[0]+(alpha[i,ix,None]*colours[i,ix]).sum(0)/max(float(alpha[i,ix].sum()),1.)).clamp(0,1)
        torch.testing.assert_close(result[i],expected)
    print('PASS: colour decomposition for sums below/above one, negative contributions and clipping')
    source_math_test(repo)
    print('CPU TESTS PASSED. Actual CUDA replay and long training still require server validation.')


if __name__=='__main__':main()
