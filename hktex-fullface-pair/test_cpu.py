"""CPU tests using upstream mutation methods and real PyTorch optimizers.
Only package-loading/GPU geometry dependencies are substituted in this test process.
The separate server smoke still runs the actual CUDA field, mesh and tracer.
"""
import ast
from abc import abstractmethod
import math
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import unittest
import numpy as np
import torch
from torch.optim.optimizer import required
from policy import select_prune,select_grow,event_steps,lr_factor,validate_config

HERE=Path(__file__).resolve().parent;REPO=HERE.parent

def extract(path,name,env):
    tree=ast.parse(path.read_text());node=next(n for n in tree.body if getattr(n,'name',None)==name)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),env)
    return env[name]

Geo=extract(REPO/'hktex/modules/geodesic_opt.py','GeodesicOpt',dict(torch=torch,optim=torch.optim,Tensor=torch.Tensor,required=required))
rotate=extract(REPO/'hktex/utils/misc.py','rotate_on_plane',dict(torch=torch))
class BaseObject:
    class Config:pass
    def __init__(self,cfg=None,**kwargs):self.cfg=SimpleNamespace(**(cfg or {}));self.configure(**kwargs)
    def configure(self,**kwargs):pass
Base=extract(REPO/'hktex/density_controllers/base.py','BaseDensityController',dict(torch=torch,math=math,abstractmethod=abstractmethod,BaseObject=BaseObject,GeodesicOpt=Geo,rotate_on_plane=rotate,dataclass=__import__('dataclasses').dataclass))
base_module=ModuleType('hktex.density_controllers.base');base_module.BaseDensityController=Base
sys.modules['hktex.density_controllers.base']=base_module
from adc import WeightedADC
from engine import raw_state,restore,digest
from runtime import sample_stratified

class Mesh:
    def __init__(self):
        self.verts=torch.tensor([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.],[4.,0.,0.],[6.,0.,0.],[4.,2.,0.],[8.,0.,0.],[10.,0.,0.],[8.,2.,0.]])
        self.faces=torch.arange(9).reshape(3,3);self.fnorms=torch.tensor([[0.,0.,1.]]).repeat(3,1)
    def get_face_vertices(self,f):return self.faces[f.long()]
    def get_vertex_locations(self,vi):return self.verts[vi.long()]
    def cartesian_to_barycentric(self,p,vi):
        t=self.get_vertex_locations(vi);xy=(p-t[:,0])[:,:2]/2
        return torch.stack([1-xy.sum(1),xy[:,0],xy[:,1]],1)
class Tracer:
    def trace(self,p,f,d,**kw):return p+d,f.clone()
    def trace_(self,p,f,d,**kw):p.add_(d)
class Texture(torch.nn.Module):
    def __init__(self,mesh):
        super().__init__();self.N_sources=18
        for n,x in dict(_kernel_colours=torch.ones(18,3)*.1,_angles=torch.ones(18)*.2,_anisotropies=torch.ones(18)*1.5,_thresholds=torch.ones(18)*.7,_sharpnesses=torch.ones(18)*10).items():setattr(self,n,torch.nn.Parameter(x))
        self._mean_colour=torch.nn.Parameter(torch.ones(1,3)*.35)
        fid=torch.arange(3).repeat_interleave(6);loc=mesh.verts[mesh.faces[fid,0]]+torch.tensor([.6,.6,0.])
        self._kernel_locations=torch.nn.Parameter(loc);self.register_buffer('_kernel_face_ids',fid);self.register_buffer('_dummy',torch.empty(0))
    @property
    def thresholds(self):return self._thresholds
    @property
    def kernel_locations(self):return self._kernel_locations
    @property
    def angles(self):return self._angles
class Network:
    def __init__(self,m):
        self.model=m;self.tracer=Tracer();self.eigalbo_interp=SimpleNamespace(barycentric_local_directions_gaussians=lambda barycentric_coords,vert_idx:torch.tensor([1.,0.,0.]).repeat(len(vert_idx),1))
    def reset(self):pass

def fixture():
    mesh=Mesh();m=Texture(mesh);network=Network(m)
    groups=[dict(name='model.'+name,params=[p],lr=.001) for name,p in m.named_parameters() if name!='_kernel_locations']
    opts=[torch.optim.Adam(groups),Geo([dict(name='model._kernel_locations',params=[m._kernel_locations],face_ids=[m._kernel_face_ids])],tracer=network.tracer,lr=.00001)]
    g=dict(vertices=mesh.verts.numpy(),faces=mesh.faces.numpy(),face_components=np.arange(3),face_regions=np.array([1,3,4]))
    a=dict(start=4,stop_inclusive=8,densify_interval=2,prune_interval=4,error_threshold=.02,selection_threshold=.001,contribution_threshold=.01,size_threshold=.18,max_densify_ratio=.2,max_prune_ratio=.2,max_kernels=40,min_kernels=6,min_head_kernels=2,min_other_component_kernels=2,split_radius_in_median_edges=1.)
    dc=WeightedADC({},mesh=mesh,model=m,optimizers=opts,network=network,geometry=g,policy=a)
    return m,network,opts,dc

def warm(m,opts):
    m.zero_grad(set_to_none=True);sum(p.square().mean() for p in m.parameters()).backward()
    for opt in opts:opt.step()

class Tests(unittest.TestCase):
    def test_schedule_exact_boundaries(self):
        import json
        c=validate_config(json.loads((HERE/'config.json').read_text()));e=event_steps(c)
        self.assertEqual(e['densify'],list(range(1000,4001,200)))
        self.assertEqual(e['prune'],list(range(1200,4001,400)))
        self.assertEqual(e['first_accumulation_step'],801)
        self.assertAlmostEqual(lr_factor(1,c),1);self.assertAlmostEqual(lr_factor(5000,c),.1)
    def test_pruning_preserves_components_and_cap(self):
        _,_,_,dc=fixture();a=dict(dc.policy,max_prune_ratio=.9)
        comp=np.repeat(np.arange(3),[10,3,3]);hits=np.zeros(16);scores=np.arange(16,dtype=float)*1e-5
        mask,_=select_prune(hits,scores,comp,[2,3,3],a)
        self.assertFalse(mask[10:].any());self.assertEqual(mask.sum(),8)
    def test_growth_respects_budget_and_size(self):
        _,_,_,dc=fixture();a=dict(dc.policy,max_kernels=20)
        clone,split,n=select_grow(np.arange(18,dtype=float),np.r_[np.ones(17)*.9,.5],a)
        self.assertEqual((clone|split).sum(),2);self.assertTrue(split[17]);self.assertTrue(clone[16])
    def test_clone_preserves_adam_and_clears_barycentric_cache(self):
        m,network,opts,dc=fixture();warm(m,opts)
        old=opts[0].state[m._kernel_colours]['exp_avg'].clone();mean=m._mean_colour
        m._kernel_locations.bary_coords=torch.ones(18,3)
        mask=np.zeros(18,bool);mask[7]=True;dc.mutate('clone',mask)
        torch.testing.assert_close(opts[0].state[m._kernel_colours]['exp_avg'][:18],old)
        self.assertEqual(float(opts[0].state[m._kernel_colours]['exp_avg'][18].abs().sum()),0)
        self.assertFalse(hasattr(m._kernel_locations,'bary_coords'));self.assertIs(m._mean_colour,mean)
        warm(m,opts);dc.optimizer_audit();self.assertEqual(m.N_sources,19)
    def test_split_then_prune_preserves_optimizer_faces_and_order(self):
        m,network,opts,dc=fixture();warm(m,opts);old=opts[0].state[m._kernel_colours]['exp_avg'].clone()
        mask=np.zeros(18,bool);mask[[6,12]]=True;dc.mutate('split',mask)
        torch.testing.assert_close(opts[0].state[m._kernel_colours]['exp_avg'][:16],old[torch.as_tensor(~mask)])
        self.assertEqual(float(opts[0].state[m._kernel_colours]['exp_avg'][16:].abs().sum()),0)
        self.assertEqual(dc.component_counts(),[6,7,7]);self.assertEqual(m.N_sources,20)
        mask=np.zeros(20,bool);mask[[0,3]]=True;dc.mutate('prune',mask)
        warm(m,opts);dc.optimizer_audit();self.assertEqual(dc.component_counts(),[4,7,7])
        self.assertIs(opts[1].param_groups[0]['face_ids'][0],m._kernel_face_ids)
    def test_weighted_adc_statistics_correct_oversampling(self):
        m,_,_,dc=fixture();labels=np.repeat(np.arange(1,5),[900,90,7,3])
        batch,w=sample_stratified(dict(regions=labels),np.random.default_rng(1))
        ids=torch.as_tensor(batch['regions']-1).reshape(1,-1,1);contrib=torch.ones_like(ids,dtype=torch.float32)
        pred=torch.as_tensor(batch['regions']/10,dtype=torch.float32)[:,None].repeat(1,3)
        dc.accumulate(3,pred,torch.zeros_like(pred),ids,contrib,torch.tensor(w))
        expected=torch.tensor([900,90,7,3],dtype=torch.float32)/1000*1024
        torch.testing.assert_close(dc._state['hits'][:4],expected,rtol=2e-5,atol=1e-4)
        torch.testing.assert_close(dc._state['error'][:4],expected*torch.arange(1,5)/10,rtol=2e-5,atol=1e-4)
    def test_adc_schedule_applies_actual_mutations_and_resets_window(self):
        m,_,opts,dc=fixture();warm(m,opts)
        dc._state['hits'].fill_(100);dc._state['contribution'].fill_(100);dc._state['error'].fill_(1)
        dc.accumulated_steps=2;rec=dc.apply(4)
        self.assertGreater(rec['split_parents']+rec['cloned'],0);self.assertEqual(dc.accumulated_steps,0)
        self.assertTrue(all(float(v.sum())==0 for v in dc._state.values()))
        warm(m,opts);dc.optimizer_audit()
    def test_exact_initialization_restore(self):
        m,n,opts,dc=fixture();s=raw_state(n);key=digest(s);warm(m,opts)
        self.assertNotEqual(digest(raw_state(n)),key);restore(n,s);self.assertEqual(digest(raw_state(n)),key)
    def test_stable_search_avoids_large_offset_cancellation_and_ties(self):
        class Index:
            config=SimpleNamespace(metric='l2');_index=object()
            _db=torch.tensor([[10000.05,10000.],[10000.01,10000.],[10000.01,10000.]])
            def _check_cuda_2d(self,x,name):assert x.ndim==2
        mod=ModuleType('hktex.knn_heat.knn_heat.faiss_knn');mod.FaissGpuFlatIndex=Index
        sys.modules[mod.__name__]=mod
        from stable_search import install
        install();d,i=Index().search(torch.tensor([[10000.,10000.]]),3)
        self.assertEqual(i.tolist(),[[1,2,0]])
        direct=((Index._db.double()-torch.tensor([[10000.,10000.]],dtype=torch.float64))**2).sum(1)
        torch.testing.assert_close(d.double(),direct[i],rtol=1e-6,atol=1e-10)

if __name__=='__main__':unittest.main()
