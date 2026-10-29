"""Exercise actual repository ADC mutation code on CPU with a planar tracer fixture."""
import argparse,ast,copy,json,math,sys,tempfile,types,unittest
from pathlib import Path
from dataclasses import dataclass,field
import torch
from adc_support import configure_adc,run_adc,install_runtime,validate_state,POLICY
from resume_support import rebind_geodesic_faces

class StripAnnotations(ast.NodeTransformer):
    def visit_AnnAssign(self,node):
        return ast.copy_location(ast.Assign(targets=[node.target],value=node.value),node) if node.value else ast.copy_location(ast.Pass(),node)
    def visit_FunctionDef(self,node):
        node.returns=None
        for a in node.args.posonlyargs+node.args.args+node.args.kwonlyargs:a.annotation=None
        if node.args.vararg:node.args.vararg.annotation=None
        if node.args.kwarg:node.args.kwarg.annotation=None
        return self.generic_visit(node)

class GeodesicOpt(torch.optim.SGD):pass
class BaseObject:
    class Config:pass
    def configure(self,*args,**kwargs):pass

def load_controllers(repo):
    ns=dict(torch=torch,math=math,dataclass=dataclass,field=field,BaseObject=BaseObject,
        abstractmethod=lambda x:x,GeodesicOpt=GeodesicOpt,Tensor=torch.Tensor,
        hktex=types.SimpleNamespace(register=lambda x:lambda c:c,is_debug=lambda:False),
        rotate_on_plane=lambda d,n,a:d)
    for name in ['base.py','error_densification.py','importance_pruning.py']:
        p=Path(repo)/'hktex/density_controllers'/name;tree=ast.parse(p.read_text())
        tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
        tree=ast.fix_missing_locations(StripAnnotations().visit(tree));exec(compile(tree,str(p),'exec'),ns)
    return ns

class Model(torch.nn.Module):
    def __init__(self,n=5000):
        super().__init__();self.N_sources=n;self.device=torch.device('cpu')
        for name in ['_angles','_anisotropies','_thresholds','_sharpnesses']:
            self.register_parameter(name,torch.nn.Parameter(torch.full((n,),.7)))
        self._kernel_locations=torch.nn.Parameter(torch.full((n,3),.2))
        self._kernel_colours=torch.nn.Parameter(torch.full((n,3),.01))
        self._mean_colour=torch.nn.Parameter(torch.full((3,),.35))
        self.register_buffer('_kernel_face_ids',torch.zeros(n,dtype=torch.int32))
        self.register_buffer('_dummy',torch.zeros(1))
    @property
    def thresholds(self):return self._thresholds
    @property
    def angles(self):return self._angles
    @property
    def kernel_locations(self):return self._kernel_locations


def fixture(out,n=5000):
    ns=load_controllers(REPO);m=Model(n)
    mesh=types.SimpleNamespace(verts=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),N_faces=1,fnorms=torch.tensor([[0.,0.,1.]]),
        get_face_vertices=lambda f:torch.tensor([[0,1,2]]).expand(len(f),3),
        cartesian_to_barycentric=lambda p,v:torch.stack([1-p[:,0]-p[:,1],p[:,0],p[:,1]],1))
    adam=torch.optim.Adam([dict(params=[p],name='model.'+n) for n,p in m.named_parameters() if n!='_kernel_locations'],lr=.001)
    geo=GeodesicOpt([dict(params=[m._kernel_locations],name='model._kernel_locations',face_ids=[m._kernel_face_ids])],lr=.001)
    for p in m.parameters():p.grad=torch.ones_like(p)
    adam.step();geo.step()
    controls=[]
    cfgs=[dict(start_iter=400,stop_iter=401,prune_interval=400,accumulation_interval=400,selection_threshold=.001,contrib_threshold=.01,min_kernels=4900),
          dict(start_iter=200,stop_iter=401,densify_interval=200,error_accumulation_interval=200,error_threshold=.02376,
               size_threshold=.18,split_radius=.05,max_densify_ratio=.02,max_kernels=5500)]
    for cls,c in zip(['ImportancePruningController','ErrorDensificationController'],cfgs):
        dc=ns[cls]();dc.cfg=types.SimpleNamespace(**c);dc.configure(mesh,m,[adam,geo]);controls.append(dc)
    eig=types.SimpleNamespace(barycentric_local_directions_gaussians=lambda barycentric_coords,vert_idx:torch.tensor([[1.,0.,0.]]).expand(len(vert_idx),3))
    tracer=types.SimpleNamespace(trace=lambda p,f,v,**kw:(p+v,f.clone()))
    net=types.SimpleNamespace(model=m,reset=lambda:None,prepare_kernels=lambda:dict(barycentric_coords=mesh.cartesian_to_barycentric(m._kernel_locations,None)),eigalbo_interp=eig,tracer=tracer)
    t=types.SimpleNamespace(model=net,mesh=mesh,optimizers=[adam,geo],density_controllers=controls,out=Path(out),observer=None,update=0)
    install_runtime(t);return t

class Tests(unittest.TestCase):
    def test_actual_clone_split_prune_and_resume_references(self):
        with tempfile.TemporaryDirectory() as td:
            t=fixture(td);m=t.model.model
            for u,n in [(10,5001),(20,5002),(30,5001)]:
                t.update=u-1;run_adc(t);self.assertEqual(m.N_sources,n);validate_state(t)
            saved=copy.deepcopy(t.optimizers[1].state_dict());t.optimizers[1].load_state_dict(saved)
            self.assertIsNot(t.optimizers[1].param_groups[0]['face_ids'][0],m._kernel_face_ids)
            rebind_geodesic_faces(t);validate_state(t)
            m._error_accumulator.fill_(1.75);m._hit_accumulator.fill_(9)
            path=Path(td)/'state.pt'
            torch.save(dict(model=m.state_dict(),opts=[o.state_dict() for o in t.optimizers]),path)
            saved=torch.load(path,weights_only=True,map_location='cpu')
            restored=fixture(td,n=m.N_sources);restored.model.model.load_state_dict(saved['model'],strict=True)
            for opt,sd in zip(restored.optimizers,saved['opts']):opt.load_state_dict(sd)
            rebind_geodesic_faces(restored);validate_state(restored)
            self.assertTrue(torch.equal(restored.model.model._error_accumulator,m._error_accumulator))
            self.assertTrue(torch.equal(restored.model.model._hit_accumulator,m._hit_accumulator))
    def test_actual_natural_decision_timing_caps_and_mixed_clone_split(self):
        with tempfile.TemporaryDirectory() as td:
            t=fixture(td);m=t.model.model
            m._error_accumulator.fill_(20.)
            with torch.no_grad():m._thresholds[:50].fill_(.95)
            t.update=199;run_adc(t)
            # 100 candidates selected; the first 50 small kernels can clone,
            # remaining selected large kernels split. All states must stay valid.
            self.assertEqual(m.N_sources,5100);validate_state(t)
            m._error_accumulator.fill_(20.);t.update=399;run_adc(t)
            self.assertEqual(m.N_sources,4998)
            decisions=[json.loads(l) for l in (Path(td)/'adc_decisions.jsonl').read_text().splitlines()]
            self.assertEqual([(x['update'],x['kind']) for x in decisions],[(200,'densify'),(400,'prune'),(400,'densify')])
    def test_policy_configuration(self):
        from omegaconf import OmegaConf
        cfg=OmegaConf.create(dict(density_controllers=[dict(density_controller_type='x.importance_pruning',args={}),dict(density_controller_type='x.error_based_densification',args={})]))
        s={};configure_adc(cfg,s)
        self.assertEqual(cfg.density_controllers[1].args.densify_interval,200)
        self.assertEqual(cfg.density_controllers[0].args.min_kernels,4900)
        self.assertEqual(s['adc_smoke'],POLICY)
    def test_real_entry_config_and_resume_identity(self):
        import importlib.util
        from omegaconf import OmegaConf
        import run_arm
        directory=Path(REPO)/'hktex-image-training-observed';sys.path.insert(0,str(directory))
        stub=types.ModuleType('image_trainer')
        stub.ExternalImageTrainer=type('Trainer',(),{'setup_images':lambda *x:None,'restore':lambda *x:None})
        stub.color3=lambda x:x;stub.cpu_state=lambda x:x
        old=sys.modules.get('image_trainer');sys.modules['image_trainer']=stub
        try:
            spec=importlib.util.spec_from_file_location('real_entry',directory/'train_images.py')
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            a=types.SimpleNamespace(arm='linear',pair='/tmp/adc-pair',resume=None)
            run_arm.install(mod,stub,a,'code')
            with tempfile.TemporaryDirectory() as td:
                root=Path(td);(root/'renderer.json').write_text('{}')
                data=types.SimpleNamespace(root=root,manifest={'renderer_config':'renderer.json'})
                st=dict(mod.PROFILES['baseline'],seed=0,manifest_sha256='m',mesh_sha256='g',source_fingerprints={'hktex/utils/lbo.py':'l','hktex/modules/eigen_albo.py':'e'})
                cfg,_,_=mod.make_config(Path(REPO),data,st,root/'cache',OmegaConf)
                self.assertTrue(st['adc']);self.assertEqual(cfg.density_controllers[1].args.densify_interval,200)
                self.assertEqual(cfg.density_controllers[0].args.start_iter,400)
                self.assertEqual(cfg.optimizers[1].scheduler.args.T_max,5000)
                initial=OmegaConf.to_container(cfg,resolve=True)
                resumed,_,_=mod.make_config(Path(REPO),data,st,root/'cache',OmegaConf,initial)
                resumed.network.model.n_sources=5101
                self.assertEqual(run_arm.make_pair_identity(initial,st),run_arm.make_pair_identity(OmegaConf.to_container(resumed,resolve=True),st))
        finally:
            if old is None:sys.modules.pop('image_trainer',None)
            else:sys.modules['image_trainer']=old

    def test_bad_optimizer_reference_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            t=fixture(td);t.optimizers[1].param_groups[0]['face_ids']=[t.model.model._kernel_face_ids.clone()]
            with self.assertRaisesRegex(RuntimeError,'Stale GeodesicOpt'):validate_state(t)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);a,rest=p.parse_known_args();REPO=a.repo
    unittest.main(argv=[sys.argv[0]]+rest,verbosity=2)
