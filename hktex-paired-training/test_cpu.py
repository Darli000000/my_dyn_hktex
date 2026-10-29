"""CPU checks of loss, config, pairing and command contracts; not a CUDA test."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
import numpy as np
import torch
from losses import dr_transform, numpy_transform, objective, validate_arm
from run_arm import install, make_pair_identity, canonical_hash
from compare_pair import compare

class TorchDr:
    clip=staticmethod(torch.clamp)
    power=staticmethod(torch.pow)
    abs=staticmethod(torch.abs)
    mean=staticmethod(torch.mean)
    dot=staticmethod(lambda x,y:(x*y).sum(-1))
    log=staticmethod(lambda x:torch.log(x) if torch.is_tensor(x) else math.log(x))
    maximum=staticmethod(lambda x,v:torch.clamp_min(x,v))

class LossTests(unittest.TestCase):
    def test_formula_positive_and_endpoint_gradients(self):
        x=torch.tensor([-1.,0.,1e-8,1e-5,.01,.3,1.,1.1],dtype=torch.float64,requires_grad=True)
        y=dr_transform(x,TorchDr)
        np.testing.assert_allclose(y.detach().numpy(),numpy_transform(x.detach().numpy()),atol=1e-13)
        # Independently evaluate the upstream formula on positive in-range RGB.
        v=x.detach().numpy()[2:7]
        expected=1.055*((np.log(v*65536+1e-6)-math.log(1e-6))/(math.log(1+1e-6)-math.log(1e-6)))**(1/2.4)-.055
        np.testing.assert_allclose(y.detach().numpy()[2:7],expected,atol=1e-13)
        y.sum().backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertEqual(float(x.grad[0]),0.)
        self.assertEqual(float(x.grad[-1]),0.)
        self.assertAlmostEqual(float(y[1].detach())+.055,1.055e-5,places=12)
    def test_both_objectives_are_rgb_mean(self):
        p=torch.tensor([[.2,.5,.8],[.1,.2,.4]],dtype=torch.float64,requires_grad=True)
        t=torch.tensor([[.1,.3,.6],[.2,.4,.5]],dtype=torch.float64)
        mi=types.SimpleNamespace(Color3f=lambda x:x)
        self.assertAlmostEqual(float(objective(p,t,'linear',mi,TorchDr).detach()),float((p-t).abs().mean().detach()))
        expected=np.abs(numpy_transform(p.detach().numpy())-numpy_transform(t.numpy())).mean()
        self.assertAlmostEqual(float(objective(p,t,'loggamma',mi,TorchDr).detach()),expected)
        objective(p,t,'loggamma',mi,TorchDr).backward()
        self.assertTrue(bool(torch.isfinite(p.grad).all()))
    def test_resume_guard(self):
        validate_arm(dict(arm='linear',code_hash='x'),'linear','x')
        for arm,code in [('loggamma','x'),('linear','y')]:
            with self.assertRaises(ValueError):validate_arm(dict(arm='linear',code_hash='x'),arm,code)

class PairTests(unittest.TestCase):
    def test_identity_excludes_only_loss_flags(self):
        s=dict(manifest_sha256='a',mesh_sha256='b',source_fingerprints={},seed=0,steps=5000,batch_pixels=256,aa=4,transport=4)
        c=dict(use_hdr=True,use_linear_gamma_space=False,network={'n':5000},lr=.01)
        b=copy.deepcopy(c);b.update(use_hdr=False,use_linear_gamma_space=True)
        self.assertEqual(make_pair_identity(c,s),make_pair_identity(b,s))
        b['lr']=.02
        self.assertNotEqual(canonical_hash(make_pair_identity(c,s)),canonical_hash(make_pair_identity(b,s)))
    def test_comparison_accepts_pair_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'initial.pt').write_bytes(b'fixture')
            sha=hashlib.sha256(b'fixture').hexdigest()
            def write(arm,signature='same'):
                d=root/'s00050'/arm;d.mkdir(parents=True,exist_ok=True)
                s=dict(seed=0,steps=5000,batch_pixels=256,aa=4,transport=4,sources=5000,adc=False,manifest_sha256='m',source_fingerprints={},
                       paired_loss=dict(arm=arm,initial_sha256=sha,code_hash='c'))
                r=dict(status='stage_complete',completed_updates=50,start_update=49,settings=s,final_eval={'mse':{'val':.1}})
                (d/'report.json').write_text(json.dumps(r))
                (d/'metrics.jsonl').write_text(json.dumps(dict(update=50,batch_signature=signature,kernels=5000,loss_l1=.1,batch_mse=.02))+'\n')
                (d/'field_metrics.jsonl').write_text(json.dumps(dict(update=50,clipped_fraction=.2))+'\n')
            write('linear');write('loggamma')
            self.assertTrue(compare(root,'s00050')['pairing_passed'])
            write('loggamma','different')
            with self.assertRaisesRegex(ValueError,'stream'):compare(root,'s00050')
            write('loggamma');(root/'initial.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'initial state changed'):compare(root,'s00050')

class ExistingConfigTests(unittest.TestCase):
    def test_real_config_and_resume(self):
        from omegaconf import OmegaConf
        repo=Path(REPO)
        directory=repo/'hktex-image-training-observed';sys.path.insert(0,str(directory))
        # Stub only the CUDA-dependent trainer import; execute the REAL make_config.
        stub=types.ModuleType('image_trainer')
        stub.ExternalImageTrainer=type('FakeTrainer',(),{'setup_images':lambda *args:None})
        stub.color3=lambda x:x;stub.cpu_state=lambda x:x
        old=sys.modules.get('image_trainer');sys.modules['image_trainer']=stub
        try:
            spec=importlib.util.spec_from_file_location('tested_entry',directory/'train_images.py')
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            a=types.SimpleNamespace(arm='loggamma',pair='/tmp/test-pair',resume=None)
            install(mod,stub,a,'samecode')
            with tempfile.TemporaryDirectory() as td:
                root=Path(td);(root/'renderer.json').write_text('{}')
                data=types.SimpleNamespace(root=root,manifest={'renderer_config':'renderer.json'})
                s=dict(mod.PROFILES['baseline'],seed=0,mesh_sha256='mesh',source_fingerprints={'hktex/utils/lbo.py':'l','hktex/modules/eigen_albo.py':'e'})
                cfg,_,_=mod.make_config(repo,data,s,root/'cache',OmegaConf)
                self.assertEqual(cfg.network.model.n_sources,5000)
                self.assertEqual(list(cfg.density_controllers),[])
                self.assertEqual(cfg.grad_spp,16)
                self.assertTrue(cfg.use_linear_gamma_space)
                for o in cfg.optimizers:
                    if o.name=='GeodesicOpt':self.assertEqual(o.scheduler.args.T_max,5000)
                    else:self.assertEqual(o.scheduler.args.step_size,1250)
                s['trainer_config']=OmegaConf.to_container(cfg,resolve=True)
                resumed,_,_=mod.make_config(repo,data,s,root/'cache',OmegaConf,s['trainer_config'])
                self.assertEqual(OmegaConf.to_container(resumed),OmegaConf.to_container(cfg))
                s['paired_loss']['arm']='linear'
                with self.assertRaisesRegex(ValueError,'loss arm'):
                    mod.make_config(repo,data,s,root/'cache',OmegaConf,s['trainer_config'])
        finally:
            if old is None:sys.modules.pop('image_trainer',None)
            else:sys.modules['image_trainer']=old

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);a,rest=p.parse_known_args();REPO=a.repo
    unittest.main(argv=[sys.argv[0]]+rest,verbosity=2)
