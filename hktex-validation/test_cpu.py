"""CPU tests; these do NOT certify CUDA/PRB end-to-end execution."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import numerics as n
import validate as v


class Gates(unittest.TestCase):
    def test_identical_low_noise(self):
        a=np.full((4,64,3),.3)
        self.assertEqual(n.compare_radiance(a,a)['status'],'passed')

    def test_wrong_scale_detected(self):
        a=np.full((4,64,3),.3)
        self.assertEqual(n.compare_radiance(a,a*.5)['status'],'failed')

    def test_noise_cannot_hide_wrong_scale(self):
        rng=np.random.default_rng(9)
        a=.3+rng.normal(0,.004,(8,1024,3));b=.1+rng.normal(0,.004,a.shape)
        self.assertEqual(n.compare_radiance(a,b)['status'],'failed')

    def test_high_noise_is_not_pass(self):
        a=np.broadcast_to(np.array([0.,.2,.4,.6])[:,None,None],(4,64,3))
        self.assertEqual(n.compare_radiance(a,a)['status'],'inconclusive')

    def test_saved_spp_required(self):
        with self.assertRaises(ValueError):n.compare_radiance(np.ones((4,10,3)),np.ones((10,3)))
        self.assertEqual(n.compare_radiance(np.ones((4,10,3)),np.ones((10,3)),256,64)['status'],'passed')

    def test_gradient_sign_and_scale(self):
        e=[.001,.003,.01]
        self.assertEqual(n.compare_gradient([.2]*3,[[.2]*3]*3,e)['status'],'passed')
        self.assertEqual(n.compare_gradient([-.2]*3,[[.2]*3]*3,e)['status'],'failed')
        self.assertEqual(n.compare_gradient([.4]*3,[[.2]*3]*3,e)['status'],'failed')

    def test_gradient_nonsmooth_inconclusive(self):
        self.assertEqual(n.compare_gradient([.2]*3,[[.1,.2,.4]]*3,[.001,.003,.01])['status'],'inconclusive')

    def test_zero_not_automatically_correct(self):
        e=[.001,.003,.01]
        self.assertEqual(n.compare_gradient([0]*3,[[0]*3]*3,e)['status'],'inconclusive')
        self.assertEqual(n.compare_gradient([0]*3,[[0]*3]*3,e,True)['status'],'passed')

    def test_patch_inside_distinct(self):
        mask=np.zeros((64,64),bool);mask[5:59,5:59]=True
        img=np.broadcast_to(np.linspace(0,1,64)[:,None,None],(64,64,3))
        a,b=n.patches(mask,img,8)
        for p in [a,b]:self.assertTrue(mask[p['y']:p['y']+8,p['x']:p['x']+8].all())
        self.assertTrue(abs(a['x']-b['x'])>=8 or abs(a['y']-b['y'])>=8)
        self.assertLess(a['mean_linear_rgb'],b['mean_linear_rgb'])
        ids=n.pixel_ids({'width':64},a)
        self.assertEqual(len(ids),64)
        self.assertEqual(ids[0],a['y']*64+a['x'])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):n.compare_gradient([np.nan]*3,[[.2]*3]*3,[.001,.003,.01])
        with self.assertRaises(ValueError):n.compare_radiance(np.full((4,3,3),np.nan),np.ones((4,3,3)))

    def test_aggregation(self):
        self.assertEqual(n.combined_status([]),'inconclusive')
        self.assertEqual(n.combined_status(['passed','inconclusive']),'inconclusive')
        self.assertEqual(n.combined_status(['failed','inconclusive']),'failed')


class Orchestration(unittest.TestCase):
    def test_fd_seed_and_parameter_restore(self):
        import torch
        v.torch=torch;v.dr=SimpleNamespace(backward=lambda x:x.backward(),suspend_grad=torch.no_grad)
        a=SimpleNamespace(seed=19,grad_repeats=3)
        with tempfile.TemporaryDirectory() as td:
            suite=v.Suite(a,Path(td));p=torch.nn.Parameter(torch.tensor([.3]))
            saved=p.detach().clone();seen=[]
            def eval_fn(grad,seed,seed_grad):
                seen.append((grad,seed,seed_grad));return p.square()
            suite.gradient_case('quadratic',[p],[torch.ones_like(p)],eval_fn)
            self.assertTrue(torch.equal(p,saved))
            self.assertEqual(suite.report['checks'][0]['status'],'passed')
            for i in range(3):
                group=seen[i*7:(i+1)*7]
                self.assertTrue(group[0][0]);self.assertNotEqual(group[0][1],group[0][2])
                self.assertTrue(all(seed==group[0][2] for _,seed,_ in group[1:]))

    def test_restore_on_error(self):
        import torch
        v.torch=torch;v.dr=SimpleNamespace(backward=lambda x:x.backward(),suspend_grad=torch.no_grad)
        with tempfile.TemporaryDirectory() as td:
            suite=v.Suite(SimpleNamespace(seed=9,grad_repeats=3),Path(td))
            p=torch.nn.Parameter(torch.tensor([.3]));b=p.detach().clone()
            def bad(grad,seed,seed_grad):
                if not grad:raise RuntimeError('deliberate FD failure')
                return p.square()
            with self.assertRaises(RuntimeError):suite.gradient_case('bad',[p],[torch.ones_like(p)],bad)
            self.assertTrue(torch.equal(p,b))


class GTMesh(unittest.TestCase):
    def test_saved_linear_texture_and_uv(self):
        import mitsuba as mi
        mi.set_variant('scalar_rgb');v.mi=mi
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'eval_gt').mkdir()
            verts=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],np.float32)
            faces=np.array([[0,1,2],[0,2,3]],np.int32)
            normals=np.tile(np.array([0,0,1],np.float32),(4,1))
            uv=np.array([[0,0],[1,0],[1,1],[0,1]],np.float32)
            np.savez(root/'eval_gt/reference.npz',vertices=verts,faces=faces,normals=normals,uv=uv,texture_linear=np.full((2,2,3),.25,np.float32))
            suite=v.Suite(SimpleNamespace(),root);suite.data=SimpleNamespace(root=root)
            mesh=suite.gt_mesh();params=mi.traverse(mesh)
            np.testing.assert_allclose(np.asarray(params['vertex_texcoords']).reshape(-1,2),np.c_[uv[:,0],1-uv[:,1]])
            si=mi.SurfaceInteraction3f();si.uv=mi.Point2f(.5,.5)
            value=mesh.bsdf().eval_diffuse_reflectance(si)
            np.testing.assert_allclose(np.asarray(value),.25,atol=1e-6)


if __name__=='__main__':unittest.main()
