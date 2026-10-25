"""CPU tests for durable snapshots, exact lineage, failure safety and camera crops."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from omegaconf import OmegaConf
from image_observer import ImageRecorder, verify_recording
from observer_common import PARAMS
from native_display import ring_mode
from render_native_images import box_target


def fake_trainer(root):
    model=torch.nn.Module()
    model.N_sources=4
    for key in PARAMS:
        if key=='_kernel_face_ids': model.register_buffer(key,torch.zeros(4,dtype=torch.int32))
        else:
            shape=(3,) if key=='_mean_colour' else ((4,3) if key in ['_kernel_locations','_kernel_colours'] else (4,))
            setattr(model,key,torch.nn.Parameter(torch.arange(np.prod(shape),dtype=torch.float32).reshape(shape)/100))
    model.cfg=OmegaConf.create(dict(out_net=False,normalize_colours=False,mass_type='one',range_enforcement_type='pgd'))
    cache=root/'mesh_eigen_albo_principal_curvatures.pt';cache.write_bytes(b'test spectral cache')
    ec=OmegaConf.create(dict(mesh_path=str(root/'mesh.npz'),precomputed_name='eigen_albo',local_frames='principal_curvatures'))
    verts=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    normals=torch.tensor([[0.,0.,1.]]*3)
    return SimpleNamespace(out=root,model=SimpleNamespace(model=model,eigalbo_interp=SimpleNamespace(cfg=ec)),
        settings=dict(observer_interval=2,source_fingerprints={}),update=0,
        mesh=SimpleNamespace(verts=verts,faces=torch.tensor([[0,1,2]]),fnorms=normals[:1],vnorms=normals),
        datamodule=SimpleNamespace(saved_normals=normals.numpy()),
        renderer=SimpleNamespace(cfg=OmegaConf.create({})),optimizers=[])


def mutation(model,mask,kind):
    for k in PARAMS:
        if k=='_mean_colour':continue
        v=getattr(model,k).detach()
        if kind=='clone':out=torch.cat([v,v[mask]])
        elif kind=='split':out=torch.cat([v[~mask],v[mask],v[mask]])
        else:out=v[~mask]
        setattr(model,k,torch.nn.Buffer(out) if k=='_kernel_face_ids' else torch.nn.Parameter(out))
    model.N_sources=len(model._kernel_face_ids)


class ObservationTests(unittest.TestCase):
    def test_recording_is_readonly_and_lineage_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            t=fake_trainer(Path(td));model=t.model.model
            before={k:getattr(model,k).detach().clone() for k in PARAMS}
            rng=torch.get_rng_state().clone();np_rng=np.random.get_state()
            rec=ImageRecorder(t)
            torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
            np.testing.assert_array_equal(np.random.get_state()[1],np_rng[1])
            for k in PARAMS:torch.testing.assert_close(getattr(model,k),before[k],rtol=0,atol=0)
            rec.mutate('clone',torch.tensor([False,True,False,False]),lambda:mutation(model,torch.tensor([False,True,False,False]),'clone'))
            rec.mutate('split',torch.tensor([True,False,False,False,True]),lambda:mutation(model,torch.tensor([True,False,False,False,True]),'split'))
            np.testing.assert_array_equal(rec.lineage.ids,[1,2,3,5,6,7,8])
            np.testing.assert_array_equal(rec.lineage.roots,[1,2,3,0,1,0,1])
            mask=torch.tensor([False,True,False,False,False,False,False])
            rec.mutate('prune',mask,lambda:mutation(model,mask,'prune'))
            t.update=2;rec.after_update(dict(loss_l1=.1,kernels=model.N_sources))
            rec.snapshot('final','test');rec.finish('training_complete')
            result=verify_recording(rec.root)
            self.assertEqual(result['final_kernels'],6);self.assertEqual(result['events'],3)
            # Moving a checkpoint starts a new recording while keeping identity.
            moved=Path(td)/'resumed';moved.mkdir();t.out=moved;t.update=2
            other=ImageRecorder(t,rec.state())
            np.testing.assert_array_equal(other.lineage.ids,rec.lineage.ids)
            self.assertEqual(other.lineage.next_id,rec.lineage.next_id)
            # Hash verification detects tampering rather than replaying wrong parameters.
            first=rec.root/'snapshots'/rec.index[0]['file'];first.write_bytes(first.read_bytes()+b'corrupt')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):verify_recording(rec.root)

    def test_failed_event_keeps_intent_and_ids(self):
        with tempfile.TemporaryDirectory() as td:
            t=fake_trainer(Path(td));rec=ImageRecorder(t);before=rec.lineage.ids.copy()
            def fail():raise RuntimeError('simulated tracer failure')
            with self.assertRaises(RuntimeError):rec.mutate('split',torch.tensor([True,False,False,False]),fail)
            np.testing.assert_array_equal(before,rec.lineage.ids)
            self.assertEqual(rec.events[0]['status'],'started')
            self.assertNotIn('after',rec.events[0])

    def test_ring_context_restores_exact_objects_on_error(self):
        with tempfile.TemporaryDirectory() as td:
            model=fake_trainer(Path(td)).model.model;model.kernel_filter_func=lambda x:x
            objs=(model.kernel_filter_func,model._kernel_colours,model._mean_colour)
            with self.assertRaises(RuntimeError):
                with ring_mode(model,np.arange(4),lambda *a,**k:None):raise RuntimeError('render failed')
            self.assertTrue(all(a is b for a,b in zip(objs,(model.kernel_filter_func,model._kernel_colours,model._mean_colour))))

    def test_box_average_linear_rgb(self):
        image=np.arange(8*8*3,dtype=np.float32).reshape(8,8,3)
        small=box_target(image,4)
        np.testing.assert_array_equal(small[2,1],image[4:6,2:4].mean((0,1)))
        with self.assertRaises(ValueError):box_target(image,3)

    def test_cropped_sensor_has_same_global_rays(self):
        import mitsuba as mi
        mi.set_variant('scalar_rgb')
        from offline_common import camera_record,sensor_dict
        record=camera_record(0,35,12,'train',512,3,40)
        full=mi.load_dict(sensor_dict(mi,record,16))
        spec=sensor_dict(mi,record,16)
        spec['film'].update(crop_offset_x=64,crop_offset_y=128,crop_width=64,crop_height=32)
        crop=mi.load_dict(spec)
        for x,y in [(0.2,0.7),(31.5,17.5),(63.8,31.7)]:
            a,_=full.sample_ray(0.,.5,mi.Point2f((x+64)/512,(y+128)/512),mi.Point2f(.5))
            b,_=crop.sample_ray(0.,.5,mi.Point2f(x/64,y/32),mi.Point2f(.5))
            np.testing.assert_allclose(np.asarray(a.d),np.asarray(b.d),atol=1e-6,rtol=0)


if __name__=='__main__':unittest.main(verbosity=2)
