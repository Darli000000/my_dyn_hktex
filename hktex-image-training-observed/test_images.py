"""CPU tests for no-texture data access, view splits, pixel footprint and cameras."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
from image_dataset import ImageDataset, pixel_samples, inside, check_contract
from offline_common import cameras, sensor_dict, project, write_json, sha256, interior_mask

WITH_MITSUBA = '--mitsuba' in sys.argv
if WITH_MITSUBA:
    sys.argv.remove('--mitsuba')


def fixture(root):
    records = cameras('full', 32, 3.7, 40)
    verts = np.array([[-.2, -.2, -.2], [.2, -.2, -.2], [0, .2, -.2]], np.float32)
    np.savez(root/'mesh.npz', vertices=verts, faces=np.array([[0, 1, 2]], np.int32),
             normals=np.tile([0, 0, 1], (3, 1)))
    center = np.zeros((32, 32), bool)
    center[8:24, 8:24] = True
    alpha = center.astype(np.float32)
    mask = interior_mask(center, alpha)
    for i, r in enumerate(records):
        # Unique camera-specific values expose accidental view/target mixing.
        rgb = np.zeros((32, 32, 3), np.float32)
        rgb[..., 0] = np.arange(32)[None, :]/32
        rgb[..., 1] = np.arange(32)[:, None]/32
        rgb[..., 2] = i/36
        np.save(root/f'{i}_rgb.npy', rgb)
        np.save(root/f'{i}_alpha.npy', alpha)
        Image.fromarray(mask.astype('uint8')*255).save(root/f'{i}_mask.png')
        Image.fromarray(np.zeros((32, 32, 3), 'uint8')).save(root/f'{i}_preview.png')
        np.savez(root/f'{i}_geometry.npz', center_mask=center, depth_cv=alpha)
        r.update(rgb=f'{i}_rgb.npy', alpha=f'{i}_alpha.npy', mask=f'{i}_mask.png',
                 geometry=f'{i}_geometry.npz', preview=f'{i}_preview.png', projection_error_px=0.)
    cfg = dict(ground_plane_config=dict(activated=False),
               emitter_config=dict(envmap_path=None, radiance=1.),
               mitsuba_mesh_config=dict(twosided=False),
               integrator_config=dict(hide_emitters=True, meta=dict(max_depth=3)))
    write_json(root/'renderer_config.json', cfg)
    m = dict(schema='hktex-offline-multiview-v1', status='complete', train_mesh='mesh.npz',
             renderer_config='renderer_config.json', views=records,
             scene_contract=dict(material='principled', specular=0., metallic=0., roughness=1.,
                                 max_depth=3, constant_environment_radiance=[1., 1., 1.], ground=False,
                                 integrator='path', hide_emitters=True, denoise=False, exposure=1., tone_mapping='none'),
             image_contract=dict(filter='box; uniform subpixel sampling', rgb='float32 HWC linear radiance'),
             sha256={p.name: sha256(p) for p in root.iterdir()},
             provenance=dict(source_mesh='/nonexistent/forbidden/original.obj'))
    write_json(root/'manifest.json', m)
    write_json(root/'replay_validation.json', dict(passed=True))
    return m


class Tests(unittest.TestCase):
    def test_pixel_footprint_and_order(self):
        r = dict(width=2048, height=2048)
        ids = np.array([0, 2048*17+5, 2048**2-1])
        pos = pixel_samples(r, ids, 4, 3, np.random.default_rng(0))
        grouped = pos.reshape(3, 4, 3, 2)
        self.assertEqual(pos.shape, (36, 2))
        np.testing.assert_array_equal(grouped[:, :, 0], grouped[:, :, 1])
        px = grouped*2048
        for i, index in enumerate(ids):
            y, x = divmod(index, 2048)
            self.assertTrue(((px[i, ..., 0] >= x) & (px[i, ..., 0] < x+1)).all())
            self.assertTrue(((px[i, ..., 1] >= y) & (px[i, ..., 1] < y+1)).all())
        # Different subpixel samples must not all be identical.
        self.assertGreater(np.std(grouped[1, :, 0, 0]), 0.)

    def test_sampling_no_uv_split_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m = fixture(root)
            # No eval_gt or original mesh exists, yet preflight and sampling work.
            self.assertTrue(check_contract(root, 32)['passed'])
            ds = ImageDataset(root)
            rng = np.random.default_rng(0)
            visited = []
            for _ in range(24):
                r, ids, target = ds.train_batch(rng, 10)
                visited.append(r['id'])
                self.assertEqual(r['split'], 'train')
                np.testing.assert_allclose(target[:, 0], (ids % 32)/32)
                np.testing.assert_allclose(target[:, 1], (ids // 32)/32)
                np.testing.assert_allclose(target[:, 2], int(r['id'].split('_')[1])/36)
            self.assertEqual(len(set(visited)), 24)
            state, rs = ds.sampler_state(), rng.bit_generator.state
            r1, ids1, t1 = ds.train_batch(rng, 10)
            ds2 = ImageDataset(root)
            ds2.load_sampler_state(state)
            rng2 = np.random.default_rng()
            rng2.bit_generator.state = rs
            r2, ids2, t2 = ds2.train_batch(rng2, 10)
            self.assertEqual(r1['id'], r2['id'])
            np.testing.assert_array_equal(ids1, ids2)
            np.testing.assert_array_equal(t1, t2)
            self.assertLessEqual(len(ds._images), 4)
            with self.assertRaises(ValueError):
                inside(root, 'eval_gt/reference.npz')
            with self.assertRaises(ValueError):
                inside(root, '../outside.npy')

    def test_reject_wrong_resolution_and_colour_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m = fixture(root)
            with self.assertRaises(ValueError):
                check_contract(root, 2048)
            m['scene_contract']['hide_emitters'] = False
            write_json(root/'manifest.json', m)
            with self.assertRaises(ValueError):
                check_contract(root, 32)

    @unittest.skipUnless(WITH_MITSUBA, 'Pass --mitsuba for real sensor checks')
    def test_actual_camera_subpixel_projection(self):
        import mitsuba as mi
        mi.set_variant('scalar_rgb')
        r = cameras('full', 2048, 3.7, 40)[7]
        sensor = mi.load_dict(sensor_dict(mi, r, 1))
        positions = pixel_samples(r, [2048*100+50, 2048*1000+800], 4, 2, np.random.default_rng(0))
        for pos in positions:
            ray, weight = sensor.sample_ray(0., .5, mi.Point2f(pos), mi.Point2f(.5))
            # Also exercise the exact differential-ray constructor used in GPU training.
            ray = mi.RayDifferential3f(ray)
            xyz = np.asarray(ray.o)+3*np.asarray(ray.d)
            uv, _ = project([xyz], r)
            np.testing.assert_allclose(uv[0], pos*2048, atol=.002)
            np.testing.assert_allclose(np.asarray(weight), [1, 1, 1], atol=1e-6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
