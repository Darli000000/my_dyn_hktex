"""CPU contract tests. --mitsuba additionally tests actual scalar Mitsuba cameras/rendering."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from offline_common import (cameras, camera_record, project, sensor_dict, srgb_to_linear,
                            linear_to_srgb, interior_mask, write_json, sha256, validate,
                            center_hits, save_view)

WITH_MITSUBA = '--mitsuba' in sys.argv
if WITH_MITSUBA:
    sys.argv.remove('--mitsuba')


class ContractTests(unittest.TestCase):
    def test_splits(self):
        rs = cameras('full', 256, 3.7, 40)
        self.assertEqual([sum(r['split'] == s for r in rs) for s in ['train', 'val', 'test']], [24, 6, 6])
        self.assertEqual(len({tuple(np.array(r['camera_to_world_mitsuba']).ravel()) for r in rs}), 36)
        for r in rs:
            c = np.array(r['camera_to_world_mitsuba'])
            np.testing.assert_allclose(c[:3, :3].T@c[:3, :3], np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.rad2deg(np.arcsin(c[1, 3]/3.7)), r['elevation_deg'])
            uv, z = project([[0, 0, 0]], r)
            np.testing.assert_allclose(uv, [[128, 128]], atol=1e-10)
            np.testing.assert_allclose(z, [3.7])

    def test_colour(self):
        x = np.linspace(0, 1, 256, dtype=np.float32)
        np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=2e-7)
        self.assertAlmostEqual(float(srgb_to_linear(.5)), .21404114, places=6)

    def test_mask(self):
        center = np.zeros((10, 10), bool)
        center[2:8, 2:8] = True
        alpha = center.astype(float)
        self.assertEqual(interior_mask(center, alpha).sum(), 16)
        alpha[5, 5] = .5
        self.assertEqual(interior_mask(center, alpha).sum(), 7)

    def test_dataset_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vertices = np.array([[-.2, -.2, -.2], [.2, .2, .2], [.2, -.2, .2]], np.float32)
            np.savez(root/'mesh.npz', vertices=vertices, faces=[[0, 1, 2]])
            center = np.zeros((32, 32), bool)
            center[8:24, 8:24] = True
            np.save(root/'rgb.npy', np.ones((32, 32, 3), np.float32)*.3)
            np.save(root/'alpha.npy', center.astype(np.float32))
            np.savez(root/'checks.npz', center_mask=center, depth_cv=np.ones((32, 32)))
            Image.fromarray(interior_mask(center, center.astype(float)).astype('uint8')*255).save(root/'mask.png')
            views = [dict(r, rgb='rgb.npy', alpha='alpha.npy', mask='mask.png', geometry='checks.npz',
                          projection_error_px=0.) for r in cameras('smoke', 32, 3.7, 40)]
            manifest = dict(status='complete', train_mesh='mesh.npz', views=views,
                            sha256={p.name: sha256(p) for p in root.iterdir()})
            write_json(root/'manifest.json', manifest)
            self.assertTrue(validate(root, write=False)['passed'])
            views[0]['K'][0][0] *= 2
            write_json(root/'manifest.json', manifest)
            self.assertFalse(validate(root, write=False)['passed'])
            views[0]['K'][0][0] /= 2
            write_json(root/'manifest.json', manifest)
            np.save(root/'rgb.npy', np.zeros((32, 32, 3), np.float32))
            self.assertFalse(validate(root, write=False)['passed'])

    @unittest.skipUnless(WITH_MITSUBA, 'Pass --mitsuba for actual scalar renderer tests')
    def test_actual_mitsuba(self):
        import mitsuba as mi
        mi.set_variant('scalar_rgb')
        # Verify real sensor rays against saved OpenCV matrices at noncentral pixels,
        # on multiple azimuth/elevation pairs (not merely projecting the origin).
        for a, e in [(0, 0), (90, 20), (210, -15), (330, 20)]:
            r = camera_record(0, a, e, 'train', 64, 3.7, 40)
            sensor = mi.load_dict(sensor_dict(mi, r, 16))
            for x, y in [(0.5, 0.5), (31.5, 20.5), (63.5, 63.5)]:
                ray, _ = sensor.sample_ray(0., .5, mi.Point2f(x/64, y/64), mi.Point2f(.5))
                p = np.asarray(ray.o)+3*np.asarray(ray.d)
                uv, _ = project([p], r)
                np.testing.assert_allclose(uv[0], [x, y], atol=2e-4)
        # The float bitmap path used by UVTextureRenderer must preserve linear input.
        tex = mi.load_dict(dict(type='bitmap', bitmap=mi.Bitmap(np.full((2, 2, 3), .18, np.float32))))
        si = mi.SurfaceInteraction3f()
        si.uv = mi.Point2f(.5)
        np.testing.assert_allclose(np.array(tex.eval_3(si)), [.18]*3, atol=1e-6)
        r = camera_record(0, 0, 0, 'train', 32, 3.7, 40)
        scene = mi.load_dict(dict(type='scene', camera=sensor_dict(mi, r, 16),
                                 integrator=dict(type='path', max_depth=3, hide_emitters=True),
                                 emitter=dict(type='constant', radiance=dict(type='rgb', value=1.)),
                                 mesh=dict(type='sphere', radius=.7,
                                           bsdf=dict(type='principled', base_color=.3, specular=0., roughness=1.))))
        a = np.asarray(mi.render(scene, spp=16, seed=123)).copy()
        b = np.asarray(mi.render(scene, spp=16, seed=123)).copy()
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (32, 32, 4))
        self.assertEqual(a[0, 0, 3], 0.)
        self.assertEqual(a[16, 16, 3], 1.)
        self.assertTrue(np.isfinite(a).all())
        mask, depth, error = center_hits(mi, scene, r)
        self.assertTrue(mask[16, 16])
        self.assertFalse(mask[0, 0])
        self.assertLess(error, .001)
        self.assertGreater(depth[16, 16], 0.)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ['rgb', 'alpha', 'masks', 'previews', 'geometry_checks']:
                (root/folder).mkdir()
            rec = save_view(mi, scene, r, root, 16, 123)
            np.testing.assert_array_equal(np.load(root/rec['rgb']), a[..., :3])
            self.assertGreater(rec['interior_fraction'], .01)
            self.assertLess(rec['projection_error_px'], .001)


if __name__ == '__main__':
    unittest.main(verbosity=2)
