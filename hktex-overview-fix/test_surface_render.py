"""CPU checks: coverage, correct depth, barycentric positions and offline regeneration."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from surface_render import camera, rasterize, project, centre_overlay
from fix_overviews import digest, save_npz


class RasterTests(unittest.TestCase):
    def setUp(self):
        self.square = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float)
        self.faces = np.array([[0, 1, 2], [0, 2, 3]])
        self.cam = dict(height=8, width=8, scale=2., centre=[0., 0., 0.])

    def test_no_diagonal_holes_and_correct_barycentric_position(self):
        v = rasterize(self.square, self.faces, self.cam)
        expected = np.zeros((8, 8), bool)
        expected[2:6, 2:6] = True
        np.testing.assert_array_equal(v['mask'], expected)
        points = (self.square[self.faces[v['face_ids']]] * v['bary'][..., None]).sum(1)
        screen = project(points, self.cam)
        y, x = np.nonzero(v['mask'])
        np.testing.assert_allclose(screen[:, :2], np.column_stack([x+.5, y+.5]), atol=1e-7)
        np.testing.assert_allclose(v['bary'].sum(1), 1, atol=1e-7)

    def test_front_back_occlusion_and_centres(self):
        verts = np.r_[self.square+[0, 0, -.5], self.square+[0, 0, .5]]
        faces = np.r_[self.faces, self.faces+4]
        front = rasterize(verts, faces, self.cam, 1)
        back = rasterize(verts, faces, self.cam, -1)
        self.assertTrue((front['face_ids'] >= 2).all())
        self.assertTrue((back['face_ids'] < 2).all())
        # Use a fine camera so hidden layer is farther than the pixel-depth tolerance.
        cam = dict(height=80, width=80, scale=20., centre=[0., 0., 0.])
        view = rasterize(verts, faces, cam, 1)
        hidden = centre_overlay(view, np.array([[0, 0, -.5]]), cam, 1)
        visible = centre_overlay(view, np.array([[0, 0, .5]]), cam, 1)
        np.testing.assert_allclose(hidden[40, 40], [.85]*3, atol=1e-7)
        np.testing.assert_allclose(visible[40, 40], [0., .32, .6], atol=1e-7)

    def test_cpu_regeneration_keeps_source_and_pixel_data(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)/'observer'
            (root/'analysis').mkdir(parents=True)
            (root/'snapshots').mkdir()
            (root/'viewer').mkdir()
            (root/'viewer'/'index.html').write_text('<html>original viewer</html>')
            (root/'manifest.json').write_text('{}')
            np.savez(root/'surface.npz', vertices=self.square, faces=self.faces)
            name = '00001_u00001_final.npz'
            np.savez(root/'snapshots'/name, _kernel_locations=np.array([[0., 0., 0.]]))
            (root/'analysis'/'metrics.json').write_text(json.dumps([
                dict(file=name, completed_updates=1, phase='final', kernels=1)]))
            out = root/'viewer-surface-v2'
            (out/'data').mkdir(parents=True)
            views = [rasterize(self.square, self.faces, camera(self.square, 96), s) for s in [1, -1]]
            values = {}
            for i, v in enumerate(views):
                values[f'gt_{i}'] = np.tile([.2, .3, .4], (len(v['face_ids']), 1)).astype(np.float32)
                values[f'predicted_{i}'] = values[f'gt_{i}'].copy()
            pixels = out/'data'/(Path(name).stem+'_pixels.npz')
            save_npz(pixels, **values, snapshot_sha256=np.asarray(digest(root/'snapshots'/name)))
            before = {p: digest(p) for p in [pixels, root/'surface.npz', root/'snapshots'/name, root/'viewer'/'index.html']}
            env = dict(os.environ, MPLCONFIGDIR=str(Path(td)/'mpl'))
            subprocess.run([sys.executable, str(Path(__file__).with_name('fix_overviews.py')),
                '--recording', str(root), '--height', '96', '--plot-only'], check=True, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for path, hash_before in before.items():
                self.assertEqual(digest(path), hash_before)
            self.assertTrue((out/'index.html').is_file())
            self.assertTrue((out/'training.gif').is_file())
            self.assertTrue((out/'original-reference'/'index.html').is_file())
            report = json.loads((out/'render_report.json').read_text())
            for v in report[Path(name).stem]:
                self.assertEqual(v['visible_pixel_mse'], 0.)


if __name__ == '__main__':
    unittest.main()
