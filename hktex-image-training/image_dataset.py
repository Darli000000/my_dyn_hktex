"""Image-only supervision. This module has no torch, HKTex or texture dependency."""
from collections import OrderedDict
from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image
from offline_common import read_json, sha256, validate, project

VERSION = '1.0.0'


def inside(root, relative):
    p = (root/relative).resolve()
    if p == root or root not in p.parents or 'eval_gt' in p.relative_to(root).parts:
        raise ValueError('Not a training/evaluation image asset: '+str(relative))
    return p


def check_contract(root, expected_size=None):
    root = Path(root).resolve()
    m = read_json(root/'manifest.json')
    if m.get('schema') != 'hktex-offline-multiview-v1':
        raise ValueError('Unsupported dataset schema')
    for r in m['views']:
        for key in ['rgb', 'alpha', 'mask', 'geometry', 'preview']:
            inside(root, r[key])
        if expected_size and (r['width'] != expected_size or r['height'] != expected_size):
            raise ValueError(f"{r['id']} is not {expected_size} x {expected_size}")
        if r['width'] != r['height']:
            raise ValueError('v1 supports square images only')
    inside(root, m['train_mesh'])
    inside(root, m['renderer_config'])
    for name in m['sha256']:
        inside(root, name)
    # Hash and examine one view at a time. No eval_gt and no original mesh/MTL.
    report = validate(root, write=False)
    if not report['passed']:
        raise ValueError(report['errors'])
    replay_path = root/'replay_validation.json'
    if not replay_path.is_file() or not read_json(replay_path).get('passed'):
        raise ValueError('Run A1 verify first: replay_validation.json must pass')
    scene = m['scene_contract']
    required = dict(material='principled', specular=0., metallic=0., roughness=1.,
                    max_depth=3, constant_environment_radiance=[1., 1., 1.], ground=False,
                    integrator='path', hide_emitters=True, denoise=False, exposure=1., tone_mapping='none')
    for k, v in required.items():
        if scene.get(k) != v:
            raise ValueError(f'Unsupported scene contract: {k}={scene.get(k)}')
    if m['image_contract'].get('filter') != 'box; uniform subpixel sampling':
        raise ValueError('Expected box-filter images')
    if not m['image_contract'].get('rgb', '').startswith('float32 HWC linear radiance'):
        raise ValueError('Expected linear float32 radiance images')
    g = np.load(inside(root, m['train_mesh']), allow_pickle=False)
    vertices, faces, normals = g['vertices'], g['faces'], g['normals']
    if normals.shape != vertices.shape or not np.isfinite(normals).all():
        raise ValueError('Invalid saved normals')
    tri = vertices[faces]
    if (np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1) <= 1e-12).any():
        raise ValueError('Degenerate mesh faces')
    cfg = read_json(inside(root, m['renderer_config']))
    checks = [cfg['ground_plane_config']['activated'] is False,
              cfg['emitter_config']['envmap_path'] is None,
              cfg['emitter_config']['radiance'] == 1.,
              cfg['mitsuba_mesh_config']['twosided'] is False,
              cfg['integrator_config']['hide_emitters'] is True,
              cfg['integrator_config']['meta']['max_depth'] == 3]
    if not all(checks):
        raise ValueError('Renderer config disagrees with scene contract')
    return dict(report, dataset=str(root), manifest_sha256=sha256(root/'manifest.json'),
                mesh_sha256=sha256(root/m['train_mesh']),
                image_sizes=sorted({(r['width'], r['height']) for r in m['views']}),
                replay_validation=read_json(replay_path),
                original_mesh_for_provenance_only=m.get('provenance', {}).get('source_mesh'),
                training_reads_original_texture=False)


class ImageDataset:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = read_json(self.root/'manifest.json')
        self.views = self.manifest['views']
        self.by_split = {s: [r for r in self.views if r['split'] == s] for s in ['train', 'val', 'test']}
        self._images, self._indices = OrderedDict(), {}
        self.read_paths = set()
        self._order = np.empty(0, dtype=int)
        self._cursor = 0

    def image(self, r):
        key = r['id']
        if key not in self._images:
            p = inside(self.root, r['rgb'])
            self.read_paths.add(str(p))
            self._images[key] = np.load(p, mmap_mode='r', allow_pickle=False)
            if len(self._images) > 4:
                self._images.popitem(last=False)
        self._images.move_to_end(key)
        return self._images[key]

    def indices(self, r):
        if r['id'] not in self._indices:
            p = inside(self.root, r['mask'])
            self.read_paths.add(str(p))
            mask = np.asarray(Image.open(p)) > 0
            ids = np.flatnonzero(mask).astype(np.uint32)
            if not len(ids):
                raise ValueError('Empty interior mask: '+r['id'])
            self._indices[r['id']] = ids
        return self._indices[r['id']]

    def target(self, r, pixel_ids):
        y, x = np.divmod(pixel_ids, r['width'])
        return np.asarray(self.image(r)[y, x], np.float32).copy()

    def train_batch(self, rng, count):
        # Cycle through a shuffled camera list; no camera starves in short runs.
        if self._cursor >= len(self._order):
            self._order = rng.permutation(len(self.by_split['train']))
            self._cursor = 0
        r = self.by_split['train'][int(self._order[self._cursor])]
        self._cursor += 1
        ids = rng.choice(self.indices(r), count, replace=True)
        return r, ids, self.target(r, ids)

    def fixed_pixels(self, r, count, seed):
        # Stable per-view seeds, independent of Python's randomized hash.
        import hashlib
        salt = int.from_bytes(hashlib.sha256(r['id'].encode()).digest()[:4], 'little')
        rng = np.random.default_rng(seed+salt)
        return rng.choice(self.indices(r), min(count, len(self.indices(r))), replace=False)

    def sampler_state(self):
        return dict(order=self._order.tolist(), cursor=self._cursor)

    def load_sampler_state(self, state):
        self._order, self._cursor = np.asarray(state['order'], dtype=int), int(state['cursor'])


def pixel_samples(record, pixel_ids, aa, transport, rng):
    """Order is [pixel, subpixel, transport], so reducing aa*transport is correct."""
    y, x = np.divmod(np.asarray(pixel_ids), record['width'])
    offsets = rng.random((len(pixel_ids), aa, 2)).astype(np.float32)
    positions = np.stack([x, y], -1)[:, None, :]+offsets
    positions /= [record['width'], record['height']]
    return np.repeat(positions[:, :, None, :], transport, axis=2).reshape(-1, 2).astype(np.float32)


def install_no_texture_guard(root):
    """Fail if any Python open attempts to read original source assets or eval_gt."""
    root = Path(root).resolve()
    m = read_json(root/'manifest.json')
    blocked = [root/'eval_gt']
    src = m.get('provenance', {}).get('source_mesh')
    if src:
        blocked.append(Path(src).resolve().parent)
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, Path)):
            return
        value = args[0].decode() if isinstance(args[0], bytes) else args[0]
        path = Path(value).resolve()
        for parent in blocked:
            if path == parent or parent in path.parents:
                raise PermissionError('Image training must not read reference texture/source assets: '+str(path))
    sys.addaudithook(audit)
    return [str(p) for p in blocked]
