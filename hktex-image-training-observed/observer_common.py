"""Read-only training observation. No forward passes, rendering, or global RNG use."""
import hashlib
import json
from pathlib import Path
import time
from functools import wraps
import numpy as np

PARAMS = ('_kernel_locations', '_kernel_colours', '_angles', '_anisotropies',
          '_thresholds', '_sharpnesses', '_mean_colour', '_kernel_face_ids')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def npz_write(path, **arrays):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def cpu(value):
    return value.detach().cpu().numpy().copy()


class Lineage:
    def __init__(self, count):
        self.ids = np.arange(count, dtype=np.int64)
        self.roots = self.ids.copy()
        self.next_id = count

    def apply(self, kind, mask):
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if len(mask) != len(self.ids):
            raise ValueError('Lineage mask length mismatch')
        selected, roots = self.ids[mask].copy(), self.roots[mask].copy()
        n = len(selected) * (2 if kind == 'split' else 1)
        children = np.arange(self.next_id, self.next_id + n, dtype=np.int64) if kind != 'prune' else np.array([], dtype=np.int64)
        if kind == 'clone':
            self.ids = np.r_[self.ids, children]
            self.roots = np.r_[self.roots, roots]
            parents = selected
        elif kind == 'split':
            self.ids = np.r_[self.ids[~mask], children]
            self.roots = np.r_[self.roots[~mask], np.tile(roots, 2)]
            parents = np.tile(selected, 2)  # actual code: positive children, then negative children
        elif kind == 'prune':
            self.ids, self.roots = self.ids[~mask], self.roots[~mask]
            parents = np.array([], dtype=np.int64)
        else:
            raise ValueError(kind)
        self.next_id += len(children)
        return {'selected_ids': selected.tolist(), 'child_ids': children.tolist(), 'parent_ids': parents.tolist()}


def sample_surface(vertices, faces, count, seed=19019):
    rng = np.random.default_rng(seed)  # never consume the training RNG
    tri = vertices[faces]
    areas = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
    fids = rng.choice(len(faces), size=count, p=areas/areas.sum())
    q = rng.random((count, 2))
    root = np.sqrt(q[:, 0])
    bary = np.c_[1-root, root*(1-q[:, 1]), root*q[:, 1]]
    pos = (vertices[faces[fids]] * bary[:, :, None]).sum(1)
    return fids.astype(np.int64), bary.astype(np.float32), pos.astype(np.float32)


