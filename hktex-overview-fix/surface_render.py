"""Orthographic, two-sided triangle rasterization at pixel centres (CPU/NumPy).

RGB is evaluated at visible surface points, never interpolated from sparse samples.
No lighting or gamma transform: diagnostic images show the model's RGB values.
"""
import numpy as np


def camera(vertices, height=384):
    lo, hi = np.min(vertices, axis=0), np.max(vertices, axis=0)
    width = int(round(height * .75))
    scale = min((width - 2) * .9 / max(hi[0]-lo[0], 1e-8),
                (height - 2) * .9 / max(hi[1]-lo[1], 1e-8))
    return dict(height=height, width=width, scale=float(scale),
                centre=((lo + hi) / 2).tolist())


def project(points, cam, sign=1):
    p = np.asarray(points, dtype=np.float64) - np.asarray(cam['centre'])
    return np.column_stack((sign*p[:, 0]*cam['scale'] + cam['width']/2,
                            -p[:, 1]*cam['scale'] + cam['height']/2,
                            sign*p[:, 2]))


def rasterize(vertices, faces, cam, sign=1):
    """Larger camera-space Z wins. Equal-depth faces use the lower face index."""
    h, w = cam['height'], cam['width']
    screen = project(vertices, cam, sign)
    depth = np.full((h, w), -np.inf, dtype=np.float64)
    face_map = np.full((h, w), -1, dtype=np.int64)
    bary_map = np.zeros((h, w, 3), dtype=np.float32)
    for fid, face in enumerate(faces):
        t = screen[face]
        x0, y0 = np.maximum(np.ceil(t[:, :2].min(0)-.5).astype(int), 0)
        x1, y1 = np.minimum(np.floor(t[:, :2].max(0)-.5).astype(int), [w-1, h-1])
        if x1 < x0 or y1 < y0:
            continue
        a, b, c = t
        den = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
        if abs(den) < 1e-14:
            continue
        xx, yy = np.meshgrid(np.arange(x0, x1+1)+.5, np.arange(y0, y1+1)+.5)
        u = ((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
        v = ((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den
        bc = np.stack([u, v, 1-u-v], axis=-1)
        inside = (bc >= -1e-10).all(-1)
        # Clamp only sub-roundoff boundary excursions, then maintain partition of unity.
        bc = np.maximum(bc, 0)
        bc /= bc.sum(-1, keepdims=True)
        z = bc @ t[:, 2]
        old = depth[y0:y1+1, x0:x1+1]
        hit = inside & (z > old)
        old[hit] = z[hit]
        face_map[y0:y1+1, x0:x1+1][hit] = fid
        bary_map[y0:y1+1, x0:x1+1][hit] = bc[hit]
    mask = face_map >= 0
    if not mask.any():
        raise ValueError('No visible triangles; check mesh axes and camera')
    return dict(face_map=face_map, bary_map=bary_map, depth=depth,
                face_ids=face_map[mask], bary=bary_map[mask], mask=mask)


def image_from_values(mask, values):
    image = np.ones((*mask.shape, 3), dtype=np.float32)
    image[mask] = values
    return image


def centre_overlay(view, centres, cam, sign):
    """Small centre markers with approximate pixel-depth occlusion, not kernel radii."""
    image = image_from_values(view['mask'], np.full((view['mask'].sum(), 3), .85))
    p = project(centres, cam, sign)
    xy = np.floor(p[:, :2]).astype(int)
    h, w = view['mask'].shape
    for (x, y), z in zip(xy, p[:, 2]):
        if not (0 <= x < w and 0 <= y < h and view['mask'][y, x]):
            continue
        # Pixel-centre depth and projected kernel depth sample slightly different points.
        if abs(z - view['depth'][y, x]) <= 1.5 / cam['scale']:
            image[y, x] = [.0, .32, .60]
    return image
