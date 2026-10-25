"""Portable data contract and Mitsuba utilities; no HKTex import at module scope."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

VERSION = '1.0.0'
BASE_COMMIT = '8c926af44a3a12c80c3fdf1dd62e4f2ba5594630'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def srgb_to_linear(x):
    x = np.asarray(x, np.float32)
    return np.where(x <= .04045, x/12.92, ((x+.055)/1.055)**2.4).astype(np.float32)


def linear_to_srgb(x):
    x = np.maximum(np.asarray(x, np.float32), 0)
    return np.where(x <= .0031308, 12.92*x, 1.055*x**(1/2.4)-.055)


def save_preview(path, rgb):
    Image.fromarray(np.round(np.clip(linear_to_srgb(rgb), 0, 1)*255).astype('uint8')).save(path)


def camera_record(index, azimuth, elevation, split, size, distance, fov):
    """Mitsuba local: left/up/forward; OpenCV: right/down/forward."""
    a, e = np.deg2rad([azimuth, elevation])
    eye = distance*np.array([np.cos(e)*np.sin(a), np.sin(e), np.cos(e)*np.cos(a)])
    forward = -eye/np.linalg.norm(eye)
    left = np.cross([0., 1., 0.], forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    c2w = np.eye(4)
    c2w[:3, :3] = np.column_stack([left, up, forward])
    c2w[:3, 3] = eye
    cv_w2c = np.diag([-1., -1., 1., 1.]) @ np.linalg.inv(c2w)
    f = size/(2*np.tan(np.deg2rad(fov)/2))
    K = [[f, 0., size/2], [0., f, size/2], [0., 0., 1.]]
    return dict(id=f'view_{index:03d}', frame_id=0, pose_id=0, split=split,
                width=size, height=size, azimuth_deg=azimuth, elevation_deg=elevation,
                fov_x_deg=fov, near_clip=.01, far_clip=1000., K=K,
                world_to_camera_cv=cv_w2c.tolist(), camera_to_world_mitsuba=c2w.tolist())


def cameras(preset, size, distance, fov):
    if preset == 'smoke':
        layout = [(90*i, 0, s) for i, s in enumerate(['train', 'train', 'val', 'test'])]
    else:
        layout = [(30*j+10*r, e, 'val' if j % 6 == 2 else 'test' if j % 6 == 5 else 'train')
                  for r, e in enumerate([-15., 0., 20.]) for j in range(12)]
    return [camera_record(i, a, e, s, size, distance, fov) for i, (a, e, s) in enumerate(layout)]


def project(points, record):
    p = np.asarray(points, np.float64)
    ext = np.asarray(record['world_to_camera_cv'])
    q = p @ ext[:3, :3].T + ext[:3, 3]
    k = np.asarray(record['K'])
    uv = q @ k.T
    return uv[:, :2]/uv[:, 2:], q[:, 2]


def sensor_dict(mi, record, spp):
    return dict(type='perspective', fov=record['fov_x_deg'], fov_axis='x',
                near_clip=record['near_clip'], far_clip=record['far_clip'],
                to_world=mi.ScalarTransform4f(np.asarray(record['camera_to_world_mitsuba'])),
                film=dict(type='hdrfilm', width=record['width'], height=record['height'],
                          pixel_format='rgba', component_format='float32', rfilter=dict(type='box')),
                sampler=dict(type='independent', sample_count=spp))


def center_hits(mi, scene, record):
    import drjit as dr
    w, h = record['width'], record['height']
    if mi.variant() == 'scalar_rgb':
        # CPU reference implementation for small integration fixtures.
        valid, points = np.zeros((h, w), bool), np.zeros((h, w, 3), np.float32)
        for y in range(h):
            for x in range(w):
                ray, _ = scene.sensors()[0].sample_ray(0., .5, mi.Point2f((x+.5)/w, (y+.5)/h), mi.Point2f(.5))
                hit = scene.ray_intersect(ray)
                valid[y, x] = hit.is_valid()
                if hit.is_valid():
                    points[y, x] = np.asarray(hit.p)
    else:
        idx = dr.arange(mi.UInt32, w*h)
        x, y = idx % w, idx // w
        pos = mi.Point2f((mi.Float(x)+.5)/w, (mi.Float(y)+.5)/h)
        ray, _ = scene.sensors()[0].sample_ray(
            0., .5, pos, mi.Point2f(.5)
        )
        hit = scene.ray_intersect(ray)
        valid = np.asarray(hit.is_valid()).reshape(h, w)
        points = np.asarray(hit.p).T.reshape(h, w, 3)
    expected = np.stack(np.meshgrid(np.arange(w)+.5, np.arange(h)+.5), axis=-1)
    if not valid.any():
        raise ValueError('Camera misses mesh: '+record['id'])
    uv, depth = project(points[valid], record)
    error = float(np.max(np.abs(uv-expected[valid])))
    if error > .02:
        raise ValueError(f'Camera convention mismatch: {error:.6f} pixels')
    z = np.zeros((h, w), np.float32)
    z[valid] = depth
    return valid, z, error


def interior_mask(center, alpha):
    """Conservative initial training mask; full images retained for later edge training."""
    mask = center & (alpha >= 1-1e-6)
    padded = np.pad(mask, 1, constant_values=False)
    return np.logical_and.reduce([padded[y:y+mask.shape[0], x:x+mask.shape[1]]
                                  for y in range(3) for x in range(3)])


def save_view(mi, scene, record, root, spp, seed):
    raw = np.asarray(mi.render(scene, spp=spp, seed=seed), dtype=np.float32)
    if raw.shape != (record['height'], record['width'], 4) or not np.isfinite(raw).all():
        raise ValueError('Expected finite linear RGBA render')
    rgb, alpha = raw[..., :3], raw[..., 3]
    center, depth, error = center_hits(mi, scene, record)
    interior = interior_mask(center, alpha)
    stem = record['id']
    paths = dict(rgb=f'rgb/{stem}.npy', alpha=f'alpha/{stem}.npy',
                 mask=f'masks/{stem}.png', geometry=f'geometry_checks/{stem}.npz',
                 preview=f'previews/{stem}.png')
    np.save(root/paths['rgb'], rgb)
    np.save(root/paths['alpha'], alpha)
    Image.fromarray(interior.astype('uint8')*255).save(root/paths['mask'])
    np.savez_compressed(root/paths['geometry'], center_mask=center, depth_cv=depth)
    save_preview(root/paths['preview'], rgb)
    return dict(record, **paths, spp=spp, seed=seed, projection_error_px=error,
                foreground_fraction=float(center.mean()), interior_fraction=float(interior.mean()))


def contact_sheet(root, records):
    thumb, label, cols = 192, 30, 6
    sheet = Image.new('RGB', (cols*thumb, ((len(records)+cols-1)//cols)*(thumb+label)), 'white')
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(records):
        x, y = (i % cols)*thumb, (i//cols)*(thumb+label)
        with Image.open(root/r['preview']) as im:
            sheet.paste(im.resize((thumb, thumb), Image.Resampling.LANCZOS), (x, y+label))
        d.text((x+4, y+6), f"{r['id']} {r['split']} el={r['elevation_deg']:g}", fill='black')
    sheet.save(root/'contact_sheet.png')


def validate(root, write=True):
    """Independent CPU-only validation; intentionally never reads eval_gt texture."""
    root = Path(root)
    m = read_json(root/'manifest.json')
    errors, warnings, views = [], [], []
    if m['status'] != 'complete':
        errors.append('Export is incomplete')
    for rel, expected in m['sha256'].items():
        if not (root/rel).is_file() or sha256(root/rel) != expected:
            errors.append('Missing/changed asset: '+rel)
    geom = np.load(root/m['train_mesh'], allow_pickle=False)
    vertices, faces = geom['vertices'], geom['faces']
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        errors.append('Invalid geometry')
    counts = {s: 0 for s in ['train', 'val', 'test']}
    seen = set()
    for r in m['views']:
        counts[r['split']] += 1
        pose = np.asarray(r['camera_to_world_mitsuba'])
        key = tuple(np.round(pose.ravel(), 10))
        if key in seen:
            errors.append('Repeated camera pose: '+r['id'])
        seen.add(key)
        expected_ext = np.diag([-1., -1., 1., 1.]) @ np.linalg.inv(pose)
        if not np.allclose(expected_ext, r['world_to_camera_cv'], atol=1e-9):
            errors.append('Inconsistent extrinsics: '+r['id'])
        f = r['width']/(2*np.tan(np.deg2rad(r['fov_x_deg'])/2))
        expected_k = [[f, 0, r['width']/2], [0, f, r['height']/2], [0, 0, 1]]
        if not np.allclose(expected_k, r['K'], atol=1e-9):
            errors.append('Inconsistent intrinsics: '+r['id'])
        rgb = np.load(root/r['rgb'], allow_pickle=False)
        alpha = np.load(root/r['alpha'], allow_pickle=False)
        mask = np.asarray(Image.open(root/r['mask'])) > 0
        checks = np.load(root/r['geometry'], allow_pickle=False)
        h, w = r['height'], r['width']
        if rgb.shape != (h, w, 3) or alpha.shape != (h, w) or mask.shape != (h, w):
            errors.append('Wrong image dimensions: '+r['id'])
            continue
        if not np.isfinite(rgb).all() or not np.isfinite(alpha).all():
            errors.append('Nonfinite image: '+r['id'])
        if alpha.min() < -1e-5 or alpha.max() > 1+1e-5:
            errors.append('Alpha outside [0,1]: '+r['id'])
        if mask.mean() < .005:
            errors.append('Insufficient foreground: '+r['id'])
        if not np.array_equal(mask, interior_mask(checks['center_mask'], alpha)):
            errors.append('Inconsistent mask: '+r['id'])
        uv, z = project(vertices, r)
        if z.min() <= r['near_clip'] or z.max() >= r['far_clip']:
            errors.append('Mesh outside depth range: '+r['id'])
        margin = float(min(uv[:, 0].min(), uv[:, 1].min(), w-uv[:, 0].max(), h-uv[:, 1].max()))
        if margin < 2:
            errors.append('Mesh is cropped or too close to image border: '+r['id'])
        views.append(dict(id=r['id'], split=r['split'], mask_fraction=float(mask.mean()),
                          framing_margin_px=margin, rgb_min=float(rgb.min()), rgb_max=float(rgb.max()),
                          projection_error_px=r['projection_error_px']))
        if r['projection_error_px'] > .02:
            errors.append('Projection check failed: '+r['id'])
    if not all(counts.values()):
        errors.append('Missing train/val/test split')
    report = dict(version=VERSION, passed=not errors, errors=errors, warnings=warnings,
                  counts=counts, views=views, scope='Data integrity and camera consistency, not training quality')
    if write:
        write_json(root/'validation.json', report)
    return report
