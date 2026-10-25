#!/usr/bin/env python3
"""Read-only HKTex checkpoint field probe. Requires the original CUDA environment.

Tests material-field forward consistency, not PRB backward or full-render parity.
Never changes training files, parameters on disk, or the eigen cache.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback

PARAMS = ('_mean_colour', '_kernel_colours', '_angles', '_anisotropies',
          '_thresholds', '_sharpnesses', '_kernel_locations', '_kernel_face_ids')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stats(values):
    import numpy as np
    a = np.asarray(values, dtype=np.float64).ravel()
    good = a[np.isfinite(a)]
    return dict(count=int(a.size), nonfinite=int(a.size-good.size),
                quantiles=dict(zip(('min', 'p01', 'p10', 'median', 'p90', 'p99', 'max'),
                                   np.quantile(good, [0, .01, .1, .5, .9, .99, 1]).tolist()))
                if good.size else {}, mean=float(good.mean()) if good.size else None)


def compare(reference, other):
    import numpy as np
    d = np.abs(np.asarray(reference)-np.asarray(other))
    return dict(abs_difference=stats(d),
                point_fraction_max_rgb_diff_gt_1e_4=float((d.max(-1) > 1e-4).mean()),
                point_fraction_max_rgb_diff_gt_1e_3=float((d.max(-1) > 1e-3).mean()),
                finite=bool(np.isfinite(d).all()))


def sample_surface(vertices, faces, count, seed):
    import numpy as np
    tri = vertices[faces]
    area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=-1)
    if not np.isfinite(area).all() or area.sum() <= 0:
        raise ValueError('Invalid mesh areas')
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(faces), size=count, p=area/area.sum())
    u, v = rng.random((2, count))
    su = np.sqrt(u)
    bary = np.stack([1-su, su*(1-v), su*v], axis=-1)
    points = (tri[ids]*bary[:, :, None]).sum(1).astype(np.float32)
    return points, ids.astype(np.int32)


def run(a, report, save):
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Run this probe in the server hktex CUDA environment')
    checkpoint = Path(a.checkpoint).resolve()
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    settings, state = ck['settings'], ck['texture']
    cfg_dict = settings['trainer_config']['network']
    ec = cfg_dict['eigen_albo']
    cache = Path(a.cache or (str(Path(ec['mesh_path']).with_suffix(''))+'_'+
                 ec.get('precomputed_name', 'eigen_albo')+'_'+
                 ec.get('local_frames', 'principal_curvatures')+'.pt')).resolve()
    if not cache.is_file():
        raise FileNotFoundError('Eigen cache missing; provide --cache PATH: '+str(cache))
    repo = Path(a.repo).resolve()
    mismatches = [p for p, h in settings['source_fingerprints'].items()
                  if not (repo/p).is_file() or sha256(repo/p) != h]
    if mismatches:
        raise RuntimeError('Source differs from checkpoint: '+', '.join(mismatches))
    report.update(checkpoint_sha256=sha256(checkpoint), cache_path=str(cache),
                  cache_sha256_now=sha256(cache), update=int(ck['update']),
                  historical_cache_hash_available=False,
                  scope='Area-uniform surface base colour; no image loss or path tracer. '
                        'Forward parity is not a backward-gradient validation.',
                  source_fingerprints_verified=True, settings=vars(a))
    save()
    sys.path.insert(0, str(repo))
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')
    import hktex
    from hktex.modules.mesh import Mesh
    from omegaconf import OmegaConf
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(a.seed)
    prefix = '_HeatKernelTexture__mesh.'
    geom = {n: state[prefix+n].numpy().copy() for n in ('verts', 'faces', 'fnorms', 'vnorms')}
    mesh = Mesh(geom['verts'], geom['faces'], geom['fnorms'], geom['vnorms'], device='cuda')
    cfg = OmegaConf.create(cfg_dict)
    cfg.model.n_sources = 1
    cfg.point_batching = 512
    # Use a temporary cache anchor; never silently recompute or overwrite the cache.
    with tempfile.TemporaryDirectory(prefix='hktex-probe-') as td:
        cfg.eigen_albo.mesh_path = str(Path(td)/'mesh.npz')
        cfg.eigen_albo.precomputed_name = 'probe'
        cfg.eigen_albo.use_precomputed = True
        cfg.eigen_albo.error_if_not_precomputed = True
        (Path(td)/f'mesh_probe_{cfg.eigen_albo.local_frames}.pt').symlink_to(cache)
        network = hktex.find(settings['trainer_config']['network_type'])(cfg, mesh=mesh)
    model, eig = network.model, network.eigalbo_interp
    for name in PARAMS:
        t = state[name].to('cuda').clone()
        setattr(model, name, torch.nn.Buffer(t) if name == '_kernel_face_ids'
                else torch.nn.Parameter(t, requires_grad=True))
    model.N_sources = len(state['_kernel_colours'])
    points, fids = sample_surface(geom['verts'], geom['faces'].astype(np.int64), a.points, a.seed)
    pts = torch.as_tensor(points, device='cuda')
    ids = torch.as_tensor(fids, device='cuda')
    arrays = dict(points=points, face_ids=fids)
    report['kernel_count'] = model.N_sources
    report['parameters'] = {n: stats(state[n].numpy()) for n in PARAMS if n != '_kernel_face_ids'}
    print('[PROBE] Loading complete; evaluate field and coverage', flush=True)
    with torch.no_grad():
        network.reset()
        network.inference_mode(True)
        network.prepare_kernels()
        vi = mesh.get_face_vertices(model.kernel_face_ids)
        bary = mesh.cartesian_to_barycentric(model.kernel_locations, vi)
        lam, _ = eig.knn_gather.gather_src(eig.grid[0], eig.grid[1], vi, bary,
                                         eig._eigen_val_knn, eig._eigen_vec_knn)
        t = float(model.diff_times.item())
        lt = (lam*t).cpu().numpy()
        arrays['kernel_lambda_times_t'] = lt
        report['spectrum'] = dict(diff_time=t, lambda_times_t=stats(lt),
                                  exp_minus_lambda_t=stats(np.exp(-lt)),
                                  note='Actual interpolated spectrum at checkpoint sources; '
                                       'these coefficients alone do not determine kernel support.')
        collected = {n: [] for n in ('rgb', 'preclamp_rgb', 'alpha_sum', 'alpha_max', 'effective_kernels')}
        for start in range(0, a.points, 512):
            info = model.prepare_points(mesh, eig, ids[start:start+512], pts=pts[start:start+512])
            colours, _, top_ids, weights = model.diffuse_heat_kernels(eig, info)
            w = weights.squeeze(-1).T
            ix = top_ids.squeeze(-1).T.long()
            total = w.sum(1)
            pre = model._mean_colour+(w[..., None]*model.kernel_colours[ix]).sum(1)/total[:, None].clamp_min(1.)
            vals = dict(rgb=model(colours), preclamp_rgb=pre, alpha_sum=total,
                        alpha_max=w.max(1).values,
                        effective_kernels=total.square()/(w.square().sum(1)+1e-20))
            for n, v in vals.items():
                collected[n].append(v.detach().cpu().numpy())
            print(f'[COVERAGE] {min(start+512, a.points)}/{a.points}', flush=True)
        arrays.update({n: np.concatenate(v) for n, v in collected.items()})
    if not all(np.isfinite(arrays[n]).all() for n in collected):
        raise FloatingPointError('Nonfinite field outputs')
    def coverage(mask):
        s, pre = arrays['alpha_sum'][mask], arrays['preclamp_rgb'][mask]
        return dict(points=int(mask.sum()), alpha_sum=stats(s),
                    fraction_sum_lt_001=float((s < .01).mean()),
                    fraction_sum_lt_01=float((s < .1).mean()),
                    fraction_sum_lt_1=float((s < 1).mean()),
                    alpha_max=stats(arrays['alpha_max'][mask]),
                    effective_kernels=stats(arrays['effective_kernels'][mask]),
                    clipped_channel_fraction=float(((pre < 0) | (pre > 1)).mean()),
                    clipped_point_fraction=float(((pre < 0) | (pre > 1)).any(-1).mean()))
    mid_y = (geom['verts'][:, 1].min()+geom['verts'][:, 1].max())/2
    report['coverage'] = {n: coverage(mask) for n, mask in {
        'all': np.ones(a.points, bool), 'lower_y_half': points[:, 1] < mid_y,
        'upper_y_half': points[:, 1] >= mid_y}.items() if mask.any()}
    report['coverage_note'] = ('sum<1 is attenuation, not automatically a bug; sum near 0 tends '
        'toward mean colour. Y partitions are spatial partitions, not semantic body labels.')
    save()

    def forward(chunk, grad, inference):
        print(f'[PARITY] chunk={chunk}, grad={grad}, inference={inference}', flush=True)
        result = []
        with torch.set_grad_enabled(grad):
            network.reset()
            network.inference_mode(inference)
            network.cfg.point_batching = chunk
            if inference:
                network.prepare_kernels()
            for start in range(0, a.points, 512):
                p = pts[start:start+512].detach().requires_grad_(grad)
                v = network(p, face_ids=ids[start:start+512])
                result.append(v.detach().cpu().numpy())
                del v, p
        network.reset()
        return np.concatenate(result)

    report['forward_parity'] = {}
    for name, chunk, grad, inference in (
        ('inference_repeat_512', 512, False, True),
        ('inference_chunk64', 64, False, True),
        ('trainlike_grad_512', 512, True, False),
        ('trainlike_grad_64', 64, True, False)):
        other = forward(chunk, grad, inference)
        arrays[name] = other
        report['forward_parity'][name] = compare(arrays['rgb'], other)
        save()
    for n in PARAMS:
        if not torch.equal(getattr(model, n).detach().cpu(), state[n]):
            raise AssertionError('In-memory model parameter changed: '+n)
    np.savez_compressed(Path(a.output)/'field_probe.npz', **arrays)
    report['parameters_unchanged'] = True
    report['status'] = 'complete'
    save()
    print('[SUCCESS]', Path(a.output)/'diagnostic.json', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--repo', default='/root/hktex-work/hktex')
    p.add_argument('--cache', help='Exact original spectral cache, if moved')
    p.add_argument('--output', required=True, help='New directory only')
    p.add_argument('--points', type=int, default=2048)
    p.add_argument('--seed', type=int, default=731)
    p.add_argument('--gpu', default='0')
    a = p.parse_args()
    if a.points < 512:
        p.error('--points must be >=512 for batch-size comparisons')
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    os.environ.setdefault('MPLBACKEND', 'Agg')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(version='1.0.0', status='running')
    def save():
        (out/'diagnostic.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    try:
        run(a, report, save)
    except Exception:
        report.update(status='failed', error=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()
