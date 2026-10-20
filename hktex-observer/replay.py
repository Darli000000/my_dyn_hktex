"""Post-training CUDA replay through HKTex's own interpolation/KNN/heat/filter code.

Produces numerical analysis files. Plotting is a separate CPU-only command.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import numpy as np
from recording import PARAMS, cpu, json_write, npz_write, sha256


def stage_values(model, eig, info):
    import torch
    idx = info['indices']
    source = eig.knn_heat._cache.evecs_src[idx]
    lam, phi = info['evals'], info['evecs']
    heat = eig.knn_heat._heat_qk_fn(lam, phi, source, model.diff_times, idx)
    self_heat = eig.knn_heat._cache.src_self_heat[idx]
    weights = info['weights']
    if weights is None: weights = torch.ones_like(heat)
    den = self_heat + eig.knn_heat.config.eps
    normalized = heat / den
    weighted = (heat * weights) / den  # actual implementation's operation order
    filtered_input = weighted ** model.cfg.power_diffused_diracs
    alpha = model.kernel_filter_func(filtered_input, epsilon=model.thresholds[idx], sharpness=model.sharpnesses[idx])
    return dict(raw=heat, normalized=normalized, weighted=weighted, alpha=alpha,
                distance_squared=info['distances'], distance_weight=weights,
                self_heat=self_heat, evals=lam, phi_query=phi, phi_source=source)


def reconstruct_colour(alpha, colours, mean, inner_k):
    import torch
    values, local = alpha.topk(k=min(inner_k or alpha.shape[1], alpha.shape[1]), dim=1)
    chosen = torch.gather(colours, 1, local[..., None].expand(-1, -1, colours.shape[-1]))
    denom = values.sum(1, keepdim=True).clamp_min(1.)
    contributions = values[..., None] * chosen / denom[..., None]
    preclamp = mean + contributions.sum(1)
    return preclamp.clamp(0., 1.), local, contributions, preclamp


def choose_roots(initial, count):
    positions = initial['_kernel_locations']
    ids = initial['stable_ids']
    chosen = [int(np.argmax(positions[:, 1]))]  # start at highest Y; spatially spread thereafter
    d2 = np.full(len(ids), np.inf)
    for _ in range(min(count, len(ids))-1):
        d2 = np.minimum(d2, np.sum((positions-positions[chosen[-1]])**2, axis=1))
        d2[chosen] = -1
        chosen.append(int(np.argmax(d2)))
    return ids[chosen].tolist()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', required=True)
    p.add_argument('--repo', default='/root/hktex-work/hktex')
    p.add_argument('--cache', help='Moved eigen cache; SHA256 must match the recording')
    p.add_argument('--gpu', default='0')
    p.add_argument('--chunk', type=int, default=64)
    p.add_argument('--snapshots', default='regular', help='regular, all (includes ADC events), or comma-separated snapshot serials')
    p.add_argument('--roots', help='Comma-separated initial stable IDs to follow; defaults to six spatially spread roots')
    p.add_argument('--probes', type=int, default=8)
    p.add_argument('--output', help='Default: <recording>/analysis')
    return p.parse_args()


def main():
    a = parse_args()
    if a.chunk < 1 or not 1 <= a.probes <= a.chunk: raise SystemExit('Require chunk >= probes >= 1')
    root, repo = Path(a.recording).resolve(), Path(a.repo).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    cache = Path(a.cache or manifest['cache_path']).resolve()
    if sha256(cache) != manifest['cache_sha256']: raise SystemExit('Eigen cache hash mismatch')
    differences = [name for name, digest in manifest['source_hashes'].items()
                   if not (repo/name).is_file() or sha256(repo/name) != digest]
    if differences: raise SystemExit('HKTex source changed since recording: ' + ', '.join(differences[:12]))
    sys.path.insert(0, str(repo))
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', a.gpu)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    import torch
    from omegaconf import OmegaConf
    from hktex.modules.mesh import Mesh
    from hktex.modules.eigen_albo_knn import EigenAlboInterpolationKNN
    from hktex.modules.heat_kernel_texture_knn import HeatKernelTextureKNN
    from hktex.utils import interpolate_barycentric_attr_from_trivertidx as interp
    if not torch.cuda.is_available(): raise SystemExit('Numerical replay requires the existing HKTex CUDA environment')
    torch.set_float32_matmul_precision(manifest['matmul_precision'])
    torch.backends.cuda.matmul.allow_tf32 = manifest['allow_tf32']
    surface = dict(np.load(root/'surface.npz', allow_pickle=False))
    mesh = Mesh(surface['vertices'], surface['faces'], surface['fnorms'], surface['vnorms'], device='cuda')
    mc = OmegaConf.create(manifest['model_config'])
    if mc.out_net or mc.normalize_colours or mc.mass_type != 'one':
        raise SystemExit('v1 replay supports the current UV config: out_net=False, normalize_colours=False, mass_type=one')
    mc.n_sources = 1
    model = HeatKernelTextureKNN(mc, mesh)
    index = json.loads((root/'snapshots.json').read_text())
    if a.snapshots == 'regular': selected = [r for r in index if r['phase'] in ('initial', 'post_update', 'final')]
    elif a.snapshots == 'all': selected = index
    else:
        wanted = {int(i) for i in a.snapshots.split(',')}
        selected = [r for r in index if int(r['file'].split('_')[0]) in wanted]
        if len(selected) != len(wanted): raise SystemExit('Unknown snapshot serial')
    if not selected: raise SystemExit('No snapshots selected')
    with np.load(root/'snapshots'/index[0]['file'], allow_pickle=False) as first:
        roots = [int(x) for x in a.roots.split(',')] if a.roots else choose_roots(first, 6)
        if not set(roots).issubset(set(first['stable_ids'].tolist())): raise SystemExit('roots must be initial stable IDs')
    out = Path(a.output).resolve() if a.output else root/'analysis'
    out.mkdir(parents=True, exist_ok=True)
    ec = OmegaConf.create(manifest['eigen_config'])
    # Read the exact saved cache through an isolated temporary symlink. Never recompute.
    with tempfile.TemporaryDirectory(prefix='hktex-replay-') as td:
        ec.mesh_path = str(Path(td)/'mesh.obj')
        ec.precomputed_name = 'replay'
        ec.use_precomputed = True
        ec.error_if_not_precomputed = True
        (Path(td)/f'mesh_replay_{ec.local_frames}.pt').symlink_to(cache)
        eig = EigenAlboInterpolationKNN(ec, mesh)
    faces = torch.tensor(surface['face_ids'], device='cuda', dtype=torch.long)
    bary = torch.tensor(surface['bary'], device='cuda', dtype=torch.float32)
    vert_idx = mesh.get_face_vertices(faces)
    query_embeddings = interp(vert_idx, bary, eig._iso_embeddings)
    rows = []
    for item in selected:
        name = Path(item['file']).stem
        print('[REPLAY]', name, flush=True)
        state = dict(np.load(root/'snapshots'/item['file'], allow_pickle=False))
        with torch.no_grad():
            for key in PARAMS:
                tensor = torch.as_tensor(state[key], device='cuda')
                if key == '_kernel_face_ids': setattr(model, key, torch.nn.Buffer(tensor))
                else: setattr(model, key, torch.nn.Parameter(tensor, requires_grad=False))
            model.N_sources = len(state['stable_ids'])
            eig.reset()
            kinfo = model.prepare_kernels(mesh, eig, save_barycentric=False)
            src_embeddings = interp(kinfo['vert_idx'], kinfo['barycentric_coords'], eig._iso_embeddings)
            representatives = []
            for rid in roots:
                found = np.flatnonzero(state['root_ids'] == rid)
                representatives.append(int(found[np.argmin(state['stable_ids'][found])]) if len(found) else -1)
            valid = [i for i in representatives if i >= 0]
            valid_roots = [rid for rid, i in zip(roots, representatives) if i >= 0]
            pieces = {k: [] for k in ['predicted','raw','normalized','weighted','alpha','distance_squared','distance_weight','euclidean','candidate','selected']}
            max_mismatch = 0.
            probe_data = None
            for start in range(0, len(faces), a.chunk):
                stop = min(start+a.chunk, len(faces))
                pts = model.prepare_points(mesh, eig, faces[start:stop], barys=bary[start:stop])
                info = dict(evals=pts['albo_evals'], evecs=pts['albo_evecs'], indices=pts['indices'],
                            weights=pts['weights'], distances=pts['distances'])
                values = stage_values(model, eig, info)
                actual, _, _, _ = model.diffuse_heat_kernels(eig, pts)
                actual = model(actual)
                rebuilt, local, contributions, preclamp = reconstruct_colour(
                    values['alpha'], model.kernel_colours[info['indices']], model._mean_colour, model.cfg.knn_inner_k)
                mismatch = float((actual-rebuilt).abs().max().item())
                max_mismatch = max(max_mismatch, mismatch)
                if not torch.isfinite(actual).all() or not torch.isfinite(rebuilt).all() or mismatch > 5e-5:
                    raise RuntimeError(f'Colour decomposition differs from actual forward: {mismatch}')
                pieces['predicted'].append(cpu(actual))
                selected_global = torch.gather(info['indices'], 1, local)
                if start == 0:
                    q = min(a.probes, stop-start)
                    probe_data = {k: cpu(v[:q]) for k,v in values.items()}
                    probe_data.update(candidate_ids=state['stable_ids'][cpu(info['indices'][:q])],
                        selected_local=cpu(local[:q]), colour_contributions=cpu(contributions[:q]),
                        preclamp=cpu(preclamp[:q]), final_rgb=cpu(actual[:q]),
                        kernel_rgb=cpu(model.kernel_colours[info['indices'][:q]]),
                        mean_rgb=cpu(model._mean_colour), probe_positions=surface['positions'][:q],
                        grid_indices=cpu(eig.grid[0][info['indices'][:q]]),
                        grid_weights=cpu(eig.grid[1][info['indices'][:q]]),
                        angles=cpu(model.angles[info['indices'][:q]]),
                        anisotropies=cpu(model.anisotropies[info['indices'][:q]]),
                        thresholds=cpu(model.thresholds[info['indices'][:q]]),
                        sharpnesses=cpu(model.sharpnesses[info['indices'][:q]]),
                        grid_angle_axis=cpu(eig._smp_coords_angle),
                        grid_anisotropy_axis=cpu(eig._smp_coords_scale))
                    probe_data['spectral_terms'] = cpu((torch.exp(-info['evals'][:q]*model.diff_times)
                        *info['evecs'][:q]*eig.knn_heat._cache.evecs_src[info['indices'][:q]]))
                    probe_data['decay'] = cpu(torch.exp(-info['evals'][:q]*model.diff_times))
                    probe_data['biharmonic_terms'] = cpu((query_embeddings[:q,None,:]-src_embeddings[info['indices'][:q]])**2)
                if valid:
                    ids = torch.tensor(valid, dtype=torch.long, device='cuda')[None].expand(stop-start,-1)
                    ev, evec, _ = eig.knn_gather.gather_queries(ids, eig.grid[0], eig.grid[1],
                        vert_idx[start:stop], bary[start:stop], eig._eigen_val_knn, eig._eigen_vec_knn)
                    dsq = ((query_embeddings[start:stop,None,:]-src_embeddings[ids])**2).sum(-1)
                    weights = eig.heat_weighting.compute(dsq) if eig.use_weighting else None
                    direct = stage_values(model,eig,dict(indices=ids,evals=ev,evecs=evec,distances=dsq,weights=weights))
                    for key in ['raw','normalized','weighted','alpha','distance_squared','distance_weight']: pieces[key].append(cpu(direct[key]))
                    pos = torch.tensor(surface['positions'][start:stop],device='cuda')
                    pieces['euclidean'].append(cpu((pos[:,None,:]-model.kernel_locations[ids]).norm(dim=-1)))
                    pieces['candidate'].append(cpu((info['indices'][...,None]==ids[:,None,:]).any(dim=1)))
                    pieces['selected'].append(cpu((selected_global[...,None]==ids[:,None,:]).any(dim=1)))
            arrays = {key: np.concatenate(parts,axis=0) for key,parts in pieces.items() if parts}
            arrays.update(root_ids=np.asarray(valid_roots), representative_ids=state['stable_ids'][valid],
                          kernel_locations=state['_kernel_locations'][valid])
            err = ((arrays['predicted']-surface['gt'])**2).mean(axis=1)
            arrays['error_mse_rgb_mean'] = err
            npz_write(out/(name+'_fields.npz'), **arrays)
            npz_write(out/(name+'_probes.npz'), **probe_data)
            mse = float(err.mean())
            row = dict(**item, mse_rgb_mean=mse, loss_rgb_sum=3*mse,
                psnr_rgb= float(-10*np.log10(max(mse,1e-30))), max_forward_decomposition_error=max_mismatch,
                roots_requested=roots, active_representatives=arrays['representative_ids'].tolist())
            rows.append(row)
            json_write(out/'metrics.json',rows)
            json_write(out/'replay.json',dict(recording=str(root), roots=roots,
                note='fields are direct representative-kernel evaluations; candidate/selected show actual KNN gates; representatives follow lowest surviving descendant ID',
                points=len(faces), probes=min(a.probes,a.chunk,len(faces)), cache_sha256=manifest['cache_sha256']))
    print('[REPLAY] Numerical analysis saved:',out,flush=True)


if __name__ == '__main__':
    main()
