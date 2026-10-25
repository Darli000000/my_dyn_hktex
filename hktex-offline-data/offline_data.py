#!/usr/bin/env python3
"""A1: export and verify a static calibrated image dataset. Does not train HKTex."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
from offline_common import (VERSION, BASE_COMMIT, write_json, read_json, sha256,
                            srgb_to_linear, cameras, project, sensor_dict, save_view,
                            contact_sheet, validate)


def boot(repo, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    repo = Path(repo).resolve()
    if not (repo/'hktex/rendering/uv_texture_renderer.py').is_file():
        raise ValueError('Not an HKTex repository: '+str(repo))
    sys.path.insert(0, str(repo))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for the original HKTex renderer')
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')
    from hktex.rendering.uv_texture_renderer import UVTextureRenderer
    from hktex.utils import load_mesh
    from omegaconf import OmegaConf
    import trimesh
    return mi, UVTextureRenderer, load_mesh, OmegaConf, trimesh


def git_info(repo):
    def run(*args):
        try:
            return subprocess.check_output(['git', '-C', str(repo), *args], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return dict(commit=run('rev-parse', 'HEAD'), working_tree_status=run('status', '--porcelain'),
                inspected_commit=BASE_COMMIT)


def prepare_mesh(load_mesh, source, trimesh):
    from scipy.spatial import cKDTree
    gt = load_mesh(str(source), merge_tex=False, bake_vert_colors=False, normalise_size=False)
    welded = load_mesh(str(source), merge_tex=True, bake_vert_colors=False, normalise_size=False)
    if not isinstance(gt, trimesh.Trimesh) or not isinstance(welded, trimesh.Trimesh):
        raise ValueError('Expected a triangle mesh')
    uv = getattr(gt.visual, 'uv', None)
    if uv is None or not np.isfinite(uv).all():
        raise ValueError('A valid UV texture is required for the exporter')
    mat = gt.visual.material
    image = getattr(mat, 'baseColorTexture', None)
    if image is None:
        image = getattr(mat, 'image', None)
    if image is None:
        raise ValueError('Texture not loaded. Check OBJ, MTL and referenced image files')
    rgba = np.asarray(image.convert('RGBA'))
    if np.any(rgba[..., 3] != 255):
        raise ValueError('This first exporter supports opaque textures only')
    rgb = rgba[..., :3].copy()
    if not np.isfinite(gt.vertices).all() or not np.isfinite(welded.vertices).all():
        raise ValueError('Nonfinite mesh coordinates')
    scale = 2./float(np.max(welded.extents))
    translation = -welded.centroid.copy()
    # Apply exactly the same affine transform to both representations.
    for mesh in (gt, welded):
        mesh.apply_translation(translation)
        mesh.apply_scale(scale)
    nearest_dist, mapping = cKDTree(welded.vertices).query(gt.vertices)
    if nearest_dist.max() > 1e-6 or not np.array_equal(mapping[gt.faces], welded.faces):
        raise ValueError('UV and welded mesh face order/corners differ; cannot safely preserve correspondence')
    normals = np.asarray(welded.vertex_normals, np.float32)
    gt_normals = normals[mapping]
    transform = np.eye(4)
    transform[:3, :3] *= scale
    transform[:3, 3] = scale*translation
    tri = np.asarray(welded.vertices)[welded.faces]
    area2 = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
    if (area2 <= 1e-12).any():
        raise ValueError('Degenerate triangles found; check mesh before export')
    info = dict(vertices=len(welded.vertices), faces=len(welded.faces), uv_vertices=len(gt.vertices),
                watertight=bool(welded.is_watertight), scale=scale,
                translation_before_scale=translation.tolist(), source_to_world=transform.tolist(),
                units='normalized: longest bounding-box extent = 2',
                uv_to_welded_max_distance=float(nearest_dist.max()),
                normals='shared welded vertex normals, mapped to UV seam vertices')
    uv = np.asarray(uv, np.float32)
    gt.visual.uv = uv.copy()
    return gt, welded, rgb, uv, gt_normals, info


def make_native_mesh(mi, renderer, mesh, linear_texture, normals):
    # Explicit linear input: the repository float-bitmap path otherwise receives
    # texture/255. Never rely on implicit sRGB decoding of float arrays.
    m = renderer.mesh_to_mitsuba(mesh, tex_img=linear_texture, full_material=False,
                                 bsdf_additional_settings=dict(specular=0., metallic=0., roughness=1.))
    params = mi.traverse(m)
    params['vertex_normals'] = np.ascontiguousarray(normals, dtype=np.float32).ravel()
    params.update()
    return m


def scene_for(mi, renderer, mi_mesh, record, spp):
    spec = renderer.configure_scene()
    spec['camera'] = sensor_dict(mi, record, spp)
    spec['mesh'] = mi_mesh
    return mi.load_dict(spec)


def export(a):
    out = Path(a.output).resolve()
    if out.exists():
        raise ValueError('Output must be a NEW directory (no overwriting/resume): '+str(out))
    source = Path(a.mesh).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    mi, Renderer, load_mesh, OmegaConf, trimesh = boot(a.repo, a.gpu)
    cfg_path = Path(a.repo)/'configs/render.yaml'
    cfg = OmegaConf.load(cfg_path)['renderer']
    cfg.camera_config.img_width = cfg.camera_config.img_height = a.size
    cfg.camera_config.sample_count = a.spp
    cfg.camera_config.sampler_type = 'independent'
    cfg.camera_config.tile_size = cfg.camera_config.tile_size_heatkernels = None
    cfg.camera_config.camera_type = 'perspective'
    cfg.ground_plane_config.activated = False
    cfg.emitter_config.envmap_path = None
    cfg.emitter_config.radiance = 1.
    cfg.integrator_config.type = 'path'
    # A visible constant environment has alpha=1 even outside the object.
    # Hide it from camera rays (it still illuminates the mesh) for usable coverage alpha.
    cfg.integrator_config.hide_emitters = True
    cfg.integrator_config.meta = dict(max_depth=3)
    cfg.mitsuba_mesh_config.twosided = False
    gt, mesh, rgb, uv, gt_normals, geometry_info = prepare_mesh(load_mesh, source, trimesh)
    records = cameras(a.preset, a.size, a.distance, a.fov)
    for r in records:
        p, z = project(mesh.vertices, r)
        if z.min() <= .01 or p.min() < 2 or p.max() > a.size-2:
            raise ValueError('Mesh does not fit camera '+r['id']+'; increase --distance')
    # Only now create the output directory. A manifest marked exporting survives failures.
    out.mkdir(parents=True)
    for folder in ['rgb', 'alpha', 'masks', 'previews', 'geometry_checks', 'eval_gt']:
        (out/folder).mkdir()
    np.savez_compressed(out/'mesh.npz', vertices=np.asarray(mesh.vertices, np.float32),
                        faces=np.asarray(mesh.faces, np.int32), normals=np.asarray(mesh.vertex_normals, np.float32))
    # Geometry-only OBJ; no MTL or UV and no second normalization during future loading.
    trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False).export(out/'mesh.obj')
    linear = srgb_to_linear(rgb.astype(np.float32)/255.)
    np.savez_compressed(out/'eval_gt/reference.npz', vertices=np.asarray(gt.vertices, np.float32),
                        faces=np.asarray(gt.faces, np.int32), uv=uv, normals=gt_normals, texture_linear=linear)
    Image.fromarray(rgb).save(out/'eval_gt/source_texture_srgb.png')
    config = OmegaConf.to_container(cfg, resolve=True)
    write_json(out/'renderer_config.json', config)
    critical = ['hktex/utils/mesh.py', 'hktex/rendering/base.py', 'hktex/rendering/uv_texture_renderer.py']
    m = dict(schema='hktex-offline-multiview-v1', version=VERSION, status='exporting',
             provenance=dict(repo=git_info(a.repo), source_mesh=str(source), source_mesh_sha256=sha256(source),
                             source_files={f: sha256(Path(a.repo)/f) for f in critical},
                             mitsuba=mi.__version__, variant=mi.variant()),
             preset=a.preset, train_mesh='mesh.npz', geometry=geometry_info,
             renderer_config='renderer_config.json',
             scene_contract=dict(material='principled', specular=0., metallic=0., roughness=1.,
                                 max_depth=3, constant_environment_radiance=[1., 1., 1.], ground=False,
                                 texture_input='sRGB decoded explicitly to linear; raw float bitmap',
                                 integrator='path', hide_emitters=True, background='black',
                                 denoise=False, exposure=1., tone_mapping='none'),
             image_contract=dict(rgb='float32 HWC linear radiance, no clipping, .npy authoritative',
                                 preview='clipped sRGB uint8, visualization only',
                                 mask='conservative opaque interior, eroded 1 pixel; NOT a full segmentation',
                                 alpha='float32 fractional path-render foreground coverage',
                                 pixel_coordinates='top-left pixel centre is (0.5,0.5); top-left image edge is (0,0)',
                                 filter='box; uniform subpixel sampling',
                                 camera_cv='right/down/forward, column-vector w2c',
                                 camera_mitsuba='left/up/forward, column-vector c2w'),
             evaluation_only='eval_gt must never be used by training or initialization',
             future_training_notes=['Use mesh.npz without re-normalizing or reordering faces.',
                                    'Match linear texture, BRDF, normals, light, max_depth and pixel footprint.',
                                    'Current online UV trainer cannot read this dataset without a new adapter.',
                                    'Use same normals for both UV and HKTex renders.',
                                    'Geometry checks contain no texture colours.'],
             sha256={}, evaluation_sha256={}, views=[])
    write_json(out/'manifest.json', m)
    renderer = Renderer(cfg)
    native_mesh = make_native_mesh(mi, renderer, gt, linear, gt_normals)
    for i, r in enumerate(records):
        print(f"[EXPORT] {i+1}/{len(records)} {r['id']} {r['split']}", flush=True)
        scene = scene_for(mi, renderer, native_mesh, r, a.spp)
        m['views'].append(save_view(mi, scene, r, out, a.spp, a.seed+i))
        write_json(out/'manifest.json', m)
        del scene
    contact_sheet(out, m['views'])
    for f in sorted(out.rglob('*')):
        if f.is_file() and f.name != 'manifest.json':
            key = 'evaluation_sha256' if 'eval_gt' in f.parts else 'sha256'
            m[key][f.relative_to(out).as_posix()] = sha256(f)
    m['status'] = 'complete'
    write_json(out/'manifest.json', m)
    report = validate(out)
    print('[VALIDATION]', report['passed'], report['counts'], flush=True)
    if not report['passed']:
        raise RuntimeError(str(report['errors']))
    print('[DONE]', out, flush=True)


def verify(a):
    root = Path(a.dataset).resolve()
    report = validate(root)
    if not report['passed']:
        raise ValueError(report['errors'])
    m = read_json(root/'manifest.json')
    for f, h in m['provenance']['source_files'].items():
        if sha256(Path(a.repo)/f) != h:
            raise ValueError('Renderer/mesh loader changed since export: '+f)
    for f, h in m['evaluation_sha256'].items():
        if sha256(root/f) != h:
            raise ValueError('Changed evaluation reference: '+f)
    mi, Renderer, _, OmegaConf, trimesh = boot(a.repo, a.gpu)
    ref = np.load(root/'eval_gt/reference.npz', allow_pickle=False)
    mesh = trimesh.Trimesh(vertices=ref['vertices'], faces=ref['faces'], process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=ref['uv'])
    renderer = Renderer(OmegaConf.create(read_json(root/m['renderer_config'])))
    native = make_native_mesh(mi, renderer, mesh, ref['texture_linear'], ref['normals'])
    selected = [next(r for r in m['views'] if r['split'] == s) for s in ['train', 'val', 'test']]
    results = []
    for r in selected:
        print('[REPLAY]', r['id'], flush=True)
        scene = scene_for(mi, renderer, native, r, r['spp'])
        same = np.asarray(mi.render(scene, spp=r['spp'], seed=r['seed']), dtype=np.float32)
        other = np.asarray(mi.render(scene, spp=r['spp'], seed=r['seed']+100000), dtype=np.float32)
        stored = np.load(root/r['rgb'], allow_pickle=False)
        alpha = np.load(root/r['alpha'], allow_pickle=False)
        mask = np.asarray(Image.open(root/r['mask'])) > 0
        err = float(np.max(np.abs(same[..., :3]-stored)))
        ae = float(np.max(np.abs(same[..., 3]-alpha)))
        mse = float(np.mean((other[..., :3][mask]-stored[mask])**2))
        # half the variance of the difference estimates single-render noise variance.
        results.append(dict(id=r['id'], same_seed_rgb_max_abs=err, same_seed_alpha_max_abs=ae,
                            passed=err < a.atol and ae < a.atol,
                            independent_seed_foreground_mse=mse,
                            estimated_single_render_noise_mse=mse/2))
    result = dict(passed=all(r['passed'] for r in results), atol=a.atol, views=results,
                  note='Same-seed replay tests the saved scene; independent seeds estimate sampling noise. '
                       'This does not test HKTex fitting or prove source texture is physical albedo.',
                  mitsuba=mi.__version__, variant=mi.variant())
    write_json(root/'replay_validation.json', result)
    print('[REPLAY VALIDATION]', result['passed'], flush=True)
    if not result['passed']:
        raise RuntimeError('Replay differs: inspect replay_validation.json before training')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('export')
    e.add_argument('--repo', default='/root/hktex-work/hktex')
    e.add_argument('--mesh', required=True)
    e.add_argument('--output', required=True)
    e.add_argument('--preset', choices=['smoke', 'full'], default='smoke')
    e.add_argument('--size', type=int, default=256)
    e.add_argument('--spp', type=int, default=64)
    e.add_argument('--distance', type=float, default=3.7)
    e.add_argument('--fov', type=float, default=40.)
    e.add_argument('--seed', type=int, default=20261025)
    e.add_argument('--gpu', default='0')
    v = sub.add_parser('validate')
    v.add_argument('--dataset', required=True)
    r = sub.add_parser('verify')
    r.add_argument('--dataset', required=True)
    r.add_argument('--repo', default='/root/hktex-work/hktex')
    r.add_argument('--gpu', default='0')
    r.add_argument('--atol', type=float, default=1e-4)
    a = p.parse_args()
    if a.command == 'export':
        if a.size < 32 or a.spp < 1 or a.distance <= 0 or not 1 < a.fov < 120 or not 0 <= a.seed < 2**31:
            p.error('Invalid size/spp/distance/fov/seed')
        export(a)
    elif a.command == 'verify':
        verify(a)
    else:
        report = validate(a.dataset)
        print(report)
        if not report['passed']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
