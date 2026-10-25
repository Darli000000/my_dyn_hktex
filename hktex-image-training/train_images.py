#!/usr/bin/env python3
"""Read calibrated images -> sample pixels/rays -> optimize HKTex with PRB."""
import argparse
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
import numpy as np
from image_dataset import VERSION, ImageDataset, check_contract, install_no_texture_guard
from offline_common import read_json, write_json, sha256

DEFAULT_DATASET = '/root/hktex-work/data/offline_multiview/smpl-male-full-20261025-003839-35340'
PROFILES = {
    'smoke': dict(steps=50, batch_pixels=64, aa=2, transport=2, sources=256,
                  eval_pixels=64, eval_every=50, save_every=25, preview_size=64, adc=False),
    'baseline': dict(steps=1000, batch_pixels=128, aa=4, transport=2, sources=1000,
                     eval_pixels=256, eval_every=250, save_every=100, preview_size=128, adc=False),
    'adc': dict(steps=1500, batch_pixels=128, aa=4, transport=2, sources=1000,
                eval_pixels=256, eval_every=250, save_every=100, preview_size=128, adc=True),
    'full': dict(steps=5000, batch_pixels=256, aa=4, transport=4, sources=1000,
                 eval_pixels=1024, eval_every=500, save_every=250, preview_size=256, adc=True),
}


def source_fingerprints(repo):
    # Changes in these files alter the model, spectral cache or ray differentiation.
    names = ['hktex/modules/mesh.py', 'hktex/utils/lbo.py', 'hktex/modules/eigen_albo.py',
             'hktex/modules/eigen_albo_knn.py', 'hktex/modules/heat_kernel_model.py',
             'hktex/modules/heat_kernel_model_knn.py', 'hktex/modules/heat_kernel_texture.py',
             'hktex/modules/heat_kernel_texture_knn.py', 'hktex/modules/geodesic_opt.py',
             'hktex/modules/tracer.py', 'hktex/rendering/torch_texture_renderer.py',
             'hktex/rendering/util.py', 'hktex/rendering/ray_renderer.py',
             'hktex/rendering/ray_integrator.py', 'hktex/density_controllers/base.py']
    return {name: sha256(repo/name) for name in names}


def make_config(repo, dataset, settings, cache_root, OmegaConf, resume_cfg=None):
    if resume_cfg is not None:
        cfg = OmegaConf.create(resume_cfg)
        # Cache anchor may live on another server; regenerate its local location.
    else:
        src = OmegaConf.load(repo/'configs/multiview_hktex_knn_ray_small.yaml')
        src.data.mesh_path = 'unused-image-training'
        cfg = OmegaConf.create(OmegaConf.to_container(src.trainer, resolve=True))
        # UvTextureMitsubaRayTrainer's extra field is not part of our base Config.
        cfg.pop('gt_spp', None)
        cfg.data_initialisation_random_ratio = 1.0
        cfg.use_linear_gamma_space = False
        cfg.use_hdr = True
        cfg.integrator_max_depth = 3
        cfg.integrator_hide_emitters = True
        cfg.network.model.n_sources = settings['sources']
        cfg.network.point_batching = 512
        cfg.network.eigen_albo.error_if_not_precomputed = False
        cfg.network.eigen_albo.use_precomputed = True
        cfg.batch_size = settings['batch_pixels']
        cfg.grad_spp = settings['aa']*settings['transport']
        # All schedulers are now stepped once per optimizer update, not per epoch.
        for o in cfg.optimizers:
            if o.name == 'GeodesicOpt':
                o.args.lr = .001
                o.scheduler.args.T_max = settings['steps']
            else:
                for p in o.params.values():
                    p.lr *= .3
                o.scheduler.args.step_size = max(1, settings['steps']//4)
        if not settings['adc']:
            cfg.density_controllers = []
        else:
            # Conservative starter schedule. Image-space thresholds need calibration;
            # enabling the code path does not guarantee a clone/split/prune occurs.
            prune, densify = cfg.density_controllers
            prune.args.start_iter = 500
            prune.args.prune_interval = 500
            prune.args.accumulation_interval = 500
            prune.args.stop_iter = int(settings['steps']*.8)
            densify.args.start_iter = 200
            densify.args.densify_interval = 200
            densify.args.error_accumulation_interval = 200
            densify.args.stop_iter = int(settings['steps']*.8)
            densify.args.max_kernels = 5000
            densify.args.max_densify_ratio = .1
            densify.args.split_radius = .05
    m = dataset.manifest
    spectral = OmegaConf.to_container(cfg.network.eigen_albo, resolve=True)
    spectral.pop('mesh_path', None)
    key_input = dict(mesh=settings['mesh_sha256'], spectral=spectral,
                     lbo=settings['source_fingerprints']['hktex/utils/lbo.py'],
                     eigen=settings['source_fingerprints']['hktex/modules/eigen_albo.py'])
    key = hashlib.sha256(json.dumps(key_input, sort_keys=True).encode()).hexdigest()[:24]
    cache = Path(cache_root)/key
    cache.mkdir(parents=True, exist_ok=True)
    cfg.network.eigen_albo.mesh_path = str(cache/'geometry.npz')
    write_json(cache/'cache_identity.json', key_input)
    renderer = read_json(dataset.root/m['renderer_config'])
    renderer['point_batching'] = None
    return cfg, OmegaConf.create(renderer), cache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['check', 'train', 'resume'])
    p.add_argument('--repo', default='/root/hktex-work/hktex')
    p.add_argument('--dataset', default=DEFAULT_DATASET)
    p.add_argument('--output', required=True, help='New directory; never overwrite earlier experiments')
    p.add_argument('--profile', choices=list(PROFILES), default='smoke')
    p.add_argument('--gpu', default='0')
    p.add_argument('--expected-size', type=int, default=2048)
    p.add_argument('--expected-mesh-name', default='SMPL_male.obj')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--checkpoint')
    p.add_argument('--cache-root', default='/root/hktex-work/hktex/outputs/image-training-cache')
    for flag in ['steps', 'batch-pixels', 'aa', 'transport', 'sources', 'preview-size']:
        p.add_argument('--'+flag, type=int)
    a = p.parse_args()
    out = Path(a.output).resolve()
    if out.exists():
        p.error('--output must be a NEW directory')
    out.mkdir(parents=True)
    write_json(out/'startup.json', dict(status='checking', args=vars(a), version=VERSION))
    try:
        print('[DATASET]', a.dataset, flush=True)
        print('[CHECK] Verifying manifest, file hashes, cameras, geometry and image contracts', flush=True)
        checks = check_contract(a.dataset, a.expected_size)
        if a.expected_mesh_name and Path(checks['original_mesh_for_provenance_only'] or '').name != a.expected_mesh_name:
            raise ValueError('Dataset source mesh is not '+a.expected_mesh_name+'; inspect manifest.json')
        write_json(out/'dataset_check.json', checks)
        print('[DATASET CHECK] passed;', checks['image_sizes'], checks['counts'], flush=True)
        print('[SOURCE PROVENANCE ONLY]', checks['original_mesh_for_provenance_only'], flush=True)
        if a.command == 'check':
            write_json(out/'startup.json', dict(status='check_passed', args=vars(a), version=VERSION))
            return
        os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
        os.environ.setdefault('MPLBACKEND', 'Agg')
        repo = Path(a.repo).resolve()
        sys.path.insert(0, str(repo))
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        torch.cuda.set_device(0)
        import mitsuba as mi
        mi.set_variant('cuda_ad_rgb')
        from omegaconf import OmegaConf
        import trimesh
        import hktex
        from image_trainer import ExternalImageTrainer
        if hktex.utils.get_device().index != 0:
            raise RuntimeError('Use a single-GPU shell without distributed RANK/LOCAL_RANK overrides')
        logging.getLogger('hktex').setLevel(logging.INFO)
        logging.getLogger('hktex').addHandler(logging.StreamHandler(sys.stdout))
        fingerprints = source_fingerprints(repo)
        state = None
        if a.command == 'resume':
            if not a.checkpoint:
                p.error('resume requires --checkpoint')
            state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
            settings = copy.deepcopy(state['settings'])
            if settings['manifest_sha256'] != checks['manifest_sha256'] or settings['source_fingerprints'] != fingerprints:
                raise ValueError('Resume requires the same dataset and model/rendering source files')
            if any(getattr(a, k) is not None for k in ['steps', 'batch_pixels', 'aa', 'transport', 'sources', 'preview_size']):
                raise ValueError('Resume uses saved settings; do not override optimizer schedule or sampling')
        else:
            settings = dict(PROFILES[a.profile], seed=a.seed, profile=a.profile, version=VERSION)
            for k in ['steps', 'batch_pixels', 'aa', 'transport', 'sources', 'preview_size']:
                if getattr(a, k) is not None:
                    settings[k] = getattr(a, k)
            if min(settings[k] for k in ['steps', 'batch_pixels', 'aa', 'transport', 'sources']) < 1:
                raise ValueError('Training dimensions/counts must be positive')
            if settings['sources'] < 32 or settings['preview_size'] < 0:
                raise ValueError('sources >= 32 and preview-size >= 0 required')
            settings.update(manifest_sha256=checks['manifest_sha256'], mesh_sha256=checks['mesh_sha256'],
                            source_fingerprints=fingerprints)
        if settings['adc'] and 'HKTEX_FIX_CURRENT_SPLIT_GEOMETRY_V1' not in (repo/'hktex/density_controllers/base.py').read_text():
            raise ValueError('ADC needs the previously applied current-split-geometry fix')
        blocked = install_no_texture_guard(a.dataset)
        print('[NO-TEXTURE GUARD]', blocked, flush=True)
        torch.manual_seed(settings['seed'])
        torch.cuda.manual_seed_all(settings['seed'])
        np.random.seed(settings['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        data = ImageDataset(a.dataset)
        with np.load(data.root/data.manifest['train_mesh'], allow_pickle=False) as g:
            v, f, n = g['vertices'].copy(), g['faces'].copy(), g['normals'].copy()
        tri = trimesh.Trimesh(vertices=v, faces=f, vertex_normals=n, process=False)
        if not np.array_equal(tri.faces, f) or not np.allclose(tri.vertices, v, atol=0, rtol=0):
            raise ValueError('Mesh construction reordered/modified geometry')
        module = SimpleNamespace(mesh=tri, saved_normals=n)
        cfg, renderer, cache = make_config(repo, data, settings, a.cache_root, OmegaConf,
                                          settings.get('trainer_config') if state else None)
        if state:
            cfg.network.model.n_sources = int(state['texture']['_kernel_colours'].shape[0])
        settings['trainer_config'] = OmegaConf.to_container(cfg, resolve=True)
        settings.update(dataset=str(data.root), cache=str(cache), forbidden_asset_roots=blocked,
                        torch_version=str(torch.__version__), mitsuba_version=mi.__version__,
                        device=torch.cuda.get_device_name(0), colour_space='linear RGB, L1 loss')
        write_json(out/'settings.json', settings)
        OmegaConf.save(cfg, out/'trainer.yaml')
        OmegaConf.save(renderer, out/'renderer.yaml')
        print('[MODEL] Building HKTex; new mesh-keyed spectral cache:', cache, flush=True)
        # Prevent concurrent runs from partially overwriting a shared eigen cache.
        import fcntl
        with open(cache/'build.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            trainer = ExternalImageTrainer(cfg, module, renderer_cfg=renderer)
        trainer.setup_images(data, out, settings)
        if state:
            trainer.restore(state)
        write_json(out/'startup.json', dict(status='training', version=VERSION, args=vars(a)))
        trainer.train_images()
        write_json(out/'startup.json', dict(status='complete', version=VERSION, args=vars(a)))
        print('[SUCCESS]', out, flush=True)
    except BaseException as exc:
        write_json(out/'startup.json', dict(status='failed', args=vars(a), version=VERSION,
                                           error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
