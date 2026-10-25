#!/usr/bin/env python3
"""Fresh, matched-budget image-training controls. No existing files are patched.

Uses the existing observed image trainer; only diffusion time differs between arms.
The baseline profile disables ADC in every arm. Retains both observer integrations.
"""
import argparse
import hashlib
import importlib.util
import inspect
import math
from pathlib import Path
import sys

VARIANTS = {'reference': 1.1103742416356491e-6, 't1e-3': 1e-3, 't1e-2': 1e-2}


def install_config_hook(module, variant, steps, provenance):
    """Interpose on config creation, before serialization/observer/model creation."""
    original = module.make_config
    def make_config(repo, dataset, settings, cache_root, OmegaConf, resume_cfg=None):
        if resume_cfg is not None:
            raise ValueError('This launcher creates fresh controlled runs; it does not resume')
        expected = dict(steps=steps, batch_pixels=128, aa=4, transport=2, sources=1000, adc=False)
        for k, v in expected.items():
            if settings.get(k) != v:
                raise ValueError(f'Unexpected {k}: {settings.get(k)!r}, expected {v!r}')
        cfg, renderer, cache = original(repo, dataset, settings, cache_root, OmegaConf)
        if len(cfg.density_controllers) != 0:
            raise ValueError('ADC must be disabled for these controls')
        current = float(cfg.network.model.diff_time)
        if not math.isclose(current, VARIANTS['reference'], rel_tol=1e-6):
            raise ValueError(f'Base diffusion time changed: {current}. Inspect config before comparing.')
        cfg.network.model.diff_time = VARIANTS[variant]
        settings['diagnostic_ablation'] = dict(
            version='1.0.0', variant=variant, changed_model_field='network.model.diff_time',
            original_diff_time=current, effective_diff_time=VARIANTS[variant],
            note='Fresh image-only baseline; ADC off in all arms. No UV colour initialization.',
            **provenance)
        print('[ABLATION]', settings['diagnostic_ablation'], flush=True)
        return cfg, renderer, cache
    module.make_config = make_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', required=True, choices=VARIANTS)
    p.add_argument('--output', required=True, help='New run directory')
    p.add_argument('--trainer-dir', default='/root/hktex-work/hktex/hktex-image-training-observed')
    p.add_argument('--repo', default='/root/hktex-work/hktex')
    p.add_argument('--dataset', default='/root/hktex-work/data/offline_multiview/smpl-male-full-20261025-003839-35340')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu', default='0')
    a = p.parse_args()
    if a.steps < 250:
        p.error('Use at least 250 updates; keep --steps equal across arms')
    directory = Path(a.trainer_dir).resolve()
    entry = directory/'train_images.py'
    if not entry.is_file():
        p.error(f'Missing {entry}; point --trainer-dir at the observed image-training package')
    text = entry.read_text()
    if '--observer-interval' not in text or 'def make_config(' not in text:
        p.error('This launcher requires the previously delivered observed image trainer')
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location('hktex_ablation_entry', entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if list(inspect.signature(module.make_config).parameters) != [
            'repo', 'dataset', 'settings', 'cache_root', 'OmegaConf', 'resume_cfg']:
        p.error('make_config interface changed; do not apply this launcher blindly')
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    provenance = dict(launcher_sha256=digest(__file__),
                      image_trainer_sources={f.name: digest(f) for f in sorted(directory.glob('*.py'))})
    install_config_hook(module, a.variant, a.steps, provenance)
    sys.argv = [str(entry), 'train', '--profile', 'baseline', '--steps', str(a.steps),
                '--batch-pixels', '128', '--aa', '4', '--transport', '2', '--sources', '1000',
                '--preview-size', '0', '--observer-interval', '250', '--seed', str(a.seed),
                '--gpu', a.gpu, '--repo', a.repo, '--dataset', a.dataset, '--output', a.output]
    module.main()


if __name__ == '__main__':
    main()
