#!/usr/bin/env python3
"""Controlled colour-only fine-tuning, with an optional pre-clamp range penalty.

Original rendering/forward clamp is unchanged. Frozen kernel geometry makes the
auxiliary surface weights constant. This is a new optimization stage, not resume.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

PARAMS = ('_mean_colour', '_kernel_colours', '_angles', '_anisotropies',
          '_thresholds', '_sharpnesses', '_kernel_locations', '_kernel_face_ids')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def raw_colour_and_penalty(colours, mean, indices, coefficients):
    import torch
    z = mean + (coefficients[..., None]*colours[indices]).sum(1)
    loss = (torch.relu(-z).square()+torch.relu(z-1).square()).mean()
    return z, loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['control', 'range'], required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--trainer-dir', default='/root/hktex-work/hktex/hktex-image-training-observed')
    p.add_argument('--repo', default='/root/hktex-work/hktex')
    p.add_argument('--dataset', default='/root/hktex-work/data/offline_multiview/smpl-male-full-20261025-003839-35340')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--range-weight', type=float, default=1.)
    p.add_argument('--probe-points', type=int, default=8192)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu', default='0')
    a = p.parse_args()
    if a.steps < 1 or not 0 < a.lr <= .01 or a.range_weight <= 0 or a.probe_points < 512:
        p.error('Require steps>=1, 0<lr<=.01, range-weight>0, probe-points>=512')
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    os.environ.setdefault('MPLBACKEND', 'Agg')
    repo, package = Path(a.repo).resolve(), Path(a.trainer_dir).resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(package))
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Run in the original hktex CUDA environment')
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    parent, texture = state['settings'], state['texture']
    for name, digest in parent['source_fingerprints'].items():
        if not (repo/name).is_file() or sha(repo/name) != digest:
            raise RuntimeError('Core source changed: '+name)
    entry = package/'train_images.py'
    if '--observer-interval' not in entry.read_text():
        raise RuntimeError('Requires the observed image trainer')
    spec = importlib.util.spec_from_file_location('colour_repair_entry', entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')
    import image_trainer
    from observer_common import sample_surface
    cls = image_trainer.ExternalImageTrainer
    from omegaconf import OmegaConf
    weight = a.range_weight if a.mode == 'range' else 0.
    original_config = module.make_config
    original_setup, original_check, original_step = cls.setup_images, cls._check_gradients, cls.step_images

    def make_config(repo, dataset, settings, cache_root, OC, resume_cfg=None):
        if resume_cfg is not None:
            raise ValueError('Use train, not resume, for this new optimization stage')
        if settings['manifest_sha256'] != parent['manifest_sha256']:
            raise ValueError('Dataset differs from the parent run')
        if settings['source_fingerprints'] != parent['source_fingerprints']:
            raise ValueError('Core source differs from parent')
        cfg, renderer, cache = original_config(repo, dataset, settings, cache_root, OC,
                                               resume_cfg=copy.deepcopy(parent['trainer_config']))
        cfg.network.model.n_sources = len(texture['_kernel_colours'])
        mc = cfg.network.model
        if mc.range_enforcement_type != 'pgd' or mc.out_net or mc.normalize_colours:
            raise ValueError('Requires PGD, no out_net and no normalize_colours')
        cfg.density_controllers = []
        cfg.batch_size = settings['batch_pixels']
        cfg.grad_spp = settings['aa']*settings['transport']
        cfg.optimizers = [dict(name='Adam', args={},
            params={'model._kernel_colours': {'lr': a.lr}},
            scheduler=dict(name='StepLR', args=dict(step_size=a.steps+1, gamma=1.)))]
        settings.update(eval_pixels=1024, eval_every=250, save_every=250)
        settings['colour_repair'] = dict(version='1.0.0', mode=a.mode,
            parent_checkpoint=str(Path(a.checkpoint).resolve()), parent_checkpoint_sha256=sha(a.checkpoint),
            parent_update=state['update'], range_weight=weight, probe_points=a.probe_points,
            colour_lr=a.lr, launcher_sha256=sha(__file__),
            trained_parameters=['_kernel_colours'], frozen_parameters=[n for n in PARAMS if n!='_kernel_colours'],
            note='New optimizer and local step counter; image loss plus optional pre-clamp range penalty. '
                 'No UV targets. Forward/renderer unchanged. Observer IDs restart in this stage.')
        return cfg, renderer, cache

    def setup(self, dataset, out, settings):
        original_setup(self, dataset, out, settings)
        model = self.model.model
        # Validate all geometry buffers before copying learned parameters.
        for n in ('verts', 'faces', 'fnorms', 'vnorms'):
            torch.testing.assert_close(getattr(self.mesh, n).detach().cpu(),
                texture['_HeatKernelTexture__mesh.'+n], rtol=0, atol=0)
        with torch.no_grad():
            for n in PARAMS:
                getattr(model, n).copy_(texture[n].to(model.device))
        for n, v in model.named_parameters():
            v.requires_grad_(n == '_kernel_colours')
            v.grad = None
        self._repair_frozen = {n: getattr(model, n).detach().clone() for n in PARAMS if n!='_kernel_colours'}
        v = self.mesh.verts.detach().cpu().numpy()
        f = self.mesh.faces.detach().cpu().numpy().astype(np.int64)
        fid, bary, pos = sample_surface(v, f, a.probe_points)
        ids, coefficients = [], []
        with torch.no_grad():
            self.model.reset()
            self.model.prepare_kernels()
            for start in range(0, a.probe_points, 256):
                info = model.prepare_points(self.mesh, self.model.eigalbo_interp,
                    torch.as_tensor(fid[start:start+256], device=model.device, dtype=torch.int32),
                    pts=torch.as_tensor(pos[start:start+256], device=model.device, dtype=torch.float32))
                actual, _, index, alpha = model.diffuse_heat_kernels(self.model.eigalbo_interp, info)
                alpha = alpha.squeeze(-1).T
                ix = index.squeeze(-1).T.long()
                coef = alpha/alpha.sum(1, keepdim=True).clamp_min(1.)
                raw, _ = raw_colour_and_penalty(model._kernel_colours, model._mean_colour, ix, coef)
                torch.testing.assert_close(raw.clamp(0, 1), actual, atol=2e-6, rtol=1e-5)
                ids.append(ix)
                coefficients.append(coef)
            self._repair_idx = torch.cat(ids).detach()
            self._repair_coef = torch.cat(coefficients).detach()
            self.model.reset()
        np.savez_compressed(self.out/'repair_probes.npz', positions=pos, face_ids=fid, bary=bary,
                            indices=self._repair_idx.cpu().numpy(), coefficients=self._repair_coef.cpu().numpy())
        self._repair_lower = torch.as_tensor(pos[:, 1] < (v[:,1].min()+v[:,1].max())/2, device=model.device)
        log_probe(self, 'initial')

    def values(self):
        m = self.model.model
        return raw_colour_and_penalty(m._kernel_colours, m._mean_colour,
                                      self._repair_idx, self._repair_coef)

    def log_probe(self, phase):
        with torch.no_grad():
            z, loss = values(self)
            if not bool(torch.isfinite(z).all()):
                raise FloatingPointError('Nonfinite repair probe')
            bad = ((z < 0) | (z > 1)).any(-1)
            row = dict(completed_updates=self.update, phase=phase, range_penalty=float(loss),
                       clipped_point_fraction=float(bad.float().mean()),
                       lower_y_clipped_point_fraction=float(bad[self._repair_lower].float().mean()),
                       below_zero_channel_fraction=(z < 0).float().mean(0).cpu().tolist(),
                       preclamp_min=float(z.min()), preclamp_max=float(z.max()))
        with open(self.out/'repair_metrics.jsonl', 'a') as f:
            f.write(json.dumps(row)+'\n')
        print('[COLOUR REPAIR]', row, flush=True)

    def check_gradients(self):
        # Called immediately after PRB image-loss backward, before optimizer.step.
        m = self.model.model
        image_norm = float(m._kernel_colours.grad.norm()) if m._kernel_colours.grad is not None else 0.
        with torch.enable_grad():
            z, loss = values(self)
            if weight:
                (weight*loss).backward()
        result = original_check(self)
        if self.update % 250 == 0:
            print('[GRADIENTS]', dict(local_update=self.update+1,
                  image_colour_norm=image_norm, combined_colour_norm=result.get('_kernel_colours'),
                  range_weight=weight), flush=True)
        return result

    def step(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        if (self.update+1) % 250 == 0 or self.update+1 == a.steps:
            for n, before in self._repair_frozen.items():
                if not torch.equal(getattr(self.model.model, n), before):
                    raise AssertionError('Frozen parameter changed: '+n)
            # Existing step increments the counter in its caller, so label explicitly.
            previous = self.update
            self.update += 1
            try:
                log_probe(self, 'post_update')
            finally:
                self.update = previous
        return result

    module.make_config = make_config
    cls.setup_images, cls._check_gradients, cls.step_images = setup, check_gradients, step
    sys.argv = [str(entry), 'train', '--profile', 'baseline', '--repo', str(repo),
                '--dataset', a.dataset, '--output', a.output, '--steps', str(a.steps),
                '--sources', str(len(texture['_kernel_colours'])), '--batch-pixels', '256',
                '--aa', '4', '--transport', '4', '--preview-size', '0',
                '--observer-interval', '250', '--seed', str(a.seed), '--gpu', a.gpu]
    module.main()


if __name__ == '__main__':
    main()
