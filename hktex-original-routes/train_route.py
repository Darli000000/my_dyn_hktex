"""Run the repository's original UV or online-ray trainer, with offline observation."""
import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

from protocol import PARAMS, setup_paths, verify_sources, write_json, install_ray_loop, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    p.add_argument('--mesh', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--route', choices=['uv', 'ray'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--updates', type=int, default=5000)
    p.add_argument('--record-every', type=int, default=250)
    p.add_argument('--gpu', default='0')
    p.add_argument('--uv-geo-scale', type=float, default=1., help='1 = author YAML; .1 is an explicit fallback, not default')
    a = p.parse_args()
    if a.updates < 4 or a.record_every < 1 or a.uv_geo_scale <= 0:
        p.error('Invalid budget/interval/scale')
    out = Path(a.output).resolve()
    if out.exists():
        p.error('--output must be new')
    repo = setup_paths(a.repo)
    pins = verify_sources(repo)
    if not Path(a.mesh).is_file():
        raise FileNotFoundError(a.mesh)
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.chdir(repo)
    out.mkdir(parents=True)
    rec = None
    started = time.monotonic()
    try:
        import torch
        import numpy as np
        import mitsuba as mi
        mi.set_variant('cuda_ad_rgb')
        import drjit as dr
        import hktex
        from omegaconf import OmegaConf
        from hktex.utils import load_config, seed_everything
        from hktex.trainers.base import BaseTrainer
        from hktex.data.importance_sampling_strategy import ImportanceSamplerDataset
        from hktex.density_controllers.base import BaseDensityController
        from hktex.density_controllers.error_densification import ErrorDensificationController
        from hktex.density_controllers.importance_pruning import ImportancePruningController
        from recording import Recorder
        from image_dataset import check_contract, ImageDataset
        import matplotlib.pyplot as plt

        def plain(c):
            return OmegaConf.to_container(OmegaConf.structured(c), resolve=True)

        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        torch.cuda.set_device(0)
        # Match author's optimisation.py, not the image-pair precision override.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        checks = check_contract(Path(a.dataset).resolve(), expected_size=2048)
        write_json(out/'dataset_check.json', checks)
        filename = 'texture_hktex_knn.yaml' if a.route == 'uv' else 'multiview_hktex_knn_ray_small.yaml'
        overrides = [f'data.mesh_path={Path(a.mesh).resolve()}', 'seed=0',
                     f'exp_root_dir={out.parent}', f'name={out.name}', 'tag=config', 'use_timestamp=False']
        if a.route == 'uv':
            overrides.append(f'optim.iters={a.updates}')
        cfg = load_config(str(repo/'configs'/filename), str(repo/'configs/render.yaml'), cli_args=overrides, n_gpus=1)
        cfg.trial_dir = str(out)
        write_json(out/'author_resolved_before_adaptation.json', plain(cfg))
        changes = ['5000 updates is a budgeted rerun of current repository author trainers, not a bitwise paper reproduction',
                   'observer/finite checks; separate eigen cache per route; final rendering performed offline',
                   'existing repository SVD/tracer compatibility fixes retained']
        if a.route == 'ray':
            # Preserve original optimizer rates and momentum. Normalize ONLY the scheduler clock.
            cfg.trainer.optimizers[0].scheduler.args.step_size = max(1, round(a.updates * 15 / 60))
            cfg.trainer.optimizers[1].scheduler.args.T_max = a.updates
            changes += ['ray epoch scheduler converted to updates: StepLR=updates*15/60; Cosine T_max=updates',
                        'ray ADC absolute update thresholds retained, not rescaled',
                        'log-gamma power argument floored at 1e-12 to avoid infinite zero derivative']
            eigen_cfg = cfg.trainer.network.eigen_albo
        else:
            cfg.trainer.optimizers[1].args.lr *= a.uv_geo_scale
            eigen_cfg = cfg.trainer.eigen_albo
            changes += ['importance wrapper exposes its inner tex_img, allowing intended UV-informed initialization']
            if a.uv_geo_scale != 1:
                changes.append(f'EXPLICIT NONDEFAULT UV GeodesicOpt LR multiplier: {a.uv_geo_scale}')
        # Cache filenames do not encode mass type or spectral config in the author code.
        # Never let these routes accidentally consume one another's cached decompositions.
        (out/'eigen-cache').mkdir()
        eigen_cfg.mesh_path = str(out/'eigen-cache/mesh.obj')
        OmegaConf.save(cfg, out/'parsed.yaml')
        seed_everything(cfg.seed)
        dm = hktex.find(cfg.data_type)(cfg.data)
        dm.prepare_data(); dm.setup('fit')
        data = ImageDataset(a.dataset)
        with np.load(Path(a.dataset)/data.manifest['train_mesh'], allow_pickle=False) as g:
            if not np.array_equal(dm.mesh.faces, g['faces']) or not np.allclose(dm.mesh.vertices, g['vertices'], atol=2e-6, rtol=0):
                raise ValueError('Source OBJ geometry/order/normalization differs from the offline dataset')
        # Strict finite guards do not replace NaNs or silently skip a bad update.
        with ExitStack() as stack:
            if a.route == 'uv' and not hasattr(ImportanceSamplerDataset, 'tex_img'):
                stack.enter_context(patch.object(ImportanceSamplerDataset, 'tex_img',
                                    property(lambda s: s.inner_dataset.tex_img), create=True))
            if a.route == 'ray':
                import hktex.utils as utils
                stack.enter_context(patch.object(utils, 'linear_to_gamma_dr',
                                    lambda x: 1.055 * dr.power(dr.maximum(x, 1e-12), 1/2.4) - .055))
            trainer = hktex.find(cfg.trainer_type)(cfg.trainer, dm, renderer_cfg=cfg.renderer)
            if a.route == 'uv':
                proxy = trainer
            else:
                proxy = SimpleNamespace(model=trainer.model.model, eigalbo_interp=trainer.model.eigalbo_interp,
                                        mesh=trainer.mesh, datamodule=dm, optimizers=trainer.optimizers)
            model = proxy.model
            rec = Recorder(proxy, cfg, repo, a.record_every, 8192)
            rec.manifest.update(route=a.route, supervision='original UV code values (RGB/255)' if a.route=='uv'
                                else 'original online GT ray rendering; no saved training images',
                                adaptations=changes, author_config=filename,
                                snapshot_semantics='initial after initialization; UV after optimizer/scheduler/PGD; ray after reset/update scheduler; ADC events before optimizer')
            write_json(rec.root/'manifest.json', rec.manifest)
            write_json(out/'protocol.json', dict(route=a.route, requested_updates=a.updates,
                original_initial_kernels=model.N_sources, adaptations=changes,
                initialisation_shared_with_image_training=False, source_fingerprints=pins,
                original_source_mesh_sha256=digest(a.mesh), observer=str(rec.root),
                meaning_of_splits='offline camera groups only; original UV/online routes have privileged source texture access'))

            def check_grads(*unused):
                for name, param in model.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        raise FloatingPointError(f'Nonfinite gradient before optimizer/ADC: {name}')

            for opt in trainer.optimizers:
                handle = opt.register_step_pre_hook(lambda opt, args, kwargs: check_grads())
                stack.callback(handle.remove)

            def after_update():
                for k in PARAMS:
                    if not torch.isfinite(getattr(model, k)).all():
                        raise FloatingPointError('Nonfinite model parameter: '+k)
                rec.after_update()

            def uv_loss(old):
                def call(self, colours, gt_colours, data=None):
                    result = old(self, colours, gt_colours, data=data)
                    if self is trainer:
                        if not torch.isfinite(result[0]): raise FloatingPointError('Nonfinite UV loss')
                        rec.on_loss(result[0], colours, gt_colours, data)
                    return result
                return call

            epochs = set()
            if a.route == 'uv':
                stack.enter_context(patch.object(BaseTrainer, 'compute_loss', uv_loss(BaseTrainer.compute_loss)))
                original_post = model.post_optimizer_step
                def post(*args, **kw):
                    result = original_post(*args, **kw); after_update(); return result
                stack.enter_context(patch.object(model, 'post_optimizer_step', post))
            else:
                trainer._route_budget = a.updates
                def ray_loss(loss, pred, gt, pts, fids, epoch):
                    value = float(loss.item())
                    if not math.isfinite(value): raise FloatingPointError('Nonfinite ray loss')
                    epochs.add(epoch)
                    rec.on_loss(torch.tensor(value), pred.torch().T.detach(), gt.torch().T.detach(),
                                dict(pos=pts, face_id=fids))
                trainer._route_loss = ray_loss
                trainer._route_after_update = after_update
                trainer._route_check_grads = check_grads
                install_ray_loop(trainer, out)
                old_batch = trainer.prepare_batch
                def checked_batch(*args, **kwargs):
                    batch = old_batch(*args, **kwargs)
                    if len(batch['batches']) == 0: raise RuntimeError('Original camera batch produced no ray batches')
                    return batch
                trainer.prepare_batch = checked_batch

            def mutation(old, kind):
                def call(self, mask, *args, **kw):
                    action = lambda: old(self, mask, *args, **kw)
                    return rec.mutate(kind, self, mask, action) if self._model is model else action()
                return call
            for name, kind in [('clone','clone'), ('split','split'), ('_remove_kernels','prune')]:
                stack.enter_context(patch.object(BaseDensityController, name, mutation(getattr(BaseDensityController,name), kind)))
            def decision(old, kind):
                def call(self, *args, **kw):
                    if self._model is model: rec.decision(self, kind)
                    return old(self, *args, **kw)
                return call
            for cls, name, kind in [(ErrorDensificationController,'_densify','densify'),
                                    (ImportancePruningController,'prune_redundant_kernels','prune')]:
                stack.enter_context(patch.object(cls,name,decision(getattr(cls,name),kind)))
            def save_figs(*args, **kwargs):
                for number in plt.get_fignums():
                    plt.figure(number).savefig(out/f'author-plot-{number}.png', dpi=140)
                    plt.close(number)
            stack.enter_context(patch.object(plt,'show',save_figs))
            # UV: one outer iteration = one update. Ray: instrumented inner update budget.
            trainer.optimise(n_iter=a.updates, debug_log_dir=None)
            if rec.completed != a.updates:
                raise RuntimeError(f'Incomplete update budget: {rec.completed}/{a.updates}')
            rec.snapshot('final','requested update budget completed')
            trainer.save_model(str(out/'author_model.pt'))
            # Interoperable evaluation artifact; intentionally not an optimizer resume checkpoint.
            network = plain(trainer.cfg.network) if a.route == 'ray' else dict(
                eigen_albo_type=trainer.cfg.eigen_albo_type, eigen_albo=plain(proxy.eigalbo_interp.cfg),
                model=plain(model.cfg), tracer_type=trainer.cfg.tracer_type,
                tracer=plain(trainer.cfg.tracer), point_batching=512)
            network['eigen_albo'] = plain(proxy.eigalbo_interp.cfg)
            settings = dict(dataset=str(Path(a.dataset).resolve()), manifest_sha256=checks['manifest_sha256'],
                mesh_sha256=checks['mesh_sha256'], source_fingerprints=pins,
                trainer_config=dict(network_type='modules.heat-kernel-model-knn', network=network),
                route=a.route, reflectance_encoding='srgb-code-values', checkpoint_kind='evaluation-only',
                cache_sha256=rec.manifest['cache_sha256'])
            torch.save(dict(update=rec.completed,settings=settings,
                texture={k:getattr(model,k).detach().cpu().clone() for k in PARAMS}),out/'checkpoint.pt')
            write_json(out/'settings.json',settings)
            rec.finish('optimisation_completed')
            write_json(out/'report.json',dict(status='success',updates=rec.completed,kernels=model.N_sources,
                ray_camera_epochs_seen=len(epochs),seconds=time.monotonic()-started,observer=str(rec.root),
                checkpoint=str(out/'checkpoint.pt'),adaptations=changes))
            print('[SUCCESS]',out,flush=True)
    except BaseException as exc:
        import traceback
        if rec is not None: rec.finish('failed',f'{type(exc).__name__}: {exc}')
        write_json(out/'failure.json',dict(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
