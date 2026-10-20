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


class Recorder:
    def __init__(self, trainer, cfg, repo, interval, eval_points):
        import torch
        from omegaconf import OmegaConf
        import trimesh
        self.trainer = trainer
        self.model = trainer.model
        mc = self.model.cfg
        ec = trainer.eigalbo_interp.cfg
        if (mc.out_net or mc.normalize_colours or mc.mass_type != 'one'
            or getattr(mc, 'range_enforcement_type', 'pgd') != 'pgd'
            or getattr(ec, 'use_euclidian_distance', False)
            or getattr(ec, 'grid_scale_map', 'log1p') != 'log1p'):
            raise ValueError('Observer v1 targets the current UV KNN/PGD/biharmonic/log1p configuration')
        self.root = Path(cfg.trial_dir) / 'observer'
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root/'snapshots').mkdir()
        (self.root/'samples').mkdir()
        self.interval, self.completed, self.loss_index = interval, 0, 0
        self.serial, self.event_serial = 0, 0
        self.lineage = Lineage(self.model.N_sources)
        self.last_loss = None
        self.status = 'running'
        self.repo = Path(repo).resolve()
        self.snapshot_index = []
        self.events = []
        self.started = time.monotonic()
        def config(x):
            return OmegaConf.to_container(OmegaConf.structured(x), resolve=True)
        ecfg = config(trainer.eigalbo_interp.cfg)
        cache = Path(str(ecfg['mesh_path']).rsplit('.', 1)[0] + '_' + ecfg['precomputed_name'] + '_' + ecfg['local_frames'] + '.pt')
        if not cache.is_file():
            raise RuntimeError('Observer needs the completed eigen cache on disk for later replay')
        vertices, faces = cpu(trainer.mesh.verts), cpu(trainer.mesh.faces).astype(np.int64)
        fids, bary, pos = sample_surface(vertices, faces, eval_points)
        tm = trainer.datamodule.mesh
        uv = np.asarray(getattr(tm, 'original_uv', tm.visual.uv))
        tf = np.asarray(getattr(tm, 'original_faces', tm.faces))
        image = getattr(tm.visual.material, 'baseColorTexture', None)
        if image is None: image = tm.visual.material.image
        quv = (uv[tf[fids]] * bary[:, :, None]).sum(1)
        gt = trimesh.visual.uv_to_color(quv, image)[:, :3].astype(np.float32) / 255
        npz_write(self.root/'surface.npz', vertices=vertices, faces=faces,
                  fnorms=cpu(trainer.mesh.fnorms), vnorms=cpu(trainer.mesh.vnorms),
                  face_ids=fids, bary=bary, positions=pos, gt=gt, uv=quv)
        paths = [p for p in (self.repo/'hktex').rglob('*.py') if '__pycache__' not in p.parts]
        paths += [self.repo/'optimisation.py']
        source_hashes = {str(p.relative_to(self.repo)): sha256(p) for p in paths}
        self.manifest = dict(version='1.0.0', status='running', repo=str(self.repo),
            config=OmegaConf.to_container(cfg, resolve=True),
            model_config=config(self.model.cfg), eigen_config=ecfg,
            cache_path=str(cache.resolve()), cache_sha256=sha256(cache), cache_bytes=cache.stat().st_size,
            source_hashes=source_hashes, torch_version=torch.__version__,
            numpy_version=np.__version__, interval=interval, eval_points=eval_points,
            matmul_precision=torch.get_float32_matmul_precision(),
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            snapshot_semantics='initial before first backward; periodic after optimizer/scheduler/PGD; event snapshots before/after that mutation, before optimizer',
            checkpoint_limit='parameters only; not an optimizer/sampler resume checkpoint')
        json_write(self.root/'manifest.json', self.manifest)
        print('[OBSERVER] Recording directory:', self.root, flush=True)

    def snapshot(self, phase, reason):
        if len(self.lineage.ids) != self.model.N_sources:
            raise RuntimeError('Lineage/model size mismatch; refusing misleading snapshot')
        name = f'{self.serial:05d}_u{self.completed:05d}_{phase}.npz'
        arrays = {k: cpu(getattr(self.model, k)) for k in PARAMS}
        for k in ('_error_accumulator', '_hit_accumulator', '_contrib_accumulator'):
            if hasattr(self.model, k): arrays[k] = cpu(getattr(self.model, k))
        arrays['stable_ids'], arrays['root_ids'] = self.lineage.ids.copy(), self.lineage.roots.copy()
        npz_write(self.root/'snapshots'/name, **arrays)
        row = dict(file=name, completed_updates=self.completed, loss_evaluation=self.loss_index,
                   phase=phase, reason=reason, kernels=len(self.lineage.ids))
        self.snapshot_index.append(row)
        json_write(self.root/'snapshots.json', self.snapshot_index)
        self.serial += 1
        return name

    def on_loss(self, loss, colours, gt, data):
        self.loss_index += 1
        self.last_loss = float(loss.detach().item())
        if self.loss_index == 1:
            self.snapshot('initial', 'after data initialization, before first backward')
        if self.loss_index == 1 or self.loss_index % self.interval == 0:
            arrays = {'predicted': cpu(colours), 'gt': cpu(gt)}
            for key in ('pos', 'face_id', 'bary', 'pool_indices'):
                if data is not None and key in data: arrays[key] = cpu(data[key])
            npz_write(self.root/'samples'/f'loss_{self.loss_index:05d}.npz', **arrays)

    def after_update(self):
        self.completed += 1
        row = dict(completed_updates=self.completed, loss_evaluation=self.loss_index,
                   training_loss_rgb_sum=self.last_loss, kernels=self.model.N_sources,
                   lr=[[float(g['lr']) for g in opt.param_groups] for opt in self.trainer.optimizers],
                   elapsed_seconds=time.monotonic()-self.started)
        with (self.root/'steps.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        if self.completed % self.interval == 0: self.snapshot('post_update', 'periodic')

    def mutate(self, kind, controller, mask, call):
        mask = cpu(mask).astype(bool)
        before = self.snapshot('before_'+kind, 'ADC event')
        selected = self.lineage.ids[mask].tolist()
        # Durable intent survives a failing tracer; lineage advances only on success.
        event = dict(event=self.event_serial, kind=kind, status='started',
                     loss_evaluation=self.loss_index, completed_updates=self.completed,
                     before=before, selected_ids=selected)
        self.events.append(event)
        json_write(self.root/'events.json', self.events)
        result = call()
        event.update(self.lineage.apply(kind, mask))
        event['after'] = self.snapshot('after_'+kind, 'ADC event')
        event['status'] = 'completed'
        self.event_serial += 1
        json_write(self.root/'events.json', self.events)
        return result

    def decision(self, controller, kind):
        from omegaconf import OmegaConf
        before = self.snapshot('decision_'+kind, 'controller scores before selection')
        row = dict(kind=kind, loss_evaluation=self.loss_index, snapshot=before,
                   controller_config=OmegaConf.to_container(OmegaConf.structured(controller.cfg), resolve=True))
        if kind == 'densify':
            err = np.abs(cpu(self.model._error_accumulator).reshape(-1))
            n = int((err > float(controller.cfg.error_threshold)).sum())
            room = max(0, int(controller.cfg.max_kernels)-self.model.N_sources)
            row.update(candidates=n, capacity_remaining=room,
                       maximum_to_add=min(n, int(self.model.N_sources*controller.cfg.max_densify_ratio), room))
        with (self.root/'decisions.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')

    def finish(self, status, error=None):
        self.manifest.update(status=status, completed_updates=self.completed, error=error)
        json_write(self.root/'manifest.json', self.manifest)


def install(repo, interval=250, eval_points=8192):
    """Process-local hooks; original methods are restored on context exit."""
    import hktex.utils as utils
    from hktex.trainers.base import BaseTrainer
    from hktex.modules.heat_kernel_texture import HeatKernelTexture
    from hktex.density_controllers.base import BaseDensityController
    from hktex.density_controllers.error_densification import ErrorDensificationController
    from hktex.density_controllers.importance_pruning import ImportancePruningController
    from contextlib import contextmanager
    state = {'cfg': None, 'rec': None}
    originals = []
    def replace(obj, name, factory):
        old = getattr(obj, name)
        originals.append((obj, name, old))
        setattr(obj, name, factory(old))
    def wrap_config(old):
        @wraps(old)
        def fn(*a, **kw):
            cfg = old(*a, **kw); state['cfg'] = cfg; return cfg
        return fn
    def wrap_optimise(old):
        @wraps(old)
        def fn(self, *a, **kw):
            rec = Recorder(self, state['cfg'], repo, interval, eval_points)
            state['rec'] = rec
            try:
                result = old(self, *a, **kw)
                rec.snapshot('final', 'optimise returned successfully')
                rec.finish('optimisation_completed')
                return result
            except BaseException as exc:
                rec.finish('failed', f'{type(exc).__name__}: {exc}')
                raise
            finally: state['rec'] = None
        return fn
    def wrap_loss(old):
        @wraps(old)
        def fn(self, colours, gt_colours, data=None):
            result = old(self, colours, gt_colours, data=data)
            rec = state['rec']
            if rec is not None and self is rec.trainer: rec.on_loss(result[0], colours, gt_colours, data)
            return result
        return fn
    def wrap_post(old):
        @wraps(old)
        def fn(self, *a, **kw):
            result = old(self, *a, **kw)
            rec = state['rec']
            if rec is not None and self is rec.model: rec.after_update()
            return result
        return fn
    def mutation(kind):
        def factory(old):
            @wraps(old)
            def fn(self, mask, *a, **kw):
                rec = state['rec']
                call = lambda: old(self, mask, *a, **kw)
                if rec is None or self._model is not rec.model: return call()
                return rec.mutate(kind, self, mask, call)
            return fn
        return factory
    def decision(kind):
        def factory(old):
            @wraps(old)
            def fn(self, *a, **kw):
                rec = state['rec']
                if rec is not None and self._model is rec.model: rec.decision(self, kind)
                return old(self, *a, **kw)
            return fn
        return factory
    @contextmanager
    def context():
        try:
            replace(utils, 'load_config', wrap_config)
            replace(BaseTrainer, 'optimise', wrap_optimise)
            replace(BaseTrainer, 'compute_loss', wrap_loss)
            replace(HeatKernelTexture, 'post_optimizer_step', wrap_post)
            for name, kind in [('clone','clone'), ('split','split'), ('_remove_kernels','prune')]:
                replace(BaseDensityController, name, mutation(kind))
            replace(ErrorDensificationController, '_densify', decision('densify'))
            replace(ImportancePruningController, 'prune_redundant_kernels', decision('prune'))
            yield
        finally:
            for obj, name, old in reversed(originals): setattr(obj, name, old)
    return context()
