"""Image-training adapter for the original observer schema. No extra forward/RNG draws."""
from pathlib import Path
import json
import numpy as np
from observer_common import PARAMS, Lineage, cpu, sample_surface, npz_write, json_write, sha256

SCHEMA = 'hktex-image-observer-v1'


class ImageRecorder:
    def __init__(self, trainer, previous=None):
        import torch
        from omegaconf import OmegaConf
        self.trainer = trainer
        self.model = trainer.model.model
        self.root = trainer.out/'observer'
        self.root.mkdir(exist_ok=False)
        for d in ['snapshots', 'samples']:
            (self.root/d).mkdir()
        self.interval = trainer.settings.get('observer_interval', 250)
        self.index, self.events = [], []
        self.lineage = Lineage(self.model.N_sources)
        if previous:
            self.lineage.ids = np.asarray(previous['ids'], np.int64)
            self.lineage.roots = np.asarray(previous['roots'], np.int64)
            self.lineage.next_id = int(previous['next_id'])
            if len(self.lineage.ids) != self.model.N_sources:
                raise ValueError('Restored lineage length differs from kernel count')
        ec = OmegaConf.to_container(OmegaConf.structured(trainer.model.eigalbo_interp.cfg), resolve=True)
        mc = OmegaConf.to_container(OmegaConf.structured(self.model.cfg), resolve=True)
        if (mc['out_net'] or mc['normalize_colours'] or mc['mass_type'] != 'one'
                or mc['range_enforcement_type'] != 'pgd'
                or ec.get('use_euclidian_distance', False)
                or ec.get('grid_scale_map', 'log1p') != 'log1p'):
            raise ValueError('Unsupported observer mathematical configuration')
        cache = Path(str(ec['mesh_path']).rsplit('.', 1)[0]+'_'+ec['precomputed_name']+'_'+ec['local_frames']+'.pt')
        if not cache.is_file():
            raise ValueError('Spectral cache must exist for offline replay')
        vertices, faces = cpu(trainer.mesh.verts), cpu(trainer.mesh.faces).astype(np.int64)
        fid, bary, pos = sample_surface(vertices, faces, 4096)
        npz_write(self.root/'surface.npz', vertices=vertices, faces=faces,
                  fnorms=cpu(trainer.mesh.fnorms), vnorms=cpu(trainer.mesh.vnorms),
                  render_normals=np.asarray(trainer.datamodule.saved_normals, np.float32),
                  face_ids=fid, bary=bary, positions=pos)
        self.manifest = dict(schema=SCHEMA, version='1.1.0', status='running',
            supervision='saved linear RGB images; no surface GT or UV',
            settings=trainer.settings, model_config=mc, eigen_config=ec,
            config=dict(renderer=OmegaConf.to_container(OmegaConf.structured(trainer.renderer.cfg), resolve=True)),
            cache_path=str(cache.resolve()), cache_sha256=sha256(cache),
            source_hashes=trainer.settings['source_fingerprints'],
            surface_sha256=sha256(self.root/'surface.npz'),
            matmul_precision=torch.get_float32_matmul_precision(),
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            start_update=trainer.update, lineage_continued=bool(previous),
            parent_recording=(previous or {}).get('recording'), interval=self.interval,
            snapshot_semantics='initial/resumed before backward; periodic after optimizer/scheduler; ADC before/after mutation before optimizer',
            field_semantics='Offline surface RGB is albedo/base colour, not image radiance. No surface GT error is defined.')
        json_write(self.root/'manifest.json', self.manifest)
        json_write(self.root/'events.json', [])
        self.snapshot('resumed' if trainer.update else 'initial', 'before next optimizer update')
        print('[OBSERVER]', self.root, flush=True)

    def state(self):
        return dict(ids=self.lineage.ids.tolist(), roots=self.lineage.roots.tolist(),
                    next_id=self.lineage.next_id, recording=str(self.root))

    def snapshot(self, phase, reason):
        if len(self.lineage.ids) != self.model.N_sources:
            raise ValueError('Lineage/model size mismatch')
        update = self.trainer.update
        name = f'{len(self.index):05d}_u{update:05d}_{phase}.npz'
        values = {k: cpu(getattr(self.model, k)) for k in PARAMS}
        for k in ['_error_accumulator', '_hit_accumulator', '_contrib_accumulator']:
            if hasattr(self.model, k): values[k] = cpu(getattr(self.model, k))
        values.update(stable_ids=self.lineage.ids.copy(), root_ids=self.lineage.roots.copy())
        for k, v in values.items():
            if not np.isfinite(v).all(): raise ValueError('Nonfinite observer array: '+k)
        npz_write(self.root/'snapshots'/name, **values)
        self.index.append(dict(file=name, completed_updates=update,
            loss_evaluation=update + int(phase.startswith(('before_', 'after_', 'decision_'))),
            phase=phase, reason=reason, kernels=len(self.lineage.ids),
            sha256=sha256(self.root/'snapshots'/name)))
        json_write(self.root/'snapshots.json', self.index)
        return name

    def mutate(self, kind, mask, call):
        mask = cpu(mask).astype(bool)
        if mask.shape != self.lineage.ids.shape:
            raise ValueError('ADC mask/lineage mismatch')
        before = self.snapshot('before_'+kind, 'ADC event')
        e = dict(event=len(self.events), kind=kind, status='started',
                 completed_updates=self.trainer.update, loss_evaluation=self.trainer.update+1,
                 selected_ids=self.lineage.ids[mask].tolist(), before=before)
        self.events.append(e)
        json_write(self.root/'events.json', self.events)
        result = call()
        e.update(self.lineage.apply(kind, mask))
        e.update(after=self.snapshot('after_'+kind, 'ADC event'), status='completed')
        json_write(self.root/'events.json', self.events)
        return result

    def decision(self, dc, kind):
        from omegaconf import OmegaConf
        name = self.snapshot('decision_'+kind, 'controller buffers before selection')
        row = dict(kind=kind, loss_evaluation=self.trainer.update+1, snapshot=name,
                   controller_config=OmegaConf.to_container(OmegaConf.structured(dc.cfg), resolve=True))
        with open(self.root/'decisions.jsonl', 'a') as f:
            f.write(json.dumps(row)+'\n')

    def sample(self, record, ids, target, predicted, seed):
        u = self.trainer.update+1
        if u == 1 or u % self.interval == 0:
            npz_write(self.root/'samples'/f'loss_{u:05d}.npz', pixel_ids=ids,
                      target_linear_rgb=target, predicted_linear_rgb=predicted,
                      view_id=np.asarray(record['id']), render_seed=np.asarray(seed))

    def after_update(self, row):
        data = dict(completed_updates=self.trainer.update, loss_evaluation=self.trainer.update,
                    training_loss_l1_rgb_mean=row['loss_l1'], kernels=row['kernels'],
                    lr=[[float(g['lr']) for g in opt.param_groups] for opt in self.trainer.optimizers])
        with open(self.root/'steps.jsonl','a') as f: f.write(json.dumps(data)+'\n')
        if self.trainer.update % self.interval == 0:
            self.snapshot('post_update','periodic')

    def finish(self, status, error=None):
        self.manifest.update(status=status, completed_updates=self.trainer.update, error=error)
        json_write(self.root/'manifest.json', self.manifest)


def verify_recording(root):
    root = Path(root)
    m = json.loads((root/'manifest.json').read_text())
    if m.get('schema') != SCHEMA: raise ValueError('Not an image observer recording')
    if sha256(root/'surface.npz') != m['surface_sha256']: raise ValueError('Surface hash mismatch')
    index = json.loads((root/'snapshots.json').read_text())
    if not index: raise ValueError('Empty recording')
    by_file = {}
    for row in index:
        p = root/'snapshots'/row['file']
        if p.resolve().parent != (root/'snapshots').resolve(): raise ValueError('Unsafe snapshot path')
        if sha256(p) != row['sha256']: raise ValueError('Snapshot hash mismatch: '+str(p))
        with np.load(p, allow_pickle=False) as d:
            n = row['kernels']; ids = d['stable_ids']
            if len(ids) != n or len(set(ids.tolist())) != n: raise ValueError('Invalid stable IDs')
            if len(d['root_ids']) != n: raise ValueError('Invalid root IDs')
            for k in PARAMS:
                if not np.isfinite(d[k]).all(): raise ValueError('Invalid parameter: '+k)
                if k != '_mean_colour' and len(d[k]) != n: raise ValueError('Invalid kernel count')
            by_file[row['file']] = (ids.copy(), d['root_ids'].copy())
    events = json.loads((root/'events.json').read_text())
    for e in events:
        if e['status'] != 'completed': continue
        ids, roots = by_file[e['before']]; after, after_roots = by_file[e['after']]
        chosen = np.isin(ids, e['selected_ids']); selected = ids[chosen]
        children = np.asarray(e['child_ids'],np.int64)
        if e['kind']=='clone':
            expected = np.r_[ids,children]; eroots=np.r_[roots,roots[chosen]]; parents=selected
        elif e['kind']=='split':
            expected=np.r_[ids[~chosen],children]; eroots=np.r_[roots[~chosen],np.tile(roots[chosen],2)]; parents=np.tile(selected,2)
        elif e['kind']=='prune':
            expected=ids[~chosen];eroots=roots[~chosen];parents=np.array([],np.int64)
        else: raise ValueError('Unknown event')
        if not (np.array_equal(expected,after) and np.array_equal(eroots,after_roots)
                and np.array_equal(parents,e['parent_ids'])): raise ValueError('Broken lineage event')
    return dict(passed=True, snapshots=len(index), events=len(events),
                final_kernels=index[-1]['kernels'], status=m['status'])


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--recording',required=True)
    p.add_argument('--checkpoint',help='Verify final snapshot equals checkpoint parameters exactly')
    a=p.parse_args()
    result=verify_recording(a.recording)
    if a.checkpoint:
        import torch
        root=Path(a.recording)
        rows=json.loads((root/'snapshots.json').read_text())
        final=[r for r in rows if r['phase']=='final'][-1]
        state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
        if state['update']!=final['completed_updates']:raise ValueError('Checkpoint update differs from final')
        with np.load(root/'snapshots'/final['file'],allow_pickle=False) as snap:
            for k in PARAMS:
                np.testing.assert_array_equal(snap[k],state['texture'][k].numpy(),err_msg='Snapshot differs: '+k)
            if state.get('observer_state'):
                np.testing.assert_array_equal(snap['stable_ids'],state['observer_state']['ids'])
                np.testing.assert_array_equal(snap['root_ids'],state['observer_state']['roots'])
        result['final_matches_checkpoint']=True
    print(json.dumps(result,indent=2))
