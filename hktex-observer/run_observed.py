"""Launch one fresh, checked SMPL texture-fitting experiment."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from omegaconf import OmegaConf
from apply_split_fix import NEW


PHASES = {
    'smoke': dict(iters=100, kernels=256, batch=512, importance=False, adc=False, resolution=128, frames=4, spp=4),
    'baseline': dict(iters=300, kernels=1000, batch=4096, importance=False, adc=False, resolution=256, frames=12, spp=8),
    'adc': dict(iters=1200, kernels=1000, batch=4096, importance=True, adc=True, resolution=256, frames=12, spp=8),
    'full': dict(iters=5000, kernels=1000, batch=4096, importance=True, adc=True, resolution=512, frames=60, spp=16),
}


def build_config(repo, phase):
    settings = PHASES[phase]
    cfg = OmegaConf.load(repo / 'configs/texture_hktex_knn.yaml')
    opts = [opt for opt in cfg.trainer.optimizers if opt.name == 'GeodesicOpt']
    if len(opts) != 1:
        raise ValueError('Expected exactly one GeodesicOpt')
    # Absolute value avoids accidentally reducing an already edited LR twice.
    opts[0].args.lr = 0.009914527998776132
    opts[0].args.momentum = 0.9
    cfg.seed = 0
    cfg.data.merge_tex = True
    cfg.data.bake_vert_colours_if_textured = False
    cfg.data.batch_size = settings['batch']
    cfg.data.num_workers = 0
    cfg.data.sampling_method = 'uniform'
    cfg.data.use_importance_sampling = settings['importance']
    cfg.data.importance_sampling_pool_size = 1000000
    cfg.data.importance_sampling_warmup_steps = 250
    cfg.trainer.model.n_sources = settings['kernels']
    cfg.trainer.data_initialisation_random_ratio = 0.5
    cfg.optim.iters = settings['iters']
    cfg.optim.save_logs = True
    cfg.optim.save_model = True
    if not settings['adc']:
        cfg.trainer.density_controllers = []
    else:
        controllers = cfg.trainer.density_controllers
        expected = ['density_controllers.importance_pruning', 'density_controllers.error_based_densification']
        if [c.density_controller_type for c in controllers] != expected:
            raise ValueError('Expected original pruning and densification controllers, in that order')
        # Fixed original 5000-step schedule also for the 1200-step component test.
        # Simply setting iters=1200 would compress interpolation-based schedules.
        for key, value in dict(start_iter=500, prune_interval=1000,
                               accumulation_interval=250, stop_iter=2500).items():
            controllers[0].args[key] = value
        for key, value in dict(start_iter=400, densify_interval=100,
                               error_accumulation_interval=75, stop_iter=3500,
                               max_kernels=5000).items():
            controllers[1].args[key] = value
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=PHASES, required=True)
    p.add_argument('--repo', default='/root/hktex-work/hktex')
    p.add_argument('--mesh', default='/root/hktex-work/data/SMPL_texture_samples/Textured_Mesh_samples/SMPL_male.obj')
    p.add_argument('--gpu', default='0')
    p.add_argument('--prepare-only', action='store_true', help='Write config without training or importing CUDA')
    p.add_argument('--record-every', type=int, default=250)
    p.add_argument('--eval-points', type=int, default=8192)
    p.add_argument('--no-recording', action='store_true', help='Rollback: use the existing V4 runner without observer hooks')
    a = p.parse_args()
    repo, mesh = Path(a.repo).resolve(), Path(a.mesh).resolve()
    if not (repo / 'optimisation.py').is_file() or not mesh.is_file():
        p.error('Missing optimisation.py or mesh file')
    here = Path(__file__).resolve().parent
    if PHASES[a.phase]['adc']:
        source = (repo / 'hktex/density_controllers/base.py').read_text()
        if NEW not in source:
            p.error('Required pruning/split fix not detected. Run apply_split_fix.py --apply and test_split_regression.py first; modified alternative fixes need review.')
    if a.record_every < 1 or a.eval_points < 64:
        p.error('record-every must be positive; eval-points must be at least 64')
    runner = here / ('checked_runner_v4.py' if a.no_recording else 'observed_runner.py')
    version = subprocess.check_output([sys.executable, str(runner), '--version'], text=True).strip()
    expected_version = 'HKTex checked runner 4.0.0' if a.no_recording else 'HKTex observed runner 1.0.0'
    if version != expected_version:
        p.error(f'Unexpected runner: {version}')
    cfg = build_config(repo, a.phase)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    run_name = f'smpl-{"control" if a.no_recording else "observed"}-{a.phase}'
    config = repo / 'configs' / f'{run_name}-{stamp}.yaml'
    OmegaConf.save(cfg, config)
    print('CONFIG:', config, flush=True)
    print('PHASE:', json.dumps(PHASES[a.phase]), flush=True)
    print('Position LR:', next(x.args.lr for x in cfg.trainer.optimizers if x.name == 'GeodesicOpt'), flush=True)
    if a.prepare_only:
        return
    conda = os.environ.get('CONDA_PREFIX')
    lib = Path(conda or '/missing-conda-env') / 'lib/libstdc++.so.6'
    if not lib.is_file():
        p.error('Activate hktex Conda environment; lib/libstdc++.so.6 must exist')
    env = dict(os.environ)
    env.update(MPLBACKEND='Agg', CUDA_LAUNCH_BLOCKING='1', PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES=a.gpu)
    env['LD_PRELOAD'] = str(lib) + (':' + env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')
    settings = PHASES[a.phase]
    command = [sys.executable, '-u', str(runner), '--repo', str(repo),
               '--config', str(config), '--rendering_config', str(repo / 'configs/render.yaml'),
               '--gpu', a.gpu, '--check-tracer', '--forward-inner-texture', '--save-previews',
               f'name={run_name}', f'tag=male_init05_lr10_{stamp}',
               f'exp_root_dir={repo / "outputs"}', f'data.mesh_path={mesh}',
               f'renderer.n_rotating_frames={settings["frames"]}',
               f'renderer.camera_config.img_width={settings["resolution"]}',
               f'renderer.camera_config.img_height={settings["resolution"]}',
               f'renderer.camera_config.sample_count={settings["spp"]}',
               'renderer.ground_plane_config.activated=False']
    if not a.no_recording:
        command[3:3] = ['--record-every', str(a.record_every), '--eval-points', str(a.eval_points)]
    log = repo.parent / 'test-logs' / f'{run_name}-{stamp}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    print('LOG:', log, flush=True)
    # Preserve the child exit status, unlike an unguarded shell pipe to tee.
    with log.open('w') as stream:
        stream.write('COMMAND: ' + json.dumps(command) + '\n')
        stream.flush()
        with subprocess.Popen(command, cwd=repo, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
            for line in child.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                stream.write(line)
                stream.flush()
            status = child.wait()
    print('EXIT STATUS:', status, flush=True)
    raise SystemExit(status)


if __name__ == '__main__':
    main()
