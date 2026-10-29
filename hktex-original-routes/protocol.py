"""Small, auditable instrumentation of the author's ray loop; no GPU imports."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

PARAMS = ('_kernel_locations', '_kernel_colours', '_angles', '_anisotropies',
          '_thresholds', '_sharpnesses', '_mean_colour', '_kernel_face_ids')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def decode_reflectance(rgb):
    import torch
    x = torch.clamp(rgb, 0., 1.)
    return torch.where(x <= .04045, x / 12.92, ((x + .055) / 1.055).pow(2.4))


def instrument_ray(source):
    """Retain renderer/loss/ADC/optimizers. Move scheduler clock and enforce budget.

    Source-shape assertions deliberately fail on unreviewed upstream loop changes.
    """
    tree = ast.parse(textwrap.dedent(source))
    fn = tree.body[0]
    outer = [n for n in fn.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
             and n.target.id == 'epoch']
    if len(outer) != 1:
        raise ValueError('Expected exactly one original epoch loop')
    outer = outer[0]
    inners = [n for n in outer.body if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple)
              and ast.unparse(n.target) == '(it, ray_idx)']
    schedules = [n for n in outer.body if isinstance(n, ast.For)
                 and ast.unparse(n.target) == 'scheduler']
    if len(inners) != 1 or len(schedules) != 1:
        raise ValueError('Original ray/scheduler structure changed')
    inner, schedule = inners[0], schedules[0]
    if ast.unparse(schedule.iter) != 'self.schedulers':
        raise ValueError('Unexpected scheduler iterator')
    outer.body.remove(schedule)
    increments = [n for n in inner.body if isinstance(n, ast.AugAssign)
                  and ast.unparse(n.target) == 'global_step']
    backwards = [n for n in inner.body if isinstance(n, ast.Expr)
                 and ast.unparse(n.value) == 'dr.backward(loss)']
    if len(increments) != 1 or len(backwards) != 1:
        raise ValueError('Original update/backward structure changed')
    budget = ast.parse('if global_step >= self._route_budget:\n    break').body[0]
    # New AST nodes for each loop, so no shared child nodes.
    outer.body.insert(0, budget)
    inner.body.insert(0, ast.parse('if global_step >= self._route_budget:\n    break').body[0])
    # Observe before any ADC decision, but after author loss computation.
    loss_stmt = [n for n in inner.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'loss' for t in n.targets)]
    if len(loss_stmt) != 1:
        raise ValueError('Expected original loss assignment')
    i = inner.body.index(loss_stmt[0]) + 1
    inner.body[i:i] = ast.parse('self._route_loss(loss, L_integrated, target, batch_hit_pts, batch_hit_fids, epoch)').body
    i = inner.body.index(backwards[0]) + 1
    inner.body[i:i] = ast.parse('self._route_check_grads()').body
    i = inner.body.index(increments[0]) + 1
    inner.body[i:i] = [schedule, *ast.parse('self._route_after_update()').body]
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'


def install_ray_loop(trainer, output):
    import types
    from hktex.trainers.mitsuba_ray_trainer import MitsubaRayTrainer
    old = MitsubaRayTrainer.optimise
    source = inspect.getsource(old)
    patched = instrument_ray(source)
    output = Path(output)
    (output/'ray_loop_original.py').write_text(textwrap.dedent(source))
    (output/'ray_loop_instrumented.py').write_text(patched)
    namespace = {}
    exec(compile(patched, str(output/'ray_loop_instrumented.py'), 'exec'), old.__globals__, namespace)
    trainer.optimise = types.MethodType(namespace['optimise'], trainer)


def setup_paths(repo):
    import sys
    repo = Path(repo).resolve()
    for sub in ('', 'hktex-observer', 'hktex-native-viewer', 'hktex-offline-data',
                'hktex-image-training-observed'):
        sys.path.insert(0, str(repo/sub))
    return repo


def verify_sources(repo):
    pins = json.loads((Path(__file__).parent/'expected_sources.json').read_text())
    bad = [p for p, h in pins.items() if not (repo/p).is_file() or digest(repo/p) != h]
    if bad:
        raise RuntimeError('Source mismatch; do not silently run a different baseline: '+', '.join(bad[:15]))
    return pins
