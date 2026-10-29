"""CPU preflight: source identities and exact update/scheduler/observation counting."""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace
from protocol import instrument_ray, verify_sources


FIXTURE = '''
def optimise(self,n_iter=100,debug_log_dir=None):
    global_step=0
    for epoch in range(n_iter):
        ray_batches=range(self.batches_per_epoch)
        for it,ray_idx in enumerate(ray_batches):
            L_integrated=target=batch_hit_pts=batch_hit_fids=None
            loss=global_step
            dr.backward(loss)
            self.optimizer_steps+=1
            global_step+=1
        for scheduler in self.schedulers:
            scheduler.step()
    return global_step
'''


def test_budget():
    compiled=instrument_ray(FIXTURE)
    ns={'dr':SimpleNamespace(backward=lambda loss:None)}
    exec(compiled,ns)
    for budget in (4,50,5000):
        for batches in (1,7,64,6001):
            s=SimpleNamespace(_route_budget=budget,batches_per_epoch=batches,optimizer_steps=0,
                schedule_steps=0,observed=0,losses=0,grad_checks=0)
            def inc(name): return lambda *args:setattr(s,name,getattr(s,name)+1)
            s.schedulers=[SimpleNamespace(step=inc('schedule_steps'))]
            s._route_after_update=inc('observed');s._route_loss=inc('losses')
            s._route_check_grads=inc('grad_checks')
            result=ns['optimise'](s,n_iter=budget)
            assert result==budget
            assert (s.optimizer_steps,s.schedule_steps,s.observed,s.losses,s.grad_checks)==(budget,)*5
    try:
        instrument_ray(FIXTURE.replace('global_step+=1','global_step=global_step+1'))
    except ValueError:
        pass
    else:
        raise AssertionError('Unreviewed loop must be rejected')


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);a=p.parse_args()
    repo=Path(a.repo).resolve()
    pins=verify_sources(repo)
    tree=ast.parse((repo/'hktex/trainers/mitsuba_ray_trainer.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MitsubaRayTrainer')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='optimise')
    patched=instrument_ray(ast.unparse(fn))
    compile(patched,'<instrumented-original-loop>','exec')
    test_budget()
    for path in Path(__file__).parent.glob('*.py'):
        compile(path.read_text(),str(path),'exec')
    print(f'PASS: {len(pins)} source hashes; real-loop transformation; 12 exact-budget cases; fail-closed source-shape guard; syntax')
    print('NOT TESTED HERE: CUDA training, DiGeo movement, Mitsuba backward, full GPU replay.')


if __name__=='__main__':main()
