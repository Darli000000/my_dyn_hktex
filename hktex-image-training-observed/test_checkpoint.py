"""CPU serialization regression using the actual checkpoint methods.

Only CUDA RNG access is replaced with CPU RNG access; this does not test GPU rendering.
"""
import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
import torch
from image_dataset import ImageDataset
from test_images import fixture


class CheckpointTest(unittest.TestCase):
    def test_complete_state_roundtrip(self):
        source = ast.parse(Path(__file__).with_name('image_trainer.py').read_text())
        funcs = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'cpu_state']
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'ExternalImageTrainer')
        funcs += [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ['save_checkpoint', 'restore']]
        ns = dict(torch=torch)
        exec(compile(ast.Module(body=funcs, type_ignores=[]), '<actual checkpoint methods>', 'exec'), ns)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            model = torch.nn.Linear(3, 3)
            opt = torch.optim.Adam(model.parameters(), lr=.01)
            sched = torch.optim.lr_scheduler.StepLR(opt, 2, .5)
            loss = model(torch.ones(4, 3)).square().mean()
            loss.backward()
            opt.step()
            sched.step()
            obj = SimpleNamespace(out=root, settings=dict(seed=0, version='1.0.0'), update=17,
                                  model=SimpleNamespace(model=model, reset=lambda: None),
                                  optimizers=[opt], schedulers=[sched], rng=np.random.default_rng(9),
                                  dataset=ImageDataset(root), first_update_delta=.01,
                                  gradient_checks=17, events=[], observer=None,
                                  observer_resume=dict(ids=[0,3], roots=[0,1], next_id=4))
            obj.dataset.train_batch(obj.rng, 10)
            expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
            with patch('torch.cuda.get_rng_state', side_effect=torch.get_rng_state):
                ns['save_checkpoint'](obj)
            state = torch.load(root/'checkpoint.pt', map_location='cpu', weights_only=True)
            with torch.no_grad():
                for p in model.parameters():
                    p.fill_(999.)
            obj.update = 0
            with patch('torch.cuda.set_rng_state', side_effect=torch.set_rng_state):
                ns['restore'](obj, state)
            self.assertEqual(obj.update, 17)
            self.assertEqual(obj.observer_resume, dict(ids=[0,3], roots=[0,1], next_id=4))
            for k, v in expected.items():
                torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
            self.assertEqual(sched.last_epoch, 1)
            self.assertTrue(all('exp_avg' in value for value in opt.state.values()))
            self.assertEqual(obj.dataset.sampler_state(), state['sampler'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
