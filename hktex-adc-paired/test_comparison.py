import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from adc_support import POLICY,scheduled_decisions
from compare_pair import compare


def write(p,x):p.write_text(json.dumps(x))
def lines(p,x):p.write_text(''.join(json.dumps(r)+'\n' for r in x))


def fixture(root,active=True):
    (root/'initial.pt').write_bytes(b'initial');sha=hashlib.sha256(b'initial').hexdigest()
    for arm in ['linear','loggamma']:
        for stage,start,end in [('s02000',0,2000),('s05000',2000,5000)]:
            d=root/stage/arm;d.mkdir(parents=True)
            s=dict(adc_full=POLICY,paired_loss=dict(arm=arm,initial_sha256=sha,code_hash='code'),
                seed=0,steps=5000,sources=5000,batch_pixels=256,aa=4,transport=4,adc=True,
                eval_pixels=1024,manifest_sha256='m',source_fingerprints={})
            write(d/'report.json',dict(settings=s,status='complete' if end==5000 else 'stage_complete',
                start_update=start,completed_updates=end,final_eval={'mse':{'val':.001}}))
            if start:write(d/'resume_repair.json',dict(passed=True))
            metrics=[dict(update=u,kernels=5000+(10*int(u>=1000)-5*int(u>=1500) if active else 0),
                batch_signature=str(u),loss_l1=.1,loss_objective=.1,batch_mse=.01,gradients={'x':.1},
                peak_cuda_gb=2.) for u in range(start+1,end+1)]
            lines(d/'metrics.jsonl',metrics)
            actions=[dict(update=u,operation=op,origin='natural',selected=n,before=b,after=e,status='complete',state_checks_passed=True)
                for u,op,n,b,e in [(1000,'split',10,5000,5010),(1500,'prune',5,5010,5005)] if start<u<=end] if active else []
            lines(d/'adc_actions.jsonl',actions)
            decisions=[dict(update=u,kind=k,threshold=POLICY['error_threshold'],min_kernels=POLICY['min_kernels'],
                selection_threshold=POLICY['selection_threshold'],contrib_threshold=POLICY['contrib_threshold'])
                for u,k in scheduled_decisions(end) if u>start]
            lines(d/'adc_decisions.jsonl',decisions)


class Tests(unittest.TestCase):
    def test_pilot_and_full(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);fixture(p)
            for u in [2000,5000]:
                r=compare(p,u);self.assertEqual(r['status'],'passed')
                self.assertEqual(r['arms']['linear']['final_kernels'],5005)
                self.assertFalse(r['quality_improvement_established'])
    def test_inactive(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);fixture(p,False)
            self.assertEqual(compare(p,2000)['status'],'functional_passed_natural_activity_inconclusive')
    def test_missing_decisions(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);fixture(p)
            (p/'s05000/loggamma/adc_decisions.jsonl').write_text('')
            with self.assertRaisesRegex(ValueError,'decisions'):compare(p,5000)
    def test_unpaired_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);fixture(p);f=p/'s02000/loggamma/metrics.jsonl'
            f.write_text(f.read_text().replace('"batch_signature": "1"','"batch_signature": "different"'))
            with self.assertRaisesRegex(ValueError,'input stream'):compare(p,2000)
    def test_forced_probes_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);fixture(p);f=p/'s02000/linear/adc_actions.jsonl'
            f.write_text(f.read_text().replace('natural','forced-smoke-probe'))
            with self.assertRaisesRegex(ValueError,'Forced probes'):compare(p,2000)


if __name__=='__main__':unittest.main(verbosity=2)
