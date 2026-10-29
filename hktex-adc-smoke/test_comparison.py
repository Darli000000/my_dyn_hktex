import sys,json,hashlib,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from compare_smoke import compare
from adc_support import POLICY
with tempfile.TemporaryDirectory() as td:
 root=Path(td);(root/'initial.pt').write_bytes(b'initial');sha=hashlib.sha256(b'initial').hexdigest()
 for arm in ['linear','loggamma']:
  for stage,start,end in [('s00250',0,250),('s00450',250,450)]:
   d=root/stage/arm;d.mkdir(parents=True)
   s=dict(adc_smoke=POLICY,paired_loss=dict(arm=arm,initial_sha256=sha,code_hash='code'),seed=0,steps=5000,sources=5000,batch_pixels=256,aa=4,transport=4,adc=True,manifest_sha256='m',source_fingerprints={})
   (d/'report.json').write_text(json.dumps(dict(settings=s,status='complete',start_update=start,completed_updates=end,final_eval={})))
   if start:(d/'resume_repair.json').write_text(json.dumps(dict(passed=True)))
   ms=[dict(update=u,kernels=5000+int(u>=10)+int(u>=20)-int(u>=30),batch_signature=str(u),loss_l1=.1,loss_objective=.1,batch_mse=.01,gradients={'x':.1}) for u in range(start+1,end+1)]
   (d/'metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in ms))
   actions=[dict(update=u,operation=op,origin='forced-smoke-probe',selected=1,before=b,after=e,status='complete',state_checks_passed=True) for u,op,b,e in [(10,'clone',5000,5001),(20,'split',5001,5002),(30,'prune',5002,5001)] if start<u<=end]
   (d/'adc_actions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in actions))
   decisions=[dict(update=u,kind=k) for u,k in [(200,'densify'),(400,'prune'),(400,'densify')] if start<u<=end]
   (d/'adc_decisions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in decisions))
 result=compare(root)
 assert result['functional_passed'] and result['status']=='functional_passed_natural_activity_inconclusive'
 (root/'s00450/loggamma/adc_decisions.jsonl').write_text('')
 try:compare(root)
 except ValueError as e:assert 'decisions' in str(e)
 else:raise AssertionError('Missing natural decisions accepted')
print('Comparison gate: inconclusive and missing-event rejection verified')
