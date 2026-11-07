"""Shared field forward, optimizer, exact-state and evaluation utilities."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from common import REGIONS,write_json,sha
from runtime import tensor,predict,metrics,pool_views,sample_stratified,DEFAULT_QUOTAS
from checks import surface_state
from previews import preview
from policy import lr_factor

def seed_all(seed):
    import random,torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def raw_state(model):
    return dict(format='hktex-real-face-exact-v1',parameters={n:p.detach().cpu().clone() for n,p in model.model.named_parameters()},
                face_ids=model.model._kernel_face_ids.detach().cpu().clone(),kernels=model.model.N_sources)

def digest(state):
    h=hashlib.sha256()
    for name,t in sorted({**state['parameters'],'_kernel_face_ids':state['face_ids']}.items()):
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.contiguous().numpy().tobytes())
    return h.hexdigest()

def restore(model,state):
    import torch
    if state['format']!='hktex-real-face-exact-v1' or state['kernels']!=model.model.N_sources:raise ValueError('Incorrect checkpoint shape/format')
    if set(state['parameters'])!=dict(model.model.named_parameters()).keys():raise ValueError('Checkpoint parameter names changed')
    with torch.no_grad():
        for name,p in model.model.named_parameters():p.copy_(state['parameters'][name])
        model.model._kernel_face_ids.copy_(state['face_ids'])
    if hasattr(model.model._kernel_locations,'bary_coords'):delattr(model.model._kernel_locations,'bary_coords')
    model.reset()
    if digest(raw_state(model))!=digest(state):raise ValueError('Checkpoint did not restore exactly')

def optimizers(model,source,c):
    import torch
    from hktex.modules.geodesic_opt import GeodesicOpt
    groups=[]
    for opt in source:
        if opt.name=='Adam':
            for name,param in opt.params.items():
                groups.append(dict(name=name,params=[getattr(model.model,name.split('.')[-1])],lr=float(param.lr)*c['adam_lr_scale']))
    adam=torch.optim.Adam(groups)
    geo=GeodesicOpt([dict(name='model._kernel_locations',params=[model.model._kernel_locations],
                         face_ids=[model.model._kernel_face_ids],lr=c['geo_lr'])],tracer=model.tracer,lr=c['geo_lr'])
    for opt in [adam,geo]:
        for group in opt.param_groups:group['initial_lr']=group['lr']
    return [adam,geo]

def forward(model,points,faces):
    """Same actual forward for BOTH arms, exposing contributions for ADC."""
    import torch
    model.reset();model.prepare_kernels();colors=[];indices=[];contribs=[]
    for start in range(0,len(points),512):
        info=model.model.prepare_points(model.mesh,model.eigalbo_interp,faces[start:start+512],pts=points[start:start+512])
        color,_,idx,weight=model.model.diffuse_heat_kernels(model.eigalbo_interp,info)
        colors.append(model.model(color));indices.append(idx);contribs.append(weight)
    return torch.cat(colors),torch.cat(indices,dim=1),torch.cat(contribs,dim=1)

def update(model,opts,g,batch,weights,step,c,adc=None):
    import torch
    f=lr_factor(step,c)
    for opt in opts:
        for group in opt.param_groups:group['lr']=group['initial_lr']*f
    model.zero_grad(set_to_none=True)
    pts=tensor(batch['points']);fi=tensor(batch['faces'],torch.int32);target=tensor(batch['rgb']);w=tensor(weights)
    pred,idx,contrib=forward(model,pts,fi);err=(pred-target).square().mean(1);loss=(err*w).sum()
    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
    if adc is not None:adc.accumulate(step,pred,target,idx,contrib,w)
    loss.backward()
    for name,p in model.model.named_parameters():
        if p.grad is None:raise ValueError('Missing parameter gradient '+name)
        if not torch.isfinite(p.grad).all():raise ValueError('Nonfinite gradient '+name)
    loc=model.model._kernel_locations;fid=model.model._kernel_face_ids
    old=loc.detach().clone();old_fid=fid.detach().clone();requested=loc.grad.detach().norm(dim=1)*opts[1].param_groups[0]['lr']
    for opt in opts:opt.step()
    model.post_optimizer_step();model.reset()
    with torch.no_grad():
        if not torch.isfinite(loc).all() or (fid<0).any() or (fid>=len(g['faces'])).any():raise ValueError('Invalid kernel position/face')
        comps=tensor(g['face_components'],torch.long)
        if not torch.equal(comps[fid.long()],comps[old_fid.long()]):raise ValueError('Geodesic update crossed a component')
        displacement=(loc-old).norm(dim=1);diag=float(np.linalg.norm(np.ptp(g['vertices'],axis=0)))
        if (displacement>requested+diag*2e-5).any():raise ValueError('Geodesic update exceeded travel budget')
        rec=dict(step=step,loss=float(loss.detach()),kernels=model.model.N_sources,lr_factor=f,
                 max_position_move=float(displacement.max()),face_changes=int((fid!=old_fid).sum()),
                 requested_above_eps=int((requested>1e-6).sum()),moved_above_numerical_floor=int((displacement>diag*2e-7).sum()))
        if step==1 or step%c['log_every']==0:
            rec['region_batch_mse']={name:float(err[tensor(batch['regions']==k,torch.bool)].detach().mean()) for k,name in REGIONS.items() if (batch['regions']==k).any()}
            surface_state(model.mesh,g,loc,fid,old_fid)
    return rec

def checkpoint(model,opts,out,step,phase,manifest_hash,adc=None):
    import torch
    out=Path(out);out.mkdir(parents=True,exist_ok=False);state=raw_state(model)
    torch.save(state,out/'state.pt');model.model.save_torch(out/'field.pt',compressed=False)
    if opts is not None:
        payload=dict(optimizers=[o.state_dict() for o in opts],step=step)
        if adc is not None:payload['adc']={k:v.detach().cpu() for k,v in adc._state.items()};payload['accumulated_steps']=adc.accumulated_steps
        torch.save(payload,out/'optimizers.pt')
    write_json(out/'checkpoint.json',dict(step=step,phase=phase,kernels=state['kernels'],parameter_sha256=digest(state),
               manifest_sha256=manifest_hash,state_sha256=sha(out/'state.pt'),format=state['format']))
    return out

class Evaluator:
    def __init__(self,views,seed):
        self.views=views;self.train=pool_views(views,'train')
        self.probe,self.weights=sample_stratified(self.train,np.random.default_rng(10000+seed),tuple(n*4 for n in DEFAULT_QUOTAS))
        self.val=pool_views(views,'val');self.test=pool_views(views,'test')
        self.legacy_val=pool_views(views,'val','legacy');self.legacy_test=pool_views(views,'test','legacy')
        ids=np.random.default_rng(20000+seed).choice(len(self.val['points']),min(4096,len(self.val['points'])),replace=False)
        self.event_probe={k:v[ids] for k,v in self.val.items()}

    def audit_before(self,model):
        p=self.event_probe;return predict(model,p['points'],p['faces'])

    def audit_after(self,model,before):
        p=self.event_probe;after=predict(model,p['points'],p['faces']);delta=np.abs(after-before)
        return dict(scope='fixed uniform validation ROI probe',pixels=len(after),mean_abs_rgb_change=float(delta.mean()),max_abs_rgb_change=float(delta.max()),
                    before=metrics(before,p['rgb'],p['regions']),after=metrics(after,p['rgb'],p['regions']))

    def run(self,model,folder,step,final=False):
        pred=predict(model,self.probe['points'],self.probe['faces']);rec=dict(step=step,kernels=model.model.N_sources)
        rec['train_probe']=metrics(pred,self.probe['rgb'],self.probe['regions'])
        mse=float((((pred.astype(np.float64)-self.probe['rgb'])**2).mean(1)*self.weights).sum())
        rec['train_probe']['all'].update(mse=mse,psnr=float(-10*np.log10(max(mse,1e-12))))
        for split in ['val']+(['test'] if final else []):
            pool=getattr(self,split);p=predict(model,pool['points'],pool['faces']);rec[split]=metrics(p,pool['rgb'],pool['regions'])
            old=getattr(self,'legacy_'+split);oldp=predict(model,old['points'],old['faces'])
            rec['legacy_'+split]=metrics(oldp,old['rgb'],old['regions'])
        rec['preview_metrics']={}
        for v in self.views:
            if v['row']['camera_index'] in ([0,7,15] if final else [0,7]):
                active=v['active'];v=dict(v,**{k:v[k][active] for k in ['ids','points','faces','rgb','regions']})
                p=predict(model,v['points'],v['faces']);preview(v,p,folder/'previews')
                rec['preview_metrics'][v['row']['id']]=metrics(p,v['rgb'],v['regions'])
        write_json(folder/'metrics.json',rec);return rec

def combine_previews(base,adc,output):
    rows=[]
    for camera in [0,7,15]:
        name=f'view_{camera:02d}';left=Image.open(base/'previews'/f'{name}.png').convert('RGB');right=Image.open(adc/'previews'/f'{name}.png').convert('RGB')
        w=left.width//3;h=left.height-30
        grid=Image.new('RGB',(3*w,h+30),'white');draw=ImageDraw.Draw(grid)
        for i,(im,title) in enumerate([(left.crop((0,30,w,h+30)),name+' GT'),(left.crop((w,30,2*w,h+30)),'5000 steps / ADC off'),(right.crop((w,30,2*w,h+30)),'5000 steps / ADC on')]):
            grid.paste(im,(i*w,30));draw.text((i*w+5,8),title,fill='black')
        rows.append(grid)
    sheet=Image.new('RGB',(rows[0].width,sum(r.height for r in rows)),'white');y=0
    for row in rows:sheet.paste(row,(0,y));y+=row.height
    sheet.save(output)
