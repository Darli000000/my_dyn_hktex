"""Pixel-weighted ADC statistics with upstream clone/split/prune mechanics."""
import numpy as np
import torch
from hktex.density_controllers.base import BaseDensityController
from checks import surface_state
from policy import select_prune,select_grow

class WeightedADC(BaseDensityController):
    def configure(self,mesh,model,optimizers,network,geometry,policy,**kwargs):
        self.network=network;self.geometry=geometry;self.policy=policy
        super().configure(mesh,model,optimizers,**kwargs)
        self.initial_counts=self.component_counts()
        head=int(geometry['face_components'][geometry['face_regions']==1][0])
        self.floors=[min(n,policy['min_head_kernels'] if ci==head else policy['min_other_component_kernels']) for ci,n in enumerate(self.initial_counts)]
        f=geometry['faces'];v=geometry['vertices']
        edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
        self.radius=float(np.median(np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1)))*policy['split_radius_in_median_edges']
        self.accumulated_steps=0

    def _post_configure(self):
        self._state={k:torch.zeros(self._model.N_sources,device=self._model._kernel_locations.device) for k in ['error','hits','contribution']}

    def component_ids(self):
        return self.geometry['face_components'][self._model._kernel_face_ids.detach().cpu().numpy()]

    def component_counts(self):
        return np.bincount(self.component_ids(),minlength=int(self.geometry['face_components'].max())+1).tolist()

    @torch.no_grad()
    def accumulate(self,step,pred,target,indices,contributions,pixel_weights):
        a=self.policy
        if not (a['start']-a['densify_interval']+1<=step<=a['stop_inclusive']):return
        # Original controller uses per-point MAE. Scale importance weights to a
        # hypothetical uniform batch of the SAME size (1024); do not count the
        # oversampled eye pixels as if they occupied 25% of the training image.
        idx=indices.squeeze(-1);weight=contributions.squeeze(-1).clamp_min(0)
        correction=pixel_weights.detach()*len(pred)
        norm=weight.sum(0,keepdim=True).clamp_min(1e-8)
        error=(pred.detach()-target).abs().mean(1)
        for key,values in [
            ('error',(weight/norm)*error[None,:]*correction[None,:]),
            ('hits',(weight>0).to(weight.dtype)*correction[None,:]),
            ('contribution',weight*correction[None,:]),
        ]:
            self._state[key].scatter_add_(0,idx.reshape(-1).long(),values.reshape(-1))
        self.accumulated_steps+=1

    def pre_backward_step(self,*args,**kwargs):
        raise RuntimeError('Use accumulate() with explicit sampling weights')

    def post_backward_step(self,*args,**kwargs):
        raise RuntimeError('ADC mutations must happen AFTER optimizer.step()')

    def optimizer_audit(self):
        self.refresh_state();n=self._model.N_sources;seen=set()
        for opt in self._optimizers:
            for group in opt.param_groups:
                name=group['name'].split('.')[-1];p=getattr(self._model,name)
                if len(group['params'])!=1 or group['params'][0] is not p:raise ValueError('Stale optimizer parameter '+name)
                seen.add(name)
                if name!='_mean_colour' and len(p)!=n:raise ValueError('Wrong kernel parameter length '+name)
                for key,value in opt.state.get(p,{}).items():
                    if torch.is_tensor(value):
                        if not torch.isfinite(value).all():raise ValueError('Nonfinite optimizer state '+name)
                        if key!='step' and value.shape!=p.shape:raise ValueError('Wrong optimizer state shape '+name)
                if 'face_ids' in group and group['face_ids'][0] is not self._model._kernel_face_ids:raise ValueError('Stale geodesic face IDs')
            if any(p not in {q for group in opt.param_groups for q in group['params']} for p in opt.state):raise ValueError('Orphan optimizer state')
        expected={name for name,p in self._model.named_parameters() if p.requires_grad}
        if seen!=expected:raise ValueError('Optimizer coverage changed')
        if len(self._model._kernel_face_ids)!=n or any(len(x)!=n for x in self._state.values()):raise ValueError('Stale ADC buffer size')
        if any(not torch.isfinite(p).all() for p in self._model.parameters()):raise ValueError('Nonfinite parameter after ADC')
        return dict(kernels=n,parameters=len(seen),optimizer_bindings_valid=True)

    @torch.no_grad()
    def mutate(self,kind,mask):
        self.refresh_state();mask=torch.as_tensor(mask,dtype=torch.bool,device=self._model._kernel_locations.device)
        if not mask.any():return
        before=self.component_ids();selected=mask.cpu().numpy();old_n=len(before)
        if kind=='prune':
            expected=before[~selected];self._remove_kernels(mask)
        elif kind=='clone':
            expected=np.concatenate([before,before[selected]]);self.clone(mask)
        elif kind=='split':
            expected=np.concatenate([before[~selected],np.tile(before[selected],2)])
            starts=self._model._kernel_locations[mask].detach().clone().repeat(2,1)
            travel=((1-self._model.thresholds[mask])*self.radius).repeat(2)
            self.split(mask,self.network.eigalbo_interp,None,self.network.tracer,self.radius)
            actual=(self._model._kernel_locations[old_n-int(mask.sum()):]-starts).norm(dim=1)
            diag=float(np.linalg.norm(np.ptp(self.geometry['vertices'],axis=0)))
            if bool((actual>travel+diag*2e-5).any()):raise ValueError('ADC split exceeded requested displacement')
        else:raise ValueError(kind)
        if not np.array_equal(expected,self.component_ids()):raise ValueError('ADC changed component membership')
        # Upstream mutation copies this attribute without resizing. It must be
        # recomputed, never reused with the new parameter/face ordering.
        loc=self._model._kernel_locations
        if hasattr(loc,'bary_coords'):delattr(loc,'bary_coords')
        self.network.reset();self.refresh_state();self.optimizer_audit()
        surface_state(self._mesh,self.geometry,loc,self._model._kernel_face_ids)

    def due(self,step):
        a=self.policy
        return a['start']<=step<=a['stop_inclusive'] and step%a['densify_interval']==0

    @torch.no_grad()
    def apply(self,step):
        if not self.due(step):return None
        a=self.policy;before=self._model.N_sources;counts=self.component_counts()
        if self.accumulated_steps!=a['densify_interval']:raise ValueError('Unexpected ADC accumulation window')
        denom=self.accumulated_steps
        rec=dict(step=step,kernels_before=before,components_before=counts,window_steps=denom,
                 split_radius=self.radius,pruned=0,cloned=0,split_parents=0,prune_candidates=0,
                 min_component_kernels=self.floors)
        if step%a['prune_interval']==0:
            mask,nc=select_prune(self._state['hits'].cpu().numpy()/denom,self._state['contribution'].cpu().numpy()/denom,self.component_ids(),self.floors,a)
            rec.update(pruned=int(mask.sum()),prune_candidates=nc);self.mutate('prune',mask)
        errors=self._state['error'].cpu().numpy()/denom
        clone,split,nc=select_grow(errors,self._model.thresholds.detach().cpu().numpy(),a)
        rec.update(densify_candidates=nc,cloned=int(clone.sum()),split_parents=int(split.sum()),
                   error_quantiles=np.quantile(errors,[0,.5,.9,.99,1]).tolist())
        if clone.any():
            self.mutate('clone',clone);split=np.r_[split,np.zeros(int(clone.sum()),bool)]
        self.mutate('split',split)
        for value in self._state.values():value.zero_()
        self.accumulated_steps=0
        rec.update(kernels_after=self._model.N_sources,components_after=self.component_counts(),optimizer_audit=self.optimizer_audit())
        if self._model.N_sources!=before-rec['pruned']+rec['cloned']+rec['split_parents']:raise ValueError('ADC count accounting failed')
        if any(n<f for n,f in zip(rec['components_after'],self.floors)):raise ValueError('Component kernel floor violated')
        return rec
