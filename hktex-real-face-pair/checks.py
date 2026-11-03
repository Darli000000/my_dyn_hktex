"""Preflight gates for the exact model and tracer subsequently used for training."""
from collections import defaultdict
import numpy as np
from common import REGIONS
from runtime import tensor

def surface_state(mesh,g,points,faces,original_faces=None):
    import torch
    with torch.no_grad():
        if not bool(torch.isfinite(points).all()) or bool((faces<0).any()) or bool((faces>=len(g['faces'])).any()):
            raise ValueError('Invalid kernel position/face ID')
        bc=mesh.cartesian_to_barycentric(points,mesh.get_face_vertices(faces))
        xyz=(mesh.get_vertex_locations(mesh.get_face_vertices(faces))*bc[:,:,None]).sum(1)
        residual=float((xyz-points).norm(dim=1).max())
        diag=float(np.linalg.norm(np.ptp(g['vertices'],axis=0)))
        if not bool(torch.isfinite(bc).all()) or float(bc.min())<-.001 or float(abs(bc.sum(1)-1).max())>.001 or residual>diag*2e-5:
            raise ValueError('Kernel left its assigned triangle surface')
        if original_faces is not None:
            labels=tensor(g['face_components'],torch.long)
            if not torch.equal(labels[faces.long()],labels[original_faces.long()]):raise ValueError('Kernel crossed a disconnected component')
    return dict(min_barycentric=float(bc.min()),max_surface_residual=residual)

def preflight(model,mesh,g,train,report,save):
    import torch
    v,f=g['vertices'],g['faces'];components=g['face_components'];diag=float(np.linalg.norm(np.ptp(v,axis=0)))
    report['tracer_config']=dict(eps=model.tracer.cfg.eps,avoid_holes=model.tracer.cfg.avoid_holes,max_iterations=model.tracer.cfg.max_iterations)
    if model.tracer.cfg.eps!=1e-6 or model.tracer.cfg.avoid_holes:raise ValueError('Incorrect tracer settings')
    rng=np.random.default_rng(19);chosen=[]
    for ci in np.unique(components):
        pool=np.flatnonzero(components==ci);chosen.extend(rng.choice(pool,min(16,len(pool)),replace=False))
    chosen=np.array(chosen);tri=v[f[chosen]];bc=np.full((len(chosen),3),1/3,np.float32);pos=(tri*bc[:,:,None]).sum(1)
    report['traces']=[]
    for fraction in [0.,.005,.05,.2]:
        direction=(tri[:,1]-tri[:,0])*fraction
        q,qi=model.tracer.trace(tensor(pos),tensor(chosen,torch.int32),tensor(direction),bary_coords=tensor(bc))
        end=q.detach().cpu().numpy();error=np.linalg.norm(end-(pos+direction),axis=1)
        limit=np.maximum(diag*2e-7,.02*np.linalg.norm(direction,axis=1))
        good=(error<=limit)&(qi.detach().cpu().numpy()==chosen)
        report['traces'].append(dict(case='in_face',edge_fraction=fraction,bad=int((~good).sum()),max_error=float(error.max())))
        save('in_face_'+str(fraction))
        if not good.all():raise ValueError('Patched tracer failed analytic in-face motion')
    edges=defaultdict(list)
    for i,face in enumerate(f):
        for a,b in zip(face,np.roll(face,-1)):edges[tuple(sorted((int(a),int(b))))].append(i)
    interior=np.array([(a,b,ff[0]) for (a,b),ff in edges.items() if len(ff)==2],np.int64)
    chosen_edges=[]
    for ci in np.unique(components):
        pool=np.flatnonzero(components[interior[:,2]]==ci)
        chosen_edges.extend(rng.choice(pool,min(16,len(pool)),replace=False))
    for name,ed in [('cross_face',interior[chosen_edges]),('open_boundary',g['boundary_edges'])]:
        if not len(ed):continue
        bc=np.zeros((len(ed),3),np.float32);directions=[]
        for j,(aa,bb,ff) in enumerate(ed):
            indices=[int(np.flatnonzero(f[ff]==aa)[0]),int(np.flatnonzero(f[ff]==bb)[0])];other=({0,1,2}-set(indices)).pop()
            bc[j,indices]=.49;bc[j,other]=.02
            directions.append(((v[aa]+v[bb])/2-v[f[ff,other]])*.1)
        direction=np.asarray(directions);fi=ed[:,2];pos=(v[f[fi]]*bc[:,:,None]).sum(1)
        q,qi=model.tracer.trace(tensor(pos),tensor(fi,torch.int32),tensor(direction),bary_coords=tensor(bc))
        rec=surface_state(mesh,g,q,qi,tensor(fi,torch.int32))
        dist=np.linalg.norm(q.detach().cpu().numpy()-pos,axis=1)
        overshoot=dist>np.linalg.norm(direction,axis=1)+diag*2e-5
        not_crossed=int((qi.detach().cpu().numpy()==fi).sum()) if name=='cross_face' else 0
        rec.update(case=name,count=len(ed),overshoots=int(overshoot.sum()),not_crossed=not_crossed)
        report['traces'].append(rec);save(name)
        if overshoot.any() or not_crossed:raise ValueError('Tracer failed '+name+' probe')
    eig=model.eigalbo_interp
    for name in ['_iso_eigen_val','_iso_eigen_vec','_eigen_val','_eigen_vec','_mass','_local_direction','_iso_embeddings']:
        if not torch.isfinite(getattr(eig,name)).all():raise ValueError('Nonfinite spectrum '+name)
    if model.model.N_sources!=5000:raise ValueError('Expected 5000 kernels')
    report['kernel_components']=np.bincount(components[model.model._kernel_face_ids.detach().cpu().numpy()],minlength=int(components.max())+1).tolist()
    report['regions']={}
    for k,name in REGIONS.items():
        pool=np.flatnonzero(train['regions']==k)
        if len(pool)<32:raise ValueError('Insufficient visible training pixels for '+name)
        ids=rng.choice(pool,min(64,len(pool)),replace=False);pts=tensor(train['points'][ids]);fi=tensor(train['faces'][ids],torch.int32)
        with torch.no_grad():
            model.reset();model.prepare_kernels();info=model.model.prepare_points(mesh,eig,fi,pts=pts)
            col,_,idx,weights=model.model.diffuse_heat_kernels(eig,info)
            weights=weights.squeeze(-1).T.abs();idx=idx.squeeze(-1).T.long()
            labels=tensor(components,torch.long);kc=labels[model.model._kernel_face_ids.long()];pc=labels[fi.long()]
            leakage=float((weights*(kc[idx]!=pc[:,None])).sum()/weights.sum().clamp_min(1e-30))
            unsupported=float((weights.sum(1)<1e-8).float().mean())
            if not torch.isfinite(col).all() or not torch.isfinite(weights).all():raise ValueError('Nonfinite '+name+' field')
        model.reset();model.zero_grad(set_to_none=True)
        target=torch.tensor([.2,.45,.6],device=pts.device).expand(len(ids),3)
        pred0 = model(pts, face_ids=fi)
        loss = (pred0.double() - target.double()).square().mean()
        loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in model.model.parameters() if p.grad is not None):raise ValueError('Nonfinite gradient for '+name)
        gradients={n:None if p.grad is None else float(p.grad.detach().double().norm()) for n,p in model.model.named_parameters()}
        rec=dict(training_pixels=len(pool),probe_points=len(ids),cross_component_weight_fraction=leakage,
            unsupported_fraction=unsupported,gradients=gradients)
        report['regions'][name]=rec;save('field_'+name)
        if leakage>1e-3 or unsupported>.05:raise ValueError('Insufficient/foreign kernel support for '+name)
        if not gradients.get('_kernel_colours'):raise ValueError('No kernel-colour gradient for '+name)
        if any(not torch.isfinite(p.grad).all() for p in model.model.parameters() if p.grad is not None):raise ValueError('Nonfinite gradient for '+name)
        # Kernel-colour finite difference; restore exactly before training.
        p=model.model._kernel_colours;flat=int(p.grad.abs().argmax());analytic=float(p.grad.flatten()[flat]);original=p.detach().clone();eps=.001
        def evaluate():
            model.reset()
            with torch.enable_grad():
                value = (model(pts, face_ids=fi).double() - target.double()).square().mean()
            return float(value.detach())

        # A finite difference is meaningful only when unchanged forwards agree.
        model.reset()
        with torch.enable_grad():
            repeated = model(pts, face_ids=fi).detach()
        model.reset()
        with torch.no_grad():
            inferred = model(pts, face_ids=fi)
        consistency = dict(
            repeat_max_abs=float((repeated - pred0.detach()).abs().max()),
            train_eval_max_abs=float((inferred - pred0.detach()).abs().max()),
        )
        rec['forward_consistency'] = consistency
        save('forward_consistency_' + name)
        if not all(np.isfinite(x) and x <= 1e-5 for x in consistency.values()):
            raise ValueError('Forward consistency failed for ' + name)
        try:
            with torch.no_grad():p.flatten()[flat]+=eps
            plus=evaluate()
            with torch.no_grad():p.copy_(original);p.flatten()[flat]-=eps
            minus=evaluate()
        finally:
            with torch.no_grad():p.copy_(original)
        numerical=(plus-minus)/(2*eps);err=abs(numerical-analytic)
        rec['finite_difference']=dict(analytic=analytic,numerical=numerical,error=err)
        save('gradient_'+name)
        if err>max(2e-5,.03*abs(analytic)):raise ValueError('Colour finite difference mismatch for '+name)
    model.zero_grad(set_to_none=True);model.reset();report['status']='passed';save('preflight_complete')
