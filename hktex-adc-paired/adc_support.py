"""Full ADC policy, mutation diagnostics and optimizer-reference checks."""
import json

POLICY=dict(kind='full-adc-paired-v1',step_count='one-based update about to execute',
    error_space='linear RGB mean absolute error for BOTH arms',
    densify_start=1000,densify_interval=200,error_window=200,
    prune_start=1500,prune_interval=500,prune_window=500,stop_exclusive=4000,
    initial_kernels=5000,max_kernels=8000,min_kernels=4500,max_densify_ratio=.05,
    error_threshold=.008,size_threshold=.1805252636362324,
    selection_threshold=.002492630717212829,contrib_threshold=.020463972379877828,
    split_radius=.05,forced_probes={},first_window='start-window < update <= start')


def scheduled_decisions(until):
    rows=[]
    for u in range(1,until+1):
        if u>=POLICY['stop_exclusive']:continue
        for kind,start,interval in [('prune',POLICY['prune_start'],POLICY['prune_interval']),
                                    ('densify',POLICY['densify_start'],POLICY['densify_interval'])]:
            if u>=start and u%interval==0:rows.append((u,kind))
    return rows


def accumulation_active(dc,u):
    cfg=dc.cfg
    window=cfg.error_accumulation_interval if hasattr(dc,'_densify') else cfg.accumulation_interval
    return u>cfg.start_iter-window and (cfg.stop_iter is None or u<cfg.stop_iter)


def configure_adc(cfg,settings):
    if len(cfg.density_controllers)!=2:raise ValueError('Expected prune and densify controllers')
    prune,dense=cfg.density_controllers
    if not prune.density_controller_type.endswith('importance_pruning') or not dense.density_controller_type.endswith('error_based_densification'):
        raise ValueError('Unexpected controller order')
    prune.args.start_iter=POLICY['prune_start'];prune.args.prune_interval=POLICY['prune_interval']
    prune.args.accumulation_interval=POLICY['prune_window'];prune.args.stop_iter=POLICY['stop_exclusive']
    prune.args.min_kernels=POLICY['min_kernels']
    dense.args.start_iter=POLICY['densify_start'];dense.args.densify_interval=POLICY['densify_interval']
    dense.args.error_accumulation_interval=POLICY['error_window'];dense.args.stop_iter=POLICY['stop_exclusive']
    dense.args.max_kernels=POLICY['max_kernels'];dense.args.max_densify_ratio=POLICY['max_densify_ratio']
    dense.args.split_radius=POLICY['split_radius']
    dense.args.error_threshold=POLICY['error_threshold'];dense.args.size_threshold=POLICY['size_threshold']
    prune.args.selection_threshold=POLICY['selection_threshold'];prune.args.contrib_threshold=POLICY['contrib_threshold']
    settings['adc_full']=dict(POLICY)


def append(t,name,row):
    with open(t.out/name,'a') as f:f.write(json.dumps(row)+'\n')


def validate_state(t):
    import torch
    m=t.model.model;n=m.N_sources
    if not POLICY['min_kernels']<=n<=POLICY['max_kernels']:raise RuntimeError('Kernel budget violated')
    params=dict(m.named_parameters())
    for name,p in params.items():
        if not bool(torch.isfinite(p).all()):raise FloatingPointError('Nonfinite '+name)
        if name!='_mean_colour' and len(p)!=n:raise RuntimeError('Parameter count mismatch '+name)
    for name in ['_kernel_face_ids','_error_accumulator','_hit_accumulator','_contrib_accumulator']:
        b=getattr(m,name)
        if len(b)!=n or b.device!=m._kernel_locations.device or not bool(torch.isfinite(b).all()):
            raise RuntimeError('Invalid buffer '+name)
    f=m._kernel_face_ids
    if not bool(((f>=0)&(f<t.mesh.N_faces)).all()):raise RuntimeError('Invalid model face IDs')
    owned=set()
    for opt in t.optimizers:
        for g in opt.param_groups:
            for p in g['params']:
                if not any(p is v for v in params.values()):raise RuntimeError('Stale optimizer parameter reference')
                owned.add(id(p))
                for key,value in opt.state.get(p,{}).items():
                    if torch.is_tensor(value) and key!='step':
                        if value.shape!=p.shape or value.device!=p.device or not bool(torch.isfinite(value).all()):
                            raise RuntimeError('Invalid optimizer state '+key)
            if 'face_ids' in g:
                if len(g['face_ids'])!=1 or g['face_ids'][0] is not f:raise RuntimeError('Stale GeodesicOpt face_ids')
    if any(p.requires_grad and id(p) not in owned for p in params.values()):raise RuntimeError('Unowned trainable parameter')
    if t.observer is not None and len(t.observer.lineage.ids)!=n:raise RuntimeError('Observer lineage length mismatch')


def stats(v):
    import torch
    v=v.detach().flatten().float()
    return dict(min=float(v.min()),median=float(v.median()),p90=float(torch.quantile(v,.9)),
                p99=float(torch.quantile(v,.99)),max=float(v.max()))


def install_runtime(t):
    """Wrap existing observer-instrumented methods, not replace their implementations."""
    t._adc_origin='natural'
    for dc in t.density_controllers:
        for name in ['clone','split','_remove_kernels']:
            original=getattr(dc,name)
            def mutate(mask,*args,_fn=original,_name=name,**kw):
                before=t.model.model.N_sources
                row=dict(update=t.update+1,origin=t._adc_origin,
                         operation='prune' if _name=='_remove_kernels' else _name,
                         selected=int(mask.sum()),before=before,status='started')
                try:
                    value=_fn(mask,*args,**kw)
                    for c in t.density_controllers:c.refresh_state()
                    validate_state(t)
                    row.update(status='complete',after=t.model.model.N_sources,state_checks_passed=True)
                    print('[ADC ACTION]',row,flush=True)
                    return value
                except BaseException as e:
                    row.update(status='failed',error=f'{type(e).__name__}: {e}');raise
                finally:append(t,'adc_actions.jsonl',row)
            setattr(dc,name,mutate)
        if hasattr(dc,'_densify'):
            original=dc._densify
            def densify(*args,_dc=dc,_fn=original,**kw):
                e=_dc._model._error_accumulator.detach().abs().flatten()
                append(t,'adc_decisions.jsonl',dict(update=t.update+1,kind='densify',
                    candidates=int((e>_dc.cfg.error_threshold).sum()),threshold=float(_dc.cfg.error_threshold),
                    errors=stats(e),kernels=_dc._model.N_sources,room=_dc.cfg.max_kernels-_dc._model.N_sources))
                return _fn(*args,**kw)
            dc._densify=densify
        if hasattr(dc,'prune_redundant_kernels'):
            original=dc.prune_redundant_kernels
            def prune(*args,_dc=dc,_fn=original,**kw):
                m=_dc._model;hits=m._hit_accumulator.detach().flatten()
                contrib=m._contrib_accumulator.detach().flatten()/_dc.cfg.accumulation_interval
                mask=(hits<_dc.cfg.selection_threshold*hits.max())|(contrib<_dc.cfg.contrib_threshold)
                append(t,'adc_decisions.jsonl',dict(update=t.update+1,kind='prune',
                    candidates=int(mask.sum()),kernels=m.N_sources,min_kernels=_dc.cfg.min_kernels,
                    selection_threshold=float(_dc.cfg.selection_threshold),contrib_threshold=float(_dc.cfg.contrib_threshold),
                    hits=stats(hits),average_contributions=stats(contrib)))
                return _fn(*args,**kw)
            dc.prune_redundant_kernels=prune


def run_adc(t):
    import torch
    u=t.update+1
    with torch.no_grad():
        # Every controller sees current parameters after preceding mutations.
        for dc in t.density_controllers:
            dc.refresh_state();t.model.reset();info=t.model.prepare_kernels()
            dc.post_backward_step(step=u,eigalbo_interp=t.model.eigalbo_interp,kernel_info=info,tracer=t.model.tracer)
        active=u<POLICY['stop_exclusive'] and (
            (u>=POLICY['densify_start'] and u%POLICY['densify_interval']==0) or
            (u>=POLICY['prune_start'] and u%POLICY['prune_interval']==0))
        if active:validate_state(t)
