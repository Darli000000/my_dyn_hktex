"""Pure CPU selection policy; runtime statistics use a uniform-pixel equivalent batch."""
import math
import numpy as np

def validate_config(c):
    a=c['adc'];steps=c['steps'];n=c['initial_kernels']
    if n!=5000 or steps<1:raise ValueError('This experiment starts with 5000 kernels')
    if not (0<a['densify_interval']<=a['start']<=a['stop_inclusive']<steps):raise ValueError('Invalid ADC window')
    if a['prune_interval']%a['densify_interval']:raise ValueError('Prune interval must be a multiple of densify interval')
    if a['start']%a['densify_interval'] or a['stop_inclusive']%a['densify_interval']:raise ValueError('ADC endpoints must align with intervals')
    if not 0<c['lr_min_factor']<=1:raise ValueError('Invalid minimum LR factor')
    if not 0<a['max_densify_ratio']<1 or not 0<a['max_prune_ratio']<1:raise ValueError('Invalid ADC caps')
    if not 0<a['min_kernels']<=n<=a['max_kernels']:raise ValueError('Invalid kernel limits')
    if not 0<=a['size_threshold']<=1 or a['split_radius_in_median_edges']<=0:raise ValueError('Invalid split policy')
    if any(a[k]<0 for k in ['error_threshold','selection_threshold','contribution_threshold']):raise ValueError('Negative threshold')
    if c['evaluate_every']<1 or c['log_every']<1:raise ValueError('Invalid logging interval')
    return c

def event_steps(c):
    a=c['adc'];ds=list(range(a['start'],a['stop_inclusive']+1,a['densify_interval']))
    return dict(densify=ds,prune=[s for s in ds if s%a['prune_interval']==0],
                first_accumulation_step=a['start']-a['densify_interval']+1)

def lr_factor(step,c):
    t=(step-1)/max(1,c['steps']-1)
    return c['lr_min_factor']+(1-c['lr_min_factor'])*.5*(1+math.cos(math.pi*t))

def select_prune(hits,contributions,components,floors,a):
    n=len(hits);candidate=(hits<a['selection_threshold']*max(float(np.max(hits)),1e-30))|(contributions<a['contribution_threshold'])
    order=np.flatnonzero(candidate);order=order[np.argsort(contributions[order],kind='stable')]
    budget=min(int(n*a['max_prune_ratio']),max(0,n-a['min_kernels']))
    counts=np.bincount(components,minlength=len(floors));mask=np.zeros(n,bool)
    for i in order:
        ci=components[i]
        if budget<=0:break
        if counts[ci]>floors[ci]:mask[i]=True;counts[ci]-=1;budget-=1
    return mask,int(candidate.sum())

def select_grow(errors,thresholds,a):
    ids=np.flatnonzero(errors>a['error_threshold']);candidates=len(ids)
    budget=max(0,min(int(len(errors)*a['max_densify_ratio']),a['max_kernels']-len(errors)))
    ids=ids[np.argsort(-errors[ids],kind='stable')[:budget]]
    chosen=np.zeros(len(errors),bool);chosen[ids]=True
    split=chosen&((1-thresholds)>a['size_threshold'])
    return chosen&~split,split,candidates
