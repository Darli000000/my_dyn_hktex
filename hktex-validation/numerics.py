"""CPU-only diagnostics. Gates are scoped engineering tolerances, not proofs."""
import numpy as np


def compare_radiance(a, b, saved_spp=None, render_spp=None, atol=.003):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 3 or len(a) < 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected finite repeated RGB measurements [repeat,pixel,3]')
    if b.ndim == 2:
        if saved_spp is None or saved_spp <= 0 or render_spp is None:
            raise ValueError('Saved-image comparison needs actual exported spp')
        if b.shape != a.shape[1:]: raise ValueError('RGB shapes differ')
        # Estimate the saved image's MC variance from independent online repeats.
        vb = a.var(0, ddof=1) * render_spp / saved_spp
        mean_b = b
    elif b.shape == a.shape:
        vb, mean_b = b.var(0, ddof=1)/len(b), b.mean(0)
    else:
        raise ValueError('RGB shapes differ')
    variance = a.var(0, ddof=1)/len(a) + vb
    delta = a.mean(0)-mean_b
    rms = float(np.sqrt(np.mean(delta**2)))
    noise = float(np.sqrt(np.mean(variance)))
    excess = float(np.sqrt(max(0, np.mean(delta**2)-np.mean(variance))))
    # Refuse a pass if the noise floor is too high to resolve a useful difference.
    if rms > 3*noise + .01: status = 'failed'
    elif noise > .01: status = 'inconclusive'
    elif rms <= noise + atol: status = 'passed'
    else: status = 'inconclusive'
    return dict(status=status, rmse=rms, estimated_mc_rmse=noise,
                excess_rmse=excess, mean_bias_rgb=delta.mean(0).tolist(),
                absolute_tolerance=atol, pixels=len(delta), repeats=len(a),
                note='Finite patch test; MC variance estimate, not a formal confidence interval')


def compare_gradient(ad, fd, epsilons, expect_zero=False):
    ad, fd = np.asarray(ad, float), np.asarray(fd, float)
    if fd.shape != (len(ad), len(epsilons)) or len(ad) < 3:
        raise ValueError('Need >=3 seeds and finite differences per epsilon')
    if not np.isfinite(ad).all() or not np.isfinite(fd).all():
        raise ValueError('Nonfinite gradient')
    checks=[]
    for i, eps in enumerate(epsilons):
        d=ad-fd[:,i]; am=float(ad.mean()); fm=float(fd[:,i].mean())
        se=float(d.std(ddof=1)/np.sqrt(len(d)))
        scale=max(abs(am),abs(fm))
        limit=2e-4+.05*scale+4*se
        signal_se=max(float(ad.std(ddof=1)),float(fd[:,i].std(ddof=1)))/np.sqrt(len(ad))
        if expect_zero:
            status='passed' if max(float(np.abs(ad).max()),float(np.abs(fd[:,i]).max()))<1e-6 else 'failed'
        elif scale<1e-4 or 4*se>.25*scale or 4*signal_se>.5*scale:
            status='inconclusive'
        else:
            status='passed' if abs(am-fm)<=limit else 'failed'
        checks.append(dict(epsilon=eps, ad_mean=am, fd_mean=fm,
                           paired_difference_se=se, tolerance=limit, status=status))
    means=fd.mean(0); scale=max(float(np.abs(means).max()),1e-4)
    stable=float(np.ptp(means)) <= 2e-4+.1*scale
    statuses=[v['status'] for v in checks]
    if not stable: status='inconclusive'
    elif statuses.count('passed')>=2 and 'failed' not in statuses: status='passed'
    elif statuses.count('failed')>=2: status='failed'
    else: status='inconclusive'
    return dict(status=status, epsilon_stable=stable, checks=checks,
                ad=ad.tolist(), fd=fd.tolist(), expected_zero=expect_zero)


def patches(mask, rgb, size=8, seed=19019):
    """Choose distinct fully interior dark and light tiles at source resolution."""
    mask=np.asarray(mask,bool); h,w=mask.shape
    if rgb.shape != (h,w,3): raise ValueError('Mask/image dimensions differ')
    ys,xs=np.nonzero(mask); rng=np.random.default_rng(seed)
    order=rng.permutation(len(ys)); candidates=[]; seen=set()
    for i in order[:min(len(order),30000)]:
        y,x=int(ys[i])-size//2,int(xs[i])-size//2
        if y<0 or x<0 or y+size>h or x+size>w or (y,x) in seen: continue
        seen.add((y,x))
        if mask[y:y+size,x:x+size].all():
            brightness=float(np.asarray(rgb[y:y+size,x:x+size]).mean())
            candidates.append((brightness,y,x))
            if len(candidates)>=1024: break
    if len(candidates)<2: raise ValueError('Too little interior for two patches')
    candidates.sort()
    dark=candidates[int(.2*(len(candidates)-1))]
    others=[c for c in candidates if abs(c[1]-dark[1])>=size or abs(c[2]-dark[2])>=size]
    if not others: raise ValueError('Cannot select two disjoint patches')
    light=others[int(.8*(len(others)-1))]
    return [dict(label=label,x=x,y=y,size=size,mean_linear_rgb=v)
            for label,(v,y,x) in zip(['dark','light'],[dark,light])]


def pixel_ids(record, patch):
    y,x=np.mgrid[patch['y']:patch['y']+patch['size'],patch['x']:patch['x']+patch['size']]
    return (y*record['width']+x).ravel()


def combined_status(values):
    values=list(values)
    if not values: return 'inconclusive'
    if 'failed' in values: return 'failed'
    return 'passed' if all(s=='passed' for s in values) else 'inconclusive'
