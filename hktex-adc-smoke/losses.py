"""Matched RGB normalization. Only the scalar loss pipeline differs between arms."""
import numpy as np

Q_FLOOR=1e-12
ARMS=('linear','loggamma')


def numpy_transform(x):
    x=np.clip(np.asarray(x,dtype=np.float64),0,1)
    q=(np.log(65536*x+1e-6)-np.log(1e-6))/(np.log(1+1e-6)-np.log(1e-6))
    return 1.055*np.maximum(q,Q_FLOOR)**(1/2.4)-.055


def dr_transform(x,dr):
    x=dr.clip(x,0.,1.)
    q=(dr.log(x*65536+1e-6)-dr.log(1e-6))/(dr.log(1.+1e-6)-dr.log(1e-6))
    # The author's fractional power has an infinite derivative at q=0.
    # This floor changes F(0) by 1.055e-5; it is a documented numerical variant.
    return 1.055*dr.power(dr.maximum(q,Q_FLOOR),1/2.4)-.055


def objective(pred,target,arm,mi,dr):
    if arm=='loggamma':pred,target=dr_transform(pred,dr),dr_transform(target,dr)
    elif arm!='linear':raise ValueError(arm)
    # SAME reduction for A and B (author's original code sums RGB instead).
    return dr.mean(dr.dot(dr.abs(pred-target),mi.Color3f(1/3)))


def validate_arm(saved,arm,code_hash):
    if saved['arm']!=arm:raise ValueError('Cannot change loss arm when resuming')
    if saved['code_hash']!=code_hash:raise ValueError('Launcher/loss code changed since checkpoint')


def objective_name(arm):
    return 'linear RGB mean L1' if arm=='linear' else 'author-style clip/log/gamma mean L1; q floor 1e-12'
