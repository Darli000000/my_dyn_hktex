"""Compare captured KNN stages by source ID, independently of candidate order."""
import numpy as np


def compare(a, b, embeddings=None):
    delta=np.abs(a['predicted'].astype(np.float64)-b['predicted'])
    point_delta=delta.max(axis=1)
    out=dict(points=len(point_delta),changed_predictions=int((point_delta>0).sum()),
        changed_over_1e_5=int((point_delta>1e-5).sum()),max_rgb_difference=float(delta.max()),
        mean_abs_rgb_difference=float(delta.mean()),worst_point=int(point_delta.argmax()))
    if 'indices' not in a or 'indices' not in b:return out
    ia,ib=a['indices'],b['indices']
    set_change=np.any(np.sort(ia,axis=1)!=np.sort(ib,axis=1),axis=1)
    order_change=np.any(ia!=ib,axis=1)
    selected_change=np.any(np.sort(a['selected'],axis=1)!=np.sort(b['selected'],axis=1),axis=1)
    out.update(candidate_order_changed=int(order_change.sum()),candidate_sets_changed=int(set_change.sum()),
        selected_sets_changed=int(selected_change.sum()),
        changed_predictions_with_same_candidate_set=int(((point_delta>1e-5)&~set_change).sum()))
    keys=[k for k in ['distance_squared','raw','normalized','weighted','alpha','self_heat'] if k in a]
    max_common={k:0. for k in keys}; changed_common={k:0 for k in keys};examples=[];same_embedding_replacements=0
    for row in range(len(ia)):
        _,xa,xb=np.intersect1d(ia[row],ib[row],return_indices=True)
        for key in keys:
            diff=np.abs(a[key][row,xa].astype(np.float64)-b[key][row,xb])
            md=float(diff.max()) if diff.size else 0.
            max_common[key]=max(max_common[key],md);changed_common[key]+=int(md>1e-5)
        if set_change[row]:
            removed=np.setdiff1d(ia[row],ib[row]);added=np.setdiff1d(ib[row],ia[row])
            ties=None
            if embeddings is not None:
                ties=sorted(v.tobytes() for v in embeddings[removed])==sorted(v.tobytes() for v in embeddings[added])
                same_embedding_replacements+=int(ties)
            if len(examples)<16:
                examples.append(dict(point=row,removed=removed.tolist(),added=added.tolist(),
                    identical_embedding_multisets=ties,max_rgb_difference=float(point_delta[row])))
    out.update(common_candidate_stage_max_differences=max_common,common_candidate_points_changed_over_1e_5=changed_common,
        changed_sets_replacing_identical_embedding_multisets=same_embedding_replacements,examples=examples)
    return out
