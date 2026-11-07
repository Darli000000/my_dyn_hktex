"""Install the proven direct spectral-distance search only inside this process."""
def install():
    import torch
    from hktex.knn_heat.knn_heat.faiss_knn import FaissGpuFlatIndex
    def search(self,queries,k):
        if self._db is None or self._index is None:raise RuntimeError('KNN index is not built')
        if self.config.metric!='l2':raise ValueError('Stable search requires squared L2')
        self._check_cuda_2d(queries,'queries')
        if queries.shape[1]!=self._db.shape[1] or not 0<k<=len(self._db):raise ValueError('Invalid KNN dimensions')
        with torch.no_grad():
            d2=torch.cdist(queries.detach().float(),self._db.detach().float(),p=2,
                           compute_mode='donot_use_mm_for_euclid_dist').square()
            idx=torch.argsort(d2,dim=1,stable=True)[:,:int(k)]
            return d2.gather(1,idx),idx
    FaissGpuFlatIndex.search=search
    return 'direct_cdist_no_mm_stable_argsort + eager_direct_differentiable_distances'
