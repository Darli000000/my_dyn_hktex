"""CPU invariants for semantic masks, missing-region sampling and original ROI evaluation."""
import unittest
import numpy as np
from prepare import region_image,make_mask,morph
from runtime import sample_stratified,pool_views

class Tests(unittest.TestCase):
    def fixture(self):
        # Four disconnected fixture triangles: head, two eyes, teeth.
        v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]*4,np.float32)
        f=np.arange(12).reshape(4,3)
        return dict(vertices=v,faces=f,face_components=np.arange(4),face_regions=np.array([1,3,4,0]),face_normals=np.tile([0,0,1.],(4,1)),boundary_edges=np.empty((0,3),int))
    def test_hair_cloth_jewelry_mouth_never_labels(self):
        g=self.fixture();s=np.array([[0,3,10,13,14,15,16,18]],np.uint8)
        self.assertFalse(region_image(s,np.zeros_like(s),g).any())
    def test_nose_lips_ears_neck_are_labeled(self):
        s=np.array([[2,11,12,8,9,17]],np.uint8)
        self.assertEqual(region_image(s,np.zeros_like(s),self.fixture()).tolist(),[[6,5,5,7,7,8]])
    def test_eye_identity_comes_from_geometry(self):
        s=np.array([[5,4,1]],np.uint8);f=np.array([[1,2,3]])
        self.assertEqual(region_image(s,f,self.fixture()).tolist(),[[3,4,0]])
    def test_internal_semantic_boundary_is_not_eroded(self):
        g=self.fixture();s=np.ones((15,15),np.uint8);s[:,7:]=2
        f=np.zeros(s.shape,int);b=np.zeros((*s.shape,3),np.float32);b[...,0]=1
        cam=dict(transform_matrix=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,2],[0,0,0,1.]]))
        mask,labels,_=make_mask(s,np.ones(s.shape),f,b,np.ones(s.shape),g,cam)
        self.assertTrue(mask[7,6:9].all());self.assertEqual(labels[7,8],6)
    def test_backfaces_and_open_boundary_excluded(self):
        g=self.fixture();s=np.ones((9,9),np.uint8);f=np.zeros(s.shape,int);b=np.zeros((*s.shape,3));b[...,0]=1
        camera={'transform_matrix':np.eye(4)};camera['transform_matrix'][2,3]=-2
        args=(s,np.ones(s.shape),f,b,np.ones(s.shape),g,camera)
        self.assertFalse(make_mask(*args)[0].any())
        camera['transform_matrix'][2,3]=2;g['boundary_edges']=np.array([[0,1,0]])
        self.assertFalse(make_mask(*args)[0].any())
    def test_sampling_is_unbiased_and_missing_optional_regions_work(self):
        labels=np.repeat(np.arange(1,7),[900,80,5,5,5,5]);pool={'regions':labels}
        batch,w=sample_stratified(pool,np.random.default_rng(4))
        self.assertEqual(len(w),1024);self.assertAlmostEqual(float(w.sum()),1.,places=6)
        for k in range(1,7):self.assertAlmostEqual(float(w[batch['regions']==k].sum()),float((labels==k).mean()),places=6)
        b2,w2=sample_stratified(pool,np.random.default_rng(4));np.testing.assert_array_equal(batch['regions'],b2['regions'])
    def test_old_roi_is_evaluated_even_when_removed_from_new_mask(self):
        v={'row':{'split':'val'},'active':np.array([True,False,True]),'legacy':np.array([True,True,False]),'regions':np.array([5,1,6]),'legacy_regions':np.array([1,2,0]),'faces':np.arange(3),'points':np.zeros((3,3)),'rgb':np.zeros((3,3))}
        new=pool_views([v],'val');old=pool_views([v],'val','legacy')
        np.testing.assert_array_equal(new['faces'],[0,2]);np.testing.assert_array_equal(old['faces'],[0,1]);np.testing.assert_array_equal(old['regions'],[1,2])
    def test_evaluator_keeps_original_roi_separate_and_audit_is_readonly(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from engine import Evaluator
        views=[]
        for split,cam in [('train',0),('val',7),('test',15)]:
            views.append(dict(row=dict(split=split,camera_index=cam,id=f'view_{cam:02d}',height=1,width=3),
                active=np.array([True,False,True]),legacy=np.array([True,True,False]),
                regions=np.array([5,1,6]),legacy_regions=np.array([1,2,0]),faces=np.arange(3),ids=np.arange(3),
                points=np.zeros((3,3)),rgb=np.array([[0.,0.,0.],[1.,1.,1.],[0.,0.,0.]])))
        evaluator=Evaluator(views,0);model=SimpleNamespace(model=SimpleNamespace(N_sources=5000))
        with patch('engine.predict',side_effect=lambda model,p,f:np.zeros((len(p),3))),tempfile.TemporaryDirectory() as td:
            r=evaluator.run(model,Path(td),5000,True)
            self.assertEqual(r['val']['all']['pixels'],2);self.assertEqual(r['val']['all']['mse'],0)
            self.assertEqual(r['legacy_val']['all']['mse'],.5);self.assertEqual(r['legacy_test']['all']['mse'],.5)
            a=evaluator.audit_after(model,evaluator.audit_before(model));self.assertEqual(a['mean_abs_rgb_change'],0)
if __name__=='__main__':unittest.main()
