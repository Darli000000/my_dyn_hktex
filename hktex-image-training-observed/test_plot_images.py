"""Synthetic schema fixture: exercises all CPU math panels with NO UV/surface GT."""
import os
from pathlib import Path
import tempfile
import sys
import unittest
import numpy as np
from observer_common import json_write,npz_write
from image_observer import ImageRecorder
from test_image_observer import fake_trainer


class PlotTest(unittest.TestCase):
    def test_math_plot_without_surface_gt(self):
        with tempfile.TemporaryDirectory() as td:
            os.environ.setdefault('MPLCONFIGDIR',td)
            t=fake_trainer(Path(td));rec=ImageRecorder(t)
            t.update=2;rec.after_update(dict(loss_l1=.1,kernels=4))
            name=rec.snapshot('final','fixture');rec.finish('training_complete')
            row=rec.index[-1];stem=Path(name).stem
            analysis=rec.root/'analysis';analysis.mkdir()
            n=4096
            fields=dict(predicted=np.full((n,3),.2),root_ids=np.array([0]),representative_ids=np.array([0]))
            for k in ['raw','normalized','weighted','alpha','distance_squared','distance_weight','euclidean']:
                fields[k]=np.full((n,1),.1)
            fields.update(candidate=np.ones((n,1),bool),selected=np.ones((n,1),bool))
            npz_write(analysis/(stem+'_fields.npz'),**fields)
            probes={k:np.ones((1,4,8))*.1 for k in ['evals','decay','phi_source','phi_query','spectral_terms','biharmonic_terms']}
            probes.update(raw=np.ones((1,4)),candidate_ids=np.arange(4)[None],selected_local=np.array([[0,1]]),
                          distance_squared=np.ones((1,4)),distance_weight=np.ones((1,4)),
                          colour_contributions=np.ones((1,2,3))*.1,mean_rgb=np.zeros(3),preclamp=np.ones((1,3))*.2,
                          final_rgb=np.ones((1,3))*.2,grid_angle_axis=np.array([0.,1.]),grid_anisotropy_axis=np.array([1.,5.]),
                          grid_indices=np.tile(np.arange(4),(1,4,1)),grid_weights=np.ones((1,4,4))*.25,
                          angles=np.ones((1,4))*.5,anisotropies=np.ones((1,4))*2)
            npz_write(analysis/(stem+'_probes.npz'),**probes)
            json_write(analysis/'metrics.json',[dict(row,max_forward_decomposition_error=0.)])
            json_write(analysis/'replay.json',dict(roots=[0]))
            import plot_image_fields
            old=sys.argv
            try:
                sys.argv=['plot_image_fields.py','--recording',str(rec.root)]
                plot_image_fields.main()
            finally:sys.argv=old
            out=rec.root/'viewer'
            for panel in ['overview','kernels','spectrum','distance','neighbours','colour','grid','parameters']:
                self.assertTrue((out/(stem+'_'+panel+'.png')).is_file())
            self.assertIn('没有 UV 表面 GT',(out/'index.html').read_text())


if __name__=='__main__':unittest.main(verbosity=2)
