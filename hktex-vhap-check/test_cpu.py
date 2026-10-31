"""Analytic tests for camera, visibility, topology and the exported data contract."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from common import project, rays, rasterize, topology, srgb_to_linear, linear_to_srgb, verify_dataset
from prepare import prepare

def camera():
    return dict(w=64,h=48,fl_x=45.,fl_y=42.,cx=31.3,cy=23.7,transform_matrix=np.eye(4).tolist())

class CPUChecks(unittest.TestCase):
    def test_camera_centres_translated_rotated(self):
        c=camera();theta=.3
        T=np.array([[np.cos(theta),0,np.sin(theta),.4],[0,1,0,-.2],[-np.sin(theta),0,np.cos(theta),.8],[0,0,0,1]])
        c['transform_matrix']=T.tolist();ids=np.array([0,31,555,3071])
        o,d=rays(c,ids);uv,z=project(o+d*3,c);y,x=np.divmod(ids,c['w'])
        np.testing.assert_allclose(uv,np.c_[x+.5,y+.5],atol=1e-10)
        self.assertTrue((z>0).all())

    def test_perspective_correct_first_hit(self):
        c=camera();v=np.array([[-1,-.7,-2],[1,-.7,-3],[0,1,-2.5]],float);f=np.array([[0,1,2]])
        # Back triangle projects to exactly the same pixels. Front must win regardless of order.
        vv=np.r_[v*2,v];ff=np.array([[0,1,2],[3,4,5]])
        fid,bc,depth=rasterize(vv,ff,c);ids=np.flatnonzero(fid>=0)
        self.assertGreater(len(ids),100);self.assertTrue((fid.ravel()[ids]==1).all())
        o,d=rays(c,ids);normal=np.cross(v[1]-v[0],v[2]-v[0]);t=((v[0]-o)@normal)/(d@normal)
        expected=o+d*t[:,None];actual=(vv[ff[fid.ravel()[ids]]]*bc.reshape(-1,3)[ids,:,None]).sum(1)
        np.testing.assert_allclose(actual,expected,atol=2e-7)
        np.testing.assert_allclose(depth.ravel()[ids],-expected[:,2],atol=3e-7)

    def test_open_components_and_bad_winding(self):
        v=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[2,0,0],[3,0,0],[2,1,0]],float)
        f=np.array([[0,1,2],[0,2,3],[4,5,6]])
        r,*_=topology(v,f);self.assertEqual(r['components'],2);self.assertEqual(r['boundary_edges'],7)
        v[3,1]=1.2  # Unequal areas avoid cancellation of averaged normals in this intentionally bad mesh.
        r,*_=topology(v,np.array([[0,1,2],[0,3,2],[4,5,6]]))
        self.assertEqual(r['inconsistent_winding_edges'],1)

    def test_colour_decode_not_identity(self):
        x=np.array([0,.04045,.5,1]);y=srgb_to_linear(x)
        self.assertAlmostEqual(float(y[2]),.21404114,places=6)
        np.testing.assert_allclose(linear_to_srgb(y),x,atol=1e-7)

    def test_single_frame_split_and_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'source';src.mkdir();out=root/'ready';out.mkdir()
            (src/'mesh.obj').write_text('v -1 -1 -2\nv 1 -1 -2\nv 1 1 -2\nv -1 1 -2\nf 1 2 3\nf 1 3 4\n')
            Image.new('RGB',(64,48),(128,128,128)).save(src/'rgb.png');Image.new('L',(64,48),255).save(src/'alpha.png')
            rows=[]
            for ci in [0,1,7,15]:
                rows.append(dict(camera(),timestep_index=0,camera_index=ci,camera_id=str(ci),file_path='rgb.png',fg_mask_path='alpha.png',mesh_path='mesh.obj'))
            (src/'transforms.json').write_text(json.dumps(dict(frames=rows)))
            np.savez(root/'region.npz',vertex_count=4,faces=np.array([[0,1,2],[0,2,3]]),allowed_faces=[True,True])
            a=argparse.Namespace(export=str(src),frame=0,val_camera=7,test_camera=15,regions=str(root/'region.npz'),geometry_only=False,alpha_threshold=.99,min_cosine=.25,erode=2)
            prepare(a,out,{});m=verify_dataset(out)
            self.assertEqual([x['split'] for x in m['views']],['train','train','val','test'])
            self.assertFalse(m['training_permitted']);self.assertTrue(m['checks']['has_region_labels'])
            rgb=np.load(out/m['views'][0]['rgb']);self.assertAlmostEqual(float(rgb[0,0,0]),.2158605,places=6)
            (out/m['views'][0]['rgb']).write_bytes(b'altered')
            with self.assertRaisesRegex(ValueError,'Changed asset'):verify_dataset(out)

if __name__=='__main__':unittest.main()
