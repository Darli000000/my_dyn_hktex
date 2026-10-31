"""CPU contract tests; these do not substitute for the server CUDA preflight."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from common import (SCHEMA, face_regions, project, rays, rasterize, srgb_to_linear,
                    linear_to_srgb, verify_dataset, sha)
from runtime import sample_stratified, metrics
from prepare import prepare


def camera(w=96,h=72):
    return dict(w=w,h=h,fl_x=60.,fl_y=60.,cx=w/2,cy=h/2,
                transform_matrix=np.eye(4).tolist())

class Contracts(unittest.TestCase):
    def test_disconnected_eyes_are_explicitly_included(self):
        faces=np.arange(15).reshape(5,3)
        groups={k:np.array([],int) for k in ['face','eye_region','left_eyeball','right_eyeball',
                  'neck','scalp','boundary','lips','lip_inside','teeth']}
        groups.update(face=np.arange(6),eye_region=np.arange(3,6),
                      left_eyeball=np.arange(6,9),right_eyeball=np.arange(9,12),neck=np.arange(12,15))
        np.testing.assert_array_equal(face_regions(faces,15,groups),[1,2,3,4,0])

    def test_pixel_centres_and_opengl_camera(self):
        r=camera();r['transform_matrix'][0][3]=.3
        ids=np.array([0,96*36+48,96*72-1]);o,d=rays(r,ids)
        uv,depth=project(o+d*2,r);y,x=np.divmod(ids,96)
        np.testing.assert_allclose(uv,np.c_[x+.5,y+.5],atol=1e-10)
        self.assertTrue((depth>0).all())

    def test_full_mesh_occludes_even_reversed_faces(self):
        # Identical projected triangles at two depths; nearer triangle has reversed winding.
        near=np.array([[-.5,-.4,-2],[.5,-.4,-2],[0,.5,-2]])
        vertices=np.vstack([near*2,near]);faces=np.array([[0,1,2],[3,5,4]])
        face,bc,_=rasterize(vertices,faces,camera());hit=face>=0
        self.assertTrue(hit.any());self.assertTrue((face[hit]==1).all())
        points=(vertices[faces[face[hit]]]*bc[hit,:,None]).sum(1)
        uv,_=project(points,camera());y,x=np.where(hit)
        np.testing.assert_allclose(uv,np.c_[x+.5,y+.5],atol=2e-5)

    def test_region_sampling_preserves_pixel_mse(self):
        # Tiny eye regions are oversampled; their objective weight must stay small.
        labels=np.repeat(np.arange(1,5),[900,90,7,3])
        pool=dict(regions=labels,rgb=np.repeat((labels/4)[:,None],3,axis=1))
        batch,w=sample_stratified(pool,np.random.default_rng(7))
        self.assertAlmostEqual(float(w.sum()),1.,places=6)
        weighted=float((batch['rgb'][:,0]**2*w).sum())
        self.assertAlmostEqual(weighted,float((pool['rgb'][:,0]**2).mean()),places=7)
        self.assertEqual(int((batch['regions']==4).sum()),128)

    def test_empty_region_metric_and_colour_roundtrip(self):
        scores=metrics(np.ones((2,3))*.2,np.zeros((2,3)),np.ones(2,int))
        self.assertAlmostEqual(scores['all']['mse'],.04)
        self.assertIsNone(scores['regions']['right_eye']['mse'])
        json.dumps(scores,allow_nan=False)
        values=np.array([0,.01,.04045,.3,1.],np.float32)
        np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(values)),values,atol=1e-6)

    def test_non_square_single_frame_prepare_and_asset_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();out=root/'prepared';out.mkdir()
            # Four disconnected visible triangles, one for each supervised region.
            v=[]
            for x,y in [(-.65,-.45),(.65,-.45),(-.65,.45),(.65,.45)]:
                v.extend([[x-.3,y-.25,-2],[x+.3,y-.25,-2],[x,y+.25,-2]])
            v=np.array(v);f=np.arange(12).reshape(4,3)
            (source/'mesh.obj').write_text(''.join('v %s %s %s\n'%tuple(p) for p in v)+
                                          ''.join('f %s %s %s\n'%tuple(t+1) for t in f))
            region=root/'regions.npz';np.savez(region,vertex_count=12,faces=f,
                                            allowed_faces=np.ones(4,bool),face_regions=np.arange(1,5,dtype=np.uint8))
            rows=[]
            for ci in [0,1,7,15]:
                Image.new('RGB',(96,72),(128,100,80)).save(source/f'rgb{ci}.png')
                Image.new('L',(96,72),255).save(source/f'mask{ci}.png')
                row=camera();row.update(timestep_index=0,camera_index=ci,camera_id=str(ci),
                    file_path=f'rgb{ci}.png',fg_mask_path=f'mask{ci}.png',mesh_path='mesh.obj');rows.append(row)
            (source/'transforms.json').write_text(json.dumps(dict(frames=rows)))
            args=SimpleNamespace(export=str(source),frame=0,regions=str(region),val_camera=7,
                                 test_camera=15,alpha_threshold=.99,min_cosine=.25,erode=1)
            report={};prepare(args,out,report);manifest=verify_dataset(out)
            self.assertEqual(report['status'],'cpu_checks_passed');self.assertEqual(manifest['schema'],SCHEMA)
            self.assertEqual([r['split'] for r in manifest['views']],['train','train','val','test'])
            for row in manifest['views']:
                labels=np.array(Image.open(out/row['regions']))
                self.assertEqual(labels.shape,(72,96));self.assertEqual(set(np.unique(labels)),{0,1,2,3,4})
            with open(out/manifest['views'][0]['regions'],'ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'Changed asset'):verify_dataset(out)

if __name__=='__main__':unittest.main()
