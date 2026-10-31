#!/usr/bin/env python3
"""Prepare one VHAP timestep for surface-appearance checks, preserving topology and cameras."""
import argparse
import json
from pathlib import Path
import traceback
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from common import SCHEMA, inside, read_obj, topology, rasterize, project, sha, srgb_to_linear, write_json
from common import REGIONS, region_counts

def diagnostic(rgb,alpha,hit,mask,path):
    h,w=mask.shape
    a=np.asarray(rgb).copy();a[mask]=(a[mask]*.55+np.array([0,255,90])*.45).astype(np.uint8)
    binary=Image.fromarray((hit*255).astype(np.uint8));edge=np.asarray(binary)!=np.asarray(binary.filter(ImageFilter.MinFilter(3)))
    a[edge]=[0,190,255]
    b=np.asarray(rgb).copy();b[~mask]=b[~mask]//4
    grid=Image.new('RGB',(w*3,h+30),'white');d=ImageDraw.Draw(grid)
    for i,(im,title) in enumerate([(rgb,'RGB (exported)'),(Image.fromarray(a),'cyan: mesh / green: candidate ROI'),(Image.fromarray(b),'candidate pixels only')]):
        grid.paste(im,(i*w,30));d.text((i*w+6,8),title,fill='black')
    grid.save(path)

def prepare(a,out,report):
    root=Path(a.export).resolve();src=root/'transforms.json';db=json.loads(src.read_text())
    rows=[r for r in db['frames'] if r['timestep_index']==a.frame]
    if not rows:raise ValueError('Missing timestep')
    rows.sort(key=lambda r:r['camera_index'])
    ids=[r['camera_index'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate camera record')
    if a.val_camera==a.test_camera or not {a.val_camera,a.test_camera}<=set(ids):raise ValueError('Distinct held-out camera indices must exist')
    if len({r['mesh_path'] for r in rows})!=1:raise ValueError('Single timestep uses multiple meshes')
    mesh_path=inside(root,rows[0]['mesh_path']);v,f=read_obj(mesh_path)
    topo,fn,vn,component,boundary=topology(v,f);report['topology']=topo
    write_json(out/'mesh_report.json',topo)
    if topo['nonmanifold_edges'] or topo['inconsistent_winding_edges']:raise ValueError('Mesh edge topology needs inspection')
    allowed=np.ones(len(f),bool)
    if a.regions:
        with np.load(a.regions,allow_pickle=False) as r:
            if int(r['vertex_count'])!=len(v) or not np.array_equal(r['faces'],f):raise ValueError('Region topology mismatch')
            allowed=r['allowed_faces'].astype(bool)
            face_regions=r['face_regions'].astype(np.uint8)
            if face_regions.shape!=(len(f),) or not np.array_equal(allowed,face_regions>0):raise ValueError('Invalid region labels')
            if allowed.shape!=(len(f),) or not allowed.any():raise ValueError('Invalid region mask')
    else:raise ValueError('Eye-inclusive region file is required')
    np.savez_compressed(out/'mesh.npz',vertices=v.astype(np.float32),faces=f.astype(np.int32),
        face_normals=fn.astype(np.float32),vertex_normals=vn.astype(np.float32),face_components=component,
        boundary_edges=boundary,allowed_faces=allowed,face_regions=face_regions)
    manifest=dict(schema=SCHEMA,timestep=a.frame,mesh='mesh.npz',views=[],
        camera_convention='OpenGL camera-to-world; x right, y up, z backward; pixel centres x+0.5,y+0.5',
        appearance_contract='fixed-illumination surface RGB; no extra lights/BSDF; not physical albedo',
        colour_contract='sRGB EOTF decoded once from VHAP colour-corrected white-composited PNG; no second correction/compositing',
        pixel_filter='centre ray approximation, no synthetic box-filter claim',
        geometry_modified=False,roi_semantics='anatomical candidate requiring visual occlusion review' if a.regions else 'geometry-only: NOT for training',
        training_permitted=False,region_names=REGIONS,split_scope='held-out appearance cameras; VHAP geometry used all views',
        provenance=dict(export_root=str(root),transforms_sha256=sha(src),source_mesh_sha256=sha(mesh_path),
            regions_sha256=sha(a.regions) if a.regions else None),sha256={})
    report['views']=[];preview_paths=[]
    for r in rows:
        ci=r['camera_index'];key=f'view_{ci:02d}';folder=out/key;folder.mkdir()
        report['current_view']=key;write_json(out/'report.json',report)
        rgb=Image.open(inside(root,r['file_path'])).convert('RGB');alpha_im=Image.open(inside(root,r['fg_mask_path'])).convert('L')
        if rgb.size!=(r['w'],r['h']) or alpha_im.size!=rgb.size:raise ValueError('RGB/alpha/intrinsic resolution mismatch')
        print('[PREPARE]',key,flush=True)
        face,bary,depth=rasterize(v,f,r);hit=face>=0;safe=np.maximum(face,0)
        xyz=np.einsum('...j,...jk->...k',bary,v[f[safe]])
        cam=np.asarray(r['transform_matrix'])[:3,3];towards=cam-xyz
        norm=np.linalg.norm(towards,axis=-1);cos=(fn[safe]*towards).sum(-1)/np.maximum(norm,1e-30)
        alpha=np.array(alpha_im,dtype=np.float32)/255
        candidate=hit&allowed[safe]&(alpha>=a.alpha_threshold)&(cos>=a.min_cosine)
        if a.erode:
            candidate=np.asarray(Image.fromarray((candidate*255).astype(np.uint8)).filter(ImageFilter.MinFilter(a.erode*2+1)))>0
        # Geometric reprojection check using perspective-correct surface coordinates.
        sample=np.flatnonzero(candidate)[::max(1,int(candidate.sum())//256)]
        if len(sample):
            uv,_=project(xyz.reshape(-1,3)[sample],r);y,x=np.divmod(sample,r['w'])
            err=float(abs(uv-np.c_[x+.5,y+.5]).max())
            if err>1e-3:raise ValueError('CPU projection inconsistency')
        else:err=None
        valid_pixels=int(candidate.sum());split='val' if ci==a.val_camera else 'test' if ci==a.test_camera else 'train'
        used=valid_pixels>=32
        if not used and split!='train':raise ValueError('Held-out camera has fewer than 32 candidate pixels')
        rgb.save(folder/'preview.png');alpha_im.save(folder/'alpha.png')
        Image.fromarray((candidate*255).astype(np.uint8)).save(folder/'candidate_mask.png')
        pixel_regions=np.where(candidate,face_regions[safe],0).astype(np.uint8)
        Image.fromarray(pixel_regions).save(folder/'regions.png')
        palette=np.array([[40,40,40],[70,200,90],[255,190,35],[60,140,255],[225,80,210]],np.uint8)
        Image.fromarray(palette[pixel_regions]).save(folder/'regions_preview.png')
        np.save(folder/'rgb_linear.npy',srgb_to_linear(np.asarray(rgb)/255),allow_pickle=False)
        np.savez_compressed(folder/'geometry.npz',face_ids=face,barycentric=bary,depth_cv=depth)
        diagnostic(rgb,alpha,hit,candidate,folder/'overlay.png');preview_paths.append(folder/'overlay.png')
        row=dict(id=key,camera_index=ci,camera_id=r['camera_id'],split=split if used else 'excluded',
            width=r['w'],height=r['h'],camera=r,rgb=f'{key}/rgb_linear.npy',alpha=f'{key}/alpha.png',
            mask=f'{key}/candidate_mask.png',regions=f'{key}/regions.png',geometry=f'{key}/geometry.npz',preview=f'{key}/preview.png',
            valid_pixels=valid_pixels,source_rgb_sha256=sha(inside(root,r['file_path'])),source_alpha_sha256=sha(inside(root,r['fg_mask_path'])))
        manifest['views'].append(row)
        report['views'].append(dict(id=key,split=row['split'],candidate_pixels=valid_pixels,mesh_pixels=int(hit.sum()),
            alpha_foreground_pixels=int((alpha>=a.alpha_threshold).sum()),cpu_reprojection_max_px=err,
            projected_faces=int(len(np.unique(face[hit]))),region_pixels=region_counts(pixel_regions)))
    if sum(r['split']=='train' for r in manifest['views'])<2:raise ValueError('Fewer than two usable training cameras')
    # Contact sheet of all cameras. Full-sized overlays remain in view folders.
    thumbs=[]
    for path in preview_paths:
        im=Image.open(path);im.thumbnail((660,340));thumbs.append((path.parent.name,im.copy()))
    sheet=Image.new('RGB',(1320,370*((len(thumbs)+1)//2)),'white');draw=ImageDraw.Draw(sheet)
    for i,(name,im) in enumerate(thumbs):
        x=(i%2)*660;y=(i//2)*370;draw.text((x+6,y+5),name,fill='black');sheet.paste(im,(x,y+25))
    sheet.save(out/'roi_contact_sheet.jpg',quality=92)
    manifest['sha256']={'mesh.npz':sha(out/'mesh.npz')}
    for r in manifest['views']:
        for k in ['rgb','alpha','mask','regions','geometry','preview']:manifest['sha256'][r[k]]=sha(out/r[k])
    manifest['checks']=dict(alpha_threshold=a.alpha_threshold,min_cosine=a.min_cosine,erosion_pixels=a.erode,
        cpu_geometry_checks_passed=True,has_region_labels=bool(a.regions),roi_visual_review_required=True)
    write_json(out/'manifest.json',manifest)
    report.update(status='cpu_checks_passed' if a.regions else 'geometry_only_inconclusive',
        candidate_roi_ready=bool(a.regions),gpu_tested=False,training_ready=False,
        next='pipeline.py checks all regions, then starts short training automatically. ROI visuals still need review.')
    write_json(out/'report.json',report)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--export',required=True);p.add_argument('--output',required=True)
    p.add_argument('--frame',type=int,default=0);p.add_argument('--regions',required=True)
    p.add_argument('--val-camera',type=int,default=7);p.add_argument('--test-camera',type=int,default=15)
    p.add_argument('--alpha-threshold',type=float,default=.99);p.add_argument('--min-cosine',type=float,default=.25);p.add_argument('--erode',type=int,default=2)
    a=p.parse_args();out=Path(a.output).resolve()
    if out.exists():p.error('Output must be new')
    if not 0<a.alpha_threshold<=1 or not 0<=a.min_cosine<1 or a.erode<0:p.error('Invalid mask thresholds')
    out.mkdir(parents=True);report=dict(status='running',stage='prepare',args=vars(a))
    try:prepare(a,out,report)
    except BaseException as e:
        report.update(status='failed',error=str(e),traceback=traceback.format_exc());write_json(out/'report.json',report);raise
    print('[SUCCESS]',out,flush=True)
if __name__=='__main__':main()
