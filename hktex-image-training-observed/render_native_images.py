"""Offline calibrated image viewer. Reuses native viewer UI/rings, never reads UV GT."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
from offline_common import read_json, sha256, sensor_dict, save_preview
from observer_common import PARAMS, json_write, npz_write
from image_observer import verify_recording
from native_display import ring_mode, contact_sheet, build_html
from image_dataset import ImageDataset, check_contract, install_no_texture_guard


def box_target(image, size):
    h,w=image.shape[:2]
    if h != w or h % size or size > h:
        raise ValueError('Render size must divide dataset image size (e.g. 256/512/1024/2048)')
    k=h//size
    return np.asarray(image).reshape(size,k,size,k,3).mean((1,3))


def selected_records(dataset, names):
    records={r['id']:r for r in dataset.views}
    wanted=names.split(',')
    if len(set(wanted))!=len(wanted) or any(n not in records for n in wanted):
        raise ValueError('Views must be distinct dataset IDs, e.g. view_000,view_002')
    return [records[n] for n in wanted]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--recording',help='New image-training observer directory')
    source.add_argument('--run',help='Existing v1.0/v1.1 image-training directory; render checkpoint only')
    p.add_argument('--repo',default='/root/hktex-work/hktex')
    p.add_argument('--dataset',help='Dataset location after moving servers')
    p.add_argument('--cache',help='Exact eigen cache .pt after moving servers')
    p.add_argument('--output',required=True,help='NEW directory')
    p.add_argument('--gpu',default='0')
    p.add_argument('--snapshots',default='final',help='final, all regular snapshots, or comma-separated serials')
    p.add_argument('--views',default='view_000,view_002',help='Actual calibrated dataset view IDs')
    p.add_argument('--size',type=int,default=512)
    p.add_argument('--spp',type=int,default=64)
    p.add_argument('--tile-size',type=int,default=64)
    p.add_argument('--point-batch',type=int,default=512)
    p.add_argument('--seed',type=int,default=21931)
    p.add_argument('--no-denoise',action='store_true')
    p.add_argument('--ring-thickness',type=float,default=.03)
    a=p.parse_args()
    if min(a.size,a.spp,a.tile_size,a.point_batch)<1 or a.ring_thickness<=0:
        p.error('Resolution, sampling and chunk settings must be positive')
    out=Path(a.output).resolve()
    if out.exists(): p.error('--output must be new; protect earlier results')
    out.mkdir(parents=True)
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    os.environ.setdefault('MPLBACKEND','Agg')
    try:
        render(a,out)
    except BaseException as exc:
        import traceback
        json_write(out/'failure.json',dict(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()))
        raise


def render(a,out):
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for native rendering')
    torch.cuda.set_device(0)
    if a.recording:
        root=Path(a.recording).resolve()
        print('[RECORDING CHECK]',verify_recording(root),flush=True)
        manifest=read_json(root/'manifest.json'); settings=manifest['settings']
        index=read_json(root/'snapshots.json')
        if a.snapshots=='final': rows=[r for r in index if r['phase']=='final'][-1:]
        elif a.snapshots=='all': rows=[r for r in index if r['phase'] in ['initial','resumed','post_update','final']]
        else:
            serials={int(n) for n in a.snapshots.split(',')}
            rows=[r for r in index if int(r['file'].split('_')[0]) in serials]
            if len(rows)!=len(serials): raise ValueError('Missing requested snapshots')
        if not rows: raise ValueError('No requested snapshots')
        def state_for(row):
            with np.load(root/'snapshots'/row['file'],allow_pickle=False) as d:
                return {k:d[k].copy() for k in (*PARAMS,'stable_ids')}
        cache=Path(a.cache or manifest['cache_path']).resolve()
        if sha256(cache)!=manifest['cache_sha256']: raise ValueError('Cache hash differs from observer')
        ec=manifest['eigen_config']; precision=manifest['matmul_precision']; tf32=manifest['allow_tf32']
        renderer_cfg=manifest['config']['renderer']
    else:
        root=Path(a.run).resolve()
        # This branch allows inspection of the already completed ADC run without retraining.
        state=torch.load(root/'checkpoint.pt',map_location='cpu',weights_only=True)
        settings=state['settings']; update=state['update']
        row=dict(file=f'checkpoint_u{update:05d}.npz',completed_updates=update,phase='checkpoint',
                 kernels=len(state['texture']['_kernel_colours']))
        rows=[row]
        lineage=state.get('observer_state')
        def state_for(row):
            values={k:state['texture'][k].numpy().copy() for k in PARAMS}
            values['stable_ids']=np.asarray(lineage['ids'],np.int64) if lineage else np.arange(row['kernels'],dtype=np.int64)
            return values
        # v1.0 never saved a cache digest. Do not silently claim its identity is verified.
        ec=settings['trainer_config']['network']['eigen_albo']
        cache=Path(a.cache or (str(ec['mesh_path']).rsplit('.',1)[0]+'_'+ec.get('precomputed_name','eigen_albo')+'_'+ec.get('local_frames','principal_curvatures')+'.pt')).resolve()
        precision='highest';tf32=False
        renderer_cfg=None
        print('[NOTE] Old checkpoint: cache SHA is recorded now; no historical cache hash available. IDs may be row indices, not lineage.',flush=True)
    repo=Path(a.repo).resolve()
    for name,digest in settings['source_fingerprints'].items():
        if not (repo/name).is_file() or sha256(repo/name)!=digest:
            raise ValueError('HKTex source changed: '+name)
    dataset_root=Path(a.dataset or settings['dataset']).resolve()
    if sha256(dataset_root/'manifest.json')!=settings['manifest_sha256']:
        raise ValueError('Dataset manifest differs from training')
    checks=check_contract(dataset_root,expected_size=2048)
    if checks['mesh_sha256']!=settings['mesh_sha256']: raise ValueError('Dataset mesh differs')
    install_no_texture_guard(dataset_root)
    data=ImageDataset(dataset_root)
    views=selected_records(data,a.views)
    if any(r['width']%a.size or a.size>r['width'] for r in views):
        raise ValueError('size must divide source image size')
    sys.path.insert(0,str(repo))
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')
    import drjit as dr
    from omegaconf import OmegaConf
    import trimesh
    import hktex
    from hktex.modules.mesh import Mesh
    from hktex.rendering.torch_texture_renderer import TorchTextureRenderer
    from hktex.utils import box_border
    torch.set_float32_matmul_precision(precision)
    torch.backends.cuda.matmul.allow_tf32=tf32
    with np.load(dataset_root/data.manifest['train_mesh'],allow_pickle=False) as g:
        vertices,faces,normals=g['vertices'].copy(),g['faces'].copy(),g['normals'].copy()
    tri=trimesh.Trimesh(vertices=vertices,faces=faces,vertex_normals=normals,process=False)
    mesh=Mesh.from_trimesh(tri,device='cuda')
    cfg=OmegaConf.create(copy.deepcopy(settings['trainer_config']['network']))
    cfg.model.n_sources=1
    cfg.point_batching=a.point_batch
    cfg.eigen_albo=OmegaConf.create(copy.deepcopy(ec))
    # Fail on missing/wrong cache instead of silently recomputing.
    cache_digest=sha256(cache)
    with tempfile.TemporaryDirectory(prefix='hktex-image-viewer-') as td:
        cfg.eigen_albo.mesh_path=str(Path(td)/'mesh.obj')
        cfg.eigen_albo.precomputed_name='replay'
        cfg.eigen_albo.use_precomputed=True
        cfg.eigen_albo.error_if_not_precomputed=True
        (Path(td)/f'mesh_replay_{cfg.eigen_albo.local_frames}.pt').symlink_to(cache)
        network=hktex.find(settings['trainer_config']['network_type'])(cfg,mesh=mesh)
    if renderer_cfg is None:
        renderer_cfg=read_json(dataset_root/data.manifest['renderer_config'])
    rc=OmegaConf.create(renderer_cfg)
    rc.point_batching=a.point_batch
    TorchTextureRenderer.mega_kernel(False,no_loops=True,no_opt_calls=True)
    renderer=TorchTextureRenderer(rc)
    json_write(out/'settings.json',dict(args=vars(a),cache_sha256=cache_digest,
        dataset_manifest_sha256=settings['manifest_sha256'],views=views,
        renderer='training TorchTextureNetwork + path; exact saved geometry/normals/cameras',
        gt='saved linear RGB, box average for lower display resolution',
        noise_note='raw output retained; denoising is presentation only; no KNN tie changes',
        sampler='independent, box pixel filter; cropped film tiles, seed changes by tile',
        precision=precision,allow_tf32=tf32))
    completed=[]
    denoiser=None if a.no_denoise else mi.OptixDenoiser(input_size=(a.size,a.size))

    def render_view(mi_mesh,record,seed):
        image=np.zeros((a.size,a.size,3),np.float32)
        r=dict(record,width=a.size,height=a.size)
        for y in range(0,a.size,a.tile_size):
            for x in range(0,a.size,a.tile_size):
                sd=sensor_dict(mi,r,a.spp)
                sd['film'].update(pixel_format='rgb',crop_offset_x=x,crop_offset_y=y,
                                  crop_width=min(a.tile_size,a.size-x),crop_height=min(a.tile_size,a.size-y))
                spec=renderer.configure_scene()
                spec['integrator']=dict(type='path',max_depth=3,hide_emitters=True)
                spec['camera']=sd;spec['mesh']=mi_mesh
                scene=mi.load_dict(spec)
                raw=mi.render(scene,seed=seed+y*a.size+x)
                tile=np.asarray(raw,dtype=np.float32).copy()
                if not np.isfinite(tile).all(): raise FloatingPointError('Nonfinite render')
                image[y:y+tile.shape[0],x:x+tile.shape[1]]=tile
                del raw,scene
        return image

    for row in rows:
        name=Path(row['file']).stem; folder=out/name;folder.mkdir()
        state=state_for(row)
        with torch.no_grad(),dr.suspend_grad():
            model=network.model
            for k in PARAMS:
                t=torch.as_tensor(state[k],device='cuda')
                setattr(model,k,torch.nn.Buffer(t) if k=='_kernel_face_ids' else torch.nn.Parameter(t,requires_grad=False))
            model.N_sources=len(state['stable_ids'])
            network.inference_mode(True)
            def render_kind(kind):
                network.reset();network.prepare_kernels()
                mi_mesh,_=renderer.mesh_to_mitsuba(tri,network,
                    bsdf_additional_settings=dict(specular=0.,metallic=0.,roughness=1.))
                params=mi.traverse(mi_mesh);params['vertex_normals']=normals.astype(np.float32).ravel();params.update()
                for i,r in enumerate(views):
                    print('[RENDER]',name,kind,r['id'],f'{a.size}px {a.spp}spp',flush=True)
                    image=render_view(mi_mesh,r,a.seed+i*1000000)
                    npz_write(folder/f'{kind}_{i:03d}.npz',linear_rgb=image)
                    save_preview(folder/f'{kind}_{i:03d}_raw.png',image)
                    display=image if denoiser is None else np.asarray(denoiser(mi.TensorXf(image)))
                    save_preview(folder/f'{kind}_{i:03d}.png',display)
            render_kind('prediction')
            with ring_mode(model,state['stable_ids'],box_border,a.ring_thickness):
                render_kind('rings')
            for k in PARAMS:
                np.testing.assert_array_equal(getattr(model,k).detach().cpu().numpy(),state[k],err_msg='Offline render mutated model')
            network.reset()
        errors=[]
        for i,r in enumerate(views):
            gt=box_target(data.image(r),a.size)
            npz_write(folder/f'gt_{i:03d}.npz',linear_rgb=gt)
            save_preview(folder/f'gt_{i:03d}.png',gt)
            save_preview(folder/f'gt_{i:03d}_raw.png',gt)
            with np.load(folder/f'prediction_{i:03d}.npz') as d:pred=d['linear_rgb']
            # Conservative valid-region mask: only entirely valid source blocks.
            from PIL import Image
            mask=np.asarray(Image.open(dataset_root/r['mask']).convert('L'))>0
            factor=r['width']//a.size
            mask=mask.reshape(a.size,factor,a.size,factor).all((1,3))
            mse=float(np.mean((pred[mask]-gt[mask])**2)) if mask.any() else None
            errors.append(dict(view=r['id'],split=r['split'],valid_pixels=int(mask.sum()),
                               raw_mse=mse,scope=f'{a.size}px box-downsampled images, not original-resolution metric'))
        json_write(folder/'image_errors.json',errors)
        contact_sheet(folder,len(views),a.size);contact_sheet(folder,len(views),a.size,raw=True)
        completed.append(dict(directory=name,label=f"Update {row['completed_updates']} | {row['phase']} | {row['kernels']} kernels | {a.views}",video=False))
        json_write(out/'completed.json',completed);build_html(out,completed,len(views),not a.no_denoise)
    print('[SUCCESS]',out/'index.html',flush=True)


if __name__=='__main__':main()
