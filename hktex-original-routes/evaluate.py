"""Common calibrated native-render evaluation, without changing trained parameters."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from protocol import setup_paths, verify_sources, write_json, digest, decode_reflectance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--views', default='view_000,view_002,view_005', help='dataset IDs, or all')
    p.add_argument('--size', type=int, default=1024)
    p.add_argument('--spp', type=int, default=128)
    p.add_argument('--tile-size', type=int, default=32)
    p.add_argument('--gpu', default='0')
    a = p.parse_args()
    repo = setup_paths(a.repo)
    verify_sources(repo)
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    os.environ.setdefault('MPLBACKEND','Agg')
    out = Path(a.output).resolve()
    if out.exists(): p.error('--output must be new')
    if min(a.size,a.spp,a.tile_size) < 1: p.error('Invalid render settings')
    import torch
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')
    from hktex.modules.heat_kernel_texture import HeatKernelTexture
    import render_native_images as native
    from offline_common import read_json
    run = Path(a.run).resolve()
    state = torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=True)
    settings = state['settings']
    encoded = settings.get('reflectance_encoding') == 'srgb-code-values'
    ec = settings['trainer_config']['network']['eigen_albo']
    cache = Path(str(ec['mesh_path']).rsplit('.',1)[0]+'_'+ec.get('precomputed_name','eigen_albo')+'_'+ec.get('local_frames','principal_curvatures')+'.pt')
    expected = settings.get('cache_sha256')
    if expected and digest(cache) != expected:
        raise ValueError('Trained eigen cache changed')
    manifest = read_json(Path(a.dataset)/'manifest.json')
    views = ','.join(r['id'] for r in manifest['views']) if a.views=='all' else a.views
    opts = SimpleNamespace(repo=str(repo),run=str(run),recording=None,dataset=a.dataset,cache=str(cache),
        output=str(out),gpu=a.gpu,snapshots='final',views=views,size=a.size,spp=a.spp,
        tile_size=a.tile_size,point_batch=512,seed=21931,no_denoise=True,ring_thickness=.03)
    original_forward = HeatKernelTexture.forward
    def field_to_linear(self,*args,**kwargs):
        rgb = original_forward(self,*args,**kwargs)
        if encoded:
            # Interpret the author's fitted RGB/255 as texture code values.
            # Decode REFLECTANCE before BSDF/path integration, never rendered radiance.
            rgb = decode_reflectance(rgb)
        return rgb
    out.mkdir(parents=True)
    contract = dict(version=1,run=str(run),checkpoint_sha256=digest(run/'checkpoint.pt'),
        dataset_manifest_sha256=digest(Path(a.dataset)/'manifest.json'),
        views=views.split(','),size=a.size,spp=a.spp,tile_size=a.tile_size,seed=21931,
        point_batch=512,precision='highest',allow_tf32=False,denoise=False,
        gt='saved linear-radiance images; box average at requested resolution',
        metric='RGB mean squared error on common conservative foreground mask',
        renderer='same calibrated cameras/shared normals/principled specular0 metallic0 roughness1/environment1/path max_depth3',
        field_adapter='clip code-value reflectance to [0,1], then sRGB decode before rendering' if encoded else 'identity (already linear reflectance)',
        scope='all saved views' if a.views=='all' else 'selected views only; not the full held-out test set',
        interpretation='quality under a common evaluation scene; original UV/online routes had privileged source access')
    write_json(out/'common_protocol.json',contract)
    try:
        # native.render sets highest/TF32 off for checkpoint replay for every route.
        with patch.object(HeatKernelTexture,'forward',field_to_linear):
            native.render(opts,out)
    except BaseException as exc:
        import traceback
        write_json(out/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__': main()
