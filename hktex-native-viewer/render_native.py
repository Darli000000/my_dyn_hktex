"""Offline snapshot rendering through HKTex's own Mitsuba renderers.

Retains raw linear renders, raw sRGB PNGs, denoised PNGs, true threshold-band
kernel rings, a contact sheet and a portable HTML gallery. No training edits.
"""
import argparse
from contextlib import contextmanager
from functools import partial
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw
from fix_overviews import Replay, read_json, write_json, load_npz, save_npz, digest, PARAMS

VERSION = '1.0.0'
KINDS = ['gt', 'prediction', 'rings']


def palette(stable_ids, seed=0):
    """Stable ID -> pseudorandom RGB; same identity keeps its colour across snapshots."""
    return np.asarray([list(hashlib.blake2s(f'{seed}:{int(i)}'.encode(), digest_size=3).digest())
                       for i in stable_ids], dtype=np.float32)/255.


@contextmanager
def ring_mode(model, stable_ids, box_border, thickness=.03):
    """Author's threshold-band method, on a separate offline model; exact object restore."""
    import torch
    original = (model.kernel_filter_func, model._kernel_colours, model._mean_colour)
    try:
        model.kernel_filter_func = partial(box_border, thickness=thickness)
        rgb = torch.as_tensor(palette(stable_ids), device=model._kernel_colours.device,
                              dtype=model._kernel_colours.dtype)
        model._kernel_colours = torch.nn.Parameter(rgb, requires_grad=False)
        model._mean_colour = torch.nn.Parameter(torch.zeros_like(model._mean_colour), requires_grad=False)
        yield
    finally:
        model.kernel_filter_func, model._kernel_colours, model._mean_colour = original


def contact_sheet(folder, frame_count, size, raw=False):
    indices = sorted(set([0, frame_count//4, frame_count//2, 3*frame_count//4]))
    sheet = Image.new('RGB', (size*3, (size+28)*len(indices)), 'white')
    draw = ImageDraw.Draw(sheet)
    suffix = '_raw' if raw else ''
    for row, i in enumerate(indices):
        for col, (key, label) in enumerate(zip(KINDS, ['GT', 'HKTex output', 'Kernel rings'])):
            with Image.open(folder/f'{key}_{i:03d}{suffix}.png') as im:
                sheet.paste(im, (col*size, row*(size+28)+28))
            draw.text((col*size+6, row*(size+28)+6), f'{label} | view {i}', fill='black')
    sheet.save(folder/('preview_views_raw.png' if raw else 'preview_views.png'))


def build_html(out, completed, frames, denoise):
    payload = json.dumps(completed, ensure_ascii=False).replace('<', '\\u003c')
    text = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>HKTex native renderer viewer</title>
<style>body{font:16px system-ui;max-width:1600px;margin:24px auto;padding:0 18px;background:#fafafa}p{line-height:1.6}select{padding:8px;margin:5px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}img{max-width:100%}h3{font-size:16px}video{max-width:100%}</style>
<h1>HKTex 原渲染器回放</h1>
<p>直接使用项目的 UVTextureRenderer / HeatKernelsRendererKNN、Mitsuba 路径追踪和 sRGB 输出。默认展示 OptiX 去噪结果，也可切换同一次渲染的未去噪图。去噪可能抹去真实细节，展示图不能代替原始表面误差评估。</p>
<p>Kernel rings 是阈值附近的热核响应带，不是核中心、完整热核数值或物理半径。采用作者 box_border 方法，颜色按稳定 ID 固定，便于跨快照跟踪；保留原 KNN/Top-K 行为。</p>
<select id="snapshot"></select><select id="view"></select><select id="style"><option value="">展示图</option><option value="_raw">未去噪对照</option></select><p id="label"></p>
<div class="grid"><div><h3>GT</h3><img id="gt"></div><div><h3>HKTex output</h3><img id="prediction"></div><div><h3>Kernel rings</h3><img id="rings"></div></div>
<p><a id="sheet">多视图总览原图</a> · <a id="sheetraw">未去噪总览原图</a></p><video id="video" controls></video>
<script>const data=PAYLOAD;const s=document.getElementById('snapshot'),v=document.getElementById('view'),st=document.getElementById('style');data.forEach((d,i)=>s.add(new Option(d.label,i)));for(let i=0;i<FRAMES;i++)v.add(new Option('view '+i,i));function show(){const d=data[s.value],i=String(v.value).padStart(3,'0');document.getElementById('label').textContent=d.label;for(const k of ['gt','prediction','rings'])document.getElementById(k).src=d.directory+'/'+k+'_'+i+st.value+'.png';document.getElementById('sheet').href=d.directory+'/preview_views.png';document.getElementById('sheetraw').href=d.directory+'/preview_views_raw.png';const video=document.getElementById('video');video.style.display=d.video?'block':'none';if(d.video)video.src=d.directory+'/comparison.mp4';}s.onchange=v.onchange=st.onchange=show;if(data.length)show();</script></html>'''
    if not denoise:
        text = text.replace('默认展示 OptiX 去噪结果', '本次未启用 OptiX，展示图与未去噪图相同')
    (out/'index.html').write_text(text.replace('PAYLOAD', payload).replace('FRAMES', str(frames)))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', required=True)
    p.add_argument('--repo', default='/root/hktex-work/hktex')
    p.add_argument('--snapshots', default='final', help='final, all regular snapshots, or serials e.g. 0,81,82')
    p.add_argument('--size', type=int, default=512, help='Width and height of each native frame')
    p.add_argument('--spp', type=int, default=16, help='Mitsuba samples per pixel')
    p.add_argument('--frames', type=int, default=4, help='Views evenly spaced over one full rotation')
    p.add_argument('--point-batch', type=int, default=512, help='HKTex shader query chunk, not image resolution')
    p.add_argument('--tile-size', type=int, default=0, help='0 = no tiling; e.g. 128 reduces renderer peak memory')
    p.add_argument('--no-denoise', action='store_true')
    p.add_argument('--ring-thickness', type=float, default=.03)
    p.add_argument('--video', action='store_true')
    p.add_argument('--fps', type=int, default=10)
    p.add_argument('--output', help='Default: observer/viewer-native')
    p.add_argument('--gpu', default='0')
    p.add_argument('--mesh')
    p.add_argument('--cache')
    a = p.parse_args()
    if min(a.size, a.spp, a.frames, a.point_batch, a.fps) < 1 or a.tile_size < 0 or not 0 < a.ring_thickness <= 1:
        p.error('Invalid resolution/sampling/frame/batch/thickness setting')
    return a


def main():
    a = parse_args()
    root = Path(a.recording).resolve()
    out = Path(a.output).resolve() if a.output else root/'viewer-native'
    protected = [root, root/'snapshots', root/'analysis', root/'viewer', root/'viewer-surface-v2']
    if any(out == p or out in p.parents or (p != root and p in out.parents) for p in protected):
        raise ValueError('Choose a separate output directory; do not overwrite recording or old viewers')
    manifest = read_json(root/'manifest.json')
    index = read_json(root/'snapshots.json')
    if a.snapshots == 'final':
        selected = [s for s in index if s['phase'] == 'final'][-1:]
    elif a.snapshots == 'all':
        selected = [s for s in index if s['phase'] in ['initial', 'post_update', 'final']]
    else:
        serials = {int(x) for x in a.snapshots.split(',')}
        selected = [s for s in index if int(s['file'].split('_')[0]) in serials]
        if len(selected) != len(serials):
            raise ValueError('Unknown snapshot serial')
    if not selected:
        raise ValueError('No snapshots selected')
    settings = dict(version=VERSION, manifest_sha256=digest(root/'manifest.json'),
                    surface_sha256=digest(root/'surface.npz'), size=a.size, spp=a.spp,
                    frames=a.frames, point_batch=a.point_batch, tile_size=a.tile_size,
                    denoise=not a.no_denoise, ring_thickness=a.ring_thickness, fps=a.fps,
                    palette='blake2s(stable_ID), seed 0', forward='original HKTex KNN / no tie-rule changes')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'settings.json').exists() and read_json(out/'settings.json') != settings:
        raise ValueError('Different settings in output directory. Choose a new --output.')
    write_json(out/'settings.json', settings)
    print('[LOAD] Checking source/cache, restoring mesh and model', flush=True)
    surface = load_npz(root/'surface.npz')
    replay = Replay(a, manifest, surface)
    import torch
    import mitsuba as mi
    import trimesh
    from omegaconf import OmegaConf
    from hktex.rendering.uv_texture_renderer import UVTextureRenderer
    from hktex.rendering.heat_kernels_renderer_knn import HeatKernelsRendererKNN
    from hktex.utils import load_mesh, box_border
    # Author's GT path reloads without UV merging when the training mesh merged UV seams.
    dc = manifest['config']['data']
    gt_mesh = load_mesh(a.mesh or dc['mesh_path'], merge_tex=False, bake_vert_colors=False)
    tri_mesh = trimesh.Trimesh(vertices=surface['vertices'], faces=surface['faces'],
                              vertex_normals=surface['vnorms'], process=False)
    cfg = OmegaConf.create(manifest['config']['renderer'])
    cfg.camera_config.img_width = cfg.camera_config.img_height = a.size
    cfg.camera_config.sample_count = a.spp
    cfg.camera_config.tile_size = a.tile_size or None
    cfg.camera_config.tile_size_heatkernels = a.tile_size or None
    cfg.point_batching = a.point_batch
    cfg.n_rotating_frames = a.frames
    write_json(out/'renderer_config.json', OmegaConf.to_container(cfg, resolve=True))
    base_angle = float(cfg.camera_config.azimuth_deg)
    elevation = float(cfg.camera_config.elevation_deg)
    # Same flags as BaseTrainer.render_result.
    HeatKernelsRendererKNN.mega_kernel(False, no_loops=True, no_opt_calls=True)
    denoiser = None if a.no_denoise else mi.OptixDenoiser(input_size=(a.size, a.size))
    renderer_gt = UVTextureRenderer(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)))
    renderer_hk = HeatKernelsRendererKNN(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)))
    mi_gt = renderer_gt.mesh_to_mitsuba(gt_mesh)
    completed = read_json(out/'completed.json') if (out/'completed.json').exists() else []
    (out/'gt-cache').mkdir(exist_ok=True)

    def render_frame(renderer, mi_mesh, i, destination):
        """Cache raw linear RGB so raw/denoised images share exactly one render."""
        raw_path = destination.with_suffix('.npz')
        if raw_path.exists():
            linear = load_npz(raw_path)['linear_rgb']
            if linear.shape != (a.size, a.size, 3) or not np.isfinite(linear).all():
                raise RuntimeError('Invalid cached linear render: '+str(raw_path))
            raw = mi.TensorXf(linear)
        else:
            renderer.change_camera_param(azimuth_deg=base_angle + 360*i/a.frames, elevation_deg=elevation)
            raw = renderer.render(mi_mesh, denoise=False)
            linear = np.asarray(mi.Bitmap(raw), dtype=np.float32)
            if not np.isfinite(linear).all():
                raise RuntimeError('Nonfinite Mitsuba render')
            save_npz(raw_path, linear_rgb=linear)
        def save_srgb(value, path):
            mi.Bitmap(value).convert(pixel_format=mi.Bitmap.PixelFormat.RGB,
                component_format=mi.Struct.Type.UInt8, srgb_gamma=True).write(str(path))
        save_srgb(raw, destination.with_name(destination.stem+'_raw.png'))
        display = raw if denoiser is None else denoiser(raw)
        save_srgb(display, destination)

    import shutil
    for item in selected:
        stem = Path(item['file']).stem
        folder = out/stem
        folder.mkdir(exist_ok=True)
        state_path = root/'snapshots'/item['file']
        state = load_npz(state_path)
        fingerprint = dict(snapshot_sha256=digest(state_path))
        if (folder/'snapshot.json').exists() and read_json(folder/'snapshot.json') != fingerprint:
            raise RuntimeError('Snapshot changed since cached rendering: '+stem)
        write_json(folder/'snapshot.json', fingerprint)
        replay.restore(state)
        mi_hk = renderer_hk.mesh_to_mitsuba(tri_mesh, replay.mesh, replay.model, replay.eig)
        label = f"Updates {item['completed_updates']} | {item['phase']} | {len(state['stable_ids'])} kernels"
        start = time.monotonic()
        with torch.no_grad():
            for i in range(a.frames):
                print(f'[GT] {stem} view {i+1}/{a.frames}', flush=True)
                cache_dest = out/'gt-cache'/f'gt_{i:03d}.png'
                render_frame(renderer_gt, mi_gt, i, cache_dest)
                for suffix in ['.png', '_raw.png']:
                    shutil.copy2(out/'gt-cache'/f'gt_{i:03d}{suffix}', folder/f'gt_{i:03d}{suffix}')
                print(f'[PREDICTION] {stem} view {i+1}/{a.frames}', flush=True)
                render_frame(renderer_hk, mi_hk, i, folder/f'prediction_{i:03d}.png')
            with ring_mode(replay.model, state['stable_ids'], box_border, a.ring_thickness):
                mi_rings = renderer_hk.mesh_to_mitsuba(tri_mesh, replay.mesh, replay.model, replay.eig)
                for i in range(a.frames):
                    print(f'[RINGS] {stem} view {i+1}/{a.frames}', flush=True)
                    render_frame(renderer_hk, mi_rings, i, folder/f'rings_{i:03d}.png')
        for key in PARAMS:
            np.testing.assert_array_equal(getattr(replay.model, key).detach().cpu().numpy(), state[key],
                                          err_msg='Offline render changed parameter '+key)
        contact_sheet(folder, a.frames, a.size)
        contact_sheet(folder, a.frames, a.size, raw=True)
        video_error = None
        if a.video:
            try:
                from hktex.utils.video import save_video
                frames = []
                for i in range(a.frames):
                    arrays = []
                    for k in KINDS:
                        with Image.open(folder/f'{k}_{i:03d}.png') as image:
                            arrays.append(np.asarray(image.convert('RGB')).copy())
                    frames.append(mi.Bitmap(np.concatenate(arrays, axis=1)))
                save_video(frames, str(folder/'comparison.mp4'), fps=a.fps)
            except Exception as exc:
                video_error = f'{type(exc).__name__}: {exc}'
                print('[VIDEO] Images saved; video export failed:', video_error, flush=True)
        row = dict(directory=stem, label=label, video=(folder/'comparison.mp4').is_file() and video_error is None,
                   seconds=time.monotonic()-start, snapshot_unchanged=True, video_error=video_error)
        completed = [r for r in completed if r['directory'] != stem] + [row]
        completed.sort(key=lambda r: r['directory'])
        write_json(out/'completed.json', completed)
        build_html(out, completed, a.frames, not a.no_denoise)
        print('[SAVED]', folder/'preview_views.png', flush=True)
        replay.eig.reset()
    print('[DONE] Open:', out/'index.html', flush=True)


if __name__ == '__main__':
    main()
