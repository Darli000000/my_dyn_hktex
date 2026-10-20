"""Create a separate surface-rendered observer viewer without modifying HKTex.

First run requires the original CUDA environment, mesh/material/texture and eigen cache.
After per-pixel data is saved, --plot-only regenerates images/HTML on CPU.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import numpy as np
from surface_render import camera, rasterize, image_from_values, centre_overlay

VERSION = '2.0.0'
PARAMS = ('_kernel_locations', '_kernel_colours', '_angles', '_anisotropies',
          '_thresholds', '_sharpnesses', '_mean_colour', '_kernel_face_ids')
PANELS = ['kernels', 'spectrum', 'distance', 'neighbours', 'colour', 'grid', 'parameters']


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def save_npz(path, **values):
    path = Path(path)
    tmp = path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **values)
    tmp.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return dict(f)


def validate_texture(surface, tm):
    """Check per-face corner geometry and all recorded UV/GT before any new prediction."""
    import trimesh
    uv = np.asarray(getattr(tm, 'original_uv', tm.visual.uv))
    faces = np.asarray(getattr(tm, 'original_faces', tm.faces))
    vertices = np.asarray(getattr(tm, 'original_vertices', tm.vertices))
    expected = surface['vertices'][surface['faces']]
    actual = vertices[faces]
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=2e-6):
        raise ValueError('Loaded mesh face/corner order or normalization differs from surface.npz')
    face_uv = uv[faces]
    check_uv = (face_uv[surface['face_ids']] * surface['bary'][..., None]).sum(1)
    if not np.allclose(check_uv, surface['uv'], rtol=0, atol=1e-7):
        raise ValueError('UV layout differs from the recorded mesh')
    image = getattr(tm.visual.material, 'baseColorTexture', None)
    if image is None:
        image = getattr(tm.visual.material, 'image', None)
    if image is None:
        raise ValueError('Texture missing; keep OBJ, MTL and its texture images together')
    gt = trimesh.visual.uv_to_color(check_uv, image)[:, :3].astype(np.float32)/255
    error = float(np.max(np.abs(gt-surface['gt'])))
    if error > 1e-6:
        raise ValueError(f'Texture does not reproduce saved GT (max difference {error})')
    return face_uv, image, error


class Replay:
    def __init__(self, a, manifest, surface):
        cache = Path(a.cache or manifest['cache_path']).resolve()
        repo = Path(a.repo).resolve()
        bad = [n for n, h in manifest['source_hashes'].items()
               if not (repo/n).is_file() or digest(repo/n) != h]
        if bad:
            raise RuntimeError('HKTex differs from recorded code: '+', '.join(bad[:12]))
        if digest(cache) != manifest['cache_sha256']:
            raise RuntimeError('Eigen cache hash differs from recording')
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', a.gpu)
        os.environ.setdefault('MPLBACKEND', 'Agg')
        sys.path.insert(0, str(repo))
        import torch
        from omegaconf import OmegaConf
        from hktex.modules.mesh import Mesh
        from hktex.modules.eigen_albo_knn import EigenAlboInterpolationKNN
        from hktex.modules.heat_kernel_texture_knn import HeatKernelTextureKNN
        from hktex.utils import load_mesh
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for new pixel predictions; use --plot-only for saved pixels')
        self.torch = torch
        torch.set_float32_matmul_precision(manifest['matmul_precision'])
        torch.backends.cuda.matmul.allow_tf32 = manifest['allow_tf32']
        self.mesh = Mesh(surface['vertices'], surface['faces'], surface['fnorms'], surface['vnorms'], device='cuda')
        mc = OmegaConf.create(manifest['model_config'])
        if mc.out_net or mc.normalize_colours or mc.mass_type != 'one':
            raise RuntimeError('This patch targets the recorded UV/KNN configuration')
        mc.n_sources = 1
        self.model = HeatKernelTextureKNN(mc, self.mesh)
        dc = manifest['config']['data']
        tm = load_mesh(a.mesh or dc['mesh_path'], merge_tex=dc.get('merge_tex', True),
                       bake_vert_colors=dc.get('bake_vert_colours_if_textured', False))
        self.face_uv, self.texture, self.gt_check = validate_texture(surface, tm)
        print('[CHECK] Mesh face order, UV and all saved GT colours match', flush=True)
        ec = OmegaConf.create(manifest['eigen_config'])
        with tempfile.TemporaryDirectory(prefix='hktex-overview-') as td:
            ec.mesh_path = str(Path(td)/'mesh.obj')
            ec.precomputed_name = 'overview'
            ec.use_precomputed = True
            ec.error_if_not_precomputed = True
            (Path(td)/f'mesh_overview_{ec.local_frames}.pt').symlink_to(cache)
            self.eig = EigenAlboInterpolationKNN(ec, self.mesh)
        self.model.eval()

    def gt(self, face_ids, bary):
        import trimesh
        uv = (self.face_uv[face_ids] * bary[..., None]).sum(1)
        return trimesh.visual.uv_to_color(uv, self.texture)[:, :3].astype(np.float32)/255

    def restore(self, state):
        torch = self.torch
        with torch.no_grad():
            for key in PARAMS:
                tensor = torch.as_tensor(state[key], device='cuda')
                setattr(self.model, key, torch.nn.Buffer(tensor) if key == '_kernel_face_ids'
                        else torch.nn.Parameter(tensor, requires_grad=False))
            self.model.N_sources = len(state['stable_ids'])
            self.eig.reset()
            self.model.prepare_kernels(self.mesh, self.eig, save_barycentric=False)

    def predict(self, view, chunk):
        torch = self.torch
        n = len(view['face_ids'])
        output = np.empty((n, 3), dtype=np.float32)
        t = time.monotonic()
        with torch.no_grad():
            for start in range(0, n, chunk):
                stop = min(start+chunk, n)
                f = torch.as_tensor(view['face_ids'][start:stop], device='cuda', dtype=torch.long)
                b = torch.as_tensor(view['bary'][start:stop], device='cuda', dtype=torch.float32)
                pts = self.model.prepare_points(self.mesh, self.eig, f, barys=b)
                rgb, _, _, _ = self.model.diffuse_heat_kernels(self.eig, pts)
                rgb = self.model(rgb)
                output[start:stop] = rgb.detach().cpu().numpy()
                if not np.isfinite(output[start:stop]).all():
                    raise RuntimeError(f'Nonfinite pixel prediction at {start}:{stop}')
                if time.monotonic()-t > 15 or stop == n:
                    print(f'[PIXELS] {stop}/{n}', flush=True)
                    t = time.monotonic()
        return output


def plot_overview(path, title, views, gt, pred, centres, cam, error_max):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    fig, axes = plt.subplots(2, 4, figsize=(12, 9), layout='constrained')
    norm = Normalize(0, error_max, clip=True)
    cmap = plt.get_cmap('magma')
    rows = []
    for row, (view, g, p, sign) in enumerate(zip(views, gt, pred, [1, -1])):
        error = np.mean((p-g)**2, axis=1)
        clipped = float(np.mean(error > error_max))
        imgs = [image_from_values(view['mask'], g), image_from_values(view['mask'], np.clip(p, 0, 1)),
                image_from_values(view['mask'], cmap(norm(error))[:, :3]),
                centre_overlay(view, centres, cam, sign)]
        for ax, img in zip(axes[row], imgs):
            ax.imshow(img, interpolation='nearest')
            ax.set_axis_off()
        titles = ['GT texture', 'HKTex prediction', f'RGB MSE | saturated {clipped:.1%}',
                  'Kernel centres (depth tested)']
        for ax, caption in zip(axes[row], titles):
            ax.set_title(caption, fontsize=10)
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes[row, 2], shrink=.65, extend='max')
        rows.append(dict(view='+Z' if sign == 1 else '-Z', visible_pixels=int(len(error)),
                         visible_pixel_mse=float(error.mean()), max_pixel_mse=float(error.max()),
                         fraction_above_colour_scale=clipped))
    fig.suptitle(title+'\nContinuous mesh surface | +Z / -Z | original forward | unlit RGB', fontsize=12)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return rows


def build_viewer(out, old, rows):
    from PIL import Image
    gallery = []
    gif = []
    for row in rows:
        stem = Path(row['file']).stem
        overview = out/(stem+'_overview.png')
        if not overview.exists():
            continue
        images = dict(overview=overview.name)
        for key in PANELS:
            src = old/(stem+'_'+key+'.png')
            if src.is_file():
                shutil.copy2(src, out/src.name)
                images[key] = src.name
        label = f"Updates {row['completed_updates']} | {row['phase']} | {row['kernels']} kernels"
        gallery.append(dict(label=label, images=images))
        if row['phase'] in ['initial', 'post_update', 'final']:
            with Image.open(overview) as im:
                frame = im.convert('RGB')
                frame.thumbnail((900, 700))
                gif.append(frame.copy())
    if gif:
        gif[0].save(out/'training.gif', save_all=True, append_images=gif[1:], duration=600, loop=0)
    extras = ''
    for name, title in [('curves.png', '原固定点评估与训练曲线'), ('lineage.png', '原热核家族记录')]:
        if (old/name).is_file():
            shutil.copy2(old/name, out/name)
            extras += f'<h2>{title}</h2><img src="{name}">'
    if gif:
        extras += '<h2>已渲染快照的演变</h2><img src="training.gif">'
    # Preserve original interactive event/decision sections as an explicitly labelled reference page.
    if (old/'index.html').is_file():
        # The original reference page requires old overview files too: keep in a separate directory.
        ref = out/'original-reference'
        ref.mkdir(exist_ok=True)
        for src in old.iterdir():
            if src.is_file() and src.suffix in ['.png', '.gif', '.html']:
                shutil.copy2(src, ref/src.name)
        extras += '<p><a href="original-reference/index.html">完整旧 viewer（含原 ADC 事件、决策与旧散点图，仅作对照）</a></p>'
    payload = json.dumps(gallery, ensure_ascii=False).replace('<', '\\u003c')
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>HKTex surface overview v2</title>
<style>body{font:16px system-ui;max-width:1350px;margin:24px auto;padding:0 18px;background:#fafafa;color:#222}p{line-height:1.6}img{max-width:100%;background:white}select{padding:8px;margin:8px}</style>
<h1>HKTex 连续表面 overview</h1>
<p>总览逐像素计算可见网格表面的 GT 与模型预测，采用相同的正交相机、深度遮挡与无光照 RGB；没有从 8192 个散点插值纹理。它不是原 Mitsuba 的光照预览，因此亮度与视角可能不同。蓝点只是核中心，不表示核半径，中心遮挡按像素深度近似判断。</p>
<p>本补丁保留原 FAISS/Top-K 行为，尚未修复并列候选的不确定性。颜色结果保存在 data/*.npz；再次绘图使用这些数值。误差图固定色标，超过上限的比例标在标题中。像素 MSE 与原均匀表面点 MSE 采用不同采样，不能直接替代。</p>
<p>其他数学面板沿用旧分析结果，其中热核场、距离场与邻居示意仍是散点投影，并未变成连续表面图。动画横轴是训练迭代，不是物理扩散时间。</p>
<select id="frame"></select><select id="mode"></select><p id="label"></p><img id="view">EXTRAS
<script>const data=PAYLOAD;const f=document.getElementById('frame'),m=document.getElementById('mode');const labels={overview:'连续表面总览',kernels:'原热核逐阶段响应（散点）',spectrum:'谱项与累加',distance:'双调和距离（含散点图）',neighbours:'邻居示意（散点）',colour:'颜色贡献',grid:'参数网格插值',parameters:'参数分布'};data.forEach((d,i)=>f.add(new Option(d.label,i)));function show(){const d=data[f.value];document.getElementById('label').textContent=d.label;document.getElementById('view').src=d.images[m.value];}function change(){m.innerHTML='';Object.keys(data[f.value].images).forEach(k=>m.add(new Option(labels[k]||k,k)));show();}f.onchange=change;m.onchange=show;if(data.length)change();</script></html>'''
    (out/'index.html').write_text(page.replace('EXTRAS', extras).replace('PAYLOAD', payload))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', required=True)
    p.add_argument('--repo', default='/root/hktex-work/hktex')
    p.add_argument('--analysis', help='Default: recording/analysis')
    p.add_argument('--viewer', help='Existing viewer for mathematical panels; default recording/viewer')
    p.add_argument('--output', help='Default: recording/viewer-surface-v2')
    p.add_argument('--snapshots', default='final', help='final, all (all entries in metrics.json), or serials e.g. 0,81,82')
    p.add_argument('--height', type=int, default=384)
    p.add_argument('--chunk', type=int, default=64)
    p.add_argument('--gpu', default='0')
    p.add_argument('--mesh', help='Moved original textured mesh; all saved GT must still match')
    p.add_argument('--cache', help='Moved original eigen cache; SHA256 must match')
    p.add_argument('--error-max', type=float, default=.02, help='Fixed display scale, not an error clamp on data')
    p.add_argument('--plot-only', action='store_true', help='CPU only; requires already-saved pixel data')
    a = p.parse_args()
    if a.height < 64 or a.chunk < 1 or not a.error_max > 0:
        p.error('Require height >= 64, chunk >= 1, error-max > 0')
    root = Path(a.recording).resolve()
    analysis = Path(a.analysis).resolve() if a.analysis else root/'analysis'
    old = Path(a.viewer).resolve() if a.viewer else root/'viewer'
    out = Path(a.output).resolve() if a.output else root/'viewer-surface-v2'
    if out == old or out == root or out == analysis or out in old.parents or old in out.parents:
        p.error('Output must be separate from recording/analysis and the old viewer')
    manifest = read_json(root/'manifest.json')
    rows = read_json(analysis/'metrics.json')
    if a.snapshots == 'all':
        selected = rows
    elif a.snapshots == 'final':
        final = [r for r in rows if r['phase'] == 'final']
        if not final:
            p.error('No final entry in metrics.json; use explicit snapshot serials')
        selected = final[-1:]
    else:
        wanted = {int(s) for s in a.snapshots.split(',')}
        selected = [r for r in rows if int(r['file'].split('_')[0]) in wanted]
        if len(selected) != len(wanted):
            p.error('Requested serial absent from analysis/metrics.json')
    if not selected:
        p.error('No snapshots selected')
    if not (old/'index.html').is_file():
        print('[NOTE] Old viewer not found; new overview page will still work. Use --viewer to retain old panels.', flush=True)
    surface = load_npz(root/'surface.npz')
    config = dict(version=VERSION, recording=str(root), manifest_sha256=digest(root/'manifest.json'),
                  surface_sha256=digest(root/'surface.npz'), camera=camera(surface['vertices'], a.height),
                  forward='original HKTex / no tie-rule modification', chunk=a.chunk)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'render_config.json').exists() and read_json(out/'render_config.json') != config:
        p.error('Output already contains a different render configuration. Choose a new --output directory.')
    write_json(out/'render_config.json', config)
    (out/'data').mkdir(exist_ok=True)
    cam = config['camera']
    print('[RASTER] Building visible surface points for +Z / -Z', flush=True)
    views = [rasterize(surface['vertices'], surface['faces'], cam, s) for s in [1, -1]]
    geo_path = out/'data'/'geometry.npz'
    save_npz(geo_path, **{f'{k}_{i}': v[k] for i, v in enumerate(views)
                         for k in ['face_map', 'bary_map', 'depth', 'face_ids', 'bary', 'mask']})
    replay = None
    gt = None
    report = read_json(out/'render_report.json') if (out/'render_report.json').exists() else {}
    for row in selected:
        name = Path(row['file']).stem
        state_path = root/'snapshots'/row['file']
        state = load_npz(state_path)
        state_hash = digest(state_path)
        data_path = out/'data'/(name+'_pixels.npz')
        if data_path.exists():
            d = load_npz(data_path)
            if str(d['snapshot_sha256'].item()) != state_hash:
                raise RuntimeError('Snapshot changed since pixel evaluation: '+name)
            gts, predictions = [d['gt_0'], d['gt_1']], [d['predicted_0'], d['predicted_1']]
            for i in range(2):
                if gts[i].shape != (len(views[i]['face_ids']), 3) or predictions[i].shape != gts[i].shape:
                    raise RuntimeError('Pixel data shape mismatch: '+name)
                if not np.isfinite(gts[i]).all() or not np.isfinite(predictions[i]).all():
                    raise RuntimeError('Nonfinite cached pixel data: '+name)
            print('[CACHE]', name, flush=True)
        else:
            if a.plot_only:
                raise RuntimeError('Pixel data missing; run CUDA replay first: '+str(data_path))
            if replay is None:
                replay = Replay(a, manifest, surface)
                gt = [replay.gt(v['face_ids'], v['bary']) for v in views]
            print('[REPLAY]', name, flush=True)
            replay.restore(state)
            predictions = [replay.predict(v, a.chunk) for v in views]
            gts = gt
            save_npz(data_path, gt_0=gts[0], gt_1=gts[1], predicted_0=predictions[0], predicted_1=predictions[1],
                     snapshot_sha256=np.asarray(state_hash))
        title = f"Updates {row['completed_updates']} | {row['phase']} | {row['kernels']} kernels"
        report[name] = plot_overview(out/(name+'_overview.png'), title, views, gts, predictions,
                                    state['_kernel_locations'], cam, a.error_max)
        write_json(out/'render_report.json', report)
        print('[PLOT]', name, flush=True)
    build_viewer(out, old, rows)
    print('[DONE] Open:', out/'index.html', flush=True)


if __name__ == '__main__':
    main()
