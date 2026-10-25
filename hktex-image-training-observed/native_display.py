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
from observer_common import PARAMS

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
<p>图片训练适配：读取保存的线性 RGB 作为 GT，使用训练网络、Mitsuba 路径追踪与数据集相机。默认展示 OptiX 去噪结果，也可切换同一次渲染的未去噪图。去噪可能抹去真实细节，展示图不能代替未去噪的图片误差评估。</p>
<p>Kernel rings 是阈值附近的热核响应带，不是核中心、完整热核数值或物理半径。采用作者 box_border 方法，颜色按稳定 ID 固定，便于跨快照跟踪；保留原 KNN/Top-K 行为。</p>
<select id="snapshot"></select><select id="view"></select><select id="style"><option value="">展示图</option><option value="_raw">未去噪对照</option></select><p id="label"></p>
<div class="grid"><div><h3>GT</h3><img id="gt"></div><div><h3>HKTex output</h3><img id="prediction"></div><div><h3>Kernel rings</h3><img id="rings"></div></div>
<p><a id="sheet">多视图总览原图</a> · <a id="sheetraw">未去噪总览原图</a></p><video id="video" controls></video>
<script>const data=PAYLOAD;const s=document.getElementById('snapshot'),v=document.getElementById('view'),st=document.getElementById('style');data.forEach((d,i)=>s.add(new Option(d.label,i)));for(let i=0;i<FRAMES;i++)v.add(new Option('view '+i,i));function show(){const d=data[s.value],i=String(v.value).padStart(3,'0');document.getElementById('label').textContent=d.label;for(const k of ['gt','prediction','rings'])document.getElementById(k).src=d.directory+'/'+k+'_'+i+st.value+'.png';document.getElementById('sheet').href=d.directory+'/preview_views.png';document.getElementById('sheetraw').href=d.directory+'/preview_views_raw.png';const video=document.getElementById('video');video.style.display=d.video?'block':'none';if(d.video)video.src=d.directory+'/comparison.mp4';}s.onchange=v.onchange=st.onchange=show;if(data.length)show();</script></html>'''
    if not denoise:
        text = text.replace('默认展示 OptiX 去噪结果', '本次未启用 OptiX，展示图与未去噪图相同')
    (out/'index.html').write_text(text.replace('PAYLOAD', payload).replace('FRAMES', str(frames)))


