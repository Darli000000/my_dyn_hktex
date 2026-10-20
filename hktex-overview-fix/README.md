# HKTex overview 连续表面补丁 v2

无需重新训练，不修改 HKTex 源码或旧 observer 工具；结果写入独立目录。

## 修复内容

- 旧 `plot_recording.py` 的 overview 用 `scatter(s=2)` 绘制 8192 个表面采样点。圆点不是热核半径，点之间的空隙不是网格孔洞。
- 新脚本光栅化原三角网格，按像素深度确定可见面及重心坐标，逐像素调用原 `prepare_points → diffuse_heat_kernels → model`，直接计算该处预测颜色。
- GT 使用该像素的面内 UV，从原纹理采样。读取模型后先验证面序、角点顺序、归一化坐标，以及所有原始评估点的 UV 和 GT；不匹配则停止。
- 正反面采用共享正交相机与相同遮挡规则；显示无光照 RGB，未执行 PBR 光照或额外 gamma 变换。因此不会逐像素等同于原 Mitsuba preview。
- 第四列蓝色标记仅表示核中心，使用像素深度近似遮挡，不表示热核影响范围。
- 误差是新可见像素上的 RGB MSE；默认色标上限 0.02，超限比例写在图中。保存的误差/颜色不因色标被截断。与原均匀表面采样 MSE 不是同一个统计量。

本补丁只修 overview。原 FAISS/Top-K 并列选择行为不变，重新计算像素时仍可能有已确认的重复性差异。其他数学面板沿用原分析，其中表面场仍是明确标注的散点图。

## 1. 放到服务器

将 `hktex-overview-fix.zip` 上传到 `/root/hktex-work/hktex/`：

```bash
conda activate hktex
cd /root/hktex-work/hktex
python -m zipfile -e hktex-overview-fix.zip .

export HKTEX_OBSERVER='/root/hktex-work/hktex/outputs/smpl-observed-full/male_init05_lr10_20261019-044000-c77569@20261019-044005/observer'
```

保留原 OBJ、MTL、纹理图片、特征分解缓存，以及完整 observer 目录。脚本默认读取 `observer/analysis/metrics.json` 与 `observer/viewer/`。

若服务器上的旧 viewer 在别处，请在后续命令添加 `--viewer /实际旧viewer目录`。旧 viewer 找不到时仍可生成 overview，但无法带上旧数学面板。若 analysis 目录移动，添加 `--analysis /实际analysis目录`。

## 2. 先只生成 final

```bash
env MPLBACKEND=Agg \
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
  python -u hktex-overview-fix/fix_overviews.py \
  --recording "$HKTEX_OBSERVER" \
  --repo /root/hktex-work/hktex \
  --snapshots final \
  --height 384 \
  --chunk 64
```

生成：

```text
observer/viewer-surface-v2/
├── index.html
├── 00082_u05000_final_overview.png
├── training.gif
├── render_config.json
├── render_report.json
├── data/
│   ├── geometry.npz
│   └── 00082_u05000_final_pixels.npz
├── …沿用的数学面板、曲线及家族图
└── original-reference/     # 完整旧 viewer，保留 ADC 事件/决策页面供对照
```

直接查看 `00082_u05000_final_overview.png`。要在本地打开 HTML，请下载整个 `viewer-surface-v2` 文件夹，不能只下载 `index.html`。

384 高度下，本次 SMPL 两个视图合计约 42858 个有效像素，比原 8192 点回放计算量大。首次会有编译时间。脚本每约 15 秒打印像素进度。

## 3. 生成所有已分析快照的 overview、动画和新 viewer

```bash
env MPLBACKEND=Agg \
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
  python -u hktex-overview-fix/fix_overviews.py \
  --recording "$HKTEX_OBSERVER" \
  --repo /root/hktex-work/hktex \
  --snapshots all \
  --height 384 \
  --chunk 64
```

`all` 指现有 `analysis/metrics.json` 中全部快照，保持与原 viewer 的分析范围一致，不自动增加原来没有分析的 ADC 快照。也可以指定 `--snapshots 0,81,82`。

已保存的像素直接复用，所以第二步生成的 final 不会再计算。中断后用同一命令继续，已完成快照会跳过模型计算。HTML 里的新帧列表覆盖当前输出目录中已经生成的 overview；原数学面板和完整旧 viewer 也会一并保留。

## 4. 之后只改图，不再跑 CUDA

完成上一步后，例如提高误差色标上限：

```bash
env MPLBACKEND=Agg \
  python -u hktex-overview-fix/fix_overviews.py \
  --recording "$HKTEX_OBSERVER" \
  --snapshots all \
  --height 384 \
  --chunk 64 \
  --error-max 0.05 \
  --plot-only
```

如果仅生成过 final，就把 `all` 换为 `final`。CPU 重画仍需 observer 中的 surface、快照、manifest、analysis/metrics，以及新目录中的像素数据。`--plot-only` 不重新查询 KNN，也不需要 CUDA、模型纹理或特征缓存。

要提高真正的渲染分辨率，必须重新计算新像素，使用新目录，例如：

```bash
env MPLBACKEND=Agg \
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
  python -u hktex-overview-fix/fix_overviews.py \
  --recording "$HKTEX_OBSERVER" \
  --snapshots final \
  --height 768 \
  --chunk 64 \
  --output "$HKTEX_OBSERVER/viewer-surface-v2-768"
```

不要在同一输出目录混用不同分辨率、chunk 或录制来源；脚本会拒绝配置不一致的缓存。更改色标不受此限制。

## 文件与撤回

- 新增 `surface_render.py`：三角面光栅化、深度遮挡、核中心标记。
- 新增 `fix_overviews.py`：原模型回放、UV/GT 校验、像素保存、图像及 HTML 生成。
- 不替换 `hktex-observer/plot_recording.py`，不修改训练脚本、模型参数、analysis 数值或原 viewer。
- 撤回：重新打开原 `observer/viewer/index.html` 即可。新目录与补丁目录可在确认不需要后自行删除。
- `gt-raster-check.png` 是本地用实际 SMPL 模型与原纹理生成的 GT 检查图，**不包含训练预测**。

## 验证与排错

本地已通过：三角形共享边覆盖、正反面深度遮挡、重心坐标重建、核中心遮挡、CPU 重画与 HTML/GIF 生成，以及输入与像素缓存不被修改的检查。实际 SMPL 的 8192 个原 GT 颜色最大差异为 0。新像素的 HKTex CUDA 计算尚需服务器运行验证；此处未声称本地已测试 GPU。

```bash
python hktex-overview-fix/test_surface_render.py
```

- `HKTex differs from recorded code`：请保留错误输出；不要跳过源码校验。脚本不会自动覆盖你现在的代码。
- `Eigen cache hash differs`：需训练时同一份缓存；路径搬迁用 `--cache /新路径/cache.pt`。
- `Loaded mesh...differs` / `Texture does not reproduce saved GT`：核对是否为原 OBJ、MTL 和纹理；路径搬迁用 `--mesh /新路径/SMPL_male.obj`。
- 缺少 `analysis/metrics.json`：指定实际 `--analysis`；本补丁不重新生成谱项等原始数学分析。
- GPU 显存不足：在一个新输出目录使用 `--chunk 32`，其他设置不变。
- 要分析新报错，请发送完整错误栈；渲染成功后可发送新的 final overview 和 `render_report.json`。
