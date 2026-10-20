# HKTex 原渲染器离线 viewer

这是独立的展示用 viewer。保留原 observer、v2 数值诊断 viewer 和训练代码，不需要重新训练。

## 为什么 v2 锯齿更多

本次 manifest 的原展示配置为 512×512、multijitter、16 samples/pixel、perspective、path integrator。原代码使用 principled 材质、环境光、OptiX 去噪，并以 `srgb_gamma=True` 输出；`preview_views.png` 是四张旋转帧的拼接，本身没有提高渲染精度。

v2 `--height 384` 对应每个视图 288×384，每像素一次网格采样，原训练式 `uv_to_color` 最近邻纹理取色，无光照、无去噪，最后 `imshow(interpolation='nearest')` 展示。因此轮廓和细纹理都可能有明显混叠。增加 PNG 的 dpi 或把旧 PNG 放大不会增加采样精度。

只提高 v2 分辨率，可以继续使用旧补丁：

```bash
env MPLBACKEND=Agg \
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
  python -u hktex-overview-fix/fix_overviews.py \
  --recording "$HKTEX_OBSERVER" \
  --snapshots final --height 768 --chunk 64 \
  --output "$HKTEX_OBSERVER/viewer-surface-v2-768"
```

这会重新计算更多表面像素，但不会变成原 Mitsuba 展示风格。

## 1. 解压

把 `hktex-native-viewer.zip` 上传到 `/root/hktex-work/hktex/`。

```bash
conda activate hktex
cd /root/hktex-work/hktex
python -m zipfile -e hktex-native-viewer.zip .

export HKTEX_OBSERVER='/root/hktex-work/hktex/outputs/smpl-observed-full/male_init05_lr10_20261019-044000-c77569@20261019-044005/observer'
```

新包自带所需的 v2 读取/校验辅助模块，不覆盖原补丁。不额外安装依赖；使用之前能成功渲染 preview 的 hktex 环境。

## 2. 首先渲染 final 四视图（推荐）

```bash
env MPLBACKEND=Agg \
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
  python -u hktex-native-viewer/render_native.py \
  --recording "$HKTEX_OBSERVER" \
  --repo /root/hktex-work/hktex \
  --snapshots final \
  --size 512 --spp 16 --frames 4 --point-batch 512
```

输出：

```text
observer/viewer-native/
├── index.html
├── settings.json
├── renderer_config.json
├── completed.json
├── gt-cache/                         # 所有快照共用的 GT 渲染
└── 00082_u05000_final/
    ├── preview_views.png            # GT / HKTex output / Kernel rings
    ├── preview_views_raw.png        # 同一次渲染的未去噪对照
    ├── gt_000.png, gt_000_raw.png
    ├── prediction_000.png, prediction_000_raw.png
    ├── rings_000.png, rings_000_raw.png
    ├── prediction_000.npz           # 原始线性 RGB
    ├── rings_000.npz                # 原始线性 RGB
    └── …其余方向
```

先打开 `00082_u05000_final/preview_views.png`。下载整个 `viewer-native` 文件夹后，可在本地打开 `index.html`，切换训练快照、方向和去噪前后结果。

四个方向覆盖一圈，继承训练配置的起始方位角；与原 60 帧视频的第 0、15、30、45 帧方向相同。每一帧独立调用原 renderer 和 OptiX，不复用原旋转视频循环中的 previous_denoised 缓冲。随机采样、KNN 并列选择与环颜色可能导致结果不逐像素相同。

## 3. 提高质量

```bash
env MPLBACKEND=Agg \
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
  python -u hktex-native-viewer/render_native.py \
  --recording "$HKTEX_OBSERVER" \
  --snapshots final \
  --size 768 --spp 64 --frames 4 --point-batch 512 \
  --output "$HKTEX_OBSERVER/viewer-native-768-spp64"
```

| 参数 | 用途 |
|---|---|
| `--size` | 每张图的实际宽高，提高空间分辨率 |
| `--spp` | 每像素采样数，改善采样噪声与轮廓采样 |
| `--frames` | 旋转方向数量，不提高单帧清晰度 |
| `--point-batch` | 模型查询批大小，主要影响显存与运行时间 |
| `--tile-size 128` | 用原渲染器分块，减少峰值渲染内存 |
| `--no-denoise` | 不去噪；默认已额外保留未去噪图片 |
| `--ring-thickness 0.03` | 热响应数值域的带宽，不是世界空间半径或像素线宽 |

768/64 相对 512/16 的像素采样总量约为 9 倍，不能简单认为速度相同。建议先用第二步的配置验证。若显存不足，用新输出目录配 `--point-batch 128 --tile-size 128`；减小 point-batch 不会降低空间分辨率。

## 4. 所有训练阶段 / 最终旋转视频

生成全部常规训练快照（initial、post_update、final；不自动包含 ADC 事件快照）：

```bash
env MPLBACKEND=Agg \
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
  python -u hktex-native-viewer/render_native.py \
  --recording "$HKTEX_OBSERVER" \
  --snapshots all \
  --size 512 --spp 16 --frames 4 --point-batch 512
```

也可用 `--snapshots 0,81,82` 指定快照。已计算的线性像素缓存复用；中断后同命令继续。改变分辨率、spp、帧数等设置时使用新 `--output`，防止混用缓存。

生成 final 的 60 帧旋转视频：

```bash
env MPLBACKEND=Agg \
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
  python -u hktex-native-viewer/render_native.py \
  --recording "$HKTEX_OBSERVER" \
  --snapshots final \
  --size 512 --spp 16 --frames 60 --point-batch 512 \
  --video --fps 10 \
  --output "$HKTEX_OBSERVER/viewer-native-turntable"
```

视频保存为该快照子目录的 `comparison.mp4`。视频编码失败不会丢失已经保存的 PNG、NPZ 与总览图；错误记录在 completed.json。

## 原作者的 kernel rings 到底是什么

`BaseTrainer.render_kernel_rings()` 暂时把 `kernel_filter_func` 改为 `box_border`、核颜色改为随机 RGB、平均颜色改为 0，再走正常渲染路径。原方法实际使用 0.03 带宽。

对过滤前响应 h 和该核阈值 epsilon：

```text
border = 1  当 epsilon - 0.03 <= h < epsilon
border = 0  其他情况
```

此 h 是进入颜色过滤器的响应：在当前配置下已进行自热核归一化和距离加权；并非原始未处理热核。环的形状与该响应场有关，且受原候选 KNN/内部 Top-K 与混色规则限制。它不是每个核一个固定大小圆圈，也不能等同于完整的热核数值图。

新脚本直接使用项目 `box_border`，默认带宽相同。为便于比较不同训练阶段，随机展示颜色改为稳定 ID 决定；与旧图的具体颜色不同是正常的。更换颜色和过滤器只发生在离线模型副本，并在 finally 中恢复原参数对象；每个快照结束后检查参数逐元素未变。

## 保留与限制

- 本工具生成独立展示 viewer，不覆盖 v2 的误差图、数学面板或训练记录。评估拟合精度继续使用未去噪数值与原表面评估；光照、sRGB 和去噪会改变外观，OptiX 也可能抹去真实高频细节。
- 不修复之前确认的 KNN 并列选择问题。高 spp/去噪可能让差异视觉上不明显，但不代表根因消失。
- 读取同一份特征缓存并校验源码、网格、UV 与 GT。搬迁路径使用 `--mesh`、`--cache`；不要跳过哈希校验。
- 无需上传完整项目。若报错，提供错误栈；若效果仍不对，提供新 preview_views.png、preview_views_raw.png 和 renderer_config.json。
- CPU 已验证环模式异常恢复、稳定 ID 配色和无缩放图片拼接。读取与前向框架沿用已在你服务器跑通的 v2；此新增 Mitsuba 回放脚本尚未在本地 CUDA 环境执行，需第二步验证。
- 撤回：继续使用原 `viewer` 或 `viewer-surface-v2` 即可。

## 实现参考

核对版本：`circle-group/hktex@9e942972c5411a364f12563eb6864520f9effe70`。

- [原渲染器：采样、film、去噪、sRGB、旋转帧](https://github.com/circle-group/hktex/blob/9e942972c5411a364f12563eb6864520f9effe70/hktex/rendering/base.py)
- [原 kernel rings 和 render_result](https://github.com/circle-group/hktex/blob/9e942972c5411a364f12563eb6864520f9effe70/hktex/trainers/base.py)
- [HKTex 在 Mitsuba 表面交点处求值](https://github.com/circle-group/hktex/blob/9e942972c5411a364f12563eb6864520f9effe70/hktex/rendering/heat_kernels_renderer_knn.py)
- [UV 纹理与材质](https://github.com/circle-group/hktex/blob/9e942972c5411a364f12563eb6864520f9effe70/hktex/rendering/uv_texture_renderer.py)
- [box_border 的响应带定义](https://github.com/circle-group/hktex/blob/9e942972c5411a364f12563eb6864520f9effe70/hktex/utils/misc.py)
