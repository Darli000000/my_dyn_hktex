# VHAP → HKTex：单帧适配与几何兼容性检查

基于 `Darli000000/my_dyn_hktex` 的 `de678ce57634fa51b5baa0e1c314eb0282fd874e` 编写。
仅新增本目录；不修改 HKTex 核心、已有训练脚本、源照片、相机或 mesh。
脚本会在运行报告中记录本机 HKTex 源文件哈希，允许识别服务器代码差异。

## 这一步回答什么

1. 一组同步照片、前景 alpha、相机和同一帧 mesh 能否构成一致的表面采样数据？
2. 原 HKTex 的 Mesh、谱预计算、KNN 查询、颜色反向传播和测地线位置优化能否处理这张 mesh？

这里不训练完整外观，不验证 PRB / Dr.Jit 可微渲染桥，不提供物理 albedo，也不宣称已经完成真实数据 trainer。
我们拟合的下一阶段目标是固定光照下的表面 RGB：像素中心射线 → 首个 mesh 命中 → HKTex 表面颜色。
先把相机、遮挡和字段优化链路验证清楚，再编写真正训练入口。

## 一次执行

将整个 `hktex-vhap-check` 文件夹放进服务器 HKTex 仓库根目录；保留现有 `VHAP` 和 `hktex` 两个 conda 环境。
需要已安装的 VHAP/FLAME 资产及已能运行原实验的 HKTex CUDA 环境。脚本不安装/升级依赖。

```bash
cd /root/hktex-work/hktex
bash hktex-vhap-check/run_checks.sh
```

默认输入：

```text
/root/hktex-work/data/vhap-export/018_EMO-1-shout+laugh-20261029_epoch30
```

默认第 0 帧、GPU 0、5000 kernels；验证相机索引 7，测试索引 15，其余有效相机训练。
这里是 `camera_index`，不是设备序列号。无有效面部像素的训练相机会被标为 excluded；验证/测试相机无足够像素会停止。
这只是外观的留出视角；VHAP 几何拟合已经使用所有视角，不是严格的端到端留出重建实验。

可通过环境变量调整路径：

```bash
EXPORT=/root/hktex-work/data/vhap-export/018_EMO-1-shout+laugh-20261029_epoch30 \
VHAP_ROOT=/root/hktex-work/VHAP GPU=0 FRAME=0 \
bash hktex-vhap-check/run_checks.sh
```

`VHAP_ENV` 默认 `VHAP`，`HKTEX_ENV` 默认 `hktex`。`RUN` 可以指定一个尚不存在的输出目录。
默认输出在仓库 `outputs/vhap-stage12-日期-时间-PID/`。启动时打印完整路径。
不覆盖已有输出，不自动补洞、合并顶点或删除部件。不会启动 5000 步训练。

## 阶段 1：数据与 ROI

`export_regions.py` 在 VHAP 环境读取 FLAME 区域标签。按本次配置使用 `FlameHead(300,100,add_teeth=True)`，逐元素校验三角面索引及顶点数与输入 OBJ 一致，再导出区域。
如果拓扑不一致，必须检查 VHAP 版本/模型选项，不能强行套标签。

`prepare.py`：

- 保持原图 550×802、相机内外参、mesh 坐标与顶点/面顺序。
- 使用导出文件的 OpenGL c2w，中心射线为 `(x+0.5,y+0.5)`；不再重复 OpenCV→OpenGL 转换。
- 用全部三角面做透视正确的 CPU z-buffer，保存首命中 face ID、重心坐标和相机深度。
- RGB 按标准 sRGB EOTF 解码一次作为工作颜色空间；这不等于确认相机的物理辐射标定，也不去除光照。保留原始 PNG 用于展示。
- 前景 alpha ≥0.99、面朝相机余弦≥0.25，叠加 FLAME face 标签并排除 neck/scalp/boundary/eye_region/眼球/lips/lip_inside/teeth，最后内缩 2 像素。
- **只限制损失区域，不裁掉遮挡用 mesh。** FLAME 标签不是照片中的头发分割；仍需查看所有视角的 overlay。尤其确认额头发际线、耳饰和侧脸遮挡。
- 写出新 schema `hktex-vhap-static-v1`，并保存资产哈希。不能直接交给旧合成 `image_dataset.py`。

`dataset/roi_contact_sheet.jpg` 中每个相机的三列是原图、叠加图、候选损失区域。青色是完整 mesh 外轮廓，绿色是候选损失区域。
完整轮廓不必覆盖真实头发/衣服；但候选面部区域不能明显错位或把这些内容当作脸部真值。
`dataset/view_XX/overlay.png` 保留全分辨率。

CPU 的重投影误差只证明计算自洽，**不证明真实相机与脸部几何没有拟合误差**；后者需要叠加图判断。
`training_permitted=false` 是故意保留的标记：这份数据包尚未通过 GPU 检查和人工 ROI 复核，也未接入正式 trainer。

## 阶段 2：原 HKTex 核心

`check_gpu.py` 使用当前仓库代码，按顺序检查：

1. Mitsuba 与 CPU 的首命中面、命中点、投影、深度，以及背景射线不应误命中。
2. 原 `Mesh(..., uv=None)` 与 GPU geodesic tracer。分别测试零移动、每个部件上的小切向移动、朝开放边界外移动；要求坐标有限、位于三角面内、没有跨不连通部件。
3. 原 small-ray 配置中的完整谱预计算（256 eigenvalues，原 anisotropy/angle grid），不通过减少谱规模掩盖兼容性问题。
4. 5000 kernels 初始化、有限颜色查询、跨不连通部件的实际 kernel 权重统计。
5. 原字段的反向传播，kernel 颜色梯度与有限差分对比；一次颜色更新降低已知诊断目标的 MSE；一次原 GeodesicOpt 位置更新仍落在合法表面。

诊断目标是固定 RGB `[0.2,0.45,0.6]`，用于检验梯度，不用于评价真实照片拟合质量。
颜色残差初始化、每步学习率、tracer 最大 256 次迭代是诊断设置，不能直接当作完整训练超参数。
不会启动 ADC，不会保存这些临时诊断参数作为训练 checkpoint。

`gpu/network.yaml` 保存实际网络配置；缓存按 mesh、配置、HKTex 源码及运行环境隔离。
首次完整谱计算和 Torch 编译可能较慢；终端和 `gpu/report.json` 的 `stage` 会显示当前检查阶段。
`peak_cuda_gb` 仅为 Torch allocator 记录，不代表整个进程（如 Mitsuba/FAISS）的全部显存。

## 如何判读

| 结果 | 解释与下一步 |
|---|---|
| `dataset/report.json`: `cpu_checks_passed` | 数据结构和 CPU 几何计算通过，继续 GPU 与 ROI 复核。 |
| `gpu/report.json`: `passed` | 当前采样范围内的核心链路通过；ROI 人工复核后可进入短训练。不是完整几何正确性证明。 |
| `inconclusive` | 看 `warnings`，例如没有解剖标签、跨部件贡献或位置梯度为零；先定位，不扩大训练。 |
| `failed` | 看 `stage` 和 traceback；可能是依赖/API、相机、谱或 tracer 问题，不能自动归因为坏 mesh。 |

验收阈值：CPU 重投影最大误差≤0.001 px；GPU≤0.02 px；GPU 命中位置/深度误差≤mesh 对角线×2e-5（最低1e-7）；所抽样的内部像素面 ID 一致。
跨部件绝对权重比例>0.001 会标为 inconclusive。这是排查阈值，不是论文标准，也不据此自动切 mesh。
自相交及顶点层面的 manifold 完整性目前未穷尽检测，报告会明确标记；有症状再增加局部检查。

先发回：`dataset/report.json`、`dataset/mesh_report.json`、`gpu/report.json`、`dataset/roi_contact_sheet.jpg`。
若失败，再加 `run.log`；若谱/模型检查失败，补 `gpu/network.yaml`。
如果 GPU 通过且面部区域对齐，下一阶段才做固定 mesh/相机、linear RGB MSE、5000 kernels、ADC 关闭的小规模拟合。

## 分开重跑

若阶段 1 已完成，可复用同一数据包，仅重跑 GPU 检查，输出目录必须换新：

```bash
conda run --no-capture-output -n hktex python hktex-vhap-check/check_gpu.py \
  --dataset /完整路径/vhap-stage12-运行目录/dataset \
  --output /完整路径/vhap-stage12-运行目录/gpu-retry \
  --gpu 0
```

没有 VHAP/FLAME 资产的本地环境可运行 `prepare.py --export ... --output ... --geometry-only`。
这种输出只是几何投影检查，会标为 `geometry_only_inconclusive`；全人物 alpha 与 mesh 的交集不能充当正式面部损失 mask。

CPU 回归检查：`python hktex-vhap-check/test_cpu.py -v`。GPU 需要服务器实际执行，本工具包不附虚构的 CUDA 通过结果。
