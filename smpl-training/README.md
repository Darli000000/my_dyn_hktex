# HKTex：SMPL 人体纹理拟合验证

适用：已在新服务器跑通苹果 smoke test 的 `hktex` 环境。此次拟合固定人体表面的 RGB 纹理，不优化 SMPL 姿态、体型或几何，也不拟合法线贴图。直接使用 OBJ，不必安装 SMPL 参数模型 SDK，不必转换 GLB。

## 1. 下载并固定模型版本

```bash
conda activate hktex
mkdir -p /root/hktex-work/data
git clone https://github.com/Meshcapade/SMPL_texture_samples.git /root/hktex-work/data/SMPL_texture_samples
git -C /root/hktex-work/data/SMPL_texture_samples checkout --detach c87d871c7498fdc98f3cabb7341b5c85c38debaa
```

如果目录已经存在，跳过 clone；保留已有文件，不要为了重试删除数据。通过 ZIP 下载也可以，但应解压完整目录结构。

本轮选择 `Textured_Mesh_samples/SMPL_male.obj`，同目录保留 `SMPL_male.mtl`、`m_01_alb.002.png`、`m_01_nrm.002.png`。MTL 的 `map_Kd` 指向颜色图；`bump` 指向法线图，当前 HKTex 的 UV 颜色采样器只读取颜色图。`Textures_preview/` 中的预览图不作为这次默认训练输入。不要交叉使用 SMPL-X 的网格或 UV。

核对版本日期：2026-10-19。仓库标注 SMPL-Body CC-BY4.0，保留原仓库授权和来源信息。

## 2. 上传本工具包

将 `smpl-training.zip` 上传到 `/root/hktex-work/hktex/`，然后：

```bash
cd /root/hktex-work/hktex
python -m zipfile -e smpl-training.zip .
python smpl-training/hktex_checked_run_v4.py --version
```

应输出 `HKTex checked runner 4.0.0`。本包复用之前的 V4 诊断器，包括同步追踪检查、初始化记录、密度操作计数、图表和预览保存。

## 3. 只读取并检查模型

```bash
python smpl-training/inspect_obj.py \
  --repo /root/hktex-work/hktex \
  --mesh /root/hktex-work/data/SMPL_texture_samples/Textured_Mesh_samples/SMPL_male.obj \
  --out /root/hktex-work/test-logs/smpl-male-static-report.json
```

此脚本从本地 `hktex/utils/mesh.py` 中提取并调用 `load_mesh`，避免导入整个 GPU 包；不修改 OBJ、纹理或预计算缓存。它检查 MTL/颜色图真实存在，并比较实际加载的纹理与 `map_Kd` 图像，防止“有 UV 但实际读到默认材质”的假通过。

本地使用 trimesh 4.11.2 和已研究的 HKTex 加载函数得到：

| 指标 | SMPL_male.obj |
|---|---:|
| OBJ 几何顶点 | 27,578 |
| OBJ 原始面 | 27,576 个四边形 |
| 自动三角化后 | 55,152 个三角形 |
| 合并几何顶点后 | 27,578 |
| 连通分量 | 1 |
| 闭合 / 绕序一致 | 是 / 是 |
| 边界边 / 非流形边 / 非流形顶点 | 0 / 0 / 0 |
| 重复面 / 近零面积面 / UV 退化面 | 0 / 0 / 0 |
| 纹理 | 4096×4096 |
| 最小三角形内角 | 约 0.243° |
| 最小角小于 1° / 5° 的面 | 1 / 39 |
| 合并前 UV 面与合并后几何面对应误差 | 0 |

完整记录见 `example_static_report.json`（其中路径是本机临时检查路径，不是服务器路径）。这不是常见的 6,890 顶点原始 SMPL 拓扑；当前目标是测试人体纹理拟合，因此可以使用。如果后续需要原始 SMPL 顶点编号或蒙皮参数，另行准备对应网格，不能直接套编号。

服务器应得到 `ready_for_smoke: true`、`failures: []`、`loaded_texture_matches_map_Kd: true`。细长三角形的 warning 是已知风险，先保留追踪检查，不自动重网格化或填洞。静态检查不覆盖自交、特征分解、CUDA 或实际追踪。

## 4. 先 smoke，再 baseline

```bash
python smpl-training/run_smpl.py --phase smoke
```

成功后再执行：

```bash
python smpl-training/run_smpl.py --phase baseline
```

默认路径就是上面的 SMPL_male.obj。更换路径时使用 `--mesh /absolute/path/to/model.obj`；更换项目路径时使用 `--repo`。这两个阶段关闭重要性采样及密度控制器，验证纹理读取、初始化、特征分解、位置更新、渲染和模型保存。

第一轮会重新计算该人体的谱数据，网格比苹果大很多，首次准备阶段可能显著更慢；不要复制苹果的 eigen 缓存来用。只有几何、拓扑及相关谱设置一致时才复用本模型缓存。用新文件名保存几何修改，避免误读旧缓存。

## 5. 启用 ADC 前确认原有修复

苹果 100 步 smoke 不覆盖之前约 1000 步遇到的“剪枝改变索引，随后分裂仍读取旧几何缓存”问题。该问题与换人体无关。

```bash
python smpl-training/apply_split_fix.py --repo /root/hktex-work/hktex --apply
python smpl-training/test_split_regression.py --repo /root/hktex-work/hktex
```

已修复时打印 `ALREADY PATCHED`；未修复时只替换经过匹配的源码块，并保留时间戳备份。若找不到预期源码块则拒绝修改，需要检查服务器当前代码，不要强行替换。回归脚本使用 CPU 和真实控制器方法，检查剪枝、克隆后分裂的几何对应及优化器状态形状；它不代替 CUDA 验证。

## 6. 组件验证，再完整训练

```bash
python smpl-training/run_smpl.py --phase adc
```

这个阶段跑 1200 步，但固定使用完整 5000 步方案的 ADC 调度：增密从零起始迭代 400 开始，每 100 步检查；剪枝在迭代 1000 首次检查。这样能覆盖原先剪枝和分裂相遇的时段，而不是通过缩短 `optim.iters` 压缩全部调度。报告里的 loss 次数通常比零起始迭代号大 1。

检查通过后运行：

```bash
python smpl-training/run_smpl.py --phase full
```

| 参数 | smoke | baseline | adc | full |
|---|---:|---:|---:|---:|
| 步数 | 100 | 300 | 1200 | 5000 |
| 初始热核 | 256 | 1000 | 1000 | 1000 |
| batch | 512 | 4096 | 4096 | 4096 |
| 重要性采样 | 关 | 关 | 开 | 开 |
| 剪枝 / 增密 | 关 | 关 | 开 | 开 |
| 最大热核 | 固定 256 | 固定 1000 | 5000 | 5000 |
| 结果视角 | 4 | 12 | 12 | 60 |
| 每栏分辨率 | 128² | 256² | 256² | 512² |
| 渲染 samples | 4 | 8 | 8 | 16 |

共同设置：seed=0，位置学习率 0.009914527998776132（原始配置的 0.1 倍），位置动量 0.9，初始化比例参数 0.5，num_workers=0，merge_tex=True，不烘焙成顶点颜色；其他 Adam 学习率和谱/KNN 参数继承本地原始 YAML。ADC 阈值及 split_radius 继承原始 YAML。重要性采样池 100 万点、预热 250 步。

初始 1000/最大 5000 个热核是用于比较和验证的起点，不是根据人体面数推导的最优容量。先确认 5000 步结果，再决定是否试初始 2000/上限 10000、更多迭代或局部采样。首轮不同时改多个超参数。位置学习率保守并不保证分裂追踪安全，分裂位移还受到 split_radius 等参数影响。

启动器自动设置 MPLBACKEND=Agg、CUDA_LAUNCH_BLOCKING=1、Conda libstdc++ 的进程级 LD_PRELOAD，保存独立 YAML 与日志。不要修改系统 C++ 库。诊断同步增加耗时，本轮不作为性能基准。

V4 的 `--forward-inner-texture` 在进程内将重要性采样包装器的 `tex_img` 转发到内部数据集，让原初始化函数能执行。这个处理和低位置学习率都属于当前实验设置；不是完全未改动的论文复现。

各阶段都是从头开始的独立实验，不会接续上一阶段的优化器状态。现有入口不提供完整状态的周期性断点续训；长实验建议在 tmux 会话运行。`--prepare-only` 可只生成配置并检查前置条件，不启动训练。

## 7. 结果查看与通过标准

终端打印本轮 `[CHECK] Experiment directory`、`Preview image` 和 `Report` 的准确路径。

```text
/root/hktex-work/hktex/outputs/smpl-full/<tag>@<时间戳>/ckpts/
  preview_views.png        # 每行：GT | HKTex output | Kernel rings
  gt_vs_out_slow.mp4       # 10 fps 转台预览
  gt_vs_out.mp4
  model.pt
  model.npz

/root/hktex-work/hktex/outputs/training-checks/<时间戳与ID>/
  report.json
  loss.csv
  plot_*.png

/root/hktex-work/test-logs/smpl-<phase>-<时间戳与ID>.log
```

前两阶段目录分别为 `outputs/smpl-smoke`、`outputs/smpl-baseline`，组件验证为 `outputs/smpl-adc`。下载 PNG/MP4 在本地查看。首先看 GT 是否确实是预期的人体纹理、相机是否包含全身，再比较 output 的面部、手指、四肢和 UV 接缝区域。512² 全身图仍不足以证明脸部细节已拟合好，后续应增加固定近景视角或独立表面采样评估。

每阶段要求 `status: completed`、loss_evaluations 与预期一致、参数有限、无追踪错误，并有渲染和模型输出。adc/full 还要查看 `sampler_batches.importance > 0`、`event_counts` 和 `density_totals`。控制器被检查不等于实际发生克隆/分裂/剪枝；某项为 0 时不能声称对应分支已经验证，也不应立即为了计数降低阈值。

训练 loss 是当前训练 batch 的 MSE；重要性采样和核数变化会影响它，不能只据此比较不同实验的拟合质量。想正式比较，应固定独立、按面积均匀采样的评估点与 GT，并使用一致视角/光照。这个包尚未加入该评估器。

失败时保留日志、对应 report.json 和实际 configs/parsed.yaml。若显存不足，先根据堆栈区分谱预计算、训练还是渲染：减 batch 不一定解决预计算占用；渲染阶段可先降分辨率/帧数。若追踪失败，保留输入检查，先定位是位置更新还是分裂操作，不通过放宽断言掩盖问题。

## 验证记录与来源

已在本地实际读取选定 OBJ、MTL 和颜色图，完成上述静态检查；已验证四阶段配置生成、1200/5000 步一致的 ADC 调度、前置补丁检查及命令行准备流程。本机未运行该人体的 CUDA 训练，是否能完整训练仍由服务器分阶段验证。

- [模型与纹理仓库，固定版本](https://github.com/Meshcapade/SMPL_texture_samples/tree/c87d871c7498fdc98f3cabb7341b5c85c38debaa)
- [配套 MTL](https://github.com/Meshcapade/SMPL_texture_samples/blob/c87d871c7498fdc98f3cabb7341b5c85c38debaa/Textured_Mesh_samples/SMPL_male.mtl)
- [HKTex UV 颜色采样](https://github.com/circle-group/hktex/blob/9e942972c5411a364f12563eb6864520f9effe70/hktex/data/uv_texture_sampler.py)
- [HKTex 基础配置](https://github.com/circle-group/hktex/blob/9e942972c5411a364f12563eb6864520f9effe70/configs/texture_hktex_knn.yaml)
