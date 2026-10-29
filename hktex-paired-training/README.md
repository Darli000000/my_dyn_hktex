# HKTex：固定 5000 kernels 的图片损失对照

适用仓库：`Darli000000/my_dyn_hktex`，已核对 main 提交 `b02d07ee7e8a7a0d127ee6ef0dd74bdda906b51d`。

这一步是**作者式损失对照**，不是完整复刻作者训练，也不是在旧的有斑点 checkpoint 上继续修补。

## 1. 实验范围

| 项目 | linear | loggamma |
|---|---|---|
| 初始化 | 同一份新建的 `initial.pt`，5000 kernels | 完全相同 |
| 监督 | 已保存的 2048×2048 线性 RGB | 同一数据集 |
| 采样 | 256 像素/update；4 次子像素 × 4 次 transport | 同一像素、视角和渲染种子序列 |
| 损失 | 原始线性 RGB mean L1 | clip → 作者的 log → 作者的 gamma → mean L1 |
| ADC | 关闭 | 关闭 |
| 优化器与调度 | 现有图片 trainer 的学习率；总 horizon=5000 | 相同 |
| 记录 | 参数快照每 250 步及阶段末尾、训练样本、梯度、固定表面探针 | 相同 |

两组顺序执行，不同时占用 GPU。训练仍然从原始 2048 图片抽取像素、生成射线、执行 HKTex/PRB 可微渲染；512/1024 的渲染命令只决定展示分辨率。

**仅修改 YAML 不够。** 当前 `image_trainer.py::step_images()` 写死了线性 RGB L1，没有执行父类 `optimise()` 中的 `use_linear_gamma_space` / `use_hdr` 分支。新增启动器在本进程内替换损失和分段训练入口，复用现有 Dataset、射线渲染、参数更新、checkpoint 和 observer；不改磁盘上的旧脚本。

本次有意没有一并更改初始化、相机、学习率、采样和 ADC，避免无法解释 A/B 差异。作者原实现还使用在线 GT 渲染、自己的初始化/相机抽样、每 epoch 调度和 ADC 误差累计。完整复刻需要逐项对齐这些逻辑，不能只复制 YAML。

### 数值约定

作者代码对应 `hktex/trainers/mitsuba_ray_trainer.py::optimise()` 及 `hktex/utils/color.py`：

```text
x = clip(radiance, 0, 1)
q = (log(65536*x + 1e-6) - log(1e-6)) / (log(1 + 1e-6) - log(1e-6))
F(x) = 1.055 * q^(1/2.4) - 0.055
```

B 组对 prediction、target 都使用这个公式，但在幂运算前加 `max(q, 1e-12)`，防止 q=0 的导数发散。它使 F(0) 相对严格公式变化约 `1.055e-5`；这是明确记录的数值保护，不是完全逐位复现。它不是常规显示 sRGB 转换。

原作者最后对 RGB **求和**；这次两组统一对 RGB **求平均**，避免仅因损失放大三倍引入梯度尺度差异。因此 B 也不应命名为“严格原版”。

统计中 `loss_l1` 一直是原始线性 RGB mean L1，`loss_objective` 才是该组实际反传的损失。不要直接比较两组 `loss_objective` 的大小。

## 2. 安装

将 `hktex-paired-training.zip` 上传到 `/root/hktex-work/`，然后：

```bash
conda activate hktex
cd /root/hktex-work/hktex
python -m zipfile -e /root/hktex-work/hktex-paired-training.zip /root/hktex-work/hktex
```

解压后新增 `hktex-paired-training/`。无需覆盖旧训练文件夹，无需更改 CUDA 或 pip 依赖。

默认路径：

```text
repo:    /root/hktex-work/hktex
dataset: /root/hktex-work/data/offline_multiview/smpl-male-full-20261025-003839-35340
```

若要改变路径，在第一次实验前设置 `HKTEX_REPO` / `HKTEX_DATASET`；不要在两组之间改变数据、源码或启动器。启动器核对源码 SHA256；不一致时会停止并列出文件，请保留差异供检查，不要绕过检查。

## 3. 先运行两组各 50 步

```bash
PAIR=/root/hktex-work/hktex/outputs/image-training/paired-5000-loss-v1
bash /root/hktex-work/hktex/hktex-paired-training/run_pair.sh smoke "$PAIR"
```

这个 smoke 已经是 **5000 kernels**，不是原先的 256-kernel smoke。第一次可能需要编译或建立谱缓存。

输出：

```text
paired-5000-loss-v1/
  initial.pt
  initial_identity.json
  logs/
  s00050/
    comparison.json
    linear/
    loggamma/
```

检查 `s00050/comparison.json` 的 `pairing_passed: true`。这仅代表两组输入、初始状态和更新序列可比，不代表训练质量已经合格。两组目录的 `report.json` 应为 `stage_complete`、`completed_updates: 50`；`loss_preflight.json` 应通过；不得出现非有限数、梯度异常或 tracer 错误。

`settings.steps=5000` 是调度总长度；`report.stage_until=50` 是本次实际停止点，不是参数错误。

不要把旧的 1000/1500/5000 步实验 checkpoint 放进这组对照。本实验从全新的共享初始化开始。

## 4. 续训到累计 1000 步

50 步两组均正常后：

```bash
PAIR=/root/hktex-work/hktex/outputs/image-training/paired-5000-loss-v1
bash /root/hktex-work/hktex/hktex-paired-training/run_pair.sh 1000 "$PAIR"
```

每组从自己的 `s00050/<arm>/checkpoint.pt` 恢复参数、优化器、调度器、训练随机状态和 observer lineage，再训练 950 步。输出到新的 `s01000/`，不会覆盖 50 步结果。

调度始终按 5000 步计算：Adam StepLR 每 1250 步乘 0.5；GeodesicOpt cosine 的 T_max=5000。没有把 50 步或 1000 步短实验的学习率提前衰减到末尾。

两组共用现有图片 trainer 的初始学习率：Adam 各参数组为原 YAML 的 0.3 倍，GeodesicOpt 为 0.001。准确配置写入各组 `trainer.yaml`。

## 5. 渲染 1000 步结果

先用 512、128 spp；不开去噪，便于看清色块是否真的消失：

```bash
PAIR=/root/hktex-work/hktex/outputs/image-training/paired-5000-loss-v1
bash /root/hktex-work/hktex/hktex-paired-training/render_pair.sh 1000 "$PAIR" 512
```

如果需要放大看细节：

```bash
bash /root/hktex-work/hktex/hktex-paired-training/render_pair.sh 1000 "$PAIR" 1024
```

脚本直接调用现有 `hktex-image-training-observed/render_native_images.py`，读取该阶段 `observer/` 的 final 快照。使用保存的几何、法线、相机和同一材质/环境，生成 GT、prediction、核环图及 HTML，不会用新 loss 变换显示颜色。

默认视角为 `view_000,view_002`。每组会打印 `[VIEWER] .../index.html`，目录结构为：

```text
s01000/linear/native-512-时间戳-PID/index.html
s01000/loggamma/native-512-时间戳-PID/index.html
```

每个快照子目录包含 `prediction_000_raw.png`、`prediction_001_raw.png` 等。图片是正常显示转换后的 raw 渲染，不经过 OptiX 去噪；训练仍使用线性浮点 RGB。

如要渲染本阶段全部常规快照：

```bash
bash /root/hktex-work/hktex/hktex-paired-training/render_pair.sh 1000 "$PAIR" 512 all
```

各阶段 observer 独立存放、lineage 继续关联。`s01000` 的 all 包含恢复点 50 和 250/500/750/1000；若想看初始 0 步，可对 50 步阶段执行：

```bash
bash /root/hktex-work/hktex/hktex-paired-training/render_pair.sh 50 "$PAIR" 512 all
```

## 6. 判读及是否继续 5000 步

请优先提供：

- `s01000/comparison.json`；
- 两组 `metrics.jsonl`、`field_metrics.jsonl`；
- 两组相同视角的 `prediction_000_raw.png` / `prediction_001_raw.png`；
- 若任何组失败，提供 `report.json`、`startup.json` 和相应日志。

观察共同的线性 val MSE、视觉斑块、固定 2048 表面探针的 `clipped_fraction`。探针指标是表面颜色越界比例，不是渲染图像误差或整个表面的精确统计。中间 eval 仅取少量固定像素和部分视角，并含 Monte Carlo 噪声，不是完整图像 PSNR。

如果两组都稳定但 1000 步还不能判断，可以接着完成累计 5000 步：

```bash
PAIR=/root/hktex-work/hktex/outputs/image-training/paired-5000-loss-v1
bash /root/hktex-work/hktex/hktex-paired-training/run_pair.sh 5000 "$PAIR"
bash /root/hktex-work/hktex/hktex-paired-training/render_pair.sh 5000 "$PAIR" 1024
```

这是另外 4000 步；输出到 `s05000/`。最终 eval 包括 train/val/test 的全部相机、每相机固定抽样像素。最终与中间 eval 的视角集合不同，不要把聚合 MSE 的跳变全解释成优化变化。

若 B 明显改善，再讨论是否引入作者式 ADC。若两组仍有相同斑块，不要仅凭 loss 下降就接入增密剪枝；先比较表面 clipping 和参数/梯度，再安排同条件在线 GT 对照或颜色参数化实验。这个实验不预设斑块的原因必然是 loss。

## 7. 数学 observer 的离线回放（可选）

已有 observer 记录无需重做；新实验继续使用图片训练的适配版，原 UV observer 没有图片监督的表面 GT，不能直接把旧 UV 启动命令替换路径来用。

例如分析 B 组 1000 步阶段：

```bash
RUN=/root/hktex-work/hktex/outputs/image-training/paired-5000-loss-v1/s01000/loggamma

env MPLBACKEND=Agg \
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
  python /root/hktex-work/hktex/hktex-image-training-observed/replay_image_fields.py \
  --recording "$RUN/observer" --repo /root/hktex-work/hktex \
  --gpu 0 --chunk 64 --snapshots regular

MPLBACKEND=Agg python /root/hktex-work/hktex/hktex-image-training-observed/plot_image_fields.py \
  --recording "$RUN/observer"
```

默认生成 `observer/analysis/` 和 `observer/viewer/`。这里的 surface RGB 是表面 base colour；图片 radiance 包含光照，不能将二者直接相减当作训练误差。

## 8. 失败、重跑与撤回

- 路径已存在时脚本会拒绝覆盖。新实验换用新的 `PAIR` 名称。
- 两组比较不通过时不要进入下一阶段，先保留日志排查。
- 如果一组显存不足，不要仅给该组降 batch；两组必须使用相同的新设置重新开始。请先发错误和显存情况。
- 如果 linear 完成而 loggamma 因临时中断失败，可以保留 linear，重跑该阶段 loggamma。先把失败的 loggamma 目录改名备份，再直接调用 `run_arm.py`；不要误用 linear 的 checkpoint。

例如重做 1000 步阶段的 loggamma（此时线性组已经完成）：

```bash
PAIR=/root/hktex-work/hktex/outputs/image-training/paired-5000-loss-v1
mv "$PAIR/s01000/loggamma" "$PAIR/s01000/loggamma-failed-$(date +%Y%m%d-%H%M%S)"
env MPLBACKEND=Agg \
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
  python /root/hktex-work/hktex/hktex-paired-training/run_arm.py \
  --arm loggamma --pair "$PAIR" --until 1000 \
  --resume "$PAIR/s00050/loggamma/checkpoint.pt" \
  --output "$PAIR/s01000/loggamma"
python /root/hktex-work/hktex/hktex-paired-training/compare_pair.py \
  --pair "$PAIR" --stage s01000
```

撤回：停止使用 `hktex-paired-training/` 的新入口，继续使用原入口即可。原文件没有被覆盖，不需要恢复 YAML 或训练脚本。保留实验输出用于比较。

## 9. 已做检查与限制

本地已通过 6 项 CPU 检查：损失公式、零值数值保护/梯度、RGB reduction、续训分组/源码检查、A/B 配对检查、调用实际旧 make_config 的参数及调度恢复检查；另做 Python/Bash 语法检查和源文件哈希检查。

CPU 检查不等于 CUDA 集成测试。本地没有服务器 CUDA 环境；服务器上 50 步两组测试仍是必需的。启动器会在第一步前使用实际 Dr.Jit CUDA 检查零值等输入的 loss 梯度是否有限，并继续沿用现有几何、参数与梯度检查。
