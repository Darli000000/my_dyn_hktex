# HKTex 图片监督：颜色截断恢复对照

这是新实验阶段，不是作者原始训练的逐字复现。网络前向、渲染器和 clamp 均不修改；range 组额外增加一个作用于 clamp 前颜色的范围损失。

## 目的与边界

已有 full 图像中，下半图前景约 46–48% 的像素红通道接近零，而对应 GT 红通道大于 0.005。旧 ADC 的材质探针也曾测到大量负值截断。当前 checkpoint 的实际表面截断比例由这个程序启动时重新测量，不能直接沿用旧比例。

两组均从同一个 full checkpoint 的参数开始，重新创建优化器，使用局部 0–1000 步计数：

| 项目 | control | range |
|---|---|---|
| 初始化 | full 第 5000 步 | 完全相同 |
| 可训练参数 | 仅核颜色残差 | 相同 |
| 全局平均色/位置/方向/各向异性/阈值/锐度 | 全部冻结 | 相同 |
| ADC | 关闭 | 关闭 |
| 颜色学习率 | 0.001，恒定 | 相同 |
| 每步监督 | 256 像素，AA=4，transport=4 | 相同 |
| 图像损失 | 原线性 RGB L1 | 相同 |
| 范围损失权重 | 0 | 1.0（实验起点，不是已验证最优值） |

范围损失为 `mean(relu(-z)^2 + relu(z-1)^2)`，z 是组合后、clamp 前的 RGB。它在越界区域提供恢复梯度，合法区间内为零。没有将负反射率送入渲染器，没有使用 UV 真值，也没有改成伪造梯度的 straight-through 操作。

固定 8192 个面积均匀表面点，缓存其原实现的核权重。由于仅训练颜色，缓存始终有效。启动时会检查缓存重建的颜色与原 forward 一致；训练中检查被冻结的参数没有变化。

这个实验隔离“颜色与截断能否修复”的问题。它不能增加核覆盖，不能修复几何错误，也不能单独验证 PRB 的全部梯度正确性。若验证误差变坏或只减少截断而斑纹不减，不应据此直接切到新 full。

## 安装及训练

将本目录放到 `/root/hktex-work/hktex/hktex-colour-repair`，保留现有 `hktex-image-training-observed`。

先运行 control，再把 `HKTEX_REPAIR_MODE=control` 改为 `range` 运行同样命令。两组都读取原 full checkpoint；不要让 range 接着 control 的结果训练。

```bash
conda activate hktex
HKTEX_REPAIR_MODE=control
HKTEX_PARENT=/root/hktex-work/hktex/outputs/image-training/full-reference-20261025-061226
HKTEX_REPAIR_OUT="/root/hktex-work/hktex/outputs/colour-repair/${HKTEX_REPAIR_MODE}-$(date +%Y%m%d-%H%M%S)"
mkdir -p /root/hktex-work/test-logs
set -o pipefail

env MPLBACKEND=Agg \
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
  python -u /root/hktex-work/hktex/hktex-colour-repair/run_colour_repair.py \
  --mode "$HKTEX_REPAIR_MODE" \
  --checkpoint "$HKTEX_PARENT/checkpoint.pt" \
  --output "$HKTEX_REPAIR_OUT" \
  --repo /root/hktex-work/hktex \
  --trainer-dir /root/hktex-work/hktex/hktex-image-training-observed \
  --dataset /root/hktex-work/data/offline_multiview/smpl-male-full-20261025-003839-35340 \
  --steps 1000 --lr 0.001 --range-weight 1.0 \
  2>&1 | tee "/root/hktex-work/test-logs/colour-repair-${HKTEX_REPAIR_MODE}-$(date +%Y%m%d-%H%M%S).log"
```

输出中 `metrics.jsonl` 的 `loss_l1` 仍只代表图像 L1。新增的 `repair_metrics.jsonl` 单独记录范围损失和表面截断比例，不能把两者混称为同一个 loss。

初始 `eval_000000.json` 应接近父实验的 `eval_005000.json`（都是同样的前两个训练/验证视角及 1024 个采样点）；不要拿它与父实验所有视角平均的 `eval_005000_final.json` 直接比较。明显偏离时先停下并提供日志。

本阶段保存 observer，使用新的本地 ID 编号；parent checkpoint 路径、哈希和父更新数写入 settings，不能把本阶段第 1000 步当作原训练第 1000 步。

## 渲染

在刚完成对应组的终端中执行；否则先把 `HKTEX_REPAIR_OUT` 设为该组实际输出路径。

```bash
env MPLBACKEND=Agg \
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
  python -u /root/hktex-work/hktex/hktex-image-training-observed/render_native_images.py \
  --run "$HKTEX_REPAIR_OUT" \
  --output "$HKTEX_REPAIR_OUT/native-512-spp64" \
  --size 512 --spp 64 --tile-size 64 --point-batch 512 \
  --views view_000,view_002
```

前向计算完全沿用原实现，因此现有 native 渲染与 observer 离线重放不需要改。

## 返回结果

每组：`repair_metrics.jsonl`、`metrics.jsonl`、`eval_000000.json`、`eval_001000_final.json`、`settings.json`，以及相同视角 raw 图片。失败时先发控制台报错。

同时看三件事：验证误差是否改善、表面负值截断是否减少、裤子颜色是否恢复。范围损失下降本身不代表图像拟合一定改善。

如果 range 显著优于 control，再考虑将范围约束用于从头训练，随后单独校准核覆盖和 ADC；不要直接把多个新参数混进一次 full。

## 验证及撤回

已做本地语法/CLI 检查，以及辅助损失的梯度数值检查、clamp 越界时恢复方向检查。本地没有 CUDA，尚未执行完整 HKTex GPU 训练。

不改原模型、viewer 或训练文件，也不覆盖父 checkpoint。回到原命令即可恢复原流程。
