# 图片训练 ADC 斑纹诊断

这个包不会修改训练代码或 checkpoint，也不会启动训练。它读取 checkpoint 中保存的网格、参数及配置，以及服务器上已有的特征分解缓存，生成独立诊断文件。

## 运行

将本目录放到 `/root/hktex-work/hktex/hktex-image-diagnosis`，激活原来的 `hktex` 环境：

```bash
conda activate hktex
HKTEX_RUN=/root/hktex-work/hktex/outputs/image-training/adc-20261025-034057-42031
HKTEX_DIAG="/root/hktex-work/hktex/outputs/image-diagnosis/adc-$(date +%Y%m%d-%H%M%S)"
mkdir -p /root/hktex-work/test-logs
set -o pipefail

env MPLBACKEND=Agg \
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
  python -u /root/hktex-work/hktex/hktex-image-diagnosis/diagnose_image_field.py \
  --repo /root/hktex-work/hktex \
  --checkpoint "$HKTEX_RUN/checkpoint.pt" \
  --output "$HKTEX_DIAG" \
  --points 2048 \
  2>&1 | tee /root/hktex-work/test-logs/image-field-diagnosis.log
```

首次执行不同分批/求导模式可能触发 Torch 编译，需要等待。输出目录必须是新目录。

自动寻找原配置记录的缓存：

```text
/root/hktex-work/hktex/outputs/image-training-cache/fd3514d3e0aa61843b4ab727/geometry_eigen_albo_principal_curvatures.pt
```

迁移服务器后若它移动了，可加 `--cache /实际路径/geometry_eigen_albo_principal_curvatures.pt`。缓存不存在时程序会停止，不会重新计算。旧 checkpoint 没有记录缓存历史哈希，因此只能记录本次使用的哈希，不能证明它与训练时完全相同。训练时的核心源码哈希不匹配也会停止。

## 返回哪些文件

- `diagnostic.json`：优先返回这个，包含热核覆盖、颜色截断、实际源点上的 λt 和前向一致性检查。
- `field_probe.npz`：保存同一批采样点、颜色、权重统计和各模式的对照颜色。需要逐点分析时再返回。
- 若失败：返回控制台日志与已写出的 `diagnostic.json`。

## 检查的含义

- 在网格上按面积固定采样 2048 个点，计算实际 top-k 核权重之和。权重和远小于 1 的区域可能主要依赖全局平均颜色；小于 1 本身并非错误。
- 检查组合后的颜色在最终 clamp 前是否越界。核的负残差颜色本身合法，不等于最终 RGB 错误。
- 检查实际插值的 λt 与 exp(-λt)，避免仅凭 diff_time 的绝对值推断扩散尺度。
- 同一批点对比缓存推理、训练式重新准备核、启用/禁用 Torch 求导，以及分批大小 512/64。这里只比较材质颜色的**前向值**，不验证路径追踪的梯度正确性，也不替代训练场景与 native 完整渲染的逐射线对照。
- 不使用 UV 真值、不优化参数、不改变 KNN 算法或特征分解。

## 已完成的本地验证及限制

已通过 Python 语法检查、命令行帮助检查，以及面积采样和数值差异汇总的 CPU 自检。当前本地环境没有 CUDA，尚未执行真实 HKTex GPU 前向；需要服务器上的这次诊断来获得结果。

`uploaded_analysis.json` 和 `loss_windows.png` 是根据本次上传的 1500 步日志与 checkpoint 生成的本地分析结果，和待运行的 GPU 诊断分开保存。

## 后续决策

1. 若同一点的前向结果明显随求导模式或分批改变，先定位该差异，不继续长训练。
2. 若前向一致，但存在较多弱覆盖/截断区域，先用相同预算、关闭 ADC 的短程对照校准扩散/阈值与学习率；这些参数不能直接照搬 UV 最优配置。
3. 若覆盖合理而拟合仍差，再做固定种子、固定像素/射线的 train/native 对照与少量视图过拟合测试，检查渲染与监督路径。
4. 确认短程拟合有效后，重新校准图像域 ADC 统计，再运行带 observer 的 full。

没有替换原文件，不需要回滚补丁。完成后可移除本诊断目录和输出目录。
