# HKTex 图片训练 + Observer / Native Viewer 适配版

版本 1.1.0，2026-10-25。参考 `Darli000000/my_dyn_hktex@8c926af44a3a12c80c3fdf1dd62e4f2ba5594630`。

本包与旧 `hktex-image-training`、`hktex-observer`、`hktex-native-viewer` **并列放置**。
不覆盖原目录，不修改 HKTex 的热核、优化器、ADC 或渲染核心代码。
复用旧 observer 的参数格式、谱分解与核谱系代码，以及 native viewer 的环渲染和 HTML；增加图片监督适配入口。
**旧版命令不能直接读取此新记录，使用下面的新命令。**

## 1. 安装

把 ZIP 上传到 `/root/hktex-work/hktex/`：

```bash
conda activate hktex
cd /root/hktex-work/hktex
unzip hktex-image-training-observed.zip
export MPLBACKEND=Agg
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
```

默认数据集仍是：

```text
/root/hktex-work/data/offline_multiview/smpl-male-full-20261025-003839-35340
```

不额外安装依赖，不使用原 OBJ / MTL / UV 纹理监督。
旧 v1.0.0 流程保持可用；撤回时改回旧目录里的运行命令即可。

## 2. 先看已完成 ADC 的清晰结果（不重新训练）

可以立即使用你已完成的 ADC 检查点：

```bash
python -u /root/hktex-work/hktex/hktex-image-training-observed/render_native_images.py \
  --run /root/hktex-work/hktex/outputs/image-training/adc-20261025-034057-42031 \
  --output /root/hktex-work/hktex/outputs/image-training/adc-20261025-034057-42031/viewer-native-512-spp64 \
  --size 512 --spp 64 --tile-size 64 --point-batch 512 \
  --views view_000,view_002
```

输出目录必须不存在。默认展示一个训练相机和一个验证相机。
打开输出目录的 `index.html`，或 `checkpoint_u01500/preview_views_raw.png` 和 `preview_views.png`。

- 每张图真正渲染为 512×512、64 spp；没有把原 128 图放大。
- 原始线性 RGB、未去噪 PNG、OptiX 去噪 PNG 都保留。
- GT 是保存的图片在线性空间做 box 平均，与数据集相机匹配；绝不再读取 UV 纹理产生 GT。
- 预测用训练时的 TorchTextureNetwork 与相同几何、法线、材质、光照/路径深度，通过 Mitsuba 原生 path 积分器渲染。
- kernel rings 沿用旧 native viewer 的 `box_border`，展示阈值响应带，不是固定大小圆点。
- 旧 ADC 没有保存稳定 ID，因此该入口只能为最终核按行分配展示颜色，不能补回历史谱系。
- 旧训练也没有保存特征缓存的历史 SHA；此入口记录当前缓存 SHA 并校验数据/源码，但不能声称验证了历史缓存未被改写。请使用原 ADC 对应的缓存。

若显存不足，先把 `--tile-size 64 --point-batch 512` 改为 `--tile-size 32 --point-batch 128`。
分块可能改变随机数序列和 KNN 并列候选，但不降低目标分辨率；不能期待逐像素相同。

512/64 的全图渲染比旧低采样预览慢。先渲染这两个视角，再考虑 1024/128；换设置要用新输出目录。
提高 spp 只减少采样方差，不能修复已经学到的纹理缺陷或 KNN 不稳定性。若未去噪的高采样图仍有明显斑点，应继续排查，不能靠去噪宣布问题解决。

## 3. 用 50 步检查新的记录接口

```bash
bash /root/hktex-work/hktex/hktex-image-training-observed/run_training.sh smoke
```

这次默认每 10 步保存快照；其余 smoke 训练设置与原来相同。
终端打印 `[OUTPUT]` 和 `[OBSERVER]`。假设新的实验目录为 `$RUN`，设置为终端实际路径：

```bash
RUN='/替换为终端OUTPUT打印的新smoke目录'
python /root/hktex-work/hktex/hktex-image-training-observed/image_observer.py \
  --recording "$RUN/observer" --checkpoint "$RUN/checkpoint.pt"
```

应看到 `passed: true`、`final_matches_checkpoint: true`；检查快照文件哈希、参数尺寸、稳定 ID、ADC 谱系，以及最终快照和检查点参数逐元素相等。
smoke 关闭 ADC，所以本次没有 ADC 事件是正常的。

建议在 full 之前再做一次低成本读取验证：

```bash
python -u /root/hktex-work/hktex/hktex-image-training-observed/render_native_images.py \
  --recording "$RUN/observer" --snapshots final \
  --output "$RUN/viewer-native-check" \
  --views view_000 --size 128 --spp 4 --tile-size 32 --no-denoise

python -u /root/hktex-work/hktex/hktex-image-training-observed/replay_image_fields.py \
  --recording "$RUN/observer" --snapshots 0 --chunk 64
```

数学回放会检查谱项、过滤/混色重建与原前向的数值一致性；只回放一个快照以验证接口。
其中 `0` 是快照序号，不是训练步数。`--snapshots all` 包括 ADC 事件，`regular` 只包含常规快照。

## 4. 开始带记录的 full

上述验证通过后：

```bash
bash /root/hktex-work/hktex/hktex-image-training-observed/run_training.sh full
```

从头训练 5000 步，初始 1000 核，每步 256 像素，每像素 4×4 光路，启用原 ADC，最多 5000 核。
输入图片仍为 2048×2048；这不是对每张图片完整训练 5000 遍，也不保证已充分收敛。
每 250 步保存 observer 快照，额外记录 ADC 决策及实际操作前后状态。
**full 默认取消训练末尾旧的小尺寸 preview；高质量展示在训练结束后独立生成，避免展示失败影响训练交付。**

新实验目录会包含：

```text
settings.json / trainer.yaml / renderer.yaml
checkpoint.pt                    # 最近一次正常完整状态，含 observer 稳定 ID
texture_float32.pt
metrics.jsonl / eval_*.json / report.json
observer/
  manifest.json                  # 模型/谱配置、缓存与源码指纹、数据集引用
  surface.npz                    # 几何、法线、固定采样点；没有 UV GT
  snapshots.json
  snapshots/*.npz                # 参数、核 ID/root ID、已有 ADC 累积缓冲区
  samples/*.npz                  # 定期保存抽样像素、相机 ID、目标/预测线性 RGB、seed
  steps.jsonl
  events.json                    # clone/split/prune 前后快照及 parent/child ID
  decisions.jsonl                # 实际进入控制器决策时的配置与缓冲快照
```

记录器不额外执行 forward/backward、不渲染、不使用训练随机数发生器；有 CPU 拷贝和磁盘写入开销。
快照用于回放，不是完整优化器检查点；恢复训练继续使用 `checkpoint.pt`。
失败的 ADC 事件保留 started 意图，不伪造 after 状态。

如果只想先测带记录的 ADC，可用本包 `run_training.sh adc`；本包不改变已有 ADC 调度。

## 5. full 完成后生成 native viewer

```bash
RUN='/替换为新的full实验目录'
python /root/hktex-work/hktex/hktex-image-training-observed/image_observer.py \
  --recording "$RUN/observer" --checkpoint "$RUN/checkpoint.pt"

python -u /root/hktex-work/hktex/hktex-image-training-observed/render_native_images.py \
  --recording "$RUN/observer" --snapshots final \
  --output "$RUN/observer/viewer-native-512-spp64" \
  --size 512 --spp 64 --tile-size 64 --point-batch 512 \
  --views view_000,view_002,view_012,view_014
```

这里用真实数据集相机，包含训练与验证视角。HTML 中显示记录的具体相机 ID；不是旧版任意旋转相机的伪 GT。
要看完整演化，把 `--snapshots final` 换为 `--snapshots all` 并换新输出目录；耗时随快照和视角数量增加。
也可传指定快照序号，如 `--snapshots 0,3,6`，先查看少数阶段。
保留每个视角的线性数组与 raw 图；`image_errors.json` 是当前展示分辨率下有效区域的原始 MSE，不是原 2048 稀疏评估指标，不应混为一谈。

新 viewer 暂不导出旋转视频；当前目标是与实际输入视角严格对应的图像与热核回放。

## 6. 原 observer 的数学诊断适配

复用旧谱项/距离/权重/颜色贡献的计算和绘图，移除无效的 UV GT / 表面拟合误差。

```bash
python -u /root/hktex-work/hktex/hktex-image-training-observed/replay_image_fields.py \
  --recording "$RUN/observer" --snapshots regular --chunk 64

python /root/hktex-work/hktex/hktex-image-training-observed/plot_image_fields.py \
  --recording "$RUN/observer"
```

生成 `observer/analysis` 数值文件与 `observer/viewer/index.html`，展示：

- 热核 raw、自热核归一化、距离加权、过滤响应及 KNN/Top-K 门控；
- 特征值、谱项、累计谱和、双调和距离分量、参数网格插值；
- 颜色贡献、核参数分布、稳定核身份及 ADC 家族关系。

这里的表面颜色是模型 base colour，图片目标是带光照的 radiance，二者不能直接相减。
因此本版不伪造表面 GT、表面 PSNR 或 MSE。实际拟合评估继续看训练目录的 `eval_*.json`。
数学 viewer 的总览仍是明确标注的稀疏采样点图，只用于定位诊断；看画质请用 native viewer。

## 7. 恢复与撤回

完整状态恢复（使用同一数据集、源码与原训练设置，输出目录必须不存在）：

```bash
python -u /root/hktex-work/hktex/hktex-image-training-observed/train_images.py resume \
  --checkpoint "$RUN/checkpoint.pt" \
  --output /root/hktex-work/hktex/outputs/image-training/resumed-observed-v1 --gpu 0
```

新 checkpoint 会恢复稳定 ID / root ID / next ID，并在新 observer 中记录父记录路径。
新目录只包含恢复后的快照；旧记录保留在旧目录。旧 v1.0 检查点没有谱系时，仅能从恢复时开始分配 ID，无法补回历史。
不要把 smoke 的 resume 当成 full；它继续原设置，不延长原计划。

撤回：继续用 `/root/hktex-work/hktex/hktex-image-training/run_training.sh` 即可。新旧目录和记录独立。

## 8. TF32 提示与验证范围

TF32 警告只提示可用的加速选项；本任务沿用 `allow_tf32=False`。不需要升级依赖或为了消除警告开启 TF32。
改变它可能改变数值及近邻选择，不宜在比较 observer 和训练结果时顺手更改。

CPU 测试见 TESTING.md。**这次新增的 observer 与 GPU native/math 回放尚需服务器 smoke 验证，不能因为旧图片训练成功就直接声称新接口已通过 CUDA 集成测试。**
