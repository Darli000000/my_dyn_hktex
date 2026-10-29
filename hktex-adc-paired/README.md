# HKTex 完整 ADC 双 loss 实验

## 推荐：一条命令全部完成

按下方安装步骤解压后，执行：

```bash
ADC_PAIR=/root/hktex-work/hktex/outputs/image-training/adc-full-pair-v1
bash /root/hktex-work/hktex/hktex-adc-paired/run_pair.sh all "$ADC_PAIR"
```

自动运行两组前 2000 步、保存和核对状态、接着完成两组至 5000 步，然后生成汇总和两组 1024/256 spp 原生 viewer 图片。第 2000 步无需人工确认；自然增密候选不足只记录为 inconclusive，不阻止 all 模式继续。实际运行错误、非有限数或配对/状态校验失败仍会退出。

中间分段只用于保存和自动恢复；两组各执行 5000 次更新，学习率计划不变。参数和训练逻辑与前一包一致。输出目录必须是新目录。后面的 pilot/finish 是可选的手动分段方式，不是 all 模式的必要步骤。

适用已检查的仓库 `Darli000000/my_dyn_hktex`，提交 `b02d07ee7e8a7a0d127ee6ef0dd74bdda906b51d`，与上一份 ADC smoke 使用同一源码校验表。只新增 `hktex-adc-paired/`，不覆盖原项目代码、旧实验或 smoke。

## 为什么可以继续

收到的 smoke_comparison.json 为 passed：两组强制 clone/split/prune、实际自然分裂/剪枝、变更核数后恢复和输入配对均通过。两组各自然分裂 1、剪枝 102，最后 4900 核。自然 clone=0 不代表 clone 实现失败；选择 clone 还是 split 取决于尺寸判据。

这证明功能链路可继续测试，尚不能证明 ADC 提升质量。旧 error_threshold=0.02376188，在第 400 步高于全部误差（最大值约 0.0142/0.0145），没有增密候选。最终 4900 是剪枝下限约束，不能解释为最优容量。

本轮采用固定 0.008 作为保守的新阈值假设。第 400 步 p99 约 0.0064/0.0070，仅提供量级参考；不能预测第 1000 步一定有多少候选。因此保存第 2000 步诊断。all 模式自动继续，训练后统一分析活跃程度；续训仍沿用最初的 5000 步学习率调度。

## 参数与实验含义

| 设置 | 本轮值 |
|---|---|
| 两组反向 loss | linear / loggamma，沿用已通过的 paired 实现 |
| 初始化 | 同一份新生成的 5000 核状态；不使用 UV 颜色初始化 |
| 总更新 | 5000；先 0→2000，再 2000→5000 |
| 增密 | 1000 开始，每 200 步，累计窗口 200 |
| 剪枝 | 1500 开始，每 500 步，累计窗口 500 |
| 停止结构变化 | update >= 4000；最后增密 3800，最后剪枝 3500 |
| 增密阈值 | 0.008，两组固定一致 |
| 单次增密净新增上限 | 当前核数的 5%，受总上限约束 |
| 核数预算 | 4500～8000；8000 是上限，不是目标 |
| split_radius | 0.05，沿用 smoke 自然分裂半径 |
| size_threshold | 0.1805252636362324，沿用配置 |
| selection_threshold / contrib_threshold | 0.002492630717212829 / 0.020463972379877828，沿用配置 |
| 显式结构探针 | 无；clone/split/prune 只由自然规则触发 |
| batch / AA / transport | 256 / 4 / 4 |
| 评估 | 固定 1024 个前景像素/视角；每 250 步 train/val 子集，最终包含 test |
| 数据 | 原 2048×2048 线性 RGB 数据，不缩成预览图训练 |
| observer | 每 200 步，结构变化前后及阶段 final 都保存 |

所有更新号从 1 开始。增密首次累计 801～1000，剪枝首次累计 1001～1500；避免把窗口起点多算一次。同一步先剪枝再增密，操作之间刷新几何及优化器引用。继续复用原控制器的参数变更、克隆、分裂和剪枝实现。

两组 ADC 都用未经过 loggamma 的线性 RGB 误差和相同阈值；只有反向优化的标量 loss 不同。两组参数逐渐不同后，实际核数与布局允许不同。保存并核对两组的相机、像素和采样种子序列。8000 核对本机显存是否足够仍需实际运行判断；peak_torch_cuda_gb 只统计 PyTorch 分配，不能代表 Mitsuba/FAISS 等全部显存。

本轮是图片 trainer 下的 ADC 探索，不是作者原始 multi-view 训练的严格复现。也没有把此前固定核实验的 initial.pt 强行接入，不能宣称与旧 baseline 已做过逐张量初始化一致性验证。保留旧 baseline 作参考；若需要严格归因 ADC 收益，后续应再跑共享本轮初始化的 ADC-off 对照。

## 安装

上传 `hktex-adc-paired.zip` 到 `/root/hktex-work/`：

```bash
conda activate hktex
cd /root/hktex-work/hktex
python -m zipfile -e /root/hktex-work/hktex-adc-paired.zip /root/hktex-work/hktex
```

默认数据集为 `/root/hktex-work/data/offline_multiview/smpl-male-full-20261025-003839-35340`。默认项目为 `/root/hktex-work/hktex`；可用 HKTEX_DATASET / HKTEX_REPO 设置路径。源码不匹配时请保留提示，不删除校验。

## 可选：手动分段运行前 2000 步

必须使用新 pair 目录，不能续接 smoke 或以前 ADC-off 的 checkpoint。

```bash
ADC_PAIR=/root/hktex-work/hktex/outputs/image-training/adc-full-pair-v1
bash /root/hktex-work/hktex/hktex-adc-paired/run_pair.sh pilot "$ADC_PAIR"
bash /root/hktex-work/hktex/hktex-adc-paired/render_pair.sh "$ADC_PAIR" 2000
```

两组依次执行。结果为 `s02000/linear/` 和 `s02000/loggamma/`，汇总为 pair 根目录的 `comparison_02000.json`。第 2000 步已覆盖增密 1000/1200/1400/1600/1800/2000，剪枝 1500/2000，包含一次同一步先剪枝再增密。

建议首先检查：

- status=passed，输入配对、状态检查通过，且两组有真实自然增长和剪枝。
- decisions 内误差分位数、候选数、实际动作数、核数预算是否合理；不能只看最终核数。
- 验证视角没有显著新色块、几何破坏，误差没有持续恶化。

若自然活动为 inconclusive，训练完成不等于故障，但不足以评价增密；不要直接调低阈值续训某一组。先分析 comparison_02000.json，再用新的 pair 同步改变两组设置。脚本不会自动调阈值或强制制造增密。

## 可选：手动续到 5000 步

确认前段结果合理后：

```bash
ADC_PAIR=/root/hktex-work/hktex/outputs/image-training/adc-full-pair-v1
bash /root/hktex-work/hktex/hktex-adc-paired/run_pair.sh finish "$ADC_PAIR"
bash /root/hktex-work/hktex/hktex-adc-paired/render_pair.sh "$ADC_PAIR" 5000
```

finish 会校验前段并要求两组都有自然增长和剪枝；它无法自动判断图片质量。恢复本组第 2000 步参数、Adam/GeodesicOpt、学习率、累计 buffer、随机状态及 observer lineage，到 5000 为止，不是再训练 5000 次。初始就按 5000 步设置 scheduler，续训不重置。

最终汇总为 `comparison_05000.json`，包含两组 ADC 活动、核数范围和 train/val/test 的固定像素评估。该评估不是完整图片的 PSNR。只有对照指标和图像支持时，才能判断 ADC 比固定核更好。

## 图片、observer 与文件

render_pair.sh 复用项目 `hktex-image-training-observed/render_native_images.py`，输出 1024、256 spp、不开去噪的原生渲染。默认训练视角 view_000 和验证视角 view_002，包含 GT/prediction/核环和 HTML。终端打印 [VIEWER]；输出位于对应阶段/arm 的 `native-1024-时间戳/`。

每段 `observer/` 保留之前兼容的记录格式，可以交给既有数学 observer/replay 流程；本包不重写 viewer。默认命令渲染阶段 final，其他训练快照保留在 observer 中。

建议回传：comparison_02000.json 或 comparison_05000.json，以及两组原始 prediction 图。需要细查时再提供对应 arm 的 adc_decisions.jsonl、adc_actions.jsonl、metrics.jsonl、field_metrics.jsonl。settings.json/trainer.yaml 是生效配置；本包的 adc_support.py 顶部 POLICY 是本轮 ADC 覆盖参数来源。不要只改原配置 YAML 期待覆盖这些值。

## 失败后恢复与回退

不删除旧结果。若第二组或后半段失败，已完成的组不必重跑。此入口限定从已完成的 2000 checkpoint 重启后半段（不支持任意中间 checkpoint 的阶段自动拼接）。例如仅 loggamma 后半段失败：

```bash
ADC_PAIR=/root/hktex-work/hktex/outputs/image-training/adc-full-pair-v1
mv "$ADC_PAIR/s05000/loggamma" "$ADC_PAIR/s05000/loggamma-failed-$(date +%Y%m%d-%H%M%S)"
env MPLBACKEND=Agg LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
  python /root/hktex-work/hktex/hktex-adc-paired/run_arm.py \
  --arm loggamma --pair "$ADC_PAIR" --until 5000 \
  --resume "$ADC_PAIR/s02000/loggamma/checkpoint.pt" \
  --output "$ADC_PAIR/s05000/loggamma"
python /root/hktex-work/hktex/hktex-adc-paired/compare_pair.py --pair "$ADC_PAIR" --until 5000
```

若目录未创建，不执行 mv。若前半段失败，用同样的单组入口，until=2000、output=s02000/该组，并省略 --resume，从共享初始状态重跑该组。

不修改原项目文件，回退只需停止使用新入口。不能在两阶段之间更改本包、loss 或参数；哈希校验会拒绝不一致的恢复。改变策略应创建新实验。

## 已做检查及限制

本地 6 项 CPU 控制器/配置测试与 5 项比较验收测试通过：原控制器实际混合 clone+split、剪枝后继续增密、预算、Adam 状态形状、累计 buffer 保存/加载、GeodesicOpt 引用、严格累计窗口、停止时间、真实配置加载、输入不配对/缺失事件/强制探针的拒绝。Python/Bash 语法和 104 个依赖源文件哈希检查通过。

CPU 几何测试使用平面追踪替身。完整包尚未在 CUDA 上运行；之前 smoke 验证了服务器上的基础 GPU 集成，新阈值和更大核数的训练稳定性、显存与质量由本轮实测确定。
