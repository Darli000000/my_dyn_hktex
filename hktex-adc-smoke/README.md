# HKTex ADC 双损失 smoke（仅功能测试）

适用已核对的仓库版本：`Darli000000/my_dyn_hktex`，main 提交 `b02d07ee7e8a7a0d127ee6ef0dd74bdda906b51d`。新目录独立，不覆盖旧训练脚本、旧 paired 结果或 observer。

本包先验证 ADC 与 checkpoint/observer 的集成；不包含完整训练入口。**smoke 有显式结构探针，不能续接成完整实验或用它的图片评价 ADC 质量。**

## 1. 本次 smoke 设置

两组：`linear`、`loggamma`。两组加载同一份新的 5000-kernel 初始化，图片/像素/渲染种子序列一致。每组训练 450 次更新，分 0→250、250→450 两个独立进程，自动测试变更核数后的续训。

训练仍使用原 2048 线性 RGB 数据，batch=256，AA=4，transport=4。沿用 5000 步学习率调度，避免短 smoke 提前把学习率衰减完。

| 更新号（从 1 开始） | 操作 |
|---|---|
| 10 | 显式克隆 1 个核，验证参数、优化器及 observer 更新 |
| 20 | 显式分裂 1 个核；使用较短的 0.005 分裂半径作为功能探针 |
| 30 | 显式删除 1 个核 |
| 200 | 按累计误差与阈值自然检查增密；可克隆或分裂，也可能没有候选 |
| 250 | 保存 checkpoint，退出，再恢复到新输出目录 |
| 400 | 同一步先自然检查剪枝，再自然检查增密，覆盖两控制器连续修改核的情况 |
| 401～450 | 不再执行结构变化，继续优化，验证变更后的训练 |

自然增密间隔、累计窗口都是 **200**；剪枝间隔及累计窗口都是 400。核数范围 4900～5500，每次自然增密净新增上限约为当前核数的 2%。自然 split_radius=0.05。

初始自然阈值沿用当前配置：error_threshold≈0.02376188、size_threshold≈0.18052526、selection_threshold≈0.00249263、contrib_threshold≈0.02046397。它们只是本次观察起点，**不保证一定产生自然增密**。脚本保存误差分位数、候选数、剩余容量，供后续校准。

自然阈值选择与显式探针分开记录：`origin=natural` 与 `origin=forced-smoke-probe`。显式探针调用原控制器的真实 clone/split/remove 方法，不修改参数文件伪造通过。

## 2. 两组 ADC 使用什么误差

- 反向传播：linear 用线性 RGB mean L1；loggamma 用之前的 clip/log/gamma mean L1（含 q>=1e-12 数值保护）。
- **两组 ADC 都用原始线性 RGB mean absolute error**，通过像素中心第一交点的核贡献分配到核。
- 因而阈值含义一致；不是把 loggamma 变换后的误差直接套在线性阈值上。
- 两组的颜色会逐渐不同，ADC 候选、实际核数和布局也可以不同；输入流必须保持一致。
- 这是两种 loss 在同一 ADC 规则下的对照，不是严格复刻作者原始 multi-view pipeline。

位置、法线、相机、光照、材质、图片目标及热核核心计算沿用现有图片 trainer。复用已有原生 viewer 与数学 observer。两组先后执行，不同时占用 GPU。

## 3. 安装与运行

将 `hktex-adc-smoke.zip` 上传到 `/root/hktex-work/`：

```bash
conda activate hktex
cd /root/hktex-work/hktex
python -m zipfile -e /root/hktex-work/hktex-adc-smoke.zip /root/hktex-work/hktex

ADC_PAIR=/root/hktex-work/hktex/outputs/image-training/adc-smoke-pair-v1
bash /root/hktex-work/hktex/hktex-adc-smoke/run_smoke.sh "$ADC_PAIR"
```

默认数据集：

```text
/root/hktex-work/data/offline_multiview/smpl-male-full-20261025-003839-35340
```

可在首次启动前通过 `HKTEX_REPO`、`HKTEX_DATASET` 修改路径。输出目录必须不存在；失败后保留原目录，下次使用新名字，或者按第 6 节只重跑失败的一段。

脚本自动设置 MPLBACKEND/LD_PRELOAD，包含已有 face_ids 恢复修复，不需要编辑旧 `run_pair.sh`。源码不匹配时会停止；请提供列出的差异，不要删除校验。

## 4. 输出与通过标准

```text
adc-smoke-pair-v1/
  initial.pt
  initial_identity.json
  logs/
  s00250/
    linear/
    loggamma/
  s00450/
    linear/
    loggamma/
  smoke_comparison.json
```

每组每段包括：

- `trainer.yaml`、`settings.json`：最终生效配置；`steps=5000` 表示调度长度，不是 smoke 实际执行步数。
- `metrics.jsonl`：loss、共同线性误差、梯度、输入摘要、实际核数。
- `adc_actions.jsonl`：真实结构变化，来源、选择数量、变更前后核数、参数和优化器状态检查结果。
- `adc_decisions.jsonl`：自然决策、候选数、误差/贡献分布；第一段主要为第 200 步，第二段为第 400 步。
- `field_metrics.jsonl`：固定表面探针的颜色截断比例。
- `observer/`：常规及 ADC 前后快照、lineage、采样记录。
- `report.json`、`startup.json`；第二段另有 `resume_repair.json`。

最终检查读取两个阶段、两组全部日志，验证：

1. 450 步连续完成，损失/梯度有限，输入序列一致。
2. 三种显式结构操作实际完成，核数改变正确。
3. 自然增密检查出现在 200/400，自然剪枝检查出现在 400。
4. 核数在预算内；变更后的优化器引用、moment 张量形状、face_ids、累计 buffer 和 observer lineage 一致。
5. 第 250 步后的恢复检查通过。

结果解释：

- `status: passed`：功能检查通过，两组都观察到自然增密与自然剪枝（不要求自然 clone 和 split 都发生）。
- `status: functional_passed_natural_activity_inconclusive`：功能与显式探针通过，但至少一组没有自然增长或没有自然剪枝。不是崩溃，也不能直接称 ADC 质量测试通过；需要分析阈值和候选统计。
- 异常退出：保留日志，不绕过引用、形状、非有限数或几何检查。

首先发回 `smoke_comparison.json`。若无法生成，请发失败日志和对应阶段的 `report.json` / `startup.json`；已有 `adc_actions.jsonl`、`adc_decisions.jsonl` 也一起发。

## 5. 生成图片（可选）

```bash
ADC_PAIR=/root/hktex-work/hktex/outputs/image-training/adc-smoke-pair-v1
bash /root/hktex-work/hktex/hktex-adc-smoke/render_smoke.sh "$ADC_PAIR"
```

生成两组第 450 步 final 快照的原生 viewer：512、128 spp、不开去噪，包含两视角 prediction/GT/核环及 HTML。终端打印 `[VIEWER]` 路径。

本次不必等图片生成就能先分析功能日志。由于是短 smoke 且有显式探针，图片不能与此前 5000 步质量直接比较。

## 6. 失败后单独重跑一段

如果某段失败，只备份该段对应 arm 的输出目录，再用本目录 `run_arm.py` 单独启动。两组必须使用同一 kit 和 pair 目录，不要临时修改阈值继续其中一组。

例如 linear 的第二段失败，而它的 250 步 checkpoint 已成功保存：

```bash
ADC_PAIR=/root/hktex-work/hktex/outputs/image-training/adc-smoke-pair-v1
mv "$ADC_PAIR/s00450/linear" "$ADC_PAIR/s00450/linear-failed-$(date +%Y%m%d-%H%M%S)"
env MPLBACKEND=Agg \
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
  python /root/hktex-work/hktex/hktex-adc-smoke/run_arm.py \
  --arm linear --pair "$ADC_PAIR" --until 450 \
  --resume "$ADC_PAIR/s00250/linear/checkpoint.pt" \
  --output "$ADC_PAIR/s00450/linear"
```

若另一组第二段尚未启动，同样运行它的第二段命令（arm/output/checkpoint 都换成 loggamma，无需执行不存在目录的 mv）。两组全部完成后：

```bash
python /root/hktex-work/hktex/hktex-adc-smoke/compare_smoke.py --pair "$ADC_PAIR"
```

## 7. 下一轮完整实验的计划（尚未执行）

smoke 通过后准备新的完整 pair，从共享新初始化开始，不沿用 smoke 的显式变更结果。

| 参数 | 完整实验计划 |
|---|---:|
| loss | linear / loggamma |
| 总更新 | 5000 |
| 初始核数 | 5000 |
| 自然增密开始 | 1000（20%） |
| 增密间隔 / 累计窗口 | **200 / 200** |
| 剪枝开始 | 1500（30%） |
| 剪枝间隔 / 累计窗口 | 500 / 500 |
| 停止 | update >= 4000，不再改变结构 |
| 核数上限 | 暂定 8000，结合 smoke 显存/统计再确认 |
| 单次增密比例上限 | 暂定 5% |
| 显式 clone/split/prune 探针 | **关闭** |

完整实验还需校准自然阈值与剪枝预算。当前 smoke 中的时间压缩只用于快速验证。完整入口不会通过简单把本脚本的 450 改成 5000 来实现。

## 8. 改动和回退

只新增 `hktex-adc-smoke/`：

- `run_arm.py`：图片训练的 ADC 步骤、双损失、阶段恢复、observer。
- `adc_support.py`：smoke 策略、显式探针、自然决策统计、状态一致性检查。
- `resume_support.py`：face_ids 恢复后重新绑定当前模型 buffer。
- `compare_smoke.py`、`run_smoke.sh`、`render_smoke.sh`：执行与验收。

原项目文件未修改。回退就是停止使用新入口，旧 fixed-kernel 实验不受影响。

本地已通过 CPU 检查，包括执行原控制器方法的克隆/分裂/剪枝、混合 clone+split、Adam 状态形状、变更核数后的状态保存/加载、累计 buffer 恢复、GeodesicOpt 引用修复和实际 make_config 的策略覆盖。CPU 几何测试使用平面追踪替身；本地无 CUDA，未验证服务器的 digeo CUDA 路径或完整训练性能。服务器本次 smoke 正是用于完成这些集成验证。
