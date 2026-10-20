# HKTex 训练记录与离线可视化 v1.0

适用于当前已通过 SMPL ADC 测试的 HKTex 环境。新增功能分为：**训练保存数据 → CUDA 数值回放 → CPU 制图/HTML**。不修改 HKTex 源文件；此前的 SVD、prune/split 修复保留。

本地已验证 CPU 数值拆解、记录开关前后的参数/Adam 状态/随机数一致性、实际控制器代码的 ID 顺序、命令解析和图表生成。**尚未在你的服务器上验证 CUDA 回放；请先完成下面的短测试。**

## 1. 放置文件

将 `hktex-observer.zip` 上传到 `/root/hktex-work/hktex/`：

```bash
conda activate hktex
cd /root/hktex-work/hktex
python -m zipfile -e hktex-observer.zip .
python hktex-observer/test_observer.py --repo /root/hktex-work/hktex
```

最后应出现 `CPU TESTS PASSED`。使用现有环境中的 torch、numpy、trimesh、Pillow、OmegaConf、matplotlib；无需升级训练依赖。

该测试读取仓库实际的 clone/split/prune、谱热核和颜色归约代码，在 CPU 测试夹具上运行。它不替代下面的真实 GPU 测试。

## 2. 先验证新增记录与回放功能

```bash
python hktex-observer/run_observed.py \
  --phase smoke \
  --record-every 25 \
  --eval-points 2048
```

默认模型：

```text
/root/hktex-work/data/SMPL_texture_samples/Textured_Mesh_samples/SMPL_male.obj
```

如模型路径不同，追加 `--mesh /你的路径/模型.obj`。默认仓库 `/root/hktex-work/hktex`，可用 `--repo` 改。启动器自动设置 Agg、Conda libstdc++ 预加载和 CUDA 同步检查，自动保存日志，并保留子进程退出状态。

训练日志会输出：

```text
[OBSERVER] Recording directory: /root/.../observer
```

将下面变量替换为**本次打印的实际路径**：

```bash
HKTEX_RECORDING='/root/这里替换为实际实验目录/observer'
```

然后运行离线数值回放。这一步仍需要原来的 CUDA 环境，但不训练、不调用优化器、不重新求解特征分解：

```bash
env MPLBACKEND=Agg \
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
  python hktex-observer/replay.py \
    --recording "$HKTEX_RECORDING" \
    --repo /root/hktex-work/hktex \
    --snapshots regular \
    --chunk 64
```

再生成图片、GIF 和 HTML（这一步只用 CPU）：

```bash
python hktex-observer/plot_recording.py --recording "$HKTEX_RECORDING"
```

成功标准：

- 训练退出状态为 0，原 V4 检查报告成功。
- `observer/manifest.json` 中 `status` 为 `optimisation_completed`。
- 回放完成，无颜色拆解一致性错误；`analysis/metrics.json` 中每个 `max_forward_decomposition_error <= 5e-5`。
- `viewer/index.html` 可打开，GT、预测、误差图和热核面板有内容。

`manifest` 的状态只表示 optimise() 完成；训练结束后的原有渲染是否成功还要看日志与进程退出状态。

## 3. 完整训练

短测试和回放通过后：

```bash
python hktex-observer/run_observed.py \
  --phase full \
  --record-every 250 \
  --eval-points 8192
```

如想先验证新增记录器在真实 GPU 上处理 ADC 事件，可把 `--phase full` 换成 `--phase adc`：1200 步会覆盖增密、克隆、分裂及剪枝。你原来的 ADC 已通过，这一步主要验证新记录器的事件快照。

| 设置 | smoke | adc | full |
|---|---:|---:|---:|
| 迭代数 | 100 | 1200 | 5000 |
| 初始核数 | 256 | 1000 | 1000 |
| 最大核数 | 固定 | 5000 | 5000 |
| batch size | 512 | 4096 | 4096 |
| 重要性采样 / ADC | 关闭 | 开启 | 开启 |
| 最终渲染 | 128² / 4 帧 / 4 spp | 256² / 12 帧 / 8 spp | 512² / 60 帧 / 16 spp |

沿用之前的 SMPL 配置：seed=0、初始化随机比例 0.5、位置 LR=0.009914527998776132、momentum=0.9；其他学习率沿用仓库 YAML。ADC 使用固定的 5000 步实验日程，避免缩短测试时压缩日程：densify 从 400 步起、间隔 100、停止值 3500、误差累计 75；prune 从 500 步起、间隔 1000、停止值 2500、累计 250。边界处是否执行由原控制器决定。

这是新开的一次完整训练，不是从之前 ADC 测试接着训练。

仅生成配置供检查，不启动训练：

```bash
python hktex-observer/run_observed.py --phase full --prepare-only
```

完整训练结束后，用本次的 `observer` 路径再次执行第 2 节的回放、制图命令。

## 4. 怎么看结果

把整个 `observer/viewer/` 文件夹下载到本地，双击 `index.html`。保留同目录 PNG/GIF；只下载 HTML 不够。页面没有外部资源依赖。

| 面板 | 用途 |
|---|---|
| 总览 | 固定均匀表面采样点的 GT、预测、误差、核中心，正反两面 |
| 热核逐阶段响应 | 原始谱和、自响应归一化、距离加权、soft-step 后 alpha、KNN 选择状态 |
| 谱项与累加 | λ、exp(-λt)、源点/目标点的 φ、逐项乘积及累加 |
| 双调和距离 | 各谱维度对距离平方的贡献、累加、距离权重，及与欧氏距离的空间对比 |
| 两种距离选邻居 | 固定目标点的双调和 KNN 与同数量欧氏近邻对比 |
| 颜色贡献 | 选中核对 RGB 的带符号贡献、基础颜色、clamp 前后颜色 |
| 参数网格插值 | 当前角度/各向异性对应的真实 4 个网格索引及权重 |
| 参数分布 | 角度、各向异性、阈值、sharpness 分布 |
| 家族变化 | 稳定核 ID 与 clone/split/prune 父子关系 |

页面底部另有训练曲线、预测演变 GIF 和 ADC 事件明细。**这是固定表面点的投影图，不是有光照的网格渲染或交互式 3D 查看器。** 原训练结束时的渲染仍保留在原实验输出中。

训练期间新增记录器不执行中间 forward，也不调用原渲染器。原 V4 的初始化/最终预览行为保持原样。保存数据仍有 CPU 拷贝、压缩和磁盘 I/O 开销；不要设置每步保存。

## 5. 数学面板的含义

当前配置的扩散时间 `t = 0.06245046192097831` 固定；动画横轴是**优化迭代**，不是逐渐增大的物理扩散时间。

实际计算链为：

```text
核的角度/各向异性 → 4 个参数网格权重
→ 插值得到源点与目标点的谱数据
→ raw = Σ exp(-λ_j t) φ_j(source) φ_j(query)
→ weighted = raw × distance_weight / (source_self_heat + eps)
→ soft-step(weighted^power, threshold, sharpness)
→ 外层 KNN 候选内按 alpha 取 inner top-k
→ 基础颜色 + Σ(alpha × kernel_RGB) / max(Σalpha, 1)
→ clamp 到 [0, 1]
```

双调和距离使用各向同性谱构造的嵌入，当前是去掉第一个模态后的 64 维 `φ_j/max(λ_j, 1e-8)`；距离平方是嵌入差的平方和。它与计算各向异性热核的谱数据用途不同。

- 热核场面板保留负值，不把负的截断谱响应擅自改成零。
- 所有快照的热核场共享色标；误差图也共享色标。距离地图用自己的色标。
- 图中的模态顺序沿用实际对齐/插值后的顺序，不能一概视为严格从低频到高频排列。
- 数学面板固定目标点，展示该时刻 alpha 最大的选中核；核可能随训练切换，标题会显示稳定 ID。这不等于 RGB 贡献绝对值最大。
- 默认追踪 6 个空间分散的初始核家族。父核分裂后选取该家族 ID 最小的存活后代展示，不把它冒充原父核；全家族消失时不再显示该代表核。
- `gate=0/1/2` 分别代表不在候选中、进入外层候选、最终被 inner top-k 选中。直接画出的代表核热核场不等于其处处参与了实际预测。
- 训练 loss 对 RGB 求和，固定点 MSE 对 RGB 求平均，两者采样分布也不同，不宜直接比较数值大小。

回放调用仓库原来的谱插值、KNN、热核求值和过滤函数；颜色拆解还与原始 forward 输出比较，超出容差就停止，不静默生成似是而非的图。

## 6. 细看一次剪枝/克隆/分裂

默认 `--snapshots regular` 回放初始、周期和最终状态。ADC 决策及每次变更前后均已额外保存，不受 `--record-every` 限制。

先查 `observer/events.json`：其中 `before`、`after` 指向具体快照。快照文件名前五位是序号，例如 `00012`。只回放某一对前后快照到独立目录：

```bash
# 示例序号；用 events.json 中本次事件的真实序号替换。
env MPLBACKEND=Agg \
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
  python hktex-observer/replay.py \
    --recording "$HKTEX_RECORDING" \
    --snapshots 12,13 \
    --output "$HKTEX_RECORDING/analysis-event"

python hktex-observer/plot_recording.py \
  --recording "$HKTEX_RECORDING" \
  --analysis "$HKTEX_RECORDING/analysis-event" \
  --output "$HKTEX_RECORDING/viewer-event"
```

也可用 `--snapshots all`，但计算与制图时间会增加。仅事件回放没有常规训练动画帧，页面的动画区域会隐藏。

更换目标点面板（默认保存前 8 个固定点的完整算术）：

```bash
python hktex-observer/plot_recording.py \
  --recording "$HKTEX_RECORDING" --probe 3 \
  --output "$HKTEX_RECORDING/viewer-probe3"
```

回放时可用 `--roots 0,7,20` 指定初始稳定 ID，建议配合新的 `--output`。若需要更多目标点算术，回放加 `--probes 16`，并保持 `--chunk >= --probes`。显存不足可减小 `--chunk`，不要修改训练参数来处理离线显存问题。

## 7. 文件与保存时机

```text
observer/
  manifest.json        配置、源码哈希、特征缓存哈希、状态
  surface.npz          固定评估点、GT、归一化网格、法线、UV
  snapshots.json       快照索引、阶段、已完成更新次数
  snapshots/*.npz      原始核参数、face ID、稳定 ID、root ID、ADC 累计量
  samples/*.npz        少量真实训练 batch 的位置/面/预测/GT 等
  steps.jsonl          每次更新的 loss、核数、学习率、累计耗时
  decisions.jsonl      ADC 判定配置、分数快照、候选量/容量
  events.json          每次变更的前后快照、选中/父/子 ID、完成状态
  analysis/            离线数值结果，含固定点评估指标与详细 NPZ
  viewer/              HTML、PNG、GIF
```

初始快照在数据初始化之后、第一次 backward 之前；周期快照在优化器、scheduler 和 PGD 限制之后；ADC 前后快照位于本轮 optimizer 更新之前。因此不同阶段即使更新次数相同，状态也可能不同。

**保留原来的 `SMPL_male_eigen_albo_principal_curvatures.pt`。** 体积较大的谱缓存仅记录路径与 SHA256，不在每个快照里重复保存。迁移时同时保留记录目录、该缓存和原版本代码；缓存换路径后回放用 `--cache /新路径/缓存.pt`。源码或缓存哈希变化会拒绝回放，不自动重算特征分解。

快照不包含完整优化器、采样器和随机数状态，**不能拿来断点续训**。它们服务于离线数值回放。当前版本面向这一套 UV/KNN/PGD、双调和距离、log1p 网格、`mass_type=one`、无输出网络且不做颜色归一化的配置。

## 8. 接入位置与撤回

所有训练接入均为当前 Python 进程内的包装，不编辑下列原文件：

| 原位置 | 追加行为 |
|---|---|
| `hktex.utils.load_config` | 记录已解析配置 |
| `BaseTrainer.optimise` | 建立记录目录、记录完成/失败状态 |
| `BaseTrainer.compute_loss` | 初始状态、少量实际 batch 数据 |
| `HeatKernelTexture.post_optimizer_step` | PGD 之后记录步数及周期快照 |
| `BaseDensityController.clone/split/_remove_kernels` | 记录变更前后及稳定 ID 映射 |
| 增密 `_densify` / 剪枝 `prune_redundant_kernels` | 记录选择前的累计分数和配置 |

本工具自己的文件：`recording.py` 负责记录，`observed_runner.py` 接入已有 V4 检查器，`run_observed.py` 生成本次配置并启动，`replay.py` 做数值回放，`plot_recording.py` 制图。

一键停用新增记录，回到原 V4 训练行为：

```bash
python hktex-observer/run_observed.py --phase full --no-recording
```

也可直接使用之前已成功的训练命令。无需还原 HKTex 源文件；不要撤销先前已经验证的 SVD、prune/split 修复。旧输出保留，不会被新实验覆盖。

若启动器提示缺少 prune/split 修复，先检查仓库是否迁移完整。包内 `apply_split_fix.py` 只是之前修复的备份和检查依据，启动器不会自动执行它；需要修复时应先看其默认 dry-run diff，再按此前流程应用并跑 `test_split_regression.py`。

## 9. 验证范围与故障材料

参考源码：`circle-group/hktex` commit `9e942972c5411a364f12563eb6864520f9effe70`，兼容此前修复后的代码。每次记录还会保存运行时实际源码哈希。

若失败，请提供：本次训练/回放的 traceback、V4 `report.json`、`observer/manifest.json`、`snapshots.json` 和 `events.json`（若存在）。暂时不必发送很大的谱缓存。

本地通过：Python 语法检查；四种训练阶段的配置生成；新入口参数解析；真实控制器代码上的记录开关对照；作者谱热核/颜色归约的 CPU 数值对照；合成数据的全部图表及 GIF/HTML 生成与布局检查。合成图只用于工具测试，未当作 SMPL 训练结果交付。

尚待服务器验证：FAISS/CUDA/torch.compile 的完整回放、实际 SMPL GPU 事件记录、5000 步完整训练。未宣称开启记录后所有 GPU 运行都能逐位一致；本工具设计上不追加训练 forward/backward、不消费训练随机数、不更改学习率或损失。
