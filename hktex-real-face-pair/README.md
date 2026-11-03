# 静态人脸 5000 步：ADC 关闭／开启配对实验

将整个 `hktex-real-face-pair` 文件夹放到服务器 HKTex 仓库根目录。无需覆盖旧的 `hktex-real-face`，不修改 HKTex 核心文件或原数据。
要求已有可用的 `hktex` conda 环境、已打 tracer eps 补丁，以及上次通过的眼部数据包。

```bash
cd /root/hktex-work/hktex
bash hktex-real-face-pair/run_pair.sh
```

默认数据路径是：
`/root/hktex-work/hktex/outputs/real-face-eyes-20261031-024447-417163/dataset`

默认输出：`outputs/real-face-adc-pair-日期-时间-PID/`。
可以覆盖 `DATASET`、`CONFIG`、`HKTEX_ENV`、`GPU`、`RUN`；`RUN` 必须是新目录。
脚本自动顺序执行：数据与数值检查 → ADC 机械操作测试 → baseline 5000 步 → adc 5000 步 → 指标和图片比较。
两组使用同一张 GPU 顺序运行，不争抢显存；正常情况下不暂停、不要求逐阶段确认。失败会停止并保留日志。

## 两组的控制变量

- 两组共用一份 `initial_state.pt` 中的精确 float32 初始参数；不是接着之前的300步训练。
- 单帧固定mesh、固定相机、同一ROI，linear RGB MSE。
- 初始5000 kernels，mean RGB 0.35、残差标准差0.003；使用原模型的表面采样/FPS。
- 每步相同采样种子、同样的512/256/128/128区域采样与像素占比校正；最后核对完整采样流哈希。
- 两组都使用直接谱空间距离搜索和稳定排序、`naive`可微距离重算。实现只在本进程替换近邻搜索，不依赖旧环境变量补丁是否存在。
- 同样的Adam初始学习率（small-ray配置×0.3）、GeodesicOpt初始0.001；两组从第1步到5000步余弦下降至初值的10%。这是完整实验的新统一调度，与300步恒定学习率试训不同。
- 验证相机7，测试相机15。每次评估完整验证ROI，测试指标只在最后计算。VHAP几何使用过全部相机，因此这是外观留出视角，不能宣称几何也完全留出。
- 几何和光照不优化；拟合固定照明下的观测颜色，不是物理albedo。
- 第999步记录两组参数差异供审计。CUDA反向归约可能有微小差异，不把“相同初始化和样本”写成“所有中间浮点结果保证逐位相同”。

## 配置位置与实际ADC时刻

所有试验参数集中在 `config.json`；运行时复制到 `protocol.json`。请在运行前修改，配对过程中不要改源码或配置。

| 项目 | 默认值 |
|---|---|
| 总步数 | 两组各5000 |
| 增密 | 1000、1200、1400…4000；包含4000 |
| 剪枝 | 1200、1600、2000…4000 |
| 统计窗口 | 最近200步；首窗口801～1000 |
| 最后收敛阶段 | 4001～5000无ADC |
| 总kernel上限／下限 | 10000／2500 |
| 每次增密／剪枝上限 | 当前kernel数的5%／5% |
| 部件保留数 | 头部至少512，其余每部件至少32；若初始更少则保留初始数量 |
| 增密误差阈值 | 0.02，见下方统计口径 |
| 大小阈值 | `1 - threshold > 0.18` 则split，否则clone |
| split基础半径 | 原mesh唯一边的长度中位数；实际移动距离再乘`1 - threshold` |

这些是本轮真实人脸ADC适配设置，不冒充作者原配置的严格复跑。
继续复用原 `BaseDensityController` 的clone/split/prune和优化器状态迁移操作，但有以下适配：

1. 训练loss是MSE；增密统计沿用作者的MAE归因思路。对top-k归一化贡献加权误差，乘`1024 × pixel_weight`，最后除以实际窗口步数。命中和贡献统计也应用同样像素权重，避免把过采样眼部当成图像面积的25%。命中只计正贡献。
2. 剪枝候选条件为低相对命中或低平均原始贡献；按贡献从小到大选取，并遵守全局下限、5%上限和部件下限。
3. 同一步先剪枝，再用同步后的统计决定clone/split；每次更新后检查数量、部件归属、三角面内坐标、优化器引用、动量形状和位置移动。
4. ADC在本步优化器更新之后执行，避免替换参数时丢掉本步梯度。重心坐标缓存随后清除并重建。
5. ADC smoke使用可丢弃的初始化副本：先4步填充优化器状态，强制clone一个头部kernel、split左右眼各一个kernel、prune三个头部kernel，再训练4步，并重新执行完整区域前向/梯度检查。测试不会污染两组正式初始化，也不修改正式阈值。
6. smoke证明操作链路有效，不代表生产阈值一定触发增密。正式候选为0时如实记录，不强行制造kernel增长。

## 自动保存与结果解读

`step_00000`、`step_00300`、`step_00500`、`step_01000`……`step_05000` 自动保存模型、指标和预览。
ADC还保存每个事件前后 `adc_snapshots/u01000_before` / `u01000_after` 等检查点；这些额外检查点默认不立即出图，避免频繁渲染。

每份检查点：
- `state.pt`：本入口使用的精确参数／位置，渲染使用它，避免原加载器的同数量重心坐标恢复问题。
- `field.pt`：原HKTex未压缩格式，保留给其他工具；本包不调用旧合成场景observer。
- `optimizers.pt`：优化器和ADC统计存档。当前入口不实现断点续训，不要把这些存档误当成已经支持resume。
- `checkpoint.json`：步数、阶段、kernel数和文件哈希。

主要结果：
```text
protocol.json                   配置、数据/源码/初始化指纹、实际ADC时刻
setup/preflight.json            实际CUDA初始检查
smoke/report.json               clone/split/prune与优化器更新测试
smoke/post_mutation_preflight.json
baseline/report.json            无ADC完整实验
adc/report.json                 ADC完整实验
adc/adc_events.jsonl             每次候选、增密、剪枝、部件数量
comparison.json                 统一指标、kernel数、控制变量核验
comparison.png                  视角0/7/15：GT、无ADC、有ADC
run.log                         连续执行日志
```

图只显示相同有效ROI；白色区域不参与训练、不是预测颜色。误差图红色为linear RGB MAE×10。眼部图是最近邻放大，不能当作原生高分辨率结果。
全局指标容易被面部面积主导，需要同时比较左右眼与眼周指标；不要只看最后一个随机batch loss。

## 训练后补渲染

默认已经有最终和每500步的图片。以下命令用于重新渲染，或额外查看ADC每次调整前后的全部快照。先把PAIR改成实际输出路径：

```bash
conda activate hktex
cd /root/hktex-work/hktex
PAIR="/root/hktex-work/hktex/outputs/real-face-adc-pair-实际时间-PID"
```

只渲染ADC组最终结果：
```bash
python hktex-real-face-pair/render.py --pair "$PAIR" --arm adc --which final \
  --output "$PAIR/render-adc-final-$(date +%Y%m%d-%H%M%S)"
```

渲染ADC组所有保存的检查点（包含增密剪枝前后）：
```bash
python hktex-real-face-pair/render.py --pair "$PAIR" --arm adc --which all \
  --output "$PAIR/render-adc-all-$(date +%Y%m%d-%H%M%S)"
```
将`--arm adc`换成`--arm baseline`即可看无ADC组。中间检查点只渲染相机0和7；最终加入测试相机15。
渲染需要原数据包和匹配的HKTex代码，不能只拷一个field文件到别处运行。

## 验证范围

本地测试使用真实PyTorch CPU优化器和仓库原ADC变更方法，测试控制变量、权重校正、clone/split/prune及状态迁移。
完整CUDA训练未在开发机器执行；服务器自动smoke是实际CUDA验收，不预先宣称通过。

```bash
python hktex-real-face-pair/test_cpu.py -v
```
