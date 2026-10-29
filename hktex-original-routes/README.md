# HKTex 原版两路线：5000 次更新与统一评估

本包配合 `Darli000000/my_dyn_hktex` 的已检查 main 版本 `b02d07ee7e8a7a0d127ee6ef0dd74bdda906b51d`。
120 个依赖源码/配置文件有 SHA256 检查；不覆盖原项目文件。所有训练 hook 只在当前 Python 进程内生效。

这是**当前仓库中作者训练路线的预算化复跑**，包含仓库已有的 SVD、tracer、ADC 修补；不是未经修改的论文原始代码复现。

## 1. 保留什么，改了什么

| 项目 | UV texture | 原版 multi-view/ray |
|---|---|---|
| 配置来源 | `configs/texture_hktex_knn.yaml` | `configs/multiview_hktex_knn_ray_small.yaml` |
| 训练类 | `UvTextureTrainer` / `BaseTrainer.optimise` | `UvTextureMitsubaRayTrainer` / `MitsubaRayTrainer.optimise` |
| 初始化核数 | **1000** | **5000** |
| 初始化 | 原配置，随机比例 0.5，读取 UV | 原配置，随机比例 1.0；不套用 image trainer 的中性颜色初始化 |
| 监督 | UV 表面采样，importance sampling 开启 | 随机相机采集命中射线，**在线渲染 GT** |
| loss | RGB 各通道 MSE 求和后平均点 | clip + log/gamma 后 RGB 各通道 L1 求和再平均射线 |
| batch | 4096 个表面点 | 每批 1024 条 ray，grad_spp=8；每轮 16 个 256² 随机相机 |
| 模型/光谱参数 | 保留 UV 配置 | 保留 ray 配置；不改成当前 image trainer 配置 |
| 初始学习率/动量 | 保留 | 保留 |
| 更新数 | 5000 | **精确 5000 次参数更新**，不是 5000 个 epoch |
| ADC | 原配置及其 5000 步比例 | 保留原 YAML 中绝对步数，不改成间隔 200 |

必要的显式适配：

1. **ray 学习率时钟换算**：原来 StepLR=15 个 epoch、Cosine T_max=60 个 epoch；这里换算成每次更新推进，StepLR=1250、Cosine T_max=5000。原学习率、gamma、eta_min 保留。不同网格每 epoch 的 ray batch 数不同，不能把 `optim.iters=5000` 直接当 5000 次更新。
2. ray 的 log/gamma 幂函数输入增加 `1e-12` 下限，避免零处无限导数。这是数值修补，不是更换 loss；边界附近仍有微小差异。
3. UV importance wrapper 暴露其内层 `tex_img`，使原作者的 UV-informed 初始化实际执行；否则原函数可能直接返回。已有 observed 入口也提供过这一修补。
4. 记录参数、loss、ADC 决策与事件；遇到 NaN/Inf 直接报告，不替换、不跳过更新。
5. 两路线分别生成 eigen cache。原文件名没有完整编码配置，不能默认复用以前的 SMPL cache。计算可能比复用旧 cache 慢。
6. 训练保持作者 `high`/TF32 设置；**统一评估**使用 `highest`/TF32 关闭，与既有 checkpoint viewer 一致。

每个 run 保存 `author_resolved_before_adaptation.json`、`parsed.yaml`、`protocol.json`。
ray 还保存 `ray_loop_original.py` 与 `ray_loop_instrumented.py`，可直接 diff 核对。

### ADC 时间点（代码采用从 0 开始的 step）

| 控制器 | UV | ray |
|---|---|---|
| 增密 start / interval / stop | 400 / 100 / 3500 | 307 / 307 / 2688 |
| 增密累计窗口 | 75 | 153 |
| 剪枝 start / interval / stop | 500 / 1000 / 2500 | 768 / 1920 / 2688 |
| 剪枝累计窗口 | 250 | 480 |
| 核数上限 | 5000 | 10000 |

触发条件是 `step >= start && step % interval == 0 && step < stop`。所以 UV 剪枝检查发生在 step 1000、2000；ray 剪枝检查在 step 1920。触发检查不代表必然删/加核。最终数量看 `report.json` 和 `observer/events.json`。

## 2. 一次运行训练、原版预览、统一对照

把 zip 上传到 `/root/hktex-work/hktex/`，然后：

```bash
conda activate hktex
cd /root/hktex-work/hktex
unzip -n hktex-original-routes.zip

export IMAGE_RUN=/root/hktex-work/hktex/outputs/image-training/adc-full-pair-20261027-024845/s05000/linear
bash /root/hktex-work/hktex/hktex-original-routes/run_all.sh
```

默认已经填好：

```text
MESH=/root/hktex-work/data/SMPL_texture_samples/Textured_Mesh_samples/SMPL_male.obj
DATASET=/root/hktex-work/data/offline_multiview/smpl-male-full-20261025-003839-35340
UPDATES=5000
SIZE=1024
SPP=128
VIEWS=view_000,view_002,view_005
GPU=0
```

脚本连续执行：源码检查 → UV 5000 → ray 5000 → 两个原版 native preview → 三条路线的统一评估 → 对照图/指标。
两条训练完成后才开始渲染；没有 2000 步暂停或人为确认。
`IMAGE_RUN` 只用于评估已训练模型，**不会重新训练或写入它**。
默认选 3 个相机（该 full 数据布局下分别属于 train/val/test）；这不是完整 test 集平均分。

例如提高采样数：

```bash
SPP=256 bash /root/hktex-work/hktex/hktex-original-routes/run_all.sh
```

需要所有 36 个保存相机都做统一评估时，可设置 `VIEWS=all`，耗时和输出量明显增加。不要把 full 训练等同于 all-view 评估。

## 3. 看哪里

启动时显示 `[OUTPUT] /root/.../original-routes-时间戳-PID`，以下以它为 `ROOT`：

```text
ROOT/
  uv/、ray/
    report.json                 状态、最终核数、更新次数
    parsed.yaml                 实际配置
    protocol.json               原配置与适配说明
    checkpoint.pt               全精度评估用；不是续训 checkpoint
    author_model.pt/.npz        作者原生保存接口
    observer/
      manifest.json
      snapshots.json / snapshots/*.npz
      steps.jsonl               训练目标值、核数、学习率
      events.json               clone/split/prune 事件（没有事件可能无文件）
      decisions.jsonl
    author-native/index.html    原版外观约定下的最终结果/热核带
  eval-uv/、eval-ray/、eval-image-linear/
    index.html
    common_protocol.json
    checkpoint_u05000/
      preview_views_raw.png
      prediction_*.npz/.png
      image_errors.json
  comparison/
    comparison_raw.png          GT | 原版UV | 原版ray | 当前image-linear
    comparison.json             同视角、同分辨率、未去噪 RGB MSE
```

`comparison.json` 的 `foreground_mse` 按有效像素数加权；`mean_view_mse` 平均各视角。两者不一定相等。
续训的 image observer 若只含 2000→5000 段，事件汇总会明确标注只覆盖该段，不能拿段首核数当训练初始化核数。

## 4. 为什么需要两套预览

原版 UV/在线 GT 使用纹理 `RGB/255` 数值；现有离线数据导出显式将 sRGB 解码为线性反射率，再渲染线性辐射亮度。直接拿原版预览 PNG 对离线 RGB 算误差，会混入颜色约定差异。

* `author-native`：保留旧 native viewer 的原版外观约定（包括原配置场景），用于观察这条路线自身的 GT/结果/热核带。
* `eval-*`：使用离线数据的相机、几何/法线、光源和材质。对原版拟合出的**表面反射率**做 sRGB→线性转换，再进 BSDF 和路径积分；image 模型已在线性空间，不重复转换。GT 来自保存的线性图片。转换不写回模型，不在渲染后的 radiance 上做逆 gamma。

所有统一评估采用同一 1024 分辨率、128 spp、随机种子、分块、RGB 平均方式、保守前景 mask、无去噪。2048 GT 按 box 平均下采样。

这些 MSE 是新评估协议的结果，**不要直接与之前 2048 分辨率、抽样像素的 eval JSON 数值混排**；对 image checkpoint 重新评估，才是本轮横向基准。
保留作者随机相机范围意味着原版 online 路线并不遵循离线 train split；UV 更能直接读取整个纹理。因此 split 标签在这里首先表示评估相机组，不能据此宣称严格相同数据条件下的泛化胜负。

## 5. 只重渲染，不重新训练

环境准备：

```bash
conda activate hktex
export MPLBACKEND=Agg
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
ROOT=/root/hktex-work/hktex/outputs/替换成这次original-routes目录
```

旧 observer/native viewer 已适配两条路线。只看最终：

```bash
python /root/hktex-work/hktex/hktex-native-viewer/render_native.py \
  --repo /root/hktex-work/hktex --recording "$ROOT/uv/observer" \
  --snapshots final --size 1024 --spp 128 --frames 4 --tile-size 32 \
  --point-batch 512 --no-denoise --output "$ROOT/uv/native-final-repeat"
```

看全部常规快照变化：

```bash
python /root/hktex-work/hktex/hktex-native-viewer/render_native.py \
  --repo /root/hktex-work/hktex --recording "$ROOT/ray/observer" \
  --snapshots all --size 512 --spp 32 --frames 4 --tile-size 32 \
  --point-batch 512 --no-denoise --output "$ROOT/ray/native-all"
```

将 `uv` / `ray` 换为另一条即可。`all` 包括 initial / periodic / final；不会默认渲染每次 ADC 事件的前后快照。
每 250 次更新保存一次，最终快照另存。记录阶段不渲染。

只重做统一评估（例如全视角）：

```bash
python /root/hktex-work/hktex/hktex-original-routes/evaluate.py \
  --repo /root/hktex-work/hktex --run "$ROOT/uv" \
  --dataset /root/hktex-work/data/offline_multiview/smpl-male-full-20261025-003839-35340 \
  --output "$ROOT/eval-uv-all" --views all --size 1024 --spp 128
```

为 ray 和 IMAGE_RUN 用完全相同参数重评，再：

```bash
python /root/hktex-work/hktex/hktex-original-routes/compare.py \
  --evaluations "author-uv=$ROOT/eval-uv-all" "author-ray=$ROOT/eval-ray-all" \
                "image-linear=$ROOT/eval-image-linear-all" \
  --output "$ROOT/comparison-all"
```

## 6. 中断、排查与撤回

* 每个训练只接受新输出目录，防止覆盖已有结果。checkpoint 是评估工件，不声称恢复 optimizer/sampler。
* 原 YAML 的 UV GeodesicOpt LR=0.0991453，比以前 SMPL observed 脚本用的 0.00991453 高 10 倍。本轮默认保留原值；若再次出现非法 face ID，应提供 `failure.json` 和日志，不自动掩盖。
* 若明确决定回到过去验证过的位移学习率，可用 `UV_GEO_SCALE=0.1` 重新开实验；它会写入配置差异，不能继续标注为未改原参数。
* `Source mismatch` 表示服务器代码与审查版本不同，请先核对具体文件；不要删除校验后继续，也不需要强制 reset 服务器代码。
* 训练/渲染任一步失败会停止后续流水线，已有结果保留。渲染失败无需重训，使用上面的单独命令。
* 撤回无需还原项目源码：停止使用新增的 `hktex-original-routes` 入口即可。所有运行时替换随进程结束恢复。

## 7. 本地验证范围

已检查两份 YAML 与 upstream 对应文件一致；检查源码指纹、真实 ray 函数的 AST 转换、不同每 epoch batch 数下的精确更新/调度/observer 计数、源码结构变化拒绝、脚本语法与比较输出。
**本地无 CUDA：尚未执行这两个 5000 步训练，也未实测 GPU 渲染或 Mitsuba backward。** 完成状态由服务器的 report/日志判定。
