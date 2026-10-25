# HKTex A2：离线多视角图片监督

版本 1.0.0，2026-10-25。基于 `Darli000000/my_dyn_hktex` 的
`8c926af44a3a12c80c3fdf1dd62e4f2ba5594630`。

这是新增的独立训练入口，不修改原项目脚本。使用原 HKTex 热核网络、FAISS、DiGeo、TorchTextureRenderer 与 PRB 射线可微渲染，目标颜色来自 A1 保存的线性 RGB 图片。

本地已验证数据读取、相机与配置、检查点读写；**未在 CUDA 上执行整个训练链路**。必须先运行 check 与 smoke，不应直接开 full。

## 1. 数据与安装

脚本默认数据集明确固定为：

```text
/root/hktex-work/data/offline_multiview/smpl-male-full-20261025-003839-35340
```

默认要求图像为 2048×2048，manifest 记录的源模型文件名为 `SMPL_male.obj`，以防误用之前的苹果结果。
这里只检查源模型的名字作为溯源信息，不加载原始模型或纹理。

将 ZIP 上传到 `/root/hktex-work/hktex/`：

```bash
conda activate hktex
cd /root/hktex-work/hktex
unzip hktex-image-training.zip
```

不需要额外 pip 安装，不使用旧 `HKTEX_MESH` 环境变量。
保留已有的 SVD 检查、tracer 检查和 split 几何修复。

## 2. 先检查数据，不启动 GPU 训练

```bash
bash /root/hktex-work/hktex/hktex-image-training/run_training.sh check
```

检查导出完成状态、文件哈希、相机内外参、几何、法线、图像尺寸/有限值、训练/验证/测试划分、颜色空间与渲染约定。
要求 A1 的 `replay_validation.json` 已通过。2048 数据文件较大，首次哈希检查可能需要一些时间。

报告保存在终端 `[OUTPUT]` 指向的新目录下的 `dataset_check.json`。
原始数据集不会被修改。`eval_gt/` 以及原始 OBJ、MTL、纹理都不参与检查或训练。

## 3. 50 次更新的 smoke

```bash
bash /root/hktex-work/hktex/hktex-image-training/run_training.sh smoke
```

默认：256 个核；50 次优化器更新；每步 64 个有效像素；每像素 2 个亚像素位置、每位置 2 条光路，即 256 条主射线；关闭剪枝/增密。
每轮打乱训练相机，依次使用，保证短实验不会一直只看到某几个视角。
初始化是与数据无关的近中性颜色与微小随机残差，不使用 UV 纹理，也不使用验证图像初始化。

输入图片始终是 2048，**不会缩小训练目标**。图片以 CPU mmap 读取，只有抽中的像素和相应射线进入 GPU。
`spp=256` 是导出监督图时的采样数，训练时可用较少光路来估计预测像素。它不等于每步必须对整张图再做 256 spp 渲染。

新特征分解缓存按几何、法线、谱配置及相关源码指纹单独建立：

```text
/root/hktex-work/hktex/outputs/image-training-cache/<hash>/
```

第一次可能需要重新求解特征分解与编译算子，之后相同几何/谱配置会复用。
不会误用旧 SMPL OBJ 对应但顶点顺序可能不同的缓存。不要为了省时间手工把旧缓存改名放进这里。

输出目录：

```text
/root/hktex-work/hktex/outputs/image-training/smoke-时间戳-PID/
```

日志：`/root/hktex-work/test-logs/image-smoke-时间戳-PID.log`。

## 4. 如何判断这一步通过

检查以下文件：

| 文件 | 含义 |
|---|---|
| `startup.json` | 环境/初始化/执行是否完成；早期失败也会记录异常 |
| `dataset_check.json` | 数据文件、尺寸与约定检查 |
| `geometry_check.json` | 在实际训练场景中，像素投影与导出深度是否匹配 |
| `settings.json`、`trainer.yaml`、`renderer.yaml` | 本次真实设置、源码哈希、禁止读取的纹理路径 |
| `metrics.jsonl` | 每步 L1、MSE、各参数梯度范数、核数和 Torch 峰值显存（不包含所有 Mitsuba/FAISS 分配） |
| `eval_*.json` | 固定像素的训练/验证误差；最终额外评估测试视角 |
| `report.json` | 总状态、更新次数、第一次参数变化、读取的数据路径 |
| `checkpoint.pt` | 最近一次完整检查点，含优化器、控制器缓冲区、调度器和采样状态 |
| `texture_float32.pt` | 完整精度、保留笛卡尔位置的纹理参数状态，不是旧压缩模型格式 |
| `preview.png` | 最终的低分辨率目标/预测对照，仅用于快速诊断 |
| `preview_*.npz` | 上述预览的线性 RGB 数组 |

验收条件：`report.status=complete`、`completed_updates=50`、第一次颜色参数变化大于零、梯度有限且第一次核颜色梯度非零、无相机/深度错误。
训练和验证 loss 应整体朝合理方向发展，但随机 batch 不要求每一步都下降。50 步和 256 个核只验证链路，不能判断最终纹理容量或质量。

`eval` 使用原分辨率下固定的 mask 内像素与固定采样种子；指标是**稀疏像素上的蒙特卡洛估计**，不是整张 2048 图的 PSNR。
线性 PSNR 的峰值约定为 1，报告不做去噪、gamma/log 变换或渲染后裁剪。
各次训练的 batch、评估采样量不相同时，不应直接比较数字。

Smoke 结束后，先发 `report.json`、`geometry_check.json`、最终 `eval_*_final.json`、`preview.png` 和日志即可。无需上传所有大图片。

## 5. 然后才进行更长训练

```bash
# 1000 核、1000 次更新，不启用 ADC：确认外部图片拟合的基线。
bash /root/hktex-work/hktex/hktex-image-training/run_training.sh baseline

# 基线正常之后，单独测试 ADC。
bash /root/hktex-work/hktex/hktex-image-training/run_training.sh adc

# 最后再做起步版完整训练。
bash /root/hktex-work/hktex/hktex-image-training/run_training.sh full
```

| 档位 | 更新次数 | 初始核数 | 每步像素 | 亚像素×光路 | ADC | 预览尺寸 |
|---|---:|---:|---:|---:|---|---:|
| smoke | 50 | 256 | 64 | 2×2 | 关闭 | 64 |
| baseline | 1000 | 1000 | 128 | 4×2 | 关闭 | 128 |
| adc | 1500 | 1000 | 128 | 4×2 | 开启，最多5000核 | 128 |
| full | 5000 | 1000 | 256 | 4×4 | 开启，最多5000核 | 256 |

这些是排查与起步配置，不是质量保证，也不表示 5000 步已经充分覆盖 2048 图片中的细节。
比如 full 总共抽取 128 万个像素样本，其中可能重复，不能理解为对每张图片完整训练了 5000 遍。
每个命令都从头建立一个新实验；不会自动继承上一个档位的参数。

ADC 沿用作者按第一命中点分配像素误差的近似，不是对所有间接光路的严格贡献归因。
`density_events.jsonl` 记录实际 clone/split/prune 调用、选择数量与核数变化。
仅开启 ADC 不保证阈值一定触发所有操作；需要根据日志判断覆盖情况，而不能强行宣称已测试完全部功能。
每个控制器执行前重新构造当前核的几何信息，保留已修复的剪枝后 split 对应逻辑。

## 6. 降低显存占用、手动参数与恢复

如显存不足，先降低每步像素或亚像素/光路数，不要重新导出整套数据。
手动示例（输出目录必须不存在）：

```bash
export MPLBACKEND=Agg
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
python -u /root/hktex-work/hktex/hktex-image-training/train_images.py train \
  --repo /root/hktex-work/hktex \
  --dataset /root/hktex-work/data/offline_multiview/smpl-male-full-20261025-003839-35340 \
  --output /root/hktex-work/hktex/outputs/image-training/smoke-small-v1 \
  --profile smoke --batch-pixels 16 --aa 2 --transport 2 --gpu 0
```

恢复中断的**同一实验设置**：

```bash
python -u /root/hktex-work/hktex/hktex-image-training/train_images.py resume \
  --dataset /root/hktex-work/data/offline_multiview/smpl-male-full-20261025-003839-35340 \
  --checkpoint /替换为原实验目录/checkpoint.pt \
  --output /root/hktex-work/hktex/outputs/image-training/resumed-v1 --gpu 0
```

恢复会检查数据集与相关源码指纹，使用保存的采样、调度器、优化器状态。
不允许借 resume 把 smoke 改为 full 或修改总步数；各档位调度不同。
这是状态恢复，不保证跨硬件或 FAISS 并列候选情况下逐位一致。
失败不会用半更新的参数覆盖最近的正常 checkpoint；失败前尚未保存的更新会丢失。

预览分辨率是独立的诊断设置，和训练输入分辨率无关。预览目标使用线性 box 平均，预测在同样的低分辨率相机下采样。
预览不用于高频细节验收；后续应增加完整原分辨率的独立评估渲染。
旧 observer 尚未适配此入口；这里先保存 loss、梯度、控制器事件和完整模型状态。

## 7. 实现位置与边界

- `image_dataset.py`：A1 数据约定校验、mmap 图片读取、仅训练相机采样、像素内采样布局、禁止读取纹理参考的守卫。
- `image_trainer.py`：继承 `MitsubaRayTrainer`，复用 HKTex 模型/渲染器，替换训练循环中的监督来源；保存检查点和评估。
- `train_images.py`：创建配置、几何与独立谱缓存，设置训练档位、初始化与恢复。
- `offline_common.py`：冻结的 A1 1.0.1 相机/数据检查工具，不依赖服务器上旧脚本的导入路径。

新模型用保存的 mesh.npz 构建，不再做归一化、重排或 UV 采样。
材质固定 specular=0、metallic=0、roughness=1，恒定环境光，无地面，隐藏背景发光体，最大路径深度3，与 A1 对应。
损失直接比较线性 RGB；没有沿用原在线 trainer 的 log/gamma 变换。
所有调度器按**优化器更新次数**前进，不按原来外层 camera batch 的 epoch 前进。
不通过原 optimisation.py 启动，因为它的在线 GT/末尾渲染流程仍然依赖 UV。

读取守卫禁止 Python 打开 `eval_gt/` 和原始模型所在的资产目录。数据集即使不包含 `eval_gt/`，训练仍应能运行；A1 的历史重放验证记录需要保留。
训练包不修改原项目、原始数据或原来的特征缓存。撤回时移走这个新增目录即可；新输出和缓存可独立保留。

## 8. 验证范围

见 TESTING.md。CPU 测试不能代替 CUDA 端的 PRB/Dr.Jit/PyTorch/HKTex 集成测试。
服务器 smoke 中首轮会检查实际核颜色梯度非零以及参数发生变化；发生异常时请保留检查，不要通过放宽容差或关闭梯度检查绕过。
