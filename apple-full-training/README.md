# 苹果完整训练验证

适用于已有服务器 `/root/hktex-work/hktex` 和已激活的 `hktex` 环境。基于此前 HKTex 提交 `9e942972c5411a364f12563eb6864520f9effe70` 的接口编写。不会修改仓库原始 Python 文件或原始配置。

## 先看已完成的 300 步结果

在服务器解压后运行：

```bash
python /root/hktex-work/hktex/apple-full-training/preview_latest.py --experiment apple-baseline
```

它会打印并保存最新实验的 `ckpts/preview_first_frame.png`。通过服务器文件管理器下载该 PNG 到本地打开。图片从左到右为原始纹理 GT、HKTex 拟合结果、热核轮廓（如果有）。它不重新训练、不重新渲染，只提取原视频第一帧。

已有视频通常在：

```text
/root/hktex-work/hktex/outputs/apple-baseline/
  init05_lr10_300steps@时间戳/ckpts/gt_vs_out.mp4
```

此前只渲染 2 帧，默认编码 30 fps，总长约 0.067 秒，播放器可能一闪而过。分辨率也只有每栏 128×128，所以先用静态图确认基本效果。

`outputs/training-checks/.../report.json` 是诊断报告目录，和上述实验渲染/模型目录不是同一个目录。

## 运行完整验证

```bash
python /root/hktex-work/hktex/apple-full-training/hktex_checked_run_v4.py --version
bash /root/hktex-work/hktex/apple-full-training/run_apple_full.sh
```

版本应为 `HKTex checked runner 4.0.0`。如使用 tmux，先进入已有会话再运行，以免 SSH 断连影响任务。

这是一轮从初始化开始的新实验，并非恢复之前 300 步的优化器状态。

| 设置 | 数值 |
|---|---|
| 训练步数 | 5000 |
| 初始热核 | 1000 |
| batch size | 4096 |
| 位置学习率 | 原始 YAML 的 0.1 倍，约 0.00991453 |
| 位置动量 | 原始配置，0.9 |
| 数据相关初始化比例参数 | 0.5 |
| 重要性采样 | 开启；100 万点池；250 步预热 |
| 剪枝 | 恢复原始 importance_pruning 控制器和阈值 |
| 增密 | 恢复原始 error_based_densification 控制器和阈值 |
| 克隆 / 分裂 | 由误差和大小阈值决定，不强制触发 |
| 密度控制器的核数上限 | 原始公式：初始核数的 5 倍，即 5000 |
| 结果渲染 | 60 视角，每栏 256×256，8 samples |
| 地面 | 关闭 |

没有沿用早期 smoke 命令中的 `trainer.density_controllers=[]` 或 `data.use_importance_sampling=False`。

这是保留低位置学习率及下述兼容处理的完整流程验证，不应称为完全不改参数的论文复现。诊断同步检查会增加开销，因此不能用这轮耗时做性能基准。

## 显式的初始化兼容处理

原 `ImportanceSamplerDataset` 保存了 `inner_dataset`，但没有提供 `tex_img`；原 `UvTextureTrainer.data_dependent_initialisation()` 读取包装对象的 `tex_img`，缺少时直接返回。

启动器显式传入 `--forward-inner-texture`。V4 仅在当前进程中为包装类增加 `tex_img` 属性，转发到 `inner_dataset.tex_img`，使原初始化函数能继续执行。不改变采样概率公式，不调整克隆/分裂阈值，也不改磁盘上的项目源文件。

报告中应看到 `forward_inner_texture: true`，初始化事件中 `has_tex_img: true`、`result: returned_normally`。位置/颜色变化值用于确认初始化是否实际改变了参数。

V4 还将项目原有 `plt.show()` 图表保存为 PNG，并新增最终多视角拼图、慢速视频和密度操作计数。

## 如何确认各组件确实运行

- `status: completed`，`loss_evaluations: 5000`。
- `sampler_batches.importance > 0`；在当前设置且完整运行时，通常 uniform=250、importance=4750。
- `event_counts.densify_check`、`event_counts.prune_check`：控制器检查调用次数。
- `event_counts.clone`、`event_counts.split`：真正调用克隆/分裂方法的次数。
- `density_totals.cloned_kernels`：被克隆的核数总和。
- `density_totals.split_parent_kernels`：被分裂的父核数总和。
- `density_totals.pruned_kernels`：实际剪掉的核数总和。
- `density_events`：每次操作前后的核数，及对应 loss 计算序号。

按原始 5000 步调度，增密在零起始迭代号 400、500、…、3400 检查，剪枝在 1000、2000 检查。报告的 `loss_evaluation` 是已完成 loss 计算的次数，通常比该迭代号多 1。

“检查被调用”不等于“一定有候选核”。苹果简单，某个实际操作总数可能为 0，这既不是自动证明错误，也不是证明该分支已经被覆盖。保留报告，后续如确实需要覆盖全部分支，再单独做组件测试，不在效果训练中随意降阈值强制触发。

分裂会额外调用追踪器，不能再要求追踪次数一定等于训练步数。

## 完成后去哪里看

终端会明确打印：

```text
[CHECK] Experiment directory: ...
[CHECK] Preview image: .../ckpts/preview_views.png
[CHECK] Preview video: .../ckpts/gt_vs_out_slow.mp4
[CHECK] Report: .../report.json
```

主要文件：

```text
outputs/apple-full/<本轮tag>@<时间戳>/
  configs/parsed.yaml       # 实际生效的配置
  configs/raw.yaml
  cmd.txt
  logs.txt
  ckpts/model.pt
  ckpts/model.npz
  ckpts/gt_vs_out.mp4        # 作者原有的并排结果视频，30 fps
  ckpts/gt_vs_out_slow.mp4   # 同样帧，10 fps，60 帧约 6 秒
  ckpts/preview_views.png   # 四个视角；每行 GT | output | kernel rings

outputs/training-checks/<本轮诊断目录>/
  report.json
  loss.csv                 # loss 与每步密度操作前的核数量
  plot_*.png               # 原代码图表的静态保存
```

先看 `preview_views.png`，再看慢速视频。注意轮廓、红绿颜色分布、果皮斑点、果柄以及不同视角的一致性。右栏热核轮廓是表示结构的可视化，不是第三份重建纹理。

原始 GLB 不会被训练自动替换成“新纹理 GLB”。打开原始苹果只能看到原模型；模型权重本身也不是图片，应看上述渲染输出。

当前入口在训练后渲染并保存结果；本包没有增加完整优化器状态的周期性断点续训。`model.pt` 不应直接当作可以恢复所有优化器、调度器和采样器状态的 checkpoint。

如果可选预览保存失败，报告会记录 `preview_error`，不将已经成功的原始训练错误标记为失败；原始 `gt_vs_out.mp4` 仍可直接查看。

## 验证范围

已完成本地语法检查和模拟集成测试：参数解析、纹理转发、初始化记录、采样计数、增密计数、PNG 保存及报告生成。未在本机运行 CUDA 训练；实际 GPU 路径需要本轮服务器验证。

参考：[原训练入口](https://github.com/circle-group/hktex/blob/9e942972c5411a364f12563eb6864520f9effe70/optimisation.py)、[默认配置](https://github.com/circle-group/hktex/blob/9e942972c5411a364f12563eb6864520f9effe70/configs/texture_hktex_knn.yaml)、[重要性采样](https://github.com/circle-group/hktex/blob/9e942972c5411a364f12563eb6864520f9effe70/hktex/data/importance_sampling_strategy.py)。
