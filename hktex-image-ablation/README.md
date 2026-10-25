# 图像训练：扩散时间对照

目的：验证当前几乎没有谱衰减的 diff_time 是否限制了拟合。不是保证效果的最终配置。

独立启动器复用已安装的 `hktex-image-training-observed`，不覆盖它或 HKTex 核心源码。observer、native viewer、图片监督与 no-UV 约束均沿用原实现。

## 对照设计

| 项目 | reference | t1e-3 | t1e-2（第三组，可稍后运行） |
|---|---|---|---|
| diff_time | 1.1103742416356491e-6 | 0.001 | 0.01 |
| 初始化 | 全新 seed=0 | 相同 | 相同 |
| 更新数 | 1000 | 1000 | 1000 |
| 核数 | 1000 | 1000 | 1000 |
| ADC | 关闭 | 关闭 | 关闭 |
| 每步像素 | 128 | 128 | 128 |
| AA / transport | 4 / 2 | 4 / 2 | 4 / 2 |
| 阈值/学习率/调度 | 原 baseline | 相同 | 相同 |

先比较 reference 与 t1e-3；有需要再运行 t1e-2。它们是控制变量实验，不能直接拿旧 ADC 1500 步当作同预算 reference。所有组从头开始；不要把旧 checkpoint 改 diff_time 后直接当作重新训练结果。

## 运行

将本目录放到 `/root/hktex-work/hktex/hktex-image-ablation`。先完成一组，再将 `HKTEX_VARIANT=reference` 改成 `t1e-3`，执行同样的命令。

```bash
conda activate hktex
HKTEX_VARIANT=reference
HKTEX_ABLATION_OUT="/root/hktex-work/hktex/outputs/image-ablation/${HKTEX_VARIANT}-$(date +%Y%m%d-%H%M%S)"
mkdir -p /root/hktex-work/test-logs
set -o pipefail

env MPLBACKEND=Agg \
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
  python -u /root/hktex-work/hktex/hktex-image-ablation/run_ablation.py \
  --variant "$HKTEX_VARIANT" \
  --trainer-dir /root/hktex-work/hktex/hktex-image-training-observed \
  --repo /root/hktex-work/hktex \
  --dataset /root/hktex-work/data/offline_multiview/smpl-male-full-20261025-003839-35340 \
  --output "$HKTEX_ABLATION_OUT" \
  --steps 1000 --seed 0 \
  2>&1 | tee "/root/hktex-work/test-logs/ablation-${HKTEX_VARIANT}-$(date +%Y%m%d-%H%M%S).log"
```

如果 observed 工具包位于其他目录，只修改 `--trainer-dir`。启动器只接受对应接口的 observed 版本，遇到不匹配会停止。原始三维训练 YAML 无需编辑。

本实验禁用训练中间的低清 preview，保留 observer。结束后使用原 native 渲染脚本，替换 `--run` 和 `--output`：

```bash
env MPLBACKEND=Agg \
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
  python -u /root/hktex-work/hktex/hktex-image-training-observed/render_native_images.py \
  --run "$HKTEX_ABLATION_OUT" \
  --output "$HKTEX_ABLATION_OUT/native-512-spp64" \
  --size 512 --spp 64 --tile-size 64 --point-batch 512 \
  --views view_000,view_002
```

比较必须保持相机、渲染种子、分辨率和 spp 相同。优先看 raw 图和保存的线性 RGB；去噪图仅用于展示。

## 返回及判断

每组请提供 `eval_001000_final.json`、`metrics.jsonl`、`settings.json`，以及相同视角的 `prediction_000_raw.png`。若有异常，补充 `report.json` 和控制台日志。

已有诊断脚本可继续用于每组新 checkpoint，比较覆盖与 clamp 比例；这一步不会训练。

判断依据是训练/验证误差、局部斑纹及材质覆盖/截断的共同变化。不要仅凭更平滑就认定更准确，也不要用 test 分数选参数。

- 扩散时间改变后拟合改善：再单独测试阈值初始化与形状学习率，然后校准 ADC。
- 仍有强烈负值截断：检查颜色步长/损失与参数化；不能直接删除 clamp 让负反射率进入渲染器。
- 覆盖与截断改善而误差仍高：补做固定射线渲染对照、梯度数值检查或少量视图过拟合。

## 验证与撤回

启动器已通过语法、CLI 和配置钩子的 CPU 检查，确认三个组只改变 diff_time，且实际配置及启动器哈希进入 settings/checkpoint。这里没有 CUDA，未运行真实训练。

没有替换任何原文件；回到原启动命令即可。删除本目录不会影响已有训练。
