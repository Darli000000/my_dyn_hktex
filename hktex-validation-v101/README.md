# HKTex 图片训练：渲染一致性 + 颜色梯度诊断

基于用户仓库 main 的 `b02d07ee7e8a7a0d127ee6ef0dd74bdda906b51d`。整个目录独立运行，不覆盖作者或图片训练代码，不执行 optimizer.step，不写入原 checkpoint，不运行增密剪枝。

本地已完成 14 项 CPU 测试、Python 语法及 shell 语法检查。**没有在本地执行 CUDA/PRB 集成测试**。本工具就是交给服务器执行的集成验证，不能把本地测试结果当成训练管线已通过验证。

## 1.0.1 修复与本次重跑

1.0.0 的诊断 GT 在同一次参数更新中写入顶点和保存法线。Mitsuba 3.7.1 会在几何更新时重算法线，覆盖该值。1.0.1 改成先提交几何，再单独提交保存法线，另加 buffer 一致性检查。训练、导出脚本和检查阈值都没有改动。

带有倾斜自定义法线的回归测试在旧实现失败，修复后通过；全部 14 项 CPU 测试通过。CUDA 上本次报告中的 16 项梯度检查已经通过，不需要为了这个修复重跑梯度。

新版压缩包解压得到独立目录 `hktex-validation-v101/`，保留原诊断目录。上传到 `/root/hktex-validation-v101.zip` 后：

```bash
conda activate hktex
unzip -n /root/hktex-validation-v101.zip -d /root/hktex-work/hktex
export HKTEX_RUN=/root/hktex-work/hktex/outputs/image-training/full-reference-20261025-061226
export HKTEX_DATASET=/root/hktex-work/data/offline_multiview/smpl-male-full-20261025-003839-35340
bash /root/hktex-work/hktex/hktex-validation-v101/run_validation.sh confirm --stage radiance
```

重跑只检查渲染/几何，增加采样以检查此前亮部的 inconclusive。保存的 GT 本身只有 256 spp，其噪声不会随在线重渲染采样增加而消失，因此部分 saved_vs_online 仍可能是 inconclusive；不要通过放宽阈值强行变绿。请发回新 report.json，必要时再检查对应 NPZ。

单阶段报告的 ready_for_controlled_5000_comparison 仍为 false，这是程序设计，不代表先前已通过的梯度作废；应合并本次渲染报告和前次梯度报告判断。

下方原版通用说明中出现的 `hktex-validation/`，运行新版时替换为 `hktex-validation-v101/`。

## 安装与运行

把压缩包上传到服务器，并解压到项目目录。得到：

```text
/root/hktex-work/hktex/hktex-validation/
    validate.py
    numerics.py
    run_validation.sh
    expected_sources.json
    test_cpu.py
    README.md
    SHA256SUMS
```

示例（压缩包实际位置请自行替换）：

```bash
conda activate hktex
cd /root/hktex-work/hktex
unzip -n /root/hktex-validation.zip -d /root/hktex-work/hktex

export HKTEX_RUN=/root/hktex-work/hktex/outputs/image-training/full-reference-20261025-061226
export HKTEX_DATASET=/root/hktex-work/data/offline_multiview/smpl-male-full-20261025-003839-35340

bash /root/hktex-work/hktex/hktex-validation/run_validation.sh quick
```

`HKTEX_RUN` 指向**训练结果目录**，不是 observer，也不是 checkpoint.pt 文件本身。建议先用原 5000 步 full-reference 的结果，不用 range 修复结果，以保留原问题。路径不存在时脚本会直接提示，不会自动挑另一个实验。

无需新安装依赖。需要当前 hktex 环境、原线性图像数据、谱缓存和 `eval_gt/reference.npz`。若迁移后谱缓存路径已改变：

```bash
bash /root/hktex-work/hktex/hktex-validation/run_validation.sh quick \
  --cache /新的完整路径/geometry_eigen_albo_principal_curvatures.pt
```

若诊断明确显示仅采样精度不足，可以提高采样后重跑：

```bash
bash /root/hktex-work/hktex/hktex-validation/run_validation.sh confirm
```

不要因任何 `inconclusive` 都反复运行 confirm：硬截断、零信号和随扰动幅度变化的导数可能需要改实验，增加 spp 不一定解决。运行 quick 后先查看具体原因。

## 输出

启动时会打印：

```text
[INPUT RUN] 原训练目录
[OUTPUT] /root/hktex-work/hktex/outputs/pipeline-validation/quick-时间戳-PID
[LOG] /root/hktex-work/test-logs/pipeline-validation-quick-时间戳-PID.log
```

结果目录包含：

- `report.json`：全部判定、梯度数值、扰动幅度、随机种子规则、文件哈希与错误堆栈。
- `geometry_check.json`：沿用图片 trainer 的全部视角几何检查。
- `dataset_check.json`：原数据契约和文件检查。
- `view_XXX_dark.npz` / `view_XXX_light.npz`：相同像素上的保存 GT、重复在线 GT、训练射线渲染及 native 渲染的线性值。

先发回 **report.json 和运行日志** 即可。只有某个 radiance 检查异常时，才需要对应的 NPZ。程序逐项保存报告；某个阶段抛出异常时会尽量继续独立阶段，最终状态仍为失败。

退出码：0=passed，2=failed，3=inconclusive。后两者不必然表示进程崩溃；查看报告区分数值判定和异常。脚本启用了 pipefail，因此不会因 tee 成功而掩盖退出码。

## 实验 A：渲染一致性

默认各选择一个 train / val / test 视角，每视角选择两个互不重叠、完整位于前景内部的小块，一个偏暗、一个偏亮。8×8 的小块位于**原始 2048×2048 像素网格**上，没有先把整张图降为 8×8，也不是用低分辨率人物来做比较。

逐项比较：

1. GT 和训练几何的交点、着色法线。
2. 保存的线性 RGB 与从 `eval_gt/reference.npz` 在线重渲染的 GT。
3. 同一 GT 的自定义 ray/PRB 前向与 native path 前向。
4. 同一 HKTex checkpoint 的训练 ray 前向与 native path 前向。native 使用 inference_mode=True，训练前向使用 False。

材质、法线、相机、环境、深度和颜色输入约定与离线数据保持一致。GT 纹理是已经线性化的数组，明确 `raw=True`，不重复做 sRGB 解码。**本阶段允许读取已知合成 GT 来诊断，但不会启动训练或修改训练的 no-texture guard。**

quick：每块每条渲染路径 4 次独立重复，每像素 64 样本；ray 端沿用 4 AA × 16 transport 的构造。confirm：8 次重复、256 样本。保存图像的 spp 从 manifest 中读取，而不是假定为某个固定值。

判定使用重复渲染估计噪声，并报告 RMSE、噪声 RMSE、扣除估计噪声后的 excess RMSE、RGB 平均偏差。默认绝对容差 0.003；噪声过大时不判通过。它是有限样本的工程筛查，**不是严格统计显著性检验，不保证网格每个像素都正确**。相近的帧、像素和通道也不能视作完全独立样本。

## 实验 B：颜色梯度

按三层检查：

1. **简单漫反射平面**：使用项目原来的 Dr.Jit/PyTorch 纹理桥接和 rb_ray/PRB，正颜色 R/G/B 分别做导数检查；另设一个严格负值截断通道，验证预期零梯度。
2. **真实 HKTex，中性颜色场**：保留 checkpoint 的核位置、形状和谱信息，仅在内存里把均值设为 [0.3,0.4,0.5]、残差置零，让输出离开硬截断区。检查直接 PyTorch 场梯度和完整 PRB 梯度。
3. **原 checkpoint 颜色场**：恢复原颜色，再重复上述检查，并记录采样点的截断比例。

真实场检查三个方向：全局红色均值、选中的高贡献核红色残差、同一批核蓝色残差。这里只验证颜色相关方向，不声称覆盖所有位置、各向异性、阈值的导数。

标量目标使用 `mean(0.2 R + 0.3 G + 0.5 B)`，不额外加 L1/log，以分离渲染导数本身的正确性。之后仍需单独比较作者式损失。梯度检查固定主射线（每像素 1 个固定子像素位置，多条 transport），不同于训练时随机 4 AA；这是为了让 +/- 扰动在相同输入上比较。

原 render_ray 的前向 seed 与反向 seed 不同。自动微分保持原调用；有限差分的 primal 使用对应的**反向重放 seed**，而不是前向 seed。扰动量为 0.001、0.003、0.01，并在多个种子下比较。简单 diffuse、max_depth=2 的控制场景避免颜色改变路径采样分布造成的干扰；实际 principled 场景仍可能因采样分布变化、截断边界等产生不稳定，报告会保留 `inconclusive`。

零信号不会自动判通过；只有专门设定的 clamp 控制项才把零当成预期。方向导数的通过容差是绝对误差 2e-4、相对误差 5%，加上配对差值的采样误差估计；还要求至少两个扰动幅度通过、无明显失败且 epsilon 稳定。大采样误差不能用来放宽到随意通过。

所有诊断扰动仅存在于独立进程内存中，完成后恢复参数；磁盘上的输入 checkpoint 只读，并在最后再次校验其哈希。

## 如何解读并进入后两步

`ready_for_controlled_5000_comparison=true` 表示本次 all 的有限检查均通过，可以开始固定 5000 核的受控对照；并不保证后续 full 质量，也不排除未覆盖区域的错误。

- `saved_vs_online` 异常：先查数据生成/颜色/场景约定，不调训练学习率。
- GT 的 ray/native 异常：先查射线采样、积分和像素对应。
- 只有 prediction ray/native 异常：重点查 HKTex inference 状态或纹理桥接。
- 简单平面梯度失败：先查桥接和 PRB 调用。
- 简单平面通过、真实场 PyTorch 梯度失败：重点查场求值/截断/局部非光滑问题。
- PyTorch 通过而真实场 PRB 失败：重点查渲染反向传播。
- 中性场通过、原 checkpoint 只有零信号或不稳定：可能是已经饱和或跨截断边界，不足以归因于桥接 bug；可据报告决定是否从头建立基线。

后两步应分开：

1. 固定 5000 核、关闭 ADC，同一数据/场景/初值/采样预算，比较线性 loss 与经数值验证的作者式 loss。作者默认在线 GT 的原样复现可另做参考，不能把它与该受控对照混为一个实验。
2. 确认颜色学习正常后，再接入 ADC，明确误差统计的颜色空间、batch 缩放和命中量含义，校准阈值并记录候选数。位置学习率等应单独逐步恢复，不一次改变所有参数。

## 回退

没有训练源码补丁需要撤回。停止使用这个目录即可；原训练、两个 observer 和 checkpoint 保持不变。输出位于独立的新目录。

## 源码一致性

`expected_sources.json` 固定了本次检查过的 main 版本源文件哈希；checkpoint 自己保存的核心源文件哈希也会检查。源码不匹配时停止并列出文件，避免验证的是另一套实现。不要为了绕过错误直接删除哈希检查；先确认改了哪些文件。

如只需重做某阶段，可使用 `--stage radiance` 或 `--stage gradients`。单阶段不会把 `ready_for_controlled_5000_comparison` 标为 true，需结合两次报告人工判断。
