# HKTex A1：静态多视角离线图片导出与验证

版本：1.0.0（2026-10-25）

针对仓库 `Darli000000/my_dyn_hktex`，检查版本：
`8c926af44a3a12c80c3fdf1dd62e4f2ba5594630`。

本包完成 A 的第一步：生成可复现的数据集，核对相机与几何，重新读取数据验证渲染。
**本包没有实现外部图片训练器，不会启动 HKTex 优化。现有 optimisation.py 也不会自动读取这些文件。**
不修改项目现有代码、不使用 eigen 缓存、不修改模型或纹理，不需要重新安装依赖。

## 1. 安装

把 `hktex-offline-data.zip` 上传到 `/root/hktex-work/hktex/`，然后：

```bash
conda activate hktex
cd /root/hktex-work/hktex
unzip hktex-offline-data.zip
```

解压后应存在：

```text
/root/hktex-work/hktex/hktex-offline-data/
  offline_data.py
  offline_common.py
  run_export.sh
  test_offline.py
  README.md
  TESTING.md
```

如果已有同名目录，先将旧目录另存，不要将不同版本脚本混在一起。

## 2. 首先运行四视角 smoke 导出

```bash
bash /root/hktex-work/hktex/hktex-offline-data/run_export.sh smoke
```

默认读取：

```text
/root/hktex-work/data/SMPL_texture_samples/Textured_Mesh_samples/SMPL_male.obj
```

OBJ 引用的 MTL 和纹理图片必须仍然位于可加载的位置。

默认：256×256、16 spp、4 个视角（2 train / 1 val / 1 test）。
脚本自动设置 `MPLBACKEND=Agg` 和之前解决 ABI 问题使用的 Conda `libstdc++.so.6` 预加载。

依次执行：

1. 读取并检查 UV 几何与合并接缝后的几何对应。
2. 同步归一化到最长包围盒边长 2，保存实际变换。
3. 导出图片、相机、mask、纯几何与诊断参考。
4. CPU 检查文件完整性、相机矩阵、图像尺寸、取景范围和数据划分。
5. **另起一个 Python 进程**，只从数据集的参考资产重建场景，重放 train/val/test 各一个视角；不再读取原 OBJ、MTL 或纹理路径。
6. 同种子重放检查一致性，换种子渲染估计蒙特卡洛噪声。

输出根目录：

```text
/root/hktex-work/data/offline_multiview/smpl-male-smoke-时间戳-PID/
```

日志：

```text
/root/hktex-work/test-logs/offline-smoke-时间戳-PID.log
```

终端开头 `[OUTPUT]`、结尾 `[SUCCESS]` 会打印实际目录。保留该路径。

## 3. 验收 smoke

首先查看：

- `contact_sheet.png`：是否是完整人体，不裁头脚，相机确实绕模型改变视角。
- `validation.json`：`passed` 应为 `true`。
- `replay_validation.json`：`passed` 应为 `true`。
- 日志：应出现 `[SUCCESS]`，不能仅凭生成了图片认为全部通过。

`projection_error_px` 应不超过 0.02 像素；这是用实际相机射线求交结果投影回像素的检查。
`same_seed_rgb_max_abs` 与 `same_seed_alpha_max_abs` 应小于 1e-4。
`independent_seed_foreground_mse` 是两次独立噪声渲染之间的 MSE，不是训练 loss。
`estimated_single_render_noise_mse` 是其一半，作为单次渲染噪声方差的粗略估计。

如失败，先发日志、`manifest.json` 和存在的验证 JSON。**不要先增大容差，也不要启动训练。**

## 4. smoke 通过后导出 36 个视角

```bash
bash /root/hktex-work/hktex/hktex-offline-data/run_export.sh full
```

默认：256×256、128 spp，24 train / 6 val / 6 test。
俯仰角为 -15°、0°、20°；每层 12 个方位，层间略微错开方位。
划分固定且相机位置不重复。它用于首轮流程验证，不保证所有身体表面都被观察到。

确认质量和耗时后，需要更高分辨率时用新目录重新导出：

```bash
HKTEX_SIZE=512 HKTEX_SPP=256 bash /root/hktex-work/hktex/hktex-offline-data/run_export.sh full
```

每次生成独立目录，绝不覆盖先前数据。中断后没有自动续跑；换新目录重新导出，以免混入不一致设置。

可覆盖的环境变量：`HKTEX_REPO`、`HKTEX_MESH`、`HKTEX_DATA_ROOT`、`HKTEX_LOG_ROOT`、`HKTEX_SIZE`、`HKTEX_SPP`、`HKTEX_GPU`。

如需手动调整距离或 FOV，使用 Python 入口；输出目录必须不存在：

```bash
export MPLBACKEND=Agg
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
python -u /root/hktex-work/hktex/hktex-offline-data/offline_data.py export \
  --repo /root/hktex-work/hktex \
  --mesh /root/hktex-work/data/SMPL_texture_samples/Textured_Mesh_samples/SMPL_male.obj \
  --output /root/hktex-work/data/offline_multiview/smpl-custom-v1 \
  --preset full --size 256 --spp 128 --distance 3.7 --fov 40 --gpu 0

python -u /root/hktex-work/hktex/hktex-offline-data/offline_data.py verify \
  --repo /root/hktex-work/hktex \
  --dataset /root/hktex-work/data/offline_multiview/smpl-custom-v1 --gpu 0
```

## 5. 输出格式及训练边界

```text
manifest.json              数据契约、实际相机矩阵、数据划分、种子、版本、文件哈希
renderer_config.json       固定渲染场景配置；实际相机以 manifest 为准
mesh.npz                   vertices / faces / normals，无 UV、无真实颜色
mesh.obj                   便于外部软件查看的纯几何副本
rgb/view_XXX.npy            float32 H×W×3，线性辐射值，未来训练的权威 RGB
alpha/view_XXX.npy          float32 覆盖率，保留边缘混合信息
masks/view_XXX.png          0/255，首轮训练使用的保守内部区域
geometry_checks/*.npz      像素中心求交 mask 与相机 Z 深度，仅几何诊断
previews/*.png              经过 sRGB 编码的图片，仅预览
contact_sheet.png           所有视角的预览拼图
validation.json             文件与相机一致性检查
replay_validation.json      独立进程重放检查
eval_gt/                    原纹理参考；只供导出后验证与独立评估
```

使用 NPY 保存 RGB 是为了直接保留线性 float32 数组，无须额外 EXR 解码依赖；它仍然是完整二维图片，**不是预先生成的训练射线数据**。PNG 也完整保存，但仅供查看，不要拿经过裁剪和量化的 PNG 代替线性训练目标。

mask 是像素中心命中、alpha 接近 1 的交集，再向内收缩 1 像素，初期避免混合边界造成误差。
它不是完整人体分割；真实轮廓信息在 alpha 和 center_mask 中。
RGB、alpha 按 box filter 亚像素积分；未来训练器必须匹配同样的像素覆盖范围，不能任意把单条亚像素射线与完整像素平均值硬比较。

相机约定：

- `K` 使用图像左上边缘为 (0,0)，左上像素中心为 (0.5,0.5)。
- `world_to_camera_cv`：列向量形式，OpenCV 右/下/前相机坐标。
- `camera_to_world_mitsuba`：列向量形式，Mitsuba 左/上/前相机坐标。
- 正方形图片，水平 FOV，无畸变。实际矩阵优先于方位/俯仰文字参数。
- `geometry.source_to_world` 描述经过项目加载器合并场景后的网格坐标到归一化世界坐标的变换。
- 使用已归一化的 mesh.npz 时不能再次归一化，不能重新排列顶点、面片。

`eval_gt` 保存用于验证的 UV 网格、线性纹理及原始纹理图片。新训练器只能读取纯几何和多视角观测，不能用它做初始化或颜色监督。
独立 `validate` 命令不读取此目录；`verify` 属于允许使用原纹理的渲染诊断。

## 6. 与旧展示渲染有意不同的地方

本包复用你仓库中的 `load_mesh`、`UVTextureRenderer.mesh_to_mitsuba` 和场景构建。
调用 `mi.render` 显式指定随机种子与 spp，不调用带默认去噪的展示入口。

- 使用真正的球面相机坐标，避开原 `elevation_deg` 绕 Z 轴旋转的行为。
- 使用统一的 welded vertex normals，并映射回 UV 接缝顶点，保证未来 GT/HKTex 几何着色法线一致。
- 将源图片视作 sRGB，显式解码到线性值再交给浮点纹理。
  原代码的 float bitmap 输入通常是图片数值除以 255，未在该位置显式解码；因此不能直接拿旧模型的渲染结果做严格同条件比较。
- 简单 principled 材质：specular=0、metallic=0、roughness=1；不读取其他 PBR 贴图。
- 白色恒定环境光强度 1，无地面，path 最大深度 3。
- `hide_emitters=True`：环境仍照亮模型，但相机背景为黑色，从而保留可用的前景 alpha。
- 不去噪、不做曝光补偿、不做 tone mapping；仅 PNG 预览做 sRGB 编码并裁剪到 [0,1]。

这些是新数据集的固定实验条件，不修改作者代码。源纹理可能已经包含烘焙阴影，本导出器不保证其为真实物理 albedo。

## 7. 下一步与撤回

A1 验收后，下一步才新增外部图片 Dataset/Trainer，连接相机射线与已保存的 RGB。
需要同步固定材质、法线、光照、线性色彩和 max_depth，禁用原 UV 初始化。
不能直接沿用旧 multiview yaml 就声称已训练这份外部数据；旧配置仍会在线渲染原 UV。

本包只新增一个脚本目录和独立数据输出。撤回时移走新增的 `hktex-offline-data` 目录即可；旧训练和输出从未被修改。新导出数据可单独保留。

## 8. 本地测试边界

见 `TESTING.md`。提供真实 Mitsuba 3.7.1 CPU 测试，但没有在你的 CUDA 服务器上执行 SMPL 数据导出。
四视角 smoke 是对你实际依赖环境、模型文件和 GPU 路径的最终验证。
