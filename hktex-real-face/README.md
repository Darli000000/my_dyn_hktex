# 真实静态人脸：眼部适配、自动验证、300 步外观试训

对应已讨论的 subject 018 / 第 0 帧。将本目录放到 HKTex 仓库根目录：

```bash
cd /root/hktex-work/hktex
bash hktex-real-face/run.sh
```

前提：已应用 `hktex-tracer-eps.patch`；保留可运行 HKTex 的 `hktex` conda 环境。
默认复用 `outputs/vhap-stage12-20261031-012229-414642/regions.npz` 中已导出的 FLAME 标签。
若该文件不存在，自动在 `VHAP` 环境从 `/root/hktex-work/VHAP` 导出，仍需要原 FLAME 资产。
源 mesh、图片、相机和旧实验均不修改。

默认输入：`/root/hktex-work/data/vhap-export/018_EMO-1-shout+laugh-20261029_epoch30`。
默认输出：`outputs/real-face-eyes-日期-时间-PID/`。可以用 `EXPORT`、`SOURCE_REGIONS`、`VHAP_ROOT`、`FRAME`、`GPU`、`RUN` 环境变量覆盖。
`RUN` 必须是新目录。`STEPS` 默认 300，入口限制 1–1000，防止误当作正式长训练入口。

## 实际执行顺序

1. 严格验证标签与 mesh 拓扑相同，生成四组互斥面标签：base_face、eye_surround、left_eye、right_eye。显式合并眼球顶点，不依赖 face 标签是否包含眼球。
2. 全 mesh 做遮挡；原相机、550×802 图片不变。前景 alpha≥0.99、几何法线与视线余弦≥0.25、合并候选区域内缩 2 像素。内缩不在区域接口处重复做，避免额外抹掉眼球。
3. 对四个区域的所有候选像素检查 GPU 首命中和面朝向。不可见眼球不强行采样；要求汇总训练视角后每组至少32个像素。
4. 原 HKTex 完整谱配置、5000 kernels；使用同一个实际训练 tracer 检查零/小/中/大步、跨三角面、所有开放边界终点。显式 `eps=1e-6`、`avoid_holes=false`、`max_iterations=256`。
5. 每区域至多64个训练像素，检查 kernel 支持、跨部件权重、有限颜色、参数梯度及颜色有限差分。若无 kernel 支持、跨部件污染或无颜色梯度，就写报告并停止，不能靠全局 mean colour 掩盖失败。
6. 全部通过后自动进行300步训练。没有逐阶段确认要求；失败时不开始训练。

新增眼部区域的照片遮挡和对齐仍需看图。自动检查证明数值和接口一致，不代表睫毛/虹膜/眼睑的几何拟合绝对准确。
这里的 `eye_surround` 按 FLAME 区域标签及新增皮肤三角面定义，并非照片眼睑分割。
base_face 的面集合与旧版本一致，但合并 ROI 后边界内缩略有不同，不能当作严格同 mask 的消融。

## 训练口径

- 固定一帧 mesh 和相机，固定光照下的共享表面 RGB；不是物理 albedo，不额外添加光照/BSDF，不使用旧合成数据 trainer 的 PRB 路径。
- RGB 按标准 sRGB EOTF 解码一次，loss 为 linear RGB MSE。此颜色空间是工作假设，不代表相机辐射标定完成。
- 初始5000 kernels，原表面采样/FPS定位；所有部件保留，不为眼球额外增加 kernel。
- 沿用旧 image trainer 的近中性初始化：mean 0.35，颜色残差 N(0,0.003)，不使用留出视角颜色初始化。
- 优化颜色、方向、各向异性、阈值、锐度和 kernel 中心；不优化 mesh 顶点。ADC关闭。
- Adam参数学习率取仓库 small-ray 配置的0.3倍；GeodesicOpt学习率0.001。300步试训保持常数学习率，实际数值写入 settings.json。
- 每批1024像素：原面部512、眼周256、左右眼各128。loss按训练集各区域像素占比做重要性校正，目标仍是全部有效训练像素的均方误差，并非四区域等权。
- 相机索引7验证、15测试，其余训练。测试误差只在最终步计算；VHAP几何已使用全部相机，所以这是外观留出视角实验。
- 每10步记录区域batch误差、位置请求/实际移动、跨面数量。每步检查有限梯度、面ID、部件归属和过长移动，定期检查三角面内坐标。
- 初始、100、200、300步保存模型、验证误差及预览。检查点使用未压缩 float32，避免小尺寸人脸的位置量化误差。图中只显示受监督区域，白色部分是未显示区域，不是预测的皮肤颜色。
- `view_XX_eyes.png` 是眼部3倍最近邻放大，顺序为 GT、预测、线性RGB平均绝对误差×10（红色）。

`preflight.json: passed` 才能进入试训；完成后 `report.json: short_training_completed`，不自动宣称适合5000步训练。
眼球高光随视角变化可能导致共享表面RGB无法完全拟合，应结合每区域误差和图片判断。
谱预计算按几何、配置和代码哈希隔离缓存，第一次运行可能重新计算；后续同配置复用。

## 看哪些文件

```text
dataset/roi_contact_sheet.jpg          全部视角候选ROI
dataset/view_07/regions_preview.png    区域颜色：绿=原面部、黄=眼周、蓝=左眼、紫=右眼
fit/preflight.json                    每区域与几何检查结果
fit/report.json                       试训汇总、区域指标、状态
fit/metrics.jsonl                     训练变化
fit/network.yaml                      实际网络与tracer配置
fit/settings.json                     实际优化设置
fit/step_00300/previews/               最终图片与眼部放大图
fit/step_00300/field.pt                原HKTex state_dict
run.log                              完整日志
```

请先回传 `fit/preflight.json`、`fit/report.json`、`fit/metrics.jsonl`、ROI总览，以及最终 view_07 与 view_15 的完整图和眼部放大图。
若检查失败，发 `fit/preflight.json` 和 `run.log`；不用强行绕过检查。

代码只做本地CPU/数据验证，发布包不宣称已执行服务器CUDA训练。检查失败报告保留具体阶段和错误。
CPU测试：`python hktex-real-face/test_cpu.py -v`。
