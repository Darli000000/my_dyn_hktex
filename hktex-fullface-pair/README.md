# 单帧完整可见脸部：无 ADC / 有 ADC 配对实验

本目录独立于旧 `hktex-real-face-pair`，不覆盖旧代码或数据，不修改 HKTex 核心代码。
继续使用之前运行成功的 `hktex` conda 环境和 tracer eps / stable KNN 适配。

## 最短运行方式（本次018第0帧）

把压缩包放到服务器仓库根目录后：

```bash
conda activate hktex
cd /root/hktex-work/hktex
unzip -n hktex-fullface-pair-v1.zip
bash hktex-fullface-pair/run_full_pair.sh
```

默认读取：
`outputs/real-face-eyes-20261031-024447-417163/dataset`

脚本顺序执行：
1. 校验旧数据及预制mask对应关系，生成全新的 `$RUN/dataset`。
2. 实际GPU首命中、表面移动、各区域前向/梯度检查。
3. 独立副本ADC clone/split/prune smoke及操作后检查。
4. baseline 5000步（linear、ADC关闭）。
5. adc 5000步（相同初始化与像素流、linear、ADC开启）。
6. expanded ROI / 原始ROI / 分区指标和最终图片比较。

正常情况下不暂停，不需要重复输入训练命令；任何阶段报错立即停止。
新输出默认位于 `outputs/real-face-full-pair-日期-时间-PID/`。
可以在命令前指定 `SOURCE_DATASET`、`HKTEX_ENV`、`GPU`、`RUN`、`CONFIG`。
`RUN`必须是新目录；仍不支持训练中断后resume。

```bash
SOURCE_DATASET="/root/hktex-work/hktex/outputs/real-face-eyes-20261031-024447-417163/dataset" \
GPU=0 bash hktex-fullface-pair/run_full_pair.sh
```

## 新mask内容和验证范围

包内 `prepared_masks.zip` 是已在提供的16视角上运行CPU人脸分割后得到的结果。它是新mask、标签、mesh标签元数据、manifest及预览的差分包，不含原RGB浮点数据，不含分割模型权重。
`apply_masks.py`从原dataset复制RGB、相机对应的几何缓存等，并逐文件核验。旧manifest必须与包内记录完全一致，不能把这套mask套用到另一帧或另一人。

实测旧监督像素862,932，新监督像素1,528,485，约增加77.1%。这是16视角投影像素数量，不是人头表面积覆盖率。
训练集分区像素为：

| 区域 | 数量 |
|---|---:|
| 基础皮肤 | 587103 |
| 眼周/眉毛 | 200713 |
| 左眼球 | 5067 |
| 右眼球 | 4329 |
| 外侧嘴唇 | 24279 |
| 鼻部 | 51312 |
| 耳廓 | 42512 |
| 颈部 | 417742 |

筛选规则：
- 完整原mesh用于第一命中和遮挡，几何、顶点和面顺序保持不变。
- 图像语义排除头发、帽子、衣服、耳饰、项链、眼镜和背景；默认排除mouth（嘴唇之间的口腔）与牙齿，保留上下唇。这不是完整口腔模型。
- 包含几何上有效的额头、脸部皮肤、眼周、鼻部、外唇、耳廓、颈部；是否属于左/右眼球由mesh连通部件决定，不依赖分割模型左右命名。
- 前景alpha至少0.99，朝向cos至少0.25；开放边界相邻三角面不监督。
- 对排除的图像遮挡区域扩张1像素；普通有效区域边缘腐蚀1像素，眼球/嘴唇不额外腐蚀。鼻部与皮肤等内部语义边界不单独腐蚀。
- 置信度是所有允许语义类别概率之和，阈值0.8；避免因为skin/nose/brow之间分类不确定而挖出细缝。
- 只使用当前单帧的图像。保留val相机7、test相机15，均不加入外观训练。

16视角叠加图已检查过，未把整片头发或衣服纳入。分割结果仍是自动提议，并非人工逐像素真值。遮挡、鼻孔、部分眼睑/边缘仍可能留空，不应强行补满。
新ROI会去掉旧ROI中少量头发边缘或不可靠像素，因此不是严格超集。

## 训练与ADC设置

`config.json`仍为每组5000步、初始5000 kernels；与上次paired保持相同模型、Adam倍率0.3、位置学习率0.001及余弦下降至10%。

每步1024个像素，八区域抽样配额依次为320/128/96/96/128/96/64/96。缺少可选区域时按比例重新分配配额，按区域实际像素占比修正loss和ADC统计。过采样用来保证小区域经常被看到，不代表把它们的平均loss强行放大。
两组同初始化文件、同采样seed，最后核对采样流hash。CUDA反向归约仍可能有非确定性；第999步参数/面索引差异继续记录，不能宣称全部更新逐位一致。

ADC沿用上一轮参数，不额外调优以混淆此次覆盖扩展：
- 增密1000、1200、…4000，共16次。
- 剪枝1200、1600、…4000，共8次。
- 每次增密/剪枝数量上限各5%，总kernel上限10000、下限2500。
- 部件最低保留数仍为头部512，其余部件32（不高于初始数）。
- 第4001～5000步停止ADC继续收敛。
- 新增每次ADC前后的固定验证ROI采样点评估：相同点、无额外优化器更新，记录RGB变化和前后MSE。它用于定位clone/split/prune带来的瞬时外观扰动，不参与阈值调节或训练梯度。

## 评估口径

`comparison.json`中的每组结果同时包含：
- `val` / `test`：新ROI；`all`为包含颈部的整体指标。
- `val.face_without_neck` / `test.face_without_neck`：按语义标签排除颈部，避免大片颈部掩盖面部误差。
- `legacy_val` / `legacy_test`：原dataset的完整旧ROI，哪怕其中一些像素不再训练也仍然评价；可与上次5000步对应结果比较。
- `regions`：八区域MSE/PSNR。

全部为linear RGB、固定照明表面颜色；不是albedo。几何来自使用全部相机的VHAP拟合，val/test只是外观留出视角，不是整个重建过程完全留出。
旧版固定ROI也可能包含头发边缘，因此legacy指标的意义是保证可比，不代表它是完美真值mask。
train_probe整体MSE做了抽样权重校正；正式结论以完整val/test ROI为准。

## 输出

```text
dataset/coverage_contact_sheet.jpg    16视角：原图 / 旧ROI / 新ROI / 区域标签
dataset/view_XX/overlay.png           单个视角高清叠加图
dataset/report.json                  覆盖变化与分区像素数量
setup/preflight.json                 CUDA初始检查
smoke/report.json                    ADC操作smoke
baseline/report.json                 无ADC报告
adc/report.json                      ADC报告
adc/adc_events.jsonl                  实际增密/剪枝及appearance_change
comparison.json                      新旧ROI统一指标、kernel数量
comparison.png                       0/7/15视角最终GT/无ADC/有ADC
run.log                              完整连续执行日志
```

仍自动保存0、300、500、1000…5000步的模型、预览和指标；ADC保存每次事件前后快照。
模型原生`field.pt`和精确恢复的`state.pt`同时保存。当前只显示新ROI；完整UV/GLB烘焙不在本包内。

只补渲染最终 / 所有检查点：

```bash
PAIR="/root/hktex-work/hktex/outputs/real-face-full-pair-实际日期-时间-PID"
python hktex-fullface-pair/render.py --pair "$PAIR" --arm adc --which final \
  --output "$PAIR/render-adc-final-$(date +%Y%m%d-%H%M%S)"
python hktex-fullface-pair/render.py --pair "$PAIR" --arm adc --which all \
  --output "$PAIR/render-adc-all-$(date +%Y%m%d-%H%M%S)"
```

将`--arm adc`换成`--arm baseline`即可渲染另一组。

## 可选：重新生成人脸mask（默认运行不需要）

公开模型：https://huggingface.co/jonathandinu/face-parsing
固定版本：`758b82e15a0178c9db39c1ff666a8b56e3a550c8`。
采用作者提供的量化ONNX；按模型卡用于非商业研究/教育。
模型在本机/服务器推理，图像不上传。新环境避免影响工作中的hktex环境：

```bash
conda create -n hktex-face-mask python=3.10 -y
conda run -n hktex-face-mask python -m pip install 'onnxruntime==1.23.2' 'numpy<3' 'Pillow>=10'
conda run --no-capture-output -n hktex-face-mask python hktex-fullface-pair/prepare.py \
  --source "/root/hktex-work/hktex/outputs/real-face-eyes-20261031-024447-417163/dataset" \
  --output "/root/hktex-work/hktex/outputs/fullface-custom-dataset"
PREPARED_DATASET="/root/hktex-work/hktex/outputs/fullface-custom-dataset" \
  bash hktex-fullface-pair/run_full_pair.sh
```

如果需要人工修订语义标签，可在一个新目录保存对应的`view_00.png`等原尺寸单通道标签图，值0..18按模型标签；用`prepare.py --semantic-overrides 该目录`重新生成新dataset，禁止直接篡改已验证manifest中的文件。
人工标签图是整张替换，不能把RGB预览当标签。已经生成过mask的16张当前图不需要重跑这一节。

## 本地验证与限制

真实16视角语义推理、mask生成与数据包重建已在CPU执行；原数据和几何不变、旧ROI保留、哈希核验均有检查。
9项原ADC/优化器CPU测试与8项新mask/采样/旧ROI测试通过。
没有在开发机器执行实际HKTex CUDA训练，服务器入口会严格执行setup与smoke。
不要把CPU生成成功等同于新区域GPU前向/梯度已经通过。
