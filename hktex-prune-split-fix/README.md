# 修复：同轮剪枝后，分裂读取旧 kernel_info

## 本次定位

日志：`apple-full-20261011-044557-33490.log`。

训练已完成 1000 次迭代，进入零起始编号 `i=1000`，也就是第 1001 次 loss 计算时：

1. 本轮开头 `prepare_knn()` 生成了对应 5000 个核的 `kernel_info`。
2. 第一个密度控制器剪掉 1 个核，模型压缩为 4999 个核。
3. 第二个控制器在同一轮执行增密，试图分裂一个核。
4. `BaseDensityController.split()` 使用**当前模型**的位置与面片编号，却按**旧缓存**的同一个下标取重心坐标和面片顶点。
5. V4 检测到位置重建误差 `0.0374811664`，超过 `0.001` 容差，因而在原始 DiGeo 调用之前停止。

前一次正常追踪的重建误差约 `1.46e-7`。本次不是一般浮点舍入误差，不应放宽容差、截断坐标或关闭检查。

调用顺序见 `hktex/trainers/base.py`；缓存读取点见 `hktex/density_controllers/base.py` 的 `split()`。单独检查面片索引范围无法发现这种下标错位，因为错误组合中的各个数值仍可能单独合法。

## 已经通过的组件

- 数据相关初始化正常执行。
- 重要性采样实际运行；失败前 uniform=250、importance=751。
- 5 次克隆成功，共新增 296 个核。
- 5 次分裂成功，共分裂 3704 个父核，净新增 3704 个核。
- 核数量由 1000 增长到 5000。
- 第一次剪枝成功将 5000 降到 4999。
- 失败的是紧随剪枝发生的下一次分裂，未完成的这次分裂不计入上述成功次数。

## 修复内容

仅修改 `hktex/density_controllers/base.py` 的一个代码块：

```python
kernel_faces = self._model._kernel_face_ids[sel]
kernel_thresholds = self._model.thresholds[sel]
kernel_locations = self._model.kernel_locations[sel]
kernel_vert_idx = self._mesh.get_face_vertices(kernel_faces)
kernel_barys = self._mesh.cartesian_to_barycentric(
    kernel_locations, kernel_vert_idx
)
```

即根据当前被选中的位置和面片重新求重心坐标，而不是从剪枝之前的 `kernel_info` 取。新的面片顶点同时用于局部方向插值。阈值、调度、位移半径和剪枝/克隆/分裂规则不变。

本补丁不声称修复所有密度控制问题，也不能用于解释先前人头上 DiGeo 返回负面片编号的问题。

## 服务器操作

将 ZIP 上传到 `/root/hktex-work/hktex/`，激活 `hktex` 环境后执行：

```bash
cd /root/hktex-work/hktex
python -m zipfile -e hktex-prune-split-fix.zip .

# 展示将应用的差异，不改文件。
python hktex-prune-split-fix/apply_fix.py --repo /root/hktex-work/hktex

# 自动备份原文件后应用；如果代码块不匹配，会停止而不覆盖源文件。
python hktex-prune-split-fix/apply_fix.py --repo /root/hktex-work/hktex --apply

# CPU 回归测试；不导入 FAISS 或 DiGeo，不运行 GPU。
python hktex-prune-split-fix/test_regression.py --repo /root/hktex-work/hktex

# 测试全部通过后，用此前相同的完整训练命令重新开始。
bash apple-full-training/run_apple_full.sh
```

补丁脚本打印 `PATCHED`、备份路径和修改后 SHA-256；重复执行打印 `ALREADY PATCHED`。不需要替换 V4 启动器、不需要重装依赖、也不需要删除苹果特征分解缓存。

脚本会在源文件旁保存 `base.py.before-split-fix-<时间戳>.bak`。需要撤销时，在没有训练运行的情况下，将该备份复制回 `base.py`；如果之后还有其他代码修改，应先比较差异再恢复。

## 验证方式

保留原 5000 步配置重跑。不要只把 `optim.iters` 改成 1100 来缩短测试：原 YAML 中剪枝/增密时机依赖总步数，那样会改变触发时机，不再重复同一条训练路径。

重点观察第 1001 次 loss 计算附近：剪枝之后能否继续执行分裂、追踪输入重建误差是否回到数值误差量级、训练是否越过此处。继续让该轮跑满 5000 步，并检查第二次剪枝及最终输出。

若发生其他报错，保留检查，将新的日志和报告发来；不要为强行运行而关掉剪枝或位置一致性检查。

此前入口在训练完成、渲染之后才保存最终模型，本轮在此之前退出；工具包没有周期性完整状态 checkpoint，不能直接从第 1000 步恢复优化器和采样状态。因此此次建议从头重跑。

## 本地验证范围

使用 CPU PyTorch 2.10.0，执行从实际源文件提取的密度控制器方法。只替换测试用的简单网格、局部方向插值和追踪器，不依赖完整 HKTex/FAISS/DiGeo 环境。

- 原实现：复现剪枝后面片顶点缓存与当前面片错位。
- 修复后：普通分裂、克隆后分裂、剪枝后分裂、剪枝+克隆后分裂通过。
- 故意将旧缓存设成 NaN/无效顶点后，修复版仍能使用当前几何正确运行。
- 核数、参数/缓冲区长度及 Adam 一阶/二阶状态尺寸通过检查。
- 已检查补丁重复应用和源文件不匹配时拒绝修改的行为。

完整 CUDA 训练尚待服务器验证。
