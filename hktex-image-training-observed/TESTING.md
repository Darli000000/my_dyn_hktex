# 验证记录：1.1.0

日期：2026-10-25。参考仓库：`Darli000000/my_dyn_hktex@8c926af44a3a12c80c3fdf1dd62e4f2ba5594630`。

## 用户服务器已提供的 1.0.0 训练证据

1500 步 ADC 完成，1500 次梯度检查。2 次 clone 共选择57个核，2次 split 共选择46个父核，2次 prune 共移除67个核；最终1036核。最终稀疏像素 MSE：train=0.00338750、val=0.00367353、test=0.00330872。
这些结果验证的是旧训练链路，不能代替新增 observer/native viewer 的集成测试。

## 本地已通过：11 项 CPU 测试

- `test_images.py --mitsuba`：4项，沿用数据隔离、采样状态、颜色/分辨率拒绝、实际 scalar Mitsuba 相机投影测试。
- `test_checkpoint.py`：1项，实际检查点保存/恢复方法 + CPU PyTorch；包含新 observer ID 状态的 weights_only 加载与恢复。CUDA 随机数接口被替换为CPU接口。
- `test_image_observer.py`：5项：
  - 记录器不改变参数或全局 NumPy/PyTorch 随机数状态；
  - 连续 clone/split/prune 的精确 ID 顺序与 root/parent 关系、恢复延续、哈希损坏拒绝；
  - 模拟 tracer 失败保留 started 意图，不推进 ID；
  - 环展示异常时恢复原参数对象；线性 box 平均；
  - 实际 Mitsuba scalar 裁剪相机与全图对应像素射线一致（后两项分别包含在独立测试中，总计5个测试方法）。
- `test_plot_images.py`：1项，人工构造无UV/无表面GT记录，实际生成全部8类数学面板、曲线、谱系和HTML。测试夹具不代表任何训练结果。
- 所有 Python 文件 AST 解析通过，启动脚本 `bash -n` 通过。

本地使用 PyTorch 2.10.0 CPU、Mitsuba 3.7.1 scalar、Dr.Jit 1.2.0。未执行 CUDA HKTex/PRB 完整训练或原生渲染。

## 服务器必须验证

1. 新入口50步smoke。
2. `image_observer.py --recording ... --checkpoint ...` 验证记录与最终检查点逐参数相等。
3. 一张128px/4spp native图，以及单快照数学回放，确认GPU侧读取和求值接口。
4. full/ADC 的事件前后快照和核身份关系；CPU模拟不能证明所有真实GPU路径。
5. 对现有ADC做512px/64spp的未去噪渲染，判断残余斑点。没有把噪声归因或画质改善当作已经验证的结论。

## 复测命令

在本包目录、已有hktex环境中：

```bash
python test_images.py --mitsuba
python test_checkpoint.py
python test_image_observer.py
python test_plot_images.py
bash -n run_training.sh
```

旧observer数学计算和native展示辅助代码被放在新目录的适配脚本中，原目录没有被修改。
