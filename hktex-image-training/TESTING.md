# 验证记录

2026-10-25，A2 1.0.0。检查参考仓库提交：
`8c926af44a3a12c80c3fdf1dd62e4f2ba5594630`。

## 已完成的本地检查

- Python 源文件语法检查与 `bash -n run_training.sh` 通过。
- `test_images.py --mitsuba`：4 项测试通过。涵盖像素内采样及光路排列、训练相机隔离、mmap/LRU 数据读取、采样状态恢复、路径与数据约定拒绝，以及真实 Mitsuba scalar 相机的 2048 亚像素射线投影。
- `test_checkpoint.py`：1 项测试通过。使用实际 CPU PyTorch 与脚本中的检查点方法，验证 `weights_only=True` 加载、参数、Adam 状态、调度器、采样状态的恢复。CUDA 随机数接口在这项测试中被替换为 CPU 接口。
- 四档配置用实际 OmegaConf 与参考仓库配置构建并检查；恢复配置的谱缓存键保持一致。
- 实际执行 `train_images.py check`，在合成的小尺寸 36 视角测试夹具上通过，划分为 24/6/6；夹具不含原始 OBJ/纹理或 `eval_gt/`。这不是对用户服务器数据的验证。

复查测试命令（在已配置的 hktex 环境内）：

```bash
cd /root/hktex-work/hktex/hktex-image-training
python test_images.py --mitsuba
python test_checkpoint.py
bash -n run_training.sh
```

本地采用 Mitsuba 3.7.1、Dr.Jit 1.2.0、PyTorch 2.10.0 CPU。这里只检查相机和数据的测试使用 scalar_rgb，不会实例化完整 CUDA HKTex 网络。

## 尚需服务器验证

- 用户实际 2048×2048 数据集的文件与相机检查。
- CUDA PRB → Dr.Jit/PyTorch → HKTex 参数的反向传播。
- DiGeo 的位置更新、FAISS 调用、显存占用及训练稳定性。
- ADC 实际 clone/split/prune 事件和变化核数后的恢复训练。
- 真实数据拟合效果、完整分辨率图像质量与收敛速度。

服务器按 `check → smoke → baseline → adc → full` 推进；前三项本地测试通过不等于完整训练已验证。
smoke 检查实际场景投影/深度、第一步核颜色梯度和参数变化，以及全程数值有限性。
ADC 开启但未触发某种事件时，不应将其记为该操作测试通过。

旧 observer 尚未接入；检查点是本入口的状态格式，不能直接当成作者的压缩模型文件传给旧 viewer。
