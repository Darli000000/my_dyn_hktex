# 验证记录

- 检查代码仓库：Darli000000/my_dyn_hktex @ 8c926af44a3a12c80c3fdf1dd62e4f2ba5594630。
- Python 语法编译、命令行 help、Bash 语法检查通过。
- `test_offline.py --mitsuba`：5 项测试通过，使用真实 Mitsuba 3.7.1 scalar_rgb CPU 后端。
- 覆盖：36 视角划分与真实俯仰角、sRGB 转换、保守 mask、数据完整性与相机参数篡改检测。
- 实际 Mitsuba 测试：多个方位/俯仰下的非中心像素射线与保存的 K/w2c 矩阵一致；浮点纹理保留线性输入；固定种子重渲染一致；隐藏环境光后的 RGBA 前景覆盖；实际求交深度/投影；RGB、mask、诊断数组的保存读取。
- 测试曾发现环境可见时背景 alpha=1，已显式使用 hide_emitters=True 修正并增加回归检查。
- 本机没有可用于 HKTex 的 CUDA GPU；没有运行仓库完整 CUDA 初始化、CUDA 向量化求交分支或实际 SMPL 文件导出，也没有执行任何训练。
- 本机 LLVM 后端不可用，测试使用 scalar_rgb；可能显示 LLVM 初始化提示，但测试不依赖 LLVM。

服务器可选的 CPU 合约检查：

```bash
python /root/hktex-work/hktex/hktex-offline-data/test_offline.py
```

最终以 `run_export.sh smoke` 在实际服务器上生成的 `validation.json`、`replay_validation.json` 与图片为准。
