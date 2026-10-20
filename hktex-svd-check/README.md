# HKTex 特征向量对齐 SVD 诊断

适用日志：SMPL 静态检查通过，在 `eigen_albo.py::_precompute_all_aniso_eigen` → `lbo.py::align_eigen` → `scipy.linalg.svd(correlation_matrix)` 抛出 `SVD did not converge`。本次还没有进入训练循环。

本补丁只处理特征向量对齐处的 SVD。原路径成功时保持原行为；发生 LinAlgError 时记录矩阵并尝试双精度 `gesvd`，检查返回结果的重构误差和正交性。非有限输入、两种方法均失败或精度检查失败都会停止，不填充假值、不跳过对齐、不全局替换 SciPy。

## 执行

上传 ZIP 到 `/root/hktex-work/hktex/`：

```bash
conda activate hktex
cd /root/hktex-work/hktex
python -m zipfile -e hktex-svd-check.zip .
python hktex-svd-check/test_svd_check.py
python hktex-svd-check/apply_svd_check.py --repo /root/hktex-work/hktex --apply
python smpl-training/run_smpl.py --phase smoke
```

安装器在原 `hktex/utils/lbo.py` 中精确匹配 SVD 调用，保留 `lbo.py.before-svd-check-时间戳.bak`；源码不符合预期则拒绝修改。重复安装输出 `ALREADY PATCHED`。不修改模型、纹理、依赖版本、训练参数或现有剪枝/分裂补丁。

本次日志中原谱缓存不存在，且源码是在全部预计算成功后才保存缓存，因此这次失败通常不会留下完整谱缓存，无需先删除文件。若重跑直接命中了已有缓存，则没有测试到此次补丁；应先核实那个缓存的来源。不要删除其他实验缓存。

## 查看结果

- 再次遇到原 SVD 失败：终端显示 `[SVD-CHECK] Original SVD failed; diagnostic: ...`。
- 备用方法成功：显示 `[SVD-CHECK] float64 gesvd passed: ...`，继续预计算；最终仍需完整跑完 100 步、渲染和保存。
- 备用方法失败：保留错误，发送以下诊断资料。
- 如果此次原路径全部成功，没有诊断文件也正常；这不能证明之前根因已经确定。

诊断目录：

```text
/root/hktex-work/hktex/outputs/eigen-svd-diagnostics/<时间戳-ID>/
  correlation.npy   # 原始失败矩阵；默认 k_eig=256 时为 256×256
  report.json       # dtype、有限性、版本、错误、备用方法的误差
```

此外保留本次训练 `.log` 和终端 `[CHECK] Report:` 对应的训练 `report.json`。两种 JSON 所在目录不同，请不要只凭文件名混淆。

## 数值设置和局限

1. 默认 SciPy SVD 使用 gesdd；备用方法使用 float64 + gesvd。成功后将结果转回原矩阵精度，检查实际返回值，避免后续谱缓存因该步骤整体提升到双精度而增加内存。
2. 不增加对角噪声，不将 NaN/Inf 替换成 0，不关闭有限性检查。
3. 备用方法对同一矩阵做分解，不重新计算上游特征向量。转成 float64 不能修复上游 float32 特征求解已损失的精度。
4. 这是数值诊断与有验证的备用路径，不证明网格、Laplacian 或特征对齐的所有数学性质正确，也不证明一定能解决该服务器问题。
5. 不因矩阵秩亏直接判模型损坏：SVD 本身可以分解秩亏矩阵。
6. 如果仍失败，应优先分析保存矩阵，再检查上游谱残差、质量矩阵、相关特征和 BLAS/LAPACK 后端；不要直接降低 k_eig、删除细长三角形或升级整套环境。
7. `Computing all eigendecompositions` 的 7 个进度项对应 0、30、60、90、120、150、180 度的外层角度循环；每个角度内还有多个各向异性参数。2/7 不能解释为已确认在第三个各向异性值失败。

本地验证覆盖正常路径、模拟默认求解器失败后调用真实 gesvd、秩亏矩阵、非有限输入、两种方法均失败和错误数值输出。未获得服务器失败矩阵，未在本机重现其实际 CUDA/谱预计算失败。

回滚：确认安装器打印的备份路径，再将该备份复制回 `hktex/utils/lbo.py`。不要用通配符覆盖其他版本。若用不同数值方案比较实验，分别记录代码版本并使用不同谱缓存名。

来源：[HKTex lbo.py](https://github.com/circle-group/hktex/blob/9e942972c5411a364f12563eb6864520f9effe70/hktex/utils/lbo.py)、[谱预计算](https://github.com/circle-group/hktex/blob/9e942972c5411a364f12563eb6864520f9effe70/hktex/modules/eigen_albo.py)、[SciPy SVD 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html)。
