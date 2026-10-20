# HKTex 同状态回放检查

用途：定位两份逐数组完全相同的 5000 步快照，为何产生不同预测。本工具只运行离线推理，不训练，不修改仓库源文件、训练配置、参数快照或已有 analysis/viewer。

## 已确认的事实

- 00081 与 00082 的全部 13 个数组逐字节相同。
- 8192 个点中有 624 个 RGB 发生变化，449 个点的最大通道差超过 1e-5；最大通道差 0.06740934。
- 已保存的 6 个代表核的热响应、权重、alpha、candidate/selected 标记均相同。注意这些只有 6 列，不是全部 50 个候选核的信息。
- 5000 个核中，1557 个属于 483 组位置完全重合的核，最大一组 19 个；每组记录的 face ID 也相同，但每组都存在不同颜色。

优先检查假设：外层 KNN 的等距离候选选择、候选顺序，以及后续 inner top-k。位置重合可能来自 clone，并不意味着这些核可以直接删除；其颜色和其他参数已不同。现有文件不足以直接证明 FAISS 是根因。

## 在服务器运行

上传 `hktex-replay-check.zip` 到 `/root/hktex-work/hktex/`：

```bash
conda activate hktex
cd /root/hktex-work/hktex
python -m zipfile -e hktex-replay-check.zip .

HKTEX_RECORDING='/root/hktex-work/hktex/outputs/smpl-observed-full/male_init05_lr10_20261019-044000-c77569@20261019-044005/observer'

set -o pipefail
env MPLBACKEND=Agg \
  LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" \
  python -u hktex-replay-check/diagnose_replay.py \
    --recording "$HKTEX_RECORDING" \
    --repo /root/hktex-work/hktex \
    --chunk 64 \
  2>&1 | tee "$HKTEX_RECORDING/replay-repeatability.log"
```

沿用原来的 CUDA 环境、TF32/编译设置和 chunk=64；第一轮会有模型/特征缓存加载及编译开销。原谱缓存只读取，不重新计算分解。源码和缓存哈希不匹配时会停止。

## 做了什么

只恢复一个最终状态，对同一批 8192 点运行三次：

1. **A_first**：准备模型和索引，执行一次完整推理。
2. **B_same_index**：保持参数、谱缓存和 FAISS 索引不变，再推理一次。
3. **C_rebuilt_index**：重新装入完全相同的参数并重建索引/派生缓存，再推理一次。

逐点保存所有 50 个候选的索引、距离、原始热响应、归一化值、加权值、alpha、self-heat，以及选中的 inner top-k 和最终颜色。对之前差异最大的 16 点与前 8 点，还保存具体谱值。比较相同候选的数值时按核索引对齐，避免把单纯排序变化误判为数值变化。

索引与谱缓存还会做内容指纹比较。候选集合发生变化时，进一步检查被替换的核是否具有完全相同的检索嵌入。

## 发回什么

结束时打印：

```text
[CHECK] Output: /root/.../observer/repeatability-日期时间
```

先发这个新目录中的 **summary.json** 即可；若失败则发 `replay-repeatability.log`。其他 NPZ 暂时保留在服务器上，必要时再取，不需要上传 1.4 GB 的特征缓存。

summary.json 的主要字段：

| 字段 | 含义 |
|---|---|
| `same_index_repeat` | A/B：同一索引重复查询的差异 |
| `rebuilt_index_repeat` | A/C：重建索引与派生缓存后的差异 |
| `candidate_order_changed` | 候选排列变化的点数 |
| `candidate_sets_changed` | 不计顺序，候选集合变化的点数 |
| `selected_sets_changed` | 最终选中的核集合变化的点数 |
| `common_candidate_stage_max_differences` | 对共有候选按索引对齐后的各阶段最大差异 |
| `changed_sets_replacing_identical_embedding_multisets` | 候选改变，但替换双方嵌入多重集合相同的点数 |
| `cache_fingerprints` | 重建前后数据库、插值网格、源谱和 self-heat 的内容指纹 |

NPZ 的 `indices` 是当前快照数组中的行索引；稳定 ID 映射保存在 `source_embeddings.npz` 的 `stable_ids`，不要混淆二者。

如果候选不变而同一候选的数值发生变化，再检查距离/谱求值的实现与执行顺序；如果候选集合因等距离核而变化，则有证据评估稳定 tie-break。不要在尚未定位前改动正式训练路径。

## 验证范围

已在本地测试：比较器对候选重排、相同嵌入替换、仅预测值比较的处理；原始上传数组比较；脚本语法与 CLI 参数解析。尚未本地运行 CUDA 检查，须在服务器验证。

诊断采集会增加 CPU 拷贝与同步，因此若这次没有复现，也不能证明旧回放没有问题；应保留旧结果。原来 `max_forward_decomposition_error` 很小，只说明一次推理内的公式拆解与原 forward 一致，不证明不同推理间的候选与结果一致。

撤回：不再运行本脚本即可，无需恢复源码。三个新结果保存在独立目录，旧输出不覆盖。完成该检查后，再继续修正 viewer 的连续网格渲染；两者是不同层面的问题。
