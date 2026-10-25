# 代码来源与适配

参考用户仓库 `https://github.com/Darli000000/my_dyn_hktex`，提交 `8c926af44a3a12c80c3fdf1dd62e4f2ba5594630`。

- `observer_common.py`：提取 `hktex-observer/recording.py` 的原参数列表、原子保存、固定采样与 Lineage。
- `replay_image_fields.py`：适配 `hktex-observer/replay.py`，保留原谱项/KNN/颜色贡献计算及前向一致性检查，移除UV表面GT误差。
- `plot_image_fields.py`：适配 `hktex-observer/plot_recording.py`，保留数学/谱系面板，明确没有表面GT，修正训练损失的语义为RGB平均L1。
- `native_display.py`：提取 `hktex-native-viewer/render_native.py` 的稳定ID配色、环模式、总览拼图与HTML，改为图片监督的文字说明。
- `image_observer.py` / `render_native_images.py`：新增图片训练记录与校准相机回放适配。训练网络和数值核心仍从服务器repo导入。
- 其余训练入口基于此前交付的 hktex-image-training 1.0.0；1.1.0增加记录接入，full关闭旧低采样预览，没有修改原loss/优化器/ADC配置。

原仓库许可证随包保留在 LICENSE。新增目录不覆盖仓库原有observer/viewer源码。
