# GPT Pro — Final Round

读完整个 zip 后请回答以下问题。

## 现状

纯经典 CV pipeline 数蚕蛹，10 张手标图（1133 只）。

最优结果：0.80x work_scale, err=9, 7/10 张图完全准确。延迟 470ms。

## 穷尽了的方向

我测了 40+ 种方法（详见 FINAL_SUMMARY.md），包括：
- 所有经典 CV 形状特征（EDT, watershed, h_maxima, concavity, ellipse fit, GMM）
- HOG + GradientBoosting（74% CV）
- CNN classification + regression（64×64 和 96×96，0.67x 和 native crop）
- 762 个手标组件作为训练数据

**决定性发现**：`round(area/200)` 的 90% 准确率无法被任何方法超越。k=2 vs k=3 的最高准确率是 51%（= 随机猜）。原因：蚕蛹是均匀棕色椭圆，缺乏可学习的视觉特征来区分 touching pair 和 touching triplet。

## 问题

1. **你认为这个 pipeline 真的到顶了吗？** 还是有我没想到的经典 CV 或轻量 ML 方法可以在不引入大模型的前提下突破 k=2 vs k=3？

2. **延迟优化**：0.80x 版本 470ms，主要在 core pipeline（color conversion + Gaussian blur + per-component peak detection）。有没有 Python 层面的加速方案？

3. **如果确实到顶了**，最小成本引入一个 instance segmentation 模块的方案是什么？不需要 Cellpose 那么重，只需要对 ambiguous component crop 做局部分割。

4. **762 个手标组件**是否足够训练一个小的 segmentation head？如果不够，需要多少？

## 文件

- `FINAL_SUMMARY.md` — 完整技术总结
- `code/` — 6 个核心源文件
- `data/` — 10 图手标 + benchmark
- `labels/` — 5 轮组件标注（762 个）
- `images/` — 0.80x 最终 overlay
