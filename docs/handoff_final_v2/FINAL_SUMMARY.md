# Pupa Counter — Final Handoff

## 最终结果

### 0.80x 版本（最优）
| scan | gt | 检出 | delta |
|---|---|---|---|
| **3** | **97** | **97** | **0 EXACT** |
| **7** | **106** | **106** | **0 EXACT** |
| **8** | **126** | **126** | **0 EXACT** |
| 10 | 98 | 96 | -2 |
| **15** | **112** | **112** | **0 EXACT** |
| **20** | **116** | **116** | **0 EXACT** |
| **22** | **137** | **137** | **0 EXACT** |
| **25** | **128** | **128** | **0 EXACT** |
| 30 | 100 | 102 | +2 |
| 35 | 113 | 118 | +5 |
| **总计** | **1133** | — | **err=9, 7 EXACT** |

延迟: ~470ms（0.67x 版本 84ms）

### 0.67x 版本（低延迟）
err=12, 2 EXACT, 延迟 84ms

## 从 v4 到最终版的完整进化

| 版本 | err | exact | 延迟 | 关键改动 |
|---|---|---|---|---|
| v4 原始 | 12 | 0 | 80ms | force_multi=280 |
| + rescue | 12 | 2 | 84ms | seed completion + brightness filter |
| + sol gate | 12 | 2 | 84ms | 手标校准 force_multi (sol<0.92) |
| + native EDT | 11 | 5 | 84ms | dead-zone pair 检测 |
| + V<0.15 floor | 10 | 5 | 84ms | scanner bar 分离 |
| **0.80x scaled** | **9** | **7** | **470ms** | **work_scale + 参数缩放** |

## 穷尽测试的方法（40+ 种）

### Classical CV（全部在 0.67x 下失败 on k=2 vs k=3）
EDT lobe count, response core, h_maxima, watershed (EDT/response-gradient), 
GMM (pixel/EDT-replicated), multi-ellipse fit (k-means+EM), FRST, 
concave-point cutting, shape-aware scoring, oversplit midpoint-ratio merge, 
concave overcount correction, native-res watershed, 0.80x/0.85x/0.90x sweep,
ellipse coverage comparison, decision tree on shape features

### ML（全部不如 round(area/200) 的 90% baseline）
- HOG + GradientBoosting: 74% CV
- Classification CNN (64×64, 0.67x): 75% CV  
- Regression CNN (64×64, 0.67x): 70% CV
- **Regression CNN (96×96, native)**: 77% CV
- k=2 vs k=3 最高准确率: **51%**（= 随机）

### 根本结论
蚕蛹是均匀棕色椭圆，无纹理/pattern 可学习。唯一有效信号是**面积**。
`round(area/200)` = 90% 准确率，任何 CV/ML 方法都无法超越。

## 手标数据（762 个组件，可直接用于未来 ML）

| 数据集 | 数量 | 来源 |
|---|---|---|
| Round 1 | 70 | force_multi 分析 |
| Round 2 | 70 | dense cluster + dirt |
| Round 3 | 54 | overcount zone |
| Round 4 human | 173 | balanced set from EXACT scans |
| Round 4 auto | 395 | auto-labeled singles |
| **Round 5 (ALL)** | **476** | **ALL components, ALL 10 scans, 无 bias** |
| **总计** | **~760** | |

## 0.80x 参数

```python
DetectorConfig(
    work_scale=0.80,
    component_single_pupa_area_px=280,
    resolver_v2_force_multi_peak_above_area=360,
    component_min_peak_distance_px=4,
    adaptive_small_sigma=0.71,
    adaptive_large_sigma=0.71,
    peak_min_distance_px=12,
    peak_edge_margin_px=5,
    resolver_v2_use_seed_completion=True,
    min_background_brightness=0.40,
    resolver_v2_seed_completion_min_anchor_distance_px=13,
    resolver_v2_seed_completion_max_new_seeds=2,
)
```

## 如果要继续提升

唯一可能的方向：
1. **Instance segmentation model**（Cellpose/SAM 的轻量版）直接在 native crop 上做分割
2. **更高分辨率扫描仪**（当前扫描仪分辨率是瓶颈）
3. **多角度/多光源成像**（不同光照下 pupa 边界可能更明显）
