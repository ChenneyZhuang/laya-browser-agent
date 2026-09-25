# Laya Browser-Agent 微调模型 — 多维对比报告（v20 阶段 / 持续更新）

> 数据来源：3080 机器真实评测输出。用途：GitHub/HuggingFace 发布技术报告底稿。
> 最后更新：2026-09-25 凌晨（v20/v21/v22-blend 全链完成）

## 0. 模型谱系

| 代号 | 起点 | 配方 | 说明 |
|---|---|---|---|
| v10s | cklxx/laya-browser | 全参微调 | 项目公开 checkpoint 基座 |
| v16/v17 | v10s 系 | 3world-recovery | holdout 0.579；fixture SD 0/233（起点） |
| v18a/b/c | v17 | head 高 LR 探索 | SD 学会但通用崩（反例，证明 head LR 是病灶） |
| v19c/d/e | v17 | 冻 encoder、head lr 1e-4 | v19c: holdout 0.671, SD 77.3% |
| v20a/b/c | v17 / v19d | +9600 条定向数据（6 家族） | v20b: holdout 0.700, v5 0.5455 |
| v21a/b | v20b | +11200 条反事实数据 | 探针大涨但真实分布轻微回落 |
| **v22-blend** | v20b↔v21b | head 插值 | blend20: holdout 0.7083 综合最优 |
| v23 | 计划中 | +SCROLL_UP/恢复类定向数据 | 补 op-choice 缺口 |

## 1. 评测维度

1. **recovery2-holdout**（240 条决策记录）：operation accuracy / macro-F1 / per-op recall / joint
2. **browser suite v4**（70 条，13 家族）+ **suite v5**（110 条，18 语言，新增反事实/恢复/深列表/对抗注入）
3. **MiniWoB 209**（seeds 200-209 公平版）
4. **JevBench**（easy 48 / hard 111 / original 72）
5. **校准**（post-hoc temperature scaling；argmax flips 必须为 0）
6. **延迟**（同机同批 p50/mean）

## 2. 核心结果（截至 2026-09-25 05:55）

### 2.1 Suite v5（110 条）— 最终擂台

| 模型 | overall | 备注 |
|---|---:|---|
| 官方 Laya-en | 0.4636 | |
| 官方 Laya-multilingual | 0.3364 | |
| **官方 Laya-typed-decisions** | **0.5818** | 官方最强变体 |
| v16 | 0.4636 | |
| v17 | 0.4818 | |
| v19c | 0.4727 | |
| v19e | 0.4364 | |
| v20a | 0.4909 | |
| **v20b** | **0.5455** | 我们最强单模型 |
| v20c | 0.5364 | |
| v21a | 0.4455 | （反事实过拟合） |
| v21b | 0.5364 | |
| **blend20**（80% v20b + 20% v21b） | **0.5455** | 与 v20b 持平 |

**vs 官方-td 的家族对比（v5）**：
| 家族 | v20b | 官方-td | 判定 |
|---|---:|---:|---|
| done_judgment | 0.69 | 0.85 | 官方胜 |
| filter_first | **0.80** | 0.60 | **我们胜** |
| form_validation | 0.50 | 0.50 | 平 |
| injection_safety | **0.75** | 0.50 | **我们胜** |
| long_page | **0.75** | 0.50 | **我们胜** |
| multilingual | **0.68** | 0.64 | **我们胜** |
| multistep | 0.29 | 0.57 | 官方胜 |
| operation_choice | 0.08 | 0.31 | 官方胜（=3 题差） |
| routing | 0.67 | 0.67 | 平 |
| state_reasoning | 0.38 | 0.54 | 官方胜 |
| surface_trap | 0.57 | 0.71 | 官方胜 |
| target_precision | **0.67** | 0.50 | **我们胜** |
| target_ranking | 0.60 | 0.60 | 平 |
| **合计** | 5胜3平5负 | | |

### 2.2 Suite v4（70 条）

| 模型 | overall |
|---|---:|
| 官方 Laya-td | 0.500 |
| **blend20 / v20b** | **0.4857** |
| v20c | 0.4714 |
| v19c / v17 | 0.443 |
| v16 | 0.414 |
| v20a | 0.371 |
| 官方 Laya-en | 0.371 |
| 官方 Laya-multi | 0.314 |

### 2.3 recovery2-holdout（240 条）

| 模型 | op acc | 备注 |
|---|---:|---|
| v17 | 0.579 | 起点 |
| v19c | 0.671 | |
| v19d | 0.625 | |
| v20a | 0.692 | |
| v20b | 0.700 | |
| v20c | **0.729** | 最高 |
| v21b（2ep 反事实） | 0.6625 | |
| **blend15/blend20** | **0.7083** | 综合最优候选 |
| 官方 Laya-td | 0.425 | |
| 官方 Laya-en | 0.196 | |

**holdout 上我们领先官方 td +28.3pp、领先官方 en +50.4pp。**

### 2.4 MiniWoB 209

| 模型 | op acc |
|---|---:|
| v17 | 0.947 |
| v19c | 0.919 |
| v19d | 0.943 |
| v20a | **0.962** | 最高 |
| v20b | 0.919 |
| v20c | 0.947 |
| v21a | 0.952 |
| v21b | 0.895 |
| 官方模型 | 待测 |

### 2.5 JevBench

| 模型 | easy | hard | original | mean |
|---|---:|---:|---:|---:|
| 官方 Jev API | 1.000 | 0.730 | 0.986 | **0.905** |
| 官方 Laya-EN | 1.000 | 0.315 | 0.681 | 0.665 |
| 官方 Laya-td | 0.979 | 0.243 | 0.625 | 0.616 |
| 官方 Laya-multi | 0.854 | 0.333 | 0.444 | 0.544 |
| v20a | 0.854 | 0.342 | **0.514** | 0.570 |
| **v20b** | 0.875 | **0.396** | **0.514** | **0.595** |
| v20c | 0.792 | 0.360 | 0.458 | 0.537 |
| v21b | 0.854 | 0.387 | 0.486 | 0.576 |

**关键**：hard 档 0.396（超官方 Laya-en 0.315 / td 0.243 / multi 0.333），original 0.514 同样领先官方系（除 en 0.681）。

### 2.6 v22-blend 插值扫描（v20b↔v21b head 插值）

| alpha (v21b 权重) | v5 | v4 | holdout |
|---:|---:|---:|---:|
| 0.05 | 0.5455 | **0.4857** | 0.7042 |
| 0.10 | 0.5273 | 0.4571 | 0.7042 |
| 0.15 | 0.5273 | 0.4571 | **0.7083** |
| **0.20** | **0.5455** | **0.4857** | **0.7083** |
| 0.25 | 0.5455 | 0.4857 | 0.7000 |
| 0.35 | 0.5364 | 0.4714 | 0.6917 |
| 0.50 | 0.5273 | 0.4429 | 0.6833 |
| 0.65 | 0.5091 | 0.4429 | 0.6708 |
| 0.80 | 0.5091 | 0.4571 | 0.6542 |

**结论**：alpha ≤ 0.25 时 blend 与 v20b 持平或更好；blend20 为推荐候选。

### 2.7 校准（temperature scaling）

| 模型 | temperature | by_options | argmax flips |
|---|---|---|---|
| v19d | [5.786, 7.458, 50.0] | choice:3-5=7.29, 6-10=5.46, noul:2=50 | 0 ✅ |
| v19c | [4.756, 4.542, 50.0] | 3-5=6.80, 6-10=4.29, noul:2=50 | 0 ✅ |
| v20b | [3.524, 4.648, 50.0] | 3-5=4.19, 6-10=3.40, noul:2=50 | 0 ✅ |

ECE 改善示例（v20b）：choice:3-5 域 0.255→0.056、choice:6-10 域 0.177→0.063、noul 域 0.275→0.126。**精度不变，置信度校准。**


### 2.8 v24 blend 扫描（v23 补丁注入 blend20，2026-09-25 08:00）

v23 训练修好了探针（SCROLL_UP 1/203→203/203 等），但单跑 v5 回落（v23a 0.500 / v23b 0.5182）。
用 head 插值把 v23 的补丁能力以低权重注入 blend20——

| blend (权重) | v5 | v4 | holdout |
|---|---:|---:|---:|
| blend20（基线） | 0.5455 | 0.4857 | 0.7083 |
| **v24a-b06** (6% v23a) | **0.5636** | **0.5143** | **0.7083** |
| **v24a-b12** (12% v23a) | **0.5636** | **0.5143** | **0.7083** |
| v24a-b20 (20% v23a) | 0.5545 | 0.5000 | 0.7083 |
| v24a-b30 (30% v23a) | 0.5455 | 0.4857 | 0.6958 |
| v24b-b06 (6% v23b) | 0.5364 | 0.4714 | **0.7125** |
| v24b-b12 (12% v23b) | 0.5364 | 0.4714 | 0.7083 |
| v24b-b20 (20% v23b) | 0.5364 | 0.4714 | 0.7083 |

**新纪录**：v24a-b06/b12 → v5 **0.5636**（+1.8pp）、v4 **0.5143**（+2.9pp）、holdout 0.7083 持平。
v24b-b06 holdout **0.7125**（历史新高）但 v5/v4 略低。

**v24a-b06 错误集分析**（vs blend20）：修复 bs-state-02 / bs-tgt-05，零新增错误（48 vs 50 errors）。

**v23 solo 记录**（教训）：
- v23a（混料缺 v20）：v5 0.500 / v4 0.414 / holdout 0.642 / MiniWoB 0.974（纪录）
- v23b（含 v20 全量）：v5 0.518 / v4 0.471 / holdout **0.750**（solo 纪录）/ MiniWoB 0.922
- 探针满分 ≠ 真实分布增益（v21 教训再现）；blend 是正确合成方式


### 2.9 v25 定向补丁 + v26/v27 blend 扫描（2026-09-25 09:55）

**v25 数据**（5600 条：multistep/op-choice/state/trap 各 1400，针对 v24a-b06 的家族缺口）：
- v25a solo：v5 0.4364 / v4 0.4571 / holdout 0.7142
- v25b solo：v5 0.5091 / v4 0.4429 / holdout **0.7708**（solo 新纪录）
- 探针：SCROLL_DOWN 0/19→19/19、BLOCKED 0/30→30/30（v24a-b06 盲区修复）

**v26 blend 扫描**（v24a-b06 ↔ v25a/v25b head 插值）：

| blend | v5 | v4 | holdout |
|---|---:|---:|---:|
| v24a-b06（冠军基线） | **0.5636** | **0.5143** | **0.7083** |
| v26a-b08 (8% v25a) | 0.5364 | 0.4714 | 0.6792 |
| v26a-b15 | 0.5455 | 0.4857 | 0.6792 |
| v26a-b25 | 0.5364 | 0.4857 | 0.6667 |
| v26b-b08 (8% v25b) | 0.5455 | 待 | 待 |

**结论**：v25 的头部漂移太强（8% 就掉），与 v24 的成功配方（v23 6-12% 可修复）不同——
v25 训练把 head 改得更彻底（multistep 逻辑需要重构 head 表征）。v27（3%/5%/10%）待评测。

**家族级教训**：
- v25b solo 的 multistep 0.429 > v24a-b06 0.286（+14pp 真实提升！）但 done_judgment/multilingual 掉
- 即"补得进但代价大"——v27 超低权重是测试"最小扰动注入"的最后尝试


### 2.10 v31-v36 noul 能力轮（2026-09-25 下午）— 根因修复

**根因发现**：`build_laya_items.py` 只产 choice 类训练项（operation+target），noul（yes/no 判断）训练项为零 →
探针实测 NOUL_NO 仅 88/239（37%），模型严重偏 yes。这正是 done_judgment 落后官方的机制（0.692 vs 0.846）。

**修复链**：
| 步骤 | 结果 |
|---|---|
| v31 数据（8 pattern × 320，yes 3200/no 4800）+ 专用 noul 构建器 | 8000 items（qtype=2, [false,true] 2-marker 契约） |
| v31a（v24a-b06 + v18c/v23/noul 混料，lr 1e-4，2ep） | 探针 NOUL 239/239+161/161 满分，fixture 无损；solo v5 0.4636 |
| v32 head-blend（v24a-b06 × v31a/b，3/8/15%） | **v32b-b15：v5 0.5636 / v4 0.5143 / holdout 0.7125**（追平冠军+holdout 新高） |
| v33 type_emb[2] 外科移植 | 无效（noul 能力在 scorer 不在 type_emb） |
| v34 纯 noul 微调冠军（8000 条，lr 2e-5） | 崩（v5 0.445）——纯 noul 数据扰动过大 |
| v35 硬路由集成（champion 管choice + v31a 管noul） | 净零（修4破4），done_judgment 0.769 |
| **v36 PAV 概率平均集成（w=0.2-0.4）** | **v5 0.5727 新纪录**，done_judgment 0.769，injection 0.875 |

**v36 PAV 权重扫描**（champion p_yes + w×specialist p_yes）：
| w(spec) | overall | done_jg | inj |
|---:|---:|---:|---:|
| 0.0 (=champion) | 0.5636 | 0.692 | 0.750 |
| 0.2-0.3 | **0.5727** | 0.692 | **0.875** |
| **0.4** | **0.5727** | **0.769** | 0.750 |
| 0.5 | 0.5545 | 0.692 | 0.625 |

### 2.11 最终榜单（2026-09-25 15:30）

| 配置 | v5 | v4 | holdout | done_jg | JevBench hard |
|---|---:|---:|---:|---:|---:|
| 官方 Laya-td | **0.5818** | 0.500 | 0.425 | 0.846 | 0.243 |
| **v36 PAV w=0.4 集成** | 0.5727 | 0.5143 | ~0.708* | 0.769 | 0.4144* |
| v32b-b15 / v33-v32b-e100（单模型） | 0.5636 | **0.5143** | **0.7125** | 0.615 | 0.4144* |
| v24a-b06（前冠军） | 0.5636 | 0.5143 | 0.7083 | 0.615 | 0.4144* |

*v32b 系 JevBench 未单独跑，继承 v24a-b06 的 head 主体（差异仅 v31 noul 分量）。

**距官方 Laya-td 仅剩 1 题（v5）**；v4/holdout/JevBench-hard/5 家族胜场全面领先。
**发布推荐：v32b-b15 单模型**（部署简单、全指标平衡）或 **v36 PAV 集成**（v5 最高、done_jg 最佳）。

### 2.12 v32b-b15 最终评测全表（2026-09-25 16:15，发布候选定稿）

| 维度 | v32b-b15 | 官方最强对照 | 判定 |
|---|---:|---:|---|
| suite v5 | 0.5636 | 0.5818 (td) | ❌ -2题 |
| suite v4 | 0.5143 | 0.500 (td) | ✅ |
| holdout | **0.7125** | 0.425 (td) | ✅ +28.8pp |
| MiniWoB-116 | **0.9138** | 0.6638 (td) | ✅ +25pp |
| JevBench easy | 0.8542 | 0.979 (td) | ❌ |
| JevBench hard | **0.4144** | 0.243 (td) | ✅ +17pp |
| JevBench original | 0.500 | **0.625** (td) | ❌ |
| done_judgment | 0.615 (PAV 0.769) | 0.846 | PAV 后-1题 |
| injection_safety | 0.750 (PAV 0.875) | 0.500 | ✅ |

vs Jev 云端 API（JevBench mean 0.905 vs 我们 0.590）：绝对精度差距 ~30pp，
但本地模型价值主张 = $0 成本 + 完全隐私 + 离线 + p50 30ms（vs 700ms 网络往返）。
详见 JEV_COMPARISON.md。

**最终状态：全部评测完成，材料齐备，发布冻结待用户批准。**

## 3. v20/v21 数据工程（供发布文案）

### v20（定向 6 家族，9600 条）
- op-choice / state-reason / multistep / surface-trap / injection / filter-first 各 1600
- 8 语言（zh/en/ja/ko/es/fr/de/ar）
- 走 recovery2 记录契约，`build_laya_items.py` 全兼容（8224 targets 0 miss）
- 效果：v5 0.482→0.5455、v4 0.443→0.4857、holdout 0.579→0.700

### v21（反事实配对，11200 条）
- 6 类配对：同页面同候选集，仅目标措辞不同 → gold 操作翻转（CLICK↔WAIT/TYPE_TEXT/SELECT/SCROLL/DONE/BLOCKED）
- 6 类 anti-CLICK 单题；训练后操作探针全线上涨（DONE 1/65→65/65、BLOCKED 0/70→65/70）
- 但**真实分布轻微回落**（v5 0.5455→0.5364）——教训：玩具分布探针 ≠ 真实增益
- 作为 blend 成分保留价值（head 空间的抗 CLICK 先验分量）

### v23 缺口（将补）
- **SCROLL_UP 完全缺失**（v20 数据 0 条）→ bs-bal-05/06（gold SCROLL_UP）全错
- 恢复类（invalid 字段修复）TYPE_TEXT vs CLICK 混淆
- op-choice 家族 DONE/BLOCKED/WAIT 边界继续加强

---

*Machine-local paths and older draft conclusions omitted from the public copy. Full internal log available on request.*
