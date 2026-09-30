# laya-browser-agent

**[English](README.md)** | [中文](README.zh-CN.md) | [日本語](README.ja.md) | [Español](README.es.md)

> 本页面是 laya-browser-agent 的中文说明。英文原版（最新）见 [README.md](README.md)。

**面向 Jev 类模型的多后端、受约束决策支持框架。** 支持 Laya MLX/PyTorch、任意
`answer(state, questions)` 同步 duck backend，以及任意 System One 形状的 HTTP
endpoint；浏览器默认推荐本项目自己的 checkpoint，并非强制使用专有模型。

[![tests](https://github.com/ChenneyZhuang/laya-browser-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/ChenneyZhuang/laya-browser-agent/actions/workflows/tests.yml/badge.svg)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-ichenney%2Flaya--browser--v32b-yellow)](https://huggingface.co/ichenney/laya-browser-v32b)

决策模型对状态回答结构化问题，返回**校准过的概率**而非生成的文字——因此它不可能幻觉出一条指令。这正是浏览器 agent "决定"那一半的理想形态：给它一张页面上可交互元素的编号表，它告诉你下一步该执行什么操作、作用于哪个元素。

本项目把这类模型接进这个角色——**完全在本地运行**，适配你已有的任何 agent。

`model="browser"` 默认加载本项目训练的
> **[ichenney/laya-browser-v32b](https://huggingface.co/ichenney/laya-browser-v32b)**——
> 子目录为 `v32b`；`model="browser-legacy"` 选择 upstream `cklxx/laya-browser`
> 的 `v10s` 路径。由于 upstream `main` 已移除这个历史目录，legacy 别名默认固定到
> revision `adf912be85ff9221ee171551778456b133c1af75`；也可以显式传入 `revision=` 覆盖。
> 已提交的历史诊断中，九项指标胜 5 负 4；对应八项正确性视图胜 5 负 3。
> 这不是新推理声明。已发布 checkpoint 还有 95-case deployment 反证，v37 尚未通过 runtime
> acceptance，本项目不声称 v37 可用。
> [完整对比数据 →](#实测数据)

## 安装

> **安装提示**：项目尚未发布到 PyPI。请先克隆仓库，再从项目根目录安装：`git clone https://github.com/ChenneyZhuang/laya-browser-agent && cd laya-browser-agent`。

```bash
# Apple Silicon（MLX，最快路径）
pip install -e '.[mlx]'

# Linux / Windows / Intel Mac（PyTorch 运行时）
pip install -e '.[torch]'

# 浏览器驱动
pip install -e '.[playwright]' && playwright install chromium
```

自检（自动识别芯片与内存，并做一次真实决策的冒烟测试）：

```bash
localdecide doctor
```

默认 v32b 模型首次使用时下载一次（约 1.3 GB；`browser-legacy` 官方 v10s 版约 650 MB），之后全部离线。

配置参数已公开：`model`、`subfolder`、`revision` 会传给已核对的 upstream runtime；HTTP
backend 还接受 `model`、`api_key`、`timeout`。自定义同步 backend 只需实现
`answer(state, questions)` 并返回 `{"answers": {...}, "usage": {...}}`。`Decider` 的
总 timeout 会约束 HTTP 剩余 timeout；同步 duck 调用不能被 Python 强制中断，晚到结果会拒绝且不再重试。

### 后端配置、超时与安全边界

下面的例子使用实际的 v32b 模型名，也展示 API key、HTTP timeout 和自定义 backend：

```python
from localdecide import Decider
from localdecide.backends.base import HTTPBackend, LayaTorchBackend

local = LayaTorchBackend(model="ichenney/laya-browser-v32b", subfolder="v32b")

class MyBackend:
    name = "my-local-backend"
    def answer(self, state, questions):
        return {"answers": {}, "usage": {}}

remote = HTTPBackend(
    "http://127.0.0.1:8791/v1/systemone",
    model="browser", api_key="example-key", timeout=10.0,
)
decider = Decider(backend=remote, timeout=12.0, retries=1)
```

本地 MLX/PyTorch backend 不会把页面状态发出本机；HTTP backend 会把状态和问题
发送到你配置的 endpoint，只应连接可信的本地或私有服务。`api_key` 会作为 Bearer
header 发送，真实 key 请放在 `LOCALDECIDE_API_KEY` 或 secret manager 中，不要写进
源码或 shell history。密码值会在观察中脱敏，`TYPE_TEXT` 会拒绝密码字段（项目不提供
vault 访问）；发送、删除、支付等不可逆动作请提供人工 `confirm` 回调。同步自定义
backend 不能被 Python 强制中断，超时后到达的结果会拒绝且不会再次调用。

## 实测数据（M4, 16 GB）

> 所有数字都在本项目的开发机上实测：Apple M4（16 GB, `laya-mlx`）做推理延迟，
> RTX 3080 做微调与批量评测，页面均为真实 Chromium。没有抄任何厂商营销数字。
> 早于 v32b 的与 Jev 对比数据用的是官方 v10s checkpoint，已标注保留作历史记录。

| 指标 | 数值 |
|---|---|
| 短文本决策 | 10–30 ms |
| 浏览器单步（限定 20 个元素） | ~333 ms |
| 吞吐量 | 最高 ~100 决策/秒 |
| 多语言目标（grounding 开启） | 9 个命中 8 个 |
| 费用 | **$0**（无 API，无计量） |
| 页面内容发送到服务器 | **零** |

### 自训 v32b（现默认模型）vs 官方模型 vs Jev 云端 API（2026-09-25 实测，同套题）

| 基准 | **v32b（自训，默认）** | 官方浏览器微调版 (v10s/td) | Jev 云端 API |
|---|---:|---:|---:|
| recovery2-holdout（240 题） | **0.7125** | 0.425 | — |
| MiniWoB（116 题） | **0.9138** | 0.6638 | — |
| browser-suite v4 | **0.5143** | 0.500 | — |
| browser-suite v5 | 0.5636 | **0.5818** | — |
| JevBench hard（111 题） | **0.4144** | 0.243 | 0.7207 |
| JevBench easy | 0.8542 | 0.979 | 1.0000 |
| 决策延迟（p50） | **27 ms**（RTX 3080） | — | 854 ms（网络往返） |

**怎么读这张表**：已提交诊断支持的历史结果合计为九项指标胜 5 负 4；对应八项正确性视图胜
5 负 3。与 Jev 云端
API 相比，绝对精度仍有差距（Jev 是云端大模型），但 v32b **免费、隐私（页面内容不出本机）、
可离线、快 31 倍**，且在 `score` 评分题（0.667 vs 0.333）和 `temporal_numeric`（0.33 vs
0.20）两类题上反超 Jev。

v32b 的关键改进：上游训练管线从不生成 `noul`（命题判断）训练项——探针实测否定类判断
准确率仅 37%。自建 8k 条 noul 语料后提升到 100%，done_judgment 家族从 0.615 提到 0.769。
完整配方与逐项数据见 [MULTIDIM_COMPARISON.md](reports/v20/MULTIDIM_COMPARISON.md)
与 [JEV_COMPARISON.md](reports/v20/JEV_COMPARISON.md)；全部 15 个版本、所有失败路径与
训练脚本存档在配套仓库 [laya-training-log](https://github.com/ChenneyZhuang/laya-training-log)。

## 快速上手

```python
from localdecide import BrowserDecider
from localdecide.drivers import PlaywrightDriver

with PlaywrightDriver(start_url="https://en.wikipedia.org/wiki/Main_Page") as driver:
    run = BrowserDecider().run(driver, "Click the 'Random article' link in the navigation.")
    print(run.stopped, run.summary()["median_decision_ms"], "ms/决策")
```

### 与官方 Jev 的正面对决：实测数据

以下对战在 v32b 诞生前完成，本地侧用的是当时的官方 v10s checkpoint，作为历史记录保留，不能当作新模型推理。`examples/diagnostics/jev_head_to_head.py`（单步）与 `jev_flow_h2h.py`（多步完整流程）把同样的任务分别喂给本地 Laya v10s 与官方 jev-1.13.0：

| 单步零上下文（12 题 / 6 语言） | 本地 v10s | 官方 Jev |
|---|---|---|
| 严格元素命中 | 4/12 | **8/12** |
| 跨语言目标 | 1/6 | **5/6** |
| 中位延迟 | **618ms** | 716ms |
| 平均置信度 | 0.90（过自信）| 0.81 |

多步流程：**两个引擎目前都无法独立完成脚本化购物流程**——难点在"输入搜索词之后、结果出现之前"的状态；本地 v10s 在商品可见时仍会重复点击 Search（p=0.84），官方 Jev 会答 BLOCKED 或以 0.45 的低置信度选对元素。本地 v10s 完整走通了中文导航流程（帮助中心→DONE），官方 Jev 到达同元素但未发 DONE。

**结论**：要开箱准确率（尤其跨语言）→ 官方 Jev 每次约 $0.000017；要隐私/离线/免费大量调用 → 本地版延迟相当、置信度校准更需打磨，且需要 harness 循环配合才能进入其训练场景。

### 已对官方 Jev API 实测验证

本项目的 `systemone` 方言已于 2026-09-22 对 TypeSafe 生产端点（`api.typesafe.ai/v1/systemone`，模型 `jev-1.13.0`）完整实测。注意：**每种题型都必须带 `criteria` 字段**——`choice` 的 `criteria` 是「选项 → 评分说明」的映射（不是字符串），`score` 的是数组。同一份请求体把 URL 换成本地 `localdecide serve` 即可用本地 Laya 模型得到相同结构的回答，切换只需改一个 base URL。



**文本分类**（`jev_text_h2h.py`，真实业务文本，21 例）：

| 任务 | 本地 v10s | 官方 Jev |
|---|---|---|
| 泳池线索 `relevant`（10 例） | **6/10** | **10/10** |
| 泳池线索质量分（0-4）| 偏低（1.0-2.0）| **校准良好（2.4-3.7）** |
| 中文短信：交易/类型/钓鱼联合（8 例） | **5/8** | **5/8** |
| 鲁棒性（空文本/5k长文/干扰措辞）| 3/3 | 3/3 |
| 中位延迟 | **36ms** | 738ms |

钓鱼识别值得警惕：「妈妈，我手机坏了…快转5000」这类经典骗局本地只给 p=0.14，红包诈骗 p=0.23——两个都会放行；官方 Jev 都是 p=0.96。**任何涉安全路由（诈骗/滥用）场景，当前本地模型不能单独信任。**

**浏览器边缘场景**（`jev_edge_h2h.py`，9 例）：旧的 4/9 对 5/9 只统计 operation；现在 scorer 分开报告 operation、target、joint。gold 已修正为 fixture 中确实未勾选的 newsletter 控件；旧的 checked-fixture 行不能当作新模型结果。安全边界仍是 harness 的禁用检查与确认门。

四组测试的实用结论：准确率与安全校准优先→官方 Jev；延迟（快 10-20 倍）、隐私、免费大量调用→本地版 + harness 守卫补偿。中文 grounding/钓鱼识别/克制能力正是微调该打的方向。
## 与 Jev / Laya 的关系

| | [TypeSafe Jev](https://docs.typesafe.ai) | [Laya](https://github.com/NandhaKishorM/laya) | **laya-browser-agent** |
|---|---|---|---|
| 权重 | 闭源，仅 API | 开放，Apache-2.0 | 运行 Laya 开放权重 |
| 运行位置 | TypeSafe 云端 | 任何 PyTorch 环境 | **你的机器**（MLX / PyTorch）|
| 接口格式 | `POST /v1/systemone` | 同一契约 | 同样支持 |
| 费用 | $0.042/百万输入 token | 免费 | 免费 |
| 浏览器工具链 | [jev-ultrafast](https://github.com/browser-use/jev-ultrafast) | — | **内置**：循环、驱动、安全护栏、技能 |
| 页面内容离开本机 | 是 | 否 | **否** |

如果你看过 Jev 的 "System One" 模型报道，想要同样的理念——结构化、带校准概率的决策而非生成文本——在你自己的浏览器 agent 上本地运行，这就是你要的接线方式。

## 五种使用方式

1. **Python 库** —— `Decider` / `BrowserDecider`
2. **浏览器 agent 循环** —— Playwright / CDP 驱动，含循环守卫与确认门
3. **MCP server** —— `localdecide-mcp`，Claude Desktop / Cursor 一段配置接入
4. **HTTP 服务** —— `localdecide serve`，支持 TypeSafe 兼容的 `/v1/systemone` 方言
5. **Agent 技能** —— `python3 install_skills.py` 装进 Claude Code / Codex / Cursor / Hermes

## 关键设计

- **模型只能从你观察到的元素里选**——模型输出永远不会变成选择器、坐标或可执行代码
- **fail-open**：后端错误、超时、格式异常都返回 `Decision(ok=False)`，不阻塞你的流程
- **置信度门**（默认 0.15）：实测模型会以 6% 置信度去提交空表单，这一门拦住它
- **空提交守卫**：实测 v32b 会以 74% 置信度在字段为空时点提交——置信度门抓不住（它很"自信"），这一门检查提议的真实后果：提交类按钮 + 与之配对的空字段 = 拒绝，连续两次则终止
- **开关守卫**：实测模型会以 90% 置信度点掉已勾选的复选框，这一门同样拦住
- **跨语言 grounding**：中文/日文/阿拉伯文等目标自动过滤异文字干扰项（1/9 → 8/9）
- **范围优先于提示词**：选项数量是延迟与准确率的最大杠杆（10 个元素 183ms，120 个 1204ms）

### 文献支撑

- [arXiv 2609.23959](https://arxiv.org/abs/2609.23959)（2026-09）：相关 typed-decision 诈骗筛查任务报告 AUROC .974、校准误差 .052；这是背景证据，不是本项目短信结果的因果解释。
- Laya 上游公开的校准基准：13 个任务族 **accuracy 0.753 @ ECE 0.030**（温度校准后）——v10s 浏览器 checkpoint 在浏览器分布之外的文本上没有继承这个校准水平（见上文泳池/短信测试）。

## 已知局限

- 零样本在你的领域上很弱——浏览器检查点好用是因为有人花了 ~5 GPU 小时微调
- 模型不能写字——`TYPE_TEXT` 需要你提供文字内容
- 折叠菜单里的元素观察不到——先点开菜单再观察
- 无法读取 canvas / 影子 DOM 内容（需要在驱动层另行处理）
- 安全护栏基于关键词，不是保证——涉及金钱/删除/发送的操作请务必提供 `confirm` 回调

完整细节见[英文 README](README.md) 的 Benchmarks、Limitations 和 Reference 部分。

## 许可

Apache-2.0（与其运行的 Laya 模型一致）
