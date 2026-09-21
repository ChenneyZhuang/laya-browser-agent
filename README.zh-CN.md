# laya-browser-agent

**[English](README.md)** | [中文](README.zh-CN.md) | [日本語](README.ja.md) | [Español](README.es.md)

> 本页面是 laya-browser-agent 的中文说明。英文原版（最新）见 [README.md](README.md)。

**浏览器 agent 决策，由 Laya 驱动 —— 开源的 System 1 模型。TypeSafe Jev 的本地开源替代：无云端、无 API key、无截图。**

[![tests](https://github.com/ChenneyZhuang/laya-browser-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/ChenneyZhuang/laya-browser-agent/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

决策模型对状态回答结构化问题，返回**校准过的概率**而非生成的文字——因此它不可能幻觉出一条指令。这正是浏览器 agent "决定"那一半的理想形态：给它一张页面上可交互元素的编号表，它告诉你下一步该执行什么操作、作用于哪个元素。

本项目把这类模型接进这个角色——**完全在本地运行**，适配你已有的任何 agent。

## 实测数据（M4, 16 GB）

| 指标 | 数值 |
|---|---|
| 短文本决策 | 10–30 ms |
| 浏览器单步（限定 20 个元素） | ~333 ms |
| 吞吐量 | 最高 ~100 决策/秒 |
| 多语言目标（grounding 开启） | 9 个命中 8 个 |
| 费用 | **$0**（无 API，无计量）|
| 页面内容发送到服务器 | **零** |

## 安装

```bash
# Apple Silicon（MLX，最快路径）
pip install 'localdecide[mlx]'

# Linux / Windows / Intel Mac（同样权重，PyTorch 运行时）
pip install 'localdecide[torch]'

# 浏览器驱动
pip install 'localdecide[playwright]' && playwright install chromium
```

自检（自动识别芯片与内存，并做一次真实决策的冒烟测试）：

```bash
localdecide doctor
```

模型首次使用时下载一次（约 650 MB），之后全部离线。

## 快速上手

```python
from localdecide import BrowserDecider
from localdecide.drivers import PlaywrightDriver

with PlaywrightDriver(start_url="https://en.wikipedia.org/wiki/Main_Page") as driver:
    run = BrowserDecider().run(driver, "Click the 'Random article' link in the navigation.")
    print(run.stopped, run.summary()["median_decision_ms"], "ms/决策")
```

## 与 Jev / Laya 的关系

| | [TypeSafe Jev](https://docs.typesafe.ai) | [Laya](https://github.com/NandhaKishorM/laya) | **laya-browser-agent** |
|---|---|---|---|
| 权重 | 闭源，仅 API | 开放，Apache-2.0 | 运行 Laya 开放权重 |
| 运行位置 | TypeSafe 云端 | 任何 PyTorch 环境 | **你的机器**（MLX / PyTorch）|
| 接口格式 | `POST /v1/systemone` | 同一契约 | 同样支持 |
| 费用 | $0.042/百万输入 token | 免费 | 免费 |
| 浏览器工具链 | [jev-ultrafast](https://github.com/browser-use/jev-ultrafast)（12.6k★）| — | **内置**：循环、驱动、安全护栏、技能 |
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
- **开关守卫**：实测模型会以 90% 置信度点掉已勾选的复选框，这一门同样拦住
- **跨语言 grounding**：中文/日文/阿拉伯文等目标自动过滤异文字干扰项（1/9 → 8/9）
- **范围优先于提示词**：选项数量是延迟与准确率的最大杠杆（10 个元素 183ms，120 个 1204ms）

## 已知局限

- 零样本在你的领域上很弱——浏览器检查点好用是因为有人花了 ~5 GPU 小时微调
- 模型不能写字——`TYPE_TEXT` 需要你提供文字内容
- 折叠菜单里的元素观察不到——先点开菜单再观察
- 无法读取 canvas / 影子 DOM 内容（需要在驱动层另行处理）
- 安全护栏基于关键词，不是保证——涉及金钱/删除/发送的操作请务必提供 `confirm` 回调

完整细节见[英文 README](README.md) 的 Benchmarks、Limitations 和 Reference 部分。

## 许可

Apache-2.0（与其运行的 Laya 模型一致）
