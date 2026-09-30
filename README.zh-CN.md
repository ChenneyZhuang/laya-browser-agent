# laya-browser-agent

**[English](README.md)** | [中文](README.zh-CN.md) | [日本語](README.ja.md) | [Español](README.es.md)

> 本页是中文摘要；模型兼容性、发布核验和完整限制见 [README.md](README.md)、[model-compatibility.md](docs/model-compatibility.md) 与 [release-verification.md](docs/release-verification.md)。

面向 Jev 类模型的多后端、受约束决策支持框架。支持 Laya MLX/PyTorch、同步
`answer(state, questions)` backend，以及 System One 形状的 HTTP endpoint。

决策模型返回对所给选项的概率估计，不生成指令；这不等于不会出错，也不等于
概率一定校准。验证、结果检查、超时和人工确认仍是安全边界。

## 模型与配置

官方 [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) 是
Laya 基础模型，不是浏览器微调版。上游浏览器参考微调是
[`cklxx/laya-browser`](https://huggingface.co/cklxx/laya-browser)。本项目明确推荐
自己的浏览器 checkpoint
[`ichenney/laya-browser-v32b`](https://huggingface.co/ichenney/laya-browser-v32b)：
使用 `model="browser"`、`subfolder="v32b"`。当前模型卡标识为 `v32b-b15`，
已核验的 Hub metadata 不可变 commit 为
`161d54d6000913ff279b0afd1ac77faef8685a9b`。需要可复现时请在配置中显式传入该
`revision`。不要把基础模型或历史 v10s 结果当成 v32b 的当前结果。

历史别名仍可显式使用：

```python
LayaTorchBackend(model="ichenney/laya-browser-v32b", subfolder="v32b",
                 revision="161d54d6000913ff279b0afd1ac77faef8685a9b")
LayaTorchBackend(model="browser-legacy", subfolder="v10s",
                 revision="adf912be85ff9221ee171551778456b133c1af75")
```

PyTorch extra 使用 `laya>=0.3.21`，因为该版本才在已检查的 loader 中支持并转发
`revision`；MLX extra 仍为 `laya-mlx>=0.1.0`。本轮只核对了发布源码、wheel metadata
和 v32b Hub metadata，没有导入 runtime、下载权重或运行模型。mock 测试只证明参数转发，不证明
checkpoint 能加载或效果良好。

## 安装与隐私

当前 PyPI 项目 URL 仍返回 404；请从 git clone 安装：

```bash
git clone https://github.com/ChenneyZhuang/laya-browser-agent
cd laya-browser-agent
pip install -e '.[mlx]'       # Apple Silicon
pip install -e '.[torch]'     # Linux / Windows / Intel Mac
pip install -e '.[playwright]' && playwright install chromium
```

本地 MLX/PyTorch backend 只在本机处理页面状态。`HTTPBackend` 会把状态和问题发到
配置的 endpoint；endpoint 可能在外部，也可能有自己的隐私、可用性和计费条款。
因此“免费/隐私”只适用于相应的本地 backend，不能推广到 HTTP。

## 证据边界与已知限制

提交的历史诊断、原始条件和方法见
[`reports/v20/MULTIDIM_COMPARISON.md`](reports/v20/MULTIDIM_COMPARISON.md) 与
[`reports/v20/JEV_COMPARISON.md`](reports/v20/JEV_COMPARISON.md)。README 不重复旧的
测试计数、静态价格或未经原始记录支持的性能范围；示例输出仅供说明。

离线 MiniWoB 或 fixture accuracy 只是诊断信号，不是端到端浏览器任务成功率。
历史 edge 记录若只统计 operation，就不能推导 target 或 joint；只有提交记录中实际
存在的字段才可报告。没有新的模型结果在此声明。

浏览器微调版不是通用 web agent：认证/密码字段、长或动态表单、折叠菜单、canvas/
shadow DOM、阻止 headless 的网站，以及语言或词汇不匹配都可能失败。保留验证、
timeout 和人工确认门。

## 设计、来源与许可

- 模型只能从观察到的选项中选择，代码负责执行、检查和不可逆操作确认。
- `TYPE_TEXT` 需要调用方提供文字；模型不负责写入字符串。
- Laya 基础模型和浏览器 checkpoint 保留各自上游的 Apache-2.0 许可、署名和来源链。
- Mind2Web、NNetNav、WebChain 等训练/评测数据仍受各自适用条款约束，本项目不重新
  许可这些数据，也不据此暗示额外权利。

完整测试、兼容性和发布边界见 [英文 README](README.md)。

## 许可

Apache-2.0（同时遵守所使用模型、代码和数据的上游条款）。
