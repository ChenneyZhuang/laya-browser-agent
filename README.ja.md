# laya-browser-agent

**[English](README.md)** | [中文](README.zh-CN.md) | [日本語](README.ja.md) | [Español](README.es.md)

> 日本語の要約です。互換性・リリース検証・詳細な制限は [README.md](README.md)、[model-compatibility.md](docs/model-compatibility.md)、[release-verification.md](docs/release-verification.md) を参照してください。

Jev 系モデル向けのマルチバックエンド制約付き意思決定支援です。Laya の MLX/PyTorch、
同期 `answer(state, questions)` backend、System One 形式の HTTP endpoint に対応します。

決定モデルは与えられた選択肢への確率推定を返します。指示文を生成しないことは、
誤りや過信がないことを意味しません。検証、結果確認、timeout、人間の確認を残してください。

## モデルと設定

公式の [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) は
Laya のベースモデルで、ブラウザ fine-tune ではありません。上流のブラウザ参考版は
[`cklxx/laya-browser`](https://huggingface.co/cklxx/laya-browser) です。推奨するのは
本プロジェクトの [`ichenney/laya-browser-v32b`](https://huggingface.co/ichenney/laya-browser-v32b)：
`model="browser"`、`subfolder="v32b"` を使います。現在の model card は
`v32b-b15` を示し、確認済みの Hub metadata の不変 commit は
`161d54d6000913ff279b0afd1ac77faef8685a9b` です。再現可能な設定ではこの
`revision` を明示してください。

```python
LayaTorchBackend(model="ichenney/laya-browser-v32b", subfolder="v32b",
                 revision="161d54d6000913ff279b0afd1ac77faef8685a9b")
LayaTorchBackend(model="browser-legacy", subfolder="v10s",
                 revision="adf912be85ff9221ee171551778456b133c1af75")
```

PyTorch extra は、確認した loader で `revision` をサポートして転送する
`laya>=0.3.21`、MLX extra は `laya-mlx>=0.1.0` です。今回は公開ソース、wheel
metadata、v32b Hub metadata を確認しましたが、runtime の import、重みのダウンロード、
推論はしていません。
mock テストは引数転送だけを検証し、checkpoint のロードや品質を証明しません。

## インストールとプライバシー

現在の PyPI プロジェクト URL はまだ 404 です。git clone からインストールしてください。

```bash
git clone https://github.com/ChenneyZhuang/laya-browser-agent
cd laya-browser-agent
pip install -e '.[mlx]'       # Apple Silicon
pip install -e '.[torch]'     # Linux / Windows / Intel Mac
pip install -e '.[playwright]' && playwright install chromium
```

ローカル MLX/PyTorch backend はページ状態を端末内で処理します。`HTTPBackend` は
設定した endpoint に状態と質問を送るため、外部 endpoint への送信、可用性、価格、
プライバシーはそのサービスの条件です。「無料・プライバシー」は対応するローカル
backend にだけ適用されます。

## 証拠の範囲と制限

歴史的な診断と raw 条件は
[`reports/v20/MULTIDIM_COMPARISON.md`](reports/v20/MULTIDIM_COMPARISON.md) と
[`reports/v20/JEV_COMPARISON.md`](reports/v20/JEV_COMPARISON.md) にあります。README では
古いテスト件数、固定価格、raw 記録で裏付けられない性能レンジを繰り返しません。例の
出力は説明用です。

オフライン MiniWoB や fixture の accuracy は診断値であり、ブラウザタスクの成功率では
ありません。歴史的 edge が operation のみを測っている場合、target や joint を意味しません。
提出済み記録に存在するフィールドだけを報告し、新しいモデル結果は主張しません。

ブラウザ fine-tune は汎用 web agent ではありません。認証/パスワード、長い動的フォーム、
折りたたみメニュー、canvas/shadow DOM、headless を拒否するサイト、語彙や言語の不一致で
失敗する可能性があります。検証、timeout、確認ゲートを有効にしてください。

## 出典とライセンス

Laya ベースモデルとブラウザ checkpoint は各 upstream の Apache-2.0、帰属、出典チェーンを
保持します。Mind2Web、NNetNav、WebChain などのデータセットには各自の適用条件があり、
本プロジェクトはそれらを再許諾せず、追加の権利を示しません。

詳細は [英語 README](README.md) を参照してください。

## ライセンス

Apache-2.0（使用するモデル、コード、データの upstream 条件にも従います）。
