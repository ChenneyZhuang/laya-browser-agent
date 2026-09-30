# laya-browser-agent

**[English](README.md)** | [中文](README.zh-CN.md) | [日本語](README.ja.md)

> これは laya-browser-agent の日本語ドキュメントです。最新情報は [README.md](README.md) をご覧ください。

**Jev 系モデル向けのマルチバックエンド制約付き意思決定支援。** Laya の MLX/PyTorch、
`answer(state, questions)` を持つ同期 duck backend、任意の System One 形式 HTTP endpoint
を利用できます。ブラウザの既定値は自作 checkpoint の推奨であり、専有モデルの強制ではありません。

[![tests](https://github.com/ChenneyZhuang/laya-browser-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/ChenneyZhuang/laya-browser-agent/actions/workflows/tests.yml/badge.svg)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-ichenney%2Flaya--browser--v32b-yellow)](https://huggingface.co/ichenney/laya-browser-v32b)

決定モデルは状態について型付きの質問に答え、生成テキストではなく**較正された確率**を返します。テキストを生成しないため、指示を幻覚することはありません。ページ上の操作可能な要素の番号付きリストを渡すと、次に実行すべき操作と対象要素を教えてくれます。これがブラウザエージェントの「決定」部分に最適な形です。

`model="browser"` は **[ichenney/laya-browser-v32b](https://huggingface.co/ichenney/laya-browser-v32b)**
の `v32b` サブフォルダを推奨します。`model="browser-legacy"` は upstream
`cklxx/laya-browser` の `v10s` パスです。upstream の `main` からこの旧ディレクトリが削除されたため、
legacy 別名はデフォルトで revision `adf912be85ff9221ee171551778456b133c1af75` に固定されます（`revision=` で上書き可能）。
コミット済みの歴史的診断は 9 指標で 5 勝 4 敗、
8 指標の正確性ビューでは 5 勝 3 敗です。新しい推論結果ではありません。公開 checkpoint には
95-case deployment の反証があり、v37 は runtime acceptance 未通過で、利用可能とは主張しません。

## インストール

> **インストール注記**：PyPI パッケージはまだ公開されていません。まずリポジトリをクローンし、プロジェクトルートからインストールしてください：`git clone https://github.com/ChenneyZhuang/laya-browser-agent && cd laya-browser-agent`。

```bash
pip install -e '.[mlx]'      # Apple Silicon
pip install -e '.[torch]'    # Linux / Windows / Intel Mac
localdecide doctor           # ハードウェア診断 + スモークテスト
```

デフォルトの v32b チェックポイントは初回使用時に一度だけダウンロード（約 1.3 GB、
公式 v10s に戻す `browser-legacy` は約 650 MB）。以後は完全オフラインで動作。

`model`、`subfolder`、`revision` は検証済み upstream runtime へ渡されます。HTTP backend は
`model`、`api_key`、`timeout` を受け付け、カスタム backend は同期 `answer(state, questions)`
を実装します。`Decider` の総 timeout は残りの HTTP timeout を制限します。同期 duck 呼び出しは
Python から強制キャンセルできないため、遅れて返った結果は拒否し、後続呼び出しは行いません。

### backend 設定、timeout、安全境界

実際の v32b モデル名、API key、HTTP timeout、カスタム backend の例です。

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

ローカルの MLX/PyTorch backend はページ状態を端末外へ送りません。HTTP backend は
設定した endpoint に状態と質問を送るため、信頼できるローカルまたはプライベートな
サービスだけを使ってください。`api_key` は Bearer header になるので、実際の key は
ソースや shell history ではなく `LOCALDECIDE_API_KEY` または secret manager に置きます。
パスワード値は観測から削除され、vault アクセスを提供しないため `TYPE_TEXT` は password
欄を拒否します。送信・削除・支払いなど不可逆な操作には人間の `confirm` callback を
指定してください。同期カスタム backend は Python から強制中断できず、timeout 後の
結果は拒否され、再呼び出しも行われません。

## 測定値（M4, 16 GB）

> すべての数値は開発機で実測：Apple M4（16 GB, `laya-mlx`）で推論レイテンシ、
> RTX 3080 でファインチューニングとバッチ評価。ベンダーのマーケティング数値の流用はなし。
> v32b より前の Jev 比較は公式 v10s チェックポイント使用（歴史的記録として保持）。

| 指標 | 値 |
|---|---|
| 短い状態の決定 | 10–30 ms |
| ブラウザステップ（20 要素） | 約 333 ms |
| スループット | 最大約 100 決定/秒 |
| 多言語ゴール（grounding 有効） | 9 中 8 命中 |
| コスト | **$0** |
| ページ内容の外部送信 | **なし** |

### 独自 v32b（現デフォルト）vs 公式モデル vs Jev API（2026-09-25 実測、同一セット）

| ベンチマーク | **v32b（独自）** | 公式ブラウザ版 | Jev API |
|---|---:|---:|---:|
| recovery2-holdout（240問） | **0.7125** | 0.425 | — |
| MiniWoB（116問） | **0.9138** | 0.6638 | — |
| browser-suite v4 | **0.5143** | 0.500 | — |
| browser-suite v5 | 0.5636 | **0.5818** | — |
| JevBench hard（111問） | **0.4144** | 0.243 | 0.7207 |
| JevBench easy | 0.8542 | 0.979 | 1.0000 |
| レイテンシ（p50） | **27 ms**（RTX 3080） | — | 854 ms（ネットワーク） |

この表に対応するコミット済み診断の歴史的集計は 9 指標で 5 勝 4 敗、8 指標の正確性ビューで 5 勝 3 敗です。Jev クラウド API には絶対精度で及ばないが、**無料・プライバシー・オフライン**という別の価値があります。詳細は [英語版 Benchmarks](README.md#benchmarks) と [JEV_COMPARISON.md](reports/v20/JEV_COMPARISON.md)。

全 15 バージョン・すべての失敗パス・トレーニングスクリプトは姉妹リポジトリ [laya-training-log](https://github.com/ChenneyZhuang/laya-training-log) に公開。

### 公式 Jev との対決：実測データ

以下の対戦は v32b 登場前の実施で、ローカル側は当時の公式 v10s チェックポイントです。歴史的記録であり、新モデル推論ではありません。`examples/diagnostics/jev_head_to_head.py`（単発）と `jev_flow_h2h.py`（マルチステップ）で比較します：

| 単発・ゼロ文脈（12 問 / 6 言語） | ローカル v10s | 公式 Jev |
|---|---|---|
| 厳密な要素命中 | 4/12 | **8/12** |
| クロスリンガル目標 | 1/6 | **5/6** |
| 中央値レイテンシ | **618ms** | 716ms |
| 平均信頼度 | 0.90（過信）| 0.81 |

マルチステップ：**両エンジンともスクリプト化された買い物フローを自力で完遂できず**。検索語入力直後（結果が出る前）の状態が難所。ローカル v10s は商品が見えていても Search を再クリック（p=0.84）、公式 Jev は BLOCKED または正要素を p=0.45 で選択。ローカル v10s は中国語ナビゲーションフローを完遂（ヘルプセンター→DONE）。

**結論**：精度重視（特に多言語）→ 公式 Jev（1 回約 $0.000017）。プライバシー・オフライン・大量呼び出し → ローカル版。レイテンシは同等だが、多言語 grounding が最大の弱点。

### 公式 Jev API での実検証

本プロジェクトの `systemone` 方言は、2026-09-22 に TypeSafe の本番エンドポイント（`api.typesafe.ai/v1/systemone`、モデル `jev-1.13.0`）で実検証済み。**すべての質問タイプに `criteria` が必須**——`choice` は「選択肢 → 説明」のマップ、`score` は配列。同じリクエストをローカルの `localdecide serve` に向ければローカル Laya モデルが同じ形式で回答し、切り替えは base URL を変えるだけ。



**テキスト分類**（`jev_text_h2h.py`、実ビジネステキスト21例）：

| タスク | ローカル v10s | 公式 Jev |
|---|---|---|
| プールリード `relevant`（10例） | **6/10** | **10/10** |
| リード質スコア（0-4）| 低め（1.0-2.0）| **良好（2.4-3.7）** |
| 中国語SMS：取引/種別/フィッシング joint（8例） | **5/8** | **5/8** |
| 堅牢性（空/5k文字/敵対表現）| 3/3 | 3/3 |
| 中央値レイテンシ | **36ms** | 738ms |

フィッシング検知は要注意：「お母さん、携帯が壊れて…5000元送って」型の詐欺をローカルは p=0.14、赤い袋詐欺は p=0.23 と両方通過させてしまう。公式 Jev は両方 p=0.96。**詐欺・悪用など安全に関わるルーティングにローカル単体は現状不適。**

**ブラウザエッジケース**（`jev_edge_h2h.py`、9例）：旧 4/9 対 5/9 は operation のみの集計です。現在の scorer は operation・target・joint を分けて出力し、gold は実際に未チェックの newsletter control に修正しました。旧 checked-fixture 行を新モデル結果として扱いません。

4バッテリーの実用結論：精度・安全キャリブレーション重視→公式 Jev。レイテンシ（10-20倍速い）・プライバシー・無料大量→ローカル＋ガード補償。中国語 grounding・フィッシング・抑制が微調整で狙うべき方向。
## 主な特徴

- モデルは観測した要素からしか選べない（セレクタや座標にはならない）
- fail-open 設計：エラー時もエージェントのフローは止まらない
- 信頼度ゲート・空送信ガード・トグルガード・ループガード内蔵
- スクリプト接地（grounding）：非ラテン文字のゴールに対し、異文字の選択肢を自動フィルタ（1/9 → 8/9）
- MCP サーバ同梱：Claude Desktop / Cursor に設定一行で接続

既知の制限やベンチマークの詳細は[英語版 README](README.md) を参照してください。

### 文献・根拠

- [arXiv 2609.23959](https://arxiv.org/abs/2609.23959)（2026-09）：関連する typed-decision 詐欺スクリーニング課題は AUROC .974、較正誤差 .052 を報告しています。これは背景情報であり、本プロジェクトの SMS 結果の因果説明ではありません。
- Laya 上流の公開較正ベンチマーク：13タスク族で **accuracy 0.753 @ ECE 0.030**（温度スケーリング後）。v10s ブラウザcheckpointはブラウザ分布外のテキストでこの較正水準を継承していない（上記プール/SMSの結果を参照）。

## ライセンス

Apache-2.0
