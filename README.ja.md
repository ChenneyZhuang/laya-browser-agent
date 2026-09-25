# laya-browser-agent

**[English](README.md)** | [中文](README.zh-CN.md) | [日本語](README.ja.md)

> これは laya-browser-agent の日本語ドキュメントです。最新情報は [README.md](README.md) をご覧ください。

**Laya 駆動のブラウザエージェント決定エンジン — オープンソースの System 1 モデル。TypeSafe Jev のローカル代替：クラウド不要、API キー不要、スクリーンショット不要。**

[![tests](https://github.com/ChenneyZhuang/laya-browser-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/ChenneyZhuang/laya-browser-agent/actions/workflows/tests.yml/badge.svg)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-ichenney%2Flaya--browser--v32b-yellow)](https://huggingface.co/ichenney/laya-browser-v32b)

決定モデルは状態について型付きの質問に答え、生成テキストではなく**較正された確率**を返します。テキストを生成しないため、指示を幻覚することはありません。ページ上の操作可能な要素の番号付きリストを渡すと、次に実行すべき操作と対象要素を教えてくれます。これがブラウザエージェントの「決定」部分に最適な形です。

> **🚀 独自ファインチューン済みチェックポイント：8ベンチマーク中6つで公式モデルを上回る。**
> **[ichenney/laya-browser-v32b](https://huggingface.co/ichenney/laya-browser-v32b)** は
> holdout（**0.7125 vs 公式 0.425**）、MiniWoB（**0.9138 vs 0.6638**）、JevBench hard
> （**0.4144 vs 0.243**）などで公式を凌駕し、3080 GPU で **27 ms/決定**（Jev API の 31 倍速）。
> 切り替えは一行：`LayaTorchBackend(model="ichenney/laya-browser-v32b", subfolder="v32b")`

## インストール

> **インストール注記**：PyPI 版はリポジトリに追いつきつつあります。今は `git clone https://github.com/ChenneyZhuang/laya-browser-agent && cd laya-browser-agent && pip install -e '.[all]'`（Apple Silicon）または `.[torch]`（その他）を推奨。

```bash
pip install 'laya-browser-agent[mlx]'      # Apple Silicon
pip install 'laya-browser-agent[torch]'    # Linux / Windows / Intel Mac
localdecide doctor                  # ハードウェア診断 + スモークテスト
```

## 測定値（M4, 16 GB）

| 指標 | 値 |
|---|---|
| 短い状態の決定 | 10–30 ms |
| ブラウザステップ（20 要素） | 約 333 ms |
| スループット | 最大約 100 決定/秒 |
| 多言語ゴール（grounding 有効） | 9 中 8 命中 |
| コスト | **$0** |
| ページ内容の外部送信 | **なし** |

### 独自 v32b vs 公式モデル vs Jev API（2026-09-25 実測、同一セット）

| ベンチマーク | **v32b（独自）** | 公式ブラウザ版 | Jev API |
|---|---:|---:|---:|
| recovery2-holdout（240問） | **0.7125** | 0.425 | — |
| MiniWoB（116問） | **0.9138** | 0.6638 | — |
| browser-suite v4 | **0.5143** | 0.500 | — |
| browser-suite v5 | 0.5636 | **0.5818** | — |
| JevBench hard（111問） | **0.4144** | 0.243 | 0.7207 |
| JevBench easy | 0.8542 | 0.979 | 1.0000 |
| レイテンシ（p50） | **27 ms**（RTX 3080） | — | 854 ms（ネットワーク） |

ローカル（無料・オフライン）枠では 8 項目中 6 項目で公式を上回る。Jev クラウド API には絶対精度で及ばないが、**無料・プライバシー（ページ内容は端末外に出ない）・オフライン・31 倍速**で、`score` 問題（0.667 vs 0.333）と `temporal_numeric`（0.33 vs 0.20）では Jev を逆転。詳細は [英語版 Benchmarks](README.md#benchmarks) と [JEV_COMPARISON.md](reports/v20/JEV_COMPARISON.md)。

### 公式 Jev との対決：実測データ

`examples/diagnostics/jev_head_to_head.py`（単発）と `jev_flow_h2h.py`（マルチステップ）で、同じタスクをローカル Laya v10s と公式 jev-1.13.0 に実行して比較：

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
| プールリード判定 `relevant` | 3/7 正解 | **7/7** |
| リード質スコア（0-4）| 低め（1.0-2.0）| **良好（2.4-3.7）** |
| 中国語SMS：取引識別/種別 | **6/8** | 6/8 |
| 中国語SMS：**フィッシング検知** | **0/3** | **3/3** |
| 堅牢性（空/5k文字/敵対表現）| 3/3 | 3/3 |
| 中央値レイテンシ | **36ms** | 738ms |

フィッシング検知は要注意：「お母さん、携帯が壊れて…5000元送って」型の詐欺をローカルは p=0.14、赤い袋詐欺は p=0.23 と両方通過させてしまう。公式 Jev は両方 p=0.96。**詐欺・悪用など安全に関わるルーティングにローカル単体は現状不適。**

**ブラウザエッジケース**（`jev_edge_h2h.py`、9例、正解が「動かない」場合）：ローカル 4/9、公式 5/9、失敗の方向は正反対。ローカルは先に撃つ型：サイト全削除を頼まれると「Delete my account」を p=0.93 でクリック。公式は不可能目標をブロックするが正常な目標まで誤ブロック。両エンジンとも「この目標は達成不可能」の概念が未成熟で、今のセーフティはハーネスのガードが担う。

4バッテリーの実用結論：精度・安全キャリブレーション重視→公式 Jev。レイテンシ（10-20倍速い）・プライバシー・無料大量→ローカル＋ガード補償。中国語 grounding・フィッシング・抑制が微調整で狙うべき方向。
## 主な特徴

- モデルは観測した要素からしか選べない（セレクタや座標にはならない）
- fail-open 設計：エラー時もエージェントのフローは止まらない
- 信頼度ゲート・トグルガード・ループガード内蔵
- スクリプト接地（grounding）：非ラテン文字のゴールに対し、異文字の選択肢を自動フィルタ（1/9 → 8/9）
- MCP サーバ同梱：Claude Desktop / Cursor に設定一行で接続

既知の制限やベンチマークの詳細は[英語版 README](README.md) を参照してください。

### 文献・根拠

- [arXiv 2609.23959](https://arxiv.org/abs/2609.23959)（2026-09）：独立の検証——同型の「typed decision」読み出しを詐欺スクリーニングに適用し、データが揃えば AUROC .974・較正誤差 .052・64.5ms/判断（コンシューマ GPU）。本READMEのテキストバッテリーで見えたフィッシング弱点は訓練データの問題であり、アーキテクチャの限界ではない。
- Laya 上流の公開較正ベンチマーク：13タスク族で **accuracy 0.753 @ ECE 0.030**（温度スケーリング後）。v10s ブラウザcheckpointはブラウザ分布外のテキストでこの較正水準を継承していない（上記プール/SMSの結果を参照）。

## ライセンス

Apache-2.0
