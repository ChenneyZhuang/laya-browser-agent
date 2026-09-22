# laya-browser-agent

**[English](README.md)** | [中文](README.zh-CN.md) | [日本語](README.ja.md)

> これは laya-browser-agent の日本語ドキュメントです。最新情報は [README.md](README.md) をご覧ください。

**Laya 駆動のブラウザエージェント決定エンジン — オープンソースの System 1 モデル。TypeSafe Jev のローカル代替：クラウド不要、API キー不要、スクリーンショット不要。**

[![tests](https://github.com/ChenneyZhuang/laya-browser-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/ChenneyZhuang/laya-browser-agent/actions/workflows/tests.yml)

決定モデルは状態について型付きの質問に答え、生成テキストではなく**較正された確率**を返します。テキストを生成しないため、指示を幻覚することはありません。ページ上の操作可能な要素の番号付きリストを渡すと、次に実行すべき操作と対象要素を教えてくれます。これがブラウザエージェントの「決定」部分に最適な形です。

## 測定値（M4, 16 GB）

| 指標 | 値 |
|---|---|
| 短い状態の決定 | 10–30 ms |
| ブラウザステップ（20 要素） | 約 333 ms |
| スループット | 最大約 100 決定/秒 |
| 多言語ゴール（grounding 有効） | 9 中 8 命中 |
| コスト | **$0** |
| ページ内容の外部送信 | **なし** |

## インストール

```bash
pip install 'laya-browser-agent[mlx]'      # Apple Silicon
pip install 'laya-browser-agent[torch]'    # Linux / Windows / Intel Mac
localdecide doctor                  # ハードウェア診断 + スモークテスト
```

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

## 主な特徴

- モデルは観測した要素からしか選べない（セレクタや座標にはならない）
- fail-open 設計：エラー時もエージェントのフローは止まらない
- 信頼度ゲート・トグルガード・ループガード内蔵
- スクリプト接地（grounding）：非ラテン文字のゴールに対し、異文字の選択肢を自動フィルタ（1/9 → 8/9）
- MCP サーバ同梱：Claude Desktop / Cursor に設定一行で接続

既知の制限やベンチマークの詳細は[英語版 README](README.md) を参照してください。

## ライセンス

Apache-2.0
