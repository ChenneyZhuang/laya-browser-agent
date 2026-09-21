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
pip install 'localdecide[mlx]'      # Apple Silicon
pip install 'localdecide[torch]'    # Linux / Windows / Intel Mac
localdecide doctor                  # ハードウェア診断 + スモークテスト
```

## 主な特徴

- モデルは観測した要素からしか選べない（セレクタや座標にはならない）
- fail-open 設計：エラー時もエージェントのフローは止まらない
- 信頼度ゲート・トグルガード・ループガード内蔵
- スクリプト接地（grounding）：非ラテン文字のゴールに対し、異文字の選択肢を自動フィルタ（1/9 → 8/9）
- MCP サーバ同梱：Claude Desktop / Cursor に設定一行で接続

既知の制限やベンチマークの詳細は[英語版 README](README.md) を参照してください。

## ライセンス

Apache-2.0
