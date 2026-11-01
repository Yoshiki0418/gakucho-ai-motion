# 同期評価レポート — <ラベル: before / after>

- 収録日: YYYY-MM-DD
- 収録コマンド: `python3 docs/sync-baseline/record.py --label <label> --avatar gakucho_v3_suit.glb`
- 収録データ: `docs/sync-baseline/recordings/<run>/`
- フロントエンド: <commit>
- バックエンド: <commit> / T2M: <model>
- アバター: <glb>

## 総括

| 指標 | 値 |
| --- | --- |
| モーション発火回数 | |
| 発話開始からの遅れ(中央値) | ms |
| 発話開始からの遅れ(最大) | ms |
| **発話が終わってから発火した回数** | |
| コンソールエラー | |

> `python3 docs/sync-baseline/analyze.py <run>` の出力をそのまま貼る。
> before / after を並べる場合は `--compare <after-run>` を付ける。

## シナリオ別

### 01-greeting — 短い挨拶 + 明確なジェスチャ

- 動画: `recordings/<run>/01-greeting/screen.webm`
- 入力: こんにちは！
- 応答テキスト:
- 計測:
  | 項目 | 値 |
  | --- | --- |
  | 発話セグメント数 | |
  | 発話長 | s |
  | モーション | (prompt) |
  | 発話開始からの遅れ | ms |
  | 発話終了に対する位置 | ms |
  | T2M 生成時間 | ms |
- 所見:
  - 時間的一致:
  - 意味的一致:

### 02-long-explanation — 長い説明の途中に動作

（同じ形式で 06 まで繰り返す）

## 定性的な所見

-
-

## 次のアクション

-
