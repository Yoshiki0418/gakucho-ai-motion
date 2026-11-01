# 応答と行動の同期 — ベースライン収録

改修の前後を同じ条件で比較できるようにするための収録キット。
「応答内容とアバターの行動が一致していない」現象を、**動画**と**数値**の両方で残す。

## なぜ数値も残すのか

動画だけだと「なんとなくズレている」以上のことが言えず、改修後に良くなったかを判定できない。
このキットはブラウザのコンソールから次の 2 種類のイベントを拾い、遅れを ms で算出する。

| ログ | 出どころ | 意味 |
| --- | --- | --- |
| `[SYNC] audio_scheduled startAt=… duration=…` | `useTextChat.ts` の `enqueueAudioBuffer` | その発話が AudioContext のタイムライン上で鳴る**予定時刻** |
| `[SYNC] motion_applied ctxNow=… scheduledUpTo=…` | `useTextChat.ts` の `motion_chunk` ハンドラ | モーションが適用された**実時刻** |

音声は AudioContext 上に正確にスケジュールされているので、この 2 つを突き合わせれば
「どの発話の、どれだけ後にモーションが出たか」が確定する。

現構成でとくに見るべき指標は **「発話が終わってから発火した回数」**。
モーションの開始時刻が T2M の完了時刻で決まっているため、ここが 0 にならない。

## セットアップ

```bash
pip install playwright
python3 -m playwright install chromium
```

フロントエンドを起動しておく。`backend: true` のシナリオはバックエンド(と T2M)も必要。

```bash
cd frontend && npm run dev      # http://localhost:4000
```

## 収録

```bash
# 全シナリオ
python3 docs/sync-baseline/record.py --label before --avatar gakucho_v3_suit.glb

# 1つだけ
python3 docs/sync-baseline/record.py --label before --only 01-greeting

# ブラウザを見ながら
python3 docs/sync-baseline/record.py --label before --headed
```

`docs/sync-baseline/recordings/<日時>-<label>/<シナリオID>/` に
`screen.webm`（1600×900）、`console.jsonl`、`scenario.json` が残る。

## 解析

```bash
python3 docs/sync-baseline/analyze.py docs/sync-baseline/recordings/20260825-2130-before

# 改修後と比較
python3 docs/sync-baseline/analyze.py <before-run> --compare <after-run>
```

## レポート化

`report-template.md` をコピーして `recordings/<run>/report.md` として埋める。
`analyze.py` の出力をそのまま貼れば数値部分は埋まる。

## シナリオ

`scenarios.json` に定義。各シナリオに「何を見るか(`watch`)」と「なぜ見るか(`why`)」を書いてある。

| ID | 内容 | バックエンド |
| --- | --- | --- |
| 01-greeting | 短い挨拶 + 明確なジェスチャ | 要 |
| 02-long-explanation | 長い説明の途中に動作 | 要 |
| 03-multiple-actions | 複数動作の連続指示 | 要 |
| 04-no-gesture | 動作指示なしの通常応答 | 要 |
| 05-consecutive-turns | 連続ターンでのモーション残留 | 要 |
| 06-offline-motion | BVH 直接読み込み(描画品質の記録) | 不要 |

## 注意

- **動画は Git に入れない。** `recordings/` 配下の生成物は `.gitignore` 済み。
  レポート(`report.md`)と `console.jsonl` だけコミットすれば比較の再現には足りる。
- Chromium の自動再生ブロックは `--autoplay-policy=no-user-gesture-required` で回避しているが、
  音声が鳴らない環境では `[SYNC] audio_scheduled` が出ず遅れを算出できない。
  `--headed` で実際に音が出ているか一度確認すること。
- 06 以外はバックエンドの応答内容に依存するため、**同じ入力でも応答文が毎回変わる**。
  before/after で応答長が大きく変わると遅れの比較が歪むので、発話長も併記して判断する。
