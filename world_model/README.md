# 物理・学習環境

本番の T2M 再生とは独立した、MuJoCo の実物理環境です。内部座標は Z-up、前方は −Y、単位は m・kg・s です。表示側の Y-up への変換は `(x,y,z) → (x,z,-y)`。自由な胴体の位置はリセット時にだけ設定し、実行中は重力・接触・関節トルクにより変化します。

## 起動と検証

既存 TTS/T2M 環境への追加インストールは行いません。今回使用した独立環境は `.runtime/world-model/venv` です。Python 3.11 の読み取り専用親環境から Torch を利用し、MuJoCo 等はこの venv に導入しました。

```bash
.runtime/world-model/venv/bin/python -m pytest -q world_model
.runtime/world-model/venv/bin/python -m world_model.training --task stand --steps 200000 --envs 8
```

長い学習はチェックポイントから再開できます。

```bash
.runtime/world-model/venv/bin/python -m world_model.training --task stand --steps 200000 --envs 8 --resume .runtime/world-model/stage1/policy.zip
```

外乱も加える場合は `--push-max 100` を指定します。学習中の力は方向・大きさ・時刻を変えて実際の身体へ加えます。位置・速度の補正ではありません。

物理環境と確認画面への記録出力は以下です。

```bash
.runtime/world-model/venv/bin/python -m world_model.publish --standing .runtime/world-model/stage1-balance --sitting .runtime/world-model/stage1-seating --walking .runtime/world-model/stage1-two-meter-gait --physical-footsteps --physical-approach --wbc-trace .runtime/world-model/verification/wbc-walk-flat-1m-rollout.json
```

出力先は `frontend/public/world-model` です。本番画面はフロントエンドのビルドに含めて配布します。`/world-model` で4段階の空間と実際に記録した姿勢を確認できます。

## 学長アバターに合わせた身体

`gakucho-avatar-profile.json` は実際の `gakucho_v3_suit.glb` を、既存の骨軸・Tポーズ・身長・靴底校正で表示して採寸したものです。住宅動作用の1.8 m正規化規約を使い、実際の頭頂までの表示高さは約1.825 m、大腿約0.453 m・すね約0.413 mです。足は靴のスキン頂点から、座面との接触は90度の股・膝の姿勢でのお尻の下端から校正しています。初期画面のBVH高さ約1.788 mと住宅動作の1.8 mの違いもプロファイルへ明記しています。

```bash
node tools/measure-gakucho-avatar.mjs
.runtime/world-model/venv/bin/python -m pytest -q world_model/test_avatar_measurement.py world_model/test_avatar_profile.py
```

物理環境へは `avatar_profile`、学習・記録出力のCLIへは `--avatar-profile frontend/public/world-model/gakucho-avatar-profile.json` を指定します。未指定なら従来の汎用人体です。骨格・靴底・頭部・骨盤の接触形状を実寸へ合わせ、体節の質量合計64.7 kg、関節の強さ・可動範囲は既存の仮定を保っています。衝突形状はスキン全頂点と同一ではなく、物理用の簡略形状です。

旧人体のPPOは新人体の検証へそのまま使えません。`body_contract.py` がプロファイルと物理身体のSHAを照合します。身体を変えて追加学習を始める場合にだけ `--allow-body-transfer` を付け、実際のPPO学習後に新しい身体契約を保存します。新身体の最終歩行評価は、従来の汎用人体と別の32条件（seed 2,000,600〜2,000,631）で行います。

```bash
.runtime/world-model/venv/bin/python -m world_model.training --task stand --steps 200000 --envs 8 --resume .runtime/world-model/stage1-balance/policy.zip --avatar-profile frontend/public/world-model/gakucho-avatar-profile.json --allow-body-transfer --output .runtime/world-model/stage1-gakucho-standing
```

新しい記録には `avatarProfile`、`physicsRestBodies`、各フレームの実body位置・回転を保存します。確認画面では同じ学長GLBを読み込み、物理結果を骨へ対応付けて表示します。「学長」「重ねて表示」「物理形状」の表示切替は物理状態・判定・再生速度を変えません。旧人体の記録は汎用人型のまま表示します。

学長の身体では、立位・着座に各212,992ステップ、歩行に2回で合計1,261,568ステップを実際に追加学習しました。新しい最終条件では、100 Nの外乱に対する立位8/8、実座面が体重を支える着座8/8、2 m先への到達・停止32/32、独立した両脚反復歩行監査31/32でした。最終歩行では物理制御への切替は0件です。未通過1件（seed 2,000,615）は到達・停止できましたが右足の反復離床回数が不足しています。

実GLBを使う全2,403フレームの表示照合では主要関節の最大位置差は約6.4 µmでした。足の形状は箱へ近似しているため、支持中の靴底と床の距離は立位・基準歩行では通常約2 mmですが、着座移行中には最大約47 mmあります。停止した着座では服のスキンが座面へ約8 mm入る差が残ります。骨の転写誤差と、簡略接触形状・姿勢によるスキン変形の差を区別し、表示の足や腰を後から床へ移動して隠していません。

学長の身体で再検証した記録の公開は次のコマンドです。

```bash
.runtime/world-model/venv/bin/python -m world_model.publish --standing .runtime/world-model/stage1-gakucho-standing --sitting .runtime/world-model/stage1-gakucho-seating --walking .runtime/world-model/stage1-gakucho-walking-refined --physical-footsteps --physical-approach --avatar-profile frontend/public/world-model/gakucho-avatar-profile.json --wbc-trace .runtime/world-model/verification/gakucho-wbc-walk-flat-1m-rollout.json
```

## 転倒例の確認

学長の体格へ適応する途中の歩行チェックポイント `stage1-gakucho-walking` には、seed 10001で8.60秒、10002で2.32秒の実際の転倒があります。現在の採用モデル `stage1-gakucho-walking-refined` と区別し、確認画面の「比較する記録」へ追加しています。「転倒直前へ」で1秒前へ移動し、「記録を再生」で等速再生できます。`/world-model?stage=1&trace=%2Fworld-model%2Fstage1-walking-fall-10002.json` から直接選択できます。

元の評価は最初の転倒判定で停止します。閲覧用の別記録では、同じPPOと制限付きPD関節トルク制御を2秒間、実MuJoCo上で続け、胴体・頭が床へ接触するところまで記録しています。後半は評価外です。位置・速度・姿勢はリセットせず、追加の外力も使いません。元の評価の転倒時刻・状態・累積報酬と一致することを照合し、骨格・身体・チェックポイント・ソースのSHAを保存します。

```bash
.runtime/world-model/venv/bin/python -m world_model.failure_examples --checkpoint-directory .runtime/world-model/stage1-gakucho-walking --historical-trace .runtime/world-model/verification/gakucho-walk-initial-validation.json --avatar-profile frontend/public/world-model/gakucho-avatar-profile.json --output .runtime/world-model/failure-examples
.runtime/world-model/venv/bin/python -m world_model.failure_gallery --examples .runtime/world-model/failure-examples/failure-examples.json
```

ギャラリー登録は現在のモデル、評価値、合格基準、初期表示記録を変えません。`publish` で評価データを再生成した場合は、続けてギャラリー登録を再実行します。

## 4段階

1. 身体と物理：重力、接触、摩擦、関節の制限、立位、歩行、着座。まず物理そのものと立位を検証し、歩行・着座を個別に評価します。
2. 家具配置：形状・寸法・向きを変えた家具と通路。固定経路に依存しない制御を、未学習の配置で評価します。
3. 複数部屋：部屋、扉、廊下をつないだ環境で、移動から着座までを連続評価します。
4. 高低差：小さな段差から階段、2階へ進めます。平面用の移動処理を流用して階段を登ったことにはしません。

## フェーズ2の学習と監査

`navigation.py` は方向転換と家具1個の回避を個別に学習する環境です。エピソードごとに床・目的地・家具の寸法と角度・左右通路・摩擦を生成し、実MuJoCoモデルを作り直します。家具課題では、出発地点と最終目的地を結ぶ直線が実家具を横切ることを確認します。経路や中継地点を方策へ渡しません。家具を身体の安全余白で拡張した可視性グラフは報酬の距離計算にだけ使います。

家具課題の難易度0では、出発から目的地まで3.10〜3.40 mの助走と、目的地への直線を端で横切る家具配置から始めます。難易度を上げると距離を短くし、家具を中央へ戻して幅・奥行・角度を広げます。最も難しい難易度1の配置と方向転換課題は変更していません。家具を端へ寄せすぎると16方向の距離センサー間に入り得るため、目的地への直線に垂直な方向で、家具の投影端から10 cmの余裕を確保します。12,000配置で直線の実遮断・初期前方rayの交差・両側通路・開始と目的地の余裕を監査し、実MuJoCoと実センサーの独立100条件でも初期家具検出100/100を確認しています。これは学習を易しくする静的配置の生成であり、実行中の身体や家具の位置補正ではありません。

`navigation_training.py` は同じ身体の59入力の歩行PPOから110入力へ明示的に移行します。既存59列と他の重みをコピーし、追加51列をゼロで初期化して、ランダムなセンサー入力32組でも元の出力を維持することを監査します。110入力の再開は `--resume110`、59入力からの初期化は `--initialize-from59` を使います。身体・プロファイル・関節20出力・観測の意味が違う方策は拒否します。

```bash
.runtime/world-model/venv/bin/python -m world_model.navigation_training --mode turn --steps 102400 --checkpoint-every 20480 --seconds 30 --difficulty-min 0 --difficulty-max .75 --seed 8471 --envs 8 --learning-rate .00002 --entropy .0002 --reset-log-std -2.6 --initialize-from59 .runtime/world-model/stage1-gakucho-walking-refined/policy.zip --avatar-profile frontend/public/world-model/gakucho-avatar-profile.json --output .runtime/world-model/stage2-gakucho-turn-ik-corrected
.runtime/world-model/venv/bin/python -m world_model.navigation_evaluation --mode turn --policy .runtime/world-model/stage2-gakucho-turn-ik-corrected/policy.zip --episodes 8 --scene-seed-base 1150100 --physics-seed-base 3150100 --difficulty-min .6 --difficulty-max 1 --seconds 30 --avatar-profile frontend/public/world-model/gakucho-avatar-profile.json --output .runtime/world-model/stage2-turn-ik-development
```

出力はPD関節目標への20関節の制限付き補正です。`steering.py` は旋回時に股関節yawを変更した後、足首位置と靴底の法線を保つ脚の逆運動学を解き直します。IKの結果はトルク目標にだけ使い、胴体・足・速度を直接移動しません。方向転換は元の歩行報酬を保持し、家具回避だけに迂回距離・接近リスクの報酬を加えます。

家具・壁の実接触を2 msごとの物理サブステップで監査し、一度の接触も履歴から消しません。到達は目的地から25 cm以内、速度25 cm/s未満、姿勢・高さ・目的階の実床を満たし、左右それぞれ20 N以上の床支持が0.5秒続くことが必要です。独立した歩容監査も通過したケースだけを成功とします。家具・壁へ1 Nを超える接触があった場合と実際の転倒は課題を終了し、理由を別々に記録します。転倒前に接触で終了した記録を「転倒回避を習得」とは扱いません。

初回の旋回学習102,400ステップでは開発8/8から最終26/32へ低下し、家具回避の追加204,800ステップは開発0/8でした。これらは未採用の試行です。旋回の足姿勢と報酬を修正し、元の採用歩行PPOから再学習しています。開発条件はモデル選択に使い、合格判定は別の未使用32条件以上で同時成功率90%以上を必要とします。家具1個への合格だけで複数家具・複数部屋・階段や任意の部屋へ対応済みとは表示しません。

初回の家具課題8件はすべて、1.12〜1.48秒で足が家具へ接触して終了しました。迂回に必要な横移動約0.6〜0.7 mに対し実際は約0.05〜0.12 mで、追加51センサー列の重みRMSも元の59列の約1%でした。助走を広げたカリキュラムは、この短い探索時間と急な回避要求への対処です。修正した方向転換制御の追加102,400ステップでは、別の同一開発8条件で、未学習の入力拡張だけの4/8に対し7/8へ改善しています。開発用の結果を最終合格には使いません。

続く102,400ステップで開発8/8となった `stage2-gakucho-turn-ik-refined` を固定し、未使用のscene seed 1,851,000〜1,851,031、physics seed 3,851,000〜3,851,031、難易度0.6〜1で一度評価しました。方向転換は生存・無接触・反復歩行・到達後停止の同時通過29/32（90.625%）で、90%以上の基準を通過しています。残る3条件は転倒です。修正後の方向転換学習は合計204,800ステップ、採用モデルの累積ステップは6,381,568です。家具がない有限な平床で最大約20.6度の目的地方向を変える課題であり、急な旋回・家具回避の習得を含みません。

```bash
.runtime/world-model/venv/bin/python -m world_model.navigation_publish --turning .runtime/world-model/stage2-turn-ik-final-evaluation --turning-model .runtime/world-model/stage2-gakucho-turn-ik-refined --obstacle .runtime/world-model/stage2-obstacle-204800-development --obstacle-model .runtime/world-model/stage2-gakucho-obstacle-initial
```

公開コマンドはフェーズ2の記録・合否・学習来歴を登録し、フェーズ1の採用モデル・合格基準・転倒比較記録を保持します。確認画面の緑のリングが実際の最終目的地です。記録ごとの合否と接触を表示します。フェーズ2の方策も、会話中の住宅動作へはまだ接続していません。

## 方向・歩調と関節制御を分ける検証

家具回避を20関節へ直接学習する再試行は、助走と観測を改善しても追加204,800ステップ後の開発8条件で0/8でした。この候補も未採用です。`command_navigation.py` では、固定した110入力・20関節出力の歩行PPOを土台に、別の上位PPOが身体を基準とする方向と先読み距離の2指示を学習します。上位入力は112個で、元の実目的地・実センサー110個と、直前の有効指示2個です。目的地への経路や中継点は入力しません。関節制御は20 ms、上位判断は100 ms、接触監査は2 msです。

短い移動指示を最終目的地として既存の参照へ渡すだけでは足上げまで縮んだため、`command_gait.py` で歩幅と足上げ時間を分離します。T2Mの周期、実寸の脚IK、上限付き関節トルクは保持します。既存歩行PPOの重みを変更せず、制御用の目標を脚の参照とモーターの目標3特徴にだけ明示的に渡します。実タスクの最終目的地と実身体の位置・速度は変更しません。開発用の閉ループ基準制御では先読み0.35 mと0.5 mの2条件で到達・停止・独立歩容監査を通過しました。これは上位方策の学習結果とは別です。

最初の上位PPO16,384判断・実81,746関節制御ステップでは、未学習の開発8条件で到達・停止3/8、反復歩行0/8でした。選んだ先読みが平均0.26 mで対応範囲より短かったため、移動指示を0.35〜0.5 mと明示的な停止へ分けます。到達報酬は実際の持続した支持付き離床を両足各2回以上行ってから与え、最終判定は独立した歩容監査も必要とします。上位判断の割引時間が長くなる分だけ終端報酬を調整し、全報酬へ同じ0.01倍を掛けて価値推定の桁を揃えます。実物理の報酬合計と学習へ渡す倍率付き報酬は別々に記録します。

長時間の途中停止はこの新しい構成の未解決課題です。既存立位モデルへの切替も開発2条件で生存1/2に留まり、採用していません。現在の上位学習は、1目的地へ歩き、実際の両足支持で0.5秒停止する課題として検証します。30秒の停止保持、複数部屋、階段、会話中のライブ制御への接続を習得済みとは扱いません。失敗例・モデルのSHA・実制御回数・20 msごとの全フレームを保存します。

## 上位方策の教師による事前学習

`command_navigation_pretraining.py` は、方向指示の探索を始めるために、実ゴールへの方位を使う解析的な教師で上位 actor だけを事前学習します。教師は現在の身体の向きと最終目的地から方向指示を選び、家具の経路・中継地点・人間の収録動作は渡しません。身体を動かすのは固定した歩行PPOと制限付き関節トルクで、実タスクのゴール・身体の位置・速度・外力を変更しません。

train scene seed 88401〜88416とphysics seed 89401〜89416で、摩擦・体節をランダム化した16条件を実行し、到達・0.5秒の両足支持停止・独立した反復歩行監査まで合格した12条件だけを使用しました。失敗4条件も実20 ms記録へ保存し、教師データから除外しています。入力112・方向／歩調2出力の教師ペアは1,071個です。開発・最終評価の部屋は教師データへ含めません。

実PPO学習32,768判断・163,376関節制御ステップの `stage2-gakucho-command-turn-normalized` から複製し、actorへ150 epoch・750 minibatchのMSE学習を行いました。criticと探索分散log_stdの重みは完全に保持し、PPO optimizerだけを再初期化します。教師とのMSEは0.7198から0.00417へ低下しました。教師の収集とsupervised epochを強化学習の判断回数へ足していません。`stage2-gakucho-command-turn-bc` の `provenance.json`、`simulation-sources`、教師データのSHAにモデルの来歴を保存します。

scene seed 1,160,100〜1,160,107、physics seed 3,160,100〜3,160,107の同じ開発8条件では、厳格な同時合格は事前学習前3/8、事前学習後のニューラル方策4/8、別に実行した解析的教師4/8でした。事前学習後の推論には教師を使いません。解析的教師の開発記録も学習データへ転用しません。この8条件は開発比較に使ったため、最終合格の判定には使えません。教師に近づけるだけで身体の旋回・離床・停止が安定するわけではなく、家具回避や任意の部屋への習得を示す結果でもありません。

旋回失敗の記録では、ゴールへ向ける指令が出ていても逆側へ曲がる場合がありました。指令の変化率を0.175から0.35 rad/sへ増やしただけのtrain2条件は0/2のままでした。歩行モーターのhip-yaw残差が小さい方向参照より大きくなるため、左右の2出力だけを±0.15 action（約±3.01度）に制限する別プロセスの比較を行いました。他18関節・モーター重み・身体状態は保持します。trainの元成功／元壁接触の2条件は2/2、同じ開発8条件では解析的教師4/8→6/8、BCニューラル方策4/8→5/8へ改善しました。後者の生存は7/8→8/8でした。この出力制限は制御アダプターの実験で、既存チェックポイントをそのままの制御として公開した結果ではありません。未使用の最終32条件は評価しておらず、合格はまだ主張できません。

教師収集・actor fitでも `--motor-hip-yaw-action-limit .15` を指定できます。制御契約が一致する明示transfer済みの上位チェックポイントと、その制限・ソースで新たに収集した教師データが必要です。古い教師データの契約や旧チェックポイントの契約を自動的に書き換えて使う処理はありません。BCの来歴の共通モーター項目を補完した際も重みのSHAと元の生成ソースは保持し、修正理由とamendment source SHAを別に保存しています。

`command_motor_limits.py` と正式な `CommandNavigationEnv` にこの制限を実装しています。上限を指定した場合だけ制御契約がversion 2になり、旧version 1の方策の暗黙転用を拒否します。明示的な移行では身体・観測・モーター・他の制御設定を照合し、actor・critic・log_stdの全テンソルと32入力での実出力を保持して、optimizerを空の状態で作り直します。移行による強化学習・物理ステップの追加は0です。正式実装は同じ開発8条件で5/8を再現し、別プロセスの比較と保存された全形状の位置・時刻が完全一致しました。これは開発結果であり、最終合格ではありません。

正式な制限付き契約version 2で別のtrain16条件（scene 88501〜88516、physics 89501〜89516）を収集し、厳格合格12条件・941教師ペアだけを再fitしました。教師シミュレーションは全7,773関節制御ステップ・77,730物理サブステップで、強化学習回数とは別です。`stage2-gakucho-command-turn-yaw-bounded-bc` は実RL32,768判断・163,376関節制御ステップを保持し、criticとlog_stdを変えずにactorだけを150 epoch学習しています。同じ開発8条件でのニューラル方策単独の結果は厳格合格5/8、生存8/8、無接触5/8でした。3失敗は壁接触です。追加教師によるMSE低下だけでは成功率は改善しておらず、この候補も未採用です。19物理ソースと教師生成ソースの実バイトを、学習来歴のSHAへ一致させて保存しています。

```bash
.runtime/world-model/venv/bin/python -m world_model.command_motor_limits --source-policy .runtime/world-model/stage2-gakucho-command-turn-bc/policy.zip --motor-policy .runtime/world-model/stage2-gakucho-turn-ik-refined/policy.zip --motor-hip-yaw-action-limit .15 --avatar-profile frontend/public/world-model/gakucho-avatar-profile.json --output .runtime/world-model/stage2-gakucho-command-turn-bc-yaw-bounded
.runtime/world-model/venv/bin/python -m world_model.command_navigation_evaluation --policy .runtime/world-model/stage2-gakucho-command-turn-bc-yaw-bounded/policy.zip --motor-policy .runtime/world-model/stage2-gakucho-turn-ik-refined/policy.zip --motor-hip-yaw-action-limit .15 --mode turn --scene-seed-base 1160100 --physics-seed-base 3160100 --episodes 8 --seconds 30 --difficulty-min 0 --difficulty-max .5 --avatar-profile frontend/public/world-model/gakucho-avatar-profile.json --output .runtime/world-model/stage2-command-turn-bc-yaw-bounded-development
```

制限前の上位モデルへ方向転換を追加65,536判断学習した候補は、同じ開発条件で3/8から2/8へ悪化したため採用していません。家具回避を追加32,768判断学習した候補も、scene 1,100,300〜1,100,307／physics 3,100,300〜3,100,307の同じ家具配置で学習前後とも0/8でした。モデル選択に使う比較は、同一の部屋・家具・物理seed・身体・モーターのSHAを確認します。これらの失敗を成功した強化学習として扱いません。

```bash
.runtime/world-model/venv/bin/python -m world_model.command_navigation_pretraining collect --motor-policy .runtime/world-model/stage2-gakucho-turn-ik-refined/policy.zip --avatar-profile frontend/public/world-model/gakucho-avatar-profile.json --episodes 16 --scene-seed-base 88401 --physics-seed-base 89401 --difficulty-max .5 --output .runtime/world-model/new-command-teacher-data
.runtime/world-model/venv/bin/python -m world_model.command_navigation_pretraining fit --policy .runtime/world-model/stage2-gakucho-command-turn-normalized/policy.zip --motor-policy .runtime/world-model/stage2-gakucho-turn-ik-refined/policy.zip --avatar-profile frontend/public/world-model/gakucho-avatar-profile.json --dataset .runtime/world-model/new-command-teacher-data --epochs 150 --output .runtime/world-model/new-command-turn-bc
```

物理環境の生成・検証と、各課題を解く方策の学習済み状態は別々に記録します。後段の環境が表示できても、前段の課題に合格していなければその課題の習得済み表示にはしません。

## 短い指令への低位モーター追加学習

`command_motor.py` は、既存の20関節モーター自体を短い指令と `command_gait_pose` へ適応させる隔離環境です。入力は110個ですが、50〜52番だけが次の関節参照と同じ短い指令点へ変わるため、旧モーターの最終目的地入力と別の意味を持ちます。`command_motor_training.py` はこの違いを明示的な契約で保存し、同じ入力数でも旧方策の暗黙転用を拒否します。真の最終目的地は報酬と評価へ保持し、実行中の胴体・足・速度・外力を書き換えません。解析的な方位ドライバーが指令を出す検証であり、学習した部屋ナビゲーターではありません。

元モーター `stage2-gakucho-turn-ik-refined` のactorとlog_stdを完全に複製し、32個の実train初期状態で実出力差0を確認します。報酬の0.01倍に合わせてcritic最終線形層だけを0.01倍にし、Adamを初期化します。次の観測と参照点の整合、既存物理環境との200 msの身体状態・トルク・接触の完全一致、全2 msのゴール保持・外力ゼロを42テストと実モーターsmokeで検証しました。初期化では実PPOステップを追加しません。

seed 90401・8環境・学習率0.00005で実102,400関節制御ステップ（1,024,000物理サブステップ）を追加学習し、累積6,483,968の `stage2-gakucho-command-motor-short-goal-adapted` を作成しました。元モーターは変更せず、新モーターも高位制御へ自動採用しません。21ソース・アセット・元チェックポイントのSHAと実制御回数を監査し、実バイトを保存します。

専用の開発scene 1,170,100〜1,170,107・physics 3,170,100〜3,170,107では、同じ部屋・身体・指令・20 ms制御と2 ms監査で、適応前と後の厳格合格はともに3/8でした。生存は6/8→8/8ですが、適応後は残る5条件で壁へ接触しています。方向追従が未達であり、この候補は未採用です。適応前の記録は親モデルの既学習6,381,568ステップと新しい短指令適応0を分け、適応後は追加102,400を明記します。未使用の最終32条件は消費していません。

```bash
.runtime/world-model/venv/bin/python -m world_model.command_motor_training --source-policy .runtime/world-model/stage2-gakucho-turn-ik-refined/policy.zip --steps 102400 --checkpoint-every 25600 --envs 8 --seed 90401 --learning-rate .00005 --reset-log-std -2.4 --avatar-profile frontend/public/world-model/gakucho-avatar-profile.json --output .runtime/world-model/new-command-motor-adaptation
.runtime/world-model/venv/bin/python -m world_model.command_motor_training --evaluate-only --resume .runtime/world-model/new-command-motor-adaptation/policy.zip --evaluation-scene-seed-base 1170100 --evaluation-physics-seed-base 3170100 --evaluation-episodes 8 --avatar-profile frontend/public/world-model/gakucho-avatar-profile.json --output .runtime/world-model/new-command-motor-development
```

## 短指令の追従報酬v2

上記102,400ステップと3/8の結果は、最終目的地の進捗・速度追従を継承した報酬v1の記録です。入力で見える短指令と報酬の目標を合わせるため、`command_motor.py` の報酬v2は消費前の短指令を固定し、実変位の指令方向への進捗、指令速度への実速度追従、実際の身体方位を評価します。速度目標は `0.35 * clip(指令距離/0.5, 0, 1) * 開始ramp` m/sです。姿勢・実支持・実離床イベント・荷重下の滑り・全10サブステップの実トルクと行動コストを保持し、真の最終目的地は停止・両足支持・反復歩行を伴う成功判定と実接触・転倒の終端に使います。

継続報酬の下限はrawで−8、0.01倍後の割引継続価値の下限は−8です。成功1500と失敗−1500は継続報酬のclampの外側で加減し、終端のscaled値はそれぞれ+15と−15になります。報酬v1モデルを `--resume` で暗黙に転用せず、`--initialize-from-command` による明示的な契約移行を必要とします。32個の実train初期状態でactor・log_stdの同値を確認し、既に0.01単位のcriticを再度縮小せず、最終value headだけをゼロにします。hidden critic、実累積制御回数を保持し、Adamは初期化します。旧ZIPと保存済みv1ソースは変更しません。

v2の60テストでは、同じ短指令・同じ身体状態で最終ゴールだけを変更した2環境を100 ms実積分し、物理状態とdense報酬の完全一致、継承した目的地報酬の相違を確認しました。独立レビューでも実転倒の19制御・190サブステップで終端コストの再計算が完全一致しました。実v1モデルから転送した200 ms smokeでは32実初期状態のactor差0、重み不変、全2 msの目的地保持・外力ゼロ、snapshotの純粋性を確認しています。

v2追加学習0の転送基準は、同じ8開発条件で元のv1モデルの厳格合格3/8、生存8/8、無接触3/8を再現しました。全8条件の物理時間・制御回数・native合否は一致し、保存した2記録の989フレームは実body・geomが完全一致します。`stage2-command-motor-reward-v2-baseline-development/reward-v1-physical-reproduction.json` に比較結果を保存しています。

seed 90501・8環境・学習率0.00005で、v2の実102,400制御（1,024,000物理サブステップ）を追加し、累積6,586,368の `stage2-gakucho-command-motor-reward-v2-adapted` を作成しました。実actor重みの変更、PPOの125更新epoch、Adamの13状態・各2000更新step、source21・全アセット・元v1 ZIPの不変を `actual-rl-weight-audit.json` で確認しました。同じ8開発条件の厳格合格は3/8→2/8、無接触3/8→2/8、生存8/8→8/8でした。残り6条件は壁接触であり、改善を示していません。この候補は未採用です。公平な比較は `development-comparison.json` に保存し、未使用の最終コホートは使っていません。

```bash
# 報酬v2の保存済みソースでの再現手順。現行v3はv1からの暗黙転送を拒否します。
.runtime/world-model/venv/bin/python -m world_model.command_motor_training --initialize-from-command .runtime/world-model/stage2-gakucho-command-motor-short-goal-adapted/policy.zip --initialize-only --keep-log-std --envs 1 --seed 90501 --avatar-profile frontend/public/world-model/gakucho-avatar-profile.json --output .runtime/world-model/new-command-reward-v2-baseline
.runtime/world-model/venv/bin/python -m world_model.command_motor_training --evaluate-only --resume .runtime/world-model/new-command-reward-v2-baseline/policy.zip --evaluation-scene-seed-base 1170100 --evaluation-physics-seed-base 3170100 --evaluation-episodes 8 --avatar-profile frontend/public/world-model/gakucho-avatar-profile.json --output .runtime/world-model/new-command-reward-v2-baseline-development
.runtime/world-model/venv/bin/python -m world_model.command_motor_training --initialize-from-command .runtime/world-model/stage2-gakucho-command-motor-short-goal-adapted/policy.zip --steps 102400 --checkpoint-every 25600 --envs 8 --seed 90501 --seconds 30 --difficulty-min 0 --difficulty-max .5 --lookahead-min .5 --lookahead-max .5 --learning-rate .00005 --entropy .0002 --reset-log-std -2.4 --avatar-profile frontend/public/world-model/gakucho-avatar-profile.json --output .runtime/world-model/new-command-reward-v2-adaptation
```

これは解析的な指令ドライバーの下で学習する関節モーターです。高位の部屋ナビゲーターへ接続した結果や、任意の部屋への合格を意味しません。v2追加学習0の転送基準とv2追加学習後を別に記録します。

家具の形状から経路を作る教師を用いた実物理の切り分けでも、同じ2個のtrain部屋・身体・物理seedで、v2基準と追加学習後はともに0/2でした。経路は関節モーターの入力へ渡さず、教師が短い指令だけを生成します。この結果を学習済みナビゲーターの評価や、以前の別モーターの0/4と混ぜません。`.runtime/world-model/stage2/command-motor-reward-v2-adapted-geometry-probe/comparison.json` に同条件の比較を保存しています。

この2条件の記録では、旋回の変化制限0.175 rad/sによって、左−0.45 radから指令の符号が変わるまで約2.57秒、右＋0.45 radへの完全切り替えまで約5.14秒かかります。教師が右の曲がり角を要求しても、その間に消費する実指令は左のままです。さらにseed 88602の追加学習後では、平均＋0.419 radの右指令を8.34秒消費しても、実方位の純変化は＋0.0087 radでした。左指令での純旋回速度−0.183 rad/sに対し、右指令では＋0.0010 rad/sであり、指令遅延と身体の左右応答を分けて検証する必要があります。左右の足は9回・11回反復していて、単に歩行が止まっている状態ではありません。これは2個の診断部屋の実測で、全家具配置への性質を証明する値ではありません。

指令距離を0.5 mから0.35 mへ減らした減速診断も、基準・追加学習後とも0/2でした。教師の家具余裕を0.7 m、先読みを1.2 mに増やした基準モデルの診断も0/2で、1件は家具、1件は壁への接触です。先読みだけで運動制約を解消したとは扱いません。全試行は身体位置への直接書き込みや外力を使わず、50 Hzの実姿勢と2 msごとの接触を記録しています。詳細は `.runtime/world-model/verification/command-motor-reward-v2-review/turn-response-analysis.json` に保存しています。

同じtrain初期状態から左右±0.45 radへ指示する4試行も実行しました。各試行は6秒・300制御・3,000物理サブステップで、初期の姿勢・速度・観測・質量・慣性・摩擦は完全一致します。v2追加学習後の固定右指令では、飽和した指令区間の実平均yaw速度が−0.074 rad/sと逆方向になり、最低uprightは0.4536でした。転倒閾値0.45を下回らなくても姿勢の合格とは扱いません。記録した全サブステップの関節トルクは、参照とactor残差から計算したPDトルクと一致していて、指令のslew遅延以外にも低位モデルの方向応答が不安定です。`.runtime/world-model/stage2/command-motor-reward-v2-signed-ramp-response/response-audit.json` と `window-analysis.json` に実測値を保存しています。

旋回slew 0.175／0.5 rad/sと股関節yaw残差の無制限／±0.15を組み合わせる、追加学習0の比較診断を実施しました。元v2モデルをstrictに検証してから、明示した2項目だけが異なるprivateモデルへactor・critic・log_stdを完全複製し、32個の実train入力で出力差0を確認しています。他18関節、身体、参照、トルク上限は保持しています。0.175・制限なし、0.5・制限なし、0.175・制限ありは各0/2で、0.5・制限ありだけが1/2でした。成功したseed 88602は14.98秒で目的地から0.222 m、実歩行・到達停止・支持・無接触をすべて満たしています。同じ組合せのseed 88601は家具への接触で失敗しました。これは幾何教師による経路補助の診断で、上位ナビゲーターを学習した結果ではありません。

その組合せを別のtrain部屋16個（scene 88781〜88796／physics 89781〜89796）へ適用すると0/16でした。家具への接触14件、壁への接触1件、転倒1件で、計4,084実制御・40,840物理サブステップを記録しています。成功した試行だけを将来の教師データ候補へ入れるため、候補NPZは0行です。追加のfit／PPO学習は0で、失敗記録を除去・合格へ変更していません。`.runtime/world-model/stage2/command-motor-private-slew-yaw-2x2-diagnostic/feasibility.json` と `.runtime/world-model/stage2/command-motor-private-combined-teacher-train16/collection-audit.json` を証拠とし、1条件での成功を汎用性の証明には使いません。

## 実旋回速度の追従報酬v3

現行の`command_motor.py`では、消費した短指令の相対角から実旋回速度の目標を作ります。指令のローカル座標（観測50〜52）と実upベクトル（41〜43）を水平へ投影すれば、世界座標の相対角と一致します。実1,200入力での差は最大`6.14e-8 rad`でした。教師がまだ消費していない要求角や、最終目的地はこの目標に使いません。

目標yaw速度は `clip(指令距離/0.15,0,1) * clip(0.3*相対角,-0.135,0.135)` rad/s、実yaw速度は制御前後の実方位差をwrapして0.02秒で割った値です。旧heading項を `4*(exp(-((実yaw速度-目標yaw速度)/0.5)^2)-1)` へ置き換えます。幅0.5 rad/sは、正常な足位相にもある瞬間yaw振動を直ちに潰さないためのsoft penaltyです。進捗項も `3*clip(実変位の指令方向成分/0.02,-1,目標速度)` とし、目標速度より速く進むことで進捗報酬をさらに稼ぐことを防ぎます。実際の速度へ上限値を書き込む制御ではありません。

姿勢・高さ・支持・離床・滑り・実トルクの項、終端の成功／失敗1500、報酬scale0.01、継続下限−8、gamma0.99は維持します。v2baselineからの明示転送でactor・log_std・hidden criticを保持し、実32入力の出力差0、最終value headのみゼロ、fresh Adam、追加制御0を確認します。v2を現行v3の`--resume`で暗黙に継続しません。既存の身体・参照・トルク上限と、旋回slew0.175 rad/s、追加股関節clipなしの条件は保持しています。

専用76テスト、200 ms／100物理サブステップの実preflight、公開診断の検証を含む全413テストが通過しています。seed90601・8環境で204,800追加制御／2,048,000物理サブステップのPPOを実行しました。同じ8開発条件の成功は3/8から5/8、衝突なしは3/8から6/8へ増えましたが、生存は8/8から7/8へ減りました。学習済みcheckpointは`a92f47f3614b4cacfcc6472f0fee712a5270800b6be04c2e9925f08f48305fe9`です。

実初期状態が一致する左右固定指令の6秒診断では、左指令の飽和区間における平均yaw速度が+0.088から+0.147 rad/sへ変化し、要求−0.135 rad/sとは逆方向でした。右指令は+0.143から+0.238 rad/sへ変化し、最小upは0.799まで低下しました。幾何教師で家具を回り込む同じ2条件は学習前後とも0/2です。通常の目的地課題の改善だけでは指令追従を認定できないため、この候補を高位ナビゲータや学長AIの対話へ採用していません。評価とnative traceは`.runtime/world-model/stage2-command-motor-reward-v3-adapted-development`、左右診断は`.runtime/world-model/stage2/command-motor-reward-v3-signed-ramp-response`、家具比較は`command-motor-reward-v3-geometry-paired-comparison.json`に保存しています。

```bash
.runtime/world-model/venv/bin/python -m world_model.command_motor_training --initialize-from-command .runtime/world-model/stage2-gakucho-command-motor-reward-v2-transfer-baseline/policy.zip --initialize-only --keep-log-std --envs 1 --seed 90601 --avatar-profile frontend/public/world-model/gakucho-avatar-profile.json --output .runtime/world-model/new-command-reward-v3-baseline
.runtime/world-model/venv/bin/python -m world_model.command_motor_training --initialize-from-command .runtime/world-model/stage2-gakucho-command-motor-reward-v2-transfer-baseline/policy.zip --steps 204800 --checkpoint-every 25600 --envs 8 --seed 90601 --avatar-profile frontend/public/world-model/gakucho-avatar-profile.json --output .runtime/world-model/new-command-reward-v3-adaptation
```

## 左右指令の学習分布を揃える検証

`command_motor_curriculum.py`の`CommandMotorCurriculumEnv`は、学習用の空の部屋で、身体の物理resetが完了した後に実際の身体の乱数生成器から指令を抽選します。20エピソードごとのbagに、相対角−0.45／+0.45 radを各5件、−0.30／−0.15／0／+0.15／+0.30 radを各2件入れてshuffleし、各エピソードの6〜8秒間、要求角を保持します。部分bagの左右比は保証しません。要求角が最大値になるエピソードは全bagの50%ですが、0.175 rad/sのslewにより最大角へ到達するまで2.57秒かかるため、実際に最大角を消費する制御の比率は別に記録します。

進行方向は現在の実方位とslew後の指令角から更新し、最終目的地の方位を追う既存の教師とは区別します。歩行lookaheadは既存範囲0.35〜0.50 mからreset時に抽選し、目的地までの実距離で上限を設けます。目的地・身体・重力・接触・トルク制御・110観測／20出力・報酬v3には変更を加えません。密な報酬は消費済み指令と実測した動作だけを使い、まだslewに反映されていない要求角は使いません。この指令生成器は学習専用で、部屋を自律移動する推論ナビゲータではありません。

`--resume --keep-log-std --local-command-curriculum`では同一報酬v3を厳格に再開し、actor・critic・log_std・Adamの全状態と制御回数・更新数を保持します。実32学習初期観測で動作出力とvalueの差0を検証し、この監査自体の追加学習制御は0です。モデルと22ソースのSHA、身体・モーション資産、エピソードごとの要求角の件数、実際に消費した左右・ゼロ・最大角の制御回数をprovenanceへ記録します。heldoutへこの学習指令生成器を流用することと、暗黙にlog_stdをリセットすることはCLIで拒否します。

```bash
.runtime/world-model/venv/bin/python -m world_model.command_motor_training --resume .runtime/world-model/stage2-gakucho-command-motor-reward-v3-adapted/policy.zip --keep-log-std --local-command-curriculum --seconds 8 --lookahead-min .35 --lookahead-max .5 --steps 512000 --checkpoint-every 32768 --envs 8 --seed 90701 --avatar-profile frontend/public/world-model/gakucho-avatar-profile.json --output .runtime/world-model/new-command-motor-local-curriculum
```

この検証は実512,000制御／5,120,000物理サブステップの追加PPOまで完了しました。seed90701・8環境・1,305学習エピソード、125 rollout、追加625更新epoch、累積7,200,768制御／875 epochです。13個すべてのAdam状態が4,000から14,000 minibatch更新へ進み、actor・critic・log_stdの全13テンソルが更新されています。最終checkpointは`5aa9536b35defac5d715f98c75f558f6f5cdaf9a26344672d02e7e4d4358252b`です。実際に消費した最大角の指令は左86,907／右84,090制御、合計33.40%でした。要求角が最大のエピソードは652/1,305で49.96%となり、エピソード比と実制御比を分けて保存しています。

同じ初期身体・部屋の6秒左右診断では、最大角を消費した区間の平均yaw速度が、左+0.1465から−0.0825 rad/s、右+0.2383から+0.1403 rad/sへ変化しました。目標は左−0.135／右+0.135 rad/sです。実2 msでの最低upは左0.9536／右0.9811、離床回数は左指令[4,5]／右指令[3,4]で、転倒・接触はありませんでした。左の平均逆旋回は解消していますが、6秒の診断と最大角区間の平均値だけで応答全体や部屋の汎用性は認定しません。実曲線と1倍速のnative記録は`.runtime/world-model/stage2/command-motor-local-curriculum-signed-ramp-response`にあります。

同じ8開発条件での到達成功は5/8のまま、生存は7/8から8/8、無接触は6/8から5/8でした。成功／失敗の入れ替わりがあり、到達性能の改善は確認できません。通常lookahead0.50 mの家具教師2条件も0/2のままです。歩行指示を0.35 mへ弱めた同じ教師条件では、旧a92が0/2、新5aaが1/2で、scene88602は20.78秒・目的地距離0.1724 mで歩容・無接触・停止・両足支持の実物理gateを通過しました。margin0.70 m／carrot1.20 mで早めに曲がる別条件は0/2でした。この成功は学習専用の幾何教師が経路を供給する診断であり、経路判断を学習した結果ではありません。

同じ低速教師で既使用の学習用16配置も比較しました。旧a92の2/16から新5aaの4/16へ変化し、新5aaは88785／88789／88792／88795で全gateを通過しました。88785／88789／88795が新しく成功し、旧a92で成功していた88781は接触へ変わりました。両モデルとも転倒0ですが、新5aaの残る12件は家具5件・南壁6件・西壁1件への接触です。beforeは実12,769制御／127,690サブステップ、afterは12,563制御／125,630サブステップでした。全16のscene・physics seed・教師route・身体・重力・制御契約・保存された初期physical frameは厳密一致しました。未保存の初期qpos／qvel配列まで比較したという主張はしません。新5aaの20 ms記録での最低upは0.94807で、16件の全2 ms姿勢の下限値は記録していません。source22、資産、全actor／critic／std／Adam、モデルの学習回数は推論前後で不変、追加RL・BCは0です。

家具16件の成功率25%は高位ナビゲータの習得や未使用部屋での汎用性を証明しません。フェーズ2の合格、高位モデルへの採用、学長AIへの接続は未実施です。before／afterの全成功・失敗native traceと物理gateは、それぞれ`.runtime/world-model/stage2/command-motor-local-curriculum-baseline-geometry-train16-lookahead035`と`command-motor-local-curriculum-geometry-train16-lookahead035`に保存しています。独立監査で全32記録の歩容・接触・到達・停止判定を再計算して一致し、全16組の保存初期physical frame・scene・route・重力・身体・制御契約も厳密一致しました。未使用部屋での評価や追加学習は、この比較では行っていません。

指示offsetのslew遅延も、モデルの学習差とは別のruntime診断で検証しました。source5aaの全actor・critic・stdを32実学習入力で誤差0のままprivate cloneへコピーし、身体・報酬v3・全20 action・参照・PD・トルク上限は維持、許可した制御契約差分はslew0.175→0.5 rad/sだけです。cloneのfresh Adamと、元モデルの7,200,768制御／875 epoch／Adam状態を保持することを明示しています。0.175側は元の低速2件のnative身体・形状・数値判定を再現し、shadow環境の全qpos・qvel・ctrl・トルクも一致しました。0.175の1/2成功に対して、0.5では2件とも西壁に接触して0/2でした。実3,189制御／31,890サブステップ、全4件で転倒0・2 ms最低up0.9805以上、追加RL・BCは0です。指示を早く切り替えるだけでは改善しなかったため、このprivate controllerは不採用です。再生速度と物理の20 ms制御／2 msサブステップは変更していません。全結果は`.runtime/world-model/stage2/command-motor-curriculum-r3-private-slew-diagnostic-verified/paired-comparison.json`、中断した旧prototype294制御は`rejected-attempt.json`へ理由を記録し、最終4件の比較には含めません。

全428テストと専用91テスト、実100 ms／50サブステップのshadow比較、独立したcheckpoint・Adam・保存ソース22・身体資産の監査が通過しています。基準モデルの8開発条件と左右の全41配列、家具教師2条件のnative bytesは旧a92と完全一致しました。`.runtime/world-model/verification/command-motor-curriculum-independent-audit`および`command-motor-local-curriculum-training-audit`に独立監査、`.runtime/world-model/stage2/command-motor-local-curriculum-slower-geometry-paired-comparison.json`に同条件0/2→1/2の比較を保存しています。

家具16組の全32記録は`verification/command-motor-train16-fairness-independent/matched-before-after-audit.json`、private slewの契約とnative記録は`verification/command-motor-r3-private-slew-contract-independent/audit.json`と`command-motor-r3-private-slew-independent/audit.json`でも独立に再検証しました。private native監査では保存された参照・状態から31,886サブステップのPDを厳密再計算し、未保存の初期full qpos／qvelに依存する先頭4サブステップは保存した不変collector hookの検証に依拠する範囲を明示しています。次の作業は、trained command motorを正しい契約で使う高位ナビゲータを別モジュールとして実装し、家具に接触しない方向・歩行距離・停止の選択を学習することです。既存のlegacy motorを前提とする高位モデルへ、今回の110/20モデルを暗黙に差し替えません。

## 既存歩行報酬の継承と実測監査

実際の呼び出し順は `SteeringHumanoidEnv → FootstepHumanoidEnv → ReferenceHumanoidEnv → _ObservedPhysicsEnv → HumanoidRoomEnv` です。`HumanoidRoomEnv.step` と観測用コピーの歩行報酬はともに `progress + 0.5 * upright - cost` ですが、`ReferenceHumanoidEnv.step` はこの戻り値の報酬を破棄し、姿勢・高さ・足の参照に加えて、係数3の進捗、係数4の目標速度追従、係数3の目的地近傍報酬を再計算します。その後 `FootstepHumanoidEnv.step` が実支持・離床・滑りによる項を加減します。家具課題で控除する3・4・3の項は、この参照層の報酬へ実際に含まれています。「存在しない速度追従項を控除していた」という仮説は独立監査で否定され、既存報酬の変更には採用していません。

同じ学長の身体・物理seed・方策出力で、元の `FootstepHumanoidEnv` と旋回を無効にした `SteeringHumanoidEnv` を40制御フレーム（0.8秒）比較しました。位置・速度・加速度・姿勢行列・関節制御・実アクチュエーター力・拘束力、観測と最終報酬は完全一致し、接触観測は実2 msサブステップ400回でした。参照層と実歩容の各項を独立に数値復元した報酬の最大誤差は `1.78e-15` です。さらに方向転換・家具課題を各40フレーム実行し、方向転換の速度追従保持というメタデータ、家具課題での3・4・3の控除、最終報酬の加減を実数値と照合しました。公開済みの29/32の物理合否や、既存モデル・保存ソースのSHAは変更していません。

監査コードと全数値は `.runtime/world-model/verification/reward-mro-audit/audit-script.py` と `audit.json` に保存しています。再実行はリポジトリのルートから `.runtime/world-model/venv/bin/python .runtime/world-model/verification/reward-mro-audit/audit-script.py` です。今後の回帰確認でも、基底の物理stepだけを読むのではなく、この継承順と最終的な報酬の数値を検証します。新しい短指令モーターで指令と追従目標を整合させる設計は、既存報酬の不具合修正とは別の変更です。

第2段階以降の観測は `SceneSensorWrapper` で追加できます。障害物までの距離、床の高さ／欠損、頭上空間を実形状から測定します。入力は59から110次元へ変わるため、現在の59入力の方策と、家具を観測する方策の学習は分けます。

`generalization.py` はエピソードごとに物理部屋を再生成し、train/heldout の scene seed を分離します。元の spawn を保持し、目標と初期の向きを明示します。59入力の方策は推論前に拒否します。階段では目標階の高さと実接触を確認し、階下で水平位置だけが一致した状態を成功にしません。

```bash
.runtime/world-model/venv/bin/python -m world_model.generalization verify --stages 2 3 4 --episodes 10
.runtime/world-model/venv/bin/python -m world_model.generalization evaluate --stages 2 3 4 --episodes 2 --seconds 1
```

同じ学長の身体で第2〜4段階の環境を再生成する場合は、上記へ `--avatar-profile frontend/public/world-model/gakucho-avatar-profile.json` を追加します。再生成時にも身体のSHAを保持し、立位・座面・上階の高さ判定を実測身体へ合わせます。別の身体で学習した110入力方策も、推論前に拒否します。

後者は policy を指定しなければ基準制御の評価です。110入力の方策を渡す場合は `--policy` を使います。空間を再生成できることは、未見部屋で行動する方策を学習済みという意味ではありません。

住宅の衝突形状は `create_home_scene()`、近傍着座課題は `create_seated_task_scene()` で作れます。住宅は元の10.12 mの床と主要家具の位置・椅子の測定座面を使い、机・椅子の脚下を空間として残します。装飾や丸みは学習用の接触形状へ近似します。

## 学習結果の読み方

以下は、学長体格に合わせる前の汎用人体での検証履歴です。新しい学長体格での結果と混ぜません。

2026-10-14の検証では、100 Nを0.2秒加えた20秒の訓練外8条件で、基準PD制御は2/8、学習した立位補正は8/8で転倒を回避しました。近傍着座は12秒の訓練外8条件と、椅子までの距離を変えた20条件を通過しています。椅子までのナビゲーションはこの結果に含みません。

累積2,867,200ステップの歩行候補は別の8条件でも到達・生存8/8でしたが、両脚反復歩行は0/8でした。実際の離床・支持・接触点の滑りを報酬に加えて学習し、累積4,915,200ステップの `stage1-two-meter-gait` を採用しています。その後の追加学習候補は検証結果が悪化したため採用していません。

採用したPPOのみの最初の32条件（seed 1,000,400〜1,000,431）では、生存31/32、到達・停止と独立した歩行監査の同時通過27/32でした。主な未達は目標の25〜33 cm手前での静止です。この問題への対処に使った条件は開発用として扱い、組み合わせた制御の最終評価には新しい32条件（seed 1,000,600〜1,000,631）を使います。

新しい32条件での最終結果は、生存32/32、到達・停止と歩行監査の同時通過31/32（96.875%）でした。学習した歩行だけの同時通過は30/32（93.75%）、物理制御による最後の接近を使った条件は1件です。残る1件（seed 1,000,604）は転倒しませんでしたが目標から約40 cmで未達となり、静止切替条件を満たさなかったため補正していません。この32件は制御方式を固定してから一度評価した結果です。100%の成功や任意の部屋への対応を示すものではありません。

`hierarchical.py` は最初の12秒を同じPPOで実行し、未到達かつ両足で0.3秒以上静止した場合だけ、`whole_body.py` の逆動力学制御へ切り替えます。追加時間は最大20秒です。最後の接近は学習済みの制御ではありません。切替時に物理状態をリセットせず、全体を関節トルクと接触で動かします。学習だけの結果と組み合わせた結果を別々に記録し、最終到達・停止と独立歩行監査の同時通過率90%以上を歩行の合格条件にします。対象は平床の2 m移動であり、家具の回避、未見部屋の経路選択、階段は含みません。

最新の合否・数値は `frontend/public/world-model/manifest.json` と同ディレクトリの `*-heldout-evaluation.json` を正とします。第2段階は同じ学長の身体で小角度の方向転換を学習・評価し、家具回避を学習中です。第3〜4段階は物理形状と観測・評価基盤を準備した状態で、複数部屋と階段を解く方策の学習は未完了です。未通過の課題を習得済みとして表示しません。

`training-progress.json` は実際の学習ステップ数、`provenance.json` はモデルの由来・チェックサム、`evaluation.json` は訓練と分けた seed での評価です。PPO 方策は PD 関節制御への制限付き補正を出力します。立位の成功を、歩行・着座・階段の成功として扱いません。

歩行は移動距離だけで判定しません。`gait_audit.py` は最初の0.5秒を除外し、両足それぞれが25 mm以上上がって60 ms以上除荷され、反対足が身体を支えることを確認します。繰り返しと左右交替、前進、姿勢の維持、目標到達をすべて満たした記録にだけ `walkingVerified: true` を付けます。

T2M参照は `references.py` で物理骨格へ対応付けます。元の軌跡・周期・チェックサムと変換誤差を保存します。`locomotion.py` の足運びはT2Mの周期と歩幅を土台に、脚の関節目標を物理向けに調整します。目標角度はトルク制御へ渡し、IKの結果を身体の位置へ直接書き込みません。

`--strict-gait-reward` は、距離だけを稼ぐ滑り移動を防ぐための実接触に基づく学習目的です。足裏の最低点、ソルバーの垂直荷重、荷重中の足の速度から離床と支持を評価します。両足が繰り返し離床するまで到達報酬を制限します。報酬の計算そのものは物理状態を変えません。合否はこの報酬ではなく、独立した `gait_audit.py` の実測で判断します。

`rollout.json` は物理計算した姿勢の記録です。ブラウザーの `/world-model` はその記録を等速で再生します。新しい学長体格の記録は同じ学長アバターで表示しますが、ブラウザー内のライブ物理計算や、会話中の行動制御への学習済み方策の接続を意味しません。

ここで追加する世界のモデルは、重力・剛体・接触を計算する物理モデルです。ニューラルな未来予測モデルを学習済みという意味ではありません。

実装で参照した仕様：[MuJoCoの物理計算](https://mujoco.readthedocs.io/en/stable/computation/index.html)、[衝突meshの仕様](https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-mesh)、[OSQPのPython API](https://osqp.org/docs/interfaces/python.html)。凹形状と床の穴は、単一meshの凸包に埋められないよう、凸な床片へ分割します。

視覚による第2段階の試験では、`depth_perception.py` が学長の頭部位置から64×48画素、水平90度、下向き15度、最大5 mの深度を計測します。3072本の個別の透視投影レイを実際の衝突形状へ投射し、最初に当たる面のみを観測します。既存の32方向レイを画像へ拡大したものではありません。自分自身を除くための別のモデル・データで問い合わせ、元の物理状態と形状を変更しません。これは理想的な深度カメラで、RGB認識やノイズのある実カメラへの対応は未検証です。

`visual_navigation.py` の上位制御は深度3フレームと局所目標・身体状態11値を受け、100 msごとに向きと歩幅・停止を選択します。部屋の地図、教師経路、形状ID、既存の51特徴は上位制御へ渡しません。下位では凍結した学長用の歩行モデル5aaが20関節を20 msごとに制御し、従来の51特徴も使用します。したがって全身を画像だけで制御するシステムではありません。接触は従来どおり2 msごとに監査し、目標の付け替え、位置の直接更新、外力で移動させる処理は行いません。

`visual_training.py` は訓練用配置のみの幾何教師による模倣を行ってから、CUDA上の深度CNNと上位PPOを学習します。教師の物理実行数、教師ラベルの最適化回数、PPOの判断数、実20 ms制御数と実2 ms積分数を別に保存します。教師のラベルは衝突のない身体運動の保証にはなりません。下位方策の重み・Adam・学習カウンタと身体の契約を保持し、28ソースと身体・モーション資産のSHAを凍結します。CUDAを利用できない場合はCPUへ黙って切り替えず終了します。

最初の試験は `.runtime/world-model/vision-navigation/cuda-depth-ppo-r1` に記録します。GPU 0のA6000で上位CNNとPPOを実行し、物理環境は8 CPUプロセスへ分けます。GPU 1の学長AI/T2Mとは分離しています。`started.json` は実行条件、`progress.json` は実際の進捗、`training.json` は更新されたCNN・Adam・実制御数の証拠です。`before/` と `after/` は同じ開発用未見8配置（1180100〜1180107）での比較、`after-blank-depth/` は身体状態を維持し画像のみ全画素1へ置換した評価です。最終評価用1800000〜1999999は今回使いません。

```bash
LD_LIBRARY_PATH="$PWD/.runtime/nvidia-595.91.07${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .runtime/world-model/venv/bin/python -u -m world_model.visual_training --motor-policy .runtime/world-model/stage2-gakucho-command-motor-local-curriculum-adapted/policy.zip --avatar-profile frontend/public/world-model/gakucho-avatar-profile.json --output .runtime/world-model/vision-navigation/new-depth-training --steps 65536 --envs 8 --bc-episodes 16 --bc-epochs 30 --eval-episodes 8 --difficulty-max .5
```

この試験は平床・家具1個・目標1個で位置や大きさ、向きなどを変えた範囲です。多数の家具、見えない角の記憶、複数部屋、階段、実家庭の画像からの自己位置推定は今回の合格対象に含めません。3フレームの履歴は短時間の情報で、長期の地図記憶ではありません。評価が完了するまで習得済みとは扱わず、会話中の学長アバターや公開manifestへの自動採用もしません。

中断前のR1では訓練用16配置で6813実制御・68130物理積分を行い、1370ラベルを収集しました。教師による停止までの成功は3/16です。CUDA上の模倣は30 epoch・180最適化ステップで完了し、`before-ppo-policy.zip` を保存しています。この時点のPPO学習は0判断です。R1の途中の未見評価3件とログを残し、`cuda-depth-ppo-r2` へそのチェックポイントから再開します。`--resume ... --bc-episodes 0` は方策全体・価値関数・標準偏差・Adamと実カウンタの一致を検証し、模倣を繰り返しません。新しい出力先とソース一式のSHAを別に保存し、完了していない試行へ上書きしません。

R2は開始時の契約検証で停止しました。カメラの取り付け座標が保存時にtupleからJSON配列へ変わるためで、座標値や物理動作への変更はありません。生成時から保存形式の配列に合わせ、実際の学長用身体・歩行モデルでの保存／再読込を含む8件の再開テストを通過しました。現在の実行先は `cuda-depth-ppo-r3`、ログは同名の `.log`、独立実行プロセスの情報は `cuda-depth-ppo-r3-launch.json` です。R2のPPO判断数は0で、失敗記録も保存しています。R3では学習前後の評価記録へ、それぞれの方策チェックポイントSHAも付与します。

R3の学習前は開発用未見8配置で全条件通過3/8、生存8/8でした。8192判断時点の同8配置では2/8、32768判断では1/8へ下がりました。各中間評価の物理実行数はPPO学習数に含めません。訓練入力1370件で画像特徴と後段をそれぞれ交換して調べると、画像特徴の変化だけで大きな旋回指示が弱まりました。これは訓練データ上の機能変化の調査で、実際の成功率や、価値関数の勾配だけが唯一の原因だという証拠ではありません。

改善候補 `visual_refinement.py` は、同じ模倣済みモデルから画像・身体特徴と行動標準偏差を固定し、移動判断と価値関数の後段を学習します。学習率は1e-5、target KLは0.01、予定16384判断、別の訓練seed92801です。元の重み・Adam状態・カウンタを保持して開始し、模倣のやり直しや教師ラベルの混入は行いません。この分岐ではCNNの追加更新は行わず、既にGPUで学習した特徴を保持します。試行は `cuda-depth-conservative-r4`、凍結対象の29ソース・資産・方策のSHAと、更新された後段・更新されなかった特徴の証拠を保存します。R3とR4は同じ開発用8配置で評価する別試行であり、異なる訓練seedも使うため、成功率の差を固定化だけの因果効果とは主張しません。

2026-10-15の視覚実験は学習・比較評価まで完了しました。GPU 0でR3は65536判断・326339制御・3263390物理積分・256 PPO epoch、R4は16384判断・81609制御・816090物理積分・64 PPO epochを実行しました。R3ではCNNを含む状態が更新され、R4では固定した特徴・標準偏差の全バイト一致と、actor/critic後段12テンソル・Adam12状態各256更新を独立確認しました。各評価の実行数と教師収集6813制御はPPO学習数へ加算していません。

| 同じ未見8配置での候補 | 回避・到達・停止など全ゲート成功 | 転倒なし |
| --- | ---: | ---: |
| 模倣学習後・PPO前 | 3/8 | 8/8 |
| R3 通常PPO後 | 1/8 | 8/8 |
| R3 画像のみ無効化 | 0/8 | 8/8 |
| R4 画像特徴・標準偏差を固定したPPO後 | 4/8 | 8/8 |
| R4 画像のみ無効化 | 0/8 | 8/8 |

R4は1180101・1180105・1180107で新たに成功し、1180100・1180106では以前の成功を失いました。8件の開発評価での小さな改善であり、任意の部屋への汎用性を示しません。画像の無効化では状態11値と下位の51特徴を維持しているため、上位が画像を利用する証拠として扱い、全身の画像のみ制御とは扱いません。失敗記録の `physicalRepeatedGait` は内部に到達判定も含むため、このゲートだけで歩行リズム自体の崩壊と解釈しません。

結果の一覧は `.runtime/world-model/vision-navigation/summary.json`、ソース・身体資産・公開manifestの監査は `release-audit.json`、各試行の `review/` に比較図と実50 Hz記録からの独立再計算があります。全体テスト513件に加え、実保存／再読込を含む再開8件と改善候補12件を通過しました。第2段階の合格・本番採用は行っておらず、最終評価用配置も使っていません。次の検証課題は、実際に衝突せず成功した経路の教師データと、旋回・減速時にも身体幅を確保する下位動作を改善することです。

次の接触診断では、R4が失敗した4配置の初期qpos/qvelと部屋形状を固定して再実行しました。元と同じCUDA推論では4件すべての終了時刻・最終距離・形状SHAを再現でき、幾何経路による指示では0/4から2/4になりました。残る2件は回避開始と家具との余裕を広げても接触しました。3件の右回避では、正方向の旋回指示が出ているのに実旋回の平均が逆向きでした。詳細は `vision-navigation/contact-r5-diagnostic/diagnosis.json`、独立監査は同ディレクトリの `trajectory-recomputation-audit.json` に保存しています。15診断軌跡・7178制御・71780物理積分であり、学習数には含めません。

同じ学長用身体・同じ歩行モデルの単独測定でも旋回に左右差があり、3〜8秒の平均は左−0.203 rad/s、右+0.091 rad/s、直進指示でも3〜4秒に−0.133 rad/sの偏りがありました。停止指示から持続的に0.1 m/s未満になるまで0.7〜1.2秒かかります。目標旋回速度は実際の旋回の上限を保証しないため、報酬式だけから通行可能性を断定しません。`motor_feasibility.py` は測定のみを行い、身体・目標・外力・関節制限を書き換えません。

`qualified_visual_training.py` は訓練用64配置の全試行を数え、実際の全50 Hz軌跡から独立した8ゲートを通過した12軌跡・2359画像状態ラベルだけを採用しました。収集は33216制御・332160物理積分で、失敗52件も集計に残します。`r5-qualified-teachers-5aa-v2/independent-accepted-trace-audit.json` は全採用gzip軌跡・カメラNPZ・身体／モデルSHA・例数を再検証した記録です。最初の収集は監査ヘッダのtask不足を独立レビューで見つけて停止し、既知の実成功軌跡による回帰テスト後にv2へ再開しました。

CUDAで画像特徴・価値関数・行動標準偏差を保持したまま判断層だけを300回更新しましたが、同じCPU推論にそろえた開発用8配置では基準3/8に対し15 epoch・30 epochの両候補が2/8へ下がりました。この候補は採用しません。教師データの成功だけで未見の身体動作の成功は保証されません。`r5-qualified-visual-5aa-v2/independent-policy-preservation-audit.json` に重み・継承PPO回数・将来PPO用Adamの明示的な再初期化の証拠を保存しています。

CPUとCUDAの推論では同一入力でも行動に最大約1e-4の差があり、それが接触の有無を変える配置がありました。比較用の `qualified_evaluation.py` は同一推論デバイス・同一初期qpos/qvel/目標/深度/状態と実際に消費された最初の行動を記録し、各軌跡を独立再監査します。新規32配置1182000〜1182031、物理seed3182000〜3182031の基準記録は `r5-new32-baseline` に保存し、候補の選択が終わるまで成績を参照しません。異なる下位モデルの比較は明示的な転送として記録します。

下位の旋回学習は元の5aaから別試行として実施しました。`motor_tracking_training.py` は瞬時旋回誤差の幅を狭めた明示的報酬v4で204800実制御、`motor_cycle_tracking_training.py` は元のv3報酬へ実際の歩行参照1周期の平均旋回追従誤差を加えた報酬v5で131072実制御をGPU 0上で学習しました。計335872制御・3358720物理積分・追加820 PPO epochで、両試行とも元のactor・critic・std・Adam・カウンタを保持して開始したことを検証しています。履歴は報酬計測のみで、世界の状態や方策入力を変更しません。`tracked_visual_navigation.py` は報酬v4/v5の完全一致する契約・身体・20関節・110入力・出自を確認してから、上位方策を明示的に接続します。旧29ソースと従来5aa専用の検証を緩めません。

瞬時誤差の学習では左右旋回が悪化し、幾何教師4配置の成功も2/4のままでした。周期平均の中間40960制御モデルは右旋回単独で0.0905から0.1237 rad/sへ改善しましたが、左旋回を弱め、学習を続けた最終モデルは右が逆方向へ退行しました。最終v5の幾何教師は2/4から1/4、同一CPU・同じ上位方策・同じ初期画像と身体状態での視覚移動は3/8から0/8へ下がりました。左右反復足運び・姿勢・生存自体は両方8/8維持しており、到達を含む合成gateだけから歩行が壊れたとは判断しません。v4・v5最終モデルとも採用しません。曲線は `vision-navigation/review/low-motor-turning-r5.png`、上位全重み・Adamと最初に実消費された行動の一致は `r5-old8-cycle-high/independent-first-action-identity.json` に保存しています。

追加の比較では、全形状を知る教師が実ゴール距離0.20 mで既存STOPをラッチしても、旧5aaは2/4、新v5は1/4のままでした。失敗の一部はその距離まで接近できず、停止指示だけでは解消しません。`goal_stop_attribution.py` はこの診断専用で、heldoutの教師行動を学習データへ混入させません。全8軌跡の初期qpos/qvel・形状・実50 Hz記録・2 ms接触・ゴール不変とラッチ後の消費指令を独立監査しています。

`motor_expert_mixture.py` は左・停止に旧5aa、正の実消費指令角に応じてv5中間モデルを滑らかに混ぜる診断です。2つの学習済み下位方策と明示的な切替規則を組み合わせており、単一モデルを新たに学習したものではありません。両方の重み・Adam・回数を凍結し、実20 msごとに角度・混合率・20関節出力を記録しました。しかし右旋回は0.0681 rad/sへ弱まり、幾何教師の成功も同じ2/4でした。この構成も採用しません。全結果は `contact-r5-mixture-diagnostic/comparison.json` と `motor-mixture-r5-probes/probes.json` に残します。

時間入力に8秒／30秒の不一致があるという調査中の仮説は撤回しました。`ReferenceHumanoidEnv` が最後の基底特徴を歩行参照の位相へ置き換えており、実際の初期状態・先頭50観測と関節出力・8秒分80物理サンプルは一致しました。人工的に位相を入れ替える介入はruntime不具合の証拠ではありません。`motor-time-domain-r5/native-input-action-equality.json` に保存しています。

追加学習候補の選択は開発用8配置と4配置の診断で終了し、従来のR4上位と5aa下位、CPU推論を維持する構成を `r5-controller-selection-lock.json` で固定しました。新32配置はこの維持モデルの汎化を測るもので、今回の追加学習で改善したという比較には使いません。`r5-summary.json` に収集・GPU学習・評価・不採用の理由を分けて記録します。学長AI/T2Mの稼働、公開manifest、元29ソースと身体・モーション資産は維持しています。フェーズ2の合格、任意の部屋への対応、本番アバターへの採用は未達です。

固定後に開示した新32配置では、回避・到達・停止など全gateの成功は6/32（18.75%）、生存32/32、無接触6/32でした。残る26件は家具22・南壁1・西壁3への接触で終了しています。到達を除いた足運び・姿勢の全gateは29/32で、到達を含む合成歩行指標6/32と分けて記録します。実14440制御・144400物理積分・2901上位判断で、評価や監査による追加学習は0です。全32 native軌跡の判定、実初期qpos/qvel・画像・状態・目標、保存30ソースを独立再検証しました。初回の基準計測では実消費した最初の方策出力は保存していないため、保存入力からのCPU再計算と区別します。全結果と範囲は `r5-new32-retained-unblind-review/review.json` に保存しています。

次の設計課題は、前進速度と旋回速度を独立して観測へ渡す新しい下位モデル契約です。現行は歩行lookahead0.35〜0.50 mと停止を切り替え、停止時には横方向の指令距離が0となって角度も失われるため、その場旋回を要求できません。左右旋回・減速・その場旋回を、元の歩行能力を保持する学習と個別gateで検証してから、合格した下位モデルに対して視覚判断を再学習する方針です。これは今回の結果から得た次の案であり、実装・学習済みとは扱いません。今回の全体570テストと追加対象46件（1件skip）、検証強化後11件は通過しました。
