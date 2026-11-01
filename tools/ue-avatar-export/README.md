# UE(MetaHuman)アバターを three.js ビューア用 glb に書き出す

`frontend/public/avatars/gakucho_v3_suit.glb`(スーツ姿・本命)と `gakucho_v1_tshirt.glb`(Tシャツ姿)は、
`~/UnrealProjects/MyProject` の MetaHuman「gakuchyou(学長)」からこのフォルダのスクリプトで生成した。
ビューア側の仕組みは `docs/ue-avatar.md` を参照。

## 生成手順(再現用)

```bash
UE=~/Applications/Linux_Unreal_Engine_5.7.0
S=$(pwd)/tools/ue-avatar-export
export GAKUCHO_GLB_OUT=~/gakucho_glb_export   # 書き出し先
export GAKUCHO_STEPS=v3mergeF,v1mergeF,hair    # 実行するステップ(下記)

$UE/Engine/Binaries/Linux/UnrealEditor-Cmd ~/UnrealProjects/MyProject/MyProject.uproject \
  -ExecutePythonScript=$S/export_gakucho_glb.py \
  -EnablePlugins=SkeletalMerging \
  -RenderOffScreen -unattended -nosplash -nopause -NoSound -nocef -graphicsadapter=0 \
  -ddc=NoZenLocalFallback -norenderthread -norhithread -ansimalloc \
  -abslog=$GAKUCHO_GLB_OUT/editor.log

# 書き出した glb を最終形へ(テクスチャ 1K 縮小 / 未使用材質除去 / 髪結合 / 目シェル非表示)
bash $S/build_final.sh $GAKUCHO_GLB_OUT/gakucho_v3_mergeF_lod0.glb gakucho_v3_suit.glb $GAKUCHO_GLB_OUT/gakucho_v3_haircards.glb
bash $S/build_final.sh $GAKUCHO_GLB_OUT/gakucho_v1_mergeF_lod0.glb gakucho_v1_tshirt.glb
cp gakucho_v3_suit.glb gakucho_v1_tshirt.glb frontend/public/avatars/
```

`GAKUCHO_STEPS` の値:

| ステップ | 内容 |
| --- | --- |
| `v3mergeF` / `v1mergeF` | 体+服+顔を SkeletalMerging で 1 メッシュにマージして書き出す(**推奨**) |
| `hair` | 髪カード(グルームの Cards 用スタティックメッシュ)を書き出す。材質は無いので `build_final.sh` 側で暗色にする |
| `v3body` `v3face` `v1body` `v1face` | 個別書き出し(顔は正常だが体・服はウェイトが壊れる。下記参照) |
| `combined` | `Content/gakuchyou_CombinedSkelMesh`(v1 相当、材質なし) |

髪は `gakucho_v3_haircards.glb` として別書き出しし、`build_final.sh` が `strip_textures.py` で暗色単色にしてから結合する
(`GAKUCHO_STEPS=hair` は前の書き出しで生成済みのものを流用可)。

## 起動フラグの意味(すべて実測で必要だったもの)

| フラグ | 理由 |
| --- | --- |
| `-RenderOffScreen` | X ディスプレイ無しで起動(SDL は dummy ドライバ)。マテリアル焼き込みには GPU(Vulkan)が必要なので `-nullrhi` は不可 |
| `-ddc=NoZenLocalFallback` | ZenServer が起動できず DDC 初期化で Fatal になるのを回避 |
| `-norenderthread -norhithread` | RHI スレッドの終了処理でクラッシュするため |
| `-ansimalloc` | mimalloc だとヒープ破壊が後で発現して原因が追えない。glibc なら即 abort し `free(): invalid size` 等が stdout に出る |
| `-EnablePlugins=SkeletalMerging` | `unreal.SkeletalMergingLibrary.merge_meshes` を使うため(プロジェクト設定は変更しない) |

## GLTFExporter(UE 5.7)の既知バグと回避策

`Engine/Plugins/Enterprise/GLTFExporter/Source/GLTFExporter/Private/Converters/GLTFAccessorConverters.cpp`

1. **ウェイトの読み出し位置がずれる**: `FGLTFBoneWeightBufferConverter` はソース側ボーンインデックスの幅を
   「出力側の最大ボーン番号が 255 を超えるか」で決めている。MetaHuman の体・服は
   「セクション内参照は 8bit だがスケルトン番号は 256 以上」なのでウェイトがデタラメになる
   (合計が 0〜8)。顔メッシュ(875 ボーン、16bit)は正常。
   → **顔メッシュと一緒にマージ**すると LOD 全体が 16bit インデックスになり正しく読める。
2. **可変影響数レイアウトでヒープ破壊**: 影響数が 4/8 未満の頂点で `VertexInfluenceCount - InfluenceOffset`
   が符号なしで負になり、配列の手前に書き込む → `free(): invalid size` で abort。
   → 書き出す全メッシュの LOD ビルド設定を `bone_influence_limit = 8` に変更してからマージする
   (`export_gakucho_glb.py` の `make_export_safe`。アセットは保存しないのでプロジェクトは変わらない)。
3. `SkeletalMergingLibrary.merge_meshes` は材質が 1 つも無いメッシュ(`gakuchyou_CombinedSkelMesh`)を
   混ぜるとアサートする。

BP(`BP_gakuchyou`)をアクターとして書き出す方法は、グルーム(髪の Niagara)のロードで落ちるため使えない。

## ツール

| ファイル | 役割 |
| --- | --- |
| `export_gakucho_glb.py` | UE 内で実行する書き出しスクリプト |
| `build_final.sh` | 縮小 → 髪結合 → 最終調整 → 検査 を一括実行 |
| `shrink_textures.py` | 画像を上限解像度へ縮小(ベースカラーは JPEG、法線/ORM は PNG 維持)+ 未使用材質の削除 |
| `merge_glb.py` | 複数 glb を 1 ファイルに結合 |
| `finalize_glb.py` | 材質未割当(WorldGridMaterial)セクション除去、目の透明シェル非表示 |
| `strip_textures.py` | テクスチャを剥がして単色化(髪カード用) |
| `inspect_glb.py` / `check_weights.py` | 構造とスキンウェイト合計の検査(ウェイト合計が 1.00 でなければ書き出し失敗) |

## ライセンス注意

MetaHuman は Epic のライセンス上、UE / Twinmotion 以外での利用に制限がある。第三者へ配信する前に条件を確認すること。
