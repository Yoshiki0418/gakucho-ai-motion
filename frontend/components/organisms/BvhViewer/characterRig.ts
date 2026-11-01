'use client'

/**
 * 外部キャラクターモデル(glTF/GLB)を読み込み、BVH骨格のポーズを毎フレーム
 * 転写(リターゲット)するためのユーティリティ。
 *
 * 設計方針:
 *   - 既存の BVH 骨格 + 独自 idle/姿勢制御はそのまま「ドライバ」として動かし続ける。
 *   - その骨格の現在ポーズを、ここで読み込んだキャラクターへ毎フレーム転写する。
 *   - これにより idle・座る/立つ・モーション再生のロジックを一切作り直さずに、
 *     見た目だけを別のアバターへ差し替えられる。
 *
 * リターゲット方式(重要):
 *   単純なローカル回転コピーではなく「各リグ自身のレストポーズからの差分」を
 *   ワールド基準に置き換えて転写する。これにより
 *     - ボーン名が違う   (Mixamo "LeftArm" ↔ UE "upperarm_l")
 *     - ボーン軸が違う   (UE はボーン長手方向が X)
 *     - レストポーズが違う (BVH は T ポーズ / UE マネキンは A ポーズ)
 *   いずれのケースでも破綻せずに転写できる。Unreal Engine から書き出した
 *   人型アバター(UE5 マネキン系 / MetaHuman 系スケルトン)をそのまま扱える。
 *
 * 複数スキン(UE アクター書き出し):
 *   体・顔・服が別々の SkeletalMeshComponent として書き出され、glb 内で別スケルトンに
 *   なっている場合は、人型ボーンが最も多く対応付いたスキンを「主リグ」とし、残りのスキンは
 *   主リグの骨格へ統合する(同名ボーンは主リグのボーンで再バインド、表情ボーンのように
 *   主リグに無いボーンは主リグ側の親ボーンへ付け替える)。UE の Leader Pose Component と
 *   同じ発想だが、毎フレームの行列コピーではなく読み込み時の一度の骨格統合で済ませる。
 *
 * モデルの差し替え:
 *   frontend/public/avatars/ に rigged な .glb を置くだけ。ビューア下部の
 *   セレクタに自動で並び、その場で切り替えられる(docs/ue-avatar.md 参照)。
 */

import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import {
  detectRigFlavor,
  mapBonesByCanonicalName,
  normalizeBoneName,
  originalBoneName,
} from './humanoidBones'

/** 既定のキャラクターモデル URL。public/avatars 配下に置く。 */
export const CHARACTER_MODEL_URL = '/avatars/character.glb'

export type CharacterTuning = {
  /** BVH 骨格の高さに対する追従スケール係数。1.0 で骨格と同じ高さ。 */
  heightScale: number
  /** Y 軸回転(度)。キャラが横/後ろを向く場合に調整する(UE 由来は 90 や 180 が必要になりやすい)。 */
  yawDegrees: number
  /** 追加の上下オフセット(BVH 単位)。足が床に沈む/浮く場合に調整。 */
  yOffset: number
}

/** 実機で見た目を微調整するためのノブ(新規ロード時の初期値)。 */
export const CHARACTER_TUNING: CharacterTuning = {
  heightScale: 1.0,
  yawDegrees: 0,
  yOffset: 0,
}

/** BVH(転写元)側のレスト情報。骨格をパースした直後に採取する。 */
export type SourceRig = {
  skeleton: THREE.Skeleton
  /** 正規名 → BVH ボーン */
  byCanonical: Map<string, THREE.Bone>
  /** ボーン → レストポーズ情報 */
  rest: Map<
    THREE.Bone,
    {
      localInverse: THREE.Quaternion
      parentWorld: THREE.Quaternion
      parentWorldInverse: THREE.Quaternion
      position: THREE.Vector3
    }
  >
  /** ルートボーン(Hips)。ルート移動の転写に使う。 */
  rootBone: THREE.Bone
  /** レストポーズでのボーン位置(ルートボーン空間)。T ポーズ合わせ込みで使う。 */
  restWorldPosition: Map<THREE.Bone, THREE.Vector3>
  /**
   * 「足が床(y=0)に接する」ルートボーン位置。ルート移動転写の基準に使う。
   *
   * BVH の Hips は OFFSET が原点なのでレスト位置は y=0 だが、モーションデータは
   * 実際の腰の高さ(約 0.93)を position チャンネルで持つ。素の y=0 を基準にすると
   * その腰の高さがまるごと差分として乗り、キャラクターが脚の長さぶん浮いてしまう。
   */
  groundedRootPosition: THREE.Vector3
  flavor: string
}

type RetargetLink = {
  canonical: string
  target: THREE.Bone
  /** 合わせ込み後のレストにおける、モデル空間でのワールド回転。 */
  targetRestWorld: THREE.Quaternion
  source: THREE.Bone
  /** 転写元レストのワールド回転の逆(ルートボーン空間)。 */
  sourceRestWorldInverse: THREE.Quaternion
}

export type LoadedCharacter = {
  /** シーンに add するラッパー(スケール/向き/位置はこのグループで制御)。 */
  root: THREE.Group
  /** glTF の scene 本体。 */
  model: THREE.Object3D
  /** 最初に見つかった SkinnedMesh(ポーズ転写のターゲット)。 */
  skinnedMesh: THREE.SkinnedMesh
  /** scale=1 時のモデル全高。 */
  naturalHeight: number
  /** このキャラ固有の表示調整値。 */
  tuning: CharacterTuning
  /** リグ種別(mixamo / unreal / vrm …)。ログ・UI 表示用。 */
  flavor: string
  /** 正規名 → モデルボーン */
  bonesByCanonical: Map<string, THREE.Bone>
  /** モデルボーン → レスト情報(バインド時に採取) */
  rest: Map<
    THREE.Bone,
    {
      local: THREE.Quaternion
      parentWorld: THREE.Quaternion
      parentWorldInverse: THREE.Quaternion
      position: THREE.Vector3
    }
  >
  /** 現在バインド中の転写リンク。bindCharacterToSource で更新される。 */
  links: RetargetLink[]
  /** 転写元のルートボーン(移動転写用)。 */
  sourceRootBone: THREE.Bone | null
  /** 転写元ルートのレスト位置。ここからの差分だけをモデルへ渡す。 */
  sourceRootRestPosition: THREE.Vector3 | null
  /** モデル側のルート相当ボーン(通常 hips)。 */
  targetRootBone: THREE.Bone | null
  /**
   * hips の親のワールドスケール(モデル空間内)。
   * glTF の Armature に 0.01 が入っている(= hips のローカル単位が cm)ことが多く、
   * ルート移動の単位換算でこれを掛けないと 100 倍ずれる。
   */
  targetRootUnitScale: number
  /** ルート移動(腰の上下・前後)を転写するか。座り動作の接地に必要。 */
  transferRootMotion: boolean
  /**
   * 合わせ込み後のレスト姿勢における、モデル空間での最下点(通常は靴底)の y。
   * モデルの原点が足元にあるとは限らないため、これを引いて床(y=0)へ接地させる。
   */
  groundOffset: number
  /**
   * 体の正面を BVH と揃えるために必要な Y 回転(度)。バインド時に自動算出する。
   * UE のキャラクターは +X が正面、BVH は +Z が正面なので、そのままだと後ろを向く。
   * `tuning.yawDegrees` はこの値に対する追加調整として足し込まれる。
   */
  autoYawDegrees: number
}

/** モデル内の SkinnedMesh をすべて列挙する。 */
const collectSkinnedMeshes = (model: THREE.Object3D): THREE.SkinnedMesh[] => {
  const meshes: THREE.SkinnedMesh[] = []
  model.traverse((object) => {
    const mesh = object as THREE.SkinnedMesh
    if (mesh.isSkinnedMesh) meshes.push(mesh)
  })
  return meshes
}

/**
 * 副リグ(主リグとは別スケルトンを持つ SkinnedMesh 群)を主リグの骨格へ統合する。
 *
 * - 主リグに同名ボーンがあるもの: そのボーンを直接使う(副リグ側のボーンは捨てる)
 * - 無いもの(表情ボーンなど): 主リグ側の対応する親ボーンの下へ付け替える
 *
 * どちらも「両リグのレストポーズがワールド空間で一致している」ことを前提にしており、
 * UE の同一アクターから書き出した体・顔・服はこれを満たす。統合後は主リグのボーンを
 * 動かすだけで全スキンが追従する。
 */
const mergeSecondaryRigs = (
  skinnedMeshes: readonly THREE.SkinnedMesh[],
  primary: THREE.SkinnedMesh,
): { secondaryRigs: number; mergedBones: number; graftedBones: number } => {
  const primaryBones = new Set<THREE.Bone>(primary.skeleton.bones)
  const primaryByName = new Map<string, THREE.Bone>()
  for (const bone of primary.skeleton.bones) {
    // GLTFLoader は重複名を "pelvis_1" に書き換えるので、glTF 上の元名で照合する
    const key = normalizeBoneName(originalBoneName(bone))
    if (!primaryByName.has(key)) primaryByName.set(key, bone)
  }

  const mergedSkeletons = new Map<THREE.Skeleton, THREE.Skeleton>()
  let secondaryRigs = 0
  let mergedBones = 0
  let graftedBones = 0

  for (const other of skinnedMeshes) {
    if (other === primary) continue
    // 同じスケルトンを共有するプリミティブ(multi-material の分割)は主リグの一部
    if (other.skeleton.bones.some((b) => primaryBones.has(b))) continue

    let merged = mergedSkeletons.get(other.skeleton)
    if (!merged) {
      secondaryRigs += 1
      const { bones, boneInverses } = other.skeleton
      const leaders = bones.map(
        (bone) => primaryByName.get(normalizeBoneName(originalBoneName(bone))) ?? null,
      )
      const boneIndex = new Map<THREE.Object3D, number>(bones.map((b, i) => [b, i]))

      bones.forEach((bone, i) => {
        const parent = bone.parent
        const parentLeader =
          parent && boneIndex.has(parent) ? leaders[boneIndex.get(parent)!] : null
        if (leaders[i]) {
          // 主リグにあるボーン: 副リグ側の階層は不要なので、最上位のものをシーンから外す
          if (!parentLeader) bone.removeFromParent()
          mergedBones += 1
        } else if (parentLeader) {
          // 主リグに無いボーン(表情ボーン等): 主リグ側の親へ付け替える。
          // add() はローカル変換を維持するので、親のレストが一致していれば位置は変わらない。
          parentLeader.add(bone)
          graftedBones += 1
        }
      })

      merged = new THREE.Skeleton(
        bones.map((bone, i) => leaders[i] ?? bone),
        boneInverses,
      )
      mergedSkeletons.set(other.skeleton, merged)
    }
    other.bind(merged, other.bindMatrix)
  }

  return { secondaryRigs, mergedBones, graftedBones }
}

/**
 * `object` から `stopAfter` までの親を遡ってワールド回転を合成する。
 * ローカル回転がレスト状態のときに呼ぶこと。
 */
const restWorldQuaternion = (
  object: THREE.Object3D | null,
  stopAfter: THREE.Object3D,
): THREE.Quaternion => {
  const q = new THREE.Quaternion()
  let current: THREE.Object3D | null = object
  while (current) {
    q.premultiply(current.quaternion)
    if (current === stopAfter) break
    current = current.parent
  }
  return q
}

/**
 * BVH 骨格のレストポーズを採取する。
 *
 * 必ず `BVHLoader.parse()` 直後、アニメーション適用や idle 姿勢の付与より
 * 前に呼ぶこと。呼ぶタイミングを誤るとレストが「動いた後の姿勢」になり、
 * 転写結果が全体的に傾く。
 */
export function createSourceRig(
  skeleton: THREE.Skeleton,
  /** そのモーションのクリップ。あればルート移動の基準を初期フレームから取る。 */
  clip?: THREE.AnimationClip,
): SourceRig {
  const bones = skeleton.bones
  const rootBone = bones[0]
  rootBone.updateMatrixWorld(true)

  const rest: SourceRig['rest'] = new Map()
  for (const bone of bones) {
    const parentWorld = restWorldQuaternion(bone.parent, rootBone)
    rest.set(bone, {
      localInverse: bone.quaternion.clone().invert(),
      parentWorld,
      parentWorldInverse: parentWorld.clone().invert(),
      position: bone.position.clone(),
    })
  }

  // レスト姿勢でのボーン位置。ルートボーン空間へ落として保持する
  // (シーン側の平行移動に影響されないようにするため)。
  const rootInverse = new THREE.Matrix4().copy(rootBone.matrixWorld).invert()
  const restWorldPosition = new Map<THREE.Bone, THREE.Vector3>()
  for (const bone of bones) {
    restWorldPosition.set(
      bone,
      bone.getWorldPosition(new THREE.Vector3()).applyMatrix4(rootInverse),
    )
  }

  // レスト姿勢での最下点(通常はつま先)を床に降ろしたときのルート位置
  let lowestY = Infinity
  for (const position of restWorldPosition.values()) {
    if (position.y < lowestY) lowestY = position.y
  }
  const groundedRootPosition = rest.get(rootBone)!.position.clone()
  if (Number.isFinite(lowestY)) groundedRootPosition.y -= lowestY

  // ルート移動の基準は、可能ならモーションの初期フレームから取る。
  //
  // 「立っている状態のルート位置」の表し方は供給元でまったく違う。
  //   Custom-T2M-Lab : 腰の高さ (0, 0.87, 0)
  //   SentiAvatar    : 撮影空間内の絶対座標 (-20.7, 98.6, 102.2)
  // 幾何から接地位置を推定すると後者で 6 体分ほど浮き上がり、画角外へ飛ぶ。
  // 初期フレームを基準にすれば、どちらでも「開始時の差分ゼロ」になる。
  const positionTrack = clip?.tracks.find((track) =>
    track.name.endsWith('.position'),
  )
  if (positionTrack && positionTrack.values.length >= 3) {
    groundedRootPosition.set(
      positionTrack.values[0],
      positionTrack.values[1],
      positionTrack.values[2],
    )
  }

  const flavor = detectRigFlavor(bones.map((b) => b.name))
  const byCanonical = mapBonesByCanonicalName(bones)
  console.log(
    `[characterRig] source rig captured: bones=${bones.length}, ` +
      `flavor=${flavor}, mapped=${byCanonical.size}, ` +
      `groundedRootY=${groundedRootPosition.y.toFixed(3)}`,
  )

  return {
    skeleton,
    byCanonical,
    rest,
    rootBone,
    restWorldPosition,
    groundedRootPosition,
    flavor,
  }
}

/** 転写の基準として使うレスト姿勢(ボーンごとのローカル回転と親のワールド回転)。 */
type RestOrientation = {
  local: THREE.Quaternion
  parentWorld: THREE.Quaternion
  parentWorldInverse: THREE.Quaternion
}

/**
 * `object` から `stopAfter` まで、いまのローカル回転を遡って `out` へ合成する。
 * 毎フレーム呼ぶので新規確保はしない。
 */
const accumulateWorldQuaternion = (
  out: THREE.Quaternion,
  object: THREE.Object3D | null,
  stopAfter: THREE.Object3D,
): THREE.Quaternion => {
  out.set(0, 0, 0, 1)
  let current: THREE.Object3D | null = object
  while (current) {
    out.premultiply(current.quaternion)
    if (current === stopAfter) break
    current = current.parent
  }
  return out
}

/** `object` から `stopAfter` まで、いまのローカル回転を遡って合成する。 */
const currentWorldQuaternion = (
  object: THREE.Object3D | null,
  stopAfter: THREE.Object3D,
): THREE.Quaternion => {
  const q = new THREE.Quaternion()
  let current: THREE.Object3D | null = object
  while (current) {
    q.premultiply(current.quaternion)
    if (current === stopAfter) break
    current = current.parent
  }
  return q
}

/**
 * 「正規名 → その直下にぶら下がる正規名の一覧」を作る。
 * 途中に対応表に無いボーン(捻りボーン等)が挟まっていても、
 * 最も近い対応ボーンを親とみなすため、リグ間で階層の粒度が違っても比較できる。
 */
const mappedChildCanonicals = (
  byCanonical: Map<string, THREE.Bone>,
): Map<string, string[]> => {
  const canonicalOf = new Map<THREE.Object3D, string>()
  for (const [canonical, bone] of byCanonical) canonicalOf.set(bone, canonical)

  const children = new Map<string, string[]>()
  for (const [canonical, bone] of byCanonical) {
    let parent: THREE.Object3D | null = bone.parent
    while (parent) {
      const parentCanonical = canonicalOf.get(parent)
      if (parentCanonical) {
        const list = children.get(parentCanonical)
        if (list) list.push(canonical)
        else children.set(parentCanonical, [canonical])
        break
      }
      parent = parent.parent
    }
  }
  return children
}

/**
 * 腰・両腿・上体から「体の正面」を XZ 平面上のベクトルとして求める。
 *
 * 腰から上体へのベクトル(up)と、右腿から左腿へのベクトル(right)の外積が正面になる。
 * T ポーズ合わせ込みは各ボーンの振り(swing)しか直さず、体軸まわりの捻りは
 * モデルの bind 値のままなので、正面の判定はこの体フレームから別途行う必要がある。
 */
const bodyForward = (
  hips: THREE.Vector3,
  leftUpLeg: THREE.Vector3,
  rightUpLeg: THREE.Vector3,
  upper: THREE.Vector3,
): THREE.Vector3 | null => {
  const right = leftUpLeg.clone().sub(rightUpLeg)
  const up = upper.clone().sub(hips)
  if (right.lengthSq() < 1e-12 || up.lengthSq() < 1e-12) return null
  const forward = right.normalize().cross(up.normalize())
  forward.y = 0
  return forward.lengthSq() < 1e-12 ? null : forward.normalize()
}

/**
 * モデルのレスト姿勢を BVH のレスト姿勢へ合わせ込み、転写の基準として採取する。
 *
 * なぜ必要か:
 *   UE のスケルトンは A ポーズ(腕が約 50 度下がった状態)がレストで、BVH は T ポーズ。
 *   「各リグ自身のレストからの差分」をそのまま渡すと、BVH が T ポーズから腕を下ろす
 *   回転量に、UE 側の A ポーズ分がそのまま上乗せされる。腕が下がりすぎて体の前で
 *   交差するのはこれが原因。
 *
 * 何をするか:
 *   ボーンごとに「子ボーンへ向かう方向」が BVH のレストと一致するよう最小回転をかけ、
 *   親から順に適用する。回転しか変えないので骨の長さ・左右の間隔は保たれる。
 *   こうして得た姿勢を基準にすれば、BVH のボーン方向がそのままモデルに再現される。
 *
 * 子が複数ある腰・胸などは向き合わせが一意に決まらないため対象外にする
 * (これらはリグが違ってもほぼ直立で揃っているため、そのままで問題ない)。
 */
const alignRestToSource = (
  character: LoadedCharacter,
  sourceRig: SourceRig,
): { rest: Map<THREE.Bone, RestOrientation>; aligned: number; maxAngleDeg: number } => {
  const { model } = character

  // 再バインド時に補正が二重に乗らないよう、まず bind ポーズへ戻す
  for (const [bone, rest] of character.rest) {
    bone.quaternion.copy(rest.local)
    bone.position.copy(rest.position)
  }

  // 両リグで「唯一の子」が一致するボーンだけを向き合わせの対象にする
  const sourceChildren = mappedChildCanonicals(sourceRig.byCanonical)
  const targetChildren = mappedChildCanonicals(character.bonesByCanonical)
  const alignTo = new Map<string, string>()
  for (const [canonical, fromSource] of sourceChildren) {
    const fromTarget = targetChildren.get(canonical)
    if (!fromTarget) continue
    const shared = fromSource.filter((child) => fromTarget.includes(child))
    if (shared.length === 1) alignTo.set(canonical, shared[0])
  }

  // 親 → 子の順に処理する(traverse は前順なので親が先に来る)
  const canonicalOf = new Map<THREE.Object3D, string>()
  for (const [canonical, bone] of character.bonesByCanonical) canonicalOf.set(bone, canonical)
  const order: { bone: THREE.Bone; canonical: string }[] = []
  model.traverse((object) => {
    const canonical = canonicalOf.get(object)
    if (canonical && alignTo.has(canonical)) {
      order.push({ bone: object as THREE.Bone, canonical })
    }
  })

  const modelInverse = new THREE.Matrix4()
  const scratch = new THREE.Vector3()
  const from = new THREE.Vector3()
  const to = new THREE.Vector3()
  const swing = new THREE.Quaternion()
  let aligned = 0
  let maxAngle = 0

  for (const { bone, canonical } of order) {
    const childCanonical = alignTo.get(canonical)
    if (!childCanonical) continue
    const targetChild = character.bonesByCanonical.get(childCanonical)
    const source = sourceRig.byCanonical.get(canonical)
    const sourceChild = sourceRig.byCanonical.get(childCanonical)
    if (!targetChild || !source || !sourceChild) continue

    const sourceFrom = sourceRig.restWorldPosition.get(source)
    const sourceTo = sourceRig.restWorldPosition.get(sourceChild)
    if (!sourceFrom || !sourceTo) continue
    to.copy(sourceTo).sub(sourceFrom)
    if (to.lengthSq() < 1e-12) continue
    to.normalize()

    // 親の補正を反映した現在の姿勢で、モデル空間での方向を測る
    model.updateMatrixWorld(true)
    modelInverse.copy(model.matrixWorld).invert()
    from.copy(targetChild.getWorldPosition(scratch).applyMatrix4(modelInverse))
    from.sub(bone.getWorldPosition(scratch).applyMatrix4(modelInverse))
    if (from.lengthSq() < 1e-12) continue
    from.normalize()

    const angle = from.angleTo(to)
    if (angle < 1e-4) continue

    // モデル空間での回転 swing をローカル回転へ変換して適用する
    swing.setFromUnitVectors(from, to)
    const parentWorld = currentWorldQuaternion(bone.parent, model)
    bone.quaternion.premultiply(
      parentWorld.clone().invert().multiply(swing).multiply(parentWorld),
    )
    aligned += 1
    maxAngle = Math.max(maxAngle, angle)
  }

  model.updateMatrixWorld(true)

  // 合わせ込んだ姿勢を転写の基準として採取する
  const rest = new Map<THREE.Bone, RestOrientation>()
  for (const bone of character.rest.keys()) {
    const parentWorld = currentWorldQuaternion(bone.parent, model)
    rest.set(bone, {
      local: bone.quaternion.clone(),
      parentWorld,
      parentWorldInverse: parentWorld.clone().invert(),
    })
  }

  return { rest, aligned, maxAngleDeg: THREE.MathUtils.radToDeg(maxAngle) }
}

/**
 * キャラクター(target)と BVH 骨格(source)を正規ボーン名で結び付ける。
 * BVH を読み込み直すたびに呼ぶ必要がある。
 */
export function bindCharacterToSource(
  character: LoadedCharacter,
  sourceRig: SourceRig,
): void {
  const links: RetargetLink[] = []
  const unmatched: string[] = []

  // モデルのレストを BVH のレスト(T ポーズ)へ合わせ込み、その姿勢を転写の基準にする。
  // これをしないと UE の A ポーズ分が二重に効いて腕が下がりすぎ、体の前で交差する。
  const { rest: alignedRest, aligned, maxAngleDeg } = alignRestToSource(character, sourceRig)

  // 合わせ込み済みレスト姿勢のうちに、モデル空間での最下点を測っておく。
  // root グループのスケール・位置は後で fit されるので、ここで取り除いておく。
  character.model.updateMatrixWorld(true)
  // Seat contact measures the deformed mesh bounds. Those cached boxes must
  // be refreshed after restoring the standing rest pose before measuring its
  // floor offset; otherwise the next bind uses the seated dimensions.
  for (const mesh of collectSkinnedMeshes(character.model)) {
    mesh.skeleton.update()
    mesh.computeBoundingBox()
  }
  const restBounds = new THREE.Box3().setFromObject(character.model)
  const rootScale = character.root.scale.y || 1
  character.groundOffset = Number.isFinite(restBounds.min.y)
    ? (restBounds.min.y - character.root.position.y) / rootScale
    : 0

  // 体の正面が BVH と揃うのに必要な Y 回転を自動算出する。
  const modelInverse = new THREE.Matrix4().copy(character.model.matrixWorld).invert()
  const targetPos = (canonical: string): THREE.Vector3 | null => {
    const bone = character.bonesByCanonical.get(canonical)
    if (!bone) return null
    return bone.getWorldPosition(new THREE.Vector3()).applyMatrix4(modelInverse)
  }
  const sourcePos = (canonical: string): THREE.Vector3 | null => {
    const bone = sourceRig.byCanonical.get(canonical)
    return bone ? sourceRig.restWorldPosition.get(bone) ?? null : null
  }
  const upperKey = ['neck', 'spine2', 'spine1', 'head'].find(
    (key) => character.bonesByCanonical.has(key) && sourceRig.byCanonical.has(key),
  )
  const targetForward =
    upperKey &&
    bodyForward(
      targetPos('hips')!,
      targetPos('leftUpLeg')!,
      targetPos('rightUpLeg')!,
      targetPos(upperKey)!,
    )
  const sourceForward =
    upperKey &&
    bodyForward(
      sourcePos('hips')!,
      sourcePos('leftUpLeg')!,
      sourcePos('rightUpLeg')!,
      sourcePos(upperKey)!,
    )
  if (targetForward && sourceForward) {
    // R_y(θ)·targetForward = sourceForward を満たす θ
    const sin =
      sourceForward.x * targetForward.z - sourceForward.z * targetForward.x
    const cos =
      targetForward.x * sourceForward.x + targetForward.z * sourceForward.z
    character.autoYawDegrees = THREE.MathUtils.radToDeg(Math.atan2(sin, cos))
  }

  // リンクは「親が先」の順に並べる。転写時に親の結果を使って子を解くため。
  const canonicalOfTarget = new Map<THREE.Object3D, string>()
  for (const [canonical, bone] of character.bonesByCanonical) canonicalOfTarget.set(bone, canonical)

  character.model.traverse((object) => {
    const canonical = canonicalOfTarget.get(object)
    if (!canonical) return
    const target = object as THREE.Bone
    const source = sourceRig.byCanonical.get(canonical)
    if (!source) {
      unmatched.push(canonical)
      return
    }
    const targetRest = alignedRest.get(target)
    const sourceRest = sourceRig.rest.get(source)
    if (!targetRest || !sourceRest) return

    links.push({
      canonical,
      target,
      targetRestWorld: targetRest.parentWorld.clone().multiply(targetRest.local),
      source,
      sourceRestWorldInverse: sourceRest.parentWorld
        .clone()
        .multiply(sourceRest.localInverse.clone().invert())
        .invert(),
    })
  })

  character.links = links
  character.sourceRootBone = sourceRig.rootBone
  // 素のレスト位置ではなく「足が床に接する」位置を基準にする(浮き対策)
  character.sourceRootRestPosition = sourceRig.groundedRootPosition.clone()
  character.targetRootBone = character.bonesByCanonical.get('hips') ?? null

  console.log(
    `[characterRig] bound ${links.length} bones ` +
      `(${character.flavor} ← ${sourceRig.flavor}), ` +
      `rest aligned to source T-pose: ${aligned} bones (max ${maxAngleDeg.toFixed(1)} deg), ` +
      `groundOffset=${character.groundOffset.toFixed(4)}, ` +
      `autoYaw=${character.autoYawDegrees.toFixed(1)}deg` +
      (unmatched.length > 0
        ? `, unmatched target bones: ${unmatched.join(', ')}`
        : ''),
  )

  if (links.length < 8) {
    console.warn(
      '[characterRig] 対応付いたボーンが少なすぎます。' +
        'モデルが人型リグでないか、ボーン命名が想定外の可能性があります。' +
        ` model bones = [${character.skinnedMesh.skeleton.bones
          .slice(0, 20)
          .map((b) => b.name)
          .join(', ')}]`,
    )
  }
}

/** テクスチャを持たない素体だけ単色化する(UE 由来のテクスチャ付きモデルは保護する)。 */
const hasTexture = (material: THREE.Material): boolean => {
  const std = material as THREE.MeshStandardMaterial
  return Boolean(std.map || std.emissiveMap || std.normalMap)
}

/** テクスチャなし素体に適用する基本色。 */
export const BODY_COLOR = 0xffffff

export async function loadCharacterModel(
  url: string = CHARACTER_MODEL_URL,
): Promise<LoadedCharacter> {
  const loader = new GLTFLoader()
  const gltf = await loader.loadAsync(url)
  const model = gltf.scene

  const materials: THREE.Material[] = []
  model.traverse((object) => {
    const mesh = object as THREE.SkinnedMesh
    // BVH 由来のポーズで一時的に視錐台外へ出ても消えないようにする
    object.frustumCulled = false
    if ((mesh.isMesh || mesh.isSkinnedMesh) && mesh.material) {
      materials.push(...(Array.isArray(mesh.material) ? mesh.material : [mesh.material]))
    }
  })
  // テクスチャを一切持たない素体モデルだけ白で統一する。
  // テクスチャ付きモデル(UE 書き出しの肌・衣服など)は、髪カードのような単色材質も含めて
  // 作者の指定色をそのまま使う。
  if (!materials.some(hasTexture)) {
    for (const m of materials) {
      const std = m as THREE.MeshStandardMaterial
      if (std.color) std.color.setHex(BODY_COLOR)
    }
  }
  const skinnedMeshes = collectSkinnedMeshes(model)
  if (skinnedMeshes.length === 0) {
    throw new Error(
      'このモデルには SkinnedMesh がありません。スケルトン付き (rigged) の glb を書き出してください。',
    )
  }

  // 主リグ = 人型ボーンが最も多く対応付いたスケルトン(体)。
  // 顔メッシュは spine_04〜head しか持たないので自然と副リグになる。
  let mesh: THREE.SkinnedMesh = skinnedMeshes[0]
  let bestMapped = -1
  for (const candidate of skinnedMeshes) {
    const mapped = mapBonesByCanonicalName(candidate.skeleton.bones).size
    if (mapped > bestMapped) {
      bestMapped = mapped
      mesh = candidate
    }
  }

  // レスト(バインド)ポーズを採取する。ポーズ転写前に行う必要がある。
  model.updateMatrixWorld(true)
  const rest: LoadedCharacter['rest'] = new Map()
  for (const bone of mesh.skeleton.bones) {
    const parentWorld = restWorldQuaternion(bone.parent, model)
    rest.set(bone, {
      local: bone.quaternion.clone(),
      parentWorld,
      parentWorldInverse: parentWorld.clone().invert(),
      position: bone.position.clone(),
    })
  }

  const bonesByCanonical = mapBonesByCanonicalName(mesh.skeleton.bones)
  const flavor = detectRigFlavor(mesh.skeleton.bones.map(originalBoneName))

  // hips のローカル単位 → モデル単位の換算係数。
  // ラッパー(root グループ)へ入れる前のこの時点でのみ、モデル空間のスケールを正しく取れる。
  let targetRootUnitScale = 1
  const hipsParent = bonesByCanonical.get('hips')?.parent
  if (hipsParent) {
    const parentScale = new THREE.Vector3()
    hipsParent.matrixWorld.decompose(
      new THREE.Vector3(),
      new THREE.Quaternion(),
      parentScale,
    )
    if (parentScale.x > 1e-9) targetRootUnitScale = parentScale.x
  }

  // 副リグ(顔・服など別スケルトンの SkinnedMesh)を主リグの骨格へ統合する。
  const { secondaryRigs, mergedBones, graftedBones } = mergeSecondaryRigs(skinnedMeshes, mesh)
  model.updateMatrixWorld(true)

  // リグ付きモデルに同梱された非スキンメッシュ(UE のグルームから書き出した髪カード等)は
  // head ボーンに取り付けて追従させる。ボーン配下でない静的メッシュのみ対象で、
  // 通常のリグ付きモデル(Mixamo 等)には静的メッシュが無いので影響しない。
  let attachedStatic = 0
  const headBone = bonesByCanonical.get('head')
  if (headBone) {
    const statics: THREE.Mesh[] = []
    model.traverse((object) => {
      const m = object as THREE.Mesh
      if (!m.isMesh || (m as THREE.SkinnedMesh).isSkinnedMesh) return
      let p: THREE.Object3D | null = m.parent
      let underBone = false
      while (p) {
        if ((p as THREE.Bone).isBone) underBone = true
        p = p.parent
      }
      if (!underBone) statics.push(m)
    })
    for (const m of statics) {
      headBone.attach(m) // ワールド変換を維持したまま親を差し替える(レストポーズ時に呼ぶこと)
      attachedStatic += 1
    }
  }

  const bbox = new THREE.Box3().setFromObject(model)
  const naturalHeight = Math.max(bbox.max.y - bbox.min.y, 1e-3)

  const root = new THREE.Group()
  root.name = 'CharacterRoot'
  root.add(model)

  console.log(
    `[characterRig] loaded "${url}": flavor=${flavor}, ` +
      `bones=${mesh.skeleton.bones.length}, mapped=${bonesByCanonical.size}, ` +
      `skinnedMeshes=${skinnedMeshes.length}, secondaryRigs=${secondaryRigs}, ` +
      `mergedBones=${mergedBones}, graftedBones=${graftedBones}, headAttached=${attachedStatic}, ` +
      `naturalHeight=${naturalHeight.toFixed(3)}`,
  )

  return {
    root,
    model,
    skinnedMesh: mesh,
    naturalHeight,
    tuning: { ...CHARACTER_TUNING },
    flavor,
    bonesByCanonical,
    rest,
    links: [],
    sourceRootBone: null,
    sourceRootRestPosition: null,
    groundOffset: 0,
    autoYawDegrees: 0,
    targetRootBone: null,
    targetRootUnitScale,
    transferRootMotion: true,
  }
}

// 毎フレーム確保を避けるための作業用オブジェクト
const _sourceWorld = new THREE.Quaternion()
const _parentWorld = new THREE.Quaternion()
const _deltaWorld = new THREE.Quaternion()
const _rootDelta = new THREE.Vector3()

/**
 * BVH 骨格の現在ポーズをキャラクターへ転写する。
 *
 * 各ボーンについて
 *   1. 転写元の「レストからのローカル差分」を求め、
 *   2. 転写元レスト姿勢のワールド基準へ持ち上げ、
 *   3. 転写先レスト姿勢のローカル基準へ落とし込み、転写先レストに掛ける。
 * という手順を踏むため、両者のボーン軸・レストポーズが違っていても成立する。
 *
 * 骨の長さや左右間隔は転写先のものが保たれるので、モデル本来のプロポーションは崩れない。
 */
export function retargetCharacter(character: LoadedCharacter): void {
  const { model, sourceRootBone, targetRootBone } = character
  if (sourceRootBone) {
    // links は親が先の順に並んでいるので、親を解いた結果をそのまま子の計算に使える。
    for (const link of character.links) {
      // 転写元の「レストからのワールド回転差分」。祖先の回転も含んだ絶対量なので、
      // 片方のリグにしか無い中間ボーン(Neck1 や spine_04 など)があっても取りこぼさない。
      accumulateWorldQuaternion(_sourceWorld, link.source, sourceRootBone)
      _deltaWorld.copy(_sourceWorld).multiply(link.sourceRestWorldInverse)

      accumulateWorldQuaternion(_parentWorld, link.target.parent, model)
      link.target.quaternion
        .copy(_parentWorld)
        .invert()
        .multiply(_deltaWorld)
        .multiply(link.targetRestWorld)
        // 四元数の積は長さがわずかに 1 からずれる。UE リグは階層が深く
        // 毎フレーム掛け続けるため、ここで正規化してドリフトを断つ。
        .normalize()
    }
  }

  // ルート移動の転写。座る/立ち上がる動作で腰の高さが変わるため、
  // これが無いとキャラが宙に浮いたまま脚だけ曲がる。
  if (!character.transferRootMotion || !sourceRootBone || !targetRootBone) return

  const targetRest = character.rest.get(targetRootBone)
  if (!targetRest) return

  // BVH 単位 → hips のローカル単位へ換算する。
  // シーン上での見た目の長さ = ローカル値 × 親のモデル内スケール × root グループのスケール
  // なので、その積で割り戻す。
  const sceneScale = (character.root.scale.x || 1) * character.targetRootUnitScale
  if (sceneScale < 1e-9) return

  _rootDelta
    .copy(sourceRootBone.position)
    .sub(character.sourceRootRestPosition ?? sourceRootBone.position)
    .divideScalar(sceneScale)

  targetRootBone.position.copy(targetRest.position).add(_rootDelta)
}

/** モデルの GPU リソースを解放する。 */
export function disposeCharacter(character: LoadedCharacter): void {
  character.root.traverse((object) => {
    const mesh = object as THREE.Mesh
    if (!mesh.isMesh) return
    mesh.geometry?.dispose?.()
    const material = mesh.material as THREE.Material | THREE.Material[]
    if (Array.isArray(material)) material.forEach((m) => m.dispose())
    else material?.dispose?.()
  })
  character.root.parent?.remove(character.root)
}

export { normalizeBoneName }
