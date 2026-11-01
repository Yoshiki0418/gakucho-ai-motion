'use client'

/**
 * 任意の人型リグ(Mixamo / Unreal Engine / VRM / Blender 等)のボーン名を
 * 共通の「正規名(canonical name)」へ寄せるための対応表。
 *
 * 目的:
 *   本プロジェクトの BVH(Mixamo 命名)で生成したモーションを、
 *   命名規則がまったく異なる UE スケルトン(pelvis / upperarm_l / calf_l …)へも
 *   そのまま転写できるようにする。
 *
 * 方式:
 *   ボーン名を正規化(名前空間除去 + 小文字化 + 記号除去)したうえで、
 *   完全一致のエイリアス表を引く。部分一致は誤爆(LeftArm と LeftForeArm 等)の
 *   温床になるため使わない。
 */

/** "mixamorig:Hips" / "Armature|Hips" / "spine_01" → "hips" / "hips" / "spine01" */
export const normalizeBoneName = (name: string): string =>
  name
    // 名前空間 ("mixamorig:", "Armature|", "Root|") を落とす
    .replace(/^.*[:|]/, '')
    .replace(/^mixamorig/i, '')
    .toLowerCase()
    .replace(/[^a-z0-9]/g, '')

/**
 * 体幹(左右を持たない)ボーン。
 * Mixamo は Spine/Spine1…、UE は spine_01/spine_02… と採番がずれるため、
 * 正規化後の "spine1"(Mixamo) と "spine01"(UE) が別文字列になることを利用して両方を吸収する。
 */
const CENTER_ALIASES: Record<string, string[]> = {
  hips: ['hips', 'hip', 'pelvis'],
  spine: ['spine', 'spine01'],
  spine1: ['spine1', 'spine02', 'chest'],
  spine2: ['spine2', 'spine03', 'upperchest'],
  spine3: ['spine3', 'spine04'],
  spine4: ['spine4', 'spine05'],
  neck: ['neck', 'neck01'],
  neck1: ['neck1', 'neck02'],
  head: ['head'],
}

/**
 * 左右を持つボーン。値は「側を示す接頭/接尾辞を除いた基底名」。
 * 例: upperArm の基底 'upperarm' から "upperarm_l"(UE) を、
 *     基底 'arm' から "LeftArm"(Mixamo) を、それぞれ拾う。
 */
const SIDE_ALIASES: Record<string, string[]> = {
  Shoulder: ['shoulder', 'clavicle'],
  UpperArm: ['arm', 'upperarm', 'armupper'],
  LowerArm: ['forearm', 'lowerarm', 'elbow'],
  Hand: ['hand', 'wrist'],
  UpLeg: ['upleg', 'thigh', 'upperleg'],
  LowerLeg: ['leg', 'calf', 'lowerleg', 'shin', 'knee'],
  Foot: ['foot', 'ankle'],
  Toe: ['toebase', 'ball', 'toe'],
}

/** 指: Mixamo "LeftHandThumb1" / UE "thumb_01_l" の双方を拾う。 */
const FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky'] as const
for (const finger of FINGER_NAMES) {
  for (const joint of [1, 2, 3]) {
    SIDE_ALIASES[`${finger[0].toUpperCase()}${finger.slice(1)}${joint}`] = [
      `hand${finger}${joint}`, // Mixamo: LeftHandThumb1
      `${finger}${joint}`,
      `${finger}0${joint}`, // UE: thumb_01_l
    ]
  }
}

/** 正規化済みボーン名 → 正規名 の逆引き表。 */
const CANONICAL_BY_FORM = new Map<string, string>()

const register = (form: string, canonical: string) => {
  // 先に登録されたものを優先し、意図しない上書きを防ぐ
  if (!CANONICAL_BY_FORM.has(form)) CANONICAL_BY_FORM.set(form, canonical)
}

for (const [canonical, forms] of Object.entries(CENTER_ALIASES)) {
  for (const form of forms) register(form, canonical)
}

for (const [part, bases] of Object.entries(SIDE_ALIASES)) {
  for (const [sideKey, prefixes] of [
    ['left', ['left', 'l']],
    ['right', ['right', 'r']],
  ] as const) {
    const canonical = `${sideKey}${part}`
    for (const base of bases) {
      for (const marker of prefixes) {
        register(`${marker}${base}`, canonical) // leftarm / larm
        register(`${base}${marker}`, canonical) // armleft / arml (UE: upperarm_l)
      }
    }
  }
}

/** 単一ボーン名の正規名を返す。人型ボーンとして解釈できない場合は null。 */
export const canonicalBoneName = (name: string): string | null =>
  CANONICAL_BY_FORM.get(normalizeBoneName(name)) ?? null

type NamedObject = { name: string; userData?: { name?: unknown } }

/**
 * glTF に書かれていた元のボーン名を返す。
 * three.js の GLTFLoader は同名ノードが複数あると "pelvis_1" のように連番を付けて
 * `name` を書き換えるが、元の名前は `userData.name` に残る。UE アクター書き出しのように
 * 体・顔・服が同じ骨格名を持つ glb では、この元名で照合しないと対応が取れない。
 */
export const originalBoneName = (bone: NamedObject): string =>
  typeof bone.userData?.name === 'string' && bone.userData.name.length > 0
    ? bone.userData.name
    : bone.name

/**
 * ボーン配列から「正規名 → ボーン」の対応を作る。
 * 同じ正規名に複数ボーンが当たった場合は最初に見つかったものを採用する。
 */
export function mapBonesByCanonicalName<T extends NamedObject>(
  bones: readonly T[],
): Map<string, T> {
  const byCanonical = new Map<string, T>()
  for (const bone of bones) {
    const canonical = canonicalBoneName(originalBoneName(bone))
    if (canonical && !byCanonical.has(canonical)) byCanonical.set(canonical, bone)
  }
  return byCanonical
}

/** ざっくりしたリグ種別の判定。ログ表示・デバッグ用。 */
export function detectRigFlavor(boneNames: readonly string[]): string {
  const forms = new Set(boneNames.map(normalizeBoneName))
  if (boneNames.some((n) => /^mixamorig/i.test(n))) return 'mixamo'
  if (forms.has('pelvis') && forms.has('upperarml')) return 'unreal'
  if (forms.has('upperchest') && forms.has('leftupperarm')) return 'vrm'
  if (forms.has('hips') && forms.has('leftarm')) return 'mixamo-like'
  return 'unknown'
}
