'use client'

import React, { useRef, useEffect, useState, useCallback } from 'react'
import { createPortal } from 'react-dom'
import * as THREE from 'three'
import { BVHLoader } from 'three/examples/jsm/loaders/BVHLoader.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { DEFAULT_BVH } from './defaultBvh'
import { bvhDurationSeconds, type ScheduledMotion } from '@/features/text-chat/hooks/useTextChat'
import {
  CHARACTER_MODEL_URL,
  bindCharacterToSource,
  createSourceRig,
  disposeCharacter,
  loadCharacterModel,
  retargetCharacter,
  type LoadedCharacter,
  type SourceRig,
} from './characterRig'
import {
  createSceneChair,
  disposeSceneChair,
  fitSceneChairToSeatedPose,
  placeSceneChair,
  type SceneChair,
} from './sceneChair'

import { createHomeEnvironment, HOME_ENVIRONMENT_CAMERA, type HomeEnvironment } from './homeEnvironment'
import { createHomeRendering, type HomeRendering } from './homeRendering'
import { createRoomNavigation, isRoomPointWalkable, planChairApproach, planRoomPath, type RoomNavigation, type RoomChairTarget } from './roomNavigation'
import { loadRoomMotionPrimitives, normalizeRoomBvh, parseRoomMotionPrimitive, type RoomMotionPrimitives } from './roomMotionPrimitives'
import { RoomTaskController, type RoomTaskBridge, type RoomTaskSnapshot } from './roomTaskController'
import { createRoomDestinations, type RoomDestination } from './roomDestinations'
import { RoomWalkController } from './roomWalkController'
import { createRoomScene, createSceneRoomNavigation, parseRoomScene, type RoomSceneDefinition, type RoomScene } from './roomScene'
import { ROOM_SCENE_EXAMPLES } from './roomSceneExamples'
import { applySeatContactTargets, captureStandingFootContacts, createSeatContactTargets, groundRoomFeet, type RoomFootContactReference } from './roomPoseContacts'

type ViewMode = 'skeleton' | 'robot' | 'character'
const PRESIDENT_AVATAR = { label: '学長（スーツ）', url: '/avatars/gakucho_v3_suit.glb' }
type EnvironmentMode = 'home' | 'laboratory'
type EnvironmentCamera = { position: THREE.Vector3; target: THREE.Vector3; fov: number }
const HOME_CAMERA_POSITION = new THREE.Vector3(...HOME_ENVIRONMENT_CAMERA.position)
const HOME_CAMERA_TARGET = new THREE.Vector3(...HOME_ENVIRONMENT_CAMERA.target)
const readRoomNavigation = (root: THREE.Object3D) => root.userData.roomSceneDefinition
  ? createSceneRoomNavigation(root) : createRoomNavigation(root)

interface BvhViewerProps {
  /** 設定画面の表示先。未指定ならビューア内で操作できる。 */
  settingsHost?: HTMLDivElement | null
  /** BVH text data injected externally (e.g. from motion generation) */
  bvhText?: string | null
  /** Text prompt that produced the BVH. Used to decide whether the final pose should persist. */
  motionPrompt?: string | null
  /**
   * 発話タイムライン上にスケジュール済みのモーション。
   * `startAt` は `getAudioTime()` と同じ時計(AudioContext)の時刻。
   * これが渡されているときは、届いた順ではなく指定時刻に再生する。
   */
  scheduledMotions?: ScheduledMotion[]
  /**
   * 発話タイムラインの現在時刻(秒)。音声が鳴っていなければ null。
   *
   * 従来はモーションを `THREE.Clock` で回していたため、音声(AudioContext)とは
   * 別の時計になり、同時に始めても必ずドリフトしていた。これを渡すことで
   * 音声と同じ時計で駆動する。
   */
  getAudioTime?: () => number | null
  /** アバターの姿勢が変わったら知らせる(会話画面が層構成方式を使えるかの判断に使う)。 */
  onPostureChange?: (posture: PostureKind) => void
}

export type PostureKind = 'standing' | 'chair_sitting' | 'floor_sitting' | 'sitting'

/** 層構成方式の動作を切り替えるとき、直前の見た目から滑らかにつなぐ時間(ms)。 */
const LAYERED_CROSSFADE_MS = 250

type CharacterCrossfade = {
  quaternions: Map<THREE.Bone, THREE.Quaternion>
  rootBone: THREE.Bone | null
  rootPosition: THREE.Vector3 | null
  startedAt: number
}
const _crossfadeQuaternion = new THREE.Quaternion()
const _crossfadePosition = new THREE.Vector3()

type MotionEndBehavior =
  | 'return-to-standing-idle'
  | 'hold-final-as-posture'
  | 'return-to-persistent-posture'

type PersistentBonePose = {
  position: THREE.Vector3
  quaternion: THREE.Quaternion
}

type PersistentPose = {
  kind: PostureKind
  bones: Map<string, PersistentBonePose>
}

type IdleBoneRole =
  | 'root'
  | 'hips'
  | 'spine'
  | 'chest'
  | 'neck'
  | 'head'
  | 'shoulder'
  | 'upperArm'
  | 'lowerArm'
  | 'hand'
  | 'thigh'
  | 'shin'
  | 'foot'
  | 'other'

type IdleBoneState = {
  bone: THREE.Bone
  role: IdleBoneRole
  side: -1 | 0 | 1
  restPosition: THREE.Vector3
  restQuaternion: THREE.Quaternion
  returnPosition: THREE.Vector3
  returnQuaternion: THREE.Quaternion
  targetPosition: THREE.Vector3
  targetQuaternion: THREE.Quaternion
}

type AvatarAnimationState = {
  rootBone: THREE.Bone
  bones: IdleBoneState[]
  phase: 'motion' | 'returning' | 'idle'
  poseKind: PostureKind
  idleTime: number
  returnElapsed: number
  returnDuration: number
  scale: number
  returnRootBaseline?: { from: number; to: number }
}

const normalizeMotionPrompt = (prompt: string) =>
  prompt
    .toLowerCase()
    .replace(/\s+/g, '')
    .replace(/[。、，,.!！?？]/g, '')

const classifyMotionPrompt = (
  prompt: string | null | undefined,
): { intent: 'stand' | 'sit' | 'other'; posture: PostureKind } => {
  const normalized = normalizeMotionPrompt(prompt ?? '')

  // 「立っている状態から」はT2M安定化用の前置きであり、起立指示ではない。
  const withoutGenerationPrefix = normalized.replace(/立っている状態から/g, '')

  if (/立ち上が|起き上が|起立|立つ|立って/.test(withoutGenerationPrefix)) {
    return { intent: 'stand', posture: 'standing' }
  }

  if (/椅子|イス|いす|腰掛/.test(normalized) && /座|腰掛/.test(normalized)) {
    return { intent: 'sit', posture: 'chair_sitting' }
  }

  if (/床|地面|あぐら|胡座|正座/.test(normalized) && /座|すわ/.test(normalized)) {
    return { intent: 'sit', posture: 'floor_sitting' }
  }

  if (/座|すわ/.test(normalized)) {
    return { intent: 'sit', posture: 'sitting' }
  }

  return { intent: 'other', posture: 'standing' }
}

/** 正規化後の骨格の全高(シーン単位)。既存の Custom-T2M-Lab の規模に合わせている。 */
const NORMALIZED_SKELETON_HEIGHT = 1.8

/**
 * BVH のスケールを供給元によらず一定にそろえる。
 *
 * ボーンのオフセットと、ルートの位置トラックを同じ係数で拡縮する。
 * 回転は単位に依存しないので触らない。
 */
const normalizeBvhScale = (
  result: { skeleton: THREE.Skeleton; clip: THREE.AnimationClip },
  targetHeight = NORMALIZED_SKELETON_HEIGHT,
) => {
  const bones = result.skeleton.bones
  if (bones.length === 0) return

  bones[0].updateMatrixWorld(true)
  const bounds = new THREE.Box3()
  const worldPosition = new THREE.Vector3()
  for (const bone of bones) {
    bone.updateWorldMatrix(true, false)
    bounds.expandByPoint(bone.getWorldPosition(worldPosition))
  }
  const height = bounds.max.y - bounds.min.y
  if (!Number.isFinite(height) || height <= 1e-6) return

  const factor = targetHeight / height
  // 1割程度のずれで拡縮しても意味がないので、明確に単位が違うときだけ直す
  if (factor > 0.5 && factor < 2) return

  for (const bone of bones) bone.position.multiplyScalar(factor)
  for (const track of result.clip.tracks) {
    if (!track.name.endsWith('.position')) continue
    for (let i = 0; i < track.values.length; i++) track.values[i] *= factor
  }
  console.log(
    `[BvhViewer] BVH スケールを正規化: 高さ ${height.toFixed(2)} → ` +
    `${targetHeight.toFixed(2)} (係数 ${factor.toFixed(4)})`
  )
}

/**
 * BVH の座標系を Y-up / +X 左 へそろえる。
 *
 * BVH は Y-up が慣習だが、供給元によっては別の軸で書き出される。
 * そのまま読むと骨格が横倒しになり、リターゲットの差分も別の軸で
 * 渡ることになって姿勢が壊れる。
 *
 * 現在つないでいる Custom-T2M-Lab / SentiAvatar はどちらも Y-up なので
 * この関数は no-op になる（安全網として残してある）。
 * SentiAvatar については当初 Z-up と誤診したが、実際はレスト姿勢が
 * 壊れていたために「腰→頭」の測定が (1,0,0) になっていただけだった。
 * 詳細は docs/sentiavatar-eval.md を参照。
 *
 * レスト姿勢の「腰→頭(上)」と「右腿→左腿(左)」から実際の体の向きを測り、
 * それを標準の軸へ移す回転を全体へ適用する。
 * オフセットと位置トラックは回転させ、回転トラックは共役変換する。
 */
const reorientBvhAxes = (result: {
  skeleton: THREE.Skeleton
  clip: THREE.AnimationClip
}) => {
  const bones = result.skeleton.bones
  if (bones.length === 0) return

  const byName = (pattern: RegExp) => bones.find((b) => pattern.test(b.name))
  const hips = byName(/^(hips?|pelvis)$/i)
  const head = byName(/^head$/i)
  const legL = byName(/^(leftupleg|thigh_l)$/i)
  const legR = byName(/^(rightupleg|thigh_r)$/i)
  if (!hips || !head || !legL || !legR) return

  bones[0].updateMatrixWorld(true)
  const world = (bone: THREE.Bone) => {
    bone.updateWorldMatrix(true, false)
    return bone.getWorldPosition(new THREE.Vector3())
  }
  const up = world(head).sub(world(hips))
  const left = world(legL).sub(world(legR))
  if (up.lengthSq() < 1e-9 || left.lengthSq() < 1e-9) return
  up.normalize()
  left.normalize()

  // すでに標準の向きなら何もしない
  if (Math.abs(up.y) > 0.95 && Math.abs(left.x) > 0.95) return

  // left を基準に up を直交化し、正規直交基底を作る
  const axisX = left.clone()
  const axisY = up.clone().addScaledVector(axisX, -up.dot(axisX)).normalize()
  const axisZ = new THREE.Vector3().crossVectors(axisX, axisY)

  // この基底を world 軸へ移す回転(基底行列の転置 = 逆回転)
  const basis = new THREE.Matrix4().makeBasis(axisX, axisY, axisZ)
  const rotation = new THREE.Quaternion().setFromRotationMatrix(basis).invert()
  const inverse = rotation.clone().invert()

  for (const bone of bones) bone.position.applyQuaternion(rotation)

  const value = new THREE.Vector3()
  const quaternion = new THREE.Quaternion()
  for (const track of result.clip.tracks) {
    if (track.name.endsWith('.position')) {
      for (let i = 0; i < track.values.length; i += 3) {
        value.set(track.values[i], track.values[i + 1], track.values[i + 2])
        value.applyQuaternion(rotation)
        track.values[i] = value.x
        track.values[i + 1] = value.y
        track.values[i + 2] = value.z
      }
    } else if (track.name.endsWith('.quaternion')) {
      // 旧座標系で表された回転を新座標系で表し直す
      for (let i = 0; i < track.values.length; i += 4) {
        quaternion.set(
          track.values[i], track.values[i + 1], track.values[i + 2], track.values[i + 3],
        )
        quaternion.premultiply(rotation).multiply(inverse)
        track.values[i] = quaternion.x
        track.values[i + 1] = quaternion.y
        track.values[i + 2] = quaternion.z
        track.values[i + 3] = quaternion.w
      }
    }
  }

  console.log(
    `[BvhViewer] BVH 座標系を変換: up=(${up.x.toFixed(2)},${up.y.toFixed(2)},${up.z.toFixed(2)}) ` +
    `left=(${left.x.toFixed(2)},${left.y.toFixed(2)},${left.z.toFixed(2)}) → Y-up`
  )
}

export const BvhViewer: React.FC<BvhViewerProps> = ({
  settingsHost,
  bvhText,
  motionPrompt,
  scheduledMotions,
  getAudioTime,
  onPostureChange,
}) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const homeEnvironmentRef = useRef<HomeEnvironment | null>(null)
  const importedRoomRef = useRef<RoomScene | null>(null)
  const importedRoomCameraRef = useRef<EnvironmentCamera | null>(null)
  const roomLoadRequestRef = useRef(0)
  const [roomSceneChoice, setRoomSceneChoice] = useState('home')
  const [roomSceneName, setRoomSceneName] = useState('リビング・ダイニング・キッチン')
  const [roomSceneLoading, setRoomSceneLoading] = useState(false)
  const [roomSceneError, setRoomSceneError] = useState('')
  const roomNavigationRef = useRef<RoomNavigation | null>(null)
  const roomPrimitivesRef = useRef<RoomMotionPrimitives | null>(null)
  const roomTaskRef = useRef<RoomTaskController | RoomWalkController | null>(null)
  const requestRoomWalkRef = useRef<((destinationId: string) => boolean) | null>(null)
  const pendingRoomDestinationRef = useRef<string | null>(null)
  const [roomDestinations, setRoomDestinations] = useState<RoomDestination[]>([])
  const roomDestinationsRef = useRef<RoomDestination[]>([])
  const [roomDestinationId, setRoomDestinationId] = useState('kitchen')
  const startRoomTaskRef = useRef<(() => void) | null>(null)
  const resumeRoomStandingRef = useRef<(() => boolean) | null>(null)
  const roomTaskContextRef = useRef<{ controller: RoomTaskController; source: SourceRig; root: THREE.Bone; state: AvatarAnimationState; mixer: THREE.AnimationMixer } | null>(null)
  const roomMotionAnchorRef = useRef(false)
  const roomPlacementYawRef = useRef(0)
  const roomCharacterPivotRef = useRef<THREE.Vector3 | null>(null)
  const roomFixedSeatRef = useRef<RoomChairTarget | null>(null)
  const roomFootReferencesRef = useRef<{ source: RoomFootContactReference | null; character: RoomFootContactReference | null }>({ source: null, character: null })
  const roomStandingPoseRef = useRef<Map<string, PersistentBonePose>>(new Map())
  const [roomTask, setRoomTask] = useState<RoomTaskSnapshot | null>(null)
  const [roomChairs, setRoomChairs] = useState<RoomChairTarget[]>([])
  const [roomChairId, setRoomChairId] = useState('dining-chair-aisle-front')
  const [roomTaskReady, setRoomTaskReady] = useState(false)
  const [roomTaskError, setRoomTaskError] = useState('')
  const [showRoomRoute, setShowRoomRoute] = useState(true)
  const homeRenderingRef = useRef<HomeRendering | null>(null)
  const laboratoryEnvironmentRef = useRef<THREE.Group | null>(null)
  const environmentModeRef = useRef<EnvironmentMode>('home')
  const environmentCamerasRef = useRef<Partial<Record<EnvironmentMode, EnvironmentCamera>>>({})
  const environmentInitializedRef = useRef(false)
  const laboratoryDefaultCameraRef = useRef<EnvironmentCamera | null>(null)
  const motionSpaceRef = useRef<THREE.Group | null>(null)
  const sourceMotionSpaceRef = useRef<THREE.Group | null>(null)
  const chairRef = useRef<SceneChair | null>(null)
  const chairWasFittedRef = useRef(false)
  const chairStandingRootRef = useRef<{ y: number; height: number } | null>(null)
  const characterFloorOffsetRef = useRef(0)
  const contactTransitionRef = useRef<{ from: number; to: number; duration: number } | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const mixerRef = useRef<THREE.AnimationMixer | null>(null)
  const clockRef = useRef(new THREE.Clock())
  // 音声タイムラインをマスタークロックにするための前回時刻。
  // null のときは音声が鳴っていないので THREE.Clock にフォールバックする。
  const lastAudioTimeRef = useRef<number | null>(null)
  // 再生待ちのスケジュール済みモーションと、適用済みの id
  const motionQueueRef = useRef<ScheduledMotion[]>([])
  const appliedMotionIdsRef = useRef<Set<string>>(new Set())
  // 再生中のスケジュール済み動作(層構成方式の差し替え判定に使う)
  const activeScheduledRef = useRef<ScheduledMotion | null>(null)
  // 層構成方式の動作を切り替えた直後の、見た目のつなぎ
  const crossfadeRef = useRef<CharacterCrossfade | null>(null)
  const getAudioTimeRef = useRef<(() => number | null) | undefined>(undefined)
  // animate ループは初期化時のクロージャなので、最新の loadBvh を ref 越しに呼ぶ
  const loadBvhRef = useRef<
    ((text: string, name: string, prompt?: string | null, segmentDuration?: number, startOffset?: number) => void)
    | undefined
  >(undefined)
  const skeletonHelperRef = useRef<THREE.SkeletonHelper | null>(null)
  const animFrameRef = useRef<number>(0)
  const controlsRef = useRef<OrbitControls | null>(null)
  const activeActionRef = useRef<THREE.AnimationAction | null>(null)
  const avatarAnimationRef = useRef<AvatarAnimationState | null>(null)
  const isPlayingRef = useRef(true)
  const persistentPoseRef = useRef<PersistentPose>({
    kind: 'standing',
    bones: new Map(),
  })
  const currentMotionEndBehaviorRef = useRef<MotionEndBehavior>('return-to-standing-idle')
  const pendingPostureKindRef = useRef<PostureKind>('standing')

  const [isPlaying, setIsPlaying] = useState(true)
  const [environmentMode, setEnvironmentMode] = useState<EnvironmentMode>('home')
  const [chairVisibility, setChairVisibility] = useState({ home: true, laboratory: false })
  const showChair = chairVisibility[environmentMode]
  const [hasFile, setHasFile] = useState(true)
  const [fileName, setFileName] = useState('Neutral Idle (Default)')
  const [viewMode, setViewMode] = useState<ViewMode>('character')
  // loadBvh / buildRobotRig / animate 内で常に最新の viewMode を参照するための ref
  const viewModeRef = useRef<ViewMode>(viewMode)
  // 外部キャラクターモデル(retarget 表示用)と、転写元の BVH スケルトン
  const characterRef = useRef<LoadedCharacter | null>(null)
  const sourceRigRef = useRef<SourceRig | null>(null)
  const retargetErroredRef = useRef(false)
  // 選択中のアバター(glb)の URL。差し替え時に再ロードする。
  const [avatarUrl, setAvatarUrl] = useState(PRESIDENT_AVATAR.url)
  const [avatarOptions, setAvatarOptions] = useState([PRESIDENT_AVATAR])
  const [avatarStatus, setAvatarStatus] = useState<string>('読込中…')
  const [avatarYaw, setAvatarYaw] = useState(0)
  const avatarYawRef = useRef(0)
  const [sceneReady, setSceneReady] = useState(false)
  // ローカル .glb を選んだときの blob URL。差し替え時に revoke する。
  const objectUrlRef = useRef<string | null>(null)
  // カメラの auto-fit は最初のモーション読込時だけ行い、以降はユーザーの視点を維持する
  const hasFittedCameraRef = useRef(false)
  useEffect(() => {
    viewModeRef.current = viewMode
  }, [viewMode])

  useEffect(() => {
    isPlayingRef.current = isPlaying
  }, [isPlaying])

  const createAvatarAnimationState = useCallback(
    (rootBone: THREE.Bone, phase: AvatarAnimationState['phase']) => {
      const bones: IdleBoneState[] = []
      let totalBoneLength = 0
      let boneLengthCount = 0

      rootBone.traverse((object) => {
        if (!(object as THREE.Bone).isBone) return
        const bone = object as THREE.Bone
        const name = bone.name.toLowerCase()
        let role: IdleBoneRole = 'other'
        if (bone === rootBone) role = 'root'
        else if (/hips?|pelvis/.test(name)) role = 'hips'
        else if (/head/.test(name)) role = 'head'
        else if (/neck/.test(name)) role = 'neck'
        else if (/forearm|lowerarm/.test(name)) role = 'lowerArm'
        else if (/(left|right)(upper)?arm$/.test(name)) role = 'upperArm'
        else if (/(left|right)(hand|wrist)$/.test(name)) role = 'hand'
        else if (/upleg|thigh/.test(name)) role = 'thigh'
        else if (/(left|right)(lowerleg|leg|shin)$/.test(name)) role = 'shin'
        else if (/(left|right)(foot|ankle)$/.test(name)) role = 'foot'
        else if (/chest|upperback|spine[ _-]?[34]/.test(name)) role = 'chest'
        else if (/spine/.test(name)) role = 'spine'
        else if (/shoulder|clavicle/.test(name)) role = 'shoulder'

        const side: -1 | 0 | 1 = /left|^l[_-]/.test(name)
          ? -1
          : /right|^r[_-]/.test(name)
            ? 1
            : 0

        if (bone !== rootBone && bone.position.lengthSq() > 1e-6) {
          totalBoneLength += bone.position.length()
          boneLengthCount++
        }

        const neutralOffset = new THREE.Euler()
        switch (role) {
          case 'root':
            neutralOffset.set(THREE.MathUtils.degToRad(-1.5), 0, 0)
            break
          case 'hips':
            neutralOffset.set(THREE.MathUtils.degToRad(1.5), 0, 0)
            break
          case 'chest':
            neutralOffset.set(THREE.MathUtils.degToRad(-1.2), 0, 0)
            break
          case 'shoulder':
            neutralOffset.set(
              0,
              THREE.MathUtils.degToRad(side * 2),
              THREE.MathUtils.degToRad(side * 3),
            )
            break
          case 'upperArm':
            neutralOffset.set(
              THREE.MathUtils.degToRad(2),
              THREE.MathUtils.degToRad(side * 5),
              THREE.MathUtils.degToRad(side * 77),
            )
            break
          case 'lowerArm':
            neutralOffset.set(
              0,
              THREE.MathUtils.degToRad(side * 11),
              THREE.MathUtils.degToRad(side * 3),
            )
            break
          case 'hand':
            neutralOffset.set(
              THREE.MathUtils.degToRad(-2),
              THREE.MathUtils.degToRad(side * 3),
              0,
            )
            break
          case 'thigh':
            neutralOffset.set(THREE.MathUtils.degToRad(-3), 0, 0)
            break
          case 'shin':
            neutralOffset.set(THREE.MathUtils.degToRad(5), 0, 0)
            break
          case 'foot':
            neutralOffset.set(THREE.MathUtils.degToRad(-2), 0, 0)
            break
          default:
            break
        }

        const neutralQuaternion = bone.quaternion
          .clone()
          .multiply(new THREE.Quaternion().setFromEuler(neutralOffset))
        const restPosition = bone.position.clone()

        bones.push({
          bone,
          role,
          side,
          restPosition,
          restQuaternion: neutralQuaternion,
          returnPosition: bone.position.clone(),
          returnQuaternion: bone.quaternion.clone(),
          targetPosition: bone.position.clone(),
          targetQuaternion: neutralQuaternion.clone(),
        })
      })

      const averageBoneLength =
        boneLengthCount > 0 ? totalBoneLength / boneLengthCount : 1

      // ニュートラル姿勢を一度仮適用し、ロボット外装を含む最下点を測る。
      // BVHのroot原点は腰にあるため、単純にゼロへ戻すと全身が地面へ沈む。
      for (const item of bones) {
        item.bone.position.copy(item.restPosition)
        item.bone.quaternion.copy(item.restQuaternion)
      }
      rootBone.updateMatrixWorld(true)

      const neutralBounds = new THREE.Box3().setFromObject(rootBone)
      const rootState = bones.find((item) => item.role === 'root')
      if (rootState && Number.isFinite(neutralBounds.min.y)) {
        const soleClearance = averageBoneLength * 0.015
        rootState.restPosition.y += -neutralBounds.min.y + soleClearance
        rootState.targetPosition.copy(rootState.restPosition)
      }

      const persistentPose = persistentPoseRef.current
      if (persistentPose.kind !== 'standing') {
        for (const item of bones) {
          const saved = persistentPose.bones.get(item.bone.name)
          if (!saved) continue
          item.restPosition.copy(saved.position)
          item.restQuaternion.copy(saved.quaternion)
          item.targetPosition.copy(saved.position)
          item.targetQuaternion.copy(saved.quaternion)
        }
      }

      avatarAnimationRef.current = {
        rootBone,
        bones,
        phase,
        poseKind: persistentPose.kind,
        idleTime: Math.random() * Math.PI * 2,
        returnElapsed: 0,
        returnDuration: 0.9,
        scale: averageBoneLength,
      }

      // 初期表示ではTポーズを挟まず、直接ニュートラルIdleを適用する。
      if (phase === 'idle') {
        for (const item of bones) {
          item.bone.position.copy(item.restPosition)
          item.bone.quaternion.copy(item.restQuaternion)
        }
        rootBone.updateMatrixWorld(true)
      } else {
        // モーション再生前はBVH本来の初期姿勢へ戻しておく。
        for (const item of bones) {
          item.bone.position.copy(item.returnPosition)
          item.bone.quaternion.copy(item.returnQuaternion)
        }
        rootBone.updateMatrixWorld(true)
      }
    },
    [],
  )

  const applyReturnPose = useCallback((state: AvatarAnimationState) => {
    for (const item of state.bones) {
      item.bone.position.copy(item.returnPosition)
      item.bone.quaternion.copy(item.returnQuaternion)
    }
    state.rootBone.updateMatrixWorld(true)
  }, [])

  const liftReturnPoseAboveFloor = useCallback((state: AvatarAnimationState) => {
    applyReturnPose(state)
    const bounds = new THREE.Box3().setFromObject(state.rootBone)
    const rootState = state.bones.find((item) => item.role === 'root')
    if (!rootState || !Number.isFinite(bounds.min.y)) return

    const soleClearance = state.scale * 0.015
    const liftY = soleClearance - bounds.min.y
    if (liftY > 0) {
      rootState.returnPosition.y += liftY
      applyReturnPose(state)
    }
  }, [applyReturnPose])

  const saveCurrentPoseAsPersistent = useCallback((
    state: AvatarAnimationState,
    kind: PostureKind,
    preserveFloorContacts = false,
  ) => {
    if (!preserveFloorContacts) liftReturnPoseAboveFloor(state)

    const bones = new Map<string, PersistentBonePose>()
    for (const item of state.bones) {
      bones.set(item.bone.name, {
        position: item.returnPosition.clone(),
        quaternion: item.returnQuaternion.clone(),
      })
      item.restPosition.copy(item.returnPosition)
      item.restQuaternion.copy(item.returnQuaternion)
      item.targetPosition.copy(item.returnPosition)
      item.targetQuaternion.copy(item.returnQuaternion)
    }

    persistentPoseRef.current = { kind, bones }
    state.poseKind = kind
  }, [liftReturnPoseAboveFloor])

  const clearPersistentPose = useCallback(() => {
    persistentPoseRef.current = {
      kind: 'standing',
      bones: new Map(),
    }
  }, [])

  const beginReturnToIdle = useCallback((rootBone: THREE.Bone) => {
    const state = avatarAnimationRef.current
    if (!state || state.rootBone !== rootBone || state.phase !== 'motion') return

    const contact = contactTransitionRef.current
    if (contact) {
      characterFloorOffsetRef.current = contact.to
      contactTransitionRef.current = null
      const character = characterRef.current
      if (character) {
        character.root.position.y = character.tuning.yOffset -
          character.groundOffset * character.root.scale.y + contact.to
      }
    }

    let maxAngularDistance = 0
    for (const item of state.bones) {
      item.returnPosition.copy(item.bone.position)
      item.returnQuaternion.copy(item.bone.quaternion)
      maxAngularDistance = Math.max(
        maxAngularDistance,
        item.returnQuaternion.angleTo(item.restQuaternion),
      )
    }

    const endBehavior = currentMotionEndBehaviorRef.current
    if (endBehavior === 'hold-final-as-posture') {
      activeActionRef.current?.stop()
      saveCurrentPoseAsPersistent(state, pendingPostureKindRef.current)
      state.returnElapsed = 0
      state.phase = 'idle'
      console.log(
        `[BvhViewer] motion finished → holding final pose as ${pendingPostureKindRef.current}`,
      )
      return
    }

    if (endBehavior === 'return-to-standing-idle') {
      const rootState = state.bones.find((item) => item.role === 'root')
      const character = characterRef.current
      if (rootState) {
        // Keep the position reached by standing up instead of sliding back to
        // the BVH origin while blending to the upright posture.
        rootState.restPosition.x = rootState.returnPosition.x
        rootState.restPosition.z = rootState.returnPosition.z
        if (character?.sourceRootRestPosition) {
          state.returnRootBaseline = {
            from: character.sourceRootRestPosition.y,
            to: rootState.restPosition.y,
          }
        }
      }
    }

    // 大きく動いた姿勢ほど少し時間をかける。ただし間延びしない範囲に制限する。
    state.returnDuration = THREE.MathUtils.clamp(
      0.62 + maxAngularDistance * 0.22,
      0.72,
      1.25,
    )
    state.returnElapsed = 0
    state.phase = 'returning'

    activeActionRef.current?.stop()

    // stop() によるバインド解除で姿勢が戻る実装差を吸収し、終了姿勢を再適用する。
    liftReturnPoseAboveFloor(state)

    console.log(
      `[BvhViewer] motion finished → returning to ${persistentPoseRef.current.kind} idle (${state.returnDuration.toFixed(2)}s)`,
    )
  }, [liftReturnPoseAboveFloor, saveCurrentPoseAsPersistent])

  const updateAvatarAnimation = useCallback((delta: number) => {
    const state = avatarAnimationRef.current
    if (!state || !isPlayingRef.current || state.phase === 'motion') return

    state.idleTime += delta
    if (state.phase === 'returning') {
      state.returnElapsed += delta
    }

    const rawProgress =
      state.phase === 'returning'
        ? THREE.MathUtils.clamp(state.returnElapsed / state.returnDuration, 0, 1)
        : 1
    // 速度・加速度とも端点で0になるため、機械的な停止感が出にくい。
    const blend =
      rawProgress * rawProgress * rawProgress *
      (rawProgress * (rawProgress * 6 - 15) + 10)
    const t = state.idleTime
    const scale = state.scale
    const seated = state.poseKind !== 'standing'
    if (state.phase === 'returning' && state.returnRootBaseline) {
      const character = characterRef.current
      if (character?.sourceRootRestPosition) {
        character.sourceRootRestPosition.y = THREE.MathUtils.lerp(
          state.returnRootBaseline.from, state.returnRootBaseline.to, blend,
        )
        if (sourceRigRef.current) {
          sourceRigRef.current.groundedRootPosition.y = character.sourceRootRestPosition.y
        }
      }
    }
    const idleRotation = new THREE.Euler()
    const idleOffset = new THREE.Quaternion()

    for (const item of state.bones) {
      const targetPosition = item.targetPosition.copy(item.restPosition)
      idleRotation.set(0, 0, 0)

      // G1らしい「姿勢制御中の微動」。人間の呼吸より控えめにする。
      switch (item.role) {
        case 'root':
          targetPosition.x += Math.sin(t * 0.72) * scale * (seated ? 0.006 : 0.018)
          targetPosition.y +=
            (Math.sin(t * 1.18) * 0.5 + Math.sin(t * 0.43) * 0.5) *
            scale *
            (seated ? 0.002 : 0.008)
          idleRotation.set(
            Math.sin(t * 0.46) * (seated ? 0.002 : 0.004),
            Math.sin(t * 0.31) * (seated ? 0.004 : 0.008),
            Math.sin(t * 0.72) * (seated ? 0.003 : 0.009),
          )
          break
        case 'hips':
          idleRotation.set(
            Math.sin(t * 0.53) * 0.006,
            Math.sin(t * 0.37) * 0.008,
            -Math.sin(t * 0.72) * 0.012,
          )
          break
        case 'spine':
          idleRotation.set(
            Math.sin(t * 1.18) * 0.006,
            Math.sin(t * 0.31) * 0.004,
            Math.sin(t * 0.72) * 0.006,
          )
          break
        case 'chest':
          idleRotation.set(
            Math.sin(t * 1.18) * 0.008,
            Math.sin(t * 0.31) * 0.006,
            Math.sin(t * 0.72) * 0.008,
          )
          break
        case 'neck':
          idleRotation.set(
            Math.sin(t * 0.41 + 0.6) * 0.008,
            Math.sin(t * 0.29 + 0.4) * 0.012,
            Math.sin(t * 0.53) * 0.004,
          )
          break
        case 'head':
          idleRotation.set(
            Math.sin(t * 0.41 + 0.9) * 0.012,
            Math.sin(t * 0.29 + 0.7) * 0.018,
            Math.sin(t * 0.53 + 0.2) * 0.006,
          )
          break
        case 'shoulder':
          idleRotation.set(
            Math.sin(t * 1.18) * 0.003,
            0,
            item.side * Math.sin(t * 0.72) * 0.004,
          )
          break
        case 'upperArm':
          idleRotation.set(
            Math.sin(t * 0.67 + item.side) * 0.003,
            0,
            item.side * Math.sin(t * 0.72) * 0.004,
          )
          break
        case 'lowerArm':
          idleRotation.set(
            0,
            item.side * Math.sin(t * 0.53 + 0.8) * 0.004,
            0,
          )
          break
        case 'thigh':
        case 'shin':
          idleRotation.set(
            Math.sin(t * 0.72 + item.side * 0.5) * (seated ? 0.0007 : 0.002),
            0,
            item.side * Math.sin(t * 0.72) * (seated ? 0.0007 : 0.002),
          )
          break
        default:
          break
      }

      idleOffset.setFromEuler(idleRotation)
      const targetQuaternion = item.targetQuaternion
        .copy(item.restQuaternion)
        .multiply(idleOffset)

      if (state.phase === 'returning') {
        item.bone.position.lerpVectors(
          item.returnPosition,
          targetPosition,
          blend,
        )
        item.bone.quaternion.slerpQuaternions(
          item.returnQuaternion,
          targetQuaternion,
          blend,
        )
      } else {
        item.bone.position.copy(targetPosition)
        item.bone.quaternion.copy(targetQuaternion)
      }
    }

    if (state.phase === 'returning' && rawProgress >= 1) {
      state.phase = 'idle'
      if (!roomTaskRef.current && persistentPoseRef.current.kind === 'standing') {
        setRoomTask((task) => task && (task.phase === 'standing' || task.phase === 'settling') ? { ...task, phase: 'idle' } : task)
      }
      console.log('[BvhViewer] idle state entered')
    }
  }, [])

  // ロボットモード用 primitive を保持し、再生成時に dispose する
  const robotRigPartsRef = useRef<THREE.Mesh[]>([])

  const disposeRobotRig = useCallback(() => {
    for (const mesh of robotRigPartsRef.current) {
      mesh.parent?.remove(mesh)
      mesh.geometry.dispose()
      const m = mesh.material as THREE.Material | THREE.Material[]
      if (Array.isArray(m)) m.forEach((mm) => mm.dispose())
      else m.dispose()
    }
    robotRigPartsRef.current = []
  }, [])

  // キャラクターを現在の BVH 骨格の高さに合わせてスケール/向き/位置を調整する。
  const standingSkeletonHeight = useCallback(() => {
    const sourceRig = sourceRigRef.current
    if (!sourceRig) return NORMALIZED_SKELETON_HEIGHT
    let minY = Infinity
    let maxY = -Infinity
    for (const position of sourceRig.restWorldPosition.values()) {
      minY = Math.min(minY, position.y)
      maxY = Math.max(maxY, position.y)
    }
    return Number.isFinite(maxY - minY) && maxY - minY > 0.1
      ? maxY - minY
      : NORMALIZED_SKELETON_HEIGHT
  }, [])

  const displayedBones = useCallback(() => {
    return viewModeRef.current === 'character' && characterRef.current
      ? characterRef.current.bonesByCanonical
      : sourceRigRef.current?.byCanonical
  }, [])

  const fitCharacterToSkeleton = useCallback(() => {
    const character = characterRef.current
    if (!character) return
    const src =
      skeletonHelperRef.current ??
      (sceneRef.current?.userData.currentRootBone as THREE.Object3D | undefined)
    if (!src) return

    src.updateMatrixWorld(true)
    const box = new THREE.Box3().setFromObject(src)
    // Seated poses are shorter. Use the standing rig, so changing avatars or yaw
    // while seated does not shrink the character or detach it from the chair.
    const height = sourceRigRef.current
      ? standingSkeletonHeight()
      : Math.max(box.max.y - box.min.y, 1e-3)
    const scale = (height / character.naturalHeight) * character.tuning.heightScale

    character.root.scale.setScalar(scale)
    // 自動判定した正面向き + UI での追加調整
    const yawDegrees = character.autoYawDegrees + character.tuning.yawDegrees + THREE.MathUtils.radToDeg(roomPlacementYawRef.current)
    character.root.rotation.set(0, THREE.MathUtils.degToRad(yawDegrees), 0)
    // モデルの原点が足元にあるとは限らないので、最下点(靴底)を床へ降ろす
    character.root.position.set(
      motionSpaceRef.current?.position.x ?? 0,
      character.tuning.yOffset - character.groundOffset * scale + characterFloorOffsetRef.current,
      motionSpaceRef.current?.position.z ?? 0,
    )
    const pivot = roomCharacterPivotRef.current
    if (pivot) {
      const offset = pivot.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), character.root.rotation.y)
      character.root.position.x -= offset.x
      character.root.position.z -= offset.z
    }
    character.root.updateMatrixWorld(true)
    console.log(
      `[BvhViewer] character fitted: bvhHeight=${height.toFixed(1)}, ` +
        `naturalHeight=${character.naturalHeight.toFixed(2)}, scale=${scale.toFixed(3)}, ` +
        `groundY=${(character.tuning.yOffset - character.groundOffset * scale).toFixed(3)}, ` +
        `yaw=${yawDegrees.toFixed(1)}deg`,
    )
  }, [standingSkeletonHeight])

  const characterFloorGap = useCallback(() => {
    const character = characterRef.current
    if (!character) return 0
    character.root.updateMatrixWorld(true)
    let lowestY = Infinity
    character.model.traverse((object) => {
      if (!(object instanceof THREE.SkinnedMesh)) return
      object.skeleton.update()
      object.computeBoundingBox()
      if (object.boundingBox) {
        lowestY = Math.min(lowestY, object.boundingBox.clone().applyMatrix4(object.matrixWorld).min.y)
      }
    })
    return Number.isFinite(lowestY) ? lowestY - 0.003 : 0
  }, [])

  const groundSeatedCharacter = useCallback(() => {
    const character = characterRef.current
    if (!character) return
    const gap = characterFloorGap()
    characterFloorOffsetRef.current -= gap
    character.root.position.y -= gap
    character.root.updateMatrixWorld(true)
  }, [characterFloorGap])

  const fitChairToDisplayedPose = useCallback(() => {
    if (roomFixedSeatRef.current) return
    const chair = chairRef.current
    const bones = displayedBones()
    if (!chair || !bones) return
    sceneRef.current?.updateMatrixWorld(true)
    if (fitSceneChairToSeatedPose(chair, bones, standingSkeletonHeight())) {
      chairWasFittedRef.current = true
      console.log(`[BvhViewer] chair fitted: seatY=${chair.seatHeight.toFixed(3)}, x=${chair.root.position.x.toFixed(3)}, z=${chair.root.position.z.toFixed(3)}`)
    }
  }, [displayedBones, standingSkeletonHeight])

  const applyFixedRoomContacts = useCallback(() => {
    const chair = roomFixedSeatRef.current
    if (!chair || persistentPoseRef.current.kind !== 'chair_sitting' || roomTaskRef.current?.ownsPose) return
    const source = sourceRigRef.current?.byCanonical
    const refs = roomFootReferencesRef.current
    if (source) {
      const target = createSeatContactTargets(source, chair, standingSkeletonHeight(), { footReference: refs.source })
      if (target) applySeatContactTargets(source, target)
    }
    const character = characterRef.current
    if (character) {
      retargetCharacter(character)
      const target = createSeatContactTargets(character.bonesByCanonical, chair, standingSkeletonHeight(), { footReference: refs.character })
      if (target) applySeatContactTargets(character.bonesByCanonical, target)
    }
  }, [standingSkeletonHeight])

  /**
   * アバター(glb)を読み込んで差し替える。
   * 既存キャラクターを破棄 → ロード → BVH 骨格へバインド → 高さフィット、の順。
   */
  const applyAvatar = useCallback(
    async (url: string, isCurrent: () => boolean) => {
      const scene = sceneRef.current
      if (!scene) return

      setAvatarStatus('読込中…')
      retargetErroredRef.current = false
      try {
        const character = await loadCharacterModel(url)
        if (!isCurrent() || sceneRef.current !== scene) {
          disposeCharacter(character)
          return
        }

        if (characterRef.current) {
          disposeCharacter(characterRef.current)
          characterRef.current = null
        }

        character.tuning.yawDegrees = avatarYawRef.current
        characterRef.current = character
        character.root.visible = viewModeRef.current === 'character'
        character.root.traverse((object) => {
          if (object instanceof THREE.Mesh) {
            object.castShadow = true
            object.receiveShadow = true
          }
        })
        scene.add(character.root)

        roomCharacterPivotRef.current = null
        const sourceRig = sourceRigRef.current
        if (sourceRig) bindCharacterToSource(character, sourceRig)
        fitCharacterToSkeleton()
        if ((roomTaskRef.current?.ownsPose || roomFixedSeatRef.current) && sourceRig) {
          // Calibrate the new avatar in the same grounded standing pose, then
          // restore the ongoing task pose within this frame.
          const saved = [...sourceRig.byCanonical.values()].map((bone) => ({ bone, position: bone.position.clone(), quaternion: bone.quaternion.clone() }))
          for (const pose of saved) {
            const upright = roomStandingPoseRef.current.get(pose.bone.name)
            if (upright) { pose.bone.position.copy(upright.position); pose.bone.quaternion.copy(upright.quaternion) }
          }
          retargetCharacter(character)
          groundSeatedCharacter()
          const hip = character.bonesByCanonical.get('hips')?.getWorldPosition(new THREE.Vector3())
          if (hip) roomCharacterPivotRef.current = hip.sub(character.root.position).applyAxisAngle(new THREE.Vector3(0, 1, 0), -character.root.rotation.y)
          roomFootReferencesRef.current.character = captureStandingFootContacts(character.bonesByCanonical)
          for (const pose of saved) { pose.bone.position.copy(pose.position); pose.bone.quaternion.copy(pose.quaternion) }
          roomTaskRef.current?.refreshCharacterContacts(roomFootReferencesRef.current.character)
          fitCharacterToSkeleton()
        }
        retargetCharacter(character)
        character.root.updateMatrixWorld(true)
        if (roomTaskRef.current?.ownsPose || roomFixedSeatRef.current) {
          roomTaskRef.current?.applyPoseContacts()
          applyFixedRoomContacts()
        } else if (persistentPoseRef.current.kind === 'chair_sitting') {
          groundSeatedCharacter()
          fitChairToDisplayedPose()
        } else if (chairRef.current && !chairWasFittedRef.current) {
          const size = standingSkeletonHeight() / NORMALIZED_SKELETON_HEIGHT
          const center = new THREE.Vector3(0, 0, -0.32 * size)
            .applyAxisAngle(new THREE.Vector3(0, 1, 0), THREE.MathUtils.degToRad(avatarYawRef.current))
            .add(motionSpaceRef.current?.position ?? new THREE.Vector3())
          placeSceneChair(chairRef.current, center, 0.46 * size, size, THREE.MathUtils.degToRad(avatarYawRef.current))
        }

        setAvatarStatus(
          `${character.flavor} リグ / ${character.links.length} ボーン連動`,
        )
        console.log(`[BvhViewer] avatar applied: ${url}`)
      } catch (err) {
        if (!isCurrent()) return
        console.error('[BvhViewer] avatar load failed:', err)
        setAvatarStatus(
          `読込失敗: ${err instanceof Error ? err.message : String(err)}`,
        )
      }
    },
    [applyFixedRoomContacts, fitCharacterToSkeleton, fitChairToDisplayedPose, groundSeatedCharacter, standingSkeletonHeight],
  )

  /**
   * BVH のボーンへ Unitree G1 をモチーフにした外装を attach する。
   * 細身の銀色フレーム、露出した黒いアクチュエータ、短い胸郭、
   * くびれた腰、横長センサーヘッド、角張った前腕・すねを特徴にする。
   */
  const buildRobotRig = useCallback((rootBone: THREE.Bone) => {
    disposeRobotRig()

    const allBones: THREE.Bone[] = []
    rootBone.traverse((o) => {
      if ((o as THREE.Bone).isBone) allBones.push(o as THREE.Bone)
    })

    const findBone = (re: RegExp) => allBones.find((b) => re.test(b.name))
    const filterBones = (re: RegExp) => allBones.filter((b) => re.test(b.name))

    // 名前検出: フィンガー等にマッチしないよう末尾アンカー
    // "LeftHand" は /hand$/ にマッチ、 "LeftHandThumb1" はマッチしない
    const handBones = filterBones(/hand$|wrist$|palm$/i)
    const footBones = filterBones(/foot$|ankle$/i)
    const shoulderBones = filterBones(/shoulder$|clavicle$/i)
    const headBone = findBone(/head$/i) ?? findBone(/neck$/i)
    const hipsBone =
      findBone(/hips?$/i) ??
      findBone(/pelvis$/i) ??
      findBone(/^root$/i) ??
      rootBone
    // chestBone は肩の親 (= 肩がぶら下がっているスパイン) を最優先で採用
    const chestBone =
      (shoulderBones[0]?.parent as THREE.Bone | undefined) ??
      findBone(/chest$/i) ??
      findBone(/spine\s*4/i) ??
      findBone(/spine\s*3/i) ??
      findBone(/spine\s*2/i) ??
      findBone(/upperback$/i) ??
      findBone(/spine\s*1/i) ??
      findBone(/spine$/i)

    // フィンガー/つま先 (= 手/足の子孫) かどうか
    const isFingerOrToe = (b: THREE.Bone) => {
      let cur: THREE.Object3D | null = b.parent
      while (cur) {
        if ((cur as THREE.Bone).isBone) {
          const bb = cur as THREE.Bone
          if (handBones.includes(bb) || footBones.includes(bb)) return true
        }
        cur = cur.parent
      }
      return false
    }

    // 平均ボーン長 (フィンガー/つま先は除外して胴体・四肢の規模感を取得)
    let total = 0
    let count = 0
    for (const b of allBones) {
      if (isFingerOrToe(b)) continue
      if (handBones.includes(b) || footBones.includes(b)) continue
      for (const c of b.children) {
        if (!(c as THREE.Bone).isBone) continue
        const cb = c as THREE.Bone
        if (isFingerOrToe(cb)) continue
        const len = cb.position.length()
        if (len > 1e-3) {
          total += len
          count++
        }
      }
    }
    const avgLen = count > 0 ? total / count : 1

    // 胸甲の高さ用に、hips→chest のスパイン連鎖の総長を計算
    let spineChainLen = 0
    if (chestBone && hipsBone) {
      let cur: THREE.Object3D | null = chestBone
      while (cur && cur !== hipsBone) {
        if ((cur as THREE.Bone).isBone) {
          spineChainLen += (cur as THREE.Bone).position.length()
        }
        cur = cur.parent
      }
    }
    if (spineChainLen < 1e-3) spineChainLen = avgLen * 3

    const limbR = avgLen * 0.19
    const cuffH = avgLen * 0.13
    const jointR = avgLen * 0.24

    const shellMat = new THREE.MeshStandardMaterial({
      color: 0xb9bdbe,
      roughness: 0.32,
      metalness: 0.72,
    })
    const innerMat = new THREE.MeshStandardMaterial({
      color: 0x111315,
      roughness: 0.3,
      metalness: 0.72,
    })
    const accentMat = new THREE.MeshStandardMaterial({
      color: 0x73787a,
      roughness: 0.25,
      metalness: 0.92,
    })
    const panelMat = new THREE.MeshStandardMaterial({
      color: 0x2a2d2f,
      roughness: 0.28,
      metalness: 0.78,
    })
    const softBlackMat = new THREE.MeshStandardMaterial({
      color: 0x090a0b,
      roughness: 0.48,
      metalness: 0.35,
    })
    const visorMat = new THREE.MeshStandardMaterial({
      color: 0x080b0d,
      roughness: 0.12,
      metalness: 0.75,
      emissive: 0x1a93a8,
      emissiveIntensity: 0.3,
    })
    const visorRimMat = new THREE.MeshStandardMaterial({
      color: 0x171a1c,
      roughness: 0.24,
      metalness: 0.9,
    })

    const tracked: THREE.Mesh[] = []
    const yAxis = new THREE.Vector3(0, 1, 0)

    const torsoBoneSet = new Set<THREE.Bone>(
      [hipsBone, chestBone].filter(Boolean) as THREE.Bone[],
    )
    if (chestBone) {
      let cur: THREE.Object3D | null = chestBone
      while (cur && cur !== hipsBone) {
        if ((cur as THREE.Bone).isBone) torsoBoneSet.add(cur as THREE.Bone)
        cur = cur.parent
      }
    }

    const addPart = (mesh: THREE.Mesh, parent: THREE.Object3D) => {
      mesh.visible = viewModeRef.current === 'robot'
      mesh.castShadow = true
      mesh.receiveShadow = true
      parent.add(mesh)
      tracked.push(mesh)
    }

    const isHandBone = (b: THREE.Bone) => handBones.includes(b)
    const isFootBone = (b: THREE.Bone) => footBones.includes(b)
    const isHeadBone = (b: THREE.Bone) => b === headBone
    const isArmBone = (b: THREE.Bone) =>
      /arm|elbow|forearm|wrist/i.test(b.name)
    const isLowerArmBone = (b: THREE.Bone) =>
      /forearm|lowerarm|elbow/i.test(b.name)
    const isLegBone = (b: THREE.Bone) =>
      /leg|thigh|upleg|knee|shin|calf/i.test(b.name)
    const isLowerLegBone = (b: THREE.Bone) =>
      /leg$|lowerleg|knee|shin|calf/i.test(b.name) &&
      !/upleg|upperleg|thigh/i.test(b.name)

    // ── 細身の四肢フレーム + 部位別の装甲パネル ──
    // 注: 胴体ボーン (torsoBoneSet) は「自分→子」の中で外向き接続 (Spine4→Neck, Hips→UpLeg 等)
    //     のみ描き、内部のスパイン連鎖 (Spine→Spine1 等) は胸甲が覆うので描かない。
    for (const bone of allBones) {
      if (isFingerOrToe(bone)) continue
      if (isHandBone(bone) || isFootBone(bone)) continue
      if (isHeadBone(bone)) continue

      for (const child of bone.children) {
        if (!(child as THREE.Bone).isBone) continue
        const childBone = child as THREE.Bone
        if (isFingerOrToe(childBone)) continue
        // 内部スパイン (torso→torso) は胸甲で覆うのでスキップ
        if (torsoBoneSet.has(bone) && torsoBoneSet.has(childBone)) continue
        // 肩オフセットはパウルドロンで覆うのでスキップ (胴体→肩)
        if (torsoBoneSet.has(bone) && shoulderBones.includes(childBone)) continue

        const childPos = childBone.position
        const dist = childPos.length()
        if (dist < 1e-3) continue
        const dir = childPos.clone().normalize()

        const arm = isArmBone(bone) || isArmBone(childBone)
        const leg = isLegBone(bone) || isLegBone(childBone)
        const lowerArm = isLowerArmBone(bone) || isLowerArmBone(childBone)
        const lowerLeg = isLowerLegBone(bone) || isLowerLegBone(childBone)
        const radiusScale = leg ? 1.05 : arm ? 0.82 : 0.72
        const segR = Math.min(limbR * radiusScale, dist * 0.28)

        // G1 は関節間の黒い構造材が見えるため、まず細い芯を通す
        const core = new THREE.Mesh(
          new THREE.CylinderGeometry(segR * 0.58, segR * 0.72, dist * 0.86, 14),
          innerMat,
        )
        core.position.copy(childPos).multiplyScalar(0.5)
        core.quaternion.setFromUnitVectors(yAxis, dir)
        addPart(core, bone)

        if (dist > avgLen * 0.48) {
          // 上腕・腿は細い銀色フレーム、前腕・すねは角張った装甲を強調
          const panelLength = dist * (lowerArm || lowerLeg ? 0.68 : 0.56)
          const panelW = segR * (lowerLeg ? 2.25 : lowerArm ? 2.05 : 1.62)
          const panelD = segR * (lowerLeg ? 1.55 : 1.28)
          const panel = new THREE.Mesh(
            new THREE.BoxGeometry(panelW, panelLength, panelD),
            shellMat,
          )
          panel.position.copy(dir).multiplyScalar(dist * 0.5)
          panel.quaternion.setFromUnitVectors(yAxis, dir)
          addPart(panel, bone)

          // 前面の黒いサービスパネル
          if (lowerArm || lowerLeg) {
            const inset = new THREE.Mesh(
              new THREE.BoxGeometry(panelW * 0.48, panelLength * 0.55, panelD * 0.08),
              panelMat,
            )
            inset.position.copy(dir).multiplyScalar(dist * 0.5)
            inset.position.z += panelD * 0.54
            inset.quaternion.copy(panel.quaternion)
            addPart(inset, bone)
          }

          const endActuator = new THREE.Mesh(
            new THREE.CylinderGeometry(segR * 1.08, segR * 1.08, cuffH, 18),
            innerMat,
          )
          endActuator.position.copy(dir).multiplyScalar(dist * 0.84)
          endActuator.quaternion.setFromUnitVectors(yAxis, dir)
          addPart(endActuator, bone)
        }
      }

      // 関節接続部のダーク球 (専用形状で覆う部位は除く)
      if (
        !shoulderBones.includes(bone) &&
        !torsoBoneSet.has(bone) &&
        !isHeadBone(bone) &&
        !isHandBone(bone) &&
        !isFootBone(bone)
      ) {
        const actuatorScale = isLegBone(bone) ? 1.12 : 0.9
        const jointBall = new THREE.Mesh(
          new THREE.SphereGeometry(jointR * actuatorScale, 18, 14),
          innerMat,
        )
        jointBall.scale.set(1.0, 0.82, 1.0)
        addPart(jointBall, bone)
      }
    }

    // ── G1風のコンパクトな肩アクチュエータ ──
    for (const sb of shoulderBones) {
      const firstChild = sb.children.find(
        (c) => (c as THREE.Bone).isBone,
      ) as THREE.Bone | undefined
      const dir =
        firstChild && firstChild.position.lengthSq() > 1e-6
          ? firstChild.position.clone().normalize()
          : new THREE.Vector3(1, 0, 0)

      const shoulderMotor = new THREE.Mesh(
        new THREE.CylinderGeometry(limbR * 1.28, limbR * 1.28, limbR * 1.12, 20),
        innerMat,
      )
      shoulderMotor.quaternion.setFromUnitVectors(yAxis, dir)
      shoulderMotor.position.copy(dir).multiplyScalar(limbR * 0.45)
      addPart(shoulderMotor, sb)

      const shoulderCap = new THREE.Mesh(
        new THREE.CylinderGeometry(limbR * 1.05, limbR * 1.05, limbR * 0.28, 20),
        shellMat,
      )
      shoulderCap.quaternion.setFromUnitVectors(yAxis, dir)
      shoulderCap.position.copy(dir).multiplyScalar(limbR * 1.05)
      addPart(shoulderCap, sb)
    }

    // 肩幅を実測。肩ボーン自体の位置だけだと幅が短くなるので、
    // 「肩 offset + 上腕 offset」= 実質の腕の付け根位置 で計測する
    let shoulderSpan = avgLen * 2.5
    if (shoulderBones.length >= 2) {
      const armRoots: THREE.Vector3[] = []
      for (const sb of shoulderBones) {
        const armChild = sb.children.find(
          (c) => (c as THREE.Bone).isBone && !shoulderBones.includes(c as THREE.Bone),
        ) as THREE.Bone | undefined
        if (armChild) {
          armRoots.push(sb.position.clone().add(armChild.position))
        } else {
          armRoots.push(sb.position.clone())
        }
      }
      if (armRoots.length >= 2) {
        const span = armRoots[0].distanceTo(armRoots[1])
        if (span > 1e-3) shoulderSpan = Math.max(span * 0.95, avgLen * 2.0)
      }
    }

    // ── 胸部: G1の短い箱型胸郭と、露出した細い腹部 ──
    if (chestBone) {
      const chestH = spineChainLen * 0.62
      const chestW = shoulderSpan * 0.76
      const chestD = Math.max(spineChainLen * 0.34, avgLen * 0.72)
      const chestCenterY = -spineChainLen * 0.24

      const chest = new THREE.Mesh(
        new THREE.BoxGeometry(chestW, chestH, chestD),
        shellMat,
      )
      chest.position.y = chestCenterY
      addPart(chest, chestBone)

      // 正面の黒い制御パネルと小さなステータス灯
      const chestPanel = new THREE.Mesh(
        new THREE.BoxGeometry(chestW * 0.42, chestH * 0.48, chestD * 0.06),
        panelMat,
      )
      chestPanel.position.set(0, chestCenterY - chestH * 0.03, chestD * 0.53)
      addPart(chestPanel, chestBone)

      const statusLight = new THREE.Mesh(
        new THREE.BoxGeometry(chestW * 0.12, chestH * 0.045, chestD * 0.025),
        visorMat,
      )
      statusLight.position.set(0, chestCenterY + chestH * 0.08, chestD * 0.57)
      addPart(statusLight, chestBone)

      const upperYoke = new THREE.Mesh(
        new THREE.BoxGeometry(chestW * 0.92, chestH * 0.16, chestD * 0.92),
        innerMat,
      )
      upperYoke.position.set(0, chestCenterY + chestH * 0.48, 0)
      addPart(upperYoke, chestBone)

      // 胸郭の下に見える二段の腹部アクチュエータ
      for (let i = 0; i < 2; i++) {
        const abSegment = new THREE.Mesh(
          new THREE.CylinderGeometry(
            chestW * (0.19 - i * 0.015),
            chestW * (0.21 - i * 0.015),
            spineChainLen * 0.16,
            12,
          ),
          i === 0 ? accentMat : innerMat,
        )
        abSegment.position.y = chestCenterY - chestH * (0.64 + i * 0.28)
        addPart(abSegment, chestBone)
      }
    }

    // ── 腰: 小さな骨盤シェルと左右の股関節モーター ──
    if (hipsBone) {
      const beltW = Math.max(shoulderSpan * 0.54, avgLen * 1.45)
      const beltH = avgLen * 0.52
      const beltD = Math.max(spineChainLen * 0.3, avgLen * 0.72)
      const waistCore = new THREE.Mesh(
        new THREE.CylinderGeometry(beltW * 0.24, beltW * 0.27, beltH * 0.9, 16),
        innerMat,
      )
      waistCore.position.y = beltH * 0.42
      addPart(waistCore, hipsBone)

      const pelvis = new THREE.Mesh(
        new THREE.BoxGeometry(beltW, beltH, beltD),
        shellMat,
      )
      pelvis.position.y = -beltH * 0.12
      addPart(pelvis, hipsBone)

      const pelvisPanel = new THREE.Mesh(
        new THREE.BoxGeometry(beltW * 0.34, beltH * 0.52, beltD * 0.05),
        panelMat,
      )
      pelvisPanel.position.set(0, -beltH * 0.12, beltD * 0.53)
      addPart(pelvisPanel, hipsBone)

      for (const side of [-1, 1]) {
        const hipMotor = new THREE.Mesh(
          new THREE.CylinderGeometry(limbR * 1.2, limbR * 1.2, limbR * 0.75, 20),
          innerMat,
        )
        hipMotor.rotation.z = Math.PI / 2
        hipMotor.position.set(side * beltW * 0.52, -beltH * 0.12, 0)
        addPart(hipMotor, hipsBone)
      }
    }

    // ── 手 ──
    for (const hb of handBones) {
      // 指が複数あるとき "親指1本目" が斜め方向にあるので、
      // すべての子 (= 指) の方向平均を掌の伸びる方向として使う
      const fingerChildren = (hb.children.filter(
        (c) => (c as THREE.Bone).isBone,
      ) as THREE.Bone[]).filter((c) => c.position.lengthSq() > 1e-6)
      let dir: THREE.Vector3
      if (fingerChildren.length > 0) {
        const sum = new THREE.Vector3()
        for (const fc of fingerChildren) {
          sum.add(fc.position.clone().normalize())
        }
        dir = sum.lengthSq() > 1e-6 ? sum.normalize() : yAxis.clone()
      } else {
        dir = yAxis.clone()
      }

      // 掌のサイズは平均ボーン長基準 (avgLen) でなく、
      // 前腕から手までの長さ (= hb.position の長さ) を基準にすると過大化を防げる
      const forearmToHand = hb.position.length()
      const handBase = forearmToHand > 1e-3 ? forearmToHand : avgLen
      const handLen = handBase * 0.5
      const handThk = handBase * 0.24
      const handWid = handBase * 0.30

      const palm = new THREE.Mesh(
        new THREE.BoxGeometry(handWid, handLen, handThk),
        shellMat,
      )
      palm.quaternion.setFromUnitVectors(yAxis, dir)
      palm.position.copy(dir).multiplyScalar(handLen * 0.5)
      addPart(palm, hb)

      const wristRing = new THREE.Mesh(
        new THREE.CylinderGeometry(limbR * 1.05, limbR * 1.05, cuffH, 24),
        accentMat,
      )
      wristRing.quaternion.setFromUnitVectors(yAxis, dir)
      addPart(wristRing, hb)

      // G1の簡素な三指グリッパーを、掌先端の黒い指ブロックで表現
      for (const side of [-1, 0, 1]) {
        const finger = new THREE.Mesh(
          new THREE.BoxGeometry(handWid * 0.19, handLen * 0.44, handThk * 0.58),
          softBlackMat,
        )
        finger.quaternion.copy(palm.quaternion)
        finger.position.copy(dir).multiplyScalar(handLen * 1.02)
        finger.position.x += side * handWid * 0.3
        addPart(finger, hb)
      }
    }

    // ── 足 ──
    for (const fb of footBones) {
      const firstChild = fb.children.find(
        (c) => (c as THREE.Bone).isBone,
      ) as THREE.Bone | undefined
      const fwd =
        firstChild && firstChild.position.lengthSq() > 1e-6
          ? firstChild.position.clone().normalize()
          : new THREE.Vector3(0, 0, 1)
      const bootLen = avgLen * 1.3
      const boot = new THREE.Mesh(
        new THREE.BoxGeometry(avgLen * 0.54, avgLen * 0.28, bootLen),
        shellMat,
      )
      const zAxis = new THREE.Vector3(0, 0, 1)
      boot.quaternion.setFromUnitVectors(zAxis, fwd)
      boot.position.copy(fwd).multiplyScalar(bootLen * 0.4)
      boot.position.y -= avgLen * 0.1
      addPart(boot, fb)

      const sole = new THREE.Mesh(
        new THREE.BoxGeometry(avgLen * 0.57, avgLen * 0.1, bootLen * 1.02),
        softBlackMat,
      )
      sole.quaternion.copy(boot.quaternion)
      sole.position.copy(boot.position)
      sole.position.y -= avgLen * 0.18
      addPart(sole, fb)

      const ankleRing = new THREE.Mesh(
        new THREE.CylinderGeometry(limbR * 1.05, limbR * 1.05, cuffH, 24),
        innerMat,
      )
      addPart(ankleRing, fb)
    }

    // ── 頭部: 顔を持たない横長センサーヘッド ──
    if (headBone) {
      const headOffset = headBone.position.length()
      const headR = Math.max(headOffset * 0.82, avgLen * 0.44)
      const headY = headR * 1.38

      const neck = new THREE.Mesh(
        new THREE.CylinderGeometry(headR * 0.29, headR * 0.34, headR * 0.82, 18),
        innerMat,
      )
      neck.position.y = headR * 0.52
      addPart(neck, headBone)

      const sensorHead = new THREE.Mesh(
        new THREE.SphereGeometry(headR, 32, 20),
        softBlackMat,
      )
      sensorHead.scale.set(1.34, 0.62, 0.82)
      sensorHead.position.y = headY
      addPart(sensorHead, headBone)

      const rearShell = new THREE.Mesh(
        new THREE.SphereGeometry(headR * 0.92, 28, 18),
        shellMat,
      )
      rearShell.scale.set(1.18, 0.58, 0.54)
      rearShell.position.set(0, headY + headR * 0.12, -headR * 0.48)
      addPart(rearShell, headBone)

      const sensorGlass = new THREE.Mesh(
        new THREE.BoxGeometry(headR * 1.72, headR * 0.32, headR * 0.11),
        visorRimMat,
      )
      sensorGlass.position.set(0, headY, headR * 0.81)
      addPart(sensorGlass, headBone)

      const centerSensor = new THREE.Mesh(
        new THREE.CylinderGeometry(headR * 0.18, headR * 0.18, headR * 0.09, 24),
        visorMat,
      )
      centerSensor.rotation.x = Math.PI / 2
      centerSensor.position.set(0, headY, headR * 0.9)
      addPart(centerSensor, headBone)
    }

    robotRigPartsRef.current = tracked
    console.log(
      `[BvhViewer] Robot rig built: parts=${tracked.length}, ` +
        `avgLen=${avgLen.toFixed(2)}, limbR=${limbR.toFixed(2)}, ` +
        `head=${headBone?.name ?? 'none'}, chest=${chestBone?.name ?? 'none'}, ` +
        `hips=${hipsBone?.name ?? 'none'}, shoulders=[${shoulderBones.map((b) => b.name).join(',')}], ` +
        `hands=[${handBones.map((b) => b.name).join(',')}], ` +
        `feet=[${footBones.map((b) => b.name).join(',')}]`,
    )
  }, [disposeRobotRig])

  const invalidateRoomImports = useCallback(() => { roomLoadRequestRef.current++ }, [])

  const setEnvironmentCamera = useCallback((view: EnvironmentCamera) => {
    const camera = cameraRef.current
    const controls = controlsRef.current
    if (!camera || !controls) return
    camera.position.copy(view.position)
    camera.fov = view.fov
    controls.target.copy(view.target)
    camera.updateProjectionMatrix()
    controls.update()
  }, [])

  const showWholeEnvironment = useCallback(() => {
    const view = environmentModeRef.current === 'home'
      ? importedRoomCameraRef.current ?? { position: HOME_CAMERA_POSITION, target: HOME_CAMERA_TARGET, fov: HOME_ENVIRONMENT_CAMERA.fov }
      : laboratoryDefaultCameraRef.current
    if (view) setEnvironmentCamera(view)
  }, [setEnvironmentCamera])

  const focusAvatar = useCallback(() => {
    const rootBone = sceneRef.current?.userData.currentRootBone as THREE.Bone | undefined
    if (!rootBone) return
    sceneRef.current?.updateMatrixWorld(true)
    const target = rootBone.getWorldPosition(new THREE.Vector3())
    target.y = Math.max(target.y, 0.9)
    setEnvironmentCamera({ position: target.clone().add(new THREE.Vector3(2.2, 1.0, 2.8)), target, fov: 48 })
  }, [setEnvironmentCamera])

  const applyEnvironment = useCallback((mode: EnvironmentMode) => {
    const scene = sceneRef.current
    const camera = cameraRef.current
    const controls = controlsRef.current
    if (!scene || !camera || !controls || !homeEnvironmentRef.current) return
    const previous = environmentModeRef.current
    if (environmentInitializedRef.current && previous !== mode) {
      environmentCamerasRef.current[previous] = {
        position: camera.position.clone(), target: controls.target.clone(), fov: camera.fov,
      }
    }
    if (previous !== mode && roomTaskRef.current?.busy) {
      pendingRoomDestinationRef.current = null
      roomTaskRef.current.cancel()
    }
    if (roomTaskRef.current) roomTaskRef.current.route.visible = mode === 'home' && roomTaskRef.current.route.userData.showRoute !== false
    environmentModeRef.current = mode
    homeEnvironmentRef.current.root.visible = mode === 'home' && !importedRoomRef.current
    if (importedRoomRef.current) importedRoomRef.current.root.visible = mode === 'home'
    if (laboratoryEnvironmentRef.current) laboratoryEnvironmentRef.current.visible = mode === 'laboratory'
    scene.background = new THREE.Color(mode === 'home' ? 0xe9e2d6 : 0x0f172a)
    homeRenderingRef.current?.setEnabled(mode === 'home')
    camera.near = 0.05
    const importedViewDistance = importedRoomCameraRef.current?.position.distanceTo(importedRoomCameraRef.current.target) ?? 0
    camera.far = Math.max(120, importedViewDistance * 5)
    controls.minDistance = mode === 'home' ? 0.8 : 0.5
    controls.maxDistance = mode === 'home' ? Math.max(22, importedViewDistance * 3) : 40
    controls.maxPolarAngle = Math.PI / 2 - 0.015
    const view = environmentCamerasRef.current[mode] ?? (mode === 'home'
      ? importedRoomCameraRef.current ?? { position: HOME_CAMERA_POSITION, target: HOME_CAMERA_TARGET, fov: HOME_ENVIRONMENT_CAMERA.fov }
      : laboratoryDefaultCameraRef.current)
    if (!environmentInitializedRef.current || previous !== mode) {
      if (view) setEnvironmentCamera(view)
    }
    camera.updateProjectionMatrix()
    environmentInitializedRef.current = true
    const floorShadow = chairRef.current?.root.getObjectByName('ChairFloorShadow')
    if (floorShadow) floorShadow.visible = mode === 'laboratory'
  }, [setEnvironmentCamera])

  // Initialize Three.js scene (deferred to ensure layout is computed)
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    let resizeObserver: ResizeObserver | undefined
    const initId = requestAnimationFrame(() => {
      const w = container.clientWidth || 640
      const h = container.clientHeight || 480

      const scene = new THREE.Scene()
      scene.background = new THREE.Color(0x0f172a)
      sceneRef.current = scene

      const motionSpace = new THREE.Group()
      motionSpace.name = 'AvatarMotionSpace'
      motionSpaceRef.current = motionSpace
      scene.add(motionSpace)
      const sourceMotionSpace = new THREE.Group()
      sourceMotionSpace.name = 'AvatarSourceMotionSpace'
      sourceMotionSpaceRef.current = sourceMotionSpace
      motionSpace.add(sourceMotionSpace)
      const chair = createSceneChair()
      chairRef.current = chair
      chairWasFittedRef.current = false
      scene.add(chair.root)

      const camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 1000)
      camera.position.set(0, 100, 300)
      cameraRef.current = camera

      const renderer = new THREE.WebGLRenderer({ antialias: true })
      renderer.setSize(w, h)
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5))
      container.appendChild(renderer.domElement)
      rendererRef.current = renderer

      const controls = new OrbitControls(camera, renderer.domElement)
      controls.target.set(0, 80, 0)
      controls.update()
      controlsRef.current = controls

      const laboratory = new THREE.Group()
      laboratory.name = 'LaboratoryEnvironment'
      laboratoryEnvironmentRef.current = laboratory
      const grid = new THREE.GridHelper(24, 48, 0x334155, 0x1e293b)
      grid.name = 'LaboratoryGrid'
      laboratory.add(grid, new THREE.AmbientLight(0xffffff, 0.8))
      const dirLight = new THREE.DirectionalLight(0xffffff, 0.5)
      dirLight.position.set(100, 200, 100)
      laboratory.add(dirLight)
      scene.add(laboratory)
      const home = createHomeEnvironment()
      homeEnvironmentRef.current = home
      scene.add(home.root)
      const sunlight = new THREE.DirectionalLight(0xffeed5, 0.8)
      sunlight.name = 'HomeWindowSunlight'
      sunlight.position.set(-3, 4.5, -10)
      sunlight.target.position.set(1, 0, 2)
      sunlight.castShadow = true
      sunlight.shadow.mapSize.set(2048, 2048)
      sunlight.shadow.camera.left = sunlight.shadow.camera.bottom = -7
      sunlight.shadow.camera.right = sunlight.shadow.camera.top = 7
      sunlight.shadow.camera.near = 0.1
      sunlight.shadow.camera.far = 24
      sunlight.shadow.bias = -0.00015
      sunlight.shadow.normalBias = 0.025
      home.lights.add(sunlight, sunlight.target)
      const rendering = createHomeRendering(renderer, scene, camera)
      homeRenderingRef.current = rendering

      // Animation loop
      let frameCount = 0
      const animate = () => {
        animFrameRef.current = requestAnimationFrame(animate)

        // --- マスタークロック ---
        // 音声が鳴っている間は AudioContext の時計で進める。こうしないと
        // モーションと音声が別の時計で走り、同時に始めても必ずドリフトする。
        const wallDelta = clockRef.current.getDelta()
        const audioNow = getAudioTimeRef.current?.() ?? null
        let delta = wallDelta
        if (audioNow !== null) {
          const previous = lastAudioTimeRef.current
          // 初回、および AudioContext が作り直されて時刻が巻き戻った場合は
          // 壁時計の値をそのまま使う(1フレームだけ)。
          delta = previous !== null && audioNow >= previous ? audioNow - previous : wallDelta
          lastAudioTimeRef.current = audioNow
        } else {
          lastAudioTimeRef.current = null
        }

        // --- スケジュール済みモーションの発火 ---
        let motionStartedThisFrame = false
        const pendingDestination = pendingRoomDestinationRef.current
        const pendingPhase = roomTaskRef.current?.phase
        if (pendingDestination && ['failed', 'cancelled'].includes(pendingPhase ?? '')) pendingRoomDestinationRef.current = null
        if (pendingDestination && pendingPhase === 'idle' && persistentPoseRef.current.kind === 'standing' && isPlayingRef.current) {
          pendingRoomDestinationRef.current = null
          motionStartedThisFrame = requestRoomWalkRef.current?.(pendingDestination) === true
        }
        const phase = avatarAnimationRef.current?.phase
        const motionBusy = phase === 'motion' || phase === 'returning'
        if (
          audioNow !== null && isPlayingRef.current &&
          !roomTaskRef.current?.busy && motionQueueRef.current.length > 0
        ) {
          let motion: ScheduledMotion | undefined
          let startOffset = 0
          const active = activeScheduledRef.current
          // 層構成方式: 発話ごとに 1 本で、発話の時刻に張り付いている。
          // 前の発話の動作が残っていても、次の発話の開始時刻で切り替える(つなぎは見た目で補う)。
          // 遅れて届いた版は、音声の現在位置から再生する。
          const layeredDue = motionQueueRef.current.filter((m) => m.layered && audioNow >= m.startAt)
          if (layeredDue.length > 0 && (!motionBusy || active?.layered)) {
            const newest = layeredDue.reduce((a, b) =>
              b.startAt > a.startAt || (b.startAt === a.startAt && b.layered!.revision > a.layered!.revision) ? b : a)
            const stale = (m: ScheduledMotion) =>
              audioNow - m.startAt >= bvhDurationSeconds(m.bvh) ||
              (motionBusy && active?.layered && (m.startAt < active.startAt ||
                (m.layered!.group === active.layered.group && m.layered!.revision <= active.layered.revision)))
            for (const m of layeredDue) {
              if (m === newest && !stale(m)) continue
              appliedMotionIdsRef.current.add(m.id)
            }
            motionQueueRef.current = motionQueueRef.current.filter((m) => !appliedMotionIdsRef.current.has(m.id))
            if (!stale(newest)) {
              motion = newest
              startOffset = Math.max(audioNow - newest.startAt, 0)
            }
          } else if (!motionBusy) {
            // 動作を最後まで等速で再生してから、次の動作を開始する。
            // 遅れて届いた複数の動作も、途中で上書きしたり捨てたりしない。
            motion = motionQueueRef.current.find((m) => !m.layered && audioNow >= m.startAt)
          }
          if (motion) {
            motionQueueRef.current = motionQueueRef.current.filter((m) => m.id !== motion!.id)
            if (!appliedMotionIdsRef.current.has(motion.id)) {
              appliedMotionIdsRef.current.add(motion.id)
              console.log(
                `[SYNC] motion_applied perf=${performance.now().toFixed(1)} ` +
                `ctxNow=${audioNow.toFixed(3)} startAt=${motion.startAt.toFixed(3)} ` +
                `lag=${((audioNow - motion.startAt) * 1000).toFixed(0)}ms ` +
                `prompt="${motion.prompt}"` +
                (motion.layered ? ` layered=${motion.layered.group}#${motion.layered.revision} offset=${startOffset.toFixed(2)}` : '')
              )
              if (motion.layered && characterRef.current && viewModeRef.current === 'character') {
                const quaternions = new Map<THREE.Bone, THREE.Quaternion>()
                for (const link of characterRef.current.links) quaternions.set(link.target, link.target.quaternion.clone())
                const rootBone = characterRef.current.targetRootBone
                crossfadeRef.current = {
                  quaternions,
                  rootBone,
                  rootPosition: rootBone ? rootBone.position.clone() : null,
                  startedAt: performance.now(),
                }
              }
              activeScheduledRef.current = motion
              loadBvhRef.current?.(
                motion.bvh,
                motion.layered ? '🧩 層構成モーション' : '🤖 AI Generated Motion',
                motion.prompt,
                motion.segmentDuration,
                startOffset,
              )
              // このフレームの経過時間は動作開始前のものなので、
              // 新しいクリップへ加算して冒頭を飛ばさない。
              delta = 0
              motionStartedThisFrame = true
            }
          }
        }

        const roomController = roomTaskRef.current
        const roomOwnsPose = roomController?.ownsPose === true
        if (roomOwnsPose) roomController!.update(isPlayingRef.current && !motionStartedThisFrame ? wallDelta : 0)
        if (mixerRef.current && !roomOwnsPose) {
          mixerRef.current.update(delta)

          frameCount++
          if (frameCount % 120 === 0) {
            const rootBone = scene.userData.currentRootBone
            const mixerTime = mixerRef.current.time
            const timeScale = mixerRef.current.timeScale
            if (rootBone) {
              console.log(
                `[BvhViewer] heartbeat: mixerTime=${mixerTime.toFixed(2)}s, ` +
                `timeScale=${timeScale}, delta=${delta.toFixed(4)}s, ` +
                `rootBonePos=[${rootBone.position.x.toFixed(2)}, ${rootBone.position.y.toFixed(2)}, ${rootBone.position.z.toFixed(2)}], ` +
                `rootBoneQuat=[${rootBone.quaternion.x.toFixed(2)}, ${rootBone.quaternion.y.toFixed(2)}, ${rootBone.quaternion.z.toFixed(2)}, ${rootBone.quaternion.w.toFixed(2)}]`
              )
            } else {
              console.log(
                `[BvhViewer] heartbeat: mixerTime=${mixerTime.toFixed(2)}s, but no rootBone in scene.userData`
              )
            }
          }
        }

        // モーション終了後のニュートラル復帰と、常時の微細なIdleを更新する。
        // Mixerより後に適用することで、復帰中はIdle制御が最終姿勢を確実に上書きする。
        if (!roomOwnsPose) updateAvatarAnimation(delta)

        // Contact placement progresses even when the character is hidden.
        const contact = contactTransitionRef.current
        const character = characterRef.current
        if (contact && avatarAnimationRef.current?.phase === 'motion') {
          const progress = THREE.MathUtils.clamp((activeActionRef.current?.time ?? 0) / contact.duration, 0, 1)
          const blend = progress * progress * (3 - 2 * progress)
          characterFloorOffsetRef.current = THREE.MathUtils.lerp(contact.from, contact.to, blend)
          if (character) {
            character.root.position.y = character.tuning.yOffset -
              character.groundOffset * character.root.scale.y + characterFloorOffsetRef.current
          }
        }

        // キャラクターモード: BVH 骨格の現在ポーズをキャラへ毎フレーム転写する。
        if (!roomOwnsPose && viewModeRef.current === 'character') {
          if (character && character.links.length > 0) {
            try {
              retargetCharacter(character)
            } catch (e) {
              if (!retargetErroredRef.current) {
                retargetErroredRef.current = true
                console.error('[BvhViewer] character retarget failed:', e)
              }
            }
            // 層構成方式の動作を切り替えた直後は、切り替え前の見た目から寄せていく
            const fade = crossfadeRef.current
            if (fade) {
              const progress = Math.min((performance.now() - fade.startedAt) / LAYERED_CROSSFADE_MS, 1)
              const weight = progress * progress * (3 - 2 * progress)
              for (const [bone, from] of fade.quaternions) {
                _crossfadeQuaternion.copy(bone.quaternion)
                bone.quaternion.slerpQuaternions(from, _crossfadeQuaternion, weight)
              }
              if (fade.rootBone && fade.rootPosition) {
                _crossfadePosition.copy(fade.rootBone.position)
                fade.rootBone.position.lerpVectors(fade.rootPosition, _crossfadePosition, weight)
              }
              if (progress >= 1) crossfadeRef.current = null
            }
          }
        }

        if (!roomOwnsPose) applyFixedRoomContacts()

        // Force update skeleton helper geometry if needed (older or specific three.js versions)
        if (skeletonHelperRef.current) {
          skeletonHelperRef.current.update?.()
        }

        controls.update()
        rendering.render(delta)
      }
      animate()

      // Resize handler
      const handleResize = () => {
        const rw = container.clientWidth
        const rh = container.clientHeight
        if (rw === 0 || rh === 0) return
        camera.aspect = rw / rh
        camera.updateProjectionMatrix()
        renderer.setSize(rw, rh)
        rendering.resize(rw, rh)
      }
      resizeObserver = new ResizeObserver(handleResize)
      resizeObserver.observe(container)

      // Load default T-pose BVH
      if (typeof DEFAULT_BVH === 'string' && DEFAULT_BVH.length > 0) {
        try {
          const loader = new BVHLoader()
          const result = loader.parse(DEFAULT_BVH)
          const rootBone = result.skeleton.bones[0]
          sourceMotionSpace.add(rootBone)
          scene.userData.currentRootBone = rootBone
          // レストポーズは idle 姿勢の付与より前に採取しなければならない。
          sourceRigRef.current = createSourceRig(result.skeleton, result.clip)

          const skHelper = new THREE.SkeletonHelper(rootBone)
          const helperMat = skHelper.material as THREE.LineBasicMaterial
          helperMat.linewidth = 2
          helperMat.color = new THREE.Color(0x38bdf8)
          skHelper.visible = viewModeRef.current === 'skeleton'
          scene.add(skHelper)
          skeletonHelperRef.current = skHelper

          // デフォルト T-Pose にもロボットリグを attach
          buildRobotRig(rootBone)
          createAvatarAnimationState(rootBone, 'idle')

          // Idle initialization lifts the source hips to put the feet on the floor.
          // Treat that grounded position as the initial root-motion baseline,
          // so selecting a character does not transfer the lift a second time.
          sourceRigRef.current.groundedRootPosition.copy(rootBone.position)
          const chairSize = standingSkeletonHeight() / NORMALIZED_SKELETON_HEIGHT
          placeSceneChair(chair, new THREE.Vector3(0, 0, -0.32 * chairSize), 0.46 * chairSize, chairSize, 0)

          // Auto-fit camera
          scene.updateMatrixWorld(true)
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            ; (skHelper as any).update?.()

          const box = new THREE.Box3().setFromObject(skHelper)
          const bSize = new THREE.Vector3()
          const bCenter = new THREE.Vector3()
          box.getSize(bSize)
          box.getCenter(bCenter)
          let maxDim = Math.max(bSize.x, bSize.y, bSize.z)

          // Fallback if maxDim is still 0 or very small
          if (maxDim < 0.1) maxDim = 2.0

          const fitDist = maxDim * 2.5
          console.log('[BvhViewer] auto-fit dimensions:', { maxDim, fitDist, bCenter, bSize })

          camera.near = fitDist * 0.001
          camera.far = fitDist * 20
          camera.updateProjectionMatrix()
          camera.position.set(bCenter.x + fitDist * 0.5, bCenter.y + maxDim * 0.5, bCenter.z + fitDist)
          controls.target.copy(bCenter)
          controls.update()
          // 初期視点を確立したので、以降のモーション読込では視点をリセットしない
          hasFittedCameraRef.current = true

          laboratoryDefaultCameraRef.current = {
            position: camera.position.clone(), target: controls.target.clone(), fov: camera.fov,
          }
          applyEnvironment(environmentModeRef.current)

          // 実際のアバター読込は avatarUrl を監視する useEffect が行う。
          // (差し替え時にシーンを作り直さずに済ませるため)
          setSceneReady(true)
        } catch (e) {
          console.error('[BvhViewer] Failed to load default BVH:', e)
        }
      }
    }) // end requestAnimationFrame

    return () => {
      console.log('[BvhViewer] cleanup: disposing renderer / controls / animation frame')
      cancelAnimationFrame(initId)
      resizeObserver?.disconnect()
      cancelAnimationFrame(animFrameRef.current)
      if (controlsRef.current) controlsRef.current.dispose()
      roomTaskRef.current?.dispose()
      roomTaskContextRef.current?.controller.dispose()
      roomTaskContextRef.current = null
      roomTaskRef.current = null
      pendingRoomDestinationRef.current = null
      roomFixedSeatRef.current = null
      roomNavigationRef.current = null
      invalidateRoomImports()
      importedRoomRef.current?.dispose()
      importedRoomRef.current = null
      importedRoomCameraRef.current = null
      roomPlacementYawRef.current = 0
      roomMotionAnchorRef.current = false
      roomCharacterPivotRef.current = null
      activeActionRef.current?.stop()
      activeActionRef.current = null
      avatarAnimationRef.current = null
      homeRenderingRef.current?.dispose()
      homeRenderingRef.current = null
      homeEnvironmentRef.current?.lights.traverse((object) => {
        if (object instanceof THREE.DirectionalLight || object instanceof THREE.SpotLight) object.shadow.dispose()
      })
      homeEnvironmentRef.current?.dispose()
      homeEnvironmentRef.current = null
      if (laboratoryEnvironmentRef.current) {
        const labGrid = laboratoryEnvironmentRef.current.getObjectByName('LaboratoryGrid') as THREE.GridHelper
        labGrid.geometry.dispose()
        for (const material of Array.isArray(labGrid.material) ? labGrid.material : [labGrid.material]) material.dispose()
        laboratoryEnvironmentRef.current.removeFromParent()
        laboratoryEnvironmentRef.current = null
      }
      environmentInitializedRef.current = false
      environmentCamerasRef.current = {}
      if (chairRef.current) {
        disposeSceneChair(chairRef.current)
        chairRef.current = null
      }
      chairWasFittedRef.current = false
      chairStandingRootRef.current = null
      characterFloorOffsetRef.current = 0
      contactTransitionRef.current = null
      motionSpaceRef.current = null
      sourceMotionSpaceRef.current = null
      if (characterRef.current) {
        disposeCharacter(characterRef.current)
        characterRef.current = null
      }
      if (rendererRef.current) {
        rendererRef.current.dispose()
        if (container.contains(rendererRef.current.domElement)) {
          container.removeChild(rendererRef.current.domElement)
        }
      }
    }
    // buildRobotRig は useCallback で安定化されている。bvhText の更新は別 useEffect で処理する
  }, [applyFixedRoomContacts, buildRobotRig, createAvatarAnimationState, updateAvatarAnimation, fitCharacterToSkeleton, standingSkeletonHeight, applyEnvironment, invalidateRoomImports])

  useEffect(() => {
    if (sceneReady) applyEnvironment(environmentMode)
  }, [environmentMode, sceneReady, applyEnvironment])

  useEffect(() => {
    if (chairRef.current) chairRef.current.root.visible = showChair && !roomFixedSeatRef.current && !roomTaskRef.current?.ownsPose
  }, [showChair, sceneReady])

  // 一時停止からの復帰も常に等速にする。
  useEffect(() => {
    if (mixerRef.current) {
      mixerRef.current.timeScale = isPlaying ? 1 : 0
    }
  }, [isPlaying])

  const loadBvh = useCallback((
    text: string,
    name: string,
    prompt?: string | null,
    /** 対応する発話の長さ(秒)。計測用で、再生速度は変更しない。 */
    segmentDurationSeconds?: number,
    /** 層構成方式で遅れて始まった動作を、音声の現在位置から再生するための開始位置(秒)。 */
    startOffsetSeconds?: number,
  ) => {
    const scene = sceneRef.current
    const camera = cameraRef.current
    const controls = controlsRef.current
    const postureIntent = classifyMotionPrompt(prompt)
    const task = roomTaskRef.current
    if (task?.busy) return
    const destinationPrompt = (prompt ?? '').replace(/椅子に座(?:っている|った)状態から/g, '')
    if (name === '🤖 AI Generated Motion' && environmentModeRef.current === 'home' &&
      /歩|移動|向か|行く|行って/.test(destinationPrompt) && !/座る|座って|座り|腰掛/.test(destinationPrompt) && requestRoomWalkRef.current) {
      const destinationId = [...roomDestinationsRef.current]
        .sort((a, b) => b.label.length - a.label.length)
        .find((target) => [target.label, ...(target.aliases ?? [])].some((label) => label && destinationPrompt.includes(label)))?.id
      if (destinationId) { requestRoomWalkRef.current(destinationId); return }
    }
    if (roomFixedSeatRef.current && postureIntent.intent === 'sit' && postureIntent.posture !== 'chair_sitting') {
      roomTaskContextRef.current?.controller.dispose()
      roomTaskContextRef.current = null
      roomFixedSeatRef.current = null
    }
    // A conversational chair request in the home uses the selected fixed seat.
    // The same basic motions power the explicit task controls and this request.
    if (name === '🤖 AI Generated Motion' && environmentModeRef.current === 'home' &&
      postureIntent.intent === 'sit' && postureIntent.posture === 'chair_sitting' &&
      persistentPoseRef.current.kind === 'standing' && roomPrimitivesRef.current && startRoomTaskRef.current) {
      startRoomTaskRef.current()
      return
    }
    if (task?.phase === 'seated' && postureIntent.intent === 'stand') {
      task.stand()
      if (avatarAnimationRef.current) avatarAnimationRef.current.phase = 'motion'
      setIsPlaying(true)
      return
    }
    if (!task && roomFixedSeatRef.current && persistentPoseRef.current.kind === 'chair_sitting' &&
      postureIntent.intent === 'stand' && resumeRoomStandingRef.current?.()) return
    if (task) {
      if (task.phase === 'seated' && roomTaskContextRef.current?.controller === task) task.route.removeFromParent()
      else { task.dispose(); roomTaskContextRef.current = null }
      roomTaskRef.current = null
      if (scene) delete scene.userData.roomTaskController
    }
    if (roomFixedSeatRef.current && postureIntent.intent === 'stand' && roomPrimitivesRef.current) {
      text = roomPrimitivesRef.current.stand.text
      name = '家具からの起立'
      setRoomTask((held) => held ? { ...held, phase: 'standing' } : held)
    }
    if (postureIntent.intent === 'stand') roomFixedSeatRef.current = null
    const previousPosture = persistentPoseRef.current.kind
    // Use the same placement frame in all display modes. The hidden character
    // must first receive the current source pose so it is not a stale anchor.
    if (characterRef.current) {
      retargetCharacter(characterRef.current)
      characterRef.current.root.updateMatrixWorld(true)
    }
    const previousBones = characterRef.current?.bonesByCanonical ?? sourceRigRef.current?.byCanonical
    const previousHip = previousBones?.get('hips')
    scene?.updateMatrixWorld(true)
    const previousHipPosition = previousHip?.getWorldPosition(new THREE.Vector3())
    const preserveChairPosition = previousPosture === 'chair_sitting'

    if (
      preserveChairPosition && postureIntent.posture === 'chair_sitting' &&
      !/座った|座っている|座りながら|手|腕|首|頭|拍手|うなず|挨拶|振/.test(prompt ?? '')
    ) {
      // Sitting on the same chair again is already satisfied. Replaying the
      // standing-to-sitting clip would move the avatar behind the seat.
      console.log('[BvhViewer] chair sitting already held; keeping the seated pose')
      return
    }

    if (postureIntent.intent === 'stand') {
      clearPersistentPose()
      currentMotionEndBehaviorRef.current = 'return-to-standing-idle'
      pendingPostureKindRef.current = 'standing'
    } else if (postureIntent.intent === 'sit') {
      currentMotionEndBehaviorRef.current = 'hold-final-as-posture'
      pendingPostureKindRef.current = postureIntent.posture
    } else if (persistentPoseRef.current.kind !== 'standing') {
      currentMotionEndBehaviorRef.current = 'return-to-persistent-posture'
      pendingPostureKindRef.current = persistentPoseRef.current.kind
    } else {
      currentMotionEndBehaviorRef.current = 'return-to-standing-idle'
      pendingPostureKindRef.current = 'standing'
    }

    console.log(
      `[BvhViewer] loadBvh called: name="${name}", text_len=${text.length}, ` +
      `prompt="${prompt ?? ''}", postureIntent=${postureIntent.intent}/${postureIntent.posture}, ` +
      `endBehavior=${currentMotionEndBehaviorRef.current}, persistent=${persistentPoseRef.current.kind}, ` +
      `refs={scene:${!!scene}, camera:${!!camera}, controls:${!!controls}}`
    )
    if (!scene || !camera || !controls) {
      console.warn(
        '[BvhViewer] loadBvh aborted: scene/camera/controls not yet initialized — ' +
        'check init useEffect timing'
      )
      return
    }

    // Remove previous skeleton & root bone
    if (skeletonHelperRef.current) {
      scene.remove(skeletonHelperRef.current)
      skeletonHelperRef.current.dispose()
      skeletonHelperRef.current = null
    }

    if (scene.userData.currentRootBone) {
      scene.userData.currentRootBone.removeFromParent()
    }
    activeActionRef.current?.stop()
    activeActionRef.current = null
    avatarAnimationRef.current = null

    // Stop any previous mixer to avoid leaked actions
    if (mixerRef.current) {
      mixerRef.current.stopAllAction()
      mixerRef.current.uncacheRoot(mixerRef.current.getRoot())
      mixerRef.current = null
    }

    const loader = new BVHLoader()
    try {
      const head = text.slice(0, 120).replace(/\n/g, '\\n')
      console.log(
        `[BvhViewer] Parsing BVH text (length=${text.length}, head="${head}")`
      )
      const prepared = name === '家具からの起立' ? parseRoomMotionPrimitive('stand', text) : null
      const result = prepared ? { skeleton: prepared.skeleton, clip: prepared.clip } : loader.parse(text)
      // 供給元によって BVH の単位が違う。Custom-T2M-Lab はメートル(腰の高さ約0.9)、
      // SentiAvatar はセンチ(約102)で出してくる。そのまま読むと骨格もキャラも
      // グリッドも 100 倍になり、カメラの画角から外れて何も見えなくなる。
      // 軸をそろえてからスケールを測る(横倒しのまま測ると高さが体の幅になる)
      reorientBvhAxes(result)
      normalizeBvhScale(result)
      // A gesture after this task stays at the room's original actor height.
      if ((roomMotionAnchorRef.current || roomFixedSeatRef.current) && !prepared) normalizeRoomBvh(result)

      const rootBone = result.skeleton.bones[0]
      const motionSpace = motionSpaceRef.current
      const sourceMotionSpace = sourceMotionSpaceRef.current
      if (sourceMotionSpace) sourceMotionSpace.add(rootBone)
      else scene.add(rootBone)
      scene.userData.currentRootBone = rootBone
      // レストポーズ採取 → キャラクターへの再バインド。
      // BVH を読み込み直すとボーン実体が入れ替わるため、毎回張り直す必要がある。
      const sourceRig = createSourceRig(result.skeleton, result.clip)
      sourceRigRef.current = sourceRig
      if (sourceMotionSpace) {
        // Retargeted root motion already subtracts the clip's initial XZ.
        // Apply the same origin to the robot/skeleton representation.
        sourceMotionSpace.position.set(-sourceRig.groundedRootPosition.x, 0, -sourceRig.groundedRootPosition.z)
        sourceMotionSpace.updateMatrixWorld(true)
      }
      if (postureIntent.posture === 'chair_sitting' && previousPosture === 'standing') {
        chairStandingRootRef.current = {
          y: sourceRig.groundedRootPosition.y,
          height: standingSkeletonHeight(),
        }
      } else if (preserveChairPosition && chairStandingRootRef.current) {
        // A seated clip starts with a low hip position. Keep the original
        // standing baseline instead of transferring its first frame as standing.
        sourceRig.groundedRootPosition.y = chairStandingRootRef.current.y *
          standingSkeletonHeight() / chairStandingRootRef.current.height
      }
      if (characterRef.current) {
        bindCharacterToSource(characterRef.current, sourceRig)
      }

      const tracks = result.clip.tracks
      console.log(
        `[BvhViewer] BVH parsed: bones=${result.skeleton.bones.length}, ` +
        `clip.duration=${result.clip.duration}s, clip.tracks=${tracks.length}, ` +
        `firstTrackName="${tracks[0]?.name}", firstTrackValuesLen=${tracks[0]?.values.length}`
      )

      const skeletonHelper = new THREE.SkeletonHelper(rootBone)
      const mat = skeletonHelper.material as THREE.LineBasicMaterial
      mat.linewidth = 2
      mat.color = new THREE.Color(0x38bdf8)
      skeletonHelper.visible = viewModeRef.current === 'skeleton'
      scene.add(skeletonHelper)
      skeletonHelperRef.current = skeletonHelper

      // ロボット表示用の primitive をボーンに attach
      buildRobotRig(rootBone)
      // アニメーション適用前のローカル姿勢をIdleの基準として保存する。
      createAvatarAnimationState(rootBone, 'motion')

      // Auto-fit camera to skeleton size
      scene.updateMatrixWorld(true)
      skeletonHelper.update?.()

      const box = new THREE.Box3().setFromObject(skeletonHelper)
      const size = new THREE.Vector3()
      const center = new THREE.Vector3()
      box.getSize(size)
      box.getCenter(center)
      let maxDim = Math.max(size.x, size.y, size.z)
      if (maxDim < 0.1) maxDim = 2.0

      const fitDistance = maxDim * 2.5
      camera.near = environmentModeRef.current === 'home' ? 0.05 : fitDistance * 0.001
      camera.far = environmentModeRef.current === 'home' ? 120 : Math.max(120, fitDistance * 20)
      camera.updateProjectionMatrix()
      // 視点(カメラ位置・注視点)の初期化は最初のモーション読込時だけ行う。
      // 以降はユーザーが操作した視点をそのまま維持し、行動描画のたびにリセットしない。
      if (!hasFittedCameraRef.current) {
        camera.position.set(center.x + fitDistance * 0.5, center.y + maxDim * 0.5, center.z + fitDistance)
        controls.target.copy(center)
        controls.update()
        hasFittedCameraRef.current = true
      }

      // 新しい骨格の高さに合わせてキャラクターを再フィットする。
      fitCharacterToSkeleton()

      const mixer = new THREE.AnimationMixer(rootBone)
      mixer.timeScale = 1
      const action = mixer.clipAction(result.clip)
      // アニメーションを一度だけ再生し、最後のフレームで停止させる
      action.setLoop(THREE.LoopOnce, 1)
      action.clampWhenFinished = true
      contactTransitionRef.current = postureIntent.intent === 'stand'
        ? { from: characterFloorOffsetRef.current, to: 0, duration: Math.max(result.clip.duration, 0.001) }
        : null

      action.play()
      mixer.update(0)
      if (characterRef.current) {
        retargetCharacter(characterRef.current)
        characterRef.current.root.updateMatrixWorld(true)
      }
      if (chairWasFittedRef.current && previousHipPosition && motionSpace) {
        const hip = (characterRef.current?.bonesByCanonical ?? sourceRig.byCanonical).get('hips')
        const firstHip = hip?.getWorldPosition(new THREE.Vector3())
        if (firstHip) {
          motionSpace.position.x += previousHipPosition.x - firstHip.x
          motionSpace.position.z += previousHipPosition.z - firstHip.z
          fitCharacterToSkeleton()
          motionSpace.updateMatrixWorld(true)
        }
      }
      if (!roomFixedSeatRef.current && postureIntent.posture === 'chair_sitting' && previousPosture !== 'chair_sitting') {
        // Predict the final contact before playback, then leave the chair fixed.
        // Add the finish listener only after this temporary sampling pass.
        mixer.setTime(result.clip.duration)
        // Account for the same floor clearance applied when saving the final
        // source pose, so the predicted seat remains correct after completion.
        const state = avatarAnimationRef.current as AvatarAnimationState | null
        if (state) {
          const bounds = new THREE.Box3().setFromObject(rootBone)
          if (Number.isFinite(bounds.min.y)) {
            rootBone.position.y += Math.max(0, state.scale * 0.015 - bounds.min.y)
            rootBone.updateMatrixWorld(true)
          }
        }
        if (characterRef.current) {
          retargetCharacter(characterRef.current)
          characterRef.current.root.updateMatrixWorld(true)
          const from = characterFloorOffsetRef.current
          const to = from - characterFloorGap()
          contactTransitionRef.current = { from, to, duration: Math.max(result.clip.duration, 0.001) }
          characterRef.current.root.position.y += to - from
          characterRef.current.root.updateMatrixWorld(true)
        }
        fitChairToDisplayedPose()
        action.stop()
        mixer.setTime(0)
        action.reset().play()
        mixer.update(0)
      }

      // 発話の長さや生成の遅れにかかわらず、BVH 本来の速度で再生する。
      if (segmentDurationSeconds && segmentDurationSeconds > 0.2 && result.clip.duration > 0) {
        console.log(
          `[BvhViewer] motion timing: clip=${result.clip.duration.toFixed(2)}s ` +
          `speech=${segmentDurationSeconds.toFixed(2)}s ` +
          'timeScale=1.00'
        )
      }

      action.play()
      if (startOffsetSeconds && startOffsetSeconds > 0.02) {
        mixer.setTime(Math.min(startOffsetSeconds, result.clip.duration))
      }
      activeActionRef.current = action
      mixerRef.current = mixer
      applyFixedRoomContacts()
      mixer.addEventListener('finished', () => {
        beginReturnToIdle(rootBone)
      })
      // 無動作率の算出に使う。実際に再生される尺は clip 長 ÷ timeScale。
      console.log(
        `[SYNC] motion_playing clip=${result.clip.duration.toFixed(2)} ` +
        `timeScale=${mixer.timeScale.toFixed(2)} ` +
        `played=${(result.clip.duration / (mixer.timeScale || 1)).toFixed(2)}`
      )
      console.log(
        `[BvhViewer] Mixer started: isRunning=${action.isRunning()}, ` +
        `enabled=${action.enabled}, weight=${action.weight}, ` +
        `clipDuration=${result.clip.duration}s, mixer.timeScale=${mixer.timeScale}`
      )

      setHasFile(true)
      setFileName(name)
      setIsPlaying(true)
      clockRef.current = new THREE.Clock()
      lastAudioTimeRef.current = getAudioTimeRef.current?.() ?? null
    } catch (e) {
      console.error('[BvhViewer] Error parsing BVH text:', e)
    }
  }, [applyFixedRoomContacts, beginReturnToIdle, buildRobotRig, characterFloorGap, clearPersistentPose, createAvatarAnimationState, fitCharacterToSkeleton, fitChairToDisplayedPose, standingSkeletonHeight])

  useEffect(() => {
    if (!sceneReady || !homeEnvironmentRef.current) return
    let cancelled = false
    Promise.all([homeEnvironmentRef.current.ready, loadRoomMotionPrimitives()]).then(([, primitives]) => {
      if (cancelled || !homeEnvironmentRef.current) return
      const room = readRoomNavigation(importedRoomRef.current?.root ?? homeEnvironmentRef.current.root)
      roomNavigationRef.current = room
      roomPrimitivesRef.current = primitives
      setRoomChairs(room.chairs)
      const destinations = createRoomDestinations(room)
      roomDestinationsRef.current = destinations
      setRoomDestinations(destinations)
      setRoomTaskReady(true)
    }).catch((error) => {
      if (!cancelled) setRoomTaskError(`動作素材の読込に失敗しました: ${String(error)}`)
    })
    return () => { cancelled = true }
  }, [sceneReady])

  const installRoomDefinition = useCallback((definition: RoomSceneDefinition | null, choice: string) => {
    const scene = sceneRef.current
    const home = homeEnvironmentRef.current
    if (!scene || !home || !roomPrimitivesRef.current || roomTaskRef.current?.busy) return false
    let next: RoomScene | null = null
    try {
      if (definition) next = createRoomScene(parseRoomScene(JSON.stringify(definition)), { floorMaterial: home.floor.material as THREE.Material })
      const room = readRoomNavigation(next?.root ?? home.root)
      const spawn = next?.spawn ?? { x: 0, z: 0, heading: 0 }
      if (!isRoomPointWalkable(room, spawn)) throw new Error('開始位置の周囲に歩ける床がありません')
      const destinations = createRoomDestinations(room)
      // Complete validation before retiring the previous room or task.
      pendingRoomDestinationRef.current = null
      for (const motion of motionQueueRef.current) appliedMotionIdsRef.current.add(motion.id)
      motionQueueRef.current = []
      roomTaskRef.current?.dispose()
      roomTaskContextRef.current?.controller.dispose()
      roomTaskRef.current = null
      roomTaskContextRef.current = null
      delete scene.userData.roomTaskController
      delete scene.userData.roomTask
      roomFixedSeatRef.current = null
      clearPersistentPose()
      chairWasFittedRef.current = false
      activeActionRef.current?.stop()
      activeActionRef.current = null
      mixerRef.current?.stopAllAction()
      contactTransitionRef.current = null
      const state = avatarAnimationRef.current
      if (state) {
        for (const item of state.bones) {
          const upright = roomStandingPoseRef.current.get(item.bone.name)
          item.bone.position.copy(upright?.position ?? item.restPosition)
          item.bone.quaternion.copy(upright?.quaternion ?? item.restQuaternion)
        }
        state.phase = 'idle'
        state.poseKind = 'standing'
      }
      const source = sourceRigRef.current
      if (source) {
        sourceMotionSpaceRef.current?.position.set(-source.groundedRootPosition.x, 0, -source.groundedRootPosition.z)
        groundRoomFeet(source.byCanonical)
      }
      roomPlacementYawRef.current = spawn.heading ?? 0
      roomCharacterPivotRef.current = null
      avatarYawRef.current = 0
      setAvatarYaw(0)
      motionSpaceRef.current?.position.set(spawn.x, 0, spawn.z)
      if (motionSpaceRef.current) motionSpaceRef.current.rotation.y = roomPlacementYawRef.current
      const character = characterRef.current
      if (character) {
        character.tuning.yawDegrees = 0
        characterFloorOffsetRef.current = 0
        fitCharacterToSkeleton()
        retargetCharacter(character)
        groundSeatedCharacter()
      }
      if (state) {
        for (const item of state.bones) {
          item.returnPosition.copy(item.bone.position)
          item.returnQuaternion.copy(item.bone.quaternion)
        }
        saveCurrentPoseAsPersistent(state, 'standing', true)
      }
      importedRoomRef.current?.dispose()
      importedRoomRef.current = next
      if (next) scene.add(next.root)
      roomNavigationRef.current = room
      roomDestinationsRef.current = destinations
      setRoomDestinations(destinations)
      setRoomDestinationId(destinations[0]?.id ?? '')
      setRoomChairs(room.chairs)
      setRoomChairId(room.defaultChairId)
      setRoomTask(null)
      setRoomTaskError('')
      setRoomSceneError('')
      setRoomSceneChoice(choice)
      setRoomSceneName(next?.definition.name ?? 'リビング・ダイニング・キッチン')
      if (next) {
        const center = new THREE.Vector3((room.bounds.minX + room.bounds.maxX) / 2, 0.8, (room.bounds.minZ + room.bounds.maxZ) / 2)
        const size = Math.max(room.bounds.maxX - room.bounds.minX, room.bounds.maxZ - room.bounds.minZ)
        importedRoomCameraRef.current = { position: center.clone().add(new THREE.Vector3(size * 0.55, size * 0.7, size * 0.72)), target: center, fov: 55 }
      } else importedRoomCameraRef.current = null
      delete environmentCamerasRef.current.home
      setEnvironmentMode('home')
      applyEnvironment('home')
      setEnvironmentCamera(importedRoomCameraRef.current ?? { position: HOME_CAMERA_POSITION, target: HOME_CAMERA_TARGET, fov: HOME_ENVIRONMENT_CAMERA.fov })
      scene.userData.roomNavigation = room
      if (chairRef.current) chairRef.current.root.visible = false
      setIsPlaying(true)
      isPlayingRef.current = true
      if (mixerRef.current) mixerRef.current.timeScale = 1
      clockRef.current = new THREE.Clock()
      return true
    } catch (error) {
      next?.dispose()
      setRoomSceneError(error instanceof Error ? error.message : String(error))
      return false
    }
  }, [applyEnvironment, clearPersistentPose, fitCharacterToSkeleton, groundSeatedCharacter, saveCurrentPoseAsPersistent, setEnvironmentCamera])

  const selectRoomScene = useCallback((choice: string) => {
    roomLoadRequestRef.current++
    setRoomSceneLoading(false)
    const example = ROOM_SCENE_EXAMPLES.find((candidate) => candidate.id === choice)
    installRoomDefinition(example?.definition ?? null, choice)
  }, [installRoomDefinition])

  const handleRoomFile = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    const request = ++roomLoadRequestRef.current
    setRoomSceneLoading(true)
    setRoomSceneError('')
    try {
      if (file.size > 1024 * 1024) throw new Error('部屋データは1MB以内にしてください')
      const definition = parseRoomScene(await file.text())
      if (request !== roomLoadRequestRef.current) return
      if (!installRoomDefinition(definition, 'imported')) setRoomSceneLoading(false)
    } catch (error) {
      if (request === roomLoadRequestRef.current) setRoomSceneError(error instanceof Error ? error.message : String(error))
    } finally {
      if (request === roomLoadRequestRef.current) setRoomSceneLoading(false)
    }
  }, [installRoomDefinition])

  const startRoomTask = useCallback((destinationId?: string) => {
    const scene = sceneRef.current
    const home = homeEnvironmentRef.current
    const cached = roomPrimitivesRef.current
    const placement = motionSpaceRef.current
    const sourcePlacement = sourceMotionSpaceRef.current
    if (!scene || !home || !cached || !placement || !sourcePlacement || roomTaskRef.current?.busy) return false
    try {
      const room = readRoomNavigation(importedRoomRef.current?.root ?? home.root)
      roomNavigationRef.current = room
      scene.updateMatrixWorld(true)
      const previousBones = displayedBones()
      const previousHip = previousBones?.get('hips')?.getWorldPosition(new THREE.Vector3()) ?? placement.position.clone()
      const start = { x: previousHip.x, z: previousHip.z }
      const startHeading = roomTaskRef.current?.heading ?? placement.rotation.y
      const destination = destinationId ? createRoomDestinations(room).find((target) => target.id === destinationId) : null
      if (destinationId && !destination) throw new Error('指定した移動先の周囲に安全な場所がありません')
      const seatPlan = destination ? null : planChairApproach(room, start, roomChairId)
      const plan = destination ? planRoomPath(room, start, destination.position) : seatPlan!
      if (!plan.ok) throw new Error(`${destination?.label ?? '椅子'}まで移動できる経路がありません (${plan.reason})`)
      const seatChair = seatPlan?.ok ? seatPlan.chair : null
      roomTaskRef.current?.dispose()
      roomTaskContextRef.current?.controller.dispose()
      roomTaskContextRef.current = null
      roomTaskRef.current = null
      roomFixedSeatRef.current = null
      roomMotionAnchorRef.current = true
      setRoomTaskError('')
      clearPersistentPose()
      activeActionRef.current?.stop()
      activeActionRef.current = null
      mixerRef.current?.stopAllAction()
      mixerRef.current?.uncacheRoot(mixerRef.current.getRoot())
      scene.userData.currentRootBone?.removeFromParent()
      skeletonHelperRef.current?.removeFromParent()
      skeletonHelperRef.current?.dispose()
      disposeRobotRig()
      const primitives = { ...cached, walk: parseRoomMotionPrimitive('walk', cached.walk.text) }
      const root = primitives.walk.root
      sourcePlacement.position.set(0, 0, 0)
      sourcePlacement.add(root)
      scene.userData.currentRootBone = root
      const source = createSourceRig(primitives.walk.skeleton, primitives.walk.clip)
      source.groundedRootPosition.fromArray(primitives.walk.metadata.restGroundedRootPosition)
      sourceRigRef.current = source
      roomPlacementYawRef.current = 0
      roomCharacterPivotRef.current = null
      avatarYawRef.current = 0
      setAvatarYaw(0)
      placement.position.set(start.x, 0, start.z)
      placement.rotation.y = 0
      const character = characterRef.current
      if (character) {
        character.tuning.yawDegrees = 0
        bindCharacterToSource(character, source)
      }
      const helper = new THREE.SkeletonHelper(root)
      ;(helper.material as THREE.LineBasicMaterial).color.set(0x38bdf8)
      helper.visible = viewModeRef.current === 'skeleton'
      scene.add(helper)
      skeletonHelperRef.current = helper
      buildRobotRig(root)
      createAvatarAnimationState(root, 'idle')
      groundRoomFeet(source.byCanonical)
      roomStandingPoseRef.current = new Map([...source.byCanonical.values()].map((bone) => [bone.name, { position: bone.position.clone(), quaternion: bone.quaternion.clone() }]))
      characterFloorOffsetRef.current = 0
      contactTransitionRef.current = null
      fitCharacterToSkeleton()
      if (character) {
        retargetCharacter(character)
        groundSeatedCharacter()
        const hip = character.bonesByCanonical.get('hips')?.getWorldPosition(new THREE.Vector3())
        if (hip) roomCharacterPivotRef.current = hip.sub(character.root.position).applyAxisAngle(new THREE.Vector3(0, 1, 0), -character.root.rotation.y)
      }
      roomFootReferencesRef.current = { source: captureStandingFootContacts(source.byCanonical), character: character ? captureStandingFootContacts(character.bonesByCanonical) : null }
      const mixer = new THREE.AnimationMixer(root)
      mixer.timeScale = 1
      mixerRef.current = mixer
      const state = avatarAnimationRef.current!
      state.phase = 'motion'
      const bridge: RoomTaskBridge = {
        mixer, sourceBones: source.byCanonical,
        getDisplayBones: () => displayedBones() ?? source.byCanonical,
        getCharacterBones: () => characterRef.current?.bonesByCanonical,
        place: (point, yaw) => {
          roomPlacementYawRef.current = yaw
          placement.position.set(point.x, 0, point.z)
          placement.rotation.y = yaw
          placement.updateMatrixWorld(true)
          const actor = characterRef.current
          if (actor) {
            actor.root.rotation.y = THREE.MathUtils.degToRad(actor.autoYawDegrees + actor.tuning.yawDegrees) + yaw
            const offset = roomCharacterPivotRef.current?.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), actor.root.rotation.y) ?? new THREE.Vector3()
            actor.root.position.x = point.x - offset.x
            actor.root.position.z = point.z - offset.z
            actor.root.updateMatrixWorld(true)
          }
        },
        retarget: () => {
          placement.updateMatrixWorld(true)
          if (characterRef.current) retargetCharacter(characterRef.current)
        },
        hold: (kind) => {
          const idle = avatarAnimationRef.current
          if (!idle) return
          for (const item of idle.bones) {
            item.returnPosition.copy(item.bone.position)
            item.returnQuaternion.copy(item.bone.quaternion)
          }
          // Room controllers already ground the source and avatar using their
          // foot contacts. Robot mesh bounds must not lift that pose again.
          saveCurrentPoseAsPersistent(idle, kind, true)
          idle.phase = 'idle'
          roomFixedSeatRef.current = kind === 'chair_sitting' ? seatChair : null
          chairWasFittedRef.current = kind === 'chair_sitting'
          chairStandingRootRef.current = { y: source.groundedRootPosition.y, height: standingSkeletonHeight() }
        },
        restoreStanding: () => {
          clearPersistentPose()
          roomFixedSeatRef.current = null
          chairWasFittedRef.current = false
          const idle = avatarAnimationRef.current
          if (!idle) return
          for (const item of idle.bones) {
            const upright = roomStandingPoseRef.current.get(item.bone.name)
            if (upright) {
              item.bone.position.copy(upright.position)
              item.bone.quaternion.copy(upright.quaternion)
              item.restPosition.copy(upright.position)
              item.restQuaternion.copy(upright.quaternion)
            }
          }
          idle.poseKind = 'standing'
          idle.phase = 'idle'
        },
        notify: (snapshot) => { scene.userData.roomTask = snapshot; setRoomTask(snapshot) },
      }
      const controller = destination
        ? new RoomWalkController(primitives, room, bridge, start, destination, startHeading)
        : new RoomTaskController(primitives, room, bridge, start, roomChairId, startHeading)
      roomTaskRef.current = controller
      roomTaskContextRef.current = controller instanceof RoomTaskController ? { controller, source, root, state, mixer } : null
      scene.userData.roomTaskController = controller
      controller.route.visible = showRoomRoute
      controller.route.userData.showRoute = showRoomRoute
      scene.add(controller.route)
      if (chairRef.current) chairRef.current.root.visible = false
      setFileName(destination ? `${destination.label}へ移動（T2M）` : '家具への歩行・着座（T2M）')
      setIsPlaying(true)
      isPlayingRef.current = true
      clockRef.current = new THREE.Clock()
      // First frame starts at zero, with the character's hip anchored to the route.
      controller.update(0)
      return true
    } catch (error) {
      setRoomTaskError(error instanceof Error ? error.message : String(error))
      console.error('[BvhViewer] room task failed:', error)
      return false
    }
  }, [roomChairId, showRoomRoute, buildRobotRig, clearPersistentPose, createAvatarAnimationState, displayedBones, disposeRobotRig, fitCharacterToSkeleton, groundSeatedCharacter, saveCurrentPoseAsPersistent, standingSkeletonHeight])

  const resumeRoomStanding = useCallback(() => {
    const context = roomTaskContextRef.current
    const scene = sceneRef.current
    const sourcePlacement = sourceMotionSpaceRef.current
    if (!context || !scene || !sourcePlacement || context.controller.phase !== 'seated' || persistentPoseRef.current.kind !== 'chair_sitting') return false
    // Restore the original room controller after a conversation gesture, so
    // standing follows the same contact blend and safe exit as the first task.
    activeActionRef.current?.stop()
    activeActionRef.current = null
    mixerRef.current?.stopAllAction()
    mixerRef.current?.uncacheRoot(mixerRef.current.getRoot())
    scene.userData.currentRootBone?.removeFromParent()
    skeletonHelperRef.current?.removeFromParent()
    skeletonHelperRef.current?.dispose()
    sourcePlacement.position.set(0, 0, 0)
    sourcePlacement.add(context.root)
    scene.userData.currentRootBone = context.root
    sourceRigRef.current = context.source
    avatarAnimationRef.current = context.state
    context.state.phase = 'motion'
    mixerRef.current = context.mixer
    context.mixer.timeScale = 1
    const helper = new THREE.SkeletonHelper(context.root)
    ;(helper.material as THREE.LineBasicMaterial).color.set(0x38bdf8)
    helper.visible = viewModeRef.current === 'skeleton'
    scene.add(helper)
    skeletonHelperRef.current = helper
    buildRobotRig(context.root)
    if (characterRef.current) bindCharacterToSource(characterRef.current, context.source)
    fitCharacterToSkeleton()
    roomTaskRef.current = context.controller
    scene.userData.roomTaskController = context.controller
    context.controller.refreshCharacterContacts(roomFootReferencesRef.current.character)
    context.controller.route.visible = showRoomRoute && environmentModeRef.current === 'home'
    scene.add(context.controller.route)
    context.controller.stand()
    setIsPlaying(true)
    isPlayingRef.current = true
    clockRef.current = new THREE.Clock()
    return true
  }, [buildRobotRig, fitCharacterToSkeleton, showRoomRoute])

  const requestRoomWalk = useCallback((destinationId: string) => {
    if (environmentModeRef.current !== 'home' || roomTaskRef.current?.busy) return false
    if (persistentPoseRef.current.kind === 'chair_sitting') {
      pendingRoomDestinationRef.current = destinationId
      const task = roomTaskRef.current
      const standing = task?.phase === 'seated' ? task.stand() : resumeRoomStandingRef.current?.()
      if (!standing) {
        pendingRoomDestinationRef.current = null
        setRoomTaskError('先に立ち上がってから移動してください')
        return false
      }
      if (avatarAnimationRef.current) avatarAnimationRef.current.phase = 'motion'
      setIsPlaying(true)
      isPlayingRef.current = true
      clockRef.current = new THREE.Clock()
      return true
    }
    if (persistentPoseRef.current.kind !== 'standing') {
      setRoomTaskError('先に立ち上がってから移動してください')
      return false
    }
    return startRoomTask(destinationId)
  }, [startRoomTask])

  useEffect(() => { requestRoomWalkRef.current = requestRoomWalk }, [requestRoomWalk])
  useEffect(() => { startRoomTaskRef.current = startRoomTask }, [startRoomTask])
  useEffect(() => { resumeRoomStandingRef.current = resumeRoomStanding }, [resumeRoomStanding])

  useEffect(() => {
    const route = roomTaskRef.current?.route
    if (route) { route.userData.showRoute = showRoomRoute; route.visible = showRoomRoute && environmentMode === 'home' }
  }, [showRoomRoute, environmentMode])

  // public/avatars 配下の .glb を列挙してセレクタに並べる(/avatar-list は Next のルート)。
  // UE から書き出した glb をそのフォルダへ置けば、ここに自動で現れる。
  useEffect(() => {
    let cancelled = false
    fetch('/avatar-list')
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(String(res.status)))))
      .then((data: { avatars?: { label: string; url: string }[] }) => {
        if (cancelled) return
        const options = data.avatars ?? []
        if (!options.length) return
        setAvatarOptions(current => [
          ...options.map(option => option.url === PRESIDENT_AVATAR.url ? PRESIDENT_AVATAR : option),
          ...current.filter(option => option.url.startsWith('blob:')),
        ])
        if (!options.some(option => option.url === PRESIDENT_AVATAR.url)) {
          const fallback = options.find(option => option.url === CHARACTER_MODEL_URL) ?? options[0]
          setAvatarUrl(current => current === PRESIDENT_AVATAR.url ? fallback.url : current)
        }
      })
      .catch((err) => {
        console.warn('[BvhViewer] avatar list unavailable:', err)
      })
    return () => {
      cancelled = true
    }
  }, [])

  // シーン初期化完了後、および選択アバター変更時にモデルを差し替える。
  useEffect(() => {
    if (!sceneReady) return
    let current = true
    void applyAvatar(avatarUrl, () => current)
    return () => { current = false }
  }, [sceneReady, avatarUrl, applyAvatar])

  // 向き調整はモデルを読み直さず、ラッパーの回転だけ更新する。
  useEffect(() => {
    avatarYawRef.current = avatarYaw
    const motionSpace = motionSpaceRef.current
    const yaw = THREE.MathUtils.degToRad(avatarYaw) + roomPlacementYawRef.current
    if (motionSpace) {
      const delta = yaw - motionSpace.rotation.y
      const chair = chairRef.current
      if (chair && Math.abs(delta) > 1e-9) {
        chair.root.position.sub(motionSpace.position)
          .applyAxisAngle(new THREE.Vector3(0, 1, 0), delta)
          .add(motionSpace.position)
        chair.root.rotation.y += delta
        chair.root.updateMatrixWorld(true)
      }
      motionSpace.rotation.y = yaw
      motionSpace.updateMatrixWorld(true)
    }
    const character = characterRef.current
    if (!character) return
    character.tuning.yawDegrees = avatarYaw
    fitCharacterToSkeleton()
  }, [avatarYaw, fitCharacterToSkeleton, sceneReady])

  // blob URL の後始末
  useEffect(() => {
    return () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current)
    }
  }, [])

  // viewMode の切替: 既存の SkeletonHelper / ロボットリグの visible を更新
  useEffect(() => {
    if (skeletonHelperRef.current) {
      skeletonHelperRef.current.visible = viewMode === 'skeleton'
    }
    for (const m of robotRigPartsRef.current) {
      m.visible = viewMode === 'robot'
    }
    if (characterRef.current) {
      characterRef.current.root.visible = viewMode === 'character'
      if (viewMode === 'character') {
        fitCharacterToSkeleton()
        retargetCharacter(characterRef.current)
        characterRef.current.root.updateMatrixWorld(true)
      }
    }
    if (viewMode === 'character' && persistentPoseRef.current.kind === 'chair_sitting' && !roomFixedSeatRef.current) {
      groundSeatedCharacter()
      fitChairToDisplayedPose()
    }
    console.log(`[BvhViewer] viewMode changed → '${viewMode}'`)
  }, [viewMode, fitCharacterToSkeleton, fitChairToDisplayedPose, groundSeatedCharacter])

  // animate ループから最新の loadBvh / getAudioTime を呼べるようにする
  useEffect(() => {
    loadBvhRef.current = loadBvh
  }, [loadBvh])

  useEffect(() => {
    getAudioTimeRef.current = getAudioTime
  }, [getAudioTime])

  // 姿勢(立位・座位)の変化を親へ知らせる。代入箇所が多いので、軽く見張る。
  useEffect(() => {
    if (!onPostureChange) return
    let last: PostureKind | null = null
    const timer = window.setInterval(() => {
      const kind = persistentPoseRef.current.kind
      if (kind !== last) {
        last = kind
        onPostureChange(kind)
      }
    }, 500)
    return () => window.clearInterval(timer)
  }, [onPostureChange])

  // スケジュール済みモーションを再生キューへ流し込む。
  // すでに適用済みのものは弾き、時刻順に並べる。
  useEffect(() => {
    if (!scheduledMotions) return
    if (scheduledMotions.length === 0) {
      motionQueueRef.current = []
      appliedMotionIdsRef.current.clear()
      return
    }
    const applied = appliedMotionIdsRef.current
    // 層構成方式は同じ発話(group)の新しい版で置き換える
    const newestRevision = new Map<string, number>()
    for (const m of scheduledMotions) {
      if (m.layered) newestRevision.set(m.layered.group, Math.max(newestRevision.get(m.layered.group) ?? -1, m.layered.revision))
    }
    motionQueueRef.current = scheduledMotions
      .filter((m) => !applied.has(m.id))
      .filter((m) => !m.layered || m.layered.revision >= (newestRevision.get(m.layered.group) ?? 0))
      .sort((a, b) => a.startAt - b.startAt)
    console.log(
      `[BvhViewer] motion queue updated: ${motionQueueRef.current.length} 件待機`
    )
  }, [scheduledMotions])

  // 旧経路: スケジュール情報が無いときだけ、届いた瞬間に再生する。
  // (BVH ファイルの手動アップロードや、scheduledMotions 未接続の呼び出し元向け)
  useEffect(() => {
    if (scheduledMotions) return
    console.log(
      `[BvhViewer] bvhText prop changed: ${bvhText ? `length=${bvhText.length}` : 'null/undefined'
      }`
    )
    if (bvhText && bvhText.length > 0) {
      loadBvh(bvhText, '🤖 AI Generated Motion', motionPrompt)
    }
  }, [bvhText, motionPrompt, loadBvh, scheduledMotions])

  /** ローカルの .glb をその場で読み込む(サーバーへ置く前の確認用)。 */
  const handleAvatarFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current)
    const url = URL.createObjectURL(file)
    objectUrlRef.current = url
    setAvatarOptions((prev) =>
      prev.some((o) => o.url === url) ? prev : [...prev, { label: `📁 ${file.name}`, url }],
    )
    setAvatarUrl(url)
    setViewMode('character')
  }, [])

  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        if (roomTaskRef.current?.busy) { pendingRoomDestinationRef.current = null; roomTaskRef.current.cancel() }
        if (!roomFixedSeatRef.current) roomMotionAnchorRef.current = false
        loadBvh(reader.result, file.name, file.name)
      }
    }
    reader.readAsText(file)
  }, [loadBvh])

  const btnStyle: React.CSSProperties = {
    padding: '6px 14px',
    borderRadius: 6,
    border: '1px solid rgba(148,163,184,0.3)',
    background: 'rgba(30,41,59,0.8)',
    color: '#e2e8f0',
    fontSize: 12,
    cursor: 'pointer',
    transition: 'background 0.2s',
  }

  const roomSelectionDisabled = !roomTaskReady || roomSceneLoading || roomTaskRef.current?.busy
  const settings = (
    <>
      <fieldset>
        <legend>アバターの表示</legend>
        <label>
          表示モデル
          <select value={viewMode} onChange={(event) => setViewMode(event.target.value as ViewMode)}>
            <option value="robot">ロボット</option>
            <option value="character">キャラクター</option>
            <option value="skeleton">骨格</option>
          </select>
        </label>
        {viewMode === 'character' && <>
          <label>
            アバターファイル
            <select value={avatarUrl} onChange={(event) => setAvatarUrl(event.target.value)}>
              {avatarOptions.map((option) => <option key={option.url} value={option.url}>{option.label}</option>)}
            </select>
          </label>
          <label>
            ローカルのアバターを開く
            <input type="file" accept=".glb,.gltf,model/gltf-binary" onChange={handleAvatarFile} />
          </label>
          <label>
            アバターの向き
            <select value={avatarYaw} disabled={roomTaskRef.current?.ownsPose || !!roomFixedSeatRef.current}
              onChange={(event) => setAvatarYaw(Number(event.target.value))}>
              {[0, 90, 180, 270].map((deg) => <option key={deg} value={deg}>{deg}°</option>)}
            </select>
          </label>
          <small role="status">{avatarStatus}</small>
        </>}
      </fieldset>
      <fieldset>
        <legend>空間</legend>
        <label>
          表示する空間
          <select value={environmentMode} onChange={(event) => setEnvironmentMode(event.target.value as EnvironmentMode)}>
            <option value="home">住宅</option>
            <option value="laboratory">検証空間</option>
          </select>
        </label>
        {environmentMode === 'home' && <>
          <label>
            部屋
            <select value={roomSceneChoice} disabled={roomSelectionDisabled}
              onChange={(event) => selectRoomScene(event.target.value)}>
              <option value="home">元の住宅</option>
              {ROOM_SCENE_EXAMPLES.map((example) => <option key={example.id} value={example.id}>{example.label}</option>)}
              {roomSceneChoice === 'imported' && <option value="imported">{roomSceneName}</option>}
            </select>
          </label>
          <label>
            部屋データを開く
            <input type="file" accept=".json,application/json" disabled={roomSelectionDisabled} onChange={handleRoomFile} />
          </label>
          {roomSceneLoading && <small role="status">部屋を読込中…</small>}
          {roomSceneError && <small role="alert">{roomSceneError}</small>}
          <button type="button" onClick={() => {
            const definition = importedRoomRef.current?.definition ?? ROOM_SCENE_EXAMPLES[0].definition
            const url = URL.createObjectURL(new Blob([JSON.stringify(definition, null, 2)], { type: 'application/json' }))
            const anchor = document.createElement('a')
            anchor.href = url
            anchor.download = 'room.json'
            anchor.click()
            setTimeout(() => URL.revokeObjectURL(url), 1000)
          }}>{roomSceneChoice === 'home' ? '部屋データの見本を保存' : '部屋データを保存'}</button>
          <label>
            <input type="checkbox" checked={showRoomRoute} onChange={(event) => setShowRoomRoute(event.target.checked)} />
            経路を表示
          </label>
        </>}
        <label>
          <input type="checkbox" checked={showChair} onChange={(event) => {
            const visible = event.target.checked
            setChairVisibility((previous) => ({ ...previous, [environmentMode]: visible }))
          }} />
          椅子を表示
        </label>
      </fieldset>
    </>
  )

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {settingsHost && createPortal(settings, settingsHost)}
      <style>{`
        .bvh-environment-panel button:disabled { opacity: .45; cursor: not-allowed; }
        .bvh-inline-settings fieldset { display: grid; gap: 6px; min-width: 0; margin: 8px 0; border: 1px solid #475569; }
        .bvh-inline-settings label { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
        .bvh-inline-settings select, .bvh-inline-settings input { min-width: 0; max-width: 100%; }
      `}</style>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />

      <div className="bvh-environment-panel" role="group" aria-label="空間" style={{
        position: 'absolute', top: 64, left: 12, display: 'flex', flexDirection: 'column',
        gap: 7, padding: '10px 12px', borderRadius: 10, color: '#e2e8f0', fontSize: 11,
        background: 'rgba(15,23,42,0.88)', backdropFilter: 'blur(8px)',
        border: '1px solid rgba(148,163,184,0.2)', maxWidth: 'calc(100% - 24px)', boxSizing: 'border-box',
        maxHeight: 'calc(100% - 210px)', overflowY: 'auto', overscrollBehavior: 'contain',
      }}>
        <span style={{ color: '#cbd5e1' }}>{environmentMode === 'home' ? roomSceneName : '検証空間（24m）'}</span>
        {settingsHost === undefined && (
          <details className="bvh-inline-settings">
            <summary>アバター・空間の設定</summary>
            {settings}
          </details>
        )}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
          <button type="button" style={btnStyle} onClick={showWholeEnvironment}>全体を見る</button>
          <button type="button" style={btnStyle} onClick={focusAvatar}>学長を見る</button>
          <button type="button" style={{ ...btnStyle, padding: '6px 8px' }} onClick={() => {
            const viewer = containerRef.current?.parentElement
            const request = document.fullscreenElement ? document.exitFullscreen() : viewer?.requestFullscreen()
            request?.catch((error) => console.warn('[BvhViewer] fullscreen unavailable', error))
          }}>全画面</button>
        </div>
        {environmentMode === 'home' && <details>
          <summary style={{ cursor: 'pointer', padding: '3px 0' }}>家具の動作（検証）</summary>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 7, paddingTop: 8, maxWidth: 280 }}>
            <select aria-label="移動先" value={roomDestinationId} disabled={roomTaskRef.current?.busy}
              onChange={(event) => setRoomDestinationId(event.target.value)}
              style={{ color: '#e2e8f0', background: '#1e293b', border: '1px solid #475569', borderRadius: 5, padding: 5 }}>
              {roomDestinations.map((destination) => <option key={destination.id} value={destination.id}>{destination.label}</option>)}
            </select>
            <button type="button" style={btnStyle} disabled={!roomTaskReady || roomTaskRef.current?.busy || roomDestinations.length === 0} onClick={() => requestRoomWalk(roomDestinationId)}>選んだ場所まで歩く</button>
            <select aria-label="座る椅子" value={roomChairId} disabled={roomTaskRef.current?.busy}
              onChange={(event) => setRoomChairId(event.target.value)}
              style={{ color: '#e2e8f0', background: '#1e293b', border: '1px solid #475569', borderRadius: 5, padding: 5 }}>
              {roomChairs.map((chair) => <option key={chair.id} value={chair.id}>{chair.label}</option>)}
            </select>
            <button type="button" style={btnStyle} disabled={!roomTaskReady || roomChairs.length === 0 || roomTaskRef.current?.busy || roomTask?.phase === 'seated' || persistentPoseRef.current.kind !== 'standing'} onClick={() => startRoomTask()}>椅子まで歩いて座る</button>
            <div style={{ display: 'flex', gap: 5 }}>
              <button type="button" style={btnStyle} disabled={roomTask?.phase !== 'seated' || !roomFixedSeatRef.current} onClick={() => {
                if (roomTaskRef.current?.stand()) {
                  if (avatarAnimationRef.current) avatarAnimationRef.current.phase = 'motion'
                  setIsPlaying(true)
                } else if (roomFixedSeatRef.current && resumeRoomStandingRef.current?.()) {
                  // The preserved controller owns the complete stand transition.
                } else if (roomFixedSeatRef.current && roomPrimitivesRef.current) {
                  setRoomTask((task) => task ? { ...task, phase: 'standing' } : task)
                  loadBvh(roomPrimitivesRef.current.stand.text, '家具からの起立', '椅子から立ち上がる')
                }
              }}>立ち上がる</button>
              <button type="button" style={btnStyle} disabled={!roomTaskRef.current?.busy} onClick={() => { pendingRoomDestinationRef.current = null; roomTaskRef.current?.cancel() }}>タスク中止</button>
            </div>
            <span role="status" aria-label="家具タスクの状態" style={{ color: '#cbd5e1', lineHeight: 1.5 }}>
              {roomTaskError || (roomTask ? ({ walking: roomTask.kind === 'walk' ? `${roomTask.chairLabel}へ歩行中` : '家具を避けて歩行中', aligning: roomTask.kind === 'walk' ? '到着して姿勢を整えています' : '椅子の向きに合わせています', sitting: '着座中', seated: `${roomTask.chairLabel}に着座しました`, standing: '立ち上がり中', settling: '立位を整えています', idle: roomTask.kind === 'walk' ? `${roomTask.chairLabel}に到着しました` : '立ち上がりました', failed: roomTask.reason ?? '停止しました', cancelled: 'タスクを中止しました' }[roomTask.phase]) : roomTaskReady ? '移動先や椅子を選んで開始できます' : '家具と動作を準備中…')}
            </span>
          </div>
        </details>}
      </div>

      {/* Controls overlay */}
      <div
        style={{
          position: 'absolute',
          bottom: 16,
          left: '50%',
          transform: 'translateX(-50%)',
          display: 'flex',
          flexWrap: 'wrap',
          justifyContent: 'center',
          width: 'max-content',
          maxWidth: 'calc(100% - 24px)',
          boxSizing: 'border-box',
          gap: 8,
          alignItems: 'center',
          padding: '8px 16px',
          borderRadius: 12,
          background: 'rgba(15,23,42,0.85)',
          backdropFilter: 'blur(8px)',
          border: '1px solid rgba(148,163,184,0.2)',
        }}
      >
        <label style={{ ...btnStyle, display: 'flex', alignItems: 'center', gap: 4 }}>
          📂 BVH ファイル
          <input
            type="file"
            accept=".bvh"
            onChange={handleFileUpload}
            style={{ display: 'none' }}
          />
        </label>

        {hasFile && (
          <>
            <button style={btnStyle} onClick={() => setIsPlaying(!isPlaying)}>
              {isPlaying ? '⏸ 一時停止' : '▶ 再生'}
            </button>
            <button
              style={btnStyle}
              onClick={() => {
                if (roomTaskRef.current?.busy) { pendingRoomDestinationRef.current = null; roomTaskRef.current.cancel(); return }
                const state = avatarAnimationRef.current
                const action = activeActionRef.current
                if (mixerRef.current && state && action) {
                  for (const item of state.bones) {
                    item.bone.position.copy(item.restPosition)
                    item.bone.quaternion.copy(item.restQuaternion)
                  }
                  state.phase = 'motion'
                  state.returnElapsed = 0
                  mixerRef.current.setTime(0)
                  action.reset()
                  action.setLoop(THREE.LoopOnce, 1)
                  action.clampWhenFinished = true
                  action.play()
                  setIsPlaying(true)
                }
              }}
            >
              ⏮ リセット
            </button>
          </>
        )}
      </div>

      {/* File name badge */}
      {hasFile && (
        <div
          style={{
            position: 'absolute',
            top: 12,
            left: 16,
            padding: '4px 10px',
            borderRadius: 6,
            background: 'rgba(15,23,42,0.75)',
            border: '1px solid rgba(148,163,184,0.3)',
            fontSize: 11,
            color: '#94a3b8',
          }}
        >
          🦴 {fileName}
        </div>
      )}

      {/* Empty state */}
      {!hasFile && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            color: '#64748b',
            fontSize: 14,
            gap: 8,
            pointerEvents: 'none',
          }}
        >
          <span style={{ fontSize: 40 }}>🦴</span>
          <span>BVH ファイルをアップロードして</span>
          <span>モーションをプレビューできます</span>
        </div>
      )}
    </div>
  )
}
