'use client'

import Link from 'next/link'
import React, { useCallback, useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { loadPhysicsAvatar, type AvatarProfile, type PhysicalBody, type PhysicsAvatar } from './physicsAvatar'

type Vec3 = [number, number, number]
type Quat = [number, number, number, number]
type RecordValue = Record<string, unknown>
type Geom = {
  id: string
  type: 'box' | 'capsule' | 'sphere' | 'plane' | 'cylinder' | 'mesh'
  position: Vec3
  quaternion: Quat
  size: number[]
  sizeConvention: 'mujoco' | 'full'
  rgba?: number[]
  body?: string
  friction?: number[]
  mesh?: { vertices: Vec3[]; faces: number[][] }
}
type Frame = { time: number; geoms: Geom[]; bodies: PhysicalBody[]; stats: RecordValue }
type FailureExample = { seed: number; fallTimeSeconds: number; settlingSeconds: number; description: string }
type NavigationRecord = { mode: string; sceneSeed: number; verified: boolean; failedGates: string[]; terminalReason: string; teacherDiagnostic: boolean }
type RecordingOption = { title: string; url: string; verified?: boolean }
/** 頭の深度カメラの仕様(world_model/depth_perception.py の DepthCameraConfig) */
type CameraSpec = { mountBody: string; horizontalFovDegrees: number; downPitchDegrees: number; width: number; height: number; nominalRangeMeters: number }
/** 経路計画が実際に使った深度画像1枚。pixels は unit m 刻み(255 = 何にも当たらない)、obstacle は地図に足した画素 */
type DepthShot = { time: number; note: string; obstacleCount: number; pixels: Uint8Array; obstacle: Uint8Array }
/** 行動の判断に使った入力(publish_results.py の perception)。kind none は画像を使っていない記録 */
type Perception = { kind: 'depth' | 'none'; summary: string; tracker: string; camera: CameraSpec; unit: number; shots: DepthShot[] }
type Trace = { frames: Frame[]; controller: string; dt: number; completeWorld: boolean; task: string; motionReference: string; physicalFootstepPrior: boolean; avatarProfile?: AvatarProfile; physicsRestBodies: PhysicalBody[]; bodyModelSha256: string; failureExample?: FailureExample; navigation?: NavigationRecord; avatarBodyMap?: Record<string, string>; viewerStatus?: string; perception?: Perception }
type Stage = {
  stage: number
  title: string
  status: string
  sceneUrl?: string
  traceUrl?: string
  traceOptions?: RecordingOption[]
  diagnosticOptions?: RecordingOption[]
  /** 学習中の最新の重みでの再生(/training-monitor が自動で更新する) */
  liveOptions?: RecordingOption[]
  /** G2 以降の評価の記録(/training-monitor/results.json) */
  resultOptions?: RecordingOption[]
  diagnosticFollowup?: { trialCount: number; verifiedTrialCount: number }
  metrics?: RecordValue
  gates?: RecordValue
  description?: string
}
type Loaded = { geoms: Geom[]; frames: Frame[]; controller: string; gravity?: Vec3; metadata: RecordValue; task: string; motionReference: string; physicalFootstepPrior: boolean; avatarProfile?: AvatarProfile; physicsRestBodies: PhysicalBody[]; bodyModelSha256: string; failureExample?: FailureExample; navigation?: NavigationRecord; avatarBodyMap?: Record<string, string>; viewerStatus?: string; perception?: Perception }
/** 全身追従器の計画(G1〜G6)の評価のまとめ。/training-monitor/results.json の progress(publish_results.py が書く) */
type ProgressRow = { goal: string; title: string; result: string; criterion: string; status: string }
type Progress = { model: string; note: string; rows: ProgressRow[] }
type BodyDisplay = 'avatar' | 'overlay' | 'physics'
type ViewApi = {
  draw: (time: number) => Frame | undefined
  focus: (bodyOnly: boolean) => void
  bodyDisplay: (value: BodyDisplay) => void
  avatarDiagnostics: () => (ReturnType<PhysicsAvatar['diagnostics']> & { visible: boolean; physicalBodiesVisible: number }) | null
}

const DEFAULT_TITLES = ['身体・重力・接触', '家具配置の変化', '複数の部屋と廊下', '段差・階段・2階']
const STATUS: Record<string, string> = {
  locked: '前段階の検証待ち', pending: '検証待ち', ready: '検証用の空間', preview: '空間の確認',
  layout: '空間の確認', validated: '物理検証済み', passed: '物理検証済み',
  trained: '学習済み', training: '学習中', failed: '検証で問題を検出', blocked: '検証待ち',
}
const record = (value: unknown): RecordValue => value && typeof value === 'object' && !Array.isArray(value) ? value as RecordValue : {}
const finite = (value: unknown, fallback: number) => typeof value === 'number' && Number.isFinite(value) ? value : fallback
const numericArray = (value: unknown): number[] => Array.isArray(value) && value.every((item) => typeof item === 'number' && Number.isFinite(item)) ? value : []
const vector = (value: unknown, fallback: Vec3 = [0, 0, 0]): Vec3 => {
  const values = numericArray(value)
  return values.length === 3 ? values as Vec3 : fallback
}
const quaternion = (value: unknown): Quat => {
  const values = numericArray(value)
  return values.length === 4 && Math.hypot(...values) > 1e-8 ? values as Quat : [1, 0, 0, 0]
}

/** MuJoCo Z-up to Three.js Y-up is a proper rotation, including the local geometry axes. */
const positionToThree = ([x, y, z]: Vec3) => new THREE.Vector3(x, z, -y)
const quaternionToThree = ([w, x, y, z]: Quat) => new THREE.Quaternion(x, z, -y, w).normalize()

function parseBodies(value: unknown): PhysicalBody[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((value) => {
    const item = record(value)
    if (typeof item.name !== 'string' || numericArray(item.position).length !== 3 || numericArray(item.quaternion).length !== 4) return []
    return [{ name: item.name, position: vector(item.position), quaternion: quaternion(item.quaternion) }]
  })
}

function parseAvatarProfile(value: unknown): AvatarProfile | undefined {
  const item = record(value), display = record(item.display)
  if (item.version !== 1 || typeof display.modelUrl !== 'string' || typeof display.modelSha256 !== 'string' ||
    !(finite(display.scale, 0) > 0) || numericArray(display.rootPosition).length !== 3) return undefined
  const joints: AvatarProfile['display']['joints'] = {}
  for (const [name, value] of Object.entries(record(display.joints))) {
    const joint = record(value)
    if (numericArray(joint.localQuaternion).length === 4 && numericArray(joint.localPosition).length === 3) {
      joints[name] = { localQuaternion: numericArray(joint.localQuaternion), localPosition: numericArray(joint.localPosition) }
    }
  }
  if (!joints.hips || !joints.leftFoot || !joints.rightFoot) return undefined
  return { display: { modelUrl: display.modelUrl, modelSha256: display.modelSha256, scale: finite(display.scale, 1),
    yawDegrees: finite(display.yawDegrees, 0), rootPosition: vector(display.rootPosition), joints }, physical: record(item.physical) }
}

function parseGeom(value: unknown, convention: Geom['sizeConvention']): Geom | null {
  const item = record(value)
  if (typeof item.id !== 'string' || !['box', 'capsule', 'sphere', 'plane', 'cylinder', 'mesh'].includes(String(item.type))) return null
  const sourceMesh = record(item.mesh)
  const vertices = sourceMesh.vertices ?? item.vertices
  const faces = sourceMesh.faces ?? item.faces
  const mesh = Array.isArray(vertices) && Array.isArray(faces) ? {
    vertices: vertices.map((vertex) => vector(vertex)),
    faces: faces.map((face) => numericArray(face)),
  } : undefined
  const yaw = finite(item.yaw, 0)
  return {
    id: item.id, type: item.type as Geom['type'], position: vector(item.position),
    quaternion: item.quaternion ? quaternion(item.quaternion) : [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)],
    size: numericArray(item.size), sizeConvention: item.sizeConvention === 'full' ? 'full' : item.sizeConvention === 'mujoco' ? 'mujoco' : convention,
    rgba: numericArray(item.rgba), body: typeof item.body === 'string' ? item.body : undefined,
    friction: numericArray(item.friction), mesh,
  }
}

function parseScene(value: unknown): { geoms: Geom[]; gravity?: Vec3; metadata: RecordValue } {
  const item = record(value)
  const input = item.geoms ?? item.geometry
  const convention = item.sizeConvention === 'mujoco' ? 'mujoco' : 'full'
  const geoms = Array.isArray(input) ? input.map((geom) => parseGeom({ ...record(geom), body: record(geom).body ?? 'world' }, convention)).filter((geom): geom is Geom => !!geom) : []
  const physics = record(item.physics)
  const gravity = numericArray(item.gravity ?? physics.gravity)
  return { geoms, gravity: gravity.length === 3 ? gravity as Vec3 : undefined, metadata: item }
}

function parseTrace(value: unknown): Trace {
  const item = record(value)
  if (item.version !== 1 || !Array.isArray(item.frames)) throw new Error('物理記録の形式が正しくありません')
  const dt = finite(item.dt, 0.02)
  const geometry = Array.isArray(item.geometry) ? item.geometry : []
  const convention = item.sizeConvention === 'mujoco' ? 'mujoco' : 'full'
  const definitions = new Map(geometry.map((geom) => [String(record(geom).id), record(geom)]))
  let previousTime = -Infinity
  const frames = item.frames.map((value, index): Frame => {
    const frame = record(value)
    const time = finite(frame.time, index * dt)
    if (time < previousTime) throw new Error('物理記録の時刻が逆転しています')
    previousTime = time
    const input = Array.isArray(frame.geoms) ? frame.geoms : []
    const geoms = input.map((value) => {
      const pose = record(value)
      const definition = definitions.get(String(pose.id))
      return parseGeom(definition ? { ...definition, ...pose } : pose, definition ? convention : 'mujoco')
    }).filter((geom): geom is Geom => !!geom)
    return { time, geoms, bodies: parseBodies(frame.bodies), stats: record(frame.stats ?? frame.metrics) }
  })
  if (!frames.length || !frames.some((frame) => frame.geoms.length)) throw new Error('物理記録に身体の形状がありません')
  const failure = record(item.failureExample)
  const failureTime = finite(failure.fallTimeSeconds, NaN)
  const failureExample = failure.evaluationExcluded === true && Number.isFinite(failureTime) ? {
    seed: finite(failure.seed, 0), fallTimeSeconds: failureTime,
    settlingSeconds: finite(failure.settlingSeconds, 0),
    description: typeof failure.description === 'string' ? failure.description : '追加学習途中のモデルで記録した転倒例です。',
  } : undefined
  const navigationData = record(item.navigation), navigationQuality = record(item.navigationQuality)
  const navigation = ['turn', 'obstacle'].includes(String(navigationData.mode)) ? {
    mode: String(navigationData.mode), sceneSeed: finite(navigationData.sceneSeed, 0),
    verified: navigationQuality.navigationVerified === true,
    failedGates: Array.isArray(navigationQuality.failedGates) ? navigationQuality.failedGates.filter((item): item is string => typeof item === 'string') : [],
    terminalReason: typeof record(frames.at(-1)?.stats.navigation).terminal_reason === 'string' ? String(record(frames.at(-1)?.stats.navigation).terminal_reason) : '',
    teacherDiagnostic: navigationData.evaluationRole === 'private_causal_diagnostic' && navigationData.isLearnedRoomNavigator === false && navigationData.highLevelLearnedActorUsed === false && navigationData.additionalTrainingTimesteps === 0,
  } : undefined
  return { frames, dt, controller: typeof item.controller === 'string' ? item.controller : '記録の制御方式を確認中',
    completeWorld: item.geometryScope === 'complete-world' || item.completeWorld === true,
    task: typeof item.task === 'string' ? item.task : typeof frames[0].stats.task === 'string' ? frames[0].stats.task : '',
    motionReference: typeof item.motionReference === 'string' ? item.motionReference : '',
    physicalFootstepPrior: item.physicalFootstepPrior === true, avatarProfile: parseAvatarProfile(item.avatarProfile),
    physicsRestBodies: parseBodies(item.physicsRestBodies), bodyModelSha256: typeof item.bodyModelSha256 === 'string' ? item.bodyModelSha256 : '', failureExample, navigation,
    avatarBodyMap: parseBodyMap(item.avatarBodyMap), viewerStatus: typeof item.viewerStatus === 'string' ? item.viewerStatus : undefined,
    perception: parsePerception(item.perception) }
}

const DEFAULT_CAMERA: CameraSpec = { mountBody: 'head', horizontalFovDegrees: 90, downPitchDegrees: 15, width: 64, height: 48, nominalRangeMeters: 5 }

function decodeBase64(text: unknown): Uint8Array {
  if (typeof text !== 'string') return new Uint8Array()
  try {
    const binary = atob(text)
    const bytes = new Uint8Array(binary.length)
    for (let index = 0; index < binary.length; index++) bytes[index] = binary.charCodeAt(index)
    return bytes
  } catch { return new Uint8Array() }
}

function parsePerception(value: unknown): Perception | undefined {
  const item = record(value), source = record(item.camera)
  if (item.kind !== 'depth' && item.kind !== 'none') return undefined
  const camera: CameraSpec = { mountBody: typeof source.mountBody === 'string' ? source.mountBody : DEFAULT_CAMERA.mountBody,
    horizontalFovDegrees: finite(source.horizontalFovDegrees, DEFAULT_CAMERA.horizontalFovDegrees), downPitchDegrees: finite(source.downPitchDegrees, DEFAULT_CAMERA.downPitchDegrees),
    width: Math.round(finite(source.width, DEFAULT_CAMERA.width)), height: Math.round(finite(source.height, DEFAULT_CAMERA.height)),
    nominalRangeMeters: finite(source.nominalRangeMeters, DEFAULT_CAMERA.nominalRangeMeters) }
  const count = camera.width * camera.height
  const shots = Array.isArray(item.shots) ? item.shots.flatMap((value): DepthShot[] => {
    const shot = record(value)
    const pixels = decodeBase64(shot.depth), packed = decodeBase64(shot.obstacle)
    if (pixels.length !== count) return []
    const obstacle = new Uint8Array(count)
    for (let index = 0; index < count; index++) obstacle[index] = ((packed[index >> 3] ?? 0) >> (index & 7)) & 1   // packbits(bitorder=little)
    return [{ time: finite(shot.time, 0), note: typeof shot.note === 'string' ? shot.note : '', obstacleCount: finite(shot.obstacleCount, 0), pixels, obstacle }]
  }).sort((a, b) => a.time - b.time) : []
  return { kind: item.kind, summary: typeof item.summary === 'string' ? item.summary : '', tracker: typeof item.tracker === 'string' ? item.tracker : '',
    camera, unit: finite(item.depthUnitMeters, 0.05), shots }
}

/** 近い=明るい黄、遠い=暗い紫。DEPTH_SCALE m より遠い点は最も暗い色 */
const DEPTH_SCALE = 8
const DEPTH_STOPS: [number, [number, number, number]][] = [[0, [255, 241, 199]], [0.22, [247, 172, 93]], [0.5, [201, 86, 116]], [1, [50, 45, 110]]]
function depthColour(meters: number): [number, number, number] {
  const t = THREE.MathUtils.clamp(meters / DEPTH_SCALE, 0, 1)
  const upper = DEPTH_STOPS.findIndex(([stop]) => stop >= t)
  const [a, from] = DEPTH_STOPS[Math.max(upper - 1, 0)], [b, to] = DEPTH_STOPS[Math.max(upper, 0)]
  const blend = b > a ? (t - a) / (b - a) : 0
  return [0, 1, 2].map((index) => Math.round(from[index] + (to[index] - from[index]) * blend)) as [number, number, number]
}

function DepthShotView({ shot, camera, unit, showObstacle }: { shot: DepthShot; camera: CameraSpec; unit: number; showObstacle: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  useEffect(() => {
    const context = canvasRef.current?.getContext('2d')
    if (!context) return
    const image = context.createImageData(camera.width, camera.height)
    for (let index = 0; index < camera.width * camera.height; index++) {
      const value = shot.pixels[index]
      let [r, g, b] = value === 255 ? [9, 13, 21] : depthColour(value * unit)
      if (showObstacle && shot.obstacle[index]) { r = Math.round(r * .3 + 255 * .7); g = Math.round(g * .3 + 46 * .7); b = Math.round(b * .3 + 46 * .7) }
      image.data.set([r, g, b, 255], index * 4)
    }
    context.putImageData(image, 0, 0)
  }, [shot, camera, unit, showObstacle])
  return <canvas ref={canvasRef} width={camera.width} height={camera.height} role="img" aria-label={`深度画像 ${camera.width}×${camera.height}`}
    style={{ display: 'block', width: '100%', aspectRatio: `${camera.width} / ${camera.height}`, imageRendering: 'pixelated', borderRadius: 8, background: '#090d15' }} />
}

function parseBodyMap(value: unknown): Record<string, string> | undefined {
  const entries = Object.entries(record(value)).filter((entry): entry is [string, string] => typeof entry[1] === 'string')
  return entries.length ? Object.fromEntries(entries) : undefined
}

/** Complete snapshots own the physical world; previews only supplement pose-only recordings. */
function geomsForRecording(sceneGeoms: Geom[], trace: Trace | null): Geom[] {
  const first = trace?.frames[0]
  if (trace?.completeWorld) return [...(first?.geoms ?? [])]
  const geoms = new Map(sceneGeoms.map((geom) => [geom.id, geom]))
  for (const geom of first?.geoms ?? []) geoms.set(geom.id, geom)
  return [...geoms.values()]
}

function createGeometry(geom: Geom): THREE.BufferGeometry {
  const size = geom.size
  const scale = geom.sizeConvention === 'mujoco' ? 2 : 1
  if (geom.type === 'box') return new THREE.BoxGeometry(Math.max(0.001, (size[0] ?? 1) * scale), Math.max(0.001, (size[2] ?? 1) * scale), Math.max(0.001, (size[1] ?? 1) * scale))
  if (geom.type === 'capsule') return new THREE.CapsuleGeometry(Math.max(0.001, size[0] ?? 0.06), Math.max(0.001, (size[1] ?? 0.1) * (geom.sizeConvention === 'mujoco' ? 2 : 1)), 6, 14)
  if (geom.type === 'sphere') return new THREE.SphereGeometry(Math.max(0.001, size[0] ?? 0.06), 16, 12)
  if (geom.type === 'cylinder') return new THREE.CylinderGeometry(size[0] ?? 0.06, size[0] ?? 0.06, Math.max(0.001, (size[1] ?? 0.1) * 2), 18)
  if (geom.type === 'plane') {
    const geometry = new THREE.PlaneGeometry(size[0] ? size[0] * 2 : 20, size[1] ? size[1] * 2 : 20)
    geometry.rotateX(-Math.PI / 2)
    return geometry
  }
  if (!geom.mesh || geom.mesh.vertices.length > 50000 || geom.mesh.faces.length > 100000) throw new Error(`${geom.id} の衝突形状を読み取れません`)
  const points = geom.mesh.vertices.flatMap((point) => positionToThree(point).toArray())
  const indices = geom.mesh.faces.flatMap((face) => {
    if (face.length !== 3 || face.some((index) => !Number.isInteger(index) || index < 0 || index >= geom.mesh!.vertices.length)) throw new Error(`${geom.id} の面の指定が正しくありません`)
    return face
  })
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3))
  geometry.setIndex(indices)
  geometry.computeVertexNormals()
  return geometry
}

const isBody = (geom: Geom) => geom.body ? geom.body !== 'world' && geom.body !== 'worldbody' : !/floor|wall|chair|table|obstacle|room|ground/i.test(geom.id)
const geomColor = (geom: Geom) => {
  if (geom.rgba && geom.rgba.length >= 3) return new THREE.Color().setRGB(geom.rgba[0], geom.rgba[1], geom.rgba[2], THREE.LinearSRGBColorSpace)
  if (!isBody(geom)) return new THREE.Color(/floor|ground/.test(geom.id) ? 0xb5afa4 : 0x7b8996)
  return new THREE.Color(/torso|pelvis|abdomen|head/.test(geom.id) ? 0xd9aa72 : /arm|hand/.test(geom.id) ? 0x6eaaa8 : 0x7298c6)
}
const metric = (stats: RecordValue, key: string, unit = '') => {
  const value = stats[key]
  if (typeof value === 'boolean') return value ? 'はい' : 'いいえ'
  if (typeof value === 'number' && Number.isFinite(value)) return `${value.toFixed(key === 'contact_count' ? 0 : 3)}${unit}`
  return '—'
}

export function WorldModelLab() {
  const [stages, setStages] = useState<Stage[]>([])
  const [progress, setProgress] = useState<Progress | null>(null)
  const [selected, setSelected] = useState(1)
  const [loaded, setLoaded] = useState<Loaded | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [stats, setStats] = useState<RecordValue>({})
  const [refresh, setRefresh] = useState(0)
  const [traceChoice, setTraceChoice] = useState<{ stage: number; url: string } | null>(null)
  const [worldGravity, setWorldGravity] = useState<Vec3 | undefined>(undefined)
  const [bodyDisplay, setBodyDisplay] = useState<BodyDisplay>('avatar')
  const [avatarStatus, setAvatarStatus] = useState('')
  const bodyDisplayRef = useRef<BodyDisplay>('avatar')
  const containerRef = useRef<HTMLDivElement | null>(null)
  const eyeContainerRef = useRef<HTMLDivElement | null>(null)
  const [showObstacle, setShowObstacle] = useState(true)
  const timeRef = useRef(0)
  const playingRef = useRef(false)
  const viewRef = useRef<ViewApi | null>(null)
  const requestedLinkAppliedRef = useRef(false)
  const currentStage = stages.find((stage) => stage.stage === selected)
  const recordingOptions = [...(currentStage?.resultOptions ?? []), ...(currentStage?.traceOptions ?? []), ...(currentStage?.diagnosticOptions ?? []), ...(currentStage?.liveOptions ?? [])]
  const currentTraceUrl = traceChoice?.stage === selected && recordingOptions.some((option) => option.url === traceChoice.url)
    ? traceChoice.url : currentStage?.traceUrl
  const startTime = loaded?.frames[0]?.time ?? 0
  const endTime = loaded?.frames.at(-1)?.time ?? 0
  const sliderEndTime = startTime + Math.ceil((endTime - startTime) * 1000) / 1000
  const hasTrace = !!loaded?.frames.length
  const hasMotion = (loaded?.frames.length ?? 0) > 1 && endTime > startTime

  useEffect(() => { playingRef.current = playing }, [playing])
  useEffect(() => { bodyDisplayRef.current = bodyDisplay; viewRef.current?.bodyDisplay(bodyDisplay) }, [bodyDisplay])

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setError('')
    fetch('/world-model/manifest.json', { signal: controller.signal, cache: 'no-store' }).then(async (response) => {
      if (!response.ok) throw new Error('検証データを準備中です。準備後に「記録を更新」を押してください。')
      const manifest = record(await response.json())
      if (manifest.version !== 1 || !Array.isArray(manifest.stages)) throw new Error('検証一覧を読み取れません')
      const next = manifest.stages.map((value): Stage => {
        const stage = record(value)
        return { stage: finite(stage.stage, 0), title: typeof stage.title === 'string' ? stage.title : '検証段階',
          status: typeof stage.status === 'string' ? stage.status : 'pending',
          sceneUrl: typeof stage.sceneUrl === 'string' ? stage.sceneUrl : undefined,
          traceUrl: typeof stage.traceUrl === 'string' ? stage.traceUrl : undefined,
          traceOptions: Array.isArray(stage.traceOptions) ? stage.traceOptions.map((value) => {
            const option = record(value)
            return { title: typeof option.title === 'string' ? option.title : '制御の記録', url: typeof option.url === 'string' ? option.url : '' }
          }).filter((option) => option.url) : undefined,
          diagnosticOptions: Array.isArray(stage.diagnosticOptions) ? stage.diagnosticOptions.map((value) => {
            const option = record(value)
            return { title: typeof option.title === 'string' ? option.title : '教師の経路補助による診断', url: typeof option.url === 'string' ? option.url : '', verified: option.verified === true }
          }).filter((option) => option.url) : undefined,
          diagnosticFollowup: stage.stage === 2 && record(record(manifest.phase2Diagnostics).followup).trialCount ? {
            trialCount: finite(record(record(manifest.phase2Diagnostics).followup).trialCount, 0),
            verifiedTrialCount: finite(record(record(manifest.phase2Diagnostics).followup).verifiedTrialCount, 0),
          } : undefined,
          metrics: record(stage.metrics), gates: record(stage.gates), description: typeof stage.description === 'string' && stage.description.trim() ? stage.description : undefined }
      }).filter((stage) => stage.stage >= 1 && stage.stage <= 4)
      // 学習中の最新の重みでの再生記録(書き出し係が止まっていれば無いだけ)
      const live = await fetch('/training-monitor/manifest.json', { signal: controller.signal, cache: 'no-store' })
        .then((liveResponse) => (liveResponse.ok ? liveResponse.json() : null)).catch(() => null)
      for (const value of Array.isArray(record(live).traceOptions) ? record(live).traceOptions as unknown[] : []) {
        const option = record(value)
        const stage = next.find((item) => item.stage === finite(option.stage, 1))
        if (stage && typeof option.url === 'string' && typeof option.title === 'string') {
          stage.liveOptions = [...(stage.liveOptions ?? []), { title: option.title, url: option.url }]
        }
      }
      // G2 以降の評価の記録とまとめ(publish_results.py が書く。フロントエンドを作り直さずに更新される)
      const results = record(await fetch('/training-monitor/results.json', { signal: controller.signal, cache: 'no-store' })
        .then((resultsResponse) => (resultsResponse.ok ? resultsResponse.json() : null)).catch(() => null))
      for (const stage of next) {
        const entry = record(record(results.stages)[String(stage.stage)])
        const options = Array.isArray(entry.options) ? entry.options.map((value) => record(value))
          .filter((option) => typeof option.url === 'string' && typeof option.title === 'string')
          .map((option) => ({ title: String(option.title), url: String(option.url) })) : []
        if (options.length) {
          stage.resultOptions = options
          stage.traceUrl = options[0].url   // 段階を開いたら最新の評価の記録を表示
        }
        if (typeof entry.status === 'string' && entry.status) stage.status = entry.status
        if (typeof entry.description === 'string' && entry.description) stage.description = entry.description
      }
      if (!controller.signal.aborted) {
        setStages(next)
        const summary = record(results.progress)
        const rows = Array.isArray(summary.rows) ? summary.rows.map((value): ProgressRow => {
          const row = record(value)
          const text = (key: string) => typeof row[key] === 'string' ? String(row[key]) : ''
          return { goal: text('goal'), title: text('title'), result: text('result'), criterion: text('criterion'), status: text('status') }
        }).filter((row) => row.goal && row.title) : []
        setProgress(rows.length ? { model: typeof summary.model === 'string' ? summary.model : '', note: typeof summary.note === 'string' ? summary.note : '', rows } : null)
        if (!requestedLinkAppliedRef.current) {
          requestedLinkAppliedRef.current = true
          const params = new URLSearchParams(window.location.search)
          const stage = next.find((item) => item.stage === Number(params.get('stage') ?? 1))
          const trace = params.get('trace')
          if (stage) {
            setSelected(stage.stage)
            if (trace && (stage.traceUrl === trace || [...(stage.resultOptions ?? []), ...(stage.traceOptions ?? []), ...(stage.diagnosticOptions ?? []), ...(stage.liveOptions ?? [])].some((option) => option.url === trace))) {
              setTraceChoice({ stage: stage.stage, url: trace })
            }
          }
        }
        const gravity = numericArray(manifest.gravity)
        setWorldGravity(gravity.length === 3 ? gravity as Vec3 : undefined)
        setLoading(false)
      }
    }).catch((error) => { if (!controller.signal.aborted) { setError(error instanceof Error ? error.message : String(error)); setLoading(false) } })
    return () => controller.abort()
  }, [refresh])

  useEffect(() => {
    if (!currentStage) {
      setLoaded(null)
      setPlaying(false)
      playingRef.current = false
      return
    }
    const controller = new AbortController()
    setLoading(true)
    setError('')
    setPlaying(false)
    playingRef.current = false
    setLoaded(null)
    const read = async (url?: string): Promise<unknown> => {
      if (!url) return null
      const response = await fetch(url, { signal: controller.signal, cache: 'no-store' })
      if (!response.ok) throw new Error('この段階の検証記録はまだ利用できません')
      return response.json()
    }
    Promise.all([read(currentStage.sceneUrl), read(currentTraceUrl)]).then(([sceneInput, traceInput]) => {
      if (controller.signal.aborted) return
      const scene = parseScene(sceneInput)
      const trace = traceInput ? parseTrace(traceInput) : null
      const first = trace?.frames[0]
      const geoms = geomsForRecording(scene.geoms, trace)
      if (!geoms.length && currentStage.status !== 'locked') throw new Error('表示できる物理形状がありません')
      timeRef.current = first?.time ?? 0
      setCurrentTime(timeRef.current)
      setStats(first?.stats ?? {})
      const physicalScene = trace?.completeWorld ? parseScene(record(traceInput).scene) : scene
      setLoaded({ geoms, frames: trace?.frames ?? [], controller: trace?.controller ?? '空間のプレビュー',
        gravity: physicalScene.gravity ?? worldGravity, metadata: physicalScene.metadata,
        task: trace?.task ?? '', motionReference: trace?.motionReference ?? '',
        physicalFootstepPrior: trace?.physicalFootstepPrior ?? false, avatarProfile: trace?.avatarProfile,
        physicsRestBodies: trace?.physicsRestBodies ?? [], bodyModelSha256: trace?.bodyModelSha256 ?? '', failureExample: trace?.failureExample, navigation: trace?.navigation,
        avatarBodyMap: trace?.avatarBodyMap, viewerStatus: trace?.viewerStatus, perception: trace?.perception })
      setLoading(false)
    }).catch((error) => { if (!controller.signal.aborted) { setError(error instanceof Error ? error.message : String(error)); setLoading(false) } })
    return () => controller.abort()
  }, [currentStage, currentTraceUrl, refresh, worldGravity])

  useEffect(() => {
    const container = containerRef.current
    if (!container || !loaded) return
    let renderer: THREE.WebGLRenderer
    try { renderer = new THREE.WebGLRenderer({ antialias: true }) }
    catch { setError('この環境では3D表示を開始できません'); return }
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x131e2c)
    const camera = new THREE.PerspectiveCamera(48, 1, 0.04, 500)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5))
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.15
    container.appendChild(renderer.domElement)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.minDistance = 0.6
    controls.maxDistance = 150
    controls.maxPolarAngle = Math.PI / 2 - 0.01
    scene.add(new THREE.HemisphereLight(0xf5f2e9, 0x596477, 2.2))
    const sun = new THREE.DirectionalLight(0xffeed5, 3)
    sun.position.set(-5, 10, 4)
    sun.castShadow = true
    sun.shadow.mapSize.set(2048, 2048)
    sun.shadow.camera.left = sun.shadow.camera.bottom = -12
    sun.shadow.camera.right = sun.shadow.camera.top = 12
    sun.shadow.camera.near = 0.5
    sun.shadow.camera.far = 35
    sun.shadow.normalBias = 0.015
    scene.add(sun, sun.target)
    const meshes = new Map<string, THREE.Mesh>()
    const definitions = new Map<string, Geom>()
    const walls: { mesh: THREE.Mesh; originalOpacity: number; normal: THREE.Vector3 }[] = []
    let goalMarker: THREE.Mesh | undefined
    if (loaded.navigation && Array.isArray(loaded.metadata.goals)) {
      const goal = loaded.metadata.goals.map(record).find((item) => item.kind === 'walk')
      if (goal && numericArray(goal.position).length === 3) {
        goalMarker = new THREE.Mesh(new THREE.RingGeometry(.16, .24, 40),
          new THREE.MeshBasicMaterial({ color: 0x78e6b6, side: THREE.DoubleSide, transparent: true, opacity: .85, depthWrite: false }))
        goalMarker.rotation.x = -Math.PI / 2
        goalMarker.position.copy(positionToThree(vector(goal.position))).add(new THREE.Vector3(0, .008, 0))
        goalMarker.name = 'NavigationGoalMarker'
        scene.add(goalMarker)
      }
    }
    let avatar: PhysicsAvatar | undefined
    let disposed = false
    setAvatarStatus(loaded.avatarProfile ? '学長アバターを読み込んでいます' : '')
    try {
      for (const geom of loaded.geoms) {
        const geometry = createGeometry(geom)
        const material = new THREE.MeshStandardMaterial({ color: geomColor(geom), roughness: isBody(geom) ? 0.58 : 0.88,
          metalness: isBody(geom) ? 0.12 : 0, side: THREE.DoubleSide, transparent: (geom.rgba?.[3] ?? 1) < 1, opacity: geom.rgba?.[3] ?? 1 })
        const mesh = new THREE.Mesh(geometry, material)
        mesh.name = geom.id
        mesh.castShadow = isBody(geom) || geom.type !== 'plane'
        mesh.receiveShadow = true
        mesh.position.copy(positionToThree(geom.position))
        mesh.quaternion.copy(quaternionToThree(geom.quaternion))
        scene.add(mesh)
        meshes.set(geom.id, mesh)
        definitions.set(geom.id, geom)
        if (/wall/i.test(geom.id) && geom.type === 'box') {
          // The thin box axis is the actual wall plane, in its rendered basis.
          const thinAxis = geom.size.indexOf(Math.min(...geom.size))
          const nativeNormal: Vec3 = thinAxis === 0 ? [1, 0, 0] : thinAxis === 1 ? [0, 1, 0] : [0, 0, 1]
          walls.push({ mesh, originalOpacity: geom.rgba?.[3] ?? 1,
            normal: positionToThree(nativeNormal).applyQuaternion(mesh.quaternion) })
        }
      }
    } catch (error) { setError(error instanceof Error ? error.message : String(error)) }
    // 学長の視点：頭の位置から、深度カメラと同じ向き(腰の向き・下向きの角度・視野)で見た映像。
    // 自分の体と表示用の印(床の赤い印)は映さない。壁の透過は俯瞰の表示だけに使う。
    const eyeContainer = eyeContainerRef.current
    const eyeSpec = loaded.perception?.camera ?? DEFAULT_CAMERA
    let eyeRenderer: THREE.WebGLRenderer | undefined
    if (eyeContainer && loaded.frames.some((frame) => frame.bodies.some((body) => body.name === eyeSpec.mountBody))) {
      try { eyeRenderer = new THREE.WebGLRenderer({ antialias: true }) } catch { eyeRenderer = undefined }
    }
    const eyeScene = new THREE.Scene()
    eyeScene.background = new THREE.Color(0x1c2937)
    const eyeCamera = new THREE.PerspectiveCamera(THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(eyeSpec.horizontalFovDegrees) / 2) * eyeSpec.height / eyeSpec.width)),
      eyeSpec.width / eyeSpec.height, 0.05, 80)
    const eyeMeshes: [THREE.Mesh, THREE.Mesh][] = []
    if (eyeRenderer && eyeContainer) {
      eyeRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5))
      eyeRenderer.outputColorSpace = THREE.SRGBColorSpace
      eyeRenderer.toneMapping = THREE.ACESFilmicToneMapping
      eyeRenderer.toneMappingExposure = 1.15
      eyeRenderer.domElement.style.display = 'block'
      eyeContainer.appendChild(eyeRenderer.domElement)
      const eyeSun = new THREE.DirectionalLight(0xffeed5, 2.2)
      eyeSun.position.set(-5, 10, 4)
      eyeScene.add(new THREE.HemisphereLight(0xf5f2e9, 0x596477, 2.4), eyeSun)
      for (const [id, mesh] of meshes) {
        const definition = definitions.get(id)!
        if (isBody(definition) || id.startsWith('seen-obstacle')) continue
        const copy = new THREE.Mesh(mesh.geometry, new THREE.MeshStandardMaterial({ color: geomColor(definition), roughness: 0.88, side: THREE.DoubleSide,
          transparent: (definition.rgba?.[3] ?? 1) < 1, opacity: definition.rgba?.[3] ?? 1 }))
        eyeScene.add(copy)
        eyeMeshes.push([mesh, copy])
      }
    }
    let eyeReady = false
    const eyeTarget = new THREE.Vector3()
    const placeEye = (from: Frame, to: Frame, blend: number) => {
      const pick = (frame: Frame, name: string) => frame.bodies.find((body) => body.name === name)
      const headFrom = pick(from, eyeSpec.mountBody), headTo = pick(to, eyeSpec.mountBody) ?? headFrom
      const pelvisFrom = pick(from, 'pelvis') ?? headFrom, pelvisTo = pick(to, 'pelvis') ?? pelvisFrom
      if (!headFrom || !headTo || !pelvisFrom || !pelvisTo) return false
      // depth_perception._camera_pose と同じ：腰の向き(MuJoCo の回転行列 R[1,0], R[0,0])、前は -Y、下向きに pitch
      const heading = ([w, x, y, z]: Quat) => Math.atan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))
      const start = heading(pelvisFrom.quaternion), end = heading(pelvisTo.quaternion)
      const yaw = start + Math.atan2(Math.sin(end - start), Math.cos(end - start)) * blend
      const pitch = THREE.MathUtils.degToRad(eyeSpec.downPitchDegrees)
      const head = [0, 1, 2].map((index) => headFrom.position[index] + (headTo.position[index] - headFrom.position[index]) * blend) as Vec3
      eyeCamera.position.copy(positionToThree(head))
      eyeCamera.up.set(0, 1, 0)
      eyeCamera.lookAt(eyeTarget.copy(positionToThree([head[0] + Math.cos(pitch) * Math.sin(yaw), head[1] - Math.cos(pitch) * Math.cos(yaw), head[2] - Math.sin(pitch)])))
      for (const [source, copy] of eyeMeshes) { copy.position.copy(source.position); copy.quaternion.copy(source.quaternion) }
      return true
    }
    const applyBodyDisplay = (value: BodyDisplay) => {
      if (avatar) avatar.root.visible = value !== 'physics'
      for (const [id, mesh] of meshes) {
        const definition = definitions.get(id)!
        if (!isBody(definition)) continue
        mesh.visible = !avatar || value !== 'avatar'
        for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
          material.transparent = !!avatar && value === 'overlay'
          material.opacity = avatar && value === 'overlay' ? .28 : definition.rgba?.[3] ?? 1
          material.depthWrite = !(avatar && value === 'overlay')
          material.needsUpdate = true
        }
      }
    }
    const followAnchor = meshes.get('humanoid_pelvis') ?? [...meshes.values()].find((mesh) => isBody(definitions.get(mesh.name)!))
    let followBody = false
    let previousAnchor = followAnchor?.position.clone()
    const cameraSide = new THREE.Vector3(), targetSide = new THREE.Vector3()
    const updateWallDisplay = () => {
      for (const { mesh, originalOpacity, normal } of walls) {
        // Only a wall between the camera and its target is translucent. Keep
        // the far room boundary and all furniture opaque while orbiting.
        const cutaway = cameraSide.copy(camera.position).sub(mesh.position).dot(normal)
          * targetSide.copy(controls.target).sub(mesh.position).dot(normal) < -1e-6
        const opacity = cutaway ? .12 : originalOpacity
        mesh.castShadow = !cutaway
        mesh.userData.renderOnlyCutawayWall = cutaway
        for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
          const transparent = cutaway || originalOpacity < 1
          if (material.opacity === opacity && material.transparent === transparent && material.depthWrite === !cutaway) continue
          material.transparent = transparent
          material.opacity = opacity
          material.depthWrite = !cutaway
          material.needsUpdate = true
        }
      }
    }
    const focus = (bodyOnly: boolean) => {
      followBody = bodyOnly
      previousAnchor = followAnchor?.position.clone()
      scene.updateMatrixWorld(true)
      avatar?.root.traverse((object) => {
        const mesh = object as THREE.SkinnedMesh
        if (mesh.isSkinnedMesh) { mesh.skeleton.update(); mesh.computeBoundingBox() }
      })
      const bounds = new THREE.Box3()
      for (const [id, mesh] of meshes) if (!bodyOnly || isBody(definitions.get(id)!)) bounds.union(new THREE.Box3().setFromObject(mesh))
      if (avatar) bounds.union(new THREE.Box3().setFromObject(avatar.root))
      if (bounds.isEmpty()) { camera.position.set(5, 4, 6); controls.target.set(0, 0.8, 0) }
      else {
        const size = bounds.getSize(new THREE.Vector3())
        const center = bounds.getCenter(new THREE.Vector3())
        const distance = Math.max(2.4, Math.max(size.x, size.y, size.z) * 1.15)
        const viewDirection = bodyOnly ? new THREE.Vector3(.95, .60, .38) : new THREE.Vector3(.80, .85, .70)
        camera.position.copy(center).addScaledVector(viewDirection, distance)
        controls.target.copy(center)
      }
      controls.update()
      updateWallDisplay()
    }
    const draw = (time: number) => {
      const frames = loaded.frames
      if (!frames.length) return undefined
      let low = 0, high = frames.length - 1
      while (low < high) { const middle = Math.ceil((low + high) / 2); if (frames[middle].time <= time) low = middle; else high = middle - 1 }
      const from = frames[low], to = frames[Math.min(low + 1, frames.length - 1)]
      const blend = to.time > from.time ? THREE.MathUtils.clamp((time - from.time) / (to.time - from.time), 0, 1) : 0
      const next = new Map(to.geoms.map((geom) => [geom.id, geom]))
      for (const geom of from.geoms) {
        const mesh = meshes.get(geom.id)
        if (!mesh) continue
        const target = next.get(geom.id) ?? geom
        mesh.position.lerpVectors(positionToThree(geom.position), positionToThree(target.position), blend)
        mesh.quaternion.slerpQuaternions(quaternionToThree(geom.quaternion), quaternionToThree(target.quaternion), blend)
      }
      avatar?.update(from.bodies, to.bodies, blend)
      if (eyeRenderer) eyeReady = placeEye(from, to, blend)
      if (followBody && followAnchor && previousAnchor) {
        const movement = followAnchor.position.clone().sub(previousAnchor)
        camera.position.add(movement)
        controls.target.add(movement)
        previousAnchor.copy(followAnchor.position)
      }
      return from
    }
    viewRef.current = { draw, focus, bodyDisplay: applyBodyDisplay,
      avatarDiagnostics: () => avatar ? { ...avatar.diagnostics(), visible: avatar.root.visible,
        physicalBodiesVisible: [...meshes.values()].filter((mesh) => isBody(definitions.get(mesh.name)!) && mesh.visible).length } : null }
    draw(timeRef.current)
    focus(true)
    if (loaded.avatarProfile && loaded.physicsRestBodies.length && loaded.frames.some((frame) => frame.bodies.length)) {
      loadPhysicsAvatar(loaded.avatarProfile, loaded.physicsRestBodies, loaded.avatarBodyMap).then((next) => {
        if (disposed) { next.dispose(); return }
        avatar = next
        scene.add(avatar.root)
        avatar.root.traverse((object) => {
          const mesh = object as THREE.Mesh
          if (mesh.isMesh) { mesh.castShadow = true; mesh.receiveShadow = true }
        })
        draw(timeRef.current)
        applyBodyDisplay(bodyDisplayRef.current)
        focus(true)
        setAvatarStatus('実測した学長の身体に対応付けて表示')
      }).catch((error) => {
        if (!disposed) setAvatarStatus(`${error instanceof Error ? error.message : String(error)} 物理形状で表示しています。`)
      })
    } else if (loaded.avatarProfile) setAvatarStatus('この記録には学長への対応付けに必要な身体姿勢がありません。物理形状で表示しています。')
    const resize = () => {
      const width = Math.max(1, container.clientWidth), height = Math.max(1, container.clientHeight)
      renderer.setSize(width, height)
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      if (eyeRenderer && eyeContainer) {
        // 枠は深度カメラと同じ縦横比なので、視野も深度カメラと一致する
        const eyeWidth = Math.max(1, eyeContainer.clientWidth), eyeHeight = Math.max(1, eyeContainer.clientHeight)
        eyeRenderer.setSize(eyeWidth, eyeHeight)
        eyeCamera.aspect = eyeWidth / eyeHeight
        eyeCamera.updateProjectionMatrix()
      }
    }
    resize()
    const observer = new ResizeObserver(resize)
    observer.observe(container)
    if (eyeContainer) observer.observe(eyeContainer)
    let frameId = 0, lastTime = performance.now(), lastNotification = 0
    const render = (now: number) => {
      frameId = requestAnimationFrame(render)
      const delta = Math.min(Math.max((now - lastTime) / 1000, 0), 0.05)
      lastTime = now
      if (playingRef.current && loaded.frames.length > 1) {
        const end = loaded.frames[loaded.frames.length - 1].time
        timeRef.current = Math.min(timeRef.current + delta, end)
        if (timeRef.current >= end) { playingRef.current = false; setPlaying(false) }
      }
      const frame = draw(timeRef.current)
      if (now - lastNotification >= 100) { lastNotification = now; setCurrentTime(timeRef.current); if (frame) setStats(frame.stats) }
      controls.update()
      updateWallDisplay()
      renderer.render(scene, camera)
      if (eyeRenderer && eyeReady) eyeRenderer.render(eyeScene, eyeCamera)
    }
    frameId = requestAnimationFrame(render)
    return () => {
      disposed = true
      viewRef.current = null
      cancelAnimationFrame(frameId)
      observer.disconnect()
      controls.dispose()
      for (const mesh of meshes.values()) { mesh.geometry.dispose(); for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) material.dispose() }
      avatar?.dispose()
      goalMarker?.geometry.dispose()
      if (goalMarker) for (const material of Array.isArray(goalMarker.material) ? goalMarker.material : [goalMarker.material]) material.dispose()
      sun.shadow.dispose()
      renderer.dispose()
      renderer.forceContextLoss()
      renderer.domElement.remove()
      for (const [, copy] of eyeMeshes) for (const material of Array.isArray(copy.material) ? copy.material : [copy.material]) material.dispose()
      if (eyeRenderer) { eyeRenderer.dispose(); eyeRenderer.forceContextLoss(); eyeRenderer.domElement.remove() }
    }
  }, [loaded])

  const seek = useCallback((time: number) => {
    const position = time >= endTime - 1e-9
      ? endTime
      : THREE.MathUtils.clamp(time, startTime, endTime)
    timeRef.current = position
    setCurrentTime(position)
    const frame = viewRef.current?.draw(position)
    if (frame) setStats(frame.stats)
  }, [startTime, endTime])
  const perception = loaded?.perception
  const eyeSpec = perception?.camera ?? DEFAULT_CAMERA
  const hasEyeView = !!loaded?.frames.some((frame) => frame.bodies.some((body) => body.name === eyeSpec.mountBody))
  const depthShots = perception?.kind === 'depth' ? perception.shots : []
  // 再生位置までに撮った最後の1枚(計画はこの画像までを地図に足して経路を引いている)
  let shotIndex = -1
  for (let index = 0; index < depthShots.length; index++) if (depthShots[index].time <= currentTime + 1e-6) shotIndex = index
  const currentShot = depthShots[Math.max(shotIndex, 0)]
  if (shotIndex < 0 && currentShot) shotIndex = 0
  const goToShot = (index: number) => {
    const shot = depthShots[index]
    if (!shot) return
    playingRef.current = false
    setPlaying(false)
    seek(shot.time)
  }
  const controllerText =loaded?.controller === 'PPO residual' ? 'PPOによる補正制御'
    : loaded?.navigation?.teacherDiagnostic ? '教師の経路補助＋関節PPO'
    : loaded?.controller === 'PPO command navigator with frozen PPO joint motor' ? '上位PPOの方向・歩調指示＋固定PPO関節制御'
    : loaded?.controller === 'zero command baseline with frozen PPO joint motor' ? '指示ゼロの基準制御＋固定PPO関節制御'
    : loaded?.controller === 'PPO navigation residual' ? 'PPOによる方向転換・回避の補正制御'
    : loaded?.controller === 'PD navigation baseline' ? '方向転換・回避の基準制御'
    : loaded?.controller === 'PPO residual comparison' ? '学習した歩行のみ・比較用'
    : loaded?.controller === 'PPO residual + whole-body goal approach' ? '学習した歩行＋必要時の物理接近'
    : loaded?.controller === 'PD baseline' ? 'PD制御の基準動作'
      : loaded?.controller === 'Whole-body inverse dynamics QP (not RL)' ? '身体全体の物理制御（学習なし）'
      : loaded?.controller === 'unpowered' ? '制御なし（重力・接触の検証）' : loaded?.controller ?? '記録を準備中'
  const friction = loaded?.geoms.find((geom) => /floor|ground/i.test(geom.id) && geom.friction?.length)?.friction
  const referenceText = loaded?.controller === 'unpowered' ? '参照姿勢の追従なし（無制御）'
    : loaded?.navigation?.teacherDiagnostic ? 'T2Mの歩行周期を土台にした関節目標。幾何計算で求めた経路から方向を指示し、学習済みPPOが関節トルクを制御'
    : loaded?.controller === 'PPO residual + whole-body goal approach' ? 'T2Mを土台にした足運び。必要な場合は、停止を確認して力学による最後の接近へ切替'
    : loaded?.controller === 'Whole-body inverse dynamics QP (not RL)' ? '実際の接地点と力学から求める関節目標'
    : loaded?.physicalFootstepPrior ? 'T2Mの周期・歩幅を土台に、足運びの関節目標を物理向けに調整'
    : loaded?.motionReference === 'cached T2M BVH' ? '保存済み T2M 動作を主要関節へ対応付け'
      : loaded?.motionReference === 'analytic stance' || loaded?.task === 'stand' ? '立位の基準姿勢'
        : loaded?.motionReference || '記録に付随する参照動作'
  const taskKey = loaded?.navigation ? loaded.navigation.mode === 'obstacle' ? 'singleFurnitureAvoidance' : 'turning'
    : loaded?.controller === 'unpowered' ? null : loaded?.task === 'walk' ? 'walking' : loaded?.task === 'sit' ? 'sitting' : loaded?.task === 'stand' ? 'standing' : null
  const taskLabel = loaded?.navigation ? loaded.navigation.mode === 'obstacle' ? '家具1個の回避' : '方向転換'
    : loaded?.task === 'walk' ? '歩行' : loaded?.task === 'sit' ? '着座' : '立位'
  const taskGate = taskKey ? currentStage?.gates?.[taskKey] : undefined
  const taskPassed = taskGate === true || taskGate === 'passed' || taskGate === 'validated'
  const failureExample = loaded?.failureExample
  const traceStatus = loaded?.viewerStatus ? loaded.viewerStatus
    : failureExample ? currentTime > failureExample.fallTimeSeconds ? '転倒後の物理続行（評価外）' : '追加学習途中の転倒例（評価外）'
    : loaded?.navigation?.teacherDiagnostic ? loaded.navigation.verified ? '教師補助の診断・通過' : '教師補助の診断・未通過'
    : loaded?.navigation ? loaded.navigation.verified ? `${taskLabel}の記録・検証通過` : `${taskLabel}の記録・未通過`
    : loaded?.controller === 'PD baseline' ? '学習後の動作と比較する基準制御'
    : loaded?.controller === 'PPO residual comparison' ? '物理制御による接近と比較する学習歩行'
    : loaded?.controller === 'Whole-body inverse dynamics QP (not RL)' ? '平床1 mの基準動作を検証'
    : taskKey ? taskPassed ? `${taskLabel}の検証を通過` : `検証中・${taskLabel}未通過`
    : loaded?.controller === 'unpowered' ? '無制御で重力・接触を検証' : ''
  const walkingEvaluation = record(currentStage?.metrics?.walkingEvaluation)
  const walkingCount = finite(walkingEvaluation.count, 0)
  const walkingCompleted = finite(walkingEvaluation.completionVerifiedRate, finite(walkingEvaluation.successRate, 0))
  const learnedCompleted = finite(walkingEvaluation.learnedOnlyCompletionVerifiedRate, NaN)
  const navigationEvaluation = record(currentStage?.metrics?.navigationEvaluation)
  const navigationCount = finite(navigationEvaluation.episodeCount, 0)
  const navigationCompleted = finite(navigationEvaluation.navigationVerifiedRate, 0)
  const commandEvaluations = record(currentStage?.metrics?.commandNavigationEvaluations)
  const turningEvaluation = record(currentStage?.metrics?.turningEvaluation)
  const turningCount = finite(turningEvaluation.episodeCount, 0)
  const commandResults = ([['turn', '方向転換'], ['obstacle', '家具1個の回避']] as const).flatMap(([mode, label]) => {
    const evaluation = record(commandEvaluations[mode])
    const count = finite(evaluation.episodeCount, 0)
    return count > 0 ? [{ mode, label, count, evaluation, passed: Math.round(finite(evaluation.navigationVerifiedRate, 0) * count) }] : []
  })
  const gateRows: [string, string][] = selected === 2 ? [['方向転換', 'turning'], ['家具1個の回避', 'singleFurnitureAvoidance'], ['複数家具の移動', 'multiFurnitureNavigation']]
    : [['立位', 'standing'], ['歩行', 'walking'], ['着座', 'sitting']]

  return (
    <main className="physics-lab" style={{ height: '100dvh', overflow: 'auto', background: '#0b1220', color: '#e6edf6', padding: '22px clamp(12px, 3vw, 36px)', boxSizing: 'border-box' }}>
      <style>{`.physics-lab *{box-sizing:border-box;color:inherit}.physics-lab button,.physics-lab a{font:inherit}.physics-lab button{cursor:pointer}.physics-lab button:disabled{opacity:.4;cursor:not-allowed}.physics-lab button:focus-visible,.physics-lab a:focus-visible,.physics-lab input:focus-visible{outline:2px solid #74b7ff;outline-offset:3px}.physics-stage-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.physics-content{display:grid;grid-template-columns:minmax(0,1fr) 290px;gap:18px}.physics-canvas{height:min(68vh,680px);min-height:430px}.physics-eyes{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}@media(max-width:850px){.physics-content{grid-template-columns:1fr}.physics-eyes{grid-template-columns:1fr}.physics-canvas{height:55vh;min-height:360px}.physics-stage-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:460px){.physics-lab{padding:14px 10px!important}.physics-canvas{min-height:330px}.physics-stage-grid{gap:7px}.physics-view-info{left:auto!important;right:12px;top:12px!important;width:100px;max-width:155px;padding:8px 9px!important;font-size:10px!important;line-height:1.4}}`}</style>
      <div style={{ maxWidth: 1440, margin: '0 auto' }}>
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, flexWrap: 'wrap', marginBottom: 22 }}>
          <div><p style={{ color: '#91a5bc', margin: '0 0 6px', fontSize: 12, letterSpacing: 1 }}>学長AI · 物理と学習の検証</p><h1 style={{ fontSize: 'clamp(22px, 3vw, 30px)', margin: 0 }}>動作が、重力のある世界で成立するか</h1><p style={{ color: '#a6b6c8', fontSize: 13, margin: '10px 0 0', lineHeight: 1.7 }}>物理シミュレーションの記録を再生。学長の身体に合わせた記録は、実際の学長アバターで表示します。</p></div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <Link href="/world-model/live" style={{ color: '#c5dcf4', border: '1px solid #34475f', borderRadius: 8, padding: '9px 13px', textDecoration: 'none', fontSize: 13 }}>ライブ</Link>
            <Link href="/world-model/training" style={{ color: '#c5dcf4', border: '1px solid #34475f', borderRadius: 8, padding: '9px 13px', textDecoration: 'none', fontSize: 13 }}>学習の様子</Link>
            <Link href="/" style={{ color: '#c5dcf4', border: '1px solid #34475f', borderRadius: 8, padding: '9px 13px', textDecoration: 'none', fontSize: 13 }}>対話・住宅画面へ</Link>
          </div>
        </header>
        <div className="physics-stage-grid" role="group" aria-label="検証段階" style={{ marginBottom: 18 }}>
          {[1, 2, 3, 4].map((number) => {
            const stage = stages.find((stage) => stage.stage === number)
            const active = selected === number
            return <button key={number} aria-pressed={active} onClick={() => setSelected(number)} style={{ textAlign: 'left', color: '#e7edf5', border: `1px solid ${active ? '#67a9eb' : '#283c52'}`, background: active ? '#18334e' : '#111e2e', borderRadius: 10, padding: '12px 14px', minHeight: 84 }}>
              <span style={{ display: 'block', color: active ? '#97ceff' : '#8fa6bd', fontSize: 11 }}>段階 {number}</span><strong style={{ display: 'block', fontSize: 14, margin: '5px 0' }}>{stage?.title ?? DEFAULT_TITLES[number - 1]}</strong><span style={{ fontSize: 11, color: '#b4c3d1' }}>{STATUS[stage?.status ?? 'locked'] ?? stage?.status}</span>
            </button>
          })}
        </div>
        {progress && <section aria-label="評価のまとめ" style={{ ...cardStyle, marginBottom: 18 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap', alignItems: 'baseline' }}>
            <strong style={{ fontSize: 14 }}>評価のまとめ（全身追従器 {progress.model}・計画 G1〜G6）</strong>
            <span style={{ color: '#9fb3c6', fontSize: 11 }}>各段階の【評価】の記録を「比較する記録」から再生できます</span>
          </div>
          <div style={{ overflowX: 'auto', marginTop: 10 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, minWidth: 640 }}>
              <thead><tr>{['段階', '内容', '結果', '基準', '状況'].map((head) => <th key={head} scope="col" style={{ textAlign: 'left', color: '#9fb3c6', fontWeight: 600, padding: '6px 8px', borderBottom: '1px solid #2c425a' }}>{head}</th>)}</tr></thead>
              <tbody>{progress.rows.map((row, index) => <tr key={`${row.goal}-${index}`} style={{ borderBottom: '1px solid #1f3348' }}>
                <td style={{ padding: '6px 8px', color: '#97ceff', whiteSpace: 'nowrap' }}>{row.goal}</td>
                <td style={{ padding: '6px 8px' }}>{row.title}</td>
                <td style={{ padding: '6px 8px', whiteSpace: 'nowrap', fontVariantNumeric: 'tabular-nums' }}>{row.result}</td>
                <td style={{ padding: '6px 8px', color: '#b2c3d3' }}>{row.criterion}</td>
                <td style={{ padding: '6px 8px', whiteSpace: 'nowrap', color: row.status === '達成' ? '#8fd6a0' : row.status === '未達' ? '#f0a59a' : '#e8cf8a' }}>{row.status}</td>
              </tr>)}</tbody>
            </table>
          </div>
          {progress.note && <p style={{ color: '#9fb3c6', fontSize: 11, margin: '8px 0 0', lineHeight: 1.6 }}>{progress.note}</p>}
        </section>}
        <div className="physics-content">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 18, minWidth: 0 }}>
          <section style={{ border: '1px solid #283b50', borderRadius: 12, overflow: 'hidden', background: '#101d2b' }}>
            <div className="physics-canvas" style={{ position: 'relative' }}>
              <div ref={containerRef} style={{ height: '100%', width: '100%' }} />
              <div className="physics-view-info" style={{ position: 'absolute', top: 14, left: 14, background: '#0b172bd9', padding: '9px 12px', borderRadius: 8, fontSize: 12 }}><strong>{controllerText}</strong><div style={{ color: '#a3b6ca', marginTop: 4 }}>{hasTrace ? '保存された物理結果 · 等速 1×' : '形状のプレビュー · 動作記録なし'}</div>{traceStatus && <div style={{ color: failureExample || loaded?.navigation && !loaded.navigation.verified ? '#f2c69e' : loaded?.navigation?.verified || taskPassed ? '#a3debd' : '#d5c398', marginTop: 5 }}>{traceStatus}</div>}{loaded?.geoms.some((geom) => /wall/i.test(geom.id)) && <div style={{ color: '#a3b6ca', marginTop: 5, fontSize: 11 }}>手前の壁を透過表示（接触判定は維持）</div>}</div>
              <div style={{ position: 'absolute', right: 12, bottom: 12, display: 'flex', gap: 7 }}><button style={buttonStyle} onClick={() => viewRef.current?.focus(true)}>身体を見る</button><button style={buttonStyle} onClick={() => viewRef.current?.focus(false)}>部屋全体</button></div>
              {(loading || error || !currentStage || currentStage.status === 'locked') && <div role="status" style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center', background: '#0b172bcc', padding: 30 }}><div><strong style={{ display: 'block', fontSize: 16 }}>{loading ? '物理記録を読み込んでいます' : error || '前の段階を検証してから進みます'}</strong><p style={{ color: '#b2c3d4', fontSize: 13, lineHeight: 1.7 }}>動作記録がない段階では、身体を動かして表示しません。</p><button style={buttonStyle} onClick={() => setRefresh((value) => value + 1)}>記録を更新</button></div></div>}
            </div>
            <div style={{ padding: '14px 16px', borderTop: '1px solid #283b50' }}>
              {loaded?.avatarProfile && <div style={{ marginBottom: 14 }}><div role="group" aria-label="身体の表示" style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>{([['avatar', '学長アバター'], ['overlay', '物理形状を重ねる'], ['physics', '物理形状のみ']] as const).map(([value, label]) => <button key={value} style={{ ...buttonStyle, background: bodyDisplay === value ? '#244e75' : buttonStyle.background }} aria-pressed={bodyDisplay === value} onClick={() => setBodyDisplay(value)}>{label}</button>)}</div><p role="status" style={{ color: '#a9c0d6', fontSize: 11, margin: '8px 0 0', lineHeight: 1.7 }}>{avatarStatus}</p></div>}
              {!!recordingOptions.length && <label style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 12, marginBottom: 12, flexWrap: 'wrap', color: '#b8cde1' }}>比較する記録<select aria-label="制御記録" value={currentTraceUrl ?? ''} onChange={(event) => setTraceChoice({ stage: selected, url: event.target.value })} style={{ background: '#152b42', color: '#e1edf9', border: '1px solid #435b76', borderRadius: 6, padding: '7px 10px', maxWidth: '100%' }}>{!!currentStage?.resultOptions?.length && <optgroup label="評価の記録（G2以降・全身追従器）">{currentStage.resultOptions.map((option) => <option key={option.url} value={option.url}>{option.title}</option>)}</optgroup>}<optgroup label="学習モデルの検証">{currentStage?.traceOptions?.map((option) => <option key={option.url} value={option.url}>{option.title}</option>)}</optgroup>{!!currentStage?.diagnosticOptions?.length && <optgroup label="教師の経路補助による診断">{currentStage.diagnosticOptions.map((option) => <option key={option.url} value={option.url}>{option.title}</option>)}</optgroup>}{!!currentStage?.liveOptions?.length && <optgroup label="学習中の最新の重み(自動更新)">{currentStage.liveOptions.map((option) => <option key={option.url} value={option.url}>{option.title}</option>)}</optgroup>}</select></label>}
              {!!currentStage?.diagnosticOptions?.length && <div role="note" style={{ border: '1px solid #626047', background: '#28281f', borderRadius: 8, padding: '12px 13px', marginBottom: 12, fontSize: 12, lineHeight: 1.8 }}><strong>教師の経路補助による診断 · {currentStage.diagnosticOptions.filter((option) => option.verified).length}/{currentStage.diagnosticOptions.length}条件で通過</strong><p style={{ margin: '5px 0 8px', color: '#d3d0b5' }}>家具1個を配置した検証用の部屋で、幾何計算による教師が経路と方向を指示し、学習済みの関節PPOが身体を動かします。住宅画面の写実的な家とは別の検証空間です。{currentStage.diagnosticFollowup && <><br />同じ設定で別の{currentStage.diagnosticFollowup.trialCount}条件を試した結果、通過は{currentStage.diagnosticFollowup.verifiedTrialCount}条件でした。部屋が変わっても安定して動ける段階には達していません。</>}<br />この診断の追加学習は0回です。経路を自力で判断する上位モデルの習得・フェーズ2の完了・対話画面への導入は、引き続き検証中です。</p><div style={{ display: 'flex', gap: 7, flexWrap: 'wrap' }}>{currentStage.diagnosticOptions.map((option) => <button key={option.url} style={{ ...buttonStyle, background: currentTraceUrl === option.url ? '#4b4930' : buttonStyle.background }} aria-pressed={currentTraceUrl === option.url} onClick={() => setTraceChoice({ stage: selected, url: option.url })}>{option.verified ? '通過した診断を見る' : '未通過の診断を見る'}</button>)}</div></div>}
              {failureExample && <div role="note" style={{ border: '1px solid #74523c', background: '#30241e', borderRadius: 8, padding: '12px 13px', marginBottom: 12, color: '#e9d2bd', fontSize: 12, lineHeight: 1.8 }}><strong>追加学習途中の転倒例 · 条件 {failureExample.seed}</strong><p style={{ margin: '5px 0 8px' }}>{failureExample.description}<br />転倒判定：{failureExample.fallTimeSeconds.toFixed(2)} 秒。現在の採用モデルの検証結果には含めません。</p><button style={buttonStyle} onClick={() => { playingRef.current = false; setPlaying(false); seek(Math.max(startTime, failureExample.fallTimeSeconds - 1)) }}>転倒直前へ</button></div>}
              {loaded?.navigation && <div role="note" style={{ border: '1px solid #365c6d', background: '#152b35', borderRadius: 8, padding: '12px 13px', marginBottom: 12, fontSize: 12, lineHeight: 1.8 }}><strong>{loaded.navigation.teacherDiagnostic ? '教師の経路補助による家具回避の診断' : taskLabel} · 部屋の条件 {loaded.navigation.sceneSeed}</strong><p style={{ margin: '5px 0 0', color: '#b6d1df' }}>緑の輪が最終目的地です。{loaded.navigation.mode === 'obstacle' ? '直進経路を遮る家具を避け、目的地で停止する課題です。家具の位置・寸法・向き、部屋の幅、床の摩擦を変えて確認します。' : '家具のない部屋で、正面からずれた目的地へ歩いて停止する課題です。'}<br />この記録：{loaded.navigation.verified ? '転倒・接触なし、歩行と到着・停止を確認' : loaded.navigation.terminalReason.includes('collision') ? '家具・壁への接触で試験終了' : loaded.navigation.terminalReason === 'fall' ? '転倒で試験終了' : '検証基準に未到達'}{loaded.navigation.teacherDiagnostic && <><br />教師が経路を補助した診断結果です。学習モデルだけによる家具回避の検証結果には加算しません。</>}</p></div>}
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <button style={{ ...buttonStyle, minWidth: 90 }} disabled={!hasMotion || loading} onClick={() => { if (!playing && currentTime >= endTime) seek(startTime); playingRef.current = !playing; setPlaying(!playing) }}>{playing ? '一時停止' : '記録を再生'}</button>
                <button style={buttonStyle} disabled={!hasTrace} onClick={() => { playingRef.current = false; setPlaying(false); seek(startTime) }}>最初へ</button>
                <span style={{ marginLeft: 'auto', whiteSpace: 'nowrap', fontSize: 12, color: '#b2c5d9', fontVariantNumeric: 'tabular-nums' }}>{(currentTime - startTime).toFixed(2)} / {(endTime - startTime).toFixed(2)} 秒</span>
              </div>
              <input aria-label="物理記録の再生位置" type="range" min={startTime} max={Math.max(sliderEndTime, startTime + 0.001)} step={0.001} value={currentTime} disabled={!hasMotion} onChange={(event) => seek(Number(event.target.value))} style={{ width: '100%', marginTop: 13, accentColor: '#84bcf4' }} />
            </div>
          </section>
          <section aria-label="学長の視点と、行動の判断に使った画像" style={{ border: '1px solid #283b50', borderRadius: 12, background: '#101d2b', padding: '14px 16px' }}>
            <div className="physics-eyes">
              <figure style={{ margin: 0, minWidth: 0 }}>
                <figcaption style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 8, flexWrap: 'wrap', marginBottom: 8 }}><strong style={{ fontSize: 14 }}>学長の視点</strong><span style={{ color: '#9fb3c6', fontSize: 11 }}>頭の位置から見た映像（再生に合わせて動きます）</span></figcaption>
                <div style={{ position: 'relative', width: '100%', aspectRatio: `${eyeSpec.width} / ${eyeSpec.height}`, borderRadius: 8, overflow: 'hidden', background: '#0d1724' }}>
                  <div ref={eyeContainerRef} style={{ position: 'absolute', inset: 0 }} />
                  {loaded && !hasEyeView && <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center', padding: 16, color: '#b2c3d4', fontSize: 12, lineHeight: 1.7 }}>この記録には頭の位置が含まれていないため、視点の映像を表示できません。</div>}
                </div>
                <p style={{ color: '#9fb3c6', fontSize: 11, lineHeight: 1.7, margin: '8px 0 0' }}>深度カメラと同じ位置・向きです（頭の付け根、腰の向き、{eyeSpec.downPitchDegrees}°下向き、横の視野{eyeSpec.horizontalFovDegrees}°）。自分の体と、床の赤い印は映しません。行動の判断には使っていない、確認用の映像です。</p>
              </figure>
              <figure style={{ margin: 0, minWidth: 0 }}>
                <figcaption style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 8, flexWrap: 'wrap', marginBottom: 8 }}><strong style={{ fontSize: 14 }}>行動の判断に使った画像</strong><span style={{ color: '#9fb3c6', fontSize: 11 }}>{perception?.kind === 'depth' ? `頭の深度カメラ ${perception.camera.width}×${perception.camera.height}・経路の計画に使用` : '画像は使っていません'}</span></figcaption>
                {perception?.kind === 'depth' && currentShot ? <>
                  <DepthShotView shot={currentShot} camera={perception.camera} unit={perception.unit} showObstacle={showObstacle} />
                  <div style={{ display: 'flex', alignItems: 'center', gap: 7, flexWrap: 'wrap', marginTop: 8, fontSize: 12 }}>
                    <button style={buttonStyle} disabled={shotIndex <= 0} onClick={() => goToShot(shotIndex - 1)}>前の撮影へ</button>
                    <button style={buttonStyle} disabled={shotIndex >= depthShots.length - 1} onClick={() => goToShot(shotIndex + 1)}>次の撮影へ</button>
                    <span style={{ color: '#d0deea', fontVariantNumeric: 'tabular-nums' }}>{shotIndex + 1}/{depthShots.length}枚目 · {(currentShot.time - startTime).toFixed(2)}秒に撮影{Math.abs(currentShot.time - currentTime) < 0.03 ? '（この瞬間）' : ''}</span>
                    <label style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 5, color: '#b8cde1' }}><input type="checkbox" checked={showObstacle} onChange={(event) => setShowObstacle(event.target.checked)} />障害物を赤で示す</label>
                  </div>
                  <p style={{ color: '#d0deea', fontSize: 12, lineHeight: 1.7, margin: '8px 0 0' }}>{currentShot.note} · 障害物として地図に足した画素 {currentShot.obstacleCount}</p>
                  <div aria-label="色の見方" style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginTop: 6, color: '#9fb3c6', fontSize: 11 }}>
                    <span>近い</span><span style={{ width: 90, height: 8, borderRadius: 4, background: `linear-gradient(90deg, ${DEPTH_STOPS.map(([stop, [r, g, b]]) => `rgb(${r},${g},${b}) ${stop * 100}%`).join(', ')})` }} /><span>遠い（{DEPTH_SCALE} m 以上）</span>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}><span style={{ width: 10, height: 10, background: '#090d15', border: '1px solid #445' }} />何にも当たらない</span>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}><span style={{ width: 10, height: 10, background: '#f04a46' }} />障害物（床から5 cm〜2 m）</span>
                  </div>
                </> : <div style={{ border: '1px dashed #34506b', borderRadius: 8, padding: '14px 15px', color: '#c2d2e1', fontSize: 12, lineHeight: 1.8, aspectRatio: perception?.kind === 'depth' ? undefined : `${eyeSpec.width} / ${eyeSpec.height}`, overflow: 'auto' }}>
                  {perception?.kind === 'depth' ? 'この記録には撮影した画像が保存されていません。' : perception ? 'この記録の行動は、画像を使わずに決めています。' : 'この記録には、判断に使った入力の情報がありません。'}
                </div>}
                {perception && <p style={{ color: '#b7c9d9', fontSize: 12, lineHeight: 1.8, margin: '10px 0 0' }}>{perception.summary}<br /><span style={{ color: '#9fb3c6' }}>{perception.tracker}</span></p>}
              </figure>
            </div>
          </section>
          </div>
          <aside style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div style={cardStyle}><strong style={{ fontSize: 14 }}>{failureExample ? '現在の採用モデルの結果' : 'この段階の結果'}</strong><p style={{ color: '#93c7f7', fontSize: 13, margin: '10px 0' }}>{STATUS[currentStage?.status ?? 'locked'] ?? currentStage?.status}</p><p style={{ color: '#b2c3d3', fontSize: 13, lineHeight: 1.8, marginBottom: 0 }}>{currentStage?.description ?? '重力と接触の下で身体を保ち、記録した結果から次の段階へ進めるかを確認します。'}</p></div>
            <div style={cardStyle}><strong style={{ fontSize: 14 }}>動作ごとの検証</strong><dl style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: '11px 8px', margin: '16px 0 0', fontSize: 12 }}>
              {gateRows.map(([label, key]) => {
                const gate = currentStage?.gates?.[key]
                const passed = gate === true || gate === 'passed' || gate === 'validated'
                const text = passed ? '検証通過' : gate === 'failed' || gate === false ? '未通過' : gate === 'training' ? '学習中' : '未検証'
                return <React.Fragment key={key}><dt style={{ color: '#a3b7cb' }}>{label}</dt><dd style={{ margin: 0, color: passed ? '#91d9bc' : '#b0bfce' }}>{text}</dd></React.Fragment>
              })}
            </dl></div>
            {walkingCount > 0 && <div style={cardStyle}><strong style={{ fontSize: 14 }}>歩行の検証範囲</strong><p style={{ color: '#b7c9d9', fontSize: 12, lineHeight: 1.9, margin: '10px 0 0' }}>平らな床で2 m先へ歩き、停止する課題。訓練外{walkingCount}条件のうち{Math.round(walkingCompleted * walkingCount)}条件で、到達・停止と両脚の反復歩行を確認しました。{Number.isFinite(learnedCompleted) && <><br />学習した歩行だけで通過：{Math.round(learnedCompleted * walkingCount)}条件<br />最後の接近に物理制御を使用：{finite(walkingEvaluation.modelBasedApproachCount, 0)}条件</>}<br />家具の回避や階段の習得は、次の段階で検証します。</p></div>}
            {navigationCount > 0 && <div style={cardStyle}><strong style={{ fontSize: 14 }}>{commandResults.length ? '関節PPOによる家具回避の検証範囲' : '家具回避の検証範囲'}</strong><p style={{ color: '#b7c9d9', fontSize: 12, lineHeight: 1.9, margin: '10px 0 0' }}>平床・家具1個の課題。訓練外{navigationCount}配置のうち{Math.round(navigationCompleted * navigationCount)}配置で、接触・転倒なし、両脚の歩行と到着・停止を確認しました。<br />接触は2ミリ秒ごとの物理計算で確認します。複数家具や複数部屋への対応は、続く課題です。</p></div>}
            {commandResults.length > 0 && <div style={cardStyle}><strong style={{ fontSize: 14 }}>方向・歩調PPOの検証</strong><p style={{ color: '#b7c9d9', fontSize: 12, lineHeight: 1.9, margin: '10px 0 0' }}>歩行の関節制御モデルを固定し、方向・歩調を指示する上位モデルだけを、成功例の模倣や強化学習で追加学習します。</p>{commandResults.map(({ mode, label, count, passed, evaluation }) => <p key={mode} style={{ color: '#b7c9d9', fontSize: 12, lineHeight: 1.9, margin: '10px 0 0' }}><strong>{label} · {evaluation.evaluationRole === 'final' ? '最終検証' : '開発検証'}</strong><br />訓練外{count}条件のうち{passed}条件で、接触・転倒なし、両脚の歩行と到着・停止を確認（{Math.round(passed / count * 100)}%）。<br />{evaluation.proficiencyPassed === true ? 'この課題の習得基準を通過' : 'この課題の習得判定には未到達'}</p>)}{turningCount > 0 && <p style={{ color: '#a9bdce', fontSize: 12, lineHeight: 1.9, margin: '10px 0 0' }}>比較する関節PPOの方向転換（{turningEvaluation.evaluationRole === 'final' ? '最終検証' : '開発検証'}）：訓練外{turningCount}条件中{Math.round(finite(turningEvaluation.navigationVerifiedRate, 0) * turningCount)}条件で通過。モデルごとに検証条件と件数が異なります。</p>}<p style={{ color: '#a9bdce', fontSize: 12, lineHeight: 1.9, margin: '10px 0 0' }}>複数家具・複数部屋・階段は未習得です。フェーズ2全体の検証は続きます。</p></div>}
            <div style={cardStyle}><strong style={{ fontSize: 14 }}>再生中の物理状態</strong><dl style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: '11px 8px', margin: '16px 0 0', fontSize: 12 }}>
              {([['腰の高さ', 'root_height', ' m'], ['接触の数', 'contact_count', ''], ['左足の接触', 'left_foot_contact', ''], ['右足の接触', 'right_foot_contact', ''], ['左足の荷重', 'left_foot_force', ' N'], ['右足の荷重', 'right_foot_force', ' N'], ['転倒', 'fall', ''], ['目標まで', 'goal_distance', ' m']] as const).map(([label, key, unit]) => <React.Fragment key={key}><dt style={{ color: '#a3b7cb' }}>{label}</dt><dd style={{ margin: 0, fontVariantNumeric: 'tabular-nums' }}>{metric(stats, key, unit)}</dd></React.Fragment>)}
              {loaded?.navigation && <><dt style={{ color: '#a3b7cb' }}>家具・壁への接触</dt><dd style={{ margin: 0 }}>{metric(stats, 'collision_occurred')}</dd></>}
            </dl></div>
            <div style={cardStyle}><strong style={{ fontSize: 14 }}>身体と環境</strong><p style={{ color: '#b2c3d3', fontSize: 12, lineHeight: 1.9, marginBottom: 0 }}>身体：主要関節を持つ剛体モデル<br />重力：{loaded?.gravity ? `${Math.hypot(...loaded.gravity).toFixed(2)} m/s²` : '記録に付随する条件を使用'}<br />床の摩擦：{friction?.length ? friction[0].toFixed(2) : '記録に付随する条件を使用'}<br />床・家具：物理計算に使う衝突形状</p></div>
            <div style={{ ...cardStyle, background: '#15293b' }}><strong style={{ fontSize: 14 }}>T2M と学習制御</strong><p style={{ color: '#d0deea', fontSize: 12, lineHeight: 1.9, marginTop: 10 }}>この記録の参照動作：{referenceText}</p><p style={{ color: '#b7c9d9', fontSize: 12, lineHeight: 1.9, marginBottom: 0 }}>対話画面の T2M は動作を生成します。物理検証では記録ごとの参照姿勢と関節制御を使い、PPO が姿勢の補正を学習します。PPO の補正制御は、学習した記録がある場合に表示します。<br /><br />ここでは VRM アバターの表示と、身体の物理検証を分けて確認できます。</p></div>
            {currentStage?.metrics && Object.keys(currentStage.metrics).length > 0 && <details style={cardStyle}><summary style={{ fontSize: 12, cursor: 'pointer' }}>検証記録の詳細</summary><pre style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', fontSize: 11, color: '#a8bdd0', lineHeight: 1.6 }}>{JSON.stringify(currentStage.metrics, null, 2)}</pre></details>}
          </aside>
        </div>
      </div>
    </main>
  )
}

const buttonStyle: React.CSSProperties = { color: '#dbe8f5', border: '1px solid #415975', background: '#1b3047', borderRadius: 7, padding: '8px 11px', fontSize: 12 }
const cardStyle: React.CSSProperties = { background: '#111f30', border: '1px solid #283e55', borderRadius: 10, padding: '16px 17px' }
