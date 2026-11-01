import * as THREE from 'three'
import { BVHLoader } from 'three/examples/jsm/loaders/BVHLoader.js'

export type RoomPrimitiveKind = 'walk' | 'sit' | 'stand'
export type RoomVector3 = [number, number, number]

export type RoomPoseGeometry = {
  hips: RoomVector3
  leftHip: RoomVector3
  rightHip: RoomVector3
  leftKnee: RoomVector3
  rightKnee: RoomVector3
  leftFoot: RoomVector3
  rightFoot: RoomVector3
  leftToe: RoomVector3
  rightToe: RoomVector3
  head: RoomVector3
  /** Lowest foot/toe joint, not the rendered shoe sole. */
  footFloorY: number
  hipToFootFloor: number
  leftKneeFlexionDegrees: number
  rightKneeFlexionDegrees: number
  /** atan2(forward.x, forward.z), measured from hips toward both knees. */
  kneeForwardHeadingRadians: number
}

export type RoomWalkLoop = {
  startFrame: number
  endFrame: number
  startSeconds: number
  endSeconds: number
  periodSeconds: number
  strideMeters: number
  speedMetersPerSecond: number
  travelHeadingRadians: number
  rootYRangeMeters: number
  boundaryMaxRotationRadians: number
  boundaryRootYErrorMeters: number
}

export type RoomPrimitiveMetadata = {
  kind: RoomPrimitiveKind
  sourceUrl: string
  provenance: 't2m_generated' | 'reversed_t2m_generated_sit'
  intentPrompt: string
  generationPrompt: string | null
  sourceDurationSeconds: number
  durationSeconds: number
  frameCount: number
  frameTimeSeconds: number
  sourceRestHeightMeters: number
  normalizedHeightMeters: number
  scale: number
  /** A grounded T-pose root, derived from the static skeleton's offsets. */
  restGroundedRootPosition: RoomVector3
  /** Prepared playback positions; the walk has X/Z removed. */
  rootStartPosition: RoomVector3
  rootEndPosition: RoomVector3
  /** Full normalized BVH positions before walk X/Z removal and trimming. */
  sourceRootStartPosition: RoomVector3
  sourceRootEndPosition: RoomVector3
  /** Use placement yaw = requested heading - nativeHeadingRadians. */
  nativeHeadingRadians: number
  firstPose: RoomPoseGeometry
  finalPose: RoomPoseGeometry
  walkLoop?: RoomWalkLoop
}

export type RoomMotionPrimitive = {
  kind: RoomPrimitiveKind
  text: string
  root: THREE.Bone
  skeleton: THREE.Skeleton
  /** Ready for the shared walk skeleton: rotations plus root position only. */
  clip: THREE.AnimationClip
  /** Full normalized clip retained for provenance, analysis or one-shot use. */
  sourceClip: THREE.AnimationClip
  metadata: RoomPrimitiveMetadata
}

export type RoomMotionPrimitives = Record<RoomPrimitiveKind, RoomMotionPrimitive>

export const ROOM_PRIMITIVE_HEIGHT = 1.8
export const ROOM_WALK_LOOP_FRAMES = { start: 260, end: 372 } as const
export const ROOM_PRIMITIVE_URLS: Record<RoomPrimitiveKind, string> = {
  walk: '/motions/room-prototype/walk.bvh',
  sit: '/motions/room-prototype/sit.bvh',
  stand: '/motions/room-prototype/stand.bvh',
}

type ParsedMotion = { skeleton: THREE.Skeleton; clip: THREE.AnimationClip }

const boneNamed = (skeleton: THREE.Skeleton, name: string) => {
  const bone = skeleton.bones.find((candidate) => candidate.name === name)
  if (!bone) throw new Error(`Room motion is missing the ${name} joint`)
  return bone
}

const positionTrack = (clip: THREE.AnimationClip, rootName: string) => {
  const track = clip.tracks.find((candidate) => candidate.name === `${rootName}.position`)
  if (!track || track.getValueSize() !== 3) throw new Error('Room motion has no root position track')
  return track
}

const rootPosition = (track: THREE.KeyframeTrack, frame: number): RoomVector3 =>
  [track.values[frame * 3], track.values[frame * 3 + 1], track.values[frame * 3 + 2]]

/** Convert the rest axes and all animation channels together, preserving FK. */
export const normalizeRoomBvh = (
  motion: ParsedMotion, targetHeight = ROOM_PRIMITIVE_HEIGHT,
) => {
  if (!Number.isFinite(targetHeight) || targetHeight <= 0) throw new Error('Invalid room motion height')
  const { skeleton, clip } = motion
  const root = skeleton.bones[0]
  if (!root) throw new Error('Room motion has no skeleton')
  root.updateMatrixWorld(true)
  const world = (name: string) => boneNamed(skeleton, name).getWorldPosition(new THREE.Vector3())
  const up = world('Head').sub(world('Hips')).normalize()
  const left = world('LeftUpLeg').sub(world('RightUpLeg')).normalize()
  if (up.lengthSq() < 0.9 || left.lengthSq() < 0.9) throw new Error('Invalid room motion rest axes')
  if (up.y < 0.95 || left.x < 0.95) {
    const y = up.clone().addScaledVector(left, -up.dot(left)).normalize()
    const z = new THREE.Vector3().crossVectors(left, y).normalize()
    if (y.lengthSq() < 0.9 || z.lengthSq() < 0.9) throw new Error('Invalid room motion coordinate basis')
    const rotation = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(left, y, z)).invert()
    const inverse = rotation.clone().invert()
    for (const bone of skeleton.bones) bone.position.applyQuaternion(rotation)
    const vector = new THREE.Vector3()
    const quaternion = new THREE.Quaternion()
    for (const track of clip.tracks) {
      if (track.name.endsWith('.position')) {
        for (let i = 0; i < track.values.length; i += 3) {
          vector.fromArray(track.values, i).applyQuaternion(rotation).toArray(track.values, i)
        }
      } else if (track.name.endsWith('.quaternion')) {
        for (let i = 0; i < track.values.length; i += 4) {
          quaternion.fromArray(track.values, i).premultiply(rotation).multiply(inverse).normalize().toArray(track.values, i)
        }
      }
    }
  }
  root.updateMatrixWorld(true)
  const bounds = new THREE.Box3()
  for (const bone of skeleton.bones) bounds.expandByPoint(bone.getWorldPosition(new THREE.Vector3()))
  const sourceHeight = bounds.max.y - bounds.min.y
  if (!Number.isFinite(sourceHeight) || sourceHeight < 1e-6) throw new Error('Invalid room motion rest height')
  const scale = targetHeight / sourceHeight
  const groundedRoot = root.position.clone()
  groundedRoot.y -= bounds.min.y
  groundedRoot.multiplyScalar(scale)
  for (const bone of skeleton.bones) bone.position.multiplyScalar(scale)
  for (const track of clip.tracks) {
    if (!track.name.endsWith('.position')) continue
    for (let i = 0; i < track.values.length; i++) track.values[i] *= scale
  }
  root.updateMatrixWorld(true)
  return { sourceHeight, scale, groundedRoot }
}

/** Keep canonical bone offsets when several clips share one installed skeleton. */
export const stripRoomChildPositionTracks = (clip: THREE.AnimationClip, rootName = 'Hips') => {
  const copy = clip.clone()
  copy.tracks = copy.tracks.filter((track) =>
    !track.name.endsWith('.position') || track.name === `${rootName}.position`,
  )
  return copy
}

/** Move the character with the room controller while retaining generated hip bob. */
export const removeRoomWalkRootXZ = (clip: THREE.AnimationClip, rootName = 'Hips') => {
  const copy = clip.clone()
  const track = positionTrack(copy, rootName)
  for (let i = 0; i < track.values.length; i += 3) {
    track.values[i] = 0
    track.values[i + 2] = 0
  }
  return copy
}

/** Retain the inclusive matching endpoint, so one full generated gait repeats. */
const clipFrameRange = (
  source: THREE.AnimationClip, start: number, end: number, frameTime: number,
) => {
  const clip = source.clone()
  for (const track of clip.tracks) {
    if (end >= track.times.length) throw new Error('Walk loop exceeds the available BVH frames')
    const size = track.getValueSize()
    track.times = new Float32Array(Array.from({ length: end - start + 1 }, (_, i) => i * frameTime))
    track.values = track.values.slice(start * size, (end + 1) * size)
  }
  clip.duration = (end - start) * frameTime
  return clip
}

/** Replay an existing generated sit in reverse; no new motion is synthesized. */
export const reverseRoomMotionClip = (source: THREE.AnimationClip) => {
  const copy = source.clone()
  for (const track of copy.tracks) {
    const size = track.getValueSize()
    const values = track.values.slice()
    const times = track.times.slice()
    for (let i = 0; i < times.length; i++) {
      const opposite = times.length - 1 - i
      track.times[i] = source.duration - times[opposite]
      for (let j = 0; j < size; j++) track.values[i * size + j] = values[opposite * size + j]
    }
  }
  return copy
}

const samplePose = (root: THREE.Bone, skeleton: THREE.Skeleton, clip: THREE.AnimationClip, time: number) => {
  const saved = skeleton.bones.map((bone) => ({ bone, position: bone.position.clone(), quaternion: bone.quaternion.clone() }))
  const mixer = new THREE.AnimationMixer(root)
  const action = mixer.clipAction(clip).setLoop(THREE.LoopOnce, 1)
  action.clampWhenFinished = true
  action.play()
  mixer.setTime(time)
  root.updateMatrixWorld(true)
  const world = (name: string) => boneNamed(skeleton, name).getWorldPosition(new THREE.Vector3())
  const hips = world('Hips'), leftHip = world('LeftUpLeg'), rightHip = world('RightUpLeg')
  const leftKnee = world('LeftLeg'), rightKnee = world('RightLeg')
  const leftFoot = world('LeftFoot'), rightFoot = world('RightFoot')
  const leftToe = world('LeftToeBase'), rightToe = world('RightToeBase')
  const forward = leftKnee.clone().add(rightKnee).multiplyScalar(0.5).sub(hips)
  const footFloorY = Math.min(leftFoot.y, rightFoot.y, leftToe.y, rightToe.y)
  const flexion = (hip: THREE.Vector3, knee: THREE.Vector3, foot: THREE.Vector3) =>
    THREE.MathUtils.radToDeg(Math.PI - hip.clone().sub(knee).angleTo(foot.clone().sub(knee)))
  const pose: RoomPoseGeometry = {
    hips: hips.toArray(), leftHip: leftHip.toArray(), rightHip: rightHip.toArray(),
    leftKnee: leftKnee.toArray(), rightKnee: rightKnee.toArray(),
    leftFoot: leftFoot.toArray(), rightFoot: rightFoot.toArray(),
    leftToe: leftToe.toArray(), rightToe: rightToe.toArray(), head: world('Head').toArray(),
    footFloorY, hipToFootFloor: hips.y - footFloorY,
    leftKneeFlexionDegrees: flexion(leftHip, leftKnee, leftFoot),
    rightKneeFlexionDegrees: flexion(rightHip, rightKnee, rightFoot),
    kneeForwardHeadingRadians: Math.atan2(forward.x, forward.z),
  }
  action.stop()
  mixer.uncacheRoot(root)
  for (const item of saved) {
    item.bone.position.copy(item.position)
    item.bone.quaternion.copy(item.quaternion)
  }
  root.updateMatrixWorld(true)
  return pose
}

const loopRotationError = (clip: THREE.AnimationClip) => {
  const start = new THREE.Quaternion(), end = new THREE.Quaternion()
  let error = 0
  for (const track of clip.tracks) {
    if (!track.name.endsWith('.quaternion')) continue
    start.fromArray(track.values, 0)
    end.fromArray(track.values, track.values.length - 4)
    // Float32 storage introduces norm rounding; normalize before comparison.
    error = Math.max(error, start.normalize().angleTo(end.normalize()))
  }
  return error
}

export const parseRoomMotionPrimitive = (kind: RoomPrimitiveKind, text: string): RoomMotionPrimitive => {
  const frames = Number(text.match(/^\s*Frames:\s*(\d+)/m)?.[1])
  const frameTime = Number(text.match(/^\s*Frame Time:\s*([^\s]+)/m)?.[1])
  if (!Number.isInteger(frames) || frames < 2 || !Number.isFinite(frameTime) || frameTime <= 0) {
    throw new Error('Room motion has invalid BVH frame metadata')
  }
  const parsed = new BVHLoader().parse(text)
  const root = parsed.skeleton.bones[0]
  const normalized = normalizeRoomBvh(parsed)
  const sourceClip = parsed.clip
  sourceClip.duration = (frames - 1) * frameTime
  const sourcePosition = positionTrack(sourceClip, root.name)
  if (sourcePosition.times.length !== frames) throw new Error('Room motion frame count differs from its tracks')
  for (const track of sourceClip.tracks) {
    if (Array.from(track.values).some((value) => !Number.isFinite(value))) throw new Error('Room motion has non-finite channels')
  }
  let clip = sourceClip.clone()
  let walkLoop: RoomWalkLoop | undefined
  if (kind === 'walk') {
    const { start, end } = ROOM_WALK_LOOP_FRAMES
    clip = clipFrameRange(sourceClip, start, end, frameTime)
    const loopPosition = positionTrack(clip, root.name)
    const first = rootPosition(loopPosition, 0)
    const last = rootPosition(loopPosition, loopPosition.times.length - 1)
    const dx = last[0] - first[0], dz = last[2] - first[2]
    const stride = Math.hypot(dx, dz)
    const heights = Array.from(loopPosition.values).filter((_, i) => i % 3 === 1)
    walkLoop = {
      startFrame: start, endFrame: end, startSeconds: start * frameTime, endSeconds: end * frameTime,
      periodSeconds: clip.duration, strideMeters: stride, speedMetersPerSecond: stride / clip.duration,
      travelHeadingRadians: Math.atan2(dx, dz), rootYRangeMeters: Math.max(...heights) - Math.min(...heights),
      boundaryMaxRotationRadians: loopRotationError(clip), boundaryRootYErrorMeters: Math.abs(last[1] - first[1]),
    }
    clip = removeRoomWalkRootXZ(clip, root.name)
  }
  clip = stripRoomChildPositionTracks(clip, root.name)
  clip.name = `RoomPrototype:${kind}`
  const firstPose = samplePose(root, parsed.skeleton, clip, 0)
  const finalPose = samplePose(root, parsed.skeleton, clip, clip.duration)
  const preparedPosition = positionTrack(clip, root.name)
  const sourceFirst = rootPosition(sourcePosition, 0)
  const sourceLast = rootPosition(sourcePosition, frames - 1)
  const seatedPose = kind === 'stand' ? firstPose : finalPose
  const metadata: RoomPrimitiveMetadata = {
    kind, sourceUrl: ROOM_PRIMITIVE_URLS[kind],
    provenance: kind === 'stand' ? 'reversed_t2m_generated_sit' : 't2m_generated',
    intentPrompt: { walk: '歩く', sit: '椅子に座る', stand: '椅子から立ち上がる' }[kind],
    generationPrompt: kind === 'walk' ? '立っている状態から、歩く。' : null,
    sourceDurationSeconds: sourceClip.duration, durationSeconds: clip.duration,
    frameCount: frames, frameTimeSeconds: frameTime,
    sourceRestHeightMeters: normalized.sourceHeight, normalizedHeightMeters: ROOM_PRIMITIVE_HEIGHT,
    scale: normalized.scale, restGroundedRootPosition: normalized.groundedRoot.toArray(),
    rootStartPosition: rootPosition(preparedPosition, 0),
    rootEndPosition: rootPosition(preparedPosition, preparedPosition.times.length - 1),
    sourceRootStartPosition: sourceFirst, sourceRootEndPosition: sourceLast,
    nativeHeadingRadians: walkLoop?.travelHeadingRadians ?? seatedPose.kneeForwardHeadingRadians,
    firstPose, finalPose, walkLoop,
  }
  return { kind, text, root, skeleton: parsed.skeleton, clip, sourceClip, metadata }
}

let cachedTexts: Promise<Record<RoomPrimitiveKind, string>> | null = null

/** Cache downloads, but return fresh rigs/clips so callers never share mixers. */
export const loadRoomMotionPrimitives = async (): Promise<RoomMotionPrimitives> => {
  if (!cachedTexts) {
    cachedTexts = Promise.all((['walk', 'sit', 'stand'] as const).map(async (kind) => {
      const response = await fetch(ROOM_PRIMITIVE_URLS[kind])
      if (!response.ok) throw new Error(`Could not load room ${kind} motion (${response.status})`)
      return [kind, await response.text()] as const
    })).then((items) => Object.fromEntries(items) as Record<RoomPrimitiveKind, string>)
      .catch((error) => { cachedTexts = null; throw error })
  }
  const texts = await cachedTexts
  return {
    walk: parseRoomMotionPrimitive('walk', texts.walk),
    sit: parseRoomMotionPrimitive('sit', texts.sit),
    stand: parseRoomMotionPrimitive('stand', texts.stand),
  }
}
