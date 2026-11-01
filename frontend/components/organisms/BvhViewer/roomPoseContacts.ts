import * as THREE from 'three'
import type { RoomChairTarget } from './roomNavigation'

export type RoomFootReference = {
  ankleClearance: number
  worldQuaternion: THREE.Quaternion
}
export type RoomFootContactReference = {
  left: RoomFootReference
  right: RoomFootReference
  forward: THREE.Vector3
  floorY: number
}
export type RoomFootTarget = {
  position: THREE.Vector3
  quaternion: THREE.Quaternion
  ankleClearance: number
}
export type RoomSeatContactTargets = {
  pelvis: THREE.Vector3
  seatCenter: THREE.Vector3
  pelvisClearance: number
  leftFoot: RoomFootTarget
  rightFoot: RoomFootTarget
  forward: THREE.Vector3
  floorY: number
}
export type RoomSeatContactOptions = {
  footReference?: RoomFootContactReference | null
  floorY?: number
  ankleClearance?: number
  pelvisClearance?: number
  forwardReach?: number
  lateralFootSpacing?: number
  rearPelvisOffset?: number
}
export type RoomLegContactMetrics = {
  upperLength: number
  lowerLength: number
  requestedReach: number
  reachableReach: number
  reachClamped: boolean
  footTargetError: number
  reachableTargetError: number
  footFloorError: number
  toeHeight: number | null
  kneePosition: [number, number, number]
  footPosition: [number, number, number]
}
export type RoomPoseContactMetrics = {
  applied: boolean
  hipSeatError: number
  hipSeatHeight: number
  maxFootTargetError: number
  maxFootFloorError: number
  maxReachError: number
  legs: { left: RoomLegContactMetrics | null; right: RoomLegContactMetrics | null }
}

type BoneMap = ReadonlyMap<string, THREE.Bone>
type Leg = { upper: THREE.Bone; knee: THREE.Bone; foot: THREE.Bone; toe?: THREE.Bone }
const EPSILON = 1e-6
const UP = new THREE.Vector3(0, 1, 0)

const getLeg = (bones: BoneMap, side: 'left' | 'right'): Leg | null => {
  const upper = bones.get(`${side}UpLeg`)
  const knee = bones.get(`${side}LowerLeg`)
  const foot = bones.get(`${side}Foot`)
  return upper && knee && foot ? { upper, knee, foot, toe: bones.get(`${side}Toe`) } : null
}
const setWorldPosition = (bone: THREE.Bone, world: THREE.Vector3) => {
  if (bone.parent) {
    bone.parent.updateWorldMatrix(true, false)
    bone.position.copy(bone.parent.worldToLocal(world.clone()))
  } else bone.position.copy(world)
  bone.updateWorldMatrix(true, true)
}
const setWorldQuaternion = (bone: THREE.Bone, quaternion: THREE.Quaternion) => {
  const local = quaternion.clone()
  if (bone.parent) local.premultiply(bone.parent.getWorldQuaternion(new THREE.Quaternion()).invert())
  bone.quaternion.copy(local.normalize())
  bone.updateWorldMatrix(true, true)
}
const horizontal = (value: THREE.Vector3, fallback = new THREE.Vector3(0, 0, 1)) => {
  const result = value.clone().setY(0)
  return result.lengthSq() > EPSILON * EPSILON ? result.normalize() : fallback.clone()
}

/** Capture once in the grounded standing pose; orientation remains correct for different bone axes. */
export const captureStandingFootContacts = (
  bones: BoneMap, standingHeight = 1.8, floorY = 0,
): RoomFootContactReference | null => {
  const left = getLeg(bones, 'left')
  const right = getLeg(bones, 'right')
  if (!left || !right) return null
  bones.get('hips')?.updateWorldMatrix(true, true)
  const scale = Math.max(0.1, standingHeight / 1.8)
  const toeDirection = new THREE.Vector3()
  for (const leg of [left, right]) {
    if (leg.toe) toeDirection.add(leg.toe.getWorldPosition(new THREE.Vector3())
      .sub(leg.foot.getWorldPosition(new THREE.Vector3())))
  }
  const ref = (foot: THREE.Bone): RoomFootReference => ({
    ankleClearance: THREE.MathUtils.clamp(foot.getWorldPosition(new THREE.Vector3()).y - floorY, 0.02 * scale, 0.13 * scale),
    worldQuaternion: foot.getWorldQuaternion(new THREE.Quaternion()).clone(),
  })
  return { left: ref(left.foot), right: ref(right.foot), forward: horizontal(toeDirection), floorY }
}

/** A fallback calibration when no standing pose was supplied; aligns each toe while preserving its bone axes. */
const fallbackFootQuaternion = (leg: Leg, forward: THREE.Vector3, ankleClearance: number) => {
  const worldQuaternion = leg.foot.getWorldQuaternion(new THREE.Quaternion())
  if (!leg.toe) return worldQuaternion
  const toeVector = leg.toe.getWorldPosition(new THREE.Vector3())
    .sub(leg.foot.getWorldPosition(new THREE.Vector3()))
  const footLength = toeVector.length()
  if (footLength < EPSILON) return worldQuaternion
  // A toe is normally below the ankle; keep it just above the floor rather than pointing upwards.
  const down = THREE.MathUtils.clamp((ankleClearance - 0.008) / footLength, 0, 0.75)
  const desired = forward.clone().multiplyScalar(Math.sqrt(1 - down * down)).addScaledVector(UP, -down)
  return new THREE.Quaternion().setFromUnitVectors(toeVector.normalize(), desired)
    .multiply(worldQuaternion).normalize()
}

/** Generate immutable world contacts for a fixed chair; geometry and actor scale are never modified. */
export const createSeatContactTargets = (
  bones: BoneMap, chair: RoomChairTarget, standingHeight = 1.8, options: RoomSeatContactOptions = {},
): RoomSeatContactTargets | null => {
  const hips = bones.get('hips')
  const left = getLeg(bones, 'left')
  const right = getLeg(bones, 'right')
  if (!hips || !left || !right) return null
  hips.updateWorldMatrix(true, true)
  const scale = Math.max(0.1, standingHeight / 1.8)
  const floorY = options.floorY ?? options.footReference?.floorY ?? 0
  const forward = horizontal(new THREE.Vector3(chair.forward.x, 0, chair.forward.z))
  const lateral = new THREE.Vector3().crossVectors(UP, forward).normalize()
  const pelvisClearance = options.pelvisClearance ?? 0.055 * scale
  const seatCenter = new THREE.Vector3(chair.position.x, chair.seatHeight, chair.position.z)
  const pelvis = seatCenter.clone().addScaledVector(forward, -(options.rearPelvisOffset ?? 0.045 * scale))
  pelvis.y += pelvisClearance
  const reach = options.forwardReach ?? 0.42 * scale
  const spacing = options.lateralFootSpacing ?? 0.1 * scale
  const reference = options.footReference
  const refYaw = reference ? Math.atan2(reference.forward.x, reference.forward.z) : 0
  const targetYaw = Math.atan2(forward.x, forward.z)
  const turn = new THREE.Quaternion().setFromAxisAngle(UP, targetYaw - refYaw)
  const target = (leg: Leg, side: 'left' | 'right', sign: number): RoomFootTarget => {
    const ref = reference?.[side]
    const ankleClearance = options.ankleClearance ?? ref?.ankleClearance ?? 0.075 * scale
    const position = pelvis.clone().addScaledVector(forward, reach).addScaledVector(lateral, sign * spacing)
    position.y = floorY + ankleClearance
    return {
      position, ankleClearance,
      quaternion: ref ? turn.clone().multiply(ref.worldQuaternion).normalize()
        : fallbackFootQuaternion(leg, forward, ankleClearance),
    }
  }
  // A +Z-facing humanoid's anatomical left is +X in this project's canonical rigs.
  return {
    pelvis, seatCenter, pelvisClearance, forward, floorY,
    leftFoot: target(left, 'left', 1), rightFoot: target(right, 'right', -1),
  }
}

/** Aim a segment through its parent's coordinates, so translated/rotated/scaled wrappers remain supported. */
const aimSegment = (joint: THREE.Bone, end: THREE.Bone, worldTarget: THREE.Vector3) => {
  joint.updateWorldMatrix(true, true)
  const origin = joint.getWorldPosition(new THREE.Vector3())
  const current = end.getWorldPosition(new THREE.Vector3())
  const desired = worldTarget.clone()
  if (joint.parent) {
    joint.parent.updateWorldMatrix(true, false)
    joint.parent.worldToLocal(origin)
    joint.parent.worldToLocal(current)
    joint.parent.worldToLocal(desired)
  }
  current.sub(origin)
  desired.sub(origin)
  if (current.lengthSq() < EPSILON * EPSILON || desired.lengthSq() < EPSILON * EPSILON) return
  const delta = new THREE.Quaternion().setFromUnitVectors(current.normalize(), desired.normalize())
  joint.quaternion.premultiply(delta).normalize()
  joint.updateWorldMatrix(true, true)
}

/** Analytic two-bone IK, preserving segment lengths and explicitly choosing the forward knee bend. */
const solveLeg = (
  leg: Leg, target: RoomFootTarget, forward: THREE.Vector3, floorY: number, strength: number,
): RoomLegContactMetrics => {
  leg.upper.updateWorldMatrix(true, true)
  const hip = leg.upper.getWorldPosition(new THREE.Vector3())
  const knee = leg.knee.getWorldPosition(new THREE.Vector3())
  const foot = leg.foot.getWorldPosition(new THREE.Vector3())
  const upperLength = hip.distanceTo(knee)
  const lowerLength = knee.distanceTo(foot)
  const blendedFoot = foot.clone().lerp(target.position, strength)
  const axis = blendedFoot.clone().sub(hip)
  const requestedReach = axis.length()
  if (requestedReach < EPSILON) axis.set(0, -1, 0)
  else axis.multiplyScalar(1 / requestedReach)
  let reachableReach = requestedReach
  const reachableFoot = hip.clone()
  const bend = forward.clone().addScaledVector(axis, -forward.dot(axis))
  if (bend.lengthSq() < EPSILON * EPSILON) {
    bend.copy(knee).sub(hip).addScaledVector(axis, -knee.clone().sub(hip).dot(axis))
  }
  if (bend.lengthSq() < EPSILON * EPSILON) bend.crossVectors(axis, new THREE.Vector3(1, 0, 0))
  if (bend.lengthSq() < EPSILON * EPSILON) bend.crossVectors(axis, new THREE.Vector3(0, 0, 1))
  bend.normalize()
  const worldScale = leg.upper.getWorldScale(new THREE.Vector3())
  const scaleRatio = Math.max(Math.abs(worldScale.x), Math.abs(worldScale.y), Math.abs(worldScale.z))
    / Math.max(EPSILON, Math.min(Math.abs(worldScale.x), Math.abs(worldScale.y), Math.abs(worldScale.z)))
  // A nonuniform ancestor changes world segment lengths as joints rotate. Recompute those
  // lengths to converge while keeping every local bone offset unchanged.
  const iterations = scaleRatio > 1.001 ? 8 : 1
  for (let iteration = 0; iteration < iterations; iteration++) {
    const currentKnee = leg.knee.getWorldPosition(new THREE.Vector3())
    const currentFoot = leg.foot.getWorldPosition(new THREE.Vector3())
    const upper = hip.distanceTo(currentKnee)
    const lower = currentKnee.distanceTo(currentFoot)
    const maximum = Math.max(EPSILON, upper + lower - EPSILON)
    const minimum = Math.min(maximum, Math.abs(upper - lower) + EPSILON)
    reachableReach = THREE.MathUtils.clamp(requestedReach, minimum, maximum)
    reachableFoot.copy(hip).addScaledVector(axis, reachableReach)
    const along = (upper * upper - lower * lower + reachableReach * reachableReach) / (2 * reachableReach)
    const perpendicular = Math.sqrt(Math.max(0, upper * upper - along * along))
    const desiredKnee = hip.clone().addScaledVector(axis, along).addScaledVector(bend, perpendicular)
    aimSegment(leg.upper, leg.knee, desiredKnee)
    aimSegment(leg.knee, leg.foot, reachableFoot)
    if (leg.foot.getWorldPosition(new THREE.Vector3()).distanceTo(reachableFoot) < 0.00001) break
  }
  const footQuaternion = leg.foot.getWorldQuaternion(new THREE.Quaternion()).slerp(target.quaternion, strength)
  setWorldQuaternion(leg.foot, footQuaternion)
  const actualFoot = leg.foot.getWorldPosition(new THREE.Vector3())
  return {
    upperLength, lowerLength, requestedReach, reachableReach,
    reachClamped: Math.abs(requestedReach - reachableReach) > 0.0001,
    footTargetError: actualFoot.distanceTo(target.position),
    reachableTargetError: actualFoot.distanceTo(reachableFoot),
    footFloorError: Math.abs(actualFoot.y - floorY - target.ankleClearance),
    toeHeight: leg.toe ? leg.toe.getWorldPosition(new THREE.Vector3()).y - floorY : null,
    kneePosition: leg.knee.getWorldPosition(new THREE.Vector3()).toArray() as [number, number, number],
    footPosition: actualFoot.toArray() as [number, number, number],
  }
}

/** Apply after animation/retargeting each frame; strength blends the transition into fixed contacts. */
export const applySeatContactTargets = (
  bones: BoneMap, targets: RoomSeatContactTargets, strength = 1,
): RoomPoseContactMetrics => {
  const hips = bones.get('hips')
  const left = getLeg(bones, 'left')
  const right = getLeg(bones, 'right')
  const amount = THREE.MathUtils.clamp(strength, 0, 1)
  if (!hips || !left || !right || amount === 0) return {
    applied: false, hipSeatError: 0, hipSeatHeight: 0,
    maxFootTargetError: 0, maxFootFloorError: 0, maxReachError: 0,
    legs: { left: null, right: null },
  }
  hips.updateWorldMatrix(true, true)
  setWorldPosition(hips, hips.getWorldPosition(new THREE.Vector3()).lerp(targets.pelvis, amount))
  const leftMetrics = solveLeg(left, targets.leftFoot, targets.forward, targets.floorY, amount)
  const rightMetrics = solveLeg(right, targets.rightFoot, targets.forward, targets.floorY, amount)
  const hip = hips.getWorldPosition(new THREE.Vector3())
  return {
    applied: true,
    hipSeatError: hip.distanceTo(targets.pelvis),
    hipSeatHeight: hip.y - targets.seatCenter.y,
    maxFootTargetError: Math.max(leftMetrics.footTargetError, rightMetrics.footTargetError),
    maxFootFloorError: Math.max(leftMetrics.footFloorError, rightMetrics.footFloorError),
    maxReachError: Math.max(leftMetrics.reachableTargetError, rightMetrics.reachableTargetError),
    legs: { left: leftMetrics, right: rightMetrics },
  }
}

/** Ground a standing pose by shifting only the pelvis; use before capturing its foot reference. */
export const groundRoomFeet = (
  bones: BoneMap, standingHeight = 1.8, floorY = 0, ankleClearance?: number,
) => {
  const hips = bones.get('hips')
  const left = getLeg(bones, 'left')
  const right = getLeg(bones, 'right')
  if (!hips || !left || !right) return 0
  hips.updateWorldMatrix(true, true)
  const scale = Math.max(0.1, standingHeight / 1.8)
  const clearance = ankleClearance ?? 0.075 * scale
  const lowest = Math.min(left.foot.getWorldPosition(new THREE.Vector3()).y, right.foot.getWorldPosition(new THREE.Vector3()).y)
  const shift = floorY + clearance - lowest
  const position = hips.getWorldPosition(new THREE.Vector3())
  position.y += shift
  setWorldPosition(hips, position)
  return shift
}
