import * as THREE from 'three'
import {
  isRoomSegmentWalkable,
  planChairApproach,
  type RoomChairTarget,
  type RoomNavigation,
  type RoomPoint,
} from './roomNavigation'
import type { RoomMotionPrimitives } from './roomMotionPrimitives'
import {
  applySeatContactTargets,
  captureStandingFootContacts,
  createSeatContactTargets,
  groundRoomFeet,
  type RoomFootContactReference,
  type RoomPoseContactMetrics,
  type RoomSeatContactTargets,
} from './roomPoseContacts'

export type RoomTaskPhase = 'walking' | 'aligning' | 'sitting' | 'seated' | 'standing' | 'settling' | 'idle' | 'failed' | 'cancelled'
export type RoomTaskSnapshot = {
  kind?: 'seat' | 'walk'
  phase: RoomTaskPhase
  chairId: string
  chairLabel: string
  position: RoomPoint
  heading: number
  elapsed: number
  path: RoomPoint[]
  pathLength: number
  distanceWalked: number
  collisions: number
  playbackRate: number
  contacts: RoomPoseContactMetrics | null
  reason?: string
}
export type RoomTaskBridge = {
  mixer: THREE.AnimationMixer
  sourceBones: Map<string, THREE.Bone>
  getDisplayBones: () => Map<string, THREE.Bone>
  getCharacterBones: () => Map<string, THREE.Bone> | undefined
  place: (position: RoomPoint, placementYaw: number) => void
  retarget: () => void
  hold: (posture: 'standing' | 'chair_sitting') => void
  restoreStanding: () => void
  notify: (snapshot: RoomTaskSnapshot) => void
}
type BonePose = { bone: THREE.Bone; position: THREE.Vector3; quaternion: THREE.Quaternion }
const smooth = (value: number) => {
  const t = THREE.MathUtils.clamp(value, 0, 1)
  return t * t * (3 - 2 * t)
}
const angleDelta = (from: number, to: number) => Math.atan2(Math.sin(to - from), Math.cos(to - from))
const pointDistance = (a: RoomPoint, b: RoomPoint) => Math.hypot(b.x - a.x, b.z - a.z)

/** Scene-aware motion adaptation. This prototype is kinematic, not an RL policy. */
export class RoomTaskController {
  readonly chair: RoomChairTarget
  readonly route: THREE.Group
  readonly path: RoomPoint[]
  readonly pathLength: number
  phase: RoomTaskPhase = 'walking'
  position: RoomPoint
  heading: number
  elapsed = 0
  distanceWalked = 0
  collisions = 0
  private waypoint = 1
  private phaseTime = 0
  private notificationTime = 0
  private action: THREE.AnimationAction
  private sourceFootReference: RoomFootContactReference | null
  private characterFootReference: RoomFootContactReference | null
  private sourceContacts: RoomSeatContactTargets | null = null
  private characterContacts: RoomSeatContactTargets | null = null
  private metrics: RoomPoseContactMetrics | null = null
  private readonly approach: RoomPoint
  private readonly standingPose: BonePose[]
  private blendFrom: BonePose[] = []
  private poseBeforeContacts: BonePose[] = []
  private contactCharacterBones: Map<string, THREE.Bone> | undefined
  private pendingHold: 'standing' | 'chair_sitting' | null = null
  private disposed = false
  private reason?: string

  constructor(
    private readonly primitives: RoomMotionPrimitives,
    private readonly room: RoomNavigation,
    private readonly bridge: RoomTaskBridge,
    start: RoomPoint,
    chairId: string,
    startHeading = 0,
  ) {
    const plan = planChairApproach(room, start, chairId)
    if (!plan.ok) throw new Error(`椅子への経路を作れません (${plan.reason})`)
    this.chair = plan.chair
    this.path = plan.waypoints.map((point) => ({ ...point }))
    this.pathLength = plan.length
    this.approach = { ...plan.approach }
    this.position = { ...start }
    this.heading = startHeading
    bridge.place(this.position, startHeading)
    groundRoomFeet(bridge.sourceBones)
    bridge.retarget()
    this.standingPose = this.capturePose()
    this.sourceFootReference = captureStandingFootContacts(bridge.sourceBones)
    const characterBones = bridge.getCharacterBones()
    this.contactCharacterBones = characterBones
    this.characterFootReference = characterBones ? captureStandingFootContacts(characterBones) : null
    bridge.mixer.timeScale = 1
    this.action = bridge.mixer.clipAction(primitives.walk.clip)
    this.action.reset().setLoop(THREE.LoopRepeat, Infinity)
    this.action.setEffectiveWeight(1).setEffectiveTimeScale(1).play()
    this.route = this.createRoute()
    this.blendFrom = this.capturePose()
    this.poseBeforeContacts = this.capturePose()
    this.emit()
  }

  get busy() {
    return ['walking', 'aligning', 'sitting', 'standing', 'settling'].includes(this.phase)
  }

  get ownsPose() {
    return this.busy || this.phase === 'seated'
  }

  snapshot(): RoomTaskSnapshot {
    return {
      kind: 'seat', phase: this.phase, chairId: this.chair.id, chairLabel: this.chair.label,
      position: { ...this.position }, heading: this.heading, elapsed: this.elapsed,
      path: this.path.map((point) => ({ ...point })), pathLength: this.pathLength,
      distanceWalked: this.distanceWalked, collisions: this.collisions,
      playbackRate: this.bridge.mixer.timeScale, contacts: this.metrics, reason: this.reason,
    }
  }

  /** Bounded simulation time prevents stalls or a resumed tab from skipping the task. */
  update(delta: number) {
    if (this.disposed || !this.ownsPose) return
    const dt = Number.isFinite(delta) ? THREE.MathUtils.clamp(delta, 0, 0.08) : 0
    if (dt === 0 || this.bridge.mixer.timeScale === 0) {
      this.applyPoseContacts()
      return
    }
    // Navigation and clips share real simulation seconds; there is no catch-up playback.
    this.bridge.mixer.timeScale = 1
    this.elapsed += dt
    this.phaseTime += dt
    this.notificationTime += dt
    if (this.phase === 'walking') this.walk(dt)
    else if (this.phase === 'aligning') this.align(dt)
    else if (this.phase === 'sitting' || this.phase === 'standing') this.transition(dt)
    else if (this.phase === 'settling') this.settle()
    else this.bridge.mixer.update(0)

    this.poseBeforeContacts = this.capturePose()
    this.applyPoseContacts()
    if (this.pendingHold) {
      const posture = this.pendingHold
      this.pendingHold = null
      if (posture === 'standing') {
        this.bridge.restoreStanding()
        this.phase = 'idle'
        this.phaseTime = 0
      }
      this.bridge.hold(posture)
      this.emit()
    }
    if (this.notificationTime >= 0.2) this.emit()
  }

  /** Repeat calls are idempotent: partial contacts always start from the same animation pose. */
  applyPoseContacts() {
    if (this.disposed || !this.ownsPose) return
    this.restorePose(this.poseBeforeContacts)
    const seatedBlend = this.phase === 'seated' ? 1
      : this.phase === 'sitting' ? smooth(this.phaseTime / this.primitives.sit.clip.duration)
        : this.phase === 'standing' ? 1 - smooth(this.phaseTime / this.primitives.stand.clip.duration) : 0
    let sourceMetrics: RoomPoseContactMetrics | null = null
    if (seatedBlend > 0 && this.sourceContacts) {
      sourceMetrics = applySeatContactTargets(this.bridge.sourceBones, this.sourceContacts, seatedBlend)
    } else {
      groundRoomFeet(this.bridge.sourceBones)
    }
    this.bridge.retarget()
    const characterBones = this.bridge.getCharacterBones()
    if (characterBones !== this.contactCharacterBones) {
      // A seated avatar is not a standing calibration pose. The viewer supplies its new reference.
      this.contactCharacterBones = characterBones
      this.characterFootReference = null
      this.characterContacts = null
    }
    this.metrics = sourceMetrics
    if (characterBones && this.characterContacts && seatedBlend > 0) {
      this.metrics = applySeatContactTargets(characterBones, this.characterContacts, seatedBlend)
    } else if (seatedBlend === 0) {
      if (characterBones) {
        const clearance = this.characterFootReference
          ? Math.min(this.characterFootReference.left.ankleClearance, this.characterFootReference.right.ankleClearance) : undefined
        groundRoomFeet(characterBones, 1.8, 0, clearance)
      }
    }
  }

  /** The caller captures this reference on the new avatar in its grounded standing pose. */
  refreshCharacterContacts(footReference: RoomFootContactReference | null) {
    this.contactCharacterBones = this.bridge.getCharacterBones()
    this.characterFootReference = footReference
    this.characterContacts = this.contactCharacterBones && this.sourceContacts
      ? createSeatContactTargets(this.contactCharacterBones, this.chair, 1.8, { footReference }) : null
    this.applyPoseContacts()
  }

  stand() {
    if (this.disposed || this.phase !== 'seated') return false
    this.beginClip('standing')
    this.poseBeforeContacts = this.capturePose()
    this.applyPoseContacts()
    return true
  }

  cancel() {
    if (this.disposed || ['idle', 'failed', 'cancelled'].includes(this.phase)) return
    if (this.phase === 'seated') {
      this.bridge.hold('chair_sitting')
    } else {
      if (['sitting', 'standing', 'settling'].includes(this.phase)) this.position = { ...this.approach }
      this.action.stop()
      this.bridge.restoreStanding()
      this.restorePose(this.standingPose)
      this.bridge.place(this.position, this.heading)
      this.bridge.retarget()
    }
    this.pendingHold = null
    this.phase = 'cancelled'
    this.emit()
  }

  dispose() {
    if (this.disposed) return
    this.disposed = true
    this.route.removeFromParent()
    this.route.traverse((object) => {
      if (object instanceof THREE.Line || object instanceof THREE.Mesh) {
        object.geometry.dispose()
        for (const material of Array.isArray(object.material) ? object.material : [object.material]) material.dispose()
      }
    })
  }

  private capturePose(): BonePose[] {
    return [...new Set(this.bridge.sourceBones.values())].map((bone) => ({
      bone, position: bone.position.clone(), quaternion: bone.quaternion.clone(),
    }))
  }

  private restorePose(poses: BonePose[]) {
    for (const pose of poses) {
      pose.bone.position.copy(pose.position)
      pose.bone.quaternion.copy(pose.quaternion)
    }
  }

  private blendPose(from: BonePose[], blend: number) {
    for (const pose of from) {
      pose.bone.position.lerpVectors(pose.position, pose.bone.position, blend)
      pose.bone.quaternion.slerpQuaternions(pose.quaternion, pose.bone.quaternion, blend)
    }
  }

  private walk(dt: number) {
    const goal = this.path[this.waypoint]
    if (!goal) {
      this.phase = 'aligning'
      this.phaseTime = 0
      this.blendFrom = this.capturePose()
      this.action.paused = true
      this.emit()
      return
    }
    const desired = Math.atan2(goal.x - this.position.x, goal.z - this.position.z)
    const difference = angleDelta(this.heading, desired)
    this.heading += THREE.MathUtils.clamp(difference, -2.2 * dt, 2.2 * dt)
    const speed = this.primitives.walk.metadata.walkLoop?.speedMetersPerSecond ?? 0.58
    const amount = Math.abs(difference) > 0.62 ? 0 : Math.min(speed * dt, pointDistance(this.position, goal))
    const remaining = pointDistance(this.position, goal)
    const next = remaining > 1e-7 ? {
      x: this.position.x + (goal.x - this.position.x) * amount / remaining,
      z: this.position.z + (goal.z - this.position.z) * amount / remaining,
    } : { ...goal }
    if (!isRoomSegmentWalkable(this.room, this.position, next)) {
      this.collisions++
      this.fail('家具への接近を検出したため停止しました')
      return
    }
    this.distanceWalked += amount
    this.position = next
    this.bridge.place(this.position, this.heading - this.primitives.walk.metadata.nativeHeadingRadians)
    this.bridge.mixer.update(dt)
    if (this.phaseTime < 0.3) this.blendPose(this.blendFrom, smooth(this.phaseTime / 0.3))
    // Reach the actual visibility-graph corner before changing segments.
    if (remaining <= amount + 1e-7) {
      this.position = { ...goal }
      this.waypoint++
    }
  }

  private align(dt: number) {
    const targetHeading = this.chair.yaw
    const difference = angleDelta(this.heading, targetHeading)
    this.heading += THREE.MathUtils.clamp(difference, -2.2 * dt, 2.2 * dt)
    this.bridge.mixer.update(0)
    for (const pose of this.standingPose) {
      const from = this.blendFrom.find((candidate) => candidate.bone === pose.bone) ?? pose
      pose.bone.position.lerpVectors(from.position, pose.position, smooth(this.phaseTime / 0.6))
      pose.bone.quaternion.slerpQuaternions(from.quaternion, pose.quaternion, smooth(this.phaseTime / 0.6))
    }
    const blend = smooth(this.phaseTime / 0.6)
    this.bridge.place(this.position, this.heading - this.primitives.walk.metadata.nativeHeadingRadians * (1 - blend))
    if (this.phaseTime >= 0.7 && Math.abs(angleDelta(this.heading, targetHeading)) < 0.025) {
      this.heading = targetHeading
      this.bridge.place(this.position, this.heading)
      this.bridge.retarget()
      this.sourceContacts = createSeatContactTargets(this.bridge.sourceBones, this.chair, 1.8,
        { footReference: this.sourceFootReference })
      const character = this.bridge.getCharacterBones()
      this.characterContacts = character ? createSeatContactTargets(character, this.chair, 1.8,
        { footReference: this.characterFootReference }) : null
      if (!this.sourceContacts) this.fail('座るための脚の関節を確認できません')
      else this.beginClip('sitting')
    }
  }

  private beginClip(phase: 'sitting' | 'standing') {
    this.blendFrom = this.capturePose()
    this.action.stop()
    this.phase = phase
    this.phaseTime = 0
    const primitive = phase === 'sitting' ? this.primitives.sit : this.primitives.stand
    this.action = this.bridge.mixer.clipAction(primitive.clip)
    this.action.reset().setLoop(THREE.LoopOnce, 1)
    this.action.clampWhenFinished = true
    this.action.setEffectiveWeight(1).setEffectiveTimeScale(1).play()
    this.bridge.mixer.update(0)
    this.blendPose(this.blendFrom, 0)
    this.emit()
  }

  private transition(dt: number) {
    const sitting = this.phase === 'sitting'
    const primitive = sitting ? this.primitives.sit : this.primitives.stand
    const progress = smooth(this.phaseTime / primitive.clip.duration)
    const pelvis = this.sourceContacts!.pelvis
    const start = sitting ? this.approach : pelvis
    const end = sitting ? pelvis : this.approach
    const next = {
      x: THREE.MathUtils.lerp(start.x, end.x, progress),
      z: THREE.MathUtils.lerp(start.z, end.z, progress),
    }
    // The seat is intentionally adjacent to the table. Keep the body radius, but
    // remove the extra walking margin while approaching its fixed contact target.
    if (!isRoomSegmentWalkable(this.room, this.position, next, {
      ignoreObstacleIds: [this.chair.obstacleId], clearance: 0,
    })) {
      this.collisions++
      this.fail('着座位置の周囲に別の家具を検出したため停止しました')
      return
    }
    this.position = next
    this.bridge.place(this.position, this.heading - primitive.metadata.nativeHeadingRadians)
    this.bridge.mixer.update(dt)
    if (this.phaseTime < 0.3) this.blendPose(this.blendFrom, smooth(this.phaseTime / 0.3))
    if (this.phaseTime + 1e-6 >= primitive.clip.duration) {
      if (sitting) {
        this.phase = 'seated'
        this.pendingHold = 'chair_sitting'
      } else {
        this.blendFrom = this.capturePose()
        this.action.stop()
        this.restorePose(this.blendFrom)
        this.phase = 'settling'
        this.position = { ...this.approach }
      }
      this.phaseTime = 0
    }
  }

  private settle() {
    const blend = smooth(this.phaseTime / 0.6)
    for (const pose of this.standingPose) {
      const from = this.blendFrom.find((candidate) => candidate.bone === pose.bone) ?? pose
      pose.bone.position.lerpVectors(from.position, pose.position, blend)
      pose.bone.quaternion.slerpQuaternions(from.quaternion, pose.quaternion, blend)
    }
    this.bridge.place(this.position, this.heading - this.primitives.stand.metadata.nativeHeadingRadians * (1 - blend))
    if (this.phaseTime + 1e-7 >= 0.6) this.pendingHold = 'standing'
  }

  private fail(reason: string) {
    this.reason = reason
    if (['sitting', 'standing', 'settling'].includes(this.phase)) this.position = { ...this.approach }
    this.action.stop()
    this.bridge.restoreStanding()
    this.restorePose(this.standingPose)
    this.bridge.place(this.position, this.heading)
    this.bridge.retarget()
    this.pendingHold = null
    this.phase = 'failed'
    this.emit()
  }

  private emit() {
    this.notificationTime = 0
    this.bridge.notify(this.snapshot())
  }

  private createRoute() {
    const root = new THREE.Group()
    root.name = 'RoomTaskRoute'
    const points = this.path.map((point) => new THREE.Vector3(point.x, 0.025, point.z))
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineDashedMaterial({ color: 0x38bdf8, dashSize: 0.12, gapSize: 0.06, transparent: true, opacity: 0.75 }))
    line.computeLineDistances()
    root.add(line)
    const marker = new THREE.Mesh(new THREE.RingGeometry(0.12, 0.18, 40),
      new THREE.MeshBasicMaterial({ color: 0x38bdf8, side: THREE.DoubleSide, transparent: true, opacity: 0.65, depthWrite: false }))
    marker.rotation.x = -Math.PI / 2
    marker.position.set(this.chair.position.x, this.chair.seatHeight + 0.012, this.chair.position.z)
    root.add(marker)
    return root
  }
}
