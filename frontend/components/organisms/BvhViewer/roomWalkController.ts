import * as THREE from 'three'
import { isRoomSegmentWalkable, planRoomPath, type RoomNavigation, type RoomPoint } from './roomNavigation'
import type { RoomDestination } from './roomDestinations'
import type { RoomMotionPrimitives } from './roomMotionPrimitives'
import type { RoomTaskBridge, RoomTaskSnapshot } from './roomTaskController'
import { captureStandingFootContacts, groundRoomFeet, type RoomFootContactReference } from './roomPoseContacts'

type WalkPhase = 'walking' | 'aligning' | 'idle' | 'failed' | 'cancelled'
type BonePose = { bone: THREE.Bone; position: THREE.Vector3; quaternion: THREE.Quaternion }
const smooth = (value: number) => {
  const t = THREE.MathUtils.clamp(value, 0, 1)
  return t * t * (3 - 2 * t)
}
const angleDelta = (from: number, to: number) => Math.atan2(Math.sin(to - from), Math.cos(to - from))
const distance = (from: RoomPoint, to: RoomPoint) => Math.hypot(to.x - from.x, to.z - from.z)

/** Follow the room's collision-free path using the T2M walk cycle, then hold standing. */
export class RoomWalkController {
  readonly kind = 'walk' as const
  readonly destination: RoomDestination
  readonly route: THREE.Group
  readonly path: RoomPoint[]
  readonly pathLength: number
  phase: WalkPhase = 'walking'
  position: RoomPoint
  heading: number
  elapsed = 0
  distanceWalked = 0
  collisions = 0
  private waypoint = 1
  private phaseTime = 0
  private notificationTime = 0
  private readonly action: THREE.AnimationAction
  private readonly standingPose: BonePose[]
  private readonly sourceFootReference: RoomFootContactReference | null
  private characterFootReference: RoomFootContactReference | null
  private contactCharacterBones: Map<string, THREE.Bone> | undefined
  private blendFrom: BonePose[]
  private poseBeforeContacts: BonePose[]
  private disposed = false
  private reason?: string

  constructor(
    private readonly primitives: RoomMotionPrimitives,
    private readonly room: RoomNavigation,
    private readonly bridge: RoomTaskBridge,
    start: RoomPoint,
    destination: RoomDestination,
    startHeading = 0,
  ) {
    if (!Number.isFinite(startHeading) || !Number.isFinite(destination.heading)) {
      throw new Error('歩行の向きが正しくありません')
    }
    const plan = planRoomPath(room, start, destination.position)
    if (!plan.ok) throw new Error(`目的地への経路を作れません (${plan.reason})`)
    this.destination = { ...destination, position: { ...destination.position } }
    this.path = plan.waypoints.map((point) => ({ ...point }))
    this.pathLength = plan.length
    this.position = { ...start }
    this.heading = startHeading
    bridge.place(this.position, this.heading)
    groundRoomFeet(bridge.sourceBones)
    bridge.retarget()
    this.standingPose = this.capturePose()
    this.sourceFootReference = captureStandingFootContacts(bridge.sourceBones)
    this.contactCharacterBones = bridge.getCharacterBones()
    this.characterFootReference = this.contactCharacterBones
      ? captureStandingFootContacts(this.contactCharacterBones) : null
    bridge.mixer.timeScale = 1
    this.action = bridge.mixer.clipAction(primitives.walk.clip)
    this.action.reset().setLoop(THREE.LoopRepeat, Infinity)
    this.action.setEffectiveWeight(1).setEffectiveTimeScale(1).play()
    this.blendFrom = this.capturePose()
    this.poseBeforeContacts = this.capturePose()
    if (plan.length < 1e-7) {
      this.phase = 'aligning'
      this.action.paused = true
    }
    this.route = this.createRoute()
    this.emit()
  }

  get busy() {
    return !this.disposed && (this.phase === 'walking' || this.phase === 'aligning')
  }

  get ownsPose() {
    return this.busy
  }

  snapshot(): RoomTaskSnapshot & { kind: 'walk' } {
    return {
      kind: 'walk', phase: this.phase, chairId: this.destination.id, chairLabel: this.destination.label,
      position: { ...this.position }, heading: this.heading, elapsed: this.elapsed,
      path: this.path.map((point) => ({ ...point })), pathLength: this.pathLength,
      distanceWalked: this.distanceWalked, collisions: this.collisions,
      playbackRate: this.bridge.mixer.timeScale, contacts: null, reason: this.reason,
    }
  }

  /** A pause or resumed browser never turns elapsed wall time into accelerated motion. */
  update(delta: number) {
    if (!this.ownsPose) return
    const dt = Number.isFinite(delta) ? THREE.MathUtils.clamp(delta, 0, 0.08) : 0
    if (dt === 0 || this.bridge.mixer.timeScale === 0) {
      this.applyPoseContacts()
      return
    }
    this.bridge.mixer.timeScale = 1
    this.action.setEffectiveTimeScale(1)
    this.elapsed += dt
    this.phaseTime += dt
    this.notificationTime += dt
    let arrived = false
    if (this.phase === 'walking') this.walk(dt)
    else arrived = this.align(dt)
    this.poseBeforeContacts = this.capturePose()
    this.applyPoseContacts()
    if (arrived) this.finishStanding('idle')
    else if (this.notificationTime >= 0.2) this.emit()
  }

  /** Retarget from a saved source pose so repeated grounding never accumulates offsets. */
  applyPoseContacts() {
    if (!this.ownsPose) return
    this.restorePose(this.poseBeforeContacts)
    this.groundPose()
  }

  refreshCharacterContacts(reference: RoomFootContactReference | null) {
    this.contactCharacterBones = this.bridge.getCharacterBones()
    this.characterFootReference = reference
    this.applyPoseContacts()
  }

  stand() {
    return false
  }

  cancel() {
    if (this.ownsPose) this.finishStanding('cancelled')
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

  private groundPose() {
    const sourceClearance = this.sourceFootReference
      ? Math.min(this.sourceFootReference.left.ankleClearance, this.sourceFootReference.right.ankleClearance) : undefined
    groundRoomFeet(this.bridge.sourceBones, 1.8, 0, sourceClearance)
    this.bridge.retarget()
    const character = this.bridge.getCharacterBones()
    if (character !== this.contactCharacterBones) {
      this.contactCharacterBones = character
      // A walking frame cannot recalibrate a replacement avatar's standing soles.
      this.characterFootReference = null
    }
    if (character) {
      const clearance = this.characterFootReference
        ? Math.min(this.characterFootReference.left.ankleClearance, this.characterFootReference.right.ankleClearance) : undefined
      groundRoomFeet(character, 1.8, 0, clearance)
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
    const remaining = distance(this.position, goal)
    const desired = remaining > 1e-7 ? Math.atan2(goal.x - this.position.x, goal.z - this.position.z) : this.heading
    const turn = angleDelta(this.heading, desired)
    this.heading += THREE.MathUtils.clamp(turn, -2.2 * dt, 2.2 * dt)
    const speed = this.primitives.walk.metadata.walkLoop?.speedMetersPerSecond ?? 0.58
    const amount = Math.abs(turn) > 0.62 ? 0 : Math.min(speed * dt, remaining)
    const next = remaining > 1e-7 ? {
      x: this.position.x + (goal.x - this.position.x) * amount / remaining,
      z: this.position.z + (goal.z - this.position.z) * amount / remaining,
    } : { ...goal }
    if (!isRoomSegmentWalkable(this.room, this.position, next)) {
      this.collisions++
      this.reason = '家具への接近を検出したため停止しました'
      this.finishStanding('failed')
      return
    }
    this.distanceWalked += amount
    this.position = next
    if (remaining <= amount + 1e-7) {
      this.position = { ...goal }
      this.waypoint++
    }
    this.bridge.place(this.position, this.heading - this.primitives.walk.metadata.nativeHeadingRadians)
    this.bridge.mixer.update(dt)
    if (this.phaseTime < 0.3) {
      const blend = smooth(this.phaseTime / 0.3)
      for (const pose of this.blendFrom) {
        pose.bone.position.lerpVectors(pose.position, pose.bone.position, blend)
        pose.bone.quaternion.slerpQuaternions(pose.quaternion, pose.bone.quaternion, blend)
      }
    }
  }

  private align(dt: number) {
    const turn = angleDelta(this.heading, this.destination.heading)
    this.heading += THREE.MathUtils.clamp(turn, -2.2 * dt, 2.2 * dt)
    this.bridge.mixer.update(0)
    const blend = smooth(this.phaseTime / 0.6)
    for (const pose of this.standingPose) {
      const from = this.blendFrom.find((candidate) => candidate.bone === pose.bone) ?? pose
      pose.bone.position.lerpVectors(from.position, pose.position, blend)
      pose.bone.quaternion.slerpQuaternions(from.quaternion, pose.quaternion, blend)
    }
    this.bridge.place(this.position, this.heading - this.primitives.walk.metadata.nativeHeadingRadians * (1 - blend))
    if (this.phaseTime + 1e-7 < 0.6 || Math.abs(angleDelta(this.heading, this.destination.heading)) > 1e-7) return false
    this.heading = this.destination.heading
    return true
  }

  private finishStanding(phase: 'idle' | 'failed' | 'cancelled') {
    this.action.stop()
    this.bridge.restoreStanding()
    this.restorePose(this.standingPose)
    this.bridge.place(this.position, this.heading)
    this.groundPose()
    this.poseBeforeContacts = this.capturePose()
    this.phase = phase
    this.phaseTime = 0
    this.bridge.hold('standing')
    this.emit()
  }

  private emit() {
    this.notificationTime = 0
    this.bridge.notify(this.snapshot())
  }

  private createRoute() {
    const root = new THREE.Group()
    root.name = 'RoomWalkRoute'
    const points = this.path.map((point) => new THREE.Vector3(point.x, 0.025, point.z))
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineDashedMaterial({ color: 0x38bdf8, dashSize: 0.12, gapSize: 0.06, transparent: true, opacity: 0.75 }))
    line.computeLineDistances()
    root.add(line)
    const marker = new THREE.Mesh(new THREE.RingGeometry(0.12, 0.18, 40),
      new THREE.MeshBasicMaterial({ color: 0x38bdf8, side: THREE.DoubleSide, transparent: true, opacity: 0.65, depthWrite: false }))
    marker.rotation.x = -Math.PI / 2
    marker.position.set(this.destination.position.x, 0.022, this.destination.position.z)
    root.add(marker)
    return root
  }
}
