import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'

export type SceneChair = {
  root: THREE.Group
  seat: THREE.Mesh
  seatHeight: number
  size: number
}

/** A reusable, floor-standing chair. Its transform is independent of the avatar. */
export const createSceneChair = (): SceneChair => {
  const root = new THREE.Group()
  root.name = 'SceneChair'
  const wood = new THREE.MeshStandardMaterial({ color: 0x9b754e, roughness: 0.58 })
  const fabric = new THREE.MeshStandardMaterial({ color: 0xb5aa97, roughness: 0.95 })
  const metal = new THREE.MeshStandardMaterial({ color: 0x45433e, roughness: 0.5, metalness: 0.45 })
  const box = new RoundedBoxGeometry(1, 1, 1, 2, 0.07)

  const add = (name: string, material: THREE.Material) => {
    const mesh = new THREE.Mesh(box, material)
    mesh.name = name
    mesh.castShadow = true
    mesh.receiveShadow = true
    root.add(mesh)
    return mesh
  }
  const seat = add('ChairSeat', fabric)
  add('ChairSeatFrame', wood)
  add('ChairBack', wood)
  add('ChairBackCushion', fabric)
  for (const side of [-1, 1]) {
    add(`ChairBackPost${side}`, metal)
    for (const end of [-1, 1]) add(`ChairLeg${side},${end}`, metal)
  }

  const shadow = new THREE.Mesh(
    new THREE.CircleGeometry(0.38, 32),
    new THREE.MeshBasicMaterial({ color: 0x020617, transparent: true, opacity: 0.25, depthWrite: false }),
  )
  shadow.name = 'ChairFloorShadow'
  shadow.rotation.x = -Math.PI / 2
  shadow.position.y = 0.002
  root.add(shadow)

  const chair = { root, seat, seatHeight: 0.46, size: 1 }
  placeSceneChair(chair, new THREE.Vector3(0, 0, -0.32), 0.46, 1, 0)
  return chair
}

export const placeSceneChair = (
  chair: SceneChair,
  center: THREE.Vector3,
  seatHeight: number,
  size: number,
  yaw: number,
) => {
  const scale = THREE.MathUtils.clamp(size, 0.2, 10)
  const top = Math.max(seatHeight, 0.16 * scale)
  chair.size = scale
  chair.seatHeight = top
  chair.root.position.set(center.x, 0, center.z)
  chair.root.rotation.set(0, yaw, 0)
  const part = (name: string, dimensions: number[], position: number[]) => {
    const mesh = chair.root.getObjectByName(name) as THREE.Mesh
    mesh.scale.set(...dimensions as [number, number, number])
    mesh.position.set(...position as [number, number, number])
  }
  const thickness = 0.065 * scale
  part('ChairSeat', [0.52 * scale, thickness, 0.48 * scale], [0, top - thickness / 2, 0])
  part('ChairSeatFrame', [0.53 * scale, 0.025 * scale, 0.49 * scale], [0, top - thickness - 0.012 * scale, 0])
  part('ChairBack', [0.5 * scale, 0.38 * scale, 0.045 * scale], [0, top + 0.255 * scale, -0.225 * scale])
  part('ChairBackCushion', [0.44 * scale, 0.29 * scale, 0.035 * scale], [0, top + 0.255 * scale, -0.185 * scale])
  const legHeight = top - thickness - 0.025 * scale
  for (const side of [-1, 1]) {
    part(`ChairBackPost${side}`, [0.026 * scale, 0.49 * scale, 0.026 * scale], [side * 0.21 * scale, top + 0.16 * scale, -0.22 * scale])
    for (const end of [-1, 1]) {
      part(`ChairLeg${side},${end}`, [0.028 * scale, legHeight, 0.028 * scale], [side * 0.21 * scale, legHeight / 2, end * 0.19 * scale])
    }
  }
  chair.root.getObjectByName('ChairFloorShadow')!.scale.setScalar(scale)
  chair.root.updateMatrixWorld(true)
}

/** Sample the displayed rig once; the chair stays fixed during and after playback. */
export const fitSceneChairToSeatedPose = (
  chair: SceneChair,
  bones: Map<string, THREE.Bone>,
  standingHeight: number,
) => {
  const leftHip = bones.get('leftUpLeg')
  const rightHip = bones.get('rightUpLeg')
  const leftKnee = bones.get('leftLowerLeg')
  const rightKnee = bones.get('rightLowerLeg')
  if (!leftHip || !rightHip || !leftKnee || !rightKnee) return false
  const hip = leftHip.getWorldPosition(new THREE.Vector3())
    .add(rightHip.getWorldPosition(new THREE.Vector3())).multiplyScalar(0.5)
  const knees = leftKnee.getWorldPosition(new THREE.Vector3())
    .add(rightKnee.getWorldPosition(new THREE.Vector3())).multiplyScalar(0.5)
  const forward = knees.sub(hip)
  forward.y = 0
  if (forward.lengthSq() < 1e-5) return false
  forward.normalize()
  const size = standingHeight / 1.8
  // The thigh joint is inside the body: support the underside, not the joint itself.
  const center = hip.clone().addScaledVector(forward, 0.065 * size)
  placeSceneChair(chair, center, hip.y - 0.055 * size, size, Math.atan2(forward.x, forward.z))
  return true
}

export const disposeSceneChair = (chair: SceneChair) => {
  const geometries = new Set<THREE.BufferGeometry>()
  const materials = new Set<THREE.Material>()
  chair.root.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return
    geometries.add(object.geometry)
    for (const material of Array.isArray(object.material) ? object.material : [object.material]) materials.add(material)
  })
  chair.root.removeFromParent()
  for (const geometry of geometries) geometry.dispose()
  for (const material of materials) material.dispose()
}
