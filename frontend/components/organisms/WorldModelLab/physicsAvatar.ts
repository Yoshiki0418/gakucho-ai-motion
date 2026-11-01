'use client'

import * as THREE from 'three'
import { disposeCharacter, loadCharacterModel, type LoadedCharacter } from '../BvhViewer/characterRig'

export type PhysicalBody = { name: string; position: [number, number, number]; quaternion: [number, number, number, number] }
export type AvatarProfile = {
  display: {
    modelUrl: string
    modelSha256: string
    scale: number
    yawDegrees: number
    rootPosition: [number, number, number]
    joints: Record<string, { localQuaternion: number[]; localPosition: number[] }>
  }
  physical: Record<string, unknown>
}

export const physicsPosition = ([x, y, z]: number[]) => new THREE.Vector3(x, z, -y)
export const physicsQuaternion = ([w, x, y, z]: number[]) => new THREE.Quaternion(x, z, -y, w).normalize()

const LEGACY_BODY_BY_BONE: Record<string, string> = {
  hips: 'pelvis', spine: 'torso', spine1: 'torso', spine2: 'torso', spine3: 'torso', spine4: 'torso',
  neck: 'torso', neck1: 'torso', head: 'head',
  leftShoulder: 'torso', rightShoulder: 'torso',
  leftUpLeg: 'left_thigh', rightUpLeg: 'right_thigh',
  leftLowerLeg: 'left_shank', rightLowerLeg: 'right_shank',
  leftFoot: 'left_foot', rightFoot: 'right_foot',
  leftToe: 'left_foot', rightToe: 'right_foot',
  leftUpperArm: 'left_upper_arm', rightUpperArm: 'right_upper_arm',
  leftLowerArm: 'left_forearm', rightLowerArm: 'right_forearm',
  leftHand: 'left_forearm', rightHand: 'right_forearm',
}

type Link = { canonical: string; bone: THREE.Bone; body: string; restBodyInverse: THREE.Quaternion; restAvatarWorld: THREE.Quaternion }
export type PhysicsAvatar = {
  root: THREE.Group
  character: LoadedCharacter
  update: (from: PhysicalBody[], to: PhysicalBody[], blend: number) => void
  diagnostics: () => { joints: Record<string, { avatar: number[]; physics: number[]; error: number }>; maximumJointError: number }
  dispose: () => void
}

/**
 * The measured rest pose is the calibration; no floor correction is applied during playback.
 * A record may supply its own bone-to-body map (the full-body tracking humanoid, whose
 * bodies already follow the GLB joints at the T-pose); the limb straightening below is
 * then skipped because it only adapts the older 20-joint control body.
 */
export async function loadPhysicsAvatar(profile: AvatarProfile, restBodies: PhysicalBody[],
  bodyMap?: Record<string, string>): Promise<PhysicsAvatar> {
  const BODY_BY_BONE = bodyMap ?? LEGACY_BODY_BY_BONE
  const display = profile.display
  if (!/^\/avatars\/[a-zA-Z0-9_.-]+\.glb$/.test(display.modelUrl) || !/^[a-f0-9]{64}$/i.test(display.modelSha256)) {
    throw new Error('学長アバターの実測情報が正しくありません')
  }
  const response = await fetch(display.modelUrl, { cache: 'force-cache' })
  if (!response.ok) throw new Error('学長アバターを読み込めません')
  const bytes = await response.arrayBuffer()
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), (value) => value.toString(16).padStart(2, '0')).join('')
  if (hash !== display.modelSha256.toLowerCase()) throw new Error('学長アバターと身体の実測情報が一致しません。記録を更新してください。')
  const blobUrl = URL.createObjectURL(new Blob([bytes], { type: 'model/gltf-binary' }))
  let character: LoadedCharacter
  try { character = await loadCharacterModel(blobUrl) } finally { URL.revokeObjectURL(blobUrl) }
  const { root, model, bonesByCanonical } = character
  root.name = 'MeasuredGakuchoAvatar'
  root.scale.setScalar(display.scale)
  root.quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), THREE.MathUtils.degToRad(display.yawDegrees))
  root.position.fromArray(display.rootPosition)
  for (const [canonical, rest] of Object.entries(display.joints)) {
    const bone = bonesByCanonical.get(canonical)
    if (!bone) continue
    bone.quaternion.fromArray(rest.localQuaternion).normalize()
    bone.position.fromArray(rest.localPosition)
  }
  root.updateMatrixWorld(true)
  const rest = new Map(restBodies.map((body) => [body.name, body]))
  const hips = bonesByCanonical.get('hips')
  const pelvis = rest.get('pelvis')
  if (!hips || !pelvis) { disposeCharacter(character); throw new Error('学長の腰と物理モデルを対応付けられません') }
  root.position.add(physicsPosition(pelvis.position).sub(hips.getWorldPosition(new THREE.Vector3())))
  root.updateMatrixWorld(true)

  // Straight control limbs retain the GLB's measured lengths. Feet retain the
  // measured shoe orientation, rather than inheriting the leg-rest correction.
  const footWorld = new Map(['leftFoot', 'rightFoot'].map((canonical) => [canonical,
    bonesByCanonical.get(canonical)?.getWorldQuaternion(new THREE.Quaternion())]))
  const align = (canonical: string, childCanonical: string, physicalEnd?: THREE.Vector3) => {
    const bone = bonesByCanonical.get(canonical), child = bonesByCanonical.get(childCanonical)
    const source = rest.get(BODY_BY_BONE[canonical]), target = rest.get(BODY_BY_BONE[childCanonical])
    if (!bone || !child || !source || (!target && !physicalEnd)) return
    root.updateMatrixWorld(true)
    const from = child.getWorldPosition(new THREE.Vector3()).sub(bone.getWorldPosition(new THREE.Vector3())).normalize()
    const to = (physicalEnd ?? physicsPosition(target!.position)).clone().sub(physicsPosition(source.position)).normalize()
    const swing = new THREE.Quaternion().setFromUnitVectors(from, to)
    const desired = swing.multiply(bone.getWorldQuaternion(new THREE.Quaternion()))
    const parent = bone.parent?.getWorldQuaternion(new THREE.Quaternion()) ?? new THREE.Quaternion()
    bone.quaternion.copy(parent.invert().multiply(desired)).normalize()
  }
  for (const side of bodyMap ? [] : ['left', 'right']) {
    align(`${side}UpLeg`, `${side}LowerLeg`)
    align(`${side}LowerLeg`, `${side}Foot`)
    align(`${side}UpperArm`, `${side}LowerArm`)
    const lower = rest.get(`${side}_forearm`)
    const physicalArms = profile.physical.arms as Record<string, { forearmLength: number }> | undefined
    if (lower && physicalArms?.[side]) align(`${side}LowerArm`, `${side}Hand`,
      physicsPosition(lower.position).add(new THREE.Vector3(0, -physicalArms[side].forearmLength, 0)))
    const foot = bonesByCanonical.get(`${side}Foot`), footRest = footWorld.get(`${side}Foot`)
    if (foot && footRest) {
      root.updateMatrixWorld(true)
      const parent = foot.parent?.getWorldQuaternion(new THREE.Quaternion()) ?? new THREE.Quaternion()
      foot.quaternion.copy(parent.invert().multiply(footRest)).normalize()
    }
  }
  root.updateMatrixWorld(true)
  const links: Link[] = []
  const canonicalByBone = new Map([...bonesByCanonical].map(([canonical, bone]) => [bone, canonical]))
  model.traverse((object) => {
    const bone = object as THREE.Bone
    const canonical = canonicalByBone.get(bone)
    const bodyName = canonical ? BODY_BY_BONE[canonical] : undefined
    const body = bodyName ? rest.get(bodyName) : undefined
    if (!canonical || !bodyName || !body) return
    links.push({ canonical, bone, body: bodyName, restBodyInverse: physicsQuaternion(body.quaternion).invert(),
      restAvatarWorld: bone.getWorldQuaternion(new THREE.Quaternion()) })
  })
  let currentBodies = rest
  const update = (from: PhysicalBody[], to: PhysicalBody[], blend: number) => {
    const targets = new Map(to.map((body) => [body.name, body]))
    const interpolated = new Map(from.map((body) => {
      const next = targets.get(body.name) ?? body
      const position = physicsPosition(body.position).lerp(physicsPosition(next.position), blend)
      const quaternion = physicsQuaternion(body.quaternion).slerp(physicsQuaternion(next.quaternion), blend)
      return [body.name, { position, quaternion }] as const
    }))
    const rootBody = interpolated.get('pelvis')
    if (!rootBody) return
    root.updateMatrixWorld(true)
    // Absolute pelvis position is transformed through the real GLB parent,
    // including its centimetre unit scale. There is no frame-by-frame snapping.
    hips.position.copy(rootBody.position)
    hips.parent?.worldToLocal(hips.position)
    hips.updateMatrixWorld(true)
    for (const link of links) {
      const body = interpolated.get(link.body)
      if (!body) continue
      const parent = link.bone.parent?.getWorldQuaternion(new THREE.Quaternion()) ?? new THREE.Quaternion()
      link.bone.quaternion.copy(parent.invert().multiply(body.quaternion.clone())
        .multiply(link.restBodyInverse).multiply(link.restAvatarWorld)).normalize()
      link.bone.updateMatrixWorld(true)
    }
    currentBodies = new Map(from.map((body) => {
      const next = targets.get(body.name) ?? body
      return [body.name, { ...body, position: body.position.map((value, axis) => value + (next.position[axis] - value) * blend) as PhysicalBody['position'] }]
    }))
    root.updateMatrixWorld(true)
  }
  const diagnostics = () => {
    const joints: Record<string, { avatar: number[]; physics: number[]; error: number }> = {}
    for (const canonical of ['hips', 'head', 'leftUpLeg', 'rightUpLeg', 'leftLowerLeg', 'rightLowerLeg', 'leftFoot', 'rightFoot', 'leftUpperArm', 'rightUpperArm', 'leftLowerArm', 'rightLowerArm']) {
      const bone = bonesByCanonical.get(canonical), body = currentBodies.get(BODY_BY_BONE[canonical])
      if (!bone || !body) continue
      const avatar = bone.getWorldPosition(new THREE.Vector3()), physics = physicsPosition(body.position)
      joints[canonical] = { avatar: avatar.toArray(), physics: physics.toArray(), error: avatar.distanceTo(physics) }
    }
    return { joints, maximumJointError: Math.max(0, ...Object.values(joints).map((joint) => joint.error)) }
  }
  root.userData.physicsAvatar = { modelSha256: hash, profile, diagnostics }
  const dispose = () => {
    const textures = new Set<THREE.Texture>()
    const skeletons = new Set<THREE.Skeleton>()
    root.traverse((object) => {
      const mesh = object as THREE.Mesh
      if (!mesh.isMesh) return
      const skinned = mesh as THREE.SkinnedMesh
      if (skinned.isSkinnedMesh) skeletons.add(skinned.skeleton)
      for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
        for (const value of Object.values(material)) {
          const texture = value as THREE.Texture | null
          if (texture?.isTexture) textures.add(texture)
        }
      }
    })
    disposeCharacter(character)
    for (const texture of textures) texture.dispose()
    for (const skeleton of skeletons) skeleton.dispose()
  }
  return { root, character, update, diagnostics, dispose }
}
