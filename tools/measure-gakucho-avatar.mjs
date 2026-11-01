#!/usr/bin/env node
/** Measure the actual rendered suit avatar, using the application's rig loader.
 * No guessed skeleton dimensions or undecoded glTF accessor bounds are used.
 * Run: node tools/measure-gakucho-avatar.mjs
 * PLAYWRIGHT_MODULE may point at an existing Playwright installation.
 */
import fs from 'node:fs/promises'
import path from 'node:path'
import http from 'node:http'
import crypto from 'node:crypto'
import { fileURLToPath } from 'node:url'
import { createRequire } from 'node:module'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const require = createRequire(path.join(root, 'frontend/package.json'))
const ts = require('typescript')
const playwrightPath = process.env.PLAYWRIGHT_MODULE || '/tmp/gakucho-motion-browser-check/node_modules/playwright/index.mjs'
const { chromium } = await import(playwrightPath)
const output = path.resolve(root, process.argv[2] || 'frontend/public/world-model/gakucho-avatar-profile.json')
const glb = path.join(root, 'frontend/public/avatars/gakucho_v3_suit.glb')
const glbBytes = await fs.readFile(glb)
const sha256 = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex')
const modules = new Map()
const rigDirectory = 'frontend/components/organisms/BvhViewer'
const sourceHashes = {}
for (const name of ['characterRig', 'humanoidBones', 'defaultBvh']) {
  const sourcePath = `${rigDirectory}/${name}.ts`
  const source = await fs.readFile(path.join(root, sourcePath), 'utf8')
  sourceHashes[sourcePath] = sha256(source)
  modules.set(`/rig/${name}.js`, ts.transpileModule(source, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
  }).outputText.replace("from './humanoidBones'", "from './humanoidBones.js'"))
}
const html = `<!doctype html><meta charset="utf-8"><script type="importmap">${JSON.stringify({
  imports: {
    three: '/vendor/build/three.module.js',
    'three/': '/vendor/',
  },
})}</script><script type="module">
import * as THREE from 'three'
import { BVHLoader } from 'three/examples/jsm/loaders/BVHLoader.js'
import { loadCharacterModel, bindCharacterToSource, createSourceRig } from '/rig/characterRig.js'
import { DEFAULT_BVH } from '/rig/defaultBvh.js'
window.measure = async () => {
  const character = await loadCharacterModel('/avatars/gakucho_v3_suit.glb')
  const bvh = new BVHLoader().parse(DEFAULT_BVH)
  bvh.skeleton.bones[0].updateMatrixWorld(true)
  const sourceBounds = new THREE.Box3()
  for (const bone of bvh.skeleton.bones) sourceBounds.expandByPoint(bone.getWorldPosition(new THREE.Vector3()))
  const nativeSourceHeight = sourceBounds.max.y - sourceBounds.min.y
  // Same unit normalization as BvhViewer. The explicit physical calibration
  // uses 1.8 m, also used by existing room-prototype motions.
  const targetHeight = 1.8
  const normalization = targetHeight / nativeSourceHeight
  if (!(normalization > .5 && normalization < 2)) {
    for (const bone of bvh.skeleton.bones) bone.position.multiplyScalar(normalization)
    for (const track of bvh.clip.tracks) if (track.name.endsWith('.position')) {
      for (let i = 0; i < track.values.length; i++) track.values[i] *= normalization
    }
  }
  const source = createSourceRig(bvh.skeleton, bvh.clip)
  bindCharacterToSource(character, source)
  const scale = targetHeight / character.naturalHeight
  const yawDegrees = character.autoYawDegrees
  character.root.scale.setScalar(scale)
  character.root.rotation.y = THREE.MathUtils.degToRad(yawDegrees)
  character.root.position.set(0, -character.groundOffset * scale, 0)
  character.root.updateMatrixWorld(true)
  character.model.traverse((object) => {
    if (object.isSkinnedMesh) object.skeleton.update()
  })
  const canonicalByObject = new Map([...character.bonesByCanonical].map(([name, bone]) => [bone, name]))
  const hips = character.bonesByCanonical.get('hips').getWorldPosition(new THREE.Vector3())
  // Horizontal centering represents the MuJoCo free pelvis at (0,0,height).
  character.root.position.x -= hips.x
  character.root.position.z -= hips.z
  character.root.updateMatrixWorld(true)
  const joints = {}
  for (const [canonical, bone] of character.bonesByCanonical) {
    let parent = bone.parent
    while (parent && !canonicalByObject.has(parent)) parent = parent.parent
    joints[canonical] = {
      boneName: bone.userData.name || bone.name,
      runtimeBoneName: bone.name,
      parentCanonical: canonicalByObject.get(parent) || null,
      position: bone.getWorldPosition(new THREE.Vector3()).toArray(),
      worldQuaternion: bone.getWorldQuaternion(new THREE.Quaternion()).toArray(),
      localQuaternion: bone.quaternion.toArray(), localPosition: bone.position.toArray(),
      parentWorldQuaternion: bone.parent.getWorldQuaternion(new THREE.Quaternion()).toArray(),
    }
  }
  const physicalVector = (v) => [v[0], -v[2], v[1]]
  const physicalJoints = Object.fromEntries(Object.entries(joints).map(([canonical, joint]) => [canonical, {
    position: physicalVector(joint.position),
    sourceBone: joint.boneName,
  }]))
  const length = (a, b) => new THREE.Vector3(...joints[a].position).distanceTo(new THREE.Vector3(...joints[b].position))
  const foot = {}
  const bounds = new THREE.Box3()
  let totalVertices = 0
  const point = new THREE.Vector3()
  const meshes = []
  character.model.traverse((object) => { if (object.isMesh) meshes.push(object) })
  for (const mesh of meshes) {
    if (mesh.isSkinnedMesh) mesh.skeleton.update()
    const position = mesh.geometry.getAttribute('position')
    for (let i = 0; i < position.count; i++) {
      mesh.getVertexPosition(i, point).applyMatrix4(mesh.matrixWorld)
      bounds.expandByPoint(point)
      totalVertices++
    }
  }
  const headBone = character.bonesByCanonical.get('head')
  const descendsFromHead = (node) => {
    while (node) { if (node === headBone) return true; node = node.parent }
    return false
  }
  const headBounds = new THREE.Box3()
  let headVertices = 0
  for (const mesh of meshes) {
    const indices = mesh.geometry.getAttribute('skinIndex')
    const weights = mesh.geometry.getAttribute('skinWeight')
    const headIndices = mesh.isSkinnedMesh
      ? new Set(mesh.skeleton.bones.map((bone, i) => descendsFromHead(bone) ? i : -1).filter((i) => i >= 0))
      : null
    if (!mesh.isSkinnedMesh && !descendsFromHead(mesh)) continue
    for (let i = 0; i < mesh.geometry.getAttribute('position').count; i++) {
      if (mesh.isSkinnedMesh) {
        let headWeight = 0
        for (let k = 0; k < 4; k++) if (headIndices.has(indices.getComponent(i, k))) headWeight += weights.getComponent(i, k)
        if (headWeight < .5) continue
      }
      mesh.getVertexPosition(i, point).applyMatrix4(mesh.matrixWorld)
      if (point.y < joints.head.position[1] - .08) continue
      headBounds.expandByPoint(point); headVertices++
    }
  }
  if (headVertices < 50 || headBounds.isEmpty()) throw new Error('Insufficient actual head mesh vertices')
  const headCenter = headBounds.getCenter(new THREE.Vector3())
  const headHalf = headBounds.getSize(new THREE.Vector3()).multiplyScalar(.5)
  const head = {
    radius: Math.max(headHalf.x, headHalf.y, headHalf.z),
    centerOffset: physicalVector(headCenter.clone().sub(new THREE.Vector3(...joints.head.position)).toArray()),
    measurement: {
      method: 'head descendant weights >= 0.5 plus attached hair; vertices at or above head bone - 0.08 m; sphere encloses measured AABB on each principal axis',
      selectedVertices: headVertices,
      boundsYUp: { min: headBounds.min.toArray(), max: headBounds.max.toArray() },
    },
  }
  // Measure the skin directly influenced by the pelvis. This is recorded as
  // contact calibration evidence; appearance cannot supply pelvis inertia.
  const pelvisBone = character.bonesByCanonical.get('hips')
  const pelvisBounds = new THREE.Box3()
  let pelvisVertices = 0
  const pelvisSelection = []
  for (const mesh of meshes) {
    if (!mesh.isSkinnedMesh) continue
    const indices = mesh.geometry.getAttribute('skinIndex')
    const weights = mesh.geometry.getAttribute('skinWeight')
    const pelvisIndex = mesh.skeleton.bones.indexOf(pelvisBone)
    if (pelvisIndex < 0) continue
    for (let i = 0; i < indices.count; i++) {
      let weight = 0
      for (let k = 0; k < 4; k++) if (indices.getComponent(i, k) === pelvisIndex) weight += weights.getComponent(i, k)
      if (weight < .35) continue
      mesh.getVertexPosition(i, point).applyMatrix4(mesh.matrixWorld)
      pelvisBounds.expandByPoint(point); pelvisVertices++
      pelvisSelection.push([mesh, i])
    }
  }
  const pelvisSkin = {
    method: 'actual rendered vertices with direct pelvis skin weight >= 0.35; rest-pose envelope, not a measured seated cushion compression',
    selectedVertices: pelvisVertices,
    boundsYUp: { min: pelvisBounds.min.toArray(), max: pelvisBounds.max.toArray() },
    relativeBoundsZUp: {
      min: [pelvisBounds.min.x, -pelvisBounds.max.z, pelvisBounds.min.y - joints.hips.position[1]],
      max: [pelvisBounds.max.x, -pelvisBounds.min.z, pelvisBounds.max.y - joints.hips.position[1]],
    },
  }
  // A geometric right-angle sitting calibration: thighs forward, shanks down.
  // Only joint rotations change; this is measurement, never a rollout patch.
  const sitDelta = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2)
  for (const side of ['left', 'right']) {
    for (const part of ['UpLeg', 'LowerLeg']) {
      const canonical = side + part
      const bone = character.bonesByCanonical.get(canonical)
      const restWorld = new THREE.Quaternion(...joints[canonical].worldQuaternion)
      const desired = part === 'UpLeg' ? sitDelta.clone().multiply(restWorld) : restWorld
      character.root.updateMatrixWorld(true)
      const parent = bone.parent.getWorldQuaternion(new THREE.Quaternion())
      bone.quaternion.copy(parent.invert().multiply(desired))
    }
  }
  character.root.updateMatrixWorld(true)
  for (const mesh of meshes) if (mesh.isSkinnedMesh) mesh.skeleton.update()
  const seatedPelvisBounds = new THREE.Box3()
  for (const [mesh, index] of pelvisSelection) {
    mesh.getVertexPosition(index, point).applyMatrix4(mesh.matrixWorld)
    seatedPelvisBounds.expandByPoint(point)
  }
  pelvisSkin.seatedCalibration = {
    pose: 'geometric 90-degree hip and knee seat pose: thighs forward, shanks down; no root translation',
    boundsYUp: { min: seatedPelvisBounds.min.toArray(), max: seatedPelvisBounds.max.toArray() },
    rootRelativeBottom: seatedPelvisBounds.min.y - joints.hips.position[1],
    pelvisWorldPosition: joints.hips.position,
    note: 'contact geometry calibration only; not a learned seated behavior or soft-tissue measurement',
  }
  const pelvisHalf = seatedPelvisBounds.getSize(new THREE.Vector3()).multiplyScalar(.5)
  const pelvisCenter = seatedPelvisBounds.getCenter(new THREE.Vector3())
  const pelvisRadius = Math.max(pelvisHalf.y, pelvisHalf.z)
  const seatOffset = joints.hips.position[1] - seatedPelvisBounds.min.y
  const pelvis = {
    radius: pelvisRadius, axisHalfLength: pelvisHalf.x - pelvisRadius,
    centerOffset: [pelvisCenter.x, -pelvisCenter.z, pelvisRadius - seatOffset],
    seatOffset,
    measurement: {
      method: 'X-axis capsule: measured seated pelvis width/depth and lowest rendered pelvis vertex; circular Y/Z simplification',
      source: 'calibration.pelvisSkin.seatedCalibration',
      massWasMeasured: false,
    },
  }
  for (const side of ['left', 'right']) for (const part of ['UpLeg', 'LowerLeg']) {
    character.bonesByCanonical.get(side + part).quaternion.fromArray(joints[side + part].localQuaternion)
  }
  character.root.updateMatrixWorld(true)
  for (const mesh of meshes) if (mesh.isSkinnedMesh) mesh.skeleton.update()
  for (const side of ['left', 'right']) {
    const footBone = character.bonesByCanonical.get(side + 'Foot')
    const toeBone = character.bonesByCanonical.get(side + 'Toe')
    const isFootBone = (bone) => {
      let node = bone
      while (node) {
        if (node === footBone || node === toeBone) return true
        node = node.parent
      }
      return false
    }
    const shoeBounds = new THREE.Box3()
    let selectedVertices = 0
    let selectedMeshes = new Set()
    for (const mesh of meshes) {
      if (!mesh.isSkinnedMesh) continue
      const indices = mesh.geometry.getAttribute('skinIndex')
      const weights = mesh.geometry.getAttribute('skinWeight')
      if (!indices || !weights) continue
      const bones = new Set(mesh.skeleton.bones.map((bone, i) => isFootBone(bone) ? i : -1).filter((i) => i >= 0))
      for (let i = 0; i < indices.count; i++) {
        let footWeight = 0
        for (let k = 0; k < 4; k++) if (bones.has(indices.getComponent(i, k))) footWeight += weights.getComponent(i, k)
        if (footWeight < .5) continue
        mesh.getVertexPosition(i, point).applyMatrix4(mesh.matrixWorld)
        // Separate shoes/sole from any weighted trouser cuff above the ankle.
        if (point.y > joints[side + 'Foot'].position[1] + .025) continue
        shoeBounds.expandByPoint(point)
        selectedVertices++
        selectedMeshes.add(mesh.name)
      }
    }
    if (selectedVertices < 50 || shoeBounds.isEmpty()) throw new Error('Insufficient actual sole vertices: ' + side)
    const center = shoeBounds.getCenter(new THREE.Vector3())
    const half = shoeBounds.getSize(new THREE.Vector3()).multiplyScalar(.5)
    const ankle = new THREE.Vector3(...joints[side + 'Foot'].position)
    foot[side] = {
      halfSize: [half.x, half.z, half.y],
      centerOffset: physicalVector(center.clone().sub(ankle).toArray()),
      measurement: {
        method: 'skinned foot/toe descendant weights >= 0.5; vertices at or below ankle + 0.025 m',
        selectedVertices, selectedMeshes: [...selectedMeshes],
        boundsYUp: { min: shoeBounds.min.toArray(), max: shoeBounds.max.toArray() },
        soleHeight: shoeBounds.min.y,
      },
    }
  }
  const rootPosition = physicalJoints.hips.position
  const hipOffsets = {}, legs = {}, arms = {}, verticalClosure = {}
  for (const side of ['left', 'right']) {
    hipOffsets[side] = physicalJoints[side + 'UpLeg'].position.map((v, i) => v - rootPosition[i])
    legs[side] = { thighLength: length(side + 'UpLeg', side + 'LowerLeg'), shankLength: length(side + 'LowerLeg', side + 'Foot') }
    arms[side] = { upperArmLength: length(side + 'UpperArm', side + 'LowerArm'), forearmLength: length(side + 'LowerArm', side + 'Hand') }
    verticalClosure[side] = rootPosition[2] + hipOffsets[side][2] - legs[side].thighLength - legs[side].shankLength + foot[side].centerOffset[2] - foot[side].halfSize[2]
  }
  return {
    version: 1, avatarId: 'gakucho_v3_suit',
    display: {
      coordinateSystem: 'Y-up, +Z-forward', quaternionOrder: 'xyzw', units: 'metres',
      modelUrl: '/avatars/gakucho_v3_suit.glb', naturalHeight: character.naturalHeight,
      normalizedHeight: targetHeight, scale, yawDegrees,
      groundOffset: character.groundOffset, rootPosition: character.root.position.toArray(),
      rootQuaternion: character.root.quaternion.toArray(),
      targetRootUnitScale: character.targetRootUnitScale,
      restPose: 'T-pose aligned by existing bindCharacterToSource to DEFAULT_BVH',
      rigFlavor: character.flavor, joints,
    },
    physical: {
      coordinateSystem: 'Z-up, -Y-forward', units: 'metres', rootHeight: rootPosition[2],
      height: bounds.max.y - bounds.min.y, referenceHeight: targetHeight,
      hipOffsets, legs, arms, foot, head, pelvis, joints: physicalJoints,
    },
    calibration: {
      method: 'existing Three GLTFLoader + loadCharacterModel + bindCharacterToSource; all dimensions from deformed mesh and world bones',
      existingDisplayHeight: nativeSourceHeight, explicitPhysicalDisplayHeight: targetHeight,
      normalizedHeightConvention: 'room-prototype normalized 1.8 m height; DEFAULT_BVH directions and rest alignment',
      wholeMeshBoundsYUp: { min: bounds.min.toArray(), max: bounds.max.toArray() },
      displayedMeshHeight: bounds.max.y - bounds.min.y,
      totalVertices, canonicalBoneCount: Object.keys(joints).length,
      straightLegSoleClosureMetres: verticalClosure,
      pelvisSkin,
      massWasMeasured: false, jointStrengthWasMeasured: false,
      dynamicAssumptions: '64.7 kg distribution and 20-hinge torque/joint bounds remain existing simplified physics assumptions',
    },
  }
}
</script>`

const server = http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url, 'http://localhost')
    if (url.pathname === '/') {
      response.writeHead(200, { 'Content-Type': 'text/html' }); response.end(html); return
    }
    if (modules.has(url.pathname)) {
      response.writeHead(200, { 'Content-Type': 'text/javascript' }); response.end(modules.get(url.pathname)); return
    }
    let filename
    if (url.pathname.startsWith('/vendor/')) filename = path.join(root, 'frontend/node_modules/three', url.pathname.slice(8))
    else if (url.pathname === '/avatars/gakucho_v3_suit.glb') filename = glb
    else { response.writeHead(404); response.end(); return }
    const bytes = await fs.readFile(filename)
    response.writeHead(200, { 'Content-Type': filename.endsWith('.js') ? 'text/javascript' : 'model/gltf-binary' })
    response.end(bytes)
  } catch (error) { response.writeHead(500); response.end(String(error)) }
})
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
let browser
try {
  browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--no-proxy-server'] })
  const page = await browser.newPage()
  page.on('console', (message) => { if (message.type() === 'log') console.log(message.text()) })
  page.on('pageerror', (error) => console.error(error))
  await page.goto(`http://127.0.0.1:${server.address().port}/`)
  await page.waitForFunction(() => typeof window.measure === 'function')
  const profile = await page.evaluate(() => window.measure())
  profile.provenance = {
    modelPath: 'frontend/public/avatars/gakucho_v3_suit.glb', modelSha256: sha256(glbBytes),
    modelBytes: glbBytes.length, measurementScript: 'tools/measure-gakucho-avatar.mjs',
    sourceSha256: sourceHashes, measurementScriptSha256: sha256(await fs.readFile(fileURLToPath(import.meta.url))),
    threeVersion: JSON.parse(await fs.readFile(path.join(root, 'frontend/node_modules/three/package.json'), 'utf8')).version,
  }
  profile.display.modelSha256 = profile.provenance.modelSha256
  await fs.mkdir(path.dirname(output), { recursive: true })
  await fs.writeFile(output, JSON.stringify(profile, null, 2) + '\n')
  console.log(JSON.stringify({ output: path.relative(root, output), rootHeight: profile.physical.rootHeight,
    naturalHeight: profile.display.naturalHeight, scale: profile.display.scale, yaw: profile.display.yawDegrees,
    legs: profile.physical.legs, foot: profile.physical.foot,
    soleClosure: profile.calibration.straightLegSoleClosureMetres }, null, 2))
} finally {
  if (browser) await browser.close()
  await new Promise((resolve) => server.close(resolve))
}
