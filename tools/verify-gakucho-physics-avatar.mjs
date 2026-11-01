#!/usr/bin/env node
// Exercise the application's actual GLB loader and physics retargeter in Chrome.
import fs from 'node:fs/promises'
import path from 'node:path'
import http from 'node:http'
import { fileURLToPath } from 'node:url'
import { createRequire } from 'node:module'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const require = createRequire(path.join(root, 'frontend/package.json'))
const ts = require('typescript')
const { chromium } = await import('/tmp/gakucho-motion-browser-check/node_modules/playwright/index.mjs')
const input = path.resolve(root, process.argv[2] || '.runtime/world-model/verification/avatar-retarget-cases.json')
const reportPath = path.resolve(root, process.argv[3] || '.runtime/world-model/verification/avatar-retarget-browser-report.json')
const modules = new Map()
for (const [modulePath, sourcePath] of [
  ['/rig/characterRig.js', 'frontend/components/organisms/BvhViewer/characterRig.ts'],
  ['/rig/humanoidBones.js', 'frontend/components/organisms/BvhViewer/humanoidBones.ts'],
  ['/rig/physicsAvatar.js', 'frontend/components/organisms/WorldModelLab/physicsAvatar.ts'],
]) {
  const source = await fs.readFile(path.join(root, sourcePath), 'utf8')
  modules.set(modulePath, ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } }).outputText
    .replace("from './humanoidBones'", "from './humanoidBones.js'")
    .replace("from '../BvhViewer/characterRig'", "from './characterRig.js'"))
}
const html = `<!doctype html><meta charset="utf-8"><style>body{margin:0;background:#152234}canvas{display:block}</style>
<script type="importmap">${JSON.stringify({ imports: { three: '/vendor/build/three.module.js', 'three/': '/vendor/' } })}</script>
<script type="module">
import * as THREE from 'three'
import { loadPhysicsAvatar, physicsPosition, physicsQuaternion } from '/rig/physicsAvatar.js'
window.verify = async () => {
  const input = await (await fetch('/cases.json')).json()
  const avatar = await loadPhysicsAvatar(input.profile, input.restBodies)
  const scene = new THREE.Scene(); scene.background = new THREE.Color(0x152234); scene.add(avatar.root)
  scene.add(new THREE.HemisphereLight(0xffffff,0x75899e,2.5))
  const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(3,5,4);scene.add(light)
  const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setSize(900,750);document.body.appendChild(renderer.domElement)
  const camera=new THREE.PerspectiveCamera(40,900/750,.03,100)
  const meshes=[];avatar.character.model.traverse(object=>{if(object.isMesh)meshes.push(object)})
  const point=new THREE.Vector3()
  const samples=[]
  for(const item of input.cases){
    avatar.update(item.frame.bodies,item.frame.bodies,0)
    for(const mesh of meshes)if(mesh.isSkinnedMesh)mesh.skeleton.update()
    const diagnostics=avatar.diagnostics()
    const soles={}
    for(const side of ['left','right']){
      const foot=avatar.character.bonesByCanonical.get(side+'Foot')
      const descends=bone=>{while(bone){if(bone===foot)return true;bone=bone.parent}return false}
      const body=item.frame.bodies.find(body=>body.name===side+'_foot')
      const bodyPos=physicsPosition(body.position),inverse=physicsQuaternion(body.quaternion).invert()
      const bounds=new THREE.Box3(),worldBounds=new THREE.Box3();let count=0
      for(const mesh of meshes){
        if(!mesh.isSkinnedMesh)continue
        const indices=mesh.geometry.getAttribute('skinIndex'),weights=mesh.geometry.getAttribute('skinWeight')
        const valid=new Set(mesh.skeleton.bones.map((bone,index)=>descends(bone)?index:-1).filter(index=>index>=0))
        for(let index=0;index<indices.count;index++){
          let weight=0;for(let component=0;component<4;component++)if(valid.has(indices.getComponent(index,component)))weight+=weights.getComponent(index,component)
          if(weight<.5)continue
          mesh.getVertexPosition(index,point).applyMatrix4(mesh.matrixWorld)
          const local=point.clone().sub(bodyPos).applyQuaternion(inverse)
          if(local.y>.025)continue
          bounds.expandByPoint(local);worldBounds.expandByPoint(point);count++
        }
      }
      const expected=input.profile.physical.foot[side]
      const expectedCenter=physicsPosition(expected.centerOffset), half=new THREE.Vector3(expected.halfSize[0],expected.halfSize[2],expected.halfSize[1])
      const expectedMin=expectedCenter.clone().sub(half),expectedMax=expectedCenter.clone().add(half)
      let physicalSoleY=Infinity
      const orientation=physicsQuaternion(body.quaternion)
      for(const x of [-1,1])for(const y of [-1,1])for(const z of [-1,1])physicalSoleY=Math.min(physicalSoleY,
        expectedCenter.clone().add(new THREE.Vector3(x*half.x,y*half.y,z*half.z)).applyQuaternion(orientation).add(bodyPos).y)
      soles[side]={count,minimumY:worldBounds.min.y,localMin:bounds.min.toArray(),localMax:bounds.max.toArray(),
        physicalSoleY,soleHeightError:worldBounds.min.y-physicalSoleY,localSoleHeightError:bounds.min.y-expectedMin.y,
        expectedMin:expectedMin.toArray(),expectedMax:expectedMax.toArray(), maximumBoundsError:Math.max(...bounds.min.clone().sub(expectedMin).toArray().map(Math.abs),...bounds.max.clone().sub(expectedMax).toArray().map(Math.abs))}
    }
    const allBounds=new THREE.Box3()
    for(const mesh of meshes)for(let index=0;index<mesh.geometry.getAttribute('position').count;index++){
      mesh.getVertexPosition(index,point).applyMatrix4(mesh.matrixWorld);allBounds.expandByPoint(point)
    }
    const pelvisBounds=new THREE.Box3();let pelvisVertices=0
    const pelvisBone=avatar.character.bonesByCanonical.get('hips')
    for(const mesh of meshes){
      if(!mesh.isSkinnedMesh)continue
      const boneIndex=mesh.skeleton.bones.indexOf(pelvisBone),indices=mesh.geometry.getAttribute('skinIndex'),weights=mesh.geometry.getAttribute('skinWeight')
      if(boneIndex<0)continue
      for(let index=0;index<indices.count;index++){
        let weight=0;for(let component=0;component<4;component++)if(indices.getComponent(index,component)===boneIndex)weight+=weights.getComponent(index,component)
        if(weight<.35)continue
        mesh.getVertexPosition(index,point).applyMatrix4(mesh.matrixWorld);pelvisBounds.expandByPoint(point);pelvisVertices++
      }
    }
    const pelvisBody=item.frame.bodies.find(body=>body.name==='pelvis')
    const physicalPelvis=input.profile.physical.pelvis
    const seatBottom=pelvisBody.position[2]+physicalPelvis.centerOffset[2]-physicalPelvis.radius
    const seat=item.frame.geoms.find(geom=>geom.id==='calibration-seat')
    const seatTop=seat?seat.position[2]+seat.size[2]:null
    samples.push({name:item.name,diagnostics,soles,meshBounds:{min:allBounds.min.toArray(),max:allBounds.max.toArray()},
      pelvis:{vertices:pelvisVertices,minimumY:pelvisBounds.min.y,physicalMinimumY:seatBottom,seatTop,
        geometricSeatError:item.name==='seated'?pelvisBounds.min.y-seatBottom:null}})
    const pelvis=physicsPosition(item.frame.bodies.find(body=>body.name==='pelvis').position)
    camera.position.copy(pelvis).add(new THREE.Vector3(2.3,1.4,3.2));camera.lookAt(pelvis.clone().add(new THREE.Vector3(0,.05,0)))
    renderer.render(scene,camera)
  }
  window.avatarVerification={avatar,scene,renderer,camera,input}
  let mismatchedAssetRejected=false
  try{await loadPhysicsAvatar({...input.profile,display:{...input.profile.display,modelSha256:'0'.repeat(64)}},input.restBodies)}
  catch(error){mismatchedAssetRejected=String(error).includes('実測情報が一致しません')}
  return {samples,modelSha256:avatar.root.userData.physicsAvatar.modelSha256,mismatchedAssetRejected}
}
</script>`
const server = http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url, 'http://localhost').pathname
    if (url === '/') { response.writeHead(200, { 'Content-Type': 'text/html' }); response.end(html); return }
    if (url === '/cases.json') { response.writeHead(200, { 'Content-Type': 'application/json' }); response.end(await fs.readFile(input)); return }
    if (url === '/favicon.ico') { response.writeHead(204); response.end(); return }
    if (modules.has(url)) { response.writeHead(200, { 'Content-Type': 'text/javascript' }); response.end(modules.get(url)); return }
    let filename
    if (url.startsWith('/vendor/')) filename = path.join(root, 'frontend/node_modules/three', url.slice('/vendor/'.length))
    else if (url.startsWith('/avatars/')) filename = path.join(root, 'frontend/public', url)
    else { response.writeHead(404); response.end(); return }
    response.writeHead(200, { 'Content-Type': filename.endsWith('.js') ? 'text/javascript' : 'model/gltf-binary' })
    response.end(await fs.readFile(filename))
  } catch (error) { response.writeHead(500); response.end(String(error)) }
})
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
let browser
try {
  browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--no-proxy-server', '--use-angle=swiftshader'] })
  const page = await browser.newPage({ viewport: { width: 900, height: 750 } })
  const errors = []
  page.on('pageerror', (error) => errors.push(String(error)))
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()) })
  await page.goto(`http://127.0.0.1:${server.address().port}/`)
  await page.waitForFunction(() => typeof window.verify === 'function')
  const report = await page.evaluate(() => window.verify())
  report.errors = errors
  await fs.mkdir(path.dirname(reportPath), { recursive: true })
  await fs.writeFile(reportPath, JSON.stringify(report, null, 2) + '\n')
  await page.screenshot({ path: reportPath.replace('.json', '.png') })
  if(errors.length || !report.mismatchedAssetRejected || report.samples.some(sample=>sample.diagnostics.maximumJointError>2.5e-5 ||
    Object.values(sample.soles).some(sole=>Math.abs(sole.localSoleHeightError)>.001 || sole.count<100) ||
    (sample.name==='seated'&&(Math.abs(sample.pelvis.geometricSeatError)>.002||Math.abs(sample.pelvis.minimumY-sample.pelvis.seatTop)>.002)))) {
    throw new Error('Actual GLB/MuJoCo retarget verification failed: '+reportPath)
  }
  console.log(JSON.stringify({ reportPath, errors, cases: report.samples.map((sample) => ({ name: sample.name,
    maximumJointError: sample.diagnostics.maximumJointError, soleBoundsError: Object.fromEntries(Object.entries(sample.soles).map(([side, sole]) => [side, sole.maximumBoundsError])) })) }, null, 2))
} finally {
  if (browser) await browser.close()
  await new Promise((resolve) => server.close(resolve))
}
