#!/usr/bin/env node
// Read-only actual PPO/WBC recorded-frame audit using the real rendered GLB.
import fs from 'node:fs/promises'
import path from 'node:path'
import http from 'node:http'
import crypto from 'node:crypto'
import { fileURLToPath } from 'node:url'
import { createRequire } from 'node:module'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const require = createRequire(path.join(root, 'frontend/package.json'))
const ts = require('typescript')
const { chromium } = await import('/tmp/gakucho-motion-browser-check/node_modules/playwright/index.mjs')
const paths = process.argv.slice(2).length ? process.argv.slice(2) : [
  '.runtime/world-model/stage1-gakucho-standing/rollout.json',
  '.runtime/world-model/stage1-gakucho-seating/rollout.json',
  '.runtime/world-model/verification/gakucho-wbc-walk-flat-1m-rollout.json',
]
const traces = await Promise.all(paths.map(async sourcePath => {
  const bytes = await fs.readFile(path.resolve(root, sourcePath))
  return { sourcePath, sourceTraceSha256: crypto.createHash('sha256').update(bytes).digest('hex'), trace: JSON.parse(bytes.toString('utf8')) }
}))
const modules = new Map()
for (const [modulePath, sourcePath] of [
  ['/rig/characterRig.js', 'frontend/components/organisms/BvhViewer/characterRig.ts'],
  ['/rig/humanoidBones.js', 'frontend/components/organisms/BvhViewer/humanoidBones.ts'],
  ['/rig/physicsAvatar.js', 'frontend/components/organisms/WorldModelLab/physicsAvatar.ts'],
]) {
  modules.set(modulePath, ts.transpileModule(await fs.readFile(path.join(root, sourcePath), 'utf8'), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
  }).outputText.replace("from './humanoidBones'", "from './humanoidBones.js'").replace("from '../BvhViewer/characterRig'", "from './characterRig.js'"))
}
const html = `<!doctype html><meta charset="utf-8"><script type="importmap">${JSON.stringify({ imports: { three: '/vendor/build/three.module.js', 'three/': '/vendor/' } })}</script>
<script type="module">
import * as THREE from 'three'
import { loadPhysicsAvatar, physicsPosition, physicsQuaternion } from '/rig/physicsAvatar.js'
window.audit = async () => {
 const records=await(await fetch('/traces.json')).json()
 const results=[]
 const distribution=values=>{
   const sorted=values.filter(Number.isFinite).sort((a,b)=>a-b)
   const percentile=p=>sorted.length?sorted[Math.min(sorted.length-1,Math.floor((sorted.length-1)*p))]:null
   return {count:sorted.length,min:sorted[0]??null,max:sorted.at(-1)??null,mean:sorted.length?sorted.reduce((a,b)=>a+b,0)/sorted.length:null,p05:percentile(.05),p50:percentile(.5),p95:percentile(.95),p99:percentile(.99)}
 }
 for(const {sourcePath,sourceTraceSha256,trace} of records){
   const avatar=await loadPhysicsAvatar(trace.avatarProfile,trace.physicsRestBodies)
   avatar.update(trace.physicsRestBodies,trace.physicsRestBodies,0)
   const meshes=[];avatar.character.model.traverse(object=>{if(object.isSkinnedMesh){object.skeleton.update();meshes.push(object)}})
   const point=new THREE.Vector3()
   const selections={left:[],right:[],pelvis:[]}
   const hips=avatar.character.bonesByCanonical.get('hips')
   for(const mesh of meshes){
     const indices=mesh.geometry.getAttribute('skinIndex'),weights=mesh.geometry.getAttribute('skinWeight')
     const hipIndex=mesh.skeleton.bones.indexOf(hips)
     for(const side of ['left','right']){
       const foot=avatar.character.bonesByCanonical.get(side+'Foot')
       const descends=bone=>{while(bone){if(bone===foot)return true;bone=bone.parent}return false}
       const boneIndices=new Set(mesh.skeleton.bones.map((bone,index)=>descends(bone)?index:-1).filter(index=>index>=0))
       const ankle=foot.getWorldPosition(new THREE.Vector3())
       for(let index=0;index<indices.count;index++){
         let weight=0;for(let component=0;component<4;component++)if(boneIndices.has(indices.getComponent(index,component)))weight+=weights.getComponent(index,component)
         if(weight<.5)continue
         mesh.getVertexPosition(index,point).applyMatrix4(mesh.matrixWorld)
         if(point.y>ankle.y+.025)continue
         selections[side].push([mesh,index])
       }
     }
     if(hipIndex>=0)for(let index=0;index<indices.count;index++){
       let weight=0;for(let component=0;component<4;component++)if(indices.getComponent(index,component)===hipIndex)weight+=weights.getComponent(index,component)
       if(weight>=.35)selections.pelvis.push([mesh,index])
     }
   }
   const rows=[],errors=[],footSupported={left:[],right:[]},footBothSupported={left:[],right:[]},seatSupported=[],seatFinal=[],footProxyDeltas={left:[],right:[]}
   const minimumY=selection=>{let minimum=Infinity;for(const [mesh,index]of selection){mesh.getVertexPosition(index,point).applyMatrix4(mesh.matrixWorld);minimum=Math.min(minimum,point.y)}return minimum}
   for(let index=0;index<trace.frames.length;index++){
     const frame=trace.frames[index]
     avatar.update(frame.bodies,frame.bodies,0)
     for(const mesh of meshes)mesh.skeleton.update()
     const diagnostics=avatar.diagnostics();errors.push(diagnostics.maximumJointError)
     const floor=frame.geoms.find(geom=>/floor|ground/.test(geom.id)&&geom.body==='world')
     const floorTop=floor?floor.position[2]+floor.size[2]:0
     const row={time:frame.time,upright:frame.stats.upright,feet:{},pelvis:null,maximumJointError:diagnostics.maximumJointError}
     const both=['left','right'].every(side=>frame.stats[side+'_foot_force']>50)
     for(const side of ['left','right']){
       const geom=frame.geoms.find(geom=>geom.id==='humanoid_'+side+'_foot')
       const orientation=physicsQuaternion(geom.quaternion),centre=physicsPosition(geom.position)
       const size=geom.size,half=new THREE.Vector3(size[0],size[2],size[1]);let proxyMinimum=Infinity
       for(const x of[-1,1])for(const y of[-1,1])for(const z of[-1,1])proxyMinimum=Math.min(proxyMinimum,new THREE.Vector3(x*half.x,y*half.y,z*half.z).applyQuaternion(orientation).add(centre).y)
       const shoeMinimum=minimumY(selections[side]),gap=shoeMinimum-floorTop,force=frame.stats[side+'_foot_force']
       const body=frame.bodies.find(body=>body.name===side+'_foot')
       const angle=new THREE.Vector3(0,1,0).applyQuaternion(physicsQuaternion(body.quaternion)).angleTo(new THREE.Vector3(0,1,0))
       row.feet[side]={force,gap,shoeMinimum,proxyMinimum,shoeProxyDelta:shoeMinimum-proxyMinimum,tiltRadians:angle}
       if(force>50){footSupported[side].push(gap);footProxyDeltas[side].push(shoeMinimum-proxyMinimum);if(both)footBothSupported[side].push(gap)}
     }
     const seat=frame.geoms.find(geom=>/seat/.test(geom.id)&&geom.body==='world')
     if(seat){
       const seatTop=seat.position[2]+seat.size[2],allSelectedMinimum=minimumY(selections.pelvis)
       const seatPosition=physicsPosition(seat.position),seatInverse=physicsQuaternion(seat.quaternion).invert()
       let minimum=Infinity,insideVertices=0,lowestPosition=null
       for(const [mesh,index]of selections.pelvis){
         mesh.getVertexPosition(index,point).applyMatrix4(mesh.matrixWorld)
         const local=point.clone().sub(seatPosition).applyQuaternion(seatInverse)
         if(Math.abs(local.x)>seat.size[0]||Math.abs(local.z)>seat.size[1])continue
         insideVertices++
         if(point.y<minimum){minimum=point.y;lowestPosition=point.toArray()}
       }
       const pelvisGeom=frame.geoms.find(geom=>geom.id==='humanoid_pelvis')
       const axis=new THREE.Vector3(0,1,0).applyQuaternion(physicsQuaternion(pelvisGeom.quaternion))
       const proxyMinimum=pelvisGeom.position[2]-pelvisGeom.size[1]*Math.abs(axis.y)-pelvisGeom.size[0]
       row.pelvis={force:frame.stats.pelvis_support_force,minimum,allSelectedMinimum,insideVertices,lowestPosition,seatTop,gap:minimum-seatTop,proxyMinimum,skinProxyDelta:minimum-proxyMinimum}
       if(frame.stats.pelvis_support_force>50&&insideVertices){seatSupported.push(row.pelvis.gap);if(frame.time>=trace.frames.at(-1).time-2)seatFinal.push(row.pelvis.gap)}
     }
     rows.push(row)
     if(index%100===0)await new Promise(resolve=>setTimeout(resolve,0))
   }
   results.push({sourcePath,sourceTraceSha256,controller:trace.controller,bodyModelSha256:trace.bodyModelSha256,avatarProfileSha256:trace.avatarProfileSha256,
     scope:'All recorded 50 Hz frames; force > 50 N defines supported; no pose or trace modification',
     frameCount:rows.length,selectedVertices:Object.fromEntries(Object.entries(selections).map(([name,vertices])=>[name,vertices.length])),
     jointErrors:distribution(errors),footSupported:Object.fromEntries(Object.entries(footSupported).map(([side,values])=>[side,distribution(values)])),
     footBothSupported:Object.fromEntries(Object.entries(footBothSupported).map(([side,values])=>[side,distribution(values)])),
     footShoeProxyDelta:Object.fromEntries(Object.entries(footProxyDeltas).map(([side,values])=>[side,distribution(values)])),
     seatSupported:distribution(seatSupported),seatFinalTwoSeconds:distribution(seatFinal),rows})
   avatar.dispose()
   console.log('audited '+sourcePath+' frames='+rows.length)
 }
 return results
}
</script>`
const server = http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url, 'http://localhost').pathname
    if (url === '/') { response.writeHead(200, { 'Content-Type': 'text/html' }); response.end(html); return }
    if (url === '/traces.json') { response.writeHead(200, { 'Content-Type': 'application/json' }); response.end(JSON.stringify(traces)); return }
    if (url === '/favicon.ico') { response.writeHead(204); response.end(); return }
    if (modules.has(url)) { response.writeHead(200, { 'Content-Type': 'text/javascript' }); response.end(modules.get(url)); return }
    const filename = url.startsWith('/vendor/') ? path.join(root, 'frontend/node_modules/three', url.slice(8)) : url.startsWith('/avatars/') ? path.join(root, 'frontend/public', url) : null
    if (!filename) { response.writeHead(404); response.end(); return }
    response.writeHead(200, { 'Content-Type': filename.endsWith('.js') ? 'text/javascript' : 'model/gltf-binary' }); response.end(await fs.readFile(filename))
  } catch (error) { response.writeHead(500); response.end(String(error)) }
})
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
let browser
try {
  browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage', '--no-proxy-server'] })
  const page = await browser.newPage()
  const errors = []
  page.on('pageerror', error => errors.push(String(error)))
  page.on('console', message => { if (message.type() === 'log') console.log(message.text()); else if (message.type() === 'error') errors.push(message.text()) })
  await page.goto(`http://127.0.0.1:${server.address().port}/`)
  await page.waitForFunction(() => typeof window.audit === 'function')
  const results = await page.evaluate(() => window.audit())
  const output = path.join(root, '.runtime/world-model/verification/avatar-actual-rollout-contact-report.json')
  await fs.writeFile(output, JSON.stringify({ version: 1, errors, results }, null, 2) + '\n')
  console.log(JSON.stringify({ output, errors, results: results.map(({ rows, ...summary }) => summary) }, null, 2))
  if (errors.length || results.some(result => result.jointErrors.max > .00003)) throw new Error('Real GLB/actual-body alignment failed')
} finally {
  if (browser) await browser.close()
  await new Promise(resolve => server.close(resolve))
}
