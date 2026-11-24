'use client'

import Image from 'next/image'
import Link from 'next/link'
import React, { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js'
import { createHouseVisual, placeHeadCamera, toThree, type HouseDescription } from './houseVisual'
import { loadPhysicsAvatar, type AvatarProfile, type PhysicalBody, type PhysicsAvatar } from '../WorldModelLab/physicsAvatar'

/** Head camera of the depth sensor (world_model/depth_perception.py): 90 degrees across, 4:3, 15 degrees down. */
const EYE = { horizontalFovDegrees: 90, aspect: 4 / 3, downPitchDegrees: 15, height: 1.612 }
const verticalFov = (horizontal: number, aspect: number) =>
  THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(horizontal) / 2) / aspect))

type View = { x: number; y: number; z?: number; floor?: number; heading: number; pitch?: number; width?: number; height?: number; hfov?: number }
type HouseApi = { ready: Promise<boolean>; render: (view: View) => string; house: HouseDescription }
declare global { interface Window { __houseLab?: HouseApi } }

/** Records of world_model/house_agent.py, published to /training-monitor/agent/. */
type AgentRunSummary = { id: string; text: string; success: boolean; finished?: boolean; claimed?: boolean | null; message?: string | null
  stoppedBy?: string | null; decisions: number; looks: number; seconds: number }
type AgentEvent = { kind: 'look' | 'decision' | 'check' | 'end'; time: number; why?: string; boxes?: string; depth?: string; map?: string
  detections?: { coco: string; conf: number; position?: number[] }[]; pose?: number[]
  vlm?: { items: { kind: 'target' | 'clue' | 'door'; what: string; id?: string }[]; seconds?: number }
  answer?: { thought?: string; action?: string; target?: string; success?: boolean; message?: string; arrived?: boolean; reason?: string }; seconds?: number
  override?: string; headTurn?: string
  objects?: string[]; frontiers?: string[]; images?: string[]; result?: Record<string, unknown> }
type AgentRun = { task: { id: string; text: string }; model: Record<string, string>; result: Record<string, unknown>; events: AgentEvent[] }
type AgentFrame = { time: number; bodies: PhysicalBody[] }
type AgentTrace = { frames: AgentFrame[]; avatarProfile: AvatarProfile; physicsRestBodies: PhysicalBody[]; avatarBodyMap: Record<string, string> }
const AGENT_BASE = '/training-monitor/agent'
const VLM_KIND: Record<string, string> = { target: '行き先', clue: '手がかり', door: '出入口' }
const ACTION_LABEL: Record<string, string> = { look_around: 'その場で見回す', go_to_object: '見つけた物・場所・出入口へ行く', explore: 'まだ見ていない所へ行く', finish: '終える' }
const COCO_JA: Record<string, string> = { refrigerator: '冷蔵庫', oven: 'コンロ', sink: 'シンク', microwave: '電子レンジ', toaster: 'トースター',
  'dining table': 'テーブル', chair: '椅子', couch: 'ソファ', tv: 'テレビ', 'potted plant': '観葉植物', vase: '花瓶', bowl: '器', clock: '時計',
  toilet: 'トイレ', bed: 'ベッド', laptop: 'ノートパソコン', book: '本', umbrella: '傘', backpack: 'かばん', 'teddy bear': 'ぬいぐるみ' }
const lastAt = <T extends { time: number }>(items: T[], time: number) => {
  let found: T | undefined
  for (const item of items) { if (item.time <= time + 1e-6) found = item; else break }
  return found
}

export function HouseLab() {
  const [house, setHouse] = useState<HouseDescription | null>(null)
  const [error, setError] = useState('')
  const [status, setStatus] = useState('家を読み込んでいます')
  const [floors, setFloors] = useState<'both' | 'first'>('both')
  const [place, setPlace] = useState('hall-1f')
  const [heading, setHeading] = useState(180)
  const overviewRef = useRef<HTMLDivElement | null>(null)
  const eyeRef = useRef<HTMLDivElement | null>(null)
  const viewRef = useRef<{ floors: 'both' | 'first'; eye: View }>({ floors: 'both', eye: { x: 0, y: 0, heading: 0 } })
  const sceneRef = useRef<THREE.Scene | null>(null)
  const [sceneReady, setSceneReady] = useState(0)
  const [runs, setRuns] = useState<AgentRunSummary[]>([])
  const [runId, setRunId] = useState('')
  const [run, setRun] = useState<AgentRun | null>(null)
  const [trace, setTrace] = useState<AgentTrace | null>(null)
  const [playTime, setPlayTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(2)
  const agentRef = useRef<{ trace: AgentTrace | null; avatar: PhysicsAvatar | null; time: number; playing: boolean; speed: number; follow: THREE.Vector3 | null }>(
    { trace: null, avatar: null, time: 0, playing: false, speed: 2, follow: null })
  const capture = typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('capture')

  useEffect(() => {
    fetch('/world-model/house2f.json', { cache: 'no-store' })
      .then((response) => { if (!response.ok) throw new Error('家の記述（house2f.json）を読み込めません'); return response.json() })
      .then((value: HouseDescription) => setHouse(value))
      .catch((reason) => setError(reason instanceof Error ? reason.message : String(reason)))
  }, [])

  const places = useMemo(() => (house?.goals ?? []).filter((goal) => goal.kind === 'walk').map((goal) => {
    const room = house?.rooms.find((item) => item.id === goal.room)
    const names: Record<string, string> = { living: 'リビング（ソファの前）', kitchen: 'キッチン', dining: 'ダイニング', entrance: '玄関ホール' }
    return { id: goal.id, label: names[goal.id] ?? room?.label ?? goal.id, floor: (room?.floor ?? 1), position: goal.position }
  }), [house])
  const current = places.find((item) => item.id === place) ?? places[0]
  useEffect(() => {
    if (!current) return
    viewRef.current.eye = { x: current.position[0], y: current.position[1], floor: current.floor, heading: THREE.MathUtils.degToRad(heading) }
  }, [current, heading])
  useEffect(() => { viewRef.current.floors = floors }, [floors])

  // ---- 指示の実行の記録（house_agent.py）
  useEffect(() => {
    if (capture) return
    fetch(`${AGENT_BASE}/index.json`, { cache: 'no-store' }).then((response) => (response.ok ? response.json() : null))
      .then((value) => {
        const list = (value?.runs ?? []) as AgentRunSummary[]
        setRuns(list)
        if (list.length) setRunId((previous) => previous || list[0].id)
      }).catch(() => undefined)
  }, [capture])
  useEffect(() => {
    if (!runId) return
    let cancelled = false
    setRun(null)
    setTrace(null)
    Promise.all([fetch(`${AGENT_BASE}/${runId}/run.json`, { cache: 'no-store' }).then((r) => r.json()),
      fetch(`${AGENT_BASE}/${runId}/trace.json`, { cache: 'no-store' }).then((r) => r.json())]).then(([record, raw]) => {
      if (cancelled) return
      const names: string[] = raw.bodyNames
      const frames: AgentFrame[] = raw.frames.map((f: { time: number; bodies: number[][] }) => ({ time: f.time,
        bodies: f.bodies.map((b, i) => ({ name: names[i], position: [b[0], b[1], b[2]], quaternion: [b[3], b[4], b[5], b[6]] })) }))
      setRun(record)
      setTrace({ frames, avatarProfile: raw.avatarProfile, physicsRestBodies: raw.physicsRestBodies, avatarBodyMap: raw.avatarBodyMap })
      setPlayTime(0)
      setPlaying(false)
    }).catch(() => { if (!cancelled) setStatus('実行の記録を読み込めませんでした') })
    return () => { cancelled = true }
  }, [runId])
  useEffect(() => { agentRef.current.trace = trace; agentRef.current.time = 0; agentRef.current.playing = false; agentRef.current.follow = null }, [trace])
  useEffect(() => { agentRef.current.speed = speed }, [speed])
  useEffect(() => {
    const scene = sceneRef.current
    if (!trace || !scene) return
    const agent = agentRef.current
    let disposed = false
    let avatar: PhysicsAvatar | null = null
    loadPhysicsAvatar(trace.avatarProfile, trace.physicsRestBodies, trace.avatarBodyMap).then((loaded) => {
      if (disposed) { loaded.dispose(); return }
      avatar = loaded
      avatar.root.traverse((object) => { const mesh = object as THREE.Mesh; if (mesh.isMesh) { mesh.castShadow = true; mesh.receiveShadow = true } })
      scene.add(avatar.root)
      avatar.update(trace.frames[0].bodies, trace.frames[0].bodies, 0)
      agent.avatar = avatar
    }).catch(() => setStatus('学長アバターを読み込めませんでした'))
    return () => {
      disposed = true
      if (avatar) { avatar.root.removeFromParent(); avatar.dispose() }
      if (agent.avatar === avatar) agent.avatar = null
    }
  }, [trace, sceneReady])
  const seekAgent = (time: number, play = false) => {
    agentRef.current.time = time
    agentRef.current.playing = play
    setPlayTime(time)
    setPlaying(play)
  }

  useEffect(() => {
    const overviewHost = overviewRef.current, eyeHost = eyeRef.current
    if (!house || !eyeHost || (!overviewHost && !capture)) return
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0xcfd8dc)
    const visual = createHouseVisual(house)
    scene.add(visual.root)
    const sun = new THREE.DirectionalLight(0xffeed5, 0.8)
    sun.position.set(-3, 4.5, -10)
    sun.target.position.set(1, 0, 2)
    scene.add(sun, sun.target)
    const makeRenderer = (keep: boolean) => {
      const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: keep })
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5))
      renderer.outputColorSpace = THREE.SRGBColorSpace
      renderer.toneMapping = THREE.ACESFilmicToneMapping
      renderer.toneMappingExposure = 0.92
      return renderer
    }
    // Each WebGL context needs its own prefiltered environment (the production home's HDR).
    const renderers: { renderer: THREE.WebGLRenderer; environment: THREE.Texture | null; pmrem: THREE.PMREMGenerator }[] = []
    const eyeRenderer = makeRenderer(true)
    eyeRenderer.domElement.style.display = 'block'
    eyeHost.appendChild(eyeRenderer.domElement)
    renderers.push({ renderer: eyeRenderer, environment: null, pmrem: new THREE.PMREMGenerator(eyeRenderer) })
    const overviewRenderer = capture ? null : makeRenderer(false)
    if (overviewRenderer && overviewHost) {
      overviewRenderer.domElement.style.display = 'block'
      overviewHost.appendChild(overviewRenderer.domElement)
      renderers.push({ renderer: overviewRenderer, environment: null, pmrem: new THREE.PMREMGenerator(overviewRenderer) })
    }
    const environmentReady = new Promise<void>((resolve) => {
      new RGBELoader().load('/environments/home/lighting/lebombo_1k.hdr', (texture) => {
        texture.mapping = THREE.EquirectangularReflectionMapping
        for (const entry of renderers) entry.environment = entry.pmrem.fromEquirectangular(texture).texture
        texture.dispose()
        resolve()
      }, undefined, () => resolve())
    })
    const draw = (entry: (typeof renderers)[number], camera: THREE.Camera, eyeView: boolean) => {
      scene.environment = entry.environment
      visual.ceilings.visible = eyeView
      visual.upper.visible = eyeView || (viewRef.current.floors === 'both' && !agentRef.current.trace)
      const avatar = agentRef.current.avatar
      if (avatar) avatar.root.visible = !eyeView   // the head camera does not see its own body
      entry.renderer.render(scene, camera)
    }

    const eyeCamera = new THREE.PerspectiveCamera(verticalFov(EYE.horizontalFovDegrees, EYE.aspect), EYE.aspect, 0.05, 60)
    const floorHeight = (floor?: number) => (floor === 2 ? house.metadata.floorHeights[1] : 0)
    const placeEye = (view: View) => {
      const hfov = view.hfov ?? EYE.horizontalFovDegrees
      const aspect = (view.width ?? 4) / (view.height ?? 3)
      eyeCamera.fov = verticalFov(hfov, aspect)
      eyeCamera.aspect = aspect
      eyeCamera.updateProjectionMatrix()
      const z = view.z ?? floorHeight(view.floor) + EYE.height
      placeHeadCamera(eyeCamera, [view.x, view.y, z], view.heading, view.pitch ?? EYE.downPitchDegrees)
    }
    const eyeSize = () => {
      const width = Math.max(1, eyeHost.clientWidth)
      eyeRenderer.setSize(width, Math.round(width / EYE.aspect))
    }
    eyeSize()

    let overviewCamera: THREE.PerspectiveCamera | null = null
    let controls: OrbitControls | null = null
    if (overviewRenderer && overviewHost) {
      overviewCamera = new THREE.PerspectiveCamera(45, 1, 0.1, 200)
      overviewCamera.position.copy(toThree(9, -17, 13))
      controls = new OrbitControls(overviewCamera, overviewRenderer.domElement)
      controls.target.copy(toThree(0, -1.6, 1.2))
      controls.enableDamping = true
      controls.update()
    }
    const resize = () => {
      eyeSize()
      if (overviewRenderer && overviewHost && overviewCamera) {
        const width = Math.max(1, overviewHost.clientWidth), height = Math.max(1, overviewHost.clientHeight)
        overviewRenderer.setSize(width, height)
        overviewCamera.aspect = width / height
        overviewCamera.updateProjectionMatrix()
      }
    }
    resize()
    const observer = new ResizeObserver(resize)
    observer.observe(eyeHost)
    if (overviewHost) observer.observe(overviewHost)

    const ready = Promise.all([visual.ready, environmentReady]).then(() => true)
    ready.then(() => setStatus(''))
    window.__houseLab = {
      ready, house,
      render: (view) => {
        const previous = eyeRenderer.getSize(new THREE.Vector2())
        eyeRenderer.setPixelRatio(1)
        eyeRenderer.setSize(view.width ?? 640, view.height ?? 480, false)
        placeEye(view)
        draw(renderers[0], eyeCamera, true)
        const image = eyeRenderer.domElement.toDataURL('image/jpeg', 0.92)
        eyeRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5))
        eyeRenderer.setSize(previous.x, previous.y)
        return image
      },
    }

    sceneRef.current = scene
    setSceneReady((value) => value + 1)
    let frame = 0, last = performance.now(), notified = 0
    const yawOf = ([w, x, y, z]: number[]) => Math.atan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))
    const loop = () => {
      frame = requestAnimationFrame(loop)
      if (capture) return
      const now = performance.now(), dt = Math.min((now - last) / 1000, 0.1)
      last = now
      const agent = agentRef.current
      let eye: View = viewRef.current.eye
      if (agent.trace && agent.avatar) {
        const frames = agent.trace.frames, end = frames[frames.length - 1].time
        if (agent.playing) {
          agent.time = Math.min(end, agent.time + dt * agent.speed)
          if (agent.time >= end) agent.playing = false
        }
        let lo = 0, hi = frames.length - 1
        while (lo < hi) { const mid = Math.ceil((lo + hi) / 2); if (frames[mid].time <= agent.time) lo = mid; else hi = mid - 1 }
        const from = frames[lo], to = frames[Math.min(lo + 1, frames.length - 1)]
        const blend = to.time > from.time ? THREE.MathUtils.clamp((agent.time - from.time) / (to.time - from.time), 0, 1) : 0
        agent.avatar.update(from.bodies, to.bodies, blend)
        const pelvis = from.bodies.find((b) => b.name === 'pelvis'), head = from.bodies.find((b) => b.name === 'head')
        if (pelvis && head) {
          const target = toThree(pelvis.position[0], pelvis.position[1], 1.0)
          if (controls && overviewCamera) {
            if (!agent.follow) {   // start: look at the body from the south-east, above
              overviewCamera.position.copy(target).add(new THREE.Vector3(3.5, 6.5, 5.5))
              controls.target.copy(target)
            } else {
              const move = target.clone().sub(agent.follow)
              overviewCamera.position.add(move)
              controls.target.add(move)
            }
            agent.follow = target
          }
          eye = { x: head.position[0], y: head.position[1], z: head.position[2], heading: yawOf(pelvis.quaternion), pitch: EYE.downPitchDegrees }
        }
        if (now - notified > 120) { notified = now; setPlayTime(agent.time); setPlaying(agent.playing) }
      }
      controls?.update()
      if (overviewCamera && renderers[1]) draw(renderers[1], overviewCamera, false)
      placeEye(eye)
      draw(renderers[0], eyeCamera, true)
    }
    loop()
    return () => {
      cancelAnimationFrame(frame)
      observer.disconnect()
      controls?.dispose()
      visual.dispose()
      sceneRef.current = null
      for (const entry of renderers) { entry.environment?.dispose(); entry.pmrem.dispose(); entry.renderer.dispose(); entry.renderer.forceContextLoss(); entry.renderer.domElement.remove() }
      if (window.__houseLab?.house === house) delete window.__houseLab
    }
  }, [house, capture])

  const counts = useMemo(() => {
    const result = new Map<string, number>()
    for (const o of house?.objects ?? []) result.set(o.label.split('（')[0], (result.get(o.label.split('（')[0]) ?? 0) + 1)
    return [...result.entries()]
  }, [house])
  const area = house?.metadata.floorArea

  return (
    <main style={{ height: '100dvh', overflow: 'auto', background: '#0b1220', color: '#e6edf6', padding: '22px clamp(12px, 3vw, 36px)', boxSizing: 'border-box' }}>
      <style>{`.house-grid{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:18px}@media(max-width:900px){.house-grid{grid-template-columns:1fr}}.house-lab button,.house-lab select{font:inherit}`}</style>
      <div className="house-lab" style={{ maxWidth: 1440, margin: '0 auto' }}>
        <header style={{ display: 'flex', justifyContent: 'space-between', gap: 16, flexWrap: 'wrap', marginBottom: 18 }}>
          <div>
            <p style={{ color: '#91a5bc', margin: '0 0 6px', fontSize: 12, letterSpacing: 1 }}>学長AI · G7 見える家</p>
            <h1 style={{ fontSize: 'clamp(22px, 3vw, 28px)', margin: 0 }}>2階建ての家{area ? `（延床 ${area.total}㎡・約 ${area.tsubo} 坪）` : ''}</h1>
            <p style={{ color: '#a6b6c8', fontSize: 13, margin: '10px 0 0', lineHeight: 1.7 }}>1階のLDKは本番の家の部屋に、一般的な配置の家具を置いています。玄関・水回り・階段・2階の寝室と子ども部屋を足しました。<br />右は、学長の頭の位置から見た映像です（深度カメラと同じ位置・向き・視野）。</p>
          </div>
          <Link href="/world-model" style={{ color: '#c5dcf4', border: '1px solid #34475f', borderRadius: 8, padding: '9px 13px', textDecoration: 'none', fontSize: 13, alignSelf: 'flex-start' }}>確認画面へ戻る</Link>
        </header>
        {(error || status) && <p role="status" style={{ color: error ? '#f2b8a8' : '#b8cde1', fontSize: 13 }}>{error || status}</p>}
        <div className="house-grid">
          <section style={{ border: '1px solid #283b50', borderRadius: 12, overflow: 'hidden', background: '#101d2b', alignSelf: 'start' }}>
            <div ref={overviewRef} style={{ height: 'min(70vh, 680px)', minHeight: 380 }} />
            <div style={{ display: 'flex', gap: 8, padding: '12px 14px', borderTop: '1px solid #283b50', flexWrap: 'wrap', alignItems: 'center', fontSize: 12 }}>
              <span style={{ color: '#9fb3c6' }}>俯瞰（天井を外して表示）</span>
              {([['both', '1階と2階'], ['first', '1階だけ']] as const).map(([value, label]) => <button key={value} onClick={() => setFloors(value)} aria-pressed={floors === value} style={{ color: '#dbe8f5', border: '1px solid #415975', background: floors === value ? '#244e75' : '#1b3047', borderRadius: 7, padding: '7px 11px' }}>{label}</button>)}
              <span style={{ color: '#9fb3c6', marginLeft: 'auto' }}>ドラッグで回転・ホイールで拡大</span>
            </div>
          </section>
          <section style={{ border: '1px solid #283b50', borderRadius: 12, background: '#101d2b', padding: '14px 16px' }}>
            <strong style={{ fontSize: 14 }}>学長の視点</strong>
            <div ref={eyeRef} style={{ width: '100%', aspectRatio: '4 / 3', marginTop: 10, borderRadius: 8, overflow: 'hidden', background: '#0d1724' }} />
            <label style={{ display: 'flex', gap: 10, alignItems: 'center', marginTop: 12, fontSize: 12, color: '#b8cde1', flexWrap: 'wrap' }}>立つ場所
              <select value={place} onChange={(event) => setPlace(event.target.value)} style={{ background: '#152b42', color: '#e1edf9', border: '1px solid #435b76', borderRadius: 6, padding: '6px 9px' }}>
                {[1, 2].map((floor) => <optgroup key={floor} label={`${floor}階`}>{places.filter((item) => item.floor === floor).map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</optgroup>)}
              </select>
            </label>
            <label style={{ display: 'flex', gap: 10, alignItems: 'center', marginTop: 10, fontSize: 12, color: '#b8cde1' }}>向き
              <input type="range" min={0} max={359} value={heading} onChange={(event) => setHeading(Number(event.target.value))} style={{ flex: 1, accentColor: '#84bcf4' }} />
              <span style={{ width: 70, fontVariantNumeric: 'tabular-nums' }}>{['南', '東', '北', '西'][Math.round(heading / 90) % 4]}向き {heading}°</span>
            </label>
            <p style={{ color: '#9fb3c6', fontSize: 11, lineHeight: 1.7, margin: '8px 0 0' }}>目の高さ {EYE.height.toFixed(2)} m（床から）・{EYE.downPitchDegrees}°下向き・横の視野 {EYE.horizontalFovDegrees}°。0° は南（玄関側）、180° は北（庭側）です。</p>
            <details style={{ marginTop: 12, fontSize: 12 }} open>
              <summary style={{ cursor: 'pointer', color: '#c5dcf4' }}>部屋と置いてある物</summary>
              <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 8 }}>
                <tbody>{(house?.rooms ?? []).filter((room) => room.id !== 'stair-well').map((room) => <tr key={room.id} style={{ borderBottom: '1px solid #1f3348' }}>
                  <td style={{ padding: '4px 6px', color: '#97ceff' }}>{room.floor}階</td><td style={{ padding: '4px 6px' }}>{room.label}</td>
                  <td style={{ padding: '4px 6px', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{(room.area / 1.62).toFixed(1)}畳</td>
                </tr>)}</tbody>
              </table>
              <p style={{ color: '#b7c9d9', lineHeight: 1.8 }}>{counts.map(([label, count]) => `${label}${count > 1 ? `×${count}` : ''}`).join('・')}</p>
            </details>
          </section>
        </div>
        {!capture && runs.length > 0 && <AgentPanel runs={runs} runId={runId} setRunId={setRunId} run={run} trace={trace} time={playTime}
          playing={playing} speed={speed} setSpeed={setSpeed} seek={seekAgent} toggle={() => seekAgent(agentRef.current.time >= (trace?.frames.at(-1)?.time ?? 0) - 1e-3 ? 0 : agentRef.current.time, !agentRef.current.playing)} />}
      </div>
    </main>
  )
}

const smallButton: React.CSSProperties = { color: '#dbe8f5', border: '1px solid #415975', background: '#1b3047', borderRadius: 7, padding: '6px 10px', fontSize: 12, cursor: 'pointer' }
const card: React.CSSProperties = { background: '#111f30', border: '1px solid #283e55', borderRadius: 10, padding: '12px 13px' }

function AgentPanel({ runs, runId, setRunId, run, trace, time, playing, speed, setSpeed, seek, toggle }: {
  runs: AgentRunSummary[]; runId: string; setRunId: (id: string) => void; run: AgentRun | null; trace: AgentTrace | null; time: number
  playing: boolean; speed: number; setSpeed: (value: number) => void; seek: (time: number, play?: boolean) => void; toggle: () => void
}) {
  const looks = (run?.events ?? []).filter((e) => e.kind === 'look')
  const decisions = (run?.events ?? []).filter((e) => e.kind === 'decision' || e.kind === 'check')
  const end = trace?.frames.at(-1)?.time ?? 0
  const look = lastAt(looks, time) ?? looks[0]
  const decision = lastAt(decisions.filter((e) => e.kind === 'decision'), time)
  const result = (run?.result ?? {}) as Record<string, unknown>
  const base = `${AGENT_BASE}/${runId}`
  const contacts = Number(result.contactCount ?? 0)
  const reason = result.success ? '成功' : result.stoppedBy ? `止まった理由：${String(result.stoppedBy)}`
    : result.reached && contacts ? `着いたが途中で ${contacts} 回接触` : result.finished ? `VLMの判断で終了：${String(result.message ?? '')}` : '判断の回数の上限'
  return (
    <section aria-label="指示の実行" style={{ marginTop: 18, border: '1px solid #283b50', borderRadius: 12, background: '#101d2b', padding: '14px 16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10, alignItems: 'baseline' }}>
        <strong style={{ fontSize: 15 }}>指示の実行（見て判断して動く・1階）</strong>
        <span style={{ color: '#9fb3c6', fontSize: 11, lineHeight: 1.6 }}>止まるたびに頭のカメラで見て（カラー画像・深度・物体検出）、ローカルVLMが次の行動を選び、全身追従器が体を動かします。上の3Dと「学長の視点」も再生に合わせて動きます。</span>
      </div>
      <div role="group" aria-label="指示" style={{ display: 'flex', gap: 7, flexWrap: 'wrap', marginTop: 10 }}>
        {runs.map((r) => <button key={r.id} onClick={() => setRunId(r.id)} aria-pressed={runId === r.id} style={{ ...smallButton, background: runId === r.id ? '#244e75' : '#1b3047' }}>
          「{r.text}」<span style={{ color: r.success ? '#8fd6a0' : '#f0a59a', marginLeft: 6 }}>{r.success ? '成功' : '未達'}</span></button>)}
      </div>
      {!run || !trace ? <p style={{ color: '#b8cde1', fontSize: 12 }}>記録を読み込んでいます</p> : <>
        <p style={{ fontSize: 12.5, margin: '12px 0 8px', color: '#d0deea' }}>
          <strong>結果：</strong><span style={{ color: result.success ? '#8fd6a0' : '#f0a59a' }}>{reason}</span>
          <span style={{ color: '#9fb3c6' }}>　判断 {String(result.decisions ?? '—')} 回・見た回数 {String(result.looks ?? '—')}・{String(result.seconds ?? '—')} 秒（物理）・VLM {run.model?.vlm}・追従器 {run.model?.tracker}</span>
        </p>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
          <button style={{ ...smallButton, minWidth: 84 }} onClick={toggle}>{playing ? '一時停止' : '再生'}</button>
          <button style={smallButton} onClick={() => seek(0)}>最初へ</button>
          {[1, 2, 4].map((value) => <button key={value} style={{ ...smallButton, background: speed === value ? '#244e75' : '#1b3047' }} aria-pressed={speed === value} onClick={() => setSpeed(value)}>{value}倍</button>)}
          <input aria-label="再生位置" type="range" min={0} max={end} step={0.02} value={Math.min(time, end)} onChange={(event) => seek(Number(event.target.value))} style={{ flex: 1, minWidth: 160, accentColor: '#84bcf4' }} />
          <span style={{ fontSize: 12, color: '#b2c5d9', fontVariantNumeric: 'tabular-nums' }}>{time.toFixed(1)} / {end.toFixed(1)} 秒</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 14, marginTop: 12 }}>
          <figure style={{ ...card, margin: 0 }}>
            <figcaption style={{ fontSize: 12.5, marginBottom: 8 }}><strong>止まって見た画像</strong><span style={{ color: '#9fb3c6' }}>（{look?.time.toFixed(1)}秒・{look?.why}）</span></figcaption>
            {look?.boxes && <Image unoptimized src={`${base}/${look.boxes}`} alt="頭のカメラのカラー画像と物体検出の枠"
              width={640} height={480} style={{ width: '100%', height: 'auto', borderRadius: 6, display: 'block' }} />}
            <p style={{ fontSize: 11.5, color: '#b7c9d9', margin: '8px 0 0', lineHeight: 1.7 }}>物体検出（黄）：{(look?.detections ?? []).map((d) => `${COCO_JA[d.coco] ?? d.coco} ${d.conf.toFixed(2)}`).join('・') || 'なし'}</p>
            <p style={{ fontSize: 11.5, color: '#b7c9d9', margin: '4px 0 0', lineHeight: 1.7 }}>VLM が画像で示した物（緑）：{(look?.vlm?.items ?? []).map((d) => `${VLM_KIND[d.kind] ?? d.kind}「${d.what}」${d.id ? `→${d.id}` : ''}`).join('・') || 'なし'}{look?.vlm?.seconds ? `（${look.vlm.seconds} 秒）` : ''}</p>
            {look?.depth && <><p style={{ fontSize: 12, margin: '10px 0 6px' }}><strong>同じ瞬間の深度画像</strong><span style={{ color: '#9fb3c6' }}>（近い＝明るい・水色＝障害物）</span></p>
              <Image unoptimized src={`${base}/${look.depth}`} alt="深度画像" width={320} height={240}
                style={{ width: '62%', height: 'auto', borderRadius: 6, display: 'block', imageRendering: 'pixelated' }} /></>}
          </figure>
          <figure style={{ ...card, margin: 0 }}>
            <figcaption style={{ fontSize: 12.5, marginBottom: 8 }}><strong>深度から作った地図と記憶した物</strong></figcaption>
            {look?.map && <Image unoptimized src={`${base}/${look.map}`} alt="地図" width={480} height={560}
              style={{ width: '100%', height: 'auto', borderRadius: 6, display: 'block', background: '#fff' }} />}
            <p style={{ fontSize: 11, color: '#9fb3c6', margin: '8px 0 0', lineHeight: 1.6 }}>赤＝障害物（頭の深度カメラ3台：正面と左右60°、と認識した家具）・灰＝見て空いていると分かった床・青点＝物体検出で記憶した物・緑四角＝VLM が示した行き先・手がかり・出入口・◆＝まだ見ていない所の候補・橙＝歩いた軌跡</p>
          </figure>
          <div style={{ ...card, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <strong style={{ fontSize: 12.5 }}>VLM（実行係）の判断</strong>
            {decision ? <>
              <p style={{ fontSize: 13, margin: 0, lineHeight: 1.7 }}>{decision.answer?.thought}</p>
              <p style={{ fontSize: 13, margin: 0 }}><strong style={{ color: '#97ceff' }}>→ {ACTION_LABEL[decision.answer?.action ?? ''] ?? decision.answer?.action}{decision.answer?.target ? `（${decision.answer.target}）` : ''}</strong>{decision.answer?.message ? `　「${decision.answer.message}」` : ''}</p>
              {decision.override && <p style={{ fontSize: 12, margin: 0, color: '#ffd27a' }}>実行の規則で変更：{decision.override}</p>}
              <p style={{ fontSize: 11, color: '#9fb3c6', margin: 0 }}>{decision.time.toFixed(1)}秒・応答 {decision.seconds} 秒・画像 {decision.images?.length ?? 0} 枚</p>
              <details style={{ fontSize: 11.5, color: '#b7c9d9' }}><summary style={{ cursor: 'pointer' }}>VLM に渡した情報</summary>
                <p style={{ margin: '6px 0 0', lineHeight: 1.7 }}>見つけた物：{(decision.objects ?? []).join('、') || 'なし'}</p>
                <p style={{ margin: '4px 0 0', lineHeight: 1.7 }}>まだ見ていない所：{(decision.frontiers ?? []).join('、') || 'なし'}</p></details>
            </> : <p style={{ fontSize: 12, color: '#9fb3c6', margin: 0 }}>まだ判断していません（開始直後に見た画像で最初の判断をします）</p>}
            <strong style={{ fontSize: 12.5, marginTop: 6 }}>判断の流れ（押すとその時点へ）</strong>
            <ol style={{ margin: 0, paddingLeft: 18, fontSize: 12, lineHeight: 1.6 }}>
              {decisions.map((d, i) => <li key={i} style={{ marginBottom: 4 }}>
                <button onClick={() => seek(d.time)} style={{ background: 'none', border: 'none', color: decision === d ? '#ffd27a' : d.kind === 'check' ? '#9fe0b4' : '#c5dcf4', padding: 0, textAlign: 'left', cursor: 'pointer', font: 'inherit' }}>
                  {d.kind === 'check' ? <>{d.time.toFixed(1)}秒 到着の確認：{d.answer?.arrived ? '着いている' : '着いていない'}（{d.answer?.reason ?? ''}）
                    {d.headTurn && <><br /><Image unoptimized src={`${base}/${d.headTurn}`} alt="頭だけを行き先へ向けて撮った画像"
                      width={160} height={120} style={{ borderRadius: 4, marginTop: 4 }} /></>}</>
                    : <>{d.time.toFixed(1)}秒 {ACTION_LABEL[d.answer?.action ?? ''] ?? d.answer?.action}{d.answer?.target ? `（${d.answer.target}）` : ''}：{(d.answer?.thought ?? '').slice(0, 46)}{(d.answer?.thought ?? '').length > 46 ? '…' : ''}</>}
                </button></li>)}
            </ol>
          </div>
        </div>
      </>}
    </section>
  )
}
