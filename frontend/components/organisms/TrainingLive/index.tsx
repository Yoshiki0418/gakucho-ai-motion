'use client'

/**
 * 学習中の重みで動く学長の身体を、9 体同時に実時間で表示する(/world-model/live)。
 *
 * 物理計算はサーバ側(world_model/tracker/live_server.py、CPU の MuJoCo)で 50Hz で回し、
 * 25Hz で姿勢を受け取る。描画側は直近の 2 フレームの間を補間して滑らかに見せる(遅れは約 40ms)。
 */

import Link from 'next/link'
import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

type GeomSpec = { body: number; type: 'box' | 'capsule' | 'sphere'; size: number[]; pos: number[]; quat: number[] }
type Layout = { bodies: number; geoms: GeomSpec[]; seatHalf: number[]; slots: string[] }
type SlotFrame = {
  title: string
  text: string
  state: 'tracking' | 'failed' | 'done' | 'pause'
  seconds: number
  length: number
  errCm: number
  failedAt: number | null
  origin: [number, number]
  seat: number[] | null
  poses: number[]
}
type Frame = { t: number; step: number; run: string; slots: SlotFrame[] }

const COLUMNS = 3
const SPACING_X = 3.3
const SPACING_Y = 4.2
const STATE_LABEL = { tracking: '追従中', failed: '崩れた', done: '最後まで追従', pause: '準備中' } as const
const STATE_COLOR = { tracking: '#5fb3ff', failed: '#ff7a45', done: '#6fdc8c', pause: '#8aa0b8' } as const

// MuJoCo(Z 上・前は -Y)の座標をそのまま使えるよう、全体を X 軸まわりに -90 度回した中に置く
const tileCenter = (index: number) => {
  const column = index % COLUMNS
  const row = Math.floor(index / COLUMNS)
  return new THREE.Vector2((column - (COLUMNS - 1) / 2) * SPACING_X, (row - 1) * SPACING_Y)
}

export function TrainingLive() {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const labelsRef = useRef<HTMLDivElement | null>(null)
  const [frame, setFrame] = useState<Frame | null>(null)
  const [connected, setConnected] = useState(false)
  const [weightsSince, setWeightsSince] = useState<{ step: number; at: number } | null>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(container.clientWidth, container.clientHeight)
    renderer.shadowMap.enabled = true
    container.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color('#061220')
    scene.fog = new THREE.Fog('#061220', 16, 34)
    const camera = new THREE.PerspectiveCamera(42, container.clientWidth / container.clientHeight, 0.05, 80)
    camera.position.set(0, 9.2, 13.6)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.target.set(0, 0.4, 0.6)
    controls.enableDamping = true
    controls.maxPolarAngle = Math.PI * 0.48
    controls.update()

    scene.add(new THREE.HemisphereLight('#b9d7ff', '#0b1626', 1.1))
    const sun = new THREE.DirectionalLight('#ffffff', 1.6)
    sun.position.set(4, 9, 6)
    sun.castShadow = true
    sun.shadow.mapSize.set(2048, 2048)
    Object.assign(sun.shadow.camera, { left: -9, right: 9, top: 9, bottom: -9 })
    scene.add(sun)

    // 床: 暗い面に細い格子、体ごとの区画は縁だけ光らせる
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.MeshStandardMaterial({ color: '#0a1a2c', roughness: 0.9 }))
    floor.rotation.x = -Math.PI / 2
    floor.receiveShadow = true
    scene.add(floor)
    const grid = new THREE.GridHelper(40, 80, '#1d3d5f', '#132a44')
    grid.position.y = 0.002
    scene.add(grid)

    const world = new THREE.Group()
    world.rotation.x = -Math.PI / 2
    scene.add(world)

    const bodyMaterial = (color: string) => new THREE.MeshStandardMaterial({ color, metalness: 0.25, roughness: 0.45, emissive: color, emissiveIntensity: 0.08 })
    type Avatar = { root: THREE.Group; bodies: THREE.Object3D[]; materials: THREE.MeshStandardMaterial[]; seat: THREE.Mesh; ring: THREE.Mesh; state: string }
    let avatars: Avatar[] = []
    let layout: Layout | null = null
    let previous: { frame: Frame; at: number } | null = null
    let latest: { frame: Frame; at: number } | null = null

    const build = (next: Layout) => {
      for (const avatar of avatars) world.remove(avatar.root)
      avatars = next.slots.map((_, index) => {
        const root = new THREE.Group()
        const center = tileCenter(index)
        // 区画の目印(状態で色が変わる輪)
        const ring = new THREE.Mesh(new THREE.RingGeometry(0.62, 0.66, 64), new THREE.MeshBasicMaterial({ color: STATE_COLOR.pause, transparent: true, opacity: 0.85 }))
        ring.position.set(center.x, center.y, 0.004)
        world.add(ring)
        const materials: THREE.MeshStandardMaterial[] = []
        const bodies = Array.from({ length: next.bodies }, () => {
          const body = new THREE.Object3D()
          root.add(body)
          return body
        })
        for (const spec of next.geoms) {
          let geometry: THREE.BufferGeometry
          if (spec.type === 'capsule') {
            geometry = new THREE.CapsuleGeometry(spec.size[0], spec.size[1] * 2, 6, 14)
            geometry.rotateX(Math.PI / 2) // three は Y 軸、MuJoCo のカプセルは Z 軸
          } else if (spec.type === 'box') {
            geometry = new THREE.BoxGeometry(spec.size[0] * 2, spec.size[1] * 2, spec.size[2] * 2)
          } else {
            geometry = new THREE.SphereGeometry(spec.size[0], 18, 12)
          }
          const material = bodyMaterial('#dce7f3')
          materials.push(material)
          const mesh = new THREE.Mesh(geometry, material)
          mesh.position.set(spec.pos[0], spec.pos[1], spec.pos[2])
          mesh.quaternion.set(spec.quat[1], spec.quat[2], spec.quat[3], spec.quat[0])
          mesh.castShadow = true
          bodies[spec.body].add(mesh)
        }
        const seat = new THREE.Mesh(
          new THREE.BoxGeometry(next.seatHalf[0] * 2, next.seatHalf[1] * 2, next.seatHalf[2] * 2),
          new THREE.MeshStandardMaterial({ color: '#8c6b4c', roughness: 0.7 }),
        )
        seat.castShadow = true
        seat.receiveShadow = true
        seat.visible = false
        root.add(seat)
        world.add(root)
        return { root, bodies, materials, seat, ring, state: 'pause' }
      })
    }

    const quatA = new THREE.Quaternion()
    const quatB = new THREE.Quaternion()
    const place = () => {
      if (!latest || !avatars.length) return
      const now = performance.now()
      const span = previous ? Math.max(latest.at - previous.at, 1) : 1
      const alpha = previous ? Math.min(Math.max((now - latest.at) / span, 0), 1) : 1
      latest.frame.slots.forEach((slot, index) => {
        const avatar = avatars[index]
        if (!avatar) return
        const before = previous?.frame.slots[index]
        const sameClip = before && before.origin[0] === slot.origin[0] && before.origin[1] === slot.origin[1]
        const center = tileCenter(index)
        avatar.root.position.set(center.x - slot.origin[0], center.y - slot.origin[1], 0)
        for (let b = 0; b < avatar.bodies.length; b += 1) {
          const o = b * 7
          const p = slot.poses
          const body = avatar.bodies[b]
          if (sameClip && before) {
            const q = before.poses
            body.position.set(
              q[o] + (p[o] - q[o]) * alpha, q[o + 1] + (p[o + 1] - q[o + 1]) * alpha, q[o + 2] + (p[o + 2] - q[o + 2]) * alpha)
            quatA.set(q[o + 4], q[o + 5], q[o + 6], q[o + 3])
            quatB.set(p[o + 4], p[o + 5], p[o + 6], p[o + 3])
            body.quaternion.slerpQuaternions(quatA, quatB, alpha)
          } else {
            body.position.set(p[o], p[o + 1], p[o + 2])
            body.quaternion.set(p[o + 4], p[o + 5], p[o + 6], p[o + 3])
          }
        }
        avatar.seat.visible = !!slot.seat
        if (slot.seat) avatar.seat.position.set(slot.seat[0], slot.seat[1], slot.seat[2])
        if (avatar.state !== slot.state) {
          avatar.state = slot.state
          const color = slot.state === 'failed' ? '#ffb08a' : slot.state === 'done' ? '#cfeedd' : '#dce7f3'
          for (const material of avatar.materials) {
            material.color.set(color)
            material.emissive.set(slot.state === 'failed' ? '#ff5a1f' : color)
            material.emissiveIntensity = slot.state === 'failed' ? 0.35 : 0.08
          }
          ;(avatar.ring.material as THREE.MeshBasicMaterial).color.set(STATE_COLOR[slot.state] ?? STATE_COLOR.pause)
        }
      })
    }

    // 区画ごとの文字(画面上の位置へ投影)
    const projected = new THREE.Vector3()
    const updateLabels = () => {
      const labels = labelsRef.current
      if (!labels || !latest) return
      const rect = renderer.domElement.getBoundingClientRect()
      latest.frame.slots.forEach((slot, index) => {
        const element = labels.children[index] as HTMLElement | undefined
        if (!element) return
        const center = tileCenter(index)
        // 区画の手前(MuJoCo の -y = カメラ側)に置く。world は X 軸 -90 度回転(MuJoCo y → three -z)
        projected.set(center.x, 0, -(center.y - 1.15)).project(camera)
        const visible = projected.z < 1
        element.style.display = visible ? 'block' : 'none'
        element.style.transform = `translate(-50%, 0) translate(${((projected.x + 1) / 2) * rect.width}px, ${((1 - projected.y) / 2) * rect.height}px)`
      })
    }

    let frameHandle = 0
    const animate = () => {
      frameHandle = requestAnimationFrame(animate)
      place()
      controls.update()
      renderer.render(scene, camera)
      updateLabels()
    }
    animate()

    const events = new EventSource('/training-live/stream')
    events.addEventListener('open', () => setConnected(true))
    events.addEventListener('error', () => setConnected(false))
    events.addEventListener('layout', (event) => {
      layout = JSON.parse((event as MessageEvent).data) as Layout
      build(layout)
    })
    let lastState = 0
    events.addEventListener('frame', (event) => {
      const next = JSON.parse((event as MessageEvent).data) as Frame
      previous = latest
      latest = { frame: next, at: performance.now() }
      setConnected(true)
      // 文字の更新は 4Hz で十分
      if (performance.now() - lastState > 250) {
        lastState = performance.now()
        setFrame(next)
        setWeightsSince((current) => (current && current.step === next.step ? current : { step: next.step, at: Date.now() }))
      }
    })

    const resize = () => {
      camera.aspect = container.clientWidth / container.clientHeight
      camera.updateProjectionMatrix()
      renderer.setSize(container.clientWidth, container.clientHeight)
    }
    const observer = new ResizeObserver(resize)
    observer.observe(container)
    return () => {
      events.close()
      cancelAnimationFrame(frameHandle)
      observer.disconnect()
      controls.dispose()
      renderer.dispose()
      container.removeChild(renderer.domElement)
    }
  }, [])

  const tracking = frame?.slots.filter((slot) => slot.state === 'tracking').length ?? 0
  return (
    <main className="training-live" style={{ height: '100dvh', display: 'flex', flexDirection: 'column', background: '#061220', color: '#e1edf9' }}>
      <style>{`.training-live :is(header, div, span, h1, p, strong) { color: inherit; }`}</style>
      <header style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', padding: '14px clamp(12px, 3vw, 28px)', borderBottom: '1px solid #16304d' }}>
        <div style={{ minWidth: 0 }}>
          <p style={{ color: '#91a5bc', margin: '0 0 4px', fontSize: 12, letterSpacing: 1 }}>学長AI · 物理と学習の検証 · ライブ</p>
          <h1 style={{ fontSize: 'clamp(18px, 2.4vw, 24px)', margin: 0 }}>学習中の重みで、9 体を実時間で動かす</h1>
          <p style={{ color: '#a6b6c8', fontSize: 12, margin: '6px 0 0', lineHeight: 1.6 }}>
            {frame ? `${frame.run}・${(frame.step / 1e8).toFixed(2)}億ステップ時点の重み` : '接続しています…'}
            {weightsSince ? `(この重みで ${Math.max(0, Math.round((Date.now() - weightsSince.at) / 60000))} 分)` : ''}
            ・学習に使っていない動作を、物理シミュレーション上で追従・{connected ? `${tracking} 体が追従中` : '未接続'}
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start', flexWrap: 'wrap' }}>
          <Link href="/world-model/training" style={{ color: '#c5dcf4', border: '1px solid #34475f', borderRadius: 8, padding: '8px 12px', textDecoration: 'none', fontSize: 13 }}>学習の様子</Link>
          <Link href="/world-model" style={{ color: '#c5dcf4', border: '1px solid #34475f', borderRadius: 8, padding: '8px 12px', textDecoration: 'none', fontSize: 13 }}>物理の検証記録</Link>
        </div>
      </header>
      <div style={{ position: 'relative', flex: 1, minHeight: 0 }}>
        <div ref={containerRef} style={{ position: 'absolute', inset: 0 }} />
        <div ref={labelsRef} style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
          {Array.from({ length: 9 }, (_, index) => {
            const slot = frame?.slots[index]
            return (
              <div key={index} style={{ position: 'absolute', left: 0, top: 0, fontSize: 11, lineHeight: 1.4, textAlign: 'center',
                background: 'rgba(6, 18, 32, 0.72)', border: '1px solid #1d3d5f', borderRadius: 6, padding: '3px 7px', whiteSpace: 'nowrap',
                display: slot ? 'block' : 'none' }}>
                {slot && (
                  <>
                    <strong style={{ fontSize: 12 }}>{slot.title}</strong>
                    <div>
                      <span style={{ color: STATE_COLOR[slot.state] ?? STATE_COLOR.pause, marginRight: 4 }}>●</span>
                      {STATE_LABEL[slot.state] ?? slot.state} {slot.seconds.toFixed(1)}/{slot.length.toFixed(1)}秒 ずれ{slot.errCm.toFixed(0)}cm
                    </div>
                  </>
                )}
              </div>
            )
          })}
        </div>
        <div style={{ position: 'absolute', right: 12, bottom: 12, fontSize: 11, color: '#a6b6c8', background: 'rgba(6, 18, 32, 0.72)', border: '1px solid #1d3d5f', borderRadius: 8, padding: '6px 10px', lineHeight: 1.6 }}>
          {(['tracking', 'done', 'failed'] as const).map((state) => (
            <div key={state}><span style={{ color: STATE_COLOR[state], marginRight: 6 }}>●</span>{STATE_LABEL[state]}</div>
          ))}
          <div style={{ color: '#8aa0b8' }}>ドラッグで視点を回せます</div>
        </div>
      </div>
    </main>
  )
}
