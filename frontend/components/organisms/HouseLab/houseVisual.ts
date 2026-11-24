'use client'

import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'

/**
 * The visible two-storey house of G7 (world_model/house2f.py writes /world-model/house2f.json).
 * The 1F LDK keeps the production home's room (garden window wall, curtains, garden, ceilings, as in
 * BvhViewer/homeEnvironment.ts) with its own typical furniture; walls, floors, ceilings, the staircase,
 * windows, lights and every listed object are built here in the production materials (Poly Haven CC0
 * wood, plaster, stone and fabric). Physics boxes are in MuJoCo
 * coordinates (Z up, metres); Three.js is Y up: (x, y, z) -> (x, z, -y), yaw about Z -> about Y.
 * Objects: size = [width, depth, height] in their own frame, the front faces local -Y (MuJoCo),
 * i.e. local +Z in Three.js.
 */
export type HouseGeom = { id: string; position: number[]; size: number[]; yaw?: number; semantic: string; entityId?: string }
export type HouseObject = {
  id: string; class: string; label: string; room: string; floor: number
  position: number[]; size: number[]; yaw: number; coco: string | null; collision: boolean; source?: string; variant?: string
}
export type HouseRoom = { id: string; label: string; floor: number; rects: number[][]; area: number }
export type HouseGoal = { id: string; position: number[]; kind: string; room?: string }
export type HouseDescription = {
  name: string; geoms: HouseGeom[]; objects: HouseObject[]; rooms: HouseRoom[]; goals: HouseGoal[]; spawn: number[]
  metadata: { floorHeights: number[]; ceilingHeights: number[]; floorArea: Record<string, number>
    stair: { firstRiserX: number; topX: number; y: number[]; rise: number; risers: number; tread: number } }
}
export type HouseVisual = {
  root: THREE.Group
  /** Parts above the 1F ceiling (2F rooms, their ceilings): hidden to look into 1F from above. */
  upper: THREE.Group
  /** Ceilings and roofs: hidden in the cutaway overview, shown for the head camera. */
  ceilings: THREE.Group
  ready: Promise<void>
  dispose: () => void
}

export const toThree = (x: number, y: number, z: number) => new THREE.Vector3(x, z, -y)

export const createHouseVisual = (house: HouseDescription, assetBase = '/environments/home'): HouseVisual => {
  const root = new THREE.Group()
  root.name = 'TwoStoreyHouse'
  const upper = new THREE.Group()
  upper.name = 'SecondFloor'
  const ceilings = new THREE.Group()
  ceilings.name = 'Ceilings'
  root.add(upper, ceilings)
  const geometries = new Set<THREE.BufferGeometry>()
  const materials = new Set<THREE.Material>()
  const textures = new Set<THREE.Texture>()
  let disposed = false
  const [, floor2] = house.metadata.floorHeights
  const [ceiling1, ceiling2] = house.metadata.ceilingHeights

  // ---------------------------------------------------------------- materials
  const material = (name: string, props: THREE.MeshStandardMaterialParameters) => {
    const value = new THREE.MeshStandardMaterial({ envMapIntensity: 0.65, ...props })
    value.name = name
    materials.add(value)
    return value
  }
  const plaster = material('HousePlaster', { color: 0xe9e4da, roughness: 0.92, side: THREE.DoubleSide })
  const ceilingPlaster = material('HouseCeiling', { color: 0xf3f0ea, roughness: 0.95, side: THREE.DoubleSide })
  const oakFloor = material('HouseOakFloor', { color: 0xffffff, roughness: 1 })
  const oak = material('HouseOak', { color: 0xc7a67b, roughness: 0.6 })
  const darkOak = material('HouseDarkOak', { color: 0x6d4a2e, roughness: 0.52 })
  const whiteWood = material('WhiteJoinery', { color: 0xf1eee8, roughness: 0.55 })
  const tile = material('LightTile', { color: 0xdedbd3, roughness: 0.42 })
  const darkTile = material('DomaStone', { color: 0x77736c, roughness: 0.8 })
  const bathTile = material('BathTile', { color: 0xc9d0d0, roughness: 0.35 })
  const ceramic = material('WhiteCeramic', { color: 0xf7f7f4, roughness: 0.18, metalness: 0.02 })
  const chrome = material('Chrome', { color: 0xd0d4d6, metalness: 0.95, roughness: 0.12 })
  const steel = material('Steel', { color: 0x9aa0a2, metalness: 0.8, roughness: 0.35 })
  const black = material('SoftBlack', { color: 0x1a1d1f, roughness: 0.45 })
  const grey = material('PlasticGrey', { color: 0x8d9296, roughness: 0.5 })
  const linenWhite = material('BedLinen', { color: 0xf2efe8, roughness: 0.95 })
  const duvetBlue = material('DuvetBlue', { color: 0x6f8aa6, roughness: 0.95 })
  const duvetSage = material('DuvetSage', { color: 0x93a58a, roughness: 0.95 })
  const duvetCoral = material('DuvetCoral', { color: 0xd9937c, roughness: 0.95 })
  const terracotta = material('Terracotta', { color: 0x9d6e54, roughness: 0.88 })
  const leaf = material('Leaf', { color: 0x4d6b45, roughness: 0.9, side: THREE.DoubleSide })
  const leafLight = material('LeafLight', { color: 0x6f8a5b, roughness: 0.92, side: THREE.DoubleSide })
  const water = material('BathWater', { color: 0x9fc6cf, roughness: 0.05, transparent: true, opacity: 0.72 })
  const glassDark = material('DoorGlass', { color: 0x2a3438, roughness: 0.1, metalness: 0.3, transparent: true, opacity: 0.85 })
  const mirror = material('Mirror', { color: 0xe8eef0, metalness: 1, roughness: 0.04 })
  const warmGlow = material('LampGlow', { color: 0xfff4dc, emissive: 0xffdca0, emissiveIntensity: 1.6, roughness: 0.7 })
  const screenOn = material('ScreenOn', { color: 0x203a4a, emissive: 0x3a6a86, emissiveIntensity: 0.6, roughness: 0.2 })
  const red = material('RandoseruRed', { color: 0xa3262a, roughness: 0.45 })
  const brownFur = material('TeddyFur', { color: 0x8a5a34, roughness: 1 })
  const brownLight = material('TeddyMuzzle', { color: 0xc89a6c, roughness: 1 })
  const navy = material('UmbrellaNavy', { color: 0x24324a, roughness: 0.7 })
  const shoeLeather = material('ShoeLeather', { color: 0x3b2a20, roughness: 0.5 })
  const sneaker = material('Sneaker', { color: 0xe7e7e7, roughness: 0.7 })
  const garments = [0x2e3e57, 0xc8b8a0, 0x7a2f35, 0xf0eee8, 0x58704f, 0x30302e, 0xb7c4d4].map((color, i) =>
    material(`Garment${i}`, { color, roughness: 0.95 }))
  const bookColours = [0xa49b86, 0x596d73, 0x805e50, 0xcbc2ad, 0x3e5a7a, 0xb4573f, 0x6d7f4a].map((color, i) =>
    material(`Book${i}`, { color, roughness: 0.9 }))

  const canvasTexture = (width: number, height: number, draw: (context: CanvasRenderingContext2D) => void) => {
    if (typeof document === 'undefined') return null
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    const context = canvas.getContext('2d')
    if (!context) return null
    draw(context)
    const texture = new THREE.CanvasTexture(canvas)
    texture.colorSpace = THREE.SRGBColorSpace
    textures.add(texture)
    return texture
  }
  const clockFace = material('ClockFace', { color: 0xffffff, roughness: 0.6 })
  const clockTexture = canvasTexture(256, 256, (c) => {
    c.fillStyle = '#fbfaf6'; c.fillRect(0, 0, 256, 256)
    c.fillStyle = '#1d1d1d'; c.font = 'bold 30px sans-serif'; c.textAlign = 'center'; c.textBaseline = 'middle'
    for (let h = 1; h <= 12; h++) {
      const a = (h / 12) * Math.PI * 2
      c.fillText(String(h), 128 + Math.sin(a) * 95, 128 - Math.cos(a) * 95)
    }
    c.lineCap = 'round'; c.strokeStyle = '#1d1d1d'
    c.lineWidth = 8; c.beginPath(); c.moveTo(128, 128); c.lineTo(128 + 45, 128 - 30); c.stroke()
    c.lineWidth = 5; c.beginPath(); c.moveTo(128, 128); c.lineTo(128 - 10, 128 - 80); c.stroke()
  })
  if (clockTexture) clockFace.map = clockTexture
  const ballMaterial = material('SoccerBall', { color: 0xffffff, roughness: 0.55 })
  const ballTexture = canvasTexture(512, 256, (c) => {
    c.fillStyle = '#f5f5f2'; c.fillRect(0, 0, 512, 256)
    c.fillStyle = '#151515'
    for (let row = 0; row < 4; row++) for (let col = 0; col < 8; col++) {
      const x = col * 64 + (row % 2) * 32 + 16, y = row * 64 + 32
      c.beginPath()
      for (let k = 0; k < 5; k++) {
        const a = k / 5 * Math.PI * 2 - Math.PI / 2
        c.lineTo(x + Math.cos(a) * 15, y + Math.sin(a) * 15)
      }
      c.fill()
    }
  })
  if (ballTexture) ballMaterial.map = ballTexture
  const windowView = material('WindowView', { color: 0xc9d8e2, emissive: 0xffffff, emissiveIntensity: 0.55, roughness: 0.3 })
  // LDK furniture
  const counterStone = material('CounterStone', { color: 0xe6e3dc, roughness: 0.35 })
  const fridgeFinish = material('FridgeFinish', { color: 0xdfe2e3, metalness: 0.55, roughness: 0.3 })
  const blackGlass = material('CeramicGlassTop', { color: 0x121416, metalness: 0.2, roughness: 0.08 })
  const appliance = material('ApplianceWhite', { color: 0xf0f0ee, roughness: 0.35 })
  const sofaFabric = material('SofaFabric', { color: 0xb9b2a5, roughness: 0.96 })
  const cushionFabric = material('CushionFabric', { color: 0xcfc1aa, roughness: 0.97 })
  const accentFabric = material('AccentFabric', { color: 0x98715d, roughness: 0.96 })
  const armchairFabric = material('ArmchairFabric', { color: 0x6f7f8c, roughness: 0.96 })
  const rugFabric = material('RugFabric', { color: 0xcabfae, roughness: 1 })
  const rugBorder = material('RugBorder', { color: 0x8a7d6b, roughness: 1 })
  const zabutonFabric = material('ZabutonFabric', { color: 0x3d4b6b, roughness: 0.95 })
  const basinShade = material('SinkBasin', { color: 0x7d8487, metalness: 0.85, roughness: 0.32 })
  const tvScreen = material('TvScreen', { color: 0x172b33, emissive: 0x28434d, emissiveIntensity: 0.35, roughness: 0.18, metalness: 0.12 })
  const tatamiMaterial = material('Tatami', { color: 0xffffff, roughness: 0.95 })
  const tatamiTexture = canvasTexture(1024, 1024, (c) => {
    // 4.5 mats: four full mats round a centre half mat (2.7 m square), rush weave and cloth borders
    const s = 1024 / 2.7, mats: [number, number, number, number][] = [
      [0, 0, 1.8, 0.9], [1.8, 0, 0.9, 1.8], [0.9, 1.8, 1.8, 0.9], [0, 0.9, 0.9, 1.8], [0.9, 0.9, 0.9, 0.9]]
    for (const [x, y, w, h] of mats) {
      c.fillStyle = '#c3bd84'; c.fillRect(x * s, y * s, w * s, h * s)
      c.strokeStyle = 'rgba(120,112,60,0.25)'; c.lineWidth = 1
      const along = w > h
      for (let k = 0; k < (along ? h : w) * s; k += 4) {
        c.beginPath()
        if (along) { c.moveTo(x * s, y * s + k); c.lineTo((x + w) * s, y * s + k) } else { c.moveTo(x * s + k, y * s); c.lineTo(x * s + k, (y + h) * s) }
        c.stroke()
      }
      c.fillStyle = '#2f3a2e'
      if (along) { c.fillRect(x * s, y * s, w * s, 9); c.fillRect(x * s, (y + h) * s - 9, w * s, 9) } else { c.fillRect(x * s, y * s, 9, h * s); c.fillRect((x + w) * s - 9, y * s, 9, h * s) }
    }
  })
  if (tatamiTexture) tatamiMaterial.map = tatamiTexture

  // ---------------------------------------------------------------- primitives
  const mesh = (name: string, geometry: THREE.BufferGeometry, surface: THREE.Material, parent: THREE.Object3D,
    position: [number, number, number] = [0, 0, 0]) => {
    geometries.add(geometry)
    const result = new THREE.Mesh(geometry, surface)
    result.name = name
    result.position.set(...position)
    result.castShadow = true
    result.receiveShadow = true
    parent.add(result)
    return result
  }
  /** Box in Three.js local coordinates (width x, height y, depth z), centred at ``position``. */
  const box = (name: string, size: [number, number, number], position: [number, number, number], surface: THREE.Material,
    parent: THREE.Object3D, radius = 0.01) => mesh(name, radius > 0
    ? new RoundedBoxGeometry(...size, 2, Math.min(radius, ...size.map((v) => v / 2.01)))
    : new THREE.BoxGeometry(...size), surface, parent, position)
  const cylinder = (name: string, top: number, bottom: number, height: number, position: [number, number, number],
    surface: THREE.Material, parent: THREE.Object3D, segments = 24) =>
    mesh(name, new THREE.CylinderGeometry(top, bottom, height, segments), surface, parent, position)
  const sphere = (name: string, scale: [number, number, number], position: [number, number, number],
    surface: THREE.Material, parent: THREE.Object3D) => {
    const result = mesh(name, new THREE.SphereGeometry(1, 20, 14), surface, parent, position)
    result.scale.set(...scale)
    return result
  }
  const lathe = (name: string, profile: [number, number][], position: [number, number, number], surface: THREE.Material,
    parent: THREE.Object3D) => mesh(name, new THREE.LatheGeometry(profile.map(([x, y]) => new THREE.Vector2(x, y)), 32),
    surface, parent, position)
  /** World-axis box from a physics geom (MuJoCo centre and size, yaw about Z). */
  const geomBox = (g: HouseGeom, surface: THREE.Material, parent: THREE.Object3D, radius = 0) => {
    const [w, d, h] = g.size
    const result = box(g.id, [w, h, d], [0, 0, 0], surface, parent, radius)
    result.position.copy(toThree(g.position[0], g.position[1], g.position[2]))
    result.rotation.y = g.yaw ?? 0
    return result
  }
  /** Horizontal plane (floor facing up, ceiling facing down) over a MuJoCo rectangle at height z. */
  const plane = (name: string, rect: number[], z: number, surface: THREE.Material, parent: THREE.Object3D, down = false, tile = 0) => {
    const [x0, y0, x1, y1] = rect
    const geometry = new THREE.PlaneGeometry(x1 - x0, y1 - y0)
    if (tile > 0) {   // texture repeats every ``tile`` metres whatever the plane's size
      const uv = geometry.attributes.uv
      for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * (x1 - x0) / tile, uv.getY(i) * (y1 - y0) / tile)
    }
    const result = mesh(name, geometry, surface, parent)
    result.rotation.x = down ? Math.PI / 2 : -Math.PI / 2
    result.position.copy(toThree((x0 + x1) / 2, (y0 + y1) / 2, z))
    result.castShadow = false
    return result
  }
  /** A group placed at an object's centre-bottom: local x = width, y = up, +z = its front. */
  const placed = (o: HouseObject, parent: THREE.Object3D) => {
    const group = new THREE.Group()
    group.name = o.id
    group.position.copy(toThree(o.position[0], o.position[1], o.position[2] - o.size[2] / 2))
    group.rotation.y = o.yaw
    group.userData = { objectClass: o.class, label: o.label, room: o.room }
    parent.add(group)
    return group
  }

  // ---------------------------------------------------------------- shell
  const levelOf = (z: number) => (z >= floor2 - 0.05 ? upper : root)
  for (const g of house.geoms) {
    if (g.id === 'home-floor' || g.id === 'home-wall-back') continue   // the garden window wall is built below
    if (g.semantic === 'wall') geomBox(g, plaster, levelOf(g.position[2] - g.size[2] / 2))
    else if (g.semantic === 'railing') {
      // Handrail on balusters, open to the stair well.
      const [w, d, h] = g.size
      const rail = new THREE.Group()
      rail.position.copy(toThree(g.position[0], g.position[1], g.position[2] - h / 2))
      rail.rotation.y = g.yaw ?? 0
      upper.add(rail)
      const along = w > d ? w : d
      const across = (v: number) => (w > d ? [v, 0, 0] : [0, 0, v]) as [number, number, number]
      box(`${g.id}-handrail`, w > d ? [along, 0.05, 0.07] : [0.07, 0.05, along], [0, h - 0.025, 0], darkOak, rail, 0.015)
      for (let s = -along / 2 + 0.06; s <= along / 2 - 0.05; s += 0.12) {
        const position = across(s)
        box(`${g.id}-baluster`, [0.025, h - 0.05, 0.025], [position[0], (h - 0.05) / 2, position[2]], whiteWood, rail, 0)
      }
    } else if (g.semantic === 'floor' && g.id.startsWith('floor-2f')) {
      // Slab sides and underside; the 2F boards are separate planes above it.
      geomBox(g, ceilingPlaster, root)
    } else if (g.semantic === 'stair') {
      const tread = geomBox(g, plaster, root)
      tread.castShadow = false
      const [w, d, h] = g.size
      const board = box(`${g.id}-board`, [w + 0.02, 0.03, d], [0, 0, 0], oak, root, 0.004)
      board.position.copy(toThree(g.position[0] - 0.01, g.position[1], g.position[2] + h / 2 - 0.012))
    }
  }

  // Floors by room: wood in living rooms, tiles in wet rooms, stone in the doma.
  const floorSurface: Record<string, THREE.Material> = {
    'toilet-1f': tile, 'toilet-2f': tile, washroom: tile, bathroom: bathTile, entrance: darkTile,
  }
  for (const room of house.rooms) {
    if (room.id === 'stairs' || room.id === 'stair-well') continue
    const z = (room.floor === 1 ? 0 : floor2) + 0.002 - (room.id === 'entrance' ? 0.12 : 0)
    const surface = floorSurface[room.id] ?? oakFloor
    room.rects.forEach((rect, i) => plane(`floor-${room.id}-${i}`, rect, z, surface,
      room.floor === 1 ? root : upper, false, surface === oakFloor ? 1.7 : 0.9))
  }
  // The raised hall edge above the doma (agarikamachi) in dark oak.
  const entrance = house.rooms.find((room) => room.id === 'entrance')
  if (entrance) {
    const [x0, , x1, y1] = entrance.rects[0]
    box('agarikamachi', [x1 - x0, 0.12, 0.06], [0, 0, 0], darkOak, root, 0.006)
      .position.copy(toThree((x0 + x1) / 2, y1 + 0.03, -0.06))
  }
  // Ceilings: 1F south block (under the slab) and 2F, all hidden in the cutaway overview.
  for (const room of house.rooms) {
    if (room.id === 'ldk') continue
    for (const [i, rect] of room.rects.entries()) {
      if (room.floor === 1 && room.id !== 'stairs') plane(`ceiling-${room.id}-${i}`, rect, ceiling1 - 0.002, ceilingPlaster, ceilings, true)
      if (room.floor === 2) plane(`ceiling-${room.id}-${i}`, rect, ceiling2, ceilingPlaster, ceilings, true)
    }
  }
  // A plain roof over the 2F and the single-storey part of the LDK (only seen from outside).
  const roof = material('Roof', { color: 0x4b4f52, roughness: 0.8, side: THREE.DoubleSide })
  plane('roof-2f', [-5.15, -7.95, 5.15, -2.85], ceiling2 + 0.25, roof, ceilings)
  plane('roof-ldk', [-5.15, -2.85, 5.15, 5.15], ceiling1 + 0.25, roof, ceilings)

  // Windows: framed openings on the inner face of the outer walls, showing the garden photograph.
  // ``wall`` names the outer wall; the window's local +z points into the room.
  const inward = { south: Math.PI, north: 0, west: Math.PI / 2 }   // south facade y=-7.8, 2F north wall y=-3.0, west wall x=-5
  const windows: { at: [number, number, number]; width: number; height: number; wall: keyof typeof inward; level: THREE.Group }[] = [
    { at: [-3.0, -3.06, floor2 + 1.5], width: 1.4, height: 1.1, wall: 'north', level: upper },
    { at: [0.5, -3.06, floor2 + 1.5], width: 1.2, height: 1.1, wall: 'north', level: upper },
    { at: [3.5, -3.06, floor2 + 1.5], width: 1.2, height: 1.1, wall: 'north', level: upper },
    { at: [-1.55, -7.74, floor2 + 1.55], width: 1.3, height: 0.9, wall: 'south', level: upper },
    { at: [4.575, -7.74, floor2 + 1.85], width: 0.5, height: 0.45, wall: 'south', level: upper },
    { at: [1.3, -7.74, floor2 + 1.3], width: 1.2, height: 1.0, wall: 'south', level: upper },
    { at: [1.3, -7.74, 1.9], width: 1.0, height: 0.9, wall: 'south', level: root },
    { at: [-2.4, -7.74, 1.75], width: 0.7, height: 0.5, wall: 'south', level: root },
    { at: [-4.94, -6.95, 1.75], width: 0.7, height: 0.6, wall: 'west', level: root },
    { at: [-4.94, -5.55, 1.8], width: 0.5, height: 0.45, wall: 'west', level: root },
  ]
  for (const [i, w] of windows.entries()) {
    const group = new THREE.Group()
    group.position.copy(toThree(...w.at))
    group.rotation.y = inward[w.wall]
    w.level.add(group)
    box(`window-${i}-view`, [w.width, w.height, 0.01], [0, 0, 0], windowView, group, 0)
    for (const [sx, sy, wx, hy] of [[0, w.height / 2, w.width + 0.08, 0.05], [0, -w.height / 2, w.width + 0.08, 0.05],
      [w.width / 2, 0, 0.05, w.height], [-w.width / 2, 0, 0.05, w.height], [0, 0, 0.03, w.height]] as const) {
      box(`window-${i}-frame`, [wx, hy, 0.05], [sx, sy, 0.02], whiteWood, group, 0.005)
    }
    box(`window-${i}-sill`, [w.width + 0.12, 0.03, 0.12], [0, -w.height / 2 - 0.04, 0.05], whiteWood, group, 0.005)
  }

  // Front door leaf, open against the hall wall, and the porch light.
  const door = new THREE.Group()
  door.position.copy(toThree(4.5, -7.78, -0.12))
  door.rotation.y = -Math.PI * 0.42
  root.add(door)
  box('front-door', [0.9, 2.12, 0.05], [-0.45, 1.06, 0], darkOak, door, 0.01)
  box('front-door-glass', [0.16, 1.4, 0.055], [-0.75, 1.15, 0], glassDark, door, 0.004)
  box('front-door-handle', [0.03, 0.5, 0.05], [-0.12, 1.05, 0.05], steel, door, 0.01)

  // Ceiling lights: a flush disc in every room; a point light only in the larger rooms (every
  // light costs in each shader, and the HDR environment already lights the small rooms).
  const lights = new THREE.Group()
  lights.name = 'HouseLights'
  root.add(lights)
  for (const room of house.rooms) {
    if (room.id === 'ldk' || room.id === 'stairs' || room.id === 'stair-well') continue
    const rect = room.rects.reduce((a, b) => ((b[2] - b[0]) * (b[3] - b[1]) > (a[2] - a[0]) * (a[3] - a[1]) ? b : a))
    const x = (rect[0] + rect[2]) / 2, y = (rect[1] + rect[3]) / 2
    const top = room.floor === 1 ? ceiling1 : ceiling2
    const disc = cylinder(`light-${room.id}`, 0.2, 0.22, 0.04, [0, 0, 0], warmGlow, room.floor === 1 ? root : upper)
    disc.position.copy(toThree(x, y, top - 0.02))
    disc.castShadow = false
    if (room.area < 4.5 || room.id === 'ldk') continue
    const light = new THREE.PointLight(0xffe6c4, 4, 7, 2)
    light.position.copy(toThree(x, y, top - 0.25))
    lights.add(light)
  }

  // ---------------------------------------------------------------- LDK room (as the production home)
  // In Three.js coordinates like BvhViewer/homeEnvironment.ts: the garden window wall at z = -5 (MuJoCo y = +5).
  const glass = new THREE.MeshPhysicalMaterial({ color: 0xd5e4df, transparent: true, opacity: 0.22, transmission: 0.7,
    thickness: 0.035, roughness: 0.06, metalness: 0, ior: 1.45, envMapIntensity: 0.65, side: THREE.DoubleSide })
  materials.add(glass)
  const curtainMaterial = material('SheerCurtains', { color: 0xf5f0e4, roughness: 1, side: THREE.DoubleSide })
  const woodCeiling = material('OakCeiling', { color: 0xe4c5a1, roughness: 0.72, side: THREE.DoubleSide })
  box('ldk-wall-north-kitchen', [3.25, 2.8, 0.1], [-3.425, 1.4, -5], plaster, root, 0)
  box('ldk-wall-north-pier', [0.45, 2.8, 0.1], [4.825, 1.4, -5], plaster, root, 0)
  box('ldk-window-header', [6.4, 0.18, 0.1], [1.4, 2.71, -5], plaster, root, 0)
  box('ldk-window-sill', [6.45, 0.1, 0.28], [1.4, 0.05, -4.94], counterStone, root, 0.015)
  for (let i = 0; i < 4; i++) {
    const x = -1.79 + i * 1.6
    box(`ldk-mullion-${i}`, [0.045, 2.56, 0.06], [x, 1.37, -4.97], black, root, 0.006)
    box(`ldk-glass-${i}`, [1.56, 2.5, 0.018], [x + 0.8, 1.37, -4.985], glass, root, 0).castShadow = false
  }
  box('ldk-mullion-end', [0.045, 2.56, 0.06], [4.61, 1.37, -4.97], black, root, 0.006)
  cylinder('ldk-curtain-rail', 0.018, 0.018, 6.62, [1.4, 2.6, -4.72], steel, root).rotation.z = Math.PI / 2
  for (const [name, centre] of [['ldk-curtain-left', -1.32], ['ldk-curtain-right', 4.16]] as const) {
    const curtain = new THREE.PlaneGeometry(0.93, 2.45, 56, 18)
    const positions = curtain.attributes.position
    for (let i = 0; i < positions.count; i++) positions.setZ(i, Math.cos((positions.getX(i) + 0.465) * Math.PI * 16) * 0.043)
    curtain.computeVertexNormals()
    mesh(name, curtain, curtainMaterial, root, [centre, 1.36, -4.68])
  }
  mesh('garden-lawn', new THREE.BoxGeometry(12, 0.08, 6), material('GardenGrass', { color: 0x65764f, roughness: 1 }), root,
    [0, -0.075, -8.1]).castShadow = false
  box('garden-patio', [8, 0.045, 1.35], [0.8, -0.008, -5.76], material('GardenPaving', { color: 0xaaa79a, roughness: 0.97 }), root, 0.008)
  const gardenSky = new THREE.MeshBasicMaterial({ color: 0xcbd9d7, toneMapped: false })
  materials.add(gardenSky)
  const backdrop = mesh('garden-backdrop', new THREE.PlaneGeometry(16, 6), gardenSky, root, [0, 2.4, -11.2])
  backdrop.castShadow = backdrop.receiveShadow = false
  mesh('ldk-ceiling-plaster', new THREE.PlaneGeometry(6.2, 10), ceilingPlaster, ceilings, [-1.9, ceiling1 - 0.002, 0]).rotation.x = Math.PI / 2
  mesh('ldk-ceiling-oak', new THREE.PlaneGeometry(3.8, 10), woodCeiling, ceilings, [3.1, ceiling1 - 0.002, 0]).rotation.x = Math.PI / 2
  box('ldk-cove', [0.012, 0.012, 9.72], [4.815, 2.72, 0], warmGlow, ceilings, 0.003)
  for (const side of [-1, 1]) box(`ldk-skirting-${side}`, [0.035, 0.085, 10], [side * 4.935, 0.0425, 0], whiteWood, root, 0.005)
  // Living, kitchen, dining, tatami corner and the sideboard wall (Three.js x, z).
  for (const [x, z, intensity] of [[2.9, -0.3, 5], [-3.6, -2.45, 4], [-0.95, -2.45, 4], [-3.6, 3.6, 3], [3.0, 3.6, 3]] as const) {
    const light = new THREE.PointLight(0xffe0b3, intensity, 6.5, 2)
    light.position.set(x, ceiling1 - 0.35, z)
    lights.add(light)
  }

  // ---------------------------------------------------------------- objects
  let chairTemplate: THREE.Object3D | null = null
  const chairSlots: THREE.Group[] = []
  const plant = (parent: THREE.Object3D, height: number, potRadius: number) => {
    lathe('pot', [[potRadius * 0.62, 0], [potRadius * 0.85, 0.015], [potRadius, height * 0.3], [potRadius * 0.96, height * 0.31]],
      [0, 0, 0], terracotta, parent)
    for (let i = 0; i < 11; i++) {
      const a = i * 2.4
      const leafMesh = sphere('leaf', [height * 0.12, height * 0.26, height * 0.025],
        [Math.sin(a) * height * 0.2, height * (0.52 + (i % 3) * 0.13), Math.cos(a) * height * 0.2], i % 2 ? leaf : leafLight, parent)
      leafMesh.rotation.set(0.55, a, Math.sin(a) * 0.6)
    }
  }
  const builders: Record<string, (o: HouseObject, g: THREE.Group) => void> = {
    'shoe cabinet': (o, g) => {
      const [w, d, h] = o.size
      box('body', [w, h, d], [0, h / 2, 0], oak, g, 0.01)
      for (let i = 0; i < 3; i++) {
        box('door', [w / 3 - 0.01, h - 0.08, 0.015], [-w / 3 + (i * w) / 3, h / 2, d / 2 + 0.006], oak, g, 0.004)
        box('handle', [0.012, 0.18, 0.02], [-w / 3 + (i * w) / 3 + w / 6 - 0.04, h / 2 + 0.1, d / 2 + 0.02], steel, g, 0.004)
      }
      box('top', [w + 0.02, 0.03, d + 0.02], [0, h + 0.015, 0], tile, g, 0.004)
      lathe('vase', [[0.04, 0], [0.06, 0.08], [0.035, 0.2], [0.03, 0.24]], [w / 3, h + 0.03, 0], ceramic, g)
    },
    'umbrella stand': (o, g) => cylinder('stand', 0.11, 0.1, o.size[2], [0, o.size[2] / 2, 0], steel, g),
    umbrella: (o, g) => {
      cylinder('canopy', 0.02, 0.06, 0.7, [0, 0.35, 0], navy, g, 12)
      const handle = mesh('handle', new THREE.TorusGeometry(0.045, 0.012, 8, 16, Math.PI), darkOak, g, [0.045, 0.75, 0])
      handle.rotation.z = Math.PI
      cylinder('shaft', 0.008, 0.008, 0.12, [0, 0.72, 0], steel, g)
    },
    shoes: (o, g) => {
      for (const [x, z, surface] of [[-0.12, 0, shoeLeather], [-0.02, 0.02, shoeLeather], [0.1, -0.02, sneaker], [0.2, 0, sneaker]] as const) {
        box('shoe', [0.09, 0.08, 0.26], [x, 0.04, z], surface, g, 0.035)
      }
    },
    toilet: (o, g) => {
      const [, d] = o.size
      box('tank', [0.4, 0.4, 0.18], [0, 0.62, -d / 2 + 0.11], ceramic, g, 0.04)
      box('tank-lid', [0.42, 0.03, 0.2], [0, 0.835, -d / 2 + 0.11], ceramic, g, 0.012)
      lathe('pedestal', [[0.12, 0], [0.13, 0.05], [0.1, 0.2], [0.13, 0.38]], [0, 0, 0.03], ceramic, g)
      const bowl = sphere('bowl', [0.19, 0.12, 0.25], [0, 0.34, 0.06], ceramic, g)
      bowl.scale.y = 0.13
      const seat = mesh('seat', new THREE.TorusGeometry(0.16, 0.025, 10, 32), ceramic, g, [0, 0.43, 0.06])
      seat.rotation.x = Math.PI / 2
      seat.scale.set(1, 1.3, 1)
      const lid = box('lid', [0.36, 0.42, 0.025], [0, 0.65, -0.1], ceramic, g, 0.012)
      lid.rotation.x = -0.12
      box('flush', [0.05, 0.02, 0.03], [0.14, 0.86, -d / 2 + 0.11], chrome, g, 0.005)
    },
    sink: (o, g) => {
      const [w, d, h] = o.size
      box('cabinet', [w, h - 0.05, d - 0.03], [0, (h - 0.05) / 2, -0.015], whiteWood, g, 0.01)
      box('counter', [w, 0.05, d], [0, h - 0.025, 0], ceramic, g, 0.01)
      const basin = sphere('basin', [w * 0.32, 0.06, d * 0.3], [0, h + 0.005, 0.03], glassDark, g)
      basin.material = material('BasinInner', { color: 0xe9ecec, roughness: 0.15 })
      const spout = cylinder('faucet', 0.015, 0.015, 0.22, [0, h + 0.11, -d / 2 + 0.08], chrome, g)
      spout.rotation.x = 0
      const arm = cylinder('faucet-arm', 0.012, 0.012, 0.15, [0, h + 0.21, -d / 2 + 0.15], chrome, g)
      arm.rotation.x = Math.PI / 2
      box('drawer-line', [w - 0.06, 0.01, 0.01], [0, h * 0.55, d / 2 - 0.02], grey, g, 0)
    },
    mirror: (o, g) => {
      const [w, d, h] = o.size
      box('mirror', [w, h, d], [0, h / 2, 0], mirror, g, 0.01)
      box('mirror-frame', [w + 0.04, h + 0.04, d * 0.5], [0, h / 2, -d * 0.3], whiteWood, g, 0.01)
      const light = cylinder('mirror-light', 0.02, 0.02, w * 0.8, [0, h + 0.05, 0.03], warmGlow, g)
      light.rotation.z = Math.PI / 2
    },
    'washing machine': (o, g) => {
      const [w, d, h] = o.size
      box('body', [w, h, d], [0, h / 2, 0], ceramic, g, 0.03)
      const ring = mesh('door-ring', new THREE.TorusGeometry(0.17, 0.025, 12, 32), chrome, g, [0, h * 0.45, d / 2 + 0.01])
      ring.rotation.set(0, 0, 0)
      const glass = cylinder('door-glass', 0.16, 0.16, 0.02, [0, h * 0.45, d / 2 + 0.005], glassDark, g)
      glass.rotation.x = Math.PI / 2
      box('panel', [w - 0.04, 0.09, 0.02], [0, h - 0.06, d / 2 + 0.002], grey, g, 0.005)
      const knob = cylinder('knob', 0.03, 0.03, 0.025, [w / 4, h - 0.06, d / 2 + 0.015], chrome, g)
      knob.rotation.x = Math.PI / 2
    },
    'hair drier': (o, g) => {
      const body = cylinder('drier', 0.035, 0.03, 0.18, [0, 0.04, 0], black, g)
      body.rotation.z = Math.PI / 2
      cylinder('drier-handle', 0.015, 0.015, 0.1, [0.04, -0.01, 0], black, g)
    },
    toothbrush: (o, g) => {
      lathe('cup', [[0.03, 0], [0.034, 0.1]], [0, 0, 0], material('Cup', { color: 0x8fb7c9, roughness: 0.3 }), g)
      for (const [x, colour] of [[-0.01, 0x2f73c9], [0.012, 0xe0564a]] as const) {
        const brush = cylinder('brush', 0.004, 0.004, 0.18, [x, 0.12, 0], material(`Brush${colour}`, { color: colour, roughness: 0.4 }), g, 8)
        brush.rotation.z = x * 8
      }
    },
    bathtub: (o, g) => {
      const [w, d, h] = o.size
      box('apron', [w, h, d], [0, h / 2, 0], ceramic, g, 0.04)
      plane('water', [-w / 2 + 0.08, -d / 2 + 0.08, w / 2 - 0.08, d / 2 - 0.08], 0, water, g)
        .position.set(0, h - 0.06, 0)
      for (const [x, z, sw, sd] of [[0, d / 2 - 0.04, w, 0.08], [0, -d / 2 + 0.04, w, 0.08], [w / 2 - 0.04, 0, 0.08, d], [-w / 2 + 0.04, 0, 0.08, d]] as const) {
        box('rim', [sw, 0.04, sd], [x, h + 0.01, z], ceramic, g, 0.015)
      }
      const tap = cylinder('bath-tap', 0.02, 0.02, 0.16, [w / 2 - 0.2, h + 0.18, -d / 2 + 0.03], chrome, g)
      tap.rotation.x = Math.PI / 2
    },
    shower: (o, g) => {
      cylinder('rail', 0.012, 0.012, 1.0, [0, -0.2, 0], chrome, g)
      const head = cylinder('head', 0.06, 0.05, 0.03, [0, 0.25, 0.06], chrome, g)
      head.rotation.x = 1.1
    },
    stool: (o, g) => {
      const [w, d, h] = o.size
      box('seat', [w, 0.04, d], [0, h - 0.02, 0], material('StoolPlastic', { color: 0xf0f0ea, roughness: 0.4 }), g, 0.015)
      for (const x of [-w / 2 + 0.03, w / 2 - 0.03]) box('leg', [0.03, h - 0.04, d - 0.04], [x, (h - 0.04) / 2, 0], grey, g, 0.008)
    },
    bed: (o, g) => {
      const [w, d, h] = o.size
      const duvet = o.id.includes('master') ? duvetBlue : o.id.endsWith('1') ? duvetCoral : duvetSage
      box('frame', [w, 0.25, d], [0, 0.17, 0], oak, g, 0.02)
      for (const x of [-w / 2 + 0.05, w / 2 - 0.05]) for (const z of [-d / 2 + 0.05, d / 2 - 0.05]) {
        box('leg', [0.06, 0.06, 0.06], [x, 0.03, z], darkOak, g, 0.008)
      }
      box('mattress', [w - 0.06, 0.2, d - 0.06], [0, 0.39, 0], linenWhite, g, 0.06)
      box('duvet', [w + 0.02, 0.08, d * 0.68], [0, h - 0.02, d * 0.14], duvet, g, 0.04)
      box('headboard', [w, 0.9, 0.06], [0, 0.55, -d / 2 + 0.03], oak, g, 0.02)
      const pillows = w > 1.2 ? [-w / 4, w / 4] : [0]
      for (const x of pillows) box('pillow', [w > 1.2 ? w / 2 - 0.12 : w - 0.25, 0.12, 0.38], [x, h + 0.03, -d / 2 + 0.3], linenWhite, g, 0.055)
    },
    nightstand: (o, g) => {
      const [w, d, h] = o.size
      box('body', [w, h, d], [0, h / 2, 0], oak, g, 0.012)
      box('drawer', [w - 0.05, h * 0.3, 0.015], [0, h * 0.72, d / 2 + 0.005], oak, g, 0.004)
      const knob = cylinder('knob', 0.012, 0.012, 0.02, [0, h * 0.72, d / 2 + 0.02], steel, g)
      knob.rotation.x = Math.PI / 2
    },
    'table lamp': (o, g) => {
      lathe('lamp-base', [[0.05, 0], [0.07, 0.03], [0.04, 0.16], [0.02, 0.2]], [0, 0, 0], ceramic, g)
      lathe('lamp-shade', [[0.13, 0], [0.09, 0.18]], [0, 0.18, 0], warmGlow, g)
    },
    dresser: (o, g) => {
      const [w, d, h] = o.size
      box('body', [w, h, d], [0, h / 2, 0], oak, g, 0.012)
      for (let i = 0; i < 4; i++) {
        box('drawer', [w - 0.05, h / 4 - 0.03, 0.015], [0, (i + 0.5) * (h / 4), d / 2 + 0.005], oak, g, 0.004)
        box('pull', [0.12, 0.015, 0.02], [0, (i + 0.5) * (h / 4) + 0.04, d / 2 + 0.02], darkOak, g, 0.004)
      }
    },
    clock: (o, g) => {
      const face = cylinder('clock-face', 0.15, 0.15, 0.03, [0, 0.15, 0], clockFace, g, 40)
      face.rotation.x = Math.PI / 2
      face.rotation.y = Math.PI   // the canvas faces the room
      const rim = mesh('clock-rim', new THREE.TorusGeometry(0.15, 0.012, 10, 40), darkOak, g, [0, 0.15, 0.01])
      rim.castShadow = false
    },
    'potted plant': (o, g) => plant(g, o.size[2], Math.min(o.size[0], o.size[1]) * 0.4),
    'clothes rack': (o, g) => {
      const [w, d, h] = o.size
      const rail = cylinder('rail', 0.015, 0.015, w, [0, h - 0.1, 0], chrome, g)
      rail.rotation.z = Math.PI / 2
      for (const x of [-w / 2, w / 2]) cylinder('post', 0.015, 0.015, h - 0.1, [x, (h - 0.1) / 2, 0], chrome, g)
      for (let i = 0; i < 12; i++) {
        const x = -w / 2 + 0.12 + i * ((w - 0.24) / 11)
        const long = i % 3 === 0
        box('garment', [0.04, long ? 1.05 : 0.7, d - 0.08], [x, h - 0.15 - (long ? 0.525 : 0.35), 0], garments[i % garments.length], g, 0.015)
      }
      box('base', [w, 0.3, d], [0, 0.15, 0], whiteWood, g, 0.01)
    },
    shelf: (o, g) => {
      const [w, d, h] = o.size
      for (const x of [-w / 2 + 0.01, w / 2 - 0.01]) box('side', [0.02, h, d], [x, h / 2, 0], whiteWood, g, 0)
      for (let i = 0; i < 5; i++) {
        box('board', [w, 0.02, d], [0, i * (h / 4.2) + 0.01, 0], whiteWood, g, 0)
        if (i < 4) for (let k = 0; k < 3; k++) {
          box('folded', [w / 3.6, 0.12, d - 0.08], [-w / 3 + (k * w) / 3, i * (h / 4.2) + 0.08, 0], garments[(i + k) % garments.length], g, 0.02)
        }
      }
    },
    desk: (o, g) => {
      const [w, d, h] = o.size
      box('top', [w, 0.03, d], [0, h - 0.015, 0], oak, g, 0.008)
      for (const x of [-w / 2 + 0.03, w / 2 - 0.03]) box('side', [0.03, h - 0.03, d - 0.04], [x, (h - 0.03) / 2, 0], whiteWood, g, 0.005)
      box('drawer', [0.4, 0.12, d - 0.04], [w / 2 - 0.25, h - 0.1, 0], whiteWood, g, 0.006)
      box('drawer-pull', [0.12, 0.015, 0.02], [w / 2 - 0.25, h - 0.1, d / 2 - 0.005], steel, g, 0.004)
    },
    chair: (o, g) => { chairSlots.push(g) },
    bookshelf: (o, g) => {
      const [w, d, h] = o.size
      box('back', [w, h, 0.015], [0, h / 2, -d / 2 + 0.008], oak, g, 0)
      for (const x of [-w / 2 + 0.01, w / 2 - 0.01]) box('side', [0.02, h, d], [x, h / 2, 0], oak, g, 0)
      for (let i = 0; i < 4; i++) {
        box('board', [w, 0.02, d], [0, i * (h / 3.4) + 0.01, 0], oak, g, 0)
        if (i < 3) for (let k = 0; k < 12; k++) {
          const height = 0.18 + ((k * 7 + i * 3) % 5) * 0.02
          box('book', [0.035, height, d - 0.06], [-w / 2 + 0.05 + k * 0.042, i * (h / 3.4) + 0.02 + height / 2, 0], bookColours[(k + i) % bookColours.length], g, 0.003)
        }
      }
    },
    book: () => { /* the bookshelf already shows its books */ },
    laptop: (o, g) => {
      const [w, d] = o.size
      box('base', [w, 0.015, d], [0, 0.008, 0], grey, g, 0.006)
      const screen = new THREE.Group()
      screen.position.set(0, 0.015, -d / 2)
      screen.rotation.x = -0.25
      g.add(screen)
      box('lid', [w, d * 0.95, 0.008], [0, d * 0.475, 0], grey, screen, 0.004)
      box('display', [w - 0.02, d * 0.85, 0.002], [0, d * 0.475, 0.005], screenOn, screen, 0)
      box('keys', [w - 0.04, 0.002, d * 0.5], [0, 0.017, d * 0.05], black, g, 0)
    },
    backpack: (o, g) => {
      const [w, d, h] = o.size
      box('body', [w, h, d], [0, h / 2, 0], red, g, 0.05)
      box('flap', [w + 0.01, h * 0.75, 0.03], [0, h * 0.6, d / 2 + 0.005], red, g, 0.04)
      box('clasp', [0.05, 0.03, 0.02], [0, h * 0.28, d / 2 + 0.025], material('Gold', { color: 0xc9a54a, metalness: 0.8, roughness: 0.3 }), g, 0.006)
    },
    'teddy bear': (o, g) => {
      sphere('body', [0.11, 0.12, 0.09], [0, 0.1, 0], brownFur, g)
      sphere('head', [0.085, 0.08, 0.08], [0, 0.25, 0.01], brownFur, g)
      sphere('muzzle', [0.035, 0.028, 0.03], [0, 0.235, 0.075], brownLight, g)
      for (const x of [-0.06, 0.06]) {
        sphere('ear', [0.03, 0.03, 0.015], [x, 0.32, 0], brownFur, g)
        sphere('arm', [0.035, 0.07, 0.035], [x * 1.7, 0.13, 0.03], brownFur, g)
        sphere('leg', [0.04, 0.035, 0.06], [x, 0.03, 0.07], brownFur, g)
        sphere('eye', [0.01, 0.01, 0.01], [x * 0.45, 0.27, 0.075], black, g)
      }
    },
    'sports ball': (o, g) => sphere('ball', [0.11, 0.11, 0.11], [0, 0.11, 0], ballMaterial, g),
    monitor: (o, g) => {
      const [w, , h] = o.size
      cylinder('foot', 0.09, 0.1, 0.015, [0, 0.008, 0], black, g)
      box('stem', [0.04, 0.12, 0.03], [0, 0.07, -0.02], black, g, 0.005)
      box('bezel', [w, h * 0.7, 0.025], [0, 0.12 + h * 0.35, 0], black, g, 0.008)
      box('screen', [w - 0.03, h * 0.7 - 0.03, 0.003], [0, 0.12 + h * 0.35, 0.014], screenOn, g, 0)
    },
  }
  const doorsAcross = (g: THREE.Group, w: number, h: number, d: number, bottom: number, surface: THREE.Material) => {
    const count = Math.max(1, Math.round(w / 0.6))
    for (let i = 0; i < count; i++) {
      const x = -w / 2 + (i + 0.5) * (w / count)
      box('door', [w / count - 0.015, h - 0.02, 0.018], [x, bottom + h / 2, d / 2 + 0.008], surface, g, 0.004)
      box('pull', [0.012, 0.14, 0.02], [x + (i % 2 ? -1 : 1) * (w / count / 2 - 0.05), bottom + h * 0.75, d / 2 + 0.022], steel, g, 0.004)
    }
  }
  const washroomSink = builders.sink   // the vanity; kitchen sinks are drawn below
  Object.assign(builders, {
    cupboard: (o: HouseObject, g: THREE.Group) => {
      const [w, d, h] = o.size
      if (o.variant === 'glass') {   // tall dish cabinet, glass upper doors, dishes on the shelves
        box('body', [w, h, d], [0, h / 2, 0], whiteWood, g, 0.01)
        doorsAcross(g, w, 0.8, d, 0.04, whiteWood)
        box('counter', [w + 0.01, 0.03, d + 0.01], [0, 0.865, 0.005], oak, g, 0.004)
        for (const x of [-w / 4, w / 4]) box('glass-door', [w / 2 - 0.02, h - 1.0, 0.012], [x, 0.92 + (h - 1.0) / 2, d / 2 + 0.006], glassDark, g, 0.004)
        for (let i = 0; i < 3; i++) for (let k = 0; k < 5; k++) {
          const plate = cylinder('plate', 0.09, 0.09, 0.012, [-w / 2 + 0.15 + k * (w - 0.3) / 4, 1.0 + i * 0.28, 0.02], ceramic, g)
          plate.rotation.x = Math.PI / 2 - 0.25
        }
        return
      }
      box('body', [w, h - 0.1, d - 0.02], [0, (h - 0.1) / 2 + 0.1, -0.01], whiteWood, g, 0.006)
      box('toe-kick', [w - 0.02, 0.1, d - 0.08], [0, 0.05, -0.04], grey, g, 0)
      box('top', [w + 0.02, 0.03, d + 0.03], [0, h - 0.015, 0.015], counterStone, g, 0.004)
      doorsAcross(g, w, h - 0.13, d, 0.1, whiteWood)
    },
    'wall cabinet': (o: HouseObject, g: THREE.Group) => {
      const [w, d, h] = o.size
      box('body', [w, h, d - 0.02], [0, h / 2, -0.01], whiteWood, g, 0.006)
      doorsAcross(g, w, h, d, 0, whiteWood)
      box('under-light', [w - 0.1, 0.008, 0.03], [0, -0.004, d / 2 - 0.07], warmGlow, g, 0)
    },
    microwave: (o: HouseObject, g: THREE.Group) => {
      const [w, d, h] = o.size
      box('body', [w, h, d], [0, h / 2, 0], appliance, g, 0.02)
      box('window', [w * 0.66, h * 0.72, 0.006], [-w * 0.13, h / 2, d / 2 + 0.002], glassDark, g, 0.006)
      box('panel', [w * 0.2, h * 0.8, 0.006], [w * 0.37, h / 2, d / 2 + 0.002], grey, g, 0.004)
      box('handle', [0.018, h * 0.6, 0.022], [w * 0.23, h / 2, d / 2 + 0.013], steel, g, 0.006)
    },
    'rice cooker': (o: HouseObject, g: THREE.Group) => {
      const [w, d, h] = o.size
      box('body', [w, h * 0.75, d], [0, h * 0.375, 0], appliance, g, 0.06)
      sphere('lid', [w / 2, h * 0.28, d / 2], [0, h * 0.72, 0], appliance, g)
      box('panel', [w * 0.6, h * 0.25, 0.01], [0, h * 0.4, d / 2], blackGlass, g, 0.01)
    },
    toaster: (o: HouseObject, g: THREE.Group) => {
      const [w, d, h] = o.size
      box('body', [w, h, d], [0, h / 2, 0], steel, g, 0.015)
      box('window', [w * 0.62, h * 0.55, 0.005], [-w * 0.12, h * 0.48, d / 2 + 0.001], glassDark, g, 0.004)
      for (const y of [0.35, 0.65]) {
        const dial = cylinder('dial', 0.016, 0.016, 0.012, [w * 0.35, h * y, d / 2 + 0.006], black, g)
        dial.rotation.x = Math.PI / 2
      }
    },
    kettle: (o: HouseObject, g: THREE.Group) => {
      const [, , h] = o.size
      cylinder('base', 0.085, 0.085, 0.02, [0, 0.01, 0], black, g)
      lathe('body', [[0.075, 0], [0.08, h * 0.5], [0.06, h * 0.85], [0.03, h * 0.9]], [0, 0.02, 0], appliance, g)
      box('handle', [0.025, h * 0.6, 0.06], [0, h * 0.45, -0.09], black, g, 0.01)
    },
    refrigerator: (o: HouseObject, g: THREE.Group) => {
      const [w, d, h] = o.size   // a typical six-door refrigerator: French doors over three drawers
      box('body', [w, h, d - 0.03], [0, h / 2, -0.015], fridgeFinish, g, 0.02)
      for (const x of [-w / 4, w / 4]) box('door', [w / 2 - 0.008, h * 0.55, 0.03], [x, h * 0.72, d / 2 - 0.015], fridgeFinish, g, 0.012)
      for (const [y, hh] of [[h * 0.375, h * 0.14], [h * 0.225, h * 0.14], [h * 0.075, h * 0.13]] as const) {
        box('drawer', [w - 0.008, hh - 0.01, 0.03], [0, y, d / 2 - 0.015], fridgeFinish, g, 0.012)
        box('drawer-pull', [w * 0.5, 0.015, 0.02], [0, y + hh / 2 - 0.03, d / 2 + 0.008], steel, g, 0.005)
      }
      for (const x of [-0.035, 0.035]) box('handle', [0.02, h * 0.3, 0.025], [x, h * 0.72, d / 2 + 0.012], steel, g, 0.008)
      box('display', [0.06, 0.04, 0.004], [-w / 4, h * 0.88, d / 2 + 0.002], screenOn, g, 0)
    },
    'trash bin': (o: HouseObject, g: THREE.Group) => {
      const [w, d, h] = o.size
      for (const x of [-w / 4, w / 4]) {
        box('bin', [w / 2 - 0.02, h - 0.03, d], [x, (h - 0.03) / 2, 0], appliance, g, 0.03)
        box('lid', [w / 2 - 0.02, 0.03, d], [x, h - 0.015, 0], grey, g, 0.012)
      }
    },
    'kitchen counter': (o: HouseObject, g: THREE.Group) => {
      // The cook's side (front, +z) has a 0.85 m worktop; the dining side is a 1.05 m panel hiding the sink.
      const [w, d] = o.size
      box('cabinet', [w, 0.72, d - 0.2], [0, 0.46, 0.1], whiteWood, g, 0.006)
      box('toe-kick', [w - 0.02, 0.1, d - 0.26], [0, 0.05, 0.07], grey, g, 0)
      box('worktop', [w + 0.02, 0.03, d - 0.18], [0, 0.835, 0.09], counterStone, g, 0.004)
      box('panel', [w + 0.02, 1.05, 0.2], [0, 0.525, -d / 2 + 0.1], oak, g, 0.01)
      box('bar-top', [w + 0.04, 0.03, 0.26], [0, 1.065, -d / 2 + 0.1], oak, g, 0.006)
      const sections = 4
      for (let i = 0; i < sections; i++) {
        const x = -w / 2 + (i + 0.5) * (w / sections)
        box('drawer', [w / sections - 0.015, 0.2, 0.018], [x, 0.7, d / 2 + 0.008], whiteWood, g, 0.004)
        box('door', [w / sections - 0.015, 0.47, 0.018], [x, 0.34, d / 2 + 0.008], whiteWood, g, 0.004)
        box('pull', [0.16, 0.012, 0.02], [x, 0.76, d / 2 + 0.022], steel, g, 0.004)
      }
    },
    sink: (o: HouseObject, g: THREE.Group) => {
      const [w, d, h] = o.size
      if (o.variant !== 'kitchen') { washroomSink(o, g); return }
      box('rim', [w + 0.03, 0.004, d + 0.03], [0, h + 0.003, 0], steel, g, 0.003)   // inset stainless sink
      box('basin', [w - 0.04, 0.004, d - 0.04], [0, h + 0.006, 0], basinShade, g, 0.02)
      cylinder('faucet', 0.016, 0.018, 0.3, [0, h + 0.15, -d / 2 - 0.04], chrome, g)
      const arc = mesh('faucet-arc', new THREE.TorusGeometry(0.09, 0.013, 8, 20, Math.PI), chrome, g, [0, h + 0.3, -d / 2 + 0.05])
      arc.rotation.y = Math.PI / 2
    },
    stove: (o: HouseObject, g: THREE.Group) => {
      const [w, d] = o.size
      box('glass', [w, 0.008, d], [0, 0.004, 0], blackGlass, g, 0.01)
      for (const [x, z, r] of [[-w / 4, 0.06, 0.09], [w / 4, 0.06, 0.09], [0, -0.13, 0.065]] as const) {
        const ring = mesh('ring', new THREE.TorusGeometry(r, 0.003, 6, 40), grey, g, [x, 0.0095, z])
        ring.rotation.x = Math.PI / 2
      }
      box('controls', [w * 0.7, 0.002, 0.04], [0, 0.0095, d / 2 - 0.04], screenOn, g, 0)
    },
    pot: (o: HouseObject, g: THREE.Group) => {
      lathe('pot', [[0.11, 0], [0.12, 0.01], [0.12, 0.12]], [0, 0, 0], steel, g)
      lathe('lid', [[0.122, 0], [0.09, 0.025], [0.02, 0.035]], [0, 0.12, 0], glassDark, g)
      for (const x of [-0.14, 0.14]) box('handle', [0.04, 0.015, 0.03], [x, 0.1, 0], black, g, 0.005)
    },
    'range hood': (o: HouseObject, g: THREE.Group) => {
      const [w, d, h] = o.size
      const top = o.position[2] + h / 2, duct = Math.max(ceiling1 - top, 0.01)
      box('hood', [w, h, d], [0, h / 2, 0], appliance, g, 0.03)
      box('duct', [0.32, duct, 0.32], [0, h + duct / 2, 0], appliance, g, 0.01)
      box('filter', [w - 0.06, 0.004, d - 0.06], [0, -0.002, 0], grey, g, 0)
      box('light', [0.2, 0.004, 0.06], [0, -0.004, d / 2 - 0.1], warmGlow, g, 0)
    },
    'dish rack': (o: HouseObject, g: THREE.Group) => {
      const [w, d, h] = o.size
      box('tray', [w, 0.02, d], [0, 0.01, 0], steel, g, 0.005)
      for (let i = 0; i < 4; i++) {
        const plate = cylinder('plate', 0.1, 0.1, 0.012, [-w / 2 + 0.07 + i * 0.065, h * 0.6, 0], ceramic, g)
        plate.rotation.z = Math.PI / 2
      }
      lathe('cup', [[0.03, 0], [0.04, 0.09]], [w / 2 - 0.06, 0.02, d / 4], ceramic, g)
    },
    'pendant light': (o: HouseObject, g: THREE.Group) => {
      const [w, , h] = o.size
      const cable = Math.max(ceiling1 - (o.position[2] + h / 2), 0.01)
      cylinder('cable', 0.003, 0.003, cable, [0, h + cable / 2, 0], black, g, 6)
      lathe('shade', [[w / 2, 0], [w / 2 * 0.95, h * 0.1], [w / 2 * 0.55, h * 0.7], [0.03, h]], [0, 0, 0],
        o.id.includes('dining') ? ceramic : black, g)
      cylinder('glow', w / 2 * 0.85, w / 2 * 0.85, 0.006, [0, 0.01, 0], warmGlow, g)
    },
    'dining table': (o: HouseObject, g: THREE.Group) => {
      const [w, d, h] = o.size
      box('top', [w, 0.035, d], [0, h - 0.0175, 0], oak, g, 0.008)
      box('apron', [w - 0.1, 0.07, d - 0.14], [0, h - 0.07, 0], oak, g, 0.004)
      for (const sx of [-1, 1]) for (const sz of [-1, 1]) box('leg', [0.05, h - 0.035, 0.05], [sx * (w / 2 - 0.06), (h - 0.035) / 2, sz * (d / 2 - 0.08)], darkOak, g, 0.006)
      for (const sx of [-1, 1]) for (const sz of [-0.25, 0.25]) box('placemat', [0.3, 0.003, 0.4], [sx * (w / 2 - 0.18), h + 0.002, sz * d], rugFabric, g, 0)
    },
    'tv board': (o: HouseObject, g: THREE.Group) => {
      const [w, d, h] = o.size
      box('body', [w, h - 0.06, d], [0, (h - 0.06) / 2 + 0.06, 0], oak, g, 0.008)
      for (const x of [-w / 2 + 0.08, w / 2 - 0.08]) box('foot', [0.05, 0.06, d - 0.06], [x, 0.03, 0], darkOak, g, 0.004)
      for (let i = 0; i < 3; i++) box('drawer', [w / 3 - 0.02, h - 0.12, 0.015], [-w / 3 + i * (w / 3), (h + 0.06) / 2, d / 2 + 0.006], oak, g, 0.004)
      box('router', [0.18, 0.04, 0.12], [w / 3, h + 0.02, -0.05], black, g, 0.008)
    },
    tv: (o: HouseObject, g: THREE.Group) => {
      const [w, , h] = o.size
      box('stand', [0.36, 0.02, 0.22], [0, 0.01, 0], black, g, 0.006)
      box('neck', [0.06, 0.08, 0.04], [0, 0.06, -0.02], black, g, 0.006)
      box('frame', [w, h - 0.08, 0.05], [0, 0.08 + (h - 0.08) / 2, -0.02], black, g, 0.008)
      box('screen', [w - 0.03, h - 0.11, 0.004], [0, 0.08 + (h - 0.08) / 2, 0.006], tvScreen, g, 0)
    },
    sofa: (o: HouseObject, g: THREE.Group) => {
      const [w, d, h] = o.size
      box('base', [w, 0.24, d], [0, 0.22, 0], sofaFabric, g, 0.06)
      box('back', [w, h - 0.1, 0.2], [0, (h - 0.1) / 2 + 0.1, -d / 2 + 0.1], sofaFabric, g, 0.07)
      for (const s of [-1, 1]) box('arm', [0.18, 0.52, d], [s * (w / 2 - 0.09), 0.36, 0], sofaFabric, g, 0.07)
      const seat = (w - 0.36) / 3
      for (let i = 0; i < 3; i++) {
        const x = -w / 2 + 0.18 + (i + 0.5) * seat
        box('seat', [seat - 0.02, 0.16, d - 0.25], [x, 0.42, 0.08], sofaFabric, g, 0.06)
        box('back-cushion', [seat - 0.03, 0.4, 0.16], [x, 0.66, -d / 2 + 0.28], cushionFabric, g, 0.07).rotation.x = -0.12
      }
      for (const [x, surface] of [[-w / 2 + 0.42, accentFabric], [w / 2 - 0.42, cushionFabric]] as const) {
        box('pillow', [0.4, 0.38, 0.13], [x, 0.66, -0.12], surface, g, 0.06).rotation.set(-0.15, 0, x > 0 ? -0.15 : 0.15)
      }
      for (const sx of [-1, 1]) for (const sz of [-1, 1]) cylinder('foot', 0.025, 0.02, 0.1, [sx * (w / 2 - 0.08), 0.05, sz * (d / 2 - 0.08)], darkOak, g, 10)
    },
    armchair: (o: HouseObject, g: THREE.Group) => {
      const [w, d, h] = o.size
      box('base', [w, 0.22, d], [0, 0.22, 0], armchairFabric, g, 0.06)
      box('back', [w, h - 0.1, 0.18], [0, (h - 0.1) / 2 + 0.1, -d / 2 + 0.09], armchairFabric, g, 0.07)
      for (const s of [-1, 1]) box('arm', [0.14, 0.5, d], [s * (w / 2 - 0.07), 0.36, 0], armchairFabric, g, 0.06)
      box('seat', [w - 0.3, 0.14, d - 0.24], [0, 0.4, 0.07], armchairFabric, g, 0.05)
      for (const sx of [-1, 1]) for (const sz of [-1, 1]) cylinder('foot', 0.022, 0.018, 0.11, [sx * (w / 2 - 0.07), 0.055, sz * (d / 2 - 0.07)], darkOak, g, 10)
    },
    'coffee table': (o: HouseObject, g: THREE.Group) => {
      const [w, d, h] = o.size
      box('top', [w, 0.03, d], [0, h - 0.015, 0], oak, g, 0.01)
      box('shelf', [w - 0.1, 0.02, d - 0.08], [0, 0.1, 0], oak, g, 0.004)
      for (const sx of [-1, 1]) for (const sz of [-1, 1]) box('leg', [0.04, h - 0.03, 0.04], [sx * (w / 2 - 0.05), (h - 0.03) / 2, sz * (d / 2 - 0.05)], darkOak, g, 0.004)
      box('remote', [0.05, 0.015, 0.18], [w / 4, h + 0.008, 0], black, g, 0.006)
      box('magazine', [0.21, 0.01, 0.28], [-w / 4, h + 0.005, 0.02], bookColours[4], g, 0.002)
    },
    rug: (o: HouseObject, g: THREE.Group) => {
      const [w, d] = o.size
      box('border', [w, 0.008, d], [0, 0.004, 0], rugBorder, g, 0.003).castShadow = false
      box('field', [w - 0.12, 0.01, d - 0.12], [0, 0.005, 0], rugFabric, g, 0.003).castShadow = false
    },
    'side table': (o: HouseObject, g: THREE.Group) => {
      const [w, d, h] = o.size
      box('top', [w, 0.025, d], [0, h - 0.0125, 0], oak, g, 0.008)
      for (const sx of [-1, 1]) for (const sz of [-1, 1]) box('leg', [0.03, h - 0.025, 0.03], [sx * (w / 2 - 0.04), (h - 0.025) / 2, sz * (d / 2 - 0.04)], darkOak, g, 0.004)
    },
    'floor lamp': (o: HouseObject, g: THREE.Group) => {
      const [, , h] = o.size
      cylinder('base', 0.15, 0.16, 0.025, [0, 0.0125, 0], steel, g)
      cylinder('pole', 0.011, 0.011, h - 0.3, [0, (h - 0.3) / 2, 0], steel, g, 8)
      lathe('shade', [[0.22, 0], [0.18, 0.3]], [0, h - 0.32, 0], warmGlow, g)
    },
    sideboard: (o: HouseObject, g: THREE.Group) => {
      const [w, d, h] = o.size
      box('body', [w, h - 0.08, d], [0, (h - 0.08) / 2 + 0.08, 0], oak, g, 0.008)
      for (const x of [-w / 2 + 0.08, w / 2 - 0.08]) box('foot', [0.05, 0.08, d - 0.06], [x, 0.04, 0], darkOak, g, 0.004)
      doorsAcross(g, w, h - 0.14, d, 0.11, oak)
      for (const [x, tilt] of [[0.1, 0.12], [0.3, -0.08]] as const) {   // photo frames
        const frame = box('photo-frame', [0.16, 0.2, 0.015], [x, h + 0.1, 0], darkOak, g, 0.004)
        frame.rotation.x = -tilt
        box('photo', [0.12, 0.15, 0.002], [x, h + 0.1, 0.009], windowView, g, 0).rotation.x = -tilt
      }
    },
    'air conditioner': (o: HouseObject, g: THREE.Group) => {
      const [w, d, h] = o.size
      box('body', [w, h, d], [0, h / 2, 0], appliance, g, 0.05)
      box('louver', [w - 0.08, 0.03, 0.02], [0, 0.04, d / 2], grey, g, 0.01)
      box('lamp', [0.03, 0.01, 0.005], [w / 2 - 0.08, h * 0.35, d / 2 + 0.002], screenOn, g, 0)
    },
    vase: (o: HouseObject, g: THREE.Group) => {
      const [w, , h] = o.size
      lathe('vase', [[w * 0.3, 0], [w * 0.5, h * 0.35], [w * 0.28, h * 0.7], [w * 0.22, h * 0.8]], [0, 0, 0], ceramic, g)
      for (let i = 0; i < 5; i++) {
        const a = i * 2.4
        cylinder('stem', 0.002, 0.002, h * 0.7, [Math.sin(a) * 0.015, h * 1.1, Math.cos(a) * 0.015], leaf, g, 5).rotation.z = Math.sin(a) * 0.2
        sphere('flower', [0.03, 0.03, 0.03], [Math.sin(a) * 0.06, h * 1.45, Math.cos(a) * 0.05], i % 2 ? accentFabric : cushionFabric, g)
      }
    },
    bowl: (o: HouseObject, g: THREE.Group) => {
      const [w] = o.size
      lathe('bowl', [[w * 0.1, 0], [w * 0.4, 0.015], [w * 0.5, 0.06], [w * 0.48, 0.08]], [0, 0, 0], ceramic, g)
      for (let i = 0; i < 4; i++) sphere('fruit', [0.038, 0.036, 0.038], [-0.05 + (i % 2) * 0.09, 0.08, -0.03 + Math.floor(i / 2) * 0.07], i % 2 ? accentFabric : terracotta, g)
    },
    tatami: (o: HouseObject, g: THREE.Group) => {
      const [w, d] = o.size
      const mat = mesh('tatami', new THREE.PlaneGeometry(w, d), tatamiMaterial, g, [0, 0.006, 0])
      mat.rotation.x = -Math.PI / 2
      mat.castShadow = false
    },
    'low table': (o: HouseObject, g: THREE.Group) => {
      const [w, d, h] = o.size
      box('top', [w, 0.04, d], [0, h - 0.02, 0], darkOak, g, 0.012)
      for (const sx of [-1, 1]) for (const sz of [-1, 1]) box('leg', [0.06, h - 0.04, 0.06], [sx * (w / 2 - 0.07), (h - 0.04) / 2, sz * (d / 2 - 0.07)], darkOak, g, 0.006)
      lathe('teapot', [[0.04, 0], [0.065, 0.05], [0.04, 0.1], [0.012, 0.11]], [0.1, h, 0], ceramic, g)
      for (const x of [-0.12, -0.22]) lathe('cup', [[0.025, 0], [0.032, 0.06]], [x, h, 0.05], ceramic, g)
    },
    cushion: (o: HouseObject, g: THREE.Group) => {
      const [w, d, h] = o.size
      box('zabuton', [w, h, d], [0, h / 2, 0], zabutonFabric, g, 0.035)
    },
    book: (o: HouseObject, g: THREE.Group) => {
      if (!o.id.startsWith('sideboard')) return   // bookshelves draw their own books
      const [w, d, h] = o.size
      for (let k = 0; k < 9; k++) {
        const height = h * (0.75 + ((k * 7) % 4) * 0.08)
        box('book', [w / 9 - 0.004, height, d], [-w / 2 + (k + 0.5) * (w / 9), height / 2, 0], bookColours[k % bookColours.length], g, 0.003)
      }
    },
  })
  for (const o of house.objects) {
    if (o.source === 'home_scene') continue
    const build = builders[o.class]
    if (!build) continue
    build(o, placed(o, o.floor === 2 ? upper : root))
  }

  // ---------------------------------------------------------------- textures and the CC0 chair
  const loadPbr = async (folder: string, surfaces: THREE.MeshStandardMaterial[], repeat: [number, number], normal: number, albedo = true) => {
    const loader = new THREE.TextureLoader()
    const maps = await Promise.all(['albedo', 'normal', 'roughness'].map(async (channel) => {
      const texture = await loader.loadAsync(`${assetBase}/${folder}/${channel}.jpg`)
      if (disposed) { texture.dispose(); return null }
      texture.wrapS = texture.wrapT = THREE.RepeatWrapping
      texture.repeat.set(...repeat)
      texture.anisotropy = 8
      if (channel === 'albedo') texture.colorSpace = THREE.SRGBColorSpace
      textures.add(texture)
      return texture
    }))
    if (disposed || maps.some((texture) => !texture)) return
    for (const surface of surfaces) {
      if (albedo) surface.map = maps[0]
      surface.normalMap = maps[1]
      surface.roughnessMap = maps[2]
      surface.normalScale.setScalar(normal)
      surface.needsUpdate = true
    }
  }
  const loadChairs = async () => {
    const gltf = await new GLTFLoader().loadAsync(`${assetBase}/models/dining-chair/chair.gltf`)
    if (disposed) return
    gltf.scene.updateMatrixWorld(true)
    const bounds = new THREE.Box3().setFromObject(gltf.scene)
    const size = bounds.getSize(new THREE.Vector3()), centre = bounds.getCenter(new THREE.Vector3())
    if (!(size.y > 0.01)) return
    gltf.scene.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return
      geometries.add(child.geometry)
      child.castShadow = child.receiveShadow = true
      for (const surface of Array.isArray(child.material) ? child.material : [child.material]) {
        materials.add(surface)
        if (surface instanceof THREE.MeshStandardMaterial) { surface.envMapIntensity = 0.65; surface.color.set(0xd6c3a5) }
        for (const value of Object.values(surface)) if (value instanceof THREE.Texture) textures.add(value)
      }
    })
    gltf.scene.position.sub(new THREE.Vector3(centre.x, bounds.min.y, centre.z))
    chairTemplate = new THREE.Group()
    chairTemplate.add(gltf.scene)
    chairTemplate.scale.setScalar(0.9 / size.y)
    for (const slot of chairSlots) slot.add(chairTemplate.clone(true))
  }
  const ready = typeof document === 'undefined' ? Promise.resolve() : Promise.allSettled([
    loadPbr('wood', [oakFloor], [1, 1], 0.24),
    loadPbr('wood', [oak, darkOak], [0.9, 1.2], 0.15),
    loadPbr('plaster', [plaster, ceilingPlaster], [2.5, 2.5], 0.08, false),
    loadPbr('stone', [tile, darkTile, bathTile], [1, 1], 0.2, false),
    loadPbr('stone', [counterStone], [1.5, 1.5], 0.15, false),
    loadPbr('fabric', [linenWhite, duvetBlue, duvetSage, duvetCoral], [2, 2], 0.3, false),
    loadPbr('fabric', [sofaFabric, cushionFabric, accentFabric, armchairFabric, rugFabric, rugBorder, zabutonFabric], [3, 3], 0.32, false),
    (async () => {   // the production home's television picture: a centred television-shaped crop
      const texture = await new THREE.TextureLoader().loadAsync(`${assetBase}/tv/mountain-lake.jpg`)
      if (disposed) { texture.dispose(); return }
      texture.colorSpace = THREE.SRGBColorSpace
      texture.repeat.set((1.2 / 0.66) / (3840 / 972), 1)
      texture.offset.x = (1 - texture.repeat.x) / 2
      textures.add(texture)
      tvScreen.color.set(0xffffff); tvScreen.emissive.set(0xffffff); tvScreen.emissiveIntensity = 0.32
      tvScreen.map = texture; tvScreen.emissiveMap = texture; tvScreen.needsUpdate = true
    })(),
    loadChairs(),
    (async () => {
      const texture = await new THREE.TextureLoader().loadAsync(`${assetBase}/garden/greenwich-park.jpg`)
      if (disposed) { texture.dispose(); return }
      texture.colorSpace = THREE.SRGBColorSpace
      texture.repeat.set(0.18, 0.3)
      texture.offset.set(0.25, 0.35)
      textures.add(texture)
      windowView.map = texture
      windowView.emissiveMap = texture
      windowView.color.set(0xffffff)
      windowView.needsUpdate = true
      // the LDK garden backdrop: the leafy part of the panorama, as in the production home
      const garden = texture.clone()
      garden.repeat.set(0.45, 0.45 * 2 / (16 / 6))
      garden.offset.set(0.2, 0.49 - garden.repeat.y / 2)
      garden.needsUpdate = true
      textures.add(garden)
      gardenSky.color.set(0xffffff)
      gardenSky.map = garden
      gardenSky.needsUpdate = true
    })(),
  ]).then(() => undefined)

  return {
    root, upper, ceilings, ready,
    dispose: () => {
      if (disposed) return
      disposed = true
      root.removeFromParent()
      for (const geometry of geometries) geometry.dispose()
      for (const surface of materials) surface.dispose()
      for (const texture of textures) texture.dispose()
    },
  }
}

/** The head camera pose as world_model/depth_perception._camera_pose: eye position, body heading
 * (forward is -Y at heading 0) and a fixed downward pitch; returns a Three.js camera. */
export const placeHeadCamera = (camera: THREE.PerspectiveCamera, eye: number[], heading: number, pitchDegrees: number) => {
  const pitch = THREE.MathUtils.degToRad(pitchDegrees)
  const forward = [Math.cos(pitch) * Math.sin(heading), -Math.cos(pitch) * Math.cos(heading), -Math.sin(pitch)]
  camera.position.copy(toThree(eye[0], eye[1], eye[2]))
  camera.up.set(0, 1, 0)
  camera.lookAt(toThree(eye[0] + forward[0], eye[1] + forward[1], eye[2] + forward[2]))
  camera.updateMatrixWorld(true)
}
