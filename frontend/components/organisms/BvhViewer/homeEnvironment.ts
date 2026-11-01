'use client'

import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'

export const HOME_ENVIRONMENT_CAMERA = {
  position: [3.6, 1.85, 4.5] as const,
  target: [0, 1.1, -1.8] as const,
  fov: 70,
  near: 0.05,
  far: 80,
}

export const HOME_ENVIRONMENT_HDR_URL = '/environments/home/lighting/lebombo_1k.hdr'

export type HomeEnvironment = {
  root: THREE.Group
  lights: THREE.Group
  floor: THREE.Mesh
  camera: typeof HOME_ENVIRONMENT_CAMERA
  ready: Promise<void>
  dispose: () => void
}

export type HomeEnvironmentOptions = {
  assetBase?: string
  /** Disable network loading when inspecting the procedural scene in isolation. */
  loadAssets?: boolean
}

type Vec3 = readonly [number, number, number]

/** A cutaway family LDK. Furniture leaves the centre and the +Z approach open. */
export const createHomeEnvironment = (
  options: HomeEnvironmentOptions = {},
): HomeEnvironment => {
  const root = new THREE.Group()
  root.name = 'HomeEnvironment'
  const lights = new THREE.Group()
  lights.name = 'HomeEnvironmentLights'
  root.add(lights)
  const geometries = new Set<THREE.BufferGeometry>()
  const materials = new Set<THREE.Material>()
  const textures = new Set<THREE.Texture>()
  const geometryCache = new Map<string, THREE.BufferGeometry>()
  let disposed = false

  const material = (name: string, props: THREE.MeshStandardMaterialParameters) => {
    const value = new THREE.MeshStandardMaterial({ envMapIntensity: 0.65, ...props })
    value.name = name
    materials.add(value)
    return value
  }
  const oak = material('WarmOak', { color: 0xc7a67b, roughness: 0.6 })
  const darkOak = material('DarkOak', { color: 0x6d4a2e, roughness: 0.52 })
  const cabinet = material('GraphiteKitchen', { color: 0x343b3b, roughness: 0.49 })
  const cabinetInset = material('KitchenRecess', { color: 0x202727, roughness: 0.6 })
  const wall = material('WarmPlaster', { color: 0xe7e2d8, roughness: 0.92 })
  const ceilingOak = material('OakCeiling', { color: 0xe4c5a1, roughness: 0.72 })
  const stone = material('HonedLightStone', { color: 0xd8d6cf, roughness: 0.37 })
  const linen = material('OatmealLinen', { color: 0xbcb5a7, roughness: 0.94 })
  const curtainMaterial = material('SheerCurtains', {
    color: 0xf5f0e4, roughness: 1, side: THREE.DoubleSide,
  })
  const cushionSand = material('SandCushions', { color: 0xcfc1aa, roughness: 0.98 })
  const cushionRust = material('RustCushions', { color: 0x98715d, roughness: 0.96 })
  const rugMaterial = material('WovenRug', { color: 0xb9b3a4, roughness: 1 })
  const black = material('SoftBlack', { color: 0x171b1b, roughness: 0.46 })
  const brushedMetal = material('BrushedSteel', { color: 0x919893, metalness: 0.82, roughness: 0.34 })
  const brass = material('AgedBrass', { color: 0x99805a, metalness: 0.8, roughness: 0.43 })
  const ceramic = material('CreamCeramic', { color: 0xd9d1bf, roughness: 0.48 })
  const terracotta = material('Terracotta', { color: 0x9d6e54, roughness: 0.88 })
  const green = material('LeafGreen', { color: 0x486345, roughness: 0.92, side: THREE.DoubleSide })
  const greenLight = material('FreshLeafGreen', { color: 0x698158, roughness: 0.96, side: THREE.DoubleSide })
  const warmGlow = material('WarmLampDiffuser', {
    color: 0xfff3d5, emissive: 0xffda91, emissiveIntensity: 1.8, roughness: 0.75,
  })
  const bookMaterials = [0xa49b86, 0x596d73, 0x805e50, 0xcbc2ad].map((color, i) =>
    material(`BookCover${i}`, { color, roughness: 0.92 }),
  )

  const cached = (key: string, build: () => THREE.BufferGeometry) => {
    let geometry = geometryCache.get(key)
    if (!geometry) {
      geometry = build()
      geometryCache.set(key, geometry)
      geometries.add(geometry)
    }
    return geometry
  }
  const mesh = (
    name: string, geometry: THREE.BufferGeometry, surface: THREE.Material,
    position: Vec3, parent: THREE.Object3D = root,
  ) => {
    geometries.add(geometry)
    const result = new THREE.Mesh(geometry, surface)
    result.name = name
    result.position.set(...position)
    result.castShadow = !(surface instanceof THREE.MeshStandardMaterial && surface.emissive.getHex() !== 0)
    result.receiveShadow = true
    parent.add(result)
    return result
  }
  const box = (
    name: string, size: Vec3, position: Vec3, surface: THREE.Material,
    radius = 0.025, parent: THREE.Object3D = root,
  ) => mesh(name, cached(`box:${size}:${radius}`, () => radius > 0
    ? new RoundedBoxGeometry(...size, 3, Math.min(radius, ...size.map((v) => v / 2)))
    : new THREE.BoxGeometry(...size)), surface, position, parent)
  const cylinder = (
    name: string, radiusTop: number, radiusBottom: number, height: number,
    position: Vec3, surface: THREE.Material, parent: THREE.Object3D = root,
  ) => mesh(name, cached(`cyl:${radiusTop}:${radiusBottom}:${height}`, () =>
    new THREE.CylinderGeometry(radiusTop, radiusBottom, height, 24)), surface, position, parent)
  const sphere = (
    name: string, size: Vec3, position: Vec3, surface: THREE.Material,
    parent: THREE.Object3D = root,
  ) => {
    const result = mesh(name, cached('sphere', () => new THREE.SphereGeometry(1, 20, 12)), surface, position, parent)
    result.scale.set(...size)
    return result
  }
  const vessel = (
    name: string, profile: readonly [number, number][], position: Vec3,
    surface: THREE.Material, parent: THREE.Object3D = root,
  ) => mesh(name, new THREE.LatheGeometry(profile.map(([x, y]) => new THREE.Vector2(x, y)), 32), surface, position, parent)

  // Keep a complete fallback even if a texture or furniture model cannot load.
  const woodData = new Uint8Array(256 * 256 * 4)
  for (let y = 0; y < 256; y++) {
    for (let x = 0; x < 256; x++) {
      const board = Math.floor(x / 32)
      const grain = Math.sin(x * 1.7 + Math.sin(y * 0.032 + board) * 2.6) * 3
      const seam = x % 32 < 1 ? -23 : 0
      const shift = Math.sin(board * 4.1) * 7 + grain + seam
      const offset = (y * 256 + x) * 4
      woodData[offset] = 191 + shift
      woodData[offset + 1] = 161 + shift
      woodData[offset + 2] = 120 + shift
      woodData[offset + 3] = 255
    }
  }
  const fallbackWood = new THREE.DataTexture(woodData, 256, 256)
  fallbackWood.colorSpace = THREE.SRGBColorSpace
  fallbackWood.wrapS = fallbackWood.wrapT = THREE.RepeatWrapping
  fallbackWood.repeat.set(5, 5)
  fallbackWood.needsUpdate = true
  textures.add(fallbackWood)
  const floorMaterial = material('OakFloor', { color: 0xffffff, map: fallbackWood, roughness: 1 })
  const floor = mesh('HomeOakFloor', new THREE.PlaneGeometry(10.12, 10.12), floorMaterial, [0, 0, 0])
  floor.rotation.x = -Math.PI / 2
  floor.castShadow = false

  // Inward-facing planes disappear from outside when orbiting the cutaway room.
  const wallPlane = (name: string, width: number, height: number, position: Vec3, yaw = 0) => {
    const panel = mesh(name, new THREE.PlaneGeometry(width, height), wall, position)
    panel.rotation.y = yaw
  }
  wallPlane('LeftWall', 10.12, 2.8, [-5, 1.4, 0], Math.PI / 2)
  wallPlane('RightWall', 10.12, 2.8, [5, 1.4, 0], -Math.PI / 2)
  wallPlane('BackWallKitchen', 3.2, 2.8, [-3.4, 1.4, -5])
  wallPlane('BackWallRightPier', 0.4, 2.8, [4.8, 1.4, -5])
  const plasterCeiling = mesh('PlasterCeiling', new THREE.PlaneGeometry(6.2, 10), wall, [-1.9, 2.8, 0])
  plasterCeiling.rotation.x = Math.PI / 2
  const woodCeiling = mesh('LivingOakCeiling', new THREE.PlaneGeometry(3.8, 10), ceilingOak, [3.1, 2.8, 0])
  woodCeiling.rotation.x = Math.PI / 2
  box('WindowHeader', [6.4, 0.14, 0.16], [1.4, 2.73, -5.02], wall, 0)
  box('WindowSill', [6.45, 0.1, 0.28], [1.4, 0.05, -4.94], stone, 0.015)
  for (const side of [-1, 1]) {
    box(`SideSkirting${side}`, [0.035, 0.085, 10], [side * 4.985, 0.0425, 0], ceramic, 0.005)
    box(`SideCeilingTrim${side}`, [0.16, 0.12, 9.9], [side * 4.92, 2.73, 0], wall, 0.012)
  }
  box('BackCeilingTrim', [9.9, 0.12, 0.15], [0, 2.73, -4.92], wall, 0.012)
  box('LivingCoveLightStrip', [0.012, 0.012, 9.72], [4.815, 2.72, 0], warmGlow, 0.003)
  box('BackCoveLightStrip', [9.7, 0.012, 0.012], [0, 2.71, -4.8], warmGlow, 0.003)
  for (const z of [-3.5, 0, 3.5]) {
    const cove = new THREE.PointLight(0xffdfb5, 0.42, 4.2, 2)
    cove.position.set(4.65, 2.57, z)
    lights.add(cove)
  }

  const glass = new THREE.MeshPhysicalMaterial({
    color: 0xd5e4df, transparent: true, opacity: 0.22, transmission: 0.7,
    thickness: 0.035, roughness: 0.06, metalness: 0, ior: 1.45, envMapIntensity: 0.65, side: THREE.DoubleSide,
  })
  materials.add(glass)
  for (let i = 0; i < 4; i++) {
    const x = -1.79 + i * 1.6
    box(`WindowMullion${i}`, [0.045, 2.56, 0.06], [x, 1.37, -4.97], black, 0.006)
    const pane = box(`WindowGlass${i}`, [1.56, 2.5, 0.018], [x + 0.8, 1.37, -4.985], glass, 0)
    pane.castShadow = false
  }
  box('WindowMullionEnd', [0.045, 2.56, 0.06], [4.61, 1.37, -4.97], black, 0.006)
  const rail = cylinder('CurtainRail', 0.018, 0.018, 6.62, [1.4, 2.68, -4.72], brushedMetal)
  rail.rotation.z = Math.PI / 2
  for (const [name, center] of [['LeftCurtain', -1.32], ['RightCurtain', 4.16]] as const) {
    const curtain = new THREE.PlaneGeometry(0.93, 2.5, 56, 18)
    const positions = curtain.attributes.position
    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i)
      positions.setZ(i, Math.cos((x + 0.465) * Math.PI * 16) * 0.043)
    }
    curtain.computeVertexNormals()
    mesh(name, curtain, curtainMaterial, [center, 1.38, -4.68])
  }

  // A shallow exterior gives the large windows actual depth rather than a flat image.
  const grass = material('GardenGrass', { color: 0x65764f, roughness: 1 })
  mesh('GardenLawn', new THREE.BoxGeometry(12, 0.08, 6), grass, [0, -0.075, -8.1]).castShadow = false
  const patio = material('GardenPaving', { color: 0xaaa79a, roughness: 0.97 })
  box('GardenPatio', [8, 0.045, 1.35], [0.8, -0.008, -5.76], patio, 0.008)
  const skyMaterial = new THREE.MeshBasicMaterial({ color: 0xcbd9d7, toneMapped: false })
  materials.add(skyMaterial)
  const sky = mesh('GardenDistantSky', new THREE.PlaneGeometry(16, 6), skyMaterial, [0, 2.4, -11.2])
  sky.castShadow = sky.receiveShadow = false

  // Left wall kitchen: dark slab fronts, an inset toe kick, stone and steel details.
  box('KitchenToeKick', [0.67, 0.11, 3.38], [-4.66, 0.055, -1.62], cabinetInset, 0.005)
  box('KitchenLowerRun', [0.67, 0.78, 3.42], [-4.64, 0.5, -1.62], cabinet, 0.015)
  box('KitchenSideCounter', [0.79, 0.055, 3.5], [-4.59, 0.92, -1.62], stone, 0.018)
  for (let i = 0; i < 5; i++) {
    const z = -2.95 + i * 0.65
    box(`KitchenLowerDoor${i}`, [0.026, 0.68, 0.63], [-4.285, 0.5, z], cabinet, 0.007)
    box(`KitchenHandle${i}`, [0.035, 0.018, 0.29], [-4.255, 0.765, z], brushedMetal, 0.006)
  }
  box('KitchenWallCabinets', [0.4, 0.7, 2.48], [-4.78, 2.12, -1.6], cabinet, 0.017)
  for (let i = 0; i < 4; i++) {
    box(`KitchenUpperDoor${i}`, [0.025, 0.66, 0.595], [-4.566, 2.12, -2.525 + i * 0.615], cabinet, 0.008)
  }
  box('KitchenTaskLight', [0.34, 0.012, 2.37], [-4.73, 1.758, -1.6], warmGlow, 0.005)
  box('KitchenBackCabinet', [2.35, 0.88, 0.66], [-3.03, 0.44, -4.63], cabinet, 0.017)
  box('KitchenBackCounter', [2.43, 0.055, 0.77], [-3.03, 0.92, -4.59], stone, 0.018)
  for (let i = 0; i < 4; i++) {
    box(`BackCabinetDoor${i}`, [0.55, 0.72, 0.023], [-3.9 + i * 0.58, 0.48, -4.283], cabinet, 0.008)
  }
  box('KitchenBackSplash', [2.43, 0.42, 0.023], [-3.03, 1.16, -4.984], stone, 0.005)
  box('Refrigerator', [0.74, 2.35, 0.81], [-4.61, 1.175, -4.45], cabinetInset, 0.038)
  box('FridgeUpperDoor', [0.032, 1.42, 0.72], [-4.223, 1.52, -4.45], brushedMetal, 0.012)
  box('FridgeLowerDoor', [0.032, 0.72, 0.72], [-4.223, 0.425, -4.45], brushedMetal, 0.012)
  box('FridgeHandle', [0.045, 0.66, 0.036], [-4.185, 1.42, -4.15], black, 0.009)

  const cooktop = box('InductionCooktop', [0.6, 0.012, 0.56], [-4.57, 0.955, -2.25], black, 0.009)
  cooktop.castShadow = false
  for (const x of [-4.73, -4.43]) for (const z of [-2.4, -2.1]) {
    const ring = mesh('CooktopRing', cached('hob-ring', () => new THREE.TorusGeometry(0.084, 0.0018, 6, 36)), brushedMetal, [x, 0.963, z])
    ring.rotation.x = -Math.PI / 2
  }
  box('ExtractorHood', [0.6, 0.12, 0.66], [-4.57, 1.72, -2.25], brushedMetal, 0.024)
  box('ExtractorFlue', [0.23, 0.76, 0.26], [-4.68, 2.15, -2.25], brushedMetal, 0.01)
  box('SinkRim', [0.63, 0.025, 0.5], [-2.67, 0.956, -4.54], brushedMetal, 0.018)
  box('SinkBowl', [0.52, 0.025, 0.38], [-2.67, 0.958, -4.54], cabinetInset, 0.038)
  const faucetPath = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-2.67, 0.96, -4.82), new THREE.Vector3(-2.67, 1.27, -4.82),
    new THREE.Vector3(-2.67, 1.32, -4.63), new THREE.Vector3(-2.67, 1.15, -4.6),
  ])
  mesh('KitchenFaucet', new THREE.TubeGeometry(faucetPath, 24, 0.014, 10, false), brushedMetal, [0, 0, 0])

  box('KitchenIslandBase', [1.12, 0.86, 2.1], [-3.04, 0.45, -1.28], cabinet, 0.035)
  box('KitchenIslandToeKick', [0.96, 0.1, 1.92], [-3.04, 0.05, -1.28], cabinetInset, 0.008)
  box('KitchenIslandStone', [1.24, 0.065, 2.23], [-3.04, 0.925, -1.28], stone, 0.024)
  for (let i = 0; i < 3; i++) {
    box(`IslandDrawer${i}`, [0.025, 0.22, 0.64], [-2.47, 0.7 - i * 0.25, -1.91], cabinet, 0.008)
  }
  box('IslandCuttingBoard', [0.34, 0.025, 0.26], [-3.12, 0.973, -0.74], darkOak, 0.027)
  vessel('IslandFruitBowl', [[0.02, 0], [0.12, 0.01], [0.16, 0.055], [0.15, 0.08]], [-3.15, 0.96, -1.8], ceramic)
  for (let i = 0; i < 4; i++) {
    sphere(`BowlFruit${i}`, [0.037, 0.035, 0.037], [-3.21 + (i % 2) * 0.095, 1.04, -1.84 + Math.floor(i / 2) * 0.07], i % 2 ? cushionRust : terracotta)
  }

  const makePlant = (name: string, position: Vec3, height: number, potRadius: number) => {
    const group = new THREE.Group()
    group.name = name
    group.position.set(...position)
    root.add(group)
    vessel(`${name}Pot`, [[potRadius * 0.6, 0], [potRadius * 0.85, 0.015], [potRadius, height * 0.29], [potRadius * 0.97, height * 0.3]], [0, 0, 0], terracotta, group)
    cylinder(`${name}Soil`, potRadius * 0.91, potRadius * 0.91, 0.012, [0, height * 0.285, 0], darkOak, group)
    for (let i = 0; i < 9; i++) {
      const angle = i * 2.4
      const stemHeight = height * (0.48 + (i % 3) * 0.15)
      const stem = cylinder(`${name}Stem${i}`, 0.005, 0.007, stemHeight, [Math.sin(angle) * 0.05, height * 0.25 + stemHeight / 2, Math.cos(angle) * 0.05], green, group)
      stem.rotation.z = Math.sin(angle) * 0.24
      const leaf = sphere(`${name}Leaf${i}`, [height * 0.11, height * 0.24, height * 0.025],
        [Math.sin(angle) * height * 0.22, height * (0.55 + (i % 3) * 0.13), Math.cos(angle) * height * 0.22], i % 2 ? green : greenLight, group)
      leaf.rotation.set(0.5, angle, Math.sin(angle) * 0.65)
    }
    return group
  }
  makePlant('KitchenHerb', [-2.01, 0.96, -4.6], 0.3, 0.065)
  makePlant('LivingPlant', [4.48, 0, -4.24], 1.42, 0.21)
  makePlant('DiningPlant', [-4.52, 0, 4.08], 1.04, 0.16)

  // Contemporary upholstered sectional, with separate cushions and thin wood feet.
  const sofa = new THREE.Group()
  sofa.name = 'LivingSofa'
  sofa.position.set(2.45, 0, -2.55)
  sofa.rotation.y = Math.PI / 2
  root.add(sofa)
  box('SofaBase', [2.55, 0.23, 0.91], [0, 0.25, 0], linen, 0.09, sofa)
  box('SofaBack', [2.49, 0.57, 0.2], [0, 0.6, -0.37], linen, 0.095, sofa)
  for (const side of [-1, 1]) {
    box(`SofaArm${side}`, [0.2, 0.41, 0.94], [side * 1.175, 0.48, 0], linen, 0.082, sofa)
    for (const z of [-0.3, 0.31]) cylinder('SofaFoot', 0.033, 0.026, 0.14, [side * 1.07, 0.07, z], darkOak, sofa)
  }
  for (let i = 0; i < 3; i++) {
    box(`SofaSeatCushion${i}`, [0.71, 0.17, 0.7], [-0.75 + i * 0.75, 0.43, 0.04], linen, 0.075, sofa)
    const back = box(`SofaBackCushion${i}`, [0.72, 0.43, 0.15], [-0.75 + i * 0.75, 0.7, -0.245], cushionSand, 0.078, sofa)
    back.rotation.x = -0.11
  }
  box('SofaChaiseBase', [0.77, 0.29, 1.48], [0.77, 0.27, 0.74], linen, 0.095, sofa)
  box('SofaChaiseCushion', [0.71, 0.17, 1.43], [0.77, 0.46, 0.77], linen, 0.078, sofa)
  for (const [i, x, color] of [[0, -0.9, cushionRust], [1, 0.64, cushionSand]] as const) {
    const pillow = box(`SofaLoosePillow${i}`, [0.38, 0.37, 0.14], [x, 0.69, -0.09], color, 0.072, sofa)
    pillow.rotation.set(-0.13, 0, i ? -0.15 : 0.15)
  }
  box('LivingRug', [2.72, 0.018, 2.8], [3.1, 0.012, -2.35], rugMaterial, 0.045)
  for (let i = 0; i < 12; i++) {
    box(`RugWeave${i}`, [2.66, 0.0018, 0.014], [3.1, 0.022, -3.54 + i * 0.21], cushionSand, 0.002)
  }
  const table = cylinder('CoffeeTableTop', 0.52, 0.52, 0.048, [3.75, 0.355, -2], darkOak)
  table.scale.z = 0.79
  cylinder('CoffeeTablePedestal', 0.13, 0.23, 0.325, [3.75, 0.1625, -2], darkOak)
  const tray = cylinder('CoffeeTableTray', 0.16, 0.17, 0.022, [3.71, 0.391, -1.97], ceramic)
  tray.scale.z = 0.82
  vessel('CoffeeCup', [[0.023, 0], [0.03, 0.055], [0.027, 0.062]], [3.67, 0.405, -2], ceramic)
  box('CoffeeTableBook', [0.23, 0.026, 0.17], [3.94, 0.393, -1.98], bookMaterials[1], 0.004)

  box('MediaConsole', [0.39, 0.37, 2.15], [4.78, 0.235, -1.97], oak, 0.024)
  for (let i = 0; i < 3; i++) box(`MediaDoor${i}`, [0.015, 0.3, 0.64], [4.575, 0.23, -2.65 + i * 0.68], oak, 0.008)
  box('TelevisionFrame', [0.065, 0.96, 1.7], [4.935, 1.28, -1.97], black, 0.014)
  const screenMaterial = new THREE.MeshStandardMaterial({
    color: 0x172b33, emissive: 0x28434d, emissiveIntensity: 0.35, roughness: 0.18, metalness: 0.12,
    envMapIntensity: 0.65,
  })
  materials.add(screenMaterial)
  const screen = mesh('TelevisionScreen', new THREE.PlaneGeometry(1.626, 0.887), screenMaterial, [4.897, 1.28, -1.97])
  screen.rotation.y = -Math.PI / 2
  screen.castShadow = false
  for (let i = 0; i < 5; i++) {
    box(`MediaBooks${i}`, [0.23, 0.025, 0.17], [4.75, 0.434 + i * 0.026, -1.17], bookMaterials[i % 4], 0.003)
  }
  vessel('MediaVase', [[0.055, 0], [0.075, 0.09], [0.042, 0.22], [0.032, 0.27]], [4.77, 0.425, -2.78], ceramic)

  // Four dining seats remain entirely left of the central activity corridor.
  const dining = new THREE.Group()
  dining.name = 'DiningArea'
  dining.position.set(-3.25, 0, 2.42)
  root.add(dining)
  box('DiningTableTop', [1.46, 0.07, 2.15], [0, 0.765, 0], oak, 0.038, dining)
  box('DiningTableApron', [1.22, 0.12, 1.91], [0, 0.68, 0], oak, 0.012, dining)
  for (const x of [-0.57, 0.57]) for (const z of [-0.88, 0.88]) {
    cylinder('DiningTableLeg', 0.032, 0.047, 0.67, [x, 0.335, z], darkOak, dining)
  }
  const diningSeats: THREE.Group[] = []
  const proceduralChairs: THREE.Group[] = []
  for (const side of [-1, 1]) for (const end of [-1, 1]) {
    const chair = new THREE.Group()
    chair.name = `DiningChair${side},${end}`
    chair.position.set(side * 1.03, 0, end * 0.63)
    chair.rotation.y = side < 0 ? Math.PI / 2 : -Math.PI / 2
    dining.add(chair)
    diningSeats.push(chair)
    const fallback = new THREE.Group()
    fallback.name = 'ProceduralDiningChair'
    chair.add(fallback)
    proceduralChairs.push(fallback)
    box('DiningChairSeat', [0.48, 0.055, 0.45], [0, 0.445, 0], oak, 0.027, fallback)
    box('DiningChairSeatPad', [0.43, 0.04, 0.4], [0, 0.488, 0], cushionSand, 0.025, fallback)
    const back = box('DiningChairBack', [0.47, 0.36, 0.06], [0, 0.75, -0.19], oak, 0.038, fallback)
    back.rotation.x = -0.09
    for (const x of [-0.18, 0.18]) for (const z of [-0.17, 0.17]) {
      const leg = cylinder('DiningChairLeg', 0.018, 0.014, z < 0 ? 0.78 : 0.44, [x, z < 0 ? 0.39 : 0.22, z], darkOak, fallback)
      leg.rotation.z = -x * 0.23
    }
  }
  vessel('DiningVase', [[0.047, 0], [0.077, 0.07], [0.052, 0.16], [0.038, 0.21]], [0, 0.804, 0], ceramic, dining)
  for (let i = 0; i < 5; i++) {
    const stem = cylinder('DiningFlowerStem', 0.002, 0.002, 0.35, [Math.sin(i * 2.4) * 0.02, 1.06, Math.cos(i * 2.4) * 0.02], green, dining)
    stem.rotation.z = Math.sin(i * 2.4) * 0.14
    sphere('DiningDriedFlower', [0.018, 0.042, 0.018], [Math.sin(i * 2.4) * 0.06, 1.24, Math.cos(i * 2.4) * 0.05], cushionSand, dining)
  }
  for (const z of [-0.62, 0.62]) {
    const plate = cylinder('DiningPlate', 0.15, 0.145, 0.015, [-0.43, 0.809, z], ceramic, dining)
    plate.receiveShadow = true
    vessel('DiningGlass', [[0.024, 0], [0.03, 0.1], [0.032, 0.12]], [-0.4, 0.82, z - 0.24], glass, dining)
  }

  const pendant = (name: string, position: Vec3, radius: number, surface: THREE.Material) => {
    cylinder(`${name}Cable`, 0.003, 0.003, 2.77 - position[1], [position[0], (2.77 + position[1]) / 2, position[2]], black)
    vessel(`${name}Shade`, [[radius, 0], [radius * 0.98, 0.026], [radius * 0.82, 0.075], [radius * 0.28, radius * 0.64], [radius * 0.1, radius * 0.74]], position, surface)
    cylinder(`${name}Diffuser`, radius * 0.89, radius * 0.89, 0.009, [position[0], position[1] + 0.009, position[2]], warmGlow)
    const light = new THREE.SpotLight(0xffe0ac, 15, 5.5, Math.PI * 0.34, 0.65, 2)
    light.name = `${name}Light`
    light.position.set(position[0], position[1] - 0.03, position[2])
    light.target.position.set(position[0], 0, position[2])
    lights.add(light, light.target)
  }
  pendant('DiningPendant', [-3.25, 2.02, 2.42], 0.29, ceramic)
  pendant('IslandPendantOne', [-3.04, 2.14, -1.94], 0.15, black)
  pendant('IslandPendantTwo', [-3.04, 2.14, -0.66], 0.15, black)
  const floorLamp = new THREE.Group()
  floorLamp.name = 'LivingFloorLamp'
  floorLamp.position.set(2.04, 0, -4.22)
  root.add(floorLamp)
  cylinder('FloorLampBase', 0.14, 0.16, 0.026, [0, 0.013, 0], brass, floorLamp)
  cylinder('FloorLampStem', 0.011, 0.011, 1.49, [0, 0.76, 0], brass, floorLamp)
  vessel('FloorLampShade', [[0.24, 0], [0.2, 0.31]], [0, 1.38, 0], curtainMaterial, floorLamp)
  const lamp = new THREE.PointLight(0xffd59a, 1.4, 3, 2)
  lamp.position.set(2.04, 1.53, -4.22)
  lights.add(lamp)

  // The viewer owns the HDR and shadow-casting window sunlight.
  const kitchenLight = new THREE.PointLight(0xffe0b3, 0.65, 3.3, 2)
  kitchenLight.position.set(-4.42, 1.8, -1.57)
  lights.add(kitchenLight)

  const trackImported = (object: THREE.Object3D) => {
    object.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return
      geometries.add(child.geometry)
      child.castShadow = child.receiveShadow = true
      for (const surface of Array.isArray(child.material) ? child.material : [child.material]) {
        materials.add(surface)
        if (surface instanceof THREE.MeshStandardMaterial) surface.envMapIntensity = 0.65
        for (const value of Object.values(surface)) {
          if (value instanceof THREE.Texture) textures.add(value)
        }
      }
    })
  }
  const dispose = () => {
    if (disposed) return
    disposed = true
    root.removeFromParent()
    for (const geometry of geometries) geometry.dispose()
    for (const surface of materials) surface.dispose()
    for (const texture of textures) texture.dispose()
  }
  const assetBase = (options.assetBase ?? '/environments/home').replace(/\/$/, '')
  const loadPbr = async (
    folder: string, surfaces: THREE.MeshStandardMaterial[], repeat: readonly [number, number], normalStrength: number,
    useAlbedo = true,
  ) => {
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
    if (disposed || maps.some((texture) => texture === null)) return
    for (const surface of surfaces) {
      if (useAlbedo) surface.map = maps[0]
      surface.normalMap = maps[1]
      surface.roughnessMap = maps[2]
      surface.normalScale.setScalar(normalStrength)
      surface.needsUpdate = true
    }
  }
  const loadDiningChairs = async () => {
    const gltf = await new GLTFLoader().loadAsync(`${assetBase}/models/dining-chair/chair.gltf`)
    trackImported(gltf.scene)
    if (disposed) {
      gltf.scene.traverse((child) => {
        if (!(child instanceof THREE.Mesh)) return
        child.geometry.dispose()
        for (const surface of Array.isArray(child.material) ? child.material : [child.material]) {
          for (const value of Object.values(surface)) if (value instanceof THREE.Texture) value.dispose()
          surface.dispose()
        }
      })
      return
    }
    gltf.scene.updateMatrixWorld(true)
    const bounds = new THREE.Box3().setFromObject(gltf.scene)
    const size = bounds.getSize(new THREE.Vector3())
    const center = bounds.getCenter(new THREE.Vector3())
    if (!Number.isFinite(size.y) || size.y <= 0.01) return
    const scale = 0.95 / size.y
    // This CC0 chair has pale painted wood; retain its grain while warming it
    // to the oak dining furniture rather than leaving a stark white silhouette.
    gltf.scene.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return
      for (const surface of Array.isArray(child.material) ? child.material : [child.material]) {
        if (surface instanceof THREE.MeshStandardMaterial) surface.color.set(0xc5a277)
      }
    })
    const prototype = new THREE.Group()
    gltf.scene.position.sub(new THREE.Vector3(center.x, bounds.min.y, center.z))
    prototype.add(gltf.scene)
    prototype.scale.setScalar(scale)
    for (let i = 0; i < diningSeats.length; i++) {
      diningSeats[i].add(prototype.clone(true))
      proceduralChairs[i].visible = false
    }
  }
  const loadScreenPhoto = async () => {
    const texture = await new THREE.TextureLoader().loadAsync(`${assetBase}/tv/mountain-lake.jpg`)
    if (disposed) { texture.dispose(); return }
    textures.add(texture)
    texture.colorSpace = THREE.SRGBColorSpace
    // The panoramic source is 3840x972: take a centred television-shaped crop.
    texture.repeat.set((1.626 / 0.887) / (3840 / 972), 1)
    texture.offset.x = (1 - texture.repeat.x) / 2
    screenMaterial.color.set(0xffffff)
    screenMaterial.emissive.set(0xffffff)
    screenMaterial.emissiveIntensity = 0.32
    screenMaterial.map = texture
    screenMaterial.emissiveMap = texture
    screenMaterial.needsUpdate = true
  }
  const loadGardenPhoto = async () => {
    const texture = await new THREE.TextureLoader().loadAsync(`${assetBase}/garden/greenwich-park.jpg`)
    if (disposed) { texture.dispose(); return }
    // The source panorama is 8K. A 2K background retains ample window detail
    // without allocating an 8K GPU texture alongside the avatar's skin maps.
    const image = texture.image as HTMLImageElement
    const canvas = document.createElement('canvas')
    canvas.width = 2048
    canvas.height = 1024
    const context = canvas.getContext('2d')
    if (context) {
      context.drawImage(image, 0, 0, canvas.width, canvas.height)
      texture.image = canvas
    }
    textures.add(texture)
    texture.colorSpace = THREE.SRGBColorSpace
    // Preserve the photograph's proportions and choose the leafy oak part of
    // the 360-degree panorama rather than stretching the complete panorama.
    texture.repeat.set(0.45, 0.45 * 2 / (16 / 6))
    texture.offset.set(0.2, 0.49 - texture.repeat.y / 2)
    texture.needsUpdate = true
    skyMaterial.color.set(0xffffff)
    skyMaterial.map = texture
    skyMaterial.needsUpdate = true
  }
  const loadFerns = async () => {
    const gltf = await new GLTFLoader().loadAsync(`${assetBase}/models/fern/fern.gltf`)
    trackImported(gltf.scene)
    if (disposed) {
      gltf.scene.traverse((child) => {
        if (!(child instanceof THREE.Mesh)) return
        child.geometry.dispose()
        for (const surface of Array.isArray(child.material) ? child.material : [child.material]) {
          for (const value of Object.values(surface)) if (value instanceof THREE.Texture) value.dispose()
          surface.dispose()
        }
      })
      return
    }
    // The glTF references an RGB diffuse JPEG; its separate leaf mask is
    // necessary to make the frond cards transparent around their silhouettes.
    const alpha = await new THREE.TextureLoader().loadAsync(`${assetBase}/models/fern/textures/fern-alpha.jpg`)
    if (disposed) { alpha.dispose(); return }
    textures.add(alpha)
    alpha.flipY = false
    gltf.scene.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return
      for (const surface of Array.isArray(child.material) ? child.material : [child.material]) {
        if (!(surface instanceof THREE.MeshStandardMaterial)) continue
        surface.alphaMap = alpha
        surface.alphaTest = 0.5
        surface.side = THREE.DoubleSide
        surface.needsUpdate = true
      }
    })
    const template = gltf.scene.getObjectByName('fern_02_a') ?? gltf.scene
    template.updateWorldMatrix(true, true)
    const bounds = new THREE.Box3().setFromObject(template)
    const size = bounds.getSize(new THREE.Vector3())
    const center = bounds.getCenter(new THREE.Vector3())
    if (!Number.isFinite(size.y) || size.y < 0.01) return
    for (const [name, potHeight, width] of [
      ['LivingPlant', 1.42 * 0.285, 0.96], ['DiningPlant', 1.04 * 0.285, 0.76],
    ] as const) {
      const plant = root.getObjectByName(name)
      if (!plant) continue
      const foliage = new THREE.Group()
      foliage.name = `${name}DetailedFern`
      const model = template.clone(true)
      model.position.sub(new THREE.Vector3(center.x, bounds.min.y, center.z))
      const scale = width / Math.max(size.x, size.z, 0.01)
      foliage.scale.setScalar(scale)
      foliage.position.y = potHeight - 0.02
      foliage.add(model)
      for (const child of plant.children) {
        if (child.name.startsWith(`${name}Leaf`) || child.name.startsWith(`${name}Stem`)) child.visible = false
      }
      plant.add(foliage)
    }
  }
  const canLoadAssets = options.loadAssets !== false && typeof document !== 'undefined'
  const ready = canLoadAssets ? Promise.allSettled([
    loadPbr('wood', [floorMaterial], [10.12 / 1.7, 10.12 / 1.7], 0.24),
    loadPbr('wood', [oak, darkOak], [0.95, 1.25], 0.15),
    loadPbr('wood', [ceilingOak], [3.8 / 1.7, 10 / 1.7], 0.13),
    loadPbr('stone', [stone], [1, 1], 0.15),
    loadPbr('plaster', [wall], [4, 4], 0.09, false),
    loadPbr('fabric', [linen, cushionSand, cushionRust, rugMaterial], [4, 4], 0.32, false),
    loadDiningChairs(),
    loadScreenPhoto(),
    loadGardenPhoto(),
    loadFerns(),
  ]).then((results) => {
    for (const result of results) {
      if (result.status === 'rejected' && !disposed) console.warn('[HomeEnvironment] asset fallback retained', result.reason)
    }
  }) : Promise.resolve()

  return { root, lights, floor, camera: HOME_ENVIRONMENT_CAMERA, ready, dispose }
}

export const disposeHomeEnvironment = (environment: HomeEnvironment) => environment.dispose()
