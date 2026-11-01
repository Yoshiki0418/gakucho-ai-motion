import * as THREE from 'three'
import type { RoomNavigation, RoomPoint } from './roomNavigation'

export type RoomSceneObstacle = {
  id: string
  label: string
  /** Box centre; chairs instead use their origin on the floor. */
  position: [number, number, number]
  size: [number, number, number]
  yaw?: number
  chair?: { seatHeight: number }
}
export type RoomSceneDestination = {
  id: string
  label: string
  position: RoomPoint
  heading?: number
  aliases?: string[]
}
export type RoomSceneDefinition = {
  version: 1
  name: string
  floor: { outline: RoomPoint[]; holes?: RoomPoint[][] }
  obstacles: RoomSceneObstacle[]
  destinations: RoomSceneDestination[]
  spawn?: RoomPoint & { heading?: number }
  units?: 'm'
  upAxis?: 'Y'
  floorHeight?: 0
}
export type RoomScene = {
  root: THREE.Group
  definition: RoomSceneDefinition
  spawn: RoomPoint & { heading: number }
  dispose: () => void
}
export type RoomSceneOptions = { floorMaterial?: THREE.Material }

const EPSILON = 1e-7
const MAX_COORDINATE = 100
const MAX_ITEMS = 128
const ACTOR_CLEARANCE = 0.24 + 0.035
const failure = (field: string, message: string): never => {
  throw new Error(`部屋JSON: ${field} ${message}`)
}
const object = (value: unknown, field: string): Record<string, unknown> => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return failure(field, 'はオブジェクトで指定してください')
  return value as Record<string, unknown>
}
const keys = (value: Record<string, unknown>, allowed: string[], field: string) => {
  const unknown = Object.keys(value).find((key) => !allowed.includes(key))
  if (unknown) failure(`${field}.${unknown}`, 'はこの形式では使用できません')
}
const text = (value: unknown, field: string, limit = 100): string => {
  if (typeof value !== 'string' || !value.trim() || value.length > limit || /[\u0000-\u001f\u007f]/.test(value)) {
    return failure(field, `は1〜${limit}文字の文字列で指定してください`)
  }
  return value.trim()
}
const number = (value: unknown, field: string, limit = MAX_COORDINATE): number => {
  if (typeof value !== 'number' || !Number.isFinite(value) || Math.abs(value) > limit) {
    return failure(field, `は絶対値${limit}以下の有限数で指定してください`)
  }
  return value
}
const array = (value: unknown, field: string, max = MAX_ITEMS): unknown[] => {
  if (!Array.isArray(value) || value.length > max) return failure(field, `は${max}件以下の配列で指定してください`)
  return value
}
const point = (value: unknown, field: string): RoomPoint => {
  const record = object(value, field)
  keys(record, ['x', 'z'], field)
  return { x: number(record.x, `${field}.x`), z: number(record.z, `${field}.z`) }
}
const tuple = (value: unknown, field: string, positive = false): [number, number, number] => {
  const values = array(value, field, 3)
  if (values.length !== 3) return failure(field, 'は3つの数値で指定してください')
  const result = values.map((item, i) => number(item, `${field}[${i}]`)) as [number, number, number]
  if (positive && result.some((item) => item <= 0)) return failure(field, 'の幅・高さ・奥行きはすべて正にしてください')
  return result
}
const pointDistance = (a: RoomPoint, b: RoomPoint) => Math.hypot(a.x - b.x, a.z - b.z)
const cross = (a: RoomPoint, b: RoomPoint, c: RoomPoint) =>
  (b.x - a.x) * (c.z - a.z) - (b.z - a.z) * (c.x - a.x)
const edgeDistance = (p: RoomPoint, a: RoomPoint, b: RoomPoint) => {
  const dx = b.x - a.x, dz = b.z - a.z
  const squared = dx * dx + dz * dz
  const t = squared > 0 ? THREE.MathUtils.clamp(((p.x - a.x) * dx + (p.z - a.z) * dz) / squared, 0, 1) : 0
  return Math.hypot(p.x - a.x - t * dx, p.z - a.z - t * dz)
}
const onSegment = (p: RoomPoint, a: RoomPoint, b: RoomPoint) => edgeDistance(p, a, b) <= EPSILON
const intersects = (a: RoomPoint, b: RoomPoint, c: RoomPoint, d: RoomPoint) => {
  const ac = cross(a, b, c), ad = cross(a, b, d), ca = cross(c, d, a), cb = cross(c, d, b)
  if (((ac > EPSILON && ad < -EPSILON) || (ac < -EPSILON && ad > EPSILON))
    && ((ca > EPSILON && cb < -EPSILON) || (ca < -EPSILON && cb > EPSILON))) return true
  return onSegment(c, a, b) || onSegment(d, a, b) || onSegment(a, c, d) || onSegment(b, c, d)
}
const inside = (p: RoomPoint, polygon: RoomPoint[], allowBoundary = false) => {
  let result = false
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const a = polygon[j], b = polygon[i]
    if (onSegment(p, a, b)) return allowBoundary
    if ((a.z > p.z) !== (b.z > p.z) && p.x < (b.x - a.x) * (p.z - a.z) / (b.z - a.z) + a.x) result = !result
  }
  return result
}
const area = (polygon: RoomPoint[]) => polygon.reduce((sum, p, i) => {
  const next = polygon[(i + 1) % polygon.length]
  return sum + p.x * next.z - next.x * p.z
}, 0) / 2
const ring = (value: unknown, field: string): RoomPoint[] => {
  const result = array(value, field).map((entry, i) => point(entry, `${field}[${i}]`))
  // Accept a conventional explicitly closed ring, while storing each vertex once.
  if (result.length > 3 && pointDistance(result[0], result[result.length - 1]) <= EPSILON) result.pop()
  if (result.length < 3) return failure(field, 'には3点以上が必要です')
  for (let i = 0; i < result.length; i++) {
    const a = result[i], b = result[(i + 1) % result.length]
    if (pointDistance(a, b) <= EPSILON) failure(field, 'に長さ0の辺があります')
    const prior = result[(i - 1 + result.length) % result.length]
    if (Math.abs(cross(prior, a, b)) <= EPSILON
      && (a.x - prior.x) * (b.x - a.x) + (a.z - prior.z) * (b.z - a.z) < 0) failure(field, 'の辺が折り返して重なっています')
    for (let j = i + 1; j < result.length; j++) {
      if (j === i + 1 || (i === 0 && j === result.length - 1)) continue
      if (intersects(a, b, result[j], result[(j + 1) % result.length])) failure(field, 'の辺が交差または接触しています')
    }
  }
  if (Math.abs(area(result)) <= EPSILON) return failure(field, 'の面積が0です')
  return result
}
const ringsIntersect = (a: RoomPoint[], b: RoomPoint[]) =>
  a.some((point, i) => b.some((other, j) => intersects(point, a[(i + 1) % a.length], other, b[(j + 1) % b.length])))
const floorContains = (p: RoomPoint, floor: RoomSceneDefinition['floor']) =>
  inside(p, floor.outline, true) && !(floor.holes ?? []).some((hole) => inside(p, hole, true))
const ringClearance = (p: RoomPoint, polygon: RoomPoint[]) => Math.min(
  ...polygon.map((vertex, i) => edgeDistance(p, vertex, polygon[(i + 1) % polygon.length])),
)
const clearSpawn = (p: RoomPoint, definition: RoomSceneDefinition) => {
  if (!floorContains(p, definition.floor) || ringClearance(p, definition.floor.outline) < ACTOR_CLEARANCE - EPSILON
    || (definition.floor.holes ?? []).some((hole) => ringClearance(p, hole) < ACTOR_CLEARANCE - EPSILON)) return false
  return definition.obstacles.every((obstacle) => {
    const yaw = obstacle.yaw ?? 0
    const dx = p.x - obstacle.position[0], dz = p.z - obstacle.position[2]
    const localX = Math.cos(yaw) * dx - Math.sin(yaw) * dz
    const localZ = Math.sin(yaw) * dx + Math.cos(yaw) * dz
    const outsideX = Math.max(Math.abs(localX) - obstacle.size[0] / 2, 0)
    const outsideZ = Math.max(Math.abs(localZ) - obstacle.size[2] / 2, 0)
    return Math.hypot(outsideX, outsideZ) > ACTOR_CLEARANCE + EPSILON
  })
}
const chooseSpawn = (definition: RoomSceneDefinition): RoomPoint & { heading: number } => {
  if (definition.spawn) {
    if (!clearSpawn(definition.spawn, definition)) return failure('spawn', 'は床・穴・障害物から十分離れた場所にしてください')
    return { ...definition.spawn, heading: definition.spawn.heading ?? 0 }
  }
  const points = definition.floor.outline
  const minX = Math.min(...points.map((p) => p.x)), maxX = Math.max(...points.map((p) => p.x))
  const minZ = Math.min(...points.map((p) => p.z)), maxZ = Math.max(...points.map((p) => p.z))
  const centre = { x: (minX + maxX) / 2, z: (minZ + maxZ) / 2 }
  const candidates = [centre]
  const nx = Math.max(1, Math.min(96, Math.ceil((maxX - minX) / 0.2)))
  const nz = Math.max(1, Math.min(96, Math.ceil((maxZ - minZ) / 0.2)))
  for (let ix = 0; ix < nx; ix++) for (let iz = 0; iz < nz; iz++) candidates.push({
    x: minX + (ix + 0.5) / nx * (maxX - minX), z: minZ + (iz + 0.5) / nz * (maxZ - minZ),
  })
  candidates.sort((a, b) => pointDistance(a, centre) - pointDistance(b, centre))
  const selected = candidates.find((candidate) => clearSpawn(candidate, definition))
  if (!selected) return failure('spawn', 'の安全な場所が見つかりません。歩ける床を確保して開始位置を指定してください')
  return { ...selected, heading: 0 }
}

const validateDefinition = (value: unknown): RoomSceneDefinition => {
  const record = object(value, 'ルート')
  keys(record, ['version', 'name', 'floor', 'obstacles', 'destinations', 'spawn', 'units', 'upAxis', 'axes', 'floorHeight'], 'ルート')
  if (record.version !== 1) return failure('version', 'は1を指定してください')
  if (record.units !== undefined && record.units !== 'm') return failure('units', 'はm（メートル）だけに対応しています')
  for (const field of ['upAxis', 'axes']) {
    if (record[field] !== undefined && record[field] !== 'Y' && record[field] !== 'Y-up') return failure(field, 'はY-upだけに対応しています')
  }
  if (record.floorHeight !== undefined && record.floorHeight !== 0) return failure('floorHeight', 'は0の平坦な床だけに対応しています')
  const floorRecord = object(record.floor, 'floor')
  keys(floorRecord, ['outline', 'holes'], 'floor')
  const outline = ring(floorRecord.outline, 'floor.outline')
  const holes = floorRecord.holes === undefined ? [] : array(floorRecord.holes, 'floor.holes', 16)
    .map((hole, i) => ring(hole, `floor.holes[${i}]`))
  if (outline.length + holes.reduce((sum, hole) => sum + hole.length, 0) > 256) return failure('floor', 'の頂点は外周と穴を合わせて256点以下にしてください')
  for (let i = 0; i < holes.length; i++) {
    const hole = holes[i]
    if (!hole.every((p) => inside(p, outline)) || ringsIntersect(hole, outline)) return failure(`floor.holes[${i}]`, 'は床の外周内に置き、外周に接触させないでください')
    for (let j = 0; j < i; j++) {
      if (ringsIntersect(hole, holes[j]) || inside(hole[0], holes[j], true) || inside(holes[j][0], hole, true)) return failure(`floor.holes[${i}]`, 'は他の穴と重ねたり内包したりできません')
    }
  }
  const entityIds = new Set<string>()
  const obstacles = array(record.obstacles, 'obstacles').map((entry, i): RoomSceneObstacle => {
    const field = `obstacles[${i}]`, item = object(entry, field)
    keys(item, ['id', 'label', 'position', 'size', 'yaw', 'chair'], field)
    const id = text(item.id, `${field}.id`, 80)
    if (entityIds.has(id)) return failure(`${field}.id`, 'が重複しています')
    entityIds.add(id)
    const position = tuple(item.position, `${field}.position`)
    const size = tuple(item.size, `${field}.size`, true)
    const yaw = item.yaw === undefined ? undefined : number(item.yaw, `${field}.yaw`, Math.PI * 100)
    let chair: RoomSceneObstacle['chair']
    if (item.chair !== undefined) {
      const data = object(item.chair, `${field}.chair`)
      keys(data, ['seatHeight'], `${field}.chair`)
      const seatHeight = number(data.seatHeight, `${field}.chair.seatHeight`)
      if (seatHeight <= 0 || seatHeight > size[1]) return failure(`${field}.chair.seatHeight`, 'は0より大きく椅子の高さ以下にしてください')
      if (Math.abs(position[1]) > EPSILON) return failure(`${field}.position[1]`, 'は椅子の床原点として0にしてください')
      chair = { seatHeight }
    }
    return { id, label: text(item.label, `${field}.label`), position, size, ...(yaw === undefined ? {} : { yaw }), ...(chair ? { chair } : {}) }
  })
  const complexity = outline.length + holes.reduce((sum, hole) => sum + hole.length, 0) + obstacles.length * 4
  if (complexity > 240) return failure('floor/obstacles', 'の形状が複雑すぎます。床の頂点数と障害物の四隅数の合計を240以下に減らしてください')
  const floor = { outline, ...(holes.length ? { holes } : {}) }
  const destinations = array(record.destinations, 'destinations').map((entry, i): RoomSceneDestination => {
    const field = `destinations[${i}]`, item = object(entry, field)
    keys(item, ['id', 'label', 'position', 'heading', 'aliases'], field)
    const id = text(item.id, `${field}.id`, 80)
    if (entityIds.has(id)) return failure(`${field}.id`, 'が障害物または目的地のIDと重複しています')
    entityIds.add(id)
    const position = point(item.position, `${field}.position`)
    if (!floorContains(position, floor)) return failure(`${field}.position`, 'は床の内側で穴の外にしてください')
    const heading = item.heading === undefined ? undefined : number(item.heading, `${field}.heading`, Math.PI * 100)
    const aliases = item.aliases === undefined ? undefined : [...new Set(array(item.aliases, `${field}.aliases`, 16)
      .map((alias, j) => text(alias, `${field}.aliases[${j}]`, 80)))]
    return { id, label: text(item.label, `${field}.label`), position, ...(heading === undefined ? {} : { heading }), ...(aliases ? { aliases } : {}) }
  })
  let spawn: RoomSceneDefinition['spawn']
  if (record.spawn !== undefined) {
    const data = object(record.spawn, 'spawn')
    keys(data, ['x', 'z', 'heading'], 'spawn')
    spawn = { x: number(data.x, 'spawn.x'), z: number(data.z, 'spawn.z'),
      ...(data.heading === undefined ? {} : { heading: number(data.heading, 'spawn.heading', Math.PI * 100) }) }
  }
  const definition: RoomSceneDefinition = {
    version: 1, name: text(record.name, 'name'), floor, obstacles, destinations,
    ...(spawn ? { spawn } : {}), ...(record.units === undefined ? {} : { units: 'm' }),
    ...(record.upAxis === undefined && record.axes === undefined ? {} : { upAxis: 'Y' }),
    ...(record.floorHeight === undefined ? {} : { floorHeight: 0 }),
  }
  chooseSpawn(definition)
  return definition
}

/** Parse only the bounded flat-room JSON format; no guessed dimensions or missing-floor fallback. */
export const parseRoomScene = (source: string): RoomSceneDefinition => {
  if (typeof source !== 'string' || source.length > 1_000_000) return failure('ファイル', 'は100万文字以下のJSONにしてください')
  let value: unknown
  try { value = JSON.parse(source) } catch { return failure('ファイル', 'をJSONとして読み取れません') }
  return validateDefinition(value)
}

/** Build real floor/furniture geometry. The caller retains ownership of a borrowed floor material. */
export const createRoomScene = (source: RoomSceneDefinition, options: RoomSceneOptions = {}): RoomScene => {
  const definition = validateDefinition(source)
  const spawn = chooseSpawn(definition)
  const root = new THREE.Group()
  root.name = 'ImportedRoomScene'
  root.userData.roomSceneDefinition = definition
  const ownedMaterials = new Set<THREE.Material>()
  const owned = (params: THREE.MeshStandardMaterialParameters) => {
    const material = new THREE.MeshStandardMaterial(params)
    ownedMaterials.add(material)
    return material
  }
  const floorMaterial = options.floorMaterial ?? owned({ color: 0xc6b49b, roughness: 0.92, side: THREE.DoubleSide })
  const furnitureMaterial = owned({ color: 0x856544, roughness: 0.68 })
  const seatMaterial = owned({ color: 0xaa9a80, roughness: 0.93 })
  const obstacleMaterial = owned({ color: 0xb4b8bc, roughness: 0.76 })
  const shape = new THREE.Shape(definition.floor.outline.map((p) => new THREE.Vector2(p.x, -p.z)))
  for (const hole of definition.floor.holes ?? []) shape.holes.push(new THREE.Path(hole.map((p) => new THREE.Vector2(p.x, -p.z))))
  const floorGeometry = new THREE.ShapeGeometry(shape)
  const uv = floorGeometry.getAttribute('uv'), positions = floorGeometry.getAttribute('position')
  const minX = Math.min(...definition.floor.outline.map((p) => p.x)), maxX = Math.max(...definition.floor.outline.map((p) => p.x))
  const minZ = Math.min(...definition.floor.outline.map((p) => p.z)), maxZ = Math.max(...definition.floor.outline.map((p) => p.z))
  for (let i = 0; i < uv.count; i++) uv.setXY(i,
    (positions.getX(i) - minX) / (maxX - minX), (-positions.getY(i) - minZ) / (maxZ - minZ))
  const floor = new THREE.Mesh(floorGeometry, floorMaterial)
  floor.name = 'RoomSceneFloor'
  floor.rotation.x = -Math.PI / 2
  floor.receiveShadow = true
  floor.userData.roomEntity = { kind: 'floor' }
  root.add(floor)
  const box = (parent: THREE.Group, name: string, size: [number, number, number], centre: [number, number, number], material: THREE.Material) => {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size), material)
    mesh.name = name
    mesh.position.set(...centre)
    mesh.castShadow = false
    mesh.receiveShadow = true
    parent.add(mesh)
    return mesh
  }
  for (const obstacle of definition.obstacles) {
    const entity = new THREE.Group()
    entity.name = `RoomEntity:${obstacle.id}`
    entity.userData.roomEntityId = obstacle.id
    entity.userData.roomEntity = { id: obstacle.id, label: obstacle.label, kind: obstacle.chair ? 'chair' : 'obstacle', seatHeight: obstacle.chair?.seatHeight }
    entity.position.set(...obstacle.position)
    entity.rotation.y = obstacle.yaw ?? 0
    root.add(entity)
    if (!obstacle.chair) {
      box(entity, 'Body', obstacle.size, [0, 0, 0], obstacleMaterial)
      continue
    }
    const [width, height, depth] = obstacle.size
    const seatHeight = obstacle.chair.seatHeight
    const thickness = Math.min(0.055, seatHeight * 0.2, height * 0.12)
    box(entity, 'Seat', [width, thickness, depth], [0, seatHeight - thickness / 2, 0], seatMaterial)
    const leg = Math.min(0.045, width * 0.13, depth * 0.13)
    const legHeight = seatHeight - thickness
    for (const side of [-1, 1]) for (const end of [-1, 1]) box(entity, 'Leg', [leg, legHeight, leg],
      [side * (width - leg) / 2, legHeight / 2, end * (depth - leg) / 2], furnitureMaterial)
    const backHeight = height - seatHeight
    if (backHeight > EPSILON) {
      const backThickness = Math.min(depth * 0.12, 0.055)
      box(entity, 'Back', [width, backHeight, backThickness], [0, seatHeight + backHeight / 2, -(depth - backThickness) / 2], furnitureMaterial)
    }
  }
  const fill = new THREE.HemisphereLight(0xfff4df, 0x5b6272, 1.5)
  fill.name = 'RoomSceneFill'
  const sun = new THREE.DirectionalLight(0xffecd5, 1.65)
  sun.name = 'RoomSceneSun'
  sun.position.set((minX + maxX) / 2 - 3, 7, (minZ + maxZ) / 2 + 2)
  sun.target.position.set((minX + maxX) / 2, 0, (minZ + maxZ) / 2)
  sun.castShadow = false
  root.add(fill, sun, sun.target)
  root.updateMatrixWorld(true)
  let disposed = false
  return { root, definition, spawn, dispose: () => {
    if (disposed) return
    disposed = true
    root.removeFromParent()
    root.traverse((node) => {
      if (node instanceof THREE.Mesh) node.geometry.dispose()
      if (node instanceof THREE.DirectionalLight || node instanceof THREE.SpotLight) node.shadow.dispose()
    })
    for (const material of ownedMaterials) material.dispose()
    root.clear()
  } }
}

const visibleBounds = (root: THREE.Object3D) => {
  const bounds = new THREE.Box3()
  const localBounds = new THREE.Box3()
  const inverseRoot = root.matrixWorld.clone().invert()
  const visit = (node: THREE.Object3D) => {
    if (!node.visible) return
    if (node instanceof THREE.Mesh) {
      if (!node.geometry.boundingBox) node.geometry.computeBoundingBox()
      if (node.geometry.boundingBox) {
        bounds.union(node.geometry.boundingBox.clone().applyMatrix4(node.matrixWorld))
        const localTransform = new THREE.Matrix4().multiplyMatrices(inverseRoot, node.matrixWorld)
        localBounds.union(node.geometry.boundingBox.clone().applyMatrix4(localTransform))
      }
    }
    for (const child of node.children) visit(child)
  }
  visit(root)
  return { bounds, localBounds }
}

/** Read tagged entities and their real world meshes, independent of household-specific names. */
export const createSceneRoomNavigation = (root: THREE.Object3D): RoomNavigation => {
  const definition = validateDefinition(root.userData.roomSceneDefinition)
  root.updateWorldMatrix(true, true)
  const up = new THREE.Vector3(0, 1, 0).transformDirection(root.matrixWorld)
  if (!Number.isFinite(up.y) || up.y < 0.999999) return failure('床の変換', 'はY-upの向きを保ってください')
  const worldPoint = (p: RoomPoint) => {
    const position = new THREE.Vector3(p.x, 0, p.z).applyMatrix4(root.matrixWorld)
    if (![position.x, position.y, position.z].every(Number.isFinite) || Math.abs(position.y) > 0.00001) {
      return failure('床の変換', 'は世界座標でY=0の平坦な床にしてください')
    }
    return { x: position.x, z: position.z }
  }
  const outline = definition.floor.outline.map(worldPoint)
  const holes = (definition.floor.holes ?? []).map((hole) => hole.map(worldPoint))
  if (Math.abs(area(outline)) <= EPSILON) return failure('床の変換', 'で歩行面が失われています')
  const obstacles: RoomNavigation['obstacles'] = []
  const chairs: RoomNavigation['chairs'] = []
  for (const item of definition.obstacles) {
    const entity = root.children.find((child) => child.userData.roomEntityId === item.id)
    if (!entity) return failure(`obstacles.${item.id}`, 'の表示形状が見つかりません')
    const { bounds, localBounds } = visibleBounds(entity)
    if (bounds.isEmpty() || localBounds.isEmpty()) return failure(`obstacles.${item.id}`, 'の表示形状がありません')
    if (![...bounds.min.toArray(), ...bounds.max.toArray(), ...localBounds.min.toArray(), ...localBounds.max.toArray()].every(Number.isFinite)) {
      return failure(`obstacles.${item.id}`, 'の表示形状の変換が不正です')
    }
    // Enclose every visible mesh in the entity's own axes, then rotate that
    // rectangle into world coordinates without filling its empty AABB corners.
    const footprint = [
      [localBounds.min.x, localBounds.min.z], [localBounds.max.x, localBounds.min.z],
      [localBounds.max.x, localBounds.max.z], [localBounds.min.x, localBounds.max.z],
    ].map(([x, z]) => {
      const corner = new THREE.Vector3(x, 0, z).applyMatrix4(entity.matrixWorld)
      return { x: corner.x, z: corner.z }
    })
    obstacles.push({ id: item.id, label: item.label,
      bounds: { minX: bounds.min.x, maxX: bounds.max.x, minZ: bounds.min.z, maxZ: bounds.max.z },
      footprint, minHeight: bounds.min.y, maxHeight: bounds.max.y })
    if (item.chair) {
      const centre = new THREE.Vector3(0, item.chair.seatHeight, 0).applyMatrix4(entity.matrixWorld)
      const direction = new THREE.Vector3(0, 0, 1).transformDirection(entity.matrixWorld)
      const length = Math.hypot(direction.x, direction.z)
      if (!Number.isFinite(length) || length < EPSILON) return failure(`obstacles.${item.id}`, 'の椅子の向きが不正です')
      const forward = { x: direction.x / length, z: direction.z / length }
      const side = { x: forward.z, z: -forward.x }
      const approach = (offset: number, rear: number) => ({ x: centre.x + side.x * offset - forward.x * rear,
        z: centre.z + side.z * offset - forward.z * rear })
      chairs.push({ id: item.id, label: item.label, obstacleId: item.id,
        position: { x: centre.x, y: centre.y, z: centre.z }, seatHeight: centre.y,
        forward, yaw: Math.atan2(forward.x, forward.z),
        approaches: [approach(0.75, 0.14), approach(-0.75, 0.14), approach(0, 0.8)] })
    }
  }
  const worldHeading = (heading: number) => {
    const forward = new THREE.Vector3(Math.sin(heading), 0, Math.cos(heading)).transformDirection(root.matrixWorld)
    if (!Number.isFinite(forward.x) || !Number.isFinite(forward.z) || Math.hypot(forward.x, forward.z) < EPSILON) return failure('向きの変換', 'が不正です')
    return Math.atan2(forward.x, forward.z)
  }
  return {
    bounds: { minX: Math.min(...outline.map((p) => p.x)), maxX: Math.max(...outline.map((p) => p.x)),
      minZ: Math.min(...outline.map((p) => p.z)), maxZ: Math.max(...outline.map((p) => p.z)) },
    walkablePolygon: outline, floorHoles: holes, obstacles, chairs,
    destinations: definition.destinations.map((destination) => ({ ...destination,
      position: worldPoint(destination.position), heading: worldHeading(destination.heading ?? 0),
      ...(destination.aliases ? { aliases: [...destination.aliases] } : {}) })),
    defaultChairId: chairs[0]?.id ?? '', avatarRadius: 0.24,
  }
}
