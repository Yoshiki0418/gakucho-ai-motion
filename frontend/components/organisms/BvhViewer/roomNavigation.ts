import * as THREE from 'three'

export type RoomPoint = { x: number; z: number }
export type RoomBounds = { minX: number; maxX: number; minZ: number; maxZ: number }
export type RoomObstacle = {
  id: string
  label: string
  bounds: RoomBounds
  /** World XZ outline of a tagged furniture box; legacy home objects use bounds instead. */
  footprint?: RoomPoint[]
  minHeight: number
  maxHeight: number
}
export type RoomChairTarget = {
  id: string
  label: string
  obstacleId: string
  position: { x: number; y: number; z: number }
  seatHeight: number
  forward: RoomPoint
  yaw: number
  /** Standing locations beside the seat, followed by a rear fallback. */
  approaches: RoomPoint[]
}
export type RoomNavigation = {
  bounds: RoomBounds
  obstacles: RoomObstacle[]
  chairs: RoomChairTarget[]
  defaultChairId: string
  avatarRadius: number
  /** A single flat floor outline in world XZ; omitted for the legacy rectangular room. */
  walkablePolygon?: RoomPoint[]
  /** Closed nonwalkable cutouts in the same floor, with either winding order. */
  floorHoles?: RoomPoint[][]
  destinations?: Array<{
    id: string
    label: string
    position: RoomPoint
    heading: number
    aliases?: string[]
  }>
}
export type RoomPathOptions = {
  radius?: number
  clearance?: number
  ignoreObstacleIds?: readonly string[]
}
export type RoomPathFailure = 'invalid-position' | 'start-blocked' | 'goal-blocked' | 'no-route'
export type RoomPath =
  | { ok: true; waypoints: RoomPoint[]; length: number }
  | { ok: false; reason: RoomPathFailure }
export type RoomChairApproach =
  | { ok: true; waypoints: RoomPoint[]; length: number; chair: RoomChairTarget; approach: RoomPoint }
  | { ok: false; reason: RoomPathFailure | 'unknown-chair' }

const EPSILON = 1e-7
const CORNER_MARGIN = 0.012
const distance = (a: RoomPoint, b: RoomPoint) => Math.hypot(b.x - a.x, b.z - a.z)
const finitePoint = (point: RoomPoint) => Number.isFinite(point.x) && Number.isFinite(point.z)
const contains = (bounds: RoomBounds, point: RoomPoint) =>
  point.x >= bounds.minX - EPSILON && point.x <= bounds.maxX + EPSILON
  && point.z >= bounds.minZ - EPSILON && point.z <= bounds.maxZ + EPSILON

const cross = (a: RoomPoint, b: RoomPoint, point: RoomPoint) =>
  (b.x - a.x) * (point.z - a.z) - (b.z - a.z) * (point.x - a.x)
const pointOnSegment = (point: RoomPoint, a: RoomPoint, b: RoomPoint) =>
  Math.abs(cross(a, b, point)) <= EPSILON
  && point.x >= Math.min(a.x, b.x) - EPSILON && point.x <= Math.max(a.x, b.x) + EPSILON
  && point.z >= Math.min(a.z, b.z) - EPSILON && point.z <= Math.max(a.z, b.z) + EPSILON
const segmentsIntersect = (a: RoomPoint, b: RoomPoint, c: RoomPoint, d: RoomPoint) => {
  const abc = cross(a, b, c)
  const abd = cross(a, b, d)
  const cda = cross(c, d, a)
  const cdb = cross(c, d, b)
  const opposite = (left: number, right: number) =>
    (left > EPSILON && right < -EPSILON) || (left < -EPSILON && right > EPSILON)
  return (opposite(abc, abd) && opposite(cda, cdb))
    || pointOnSegment(c, a, b) || pointOnSegment(d, a, b)
    || pointOnSegment(a, c, d) || pointOnSegment(b, c, d)
}
const pointSegmentDistance = (point: RoomPoint, a: RoomPoint, b: RoomPoint) => {
  const dx = b.x - a.x
  const dz = b.z - a.z
  const lengthSquared = dx * dx + dz * dz
  const t = lengthSquared > EPSILON * EPSILON
    ? THREE.MathUtils.clamp(((point.x - a.x) * dx + (point.z - a.z) * dz) / lengthSquared, 0, 1) : 0
  return Math.hypot(point.x - a.x - t * dx, point.z - a.z - t * dz)
}
const segmentDistance = (a: RoomPoint, b: RoomPoint, c: RoomPoint, d: RoomPoint) =>
  segmentsIntersect(a, b, c, d) ? 0 : Math.min(
    pointSegmentDistance(a, c, d), pointSegmentDistance(b, c, d),
    pointSegmentDistance(c, a, b), pointSegmentDistance(d, a, b),
  )
const polygonContains = (polygon: readonly RoomPoint[], point: RoomPoint) => {
  if (polygon.length < 3 || polygon.some((vertex) => !finitePoint(vertex))) return false
  let inside = false
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const a = polygon[j]
    const b = polygon[i]
    if (pointOnSegment(point, a, b)) return true
    if ((a.z > point.z) !== (b.z > point.z)
      && point.x < (b.x - a.x) * (point.z - a.z) / (b.z - a.z) + a.x) inside = !inside
  }
  return inside
}
type FloorEdge = readonly [RoomPoint, RoomPoint]
const polygonEdges = (polygon: readonly RoomPoint[]): FloorEdge[] =>
  polygon.map((point, i) => [point, polygon[(i + 1) % polygon.length]] as const)

/** Exact flat-floor capsule constraints; endpoint checks alone miss concave exits and holes. */
const floorConstraints = (room: RoomNavigation, padding: number) => {
  const outline = room.walkablePolygon
  const holes = room.floorHoles ?? []
  const edges = [...(outline ? polygonEdges(outline) : []), ...holes.flatMap(polygonEdges)]
  const pointClear = (point: RoomPoint) =>
    (!outline || polygonContains(outline, point))
    && holes.every((hole) => !polygonContains(hole, point))
    && edges.every(([a, b]) => pointSegmentDistance(point, a, b) + EPSILON >= padding)
  const segmentClear = (from: RoomPoint, to: RoomPoint) => pointClear(from) && pointClear(to)
    && edges.every(([a, b]) => !segmentsIntersect(from, to, a, b)
      && segmentDistance(from, to, a, b) + EPSILON >= padding)
  return { pointClear, segmentClear }
}

/** Furniture outlines retain their real orientation instead of blocking their larger world AABB. */
const footprintConstraints = (footprints: readonly RoomPoint[][], padding: number) => {
  const edges = footprints.flatMap(polygonEdges)
  const pointClear = (point: RoomPoint) => !footprints.some((polygon) => polygonContains(polygon, point))
    && edges.every(([a, b]) => pointSegmentDistance(point, a, b) + EPSILON >= padding)
  const segmentClear = (from: RoomPoint, to: RoomPoint) => pointClear(from) && pointClear(to)
    && edges.every(([a, b]) => !segmentsIntersect(from, to, a, b)
      && segmentDistance(from, to, a, b) + EPSILON >= padding)
  return { pointClear, segmentClear }
}

/** Offset corner candidates stay inside the outline and outside cutouts after point validation. */
const floorCornerCandidates = (polygon: readonly RoomPoint[], padding: number, interior: boolean) => {
  if (polygon.length < 3) return []
  const signedArea = polygon.reduce((sum, point, i) => {
    const next = polygon[(i + 1) % polygon.length]
    return sum + point.x * next.z - next.x * point.z
  }, 0)
  const side = (signedArea >= 0 ? 1 : -1) * (interior ? 1 : -1)
  const offset = padding + CORNER_MARGIN
  const candidates: RoomPoint[] = []
  for (let i = 0; i < polygon.length; i++) {
    const previous = polygon[(i + polygon.length - 1) % polygon.length]
    const point = polygon[i]
    const next = polygon[(i + 1) % polygon.length]
    const lengthIn = distance(previous, point)
    const lengthOut = distance(point, next)
    if (lengthIn < EPSILON || lengthOut < EPSILON) continue
    const n1 = { x: -(point.z - previous.z) / lengthIn * side, z: (point.x - previous.x) / lengthIn * side }
    const n2 = { x: -(next.z - point.z) / lengthOut * side, z: (next.x - point.x) / lengthOut * side }
    const denominator = 1 + n1.x * n2.x + n1.z * n2.z
    if (denominator > EPSILON) candidates.push({
      x: point.x + (n1.x + n2.x) * offset / denominator,
      z: point.z + (n1.z + n2.z) * offset / denominator,
    })
    candidates.push(
      { x: point.x + n1.x * offset, z: point.z + n1.z * offset },
      { x: point.x + n2.x * offset, z: point.z + n2.z * offset },
    )
    // Extra arc samples help acute/reflex corners where a long miter falls outside the floor.
    for (let direction = 0; direction < 8; direction++) {
      const angle = direction * Math.PI / 4
      candidates.push({ x: point.x + Math.cos(angle) * offset * Math.SQRT2, z: point.z + Math.sin(angle) * offset * Math.SQRT2 })
    }
  }
  return candidates
}

/** Box3.setFromObject includes hidden procedural fallbacks; only visible meshes are physical here. */
const visibleBounds = (object: THREE.Object3D) => {
  const bounds = new THREE.Box3()
  const visit = (node: THREE.Object3D) => {
    if (!node.visible) return
    if (node instanceof THREE.Mesh) {
      if (!node.geometry.boundingBox) node.geometry.computeBoundingBox()
      if (node.geometry.boundingBox) bounds.union(node.geometry.boundingBox.clone().applyMatrix4(node.matrixWorld))
    }
    for (const child of node.children) visit(child)
  }
  visit(object)
  return bounds
}

/** Find the largest upward horizontal surface at chair-seat height in the displayed mesh. */
const chairSeat = (chair: THREE.Object3D, fallback: THREE.Vector3) => {
  const planes = new Map<number, { area: number; x: number; y: number; z: number }>()
  const a = new THREE.Vector3()
  const b = new THREE.Vector3()
  const c = new THREE.Vector3()
  const ab = new THREE.Vector3()
  const ac = new THREE.Vector3()
  const normal = new THREE.Vector3()
  const floorY = chair.getWorldPosition(new THREE.Vector3()).y
  const visit = (node: THREE.Object3D) => {
    if (!node.visible) return
    if (node instanceof THREE.Mesh) {
      const positions = node.geometry.getAttribute('position')
      const indices = node.geometry.getIndex()
      if (positions) {
        const count = indices ? indices.count : positions.count
        for (let i = 0; i + 2 < count; i += 3) {
          a.fromBufferAttribute(positions, indices ? indices.getX(i) : i).applyMatrix4(node.matrixWorld)
          b.fromBufferAttribute(positions, indices ? indices.getX(i + 1) : i + 1).applyMatrix4(node.matrixWorld)
          c.fromBufferAttribute(positions, indices ? indices.getX(i + 2) : i + 2).applyMatrix4(node.matrixWorld)
          normal.crossVectors(ab.subVectors(b, a), ac.subVectors(c, a))
          const twiceArea = normal.length()
          if (twiceArea < 0.00001 || normal.y / twiceArea < 0.9) continue
          const height = (a.y + b.y + c.y) / 3
          if (height - floorY < 0.34 || height - floorY > 0.62) continue
          const key = Math.round(height / 0.01)
          const plane = planes.get(key) ?? { area: 0, x: 0, y: 0, z: 0 }
          plane.area += twiceArea
          plane.x += (a.x + b.x + c.x) / 3 * twiceArea
          plane.y += height * twiceArea
          plane.z += (a.z + b.z + c.z) / 3 * twiceArea
          planes.set(key, plane)
        }
      }
    }
    for (const child of node.children) visit(child)
  }
  visit(chair)
  const largestArea = Math.max(0, ...[...planes.values()].map((plane) => plane.area))
  // Prefer the pad above the wooden slab, whose hidden top triangles can have more area.
  const surface = [...planes.values()].filter((plane) => plane.area >= largestArea * 0.5)
    .sort((left, right) => right.y / right.area - left.y / left.area)[0]
  return surface && surface.area > 0.025
    ? new THREE.Vector3(surface.x / surface.area, surface.y / surface.area, surface.z / surface.area)
    : fallback
}

/** Snapshot semantic furniture transforms after home.ready resolves; no visual transforms are changed. */
export const createRoomNavigation = (homeRoot: THREE.Object3D): RoomNavigation => {
  homeRoot.updateWorldMatrix(true, true)
  const floor = homeRoot.getObjectByName('HomeOakFloor')
  const floorBounds = floor ? visibleBounds(floor) : new THREE.Box3(
    new THREE.Vector3(-5.06, 0, -5.06), new THREE.Vector3(5.06, 0, 5.06),
  )
  const bounds = {
    minX: floorBounds.min.x + 0.06, maxX: floorBounds.max.x - 0.06,
    minZ: floorBounds.min.z + 0.06, maxZ: floorBounds.max.z - 0.06,
  }
  const obstacles: RoomObstacle[] = []
  const supportingParts: Record<string, readonly string[]> = {
    KitchenSideCounter: ['KitchenLowerRun', 'KitchenToeKick'],
    KitchenBackCounter: ['KitchenBackCabinet'],
    KitchenIslandStone: ['KitchenIslandBase', 'KitchenIslandToeKick'],
    CoffeeTableTop: ['CoffeeTablePedestal'],
    DiningTableTop: ['DiningTableApron', 'DiningTableLeg'],
  }
  const addObstacle = (name: string, id: string, label: string) => {
    const object = homeRoot.getObjectByName(name)
    if (!object) return
    const box = visibleBounds(object)
    if (box.isEmpty()) return
    const parts = new Set(supportingParts[name] ?? [])
    homeRoot.traverse((node) => {
      if (parts.has(node.name)) box.union(visibleBounds(node))
    })
    obstacles.push({
      id, label,
      bounds: { minX: box.min.x, maxX: box.max.x, minZ: box.min.z, maxZ: box.max.z },
      minHeight: box.min.y, maxHeight: box.max.y,
    })
  }
  for (const [name, id, label] of [
    ['KitchenSideCounter', 'kitchen-side', '壁側キッチン'],
    ['KitchenBackCounter', 'kitchen-back', '奥側キッチン'],
    ['Refrigerator', 'refrigerator', '冷蔵庫'],
    ['KitchenIslandStone', 'kitchen-island', 'キッチンの作業台'],
    ['LivingSofa', 'sofa', 'ソファ'],
    ['CoffeeTableTop', 'coffee-table', 'ローテーブル'],
    ['MediaConsole', 'media-console', 'テレビ台'],
    ['DiningTableTop', 'dining-table', '食卓'],
    ['LivingPlant', 'living-plant', 'リビングの鉢植え'],
    ['DiningPlant', 'dining-plant', '食卓側の鉢植え'],
    ['LivingFloorLamp', 'floor-lamp', 'フロアランプ'],
  ]) addObstacle(name, id, label)

  const chairs: RoomChairTarget[] = []
  for (const side of [-1, 1]) for (const end of [-1, 1]) {
    const name = `DiningChair${side},${end}`
    const object = homeRoot.getObjectByName(name)
    if (!object) continue
    const id = `dining-chair-${side < 0 ? 'wall' : 'aisle'}-${end < 0 ? 'back' : 'front'}`
    const label = `食卓の椅子（${side < 0 ? '壁側' : '通路側'}・${end < 0 ? '奥' : '手前'}）`
    addObstacle(name, id, label)
    const origin = object.getWorldPosition(new THREE.Vector3())
    const surface = chairSeat(object, origin.clone().add(new THREE.Vector3(0, 0.508, 0)))
    const direction = new THREE.Vector3(0, 0, 1).applyQuaternion(object.getWorldQuaternion(new THREE.Quaternion()))
    const horizontalLength = Math.hypot(direction.x, direction.z)
    const forward = { x: direction.x / horizontalLength, z: direction.z / horizontalLength }
    const sideVector = { x: forward.z, z: -forward.x }
    const approach = (sideDistance: number, rearDistance: number) => ({
      x: surface.x + sideVector.x * sideDistance - forward.x * rearDistance,
      z: surface.z + sideVector.z * sideDistance - forward.z * rearDistance,
    })
    chairs.push({
      id, label, obstacleId: id,
      position: { x: surface.x, y: surface.y, z: surface.z }, seatHeight: surface.y,
      forward, yaw: Math.atan2(forward.x, forward.z),
      approaches: [approach(0.75, 0.14), approach(-0.75, 0.14), approach(0, 0.8)],
    })
  }
  return { bounds, obstacles, chairs, defaultChairId: 'dining-chair-aisle-front', avatarRadius: 0.24 }
}

const collisionBounds = (room: RoomNavigation, options: RoomPathOptions) => {
  const radius = Math.max(0, options.radius ?? room.avatarRadius)
  const clearance = Math.max(0, options.clearance ?? 0.035)
  const inflate = radius + clearance
  const ignored = new Set(options.ignoreObstacleIds ?? [])
  const active = room.obstacles.filter((obstacle) => !ignored.has(obstacle.id))
  const hasFootprint = (obstacle: RoomObstacle) => !!obstacle.footprint
    && obstacle.footprint.length >= 3 && obstacle.footprint.every(finitePoint)
  return {
    padding: inflate,
    boundary: {
      minX: room.bounds.minX + inflate, maxX: room.bounds.maxX - inflate,
      minZ: room.bounds.minZ + inflate, maxZ: room.bounds.maxZ - inflate,
    },
    footprints: active.filter(hasFootprint).map((obstacle) => obstacle.footprint!),
    obstacles: active.filter((obstacle) => !hasFootprint(obstacle)).map(({ bounds }) => ({
      minX: bounds.minX - inflate, maxX: bounds.maxX + inflate,
      minZ: bounds.minZ - inflate, maxZ: bounds.maxZ + inflate,
    })),
  }
}

/** Exact segment/AABB intersection, including touching; no sampling can miss thin obstacles. */
const intersects = (from: RoomPoint, to: RoomPoint, bounds: RoomBounds) => {
  let entering = 0
  let exiting = 1
  for (const axis of ['x', 'z'] as const) {
    const delta = to[axis] - from[axis]
    const min = axis === 'x' ? bounds.minX : bounds.minZ
    const max = axis === 'x' ? bounds.maxX : bounds.maxZ
    if (Math.abs(delta) < EPSILON) {
      if (from[axis] < min - EPSILON || from[axis] > max + EPSILON) return false
    } else {
      const near = (min - from[axis]) / delta
      const far = (max - from[axis]) / delta
      entering = Math.max(entering, Math.min(near, far))
      exiting = Math.min(exiting, Math.max(near, far))
      if (entering > exiting + EPSILON) return false
    }
  }
  return entering <= exiting + EPSILON
}

export const isRoomPointWalkable = (room: RoomNavigation, point: RoomPoint, options: RoomPathOptions = {}) => {
  if (!finitePoint(point)) return false
  const { boundary, obstacles, footprints, padding } = collisionBounds(room, options)
  return contains(boundary, point) && !obstacles.some((bounds) => contains(bounds, point))
    && floorConstraints(room, padding).pointClear(point)
    && footprintConstraints(footprints, padding).pointClear(point)
}

export const isRoomSegmentWalkable = (
  room: RoomNavigation, from: RoomPoint, to: RoomPoint, options: RoomPathOptions = {},
) => {
  if (!finitePoint(from) || !finitePoint(to)) return false
  const { boundary, obstacles, footprints, padding } = collisionBounds(room, options)
  return contains(boundary, from) && contains(boundary, to)
    && !obstacles.some((bounds) => intersects(from, to, bounds))
    && floorConstraints(room, padding).segmentClear(from, to)
    && footprintConstraints(footprints, padding).segmentClear(from, to)
}

/** Deterministic visibility graph around inflated furniture; returns no route instead of teleporting. */
export const planRoomPath = (
  room: RoomNavigation, start: RoomPoint, goal: RoomPoint, options: RoomPathOptions = {},
): RoomPath => {
  if (!finitePoint(start) || !finitePoint(goal)) return { ok: false, reason: 'invalid-position' }
  const { boundary, obstacles, footprints, padding } = collisionBounds(room, options)
  const floor = floorConstraints(room, padding)
  const furniture = footprintConstraints(footprints, padding)
  const pointClear = (point: RoomPoint) =>
    contains(boundary, point) && !obstacles.some((bounds) => contains(bounds, point))
    && floor.pointClear(point) && furniture.pointClear(point)
  const segmentClear = (from: RoomPoint, to: RoomPoint) =>
    !obstacles.some((bounds) => intersects(from, to, bounds))
    && floor.segmentClear(from, to) && furniture.segmentClear(from, to)
  if (!pointClear(start)) return { ok: false, reason: 'start-blocked' }
  if (!pointClear(goal)) return { ok: false, reason: 'goal-blocked' }
  if (segmentClear(start, goal)) return {
    ok: true, waypoints: [{ ...start }, { ...goal }], length: distance(start, goal),
  }
  const nodes: RoomPoint[] = [{ ...start }, { ...goal }]
  for (const obstacle of obstacles) {
    for (const x of [obstacle.minX - CORNER_MARGIN, obstacle.maxX + CORNER_MARGIN]) {
      for (const z of [obstacle.minZ - CORNER_MARGIN, obstacle.maxZ + CORNER_MARGIN]) {
        const point = { x, z }
        if (pointClear(point) && !nodes.some((node) => distance(node, point) < EPSILON)) nodes.push(point)
      }
    }
  }
  const appendNode = (point: RoomPoint) => {
    if (finitePoint(point) && pointClear(point) && !nodes.some((node) => distance(node, point) < EPSILON)) nodes.push(point)
  }
  if (room.walkablePolygon) for (const point of floorCornerCandidates(room.walkablePolygon, padding, true)) appendNode(point)
  for (const hole of room.floorHoles ?? []) for (const point of floorCornerCandidates(hole, padding, false)) appendNode(point)
  for (const footprint of footprints) for (const point of floorCornerCandidates(footprint, padding, false)) appendNode(point)
  const lengths = nodes.map(() => Number.POSITIVE_INFINITY)
  const previous = nodes.map(() => -1)
  const visited = nodes.map(() => false)
  lengths[0] = 0
  for (let iteration = 0; iteration < nodes.length; iteration++) {
    let current = -1
    for (let i = 0; i < nodes.length; i++) {
      if (!visited[i] && (current < 0 || lengths[i] < lengths[current])) current = i
    }
    if (current < 0 || !Number.isFinite(lengths[current])) break
    if (current === 1) {
      const waypoints: RoomPoint[] = []
      for (let index = 1; index >= 0; index = previous[index]) waypoints.unshift(nodes[index])
      return { ok: true, waypoints, length: lengths[1] }
    }
    visited[current] = true
    for (let next = 0; next < nodes.length; next++) {
      if (visited[next] || !segmentClear(nodes[current], nodes[next])) continue
      const candidate = lengths[current] + distance(nodes[current], nodes[next])
      if (candidate + EPSILON < lengths[next]) {
        lengths[next] = candidate
        previous[next] = current
      }
    }
  }
  return { ok: false, reason: 'no-route' }
}

export const planChairApproach = (
  room: RoomNavigation, start: RoomPoint, chairId = room.defaultChairId, options: RoomPathOptions = {},
): RoomChairApproach => {
  const chair = room.chairs.find((candidate) => candidate.id === chairId)
  if (!chair) return { ok: false, reason: 'unknown-chair' }
  // A reachable standing approach cannot make a seat over a floor edge or cutout safe.
  if (room.walkablePolygon && !floorConstraints(room, collisionBounds(room, options).padding)
    .pointClear({ x: chair.position.x, z: chair.position.z })) return { ok: false, reason: 'goal-blocked' }
  let best: Extract<RoomChairApproach, { ok: true }> | undefined
  let lastReason: RoomPathFailure = 'no-route'
  // Side approaches can reach a fixed seat without walking through its backrest.
  for (const approach of chair.approaches.slice(0, 2)) {
    const path = planRoomPath(room, start, approach, options)
    if (path.ok) {
      if (!best || path.length < best.length) best = { ...path, chair, approach }
    } else lastReason = path.reason
  }
  return best ?? { ok: false, reason: lastReason }
}
