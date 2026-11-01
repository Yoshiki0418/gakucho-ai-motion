import {
  isRoomPointWalkable,
  planRoomPath,
  type RoomBounds,
  type RoomNavigation,
  type RoomPoint,
} from './roomNavigation'

export type RoomDestination = {
  id: string
  label: string
  position: RoomPoint
  heading: number
  aliases?: string[]
}

const center = (bounds: RoomBounds): RoomPoint => ({
  x: (bounds.minX + bounds.maxX) / 2,
  z: (bounds.minZ + bounds.maxZ) / 2,
})
const distance = (a: RoomPoint, b: RoomPoint) => Math.hypot(a.x - b.x, a.z - b.z)

/** Named standing destinations come from actual furniture bounds, not fixed room coordinates. */
export const createRoomDestinations = (room: RoomNavigation): RoomDestination[] => {
  const radius = room.avatarRadius
  const clearance = 0.035 + 0.06
  const standoff = radius + clearance + 0.012
  const hub = center(room.bounds)
  const hubAvailable = isRoomPointWalkable(room, hub)
  const destinations: RoomDestination[] = []
  const boundsOf = (id: string) => room.obstacles.find((obstacle) => obstacle.id === id)?.bounds
  const add = (id: string, label: string, preferred: RoomPoint, lookAt: RoomPoint, aliases?: string[]) => {
    const candidates = [preferred]
    // If a later furniture placement blocks the preferred point, find the nearest safe
    // standing point in its neighbourhood; never put the avatar inside a furniture box.
    for (let ix = -12; ix <= 12; ix++) for (let iz = -12; iz <= 12; iz++) {
      if (ix || iz) candidates.push({ x: preferred.x + ix * 0.12, z: preferred.z + iz * 0.12 })
    }
    candidates.sort((a, b) => distance(a, preferred) - distance(b, preferred))
    const position = candidates.find((point) => isRoomPointWalkable(room, point, { radius, clearance })
      // Imported destinations are tested from the actual actor when requested.
      // A room's centre can belong to a different partition than its spawn.
      && (room.destinations !== undefined || !hubAvailable || planRoomPath(room, hub, point).ok))
    if (!position) return
    destinations.push({ id, label, position, heading: Math.atan2(lookAt.x - position.x, lookAt.z - position.z), aliases })
  }

  if (room.destinations) {
    for (const target of room.destinations) {
      add(target.id, target.label, target.position, {
        x: target.position.x + Math.sin(target.heading),
        z: target.position.z + Math.cos(target.heading),
      }, target.aliases)
    }
    return destinations
  }

  const island = boundsOf('kitchen-island')
  if (island) add('kitchen', 'キッチン（アイランド横）', {
    x: island.maxX + standoff, z: center(island).z,
  }, center(island), ['キッチン', '台所'])

  const table = boundsOf('dining-table')
  if (table) add('dining', 'ダイニング（食卓横）', {
    x: table.maxX + standoff, z: center(table).z,
  }, center(table), ['ダイニング', '食卓'])

  const sofa = boundsOf('sofa')
  if (sofa) {
    const media = boundsOf('media-console')
    // The coffee table and chaise occupy the middle of the sofa/TV gap. Stop at
    // its open front corner and face the TV, leaving all furniture in place.
    add('living', 'リビング（ソファ前）', {
      x: sofa.maxX - standoff, z: sofa.maxZ + standoff,
    }, media ? center(media) : center(sofa), ['リビング', 'ソファ'])
  }
  return destinations
}
