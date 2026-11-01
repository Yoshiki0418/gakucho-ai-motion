import type { RoomSceneDefinition } from './roomScene'

export const ROOM_SCENE_EXAMPLES: { id: string; label: string; definition: RoomSceneDefinition }[] = [
  {
    id: 'l-shaped', label: 'L字形の部屋',
    definition: {
      version: 1, name: 'L字形の部屋',
      floor: { outline: [{ x: -5, z: -4 }, { x: 5, z: -4 }, { x: 5, z: 0 }, { x: 0, z: 0 }, { x: 0, z: 4 }, { x: -5, z: 4 }] },
      spawn: { x: -3, z: 2.9 },
      obstacles: [
        { id: 'worktop', label: 'キッチンの作業台', position: [3, 0.45, -3], size: [1.6, 0.9, 0.8] },
        { id: 'table', label: '食卓', position: [-3.3, 0.38, 1.2], size: [1.6, 0.76, 1.2] },
        { id: 'seat', label: '食卓の椅子', position: [-1.7, 0, 1.2], size: [0.5, 0.9, 0.5], yaw: -Math.PI / 2, chair: { seatHeight: 0.45 } },
        { id: 'sofa', label: 'ソファ', position: [-3.2, 0.45, -2.4], size: [2.2, 0.9, 0.9] },
      ],
      destinations: [
        { id: 'worktop-approach', label: 'キッチン', position: { x: 3, z: -1.9 }, heading: Math.PI, aliases: ['台所'] },
        { id: 'table-approach', label: '食卓', position: { x: -0.85, z: 1.2 }, heading: -Math.PI / 2, aliases: ['ダイニング'] },
        { id: 'sofa-approach', label: 'ソファ前', position: { x: -3.2, z: -1.25 }, heading: Math.PI, aliases: ['リビング', 'ソファ'] },
        { id: 'window-approach', label: '窓際', position: { x: 4, z: -0.75 }, heading: Math.PI / 2, aliases: ['窓辺'] },
      ],
    },
  },
  {
    id: 'partitioned', label: '間仕切りのある部屋',
    definition: {
      version: 1, name: '間仕切りのある部屋',
      floor: { outline: [{ x: -6, z: -4 }, { x: 6, z: -4 }, { x: 6, z: 4 }, { x: -6, z: 4 }] },
      spawn: { x: -4, z: 0 },
      obstacles: [
        { id: 'divider-back', label: '奥の間仕切り', position: [0, 1.2, -2.4], size: [0.18, 2.4, 3.2] },
        { id: 'divider-front', label: '手前の間仕切り', position: [0, 1.2, 2.4], size: [0.18, 2.4, 3.2] },
        { id: 'angled-table', label: '斜めの作業台', position: [3, 0.4, -1.6], size: [2, 0.8, 0.9], yaw: 0.5 },
        { id: 'chair', label: '作業用の椅子', position: [3.2, 0, 0.6], size: [0.52, 0.95, 0.5], yaw: Math.PI, chair: { seatHeight: 0.48 } },
      ],
      destinations: [
        { id: 'east-room', label: '東側の部屋', position: { x: 4.5, z: 1.8 }, heading: 0 },
        { id: 'west-room', label: '西側の部屋', position: { x: -4, z: 0 }, heading: Math.PI / 2 },
        { id: 'desk-side', label: '作業台の横', position: { x: 4.75, z: -1.6 }, heading: -Math.PI / 2 },
      ],
    },
  },
  {
    id: 'courtyard', label: '中央に中庭のある部屋',
    definition: {
      version: 1, name: '中央に中庭のある部屋',
      floor: {
        outline: [{ x: -6, z: -4 }, { x: 6, z: -4 }, { x: 6, z: 4 }, { x: -6, z: 4 }],
        holes: [[{ x: -1.5, z: -1.2 }, { x: 1.5, z: -1.2 }, { x: 1.5, z: 1.2 }, { x: -1.5, z: 1.2 }]],
      },
      spawn: { x: -4, z: 0 },
      obstacles: [
        { id: 'cabinet', label: '収納棚', position: [4.8, 0.6, -2.5], size: [1.3, 1.2, 0.6] },
        { id: 'seat', label: '窓際の椅子', position: [-3.5, 0, -2.5], size: [0.5, 0.9, 0.5], yaw: Math.PI / 2, chair: { seatHeight: 0.44 } },
      ],
      destinations: [
        { id: 'east-side', label: '中庭の東側', position: { x: 4, z: 0 }, heading: -Math.PI / 2 },
        { id: 'west-side', label: '中庭の西側', position: { x: -4, z: 0 }, heading: Math.PI / 2 },
        { id: 'cabinet-side', label: '収納棚の前', position: { x: 4.8, z: -1.5 }, heading: Math.PI },
      ],
    },
  },
]
