'use client'

import React, { useRef, useEffect, useState, useCallback } from 'react'
import * as THREE from 'three'
import { BVHLoader } from 'three/examples/jsm/loaders/BVHLoader.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { DEFAULT_BVH } from './defaultBvh'

interface BvhViewerProps {
  /** BVH text data injected externally (e.g. from motion generation) */
  bvhText?: string | null
}

export const BvhViewer: React.FC<BvhViewerProps> = ({ bvhText }) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const mixerRef = useRef<THREE.AnimationMixer | null>(null)
  const clockRef = useRef(new THREE.Clock())
  const skeletonHelperRef = useRef<THREE.SkeletonHelper | null>(null)
  const animFrameRef = useRef<number>(0)
  const controlsRef = useRef<OrbitControls | null>(null)

  const [isPlaying, setIsPlaying] = useState(true)
  const [hasFile, setHasFile] = useState(true)
  const [fileName, setFileName] = useState('T-Pose (Default)')
  const [viewMode, setViewMode] = useState<'skeleton' | 'robot'>('robot')
  // loadBvh / buildRobotRig 内で常に最新の viewMode を参照するための ref
  const viewModeRef = useRef<'skeleton' | 'robot'>(viewMode)
  useEffect(() => {
    viewModeRef.current = viewMode
  }, [viewMode])

  // ロボットモード用 primitive を保持し、再生成時に dispose する
  const robotRigPartsRef = useRef<THREE.Mesh[]>([])

  const disposeRobotRig = useCallback(() => {
    for (const mesh of robotRigPartsRef.current) {
      mesh.parent?.remove(mesh)
      mesh.geometry.dispose()
      const m = mesh.material as THREE.Material | THREE.Material[]
      if (Array.isArray(m)) m.forEach((mm) => mm.dispose())
      else m.dispose()
    }
    robotRigPartsRef.current = []
  }, [])

  /**
   * 与えられた rootBone のボーン階層に沿って、白基調 + ダーク関節の
   * ヒューマノイドロボット風 primitive を attach する。
   *  - 各四肢: 太めの白カプセル (シェル) + 両端のダーク・カフ + 中央のシルバーリング
   *  - 各関節: ダークの球で接続部を覆う
   *  - 肩: 大きめの白パウルドロン
   *  - 胸部: 厚みのある白い胸甲 + 中央シーム
   *  - 腰: ダークのベルト装甲
   *  - 頭部: ダークの卵形ヘッド + バイザー (シアン発光)
   *  - 手・足: ダークの mitten / boot ブロック
   */
  const buildRobotRig = useCallback((rootBone: THREE.Bone) => {
    disposeRobotRig()

    const allBones: THREE.Bone[] = []
    rootBone.traverse((o) => {
      if ((o as THREE.Bone).isBone) allBones.push(o as THREE.Bone)
    })

    const findBone = (re: RegExp) => allBones.find((b) => re.test(b.name))
    const filterBones = (re: RegExp) => allBones.filter((b) => re.test(b.name))

    // 名前検出: フィンガー等にマッチしないよう末尾アンカー
    // "LeftHand" は /hand$/ にマッチ、 "LeftHandThumb1" はマッチしない
    const handBones = filterBones(/hand$|wrist$|palm$/i)
    const footBones = filterBones(/foot$|ankle$/i)
    const shoulderBones = filterBones(/shoulder$|clavicle$/i)
    const headBone = findBone(/head$/i) ?? findBone(/neck$/i)
    const hipsBone =
      findBone(/hips?$/i) ??
      findBone(/pelvis$/i) ??
      findBone(/^root$/i) ??
      rootBone
    // chestBone は肩の親 (= 肩がぶら下がっているスパイン) を最優先で採用
    const chestBone =
      (shoulderBones[0]?.parent as THREE.Bone | undefined) ??
      findBone(/chest$/i) ??
      findBone(/spine\s*4/i) ??
      findBone(/spine\s*3/i) ??
      findBone(/spine\s*2/i) ??
      findBone(/upperback$/i) ??
      findBone(/spine\s*1/i) ??
      findBone(/spine$/i)

    // フィンガー/つま先 (= 手/足の子孫) かどうか
    const isFingerOrToe = (b: THREE.Bone) => {
      let cur: THREE.Object3D | null = b.parent
      while (cur) {
        if ((cur as THREE.Bone).isBone) {
          const bb = cur as THREE.Bone
          if (handBones.includes(bb) || footBones.includes(bb)) return true
        }
        cur = cur.parent
      }
      return false
    }

    // 平均ボーン長 (フィンガー/つま先は除外して胴体・四肢の規模感を取得)
    let total = 0
    let count = 0
    for (const b of allBones) {
      if (isFingerOrToe(b)) continue
      if (handBones.includes(b) || footBones.includes(b)) continue
      for (const c of b.children) {
        if (!(c as THREE.Bone).isBone) continue
        const cb = c as THREE.Bone
        if (isFingerOrToe(cb)) continue
        const len = cb.position.length()
        if (len > 1e-3) {
          total += len
          count++
        }
      }
    }
    const avgLen = count > 0 ? total / count : 1

    // 胸甲の高さ用に、hips→chest のスパイン連鎖の総長を計算
    let spineChainLen = 0
    if (chestBone && hipsBone) {
      let cur: THREE.Object3D | null = chestBone
      while (cur && cur !== hipsBone) {
        if ((cur as THREE.Bone).isBone) {
          spineChainLen += (cur as THREE.Bone).position.length()
        }
        cur = cur.parent
      }
    }
    if (spineChainLen < 1e-3) spineChainLen = avgLen * 3

    const limbR = avgLen * 0.28
    const cuffR = limbR * 1.18
    const cuffH = avgLen * 0.12
    const ringR = limbR * 1.12
    const ringH = cuffH * 0.4
    const jointR = limbR * 1.05

    const shellMat = new THREE.MeshStandardMaterial({
      color: 0xeaecef,
      roughness: 0.42,
      metalness: 0.18,
    })
    const innerMat = new THREE.MeshStandardMaterial({
      color: 0x16191f,
      roughness: 0.34,
      metalness: 0.82,
    })
    const accentMat = new THREE.MeshStandardMaterial({
      color: 0x9aa1a8,
      roughness: 0.28,
      metalness: 0.9,
    })
    const visorMat = new THREE.MeshStandardMaterial({
      color: 0x05080c,
      roughness: 0.08,
      metalness: 0.4,
      emissive: 0x00d4ff,
      emissiveIntensity: 0.85,
    })
    const visorRimMat = new THREE.MeshStandardMaterial({
      color: 0x101318,
      roughness: 0.25,
      metalness: 0.88,
    })

    const tracked: THREE.Mesh[] = []
    const yAxis = new THREE.Vector3(0, 1, 0)

    const torsoBoneSet = new Set<THREE.Bone>(
      [hipsBone, chestBone].filter(Boolean) as THREE.Bone[],
    )
    if (chestBone) {
      let cur: THREE.Object3D | null = chestBone
      while (cur && cur !== hipsBone) {
        if ((cur as THREE.Bone).isBone) torsoBoneSet.add(cur as THREE.Bone)
        cur = cur.parent
      }
    }

    const addPart = (mesh: THREE.Mesh, parent: THREE.Object3D) => {
      mesh.visible = viewModeRef.current === 'robot'
      parent.add(mesh)
      tracked.push(mesh)
    }

    const isHandBone = (b: THREE.Bone) => handBones.includes(b)
    const isFootBone = (b: THREE.Bone) => footBones.includes(b)
    const isHeadBone = (b: THREE.Bone) => b === headBone

    // ── 各ボーンと子ボーンの間にシェル + カフを生成 ──
    // 注: 胴体ボーン (torsoBoneSet) は「自分→子」の中で外向き接続 (Spine4→Neck, Hips→UpLeg 等)
    //     のみ描き、内部のスパイン連鎖 (Spine→Spine1 等) は胸甲が覆うので描かない。
    for (const bone of allBones) {
      if (isFingerOrToe(bone)) continue
      if (isHandBone(bone) || isFootBone(bone)) continue
      if (isHeadBone(bone)) continue

      for (const child of bone.children) {
        if (!(child as THREE.Bone).isBone) continue
        const childBone = child as THREE.Bone
        if (isFingerOrToe(childBone)) continue
        // 内部スパイン (torso→torso) は胸甲で覆うのでスキップ
        if (torsoBoneSet.has(bone) && torsoBoneSet.has(childBone)) continue
        // 肩オフセットはパウルドロンで覆うのでスキップ (胴体→肩)
        if (torsoBoneSet.has(bone) && shoulderBones.includes(childBone)) continue

        const childPos = childBone.position
        const dist = childPos.length()
        if (dist < 1e-3) continue
        const dir = childPos.clone().normalize()

        // 短いオフセットでもカプセルがはみ出さないよう、半径を距離に応じて縮める
        const segLimbR = Math.min(limbR, dist * 0.45)
        const segCuffR = segLimbR * 1.14

        const cylLen = Math.max(dist - 2 * segLimbR, dist * 0.05)
        const shell = new THREE.Mesh(
          new THREE.CapsuleGeometry(segLimbR, cylLen, 8, 18),
          shellMat,
        )
        shell.position.copy(childPos).multiplyScalar(0.5)
        shell.quaternion.setFromUnitVectors(yAxis, dir)
        addPart(shell, bone)

        // 短すぎるセグメント (肩 offset、首根、hip→thigh 等) ではカフとリングを省く
        const drawCuffs = dist > limbR * 1.4
        if (drawCuffs) {
          const cuff1 = new THREE.Mesh(
            new THREE.CylinderGeometry(segCuffR, segCuffR, cuffH, 24),
            innerMat,
          )
          cuff1.position.copy(dir).multiplyScalar(dist * 0.16)
          cuff1.quaternion.setFromUnitVectors(yAxis, dir)
          addPart(cuff1, bone)

          const cuff2 = new THREE.Mesh(
            new THREE.CylinderGeometry(segCuffR, segCuffR, cuffH, 24),
            innerMat,
          )
          cuff2.position.copy(dir).multiplyScalar(dist * 0.84)
          cuff2.quaternion.setFromUnitVectors(yAxis, dir)
          addPart(cuff2, bone)

          const ring = new THREE.Mesh(
            new THREE.CylinderGeometry(segCuffR * 0.95, segCuffR * 0.95, ringH, 24),
            accentMat,
          )
          ring.position.copy(dir).multiplyScalar(dist * 0.5)
          ring.quaternion.setFromUnitVectors(yAxis, dir)
          addPart(ring, bone)
        }
      }

      // 関節接続部のダーク球 (専用形状で覆う部位は除く)
      if (
        !shoulderBones.includes(bone) &&
        !torsoBoneSet.has(bone) &&
        !isHeadBone(bone) &&
        !isHandBone(bone) &&
        !isFootBone(bone)
      ) {
        const jointBall = new THREE.Mesh(
          new THREE.SphereGeometry(jointR, 22, 16),
          innerMat,
        )
        addPart(jointBall, bone)
      }
    }

    // ── 肩パウルドロン ──
    for (const sb of shoulderBones) {
      const firstChild = sb.children.find(
        (c) => (c as THREE.Bone).isBone,
      ) as THREE.Bone | undefined
      const dir =
        firstChild && firstChild.position.lengthSq() > 1e-6
          ? firstChild.position.clone().normalize()
          : new THREE.Vector3(1, 0, 0)

      const pauldron = new THREE.Mesh(
        new THREE.SphereGeometry(limbR * 1.55, 28, 22),
        shellMat,
      )
      pauldron.scale.set(1.0, 0.85, 1.0)
      pauldron.position.copy(dir).multiplyScalar(limbR * 0.55)
      addPart(pauldron, sb)

      const decal = new THREE.Mesh(
        new THREE.CylinderGeometry(limbR * 0.5, limbR * 0.5, limbR * 0.16, 24),
        innerMat,
      )
      decal.quaternion.setFromUnitVectors(yAxis, dir)
      decal.position.copy(dir).multiplyScalar(limbR * 1.5)
      addPart(decal, sb)
    }

    // 肩幅を実測。肩ボーン自体の位置だけだと幅が短くなるので、
    // 「肩 offset + 上腕 offset」= 実質の腕の付け根位置 で計測する
    let shoulderSpan = avgLen * 2.5
    if (shoulderBones.length >= 2) {
      const armRoots: THREE.Vector3[] = []
      for (const sb of shoulderBones) {
        const armChild = sb.children.find(
          (c) => (c as THREE.Bone).isBone && !shoulderBones.includes(c as THREE.Bone),
        ) as THREE.Bone | undefined
        if (armChild) {
          armRoots.push(sb.position.clone().add(armChild.position))
        } else {
          armRoots.push(sb.position.clone())
        }
      }
      if (armRoots.length >= 2) {
        const span = armRoots[0].distanceTo(armRoots[1])
        if (span > 1e-3) shoulderSpan = Math.max(span * 0.95, avgLen * 2.0)
      }
    }

    // ── 胸甲 (hips→chest のスパイン連鎖全体を覆う) ──
    if (chestBone) {
      const chestH = spineChainLen * 1.05
      const chestW = shoulderSpan * 0.82
      const chestD = Math.max(spineChainLen * 0.42, avgLen * 0.85)
      // chestBone (= スパイン連鎖最上端) のローカル -Y 方向に中心を取り、胴体全体を覆う
      const chestCenterY = -spineChainLen * 0.45

      const chest = new THREE.Mesh(
        new THREE.BoxGeometry(chestW, chestH, chestD),
        shellMat,
      )
      chest.position.y = chestCenterY
      addPart(chest, chestBone)

      // 中央の縦シーム (ダーク帯)
      const seam = new THREE.Mesh(
        new THREE.BoxGeometry(chestW * 0.07, chestH * 0.92, chestD * 0.05),
        innerMat,
      )
      seam.position.set(0, chestCenterY, chestD * 0.5 + chestD * 0.005)
      addPart(seam, chestBone)

      // 首〜肩のダーク・ヨーク (胸甲の上端付近)
      const yoke = new THREE.Mesh(
        new THREE.BoxGeometry(chestW * 0.82, chestH * 0.18, chestD * 0.95),
        innerMat,
      )
      yoke.position.set(0, chestCenterY + chestH * 0.46, 0)
      addPart(yoke, chestBone)

      // 腹部側のダーク帯 (胸甲下端のセパレータ)
      const abdoBand = new THREE.Mesh(
        new THREE.BoxGeometry(chestW * 0.95, chestH * 0.1, chestD * 1.0),
        innerMat,
      )
      abdoBand.position.set(0, chestCenterY - chestH * 0.5, 0)
      addPart(abdoBand, chestBone)
    }

    // ── 腰ベルト ──
    if (hipsBone) {
      const beltW = Math.max(shoulderSpan * 0.70, avgLen * 1.7)
      const beltH = avgLen * 0.7
      const beltD = Math.max(spineChainLen * 0.42, avgLen * 0.9)
      const belt = new THREE.Mesh(
        new THREE.BoxGeometry(beltW, beltH, beltD),
        innerMat,
      )
      belt.position.y = -beltH * 0.1
      addPart(belt, hipsBone)

      // 腹部の小さなアクセント発光
      const accent = new THREE.Mesh(
        new THREE.BoxGeometry(beltW * 0.22, beltH * 0.18, beltD * 0.04),
        visorMat,
      )
      accent.position.set(0, -beltH * 0.1, beltD * 0.5 + 0.001)
      addPart(accent, hipsBone)
    }

    // ── 手 ──
    for (const hb of handBones) {
      // 指が複数あるとき "親指1本目" が斜め方向にあるので、
      // すべての子 (= 指) の方向平均を掌の伸びる方向として使う
      const fingerChildren = (hb.children.filter(
        (c) => (c as THREE.Bone).isBone,
      ) as THREE.Bone[]).filter((c) => c.position.lengthSq() > 1e-6)
      let dir: THREE.Vector3
      if (fingerChildren.length > 0) {
        const sum = new THREE.Vector3()
        for (const fc of fingerChildren) {
          sum.add(fc.position.clone().normalize())
        }
        dir = sum.lengthSq() > 1e-6 ? sum.normalize() : yAxis.clone()
      } else {
        dir = yAxis.clone()
      }

      // 掌のサイズは平均ボーン長基準 (avgLen) でなく、
      // 前腕から手までの長さ (= hb.position の長さ) を基準にすると過大化を防げる
      const forearmToHand = hb.position.length()
      const handBase = forearmToHand > 1e-3 ? forearmToHand : avgLen
      const handLen = handBase * 0.5
      const handThk = handBase * 0.24
      const handWid = handBase * 0.30

      const palm = new THREE.Mesh(
        new THREE.BoxGeometry(handWid, handLen, handThk),
        innerMat,
      )
      palm.quaternion.setFromUnitVectors(yAxis, dir)
      palm.position.copy(dir).multiplyScalar(handLen * 0.5)
      addPart(palm, hb)

      const wristRing = new THREE.Mesh(
        new THREE.CylinderGeometry(limbR * 1.05, limbR * 1.05, cuffH, 24),
        accentMat,
      )
      wristRing.quaternion.setFromUnitVectors(yAxis, dir)
      addPart(wristRing, hb)
    }

    // ── 足 ──
    for (const fb of footBones) {
      const firstChild = fb.children.find(
        (c) => (c as THREE.Bone).isBone,
      ) as THREE.Bone | undefined
      const fwd =
        firstChild && firstChild.position.lengthSq() > 1e-6
          ? firstChild.position.clone().normalize()
          : new THREE.Vector3(0, 0, 1)
      const bootLen = avgLen * 1.3
      const boot = new THREE.Mesh(
        new THREE.BoxGeometry(avgLen * 0.5, avgLen * 0.4, bootLen),
        innerMat,
      )
      const zAxis = new THREE.Vector3(0, 0, 1)
      boot.quaternion.setFromUnitVectors(zAxis, fwd)
      boot.position.copy(fwd).multiplyScalar(bootLen * 0.4)
      boot.position.y -= avgLen * 0.14
      addPart(boot, fb)

      const ankleRing = new THREE.Mesh(
        new THREE.CylinderGeometry(limbR * 1.05, limbR * 1.05, cuffH, 24),
        accentMat,
      )
      addPart(ankleRing, fb)
    }

    // ── 頭部 ──
    if (headBone) {
      // 頭サイズは「Head ボーン自身のオフセット (= neck 最終→head 距離)」を基準にし、
      // BVH ごとのスケール差で頭が肥大化するのを防ぐ
      const headOffset = headBone.position.length()
      const headR = Math.max(headOffset * 1.05, avgLen * 0.5)
      const headH = headR * 1.18

      const head = new THREE.Mesh(
        new THREE.SphereGeometry(headR, 36, 28),
        innerMat,
      )
      head.scale.set(1.0, 1.18, 1.06)
      head.position.y = headH * 0.95
      addPart(head, headBone)

      const visorRimW = headR * 1.55
      const visorRimH = headR * 0.62
      const visorRimD = headR * 0.28
      const visorRim = new THREE.Mesh(
        new THREE.BoxGeometry(visorRimW, visorRimH, visorRimD),
        visorRimMat,
      )
      visorRim.position.set(0, headH * 1.05, headR * 0.78)
      addPart(visorRim, headBone)

      const visor = new THREE.Mesh(
        new THREE.BoxGeometry(visorRimW * 0.92, visorRimH * 0.46, visorRimD * 0.55),
        visorMat,
      )
      visor.position.set(0, headH * 1.05, headR * 0.78 + visorRimD * 0.5)
      addPart(visor, headBone)

      const chinLine = new THREE.Mesh(
        new THREE.BoxGeometry(headR * 0.85, headR * 0.07, headR * 0.05),
        accentMat,
      )
      chinLine.position.set(0, headH * 0.55, headR * 0.85)
      addPart(chinLine, headBone)

      const back = new THREE.Mesh(
        new THREE.SphereGeometry(headR * 0.98, 24, 18),
        accentMat,
      )
      back.scale.set(0.95, 1.05, 0.55)
      back.position.set(0, headH * 0.95, -headR * 0.55)
      addPart(back, headBone)
    }

    robotRigPartsRef.current = tracked
    console.log(
      `[BvhViewer] Robot rig built: parts=${tracked.length}, ` +
        `avgLen=${avgLen.toFixed(2)}, limbR=${limbR.toFixed(2)}, ` +
        `head=${headBone?.name ?? 'none'}, chest=${chestBone?.name ?? 'none'}, ` +
        `hips=${hipsBone?.name ?? 'none'}, shoulders=[${shoulderBones.map((b) => b.name).join(',')}], ` +
        `hands=[${handBones.map((b) => b.name).join(',')}], ` +
        `feet=[${footBones.map((b) => b.name).join(',')}]`,
    )
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [disposeRobotRig])

  // Initialize Three.js scene (deferred to ensure layout is computed)
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const initId = requestAnimationFrame(() => {
      const w = container.clientWidth || 640
      const h = container.clientHeight || 480

      const scene = new THREE.Scene()
      scene.background = new THREE.Color(0x0f172a)
      sceneRef.current = scene

      const camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 1000)
      camera.position.set(0, 100, 300)
      cameraRef.current = camera

      const renderer = new THREE.WebGLRenderer({ antialias: true })
      renderer.setSize(w, h)
      renderer.setPixelRatio(window.devicePixelRatio)
      container.appendChild(renderer.domElement)
      rendererRef.current = renderer

      const controls = new OrbitControls(camera, renderer.domElement)
      controls.target.set(0, 80, 0)
      controls.update()
      controlsRef.current = controls

      // Grid
      const grid = new THREE.GridHelper(400, 20, 0x334155, 0x1e293b)
      scene.add(grid)

      // Lights
      scene.add(new THREE.AmbientLight(0xffffff, 0.8))
      const dirLight = new THREE.DirectionalLight(0xffffff, 0.5)
      dirLight.position.set(100, 200, 100)
      scene.add(dirLight)

      // Animation loop
      let frameCount = 0
      const animate = () => {
        animFrameRef.current = requestAnimationFrame(animate)
        const delta = clockRef.current.getDelta()
        if (mixerRef.current) {
          mixerRef.current.update(delta)

          frameCount++
          if (frameCount % 120 === 0) {
            const rootBone = scene.userData.currentRootBone
            const mixerTime = mixerRef.current.time
            const timeScale = mixerRef.current.timeScale
            if (rootBone) {
              console.log(
                `[BvhViewer] heartbeat: mixerTime=${mixerTime.toFixed(2)}s, ` +
                `timeScale=${timeScale}, delta=${delta.toFixed(4)}s, ` +
                `rootBonePos=[${rootBone.position.x.toFixed(2)}, ${rootBone.position.y.toFixed(2)}, ${rootBone.position.z.toFixed(2)}], ` +
                `rootBoneQuat=[${rootBone.quaternion.x.toFixed(2)}, ${rootBone.quaternion.y.toFixed(2)}, ${rootBone.quaternion.z.toFixed(2)}, ${rootBone.quaternion.w.toFixed(2)}]`
              )
            } else {
              console.log(
                `[BvhViewer] heartbeat: mixerTime=${mixerTime.toFixed(2)}s, but no rootBone in scene.userData`
              )
            }
          }
        }

        // Force update skeleton helper geometry if needed (older or specific three.js versions)
        if (skeletonHelperRef.current) {
          skeletonHelperRef.current.update?.()
        }

        controls.update()
        renderer.render(scene, camera)
      }
      animate()

      // Resize handler
      const handleResize = () => {
        const rw = container.clientWidth
        const rh = container.clientHeight
        if (rw === 0 || rh === 0) return
        camera.aspect = rw / rh
        camera.updateProjectionMatrix()
        renderer.setSize(rw, rh)
      }
      const resizeObserver = new ResizeObserver(handleResize)
      resizeObserver.observe(container)

      // Load default T-pose BVH
      if (typeof DEFAULT_BVH === 'string' && DEFAULT_BVH.length > 0) {
        try {
          const loader = new BVHLoader()
          const result = loader.parse(DEFAULT_BVH)
          const rootBone = result.skeleton.bones[0]
          scene.add(rootBone)
          scene.userData.currentRootBone = rootBone

          const skHelper = new THREE.SkeletonHelper(rootBone)
          const helperMat = skHelper.material as THREE.LineBasicMaterial
          helperMat.linewidth = 2
          helperMat.color = new THREE.Color(0x38bdf8)
          // 初期 viewMode は 'robot' 固定なので骨格は非表示で開始
          skHelper.visible = false
          scene.add(skHelper)
          skeletonHelperRef.current = skHelper

          // デフォルト T-Pose にもロボットリグを attach
          buildRobotRig(rootBone)

          // Auto-fit camera
          scene.updateMatrixWorld(true)
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            ; (skHelper as any).update?.()

          const box = new THREE.Box3().setFromObject(skHelper)
          const bSize = new THREE.Vector3()
          const bCenter = new THREE.Vector3()
          box.getSize(bSize)
          box.getCenter(bCenter)
          let maxDim = Math.max(bSize.x, bSize.y, bSize.z)

          // Fallback if maxDim is still 0 or very small
          if (maxDim < 0.1) maxDim = 2.0

          const fitDist = maxDim * 2.5
          console.log('[BvhViewer] auto-fit dimensions:', { maxDim, fitDist, bCenter, bSize })

          camera.near = fitDist * 0.001
          camera.far = fitDist * 20
          camera.updateProjectionMatrix()
          camera.position.set(bCenter.x + fitDist * 0.5, bCenter.y + maxDim * 0.5, bCenter.z + fitDist)
          controls.target.copy(bCenter)
          controls.update()

          // Adjust grid
          scene.remove(grid)
          const autoGrid = new THREE.GridHelper(maxDim * 4, 20, 0x334155, 0x1e293b)
          autoGrid.position.y = box.min.y
          scene.add(autoGrid)
        } catch (e) {
          console.error('[BvhViewer] Failed to load default BVH:', e)
        }
      }
    }) // end requestAnimationFrame

    return () => {
      console.log('[BvhViewer] cleanup: disposing renderer / controls / animation frame')
      cancelAnimationFrame(initId)
      cancelAnimationFrame(animFrameRef.current)
      if (controlsRef.current) controlsRef.current.dispose()
      if (rendererRef.current) {
        rendererRef.current.dispose()
        if (container.contains(rendererRef.current.domElement)) {
          container.removeChild(rendererRef.current.domElement)
        }
      }
    }
    // 初期化はマウント時のみ。bvhText の更新は別 useEffect で loadBvh を呼ぶ
  }, [])

  // Update mixer play/pause
  useEffect(() => {
    if (mixerRef.current) {
      mixerRef.current.timeScale = isPlaying ? 1 : 0
    }
  }, [isPlaying])

  const loadBvh = useCallback((text: string, name: string) => {
    const scene = sceneRef.current
    const camera = cameraRef.current
    const controls = controlsRef.current
    console.log(
      `[BvhViewer] loadBvh called: name="${name}", text_len=${text.length}, ` +
      `refs={scene:${!!scene}, camera:${!!camera}, controls:${!!controls}}`
    )
    if (!scene || !camera || !controls) {
      console.warn(
        '[BvhViewer] loadBvh aborted: scene/camera/controls not yet initialized — ' +
        'check init useEffect timing'
      )
      return
    }

    // Remove previous skeleton & root bone
    if (skeletonHelperRef.current) {
      scene.remove(skeletonHelperRef.current)
      skeletonHelperRef.current.dispose()
      skeletonHelperRef.current = null
    }

    if (scene.userData.currentRootBone) {
      scene.remove(scene.userData.currentRootBone)
    }

    // Stop any previous mixer to avoid leaked actions
    if (mixerRef.current) {
      mixerRef.current.stopAllAction()
      mixerRef.current.uncacheRoot(mixerRef.current.getRoot())
      mixerRef.current = null
    }

    const loader = new BVHLoader()
    try {
      const head = text.slice(0, 120).replace(/\n/g, '\\n')
      console.log(
        `[BvhViewer] Parsing BVH text (length=${text.length}, head="${head}")`
      )
      const result = loader.parse(text)

      const rootBone = result.skeleton.bones[0]
      scene.add(rootBone)
      scene.userData.currentRootBone = rootBone

      const tracks = result.clip.tracks
      console.log(
        `[BvhViewer] BVH parsed: bones=${result.skeleton.bones.length}, ` +
        `clip.duration=${result.clip.duration}s, clip.tracks=${tracks.length}, ` +
        `firstTrackName="${tracks[0]?.name}", firstTrackValuesLen=${tracks[0]?.values.length}`
      )

      const skeletonHelper = new THREE.SkeletonHelper(rootBone)
      const mat = skeletonHelper.material as THREE.LineBasicMaterial
      mat.linewidth = 2
      mat.color = new THREE.Color(0x38bdf8)
      skeletonHelper.visible = viewModeRef.current === 'skeleton'
      scene.add(skeletonHelper)
      skeletonHelperRef.current = skeletonHelper

      // ロボット表示用の primitive をボーンに attach
      buildRobotRig(rootBone)

      // Auto-fit camera to skeleton size
      scene.updateMatrixWorld(true)
      skeletonHelper.update?.()

      const box = new THREE.Box3().setFromObject(skeletonHelper)
      const size = new THREE.Vector3()
      const center = new THREE.Vector3()
      box.getSize(size)
      box.getCenter(center)
      let maxDim = Math.max(size.x, size.y, size.z)
      if (maxDim < 0.1) maxDim = 2.0

      const fitDistance = maxDim * 2.5
      camera.near = fitDistance * 0.001
      camera.far = fitDistance * 20
      camera.updateProjectionMatrix()
      camera.position.set(center.x + fitDistance * 0.5, center.y + maxDim * 0.5, center.z + fitDistance)
      controls.target.copy(center)
      controls.update()

      // Update grid to match skeleton scale
      const existingGrid = scene.children.find((c) => c instanceof THREE.GridHelper)
      if (existingGrid) scene.remove(existingGrid)
      const gridSize = maxDim * 4
      const newGrid = new THREE.GridHelper(gridSize, 20, 0x334155, 0x1e293b)
      newGrid.position.y = 0 // Keep grid at true floor level
      scene.add(newGrid)

      const mixer = new THREE.AnimationMixer(rootBone)
      const action = mixer.clipAction(result.clip)
      // アニメーションを一度だけ再生し、最後のフレームで停止させる
      action.setLoop(THREE.LoopOnce, 1)
      action.clampWhenFinished = true
      action.play()
      mixerRef.current = mixer
      console.log(
        `[BvhViewer] Mixer started: isRunning=${action.isRunning()}, ` +
        `enabled=${action.enabled}, weight=${action.weight}, ` +
        `clipDuration=${result.clip.duration}s, mixer.timeScale=${mixer.timeScale}`
      )

      setHasFile(true)
      setFileName(name)
      setIsPlaying(true)
      clockRef.current = new THREE.Clock()
    } catch (e) {
      console.error('[BvhViewer] Error parsing BVH text:', e)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [buildRobotRig])

  // viewMode の切替: 既存の SkeletonHelper / ロボットリグの visible を更新
  useEffect(() => {
    if (skeletonHelperRef.current) {
      skeletonHelperRef.current.visible = viewMode === 'skeleton'
    }
    for (const m of robotRigPartsRef.current) {
      m.visible = viewMode === 'robot'
    }
    console.log(`[BvhViewer] viewMode changed → '${viewMode}'`)
  }, [viewMode])

  // Auto-load BVH from external prop (agent motion generation)
  useEffect(() => {
    console.log(
      `[BvhViewer] bvhText prop changed: ${bvhText ? `length=${bvhText.length}` : 'null/undefined'
      }`
    )
    if (bvhText && bvhText.length > 0) {
      loadBvh(bvhText, '🤖 AI Generated Motion')
    }
  }, [bvhText, loadBvh])

  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        loadBvh(reader.result, file.name)
      }
    }
    reader.readAsText(file)
  }, [loadBvh])

  const btnStyle: React.CSSProperties = {
    padding: '6px 14px',
    borderRadius: 6,
    border: '1px solid rgba(148,163,184,0.3)',
    background: 'rgba(30,41,59,0.8)',
    color: '#e2e8f0',
    fontSize: 12,
    cursor: 'pointer',
    transition: 'background 0.2s',
  }

  const segBase: React.CSSProperties = {
    padding: '6px 12px',
    border: '1px solid rgba(148,163,184,0.3)',
    fontSize: 12,
    cursor: 'pointer',
    transition: 'all 0.15s',
  }
  const segActive: React.CSSProperties = {
    background: 'rgba(56,189,248,0.18)',
    color: '#38bdf8',
    borderColor: 'rgba(56,189,248,0.6)',
  }
  const segInactive: React.CSSProperties = {
    background: 'rgba(30,41,59,0.8)',
    color: '#94a3b8',
  }

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />

      {/* Controls overlay */}
      <div
        style={{
          position: 'absolute',
          bottom: 16,
          left: '50%',
          transform: 'translateX(-50%)',
          display: 'flex',
          gap: 8,
          alignItems: 'center',
          padding: '8px 16px',
          borderRadius: 12,
          background: 'rgba(15,23,42,0.85)',
          backdropFilter: 'blur(8px)',
          border: '1px solid rgba(148,163,184,0.2)',
        }}
      >
        <label style={{ ...btnStyle, display: 'flex', alignItems: 'center', gap: 4 }}>
          📂 BVH ファイル
          <input
            type="file"
            accept=".bvh"
            onChange={handleFileUpload}
            style={{ display: 'none' }}
          />
        </label>

        {/* 表示モード切替 (ロボット / 骨格) */}
        <div
          style={{
            display: 'flex',
            borderRadius: 6,
            overflow: 'hidden',
            border: '1px solid rgba(148,163,184,0.3)',
          }}
        >
          <button
            type="button"
            onClick={() => setViewMode('robot')}
            style={{
              ...segBase,
              ...(viewMode === 'robot' ? segActive : segInactive),
              borderTop: 'none',
              borderBottom: 'none',
              borderLeft: 'none',
              borderRight: '1px solid rgba(148,163,184,0.3)',
            }}
          >
            🤖 ロボット
          </button>
          <button
            type="button"
            onClick={() => setViewMode('skeleton')}
            style={{
              ...segBase,
              ...(viewMode === 'skeleton' ? segActive : segInactive),
              border: 'none',
            }}
          >
            🦴 骨格
          </button>
        </div>

        {hasFile && (
          <>
            <button style={btnStyle} onClick={() => setIsPlaying(!isPlaying)}>
              {isPlaying ? '⏸ 一時停止' : '▶ 再生'}
            </button>
            <button
              style={btnStyle}
              onClick={() => {
                if (mixerRef.current) {
                  mixerRef.current.setTime(0)
                  setIsPlaying(true)
                }
              }}
            >
              ⏮ リセット
            </button>
          </>
        )}
      </div>

      {/* File name badge */}
      {hasFile && (
        <div
          style={{
            position: 'absolute',
            top: 12,
            left: 16,
            padding: '4px 10px',
            borderRadius: 6,
            background: 'rgba(15,23,42,0.75)',
            border: '1px solid rgba(148,163,184,0.3)',
            fontSize: 11,
            color: '#94a3b8',
          }}
        >
          🦴 {fileName}
        </div>
      )}

      {/* Empty state */}
      {!hasFile && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            color: '#64748b',
            fontSize: 14,
            gap: 8,
            pointerEvents: 'none',
          }}
        >
          <span style={{ fontSize: 40 }}>🦴</span>
          <span>BVH ファイルをアップロードして</span>
          <span>モーションをプレビューできます</span>
        </div>
      )}
    </div>
  )
}
