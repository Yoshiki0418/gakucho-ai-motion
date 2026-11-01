import * as THREE from 'three'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { SSAOPass } from 'three/examples/jsm/postprocessing/SSAOPass.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'
import { RectAreaLightUniformsLib } from 'three/examples/jsm/lights/RectAreaLightUniformsLib.js'

export type HomeRendering = {
  ready: Promise<void>
  setEnabled: (enabled: boolean) => void
  render: (delta: number) => void
  resize: (width: number, height: number) => void
  dispose: () => void
}

/** Local HDR lighting and contact shading; the laboratory still renders directly. */
export const createHomeRendering = (
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  camera: THREE.PerspectiveCamera,
): HomeRendering => {
  RectAreaLightUniformsLib.init()
  let disposed = false
  let enabled = false
  const pmrem = new THREE.PMREMGenerator(renderer)
  const room = new RoomEnvironment(renderer)
  const fallback = pmrem.fromScene(room, 0.04)
  room.dispose()
  let environment = fallback.texture
  let hdrTarget: THREE.WebGLRenderTarget | null = null

  const size = renderer.getSize(new THREE.Vector2())
  const target = new THREE.WebGLRenderTarget(size.x, size.y, {
    type: THREE.HalfFloatType,
    samples: renderer.capabilities.isWebGL2 ? 4 : 0,
  })
  const composer = new EffectComposer(renderer, target)
  const renderPass = new RenderPass(scene, camera)
  const ambient = new SSAOPass(scene, camera, size.x, size.y, 16)
  ambient.kernelRadius = 0.24
  ambient.minDistance = 0.001
  ambient.maxDistance = 0.045
  ambient.enabled = renderer.capabilities.isWebGL2
  const output = new OutputPass()
  composer.addPass(renderPass)
  composer.addPass(ambient)
  composer.addPass(output)
  composer.setSize(size.x, size.y)
  // The beauty pass updates shadows once; the AO normals pass reuses that map.
  renderer.shadowMap.autoUpdate = false

  const ready = new Promise<void>((resolve) => {
    new RGBELoader().load('/environments/home/lighting/lebombo_1k.hdr', (texture) => {
      if (disposed) {
        texture.dispose()
        resolve()
        return
      }
      texture.mapping = THREE.EquirectangularReflectionMapping
      hdrTarget = pmrem.fromEquirectangular(texture)
      environment = hdrTarget.texture
      texture.dispose()
      if (enabled) scene.environment = environment
      resolve()
    }, undefined, (error) => {
      console.warn('[HomeEnvironment] HDR lighting unavailable; using indoor lighting', error)
      resolve()
    })
  })

  return {
    ready,
    setEnabled: (value) => {
      enabled = value
      renderer.shadowMap.enabled = value
      renderer.shadowMap.type = THREE.PCFSoftShadowMap
      renderer.toneMapping = value ? THREE.ACESFilmicToneMapping : THREE.NoToneMapping
      renderer.toneMappingExposure = value ? 0.92 : 1
      renderer.outputColorSpace = THREE.SRGBColorSpace
      scene.environment = value ? environment : null
    },
    render: (delta) => {
      if (disposed) return
      if (enabled) {
        renderer.shadowMap.needsUpdate = true
        // Projection changes when users switch cameras or resize the viewport.
        ambient.ssaoMaterial.uniforms.cameraNear.value = camera.near
        ambient.ssaoMaterial.uniforms.cameraFar.value = camera.far
        ambient.ssaoMaterial.uniforms.cameraProjectionMatrix.value.copy(camera.projectionMatrix)
        ambient.ssaoMaterial.uniforms.cameraInverseProjectionMatrix.value.copy(camera.projectionMatrixInverse)
        composer.render(delta)
      } else {
        renderer.render(scene, camera)
      }
    },
    resize: (width, height) => composer.setSize(width, height),
    dispose: () => {
      disposed = true
      if (scene.environment === environment) scene.environment = null
      ambient.dispose()
      ambient.ssaoMaterial.dispose()
      ambient.noiseTexture.dispose()
      output.dispose()
      composer.dispose()
      fallback.dispose()
      hdrTarget?.dispose()
      pmrem.dispose()
    },
  }
}
