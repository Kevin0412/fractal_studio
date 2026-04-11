<script setup lang="ts">
// ThreeDViewer.vue — three.js GLB viewer with orbit controls.
// Loads a GLB from a backend artifact URL; replaces scene mesh on prop change.

import { onMounted, onBeforeUnmount, ref, watch } from 'vue'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

const props = defineProps<{
  glbUrl: string | null
  loading?: boolean
}>()

const canvasEl = ref<HTMLCanvasElement | null>(null)

let renderer: THREE.WebGLRenderer | null = null
let scene: THREE.Scene | null = null
let camera: THREE.PerspectiveCamera | null = null
let controls: OrbitControls | null = null
let animId: number | null = null
let meshGroup: THREE.Group | null = null
let ro: ResizeObserver | null = null

function initThree() {
  const el = canvasEl.value!
  renderer = new THREE.WebGLRenderer({ canvas: el, antialias: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setClearColor(0x0a0b0d, 1)

  scene = new THREE.Scene()

  camera = new THREE.PerspectiveCamera(45, el.clientWidth / el.clientHeight, 0.001, 100)
  camera.position.set(0, 0.6, 2.5)

  // Lighting: key + fill, no rim glow — instrumentation aesthetic
  const key = new THREE.DirectionalLight(0xd7dae0, 1.6)
  key.position.set(1, 2, 2)
  scene.add(key)
  const fill = new THREE.DirectionalLight(0xd7dae0, 0.4)
  fill.position.set(-2, 1, -1)
  scene.add(fill)
  const ambient = new THREE.AmbientLight(0xd7dae0, 0.2)
  scene.add(ambient)

  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.dampingFactor = 0.08
  controls.minDistance = 0.05
  controls.maxDistance = 20

  ro = new ResizeObserver(() => resizeRenderer())
  ro.observe(el.parentElement!)

  resizeRenderer()
  loop()
}

function resizeRenderer() {
  if (!renderer || !camera || !canvasEl.value) return
  const parent = canvasEl.value.parentElement!
  const w = parent.clientWidth
  const h = parent.clientHeight
  renderer.setSize(w, h, false)
  camera.aspect = w / h
  camera.updateProjectionMatrix()
}

function loop() {
  animId = requestAnimationFrame(loop)
  controls!.update()
  renderer!.render(scene!, camera!)
}

function clearMesh() {
  if (meshGroup && scene) {
    scene.remove(meshGroup)
    meshGroup.traverse(o => {
      if (o instanceof THREE.Mesh) {
        o.geometry.dispose()
        if (Array.isArray(o.material)) o.material.forEach(m => m.dispose())
        else o.material.dispose()
      }
    })
    meshGroup = null
  }
}

const loader = new GLTFLoader()

async function loadGlb(url: string) {
  clearMesh()
  try {
    const gltf = await loader.loadAsync(url)
    meshGroup = new THREE.Group()

    gltf.scene.traverse(child => {
      if (child instanceof THREE.Mesh) {
        // Replace whatever material came with the GLB with our instrumentation look
        const mat = new THREE.MeshStandardMaterial({
          color: 0xc8cdd5,
          roughness: 0.7,
          metalness: 0.1,
          side: THREE.DoubleSide,
        })
        child.material = mat
        // Amber wireframe overlay on hover would need a second mesh; skip for now
        meshGroup!.add(child.clone())
      }
    })

    // Center and normalize to unit bounding box
    const box = new THREE.Box3().setFromObject(meshGroup)
    const center = box.getCenter(new THREE.Vector3())
    const size = box.getSize(new THREE.Vector3()).length()
    meshGroup.position.sub(center)
    if (size > 0) meshGroup.scale.setScalar(2.0 / size)

    scene!.add(meshGroup)

    // Reset camera to see the whole mesh
    controls!.reset()
    camera!.position.set(0, 0.6, 2.5)
    controls!.target.set(0, 0, 0)
    controls!.update()
  } catch (e) {
    console.error('ThreeDViewer: GLB load failed', e)
  }
}

watch(() => props.glbUrl, (url) => {
  if (url) loadGlb(url)
  else clearMesh()
})

onMounted(() => {
  initThree()
  if (props.glbUrl) loadGlb(props.glbUrl)
})

onBeforeUnmount(() => {
  if (animId !== null) cancelAnimationFrame(animId)
  ro?.disconnect()
  controls?.dispose()
  renderer?.dispose()
  clearMesh()
})
</script>

<template>
  <div class="viewer-wrap">
    <canvas ref="canvasEl" class="canvas" />
    <div v-if="loading" class="overlay">
      <span class="spinner">loading mesh…</span>
    </div>
    <div v-if="!glbUrl && !loading" class="overlay">
      <span class="hint">no mesh — select mode and compute</span>
    </div>
  </div>
</template>

<style scoped>
.viewer-wrap {
  position: relative;
  width: 100%;
  height: 100%;
  background: var(--bg);
  overflow: hidden;
}

.canvas {
  display: block;
  width: 100%;
  height: 100%;
}

.overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: none;
}

.spinner, .hint {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.spinner::before { content: '⬡ '; color: var(--accent); }
</style>
