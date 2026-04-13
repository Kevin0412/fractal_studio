<script setup lang="ts">
// ThreeDViewer.vue — three.js viewer.
// Supports two render modes:
//   GLB mode  : loads a glTF/GLB mesh artifact URL.
//   Voxel mode: renders a Minecraft-style InstancedMesh from a field byte array.
//               Each inside voxel is an axis-aligned cube; color encodes depth
//               (byte=1 → dark amber, byte=255 → bright amber / surface).

import { onMounted, onBeforeUnmount, ref, watch } from 'vue'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { TransitionVoxelResponse } from '../api'

const props = defineProps<{
  glbUrl?: string | null
  voxelData?: TransitionVoxelResponse | null
  loading?: boolean
}>()

const canvasEl = ref<HTMLCanvasElement | null>(null)

let renderer: THREE.WebGLRenderer | null = null
let scene: THREE.Scene | null = null
let camera: THREE.PerspectiveCamera | null = null
let controls: OrbitControls | null = null
let animId: number | null = null
let meshGroup: THREE.Group | null = null
let voxelMesh: THREE.InstancedMesh | null = null
let ro: ResizeObserver | null = null

function initThree() {
  const el = canvasEl.value!
  renderer = new THREE.WebGLRenderer({ canvas: el, antialias: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setClearColor(0x0a0b0d, 1)

  scene = new THREE.Scene()

  camera = new THREE.PerspectiveCamera(45, el.clientWidth / el.clientHeight, 0.001, 100)
  camera.position.set(0, 0.8, 3.2)

  // Key + fill + ambient — instrumentation aesthetic
  const key = new THREE.DirectionalLight(0xd7dae0, 1.8)
  key.position.set(1.5, 2, 2)
  scene.add(key)
  const fill = new THREE.DirectionalLight(0xd7dae0, 0.5)
  fill.position.set(-2, 1, -1)
  scene.add(fill)
  const ambient = new THREE.AmbientLight(0xd7dae0, 0.25)
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

// ── GLB ───────────────────────────────────────────────────────────────────────

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
  clearVoxels()
  try {
    const gltf = await loader.loadAsync(url)
    meshGroup = new THREE.Group()

    gltf.scene.traverse(child => {
      if (child instanceof THREE.Mesh) {
        const mat = new THREE.MeshStandardMaterial({
          color: 0xc8cdd5,
          roughness: 0.7,
          metalness: 0.1,
          side: THREE.DoubleSide,
        })
        child.material = mat
        meshGroup!.add(child.clone())
      }
    })

    centerAndNormalize(meshGroup)
    scene!.add(meshGroup)
    resetCamera()
  } catch (e) {
    console.error('ThreeDViewer: GLB load failed', e)
  }
}

// ── Voxels ────────────────────────────────────────────────────────────────────

function clearVoxels() {
  if (voxelMesh && scene) {
    scene.remove(voxelMesh)
    voxelMesh.geometry.dispose()
    ;(voxelMesh.material as THREE.Material).dispose()
    voxelMesh = null
  }
}

// Amber palette: deep inside = dark, surface = bright
// We lerp from darkAmber (#1a0a00) to brightAmber (#f0a030).
const DARK   = new THREE.Color(0x1a0800)
const BRIGHT = new THREE.Color(0xf0a030)
const TMP    = new THREE.Color()

function buildVoxels(data: TransitionVoxelResponse) {
  clearMesh()
  clearVoxels()

  const N = data.resolution
  const iso = data.isoLevel
  const extent = data.extent

  // Decode base64 → Uint8Array
  const binStr = atob(data.fieldB64)
  const bytes  = new Uint8Array(binStr.length)
  for (let i = 0; i < binStr.length; i++) bytes[i] = binStr.charCodeAt(i)

  // Count inside voxels
  let count = 0
  for (let i = 0; i < bytes.length; i++) if (bytes[i] > 0) count++
  if (count === 0) return

  // Cube geometry — slight gap (Minecraft gap look): scale to 0.92 of cell size
  const cellSize = (extent * 2) / N
  const geo = new THREE.BoxGeometry(cellSize * 0.92, cellSize * 0.92, cellSize * 0.92)
  const mat = new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0.05 })

  voxelMesh = new THREE.InstancedMesh(geo, mat, count)
  voxelMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
  voxelMesh.instanceColor = new THREE.InstancedBufferAttribute(
    new Float32Array(count * 3), 3
  )

  const dummy = new THREE.Object3D()
  const origin = -extent + cellSize * 0.5  // center of first cell

  let idx = 0
  for (let zi = 0; zi < N; zi++) {
    for (let yi = 0; yi < N; yi++) {
      for (let xi = 0; xi < N; xi++) {
        const byteVal = bytes[xi + N * (yi + N * zi)]
        if (byteVal === 0) continue

        // Position in world space (centered at origin)
        dummy.position.set(
          origin + xi * cellSize,
          origin + yi * cellSize,
          origin + zi * cellSize
        )
        dummy.updateMatrix()
        voxelMesh.setMatrixAt(idx, dummy.matrix)

        // Color: byte=1 (deep) → DARK, byte=255 (surface) → BRIGHT
        const t = (byteVal - 1) / 254
        TMP.copy(DARK).lerp(BRIGHT, Math.pow(t, 0.6))  // power < 1 → more bright surface
        voxelMesh.setColorAt(idx, TMP)

        idx++
      }
    }
  }

  voxelMesh.instanceMatrix.needsUpdate = true
  if (voxelMesh.instanceColor) voxelMesh.instanceColor.needsUpdate = true

  scene!.add(voxelMesh)
  resetCamera()
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function centerAndNormalize(group: THREE.Group) {
  const box = new THREE.Box3().setFromObject(group)
  const center = box.getCenter(new THREE.Vector3())
  const size = box.getSize(new THREE.Vector3()).length()
  group.position.sub(center)
  if (size > 0) group.scale.setScalar(2.0 / size)
}

function resetCamera() {
  controls!.reset()
  camera!.position.set(0, 0.8, 3.2)
  controls!.target.set(0, 0, 0)
  controls!.update()
}

// ── Watchers ──────────────────────────────────────────────────────────────────

watch(() => props.glbUrl, (url) => {
  if (url) loadGlb(url)
  else { clearMesh(); clearVoxels() }
})

watch(() => props.voxelData, (data) => {
  if (data) buildVoxels(data)
  else clearVoxels()
})

onMounted(() => {
  initThree()
  if (props.glbUrl)    loadGlb(props.glbUrl)
  if (props.voxelData) buildVoxels(props.voxelData)
})

onBeforeUnmount(() => {
  if (animId !== null) cancelAnimationFrame(animId)
  ro?.disconnect()
  controls?.dispose()
  renderer?.dispose()
  clearMesh()
  clearVoxels()
})
</script>

<template>
  <div class="viewer-wrap">
    <canvas ref="canvasEl" class="canvas" />
    <div v-if="loading" class="overlay">
      <span class="spinner">computing…</span>
    </div>
    <div v-if="!glbUrl && !voxelData && !loading" class="overlay">
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
