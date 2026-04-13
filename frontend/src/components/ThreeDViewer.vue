<script setup lang="ts">
// ThreeDViewer.vue — three.js viewer with two render modes:
//
//   GLB mode  : loads a glTF/GLB mesh artifact URL.
//   Voxel mode: Minecraft-style surface mesh.
//               For each inside voxel, only the faces adjacent to an empty
//               neighbour are emitted into a single BufferGeometry, so zero
//               hidden geometry is uploaded to the GPU.
//               Color encodes depth: byte=1 (deep) → dark amber,
//                                    byte=255 (near surface) → bright amber.

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
let scene:    THREE.Scene | null = null
let camera:   THREE.PerspectiveCamera | null = null
let controls: OrbitControls | null = null
let animId:   number | null = null
let meshGroup: THREE.Group | null = null
let voxelMesh: THREE.Mesh | null = null
let ro: ResizeObserver | null = null

// ── Init ─────────────────────────────────────────────────────────────────────

function initThree() {
  const el = canvasEl.value!
  renderer = new THREE.WebGLRenderer({ canvas: el, antialias: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setClearColor(0x0a0b0d, 1)

  scene = new THREE.Scene()

  camera = new THREE.PerspectiveCamera(45, el.clientWidth / el.clientHeight, 0.001, 100)
  camera.position.set(0, 0.8, 3.2)

  const key  = new THREE.DirectionalLight(0xd7dae0, 1.8)
  key.position.set(1.5, 2, 2)
  scene.add(key)
  const fill = new THREE.DirectionalLight(0xd7dae0, 0.5)
  fill.position.set(-2, 1, -1)
  scene.add(fill)
  scene.add(new THREE.AmbientLight(0xd7dae0, 0.25))

  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping  = true
  controls.dampingFactor  = 0.08
  controls.minDistance    = 0.05
  controls.maxDistance    = 20

  ro = new ResizeObserver(() => resizeRenderer())
  ro.observe(el.parentElement!)
  resizeRenderer()
  loop()
}

function resizeRenderer() {
  if (!renderer || !camera || !canvasEl.value) return
  const p = canvasEl.value.parentElement!
  renderer.setSize(p.clientWidth, p.clientHeight, false)
  camera.aspect = p.clientWidth / p.clientHeight
  camera.updateProjectionMatrix()
}

function loop() {
  animId = requestAnimationFrame(loop)
  controls!.update()
  renderer!.render(scene!, camera!)
}

function resetCamera() {
  controls!.reset()
  camera!.position.set(0, 0.8, 3.2)
  controls!.target.set(0, 0, 0)
  controls!.update()
}

// ── GLB ──────────────────────────────────────────────────────────────────────

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
  clearMesh(); clearVoxels()
  try {
    const gltf = await loader.loadAsync(url)
    meshGroup = new THREE.Group()
    gltf.scene.traverse(child => {
      if (child instanceof THREE.Mesh) {
        child.material = new THREE.MeshStandardMaterial({
          color: 0xc8cdd5, roughness: 0.7, metalness: 0.1, side: THREE.DoubleSide,
        })
        meshGroup!.add(child.clone())
      }
    })
    // Normalize to unit bounding box
    const box    = new THREE.Box3().setFromObject(meshGroup)
    const center = box.getCenter(new THREE.Vector3())
    const size   = box.getSize(new THREE.Vector3()).length()
    meshGroup.position.sub(center)
    if (size > 0) meshGroup.scale.setScalar(2.0 / size)
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

// 6 face definitions: neighbour delta, outward normal, 4 corner offsets (half-cell units).
// Winding is CCW when viewed from outside (back-face culling compatible).
const FACE_DEFS = [
  { d: [ 1, 0, 0], n: [ 1, 0, 0], v: [[ 0.5,-0.5,-0.5],[ 0.5,-0.5, 0.5],[ 0.5, 0.5, 0.5],[ 0.5, 0.5,-0.5]] },
  { d: [-1, 0, 0], n: [-1, 0, 0], v: [[-0.5,-0.5, 0.5],[-0.5,-0.5,-0.5],[-0.5, 0.5,-0.5],[-0.5, 0.5, 0.5]] },
  { d: [ 0, 1, 0], n: [ 0, 1, 0], v: [[-0.5, 0.5,-0.5],[-0.5, 0.5, 0.5],[ 0.5, 0.5, 0.5],[ 0.5, 0.5,-0.5]] },
  { d: [ 0,-1, 0], n: [ 0,-1, 0], v: [[-0.5,-0.5, 0.5],[-0.5,-0.5,-0.5],[ 0.5,-0.5,-0.5],[ 0.5,-0.5, 0.5]] },
  { d: [ 0, 0, 1], n: [ 0, 0, 1], v: [[-0.5,-0.5, 0.5],[ 0.5,-0.5, 0.5],[ 0.5, 0.5, 0.5],[-0.5, 0.5, 0.5]] },
  { d: [ 0, 0,-1], n: [ 0, 0,-1], v: [[ 0.5,-0.5,-0.5],[-0.5,-0.5,-0.5],[-0.5, 0.5,-0.5],[ 0.5, 0.5,-0.5]] },
] as const

const DARK   = new THREE.Color(0x1a0800)
const BRIGHT = new THREE.Color(0xf0a030)
const TMP    = new THREE.Color()

function buildVoxels(data: TransitionVoxelResponse) {
  clearMesh(); clearVoxels()

  const N      = data.resolution
  const extent = data.extent

  // Decode base64 → Uint8Array (byte=0 outside, 1-255 inside/depth)
  const binStr = atob(data.fieldB64)
  const bytes  = new Uint8Array(binStr.length)
  for (let i = 0; i < binStr.length; i++) bytes[i] = binStr.charCodeAt(i)

  // Inline accessor — returns true if the voxel at (xi,yi,zi) is OUTSIDE.
  // Out-of-bounds coords are treated as outside (open boundary).
  function isOutside(xi: number, yi: number, zi: number): boolean {
    if (xi < 0 || xi >= N || yi < 0 || yi >= N || zi < 0 || zi >= N) return true
    return bytes[xi + N * (yi + N * zi)] === 0
  }

  const cellSize = (extent * 2) / N
  const origin   = -extent + cellSize * 0.5   // center of voxel [0,0,0]

  // Geometry buffers — grown dynamically.
  const pos: number[] = []
  const nor: number[] = []
  const col: number[] = []
  const idx: number[] = []

  for (let zi = 0; zi < N; zi++) {
    for (let yi = 0; yi < N; yi++) {
      for (let xi = 0; xi < N; xi++) {
        const depth = bytes[xi + N * (yi + N * zi)]
        if (depth === 0) continue   // outside — skip entirely

        // Voxel center in world space
        const cx = origin + xi * cellSize
        const cy = origin + yi * cellSize
        const cz = origin + zi * cellSize

        // Depth → amber color (power curve keeps surface bright)
        const t = (depth - 1) / 254
        TMP.copy(DARK).lerp(BRIGHT, Math.pow(t, 0.55))
        const { r, g, b } = TMP

        for (const face of FACE_DEFS) {
          // Skip this face if the neighbour in that direction is also inside.
          if (!isOutside(xi + face.d[0], yi + face.d[1], zi + face.d[2])) continue

          // Emit 4 vertices + 2 triangles for this exposed face.
          const base = pos.length / 3
          for (const [vx, vy, vz] of face.v) {
            pos.push(cx + vx * cellSize, cy + vy * cellSize, cz + vz * cellSize)
            nor.push(face.n[0], face.n[1], face.n[2])
            col.push(r, g, b)
          }
          idx.push(base, base + 1, base + 2,   base, base + 2, base + 3)
        }
      }
    }
  }

  if (pos.length === 0) return

  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  geo.setAttribute('normal',   new THREE.Float32BufferAttribute(nor, 3))
  geo.setAttribute('color',    new THREE.Float32BufferAttribute(col, 3))
  geo.setIndex(idx)

  voxelMesh = new THREE.Mesh(
    geo,
    new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85, metalness: 0.05 })
  )
  scene!.add(voxelMesh)
  resetCamera()
}

// ── Watchers ──────────────────────────────────────────────────────────────────

watch(() => props.glbUrl, url => {
  if (url) loadGlb(url)
  else { clearMesh(); clearVoxels() }
})

watch(() => props.voxelData, data => {
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
  position: relative; width: 100%; height: 100%;
  background: var(--bg); overflow: hidden;
}
.canvas { display: block; width: 100%; height: 100%; }
.overlay {
  position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
  pointer-events: none;
}
.spinner, .hint {
  font-family: var(--mono); font-size: 11px;
  color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.08em;
}
.spinner::before { content: '⬡ '; color: var(--accent); }
</style>
