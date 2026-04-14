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
// Face culling and CCW winding are computed in the C++ backend.
// The response carries three compact base64 arrays:
//   posB64   — float32[faceCount * 4 * 3]   vertex positions (world space)
//   normB64  — int8[faceCount * 3]           one outward normal per face
//   depthB64 — uint8[faceCount]              depth byte (1=deep, 255=surface)

function clearVoxels() {
  if (voxelMesh && scene) {
    scene.remove(voxelMesh)
    voxelMesh.geometry.dispose()
    ;(voxelMesh.material as THREE.Material).dispose()
    voxelMesh = null
  }
}

const DARK   = new THREE.Color(0x1a0800)
const BRIGHT = new THREE.Color(0xf0a030)
const TMP    = new THREE.Color()

function decodeB64Bytes(b64: string): Uint8Array {
  const bin = atob(b64)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

function buildVoxels(data: TransitionVoxelResponse) {
  clearMesh(); clearVoxels()

  // Decode the three compact arrays from the backend
  const posBytes   = decodeB64Bytes(data.posB64)
  const normBytes  = decodeB64Bytes(data.normB64)
  const depthBytes = decodeB64Bytes(data.depthB64)

  const faceCount  = depthBytes.length
  if (faceCount === 0) return

  // posF32: positions are already in world space, 4 vertices × 3 floats per face
  const posF32 = new Float32Array(posBytes.buffer, posBytes.byteOffset, posBytes.byteLength / 4)

  // Normals: int8 per-face (3 components) → expand to per-vertex float (×4)
  const norF32 = new Float32Array(faceCount * 4 * 3)
  const normI8 = new Int8Array(normBytes.buffer, normBytes.byteOffset, normBytes.byteLength)
  for (let f = 0; f < faceCount; f++) {
    const nx = normI8[f * 3 + 0]
    const ny = normI8[f * 3 + 1]
    const nz = normI8[f * 3 + 2]
    for (let v = 0; v < 4; v++) {
      norF32[(f * 4 + v) * 3 + 0] = nx
      norF32[(f * 4 + v) * 3 + 1] = ny
      norF32[(f * 4 + v) * 3 + 2] = nz
    }
  }

  // Colors: depth byte → amber lerp, expand to per-vertex (×4)
  const colF32 = new Float32Array(faceCount * 4 * 3)
  for (let f = 0; f < faceCount; f++) {
    const t = Math.pow((depthBytes[f] - 1) / 254, 0.55)
    TMP.copy(DARK).lerp(BRIGHT, t)
    for (let v = 0; v < 4; v++) {
      colF32[(f * 4 + v) * 3 + 0] = TMP.r
      colF32[(f * 4 + v) * 3 + 1] = TMP.g
      colF32[(f * 4 + v) * 3 + 2] = TMP.b
    }
  }

  // Indices: 0,1,2,0,2,3 per face; use Uint32 to handle > 64k vertices
  const idxArr = new Uint32Array(faceCount * 6)
  for (let f = 0; f < faceCount; f++) {
    const b = f * 4
    idxArr[f * 6 + 0] = b;     idxArr[f * 6 + 1] = b + 1; idxArr[f * 6 + 2] = b + 2
    idxArr[f * 6 + 3] = b;     idxArr[f * 6 + 4] = b + 2; idxArr[f * 6 + 5] = b + 3
  }

  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.BufferAttribute(posF32,  3))
  geo.setAttribute('normal',   new THREE.BufferAttribute(norF32,  3))
  geo.setAttribute('color',    new THREE.BufferAttribute(colF32,  3))
  geo.setIndex(new THREE.BufferAttribute(idxArr, 1))

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
