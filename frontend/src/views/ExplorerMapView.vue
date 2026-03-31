<template>
  <div>
    <ParamEditor
      title="Fractal Explorer Parameters"
      :iterations="iterations"
      :color-map="colorMap"
      :center-re="centerRe"
      :center-im="centerIm"
      :scale="scale"
      @style-change="onStyleChange"
      @apply-viewport="onApplyViewport"
    />
    <FormulaPanel />
    <section class="view-card">
      <h3>{{ t('explorerMap') }}</h3>
      <p>
        Render Source:
        <select v-model="renderSource" @change="onRenderSourceChange">
          <option value="map">Map Variants</option>
          <option value="hs">HS Families Images</option>
        </select>
      </p>
      <p>Center: {{ centerRe }} + {{ centerIm }}i</p>
      <p>Scale: {{ scale }}</p>
      <p>Iterations: {{ iterations }}</p>
      <p>Color: {{ colorMap }}</p>
      <p>
        Variant:
        <select v-model.number="variety" @change="onVariantChange">
          <option v-for="v in variants" :key="v.value" :value="v.value">{{ v.label }}</option>
        </select>
      </p>
      <p>Drag: pan. Wheel: zoom. Click: recenter.</p>
      <p v-if="renderSource === 'hs'">
        HS Stage:
        <select v-model="hsSelectedArtifactId" @change="onHsImageChange">
          <option v-for="item in hsItems" :key="item.artifactId" :value="item.artifactId">{{ item.name }}</option>
        </select>
      </p>
      <div
        ref="mapCanvasRef"
        class="map-canvas"
        :class="{ fullscreen }"
        @mousedown="startDrag"
        @mousemove="onMove"
        @mouseup="stopDrag"
        @mouseleave="onMouseLeave"
        @wheel.prevent="onWheel"
        @click="onClick"
      >
        <button class="fullscreen-btn" @click.stop="toggleFullscreen">
          {{ fullscreen ? 'Exit Fullscreen' : 'Fullscreen' }}
        </button>
        <img v-if="imageUrl" :src="imageUrl" alt="fractal-map" class="map-image" />
        <div class="map-grid"></div>
        <div class="map-crosshair">+</div>
        <div class="mouse-overlay">Mouse: {{ mouseText }}</div>
        <div v-if="loading" class="map-overlay">rendering...</div>
        <div v-if="errorMessage" class="map-overlay error">{{ errorMessage }}</div>
      </div>
      <p>
        <a :href="`/special-points`">Open Special Points</a>
        |
        <a :href="`/artifacts?kind=image`">Open Images</a>
        |
        <a :href="`/stl-gallery`">Open 3D Models</a>
      </p>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { artifactContentUrl, getArtifacts, invokeModule, postMapRender, type ArtifactItem } from '../api'
import { t } from '../i18n'
import ParamEditor from '../components/ParamEditor.vue'
import FormulaPanel from '../components/FormulaPanel.vue'

const route = useRoute()
const router = useRouter()

const variants = [
  { value: 0, label: 'Classic Mandelbrot' },
  { value: 1, label: 'Conjugate Mandelbrot' },
  { value: 2, label: 'Burning Ship' },
  { value: 3, label: 'Duck Fold' },
  { value: 4, label: 'Bell Mirror' },
  { value: 5, label: 'Fish Tail' },
  { value: 6, label: 'Vase Mirror' },
  { value: 7, label: 'Bird Wing' },
  { value: 8, label: 'Mask Fold' },
  { value: 9, label: 'Ship Fold' },
]

const centerRe = ref(0)
const centerIm = ref(0)
const scale = ref(4)
const iterations = ref(1024)
const colorMap = ref('classic_cos')
const variety = ref(0)
const renderSource = ref<'map' | 'hs'>('map')
const hsItems = ref<ArtifactItem[]>([])
const hsSelectedArtifactId = ref('')

const dragging = ref(false)
const lastX = ref(0)
const lastY = ref(0)

const imageUrl = ref('')
const loading = ref(false)
const errorMessage = ref('')
const mapCanvasRef = ref<HTMLElement | null>(null)
const fullscreen = ref(false)
const effectiveWidth = ref(1600)
const effectiveHeight = ref(1000)
const mouseRe = ref<number | null>(null)
const mouseIm = ref<number | null>(null)

const mouseText = computed(() => {
  if (mouseRe.value == null || mouseIm.value == null) return '—'
  return `${mouseRe.value.toFixed(12)} + ${mouseIm.value.toFixed(12)}i`
})

let renderTimer: number | null = null
let requestSeq = 0

const syncQuery = () => {
  void router.replace({
    path: '/explorer-map',
    query: {
      centerRe: String(centerRe.value),
      centerIm: String(centerIm.value),
      scale: String(scale.value),
      iterations: String(iterations.value),
      colorMap: colorMap.value,
      variety: String(variety.value),
      source: renderSource.value,
      hsArtifactId: hsSelectedArtifactId.value,
    },
  })
}

const getRenderSize = () => {
  const target = mapCanvasRef.value
  if (!target) {
    return { width: 1600, height: 1000 }
  }
  const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 3))
  const width = Math.max(256, Math.min(2048, Math.round(target.clientWidth * dpr)))
  const height = Math.max(256, Math.min(2048, Math.round(target.clientHeight * dpr)))
  return { width, height }
}

const getImageRect = () => {
  const target = mapCanvasRef.value
  if (!target) return null
  const rect = target.getBoundingClientRect()
  const canvasAspect = rect.width / Math.max(1, rect.height)
  const imageAspect = effectiveWidth.value / Math.max(1, effectiveHeight.value)

  if (canvasAspect > imageAspect) {
    const displayHeight = rect.height
    const displayWidth = displayHeight * imageAspect
    const left = rect.left + (rect.width - displayWidth) / 2
    return { left, top: rect.top, width: displayWidth, height: displayHeight }
  }

  const displayWidth = rect.width
  const displayHeight = displayWidth / imageAspect
  const top = rect.top + (rect.height - displayHeight) / 2
  return { left: rect.left, top, width: displayWidth, height: displayHeight }
}

const pointerToComplex = (clientX: number, clientY: number) => {
  const imageRect = getImageRect()
  if (!imageRect) return null
  if (clientX < imageRect.left || clientX > imageRect.left + imageRect.width) return null
  if (clientY < imageRect.top || clientY > imageRect.top + imageRect.height) return null

  const nx = (clientX - imageRect.left) / imageRect.width
  const ny = (clientY - imageRect.top) / imageRect.height
  const aspect = effectiveWidth.value / Math.max(1, effectiveHeight.value)
  const spanRe = scale.value * aspect
  const spanIm = scale.value
  const re = centerRe.value + (nx - 0.5) * spanRe
  const im = centerIm.value - (ny - 0.5) * spanIm
  return { re, im, nx, ny }
}

const loadHsImages = async () => {
  if (hsItems.value.length > 0) return

  const reports = await getArtifacts({ kind: 'report' })
  const hsRunIds: string[] = []
  for (const item of reports.items) {
    if (item.name === 'hs_family_matrix.json' && !hsRunIds.includes(item.runId)) {
      hsRunIds.push(item.runId)
    }
  }

  for (let i = hsRunIds.length - 1; i >= 0; i -= 1) {
    const list = await getArtifacts({ kind: 'image', runId: hsRunIds[i] })
    if (list.items.length > 0) {
      hsItems.value = list.items
      hsSelectedArtifactId.value = hsItems.value[0].artifactId
      return
    }
  }

  errorMessage.value = 'No HS family images found. Run HS Families once, then switch back here.'
}

const renderMap = async () => {
  const seq = ++requestSeq
  loading.value = true
  errorMessage.value = ''
  try {
    if (renderSource.value === 'hs') {
      await loadHsImages()
      if (seq !== requestSeq) return
      imageUrl.value = hsSelectedArtifactId.value.length > 0 ? artifactContentUrl(hsSelectedArtifactId.value) : ''
      if (hsItems.value.length > 0) {
        effectiveWidth.value = 1600
        effectiveHeight.value = 1000
      }
      return
    }

    const { width, height } = getRenderSize()
    const res = await postMapRender({
      centerRe: centerRe.value,
      centerIm: centerIm.value,
      scale: scale.value,
      width,
      height,
      variety: variety.value,
      iterations: iterations.value,
      colorMap: colorMap.value,
    })
    if (seq !== requestSeq) return
    effectiveWidth.value = res.effective.width
    effectiveHeight.value = res.effective.height
    imageUrl.value = artifactContentUrl(res.artifactId)
  } catch (err) {
    if (seq !== requestSeq) return
    errorMessage.value = err instanceof Error ? err.message : 'render failed'
  } finally {
    if (seq === requestSeq) {
      loading.value = false
    }
  }
}

const requestRenderDebounced = () => {
  if (renderTimer != null) {
    window.clearTimeout(renderTimer)
  }
  renderTimer = window.setTimeout(() => {
    void renderMap()
  }, 180)
}

const onStyleChange = (payload: { iterations: number; colorMap: string }) => {
  iterations.value = payload.iterations
  colorMap.value = payload.colorMap
  syncQuery()
  requestRenderDebounced()
}

const onApplyViewport = (payload: { centerRe: number; centerIm: number; scale: number }) => {
  centerRe.value = payload.centerRe
  centerIm.value = payload.centerIm
  scale.value = payload.scale
  syncQuery()
  void renderMap()
}

const onVariantChange = () => {
  centerRe.value = 0
  centerIm.value = 0
  scale.value = 4
  syncQuery()
  void renderMap()
}

const onRenderSourceChange = () => {
  syncQuery()
  void renderMap()
}

const onHsImageChange = () => {
  imageUrl.value = hsSelectedArtifactId.value.length > 0 ? artifactContentUrl(hsSelectedArtifactId.value) : ''
}

const startDrag = (e: MouseEvent) => {
  dragging.value = true
  lastX.value = e.clientX
  lastY.value = e.clientY
}

const stopDrag = () => {
  if (dragging.value) {
    dragging.value = false
    syncQuery()
    void renderMap()
  }
}

const onDrag = (e: MouseEvent) => {
  if (renderSource.value !== 'map') return
  if (!dragging.value) return
  const imageRect = getImageRect()
  if (!imageRect) return
  const dx = e.clientX - lastX.value
  const dy = e.clientY - lastY.value
  const aspect = effectiveWidth.value / Math.max(1, effectiveHeight.value)
  centerRe.value -= (dx / imageRect.width) * scale.value * aspect
  centerIm.value += (dy / imageRect.height) * scale.value
  lastX.value = e.clientX
  lastY.value = e.clientY
  requestRenderDebounced()
}

const onMove = (e: MouseEvent) => {
  const c = pointerToComplex(e.clientX, e.clientY)
  if (!c) {
    mouseRe.value = null
    mouseIm.value = null
  } else {
    mouseRe.value = c.re
    mouseIm.value = c.im
  }
  onDrag(e)
}

const onMouseLeave = () => {
  mouseRe.value = null
  mouseIm.value = null
  stopDrag()
}

const onWheel = (e: WheelEvent) => {
  if (renderSource.value !== 'map') return
  if (e.deltaY < 0) {
    scale.value *= 0.9
  } else {
    scale.value *= 1.1
  }
  syncQuery()
  requestRenderDebounced()
}

const onClick = (e: MouseEvent) => {
  if (renderSource.value !== 'map') return
  const c = pointerToComplex(e.clientX, e.clientY)
  if (!c) return
  centerRe.value = c.re
  centerIm.value = c.im
  syncQuery()
  void renderMap()
}

const onFullscreenChange = () => {
  fullscreen.value = document.fullscreenElement === mapCanvasRef.value
  requestRenderDebounced()
}

const toggleFullscreen = async () => {
  try {
    if (!mapCanvasRef.value) return
    if (document.fullscreenElement === mapCanvasRef.value) {
      await document.exitFullscreen()
    } else {
      await mapCanvasRef.value.requestFullscreen()
    }
  } catch {
    // ignore
  }
}

onMounted(async () => {
  document.addEventListener('fullscreenchange', onFullscreenChange)
  const q = route.query
  if (typeof q.centerRe === 'string') centerRe.value = Number(q.centerRe)
  if (typeof q.centerIm === 'string') centerIm.value = Number(q.centerIm)
  if (typeof q.scale === 'string') scale.value = Number(q.scale)
  if (typeof q.iterations === 'string') iterations.value = Number(q.iterations)
  if (typeof q.colorMap === 'string' && q.colorMap.length > 0) colorMap.value = q.colorMap
  if (typeof q.variety === 'string') {
    const parsed = Number(q.variety)
    variety.value = Number.isFinite(parsed) ? parsed : 0
  }
  if (q.source === 'hs' || q.source === 'map') {
    renderSource.value = q.source
  }
  if (typeof q.hsArtifactId === 'string') {
    hsSelectedArtifactId.value = q.hsArtifactId
  }
  await renderMap()
})

watch(
  () => route.query,
  async (q) => {
    if (typeof q.centerRe === 'string') centerRe.value = Number(q.centerRe)
    if (typeof q.centerIm === 'string') centerIm.value = Number(q.centerIm)
    if (typeof q.scale === 'string') scale.value = Number(q.scale)
    if (typeof q.iterations === 'string') iterations.value = Number(q.iterations)
    if (typeof q.colorMap === 'string' && q.colorMap.length > 0) colorMap.value = q.colorMap
    if (typeof q.variety === 'string') {
      const parsed = Number(q.variety)
      variety.value = Number.isFinite(parsed) ? parsed : 0
    }
    if (q.source === 'hs' || q.source === 'map') {
      renderSource.value = q.source
    }
    if (typeof q.hsArtifactId === 'string') {
      hsSelectedArtifactId.value = q.hsArtifactId
      imageUrl.value = hsSelectedArtifactId.value.length > 0 ? artifactContentUrl(hsSelectedArtifactId.value) : imageUrl.value
    }
  },
)

onBeforeUnmount(() => {
  document.removeEventListener('fullscreenchange', onFullscreenChange)
})
</script>

<style scoped>
.map-canvas {
  position: relative;
  height: 560px;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid #3a4a5a;
  background: #070b11;
  cursor: crosshair;
}
.map-canvas.fullscreen {
  height: 100vh;
  border-radius: 0;
  border: 0;
}
.fullscreen-btn {
  position: absolute;
  left: 8px;
  top: 8px;
  z-index: 4;
  border: 1px solid #4d6075;
  background: rgba(12, 18, 26, 0.8);
  color: #d9e5f2;
  border-radius: 6px;
  padding: 4px 8px;
  font-size: 12px;
}
.map-image {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  image-rendering: auto;
}
.map-grid {
  position: absolute;
  inset: 0;
  background-image: linear-gradient(rgba(255, 255, 255, 0.06) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.06) 1px, transparent 1px);
  background-size: 32px 32px;
}
.map-crosshair {
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  color: #fff;
  font-size: 20px;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.8);
}
.map-overlay {
  position: absolute;
  right: 8px;
  top: 8px;
  color: #fff;
  background: rgba(0, 0, 0, 0.6);
  border-radius: 6px;
  padding: 4px 8px;
  font-size: 12px;
}
.map-overlay.error {
  background: rgba(120, 0, 0, 0.7);
}
.mouse-overlay {
  position: absolute;
  left: 8px;
  bottom: 8px;
  color: #fff;
  background: rgba(0, 0, 0, 0.6);
  border-radius: 6px;
  padding: 4px 8px;
  font-size: 12px;
  z-index: 3;
}
</style>
