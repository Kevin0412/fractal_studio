<template>
  <div>
    <section class="view-card controls">
      <h3>{{ t('explorerMap') }}: Mandelbrot ↔ Julia</h3>
      <label>
        Variant
        <select v-model.number="variety" @change="onVariantChange">
          <option v-for="v in variants" :key="v.value" :value="v.value">{{ v.label }}</option>
        </select>
      </label>
      <label>
        Iterations
        <input v-model.number="iterations" type="number" min="1" @change="onStyleChange" />
      </label>
      <label>
        Color Mapping
        <select v-model="colorMap" @change="onStyleChange">
          <option value="classic_cos">classic_cos</option>
          <option value="mod17">mod17</option>
          <option value="hsv_wheel">hsv_wheel</option>
          <option value="tri765">tri765</option>
          <option value="grayscale">grayscale</option>
        </select>
      </label>
      <label>
        Mode
        <select v-model="viewMode" @change="onModeChange">
          <option value="explore">Explore Fractal</option>
          <option value="fractal-julia">Fractal + Julia</option>
        </select>
      </label>
      <p v-if="viewMode === 'fractal-julia'">Selected Julia c: {{ juliaRe.toFixed(12) }} + {{ juliaIm.toFixed(12) }}i</p>
      <p v-if="viewMode === 'fractal-julia'">Left click picks Julia c and recenters left map. Drag/Wheel works on both panes.</p>
      <p v-else>Left click recenters. Drag/Wheel to explore.</p>
      <p class="export-row">
        <button type="button" @click="exportPanePng('left')">Export Left PNG</button>
        <button v-if="viewMode === 'fractal-julia'" type="button" @click="exportPanePng('right')">Export Right PNG</button>
      </p>
    </section>

    <section class="dual-grid" :class="{ single: viewMode === 'explore' }">
      <article class="view-card pane-card">
        <h4>Left: Mandelbrot / Variant</h4>
        <p>Center: {{ left.centerRe }} + {{ left.centerIm }}i</p>
        <p>Scale: {{ left.scale }}</p>
        <div
          ref="leftRef"
          class="map-canvas"
          @mousedown="startDrag('left', $event)"
          @mousemove="onMove('left', $event)"
          @mouseup="stopDrag('left')"
          @mouseleave="onLeave('left')"
          @wheel.prevent="onWheel('left', $event)"
          @click="onLeftClick"
        >
          <img v-if="left.imageUrl" :src="left.imageUrl" alt="left-map" class="map-image" />
          <div class="map-grid"></div>
          <div class="map-crosshair">+</div>
          <div class="mouse-overlay">Mouse: {{ leftMouseText }}</div>
          <div v-if="left.loading" class="map-overlay">rendering...</div>
          <div v-if="left.errorMessage" class="map-overlay error">{{ left.errorMessage }}</div>
        </div>
      </article>

      <article class="view-card pane-card" v-if="viewMode === 'fractal-julia'">
        <h4>Right: Julia</h4>
        <p>Julia c: {{ juliaRe.toFixed(12) }} + {{ juliaIm.toFixed(12) }}i</p>
        <p>Center: {{ right.centerRe }} + {{ right.centerIm }}i</p>
        <p>Scale: {{ right.scale }}</p>
        <div
          ref="rightRef"
          class="map-canvas"
          @mousedown="startDrag('right', $event)"
          @mousemove="onMove('right', $event)"
          @mouseup="stopDrag('right')"
          @mouseleave="onLeave('right')"
          @wheel.prevent="onWheel('right', $event)"
          @click="onRightClick"
        >
          <img v-if="right.imageUrl" :src="right.imageUrl" alt="right-map" class="map-image" />
          <div class="map-grid"></div>
          <div class="map-crosshair">+</div>
          <div class="mouse-overlay">Mouse: {{ rightMouseText }}</div>
          <div v-if="right.loading" class="map-overlay">rendering...</div>
          <div v-if="right.errorMessage" class="map-overlay error">{{ right.errorMessage }}</div>
        </div>
      </article>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { artifactContentUrl, postMapRender } from '../api'
import { t } from '../i18n'

type PaneKey = 'left' | 'right'
type PaneState = {
  centerRe: number
  centerIm: number
  scale: number
  imageUrl: string
  loading: boolean
  errorMessage: string
  effectiveWidth: number
  effectiveHeight: number
  dragging: boolean
  lastX: number
  lastY: number
  mouseRe: number | null
  mouseIm: number | null
  requestSeq: number
  timer: number | null
}

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

const leftRef = ref<HTMLElement | null>(null)
const rightRef = ref<HTMLElement | null>(null)

const makePane = (scale = 4): PaneState => ({
  centerRe: 0,
  centerIm: 0,
  scale,
  imageUrl: '',
  loading: false,
  errorMessage: '',
  effectiveWidth: 1600,
  effectiveHeight: 1000,
  dragging: false,
  lastX: 0,
  lastY: 0,
  mouseRe: null,
  mouseIm: null,
  requestSeq: 0,
  timer: null,
})

const left = reactive(makePane(4))
const right = reactive(makePane(4))

const iterations = ref(1024)
const colorMap = ref('classic_cos')
const variety = ref(0)
const juliaRe = ref(0)
const juliaIm = ref(0)
const viewMode = ref<'explore' | 'fractal-julia'>('explore')

const leftMouseText = computed(() => (left.mouseRe == null || left.mouseIm == null ? '—' : `${left.mouseRe.toFixed(12)} + ${left.mouseIm.toFixed(12)}i`))
const rightMouseText = computed(() => (right.mouseRe == null || right.mouseIm == null ? '—' : `${right.mouseRe.toFixed(12)} + ${right.mouseIm.toFixed(12)}i`))

const getRef = (pane: PaneKey) => (pane === 'left' ? leftRef.value : rightRef.value)
const getPane = (pane: PaneKey) => (pane === 'left' ? left : right)

const syncQuery = () => {
  void router.replace({
    path: '/explorer-map',
    query: {
      variety: String(variety.value),
      iterations: String(iterations.value),
      colorMap: colorMap.value,
      juliaRe: String(juliaRe.value),
      juliaIm: String(juliaIm.value),
      leftCenterRe: String(left.centerRe),
      leftCenterIm: String(left.centerIm),
      leftScale: String(left.scale),
      rightCenterRe: String(right.centerRe),
      rightCenterIm: String(right.centerIm),
      rightScale: String(right.scale),
      mode: viewMode.value,
    },
  })
}

const getRenderSize = (pane: PaneKey) => {
  const target = getRef(pane)
  if (!target) return { width: 1600, height: 1000 }
  const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 3))
  const width = Math.max(256, Math.min(2048, Math.round(target.clientWidth * dpr)))
  const height = Math.max(256, Math.min(2048, Math.round(target.clientHeight * dpr)))
  return { width, height }
}

const getImageRect = (pane: PaneKey) => {
  const p = getPane(pane)
  const target = getRef(pane)
  if (!target) return null
  const rect = target.getBoundingClientRect()
  const canvasAspect = rect.width / Math.max(1, rect.height)
  const imageAspect = p.effectiveWidth / Math.max(1, p.effectiveHeight)
  if (canvasAspect > imageAspect) {
    const displayHeight = rect.height
    const displayWidth = displayHeight * imageAspect
    const leftPos = rect.left + (rect.width - displayWidth) / 2
    return { left: leftPos, top: rect.top, width: displayWidth, height: displayHeight }
  }
  const displayWidth = rect.width
  const displayHeight = displayWidth / imageAspect
  const topPos = rect.top + (rect.height - displayHeight) / 2
  return { left: rect.left, top: topPos, width: displayWidth, height: displayHeight }
}

const pointerToComplex = (pane: PaneKey, clientX: number, clientY: number) => {
  const p = getPane(pane)
  const imageRect = getImageRect(pane)
  if (!imageRect) return null
  if (clientX < imageRect.left || clientX > imageRect.left + imageRect.width) return null
  if (clientY < imageRect.top || clientY > imageRect.top + imageRect.height) return null
  const nx = (clientX - imageRect.left) / imageRect.width
  const ny = (clientY - imageRect.top) / imageRect.height
  const aspect = p.effectiveWidth / Math.max(1, p.effectiveHeight)
  const spanRe = p.scale * aspect
  const spanIm = p.scale
  const re = p.centerRe + (nx - 0.5) * spanRe
  const im = p.centerIm - (ny - 0.5) * spanIm
  return { re, im }
}

const renderPane = async (pane: PaneKey) => {
  const p = getPane(pane)
  const seq = ++p.requestSeq
  p.loading = true
  p.errorMessage = ''
  try {
    const { width, height } = getRenderSize(pane)
    const res = await postMapRender({
      centerRe: p.centerRe,
      centerIm: p.centerIm,
      scale: p.scale,
      width,
      height,
      variety: variety.value,
      iterations: iterations.value,
      colorMap: colorMap.value,
      mode: pane === 'left' ? 'mandelbrot' : 'julia',
      juliaRe: juliaRe.value,
      juliaIm: juliaIm.value,
    })
    if (seq !== p.requestSeq) return
    p.effectiveWidth = res.effective.width
    p.effectiveHeight = res.effective.height
    p.imageUrl = artifactContentUrl(res.artifactId)
  } catch (err) {
    if (seq !== p.requestSeq) return
    p.errorMessage = err instanceof Error ? err.message : 'render failed'
  } finally {
    if (seq === p.requestSeq) p.loading = false
  }
}

const requestRenderDebounced = (pane: PaneKey) => {
  const p = getPane(pane)
  if (p.timer != null) window.clearTimeout(p.timer)
  p.timer = window.setTimeout(() => {
    void renderPane(pane)
  }, 150)
}

const onStyleChange = () => {
  syncQuery()
  requestRenderDebounced('left')
  if (viewMode.value === 'fractal-julia') {
    requestRenderDebounced('right')
  }
}

const onVariantChange = () => {
  left.centerRe = 0
  left.centerIm = 0
  left.scale = 4
  syncQuery()
  void renderPane('left')
  if (viewMode.value === 'fractal-julia') {
    void renderPane('right')
  }
}

const onModeChange = () => {
  syncQuery()
  void renderPane('left')
  if (viewMode.value === 'fractal-julia') {
    void renderPane('right')
  }
}

const startDrag = (pane: PaneKey, e: MouseEvent) => {
  const p = getPane(pane)
  p.dragging = true
  p.lastX = e.clientX
  p.lastY = e.clientY
}

const stopDrag = (pane: PaneKey) => {
  const p = getPane(pane)
  if (!p.dragging) return
  p.dragging = false
  syncQuery()
  void renderPane(pane)
}

const onMove = (pane: PaneKey, e: MouseEvent) => {
  const p = getPane(pane)
  const c = pointerToComplex(pane, e.clientX, e.clientY)
  if (c) {
    p.mouseRe = c.re
    p.mouseIm = c.im
  } else {
    p.mouseRe = null
    p.mouseIm = null
  }

  if (!p.dragging) return
  const imageRect = getImageRect(pane)
  if (!imageRect) return
  const dx = e.clientX - p.lastX
  const dy = e.clientY - p.lastY
  const aspect = p.effectiveWidth / Math.max(1, p.effectiveHeight)
  p.centerRe -= (dx / imageRect.width) * p.scale * aspect
  p.centerIm += (dy / imageRect.height) * p.scale
  p.lastX = e.clientX
  p.lastY = e.clientY
  requestRenderDebounced(pane)
}

const onLeave = (pane: PaneKey) => {
  const p = getPane(pane)
  p.mouseRe = null
  p.mouseIm = null
  stopDrag(pane)
}

const onWheel = (pane: PaneKey, e: WheelEvent) => {
  const p = getPane(pane)
  if (e.deltaY < 0) {
    p.scale *= 0.9
  } else {
    p.scale *= 1.1
  }
  syncQuery()
  requestRenderDebounced(pane)
}

const onLeftClick = (e: MouseEvent) => {
  const c = pointerToComplex('left', e.clientX, e.clientY)
  if (!c) return

  left.centerRe = c.re
  left.centerIm = c.im

  if (viewMode.value === 'fractal-julia') {
    juliaRe.value = c.re
    juliaIm.value = c.im
    syncQuery()
    void renderPane('left')
    void renderPane('right')
    return
  }

  syncQuery()
  void renderPane('left')
}

const onRightClick = (e: MouseEvent) => {
  const c = pointerToComplex('right', e.clientX, e.clientY)
  if (!c) return
  right.centerRe = c.re
  right.centerIm = c.im
  syncQuery()
  void renderPane('right')
}

const exportPanePng = (pane: PaneKey) => {
  const p = getPane(pane)
  if (!p.imageUrl) return
  const a = document.createElement('a')
  const stamp = Date.now()
  a.href = p.imageUrl
  a.download = pane === 'left' ? `fractal-left-${stamp}.png` : `julia-right-${stamp}.png`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
}

onMounted(async () => {
  const q = route.query
  if (typeof q.variety === 'string') variety.value = Number(q.variety)
  if (typeof q.iterations === 'string') iterations.value = Number(q.iterations)
  if (typeof q.colorMap === 'string' && q.colorMap.length > 0) colorMap.value = q.colorMap
  if (typeof q.juliaRe === 'string') juliaRe.value = Number(q.juliaRe)
  if (typeof q.juliaIm === 'string') juliaIm.value = Number(q.juliaIm)
  if (typeof q.leftCenterRe === 'string') left.centerRe = Number(q.leftCenterRe)
  if (typeof q.leftCenterIm === 'string') left.centerIm = Number(q.leftCenterIm)
  if (typeof q.leftScale === 'string') left.scale = Number(q.leftScale)
  if (typeof q.rightCenterRe === 'string') right.centerRe = Number(q.rightCenterRe)
  if (typeof q.rightCenterIm === 'string') right.centerIm = Number(q.rightCenterIm)
  if (typeof q.rightScale === 'string') right.scale = Number(q.rightScale)
  if (q.mode === 'explore' || q.mode === 'fractal-julia') {
    viewMode.value = q.mode
  }

  await renderPane('left')
  if (viewMode.value === 'fractal-julia') {
    await renderPane('right')
  }
})
</script>

<style scoped>
.controls {
  margin-bottom: 12px;
}
label {
  display: block;
  margin-bottom: 8px;
}
.export-row {
  display: flex;
  gap: 8px;
  margin-top: 10px;
}
button {
  background: #2f6f92;
  color: #fff;
  border: none;
  border-radius: 8px;
  padding: 8px 12px;
  cursor: pointer;
}
input,
select {
  width: 100%;
  margin-top: 4px;
}
.dual-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}
.dual-grid.single {
  grid-template-columns: 1fr;
}
.pane-card {
  min-width: 0;
}
.map-canvas {
  position: relative;
  height: 540px;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid #3a4a5a;
  background: #070b11;
  cursor: crosshair;
}
.map-image {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
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
  font-size: 18px;
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
@media (max-width: 1100px) {
  .dual-grid {
    grid-template-columns: 1fr;
  }
}
</style>
