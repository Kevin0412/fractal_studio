<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref, watch } from 'vue'
import { api, type MapRenderRequest, type Metric, type ColorMap } from '../api'

// MapCanvas renders the fractal by requesting a full frame from the backend at
// the exact canvas pixel dimensions. Every param/pan/zoom change triggers a
// debounced full re-render; the previous frame remains visible while loading.

const props = defineProps<{
  centerRe: number
  centerIm: number
  scale: number
  iterations: number
  variant: string   // Variant literal or "custom:HASH"
  metric: Metric
  colorMap: ColorMap
  smooth?: boolean
  transitionTheta: number | null
  transitionThetaMilliDeg?: number | null
  transitionFrom?: string
  transitionTo?: string
  julia?: boolean
  juliaRe?: number
  juliaIm?: number
  engine?: string
  scalarType?: string
}>()

const emit = defineEmits<{
  (e: 'viewport-change', v: { centerRe: number; centerIm: number; scale: number }): void
  (e: 'rendered', meta: { generatedMs: number; artifactId: string; engineUsed?: string; scalarUsed?: string }): void
  (e: 'click-world', pos: { re: number; im: number }): void
}>()

// ── DOM refs ──────────────────────────────────────────────────────────────────
const wrapper  = ref<HTMLDivElement | null>(null)
const canvasEl = ref<HTMLCanvasElement | null>(null)
const pending  = ref(false)
const error    = ref('')
const domW     = ref(0)
const domH     = ref(0)
let   ro: ResizeObserver | null = null
let   renderTimer: ReturnType<typeof setTimeout> | null = null
let   currentRender: AbortController | null = null
let   renderSeq = 0

// ── Full-frame render ─────────────────────────────────────────────────────────

function clearCanvas() {
  const ctx = canvasEl.value?.getContext('2d')
  if (!ctx) return
  ctx.clearRect(0, 0, domW.value, domH.value)
}

function invalidateCurrentRender() {
  currentRender?.abort()
  currentRender = null
  renderSeq += 1
}

async function renderFrame() {
  if (domW.value < 16 || domH.value < 16) return
  pending.value = true
  error.value   = ''

  // Abort any in-flight render
  currentRender?.abort()
  const controller = new AbortController()
  currentRender = controller
  const seq = ++renderSeq
  const requestId = `${Date.now()}-${seq}`

  const req: MapRenderRequest = {
    requestId,
    centerRe:   props.centerRe,
    centerIm:   props.centerIm,
    scale:      props.scale,
    width:      domW.value,
    height:     domH.value,
    iterations: props.iterations,
    variant:    props.variant,
    metric:     props.metric,
    colorMap:   props.colorMap,
    smooth:     props.smooth,
    julia:      props.julia,
    juliaRe:    props.juliaRe ?? 0,
    juliaIm:    props.juliaIm ?? 0,
  }
  if (props.transitionThetaMilliDeg !== null && props.transitionThetaMilliDeg !== undefined) {
    req.transitionThetaMilliDeg = props.transitionThetaMilliDeg
    req.transitionTheta = props.transitionThetaMilliDeg * Math.PI / 180000
  } else if (props.transitionTheta !== null) {
    req.transitionTheta = props.transitionTheta
  }
  if (props.transitionFrom)           (req as any).transitionFrom  = props.transitionFrom
  if (props.transitionTo)             (req as any).transitionTo    = props.transitionTo
  if (props.engine)                   (req as any).engine           = props.engine
  if (props.scalarType)               (req as any).scalarType       = props.scalarType

  try {
    const resp = await api.mapRender(req, controller.signal) as any
    if (seq !== renderSeq || (resp.requestId && resp.requestId !== requestId)) return
    const imgUrl = api.artifactContentUrl(resp.artifactId)
    await new Promise<void>((res, rej) => {
      const img = new Image()
      img.onload = () => {
        if (seq !== renderSeq) { res(); return }
        const ctx = canvasEl.value?.getContext('2d')
        if (ctx) ctx.drawImage(img, 0, 0, domW.value, domH.value)
        res()
      }
      img.onerror = rej
      img.src = imgUrl
    })
    if (seq !== renderSeq) return
    emit('rendered', {
      generatedMs: resp.generatedMs,
      artifactId:  resp.artifactId,
      engineUsed:  resp.effective?.engine ?? resp.engineUsed,
      scalarUsed:  resp.effective?.scalar ?? resp.scalarUsed,
    })
  } catch (e: any) {
    if (seq === renderSeq && e?.name !== 'AbortError') error.value = e?.message ?? String(e)
  } finally {
    if (currentRender === controller) currentRender = null
    if (seq === renderSeq) pending.value = false
  }
}

function scheduleRender(delay = 200) {
  invalidateCurrentRender()
  if (domW.value >= 16 && domH.value >= 16) {
    pending.value = true
    error.value = ''
    clearCanvas()
  }
  if (renderTimer) clearTimeout(renderTimer)
  renderTimer = setTimeout(renderFrame, delay)
}

// ── Watchers ──────────────────────────────────────────────────────────────────

watch(() => [
  props.centerRe, props.centerIm, props.scale,
  props.variant, props.metric, props.colorMap, props.smooth,
  props.iterations, props.julia, props.juliaRe, props.juliaIm,
  props.engine, props.scalarType, props.transitionTheta, props.transitionThetaMilliDeg,
  props.transitionFrom, props.transitionTo,
  domW.value, domH.value,
], () => scheduleRender())

// ── Lifecycle ─────────────────────────────────────────────────────────────────

onMounted(() => {
  if (!wrapper.value || !canvasEl.value) return
  ro = new ResizeObserver(entries => {
    for (const e of entries) {
      const w = Math.round(e.contentRect.width)
      const h = Math.round(e.contentRect.height)
      if (w !== domW.value || h !== domH.value) {
        domW.value = w
        domH.value = h
        if (canvasEl.value) { canvasEl.value.width = w; canvasEl.value.height = h }
      }
    }
  })
  ro.observe(wrapper.value)
  const w = Math.round(wrapper.value.clientWidth)
  const h = Math.round(wrapper.value.clientHeight)
  domW.value = w
  domH.value = h
  if (canvasEl.value) { canvasEl.value.width = w; canvasEl.value.height = h }
  scheduleRender(0)
})

onBeforeUnmount(() => {
  ro?.disconnect()
  if (renderTimer) clearTimeout(renderTimer)
  invalidateCurrentRender()
})

// ── Interaction ───────────────────────────────────────────────────────────────

function onWheel(e: WheelEvent) {
  if (!wrapper.value) return
  const rect    = wrapper.value.getBoundingClientRect()
  const px      = (e.clientX - rect.left) / rect.width
  const py      = (e.clientY - rect.top)  / rect.height
  const aspect  = rect.width / rect.height
  const wx      = props.centerRe + (px - 0.5) * props.scale * aspect
  const wy      = props.centerIm + (0.5 - py) * props.scale
  const factor  = e.deltaY > 0 ? 1.25 : 0.8
  const newScale = props.scale * factor
  emit('viewport-change', {
    centerRe: wx - (px - 0.5) * newScale * aspect,
    centerIm: wy + (py - 0.5) * newScale,
    scale:    newScale,
  })
}

let dragging  = false
let dragMoved = false
let dragStart = { x: 0, y: 0, cx: 0, cy: 0, sc: 0 }

function screenToWorld(e: MouseEvent): { re: number; im: number } | null {
  if (!wrapper.value) return null
  const rect   = wrapper.value.getBoundingClientRect()
  const aspect = rect.width / rect.height
  const px     = (e.clientX - rect.left) / rect.width
  const py     = (e.clientY - rect.top)  / rect.height
  return {
    re: props.centerRe + (px - 0.5) * props.scale * aspect,
    im: props.centerIm - (py - 0.5) * props.scale,
  }
}

function onMouseDown(e: MouseEvent) {
  dragging  = true
  dragMoved = false
  dragStart = { x: e.clientX, y: e.clientY, cx: props.centerRe, cy: props.centerIm, sc: props.scale }
}

function onMouseMove(e: MouseEvent) {
  if (!dragging || !wrapper.value) return
  const dx = e.clientX - dragStart.x
  const dy = e.clientY - dragStart.y
  if (!dragMoved && Math.hypot(dx, dy) < 5) return
  dragMoved = true
  const rect   = wrapper.value.getBoundingClientRect()
  const aspect = rect.width / rect.height
  emit('viewport-change', {
    centerRe: dragStart.cx - (dx / rect.width)  * dragStart.sc * aspect,
    centerIm: dragStart.cy + (dy / rect.height) * dragStart.sc,
    scale:    dragStart.sc,
  })
}

function onMouseUp(e: MouseEvent) {
  if (dragging && !dragMoved) {
    const w = screenToWorld(e)
    if (w) emit('click-world', w)
  }
  dragging = false
}
</script>

<template>
  <div class="map-wrap"
       ref="wrapper"
       @wheel.prevent="onWheel"
       @mousedown="onMouseDown"
       @mousemove="onMouseMove"
       @mouseup="onMouseUp"
       @mouseleave="onMouseUp">
    <canvas ref="canvasEl" />
    <div v-if="pending" class="overlay">rendering…</div>
    <div v-if="error"   class="overlay error">{{ error }}</div>
  </div>
</template>

<style scoped>
.map-wrap {
  position: relative;
  width: 100%;
  height: 100%;
  background: var(--bg, #0a0b0d);
  cursor: grab;
  user-select: none;
  overflow: hidden;
}

.map-wrap:active { cursor: grabbing; }

canvas {
  display: block;
  width: 100%;
  height: 100%;
  image-rendering: pixelated;
}

.overlay {
  position: absolute;
  left: 50%; top: 8px;
  transform: translateX(-50%);
  background: var(--panel);
  border: 1px solid var(--rule);
  padding: 4px 10px;
  font-family: var(--mono);
  font-size: var(--fs-label);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-dim);
  pointer-events: none;
}

.overlay.error {
  color: var(--bad);
  border-color: var(--bad);
}
</style>
