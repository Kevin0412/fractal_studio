<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref, watch } from 'vue'
import { api, type MapRenderRequest, type Variant, type Metric, type ColorMap } from '../api'

// MapCanvas renders at its actual DOM pixel size — no fixed width/height props.
// A ResizeObserver tracks the container and triggers a re-render on resize.

const props = defineProps<{
  centerRe: number
  centerIm: number
  scale: number
  iterations: number
  variant: Variant
  metric: Metric
  colorMap: ColorMap
  smooth?: boolean
  transitionTheta: number | null
  engine?: string
  scalarType?: string
}>()

const emit = defineEmits<{
  (e: 'viewport-change', v: { centerRe: number; centerIm: number; scale: number }): void
  (e: 'rendered', meta: { generatedMs: number; artifactId: string; engineUsed?: string; scalarUsed?: string }): void
}>()

const imgUrl  = ref<string>('')
const pending = ref<boolean>(false)
const error   = ref<string>('')
const wrapper = ref<HTMLDivElement | null>(null)
const img     = ref<HTMLImageElement | null>(null)

// Actual DOM size, updated by ResizeObserver
const domW = ref(0)
const domH = ref(0)

let renderTimer: any = null
let ro: ResizeObserver | null = null

async function triggerRender() {
  const W = domW.value
  const H = domH.value
  if (W < 16 || H < 16) return   // not yet laid out
  pending.value = true
  error.value = ''
  const req: MapRenderRequest = {
    centerRe:   props.centerRe,
    centerIm:   props.centerIm,
    scale:      props.scale,
    width:      W,
    height:     H,
    iterations: props.iterations,
    variant:    props.variant,
    metric:     props.metric,
    colorMap:   props.colorMap,
  }
  if (props.transitionTheta !== null) req.transitionTheta = props.transitionTheta
  if (props.smooth)      req.smooth      = props.smooth
  if (props.engine)     (req as any).engine     = props.engine
  if (props.scalarType) (req as any).scalarType = props.scalarType
  try {
    const resp = await api.mapRender(req) as any
    imgUrl.value = api.artifactContentUrl(resp.artifactId)
    emit('rendered', {
      generatedMs: resp.generatedMs,
      artifactId:  resp.artifactId,
      engineUsed:  resp.engineUsed,
      scalarUsed:  resp.scalarUsed,
    })
  } catch (e: any) {
    error.value = e?.message || String(e)
  } finally {
    pending.value = false
  }
}

function requestRender() {
  if (renderTimer) clearTimeout(renderTimer)
  renderTimer = setTimeout(triggerRender, 120)
}

watch(() => [
  props.centerRe, props.centerIm, props.scale,
  props.iterations, props.variant, props.metric, props.colorMap, props.smooth,
  props.transitionTheta, props.engine, props.scalarType,
  domW.value, domH.value,
], requestRender)

onMounted(() => {
  if (!wrapper.value) return
  ro = new ResizeObserver(entries => {
    for (const e of entries) {
      const w = Math.round(e.contentRect.width)
      const h = Math.round(e.contentRect.height)
      if (w !== domW.value || h !== domH.value) {
        domW.value = w
        domH.value = h
      }
    }
  })
  ro.observe(wrapper.value)
  // Seed from current size immediately
  domW.value = Math.round(wrapper.value.clientWidth)
  domH.value = Math.round(wrapper.value.clientHeight)
  triggerRender()
})

onBeforeUnmount(() => {
  ro?.disconnect()
  if (renderTimer) clearTimeout(renderTimer)
})

// ---- interaction ----

function onWheel(e: WheelEvent) {
  e.preventDefault()
  if (!wrapper.value) return
  const rect = wrapper.value.getBoundingClientRect()
  const px = (e.clientX - rect.left) / rect.width
  const py = (e.clientY - rect.top)  / rect.height
  // world position of cursor before zoom
  const aspect = rect.width / rect.height
  const spanIm = props.scale
  const spanRe = props.scale * aspect
  const wx = props.centerRe + (px - 0.5) * spanRe
  const wy = props.centerIm + (0.5 - py) * spanIm

  const factor = e.deltaY > 0 ? 1.25 : 0.8
  const newScale = props.scale * factor
  // keep cursor anchored to the same world point
  const newSpanIm = newScale
  const newSpanRe = newScale * aspect
  const newCenterRe = wx - (px - 0.5) * newSpanRe
  const newCenterIm = wy + (py - 0.5) * newSpanIm

  emit('viewport-change', { centerRe: newCenterRe, centerIm: newCenterIm, scale: newScale })
}

let dragging = false
let dragStart = { x: 0, y: 0, cx: 0, cy: 0, sc: 0 }

function onMouseDown(e: MouseEvent) {
  dragging = true
  dragStart = {
    x: e.clientX, y: e.clientY,
    cx: props.centerRe, cy: props.centerIm, sc: props.scale,
  }
}

function onMouseMove(e: MouseEvent) {
  if (!dragging || !wrapper.value) return
  const rect = wrapper.value.getBoundingClientRect()
  const aspect = rect.width / rect.height
  const dx = (e.clientX - dragStart.x) / rect.width
  const dy = (e.clientY - dragStart.y) / rect.height
  const newCenterRe = dragStart.cx - dx * dragStart.sc * aspect
  const newCenterIm = dragStart.cy + dy * dragStart.sc
  emit('viewport-change', { centerRe: newCenterRe, centerIm: newCenterIm, scale: dragStart.sc })
}

function onMouseUp() { dragging = false }
</script>

<template>
  <div class="map-wrap"
       ref="wrapper"
       @wheel="onWheel"
       @mousedown="onMouseDown"
       @mousemove="onMouseMove"
       @mouseup="onMouseUp"
       @mouseleave="onMouseUp">
    <img v-if="imgUrl" ref="img" :src="imgUrl" :class="{ stale: pending }" draggable="false" />
    <div v-if="pending" class="overlay">rendering…</div>
    <div v-if="error" class="overlay error">{{ error }}</div>
  </div>
</template>

<style scoped>
.map-wrap {
  position: relative;
  width: 100%;
  height: 100%;
  background: #000;
  cursor: grab;
  user-select: none;
  overflow: hidden;
}

.map-wrap:active { cursor: grabbing; }

img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
  image-rendering: pixelated;
  transition: opacity 0.15s;
}

img.stale { opacity: 0.65; }

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
}

.overlay.error {
  color: var(--bad);
  border-color: var(--bad);
}
</style>
