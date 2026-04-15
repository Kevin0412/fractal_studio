<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref, watch } from 'vue'
import { api, type MapRenderRequest, type Metric, type ColorMap } from '../api'

// MapCanvas renders the fractal using one of two strategies:
//   • Normal mode (no transitionTheta): tile-based LRU cache with JS colorization.
//     Colormap/smooth changes are instant — no re-fetch.
//   • Transition mode (transitionTheta set): single full-frame fetch from mapRender
//     (transition kernel outputs BGR directly; tile approach doesn't apply).

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

const TILE_PX  = 256
const MAX_CACHE = 256

// ── DOM refs ─────────────────────────────────────────────────────────────────
const wrapper   = ref<HTMLDivElement | null>(null)
const canvasEl  = ref<HTMLCanvasElement | null>(null)
const pending   = ref(false)
const error     = ref('')
const domW      = ref(0)
const domH      = ref(0)
let   ro: ResizeObserver | null = null

// ── Transition-mode full-frame state ─────────────────────────────────────────
let transitionTimer: ReturnType<typeof setTimeout> | null = null

// ── Tile cache ───────────────────────────────────────────────────────────────
interface TileEntry {
  key: string
  lastUsed: number
  width: number
  height: number
  metric: string
  // Escape metric raw data:
  rawU32?: Uint32Array   // iter counts [W*H]
  rawF32?: Float32Array  // |z|² at escape [W*H], 0 if bounded
  maxIter?: number
  // Non-escape metric raw data:
  rawF64?: Float64Array  // field values [W*H]
  fieldMin?: number
  fieldMax?: number
  // Colorized cache (invalidated on colormap/smooth change):
  imageData?: ImageData
  colorKey?: string      // "${colorMap}:${smooth?1:0}" when imageData was built
}

const tileCache = new Map<string, TileEntry>()
const inFlight  = new Set<string>()   // tile keys currently being fetched

// ── Colormap helpers (ported from colormap.hpp) ───────────────────────────────

function clamp255(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : Math.round(v)
}

function cosColor(n: number, freq: number): number {
  return 128 - 128 * Math.cos(freq * n * Math.PI)
}

function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  const c  = v * s
  const hh = h / 60
  const x  = c * (1 - Math.abs(hh % 2 - 1))
  let rr = 0, gg = 0, bb = 0
  if      (hh < 1) { rr = c; gg = x }
  else if (hh < 2) { rr = x; gg = c }
  else if (hh < 3) { gg = c; bb = x }
  else if (hh < 4) { gg = x; bb = c }
  else if (hh < 5) { rr = x; bb = c }
  else             { rr = c; bb = x }
  const m = v - c
  return [clamp255((rr+m)*255), clamp255((gg+m)*255), clamp255((bb+m)*255)]
}

function rainbowFromIndex(idx: number): [number, number, number] {
  if (idx <= 0)    return [0, 0, 0]
  if (idx >= 1785) return [255, 255, 255]
  let a0 = idx, a1 = 0, a2 = 0
  if      (255  < a0 && a0 <  510) { a1 = a0 - 255;  a0 = 510  - a0 }
  else if (509  < a0 && a0 <  765) { a1 = 255;        a0 = a0   - 510 }
  else if (764  < a0 && a0 < 1020) { a2 = a0 - 765;  a1 = 1020 - a0; a0 = a1 }
  else if (1019 < a0 && a0 < 1275) { a2 = 255;        a0 = a0   - 1020 }
  else if (1274 < a0 && a0 < 1530) { a2 = 255; a1 = a0 - 1275; a0 = 1530 - a0 }
  else if (a0 > 1529)              { a2 = 255; a1 = 255; a0 = a0 - 1530 }
  return [clamp255(a1), clamp255(a2), clamp255(a0)]
}

function smoothMu(iter: number, norm: number): number {
  if (norm > 1.0) {
    const mu = iter + 1 - Math.log2(Math.log2(norm))
    return mu > 0 ? mu : 0
  }
  return iter
}

type RGB = [number, number, number]

function colorizeEscape(iter: number, maxIter: number, norm: number, cm: ColorMap, smooth: boolean): RGB {
  if (iter >= maxIter) return [255, 255, 255]
  if (smooth) {
    const mu = smoothMu(iter, norm)
    const t  = (mu / 32) % 1
    if (cm === 'hsv_wheel') return hsvToRgb(t * 360, 1, 1)
    if (cm === 'tri765') {
      const mf = t * 765, m = Math.floor(mf), d = Math.floor((mf - m) * 255)
      const band = Math.floor(m / 255) % 3
      if (band === 0) return [clamp255(255-d), clamp255(d), 255]
      if (band === 1) return [clamp255(d), 255, clamp255(255-d)]
      return [255, clamp255(255-d), clamp255(d)]
    }
    if (cm === 'grayscale') { const v = clamp255(t * 255); return [v, v, v] }
    if (cm === 'mod17')     { const v = clamp255(Math.floor(mu) % 17 * 15); return [v, v, v] }
    // classic_cos (default)
    return [clamp255(cosColor(t, 53)), clamp255(cosColor(t, 27)), clamp255(cosColor(t, 139))]
  }
  // Non-smooth
  const n = (iter + 1) / (maxIter + 2)
  if (cm === 'mod17') {
    return [clamp255(iter % 256), clamp255(Math.floor(iter / 256)), clamp255((iter % 17) * 17)]
  }
  if (cm === 'hsv_wheel') return hsvToRgb((iter % 1440) / 4, 1, 1)
  if (cm === 'tri765') {
    const m = iter % 765, band = Math.floor(m / 255), d = m % 255
    if (band === 0) return [clamp255(255-d), clamp255(d), 255]
    if (band === 1) return [clamp255(d), 255, clamp255(255-d)]
    return [255, clamp255(255-d), clamp255(d)]
  }
  if (cm === 'grayscale') { const v = clamp255(n * 255); return [v, v, v] }
  return [clamp255(cosColor(n, 53)), clamp255(cosColor(n, 27)), clamp255(cosColor(n, 139))]
}

function colorizeFieldNorm(v01: number, cm: ColorMap): RGB {
  if (v01 < 0) v01 = 0; if (v01 > 1) v01 = 1
  const PI = Math.PI
  if (cm === 'grayscale')  { const v = clamp255(v01 * 255); return [v, v, v] }
  if (cm === 'hsv_wheel')  return hsvToRgb(v01 * 360, 1, 1)
  if (cm === 'tri765') {
    const m = Math.floor(v01 * 765), band = Math.floor(m / 255) % 3, d = m % 255
    if (band === 0) return [clamp255(255-d), clamp255(d), 255]
    if (band === 1) return [clamp255(d), 255, clamp255(255-d)]
    return [255, clamp255(255-d), clamp255(d)]
  }
  if (cm === 'mod17')      { const v = Math.min(16, Math.floor(v01 * 17)) * 15; return [v, v, v] }
  if (cm === 'hs_rainbow') return rainbowFromIndex(Math.min(1785, Math.floor(v01 * 1785)))
  // classic_cos
  return [
    clamp255(128 - 128*Math.cos(v01*2*PI)),
    clamp255(128 - 128*Math.cos(v01*2*PI + 2.094395)),
    clamp255(128 - 128*Math.cos(v01*2*PI + 4.188790)),
  ]
}

function colorizeFieldHsRainbow(x: number): RGB {
  if (x <= 0 || !isFinite(x)) return [255, 255, 255]
  const raw = (36/35 - Math.log2(x)) * 35
  return rainbowFromIndex(Math.max(0, Math.min(1785, Math.floor(raw))))
}

function colorizeFieldSmooth(x: number, cm: ColorMap): RGB {
  if (x <= 0) return [255, 255, 255]
  const bv = 2 - Math.log2(x)
  if (cm === 'hsv_wheel') {
    const idx = Math.max(0, Math.floor(180 * bv))
    return hsvToRgb((idx % 1440) / 4, 1, 1)
  }
  if (cm === 'tri765') {
    const idx = Math.max(0, Math.floor(96 * bv))
    const m = idx % 765, band = Math.floor(m / 255), d = m % 255
    if (band === 0) return [clamp255(255-d), clamp255(d), 255]
    if (band === 1) return [clamp255(d), 255, clamp255(255-d)]
    return [255, clamp255(255-d), clamp255(d)]
  }
  if (cm === 'hs_rainbow') return rainbowFromIndex(Math.max(0, Math.min(1785, Math.floor(35 * bv))))
  // grayscale + others
  const v = Math.max(0, Math.floor(32 * bv)) % 256
  return [v, v, v]
}

// ── Base64 decode ─────────────────────────────────────────────────────────────

function b64ToBuffer(b64: string): ArrayBuffer {
  const binStr = atob(b64)
  const buf    = new Uint8Array(binStr.length)
  for (let i = 0; i < binStr.length; i++) buf[i] = binStr.charCodeAt(i)
  return buf.buffer
}

// ── Tile key ──────────────────────────────────────────────────────────────────

function makeTileKey(tileCenterRe: number, tileCenterIm: number, tileSpan: number): string {
  const p = props
  const jKey = p.julia ? `j:${p.juliaRe?.toExponential(6)}:${p.juliaIm?.toExponential(6)}` : ''
  return [
    p.variant, p.metric, jKey,
    p.iterations, p.scalarType ?? 'auto',
    tileCenterRe.toExponential(10),
    tileCenterIm.toExponential(10),
    tileSpan.toExponential(8),
  ].join(':')
}

// ── LRU eviction ─────────────────────────────────────────────────────────────

function evictLRU() {
  if (tileCache.size <= MAX_CACHE) return
  let oldest: TileEntry | null = null
  for (const e of tileCache.values()) {
    if (!oldest || e.lastUsed < oldest.lastUsed) oldest = e
  }
  if (oldest) tileCache.delete(oldest.key)
}

// ── Colorize one tile entry (lazy, invalidated by colorKey change) ─────────────

function getColorKey(): string {
  return `${props.colorMap}:${props.smooth ? 1 : 0}`
}

function colorizeTile(entry: TileEntry): ImageData {
  const { width, height, metric } = entry
  const rgba = new Uint8ClampedArray(width * height * 4)
  const cm   = props.colorMap
  const sm   = !!props.smooth

  if (metric === 'escape' && entry.rawU32 && entry.rawF32) {
    const u32 = entry.rawU32, f32 = entry.rawF32, mi = entry.maxIter!
    for (let i = 0; i < width * height; i++) {
      const [r, g, b] = colorizeEscape(u32[i], mi, f32[i], cm, sm)
      rgba[i*4]=r; rgba[i*4+1]=g; rgba[i*4+2]=b; rgba[i*4+3]=255
    }
  } else if (metric !== 'escape' && entry.rawF64) {
    const f64 = entry.rawF64
    const fmin = entry.fieldMin!, fmax = entry.fieldMax!
    const denom = fmax > fmin ? fmax - fmin : 1
    for (let i = 0; i < width * height; i++) {
      const x = f64[i]
      let rgb: RGB
      if (cm === 'hs_rainbow') {
        rgb = colorizeFieldHsRainbow(x)
      } else if (sm) {
        rgb = colorizeFieldSmooth(x, cm)
      } else {
        rgb = colorizeFieldNorm(Math.min(1, Math.max(0, (x - fmin) / denom)), cm)
      }
      rgba[i*4]=rgb[0]; rgba[i*4+1]=rgb[1]; rgba[i*4+2]=rgb[2]; rgba[i*4+3]=255
    }
  }

  return new ImageData(rgba, width, height)
}

// ── Canvas drawing ────────────────────────────────────────────────────────────

function drawCanvas() {
  const cv = canvasEl.value
  if (!cv || domW.value < 1 || domH.value < 1) return
  const ctx = cv.getContext('2d')!
  ctx.clearRect(0, 0, domW.value, domH.value)

  if (props.transitionTheta !== null) return  // handled by transition path

  const ppu      = domH.value / props.scale          // pixels per complex unit (im)
  const span_re  = props.scale * domW.value / domH.value
  const tileSpan = TILE_PX / ppu                     // complex units per tile
  const viewLeft = props.centerRe - span_re / 2
  const viewTop  = props.centerIm + props.scale / 2

  const colorKey = getColorKey()
  const now      = Date.now()

  const nTX = Math.ceil(domW.value  / TILE_PX) + 2
  const nTY = Math.ceil(domH.value  / TILE_PX) + 2

  // Snap grid to tileSpan
  const gridRe = Math.floor(props.centerRe / tileSpan) * tileSpan
  const gridIm = Math.floor(props.centerIm / tileSpan) * tileSpan

  for (let ty = -1; ty <= nTY; ty++) {
    for (let tx = -1; tx <= nTX; tx++) {
      const tileCenterRe = gridRe + tx * tileSpan
      const tileCenterIm = gridIm + ty * tileSpan

      const px = Math.round((tileCenterRe - tileSpan/2 - viewLeft) * ppu)
      const py = Math.round((viewTop - tileCenterIm - tileSpan/2)  * ppu)

      if (px + TILE_PX < 0 || px >= domW.value || py + TILE_PX < 0 || py >= domH.value) continue

      const key = makeTileKey(tileCenterRe, tileCenterIm, tileSpan)
      const entry = tileCache.get(key)

      if (entry) {
        entry.lastUsed = now
        if (entry.colorKey !== colorKey) {
          entry.imageData = colorizeTile(entry)
          entry.colorKey  = colorKey
        }
        if (entry.imageData) ctx.putImageData(entry.imageData, px, py)
      } else if (!inFlight.has(key)) {
        fetchTile(key, tileCenterRe, tileCenterIm, tileSpan)
      }
    }
  }
}

// ── Tile fetch ────────────────────────────────────────────────────────────────

async function fetchTile(
  key: string,
  tileCenterRe: number,
  tileCenterIm: number,
  tileSpan: number,
) {
  if (inFlight.has(key) || tileCache.has(key)) return
  inFlight.add(key)
  error.value = ''
  pending.value = true

  try {
    const resp = await api.mapField({
      centerRe:   tileCenterRe,
      centerIm:   tileCenterIm,
      scale:      tileSpan,
      width:      TILE_PX,
      height:     TILE_PX,
      iterations: props.iterations,
      variant:    props.variant,
      metric:     props.metric,
      julia:      props.julia,
      juliaRe:    props.juliaRe ?? 0,
      juliaIm:    props.juliaIm ?? 0,
      engine:     props.engine ?? 'auto',
      scalarType: props.scalarType ?? 'auto',
    })

    const entry: TileEntry = {
      key,
      lastUsed: Date.now(),
      width:    resp.width,
      height:   resp.height,
      metric:   resp.metric,
      maxIter:  resp.maxIter,
    }

    if (resp.metric === 'escape' && resp.iterB64 && resp.finalMagB64) {
      entry.rawU32 = new Uint32Array(b64ToBuffer(resp.iterB64))
      entry.rawF32 = new Float32Array(b64ToBuffer(resp.finalMagB64))
    } else if (resp.fieldB64) {
      entry.rawF64    = new Float64Array(b64ToBuffer(resp.fieldB64))
      entry.fieldMin  = resp.fieldMin ?? 0
      entry.fieldMax  = resp.fieldMax ?? 1
    }

    tileCache.set(key, entry)
    evictLRU()

    emit('rendered', {
      generatedMs: resp.generatedMs,
      artifactId:  '',
      scalarUsed:  resp.scalarUsed,
      engineUsed:  'openmp',
    })
  } catch (e: any) {
    error.value = e?.message ?? String(e)
  } finally {
    inFlight.delete(key)
    if (inFlight.size === 0) pending.value = false
    drawCanvas()
  }
}

// ── Transition-mode full-frame fetch ──────────────────────────────────────────

async function fetchTransitionFrame() {
  if (domW.value < 16 || domH.value < 16) return
  pending.value = true
  error.value = ''
  const req: MapRenderRequest = {
    centerRe:        props.centerRe,
    centerIm:        props.centerIm,
    scale:           props.scale,
    width:           domW.value,
    height:          domH.value,
    iterations:      props.iterations,
    variant:         props.variant,
    metric:          props.metric,
    colorMap:        props.colorMap,
    smooth:          props.smooth,
    transitionTheta: props.transitionTheta!,
  }
  if (props.engine)     (req as any).engine     = props.engine
  if (props.scalarType) (req as any).scalarType = props.scalarType
  try {
    const resp = await api.mapRender(req) as any
    const imgUrl = api.artifactContentUrl(resp.artifactId)
    await new Promise<void>((res, rej) => {
      const img = new Image()
      img.onload = () => {
        const ctx = canvasEl.value?.getContext('2d')
        if (ctx) ctx.drawImage(img, 0, 0, domW.value, domH.value)
        res()
      }
      img.onerror = rej
      img.src = imgUrl
    })
    emit('rendered', { generatedMs: resp.generatedMs, artifactId: resp.artifactId, engineUsed: resp.engineUsed, scalarUsed: resp.scalarUsed })
  } catch (e: any) {
    error.value = e?.message ?? String(e)
  } finally {
    pending.value = false
  }
}

// ── Viewport-change handler ───────────────────────────────────────────────────

function scheduleLayout() {
  if (props.transitionTheta !== null) {
    if (transitionTimer) clearTimeout(transitionTimer)
    transitionTimer = setTimeout(fetchTransitionFrame, 120)
  } else {
    drawCanvas()
  }
}

// Invalidate colorized caches + redraw when colormap/smooth changes.
function onColormapChange() {
  if (props.transitionTheta !== null) {
    scheduleLayout()   // transition always re-fetches anyway
    return
  }
  const colorKey = getColorKey()
  for (const e of tileCache.values()) {
    if (e.colorKey !== colorKey) {
      e.imageData = undefined
      e.colorKey  = undefined
    }
  }
  drawCanvas()
}

// Invalidate all tiles (params that affect compute, not just color).
function invalidateAll() {
  tileCache.clear()
  inFlight.clear()
  pending.value = false
}

// ── Watchers ──────────────────────────────────────────────────────────────────

watch(() => [
  props.centerRe, props.centerIm, props.scale,
  props.variant, props.metric,
  props.iterations, props.julia, props.juliaRe, props.juliaIm,
  props.engine, props.scalarType,
  props.transitionTheta,
  domW.value, domH.value,
], () => {
  invalidateAll()
  scheduleLayout()
})

watch(() => [props.colorMap, props.smooth], onColormapChange)

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
        if (canvasEl.value) {
          canvasEl.value.width  = w
          canvasEl.value.height = h
        }
      }
    }
  })
  ro.observe(wrapper.value)
  const w = Math.round(wrapper.value.clientWidth)
  const h = Math.round(wrapper.value.clientHeight)
  domW.value = w
  domH.value = h
  if (canvasEl.value) { canvasEl.value.width = w; canvasEl.value.height = h }
  scheduleLayout()
})

onBeforeUnmount(() => {
  ro?.disconnect()
  if (transitionTimer) clearTimeout(transitionTimer)
})

// ── Interaction ───────────────────────────────────────────────────────────────

function onWheel(e: WheelEvent) {
  e.preventDefault()
  if (!wrapper.value) return
  const rect   = wrapper.value.getBoundingClientRect()
  const px     = (e.clientX - rect.left) / rect.width
  const py     = (e.clientY - rect.top)  / rect.height
  const aspect = rect.width / rect.height
  const wx     = props.centerRe + (px - 0.5) * props.scale * aspect
  const wy     = props.centerIm + (0.5 - py) * props.scale
  const factor = e.deltaY > 0 ? 1.25 : 0.8
  const newScale  = props.scale * factor
  const newCenterRe = wx - (px - 0.5) * newScale * aspect
  const newCenterIm = wy + (py - 0.5) * newScale
  emit('viewport-change', { centerRe: newCenterRe, centerIm: newCenterIm, scale: newScale })
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
    scale: dragStart.sc,
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
       @wheel="onWheel"
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
