<script setup lang="ts">
import { inject, onMounted, ref, watch, computed } from 'vue'
import MapCanvas from '../components/MapCanvas.vue'
import SpecialPointList from '../components/SpecialPointList.vue'
import {
  api, VARIANTS, METRICS, COLORMAPS, VARIANT_LABELS,
  type Variant, type Metric, type ColorMap, type SpecialPoint,
  type LnMapResponse,
} from '../api'
import type { StatusState } from '../types'
import { t, lang } from '../i18n'

// Metric display labels
const METRIC_LABELS: Record<string, { en: string; zh: string }> = {
  escape:             { en: 'Escape time',   zh: '逃逸时间' },
  min_abs:            { en: 'Min |z|',       zh: '最小 |z|' },
  max_abs:            { en: 'Max |z|',       zh: '最大 |z|' },
  envelope:           { en: 'Envelope',      zh: '包络' },
  min_pairwise_dist:  { en: 'Min pairwise',  zh: '最小轨道距' },
}

const COLORMAP_LABELS: Record<string, { en: string; zh: string }> = {
  classic_cos: { en: 'Classic Cos', zh: '经典余弦' },
  mod17:       { en: 'Mod-17',      zh: 'Mod-17' },
  hsv_wheel:   { en: 'HSV Wheel',   zh: 'HSV 色轮' },
  tri765:      { en: 'Tri-765',     zh: 'Tri-765' },
  grayscale:   { en: 'Grayscale',   zh: '灰度' },
  hs_rainbow:  { en: 'HS Rainbow',  zh: '隐结构彩虹' },
}

const status = inject<StatusState>('status')!

// ── Left / Mandelbrot viewport ────────────────────────────────────────────────
const centerRe   = ref(-0.75)
const centerIm   = ref( 0.0)
const scale      = ref( 3.0)
const iterations = ref(1024)

const variant  = ref<Variant>('mandelbrot')
const metric   = ref<Metric>('escape')
const colorMap = ref<ColorMap>('classic_cos')
const smooth   = ref(false)

const transitionOn = ref(false)
const thetaDeg     = ref(0)

// ── Julia mode ────────────────────────────────────────────────────────────────
const juliaOn  = ref(false)
const juliaRe  = ref(-0.7)
const juliaIm  = ref(0.27)

// Right / Julia viewport (independent)
const jCenterRe = ref(0.0)
const jCenterIm = ref(0.0)
const jScale    = ref(4.0)

// Format c for display
const juliaLabel = computed(() => {
  const sign = juliaIm.value >= 0 ? '+' : ''
  return `${juliaRe.value.toPrecision(10)} ${sign}${juliaIm.value.toPrecision(10)}i`
})

// Left-canvas click: pick julia c AND recenter left map
function onPickJulia(pos: { re: number; im: number }) {
  juliaRe.value = pos.re
  juliaIm.value = pos.im
  centerRe.value = pos.re
  centerIm.value = pos.im
}

function onJuliaViewport(v: { centerRe: number; centerIm: number; scale: number }) {
  jCenterRe.value = v.centerRe
  jCenterIm.value = v.centerIm
  jScale.value    = v.scale
}

// ── Engine / scalar ───────────────────────────────────────────────────────────
const engineMode = ref<'auto' | 'openmp' | 'avx512' | 'cuda' | 'hybrid'>('auto')
const scalarMode = ref<'auto' | 'fp64' | 'fx64'>('auto')

// ── Status rail sync ─────────────────────────────────────────────────────────
const lastMs         = ref<number | null>(null)
const lastArtifactId = ref<string>('')
const lastEngine     = ref('')
const lastScalar     = ref('')

function syncStatus() {
  status.cRe      = centerRe.value
  status.cIm      = centerIm.value
  status.zoom     = scale.value
  status.iter     = iterations.value
  status.variant  = variant.value
  status.metric   = metric.value
  status.renderMs = lastMs.value
  status.engine   = lastEngine.value || engineMode.value
  status.scalar   = lastScalar.value || scalarMode.value
  status.message  = 'ready'
}

watch([centerRe, centerIm, scale, iterations, variant, metric, lastMs], syncStatus, { immediate: true })

onMounted(() => {
  const pending = sessionStorage.getItem('fs_pending_center')
  if (pending) {
    try {
      const c = JSON.parse(pending)
      if (typeof c.re === 'number' && typeof c.im === 'number') {
        centerRe.value = c.re
        centerIm.value = c.im
        scale.value    = 0.01
      }
    } catch {}
    sessionStorage.removeItem('fs_pending_center')
  }
})

function onViewportChange(v: { centerRe: number; centerIm: number; scale: number }) {
  centerRe.value = v.centerRe
  centerIm.value = v.centerIm
  scale.value    = v.scale
}

function onRendered(meta: { generatedMs: number; artifactId: string; engineUsed?: string; scalarUsed?: string }) {
  lastMs.value         = meta.generatedMs
  lastArtifactId.value = meta.artifactId
  lastEngine.value     = meta.engineUsed ?? ''
  lastScalar.value     = meta.scalarUsed ?? ''
  syncStatus()
}

function resetView() {
  centerRe.value = 0.0
  centerIm.value = 0.0
  scale.value    = 4.0
}

function onImportPoint(p: SpecialPoint) {
  centerRe.value = p.real
  centerIm.value = p.imag
  scale.value    = 0.01
}

function exportPng() {
  if (!lastArtifactId.value) return
  window.open(api.artifactDownloadUrl(lastArtifactId.value), '_blank')
}

// ── ln-map & video export ─────────────────────────────────────────────────────
const lnBusy         = ref(false)
const lnStatus       = ref('')
const lastLnArtifact = ref<LnMapResponse | null>(null)

async function exportLnMap() {
  lnBusy.value  = true
  lnStatus.value = 'rendering ln-map…'
  lastLnArtifact.value = null
  try {
    const resp = await api.lnMap({
      centerRe: centerRe.value,
      centerIm: centerIm.value,
      widthS: 960,
      depthOctaves: 20,
      variant: variant.value,
      colorMap: colorMap.value,
      iterations: Math.max(iterations.value, 2048),
    })
    lastLnArtifact.value = resp
    lnStatus.value = `ln-map: ${resp.widthS}×${resp.heightT}, ${resp.generatedMs.toFixed(0)}ms`
    window.open(api.artifactDownloadUrl(resp.artifactId), '_blank')
  } catch (e: any) {
    lnStatus.value = 'failed: ' + (e?.message || e)
  } finally {
    lnBusy.value = false
  }
}

const videoModalOpen  = ref(false)
const videoFps        = ref(30)
const videoDuration   = ref(8.0)
const videoWidth      = ref(720)
const videoHeight     = ref(720)
const videoBusy       = ref(false)
const videoStatus     = ref('')
const videoArtifactId = ref('')

function openVideoModal() {
  if (!lastLnArtifact.value) return
  videoModalOpen.value  = true
  videoStatus.value     = ''
  videoArtifactId.value = ''
}

async function exportVideo() {
  if (!lastLnArtifact.value) return
  videoBusy.value   = true
  videoStatus.value = 'generating video…'
  videoArtifactId.value = ''
  try {
    const resp = await api.videoZoom({
      lnMapArtifactId: lastLnArtifact.value.artifactId,
      fps: videoFps.value,
      durationSec: videoDuration.value,
      width: videoWidth.value,
      height: videoHeight.value,
    })
    videoArtifactId.value = resp.artifactId
    videoStatus.value = `${resp.frameCount} frames · ${resp.generatedMs.toFixed(0)}ms`
  } catch (e: any) {
    videoStatus.value = 'failed: ' + (e?.message || e)
  } finally {
    videoBusy.value = false
  }
}

function downloadVideo() {
  if (!videoArtifactId.value) return
  window.open(api.artifactDownloadUrl(videoArtifactId.value), '_blank')
}
</script>

<template>
  <div class="map-view">

    <!-- ── Controls bar ──────────────────────────────────────────────────── -->
    <div class="controls">
      <div class="group">
        <label>{{ t('variant') }}</label>
        <select v-model="variant" :disabled="transitionOn">
          <option v-for="v in VARIANTS" :key="v" :value="v">{{ VARIANT_LABELS[v][lang] }}</option>
        </select>
      </div>

      <div class="group">
        <label>{{ t('metric') }}</label>
        <select v-model="metric">
          <option v-for="m in METRICS" :key="m" :value="m">{{ METRIC_LABELS[m]?.[lang] ?? m }}</option>
        </select>
      </div>

      <div class="group">
        <label>{{ t('colormap') }}</label>
        <select v-model="colorMap">
          <option v-for="c in COLORMAPS" :key="c" :value="c">{{ COLORMAP_LABELS[c]?.[lang] ?? c }}</option>
        </select>
      </div>

      <div class="group">
        <label>
          <input type="checkbox" v-model="smooth" style="width:auto;margin-right:6px" />
          {{ t('smooth') }}
        </label>
      </div>

      <div class="group">
        <label>{{ t('iterations') }}</label>
        <input type="number" v-model.number="iterations" min="16" max="1000000" step="128" />
      </div>

      <div class="group transition-group">
        <label>
          <input type="checkbox" v-model="transitionOn" style="width:auto;margin-right:6px" />
          {{ t('transition') }}
        </label>
        <div v-if="transitionOn" class="theta-row">
          <input type="range" min="0" max="90" step="0.1" v-model.number="thetaDeg" />
          <span class="num">{{ thetaDeg.toFixed(1) }}°</span>
        </div>
      </div>

      <div class="group">
        <label>
          <input type="checkbox" v-model="juliaOn" style="width:auto;margin-right:6px" />
          {{ t('julia') }}
        </label>
      </div>

      <div class="group">
        <label>{{ t('engine') }}</label>
        <select v-model="engineMode">
          <option value="auto">auto</option>
          <option value="cuda">cuda</option>
          <option value="avx512">avx512</option>
          <option value="hybrid">hybrid</option>
          <option value="openmp">openmp</option>
        </select>
      </div>

      <div class="group">
        <label>{{ t('scalar') }}</label>
        <select v-model="scalarMode">
          <option value="auto">auto</option>
          <option value="fp64">fp64</option>
          <option value="fx64">fx64</option>
        </select>
      </div>

      <div class="spacer"></div>

      <button @click="resetView" :title="t('reset')">⌂ {{ t('reset') }}</button>
      <button @click="exportPng" :disabled="!lastArtifactId">{{ t('export_png') }}</button>
      <button @click="exportLnMap" :disabled="lnBusy">{{ t('export_lnmap') }}</button>
      <button @click="openVideoModal" :disabled="!lastLnArtifact" title="Export zoom video">video →</button>
    </div>

    <!-- ── ln-map status strip ───────────────────────────────────────────── -->
    <div v-if="lnStatus" class="ln-status mono">{{ lnStatus }}</div>

    <!-- ── Julia info strip ─────────────────────────────────────────────── -->
    <div v-if="juliaOn" class="julia-strip mono">
      <span class="julia-label">{{ t('julia_selected_c') }}:</span>
      <span class="julia-val">{{ juliaLabel }}</span>
      <span class="julia-hint">{{ t('julia_hint') }}</span>
    </div>

    <!-- ── Main stage: dual-pane or single ──────────────────────────────── -->
    <div class="stage">

      <!-- Single-pane mode (no Julia) -->
      <template v-if="!juliaOn">
        <MapCanvas
          :centerRe="centerRe" :centerIm="centerIm" :scale="scale"
          :iterations="iterations" :variant="variant" :metric="metric"
          :colorMap="colorMap" :smooth="smooth"
          :transitionTheta="transitionOn ? thetaDeg * Math.PI / 180 : null"
          :engine="engineMode" :scalarType="scalarMode"
          @viewport-change="onViewportChange"
          @rendered="onRendered"
        />
        <aside class="points">
          <SpecialPointList @import-point="onImportPoint" />
        </aside>
      </template>

      <!-- Dual-pane Julia mode -->
      <template v-else>
        <div class="dual-pane">
          <!-- Left: Mandelbrot / variant — click picks julia c -->
          <div class="pane">
            <div class="pane-header mono">
              <span class="pane-title">{{ t('julia_left') }}: {{ VARIANT_LABELS[variant][lang] }}</span>
              <span class="pane-meta">
                {{ t('center') }}: {{ centerRe.toPrecision(10) }} + {{ centerIm.toPrecision(10) }}i
                &nbsp;·&nbsp;{{ t('scale') }}: {{ scale.toPrecision(6) }}
              </span>
            </div>
            <div class="pane-canvas">
              <MapCanvas
                :centerRe="centerRe" :centerIm="centerIm" :scale="scale"
                :iterations="iterations" :variant="variant" :metric="metric"
                :colorMap="colorMap" :smooth="smooth"
                :transitionTheta="transitionOn ? thetaDeg * Math.PI / 180 : null"
                :engine="engineMode" :scalarType="scalarMode"
                @viewport-change="onViewportChange"
                @rendered="onRendered"
                @click-world="onPickJulia"
              />
            </div>
          </div>

          <!-- Right: Julia set J(c) — own viewport -->
          <div class="pane">
            <div class="pane-header mono">
              <span class="pane-title">{{ t('julia_right') }}</span>
              <span class="pane-meta">
                julia c: {{ juliaRe.toPrecision(10) }} + {{ juliaIm.toPrecision(10) }}i
                &nbsp;·&nbsp;{{ t('center') }}: {{ jCenterRe.toPrecision(6) }} + {{ jCenterIm.toPrecision(6) }}i
                &nbsp;·&nbsp;{{ t('scale') }}: {{ jScale.toPrecision(6) }}
              </span>
            </div>
            <div class="pane-canvas">
              <MapCanvas
                :centerRe="jCenterRe" :centerIm="jCenterIm" :scale="jScale"
                :iterations="iterations" :variant="variant" :metric="metric"
                :colorMap="colorMap" :smooth="smooth"
                :transitionTheta="null"
                :julia="true" :juliaRe="juliaRe" :juliaIm="juliaIm"
                :engine="engineMode" :scalarType="scalarMode"
                @viewport-change="onJuliaViewport"
              />
            </div>
          </div>
        </div>
      </template>
    </div>

    <!-- ── Video export modal ────────────────────────────────────────────── -->
    <Teleport to="body">
      <div v-if="videoModalOpen" class="modal-backdrop" @click.self="videoModalOpen = false">
        <div class="modal">
          <div class="modal-title">{{ t('video_title') }}</div>
          <div class="modal-body">
            <div class="mrow">
              <label>{{ t('video_fps') }}</label>
              <input type="number" v-model.number="videoFps" min="1" max="60" step="1" />
            </div>
            <div class="mrow">
              <label>{{ t('video_duration') }}</label>
              <input type="number" v-model.number="videoDuration" min="1" max="300" step="1" />
            </div>
            <div class="mrow">
              <label>{{ t('video_width') }}</label>
              <input type="number" v-model.number="videoWidth" min="128" max="1920" step="64" />
            </div>
            <div class="mrow">
              <label>{{ t('video_height') }}</label>
              <input type="number" v-model.number="videoHeight" min="128" max="1080" step="64" />
            </div>
            <div v-if="lastLnArtifact" class="mrow source mono">
              {{ t('video_source') }}: {{ lastLnArtifact.artifactId }}<br/>
              {{ lastLnArtifact.widthS }}×{{ lastLnArtifact.heightT }} · {{ lastLnArtifact.depthOctaves }} oct
            </div>
          </div>
          <div class="modal-footer">
            <button @click="videoModalOpen = false" class="btn-cancel">{{ t('video_cancel') }}</button>
            <button @click="exportVideo" :disabled="videoBusy" class="btn-go">
              {{ videoBusy ? t('loading') : t('video_render') }}
            </button>
          </div>
          <div v-if="videoStatus" class="modal-status mono">{{ videoStatus }}</div>
          <div v-if="videoArtifactId" class="modal-footer">
            <button @click="downloadVideo" class="btn-go">{{ t('video_download') }}</button>
          </div>
        </div>
      </div>
    </Teleport>

  </div>
</template>

<style scoped>
.map-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

/* ── Controls ── */
.controls {
  display: flex;
  align-items: flex-end;
  gap: 14px;
  padding: 10px 14px;
  border-bottom: 1px solid var(--rule);
  background: var(--panel);
  flex-wrap: wrap;
  flex-shrink: 0;
}

.group {
  display: flex;
  flex-direction: column;
  min-width: 100px;
}

.group.transition-group { min-width: 200px; }

.theta-row {
  display: flex;
  align-items: center;
  gap: 8px;
}
.theta-row input[type="range"] { flex: 1; }

.spacer { flex: 1; }

/* ── ln-map status ── */
.ln-status {
  padding: 5px 14px;
  color: var(--text-dim);
  font-size: var(--fs-label);
  border-bottom: 1px solid var(--rule);
  background: var(--bg-raised);
  flex-shrink: 0;
}

/* ── Julia strip ── */
.julia-strip {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 5px 14px;
  background: var(--accent-weak);
  border-bottom: 1px solid var(--accent-edge);
  font-size: var(--fs-label);
  flex-shrink: 0;
}
.julia-label { color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.08em; }
.julia-val   { color: var(--accent); }
.julia-hint  { color: var(--text-faint); margin-left: auto; }

/* ── Stage ── */
.stage {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: 1fr 320px;
}

/* single-pane: canvas takes full grid, points panel on right */
.stage > .map-canvas-wrap,
.stage > canvas { grid-column: 1; }

.points {
  border-left: 1px solid var(--rule);
  padding: 12px 14px;
  background: var(--bg-raised);
  overflow-y: auto;
}

/* ── Dual pane ── */
.dual-pane {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 1fr 1fr;
  height: 100%;
  min-height: 0;
}

.pane {
  display: flex;
  flex-direction: column;
  min-height: 0;
  border-right: 1px solid var(--rule);
}
.pane:last-child { border-right: none; }

.pane-header {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 6px 10px;
  background: var(--panel);
  border-bottom: 1px solid var(--rule);
  flex-shrink: 0;
}
.pane-title {
  font-size: var(--fs-label);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--accent);
}
.pane-meta {
  font-size: 10px;
  color: var(--text-dim);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.pane-canvas {
  flex: 1;
  min-height: 0;
  position: relative;
}

/* ── Video modal ── */
.modal-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
}
.modal {
  background: var(--panel);
  border: 1px solid var(--rule);
  width: 340px;
  padding: 24px 28px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.modal-title {
  font-family: var(--mono);
  font-size: 12px;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.1em;
}
.modal-body { display: flex; flex-direction: column; gap: 10px; }
.mrow {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}
.mrow label {
  font-size: 11px;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  min-width: 90px;
}
.mrow input[type="number"] { width: 100px; }
.mrow.source {
  flex-direction: column;
  align-items: flex-start;
  font-size: 10px;
  color: var(--text-dim);
  line-height: 1.5;
}
.modal-footer { display: flex; gap: 10px; justify-content: flex-end; }
.btn-cancel {
  background: transparent;
  border: 1px solid var(--rule);
  color: var(--text-dim);
  padding: 6px 14px;
  font-family: var(--mono);
  font-size: 12px;
  cursor: pointer;
}
.btn-go {
  background: var(--accent);
  color: #000;
  border: none;
  padding: 6px 16px;
  font-family: var(--mono);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}
.btn-go:disabled { opacity: 0.5; cursor: default; }
.modal-status { font-size: 10px; color: var(--text-dim); }
</style>
