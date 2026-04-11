<script setup lang="ts">
import { inject, onMounted, ref, watch } from 'vue'
import MapCanvas from '../components/MapCanvas.vue'
import SpecialPointList from '../components/SpecialPointList.vue'
import {
  api, VARIANTS, METRICS, COLORMAPS,
  type Variant, type Metric, type ColorMap, type SpecialPoint,
  type LnMapResponse, type MapRenderResponse,
} from '../api'
import type { StatusState } from '../types'
import { t } from '../i18n'

const status = inject<StatusState>('status')!

const centerRe = ref(-0.75)
const centerIm = ref( 0.0)
const scale    = ref( 3.0)
const iterations = ref(1024)

const variant  = ref<Variant>('mandelbrot')
const metric   = ref<Metric>('escape')
const colorMap = ref<ColorMap>('classic_cos')
const smooth   = ref(false)   // ln-smooth continuous coloring

const transitionOn = ref(false)
const thetaDeg     = ref(0)   // 0–90 degrees

// Phase 3: engine + scalar type selection
const engineMode  = ref<'auto' | 'openmp' | 'avx512' | 'cuda' | 'hybrid'>('auto')
const scalarMode  = ref<'auto' | 'fp64' | 'fx64'>('auto')

const lastMs = ref<number | null>(null)
const lastArtifactId = ref<string>('')
const lastEngine = ref('')
const lastScalar = ref('')

function syncStatus() {
  status.cRe     = centerRe.value
  status.cIm     = centerIm.value
  status.zoom    = scale.value
  status.iter    = iterations.value
  status.variant = variant.value
  status.metric  = metric.value
  status.renderMs = lastMs.value
  status.engine  = lastEngine.value || engineMode.value
  status.scalar  = lastScalar.value || scalarMode.value
  status.message = 'ready'
}

watch([centerRe, centerIm, scale, iterations, variant, metric, lastMs], syncStatus, { immediate: true })

// If the PointsView navigated here with a pending center, adopt it.
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
  lastMs.value = meta.generatedMs
  lastArtifactId.value = meta.artifactId
  lastEngine.value = meta.engineUsed ?? ''
  lastScalar.value = meta.scalarUsed ?? ''
  syncStatus()
}

function resetView() {
  centerRe.value =  0.0
  centerIm.value =  0.0
  scale.value    =  4.0
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

const lnBusy   = ref(false)
const lnStatus = ref('')
const lastLnArtifact = ref<LnMapResponse | null>(null)

async function exportLnMap() {
  lnBusy.value = true
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

// ---- Video export modal ----
const videoModalOpen = ref(false)
const videoFps       = ref(30)
const videoDuration  = ref(8.0)
const videoWidth     = ref(720)
const videoHeight    = ref(720)
const videoBusy      = ref(false)
const videoStatus    = ref('')
const videoArtifactId = ref('')

function openVideoModal() {
  if (!lastLnArtifact.value) return
  videoModalOpen.value = true
  videoStatus.value = ''
  videoArtifactId.value = ''
}

async function exportVideo() {
  if (!lastLnArtifact.value) return
  videoBusy.value  = true
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
    <div class="controls">
      <div class="group">
        <label>{{ t('variant') }}</label>
        <select v-model="variant" :disabled="transitionOn">
          <option v-for="v in VARIANTS" :key="v" :value="v">{{ v }}</option>
        </select>
      </div>

      <div class="group">
        <label>{{ t('metric') }}</label>
        <select v-model="metric">
          <option v-for="m in METRICS" :key="m" :value="m">{{ m }}</option>
        </select>
      </div>

      <div class="group">
        <label>{{ t('colormap') }}</label>
        <select v-model="colorMap">
          <option v-for="c in COLORMAPS" :key="c" :value="c">{{ c }}</option>
        </select>
      </div>

      <div class="group">
        <label>
          <input type="checkbox" v-model="smooth" style="width:auto; margin-right:6px;" />
          {{ t('smooth') }}
        </label>
      </div>

      <div class="group">
        <label>{{ t('iterations') }}</label>
        <input type="number" v-model.number="iterations" min="16" max="1000000" step="128" />
      </div>

      <div class="group transition-group">
        <label>
          <input type="checkbox" v-model="transitionOn" style="width:auto; margin-right:6px;" />
          {{ t('transition') }} m↔b
        </label>
        <div v-if="transitionOn" class="theta-row">
          <input type="range" min="0" max="90" step="0.1" v-model.number="thetaDeg" />
          <span class="num">{{ thetaDeg.toFixed(1) }}°</span>
        </div>
      </div>

      <div class="group">
        <label>engine</label>
        <select v-model="engineMode">
          <option value="auto">auto</option>
          <option value="cuda">cuda</option>
          <option value="avx512">avx512</option>
          <option value="hybrid">hybrid</option>
          <option value="openmp">openmp</option>
        </select>
      </div>

      <div class="group">
        <label>scalar</label>
        <select v-model="scalarMode">
          <option value="auto">auto</option>
          <option value="fp64">fp64</option>
          <option value="fx64">fx64</option>
        </select>
      </div>

      <div class="spacer"></div>

      <button @click="resetView" title="Reset to 0+0i, scale 4">⌂ reset</button>
      <button @click="exportPng" :disabled="!lastArtifactId">{{ t('export_png') }}</button>
      <button @click="exportLnMap" :disabled="lnBusy">{{ t('export_lnmap') }}</button>
      <button @click="openVideoModal" :disabled="!lastLnArtifact" title="Export zoom video from ln-map">video →</button>
    </div>
    <div v-if="lnStatus" class="ln-status mono">{{ lnStatus }}</div>

    <!-- Video export modal -->
    <Teleport to="body">
      <div v-if="videoModalOpen" class="modal-backdrop" @click.self="videoModalOpen = false">
        <div class="modal">
          <div class="modal-title">export zoom video</div>
          <div class="modal-body">
            <div class="mrow">
              <label>fps</label>
              <input type="number" v-model.number="videoFps" min="1" max="60" step="1" />
            </div>
            <div class="mrow">
              <label>duration (s)</label>
              <input type="number" v-model.number="videoDuration" min="1" max="300" step="1" />
            </div>
            <div class="mrow">
              <label>width px</label>
              <input type="number" v-model.number="videoWidth" min="128" max="1920" step="64" />
            </div>
            <div class="mrow">
              <label>height px</label>
              <input type="number" v-model.number="videoHeight" min="128" max="1080" step="64" />
            </div>
            <div class="mrow source mono" v-if="lastLnArtifact">
              source: {{ lastLnArtifact.artifactId }}<br/>
              {{ lastLnArtifact.widthS }}×{{ lastLnArtifact.heightT }} · {{ lastLnArtifact.depthOctaves }} oct
            </div>
          </div>
          <div class="modal-footer">
            <button @click="videoModalOpen = false" class="btn-cancel">cancel</button>
            <button @click="exportVideo" :disabled="videoBusy" class="btn-go">
              {{ videoBusy ? 'rendering…' : 'render' }}
            </button>
          </div>
          <div v-if="videoStatus" class="modal-status mono">{{ videoStatus }}</div>
          <div v-if="videoArtifactId" class="modal-footer">
            <button @click="downloadVideo" class="btn-go">download mp4</button>
          </div>
        </div>
      </div>
    </Teleport>

    <div class="stage">
      <MapCanvas
        :centerRe="centerRe"
        :centerIm="centerIm"
        :scale="scale"
        :iterations="iterations"
        :variant="variant"
        :metric="metric"
        :colorMap="colorMap"
        :smooth="smooth"
        :transitionTheta="transitionOn ? thetaDeg * Math.PI / 180 : null"
        :engine="engineMode"
        :scalarType="scalarMode"
        @viewport-change="onViewportChange"
        @rendered="onRendered"
      />
    </div>

    <aside class="points">
      <SpecialPointList @import-point="onImportPoint" />
    </aside>
  </div>
</template>

<style scoped>
.map-view {
  display: grid;
  grid-template-columns: 1fr 320px;
  grid-template-rows: auto auto 1fr;
  height: 100%;
  gap: 0;
}

.controls {
  grid-column: 1 / -1;
  display: flex;
  align-items: flex-end;
  gap: 14px;
  padding: 12px 14px;
  border-bottom: 1px solid var(--rule);
  background: var(--panel);
  flex-wrap: wrap;
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

.ln-status {
  grid-column: 1 / -1;
  padding: 6px 14px;
  color: var(--text-dim);
  font-size: var(--fs-label);
  border-bottom: 1px solid var(--rule);
  background: var(--bg-raised);
}

.stage {
  position: relative;
  min-height: 0;
}

.points {
  border-left: 1px solid var(--rule);
  padding: 12px 14px;
  background: var(--bg-raised);
  overflow: auto;
}

/* ---- Video modal ---- */
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

.modal-body {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

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

.mrow input[type="number"] {
  width: 100px;
}

.mrow.source {
  flex-direction: column;
  align-items: flex-start;
  font-size: 10px;
  color: var(--text-dim);
  line-height: 1.5;
}

.modal-footer {
  display: flex;
  gap: 10px;
  justify-content: flex-end;
}

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

.btn-go:disabled {
  opacity: 0.5;
  cursor: default;
}

.modal-status {
  font-size: 10px;
  color: var(--text-dim);
}
</style>
