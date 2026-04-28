<script setup lang="ts">
import { inject, onMounted, ref, watch, computed } from 'vue'
import MapCanvas from '../components/MapCanvas.vue'
import SpecialPointList from '../components/SpecialPointList.vue'
import {
  api, VARIANTS, METRICS, COLORMAPS, VARIANT_LABELS,
  type Metric, type ColorMap, type SpecialPoint,
  type VideoExportResponse, type VideoPreviewResponse, type RunProgress, type RunStatusResponse, type CustomVariant,
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

type ExportPreset = {
  key: string
  label: { en: string; zh: string }
  width: number
  height: number
}

const EXPORT_PRESETS: ExportPreset[] = [
  { key: 'fhd',       label: { en: 'FHD 16:9',       zh: 'FHD 16:9' },       width: 1920, height: 1080 },
  { key: 'qhd',       label: { en: 'QHD 16:9',       zh: 'QHD 16:9' },       width: 2560, height: 1440 },
  { key: '4k',        label: { en: '4K UHD 16:9',    zh: '4K UHD 16:9' },    width: 3840, height: 2160 },
  { key: '8k',        label: { en: '8K UHD 16:9',    zh: '8K UHD 16:9' },    width: 7680, height: 4320 },
  { key: 'wuxga',     label: { en: 'WUXGA 16:10',    zh: 'WUXGA 16:10' },    width: 1920, height: 1200 },
  { key: 'wqxga',     label: { en: 'WQXGA 16:10',    zh: 'WQXGA 16:10' },    width: 2560, height: 1600 },
  { key: 'uwqhd',     label: { en: 'UWQHD 21:9',     zh: 'UWQHD 21:9' },     width: 3440, height: 1440 },
  { key: 'phone_fhd', label: { en: 'Phone 9:16 FHD', zh: '手机 9:16 FHD' },  width: 1080, height: 1920 },
  { key: 'phone_qhd', label: { en: 'Phone 9:16 QHD', zh: '手机 9:16 QHD' },  width: 1440, height: 2560 },
]

// ── Left / Mandelbrot viewport ────────────────────────────────────────────────
const centerRe   = ref(-0.75)
const centerIm   = ref( 0.0)
const scale      = ref( 3.0)
const iterations = ref(1024)

const variant  = ref<string>('mandelbrot')  // Variant literal or "custom:HASH"
const metric   = ref<Metric>('escape')
const colorMap = ref<ColorMap>('classic_cos')
const smooth   = ref(false)

// ── Custom variants ───────────────────────────────────────────────────────────
const customVariants     = ref<CustomVariant[]>([])
const showCustomPanel    = ref(false)
const customFormula      = ref('z^2 + c')
const customName         = ref('my_variant')
const customBailoutDirty = ref(false)
const customBailout      = ref(suggestCustomBailout(customFormula.value))
const customCompiling    = ref(false)
const customCompileMsg   = ref('')

function suggestCustomBailout(formula: string): number {
  const s = formula.replace(/\s+/g, '').toLowerCase()
  const power = s === 'z*z+c'
    ? 2
    : s === 'z*z*z+c'
      ? 3
      : Number(s.match(/^z\^(\d+)\+c$/)?.[1] ?? s.match(/^pow\(z,(\d+)\)\+c$/)?.[1])
  return Number.isFinite(power) && power >= 2 ? Math.pow(2, 1 / (power - 1)) : 2
}

watch(customFormula, (formula) => {
  if (!customBailoutDirty.value) customBailout.value = suggestCustomBailout(formula)
})

async function loadCustomVariants() {
  try {
    const r = await api.variantList()
    customVariants.value = r.custom
  } catch {}
}

async function compileCustom() {
  customCompiling.value = true
  customCompileMsg.value = ''
  try {
    const r = await api.variantCompile(
      customFormula.value,
      customName.value,
      customBailoutDirty.value ? customBailout.value : undefined,
    )
    if (r.ok && r.variantId) {
      await loadCustomVariants()
      variant.value        = r.variantId
      showCustomPanel.value = false
      customCompileMsg.value = ''
    } else {
      customCompileMsg.value = r.error ?? 'compile failed'
    }
  } catch (e: any) {
    customCompileMsg.value = e?.message ?? 'error'
  } finally {
    customCompiling.value = false
  }
}

async function deleteCustom(variantId: string) {
  await api.variantDelete(variantId)
  await loadCustomVariants()
  if (variant.value === variantId) variant.value = 'mandelbrot'
}

function onVariantSelect(val: string) {
  if (val === '__new_custom__') {
    showCustomPanel.value = true
    // keep previous variant active until compile succeeds
  } else {
    variant.value = val
    showCustomPanel.value = false
  }
}

const transitionOn = ref(false)
const thetaDeg     = ref(0)
const transitionFrom = ref<string>('mandelbrot')
const transitionTo   = ref<string>('burning_ship')
const AXIS_TRANSITION_VARIANTS = VARIANTS.slice(0, 10)

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
const engineMode = ref<'auto' | 'openmp' | 'avx2' | 'avx512' | 'cuda' | 'hybrid'>('auto')
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
  loadCustomVariants()

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

const pngPresetKey = ref('fhd')
const videoPresetKey = ref('fhd')

const pngPreset = computed(() =>
  EXPORT_PRESETS.find(p => p.key === pngPresetKey.value) ?? EXPORT_PRESETS[0]
)
const videoPreset = computed(() =>
  EXPORT_PRESETS.find(p => p.key === videoPresetKey.value) ?? EXPORT_PRESETS[0]
)

async function exportPng() {
  try {
    const resp = await api.mapRender({
      taskType:   'still_export',
      centerRe:   centerRe.value,
      centerIm:   centerIm.value,
      scale:      scale.value,
      width:      pngPreset.value.width,
      height:     pngPreset.value.height,
      iterations: iterations.value,
      variant:    variant.value,
      metric:     metric.value,
      colorMap:   colorMap.value,
      smooth:     smooth.value,
      julia:      juliaOn.value,
      juliaRe:    juliaRe.value,
      juliaIm:    juliaIm.value,
      transitionTheta: transitionOn.value ? thetaDeg.value * Math.PI / 180 : undefined,
      transitionFrom:  transitionOn.value ? transitionFrom.value : undefined,
      transitionTo:    transitionOn.value ? transitionTo.value : undefined,
    }) as any
    window.open(api.artifactDownloadUrl(resp.artifactId), '_blank')
  } catch (e: any) {
    console.error('export PNG failed:', e?.data?.error ?? e)
  }
}

// ── Unified video export (ln-map + final frame + video in one dialog) ─────────
const exportModalOpen = ref(false)
const exportDepth     = ref(20)
const exportFps       = ref(30)
const exportSecondsPerOctave = ref(0.4)
const exportQualityPreset = ref<'draft' | 'balanced' | 'high' | 'full'>('balanced')
const exportW         = ref(1920)
const exportH         = ref(1080)
const exportBusy      = ref(false)
const exportPreviewBusy = ref(false)
const exportStatus    = ref('')
const exportPreviewStatus = ref('')
const exportResult    = ref<VideoExportResponse | null>(null)
const exportPreviewResult = ref<VideoPreviewResponse | null>(null)
const exportJobId     = ref('')
const exportProgress  = ref<RunProgress>({})
const exportDepthDirty = ref(false)

const exportEstimatedDuration = computed(() =>
  Math.max(0, exportDepth.value) * Math.max(0, exportSecondsPerOctave.value)
)
const exportEstimatedFrames = computed(() =>
  Math.max(2, Math.round(exportEstimatedDuration.value * Math.max(1, exportFps.value)))
)
const visiblePreview = computed(() => exportPreviewResult.value ?? exportResult.value)
const exportProgressDetail = computed(() => {
  const p = exportProgress.value
  if (!p.stage) return ''
  if (p.stage === 'ln_map') {
    return `ln-map ${p.current || 0}/${p.total || 0} rows · octave ${(p.depthOctave || 0).toFixed(2)}/${(p.totalDepthOctaves || 0).toFixed(2)}`
  }
  if (p.stage === 'video_warp_encode') {
    return `encode ${p.current || 0}/${p.total || 0} frames`
  }
  if (p.stage === 'final_frame') return `final frame ${p.current || 0}/${p.total || 1}`
  return p.stage
})
const exportMemoryEstimateMiB = computed(() => {
  const fullWidth = Math.ceil(Math.sqrt(exportW.value * exportW.value + exportH.value * exportH.value) * Math.PI)
  const scaleByPreset: Record<string, number> = { draft: 0.35, balanced: 0.55, high: 0.75, full: 1.0 }
  const actualWidth = Math.ceil(fullWidth * (scaleByPreset[exportQualityPreset.value] ?? 0.55))
  const heightT = Math.ceil((2 + exportDepth.value) * Math.LN2 / (Math.PI * 2) * actualWidth)
  const pixels = exportW.value * exportH.value
  const bytes = actualWidth * heightT * 3 + pixels * (3 * 4 + 4 * 5 + 1)
  return bytes / 1024 / 1024
})

watch(videoPreset, p => {
  exportW.value = p.width
  exportH.value = p.height
  if (exportModalOpen.value && !exportDepthDirty.value) syncExportDepthToCurrentView()
}, { immediate: true })

watch([exportW, exportH], () => {
  if (exportModalOpen.value && !exportDepthDirty.value) syncExportDepthToCurrentView()
  if (exportModalOpen.value) clearExportPreview()
})

function defaultExportDepthForView() {
  const aspect = Math.max(1e-9, exportW.value / Math.max(1, exportH.value))
  const rMax = Math.sqrt(aspect * aspect + 1)
  const kTopStart = Math.log(4) - Math.log(rMax)
  const kTopEnd = Math.log(Math.max(scale.value, 1e-300) * 0.5)
  const depth = (kTopStart - kTopEnd) / Math.LN2
  if (!Number.isFinite(depth)) return 20
  return Math.min(120, Math.max(0.05, depth))
}

function syncExportDepthToCurrentView() {
  exportDepth.value = Number(defaultExportDepthForView().toFixed(2))
}

function clearExportPreview() {
  exportPreviewStatus.value = ''
  exportPreviewResult.value = null
}

function progressRatio(stage: string) {
  if (exportProgress.value.stage !== stage) {
    if (stage === 'final_frame' && ['ln_map', 'video_warp_encode'].includes(exportProgress.value.stage || '')) return 1
    if (stage === 'ln_map' && exportProgress.value.stage === 'video_warp_encode') return 1
    return 0
  }
  const total = Math.max(1, exportProgress.value.total || 1)
  return Math.max(0, Math.min(1, (exportProgress.value.current || 0) / total))
}

function onExportDepthInput() {
  exportDepthDirty.value = true
  clearExportPreview()
}

function openExportModal() {
  exportDepthDirty.value = false
  syncExportDepthToCurrentView()
  exportModalOpen.value = true
  exportStatus.value    = ''
  exportPreviewStatus.value = ''
  exportResult.value    = null
  exportPreviewResult.value = null
  exportJobId.value = ''
  exportProgress.value = {}
}

function videoRequestBase() {
  return {
    centerRe:     centerRe.value,
    centerIm:     centerIm.value,
    julia:        juliaOn.value,
    juliaRe:      juliaRe.value,
    juliaIm:      juliaIm.value,
    variant:      variant.value,
    colorMap:     colorMap.value,
    iterations:   Math.max(iterations.value, 2048),
    depthOctaves: exportDepth.value,
    fps:          exportFps.value,
    secondsPerOctave: exportSecondsPerOctave.value,
    targetScale:  !exportDepthDirty.value && exportDepth.value > 0.05 ? scale.value : undefined,
    qualityPreset: exportQualityPreset.value,
    background: true,
    width:        exportW.value,
    height:       exportH.value,
  }
}

function previewSizeForExport() {
  const maxSide = 720
  const longSide = Math.max(exportW.value, exportH.value, 1)
  const ratio = Math.min(1, maxSide / longSide)
  return {
    previewWidth: Math.max(64, Math.round(exportW.value * ratio)),
    previewHeight: Math.max(64, Math.round(exportH.value * ratio)),
  }
}

async function runPreview() {
  exportPreviewBusy.value = true
  exportPreviewStatus.value = 'previewing…'
  exportPreviewResult.value = null
  try {
    const resp = await api.videoPreview({
      ...videoRequestBase(),
      ...previewSizeForExport(),
    })
    exportPreviewResult.value = resp
    exportPreviewStatus.value = `${resp.width}×${resp.height} · depth ${resp.depthOctaves.toFixed(2)} · ${resp.generatedMs.toFixed(0)} ms`
  } catch (e: any) {
    exportPreviewStatus.value = 'failed: ' + (e?.data?.error || e?.message || e)
  } finally {
    exportPreviewBusy.value = false
  }
}

async function runExport() {
  exportBusy.value   = true
  exportStatus.value = 'queued…'
  exportResult.value = null
  exportProgress.value = {}
  try {
    const resp = await api.videoExport(videoRequestBase())
    exportJobId.value = resp.runId
    exportStatus.value = `${resp.runId} · ${resp.frameCount} frames · ${resp.durationSec.toFixed(2)}s`
    await pollVideoExport(resp)
  } catch (e: any) {
    exportStatus.value = 'failed: ' + (e?.data?.error || e?.message || e)
  } finally {
    exportBusy.value = false
  }
}

function artifactByName(status: RunStatusResponse, name: string) {
  return status.artifacts.find(a => a.name === name)
}

async function pollVideoExport(initial: VideoExportResponse) {
  for (;;) {
    await new Promise(resolve => setTimeout(resolve, 700))
    const status = await api.runStatus(initial.runId)
    exportProgress.value = status.progress || {}
    if (status.status === 'failed') {
      const msg = status.progress?.errorMessage || 'video export failed'
      exportStatus.value = `failed: ${status.progress?.failedStage || status.progress?.stage || 'video_export'} · ${msg}`
      return
    }
    if (status.status === 'cancelled') {
      exportStatus.value = 'cancelled'
      return
    }
    if (status.status !== 'completed') continue

    const video = artifactByName(status, 'zoom.mp4') || status.artifacts.find(a => a.kind === 'video')
    const lnMap = artifactByName(status, 'ln_map.png')
    const finalFrame = artifactByName(status, 'final_frame.png')
    const startFrame = artifactByName(status, 'start_frame.png')
    const endFrame = artifactByName(status, 'end_frame.png')
    exportResult.value = {
      ...initial,
      status: 'completed',
      videoArtifactId: video?.artifactId,
      videoUrl: video?.contentUrl,
      videoDownloadUrl: video?.downloadUrl,
      lnMapArtifactId: lnMap?.artifactId,
      lnMapDownloadUrl: lnMap?.downloadUrl,
      finalFrameArtifactId: finalFrame?.artifactId,
      finalFrameDownloadUrl: finalFrame?.downloadUrl,
      startFrameArtifactId: startFrame?.artifactId,
      startFrameUrl: startFrame?.contentUrl,
      startFrameDownloadUrl: startFrame?.downloadUrl,
      endFrameArtifactId: endFrame?.artifactId,
      endFrameUrl: endFrame?.contentUrl,
      endFrameDownloadUrl: endFrame?.downloadUrl,
      generatedMs: status.finishedAt && status.startedAt ? status.finishedAt - status.startedAt : undefined,
    }
    exportStatus.value = `completed · ${initial.frameCount} frames`
    return
  }
}
</script>

<template>
  <div class="map-view">

    <!-- ── Controls bar ──────────────────────────────────────────────────── -->
    <div class="controls">
      <div class="group">
        <label>{{ t('variant') }}</label>
        <select :value="variant" @change="onVariantSelect(($event.target as HTMLSelectElement).value)" :disabled="transitionOn">
          <option v-for="v in VARIANTS" :key="v" :value="v">{{ VARIANT_LABELS[v][lang] }}</option>
          <template v-if="customVariants.length">
            <option disabled>──────</option>
            <option v-for="cv in customVariants" :key="cv.variantId" :value="cv.variantId">
              ✦ {{ cv.name }}
            </option>
          </template>
          <option value="__new_custom__">{{ t('custom_new') }}</option>
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
        <div v-if="transitionOn" class="theta-row">
          <select v-model="transitionFrom">
            <option v-for="v in AXIS_TRANSITION_VARIANTS" :key="'from-' + v" :value="v">{{ VARIANT_LABELS[v][lang] }}</option>
          </select>
          <span class="num">→</span>
          <select v-model="transitionTo">
            <option v-for="v in AXIS_TRANSITION_VARIANTS" :key="'to-' + v" :value="v">{{ VARIANT_LABELS[v][lang] }}</option>
          </select>
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
          <option value="avx2">avx2</option>
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

      <div class="group export-preset-group">
        <label>{{ lang === 'en' ? 'Wallpaper' : '壁纸尺寸' }}</label>
        <select v-model="pngPresetKey">
          <option v-for="p in EXPORT_PRESETS" :key="p.key" :value="p.key">
            {{ p.label[lang] }} · {{ p.width }}×{{ p.height }}
          </option>
        </select>
      </div>

      <button @click="resetView" :title="t('reset')">⌂ {{ t('reset') }}</button>
      <button @click="exportPng">{{ t('export_png') }}</button>
      <button @click="openExportModal">{{ t('export_video') }}</button>
    </div>

    <!-- ── Custom formula editor ─────────────────────────────────────────── -->
    <div v-if="showCustomPanel" class="custom-panel">
      <div class="custom-header mono">
        <span>{{ t('custom_new') }}</span>
        <button class="custom-close" @click="showCustomPanel = false">✕</button>
      </div>
      <div class="custom-hint mono">{{ t('custom_hint') }}</div>
      <div class="custom-row">
        <label>{{ t('custom_formula') }}</label>
        <input v-model="customFormula" class="formula-input mono" placeholder="z^2 + c" spellcheck="false" />
        <label style="margin-left:12px">{{ t('custom_name') }}</label>
        <input v-model="customName" placeholder="my_variant" style="width:120px" />
        <label style="margin-left:12px">{{ t('custom_bailout') }}</label>
        <input type="number" v-model.number="customBailout" min="0.1" max="1000000" step="0.001" style="width:80px" @input="customBailoutDirty = true" />
        <button class="btn-compile" @click="compileCustom" :disabled="customCompiling">
          {{ customCompiling ? t('loading') : t('custom_compile') }}
        </button>
      </div>
      <div v-if="customCompileMsg" class="custom-msg mono">{{ customCompileMsg }}</div>
      <div v-if="customVariants.length" class="custom-list">
        <div v-for="cv in customVariants" :key="cv.variantId" class="custom-item mono">
          <span class="cv-name">{{ cv.name }}</span>
          <span class="cv-formula">{{ cv.formula }}</span>
          <button class="cv-use" @click="variant = cv.variantId; showCustomPanel = false">use</button>
          <button class="cv-del" @click="deleteCustom(cv.variantId)">{{ t('custom_delete') }}</button>
        </div>
      </div>
    </div>

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
          :transitionFrom="transitionFrom" :transitionTo="transitionTo"
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
              <span class="pane-title">{{ t('julia_left') }}: {{ (VARIANT_LABELS as any)[variant]?.[lang] ?? variant }}</span>
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
                :transitionFrom="transitionFrom" :transitionTo="transitionTo"
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

    <!-- ── Unified video export modal ───────────────────────────────────── -->
    <Teleport to="body">
      <div v-if="exportModalOpen" class="modal-backdrop" @click.self="exportModalOpen = false">
        <div class="modal">
          <div class="modal-title">
            {{ juliaOn ? t('export_julia_video') : t('export_video') }}
          </div>
          <div v-if="juliaOn" class="mrow source mono" style="margin-bottom:6px">
            Julia c: {{ juliaRe.toPrecision(8) }} + {{ juliaIm.toPrecision(8) }}i
          </div>
          <div class="modal-body">
            <div class="mrow">
              <label>{{ t('video_depth') }}</label>
              <input type="number" v-model.number="exportDepth" min="0.05" max="120" step="0.05" @input="onExportDepthInput" />
            </div>
            <div class="mrow">
              <label>{{ t('video_fps') }}</label>
              <input type="number" v-model.number="exportFps" min="1" max="120" step="1" />
            </div>
            <div class="mrow">
              <label>{{ t('video_seconds_per_octave') }}</label>
              <input type="number" v-model.number="exportSecondsPerOctave" min="0.05" max="60" step="0.05" />
            </div>
            <div class="mrow estimate">
              <label>{{ t('video_estimate') }}</label>
              <span class="mono">{{ exportEstimatedDuration.toFixed(2) }}s · {{ exportEstimatedFrames }} frames · {{ exportMemoryEstimateMiB.toFixed(0) }} MiB</span>
            </div>
            <div class="mrow">
              <label>Quality</label>
              <select v-model="exportQualityPreset">
                <option value="draft">draft</option>
                <option value="balanced">balanced</option>
                <option value="high">high</option>
                <option value="full">full</option>
              </select>
            </div>
            <div class="mrow">
              <label>{{ lang === 'en' ? 'Preset' : '预设' }}</label>
              <select v-model="videoPresetKey">
                <option v-for="p in EXPORT_PRESETS" :key="p.key" :value="p.key">
                  {{ p.label[lang] }} · {{ p.width }}×{{ p.height }}
                </option>
              </select>
            </div>
            <div class="mrow">
              <label>{{ t('video_width') }}</label>
              <input type="number" v-model.number="exportW" min="128" max="7680" step="64" />
            </div>
            <div class="mrow">
              <label>{{ t('video_height') }}</label>
              <input type="number" v-model.number="exportH" min="128" max="4320" step="64" />
            </div>
          </div>
          <div class="modal-footer">
            <button @click="exportModalOpen = false" class="btn-cancel">{{ t('video_cancel') }}</button>
            <button @click="runPreview" :disabled="exportBusy || exportPreviewBusy" class="btn-preview">
              {{ exportPreviewBusy ? t('loading') : t('video_preview') }}
            </button>
            <button @click="runExport" :disabled="exportBusy || exportPreviewBusy" class="btn-go">
              {{ exportBusy ? t('loading') : t('video_render') }}
            </button>
          </div>
          <div v-if="exportPreviewStatus" class="modal-status mono">{{ exportPreviewStatus }}</div>
          <div v-if="exportStatus" class="modal-status mono">{{ exportStatus }}</div>
          <div v-if="exportBusy || exportJobId" class="progress-stack">
            <div class="progress-row">
              <span>final</span>
              <progress :value="progressRatio('final_frame')" max="1"></progress>
            </div>
            <div class="progress-row">
              <span>ln-map</span>
              <progress :value="progressRatio('ln_map')" max="1"></progress>
            </div>
            <div class="progress-row">
              <span>encode</span>
              <progress :value="progressRatio('video_warp_encode')" max="1"></progress>
            </div>
          </div>
          <div v-if="exportProgressDetail" class="modal-status mono">{{ exportProgressDetail }}</div>
          <div v-if="visiblePreview" class="modal-body" style="gap:6px">
            <div v-if="visiblePreview.startFrameUrl || visiblePreview.endFrameUrl" class="preview-grid">
              <a v-if="visiblePreview.startFrameUrl"
                 :href="api.baseUrl + (visiblePreview.startFrameDownloadUrl || visiblePreview.startFrameUrl)"
                 class="preview-item" download>
                <span>{{ t('video_start_frame') }}</span>
                <img :src="api.baseUrl + visiblePreview.startFrameUrl" alt="" />
              </a>
              <a v-if="visiblePreview.endFrameUrl"
                 :href="api.baseUrl + (visiblePreview.endFrameDownloadUrl || visiblePreview.endFrameUrl)"
                 class="preview-item" download>
                <span>{{ t('video_end_frame') }}</span>
                <img :src="api.baseUrl + visiblePreview.endFrameUrl" alt="" />
              </a>
            </div>
          </div>
          <div v-if="exportResult" class="modal-body" style="gap:6px">
            <a v-if="exportResult.videoDownloadUrl" :href="api.baseUrl + exportResult.videoDownloadUrl" class="dl-link" download>↓ {{ t('video_download') }}</a>
            <a v-if="exportResult.lnMapDownloadUrl" :href="api.baseUrl + exportResult.lnMapDownloadUrl" class="dl-link" download>↓ ln-map PNG</a>
            <a v-if="exportResult.finalFrameDownloadUrl" :href="api.baseUrl + exportResult.finalFrameDownloadUrl" class="dl-link" download>↓ {{ t('export_png') }}</a>
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
.group.export-preset-group { min-width: 190px; }

.theta-row {
  display: flex;
  align-items: center;
  gap: 8px;
}
.theta-row input[type="range"] { flex: 1; }

.spacer { flex: 1; }

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
  width: min(420px, calc(100vw - 32px));
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
.mrow.estimate span {
  font-size: 10px;
  color: var(--text-dim);
  white-space: nowrap;
}
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
.btn-preview {
  background: transparent;
  border: 1px solid var(--accent);
  color: var(--accent);
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
.btn-go:disabled,
.btn-preview:disabled { opacity: 0.5; cursor: default; }
.modal-status { font-size: 10px; color: var(--text-dim); }
.progress-stack {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.progress-row {
  display: grid;
  grid-template-columns: 58px 1fr;
  align-items: center;
  gap: 8px;
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text-dim);
}
.progress-row progress {
  width: 100%;
  height: 7px;
  accent-color: var(--accent);
}
.preview-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
  margin-bottom: 4px;
}
.preview-item {
  display: flex;
  flex-direction: column;
  gap: 5px;
  color: var(--text-dim);
  font-family: var(--mono);
  font-size: 10px;
  text-decoration: none;
}
.preview-item img {
  width: 100%;
  height: 120px;
  object-fit: contain;
  border: 1px solid var(--rule);
  background: #000;
}
.preview-item:hover span { color: var(--accent); }
.dl-link {
  display: block;
  padding: 5px 0;
  font-family: var(--mono);
  font-size: 11px;
  color: var(--accent);
  text-decoration: none;
}
.dl-link:hover { text-decoration: underline; }

/* ── Custom formula panel ── */
.custom-panel {
  background: var(--panel);
  border-bottom: 1px solid var(--rule);
  padding: 10px 14px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  flex-shrink: 0;
}
.custom-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 11px;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.custom-close {
  background: none;
  border: none;
  color: var(--text-dim);
  cursor: pointer;
  font-size: 14px;
  padding: 0 4px;
}
.custom-close:hover { color: var(--text); }
.custom-hint {
  font-size: 10px;
  color: var(--text-dim);
  line-height: 1.5;
}
.custom-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.custom-row label {
  font-size: 11px;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  white-space: nowrap;
}
.formula-input {
  flex: 1;
  min-width: 180px;
}
.btn-compile {
  background: var(--accent);
  color: #000;
  border: none;
  padding: 5px 14px;
  font-family: var(--mono);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}
.btn-compile:disabled { opacity: 0.5; cursor: default; }
.custom-msg {
  font-size: 10px;
  color: var(--bad);
  white-space: pre-wrap;
}
.custom-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
  border-top: 1px solid var(--rule);
  padding-top: 6px;
}
.custom-item {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 11px;
}
.cv-name { color: var(--accent); min-width: 100px; }
.cv-formula { color: var(--text-dim); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cv-use, .cv-del {
  background: none;
  border: 1px solid var(--rule);
  color: var(--text-dim);
  font-family: var(--mono);
  font-size: 10px;
  padding: 2px 8px;
  cursor: pointer;
}
.cv-use:hover { color: var(--accent); border-color: var(--accent); }
.cv-del:hover { color: var(--bad);    border-color: var(--bad); }
</style>
