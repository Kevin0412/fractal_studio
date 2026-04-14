<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import ThreeDViewer from '../components/ThreeDViewer.vue'
import { api, VARIANTS, VARIANT_LABELS, type Variant, type HsStage, type TransitionVoxelResponse, type HsFieldResponse, type MeshResponse } from '../api'
import { t, lang } from '../i18n'

type Mode = 'hs' | 'transition'
type TxRenderMode = 'mesh' | 'voxel'

const mode    = ref<Mode>('hs')
const txRMode = ref<TxRenderMode>('voxel')  // default to voxel for M↔B

// HS mode params
const hsMetric  = ref<HsStage>('min_abs')
const hsVariant = ref<Variant>('mandelbrot')
const hsRes     = ref(256)
const hsIter    = ref(512)
const hsCenterRe = ref(-0.75)
const hsCenterIm = ref(0.0)
const hsScale    = ref(3.0)

// HS z-scale: sign (convex=+1 / concave=-1) + log exponent
const hsZSign = ref<1 | -1>(1)           // 1 = convex (height up), -1 = concave (inverted)
const hsZExp  = ref(-1.0)                 // log10 magnitude: range -15..0 → 10^exp
const hsZScale = computed(() => hsZSign.value * Math.pow(10, hsZExp.value))

// Transition mode params
const txRes   = ref(64)
const txTheta = ref(0.0)
const txIso   = ref(0.48)
const txIter  = ref(128)
const txCenterRe = ref(-0.75)
const txCenterIm = ref(0.0)
const txScale    = ref(3.0)

// State
const hsFieldData  = ref<HsFieldResponse | null>(null)
const glbUrl       = ref<string | null>(null)
const voxelData    = ref<TransitionVoxelResponse | null>(null)
const stlUrl       = ref<string | null>(null)
const loading      = ref(false)
const stlLoading   = ref(false)
const info         = ref('')
const error        = ref('')

const HS_METRICS: HsStage[] = ['min_abs', 'max_abs', 'envelope', 'min_pairwise_dist']

const HS_METRIC_LABELS: Record<HsStage, { en: string; zh: string }> = {
  min_abs:            { en: 'Min abs(z) (HS-base)',       zh: '最小 abs(z)（HS 基础）' },
  max_abs:            { en: 'Max abs(z) (envelope hi)',   zh: '最大 abs(z)（包络高）' },
  envelope:           { en: 'Envelope',                   zh: '包络' },
  min_pairwise_dist:  { en: 'Min pairwise (recurrence)',  zh: '最小轨道距（递归）' },
}

// ── HS field auto-compute (debounced) ─────────────────────────────────────────
let debounceTimer: ReturnType<typeof setTimeout> | null = null

function scheduleHsCompute() {
  if (mode.value !== 'hs') return
  if (debounceTimer) clearTimeout(debounceTimer)
  debounceTimer = setTimeout(computeHsField, 500)
}

watch([hsMetric, hsVariant, hsRes, hsIter, hsCenterRe, hsCenterIm, hsScale], scheduleHsCompute)

// ── HS field fetch ────────────────────────────────────────────────────────────

async function computeHsField() {
  loading.value = true
  error.value   = ''
  info.value    = 'computing HS field…'
  hsFieldData.value = null
  stlUrl.value  = null
  try {
    const r = await api.hsField({
      centerRe:   hsCenterRe.value,
      centerIm:   hsCenterIm.value,
      scale:      hsScale.value,
      resolution: hsRes.value,
      metric:     hsMetric.value,
      variant:    hsVariant.value,
      iterations: hsIter.value,
    })
    hsFieldData.value = r
    info.value = `${r.width}×${r.height} field · range [${r.fieldMin.toFixed(3)}, ${r.fieldMax.toFixed(3)}] · ${r.generatedMs.toFixed(0)}ms`
  } catch (e: any) {
    error.value = e?.message ?? String(e)
    info.value  = ''
  } finally {
    loading.value = false
  }
}

// ── HS STL export ─────────────────────────────────────────────────────────────

async function exportHsStl() {
  stlLoading.value = true
  error.value = ''
  stlUrl.value = null
  try {
    const r: MeshResponse = await api.hsMesh({
      centerRe:   hsCenterRe.value,
      centerIm:   hsCenterIm.value,
      scale:      hsScale.value,
      resolution: hsRes.value,
      metric:     hsMetric.value,
      variant:    hsVariant.value,
      iterations: hsIter.value,
    })
    stlUrl.value = api.artifactDownloadUrl(r.stlArtifactId)
  } catch (e: any) {
    error.value = e?.message ?? String(e)
  } finally {
    stlLoading.value = false
  }
}

// ── Transition compute ────────────────────────────────────────────────────────

async function computeTransitionVoxels() {
  loading.value   = true
  error.value     = ''
  info.value      = 'computing voxel field…'
  glbUrl.value    = null
  voxelData.value = null
  stlUrl.value    = null
  try {
    const r = await api.transitionVoxels({
      resolution: txRes.value,
      iso:        txIso.value,
      iterations: txIter.value,
    })
    voxelData.value = r
    info.value = `${r.faceCount.toLocaleString()} faces · ${r.resolution}³ grid · ${r.generatedMs.toFixed(0)}ms`
  } catch (e: any) {
    error.value = e?.message ?? String(e)
    info.value  = ''
  } finally {
    loading.value = false
  }
}

async function computeTransitionMesh() {
  loading.value   = true
  error.value     = ''
  info.value      = 'marching cubes…'
  glbUrl.value    = null
  voxelData.value = null
  stlUrl.value    = null
  try {
    const r = await api.transitionMesh({
      centerRe:   txCenterRe.value,
      centerIm:   txCenterIm.value,
      scale:      txScale.value,
      resolution: txRes.value,
      theta:      txTheta.value,
      iso:        txIso.value,
      iterations: txIter.value,
    })
    glbUrl.value = api.artifactContentUrl(r.glbArtifactId)
    stlUrl.value = api.artifactDownloadUrl(r.stlArtifactId)
    info.value = `${r.vertexCount} verts · ${r.triangleCount} tri · field ${(r.fieldMs ?? 0).toFixed(0)}ms · MC ${(r.mcMs ?? 0).toFixed(0)}ms`
  } catch (e: any) {
    error.value = e?.message ?? String(e)
    info.value  = ''
  } finally {
    loading.value = false
  }
}

function compute() {
  if (mode.value === 'hs') {
    computeHsField()
  } else if (txRMode.value === 'voxel') {
    computeTransitionVoxels()
  } else {
    computeTransitionMesh()
  }
}

// When switching to HS mode, auto-compute if no field yet
watch(mode, (m) => {
  if (m === 'hs' && !hsFieldData.value && !loading.value) {
    computeHsField()
  }
})
</script>

<template>
  <div class="three-view">
    <!-- Left control strip -->
    <aside class="controls">
      <!-- Mode toggle -->
      <div class="mode-row">
        <button :class="['mode-btn', mode === 'hs' ? 'active' : '']" @click="mode = 'hs'">{{ t('three_mode_hs') }}</button>
        <button :class="['mode-btn', mode === 'transition' ? 'active' : '']" @click="mode = 'transition'">{{ t('three_mode_tx') }}</button>
      </div>

      <!-- HS params -->
      <template v-if="mode === 'hs'">
        <div class="group">
          <label>{{ t('three_metric') }}</label>
          <select v-model="hsMetric">
            <option v-for="m in HS_METRICS" :key="m" :value="m">{{ HS_METRIC_LABELS[m][lang] }}</option>
          </select>
        </div>
        <div class="group">
          <label>{{ t('variant') }}</label>
          <select v-model="hsVariant">
            <option v-for="v in VARIANTS" :key="v" :value="v">{{ VARIANT_LABELS[v][lang] }}</option>
          </select>
        </div>
        <div class="group">
          <label>{{ t('three_resolution') }}</label>
          <input type="number" v-model.number="hsRes" min="32" max="4096" step="64" />
        </div>
        <div class="group">
          <label>{{ t('iterations') }}</label>
          <input type="number" v-model.number="hsIter" min="64" max="10000" step="128" />
        </div>
        <div class="group">
          <label>{{ t('three_center_re') }}</label>
          <input type="number" v-model.number="hsCenterRe" step="0.01" />
        </div>
        <div class="group">
          <label>{{ t('three_center_im') }}</label>
          <input type="number" v-model.number="hsCenterIm" step="0.01" />
        </div>
        <div class="group">
          <label>{{ t('three_scale') }}</label>
          <input type="number" v-model.number="hsScale" min="0.0001" step="0.1" />
        </div>

        <!-- Z-scale controls -->
        <div class="rule"></div>
        <div class="group">
          <label>Z-SCALE</label>
          <div class="zscale-row">
            <label class="inline-label">
              <input type="checkbox" :checked="hsZSign === -1" @change="hsZSign = (hsZSign === 1 ? -1 : 1)" />
              {{ lang === 'en' ? 'Concave' : '凹面' }}
            </label>
            <span class="num">{{ (hsZScale >= 0 ? '' : '−') + Math.pow(10, hsZExp).toExponential(1) }}</span>
          </div>
          <input type="range" min="-15" max="0" step="0.1" v-model.number="hsZExp" />
          <span class="num dim">10<sup>{{ hsZExp.toFixed(1) }}</sup></span>
        </div>

        <!-- STL export -->
        <button class="stl-btn" @click="exportHsStl" :disabled="stlLoading">
          {{ stlLoading ? (lang === 'en' ? 'Exporting…' : '导出中…') : (lang === 'en' ? 'Export STL' : '导出 STL') }}
        </button>
        <a v-if="stlUrl" :href="stlUrl" download class="stl-link mono">
          {{ lang === 'en' ? '⬇ download STL' : '⬇ 下载 STL' }}
        </a>
      </template>

      <!-- Transition params -->
      <template v-else>
        <!-- Render-mode toggle: voxel (Minecraft) vs smooth mesh -->
        <div class="mode-row" style="margin-bottom:10px">
          <button :class="['mode-btn', txRMode === 'voxel' ? 'active' : '']" @click="txRMode = 'voxel'">⬜ VOXEL</button>
          <button :class="['mode-btn', txRMode === 'mesh'  ? 'active' : '']" @click="txRMode = 'mesh'">◈ MESH</button>
        </div>

        <div class="group">
          <label>
            {{ t('three_iso') }}
            <span class="dim"> ({{ txRMode === 'voxel' ? 'core depth' : 'surface' }})</span>
          </label>
          <input type="range" min="0.05" max="0.48" step="0.01" v-model.number="txIso" />
          <span class="num">{{ txIso.toFixed(2) }}{{ txRMode === 'voxel' ? (txIso >= 0.45 ? ' (all)' : '') : '' }}</span>
        </div>
        <div class="group">
          <label>{{ t('three_resolution') }}</label>
          <input v-if="txRMode === 'voxel'"
                 type="number" v-model.number="txRes" min="16" step="16" />
          <input v-else
                 type="number" v-model.number="txRes" min="32" max="1024" step="32" />
          <span class="num dim">{{ txRMode === 'voxel' ? txRes + '³ vox' : txRes + '³ MC' }}</span>
        </div>
        <div class="group">
          <label>{{ t('iterations') }}</label>
          <input type="number" v-model.number="txIter" min="32" max="2000" step="32" />
        </div>

        <!-- STL export for M↔B mesh mode -->
        <a v-if="stlUrl" :href="stlUrl" download class="stl-link mono">
          {{ lang === 'en' ? '⬇ download STL' : '⬇ 下载 STL' }}
        </a>
      </template>

      <button class="compute-btn" @click="compute" :disabled="loading">
        {{ loading ? t('three_computing') : t('three_compute') }}
      </button>

      <div v-if="info"  class="info mono">{{ info }}</div>
      <div v-if="error" class="err mono">{{ error }}</div>
    </aside>

    <!-- Viewer canvas -->
    <div class="viewer">
      <ThreeDViewer
        :glbUrl="glbUrl"
        :voxelData="voxelData"
        :hsFieldData="hsFieldData"
        :zScale="hsZScale"
        :viewMode="mode"
        :loading="loading" />
    </div>
  </div>
</template>

<style scoped>
.three-view {
  display: grid;
  grid-template-columns: 240px 1fr;
  height: 100%;
  overflow: hidden;
}

.controls {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 14px 12px;
  border-right: 1px solid var(--rule);
  background: var(--panel);
  overflow-y: auto;
}

.mode-row {
  display: flex;
  gap: 6px;
  margin-bottom: 6px;
}

.mode-btn {
  flex: 1;
  padding: 5px;
  background: var(--rule);
  color: var(--text-dim);
  border: 1px solid transparent;
  font-family: var(--mono);
  font-size: 11px;
  cursor: pointer;
  transition: border-color 0.1s, color 0.1s;
}

.mode-btn.active {
  border-color: var(--accent);
  color: var(--accent);
}

.group {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.group label {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--text-dim);
}

.group select,
.group input[type="number"] {
  width: 100%;
}

.group input[type="range"] { width: 100%; }

.zscale-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 6px;
}

.inline-label {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  color: var(--text-dim);
  text-transform: none;
  letter-spacing: normal;
  cursor: pointer;
}

.num {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text-dim);
  align-self: flex-end;
}

.dim { color: var(--text-faint); }

.compute-btn {
  margin-top: 8px;
  padding: 8px;
  background: var(--accent);
  color: #000;
  font-family: var(--mono);
  font-size: 12px;
  border: none;
  cursor: pointer;
  font-weight: 600;
  letter-spacing: 0.05em;
}

.compute-btn:disabled {
  opacity: 0.5;
  cursor: default;
}

.stl-btn {
  padding: 6px 8px;
  background: transparent;
  color: var(--text-dim);
  border: 1px solid var(--rule);
  font-family: var(--mono);
  font-size: 11px;
  cursor: pointer;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.stl-btn:hover:not(:disabled) { border-color: var(--accent-edge); color: var(--accent); }
.stl-btn:disabled { opacity: 0.5; cursor: default; }

.stl-link {
  font-size: 11px;
  color: var(--accent);
  text-decoration: none;
  padding: 3px 0;
}

.info, .err {
  font-size: 10px;
  line-height: 1.4;
  color: var(--text-dim);
  word-break: break-all;
  margin-top: 4px;
}

.err { color: var(--bad); }

.rule {
  border-top: 1px solid var(--rule);
  margin: 4px 0;
}

.viewer {
  position: relative;
  min-height: 0;
  height: 100%;
}
</style>
