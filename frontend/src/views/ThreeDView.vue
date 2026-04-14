<script setup lang="ts">
import { ref } from 'vue'
import ThreeDViewer from '../components/ThreeDViewer.vue'
import { api, VARIANTS, VARIANT_LABELS, type Variant, type HsStage, type TransitionVoxelResponse } from '../api'
import { t, lang } from '../i18n'

type Mode = 'hs' | 'transition'
type TxRenderMode = 'mesh' | 'voxel'

const mode    = ref<Mode>('hs')
const txRMode = ref<TxRenderMode>('voxel')  // default to voxel for M↔B

// HS mode params
const hsMetric  = ref<HsStage>('min_abs')
const hsVariant = ref<Variant>('mandelbrot')
const hsRes     = ref(512)
const hsIter    = ref(512)
const hsCenterRe = ref(-0.75)
const hsCenterIm = ref(0.0)
const hsScale    = ref(3.0)

// Transition mode params
const txRes   = ref(64)    // good default for voxel: fast, ~50k exposed faces
const txTheta = ref(0.0)
const txIso   = ref(0.48)  // 0.48 ≈ all inside voxels; lower = deep-core only
const txIter  = ref(128)
const txCenterRe = ref(-0.75)
const txCenterIm = ref(0.0)
const txScale    = ref(3.0)

const glbUrl    = ref<string | null>(null)
const voxelData = ref<TransitionVoxelResponse | null>(null)
const loading   = ref(false)
const info      = ref('')
const error     = ref('')

const HS_METRICS: HsStage[] = ['min_abs', 'max_abs', 'envelope', 'min_pairwise_dist']

const HS_METRIC_LABELS: Record<HsStage, { en: string; zh: string }> = {
  min_abs:            { en: 'Min |z|  (HS-base)',        zh: '最小 |z|（HS 基础）' },
  max_abs:            { en: 'Max |z|  (envelope hi)',    zh: '最大 |z|（包络高）' },
  envelope:           { en: 'Envelope',                  zh: '包络' },
  min_pairwise_dist:  { en: 'Min pairwise (recurrence)', zh: '最小轨道距（递归）' },
}

async function computeHsMesh() {
  loading.value = true
  error.value   = ''
  info.value    = 'computing HS field + mesh…'
  glbUrl.value  = null
  try {
    const r = await api.hsMesh({
      centerRe:   hsCenterRe.value,
      centerIm:   hsCenterIm.value,
      scale:      hsScale.value,
      resolution: hsRes.value,
      metric:     hsMetric.value,
      variant:    hsVariant.value,
      iterations: hsIter.value,
    })
    glbUrl.value = api.artifactContentUrl(r.glbArtifactId)
    info.value = `${r.vertexCount} verts · ${r.triangleCount} tri · ${(r.generatedMs ?? 0).toFixed(0)}ms`
  } catch (e: any) {
    error.value = e?.message ?? String(e)
    info.value  = ''
  } finally {
    loading.value = false
  }
}

async function computeTransitionVoxels() {
  loading.value   = true
  error.value     = ''
  info.value      = 'computing voxel field…'
  glbUrl.value    = null
  voxelData.value = null
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
    computeHsMesh()
  } else if (txRMode.value === 'voxel') {
    computeTransitionVoxels()
  } else {
    computeTransitionMesh()
  }
}
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
      </template>

      <!-- Transition params -->
      <template v-else>
        <!-- Render-mode toggle: voxel (Minecraft) vs smooth mesh -->
        <div class="mode-row" style="margin-bottom:10px">
          <button :class="['mode-btn', txRMode === 'voxel' ? 'active' : '']" @click="txRMode = 'voxel'">⬜ VOXEL</button>
          <button :class="['mode-btn', txRMode === 'mesh'  ? 'active' : '']" @click="txRMode = 'mesh'">◈ MESH</button>
        </div>

        <!-- Iso: threshold for inside. 0.5 = all inside voxels; lower = deep core only -->
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
                 type="number" v-model.number="txRes" min="16" max="512" step="16" />
          <input v-else
                 type="number" v-model.number="txRes" min="32" max="1024" step="32" />
          <span class="num dim">{{ txRMode === 'voxel' ? txRes + '³ vox' : txRes + '³ MC' }}</span>
        </div>
        <div class="group">
          <label>{{ t('iterations') }}</label>
          <input type="number" v-model.number="txIter" min="32" max="2000" step="32" />
        </div>
      </template>

      <button class="compute-btn" @click="compute" :disabled="loading">
        {{ loading ? t('three_computing') : t('three_compute') }}
      </button>

      <div v-if="info"  class="info mono">{{ info }}</div>
      <div v-if="error" class="err mono">{{ error }}</div>
    </aside>

    <!-- Viewer canvas -->
    <div class="viewer">
      <ThreeDViewer :glbUrl="glbUrl" :voxelData="voxelData" :loading="loading" />
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

.info, .err {
  font-size: 10px;
  line-height: 1.4;
  color: var(--text-dim);
  word-break: break-all;
  margin-top: 4px;
}

.err { color: var(--bad); }

.viewer {
  position: relative;
  min-height: 0;
  height: 100%;
}
</style>
