<script setup lang="ts">
import { ref } from 'vue'
import ThreeDViewer from '../components/ThreeDViewer.vue'
import { api, VARIANTS, type Variant, type HsStage } from '../api'
import { t } from '../i18n'

type Mode = 'hs' | 'transition'

const mode = ref<Mode>('hs')

// HS mode params
const hsMetric  = ref<HsStage>('min_abs')
const hsVariant = ref<Variant>('mandelbrot')
const hsRes     = ref(128)
const hsIter    = ref(512)
const hsCenterRe = ref(-0.75)
const hsCenterIm = ref(0.0)
const hsScale    = ref(3.0)

// Transition mode params
const txRes   = ref(96)
const txTheta = ref(0.0)
const txIso   = ref(0.5)
const txIter  = ref(128)
const txCenterRe = ref(-0.75)
const txCenterIm = ref(0.0)
const txScale    = ref(3.0)

const glbUrl  = ref<string | null>(null)
const loading = ref(false)
const info    = ref('')
const error   = ref('')

const HS_METRICS: { value: HsStage; label: string }[] = [
  { value: 'min_abs',           label: 'min |z|  (HS-base)' },
  { value: 'max_abs',           label: 'max |z|  (envelope hi)' },
  { value: 'envelope',          label: 'envelope' },
  { value: 'min_pairwise_dist', label: 'min pairwise dist (recurrence)' },
]

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

async function computeTransitionMesh() {
  loading.value = true
  error.value   = ''
  info.value    = 'marching cubes…'
  glbUrl.value  = null
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
  if (mode.value === 'hs') computeHsMesh()
  else computeTransitionMesh()
}
</script>

<template>
  <div class="three-view">
    <!-- Left control strip -->
    <aside class="controls">
      <!-- Mode toggle -->
      <div class="mode-row">
        <button :class="['mode-btn', mode === 'hs' ? 'active' : '']" @click="mode = 'hs'">HS</button>
        <button :class="['mode-btn', mode === 'transition' ? 'active' : '']" @click="mode = 'transition'">M↔B</button>
      </div>

      <!-- HS params -->
      <template v-if="mode === 'hs'">
        <div class="group">
          <label>metric</label>
          <select v-model="hsMetric">
            <option v-for="m in HS_METRICS" :key="m.value" :value="m.value">{{ m.label }}</option>
          </select>
        </div>
        <div class="group">
          <label>{{ t('variant') }}</label>
          <select v-model="hsVariant">
            <option v-for="v in VARIANTS" :key="v" :value="v">{{ v }}</option>
          </select>
        </div>
        <div class="group">
          <label>resolution</label>
          <input type="number" v-model.number="hsRes" min="32" max="512" step="32" />
        </div>
        <div class="group">
          <label>{{ t('iterations') }}</label>
          <input type="number" v-model.number="hsIter" min="64" max="10000" step="128" />
        </div>
        <div class="group">
          <label>center Re</label>
          <input type="number" v-model.number="hsCenterRe" step="0.01" />
        </div>
        <div class="group">
          <label>center Im</label>
          <input type="number" v-model.number="hsCenterIm" step="0.01" />
        </div>
        <div class="group">
          <label>scale</label>
          <input type="number" v-model.number="hsScale" min="0.0001" step="0.1" />
        </div>
      </template>

      <!-- Transition params -->
      <template v-else>
        <div class="group">
          <label>θ (M↔B)</label>
          <input type="range" min="0" max="1.5707963267948966" step="0.01" v-model.number="txTheta" />
          <span class="num">{{ txTheta.toFixed(3) }}</span>
        </div>
        <div class="group">
          <label>iso level</label>
          <input type="range" min="0.1" max="0.9" step="0.05" v-model.number="txIso" />
          <span class="num">{{ txIso.toFixed(2) }}</span>
        </div>
        <div class="group">
          <label>resolution</label>
          <input type="number" v-model.number="txRes" min="32" max="256" step="16" />
        </div>
        <div class="group">
          <label>{{ t('iterations') }}</label>
          <input type="number" v-model.number="txIter" min="32" max="2000" step="32" />
        </div>
        <div class="group">
          <label>center Re</label>
          <input type="number" v-model.number="txCenterRe" step="0.01" />
        </div>
        <div class="group">
          <label>center Im</label>
          <input type="number" v-model.number="txCenterIm" step="0.01" />
        </div>
        <div class="group">
          <label>scale</label>
          <input type="number" v-model.number="txScale" min="0.0001" step="0.1" />
        </div>
      </template>

      <button class="compute-btn" @click="compute" :disabled="loading">
        {{ loading ? 'computing…' : 'compute' }}
      </button>

      <div v-if="info"  class="info mono">{{ info }}</div>
      <div v-if="error" class="err mono">{{ error }}</div>
    </aside>

    <!-- Viewer canvas -->
    <div class="viewer">
      <ThreeDViewer :glbUrl="glbUrl" :loading="loading" />
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
