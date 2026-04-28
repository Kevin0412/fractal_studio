<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { api, type SpecialPointEnumResult, type SpecialPointEnumResponse, type SpecialPointKind, type SpecialPointViewport } from '../api'

const props = defineProps<{
  viewport?: SpecialPointViewport
  hoveredId?: string
  selectedId?: string
}>()

const emit = defineEmits<{
  (e: 'import-point', p: SpecialPointEnumResult): void
  (e: 'hover-point', id: string): void
  (e: 'select-point', p: SpecialPointEnumResult): void
  (e: 'results-updated', points: SpecialPointEnumResult[]): void
  (e: 'use-julia', p: SpecialPointEnumResult): void
}>()

const kind = ref<SpecialPointKind>('center')
const periodMin = ref(1)
const periodMax = ref(8)
const preperiodMin = ref(1)
const preperiodMax = ref(3)
const includeVariantExistence = ref(true)
const includeRejectedDebug = ref(false)
const visibleOnly = ref(false)
const seedsPerBatch = ref(2048)
const maxSeedBatches = ref(80)
const running = ref(false)
const message = ref('')
const result = ref<SpecialPointEnumResponse | null>(null)
const variantFilter = ref('')

watch(kind, (next) => {
  if (next === 'misiurewicz') {
    preperiodMin.value = 1
    preperiodMax.value = Math.min(preperiodMax.value, 4)
    periodMin.value = 1
    periodMax.value = Math.min(periodMax.value, 4)
  } else {
    periodMin.value = 1
    periodMax.value = Math.max(periodMax.value, 8)
  }
})

function expectedCenterCount(p: number): number {
  const counts: Record<number, number> = {}
  for (let n = 1; n <= p; n++) {
    let c = Math.pow(2, n - 1)
    for (let d = 1; d < n; d++) if (n % d === 0) c -= counts[d]
    counts[n] = c
  }
  return counts[p] ?? 0
}

function expectedMisiurewiczCount(m: number, p: number): number {
  if (m < 1 || p < 1 || m > 6 || p > 6 || m + p > 10) return -1
  if (m === 1) return 0
  let count = 2 * expectedCenterCount(p) * Math.pow(2, m - 2)
  if ((m - 1) % p === 0) count -= expectedCenterCount(p)
  return count
}

const expectedInfo = computed(() => {
  let total = 0
  if (kind.value === 'center') {
    if (periodMin.value < 1 || periodMax.value < periodMin.value || periodMax.value > 10) {
      return { ok: false, total: 0, text: 'period range must be 1..10' }
    }
    for (let p = periodMin.value; p <= periodMax.value; p++) total += expectedCenterCount(p)
  } else {
    if (preperiodMin.value < 1 || preperiodMax.value < preperiodMin.value || preperiodMax.value > 6 ||
        periodMin.value < 1 || periodMax.value < periodMin.value || periodMax.value > 6 ||
        preperiodMax.value + periodMax.value > 10) {
      return { ok: false, total: 0, text: 'preperiod 1..6, period 1..6, sum <= 10' }
    }
    for (let m = preperiodMin.value; m <= preperiodMax.value; m++) {
      for (let p = periodMin.value; p <= periodMax.value; p++) {
        const c = expectedMisiurewiczCount(m, p)
        if (c < 0) return { ok: false, total: 0, text: `count unavailable for m=${m}, p=${p}` }
        total += c
      }
    }
  }
  if (total > 3000) return { ok: false, total, text: 'expected count exceeds 3000' }
  return { ok: true, total, text: `${total} expected` }
})

const points = computed(() => result.value?.points ?? [])
const visiblePoints = computed(() => {
  if (!variantFilter.value) return points.value
  return points.value.filter(p => p.variants?.some(v => v.variant_name === variantFilter.value && v.exists))
})
const variants = computed(() => {
  const names = new Set<string>()
  for (const p of points.value) for (const v of p.variants || []) if (v.exists) names.add(v.variant_name)
  return [...names]
})
const grouped = computed(() => {
  const groups = new Map<string, SpecialPointEnumResult[]>()
  for (const p of visiblePoints.value) {
    const key = kind.value === 'center' ? `period ${p.period}` : `m${p.preperiod} p${p.period}`
    if (!groups.has(key)) groups.set(key, [])
    groups.get(key)!.push(p)
  }
  return [...groups.entries()]
})

async function enumerate() {
  if (!expectedInfo.value.ok) return
  running.value = true
  message.value = 'enumerating...'
  result.value = null
  emit('results-updated', [])
  try {
    const resp = await api.specialPointsEnumerate({
      kind: kind.value,
      periodMin: periodMin.value,
      periodMax: periodMax.value,
      preperiodMin: preperiodMin.value,
      preperiodMax: preperiodMax.value,
      includeVariantExistence: includeVariantExistence.value,
      includeRejectedDebug: includeRejectedDebug.value,
      visibleOnly: visibleOnly.value,
      seedsPerBatch: seedsPerBatch.value,
      maxSeedBatches: maxSeedBatches.value,
      viewport: props.viewport,
    })
    result.value = resp
    emit('results-updated', resp.points)
    message.value = `${resp.status}: ${resp.acceptedCount}/${resp.expectedCount} roots, ${resp.seedCount} seeds`
  } catch (e: any) {
    message.value = 'failed: ' + (e?.data?.error || e?.message || e)
  } finally {
    running.value = false
  }
}

function copyPoint(p: SpecialPointEnumResult) {
  navigator.clipboard?.writeText(`${p.re}, ${p.im}`)
}

function addBookmark(p: SpecialPointEnumResult) {
  const raw = localStorage.getItem('fs_special_point_bookmarks')
  const items = raw ? JSON.parse(raw) : []
  items.push({ id: p.id, re: p.re, im: p.im, kind: p.kind, period: p.period, preperiod: p.preperiod, createdAt: new Date().toISOString() })
  localStorage.setItem('fs_special_point_bookmarks', JSON.stringify(items.slice(-500)))
}

function selectPoint(p: SpecialPointEnumResult) {
  emit('select-point', p)
  emit('import-point', p)
}

defineExpose({ enumerate, refresh: enumerate, points })
</script>

<template>
  <div class="sp-panel">
    <div class="head">
      <span class="panel-title">Special Points</span>
      <button class="primary" @click="enumerate" :disabled="running || !expectedInfo.ok">
        {{ running ? 'running' : 'enumerate' }}
      </button>
    </div>

    <div class="controls-grid">
      <label>mode</label>
      <select v-model="kind">
        <option value="center">Hyperbolic centers</option>
        <option value="misiurewicz">Misiurewicz points</option>
      </select>
      <label v-if="kind === 'misiurewicz'">preperiod</label>
      <div v-if="kind === 'misiurewicz'" class="pair">
        <input type="number" v-model.number="preperiodMin" min="1" max="6" />
        <input type="number" v-model.number="preperiodMax" min="1" max="6" />
      </div>
      <label>period</label>
      <div class="pair">
        <input type="number" v-model.number="periodMin" min="1" max="10" />
        <input type="number" v-model.number="periodMax" min="1" max="10" />
      </div>
      <label>seeds</label>
      <div class="pair">
        <input type="number" v-model.number="seedsPerBatch" min="1" max="10000" />
        <input type="number" v-model.number="maxSeedBatches" min="1" max="200" />
      </div>
    </div>

    <div class="opts">
      <label><input type="checkbox" v-model="includeVariantExistence" /> variants</label>
      <label><input type="checkbox" v-model="visibleOnly" /> visible only</label>
      <label><input type="checkbox" v-model="includeRejectedDebug" /> rejected</label>
    </div>

    <div class="status mono" :class="{ bad: !expectedInfo.ok }">{{ expectedInfo.text }}</div>
    <div v-if="message" class="status mono">{{ message }}</div>
    <div v-if="result && !result.complete" class="status warn mono">
      incomplete: {{ result.acceptedCount }}/{{ result.expectedCount }} roots
    </div>

    <div v-if="variants.length" class="filters">
      <button :class="{ active: !variantFilter }" @click="variantFilter = ''">all</button>
      <button v-for="v in variants" :key="v" :class="{ active: variantFilter === v }" @click="variantFilter = v">{{ v }}</button>
    </div>

    <div v-if="grouped.length" class="groups">
      <section v-for="[name, pts] in grouped" :key="name" class="group-block">
        <div class="group-title mono">{{ name }} · {{ pts.length }}</div>
        <div
          v-for="p in pts"
          :key="p.id"
          class="point-row"
          :class="{ hover: hoveredId === p.id, selected: selectedId === p.id }"
          @mouseenter="$emit('hover-point', p.id)"
          @mouseleave="$emit('hover-point', '')"
          @click="selectPoint(p)">
          <div class="coord mono">
            <span>{{ p.re.toFixed(10) }}</span>
            <span>{{ p.im.toFixed(10) }}</span>
          </div>
          <div class="meta mono">res {{ p.residual.toExponential(1) }} · {{ p.newtonIterations }} it</div>
          <div v-if="p.variants?.length" class="tags">
            <button
              v-for="v in p.variants.filter(v => v.exists)"
              :key="v.variant_name"
              class="tag"
              @click.stop="variantFilter = v.variant_name">
              {{ v.variant_name }}
            </button>
          </div>
          <div class="actions">
            <button @click.stop="copyPoint(p)">copy</button>
            <button @click.stop="$emit('use-julia', p)">julia c</button>
            <button @click.stop="addBookmark(p)">bookmark</button>
          </div>
        </div>
      </section>
    </div>
    <div v-else class="empty mono">No enumeration results yet.</div>
  </div>
</template>

<style scoped>
.sp-panel { display: flex; flex-direction: column; gap: 10px; }
.head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.controls-grid {
  display: grid;
  grid-template-columns: 72px 1fr;
  align-items: center;
  gap: 6px 8px;
}
.controls-grid label,
.opts label {
  color: var(--text-dim);
  font-size: var(--fs-label);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.opts { display: flex; flex-wrap: wrap; gap: 8px; }
.opts input { width: auto; margin-right: 4px; }
.status { color: var(--text-dim); font-size: 10px; }
.status.bad { color: var(--bad); }
.status.warn { color: #d5ad45; }
.filters { display: flex; flex-wrap: wrap; gap: 5px; }
.filters button,
.tag {
  padding: 2px 6px;
  font-size: 9px;
}
.filters button.active { border-color: var(--accent); color: var(--accent); }
.groups { display: flex; flex-direction: column; gap: 10px; }
.group-title {
  color: var(--accent);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 5px;
}
.point-row {
  border: 1px solid var(--rule);
  padding: 7px;
  margin-bottom: 5px;
  cursor: pointer;
  background: rgba(255,255,255,0.015);
}
.point-row.hover,
.point-row:hover { border-color: var(--accent); background: var(--accent-weak); }
.point-row.selected { border-color: var(--accent); box-shadow: inset 2px 0 0 var(--accent); }
.coord {
  display: grid;
  grid-template-columns: 1fr;
  gap: 2px;
  font-size: 10px;
  color: var(--text);
}
.meta { color: var(--text-faint); font-size: 9px; margin-top: 4px; }
.tags { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 5px; }
.actions { display: flex; gap: 5px; margin-top: 6px; }
.actions button { padding: 2px 5px; font-size: 9px; }
.empty { color: var(--text-faint); font-size: 10px; padding: 8px 0; }
.num { font-variant-numeric: tabular-nums; }
</style>
