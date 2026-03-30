<template>
  <div>
    <section class="view-card">
      <h2>{{ t('specialPoints') }}</h2>
      <label>
        Mode
        <select v-model="mode">
          <option value="auto">{{ t('specialPointModeAuto') }}</option>
          <option value="seed">{{ t('specialPointModeSeed') }}</option>
        </select>
      </label>
      <label>
        Family
        <input v-model="family" type="text" />
      </label>
      <div v-if="mode === 'auto'">
        <label>
          Point type
          <select v-model="pointType">
            <option value="misiurewicz">misiurewicz</option>
            <option value="hyperbolic">hyperbolic</option>
          </select>
        </label>
        <label>k (pre-period)<input v-model.number="k" type="number" min="0" /></label>
        <label>p (period)<input v-model.number="p" type="number" min="1" /></label>
      </div>
      <div v-else>
        <label>k (pre-period)<input v-model.number="k" type="number" min="0" /></label>
        <label>p (period)<input v-model.number="p" type="number" min="1" /></label>
        <label>max iter<input v-model.number="maxIter" type="number" min="1" /></label>
        <label>seed real<input v-model.number="seedRe" type="number" step="0.000001" /></label>
        <label>seed imag<input v-model.number="seedIm" type="number" step="0.000001" /></label>
      </div>
      <button type="button" @click="run">{{ t('run') }}</button>
    </section>
    <section class="view-card">
      <h3>Saved Points</h3>
      <table>
        <thead>
          <tr>
            <th>time</th>
            <th>family</th>
            <th>mode</th>
            <th>type</th>
            <th>k</th>
            <th>p</th>
            <th>real</th>
            <th>imag</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="point in points" :key="point.id">
            <td>{{ point.createdAt }}</td>
            <td>{{ point.family }}</td>
            <td>{{ point.sourceMode }}</td>
            <td>{{ point.pointType }}</td>
            <td>{{ point.k }}</td>
            <td>{{ point.p }}</td>
            <td>{{ point.real }}</td>
            <td>{{ point.imag }}</td>
          </tr>
        </tbody>
      </table>
    </section>
    <JobStatusPanel :status="status" :run-id="runId" />
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import JobStatusPanel from '../components/JobStatusPanel.vue'
import { getSpecialPoints, postSpecialPointsAuto, postSpecialPointsSeed, type SpecialPoint } from '../api'
import { t } from '../i18n'

const status = ref('idle')
const runId = ref('')
const mode = ref<'auto' | 'seed'>('auto')
const family = ref('mandelbrot')
const pointType = ref('misiurewicz')
const k = ref(1)
const p = ref(1)
const maxIter = ref(256)
const seedRe = ref(0)
const seedIm = ref(0)
const points = ref<SpecialPoint[]>([])

const loadPoints = async () => {
  const res = await getSpecialPoints()
  points.value = res.items
}

const run = async () => {
  status.value = 'running'
  runId.value = ''
  try {
    if (mode.value === 'auto') {
      const res = await postSpecialPointsAuto({
        family: family.value,
        pointType: pointType.value,
        k: k.value,
        p: p.value,
      })
      status.value = `completed-auto-${res.count}`
      runId.value = res.points[0]?.id ?? ''
    } else {
      const res = await postSpecialPointsSeed({
        family: family.value,
        k: k.value,
        p: p.value,
        maxIter: maxIter.value,
        seed: {
          re: seedRe.value,
          im: seedIm.value,
        },
      })
      status.value = 'completed-seed-1'
      runId.value = res.point.id
    }
    await loadPoints()
  } catch {
    status.value = 'backend-unreachable'
  }
}

onMounted(async () => {
  try {
    await loadPoints()
  } catch {
    status.value = 'backend-unreachable'
  }
})
</script>

<style scoped>
label {
  display: block;
  margin-bottom: 10px;
}
button {
  background: #2f6f92;
  color: #fff;
  border: none;
  border-radius: 8px;
  padding: 8px 12px;
  cursor: pointer;
}
table {
  width: 100%;
  border-collapse: collapse;
}
th,
td {
  border-bottom: 1px solid #ddd;
  padding: 6px 8px;
  text-align: left;
  font-size: 13px;
}
</style>
