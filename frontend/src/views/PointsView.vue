<script setup lang="ts">
import { ref } from 'vue'
import { api } from '../api'
import SpecialPointList from '../components/SpecialPointList.vue'
import { useRouter } from 'vue-router'
import { t } from '../i18n'

const router = useRouter()

const k = ref(0)
const p = ref(3)
const busy = ref(false)
const message = ref('')
const lastCount = ref<number | null>(null)

const list = ref<InstanceType<typeof SpecialPointList> | null>(null)

async function autoSolve() {
  busy.value = true
  message.value = 'solving…'
  try {
    const pointType = k.value === 0 ? 'hyperbolic' : 'misiurewicz'
    const resp = await api.specialPointsAuto(k.value, p.value, pointType)
    lastCount.value = resp.count
    message.value = `found ${resp.count} point(s) for k=${k.value} p=${p.value}`
    list.value?.refresh()
  } catch (e: any) {
    message.value = 'failed: ' + (e?.message || e)
  } finally {
    busy.value = false
  }
}

// Seed form
const seedK = ref(1)
const seedP = ref(2)
const seedRe = ref(-0.1)
const seedIm = ref(0.75)

async function seedSolve() {
  busy.value = true
  message.value = 'newton iterating…'
  try {
    const resp = await api.specialPointsSeed(seedK.value, seedP.value, seedRe.value, seedIm.value)
    message.value = resp.converged
      ? `converged → (${resp.points[0].real.toFixed(8)}, ${resp.points[0].imag.toFixed(8)})`
      : 'did not converge'
    list.value?.refresh()
  } catch (e: any) {
    message.value = 'failed: ' + (e?.message || e)
  } finally {
    busy.value = false
  }
}

function onImportPoint(pt: any) {
  // Persist intent and navigate; MapView reads from sessionStorage on mount
  // via its special-points list refresh, but since SpecialPointList only
  // imports to its host, we go through a queryless nav and rely on the
  // user re-clicking in the map list too. Simplest and most robust:
  sessionStorage.setItem('fs_pending_center', JSON.stringify({
    re: pt.real, im: pt.imag,
  }))
  router.push('/')
}
</script>

<template>
  <div class="wrap">
    <div class="col">
      <div class="panel">
        <div class="panel-title">{{ t('points_auto') }}</div>
        <div class="row">
          <div><label>{{ t('points_k') }}</label><input type="number" v-model.number="k" min="0" /></div>
          <div><label>{{ t('points_p') }}</label><input type="number" v-model.number="p" min="1" /></div>
        </div>
        <button class="primary" @click="autoSolve" :disabled="busy">{{ t('points_auto') }}</button>
      </div>

      <div class="panel">
        <div class="panel-title">{{ t('points_seed') }}</div>
        <div class="row">
          <div><label>{{ t('points_table_k') }}</label><input type="number" v-model.number="seedK" min="0" /></div>
          <div><label>{{ t('points_table_p') }}</label><input type="number" v-model.number="seedP" min="1" /></div>
        </div>
        <div class="row">
          <div><label>{{ t('points_seed_re') }}</label><input type="number" v-model.number="seedRe" step="0.01" /></div>
          <div><label>{{ t('points_seed_im') }}</label><input type="number" v-model.number="seedIm" step="0.01" /></div>
        </div>
        <button class="primary" @click="seedSolve" :disabled="busy">{{ t('points_seed') }}</button>
      </div>

      <div v-if="message" class="msg mono">{{ message }}</div>
    </div>

    <div class="col wide">
      <SpecialPointList ref="list" @import-point="onImportPoint" />
    </div>
  </div>
</template>

<style scoped>
.wrap {
  display: grid;
  grid-template-columns: 320px 1fr;
  gap: 1px;
  height: 100%;
  overflow: hidden;
}

.col {
  padding: 18px;
  overflow: auto;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.col.wide { background: var(--bg-raised); }

.row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-bottom: 10px;
}

.msg {
  color: var(--text-dim);
  padding: 6px 0;
  font-size: var(--fs-label);
}

.panel { padding: 14px; }
</style>
