<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { api, type SpecialPoint } from '../api'
import { t } from '../i18n'

defineEmits<{
  (e: 'import-point', p: SpecialPoint): void
}>()

const items = ref<SpecialPoint[]>([])
const loading = ref(false)

async function refresh() {
  loading.value = true
  try {
    const r = await api.specialPointsList()
    items.value = r.items
  } finally {
    loading.value = false
  }
}

defineExpose({ refresh })

onMounted(refresh)
</script>

<template>
  <div>
    <div class="head">
      <span class="panel-title">{{ t('points_table_c') }} ({{ items.length }})</span>
      <button @click="refresh" :disabled="loading">{{ t('render') }}</button>
    </div>
    <table class="pts" v-if="items.length">
      <thead>
        <tr>
          <th>{{ t('points_table_k') }}/{{ t('points_table_p') }}</th>
          <th>Re</th>
          <th>Im</th>
          <th>type</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="it in items" :key="it.id">
          <td class="num">{{ it.k }}/{{ it.p }}</td>
          <td class="num">{{ it.real.toFixed(6) }}</td>
          <td class="num">{{ it.imag.toFixed(6) }}</td>
          <td class="mono dim">{{ it.pointType }}</td>
          <td><button class="primary tiny" @click="$emit('import-point', it)">{{ t('points_import') }}</button></td>
        </tr>
      </tbody>
    </table>
    <div v-else class="empty">{{ t('points_none') }}</div>
  </div>
</template>

<style scoped>
.head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}

.pts {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--fs-mono);
  font-family: var(--mono);
}

.pts th {
  font-size: var(--fs-label);
  color: var(--text-faint);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  text-align: left;
  padding: 4px 6px;
  border-bottom: 1px solid var(--rule);
  font-weight: normal;
}

.pts td {
  padding: 4px 6px;
  border-bottom: 1px solid var(--rule);
  color: var(--text);
}

.pts tr:hover td { background: var(--accent-weak); }

.num { font-variant-numeric: tabular-nums; text-align: right; }
.dim { color: var(--text-dim); }

.tiny {
  padding: 2px 6px;
  font-size: 9px;
}

.empty {
  color: var(--text-faint);
  font-family: var(--mono);
  font-size: var(--fs-label);
  padding: 8px 0;
}
</style>
