<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { api } from '../api'
import type { StatusState } from '../types'
import { t, lang, setLang } from '../i18n'

defineProps<{ status: StatusState }>()

const hw = ref<any>(null)
let interval: any

async function refresh() {
  try {
    hw.value = await api.hardware()
  } catch {}
}

onMounted(() => {
  refresh()
  interval = setInterval(refresh, 10000)
})

onUnmounted(() => { if (interval) clearInterval(interval) })

function fmt(n: number | null, digits = 6): string {
  if (n === null || !isFinite(n)) return '—'
  return n.toFixed(digits)
}

function fmtSci(n: number | null): string {
  if (n === null || !isFinite(n)) return '—'
  return n.toExponential(3)
}

function fmtMs(n: number | null): string {
  if (n === null || !isFinite(n)) return '—'
  return n.toFixed(1) + 'ms'
}
</script>

<template>
  <aside class="rail">
    <!-- live bar -->
    <div class="live">
      <span class="dot"></span>
      <span class="mono">{{ status.message }}</span>
      <span class="spacer"></span>
      <span class="lang" @click="setLang(lang === 'en' ? 'zh' : 'en')">{{ lang.toUpperCase() }}</span>
    </div>

    <!-- engine + timing -->
    <div class="panel">
      <div class="panel-title">engine</div>
      <div class="row"><span class="k">engine</span><span class="v mono">{{ status.engine }}</span></div>
      <div class="row"><span class="k">{{ t('status_time') }}</span><span class="v num">{{ fmtMs(status.renderMs) }}</span></div>
    </div>

    <!-- viewport numerics -->
    <div class="panel">
      <div class="panel-title">viewport</div>
      <div class="row"><span class="k">c.re</span><span class="v num">{{ fmt(status.cRe, 10) }}</span></div>
      <div class="row"><span class="k">c.im</span><span class="v num">{{ fmt(status.cIm, 10) }}</span></div>
      <div class="row"><span class="k">zoom</span><span class="v num">{{ fmtSci(status.zoom) }}</span></div>
      <div class="row"><span class="k">iter</span><span class="v num">{{ status.iter ?? '—' }}</span></div>
      <div class="row"><span class="k">variant</span><span class="v mono">{{ status.variant }}</span></div>
      <div class="row"><span class="k">metric</span><span class="v mono">{{ status.metric }}</span></div>
    </div>

    <!-- hardware -->
    <div class="panel">
      <div class="panel-title">hardware</div>
      <div v-if="hw">
        <div class="row"><span class="k">cpu</span><span class="v mono hwcpu">{{ hw.cpuModel }}</span></div>
        <div class="row"><span class="k">cores</span><span class="v num">{{ hw.cpuPhysicalCores }} / {{ hw.cpuLogicalCores }}</span></div>
        <div class="row"><span class="k">mem</span><span class="v num">{{ Math.round(hw.memoryAvailableMiB/1024) }} / {{ Math.round(hw.memoryTotalMiB/1024) }} GiB</span></div>
        <div class="row"><span class="k">gpu</span><span class="v mono hwcpu">{{ hw.gpuModel }}</span></div>
        <div class="row"><span class="k">vram</span><span class="v mono">{{ hw.gpuMemory }}</span></div>
      </div>
    </div>
  </aside>
</template>

<style scoped>
.rail {
  background: var(--bg);
  overflow: auto;
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.live {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  border-bottom: 1px solid var(--rule);
  font-size: var(--fs-label);
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

.dot {
  width: 6px; height: 6px;
  background: var(--good);
  border-radius: 50%;
  box-shadow: 0 0 6px var(--good);
}

.spacer { flex: 1; }

.lang {
  cursor: pointer;
  color: var(--accent);
  font-family: var(--mono);
}

.panel {
  background: var(--panel);
  border: none;
  border-bottom: 1px solid var(--rule);
  padding: 12px 14px;
}

.row {
  display: grid;
  grid-template-columns: 56px 1fr;
  align-items: baseline;
  padding: 2px 0;
  font-size: var(--fs-mono);
}

.k {
  color: var(--text-faint);
  font-family: var(--mono);
  font-size: var(--fs-label);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.v {
  color: var(--text);
  text-align: right;
}

.num { font-variant-numeric: tabular-nums; }

.hwcpu {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: 10px;
}
</style>
