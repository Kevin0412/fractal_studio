<template>
  <div>
    <section class="view-card">
      <h2>{{ t('dashboard') }}</h2>
      <p><strong>{{ t('backend') }}:</strong> {{ backendStatus }}</p>
      <p><strong>{{ t('openmp') }}:</strong> {{ openmp }}</p>
      <p><strong>{{ t('cuda') }}:</strong> {{ cuda }}</p>
      <button type="button" @click="refresh">{{ t('refreshHealth') }}</button>
    </section>
    <HardwarePanel
      :cpu-model="hardware.cpuModel"
      :cpu-logical-cores="hardware.cpuLogicalCores"
      :cpu-physical-cores="hardware.cpuPhysicalCores"
      :memory-total-mi-b="hardware.memoryTotalMiB"
      :memory-available-mi-b="hardware.memoryAvailableMiB"
      :gpu-model="hardware.gpuModel"
      :gpu-memory="hardware.gpuMemory"
    />
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { getSystemCheck, getSystemHardware } from '../api'
import HardwarePanel from '../components/HardwarePanel.vue'
import { t } from '../i18n'

const backendStatus = ref('checking')
const openmp = ref('unknown')
const cuda = ref('unknown')
const hardware = ref({
  cpuModel: 'unknown',
  cpuLogicalCores: 0,
  cpuPhysicalCores: 0,
  memoryTotalMiB: 0,
  memoryAvailableMiB: 0,
  gpuModel: 'unknown',
  gpuMemory: 'unknown',
})

const refresh = async () => {
  backendStatus.value = 'checking'
  try {
    const check = await getSystemCheck()
    const info = await getSystemHardware()
    backendStatus.value = 'online'
    openmp.value = check.openmp ? 'available' : 'missing'
    cuda.value = check.cuda ? 'available' : 'missing'
    hardware.value = info
  } catch {
    backendStatus.value = 'offline'
    openmp.value = 'unknown'
    cuda.value = 'unknown'
  }
}

onMounted(refresh)
</script>

<style scoped>
button {
  background: #2f6f92;
  color: #fff;
  border: none;
  border-radius: 8px;
  padding: 8px 12px;
  cursor: pointer;
}
</style>
