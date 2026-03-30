<template>
  <section class="view-card">
    <h3>{{ t('dashboard') }}</h3>
    <p><strong>{{ t('cpuModel') }}:</strong> {{ cpuModel }}</p>
    <p><strong>{{ t('cpuLogicalCores') }}:</strong> {{ cpuLogicalCores }}</p>
    <p><strong>{{ t('cpuPhysicalCores') }}:</strong> {{ cpuPhysicalCores }}</p>
    <p><strong>{{ t('memoryTotal') }}:</strong> {{ memoryTotalMiB }}</p>
    <p><strong>{{ t('memoryAvailable') }}:</strong> {{ memoryAvailableMiB }}</p>
    <p><strong>{{ t('gpuModel') }}:</strong> {{ gpuModel }}</p>
    <p><strong>{{ t('gpuMemory') }}:</strong> {{ gpuMemory }}</p>
  </section>
</template>

<script setup lang="ts">
import { t } from '../i18n'

defineProps<{
  cpuModel: string
  cpuLogicalCores: number
  cpuPhysicalCores: number
  memoryTotalMiB: number
  memoryAvailableMiB: number
  gpuModel: string
  gpuMemory: string
}>()
</script>
