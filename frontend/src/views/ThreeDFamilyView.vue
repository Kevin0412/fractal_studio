<template>
  <div>
    <RunConfigForm title="HS Families" description="Compile and run hidden-structure family stages" @run="run" />
    <JobStatusPanel :status="status" :run-id="runId" />
    <GalleryGrid title="HS Stage Coverage" :items="stages.map((s) => s.label)" />

    <section class="view-card">
      <h3>Explore HS on Map</h3>
      <ul>
        <li v-for="stage in stages" :key="stage.label">
          {{ stage.label }}
          <button class="map-btn" @click="openOnMap(stage.variety)">Explore on Map</button>
        </li>
      </ul>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import GalleryGrid from '../components/GalleryGrid.vue'
import RunConfigForm from '../components/RunConfigForm.vue'
import JobStatusPanel from '../components/JobStatusPanel.vue'
import { invokeModule } from '../api'

const router = useRouter()

const stages = [
  { label: 'HS-Base', variety: 0 },
  { label: 'HS-Variant', variety: 1 },
  { label: 'HS-Recurrence', variety: 2 },
  { label: 'HS-Symmetry', variety: 3 },
  { label: 'HS-Envelope', variety: 4 },
  { label: 'HS-MultiChannel', variety: 9 },
]

const status = ref('idle')
const runId = ref('')

const openOnMap = (variety: number) => {
  void router.push({
    path: '/explorer-map',
    query: {
      centerRe: '0',
      centerIm: '0',
      scale: '4',
      iterations: '2048',
      colorMap: 'classic',
      variety: String(variety),
    },
  })
}

const run = async () => {
  status.value = 'running'
  try {
    const res = await invokeModule('hidden-structure-family')
    status.value = res.status
    runId.value = res.runId
  } catch {
    status.value = 'backend-unreachable'
  }
}
</script>

<style scoped>
.map-btn {
  margin-left: 8px;
}
</style>
