<template>
  <div>
    <RunConfigForm title="Transition Conversion" description="Mandelbrot ↔ Burning Ship bridge sweep" @run="run" />
    <JobStatusPanel :status="status" :run-id="runId" />
    <section class="view-card" v-if="imageUrl">
      <h3>Latest Transition Image</h3>
      <img :src="imageUrl" alt="transition" class="preview-image" />
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import RunConfigForm from '../components/RunConfigForm.vue'
import JobStatusPanel from '../components/JobStatusPanel.vue'
import { artifactContentUrl, getArtifacts, invokeModule } from '../api'

const status = ref('idle')
const runId = ref('')
const imageUrl = ref('')

const run = async () => {
  status.value = 'running'
  try {
    const res = await invokeModule('transition-conversion')
    status.value = res.status
    runId.value = res.runId
    const artifacts = await getArtifacts({ runId: res.runId, kind: 'image' })
    imageUrl.value = artifacts.items[0] ? artifactContentUrl(artifacts.items[0].artifactId) : ''
  } catch {
    status.value = 'backend-unreachable'
  }
}
</script>

<style scoped>
.preview-image {
  width: 100%;
  border-radius: 8px;
}
</style>
