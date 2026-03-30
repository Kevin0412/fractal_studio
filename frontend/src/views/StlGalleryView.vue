<template>
  <div>
    <RunConfigForm title="STL Export" description="Generate and index STL outputs" @run="run" />
    <JobStatusPanel :status="status" :run-id="runId" />
    <section class="view-card">
      <h3>STL Artifacts</h3>
      <div v-if="runId === ''">Run STL Export to generate models.</div>
      <div v-else>
        <p><a :href="`/artifacts?runId=${runId}&kind=stl`">Open STL artifacts</a></p>
        <model-viewer
          v-if="stlSrc"
          :src="stlSrc"
          camera-controls
          auto-rotate
          style="width: 100%; height: 420px; background: #111; border-radius: 8px"
        />
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import RunConfigForm from '../components/RunConfigForm.vue'
import JobStatusPanel from '../components/JobStatusPanel.vue'
import { getArtifacts, invokeModule, artifactContentUrl } from '../api'

const status = ref('idle')
const runId = ref('')
const stlArtifactId = ref('')

const stlSrc = computed(() => {
  if (stlArtifactId.value === '') {
    return ''
  }
  return artifactContentUrl(stlArtifactId.value)
})

const loadStlForRun = async (id: string) => {
  const res = await getArtifacts({ runId: id, kind: 'stl' })
  stlArtifactId.value = res.items[0]?.artifactId ?? ''
}

const run = async () => {
  status.value = 'running'
  try {
    const res = await invokeModule('stl-export')
    status.value = res.status
    runId.value = res.runId
    await loadStlForRun(res.runId)
  } catch {
    status.value = 'backend-unreachable'
  }
}

onMounted(async () => {
  if (runId.value !== '') {
    await loadStlForRun(runId.value)
  }
})
</script>
