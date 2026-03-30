<template>
  <div>
    <section class="view-card">
      <h2>{{ t('artifacts') }}</h2>
      <label>
        Kind
        <select v-model="kind" @change="loadArtifacts">
          <option value="">all</option>
          <option value="image">image</option>
          <option value="video">video</option>
          <option value="stl">stl</option>
          <option value="report">report</option>
        </select>
      </label>
      <label>
        Run ID
        <input v-model="runId" type="text" placeholder="optional" />
      </label>
      <button type="button" @click="loadArtifacts">Refresh</button>
    </section>

    <section class="view-card">
      <h3>Artifacts</h3>
      <table>
        <thead>
          <tr>
            <th>runId</th>
            <th>name</th>
            <th>kind</th>
            <th>preview</th>
            <th>download</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="item in items" :key="item.artifactId">
            <td>{{ item.runId }}</td>
            <td>{{ item.name }}</td>
            <td>{{ item.kind }}</td>
            <td>
              <button type="button" @click="select(item)">open</button>
            </td>
            <td>
              <a :href="artifactDownloadUrl(item.downloadPath)" target="_blank" rel="noreferrer">download</a>
            </td>
          </tr>
        </tbody>
      </table>
    </section>

    <section v-if="selected != null" class="view-card">
      <h3>Preview: {{ selected.name }}</h3>
      <div v-if="selected.kind === 'image'">
        <img :src="artifactContentUrl(selected.artifactId)" :alt="selected.name" class="preview-image" />
      </div>
      <div v-else-if="selected.kind === 'video'">
        <video :src="artifactContentUrl(selected.artifactId)" controls class="preview-video"></video>
      </div>
      <div v-else>
        <p>Preview not available for this type. Use download.</p>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { artifactContentUrl, artifactDownloadUrl, getArtifacts, type ArtifactItem } from '../api'
import { t } from '../i18n'

const route = useRoute()

const kind = ref('')
const runId = ref('')
const items = ref<ArtifactItem[]>([])
const selected = ref<ArtifactItem | null>(null)

const select = (item: ArtifactItem) => {
  selected.value = item
}

const loadArtifacts = async () => {
  const res = await getArtifacts({
    kind: kind.value,
    runId: runId.value,
  })
  items.value = res.items
  selected.value = res.items[0] ?? null
}

onMounted(async () => {
  const routeRunId = route.query.runId
  if (typeof routeRunId === 'string') {
    runId.value = routeRunId
  }
  await loadArtifacts()
})

watch(
  () => route.query.runId,
  async (value) => {
    runId.value = typeof value === 'string' ? value : ''
    await loadArtifacts()
  },
)
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
.preview-image {
  max-width: 100%;
  border-radius: 8px;
}
.preview-video {
  max-width: 100%;
  border-radius: 8px;
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
