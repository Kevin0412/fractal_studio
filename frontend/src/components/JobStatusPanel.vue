<template>
  <section class="view-card">
    <h3>Job Status</h3>
    <p><strong>Run ID:</strong> {{ runId || 'N/A' }}</p>
    <p><strong>Status:</strong> {{ status }}</p>
    <RouterLink v-if="runId" :to="{ path: '/artifacts', query: { runId } }">Open artifacts for this run</RouterLink>
  </section>
</template>

<script setup lang="ts">
import { RouterLink } from 'vue-router'

defineProps<{ runId?: string; status: string }>()
</script>
