<script setup lang="ts">
import NavRail    from './components/NavRail.vue'
import StatusRail from './components/StatusRail.vue'
import { provide, reactive } from 'vue'
import type { StatusState } from './types'

const status = reactive<StatusState>({
  cpu: null,
  gpu: null,
  renderMs: null,
  engine: 'openmp',
  scalar: 'fp64',
  cRe: null,
  cIm: null,
  zoom: null,
  iter: null,
  variant: 'mandelbrot',
  metric: 'escape',
  message: 'ready',
})

provide('status', status)
</script>

<template>
  <div class="shell">
    <NavRail />
    <main class="main">
      <router-view />
    </main>
    <StatusRail :status="status" />
  </div>
</template>

<style scoped>
.shell {
  display: grid;
  grid-template-columns: var(--nav-w) 1fr var(--rail-w);
  height: 100vh;
  overflow: hidden;
}

.main {
  border-left:  1px solid var(--rule);
  border-right: 1px solid var(--rule);
  overflow: auto;
}
</style>
