<template>
  <div>
    <RunConfigForm title="Run All Discoveries" description="Execute all modules and build bilingual conclusions" @run="runAll" />
    <JobStatusPanel :status="status" :run-id="runId" />
    <ConclusionCards :en="en" :zh="zh" />
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import ConclusionCards from '../components/ConclusionCards.vue'
import RunConfigForm from '../components/RunConfigForm.vue'
import JobStatusPanel from '../components/JobStatusPanel.vue'
import { invokeModule } from '../api'

const status = ref('idle')
const runId = ref('')
const en = ref('Variant atlas, hidden-structure mappings, transition conversion, and STL pipeline are unified.')
const zh = ref('变体图谱、隐藏结构映射、过渡转换与 STL 流程已统一。')

const runAll = async () => {
  status.value = 'running'
  try {
    const res = await invokeModule('run-all')
    status.value = res.status
    runId.value = res.runId
    en.value = 'Unified run completed across atlas, hidden-structure, transition, special-points, and STL.'
    zh.value = '已完成图谱、隐藏结构、过渡转换、特殊点与 STL 的统一运行。'
  } catch {
    status.value = 'backend-unreachable'
  }
}
</script>
