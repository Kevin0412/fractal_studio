<template>
  <section class="view-card">
    <h3>{{ title }}</h3>
    <label>
      Iterations
      <input v-model.number="iterations" type="number" min="1" />
    </label>
    <label>
      Color Mapping
      <input v-model="colorMap" type="text" />
    </label>
    <label>
      Center (re)
      <input v-model.number="centerRe" type="number" step="0.000001" />
    </label>
    <label>
      Center (im)
      <input v-model.number="centerIm" type="number" step="0.000001" />
    </label>
    <label>
      Scale
      <input v-model.number="scale" type="number" step="0.000001" />
    </label>
    <button type="button" @click="apply">{{ t('run') }}</button>
  </section>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { t } from '../i18n'

const props = defineProps<{ title: string }>()
const emit = defineEmits<{ apply: [payload: { iterations: number; colorMap: string; centerRe: number; centerIm: number; scale: number }] }>()

const iterations = ref(1024)
const colorMap = ref('classic')
const centerRe = ref(0)
const centerIm = ref(0)
const scale = ref(4)

const apply = () => {
  emit('apply', {
    iterations: iterations.value,
    colorMap: colorMap.value,
    centerRe: centerRe.value,
    centerIm: centerIm.value,
    scale: scale.value,
  })
}
</script>

<style scoped>
label {
  display: block;
  margin-bottom: 10px;
}
input {
  width: 100%;
  margin-top: 4px;
}
button {
  background: #2f6f92;
  color: #fff;
  border: none;
  border-radius: 8px;
  padding: 8px 12px;
  cursor: pointer;
}
</style>
