<template>
  <section class="view-card">
    <h3>{{ title }}</h3>
    <label>
      Iterations
      <input v-model.number="iterations" type="number" min="1" @change="emitStyle" />
    </label>
    <label>
      Color Mapping
      <select v-model="colorMap" @change="emitStyle">
        <option value="classic_cos">classic_cos</option>
        <option value="mod17">mod17</option>
        <option value="hsv_wheel">hsv_wheel</option>
        <option value="tri765">tri765</option>
        <option value="grayscale">grayscale</option>
      </select>
    </label>
    <ul class="formula-list">
      <li>mod17: R=iter%256, G=iter/256, B=(iter%17)*17</li>
      <li>hsv_wheel: H=iter%360, S=1, V=1</li>
      <li>tri765: piecewise RGB over m=int(iter*765/maxIter)</li>
      <li>inside-set: white (255,255,255)</li>
    </ul>
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
      <input v-model.number="scale" type="number" step="0.000001" min="1e-18" />
    </label>
    <button type="button" @click="applyViewport">{{ t('run') }}</button>
  </section>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { t } from '../i18n'

const props = defineProps<{
  title: string
  iterations: number
  colorMap: string
  centerRe: number
  centerIm: number
  scale: number
}>()

const emit = defineEmits<{
  styleChange: [payload: { iterations: number; colorMap: string }]
  applyViewport: [payload: { centerRe: number; centerIm: number; scale: number }]
}>()

const iterations = ref(props.iterations)
const colorMap = ref(props.colorMap)
const centerRe = ref(props.centerRe)
const centerIm = ref(props.centerIm)
const scale = ref(props.scale)

watch(
  () => props.iterations,
  (v) => {
    iterations.value = v
  },
)
watch(
  () => props.colorMap,
  (v) => {
    colorMap.value = v
  },
)
watch(
  () => props.centerRe,
  (v) => {
    centerRe.value = v
  },
)
watch(
  () => props.centerIm,
  (v) => {
    centerIm.value = v
  },
)
watch(
  () => props.scale,
  (v) => {
    scale.value = v
  },
)

const emitStyle = () => {
  emit('styleChange', {
    iterations: iterations.value,
    colorMap: colorMap.value,
  })
}

const applyViewport = () => {
  emit('applyViewport', {
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
input,
select {
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
.formula-list {
  margin: 8px 0 12px;
  font-size: 12px;
  opacity: 0.9;
  padding-left: 18px;
}
</style>
