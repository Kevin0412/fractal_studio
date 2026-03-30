<template>
  <section class="view-card">
    <h3>{{ title }}</h3>
    <p>{{ description }}</p>
    <button type="button" @click="$emit('run')">Run</button>
  </section>
</template>

<script setup lang="ts">
defineProps<{ title: string; description: string }>()
defineEmits<{ run: [] }>()
</script>

<style scoped>
button {
  background: #2f6f92;
  color: #fff;
  border: none;
  border-radius: 8px;
  padding: 8px 12px;
  cursor: pointer;
}
</style>
