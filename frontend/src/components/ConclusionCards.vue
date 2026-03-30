<template>
  <section class="view-card">
    <h3>Conclusions</h3>
    <article>
      <h4>EN</h4>
      <p>{{ en }}</p>
    </article>
    <article>
      <h4>中文</h4>
      <p>{{ zh }}</p>
    </article>
  </section>
</template>

<script setup lang="ts">
defineProps<{ en: string; zh: string }>()
</script>
