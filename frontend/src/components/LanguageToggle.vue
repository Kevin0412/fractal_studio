<template>
  <button type="button" class="lang-btn" @click="toggleLanguage()">{{ t('language') }}</button>
</template>

<script setup lang="ts">
import { t, toggleLanguage } from '../i18n'
</script>

<style scoped>
.lang-btn {
  background: #2f6f92;
  color: #fff;
  border: none;
  border-radius: 8px;
  padding: 8px 10px;
  cursor: pointer;
}
</style>
