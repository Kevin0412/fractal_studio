<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="sidebar-top">
        <h1>{{ t('appTitle') }}</h1>
        <LanguageToggle />
      </div>
      <nav>
        <RouterLink to="/">{{ t('dashboard') }}</RouterLink>
        <RouterLink to="/atlas">{{ t('atlas') }}</RouterLink>
        <RouterLink to="/special-points">{{ t('specialPoints') }}</RouterLink>
        <RouterLink to="/transition-conversion">{{ t('transition') }}</RouterLink>
        <RouterLink to="/hidden-structure">{{ t('hiddenStructure') }}</RouterLink>
        <RouterLink to="/hidden-structure-families">{{ t('hsFamilies') }}</RouterLink>
        <RouterLink to="/stl-gallery">{{ t('stlGallery') }}</RouterLink>
        <RouterLink to="/explorer-map">{{ t('explorerMap') }}</RouterLink>
        <RouterLink to="/artifacts">{{ t('artifacts') }}</RouterLink>
        <RouterLink to="/conclusions">{{ t('conclusions') }}</RouterLink>
      </nav>
    </aside>
    <main class="content">
      <RouterView />
    </main>
  </div>
</template>

<script setup lang="ts">
import LanguageToggle from './components/LanguageToggle.vue'
import { t } from './i18n'
</script>
