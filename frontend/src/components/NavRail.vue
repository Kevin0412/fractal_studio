<script setup lang="ts">
import { RouterLink } from 'vue-router'
import { t, lang, toggleLang } from '../i18n'
import { isLight, toggleTheme } from '../theme'

const items = [
  { to: '/',       glyph: 'MP', label: 'nav_map' },
  { to: '/points', glyph: 'PT', label: 'nav_points' },
  { to: '/3d',     glyph: '3D', label: 'nav_3d' },
  { to: '/runs',   glyph: 'RN', label: 'nav_runs' },
  { to: '/system', glyph: 'SY', label: 'nav_system' },
]
</script>

<template>
  <nav class="navrail">
    <div class="brand mono">fs</div>
    <RouterLink
      v-for="item in items"
      :key="item.to"
      :to="item.to"
      class="nav-item"
      active-class="active">
      <span class="glyph mono">{{ item.glyph }}</span>
      <span class="tip">{{ t(item.label) }}</span>
    </RouterLink>

    <div class="spacer"></div>

    <button class="theme-btn nav-item" @click="toggleLang" title="Toggle language / 切换语言">
      <span class="glyph mono">{{ lang === 'en' ? 'ZH' : 'EN' }}</span>
      <span class="tip">{{ lang === 'en' ? '中文' : 'English' }}</span>
    </button>

    <button class="theme-btn nav-item" @click="toggleTheme" :title="isLight ? 'Switch to dark mode' : 'Switch to light mode'">
      <span class="glyph mono">{{ isLight ? 'DK' : 'LT' }}</span>
      <span class="tip">{{ isLight ? t('nav_dark') : t('nav_light') }}</span>
    </button>
  </nav>
</template>

<style scoped>
.navrail {
  display: flex;
  flex-direction: column;
  align-items: center;
  background: var(--bg);
  padding-top: 18px;
  padding-bottom: 12px;
  gap: 4px;
  height: 100%;
}

.spacer { flex: 1; }

.theme-btn {
  border: none;
  background: transparent;
  cursor: pointer;
  padding: 0;
  text-transform: none;
  letter-spacing: normal;
  font-size: inherit;
}

.theme-btn:hover { color: var(--accent); }

.brand {
  font-family: var(--mono);
  font-size: 14px;
  color: var(--accent);
  margin-bottom: 20px;
  letter-spacing: 0.02em;
}

.nav-item {
  position: relative;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-faint);
  border: 1px solid transparent;
}

.nav-item:hover { color: var(--text); }

.nav-item.active {
  color: var(--accent);
  border-color: var(--rule-hi);
  background: var(--accent-weak);
}

.glyph {
  font-size: 11px;
  letter-spacing: 0.02em;
  text-transform: uppercase;
}

.tip {
  position: absolute;
  left: 46px;
  background: var(--panel);
  border: 1px solid var(--rule);
  padding: 4px 8px;
  font-size: var(--fs-label);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-dim);
  white-space: nowrap;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.12s;
  z-index: 20;
}

.nav-item:hover .tip { opacity: 1; }
</style>
