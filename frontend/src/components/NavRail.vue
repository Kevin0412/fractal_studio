<script setup lang="ts">
import { RouterLink } from 'vue-router'
import { t } from '../i18n'

// Simple two-letter glyph icons in monospace — matches the instrumentation aesthetic.
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
  </nav>
</template>

<style scoped>
.navrail {
  display: flex;
  flex-direction: column;
  align-items: center;
  background: var(--bg);
  padding-top: 18px;
  gap: 4px;
}

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
