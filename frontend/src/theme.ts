// theme.ts — global reactive dark/light theme state.
//
// Import { isLight, toggleTheme } anywhere that needs to react to theme changes
// (e.g. NavRail for the toggle button, ThreeDViewer for WebGL clear colour).

import { ref, watch } from 'vue'

export const isLight = ref(false)

export function toggleTheme() {
  isLight.value = !isLight.value
  if (isLight.value) {
    document.documentElement.setAttribute('data-theme', 'light')
  } else {
    document.documentElement.removeAttribute('data-theme')
  }
}

// BG colours to use as WebGL clear colour (must match tokens.css)
export const CLEAR_DARK  = 0x0a0b0d
export const CLEAR_LIGHT = 0xf2f2ee

export function clearColor(): number {
  return isLight.value ? CLEAR_LIGHT : CLEAR_DARK
}
