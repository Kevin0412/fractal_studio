// i18n.ts — tiny two-language store (EN / 中文) with a reactive locale ref.

import { ref } from 'vue'

export type Lang = 'en' | 'zh'

export const lang = ref<Lang>((localStorage.getItem('fsd_lang') as Lang) || 'en')

export function setLang(l: Lang) {
  lang.value = l
  localStorage.setItem('fsd_lang', l)
}

type Dict = Record<string, { en: string; zh: string }>

const dict: Dict = {
  nav_map:        { en: 'Map',         zh: '图谱' },
  nav_points:     { en: 'Points',      zh: '特殊点' },
  nav_3d:         { en: '3D',          zh: '三维' },
  nav_runs:       { en: 'Runs',        zh: '运行记录' },
  nav_system:     { en: 'System',      zh: '系统' },

  variant:        { en: 'Variant',     zh: '变体' },
  metric:         { en: 'Metric',      zh: '指标' },
  colormap:       { en: 'Colormap',    zh: '色图' },
  smooth:         { en: 'Ln-smooth',   zh: '对数平滑着色' },
  iterations:     { en: 'Iterations',  zh: '迭代' },
  scale:          { en: 'Scale',       zh: '缩放' },
  center:         { en: 'Center',      zh: '中心' },
  theta:          { en: 'θ (rad)',     zh: 'θ (弧度)' },
  transition:     { en: 'Transition',  zh: '过渡' },

  render:         { en: 'Render',      zh: '渲染' },
  export_png:     { en: 'Export PNG',  zh: '导出 PNG' },
  export_lnmap:   { en: 'Export ln-map', zh: '导出 ln 图' },
  export_video:   { en: 'Export video (P2)', zh: '导出视频 (阶段2)' },

  points_k:       { en: 'Pre-period k', zh: '前周期 k' },
  points_p:       { en: 'Period p',    zh: '周期 p' },
  points_auto:    { en: 'Auto-solve',  zh: '自动求解' },
  points_seed:    { en: 'Seed',        zh: '种子求解' },
  points_import:  { en: '→ Map',       zh: '→ 映射到图谱' },

  status_cpu:     { en: 'cpu',         zh: 'cpu' },
  status_gpu:     { en: 'gpu',         zh: 'gpu' },
  status_time:    { en: 'render',      zh: '渲染' },
  status_ready:   { en: 'ready',       zh: '就绪' },
}

export function t(key: string): string {
  const entry = dict[key]
  return entry ? entry[lang.value] : key
}
