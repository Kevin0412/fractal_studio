import { ref } from 'vue'

export type Language = 'en' | 'zh'

export const language = ref<Language>('en')

const dictionary: Record<Language, Record<string, string>> = {
  en: {
    appTitle: 'Fractal Studio',
    dashboard: 'Dashboard',
    atlas: 'Atlas',
    specialPoints: 'Special Points',
    transition: 'Transition',
    hiddenStructure: 'Hidden Structure',
    hsFamilies: 'HS Families',
    stlGallery: 'STL Gallery',
    conclusions: 'Conclusions',
    explorerMap: 'Explorer Map',
    artifacts: 'Artifacts',
    language: '中文',
    refreshHealth: 'Refresh Health',
    backend: 'Backend',
    openmp: 'OpenMP',
    cuda: 'CUDA',
    cpuModel: 'CPU Model',
    cpuLogicalCores: 'CPU Logical Cores',
    cpuPhysicalCores: 'CPU Physical Cores',
    memoryTotal: 'Memory Total (MiB)',
    memoryAvailable: 'Memory Available (MiB)',
    gpuModel: 'GPU Model',
    gpuMemory: 'GPU Memory',
    specialPointModeAuto: 'Auto Mode (k,p)',
    specialPointModeSeed: 'Seed Mode (complex)',
    run: 'Run',
    formulas: 'Variant Formulas',
  },
  zh: {
    appTitle: '分形工作台',
    dashboard: '仪表盘',
    atlas: '图谱',
    specialPoints: '特殊点',
    transition: '过渡',
    hiddenStructure: '隐藏结构',
    hsFamilies: 'HS 家族',
    stlGallery: 'STL 展示',
    conclusions: '结论',
    explorerMap: '探索地图',
    artifacts: '成果下载',
    language: 'EN',
    refreshHealth: '刷新状态',
    backend: '后端',
    openmp: 'OpenMP',
    cuda: 'CUDA',
    cpuModel: 'CPU 型号',
    cpuLogicalCores: 'CPU 逻辑核心',
    cpuPhysicalCores: 'CPU 物理核心',
    memoryTotal: '总内存 (MiB)',
    memoryAvailable: '可用内存 (MiB)',
    gpuModel: 'GPU 型号',
    gpuMemory: 'GPU 显存',
    specialPointModeAuto: '自动模式 (k,p)',
    specialPointModeSeed: '种子模式 (复数)',
    run: '运行',
    formulas: '变体公式',
  },
}

export function t(key: string): string {
  const entry = dictionary[language.value][key]
  if (entry == null) {
    return key
  }
  return entry
}

export function toggleLanguage(): void {
  if (language.value === 'en') {
    language.value = 'zh'
  } else {
    language.value = 'en'
  }
}
