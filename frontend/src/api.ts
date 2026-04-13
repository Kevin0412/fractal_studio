// api.ts — typed client for the native fractal_studio backend.
//
// All endpoints are POST JSON / GET query-string. Returns parsed JSON.

const BASE =
  (import.meta as any).env?.VITE_BACKEND_URL ??
  `http://${location.hostname}:18080`

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`${path}: ${res.status} ${await res.text()}`)
  return res.json() as Promise<T>
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(BASE + path)
  if (!res.ok) throw new Error(`${path}: ${res.status}`)
  return res.json() as Promise<T>
}

// ---- Types ----

export type Variant =
  | 'mandelbrot' | 'tricorn' | 'burning_ship' | 'celtic' | 'heart'
  | 'buffalo' | 'perp_buffalo' | 'celtic_ship' | 'mandelceltic' | 'perp_ship'

export const VARIANTS: Variant[] = [
  'mandelbrot', 'tricorn', 'burning_ship', 'celtic', 'heart',
  'buffalo', 'perp_buffalo', 'celtic_ship', 'mandelceltic', 'perp_ship',
]

// Human-readable display names, used in dropdowns and info panels
export const VARIANT_LABELS: Record<Variant, { en: string; zh: string }> = {
  mandelbrot:   { en: 'Mandelbrot',        zh: 'Mandelbrot' },
  tricorn:      { en: 'Tricorn',           zh: '三角形' },
  burning_ship: { en: 'Burning Ship',      zh: '燃烧船' },
  celtic:       { en: 'Celtic',            zh: '凯尔特' },
  heart:        { en: 'Heart',             zh: '心形' },
  buffalo:      { en: 'Buffalo',           zh: '野牛' },
  perp_buffalo: { en: 'Perp. Buffalo',     zh: '垂直野牛' },
  celtic_ship:  { en: 'Celtic Ship',       zh: '凯尔特船' },
  mandelceltic: { en: 'Mandelceltic',      zh: '曼德凯尔特' },
  perp_ship:    { en: 'Perp. Ship',        zh: '垂直船' },
}

export type Metric = 'escape' | 'min_abs' | 'max_abs' | 'envelope' | 'min_pairwise_dist'

export const METRICS: Metric[] = ['escape', 'min_abs', 'max_abs', 'envelope', 'min_pairwise_dist']

export type ColorMap = 'classic_cos' | 'mod17' | 'hsv_wheel' | 'tri765' | 'grayscale' | 'hs_rainbow'

export const COLORMAPS: ColorMap[] = ['classic_cos', 'mod17', 'hsv_wheel', 'tri765', 'grayscale', 'hs_rainbow']

export interface MapRenderRequest {
  centerRe: number
  centerIm: number
  scale: number          // height in complex units
  width: number
  height: number
  iterations: number
  variant?: Variant
  metric?: Metric
  colorMap?: ColorMap
  smooth?: boolean           // ln-smooth continuous coloring (μ = iter + 1 − log₂(log₂(|z|²)))
  bailout?: number
  julia?: boolean
  juliaRe?: number
  juliaIm?: number
  transitionTheta?: number  // 0..π; when set, transition kernel is used instead of the variant kernel
}

export interface MapRenderResponse {
  runId: string
  status: string
  artifactId: string
  imagePath: string
  generatedMs: number
  width: number
  height: number
  effective: Record<string, any>
}

export interface SpecialPoint {
  id: string
  family: string
  pointType: string
  k: number
  p: number
  real: number
  imag: number
  sourceMode: string
  createdAt: string
}

export interface LnMapRequest {
  centerRe: number
  centerIm: number
  widthS: number
  depthOctaves: number
  variant?: Variant
  colorMap?: ColorMap
  iterations?: number
}

export interface LnMapResponse {
  runId: string
  status: string
  artifactId: string
  imagePath: string
  widthS: number
  heightT: number
  depthOctaves: number
  generatedMs: number
}

export type HsStage = 'min_abs' | 'max_abs' | 'envelope' | 'min_pairwise_dist'

export interface HsMeshRequest {
  centerRe?: number
  centerIm?: number
  scale?: number
  width?: number
  height?: number
  resolution?: number
  metric?: HsStage
  variant?: Variant
  iterations?: number
}

export interface MeshResponse {
  runId: string
  status: string
  glbArtifactId: string
  glbUrl: string
  stlArtifactId: string
  stlUrl: string
  vertexCount: number
  triangleCount: number
  generatedMs?: number
  fieldMs?: number
  mcMs?: number
}

export interface TransitionMeshRequest {
  centerRe?: number
  centerIm?: number
  scale?: number
  resolution?: number
  theta?: number
  iso?: number
  iterations?: number
}

export interface TransitionVoxelRequest {
  centerX?: number
  centerY?: number
  centerZ?: number
  extent?: number
  resolution?: number   // default 64, max 256
  iso?: number
  iterations?: number
}

export interface TransitionVoxelResponse {
  runId: string
  status: string
  resolution: number
  isoLevel: number
  extent: number
  centerX: number
  centerY: number
  centerZ: number
  insideCount: number
  generatedMs: number
  fieldB64: string      // base64 Uint8Array, length = resolution³; 0=outside, 1-255=inside depth
}

export interface VideoZoomRequest {
  lnMapArtifactId: string
  fps?: number
  durationSec?: number
  width?: number
  height?: number
  startLnRadius?: number
  depthOctaves?: number
}

export interface VideoZoomResponse {
  runId: string
  status: string
  artifactId: string
  videoUrl: string
  downloadUrl: string
  frameCount: number
  fps: number
  durationSec: number
  width: number
  height: number
  generatedMs: number
}

export interface Hardware {
  cpuModel: string
  cpuLogicalCores: number
  cpuPhysicalCores: number
  memoryTotalMiB: number
  memoryAvailableMiB: number
  gpuModel: string
  gpuMemory: string
}

export interface RunRow {
  id: string
  module: string
  status: string
  startedAt: number
  finishedAt: number
  outputDir: string
}

export interface ArtifactRow {
  artifactId: string
  runId: string
  name: string
  kind: string
  sizeBytes: number
  downloadPath: string
  contentPath: string
}

// ---- API methods ----

export const api = {
  baseUrl: BASE,

  systemCheck: () => getJson<{ openmp: boolean; cuda: boolean }>('/api/system/check'),
  hardware:    () => getJson<Hardware>('/api/system/hardware'),

  mapRender:  (req: MapRenderRequest) => postJson<MapRenderResponse>('/api/map/render', req),
  lnMap:      (req: LnMapRequest)     => postJson<LnMapResponse>('/api/map/ln', req),

  specialPointsAuto: (k: number, p: number, pointType?: string) =>
    postJson<{ mode: string; k: number; p: number; count: number; points: SpecialPoint[] }>(
      '/api/special-points/auto', { k, p, pointType }),

  specialPointsSeed: (k: number, p: number, re: number, im: number) =>
    postJson<{ mode: string; converged: boolean; points: SpecialPoint[] }>(
      '/api/special-points/seed', { k, p, re, im }),

  specialPointsList: (family?: string) =>
    getJson<{ items: SpecialPoint[] }>(
      `/api/special-points${family ? `?family=${encodeURIComponent(family)}` : ''}`),

  hsMesh: (req: HsMeshRequest) => postJson<MeshResponse>('/api/hs/mesh', req),
  transitionMesh:   (req: TransitionMeshRequest)  => postJson<MeshResponse>('/api/transition/mesh', req),
  transitionVoxels: (req: TransitionVoxelRequest) => postJson<TransitionVoxelResponse>('/api/transition/voxels', req),
  videoZoom: (req: VideoZoomRequest) => postJson<VideoZoomResponse>('/api/video/zoom', req),

  runs: (limit = 50) => getJson<{ items: RunRow[] }>(`/api/runs?limit=${limit}`),

  artifacts: (kind?: string, runId?: string) => {
    const q = new URLSearchParams()
    if (kind)  q.set('kind',  kind)
    if (runId) q.set('runId', runId)
    const s = q.toString()
    return getJson<{ items: ArtifactRow[] }>(`/api/artifacts${s ? '?' + s : ''}`)
  },

  artifactContentUrl: (artifactId: string) =>
    `${BASE}/api/artifacts/content?artifactId=${encodeURIComponent(artifactId)}`,

  artifactDownloadUrl: (artifactId: string) =>
    `${BASE}/api/artifacts/download?artifactId=${encodeURIComponent(artifactId)}`,
}
