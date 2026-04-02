export type ModuleRunResponse = {
  runId: string
  status: string
  artifactCount: number
  error?: string
}

export type SystemCheckResponse = {
  openmp: boolean
  cuda: boolean
}

export type SystemHardwareResponse = {
  cpuModel: string
  cpuLogicalCores: number
  cpuPhysicalCores: number
  memoryTotalMiB: number
  memoryAvailableMiB: number
  gpuModel: string
  gpuMemory: string
}

export type SpecialPoint = {
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

export type SpecialPointsListResponse = {
  items: SpecialPoint[]
}

export type SpecialPointsAutoRequest = {
  family: string
  pointType: string
  k: number
  p: number
}

export type SpecialPointsAutoResponse = {
  mode: 'auto'
  k: number
  p: number
  count: number
  points: Array<{ id: string; real: number; imag: number }>
}

export type SpecialPointsSeedRequest = {
  family: string
  k: number
  p: number
  maxIter: number
  seed: {
    re: number
    im: number
  }
}

export type SpecialPointsSeedResponse = {
  mode: 'seed'
  point: {
    id: string
    real: number
    imag: number
    digits: number
    residualReal: number
    residualImag: number
  }
}

export type ArtifactItem = {
  artifactId: string
  runId: string
  name: string
  kind: string
  downloadPath: string
}

export type ArtifactsListResponse = {
  items: ArtifactItem[]
}

export type MapRenderRequest = {
  centerRe: number
  centerIm: number
  scale: number
  width: number
  height: number
  variety: number
  iterations: number
  colorMap: string
  mode?: 'mandelbrot' | 'julia'
  juliaRe?: number
  juliaIm?: number
}

export type MapRenderResponse = {
  runId: string
  status: string
  artifactId: string
  imagePath: string
  effective: {
    centerRe: number
    centerIm: number
    scale: number
    width: number
    height: number
    variety: number
    mode?: 'mandelbrot' | 'julia'
    juliaRe?: number
    juliaIm?: number
  }
  notes: {
    iterationsApplied: boolean
    colorMapApplied: boolean
  }
}

const baseUrl = (import.meta.env.VITE_BACKEND_URL as string | undefined) ?? 'http://127.0.0.1:18080'

export async function invokeModule(moduleName: string): Promise<ModuleRunResponse> {
  const res = await fetch(`${baseUrl}/api/modules/${moduleName}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({}),
  })

  if (!res.ok) {
    throw new Error(`request failed: ${res.status}`)
  }

  return (await res.json()) as ModuleRunResponse
}

export async function getSystemCheck(): Promise<SystemCheckResponse> {
  const res = await fetch(`${baseUrl}/api/system/check`)
  if (!res.ok) {
    throw new Error(`request failed: ${res.status}`)
  }
  return (await res.json()) as SystemCheckResponse
}

export async function getSystemHardware(): Promise<SystemHardwareResponse> {
  const res = await fetch(`${baseUrl}/api/system/hardware`)
  if (!res.ok) {
    throw new Error(`request failed: ${res.status}`)
  }
  return (await res.json()) as SystemHardwareResponse
}

export async function postSpecialPointsAuto(payload: SpecialPointsAutoRequest): Promise<SpecialPointsAutoResponse> {
  const res = await fetch(`${baseUrl}/api/special-points/auto`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    throw new Error(`request failed: ${res.status}`)
  }
  return (await res.json()) as SpecialPointsAutoResponse
}

export async function postSpecialPointsSeed(payload: SpecialPointsSeedRequest): Promise<SpecialPointsSeedResponse> {
  const res = await fetch(`${baseUrl}/api/special-points/seed`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    throw new Error(`request failed: ${res.status}`)
  }
  return (await res.json()) as SpecialPointsSeedResponse
}

export async function getSpecialPoints(params?: { family?: string; k?: number; p?: number }): Promise<SpecialPointsListResponse> {
  const query = new URLSearchParams()
  if (params?.family != null && params.family !== '') {
    query.set('family', params.family)
  }
  if (params?.k != null) {
    query.set('k', String(params.k))
  }
  if (params?.p != null) {
    query.set('p', String(params.p))
  }
  const suffix = query.toString().length > 0 ? `?${query.toString()}` : ''
  const res = await fetch(`${baseUrl}/api/special-points${suffix}`)
  if (!res.ok) {
    throw new Error(`request failed: ${res.status}`)
  }
  return (await res.json()) as SpecialPointsListResponse
}

export async function getArtifacts(params?: { kind?: string; runId?: string }): Promise<ArtifactsListResponse> {
  const query = new URLSearchParams()
  if (params?.kind != null && params.kind !== '') {
    query.set('kind', params.kind)
  }
  if (params?.runId != null && params.runId !== '') {
    query.set('runId', params.runId)
  }
  const suffix = query.toString().length > 0 ? `?${query.toString()}` : ''
  const res = await fetch(`${baseUrl}/api/artifacts${suffix}`)
  if (!res.ok) {
    throw new Error(`request failed: ${res.status}`)
  }
  return (await res.json()) as ArtifactsListResponse
}

export function artifactDownloadUrl(downloadPath: string): string {
  return `${baseUrl}${downloadPath}`
}

export function artifactContentUrl(artifactId: string): string {
  const query = new URLSearchParams({ artifactId })
  return `${baseUrl}/api/artifacts/content?${query.toString()}`
}

export async function postMapRender(payload: MapRenderRequest): Promise<MapRenderResponse> {
  const res = await fetch(`${baseUrl}/api/map/render`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    throw new Error(`request failed: ${res.status}`)
  }
  return (await res.json()) as MapRenderResponse
}
