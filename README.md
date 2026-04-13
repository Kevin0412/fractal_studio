# Fractal Studio

An interactive fractal explorer with a native C++ compute backend and a Vue 3 instrumentation-style frontend.

---

## Features

- **10 Mandelbrot-family variants** — all rendered natively in C++, no external shell-outs
- **Multiple metrics** — escape time, min |z|, max |z|, envelope, min pairwise orbit distance
- **Dual-pane Julia explorer** — left pane picks c by click and recenters; right pane shows the Julia set J(c) with independent viewport
- **Mandelbrot ↔ Burning Ship transition** — continuous 3D rotation in the (x, y, z) iteration space, controlled by a θ slider
- **3D mesh viewer** — hidden-structure (HS) height fields and transition volumes rendered as glTF meshes in three.js
- **Ln-map zoom video export** — render a single logarithmic strip image once, generate arbitrarily long smooth zoom videos from it (zero fractal re-computation per video)
- **Dark instrumentation UI** — near-black canvas, amber accent, monospace numerics, bilingual EN/中文
- **Multi-engine rendering** — select CUDA, AVX-512, OpenMP, or hybrid per render

---

## Engine Support Matrix

| Engine | All 10 variants | Julia mode | Escape | Min/Max |z| · Envelope | Min pairwise dist | Fixed-point fx64 |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|
| **OpenMP**       | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **CUDA**         | ✓ | ✓ | ✓ | ✓ | — | ✓ (all variants) |
| **AVX-512 fp64** | ✓ | ✓ | ✓ | ✓ | — | — |
| **AVX-512 fx64** | Mandelbrot only | ✓ | ✓ | — | — | ✓ |
| **Hybrid**       | ✓ | ✓ | ✓ | ✓ | — | ✓ |

- **auto** mode picks CUDA → AVX-512 → OpenMP in priority order based on runtime availability and metric support.
- **fx64** (1s·6i·57f fixed-point) engages automatically when scale < 1e-13 for ~17 decimal digits of precision vs fp64's ~15.
- **CUDA Graphs** are used for launch-overhead-free tile streaming when the GPU is active.
- **MinPairwiseDist** (O(n²) orbit buffer) runs only on OpenMP; the vectorized paths skip it.

---

## Variants

| API name | Display name | Formula |
|---|---|---|
| `mandelbrot`   | Mandelbrot        | z² + c |
| `tricorn`      | Tricorn           | conj(z)² + c |
| `burning_ship` | Burning Ship      | (\|Re z\| + \|Im z\|·i)² + c |
| `celtic`       | Celtic            | (Re z + \|Im z\|·i)² + c |
| `heart`        | Heart             | (\|Re z\| − Im z·i)² + c |
| `buffalo`      | Buffalo           | z²→\|Re(z²)\| + Im(z²)·i + c |
| `perp_buffalo` | Perp. Buffalo     | z²→\|Re(z²)\| − Im(z²)·i + c |
| `celtic_ship`  | Celtic Ship       | z²→\|Re(z²)\| + \|Im(z²)\|·i + c |
| `mandelceltic` | Mandelceltic      | (Re+\|Im\|·i)²→\|Re\|+Im·i + c |
| `perp_ship`    | Perp. Ship        | (\|Re\|+Im·i)²→\|Re\|−Im·i + c |

---

## Mandelbrot ↔ Burning Ship Transition

The transition uses a 3D iteration space (the user's original innovation):

```
x' = x² − y² − z² + x₀
y' = 2xy + y₀
z' = 2|xz| + z₀
```

A 2D map at rotation angle θ around the x-axis maps screen pixel (u, v) to seed (x₀, y₀, z₀) = (u, v·cosθ, v·sinθ).

- θ = 0 → standard Mandelbrot (xy-plane)
- θ = π/2 → Burning Ship (xz-plane)
- 0 < θ < π/2 → continuous morphing bridge

The same volume can be rendered as a 3D mesh (marching cubes) in the 3D viewer.

---

## Zoom Video Export

The ln-map approach renders each zoom level exactly once:

1. **Export ln-map** — renders a tall strip image in (θ, ln r) coordinates. One render covers the full zoom depth.
2. **Export zoom video** — slides a window down the strip and exp-warps each frame into Cartesian coordinates, then pipes raw frames to `ffmpeg`.

A 40-octave zoom video at 1080p/30fps requires only one ln-map render (~seconds) rather than thousands of individual frame renders.

---

## Build

### Backend

Requirements: CMake ≥ 3.20, C++20 compiler, OpenCV, sqlite3. Optional: CUDA toolkit ≥ 12, OpenMP.

```bash
cd fractal_studio/backend
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/fractal_studio_backend   # starts on :18080
```

CUDA support is detected automatically if `nvcc` is on PATH. AVX-512 support is detected at runtime via CPUID.

### Frontend

Requirements: Node.js ≥ 18.

```bash
cd fractal_studio/frontend
npm install
npm run dev      # dev server (Vite, port 5173)
npm run build    # production build into dist/
```

Set `VITE_BACKEND_URL=http://host:18080` to point the frontend at a non-local backend.

---

## API Summary

All endpoints are `POST /api/...` with JSON body or `GET /api/...` with query params.

| Endpoint | Description |
|---|---|
| `POST /api/map/render` | Render a fractal map image |
| `POST /api/map/ln` | Render a logarithmic strip image for zoom video |
| `POST /api/special-points/auto` | Auto-solve periodic/preperiodic points (k, p) |
| `POST /api/special-points/seed` | Newton-converge from a seed point |
| `GET  /api/special-points` | List computed special points |
| `POST /api/hs/mesh` | Compute HS height field + marching-cubes mesh |
| `POST /api/transition/mesh` | Compute 3D transition volume + marching-cubes mesh |
| `POST /api/video/zoom` | Generate zoom video from an existing ln-map artifact |
| `GET  /api/runs` | Run history |
| `GET  /api/artifacts` | Artifact list + download URLs |
| `GET  /api/system/hardware` | CPU/GPU/RAM info |
| `GET  /api/system/check` | OpenMP/CUDA availability |

### `POST /api/map/render` key fields

```json
{
  "centerRe": -0.75, "centerIm": 0.0, "scale": 3.0,
  "width": 1024, "height": 1024, "iterations": 1024,
  "variant": "mandelbrot",
  "metric": "escape",
  "colorMap": "classic_cos",
  "smooth": false,
  "julia": false, "juliaRe": 0.0, "juliaIm": 0.0,
  "transitionTheta": null,
  "engine": "auto",
  "scalarType": "auto"
}
```

`engine`: `"auto"` | `"cuda"` | `"avx512"` | `"hybrid"` | `"openmp"`  
`scalarType`: `"auto"` | `"fp64"` | `"fx64"`

---

## Architecture

```
fractal_studio/
  backend/
    src/
      compute/
        variants.hpp          # 10 variant step functions (tag-dispatched)
        escape_time.hpp       # per-pixel iteration core (all metrics)
        map_kernel.cpp        # OpenMP renderer
        map_kernel_avx512.cpp # AVX-512 renderer (fp64 + fx64)
        cuda/
          map_kernel.cu       # CUDA renderer (all variants, Julia, CUDA Graphs)
          fx64.cuh            # fixed-point int64 on CUDA
        hs/
          heightfield_mesh.cpp  # HS field → triangle mesh
        marching_cubes.cpp    # general MC for transition volume
        transition_volume.cpp # 3D transition iteration
        newton/
          mandelbrot_sp.cpp   # Newton solver for special points
        scalar/
          fx64.hpp            # 1s·6i·57f fixed-point type
        colormap.cpp          # 6 colormaps
        mesh_io.cpp           # STL + glTF (GLB) writers
        video/
          ln_warp.cpp         # frame generator for zoom video
      api/
        routes_map.cpp
        routes_hs.cpp
        routes_mesh.cpp
        routes_video.cpp
        routes_points.cpp
        routes_artifacts.cpp
      core/
        http_server.cpp       # minimal socket HTTP server
        db.cpp                # sqlite3 run/artifact store
        job_runner.cpp        # async job execution
  frontend/
    src/
      views/
        MapView.vue           # map explorer + Julia dual-pane + video export
        ThreeDView.vue        # HS + transition 3D viewer
        PointsView.vue        # special points compute + import to map
        RunsView.vue          # run history + artifact downloads
        SystemView.vue        # hardware info
      components/
        MapCanvas.vue         # WebGL tile renderer + pan/zoom
        ThreeDViewer.vue      # three.js orbit viewer
        StatusRail.vue        # live CPU/GPU/render stats
        NavRail.vue           # left nav (EN/ZH + dark/light toggle)
      api.ts                  # typed backend client
      i18n.ts                 # EN/中文 reactive store
```

---

## License

Internal research tool.
