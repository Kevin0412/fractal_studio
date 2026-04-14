# Fractal Studio / 分形工作室

> **EN**: An interactive fractal explorer with a native C++ compute backend and a Vue 3 instrumentation-style frontend.
> **中文**：交互式分形探索工具，后端为原生 C++ 计算引擎，前端为 Vue 3 暗色仪表盘风格界面。

---

## Features / 功能特性

- **10 Mandelbrot-family variants** — all rendered natively in C++, no external shell-outs
  **10 种 Mandelbrot 族变体** — 全部原生 C++ 渲染，无外部进程调用
- **Multiple metrics** — escape time, min |z|, max |z|, envelope, min pairwise orbit distance
  **多种度量指标** — 逃逸时间、min |z|、max |z|、包络、最小轨道距
- **Dual-pane Julia explorer** — left pane picks c by click and recenters; right pane shows the Julia set J(c) with independent viewport
  **双窗格 Julia 探索器** — 左侧点选 c 值并重定中心，右侧独立视口展示对应 Julia 集
- **Mandelbrot ↔ Burning Ship transition** — continuous 3D rotation in the (x, y, z) iteration space, controlled by a θ slider
  **Mandelbrot ↔ Burning Ship 过渡** — 在 (x, y, z) 迭代空间通过 θ 滑块连续旋转
- **3D viewer** — HS height fields and transition volumes as Minecraft-style voxels or smooth marching-cubes meshes in three.js
  **三维查看器** — HS 高度场与过渡体积以 Minecraft 体素或平滑 MC 网格在 three.js 中渲染
- **Ln-map zoom video export** — render a single logarithmic strip image once, generate arbitrarily long smooth zoom videos from it (zero fractal re-computation per video)
  **对数图缩放视频导出** — 仅渲染一次对数条带图，可无限次生成不同时长的缩放视频（无需重复计算分形）
- **Dark instrumentation UI** — near-black canvas, amber accent, monospace numerics, bilingual EN/中文
  **暗色仪表盘界面** — 近黑色画布、琥珀色主题、等宽数字显示，中英文双语
- **Multi-engine rendering** — select CUDA, AVX-512, OpenMP, or hybrid per render
  **多引擎渲染** — 可选 CUDA、AVX-512、OpenMP 或混合模式

---

## Engine Support Matrix / 引擎支持矩阵

| Engine / 引擎 | All 10 variants | Julia mode | Escape | Min/Max \|z\| · Envelope | Min pairwise dist | Fixed-point fx64 |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|
| **OpenMP**       | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **CUDA**         | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| **AVX-512 fp64** | ✓ | ✓ | ✓ | ✓ | — | — |
| **AVX-512 fx64** | Mandelbrot only | ✓ | ✓ | — | — | ✓ |
| **Hybrid**       | ✓ | ✓ | ✓ | ✓ | — | ✓ |

- **auto** mode picks CUDA → AVX-512 → OpenMP in priority order based on runtime availability and metric support.
  **auto** 模式按 CUDA → AVX-512 → OpenMP 优先级自动选择，取决于运行时可用性与度量支持情况。
- **fx64** (1s·6i·57f fixed-point) engages automatically when scale < 1e-13 for ~17 decimal digits of precision vs fp64's ~15.
  **fx64**（1符·6整·57小数定点数）在 scale < 1e-13 时自动启用，精度约 17 位，优于 fp64 的 15 位。
- **CUDA Graphs** are used for launch-overhead-free tile streaming when the GPU is active.
  GPU 活跃时使用 **CUDA Graphs** 实现零启动开销的分块流式渲染。
- **MinPairwiseDist** (O(n²) orbit buffer) runs only on OpenMP; the vectorized paths skip it.
  **最小轨道距**（O(n²)轨道缓冲）仅在 OpenMP 路径支持，向量化路径跳过。

---

## Variants / 变体

| API name | Display name / 显示名称 | Formula / 公式 |
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

## Mandelbrot ↔ Burning Ship Transition / M↔B 过渡

The transition uses a 3D iteration space (the author's original research):
过渡使用作者原创的三维迭代空间：

```
x' = x² − y² − z² + x₀
y' = 2xy + y₀
z' = 2|xz| + z₀
```

A 2D map at rotation angle θ around the x-axis maps screen pixel (u, v) to seed (x₀, y₀, z₀) = (u, v·cosθ, v·sinθ).
绕 x 轴旋转角度 θ 时，屏幕像素 (u, v) 对应种子 (x₀, y₀, z₀) = (u, v·cosθ, v·sinθ)。

- θ = 0 → standard Mandelbrot (xy-plane) / 标准 Mandelbrot（xy 平面）
- θ = π/2 → Burning Ship (xz-plane) / Burning Ship（xz 平面）
- 0 < θ < π/2 → continuous morphing bridge / 连续过渡桥接

The 3D viewer can render the full volume as a Minecraft-style voxel model (only exposed faces are sent — no hidden geometry) or as a smooth marching-cubes mesh.
三维查看器可将完整体积渲染为 Minecraft 风格体素模型（仅发送暴露面，无隐藏几何）或平滑 MC 网格。

---

## Zoom Video Export / 缩放视频导出

The ln-map approach renders each zoom level exactly once:
对数图方法对每个缩放层次只渲染一次：

1. **Export ln-map / 导出对数图** — renders a tall strip image in (θ, ln r) coordinates. One render covers the full zoom depth.
   在 (θ, ln r) 坐标系渲染高条带图，单次渲染覆盖全部缩放深度。
2. **Export zoom video / 导出缩放视频** — slides a window down the strip and exp-warps each frame into Cartesian coordinates, then pipes raw frames to `ffmpeg`.
   沿条带滑动窗口，对每帧进行 exp 变形还原为直角坐标，再通过管道送入 `ffmpeg` 编码。

A 40-octave zoom video at 1080p/30fps requires only one ln-map render (~seconds) rather than thousands of individual frame renders.
40 倍频程的 1080p/30fps 缩放视频只需一次对数图渲染（数秒），而非数千次独立帧渲染。

---

## Build / 构建

### Backend / 后端

Requirements: CMake ≥ 3.20, C++20 compiler, OpenCV, sqlite3. Optional: CUDA toolkit ≥ 12, OpenMP.
依赖：CMake ≥ 3.20，C++20 编译器，OpenCV，sqlite3。可选：CUDA toolkit ≥ 12，OpenMP。

```bash
cd fractal_studio/backend
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/fractal_studio_backend   # starts on :18080 / 监听 :18080
```

CUDA support is detected automatically if `nvcc` is on PATH. AVX-512 support is detected at runtime via CPUID.
若 `nvcc` 在 PATH 中则自动检测并启用 CUDA；AVX-512 通过运行时 CPUID 检测。

### Frontend / 前端

Requirements: Node.js ≥ 18.
依赖：Node.js ≥ 18。

```bash
cd fractal_studio/frontend
npm install
npm run dev      # dev server (Vite, port 5173) / 开发服务器（Vite，端口 5173）
npm run build    # production build into dist/ / 生产构建输出至 dist/
```

Set `VITE_BACKEND_URL=http://host:18080` to point the frontend at a non-local backend.
设置 `VITE_BACKEND_URL=http://host:18080` 可将前端指向非本地后端。

---

## API Summary / API 摘要

All endpoints are `POST /api/...` with JSON body or `GET /api/...` with query params.
所有端点均为带 JSON 请求体的 `POST /api/...` 或带查询参数的 `GET /api/...`。

| Endpoint | Description / 说明 |
|---|---|
| `POST /api/map/render` | Render a fractal map image / 渲染分形图 |
| `POST /api/map/ln` | Render a logarithmic strip image for zoom video / 渲染对数条带图 |
| `POST /api/special-points/auto` | Auto-solve periodic/preperiodic points (k, p) / 自动求解周期/前周期点 |
| `POST /api/special-points/seed` | Newton-converge from a seed point / 从种子点牛顿迭代收敛 |
| `GET  /api/special-points` | List computed special points / 列出已计算特殊点 |
| `POST /api/hs/mesh` | Compute HS height field + marching-cubes mesh / 计算 HS 高度场网格 |
| `POST /api/transition/mesh` | Compute 3D transition volume + marching-cubes mesh / 计算三维过渡体积网格 |
| `POST /api/transition/voxels` | Compute voxel body (exposed faces only, backend-culled) / 计算体素体（仅暴露面，后端剔除） |
| `POST /api/video/zoom` | Generate zoom video from an existing ln-map artifact / 从对数图生成缩放视频 |
| `GET  /api/runs` | Run history / 运行历史 |
| `GET  /api/artifacts` | Artifact list + download URLs / 产物列表与下载链接 |
| `GET  /api/system/hardware` | CPU/GPU/RAM info / 硬件信息 |
| `GET  /api/system/check` | OpenMP/CUDA availability / OpenMP/CUDA 可用性 |

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

## Architecture / 架构

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

## License / 许可

Internal research tool. / 内部研究工具。
