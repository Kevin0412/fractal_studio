cmake_minimum_required(VERSION 3.16)
project(fractal_studio_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)

add_executable(fractal_studio_backend
  src/main.cpp
  src/core/path_guard.cpp
  src/core/job_runner.cpp
  src/core/http_server.cpp
  src/core/hardware_probe.cpp
  src/core/db.cpp
  src/api/routes_runs.cpp
  src/api/routes_modules.cpp
  src/api/routes_points.cpp
  src/api/routes_map.cpp
  src/api/routes_artifacts.cpp
  src/adapters/legacy_variant_adapter.cpp
  src/adapters/legacy_map_adapter.cpp
  src/adapters/legacy_newton_adapter.cpp
  src/adapters/legacy_transition_adapter.cpp
  src/adapters/legacy_hidden_structure_adapter.cpp
  src/adapters/legacy_3d_family_adapter.cpp
  src/adapters/legacy_stl_adapter.cpp
  src/adapters/cuda_probe.cpp
  src/adapters/openmp_probe.cpp
  src/report/conclusion_builder.cpp
)

target_include_directories(fractal_studio_backend PRIVATE src/include)
target_link_libraries(fractal_studio_backend PRIVATE OpenMP::OpenMP_CXX sqlite3)
