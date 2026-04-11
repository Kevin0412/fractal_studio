cmake_minimum_required(VERSION 3.18)
project(fractal_studio_backend LANGUAGES CXX)

# ---- Optional CUDA ----
# Enable CUDA if nvcc is available. The CUDA sources are only compiled when
# this succeeds, and the rest of the build is unaffected if CUDA is absent.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(HAS_CUDA ON)
    message(STATUS "CUDA enabled: ${CMAKE_CUDA_COMPILER}")
else()
    set(HAS_CUDA OFF)
    message(STATUS "CUDA not found — GPU kernels disabled")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP QUIET)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc videoio)

add_executable(fractal_studio_backend
    src/main.cpp

    # Core
    src/core/path_guard.cpp
    src/core/job_runner.cpp
    src/core/http_server.cpp
    src/core/hardware_probe.cpp
    src/core/db.cpp

    # API routes
    src/api/routes_runs.cpp
    src/api/routes_modules.cpp
    src/api/routes_map.cpp
    src/api/routes_ln.cpp
    src/api/routes_mesh.cpp
    src/api/routes_video.cpp
    src/api/routes_points.cpp
    src/api/routes_artifacts.cpp
    src/api/routes_benchmark.cpp

    # Native compute
    src/compute/map_kernel.cpp
    src/compute/map_kernel_avx512.cpp
    src/compute/transition_kernel.cpp
    src/compute/image_io.cpp
    src/compute/mesh_io.cpp
    src/compute/marching_cubes.cpp
    src/compute/transition_volume.cpp
    src/compute/hs/heightfield_mesh.cpp
    src/compute/newton/mandelbrot_sp.cpp
    src/compute/tile_scheduler.cpp

    # Hardware probes
    src/adapters/cuda_probe.cpp
    src/adapters/openmp_probe.cpp
)

# Conditionally compile the CUDA kernel and link it in.
if(HAS_CUDA)
    add_library(fractal_cuda STATIC
        src/compute/cuda/map_kernel.cu
    )
    target_include_directories(fractal_cuda PRIVATE
        src
        src/include
        ${OpenCV_INCLUDE_DIRS}
    )
    set_target_properties(fractal_cuda PROPERTIES
        CUDA_ARCHITECTURES "89"  # RTX 4070 (Ada Lovelace = sm_89)
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_compile_options(fractal_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3>
    )
    target_link_libraries(fractal_studio_backend PRIVATE fractal_cuda cuda)
    target_compile_definitions(fractal_studio_backend PRIVATE HAS_CUDA_KERNEL=1)
endif()

target_include_directories(fractal_studio_backend PRIVATE
    src/include
    src
    ${OpenCV_INCLUDE_DIRS}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(fractal_studio_backend PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled")
else()
    message(STATUS "OpenMP not found — parallel rendering will be single-threaded")
endif()

target_link_libraries(fractal_studio_backend PRIVATE
    sqlite3
    ${OpenCV_LIBS}
)

target_compile_options(fractal_studio_backend PRIVATE
    -Wall -Wextra -Wno-unused-parameter -O3
)

# AVX-512 kernel: compile with AVX-512F + IFMA52 intrinsics enabled when the
# compiler supports them. avx512_available() does a runtime CPUID check so the
# binary falls back to OpenMP on CPUs without AVX-512. On compilers/arches that
# don't support the flags (ARM, old GCC), the file still compiles — all
# intrinsic code is inside #if defined(__AVX512F__) guards and avx512_available()
# returns false unconditionally.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512f" COMPILER_HAS_AVX512F)
if(COMPILER_HAS_AVX512F)
    set_source_files_properties(src/compute/map_kernel_avx512.cpp PROPERTIES
        COMPILE_FLAGS "-mavx512f -mavx512dq -mavx512ifma")
    message(STATUS "AVX-512 compile flags enabled")
else()
    message(STATUS "Compiler lacks AVX-512 support — AVX-512 path disabled at compile time")
endif()
