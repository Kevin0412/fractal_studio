// compute/ln_map.cpp

#include "ln_map.hpp"

#include "engine_select.hpp"
#include "escape_time.hpp"
#include "map_kernel_avx2.hpp"
#include "map_kernel_avx512.hpp"
#include "parallel.hpp"

#if defined(HAS_CUDA_KERNEL)
#  include "cuda/ln_map.cuh"
#  define USE_CUDA_LN_MAP 1
#else
#  define USE_CUDA_LN_MAP 0
#endif

#include <opencv2/core.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace fsd::compute {
namespace {

constexpr double TAU = 6.283185307179586;
constexpr double LN_FOUR = 1.3862943611198906;

struct TrigColumns {
    std::vector<double> cos_col;
    std::vector<double> sin_col;
};

void ensure_ln_out(const LnMapParams& p, cv::Mat& out) {
    if (out.empty() || out.rows != p.height_t || out.cols != p.width_s || out.type() != CV_8UC3) {
        out.create(p.height_t, p.width_s, CV_8UC3);
    }
}

std::pair<int, int> clamp_rows(const LnMapParams& p, int row_start, int row_count) {
    const int start = std::max(0, std::min(row_start, p.height_t));
    const int end = std::max(start, std::min(p.height_t, start + std::max(0, row_count)));
    return {start, end};
}

TrigColumns make_trig_columns(int s) {
    TrigColumns cols;
    cols.cos_col.resize(static_cast<size_t>(s));
    cols.sin_col.resize(static_cast<size_t>(s));
    for (int x = 0; x < s; x++) {
        const double th = TAU * static_cast<double>(x) / static_cast<double>(s);
        cols.cos_col[static_cast<size_t>(x)] = std::cos(th);
        cols.sin_col[static_cast<size_t>(x)] = std::sin(th);
    }
    return cols;
}

template <Variant V>
void render_ln_variant_openmp_rows_impl(
    const LnMapParams& p,
    cv::Mat& out,
    int row_start,
    int row_end,
    const TrigColumns& cols,
    bool threaded,
    const LnMapProgress& on_row_done
) {
    const Cx<double> c_julia{p.julia_re, p.julia_im};
    const int s = p.width_s;
    std::atomic<int> rows_done{0};

    auto render_row = [&](int row) {
        uint8_t* rowp = out.ptr<uint8_t>(row);
        const double k = LN_FOUR - static_cast<double>(row) * TAU / static_cast<double>(s);
        const double r_mag = std::exp(k);
        for (int x = 0; x < s; x++) {
            const double pre = p.center_re + r_mag * cols.cos_col[static_cast<size_t>(x)];
            const double pim = p.center_im + r_mag * cols.sin_col[static_cast<size_t>(x)];
            Cx<double> z0, c;
            if (p.julia) {
                z0 = {pre, pim};
                c = c_julia;
            } else {
                z0 = {0.0, 0.0};
                c = {pre, pim};
            }
            const IterResult ir = iterate_masked<
                IterResultField::Iter | IterResultField::Escaped,
                V, double>(z0, c, p.iterations, p.bailout, p.bailout_sq);
            uint8_t* px = rowp + 3 * x;
            const int it = ir.escaped ? ir.iter : p.iterations;
            colorize_escape_bgr(it, p.iterations, p.colormap, 0.0, false, px[0], px[1], px[2]);
        }
        if (on_row_done) {
            const int done = rows_done.fetch_add(1, std::memory_order_relaxed) + 1;
            if (done == row_end - row_start || (done % 16) == 0) on_row_done(done);
        }
    };

    if (threaded) {
        const int thread_count = default_render_threads();
        #pragma omp parallel for num_threads(thread_count) schedule(dynamic, 8)
        for (int row = row_start; row < row_end; row++) {
            render_row(row);
        }
    } else {
        for (int row = row_start; row < row_end; row++) {
            render_row(row);
        }
    }
}

void dispatch_openmp_rows_impl(
    const LnMapParams& p,
    cv::Mat& out,
    int row_start,
    int row_end,
    const TrigColumns& cols,
    bool threaded,
    const LnMapProgress& on_row_done
) {
    switch (p.variant) {
        case Variant::Mandelbrot: render_ln_variant_openmp_rows_impl<Variant::Mandelbrot>(p, out, row_start, row_end, cols, threaded, on_row_done); break;
        case Variant::Tri:        render_ln_variant_openmp_rows_impl<Variant::Tri>       (p, out, row_start, row_end, cols, threaded, on_row_done); break;
        case Variant::Boat:       render_ln_variant_openmp_rows_impl<Variant::Boat>      (p, out, row_start, row_end, cols, threaded, on_row_done); break;
        case Variant::Duck:       render_ln_variant_openmp_rows_impl<Variant::Duck>      (p, out, row_start, row_end, cols, threaded, on_row_done); break;
        case Variant::Bell:       render_ln_variant_openmp_rows_impl<Variant::Bell>      (p, out, row_start, row_end, cols, threaded, on_row_done); break;
        case Variant::Fish:       render_ln_variant_openmp_rows_impl<Variant::Fish>      (p, out, row_start, row_end, cols, threaded, on_row_done); break;
        case Variant::Vase:       render_ln_variant_openmp_rows_impl<Variant::Vase>      (p, out, row_start, row_end, cols, threaded, on_row_done); break;
        case Variant::Bird:       render_ln_variant_openmp_rows_impl<Variant::Bird>      (p, out, row_start, row_end, cols, threaded, on_row_done); break;
        case Variant::Mask:       render_ln_variant_openmp_rows_impl<Variant::Mask>      (p, out, row_start, row_end, cols, threaded, on_row_done); break;
        case Variant::Ship:       render_ln_variant_openmp_rows_impl<Variant::Ship>      (p, out, row_start, row_end, cols, threaded, on_row_done); break;
        case Variant::SinZ:       render_ln_variant_openmp_rows_impl<Variant::SinZ>      (p, out, row_start, row_end, cols, threaded, on_row_done); break;
        case Variant::CosZ:       render_ln_variant_openmp_rows_impl<Variant::CosZ>      (p, out, row_start, row_end, cols, threaded, on_row_done); break;
        case Variant::ExpZ:       render_ln_variant_openmp_rows_impl<Variant::ExpZ>      (p, out, row_start, row_end, cols, threaded, on_row_done); break;
        case Variant::SinhZ:      render_ln_variant_openmp_rows_impl<Variant::SinhZ>     (p, out, row_start, row_end, cols, threaded, on_row_done); break;
        case Variant::CoshZ:      render_ln_variant_openmp_rows_impl<Variant::CoshZ>     (p, out, row_start, row_end, cols, threaded, on_row_done); break;
        case Variant::TanZ:       render_ln_variant_openmp_rows_impl<Variant::TanZ>      (p, out, row_start, row_end, cols, threaded, on_row_done); break;
        case Variant::Custom:     throw std::runtime_error("ln-map custom variants are not supported");
    }
}

#if USE_CUDA_LN_MAP
fsd_cuda::CudaLnMapParams make_cuda_params(const LnMapParams& p) {
    fsd_cuda::CudaLnMapParams cp;
    cp.julia = p.julia;
    cp.center_re = p.center_re;
    cp.center_im = p.center_im;
    cp.julia_re = p.julia_re;
    cp.julia_im = p.julia_im;
    cp.width_s = p.width_s;
    cp.height_t = p.height_t;
    cp.iterations = p.iterations;
    cp.bailout = p.bailout;
    cp.bailout_sq = p.bailout_sq;
    cp.variant_id = static_cast<int>(p.variant);
    cp.colormap_id = static_cast<int>(p.colormap);
    return cp;
}

int cuda_progress_chunk_rows(const LnMapParams& p) {
    const int h = std::max(1, p.height_t);
    const bool low_end = runtime_capabilities().cuda_low_end;
    const int target_updates = low_end ? 48 : 64;
    const int min_rows = low_end ? 8 : 16;
    const int max_rows = low_end ? 64 : 256;
    return std::clamp((h + target_updates - 1) / target_updates, min_rows, max_rows);
}

LnMapStats render_ln_map_cuda_with_progress(
    const LnMapParams& p,
    cv::Mat& out,
    const LnMapProgress& on_row_done
) {
    ensure_ln_out(p, out);
    const fsd_cuda::CudaLnMapParams cp = make_cuda_params(p);
    const int chunk_rows = cuda_progress_chunk_rows(p);

    double elapsed_ms = 0.0;
    for (int row0 = 0; row0 < p.height_t; row0 += chunk_rows) {
        const int rows = std::min(chunk_rows, p.height_t - row0);
        const auto stats = fsd_cuda::cuda_render_ln_map_rows(cp, out, row0, rows);
        elapsed_ms += stats.elapsed_ms;
        if (on_row_done) on_row_done(row0 + rows);
    }

    return {elapsed_ms, p.width_s * p.height_t, "cuda", "fp64"};
}
#endif

bool should_try_cuda(const LnMapParams& p) {
#if USE_CUDA_LN_MAP
    if (!ln_map_variant_supported_by_simd(p.variant)) return false;
    if (p.engine == "cuda" || p.engine == "hybrid") return fsd_cuda::cuda_ln_map_available();
    if (p.engine != "auto") return false;
    const auto caps = runtime_capabilities();
    return caps.cuda_runtime && !caps.cuda_low_end;
#else
    (void)p;
    return false;
#endif
}

bool should_try_avx512(const LnMapParams& p) {
    if (!ln_map_variant_supported_by_simd(p.variant)) return false;
    if (p.engine == "avx512") return avx512_available();
    if (p.engine != "auto") return false;
    return avx512_available();
}

bool should_try_avx2(const LnMapParams& p) {
    if (!ln_map_variant_supported_by_simd(p.variant)) return false;
    if (p.engine == "avx2") return ln_map_avx2_available();
    if (p.engine != "auto") return false;
    return ln_map_avx2_available();
}

enum class LnCpuBackend {
    Avx512,
    Avx2,
    Openmp,
};

LnCpuBackend select_ln_cpu_backend(const LnMapParams& p) {
    if (ln_map_variant_supported_by_simd(p.variant) && avx512_available()) return LnCpuBackend::Avx512;
    if (ln_map_variant_supported_by_simd(p.variant) && ln_map_avx2_available()) return LnCpuBackend::Avx2;
    return LnCpuBackend::Openmp;
}

const char* ln_cpu_backend_name(LnCpuBackend backend) {
    switch (backend) {
        case LnCpuBackend::Avx512: return "avx512";
        case LnCpuBackend::Avx2: return "avx2";
        case LnCpuBackend::Openmp: return "openmp";
    }
    return "openmp";
}

void render_ln_cpu_rows_serial(
    const LnMapParams& p,
    cv::Mat& out,
    int row_start,
    int row_count,
    LnCpuBackend backend
) {
    if (row_count <= 0) return;
    switch (backend) {
        case LnCpuBackend::Avx512:
            render_ln_map_avx512_rows(p, out, row_start, row_count, nullptr);
            return;
        case LnCpuBackend::Avx2:
            render_ln_map_avx2_rows(p, out, row_start, row_count, nullptr);
            return;
        case LnCpuBackend::Openmp: {
            const auto [start, end] = clamp_rows(p, row_start, row_count);
            const TrigColumns cols = make_trig_columns(p.width_s);
            dispatch_openmp_rows_impl(p, out, start, end, cols, false, nullptr);
            return;
        }
    }
}

LnMapStats render_ln_map_hybrid(const LnMapParams& p, cv::Mat& out, const LnMapProgress& on_row_done) {
    if (!ln_map_variant_supported_by_simd(p.variant)) {
        return render_ln_map_openmp(p, out, on_row_done);
    }

    ensure_ln_out(p, out);
    const auto t0 = std::chrono::steady_clock::now();
    const LnCpuBackend cpu_backend = select_ln_cpu_backend(p);
    std::atomic<int> next_row{0};
    std::atomic<int> rows_done{0};
    std::atomic<int> gpu_rows{0};
    std::atomic<int> cpu_rows{0};
    std::atomic<bool> gpu_available{true};
    std::mutex progress_mu;
    const int h = p.height_t;
    const int gpu_batch = runtime_capabilities().cuda_low_end ? 16 : 48;
    const int cpu_batch = cpu_backend == LnCpuBackend::Openmp ? 2 : 4;

    auto notify = [&](int delta) {
        if (!on_row_done || delta <= 0) return;
        const int done = rows_done.fetch_add(delta, std::memory_order_relaxed) + delta;
        std::lock_guard<std::mutex> lock(progress_mu);
        on_row_done(std::min(done, h));
    };

    auto render_cpu = [&](int row0, int rows) {
        render_ln_cpu_rows_serial(p, out, row0, rows, cpu_backend);
        cpu_rows.fetch_add(rows, std::memory_order_relaxed);
        notify(rows);
    };

#if USE_CUDA_LN_MAP
    std::thread gpu_thread;
    if (fsd_cuda::cuda_ln_map_available()) {
        gpu_thread = std::thread([&]() {
            fsd_cuda::CudaLnMapParams cp = make_cuda_params(p);
            while (true) {
                const int row0 = next_row.fetch_add(gpu_batch, std::memory_order_relaxed);
                if (row0 >= h) break;
                const int rows = std::min(gpu_batch, h - row0);
                if (!gpu_available.load(std::memory_order_relaxed)) {
                    render_cpu(row0, rows);
                    continue;
                }
                try {
                    fsd_cuda::cuda_render_ln_map_rows(cp, out, row0, rows);
                    gpu_rows.fetch_add(rows, std::memory_order_relaxed);
                    notify(rows);
                } catch (...) {
                    gpu_available.store(false, std::memory_order_relaxed);
                    render_cpu(row0, rows);
                }
            }
        });
    } else {
        gpu_available.store(false, std::memory_order_relaxed);
    }
#else
    gpu_available.store(false, std::memory_order_relaxed);
#endif

    const int cpu_threads = std::max(1, std::min(default_render_threads(), 8));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(cpu_threads));
    for (int i = 0; i < cpu_threads; ++i) {
        workers.emplace_back([&]() {
            while (true) {
                const int row0 = next_row.fetch_add(cpu_batch, std::memory_order_relaxed);
                if (row0 >= h) break;
                const int rows = std::min(cpu_batch, h - row0);
                render_cpu(row0, rows);
            }
        });
    }

    for (auto& worker : workers) worker.join();
#if USE_CUDA_LN_MAP
    if (gpu_thread.joinable()) gpu_thread.join();
#endif

    const auto t1 = std::chrono::steady_clock::now();
    LnMapStats stats;
    stats.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    stats.pixel_count = p.width_s * p.height_t;
    stats.scalar_used = "fp64";
    const int gpu = gpu_rows.load(std::memory_order_relaxed);
    const int cpu = cpu_rows.load(std::memory_order_relaxed);
    if (gpu > 0 && cpu > 0) {
        stats.engine_used = std::string("hybrid_cuda_") + ln_cpu_backend_name(cpu_backend);
    } else if (gpu > 0) {
        stats.engine_used = "cuda";
    } else {
        stats.engine_used = ln_cpu_backend_name(cpu_backend);
    }
    return stats;
}

} // namespace

bool ln_map_variant_supported_by_simd(Variant v) {
    const int id = static_cast<int>(v);
    return id >= 0 && id <= 9;
}

LnMapStats render_ln_map_openmp_rows(const LnMapParams& p, cv::Mat& out, int row_start, int row_count, const LnMapProgress& on_row_done) {
    ensure_ln_out(p, out);
    const auto [start, end] = clamp_rows(p, row_start, row_count);
    const auto t0 = std::chrono::steady_clock::now();
    const TrigColumns cols = make_trig_columns(p.width_s);
    dispatch_openmp_rows_impl(p, out, start, end, cols, true, on_row_done);
    const auto t1 = std::chrono::steady_clock::now();
    LnMapStats stats;
    stats.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    stats.pixel_count = p.width_s * (end - start);
    stats.engine_used = "openmp";
    stats.scalar_used = "fp64";
    return stats;
}

LnMapStats render_ln_map_openmp(const LnMapParams& p, cv::Mat& out, const LnMapProgress& on_row_done) {
    return render_ln_map_openmp_rows(p, out, 0, p.height_t, on_row_done);
}

LnMapStats render_ln_map(const LnMapParams& p, cv::Mat& out, const LnMapProgress& on_row_done) {
    if (p.engine == "hybrid") {
        return render_ln_map_hybrid(p, out, on_row_done);
    }

    if (should_try_cuda(p)) {
#if USE_CUDA_LN_MAP
        try {
            if (on_row_done) {
                return render_ln_map_cuda_with_progress(p, out, on_row_done);
            }
            const auto stats = fsd_cuda::cuda_render_ln_map(make_cuda_params(p), out);
            return {stats.elapsed_ms, p.width_s * p.height_t, "cuda", "fp64"};
        } catch (...) {
            if (p.engine == "cuda") throw;
        }
#endif
    }

    if (should_try_avx512(p)) {
        try {
            return render_ln_map_avx512(p, out, on_row_done);
        } catch (...) {
            if (p.engine == "avx512") throw;
        }
    }

    if (should_try_avx2(p)) {
        try {
            return render_ln_map_avx2(p, out, on_row_done);
        } catch (...) {
            if (p.engine == "avx2") throw;
        }
    }

    return render_ln_map_openmp(p, out, on_row_done);
}

} // namespace fsd::compute
