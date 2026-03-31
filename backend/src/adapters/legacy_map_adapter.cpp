#include "adapters.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace fsd {

namespace {

std::mutex gMapBuildMutex;
bool gMapExeReady = false;

std::string managedMapRendererSource() {
    return R"SRC(#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <omp.h>
#include "svpng/svpng.inc"

static int clamp255(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return v;
}

static int color_r(double n) {
    const double a = 128.0 - 128.0 * cos(53.0 * n * 3.141592653589793);
    return clamp255((int)a);
}

static int color_g(double n) {
    const double a = 128.0 - 128.0 * cos(27.0 * n * 3.141592653589793);
    return clamp255((int)a);
}

static int color_b(double n) {
    const double a = 128.0 - 128.0 * cos(139.0 * n * 3.141592653589793);
    return clamp255((int)a);
}

static void hsv_to_rgb(double h, double s, double v, int* r, int* g, int* b) {
    const double c = v * s;
    const double hh = h / 60.0;
    const double x = c * (1.0 - fabs(fmod(hh, 2.0) - 1.0));

    double rr = 0.0;
    double gg = 0.0;
    double bb = 0.0;

    if (hh >= 0.0 && hh < 1.0) {
        rr = c; gg = x; bb = 0.0;
    } else if (hh < 2.0) {
        rr = x; gg = c; bb = 0.0;
    } else if (hh < 3.0) {
        rr = 0.0; gg = c; bb = x;
    } else if (hh < 4.0) {
        rr = 0.0; gg = x; bb = c;
    } else if (hh < 5.0) {
        rr = x; gg = 0.0; bb = c;
    } else {
        rr = c; gg = 0.0; bb = x;
    }

    const double m = v - c;
    *r = clamp255((int)((rr + m) * 255.0));
    *g = clamp255((int)((gg + m) * 255.0));
    *b = clamp255((int)((bb + m) * 255.0));
}

static void colorize(int iter, int max_iter, int palette, unsigned char* r, unsigned char* g, unsigned char* b) {
    if (iter >= max_iter) {
        *r = 255;
        *g = 255;
        *b = 255;
        return;
    }

    const double n = ((double)iter + 1.0) / ((double)max_iter + 2.0);

    if (palette == 1) {
        const int rr = iter % 256;
        const int gg = iter / 256;
        const int bb = (iter % 17) * 17;
        *r = (unsigned char)clamp255(rr);
        *g = (unsigned char)clamp255(gg);
        *b = (unsigned char)clamp255(bb);
        return;
    }

    if (palette == 2) {
        const double h = fmod((double)iter, 1440.0) / 4.0;
        int rr = 0;
        int gg = 0;
        int bb = 0;
        hsv_to_rgb(h, 1.0, 1.0, &rr, &gg, &bb);
        *r = (unsigned char)rr;
        *g = (unsigned char)gg;
        *b = (unsigned char)bb;
        return;
    }

    if (palette == 3) {
        const int m = iter % 765;
        const int band = m / 255;
        const int d = m % 255;
        int rr = 255;
        int gg = 255;
        int bb = 255;
        if (band == 0) {
            rr = 255 - d;
            gg = d;
            bb = 255;
        } else if (band == 1) {
            rr = d;
            gg = 255;
            bb = 255 - d;
        } else {
            rr = 255;
            gg = 255 - d;
            bb = d;
        }
        *r = (unsigned char)clamp255(rr);
        *g = (unsigned char)clamp255(gg);
        *b = (unsigned char)clamp255(bb);
        return;
    }

    if (palette == 4) {
        const int v = clamp255((int)(n * 255.0));
        *r = (unsigned char)v;
        *g = (unsigned char)v;
        *b = (unsigned char)v;
        return;
    }

    *r = (unsigned char)color_r(n);
    *g = (unsigned char)color_g(n);
    *b = (unsigned char)color_b(n);
}

static int mandelbrot(complex double c, int max) {
    complex double n = 0.0 + 0.0 * I;
    for (int z = 0; z < max; z++) {
        n = n * n + c;
        if (cabs(n) > 2.0) return z;
    }
    return max;
}

static int tri(complex double c, int max) {
    complex double n = 0.0 + 0.0 * I;
    for (int z = 0; z < max; z++) {
        n = conj(n * n) + c;
        if (cabs(n) > 2.0) return z;
    }
    return max;
}

static int boat(complex double c, int max) {
    complex double n = 0.0 + 0.0 * I;
    for (int z = 0; z < max; z++) {
        n = cabs(creal(n)) + cabs(cimag(n)) * I;
        n = n * n + c;
        if (cabs(n) > 2.0) return z;
    }
    return max;
}

static int duck(complex double c, int max) {
    complex double n = 0.0 + 0.0 * I;
    for (int z = 0; z < max; z++) {
        n = creal(n) + cabs(cimag(n)) * I;
        n = n * n + c;
        if (cabs(n) > 2.0) return z;
    }
    return max;
}

static int bell(complex double c, int max) {
    complex double n = 0.0 + 0.0 * I;
    for (int z = 0; z < max; z++) {
        n = cabs(creal(n)) - cimag(n) * I;
        n = n * n + c;
        if (cabs(n) > 2.0) return z;
    }
    return max;
}

static int fish(complex double c, int max) {
    complex double n = 0.0 + 0.0 * I;
    for (int z = 0; z < max; z++) {
        n = n * n;
        n = cabs(creal(n)) + cimag(n) * I;
        n = n + c;
        if (cabs(n) > 2.0) return z;
    }
    return max;
}

static int vase(complex double c, int max) {
    complex double n = 0.0 + 0.0 * I;
    for (int z = 0; z < max; z++) {
        n = n * n;
        n = cabs(creal(n)) - cimag(n) * I;
        n = n + c;
        if (cabs(n) > 2.0) return z;
    }
    return max;
}

static int bird(complex double c, int max) {
    complex double n = 0.0 + 0.0 * I;
    for (int z = 0; z < max; z++) {
        n = n * n;
        n = cabs(creal(n)) + cabs(cimag(n)) * I;
        n = n + c;
        if (cabs(n) > 2.0) return z;
    }
    return max;
}

static int mask(complex double c, int max) {
    complex double n = 0.0 + 0.0 * I;
    for (int z = 0; z < max; z++) {
        n = creal(n) + cabs(cimag(n)) * I;
        n = n * n;
        n = cabs(creal(n)) + cimag(n) * I;
        n = n + c;
        if (cabs(n) > 2.0) return z;
    }
    return max;
}

static int ship(complex double c, int max) {
    complex double n = 0.0 + 0.0 * I;
    for (int z = 0; z < max; z++) {
        n = cabs(creal(n)) + cimag(n) * I;
        n = n * n;
        n = cabs(creal(n)) - cimag(n) * I;
        n = n + c;
        if (cabs(n) > 2.0) return z;
    }
    return max;
}

static int iterate_point(complex double c, int max_iter, int variety) {
    switch (variety) {
        case 0: return mandelbrot(c, max_iter);
        case 1: return tri(c, max_iter);
        case 2: return boat(c, max_iter);
        case 3: return duck(c, max_iter);
        case 4: return bell(c, max_iter);
        case 5: return fish(c, max_iter);
        case 6: return vase(c, max_iter);
        case 7: return bird(c, max_iter);
        case 8: return mask(c, max_iter);
        case 9: return ship(c, max_iter);
        default: return mandelbrot(c, max_iter);
    }
}

int main(int argc, char** argv) {
    if (argc != 10) {
        return 1;
    }

    const double scale = atof(argv[1]);
    const double center_re = atof(argv[2]);
    const double center_im = atof(argv[3]);
    const char* filename = argv[4];
    const int w = atoi(argv[5]);
    const int h = atoi(argv[6]);
    const int variety = atoi(argv[7]);
    const int max_iter = atoi(argv[8]);
    const int palette = atoi(argv[9]);

    if (w <= 0 || h <= 0 || scale <= 0.0 || max_iter <= 0) {
        return 2;
    }

    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        return 3;
    }

    unsigned char* rgb = (unsigned char*)malloc((size_t)w * (size_t)h * 3);
    if (!rgb) {
        fclose(fp);
        return 4;
    }

    const double aspect = (double)w / (double)h;
    const double span_im = scale;
    const double span_re = scale * aspect;

    const double re_min = center_re - span_re * 0.5;
    const double im_max = center_im + span_im * 0.5;

    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            const double re = re_min + ((double)x + 0.5) / (double)w * span_re;
            const double im = im_max - ((double)y + 0.5) / (double)h * span_im;
            const complex double c = re + im * I;
            const int iter = iterate_point(c, max_iter, variety);
            const size_t idx = ((size_t)y * (size_t)w + (size_t)x) * 3;
            colorize(iter, max_iter, palette, &rgb[idx + 0], &rgb[idx + 1], &rgb[idx + 2]);
        }
    }

    svpng(fp, (unsigned)w, (unsigned)h, rgb, 0);
    free(rgb);
    fclose(fp);
    return 0;
}
)SRC";
}

} // namespace

Artifact runManagedMapRender(const fs::path& repoRoot, const std::string& runDir, const MapRenderParams& params) {
    const fs::path legacySvpngInc = repoRoot / "C_mandelbrot" / "svpng" / "svpng.inc";
    if (!fs::exists(legacySvpngInc)) {
        throw std::runtime_error("legacy svpng include not found");
    }

    const fs::path managedDir = repoRoot / "fractal_studio" / "runtime" / "managed_sources";
    fs::create_directories(managedDir);

    const fs::path managedSvpngDir = managedDir / "svpng";
    fs::create_directories(managedSvpngDir);

    const fs::path managedSrc = managedDir / "map_renderer_openmp.c";
    const fs::path exePath = managedDir / "map_renderer_openmp.out";
    const fs::path imagePath = fs::path(runDir) / "map.png";

    {
        std::lock_guard<std::mutex> lock(gMapBuildMutex);
        if (!gMapExeReady || !fs::exists(exePath)) {
            fs::copy_file(legacySvpngInc, managedSvpngDir / "svpng.inc", fs::copy_options::overwrite_existing);
            std::ofstream out(managedSrc, std::ios::trunc);
            if (!out) {
                throw std::runtime_error("failed to write managed map source");
            }
            out << managedMapRendererSource();
            out.close();

            const std::string compileCmd = "gcc \"" + managedSrc.string() + "\" -o \"" + exePath.string() + "\" -O3 -fopenmp -lm";
            if (std::system(compileCmd.c_str()) != 0) {
                throw std::runtime_error("managed map compile failed");
            }
            gMapExeReady = true;
        }
    }

    const double safeScale = params.scale > 0.0 ? params.scale : 4.0;
    const int width = std::max(256, std::min(2048, params.width));
    const int height = std::max(256, std::min(2048, params.height));
    const int variety = std::max(0, std::min(9, params.variety));
    const int iterations = std::max(1, std::min(65535, params.iterations));

    int palette = 0;
    if (params.colorMap == "mod17") {
        palette = 1;
    } else if (params.colorMap == "hsv_wheel") {
        palette = 2;
    } else if (params.colorMap == "tri765") {
        palette = 3;
    } else if (params.colorMap == "grayscale") {
        palette = 4;
    }

    std::ostringstream scaleSs;
    scaleSs << std::setprecision(17) << safeScale;
    std::ostringstream reSs;
    reSs << std::setprecision(17) << params.centerRe;
    std::ostringstream imSs;
    imSs << std::setprecision(17) << params.centerIm;

    const std::string runCmd =
        "\"" + exePath.string() + "\" " +
        scaleSs.str() + " " +
        reSs.str() + " " +
        imSs.str() + " \"" + imagePath.string() + "\" " +
        std::to_string(width) + " " +
        std::to_string(height) + " " +
        std::to_string(variety) + " " +
        std::to_string(iterations) + " " +
        std::to_string(palette);

    if (std::system(runCmd.c_str()) != 0) {
        throw std::runtime_error("managed map execution failed");
    }

    if (!fs::exists(imagePath)) {
        throw std::runtime_error("map image not generated");
    }

    return Artifact{"explorer-map", imagePath.string(), "image"};
}

} // namespace fsd
