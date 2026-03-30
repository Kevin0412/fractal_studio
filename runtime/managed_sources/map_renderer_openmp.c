#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <omp.h>
#include "svpng/svpng.inc"

static int color_r(double n) {
    double a = 128.0 - 128.0 * cos(53.0 * n * 3.141592653589793);
    return a < 256.0 ? (int)a : 255;
}

static int color_g(double n) {
    double a = 128.0 - 128.0 * cos(27.0 * n * 3.141592653589793);
    return a < 256.0 ? (int)a : 255;
}

static int color_b(double n) {
    double a = 128.0 - 128.0 * cos(139.0 * n * 3.141592653589793);
    return a < 256.0 ? (int)a : 255;
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
    if (argc != 9) {
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
            const double n = ((double)iter + 1.0) / ((double)max_iter + 2.0);

            const size_t idx = ((size_t)y * (size_t)w + (size_t)x) * 3;
            rgb[idx + 0] = (unsigned char)color_r(n);
            rgb[idx + 1] = (unsigned char)color_g(n);
            rgb[idx + 2] = (unsigned char)color_b(n);
        }
    }

    svpng(fp, (unsigned)w, (unsigned)h, rgb, 0);
    free(rgb);
    fclose(fp);
    return 0;
}
