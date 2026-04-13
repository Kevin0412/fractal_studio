// compute/variants.hpp
//
// The 10 Mandelbrot-family variant step functions, tag-dispatched so the
// compiler inlines the chosen body into the escape-time loop. Math ported
// verbatim from C_mandelbrot/Mandelbrot_python_ln.c and the legacy managed map
// renderer in fractal_studio/backend/src/adapters/legacy_map_adapter.cpp.
//
// Variant catalog (see README.md for full descriptions):
//   mandelbrot   — z² + c                                (standard)
//   tricorn      — conj(z²) + c                          (Mandelbar)
//   burning_ship — (|Re z| + |Im z|·i)² + c             (Burning Ship)
//   celtic       — (Re z + |Im z|·i)² + c               (half-fold Im only)
//   heart        — (|Re z| − Im z·i)² + c               (fold Re, negate Im)
//   buffalo      — z² → |Re(z²)| + Im(z²)·i + c        (Celtic Mandelbrot)
//   perp_buffalo — z² → |Re(z²)| − Im(z²)·i + c        (perpendicular buffalo)
//   celtic_ship  — z² → |Re(z²)| + |Im(z²)|·i + c      (Celtic+Ship combined)
//   mandelceltic — (Re+|Im|·i)² → |Re|+Im·i + c         (pre+post fold Im)
//   perp_ship    — (|Re|+Im·i)² → |Re|−Im·i + c         (perpendicular ship)

#pragma once

#include "complex.hpp"

namespace fsd::compute {

enum class Variant {
    Mandelbrot  = 0,  // z² + c
    Tri         = 1,  // conj(z²) + c                       (tricorn / Mandelbar)
    Boat        = 2,  // (|Re| + |Im|·i)² + c               (Burning Ship)
    Duck        = 3,  // (Re + |Im|·i)² + c                 (Celtic — fold Im only)
    Bell        = 4,  // (|Re| − Im·i)² + c                 (Heart — fold Re, negate Im)
    Fish        = 5,  // z²→|Re(z²)|+Im(z²)·i+c            (Buffalo / Celtic Mandelbrot)
    Vase        = 6,  // z²→|Re(z²)|−Im(z²)·i+c            (Perpendicular Buffalo)
    Bird        = 7,  // z²→|Re(z²)|+|Im(z²)|·i+c          (Celtic Ship)
    Mask        = 8,  // (Re+|Im|·i)²→|Re|+Im·i+c          (Mandelceltic)
    Ship        = 9,  // (|Re|+Im·i)²→|Re|−Im·i+c          (Perpendicular Ship)
};

inline const char* variant_name(Variant v) {
    switch (v) {
        case Variant::Mandelbrot: return "mandelbrot";
        case Variant::Tri:        return "tricorn";
        case Variant::Boat:       return "burning_ship";
        case Variant::Duck:       return "celtic";
        case Variant::Bell:       return "heart";
        case Variant::Fish:       return "buffalo";
        case Variant::Vase:       return "perp_buffalo";
        case Variant::Bird:       return "celtic_ship";
        case Variant::Mask:       return "mandelceltic";
        case Variant::Ship:       return "perp_ship";
    }
    return "mandelbrot";
}

inline bool variant_from_name(const char* name, Variant& out) {
    struct Entry { const char* n; Variant v; };
    static constexpr Entry table[] = {
        {"mandelbrot",  Variant::Mandelbrot},
        {"tricorn",     Variant::Tri},
        {"burning_ship",Variant::Boat},
        {"celtic",      Variant::Duck},
        {"heart",       Variant::Bell},
        {"buffalo",     Variant::Fish},
        {"perp_buffalo",Variant::Vase},
        {"celtic_ship", Variant::Bird},
        {"mandelceltic",Variant::Mask},
        {"perp_ship",   Variant::Ship},
        // Legacy aliases (keep old names working)
        {"tri",         Variant::Tri},
        {"boat",        Variant::Boat},
        {"duck",        Variant::Duck},
        {"bell",        Variant::Bell},
        {"fish",        Variant::Fish},
        {"vase",        Variant::Vase},
        {"bird",        Variant::Bird},
        {"mask",        Variant::Mask},
        {"ship",        Variant::Ship},
    };
    for (const auto& e : table) {
        const char* a = e.n;
        const char* b = name;
        while (*a && *b && *a == *b) { ++a; ++b; }
        if (*a == 0 && *b == 0) { out = e.v; return true; }
    }
    return false;
}

// Tag-dispatched step. The compiler collapses the switch on V away when V is a
// template constant, so each instantiation gets a tight loop for its variant.
template <Variant V, typename S>
inline Cx<S> variant_step(Cx<S> z, const Cx<S>& c) {
    if constexpr (V == Variant::Mandelbrot) {
        return z.sqr() + c;
    } else if constexpr (V == Variant::Tri) {
        return conj_sqr(z) + c;
    } else if constexpr (V == Variant::Boat) {
        Cx<S> w{scalar_abs(z.re), scalar_abs(z.im)};
        return w.sqr() + c;
    } else if constexpr (V == Variant::Duck) {
        Cx<S> w{z.re, scalar_abs(z.im)};
        return w.sqr() + c;
    } else if constexpr (V == Variant::Bell) {
        Cx<S> w{scalar_abs(z.re), -z.im};
        return w.sqr() + c;
    } else if constexpr (V == Variant::Fish) {
        Cx<S> w = z.sqr();
        w.re = scalar_abs(w.re);
        return w + c;
    } else if constexpr (V == Variant::Vase) {
        Cx<S> w = z.sqr();
        w.re = scalar_abs(w.re);
        w.im = -w.im;
        return w + c;
    } else if constexpr (V == Variant::Bird) {
        Cx<S> w = z.sqr();
        w.re = scalar_abs(w.re);
        w.im = scalar_abs(w.im);
        return w + c;
    } else if constexpr (V == Variant::Mask) {
        Cx<S> w{z.re, scalar_abs(z.im)};
        w = w.sqr();
        w.re = scalar_abs(w.re);
        return w + c;
    } else if constexpr (V == Variant::Ship) {
        Cx<S> w{scalar_abs(z.re), z.im};
        w = w.sqr();
        w.re = scalar_abs(w.re);
        w.im = -w.im;
        return w + c;
    }
}

} // namespace fsd::compute
