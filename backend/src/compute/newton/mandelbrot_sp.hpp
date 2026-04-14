// compute/newton/mandelbrot_sp.hpp
//
// Native port of C_mandelbrot/special_points_of_mandelbrot_set.py.
//
// Two entry points:
//
//   1. auto_solve(k, p)  — enumerates all (k, p) preperiodic / periodic roots
//      by building the polynomial g(k, p)(c) and finding its zeros via Newton
//      iteration with deflation.
//
//        · f(n)(c) := P_n(0; c) = iterate z² + c starting from z=0, n times.
//          f(0)=0, f(1)=c, f(2)=c²+c, f(3)=(c²+c)²+c, …
//          (Note: mapping to the Python: `f(t)` there is our f(t).)
//        · g(0, p)(c)   := f(p − 1)(c)
//          g(k>0, p)(c) := f(k + p − 1)(c) + f(k − 1)(c)
//          Then divide out lower-period factors by trial-dividing
//          g(t, factor) for every (t, factor) with factor | p and
//          (t, factor) != (k, p) and t ≤ k.
//        · Collect roots with Newton iteration + synthetic (linear) deflation.
//
//   2. seed_solve(k, p, seed) — runs Newton iteration on g(k, p) starting
//      from the given seed and returns the converged root.
//
// We operate on complex<double>. For modest (k, p) — degrees up to a few
// hundred — this is fine; beyond that the polynomial approach gets impractical
// and we'd need a different algorithm (distance estimator, atom period
// detection) which is out of scope for Phase 1.

#pragma once

#include <complex>
#include <string>
#include <vector>

namespace fsd::compute::newton_sp {

struct SolvedPoint {
    int k;
    int p;
    std::complex<double> c;
    std::string variant = "mandelbrot";
    std::string source;  // "auto" or "seed"
};

// Enumerate roots of g(k, p). Runs Newton + deflation until the polynomial
// degree drops to 1 or `max_attempts` random seeds fail to converge.
std::vector<SolvedPoint> auto_solve(int k, int p);

// Newton-iterate g(k, p) from `seed`. Returns the converged root or nullopt
// (represented as an empty vector) if iteration diverges.
std::vector<SolvedPoint> seed_solve(int k, int p, std::complex<double> seed);

} // namespace fsd::compute::newton_sp
