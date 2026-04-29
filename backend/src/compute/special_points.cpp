#include "special_points.hpp"

#include "variants.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace fsd::compute {
namespace {

using Z = std::complex<double>;

constexpr double PI = 3.14159265358979323846264338327950288;

struct Task {
    SpecialPointKind kind;
    int preperiod;
    int period;
    int expected;
};

struct SeedSolveOutcome {
    int seed_count = 0;
    int newton_success_count = 0;
    int rejected_count = 0;
    std::vector<SpecialPointResult> accepted;
    std::vector<SpecialPointResult> rejected_debug;
};

bool finite(Z z) {
    return std::isfinite(z.real()) && std::isfinite(z.imag());
}

Z normalize_root(Z z, double eps) {
    double re = z.real();
    double im = z.imag();
    if (std::abs(im) < eps) im = 0.0;
    if (std::abs(re) < eps) re = 0.0;
    return {re, im};
}

std::string format_id(SpecialPointKind kind, int preperiod, int period, Z c) {
    std::ostringstream ss;
    ss << (kind == SpecialPointKind::HyperbolicCenter ? "center" : "misiurewicz")
       << "-m" << preperiod << "-p" << period << "-";
    ss << std::fixed << std::setprecision(12) << c.real();
    if (c.imag() >= 0.0) ss << "+";
    ss << std::fixed << std::setprecision(12) << c.imag() << "i";
    return ss.str();
}

int mobius_center_count(int p) {
    int count = 1 << (p - 1);
    for (int d = 1; d < p; ++d) {
        if (p % d == 0) count -= expected_center_count(d);
    }
    return count;
}

double halton(unsigned long long index, int base) {
    double f = 1.0;
    double r = 0.0;
    while (index > 0) {
        f /= static_cast<double>(base);
        r += f * static_cast<double>(index % static_cast<unsigned long long>(base));
        index /= static_cast<unsigned long long>(base);
    }
    return r;
}

Z seed_for(unsigned long long index) {
    // Deterministic square sampling, then the caller rejects points outside
    // |c| <= 2. This keeps the seed policy reproducible while matching the
    // requested random(-2, 2) square semantics.
    const double u = halton(index + 1, 2);
    const double v = halton(index + 1, 3);
    return {4.0 * u - 2.0, 4.0 * v - 2.0};
}

bool same_root(const SpecialPointResult& a, const SpecialPointResult& b, double eps) {
    return a.kind == b.kind
        && a.preperiod == b.preperiod
        && a.period == b.period
        && std::abs(Z(a.re, a.im) - Z(b.re, b.im)) < eps;
}

template <Variant V>
Z variant_step_std(Z z, Z c) {
    Cx<double> zz{z.real(), z.imag()};
    Cx<double> cc{c.real(), c.imag()};
    const Cx<double> out = variant_step<V, double>(zz, cc);
    return {out.re, out.im};
}

Z variant_step_switch(Variant v, Z z, Z c) {
    switch (v) {
        case Variant::Mandelbrot: return variant_step_std<Variant::Mandelbrot>(z, c);
        case Variant::Tri:        return variant_step_std<Variant::Tri>(z, c);
        case Variant::Boat:       return variant_step_std<Variant::Boat>(z, c);
        case Variant::Duck:       return variant_step_std<Variant::Duck>(z, c);
        case Variant::Bell:       return variant_step_std<Variant::Bell>(z, c);
        case Variant::Fish:       return variant_step_std<Variant::Fish>(z, c);
        case Variant::Vase:       return variant_step_std<Variant::Vase>(z, c);
        case Variant::Bird:       return variant_step_std<Variant::Bird>(z, c);
        case Variant::Mask:       return variant_step_std<Variant::Mask>(z, c);
        case Variant::Ship:       return variant_step_std<Variant::Ship>(z, c);
        default:                  return variant_step_std<Variant::Mandelbrot>(z, c);
    }
}

std::string display_variant_name(Variant v) {
    switch (v) {
        case Variant::Mandelbrot: return "Mandelbrot";
        case Variant::Tri:        return "Tri";
        case Variant::Boat:       return "Boat";
        case Variant::Duck:       return "Duck";
        case Variant::Bell:       return "Bell";
        case Variant::Fish:       return "Fish";
        case Variant::Vase:       return "Vase";
        case Variant::Bird:       return "Bird";
        case Variant::Mask:       return "Mask";
        case Variant::Ship:       return "Ship";
        default:                  return "Unsupported";
    }
}

OrbitClassification classify_orbit_with_step(
    Z c,
    int max_iter,
    double eps,
    const std::function<Z(Z, Z)>& step
) {
    OrbitClassification out;
    out.orbit.reserve(static_cast<size_t>(max_iter) + 1);
    Z z{0.0, 0.0};
    out.orbit.push_back(z);
    for (int n = 1; n <= max_iter; ++n) {
        z = step(z, c);
        if (!finite(z)) break;
        out.orbit.push_back(z);
    }

    for (int i = 0; i < static_cast<int>(out.orbit.size()); ++i) {
        for (int j = i + 1; j < static_cast<int>(out.orbit.size()); ++j) {
            const double err = std::abs(out.orbit[j] - out.orbit[i]);
            if (err < eps) {
                out.found_repeat = true;
                out.preperiod = i;
                out.period = j - i;
                out.repeat_error = err;
                out.is_center = i == 0;
                out.is_misiurewicz = i > 0;
                return out;
            }
        }
    }
    return out;
}

bool matches_request(
    const OrbitClassification& actual,
    SpecialPointKind kind,
    int requested_preperiod,
    int requested_period
) {
    if (kind == SpecialPointKind::HyperbolicCenter) {
        return actual.is_center && actual.period == requested_period;
    }
    return actual.is_misiurewicz
        && actual.preperiod == requested_preperiod
        && actual.period == requested_period;
}

std::string rejection_reason(
    const OrbitClassification& actual,
    SpecialPointKind kind,
    int requested_preperiod,
    int requested_period
) {
    if (!actual.found_repeat) return "no_repeat";
    if (kind == SpecialPointKind::HyperbolicCenter) {
        if (!actual.is_center) return "not_center";
        if (actual.period < requested_period) return "degenerated_period";
        return "wrong_period";
    }
    if (actual.is_center) return "center_not_misiurewicz";
    if (!actual.is_misiurewicz) return "not_misiurewicz";
    if (actual.preperiod < requested_preperiod) return "early_preperiod";
    if (actual.period < requested_period) return "degenerated_period";
    if (actual.preperiod != requested_preperiod) return "wrong_preperiod";
    return "wrong_period";
}

SpecialPointResult make_base_result(SpecialPointKind kind, int preperiod, int period, Z c) {
    c = normalize_root(c, 1e-13);
    SpecialPointResult out;
    out.kind = kind;
    out.preperiod = preperiod;
    out.period = period;
    out.re = c.real();
    out.im = c.imag();
    out.id = format_id(kind, preperiod, period, c);
    return out;
}

SpecialPointResult polish_result(const SpecialPointResult& input, const SpecialPointEnumRequest& req) {
    SpecialPointResult out = input;
    if (!input.accepted) return out;
    const Z c0(input.re, input.im);
    if (input.kind == SpecialPointKind::HyperbolicCenter) {
        out = newton_solve_center(c0, input.period, req);
    } else {
        out = newton_solve_misiurewicz(c0, input.preperiod, input.period, req);
    }
    return out;
}

void add_or_replace_root(std::vector<SpecialPointResult>& roots, const SpecialPointResult& root, double eps) {
    for (auto& existing : roots) {
        if (!same_root(existing, root, eps)) continue;
        if (root.residual < existing.residual) existing = root;
        return;
    }
    roots.push_back(root);
}

SeedSolveOutcome solve_enum_seed(Z seed, const Task& task, const SpecialPointEnumRequest& req) {
    SeedSolveOutcome out;
    if (std::norm(seed) > 4.0) return out;
    out.seed_count = 1;

    SpecialPointResult root = task.kind == SpecialPointKind::HyperbolicCenter
        ? newton_solve_center(seed, task.period, req)
        : newton_solve_misiurewicz(seed, task.preperiod, task.period, req);

    if (root.converged) ++out.newton_success_count;
    if (!root.accepted) {
        ++out.rejected_count;
        if (req.include_rejected_debug) out.rejected_debug.push_back(root);
        return out;
    }

    root = polish_result(root, req);
    if (!root.accepted) {
        ++out.rejected_count;
        if (req.include_rejected_debug) out.rejected_debug.push_back(root);
        return out;
    }

    root.visible = point_in_viewport(req.viewport, Z(root.re, root.im));
    if (!req.visible_only || root.visible) out.accepted.push_back(root);

    if (std::abs(root.im) > req.root_merge_eps) {
        const Z conj_seed(root.re, -root.im);
        SpecialPointResult conj_root = task.kind == SpecialPointKind::HyperbolicCenter
            ? newton_solve_center(conj_seed, task.period, req)
            : newton_solve_misiurewicz(conj_seed, task.preperiod, task.period, req);
        if (conj_root.accepted) {
            conj_root = polish_result(conj_root, req);
            conj_root.visible = point_in_viewport(req.viewport, Z(conj_root.re, conj_root.im));
            if (!req.visible_only || conj_root.visible) out.accepted.push_back(conj_root);
        }
    }
    return out;
}

void merge_enum_outcome(SpecialPointEnumResponse& resp, const SeedSolveOutcome& outcome, const SpecialPointEnumRequest& req) {
    resp.seed_count += outcome.seed_count;
    resp.newton_success_count += outcome.newton_success_count;
    resp.rejected_count += outcome.rejected_count;
    for (const auto& root : outcome.accepted) {
        add_or_replace_root(resp.points, root, req.root_merge_eps);
    }
    if (req.include_rejected_debug) {
        for (const auto& rejected : outcome.rejected_debug) {
            if (resp.rejected_debug.size() >= 256) break;
            resp.rejected_debug.push_back(rejected);
        }
    }
}

SeedSolveOutcome solve_search_seed(
    Z seed,
    const Task& task,
    const SpecialPointEnumRequest& opt,
    const SpecialPointSearchRequest& req
) {
    SeedSolveOutcome out;
    if (std::norm(seed) > 4.0) return out;
    out.seed_count = 1;

    SpecialPointResult root = task.kind == SpecialPointKind::HyperbolicCenter
        ? newton_solve_center(seed, task.period, opt)
        : newton_solve_misiurewicz(seed, task.preperiod, task.period, opt);
    if (root.converged) ++out.newton_success_count;
    if (!root.accepted) {
        ++out.rejected_count;
        return out;
    }

    root.visible = point_in_viewport(req.viewport, Z(root.re, root.im));
    if (!req.visible_only || root.visible) out.accepted.push_back(root);
    return out;
}

void merge_search_outcome(SpecialPointSearchResponse& resp, const SeedSolveOutcome& outcome, double merge_eps) {
    resp.seed_count += outcome.seed_count;
    resp.newton_success_count += outcome.newton_success_count;
    resp.rejected_count += outcome.rejected_count;
    for (const auto& root : outcome.accepted) {
        add_or_replace_root(resp.points, root, merge_eps);
    }
}

std::vector<Task> build_tasks(const SpecialPointEnumRequest& req) {
    std::vector<Task> tasks;
    if (req.kind == SpecialPointKind::HyperbolicCenter) {
        for (int p = req.period_min; p <= req.period_max; ++p) {
            tasks.push_back({req.kind, 0, p, expected_center_count(p)});
        }
    } else {
        for (int m = req.preperiod_min; m <= req.preperiod_max; ++m) {
            for (int p = req.misiurewicz_period_min; p <= req.misiurewicz_period_max; ++p) {
                tasks.push_back({req.kind, m, p, expected_misiurewicz_count(m, p)});
            }
        }
    }
    return tasks;
}

SpecialPointEnumRequest options_from_search(const SpecialPointSearchRequest& req) {
    SpecialPointEnumRequest out;
    out.kind = req.kind;
    out.period_min = req.period_min;
    out.period_max = req.period_max;
    out.preperiod_min = req.preperiod_min;
    out.preperiod_max = req.preperiod_max;
    out.misiurewicz_period_min = req.period_min;
    out.misiurewicz_period_max = req.period_max;
    out.max_newton_iter = req.max_newton_iter;
    out.newton_eps = req.newton_eps;
    out.classify_eps = req.classify_eps;
    out.root_merge_eps = req.root_merge_eps;
    out.include_variant_existence = req.include_variant_compatibility;
    out.visible_only = req.visible_only;
    out.viewport = req.viewport;
    return out;
}

Z viewport_seed(const SpecialPointViewport& viewport, unsigned long long index, int period) {
    const double aspect = static_cast<double>(std::max(1, viewport.width)) / static_cast<double>(std::max(1, viewport.height));
    const double half_h = viewport.scale * 0.5;
    const double half_w = half_h * aspect;
    const double left = viewport.center_re - half_w;
    const double bottom = viewport.center_im - half_h;
    const double u = halton(index + 1 + static_cast<unsigned long long>(period) * 1009ULL, 2);
    const double v = halton(index + 1 + static_cast<unsigned long long>(period) * 9176ULL, 3);
    return {left + u * (2.0 * half_w), bottom + v * (2.0 * half_h)};
}

} // namespace

std::string special_point_kind_name(SpecialPointKind kind) {
    return kind == SpecialPointKind::HyperbolicCenter ? "center" : "misiurewicz";
}

std::pair<Z, Z> eval_center_f_df(Z c, int period) {
    Z z{0.0, 0.0};
    Z dz{0.0, 0.0};
    for (int i = 0; i < period; ++i) {
        dz = 2.0 * z * dz + Z{1.0, 0.0};
        z = z * z + c;
    }
    return {z, dz};
}

std::pair<Z, Z> eval_misiurewicz_f_df(Z c, int preperiod, int period) {
    Z z{0.0, 0.0};
    Z dz{0.0, 0.0};
    Z z_m{0.0, 0.0};
    Z dz_m{0.0, 0.0};
    const int total = preperiod + period;
    for (int i = 0; i < total; ++i) {
        dz = 2.0 * z * dz + Z{1.0, 0.0};
        z = z * z + c;
        if (i + 1 == preperiod) {
            z_m = z;
            dz_m = dz;
        }
    }
    return {z - z_m, dz - dz_m};
}

OrbitClassification classify_critical_orbit_mandelbrot(Z c, int max_iter, double eps) {
    return classify_orbit_with_step(c, max_iter, eps, [](Z z, Z cc) {
        return z * z + cc;
    });
}

OrbitClassification classify_critical_orbit(Z c, int max_iter, double eps) {
    return classify_critical_orbit_mandelbrot(c, max_iter, eps);
}

std::vector<VariantExistence> classify_variant_existence(
    Z c,
    SpecialPointKind requested_kind,
    int requested_preperiod,
    int requested_period,
    double eps
) {
    static constexpr Variant variants[] = {
        Variant::Mandelbrot, Variant::Tri,  Variant::Boat, Variant::Duck, Variant::Bell,
        Variant::Fish,       Variant::Vase, Variant::Bird, Variant::Mask, Variant::Ship,
    };

    const OrbitClassification mandel = classify_critical_orbit_mandelbrot(
        c, std::max(8, requested_preperiod + requested_period + 8), eps);

    std::vector<VariantExistence> out;
    out.reserve(std::size(variants));
    for (Variant v : variants) {
        VariantExistence item;
        item.variant_name = display_variant_name(v);

        double max_step_error = 0.0;
        item.same_orbit_as_mandelbrot = true;
        if (mandel.orbit.size() >= 2) {
            for (size_t i = 0; i + 1 < mandel.orbit.size(); ++i) {
                const Z variant_next = variant_step_switch(v, mandel.orbit[i], c);
                const double err = std::abs(variant_next - mandel.orbit[i + 1]);
                max_step_error = std::max(max_step_error, err);
                if (err >= eps) item.same_orbit_as_mandelbrot = false;
            }
        }

        const OrbitClassification actual = classify_orbit_with_step(
            c, std::max(8, requested_preperiod + requested_period + 8), eps,
            [v](Z z, Z cc) { return variant_step_switch(v, z, cc); });

        item.exists = matches_request(actual, requested_kind, requested_preperiod, requested_period);
        item.actual_preperiod = actual.preperiod;
        item.actual_period = actual.period;
        item.repeat_error = actual.repeat_error;
        item.reason = item.exists ? "ok" : "not_same_orbit_or_not_matching_period";
        if (!item.same_orbit_as_mandelbrot) {
            item.reason = "not_same_orbit_or_not_matching_period";
        }
        item.repeat_error = std::max(item.repeat_error, max_step_error);
        out.push_back(item);
    }
    return out;
}

SpecialPointResult newton_solve_center(Z initial, int period, const SpecialPointEnumRequest& req) {
    SpecialPointResult out = make_base_result(SpecialPointKind::HyperbolicCenter, 0, period, initial);
    Z c = initial;
    for (int iter = 0; iter < req.max_newton_iter; ++iter) {
        if (!finite(c)) {
            out.reason = "non_finite";
            return out;
        }
        if (std::abs(c) > 4.0) {
            out.reason = "escaped_parameter_region";
            return out;
        }
        const auto [f, df] = eval_center_f_df(c, period);
        out.residual = std::abs(f);
        out.newton_iterations = iter;
        if (out.residual < req.newton_eps) {
            out.converged = true;
            if (std::abs(df) < 1e-20) break;
            const Z step = f / df;
            if (std::abs(step) < 1e-14) break;
            c -= step;
            continue;
        }
        if (std::abs(df) < 1e-20) {
            out.reason = "derivative_too_small";
            return out;
        }
        const Z step = f / df;
        c -= step;
        if (std::abs(step) < 1e-16 && out.residual >= req.newton_eps) {
            out.reason = "stalled";
            return out;
        }
    }

    const auto [f, df] = eval_center_f_df(c, period);
    (void)df;
    const int iterations_used = out.newton_iterations;
    out = make_base_result(SpecialPointKind::HyperbolicCenter, 0, period, c);
    out.newton_iterations = iterations_used;
    out.residual = std::abs(f);
    out.converged = out.residual < req.newton_eps;
    out.actual = classify_critical_orbit_mandelbrot(c, std::max(16, period * 3 + 8), req.classify_eps);
    out.accepted = out.converged && matches_request(out.actual, out.kind, 0, period);
    out.reason = out.accepted ? "ok" : rejection_reason(out.actual, out.kind, 0, period);
    if (req.include_variant_existence && out.accepted) {
        out.variants = classify_variant_existence(c, out.kind, 0, period, req.classify_eps);
    }
    return out;
}

SpecialPointResult find_hyperbolic_center_near(Z initial, int period, const SpecialPointSearchRequest& req) {
    SpecialPointEnumRequest opt = options_from_search(req);
    return newton_solve_center(initial, period, opt);
}

SpecialPointResult newton_solve_misiurewicz(Z initial, int preperiod, int period, const SpecialPointEnumRequest& req) {
    SpecialPointResult out = make_base_result(SpecialPointKind::Misiurewicz, preperiod, period, initial);
    Z c = initial;
    for (int iter = 0; iter < req.max_newton_iter; ++iter) {
        if (!finite(c)) {
            out.reason = "non_finite";
            return out;
        }
        if (std::abs(c) > 4.0) {
            out.reason = "escaped_parameter_region";
            return out;
        }
        const auto [f, df] = eval_misiurewicz_f_df(c, preperiod, period);
        out.residual = std::abs(f);
        out.newton_iterations = iter;
        if (out.residual < req.newton_eps) {
            out.converged = true;
            if (std::abs(df) < 1e-20) break;
            const Z step = f / df;
            if (std::abs(step) < 1e-14) break;
            c -= step;
            continue;
        }
        if (std::abs(df) < 1e-20) {
            out.reason = "derivative_too_small";
            return out;
        }
        const Z step = f / df;
        c -= step;
        if (std::abs(step) < 1e-16 && out.residual >= req.newton_eps) {
            out.reason = "stalled";
            return out;
        }
    }

    const auto [f, df] = eval_misiurewicz_f_df(c, preperiod, period);
    (void)df;
    const int iterations_used = out.newton_iterations;
    out = make_base_result(SpecialPointKind::Misiurewicz, preperiod, period, c);
    out.newton_iterations = iterations_used;
    out.residual = std::abs(f);
    out.converged = out.residual < req.newton_eps;
    out.actual = classify_critical_orbit_mandelbrot(
        c, std::max(16, (preperiod + period) * 3 + 8), req.classify_eps);
    out.accepted = out.converged && matches_request(out.actual, out.kind, preperiod, period);
    out.reason = out.accepted ? "ok" : rejection_reason(out.actual, out.kind, preperiod, period);
    if (req.include_variant_existence && out.accepted) {
        out.variants = classify_variant_existence(c, out.kind, preperiod, period, req.classify_eps);
    }
    return out;
}

int expected_center_count(int period) {
    if (period < 1 || period > 30) return -1;
    return mobius_center_count(period);
}

int expected_misiurewicz_count(int preperiod, int period) {
    if (preperiod < 1 || period < 1 || preperiod > 6 || period > 6 || preperiod + period > 10) {
        return -1;
    }
    if (preperiod == 1) return 0;

    // Exact parameter count for the quadratic critical orbit definition used
    // here: z_0 = 0, exact preperiod m, exact period p. The primitive period-p
    // point count of z -> z^2 + c is 2 * expected_center_count(p). For m >= 2,
    // each primitive periodic point has 2^(m-2) critical preimage choices,
    // except when the critical orbit itself is a period-p center
    // (p | (m - 1)); those centers are not Misiurewicz points.
    const int primitive_periodic_points = 2 * expected_center_count(period);
    int count = primitive_periodic_points * (1 << (preperiod - 2));
    if ((preperiod - 1) % period == 0) count -= expected_center_count(period);
    return count;
}

bool point_in_viewport(const SpecialPointViewport& viewport, Z c) {
    if (!viewport.enabled) return true;
    const double aspect = static_cast<double>(std::max(1, viewport.width)) / static_cast<double>(std::max(1, viewport.height));
    const double half_h = viewport.scale * 0.5;
    const double half_w = half_h * aspect;
    return c.real() >= viewport.center_re - half_w
        && c.real() <= viewport.center_re + half_w
        && c.imag() >= viewport.center_im - half_h
        && c.imag() <= viewport.center_im + half_h;
}

SpecialPointEnumResponse enumerate_special_points(
    const SpecialPointEnumRequest& req,
    const SpecialPointProgressCallback& progress
) {
    const std::vector<Task> tasks = build_tasks(req);
    SpecialPointEnumResponse resp;
    resp.status = "running";
    for (const Task& t : tasks) {
        if (t.expected < 0) {
            throw std::runtime_error("expected count unavailable for requested Misiurewicz parameter");
        }
        resp.expected_count += t.expected;
    }

    int task_index = 0;
    for (const Task& task : tasks) {
        ++task_index;
        const int accepted_before_task = static_cast<int>(resp.points.size());
        const unsigned long long task_seed_offset =
            1000003ULL * static_cast<unsigned long long>(task.period)
            + 9176ULL * static_cast<unsigned long long>(task.preperiod + 1);

        const Z anchors[] = {
            {0.0, 0.0}, {-1.0, 0.0}, {-2.0, 0.0}, {1.0, 0.0}, {0.25, 0.0},
            {0.0, 1.0}, {0.0, -1.0}, {-0.75, 0.75}, {-0.75, -0.75},
        };
        for (Z seed : anchors) {
            if (static_cast<int>(resp.points.size()) - accepted_before_task >= task.expected) break;
            merge_enum_outcome(resp, solve_enum_seed(seed, task, req), req);
        }

        for (int batch = 0; batch < req.max_seed_batches; ++batch) {
            if (static_cast<int>(resp.points.size()) - accepted_before_task >= task.expected) break;
            if (progress && !progress(task_index, static_cast<int>(tasks.size()),
                                      static_cast<int>(resp.points.size()), resp.expected_count,
                                      resp.seed_count, batch)) {
                resp.status = "cancelled";
                resp.complete = false;
                return resp;
            }

            std::vector<Z> seeds;
            seeds.reserve(static_cast<size_t>(req.seeds_per_batch));
            for (int i = 0; i < req.seeds_per_batch; ++i) {
                const unsigned long long seed_index = task_seed_offset
                    + static_cast<unsigned long long>(batch) * static_cast<unsigned long long>(req.seeds_per_batch)
                    + static_cast<unsigned long long>(i);
                seeds.push_back(seed_for(seed_index));
            }

            std::vector<SeedSolveOutcome> outcomes(seeds.size());
            #pragma omp parallel for schedule(dynamic, 16)
            for (int i = 0; i < static_cast<int>(seeds.size()); ++i) {
                outcomes[static_cast<size_t>(i)] = solve_enum_seed(seeds[static_cast<size_t>(i)], task, req);
            }

            for (const auto& outcome : outcomes) {
                merge_enum_outcome(resp, outcome, req);
            }
        }
    }

    std::sort(resp.points.begin(), resp.points.end(), [](const auto& a, const auto& b) {
        if (a.period != b.period) return a.period < b.period;
        if (a.preperiod != b.preperiod) return a.preperiod < b.preperiod;
        if (a.re != b.re) return a.re < b.re;
        return a.im < b.im;
    });

    resp.accepted_count = static_cast<int>(resp.points.size());
    resp.complete = !req.visible_only && resp.accepted_count == resp.expected_count;
    resp.status = resp.complete ? "completed" : "incomplete";
    if (!resp.complete) {
        resp.warning = "not all expected roots were found within the seed budget";
        if (req.visible_only) resp.warning = "visibleOnly filters results; complete is only true for unfiltered enumeration";
    }
    return resp;
}

SpecialPointSearchResponse search_special_points(
    const SpecialPointSearchRequest& req,
    const SpecialPointSearchProgressCallback& progress
) {
    if (!req.viewport.enabled) {
        throw std::runtime_error("viewport is required for special point search");
    }

    SpecialPointSearchResponse resp;
    resp.status = "searching";

    std::vector<Task> tasks;
    if (req.kind == SpecialPointKind::HyperbolicCenter) {
        for (int p = req.period_min; p <= req.period_max; ++p) {
            tasks.push_back({req.kind, 0, p, 0});
        }
    } else {
        for (int m = req.preperiod_min; m <= req.preperiod_max; ++m) {
            for (int p = req.period_min; p <= req.period_max; ++p) {
                tasks.push_back({req.kind, m, p, 0});
            }
        }
    }
    const int task_count = std::max(1, static_cast<int>(tasks.size()));
    const int seeds_per_task = std::max(1, req.seed_budget / task_count);
    const double merge_eps = std::max(req.root_merge_eps, req.viewport.scale * 1e-9);
    const SpecialPointEnumRequest opt = options_from_search(req);

    const Z anchors[] = {
        {req.viewport.center_re, req.viewport.center_im},
        {req.viewport.center_re + req.viewport.scale * 0.025, req.viewport.center_im},
        {req.viewport.center_re - req.viewport.scale * 0.025, req.viewport.center_im},
        {req.viewport.center_re, req.viewport.center_im + req.viewport.scale * 0.025},
        {req.viewport.center_re, req.viewport.center_im - req.viewport.scale * 0.025},
        {0.0, 0.0}, {-1.0, 0.0}, {-0.12256116687665362, 0.7448617666197442},
        {-0.12256116687665362, -0.7448617666197442},
    };

    std::atomic<bool> cancel_requested{false};
    std::atomic<bool> misiurewicz_found{false};
    std::atomic<int> progress_seed_count{0};
    std::atomic<int> progress_accepted_count{0};
    std::vector<SpecialPointSearchResponse> locals(tasks.size());

    #pragma omp parallel for schedule(dynamic, 1)
    for (int task_i = 0; task_i < static_cast<int>(tasks.size()); ++task_i) {
        if (cancel_requested.load(std::memory_order_relaxed)) continue;
        if (req.kind == SpecialPointKind::Misiurewicz &&
            misiurewicz_found.load(std::memory_order_relaxed)) {
            continue;
        }

        const Task& task = tasks[static_cast<size_t>(task_i)];
        bool proceed = true;
        if (progress) {
            #pragma omp critical(special_point_search_progress)
            {
                proceed = progress(task.period, task_i + 1, task_count, 0, 0);
            }
        }
        if (!proceed) {
            cancel_requested.store(true, std::memory_order_relaxed);
            continue;
        }

        SpecialPointSearchResponse local;
        local.status = "searching";
        auto merge_local = [&](const SeedSolveOutcome& outcome) {
            merge_search_outcome(local, outcome, merge_eps);
            if (req.kind == SpecialPointKind::Misiurewicz && !local.points.empty()) {
                misiurewicz_found.store(true, std::memory_order_relaxed);
            }
        };

        for (Z seed : anchors) {
            if (cancel_requested.load(std::memory_order_relaxed)) break;
            if (req.kind == SpecialPointKind::Misiurewicz &&
                misiurewicz_found.load(std::memory_order_relaxed) && local.points.empty()) {
                break;
            }
            if (!point_in_viewport(req.viewport, seed)) continue;
            merge_local(solve_search_seed(seed, task, opt, req));
            if (req.kind == SpecialPointKind::Misiurewicz && !local.points.empty()) break;
        }

        for (int i = 0; i < seeds_per_task; ++i) {
            if (cancel_requested.load(std::memory_order_relaxed)) break;
            if (req.kind == SpecialPointKind::Misiurewicz &&
                misiurewicz_found.load(std::memory_order_relaxed) && local.points.empty()) {
                break;
            }
            const unsigned long long seed_index =
                static_cast<unsigned long long>(i)
                + 1000003ULL * static_cast<unsigned long long>(task_i + 1);
            const Z seed = viewport_seed(req.viewport, seed_index, task.period);
            merge_local(solve_search_seed(seed, task, opt, req));
            if (req.kind == SpecialPointKind::Misiurewicz && !local.points.empty()) break;
        }

        local.accepted_count = static_cast<int>(local.points.size());
        locals[static_cast<size_t>(task_i)] = std::move(local);

        const int accepted_now = progress_accepted_count.fetch_add(
            locals[static_cast<size_t>(task_i)].accepted_count,
            std::memory_order_relaxed) + locals[static_cast<size_t>(task_i)].accepted_count;
        const int seed_now = progress_seed_count.fetch_add(
            locals[static_cast<size_t>(task_i)].seed_count,
            std::memory_order_relaxed) + locals[static_cast<size_t>(task_i)].seed_count;
        if (progress && !cancel_requested.load(std::memory_order_relaxed)) {
            bool proceed_after_task = true;
            #pragma omp critical(special_point_search_progress)
            {
                proceed_after_task = progress(task.period, task_i + 1, task_count, accepted_now, seed_now);
            }
            if (!proceed_after_task) cancel_requested.store(true, std::memory_order_relaxed);
        }
    }

    if (cancel_requested.load(std::memory_order_relaxed)) {
        resp.status = "cancelled";
        return resp;
    }

    for (const auto& local : locals) {
        resp.seed_count += local.seed_count;
        resp.newton_success_count += local.newton_success_count;
        resp.rejected_count += local.rejected_count;
        for (const auto& root : local.points) {
            add_or_replace_root(resp.points, root, merge_eps);
        }
    }
    std::sort(resp.points.begin(), resp.points.end(), [](const auto& a, const auto& b) {
        if (a.period != b.period) return a.period < b.period;
        if (a.re != b.re) return a.re < b.re;
        return a.im < b.im;
    });
    if (req.kind == SpecialPointKind::Misiurewicz && resp.points.size() > 1) {
        resp.points.resize(1);
    }

    resp.accepted_count = static_cast<int>(resp.points.size());
    resp.status = "completed";
    if (resp.accepted_count == 0) {
        resp.warning = req.kind == SpecialPointKind::Misiurewicz
            ? "no matching visible Misiurewicz point found within the sampled viewport budget"
            : "no visible hyperbolic center found within the sampled viewport budget";
    } else {
        resp.warning = req.kind == SpecialPointKind::Misiurewicz
            ? "sampled Misiurewicz discovery; stops after the first visible matching point"
            : "sampled viewport discovery; not guaranteed exhaustive";
    }
    return resp;
}

} // namespace fsd::compute
