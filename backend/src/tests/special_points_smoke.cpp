#include "compute/special_points.hpp"

#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using fsd::compute::SpecialPointEnumRequest;
using fsd::compute::SpecialPointKind;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

SpecialPointEnumRequest base_request() {
    SpecialPointEnumRequest req;
    req.max_newton_iter = 80;
    req.max_seed_batches = 120;
    req.seeds_per_batch = 4096;
    req.newton_eps = 1e-13;
    req.classify_eps = 1e-10;
    req.root_merge_eps = 1e-9;
    req.include_variant_existence = true;
    return req;
}

void expect_center_period(int period, int expected) {
    SpecialPointEnumRequest req = base_request();
    req.kind = SpecialPointKind::HyperbolicCenter;
    req.period_min = period;
    req.period_max = period;
    const auto resp = fsd::compute::enumerate_special_points(req);
    require(resp.complete, "center enumeration did not complete for period " + std::to_string(period));
    require(resp.expected_count == expected, "unexpected center expected count for period " + std::to_string(period));
    require(resp.accepted_count == expected, "unexpected center accepted count for period " + std::to_string(period));
    for (const auto& p : resp.points) {
        require(p.accepted, "center point was not accepted");
        require(p.actual.is_center, "center point did not classify as center");
        require(p.actual.period == period, "center point classified with wrong period");
        require(!p.variants.empty(), "center point missing variant existence results");
        require(p.variants.front().variant_name == "Mandelbrot" && p.variants.front().exists,
                "Mandelbrot variant did not accept Mandelbrot point");
    }
}

void expect_misiurewicz_2_1() {
    SpecialPointEnumRequest req = base_request();
    req.kind = SpecialPointKind::Misiurewicz;
    req.preperiod_min = 2;
    req.preperiod_max = 2;
    req.misiurewicz_period_min = 1;
    req.misiurewicz_period_max = 1;
    const auto resp = fsd::compute::enumerate_special_points(req);
    require(resp.complete, "misiurewicz m=2 p=1 enumeration did not complete");
    require(resp.expected_count == 1, "unexpected misiurewicz m=2 p=1 expected count");
    require(resp.accepted_count == 1, "unexpected misiurewicz m=2 p=1 accepted count");
    const auto& p = resp.points.front();
    require(p.actual.is_misiurewicz, "misiurewicz point did not classify as Misiurewicz");
    require(p.actual.preperiod == 2 && p.actual.period == 1, "wrong misiurewicz classification");
    require(std::abs(p.re + 2.0) < 1e-8 && std::abs(p.im) < 1e-8,
            "misiurewicz m=2 p=1 root is not c=-2: " + std::to_string(p.re) + ", " + std::to_string(p.im));
}

} // namespace

int main() {
    try {
        require(fsd::compute::expected_center_count(1) == 1, "center count p=1");
        require(fsd::compute::expected_center_count(2) == 1, "center count p=2");
        require(fsd::compute::expected_center_count(3) == 3, "center count p=3");
        require(fsd::compute::expected_center_count(4) == 6, "center count p=4");

        require(fsd::compute::expected_misiurewicz_count(1, 1) == 0, "misiurewicz count m=1 p=1");
        require(fsd::compute::expected_misiurewicz_count(2, 1) == 1, "misiurewicz count m=2 p=1");
        require(fsd::compute::expected_misiurewicz_count(2, 2) == 2, "misiurewicz count m=2 p=2");
        require(fsd::compute::expected_misiurewicz_count(3, 2) == 3, "misiurewicz count m=3 p=2");
        require(fsd::compute::expected_misiurewicz_count(4, 3) == 21, "misiurewicz count m=4 p=3");

        SpecialPointEnumRequest req = base_request();
        const auto deg_center = fsd::compute::newton_solve_center({0.0, 0.0}, 2, req);
        require(deg_center.converged, "degenerate center did not converge");
        require(!deg_center.accepted, "period-1 center accepted as period 2");
        require(deg_center.actual.period == 1, "degenerate center actual period not detected");

        req.kind = SpecialPointKind::Misiurewicz;
        const auto deg_misi = fsd::compute::newton_solve_misiurewicz({0.0, 0.0}, 2, 1, req);
        require(deg_misi.converged, "degenerate Misiurewicz seed did not converge");
        require(!deg_misi.accepted, "center accepted as Misiurewicz");
        require(deg_misi.actual.is_center, "degenerate Misiurewicz seed did not classify as center");

        expect_center_period(1, 1);
        expect_center_period(2, 1);
        expect_center_period(3, 3);
        expect_center_period(4, 6);
        expect_misiurewicz_2_1();
    } catch (const std::exception& e) {
        std::cerr << "special_points_smoke failed: " << e.what() << '\n';
        return 1;
    }
    std::cout << "special_points_smoke passed\n";
    return 0;
}
