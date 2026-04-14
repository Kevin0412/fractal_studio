// compute/marching_cubes.hpp
//
// Classic Lorensen/Cline marching cubes. Input: a dense scalar field on a
// regular 3D grid (Nx × Ny × Nz) and an iso-level. Output: an indexed
// triangle mesh embedded in [-0.5, +0.5] × [-0.5, +0.5] × [-0.5, +0.5].
//
// The tables (256-entry `edgeTable` and 256×16 `triTable`) are Paul Bourke's
// well-known ones (public domain).

#pragma once

#include "mesh.hpp"

#include <vector>

namespace fsd::compute {

struct McField {
    int Nx = 0, Ny = 0, Nz = 0;
    std::vector<float> data;  // size = Nx*Ny*Nz, row-major: data[x + Nx*(y + Ny*z)]

    float at(int x, int y, int z) const {
        return data[static_cast<size_t>(x) + static_cast<size_t>(Nx) *
               (static_cast<size_t>(y) + static_cast<size_t>(Ny) * static_cast<size_t>(z))];
    }
};

// Extract an iso-surface at `iso`. Mesh is centered in a unit cube
// [-0.5, +0.5]^3 regardless of field resolution.
Mesh marchingCubes(const McField& field, float iso);

} // namespace fsd::compute
