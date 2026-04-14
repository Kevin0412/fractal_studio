// compute/mesh_io.hpp
//
// STL binary writer and minimal binary glTF (GLB) writer. Both take a Mesh
// (indexed triangles) and a path. We don't bother with texture coordinates or
// vertex normals — the three.js viewer uses flat shading on the GPU.

#pragma once

#include "mesh.hpp"

#include <string>

namespace fsd::compute {

void writeStlBinary(const std::string& path, const Mesh& mesh);

// Write a self-contained binary glTF. JSON chunk describes the scene and
// references the BIN chunk for the vertex/index buffers. Result is loadable
// by three.js GLTFLoader with no external resources.
void writeGlb(const std::string& path, const Mesh& mesh);

} // namespace fsd::compute
