// compute/mesh_io.cpp

#include "mesh_io.hpp"

#include "../third_party/nlohmann/json.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace fsd::compute {

namespace {

void ensureParent(const std::string& path) {
    std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
}

void finalizeTempFile(const std::filesystem::path& tmp, const std::filesystem::path& finalPath, const char* label) {
    std::error_code ec;
    std::filesystem::rename(tmp, finalPath, ec);
    if (ec) {
        std::filesystem::remove(finalPath, ec);
        ec.clear();
        std::filesystem::rename(tmp, finalPath, ec);
    }
    if (ec) {
        std::filesystem::remove(tmp, ec);
        throw std::runtime_error(std::string(label) + ": cannot finalize " + finalPath.string());
    }
}

// Put an integer into a byte buffer at offset (little-endian).
inline void putU32(std::vector<uint8_t>& buf, size_t off, uint32_t v) {
    buf[off + 0] = static_cast<uint8_t>(v);
    buf[off + 1] = static_cast<uint8_t>(v >> 8);
    buf[off + 2] = static_cast<uint8_t>(v >> 16);
    buf[off + 3] = static_cast<uint8_t>(v >> 24);
}

} // namespace

void writeStlBinary(const std::string& path, const Mesh& mesh) {
    ensureParent(path);
    const std::filesystem::path finalPath(path);
    const std::filesystem::path tmpPath = path + ".tmp";
    std::ofstream out(tmpPath, std::ios::binary);
    if (!out) throw std::runtime_error("writeStlBinary: cannot open " + tmpPath.string());

    // 80-byte header
    char header[80] = {0};
    std::strncpy(header, "fractal_studio HS mesh", 79);
    out.write(header, 80);

    const size_t nv = mesh.vertices.size();
    // Count only triangles with valid indices (skip any degenerate ones).
    uint32_t triCount = 0;
    for (size_t t = 0; t < mesh.triangleCount(); t++) {
        if (mesh.indices[3*t]   < nv &&
            mesh.indices[3*t+1] < nv &&
            mesh.indices[3*t+2] < nv) triCount++;
    }
    out.write(reinterpret_cast<const char*>(&triCount), sizeof(triCount));

    for (size_t t = 0; t < mesh.triangleCount(); t++) {
        const uint32_t ia = mesh.indices[3 * t + 0];
        const uint32_t ib = mesh.indices[3 * t + 1];
        const uint32_t ic = mesh.indices[3 * t + 2];
        if (ia >= nv || ib >= nv || ic >= nv) continue;  // skip degenerate triangle
        const auto& a = mesh.vertices[ia];
        const auto& b = mesh.vertices[ib];
        const auto& c = mesh.vertices[ic];

        // Normal via cross product.
        const float ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z;
        const float vx = c.x - a.x, vy = c.y - a.y, vz = c.z - a.z;
        float nx = uy * vz - uz * vy;
        float ny = uz * vx - ux * vz;
        float nz = ux * vy - uy * vx;
        const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (len > 0) { nx /= len; ny /= len; nz /= len; }

        out.write(reinterpret_cast<const char*>(&nx), 4);
        out.write(reinterpret_cast<const char*>(&ny), 4);
        out.write(reinterpret_cast<const char*>(&nz), 4);
        out.write(reinterpret_cast<const char*>(&a), 12);
        out.write(reinterpret_cast<const char*>(&b), 12);
        out.write(reinterpret_cast<const char*>(&c), 12);

        const uint16_t attr = 0;
        out.write(reinterpret_cast<const char*>(&attr), 2);
    }
    out.close();
    if (!out) {
        std::error_code ec;
        std::filesystem::remove(tmpPath, ec);
        throw std::runtime_error("writeStlBinary: write failed " + path);
    }
    finalizeTempFile(tmpPath, finalPath, "writeStlBinary");
}

// Minimal GLB: container of JSON + BIN chunks. Buffer has two bufferViews
// (positions and indices) referenced by two accessors and one primitive.
void writeGlb(const std::string& path, const Mesh& mesh) {
    ensureParent(path);
    if (mesh.vertices.empty() || mesh.indices.empty()) {
        throw std::runtime_error("writeGlb: empty mesh");
    }

    const uint32_t vertexCount = static_cast<uint32_t>(mesh.vertices.size());
    const uint32_t indexCount  = static_cast<uint32_t>(mesh.indices.size());

    const size_t posBytes = vertexCount * 12;   // 3 × float
    const size_t idxBytes = indexCount * 4;     // uint32

    // Each bufferView must be 4-byte aligned (we chose u32 indices and f32
    // positions, so this is automatic).
    const size_t posOffset = 0;
    const size_t idxOffset = posBytes;
    const size_t binBytes  = posBytes + idxBytes;

    // Compute bbox for accessor.min / accessor.max — glTF requires these for
    // position accessors.
    Bbox bb = computeBbox(mesh);

    nlohmann::json gltf;
    gltf["asset"] = {{"version", "2.0"}, {"generator", "fractal_studio"}};
    gltf["scene"] = 0;
    gltf["scenes"] = nlohmann::json::array({{{"nodes", nlohmann::json::array({0})}}});
    gltf["nodes"]  = nlohmann::json::array({{{"mesh", 0}}});
    gltf["meshes"] = nlohmann::json::array({
        {
            {"primitives", nlohmann::json::array({
                {
                    {"attributes", {{"POSITION", 0}}},
                    {"indices", 1},
                    {"mode", 4},  // TRIANGLES
                },
            })},
        },
    });
    gltf["accessors"] = nlohmann::json::array({
        {
            {"bufferView", 0},
            {"componentType", 5126},  // FLOAT
            {"count", vertexCount},
            {"type", "VEC3"},
            {"min", nlohmann::json::array({bb.lo.x, bb.lo.y, bb.lo.z})},
            {"max", nlohmann::json::array({bb.hi.x, bb.hi.y, bb.hi.z})},
        },
        {
            {"bufferView", 1},
            {"componentType", 5125},  // UNSIGNED_INT
            {"count", indexCount},
            {"type", "SCALAR"},
        },
    });
    gltf["bufferViews"] = nlohmann::json::array({
        {
            {"buffer", 0},
            {"byteOffset", posOffset},
            {"byteLength", posBytes},
            {"target", 34962},  // ARRAY_BUFFER
        },
        {
            {"buffer", 0},
            {"byteOffset", idxOffset},
            {"byteLength", idxBytes},
            {"target", 34963},  // ELEMENT_ARRAY_BUFFER
        },
    });
    gltf["buffers"] = nlohmann::json::array({
        {{"byteLength", binBytes}},
    });

    std::string jsonStr = gltf.dump();
    // JSON chunk content must be 4-byte aligned and padded with spaces.
    while (jsonStr.size() % 4 != 0) jsonStr.push_back(' ');
    const uint32_t jsonChunkLen = static_cast<uint32_t>(jsonStr.size());

    // BIN chunk content must be 4-byte aligned and padded with \0.
    std::vector<uint8_t> binChunk(binBytes, 0);
    std::memcpy(binChunk.data() + posOffset, mesh.vertices.data(), posBytes);
    std::memcpy(binChunk.data() + idxOffset, mesh.indices.data(),  idxBytes);
    while (binChunk.size() % 4 != 0) binChunk.push_back(0);
    const uint32_t binChunkLen = static_cast<uint32_t>(binChunk.size());

    // GLB header: magic "glTF" (4), version 2 (4), total length (4).
    // Each chunk: length (4), type (4), data.
    const uint32_t totalLen =
        12 +                         // header
        8 + jsonChunkLen +           // JSON chunk header + data
        8 + binChunkLen;             // BIN chunk header + data

    const std::filesystem::path finalPath(path);
    const std::filesystem::path tmpPath = path + ".tmp";
    std::ofstream out(tmpPath, std::ios::binary);
    if (!out) throw std::runtime_error("writeGlb: cannot open " + tmpPath.string());

    auto writeU32 = [&](uint32_t v) {
        uint8_t b[4] = {
            static_cast<uint8_t>(v),
            static_cast<uint8_t>(v >> 8),
            static_cast<uint8_t>(v >> 16),
            static_cast<uint8_t>(v >> 24),
        };
        out.write(reinterpret_cast<const char*>(b), 4);
    };

    out.write("glTF", 4);
    writeU32(2);
    writeU32(totalLen);

    // JSON chunk
    writeU32(jsonChunkLen);
    out.write("JSON", 4);
    out.write(jsonStr.data(), jsonStr.size());

    // BIN chunk
    writeU32(binChunkLen);
    out.write("BIN\0", 4);
    out.write(reinterpret_cast<const char*>(binChunk.data()), binChunk.size());
    out.close();
    if (!out) {
        std::error_code ec;
        std::filesystem::remove(tmpPath, ec);
        throw std::runtime_error("writeGlb: write failed " + path);
    }
    finalizeTempFile(tmpPath, finalPath, "writeGlb");
}

} // namespace fsd::compute
