// compute/image_io.hpp
//
// Thin OpenCV wrapper for writing/reading PNG artifacts. We don't need a
// lot — just one writer that takes a path and a BGR cv::Mat and does
// deterministic PNG output, plus a reader for the ln-map video pipeline.

#pragma once

#include <opencv2/core.hpp>

#include <string>

namespace fsd::compute {

// Write a BGR CV_8UC3 Mat to path. Creates parent dirs if needed.
// Uses PNG compression level 3 (fast, moderate size).
// Returns the written file path.
std::string write_png(const std::string& path, const cv::Mat& bgr);

// Read a PNG into a BGR CV_8UC3 Mat. Throws std::runtime_error on failure.
cv::Mat read_png(const std::string& path);

} // namespace fsd::compute
