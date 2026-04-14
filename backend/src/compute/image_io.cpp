// compute/image_io.cpp

#include "image_io.hpp"

#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <stdexcept>
#include <vector>

namespace fsd::compute {

std::string write_png(const std::string& path, const cv::Mat& bgr) {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::runtime_error("write_png: expected non-empty CV_8UC3 Mat");
    }
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }
    const std::vector<int> params = {
        cv::IMWRITE_PNG_COMPRESSION, 3,
    };
    if (!cv::imwrite(path, bgr, params)) {
        throw std::runtime_error("write_png: imwrite failed for " + path);
    }
    return path;
}

cv::Mat read_png(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("read_png: imread failed for " + path);
    }
    return img;
}

} // namespace fsd::compute
