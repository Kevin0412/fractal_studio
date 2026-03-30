#include "report.hpp"

#include <sstream>

namespace fsd {

std::string buildConclusions(const RunRecord& run) {
    std::ostringstream ss;
    ss << "{\"runId\":\"" << run.id
       << "\",\"summary_en\":\"Unified run completed across atlas, hidden-structure, transition, special-points, and STL.\""
       << ",\"summary_zh\":\"已完成图谱、隐藏结构、过渡转换、特殊点与 STL 的统一运行。\""
       << "}";
    return ss.str();
}

} // namespace fsd
