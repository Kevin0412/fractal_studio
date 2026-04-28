#include "resource_manager.hpp"

#include <algorithm>
#include <utility>

namespace fsd {

ResourceManager::Lease::Lease(ResourceManager* owner, std::string runId, std::string taskType, std::vector<std::string> locks)
    : owner_(owner), runId_(std::move(runId)), taskType_(std::move(taskType)), locks_(std::move(locks)) {}

ResourceManager::Lease::Lease(Lease&& other) noexcept
    : owner_(other.owner_), runId_(std::move(other.runId_)), taskType_(std::move(other.taskType_)), locks_(std::move(other.locks_)) {
    other.owner_ = nullptr;
}

ResourceManager::Lease& ResourceManager::Lease::operator=(Lease&& other) noexcept {
    if (this != &other) {
        release();
        owner_ = other.owner_;
        runId_ = std::move(other.runId_);
        taskType_ = std::move(other.taskType_);
        locks_ = std::move(other.locks_);
        other.owner_ = nullptr;
    }
    return *this;
}

ResourceManager::Lease::~Lease() {
    release();
}

void ResourceManager::Lease::release() {
    if (!owner_) return;
    owner_->release(runId_, taskType_, locks_);
    owner_ = nullptr;
}

ResourceManager::ResourceManager() {
    resources_["video_export"] = ResourceState{0, 1, "", ""};
    resources_["transition_volume"] = ResourceState{0, 1, "", ""};
    resources_["benchmark"] = ResourceState{0, 1, "", ""};
    resources_["cuda_heavy"] = ResourceState{0, 1, "", ""};
    resources_["cpu_heavy"] = ResourceState{0, 1, "", ""};
}

bool ResourceManager::tryAcquire(
    const std::string& runId,
    const std::string& taskType,
    const std::vector<std::string>& locks,
    Lease& out,
    std::string& conflictLock,
    std::string& activeRunId
) {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& name : locks) {
        auto& r = resources_[name];
        if (r.limit <= 0) r.limit = 1;
        if (r.active >= r.limit) {
            conflictLock = name;
            activeRunId = r.activeRunId;
            return false;
        }
    }
    for (const auto& name : locks) {
        auto& r = resources_[name];
        r.active++;
        r.activeRunId = runId;
        r.taskType = taskType;
    }
    out = Lease(this, runId, taskType, locks);
    return true;
}

std::vector<ResourceLockSnapshot> ResourceManager::snapshot() const {
    std::vector<ResourceLockSnapshot> out;
    std::lock_guard<std::mutex> lk(mu_);
    out.reserve(resources_.size());
    for (const auto& [name, r] : resources_) {
        out.push_back(ResourceLockSnapshot{name, r.active, r.limit, r.activeRunId, r.taskType});
    }
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.name < b.name; });
    return out;
}

void ResourceManager::release(const std::string& runId, const std::string&, const std::vector<std::string>& locks) {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& name : locks) {
        auto it = resources_.find(name);
        if (it == resources_.end()) continue;
        auto& r = it->second;
        if (r.active > 0) r.active--;
        if (r.active == 0 || r.activeRunId == runId) {
            r.activeRunId.clear();
            r.taskType.clear();
        }
    }
}

ResourceManager& resourceManager() {
    static ResourceManager mgr;
    return mgr;
}

} // namespace fsd
