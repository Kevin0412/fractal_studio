#pragma once

#include "types.hpp"

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace fsd {

struct ResourceLockSnapshot {
    std::string name;
    int active = 0;
    int limit = 1;
    std::string activeRunId;
    std::string taskType;
};

class ResourceManager {
public:
    class Lease {
    public:
        Lease() = default;
        Lease(ResourceManager* owner, std::string runId, std::string taskType, std::vector<std::string> locks);
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept;
        Lease& operator=(Lease&& other) noexcept;
        ~Lease();
        const std::vector<std::string>& locks() const { return locks_; }
        explicit operator bool() const { return owner_ != nullptr; }
        void release();

    private:
        ResourceManager* owner_ = nullptr;
        std::string runId_;
        std::string taskType_;
        std::vector<std::string> locks_;
    };

    ResourceManager();

    bool tryAcquire(
        const std::string& runId,
        const std::string& taskType,
        const std::vector<std::string>& locks,
        Lease& out,
        std::string& conflictLock,
        std::string& activeRunId
    );

    std::vector<ResourceLockSnapshot> snapshot() const;

private:
    struct ResourceState {
        int active = 0;
        int limit = 1;
        std::string activeRunId;
        std::string taskType;
    };

    void release(const std::string& runId, const std::string& taskType, const std::vector<std::string>& locks);

    mutable std::mutex mu_;
    std::unordered_map<std::string, ResourceState> resources_;
};

ResourceManager& resourceManager();

} // namespace fsd
