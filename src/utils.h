#pragma once

#include <optional>
#include <volk.h>
#include <algorithm>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

inline std::string ToLowercase(const std::string &s) {
    std::string out = s;
    std::transform(s.begin(), s.end(), out.begin(), ::tolower);
    return out;
}

inline bool SupportsApiVersion(uint32_t actual_version, uint32_t requested_version) {
    uint32_t actual_variant = VK_API_VERSION_VARIANT(actual_version);
    uint32_t requested_variant = VK_API_VERSION_VARIANT(requested_version);
    if (actual_variant != requested_variant)
        return actual_variant > requested_variant;

    uint32_t actual_major = VK_API_VERSION_MAJOR(actual_version);
    uint32_t requested_major = VK_API_VERSION_MAJOR(requested_version);
    if (actual_major != requested_major)
        return actual_major > requested_major;

    uint32_t actual_minor = VK_API_VERSION_MINOR(actual_version);
    uint32_t requested_minor = VK_API_VERSION_MINOR(requested_version);
    if (actual_minor != requested_minor)
        return actual_minor > requested_minor;

    uint32_t actual_patch = VK_API_VERSION_MINOR(actual_version);
    uint32_t requested_patch = VK_API_VERSION_MINOR(requested_version);
    return actual_patch >= requested_patch;
}

inline std::string CurrentTimeString() {
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
    return std::string(buf);
}

static std::optional<fs::path> ResolvePath(
    const fs::path &candidate,
    const fs::path &base_dir)
{
    if (fs::exists(candidate)) {
        return fs::absolute(candidate);
    }

    fs::path joined = base_dir / candidate;
    if (fs::exists(joined)) {
        return fs::absolute(joined);
    }

    return std::nullopt;
}
