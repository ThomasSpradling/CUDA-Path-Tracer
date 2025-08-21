#pragma once

#include "exception.h"
#include <vulkan/vk_enum_string_helper.h>

#define VK_CHECK(expr) \
    if (VkResult result = (expr); result != VK_SUCCESS) { \
        PT_ERROR("Call '" + std::string(#expr) + "' returned " + std::string(string_VkResult(result)) + "."); \
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
