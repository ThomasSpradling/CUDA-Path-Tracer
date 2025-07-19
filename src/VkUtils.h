#pragma once

#include "exception.h"
#include <vulkan/vk_enum_string_helper.h>

#define VK_CHECK(expr) \
    if (VkResult result = (expr); result != VK_SUCCESS) { \
        PT_ERROR("Call '" + std::string(#expr) + "' returned " + std::string(string_VkResult(result)) + "."); \
    }
