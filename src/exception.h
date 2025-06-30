#pragma once

#include <vulkan/vk_enum_string_helper.h>
#include <sstream>
#include <stdexcept>
#include <format>

class Exception : public std::runtime_error {
public:
    Exception(const std::string &err, const std::string &filename, int line)
        : std::runtime_error(err)
    {
        std::ostringstream oss;
        oss << "[" << filename << ":" << line << "]: " << err;
        m_message = oss.str();
    }

    virtual const char *what() const noexcept override {
        return m_message.c_str();
    }
private:
    std::string m_message = "";
};

#define ERROR(arg) throw Exception(arg, __FILE__, __LINE__);

#define NASSERT(expr) \
    if (!(expr)) ERROR(std::format("Assertion failed:\n\t {}", #expr));

#define ASSERT(expr, arg) \
    if (!(expr)) ERROR(std::format("Assertion failed:\n\t {} -- {}", #expr, arg));

#define VK_CHECK(expr) \
    if (VkResult result = expr; result != VK_SUCCESS) { \
        ERROR(std::format("Call '{}' returned {}.", #expr, string_VkResult(result))); \
    }
