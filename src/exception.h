#pragma once

#include <cuda_runtime_api.h>
#include <vulkan/vk_enum_string_helper.h>
#include <sstream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <driver_types.h>
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

#define PT_ERROR(arg) throw Exception(arg, __FILE__, __LINE__);

#define PT_QASSERT(expr) \
    if (!(expr)) PT_ERROR(std::format("Assertion failed:\n\t {}", #expr));

#define PT_ASSERT(expr, arg) \
    if (!(expr)) PT_ERROR(std::format("Assertion failed:\n\t {} -- {}", #expr, arg));

#define VK_CHECK(expr) \
    if (VkResult result = (expr); result != VK_SUCCESS) { \
        PT_ERROR(std::format("Call '{}' returned {}.", #expr, string_VkResult(result))); \
    }

#define CUDA_CHECK(expr) \
    if (cudaError_t err = (expr); err != cudaSuccess) { \
        PT_ERROR(std::format("Call '{}' returned {}.", #expr, cudaGetErrorString(err))); \
    }

