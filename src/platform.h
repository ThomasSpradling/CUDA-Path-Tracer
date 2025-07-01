#pragma once
#include <utility>

#include "platform_select.h"

#include <cuda_runtime_api.h>
#include <driver_types.h>

#define VK_USE_PLATFORM_WIN32_KHR
#include <volk.h>
#include <vulkan/vulkan_win32.h>
#undef VK_USE_PLATFORM_WIN32_KHR

#ifdef PLATFORM_WINDOWS
    #define WIN32_LEAN_AND_MEAN
    #define NOMINMAX
    #include <Windows.h>
#endif

cudaExternalMemory_t ImportCudaExternalMemory(VkDevice device, VkDeviceMemory memory, VkDeviceSize size);
std::pair<cudaExternalSemaphore_t, VkSemaphore> ImportCudaExternalSemaphore(VkDevice device);

