#include "platform.h"
#include "../utils/cuda_utils.h"
#include "../utils/vk_utils.h"
#include "../utils/exception.h"
    
cudaExternalMemory_t ImportCudaExternalMemory(VkDevice device, VkDeviceMemory memory, VkDeviceSize size) {
    cudaExternalMemory_t external_memory;
    
    cudaExternalMemoryHandleDesc memory_desc {};
#ifdef PLATFORM_WINDOWS
    HANDLE memory_handle;

    VkMemoryGetWin32HandleInfoKHR get_handle {
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
        .memory = memory,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
    };
    vkGetMemoryWin32HandleKHR(device, &get_handle, &memory_handle);

    memory_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    memory_desc.size = size;
    memory_desc.handle.win32.handle = memory_handle;
#elif defined(PLATFORM_LINUX)
    int memory_fd;
    VkMemoryGetFdInfoKHR get_fd {
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
        .memory = memory,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    vkGetMemoryFdKHR(device, &get_fd, &memory_fd);

    memory_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    memory_desc.size = size;
    memory_desc.handle.fd = memory_fd;
#endif
    CUDA_CHECK(cudaImportExternalMemory(&external_memory, &memory_desc));

#ifdef PLATFORM_WINDOWS
    CloseHandle(memory_handle);
#elif defined(PLATFORM_LINUX)
    close(memory_fd);
#endif

    return external_memory;
}

std::pair<cudaExternalSemaphore_t, VkSemaphore> ImportCudaExternalSemaphore(VkDevice device) {
    cudaExternalSemaphore_t cuda_semaphore;
    VkSemaphore vulkan_semaphore;
    
    VkExportSemaphoreCreateInfo export_info {
        .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
    };

#ifdef PLATFORM_WINDOWS
    export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#elif defined(PLATFORM_LINUX)
    export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

    VkSemaphoreTypeCreateInfo timeline_info {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .pNext = &export_info,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = 0,
    };

    VkSemaphoreCreateInfo semaphore_create_info {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &timeline_info
    };
    VK_CHECK(vkCreateSemaphore(device, &semaphore_create_info, nullptr, &vulkan_semaphore));

    cudaExternalSemaphoreHandleDesc semaphore_desc {};
#ifdef PLATFORM_WINDOWS
    HANDLE semaphore_handle;
    VkSemaphoreGetWin32HandleInfoKHR get_handle_semaphore {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR,
        .semaphore = vulkan_semaphore,
        .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT,
    };
    vkGetSemaphoreWin32HandleKHR(device, &get_handle_semaphore, &semaphore_handle);
    
    semaphore_desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
    semaphore_desc.handle.win32.handle = semaphore_handle;
#elif defined(PLATFORM_LINUX)
    int semaphore_fd;
    VkSemaphoreGetFdInfoKHR get_fd_semaphore {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
        .semaphore = vulkan_semaphore,
        .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    vkGetSemaphoreFdKHR(device, &get_fd_semaphore, &semaphore_fd);
    
    semaphore_desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
    semaphore_desc.handle.fd = semaphore_fd;
#endif
    CUDA_CHECK(cudaImportExternalSemaphore(&cuda_semaphore, &semaphore_desc));

#ifdef PLATFORM_WINDOWS
    CloseHandle(semaphore_handle);
#elif defined(PLATFORM_LINUX)
    close(semaphore_fd);
#endif

    return std::make_pair(cuda_semaphore, vulkan_semaphore);
}
