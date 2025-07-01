#include "platform.h"

#include "VulkanRenderer.h"
#include "exception.h"
#include "utils.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <format>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

VulkanRenderer::VulkanRenderer(const RendererProperties &props)
    : m_props(props)
{
    m_window_extent.width = props.window_width;
    m_window_extent.height = props.window_height;

    VK_CHECK(volkInitialize());
    InitGLFW();
    InitVulkan();
    InitImGui();
}

VulkanRenderer::~VulkanRenderer() {
    vkDeviceWaitIdle(m_context.device);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    DestroyPerFrameData();
    DestroyCudaObjects();

    vkDestroyCommandPool(m_context.device, m_context.command_pool, nullptr);
    vkDestroyDescriptorPool(m_context.device, m_ui.descriptor_pool, nullptr);

    if (m_props.enable_present) {
        vkDestroySwapchainKHR(m_context.device, m_context.swapchain, nullptr);
    }

    vmaDestroyAllocator(m_context.allocator);
    vkDestroyDevice(m_context.device, nullptr);
    if (m_props.enable_present) {
        vkDestroySurfaceKHR(m_context.instance, m_context.surface, nullptr);
    }

    if (m_props.enable_validation) {
        vkDestroyDebugUtilsMessengerEXT(m_context.instance, m_context.debug_messenger, nullptr);
    }
    vkDestroyInstance(m_context.instance, nullptr);
}

std::optional<Frame> VulkanRenderer::BeginFrame() {
    PT_ASSERT(m_current_frame < m_swapchain_image_count, "Invalid frame number!");

    const PerFrameData &frame = m_frame_data[m_current_frame];

    // Create new swapchain on resize
    if (m_context.resize_requested) {
        ResizeSwapChain();
        return std::nullopt;
    }

    vkWaitForFences(m_context.device, 1, &frame.render_fence, VK_TRUE, 1'000'000'000);
    vkResetFences(m_context.device, 1, &frame.render_fence);

    // Acquire next swapchain image and signal
    // acquire semaphore when done

    uint32_t image_index;
    VkResult res = vkAcquireNextImageKHR(
        m_context.device,
        m_context.swapchain,
        1'000'000'000,
        frame.swapchain_acquire_semaphore,
        VK_NULL_HANDLE,
        &image_index
    );

    if (res == VK_ERROR_OUT_OF_DATE_KHR) {
        m_context.resize_requested = true;
        return std::nullopt;
    }

    if (res != VK_SUBOPTIMAL_KHR) {
        VK_CHECK(res);
    }

    m_swapchain_image_index = image_index;

    VkCommandBufferBeginInfo command_buffer_begin_info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    vkBeginCommandBuffer(frame.command_buffer, &command_buffer_begin_info);
    
    return Frame {
        .cmd = frame.command_buffer,
        .image = frame.draw_image,
        .image_view = frame.draw_image_view,
        .extent = m_context.swapchain_extent
    };
}

void VulkanRenderer::EndFrame() {
    const PerFrameData &frame = m_frame_data[m_current_frame];

    VkImageSubresourceRange image_range {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
    
    // Transition images to proper layouts
    {
        VkImageMemoryBarrier2 draw_image_barrier {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .pNext = nullptr,
            .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            // .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            // .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            // .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .image = frame.draw_image,
            .subresourceRange = image_range,
        };

        VkImageMemoryBarrier2 swapchain_image_barrier {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .pNext = nullptr,
            .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .image = m_context.swapchain_images[m_swapchain_image_index],
            .subresourceRange = image_range,
        };

        std::vector<VkImageMemoryBarrier2> image_barriers { draw_image_barrier, swapchain_image_barrier };

        VkDependencyInfo dependency_info {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = nullptr,
            .imageMemoryBarrierCount = static_cast<uint32_t>(image_barriers.size()),
            .pImageMemoryBarriers = image_barriers.data(),
        };

        // Transition draw image COLOR_ATTACHMENT_OPTIMAL -> TRANSFER_SRC_OPTIMAL
        // Transition swapchain image -> VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        vkCmdPipelineBarrier2(frame.command_buffer, &dependency_info);
    }

    // Copy image to swapchain
    {
        VkImageSubresourceLayers image_layers {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };

        VkImageCopy copy_region {
            .srcSubresource = image_layers,
            .srcOffset = { 0, 0, 0 },
            .dstSubresource = image_layers,
            .dstOffset = { 0, 0, 0 },
            .extent = {
                .width = m_context.swapchain_extent.width,
                .height = m_context.swapchain_extent.height,
                .depth = 1
            },
        };

        vkCmdCopyImage(
            frame.command_buffer,
            frame.draw_image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            m_context.swapchain_images[m_swapchain_image_index],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &copy_region
        );
    }

    // Transition swapchain image
    {
        VkImageMemoryBarrier2 memory_barrier {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .pNext = nullptr,
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .image = m_context.swapchain_images[m_swapchain_image_index],
            .subresourceRange = image_range,
        };

        VkDependencyInfo dependency_info {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = nullptr,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &memory_barrier,
        };

        vkCmdPipelineBarrier2(frame.command_buffer, &dependency_info);        
    }

    vkEndCommandBuffer(frame.command_buffer);

    // Submit and present. Hide latency by using semaphores

    uint64_t wait_value = m_frame_counter * 2 + 2;
    uint64_t signal_value = m_frame_counter * 2 + 3;

    std::vector<VkSemaphore> wait_semaphores {
        m_context.cuda_semaphore,
        frame.swapchain_acquire_semaphore,
    };
    std::vector<VkPipelineStageFlags> wait_stages {
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
    };
    std::vector<VkSemaphore> signal_semaphores {
        m_context.cuda_semaphore,
        frame.draw_complete_semaphore,
    };

    std::vector<uint64_t> wait_values = { wait_value, 0 };
    std::vector<uint64_t> signal_values = { signal_value, 0 };

    VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkTimelineSemaphoreSubmitInfo timeline {
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .waitSemaphoreValueCount = 2,
        .pWaitSemaphoreValues = wait_values.data(),
        .signalSemaphoreValueCount = 2,
        .pSignalSemaphoreValues = signal_values.data(),
    };

    VkSubmitInfo graphics_submit {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = &timeline,
        .waitSemaphoreCount = static_cast<uint32_t>(wait_semaphores.size()),
        .pWaitSemaphores = wait_semaphores.data(),
        .pWaitDstStageMask = wait_stages.data(),
        .commandBufferCount = 1,
        .pCommandBuffers = &frame.command_buffer,
        .signalSemaphoreCount = static_cast<uint32_t>(signal_semaphores.size()),
        .pSignalSemaphores = signal_semaphores.data(),
    };
    VK_CHECK(vkQueueSubmit(m_context.graphics_queue, 1, &graphics_submit, frame.render_fence));

    VkPresentInfoKHR present_info {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &frame.draw_complete_semaphore,
        .swapchainCount = 1,
        .pSwapchains = &m_context.swapchain,
        .pImageIndices = &m_swapchain_image_index,
    };

    VkResult res = vkQueuePresentKHR(m_context.graphics_queue, &present_info);
    if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR) {
        m_context.resize_requested = true;
    } else {
        VK_CHECK(res);
    }

    m_frame_counter++;
    m_current_frame = (m_current_frame + 1) % m_swapchain_image_count;
}

void VulkanRenderer::BeginRenderingUI() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void VulkanRenderer::EndRenderingUI() {
    const PerFrameData &frame = m_frame_data[m_current_frame];

    ImGui::Render();

    // Transition image to color attachment layout
    {
        VkImageSubresourceRange range {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };

        VkImageMemoryBarrier2 barrier {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .image = frame.draw_image,
            .subresourceRange = range,
        };

        VkDependencyInfo dependency {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier,
        };

        vkCmdPipelineBarrier2(frame.command_buffer, &dependency);
    }

    VkRenderPassBeginInfo rp_begin{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = m_ui.render_pass,
        .framebuffer = m_ui.frame_buffers[m_current_frame],
        .renderArea = { {0, 0}, m_context.swapchain_extent },
        .clearValueCount = 0,
        .pClearValues = nullptr
    };

    vkCmdBeginRenderPass(frame.command_buffer, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), frame.command_buffer);

    vkCmdEndRenderPass(frame.command_buffer);
}

void VulkanRenderer::Draw() {
    const PerFrameData &frame = m_frame_data[m_current_frame];

    // Transition draw image to TRANSFER_DST_OPTIMAL 
    {
        VkImageSubresourceRange image_range = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };

        VkImageMemoryBarrier2 image_barrier {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .image = frame.draw_image,
            .subresourceRange = image_range,
        };

        VkDependencyInfo dependency_info {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &image_barrier,
        };

        vkCmdPipelineBarrier2(frame.command_buffer, &dependency_info);
    }

    // VkImageSubresourceRange range {
    //     .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    //     .baseMipLevel = 0,
    //     .levelCount = 1,
    //     .baseArrayLayer = 0,
    //     .layerCount = 1,
    // };

    // VkClearColorValue color {
    //     .float32 = { 0,0,0, 1.0f },
    // };
    // vkCmdClearColorImage(frame.command_buffer, frame.draw_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);

    VkBufferImageCopy copy {
        .bufferOffset = 0,
        .bufferRowLength = 0,   // tightly packed
        .bufferImageHeight = 0,
        .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .imageExtent = {
            m_context.swapchain_extent.width,
            m_context.swapchain_extent.height,
            1
        }
    };
    vkCmdCopyBufferToImage(frame.command_buffer, m_draw_buffer, frame.draw_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
}

void VulkanRenderer::InitGLFW() {
    if (!glfwInit())
        PT_ERROR("Error initializing GLFW!");

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    m_window = glfwCreateWindow(m_window_extent.width, m_window_extent.height, "Path Tracer", nullptr, nullptr);
    if (!m_window) {
        PT_ERROR("Error creating GLFW window!");
    }
}

void VulkanRenderer::InitVulkan() {
    InitVulkanInstance();
    InitVulkanDevice();

    VkCommandPoolCreateInfo command_pool_create_info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = m_context.graphics_queue_family,
    };

    VK_CHECK(vkCreateCommandPool(m_context.device, &command_pool_create_info, nullptr, &m_context.command_pool));

    if (m_props.enable_present) {
        CreateSwapChain(m_window_extent);
    } else {
        m_swapchain_image_count = 3;
    }

    CreatePerFrameData();
    CreateCudaObjects();
}

void VulkanRenderer::InitImGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(m_window, true);

    m_ui.frame_buffers.resize(m_swapchain_image_count);

    // Create descriptor pools
    {
        VkDescriptorPoolSize pool_size = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 };
        VkDescriptorPoolCreateInfo create_info {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = pool_size.descriptorCount,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
        };
        VK_CHECK(vkCreateDescriptorPool(m_context.device, &create_info, nullptr, &m_ui.descriptor_pool));
    }

    ImGui_ImplVulkan_InitInfo init_info {
        .ApiVersion = m_props.vulkan_version,
        .Instance = m_context.instance,
        .PhysicalDevice = m_context.physical_device,
        .Device = m_context.device,
        .QueueFamily = m_context.graphics_queue_family,
        .Queue = m_context.graphics_queue,
        .DescriptorPool = m_ui.descriptor_pool,
        .RenderPass = m_ui.render_pass,
        .MinImageCount = m_swapchain_image_count,
        .ImageCount = m_swapchain_image_count,
        .MSAASamples = VK_SAMPLE_COUNT_1_BIT,
        .PipelineCache = VK_NULL_HANDLE,
        .Subpass = 0,
        .Allocator = nullptr,
        .CheckVkResultFn = +[](VkResult err){ VK_CHECK(err); },
    };

    ImGui_ImplVulkan_Init(&init_info);
}

void VulkanRenderer::ResizeSwapChain() {
    vkDeviceWaitIdle(m_context.device);

    int width, height;
    glfwGetWindowSize(m_window, &width, &height);
    m_window_extent.width = width;
    m_window_extent.height = height;

    m_context.resize_requested = false;
    DestroyPerFrameData();
    DestroyCudaObjects();

    CreateSwapChain(m_window_extent);
    CreatePerFrameData();
    m_frame_counter = 0;
    m_current_frame = 0;
    CreateCudaObjects();
}

void VulkanRenderer::CreateCudaObjects() {
    const VkDeviceSize buffer_size = m_context.swapchain_extent.width * m_context.swapchain_extent.height * 4;

    // Create Vulkan Buffer
    VkExternalMemoryBufferCreateInfo external_buffer_info {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
#ifdef PLATFORM_WINDOWS
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
#elif defined(PLATFORM_LINUX)
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
#endif
    };

    VkBufferCreateInfo buffer_info {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &external_buffer_info,
        .size = buffer_size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    };

    VmaAllocationCreateInfo allocation {
        .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };

    uint32_t memory_type_index = 0;
    VK_CHECK(vmaFindMemoryTypeIndexForBufferInfo(
        m_context.allocator,
        &buffer_info,          // VkBufferCreateInfo
        &allocation,           // VmaAllocationCreateInfo (flags, usage, etc)
        &memory_type_index
    ));
    
    // Must create a VMA memory pool to allow for exportable memory
    // for use with CUDA
    // See: https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/other_api_interop.html
    VmaPoolCreateInfo pool_create_info {
        .memoryTypeIndex = memory_type_index,
        .pMemoryAllocateNext = (void *) &m_context.export_alloc_info,
    };
    vmaCreatePool(m_context.allocator, &pool_create_info, &m_context.exportable_pool);

    VmaAllocationInfo allocation_info;
    allocation.pool = m_context.exportable_pool;
    vmaCreateBuffer(m_context.allocator, &buffer_info, &allocation, &m_draw_buffer, &m_draw_allocation, &allocation_info);
    
    // Get memory and semaphore to CUDA
    m_cuda_context.external_memory = ImportCudaExternalMemory(m_context.device, allocation_info.deviceMemory, buffer_size);

    cudaExternalMemoryBufferDesc buffer_desc = {
        .offset = 0,
        .size = buffer_size,
    };
    CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&m_cuda_ptr, m_cuda_context.external_memory, &buffer_desc));

    std::tie(m_cuda_context.external_semaphore, m_context.cuda_semaphore) = ImportCudaExternalSemaphore(m_context.device);

    VkSemaphoreSignalInfo signal_info {
        .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
        .semaphore = m_context.cuda_semaphore,
        .value     = 1
    };
    VK_CHECK(vkSignalSemaphore(m_context.device, &signal_info));
}

void VulkanRenderer::DestroyCudaObjects() {
    CUDA_CHECK(cudaDestroyExternalMemory(m_cuda_context.external_memory));
    CUDA_CHECK(cudaDestroyExternalSemaphore(m_cuda_context.external_semaphore));
    vkDestroySemaphore(m_context.device, m_context.cuda_semaphore, nullptr);
    vmaDestroyBuffer(m_context.allocator, m_draw_buffer, m_draw_allocation);
    vmaDestroyPool(m_context.allocator, m_context.exportable_pool);
}

void VulkanRenderer::CreatePerFrameData() {
    m_ui.frame_buffers.resize(m_swapchain_image_count);
    m_frame_data.resize(m_swapchain_image_count);

    VkSemaphoreCreateInfo semaphore_create_info {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = nullptr,
    };

    VkFenceCreateInfo fence_create_info {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };

    for (uint32_t i = 0; i < m_swapchain_image_count; ++i) {
        VK_CHECK(vkCreateSemaphore(m_context.device, &semaphore_create_info, nullptr, &m_frame_data[i].draw_complete_semaphore));
        VK_CHECK(vkCreateSemaphore(m_context.device, &semaphore_create_info, nullptr, &m_frame_data[i].swapchain_acquire_semaphore));
        VK_CHECK(vkCreateFence(m_context.device, &fence_create_info, nullptr, &m_frame_data[i].render_fence));

        VkCommandBufferAllocateInfo command_buffer_allocate_info {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = m_context.command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VK_CHECK(vkAllocateCommandBuffers(m_context.device, &command_buffer_allocate_info, &m_frame_data[i].command_buffer));
    
        // Create draw image and image view
        {
            VkImageCreateInfo image_info {
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .flags = 0,
                .imageType = VK_IMAGE_TYPE_2D,
                .format = m_image_formats.color,
                .extent = {
                    .width = m_context.swapchain_extent.width,
                    .height = m_context.swapchain_extent.height,
                    .depth = 1,
                },
                .mipLevels = 1,
                .arrayLayers = 1,
                // .initialLayout 
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            };

            VmaAllocationCreateInfo allocation_info {
                .flags = 0,
                .usage = VMA_MEMORY_USAGE_AUTO,
            };
            vmaCreateImage(m_context.allocator, &image_info, &allocation_info, &m_frame_data[i].draw_image, &m_frame_data[i].draw_image_allocation, nullptr);
        
            VkImageSubresourceRange view_range {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            };

            VkImageViewCreateInfo image_view_info {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = m_frame_data[i].draw_image,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = m_image_formats.color,
                .components = {
                    VK_COMPONENT_SWIZZLE_R,
                    VK_COMPONENT_SWIZZLE_G,
                    VK_COMPONENT_SWIZZLE_B,
                    VK_COMPONENT_SWIZZLE_A
                },
                .subresourceRange = view_range,
            };
            vkCreateImageView(m_context.device, &image_view_info, nullptr, &m_frame_data[i].draw_image_view);
        }
    }

    // Create ImGui render pass
    {
        VkAttachmentDescription attachment {
            .format = m_image_formats.color,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        };

        const VkAttachmentReference color_ref {
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        };

        const VkSubpassDescription subpass{
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = &color_ref,
        };

        const VkSubpassDependency dependency {
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
        };

        VkRenderPassCreateInfo create_info {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = &attachment,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 1,
            .pDependencies = &dependency,
        };

        VK_CHECK(vkCreateRenderPass(m_context.device, &create_info, nullptr, &m_ui.render_pass));
    }

    // Create ImGui framebuffers
    for (uint32_t i = 0; i < m_swapchain_image_count; ++i) {
        std::vector<VkImageView> attachments = {
            m_frame_data[i].draw_image_view
        };

        VkFramebufferCreateInfo create_info {
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = m_ui.render_pass,
            .attachmentCount = 1,
            .pAttachments = attachments.data(),
            .width = m_context.swapchain_extent.width,
            .height = m_context.swapchain_extent.height,
            .layers = 1,
        };

        VK_CHECK(vkCreateFramebuffer(m_context.device, &create_info, nullptr, &m_ui.frame_buffers[i]));
    }
}

void VulkanRenderer::DestroyPerFrameData() {
    for (int i = 0; i < m_swapchain_image_count; ++i) {
        vkDestroySemaphore(m_context.device, m_frame_data[i].draw_complete_semaphore, nullptr);
        vkDestroySemaphore(m_context.device, m_frame_data[i].swapchain_acquire_semaphore, nullptr);
        vkDestroyFence(m_context.device, m_frame_data[i].render_fence, nullptr);

        vkFreeCommandBuffers(m_context.device, m_context.command_pool, 1, &m_frame_data[i].command_buffer);

        vkDestroyImageView(m_context.device, m_frame_data[i].draw_image_view, nullptr);
        vmaDestroyImage(m_context.allocator, m_frame_data[i].draw_image, m_frame_data[i].draw_image_allocation);
    
        vkDestroyFramebuffer(m_context.device, m_ui.frame_buffers[i], nullptr);
    }

    vkDestroyRenderPass(m_context.device, m_ui.render_pass, nullptr);

}

void VulkanRenderer::InitVulkanInstance() {
    //// Init Instance ////

    std::vector<std::string> requested_extensions {};
    std::vector<std::string> requested_layers {};

    if (m_props.enable_validation) {
        requested_layers.push_back("VK_LAYER_KHRONOS_validation");
        requested_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    if (m_props.enable_present) {
        uint32_t count;
        const char **glfw_extensions = glfwGetRequiredInstanceExtensions(&count);
        for (uint32_t i = 0; i < count; ++i) {
            requested_extensions.push_back(glfw_extensions[i]);
        }
    }

    // Init instance
    VkApplicationInfo application_info {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = nullptr,
        .pApplicationName = "CUDA Path Tracer",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = m_props.vulkan_version,
    };

    // Check that extensions are all supported
    std::vector<const char *> extensions;
    {
        uint32_t extension_count = 0;
        VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr));
        PT_QASSERT(extension_count > 0);
    
        std::vector<VkExtensionProperties> supported_extensions(extension_count);
        VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, supported_extensions.data()));
    
        const auto add_if_supported = [&](const char *extension_name) {
            for (auto extension : supported_extensions) {
                if (std::strcmp(extension.extensionName, extension_name) == 0) {
                    extensions.push_back(extension_name);
                    return;
                }
            }

            PT_ERROR(std::format("Required extension unavailable: {}", extension_name));
        };
        
        for (auto &ext : requested_extensions) {
            add_if_supported(ext.c_str());
        }
    }

    // Check that layers are all supported
    std::vector<const char *> layers;
    {
        uint32_t layer_count = 0;
        VK_CHECK(vkEnumerateInstanceLayerProperties(&layer_count, nullptr));
        PT_QASSERT(layer_count > 0);
    
        std::vector<VkLayerProperties> supported_layers(layer_count);
        VK_CHECK(vkEnumerateInstanceLayerProperties(&layer_count, supported_layers.data()));
    
        const auto add_if_supported = [&](const char *layer_name) {
            for (auto layer : supported_layers) {
                if (std::strcmp(layer.layerName, layer_name) == 0) {
                    layers.push_back(layer_name);
                    return;
                }
            }

            PT_ERROR(std::format("Required layer unavailable: {}", layer_name));
        };
        
        for (auto &layer : requested_layers) {
            add_if_supported(layer.c_str());
        }
    }

    VkInstanceCreateInfo instance_create_info {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .pApplicationInfo = &application_info,
        .enabledLayerCount = static_cast<uint32_t>(layers.size()),
        .ppEnabledLayerNames = layers.data(),
        .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };

    std::cout << "Enabling extensions: " << std::endl;
    for (const auto &extension_name : extensions) {
        std::cout << " - " << extension_name << std::endl;
    }
    std::cout << "Enabling layers: " << std::endl;
    for (const auto &layer_name : layers) {
        std::cout << " - " << layer_name << std::endl;
    }

    //// Prepare Debug Messenger ////

    VkDebugUtilsMessengerCreateInfoEXT debug_messenger_create_info {};
    if (m_props.enable_validation) {
        // Set up debug messenger
        debug_messenger_create_info = {
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .pNext = nullptr,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = VulkanRenderer::DebugCallback,
            .pUserData = nullptr,
        };

        instance_create_info.pNext = &debug_messenger_create_info;
    }

    VK_CHECK(vkCreateInstance(&instance_create_info, nullptr, &m_context.instance));

    volkLoadInstance(m_context.instance);

    if (m_props.enable_validation) {
        VK_CHECK(vkCreateDebugUtilsMessengerEXT(m_context.instance, &debug_messenger_create_info, nullptr, &m_context.debug_messenger));
    }

    //// Init Surface ////
    if (m_props.enable_present) {
        glfwCreateWindowSurface(m_context.instance, m_window, nullptr, &m_context.surface);
    }
}

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanRenderer::DebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    VkDebugUtilsMessageTypeFlagsEXT message_type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void* user_data)
{
    std::string prefix;
    if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT)
        prefix = "VERBOSE";
    else if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
        prefix = "INFO";
    else if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        prefix = "WARNING";
    else if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        prefix = "ERROR";

    fprintf(stderr, "Validation Message [%s]: {%s}", prefix.c_str(), callback_data->pMessage);

    return VK_FALSE;
}

void VulkanRenderer::InitVulkanDevice() {
    bool has13 = SupportsApiVersion(m_props.vulkan_version, VK_MAKE_API_VERSION(0, 1, 3, 0));

    std::vector<std::string> requested_extensions;
    if (m_props.enable_present) {
        requested_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    requested_extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
    requested_extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);

#ifdef PLATFORM_WINDOWS
    requested_extensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    requested_extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#else
    requested_extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    requested_extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif

    if (!has13) {
        requested_extensions.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
    }

    //// Choose Physical Device ////

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(m_context.instance, &device_count, nullptr);
    PT_QASSERT(device_count > 0);

    std::vector<VkPhysicalDevice> physical_devices(device_count);
    VK_CHECK(vkEnumeratePhysicalDevices(m_context.instance, &device_count, physical_devices.data()));

    std::cout << "Available devices: " << std::endl;
    for (auto device : physical_devices) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);

        std::cout << " - " << properties.deviceName << std::endl;
    }

    VkPhysicalDevice best_device = VK_NULL_HANDLE;
    int best_score = -1;

    // Very simple selection algorithm to try to bias discrete GPUs
    uint32_t device_group_count;
    vkEnumeratePhysicalDeviceGroups(m_context.instance, &device_group_count, nullptr);

    std::vector<VkPhysicalDeviceGroupProperties> device_groups(device_group_count);
    vkEnumeratePhysicalDeviceGroups(m_context.instance, &device_group_count, device_groups.data());

    for (const auto &group : device_groups) {
        // For CUDA interop, we require that the found device is the only one in its group
        // See: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#matching-device-uuids
        if (group.physicalDeviceCount == 1) {
            VkPhysicalDeviceProperties properties;
            VkPhysicalDeviceFeatures features;
            vkGetPhysicalDeviceProperties(group.physicalDevices[0], &properties);
            vkGetPhysicalDeviceFeatures(group.physicalDevices[0], &features);

            int score = 0;
            if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                score += 1000;
            } else if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
                score += 100;
            }

            score += static_cast<int>(properties.limits.maxImageDimension2D);

            if (score > best_score) {
                best_score = score;
                best_device = group.physicalDevices[0];
            }
        }
    }

    VkPhysicalDeviceProperties device_properties;
    vkGetPhysicalDeviceProperties(best_device, &device_properties);
    std::cout << "Chose physical device: " << device_properties.deviceName << std::endl;

    m_context.physical_device = best_device;

    //// Create Software Device ////

    // Choose graphics queue
    std::vector<VkDeviceQueueCreateInfo> queue_infos;
    {
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(m_context.physical_device, &queue_family_count, nullptr);
        PT_QASSERT(queue_family_count > 0);
    
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(m_context.physical_device, &queue_family_count, queue_families.data());
    
        for (uint32_t i = 0; i < queue_family_count; ++i) {
            if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                VkBool32 present_support = VK_FALSE;
                vkGetPhysicalDeviceSurfaceSupportKHR(m_context.physical_device, i, m_context.surface, &present_support);
                
                if (present_support) {
                    m_context.graphics_queue_family = i;
                    break;
                }
            }
        }

        PT_ASSERT(m_context.graphics_queue_family != std::numeric_limits<uint32_t>::max(), "Cannot find a queue with both present and graphics capability!");

        float queue_priority = 1.0f;
        VkDeviceQueueCreateInfo queue_create_info {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = m_context.graphics_queue_family,
            .queueCount = 1,
            .pQueuePriorities = &queue_priority,
        };
        queue_infos.push_back(queue_create_info);
    }

    // Handling Device Extensions
    std::vector<const char *> extensions {};
    {
        uint32_t extension_count = 0;
        VK_CHECK(vkEnumerateDeviceExtensionProperties(m_context.physical_device, nullptr, &extension_count, nullptr));
        
        std::vector<VkExtensionProperties> supported_extensions(extension_count);
        VK_CHECK(vkEnumerateDeviceExtensionProperties(m_context.physical_device, nullptr, &extension_count, supported_extensions.data()));
    
        const auto add_if_supported = [&](const char *extension_name) {
            for (auto extension : supported_extensions) {
                if (std::strcmp(extension.extensionName, extension_name) == 0) {
                    extensions.push_back(extension_name);
                    return;
                }
            }

            PT_ERROR(std::format("Required extension unavailable: {}", extension_name));
        };
        
        for (auto &ext : requested_extensions) {
            add_if_supported(ext.c_str());
        }

        std::cout << "Enabling device extensions: " << std::endl;
        for (const auto &extension_name : extensions) {
            std::cout << " - " << extension_name << std::endl;
        }
    }

    std::vector<VkBaseOutStructure *> feature_chain;
    VkPhysicalDeviceVulkan12Features feat12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .timelineSemaphore = VK_TRUE,
        .bufferDeviceAddress = VK_TRUE,
    };

    VkPhysicalDeviceVulkan13Features feat13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = nullptr,
        .synchronization2 = VK_TRUE,
        .dynamicRendering = VK_TRUE,
    };

    VkPhysicalDeviceDynamicRenderingFeatures dynamic_rendering_features {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR,
        .pNext = nullptr,
        .dynamicRendering = VK_TRUE
    };

    feature_chain.push_back(reinterpret_cast<VkBaseOutStructure *>(&feat12));
    if (has13) {
        feature_chain.push_back(reinterpret_cast<VkBaseOutStructure *>(&feat13));
    } else {
        feature_chain.push_back(reinterpret_cast<VkBaseOutStructure*>(&dynamic_rendering_features));
    }

    // Connect feature linked list
    for (size_t i = 1; i < feature_chain.size(); ++i) {
        feature_chain[i - 1]->pNext = feature_chain[i];
    }

    VkDeviceCreateInfo device_create_info {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = feature_chain.empty() ? nullptr : feature_chain.front(),
        .queueCreateInfoCount = static_cast<uint32_t>(queue_infos.size()),
        .pQueueCreateInfos = queue_infos.data(),
        .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
        .pEnabledFeatures = nullptr,
    };

    VK_CHECK(vkCreateDevice(m_context.physical_device, &device_create_info, nullptr, &m_context.device));
    volkLoadDevice(m_context.device);
    vkGetDeviceQueue(m_context.device, m_context.graphics_queue_family, 0, &m_context.graphics_queue);

    //// Set up Vulkan Memory Allocator ////
    VmaVulkanFunctions vulkan_functions = {
        .vkGetInstanceProcAddr = vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr = vkGetDeviceProcAddr,
    };

    VmaAllocatorCreateInfo allocatorInfo = {
        .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = m_context.physical_device,
        .device = m_context.device,
        .pVulkanFunctions = &vulkan_functions,
        .instance = m_context.instance,
    };
    vmaCreateAllocator(&allocatorInfo, &m_context.allocator);

    //// Pick Supported Formats ////

    const auto pick_supported_format = [&](const std::vector<VkFormat> &candidates, VkFormatFeatureFlags feature) {
        for (VkFormat format : candidates) {
            VkFormatProperties properties;
            vkGetPhysicalDeviceFormatProperties(m_context.physical_device, format, &properties);
            if (properties.optimalTilingFeatures & feature) {
                return format;
            }
        }
        return VK_FORMAT_UNDEFINED;
    };

    const std::vector<VkFormat> color_candidates = {
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_FORMAT_R8G8B8A8_SRGB
    };
    m_image_formats.color = pick_supported_format(color_candidates, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT);
    PT_ASSERT(m_image_formats.color != VK_FORMAT_UNDEFINED, "Cannot find valid color format!");
    
    const std::vector<VkFormat> hdr_color_candidates = {
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_A2B10G10R10_UNORM_PACK32,
    };
    m_image_formats.hdr = pick_supported_format(hdr_color_candidates, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT);
    PT_ASSERT(m_image_formats.hdr != VK_FORMAT_UNDEFINED, "Cannot find valid hdr format!");
}

void VulkanRenderer::InitSwapChain() {
    CreateSwapChain({ .width = m_props.window_width, .height = m_props.window_height });
}

void VulkanRenderer::CreateSwapChain(const VkExtent2D &extent) {
    VkSwapchainKHR old_swapchain = m_context.swapchain;

    VkSurfaceCapabilitiesKHR surface_capabilities;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_context.physical_device, m_context.surface, &surface_capabilities));

    // Choose present format
    VkSurfaceFormatKHR swapchain_format;
    {
        uint32_t surface_formats_count = 0;
        VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(m_context.physical_device, m_context.surface, &surface_formats_count, nullptr));
        PT_QASSERT(surface_formats_count > 0);

        std::vector<VkSurfaceFormatKHR> surface_formats(surface_formats_count);
        VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(m_context.physical_device, m_context.surface, &surface_formats_count, surface_formats.data()));
    
        bool found = false;
        for (const auto &format : surface_formats) {
            if (format.format == VK_FORMAT_R8G8B8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                swapchain_format = format;
                found = true;
                break;
            }
        }
        
        if (!found) {
            swapchain_format = surface_formats[0];
        }
    }

    // Choose present mode
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    {
        uint32_t present_modes_count = 0;
        VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(m_context.physical_device, m_context.surface, &present_modes_count, nullptr));
        PT_QASSERT(present_modes_count > 0);
    
        std::vector<VkPresentModeKHR> present_modes(present_modes_count);
        VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(m_context.physical_device, m_context.surface, &present_modes_count, present_modes.data()));
    
        if (!m_props.enable_vsync) {
            for (const auto &mode : present_modes) {
                if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
                    present_mode = VK_PRESENT_MODE_FIFO_KHR;
            }
        }
    }

    // Choose extent
    m_context.swapchain_extent = surface_capabilities.currentExtent;
    if (surface_capabilities.currentExtent.width == UINT32_MAX) {
        m_context.swapchain_extent = extent;
        m_context.swapchain_extent.width = std::clamp(m_context.swapchain_extent.width, surface_capabilities.minImageExtent.width, surface_capabilities.maxImageExtent.width);
        m_context.swapchain_extent.height = std::clamp(m_context.swapchain_extent.height, surface_capabilities.minImageExtent.height, surface_capabilities.maxImageExtent.height);
    }

    // Choose image count
    uint32_t image_count = surface_capabilities.minImageCount + 1;
    if (surface_capabilities.maxImageCount > 0 && image_count > surface_capabilities.maxImageCount) {
        image_count = surface_capabilities.maxImageCount;
    }

    // Choose pre-transform
    VkSurfaceTransformFlagBitsKHR pre_transform;
    if (surface_capabilities.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) {
        pre_transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    }

    // Choose swapchain blending mode
    VkCompositeAlphaFlagBitsKHR composite_alpha;
    {
        std::vector<VkCompositeAlphaFlagBitsKHR> composite_alpha_flags {
            VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
            VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
            VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
        };
    
        for (auto flag : composite_alpha_flags) {
            if (surface_capabilities.supportedCompositeAlpha & flag) {
                composite_alpha = flag;
                break;
            }
        }
    }

    VkSwapchainCreateInfoKHR swapchain_create_info {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext = nullptr,
        .surface = m_context.surface,
        .minImageCount = image_count,
        .imageFormat = swapchain_format.format,
        .imageColorSpace = swapchain_format.colorSpace,
        .imageExtent = m_context.swapchain_extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = pre_transform,
        .compositeAlpha = composite_alpha,
        .presentMode = present_mode,
        .clipped = VK_TRUE,
        .oldSwapchain = old_swapchain,
    };

    PT_QASSERT(surface_capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    PT_QASSERT(surface_capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    swapchain_create_info.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    swapchain_create_info.imageUsage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VK_CHECK(vkCreateSwapchainKHR(m_context.device, &swapchain_create_info, nullptr, &m_context.swapchain));

    if (old_swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(m_context.device, old_swapchain, nullptr);
    }

    VK_CHECK(vkGetSwapchainImagesKHR(m_context.device, m_context.swapchain, &image_count, nullptr));

    m_context.swapchain_images.resize(image_count);
    VK_CHECK(vkGetSwapchainImagesKHR(m_context.device, m_context.swapchain, &image_count, m_context.swapchain_images.data()));

    m_swapchain_image_count = m_context.swapchain_images.size();
}
