#pragma once

#include <volk.h>
// #define IMGUI_IMPL_VULKAN_USE_VOLK
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <GLFW/glfw3.h>
#include <optional>

#include <vector>
#include <glm/glm.hpp>

#include <vk_mem_alloc.h>

struct RendererProperties {
    uint32_t vulkan_version = VK_MAKE_API_VERSION(0, 1, 3, 0);
    
    bool enable_validation = false;
    bool enable_present = true;
    bool enable_vsync = false;

    uint32_t window_width;
    uint32_t window_height;
};

struct Frame {
    VkCommandBuffer cmd;
    VkImage image;
    VkImageView image_view;
    VkExtent2D extent;
};

class VulkanRenderer {
public:
    VulkanRenderer(const RendererProperties &props);
    ~VulkanRenderer();

    GLFWwindow *GetWindow() const { return m_window; }

    void Draw();

    std::optional<Frame> BeginFrame();
    void EndFrame();

    void BeginRenderingUI();
    void EndRenderingUI();
private:
    struct VulkanContext {
        VkInstance instance = VK_NULL_HANDLE;
        VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;

        VkPhysicalDevice physical_device = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        VmaAllocator allocator = VK_NULL_HANDLE;
        VkQueue graphics_queue = VK_NULL_HANDLE;
        uint32_t graphics_queue_family = std::numeric_limits<uint32_t>::max();
        // VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

        VkSurfaceKHR surface = VK_NULL_HANDLE;
        VkSwapchainKHR swapchain = VK_NULL_HANDLE;
        VkExtent2D swapchain_extent;
        bool resize_requested = false;
        VkCommandPool command_pool = VK_NULL_HANDLE;

        std::vector<VkImage> swapchain_images;
    };

    struct PerFrameData {
        VkSemaphore swapchain_acquire_semaphore = VK_NULL_HANDLE;
        VkSemaphore draw_complete_semaphore = VK_NULL_HANDLE;
        VkFence render_fence = VK_NULL_HANDLE;

        VkCommandBuffer command_buffer = VK_NULL_HANDLE;

        VkImage draw_image;
        VkImageView draw_image_view;
        VmaAllocation draw_image_allocation;
    };

    struct ImageFormats {
        VkFormat color;
        VkFormat hdr;
    };

    struct ImGuiContext {
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        std::vector<VkFramebuffer> frame_buffers;
        VkRenderPass render_pass = VK_NULL_HANDLE;

        // VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
        // VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

        // VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        // VkPipeline pipeline = VK_NULL_HANDLE;

        // VkImage font_texture_image = VK_NULL_HANDLE;
        // VkSampler font_texture_sampler = VK_NULL_HANDLE;
        // VkImageView font_texture_view = VK_NULL_HANDLE;
        // VmaAllocation font_texture_allocation = nullptr;

        // VkBuffer vertex_buffer = VK_NULL_HANDLE;
        // VmaAllocation vertex_buffer_allocation = nullptr;

        // VkBuffer index_buffer = VK_NULL_HANDLE;
        // VmaAllocation index_buffer_allocation = nullptr;
    };

    // struct VulkanProperties {
    //     VkFormat m_color_format;
    //     VkFormat m_hdr_format;
    // };
private:
    const RendererProperties &m_props;
    // uint32_t m_iteration = 1;

    uint32_t m_swapchain_image_index = 0;
    uint32_t m_swapchain_image_count = 0;
    uint32_t m_current_frame = 0;

    ImageFormats m_image_formats;

    VulkanContext m_context {};
    ImGuiContext m_ui {};

    std::vector<PerFrameData> m_frame_data {};

    GLFWwindow *m_window;
    VkExtent2D m_window_extent;

private:
    void InitGLFW();
    void InitVulkan();
    void InitImGui();
    // void InitCUDA();

    void InitVulkanInstance();
    void InitVulkanDevice();

    void CreatePerFrameData();
    void DestroyPerFrameData();
    void InitSwapChain();

    void CreateSwapChain(const VkExtent2D &extent);

    static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
        VkDebugUtilsMessageTypeFlagsEXT message_type,
        const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
        void* user_data);
};
