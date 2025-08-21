#pragma once

#include <memory>
#include <string>
#include "kernel/integrators/integrator.h"
#include "vulkan_renderer.h"
#include "scene.h"
#include "vulkan_renderer.h"

struct PathTracerProperties {
    RendererProperties renderer_properties;
    std::string input_scene_name;
    bool stop_at_max_iterations = true; // saves and closes after we achieve the max iterations
};

class Application {
public:
    Application(PathTracerProperties props);
    ~Application() = default;
    void Run();
private:
    struct CameraState {
        uint32_t width, height;
        float phi = 0, theta = 0;
        float zoom = 0;

        bool dirty = false;
    };

    struct InputState {
        glm::dvec2 mouse_pos { 0, 0 };

        bool left_mouse_pressed = false;
        bool right_mouse_pressed = false;
        bool middle_mouse_pressed = false;
    };
private:
    PathTracerProperties m_props;
    uint32_t m_iteration = 0;
    std::unique_ptr<VulkanRenderer> m_renderer;

    std::unique_ptr<Scene> m_scene;

    std::unique_ptr<Integrator> m_integrator;

    IntegratorState m_integrator_state;

    CameraState m_camera_state{};
    InputState m_input_state{};

    bool m_paused = false;
private:
    void SaveImage(const void *device_image, glm::ivec2 resolution);

    void SetupCamera();
    void SynchAndCompute(const glm::ivec2 &resolution);
    void SetIntegrator(IntegratorType type);
    
    void RenderImGui();

    static void KeyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
    static void CursorCallback(GLFWwindow *window, double xpos, double ypos);
    static void MouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
    static void ScrollCallback(GLFWwindow* window, double /*xoff*/, double yoff);
    static void WindowResizeCallback(GLFWwindow *window, int width, int height);
    static void WindowIconifyCallback(GLFWwindow *window, int iconified);
};
