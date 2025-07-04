#pragma once

#include <cuda_runtime.h>
#include <driver_types.h>
#include "PathTracer.h"
#include "VulkanRenderer.h"
#include "scene.h"

struct PathTracerProperties {
    RendererProperties renderer_properties;
    std::string input_scene_name;
};

class PathTracerApplication {
public:
    PathTracerApplication(const PathTracerProperties &props);
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
    const PathTracerProperties &m_props;
    uint32_t m_iteration = 0;
    VulkanRenderer m_renderer;

    Scene m_scene;
    PathTracer m_path_tracer;
    CameraState m_camera_state;
    InputState m_input_state{};

    const std::string m_start_time_string;
private:
    void SaveImage();

    void SetupCamera();
    void SynchAndCompute();

    void RenderImGui();

    static void KeyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
    static void CursorCallback(GLFWwindow *window, double xpos, double ypos);
    static void MouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
};