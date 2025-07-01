#pragma once

#include <cuda_runtime.h>
#include <driver_types.h>
#include "PathTracer.h"
#include "VulkanRenderer.h"

struct PathTracerProperties {
    RendererProperties renderer_properties;
};

class PathTracerApplication {
public:
    PathTracerApplication(const PathTracerProperties &props);
    void Run();
private:
    const PathTracerProperties &m_props;
    uint32_t m_iteration = 0;
    VulkanRenderer m_renderer;

    PathTracer m_path_tracer;
private:
    void SynchAndCompute();

    static void KeyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
    static void CursorCallback(GLFWwindow *window, double xpos, double ypos);
    static void MouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
};