#include "PathTracer.h"
#include "VulkanRenderer.h"


#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include "exception.h"
#include <format>
#include <string>

PathTracer::PathTracer(const PathTracerProperties &props)
    : m_props(props)
    , m_renderer(props.renderer_properties)
{}

void PathTracer::Run() {
    while (!glfwWindowShouldClose(m_renderer.GetWindow())) {
        glfwPollEvents();

        std::string title = std::format("Path Tracer | {} Iterations", m_iteration);
        glfwSetWindowTitle(m_renderer.GetWindow(), title.c_str());

        // m_renderer.Present();

        if (std::optional<Frame> frame = m_renderer.BeginFrame(); frame) {
            m_renderer.Draw();

            m_renderer.BeginRenderingUI();
            ImGui::ShowDemoWindow();
            m_renderer.EndRenderingUI();

            m_renderer.EndFrame();
        }

        // Render ImgGui

        // glfwSwapBuffers(m_window);
    }
}
