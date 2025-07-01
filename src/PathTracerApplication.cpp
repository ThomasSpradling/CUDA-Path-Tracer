#include "PathTracerApplication.h"
#include "VulkanRenderer.h"

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include "exception.h"
#include <format>
#include <iostream>
#include <string>

PathTracerApplication::PathTracerApplication(const PathTracerProperties &props)
    : m_props(props)
    , m_renderer(props.renderer_properties)
    , m_path_tracer(m_renderer.GetExtent())
{
    glfwSetKeyCallback(m_renderer.GetWindow(), KeyCallback);
    glfwSetMouseButtonCallback(m_renderer.GetWindow(), MouseButtonCallback);
    glfwSetCursorPosCallback(m_renderer.GetWindow(), CursorCallback);
}

void PathTracerApplication::Run() {
    while (!glfwWindowShouldClose(m_renderer.GetWindow())) {
        glfwPollEvents();

        std::string title = std::format("Path Tracer | {} Iterations", m_iteration);
        glfwSetWindowTitle(m_renderer.GetWindow(), title.c_str());

        SynchAndCompute();
        if (std::optional<Frame> frame = m_renderer.BeginFrame(); frame) {
            m_renderer.Draw();

            m_renderer.BeginRenderingUI();
            ImGui::ShowDemoWindow();
            m_renderer.EndRenderingUI();

            m_renderer.EndFrame();
        }
    }
}

void PathTracerApplication::SynchAndCompute() {
    cudaExternalSemaphoreWaitParams semaphore_wait_params {};
    semaphore_wait_params.params.fence.value = m_renderer.GetFrameCounter() * 2 + 1;
    cudaWaitExternalSemaphoresAsync(&m_renderer.GetCudaSemaphore(), &semaphore_wait_params, 1, 0);
    
    m_path_tracer.PathTrace(reinterpret_cast<uchar4*>(m_renderer.GetCudaPtr()));

    cudaExternalSemaphoreSignalParams semaphore_signal_params {};
    semaphore_signal_params.params.fence.value = m_renderer.GetFrameCounter() * 2 + 2;
    CUDA_CHECK(cudaSignalExternalSemaphoresAsync(&m_renderer.GetCudaSemaphore(), &semaphore_signal_params, 1, 0));
}

void PathTracerApplication::KeyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
    if (ImGui::GetIO().WantCaptureKeyboard)
        return;
    
    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
        }
    }
}

void PathTracerApplication::MouseButtonCallback(GLFWwindow *window, int button, int action, int mods) {
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    if (ImGui::GetIO().WantCaptureMouse)
        return;
}

void PathTracerApplication::CursorCallback(GLFWwindow *window, double xpos, double ypos) {
    ImGui_ImplGlfw_CursorPosCallback(window, xpos, ypos);
    if (ImGui::GetIO().WantCaptureMouse)
        return;
}

