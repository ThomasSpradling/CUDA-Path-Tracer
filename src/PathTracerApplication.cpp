#include "PathTracerApplication.h"
#include "VulkanRenderer.h"

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include "exception.h"
#include "glm/geometric.hpp"
#include <format>
#include <numbers>
#include <ratio>
#include <string>

PathTracerApplication::PathTracerApplication(const PathTracerProperties &props)
    : m_props(props)
    , m_scene(props.input_scene_name)
    , m_renderer(props.renderer_properties)
    , m_path_tracer(m_renderer.GetExtent(), m_scene)
    , m_start_time_string(CurrentTimeString())
{
    SetupCamera();

    glfwSetWindowUserPointer(m_renderer.GetWindow(), this);
    glfwSetKeyCallback(m_renderer.GetWindow(), KeyCallback);
    glfwSetMouseButtonCallback(m_renderer.GetWindow(), MouseButtonCallback);
    glfwSetCursorPosCallback(m_renderer.GetWindow(), CursorCallback);
}

void PathTracerApplication::Run() {
    while (!glfwWindowShouldClose(m_renderer.GetWindow())) {
        glfwPollEvents();

        if (m_camera_state.dirty) {
            m_iteration = 0;
            Camera &cam = m_scene.State().camera;
            cam.position.x = m_camera_state.zoom * glm::sin(m_camera_state.phi) * glm::sin(m_camera_state.theta);
            cam.position.y = m_camera_state.zoom * glm::cos(m_camera_state.theta);
            cam.position.z = m_camera_state.zoom * glm::cos(m_camera_state.phi) * glm::sin(m_camera_state.theta);

            cam.position += cam.look_at;

            cam.front  = glm::normalize(cam.look_at - cam.position);
            cam.right  = glm::normalize(glm::cross(glm::vec3(0,1,0), cam.front));
            cam.up     = glm::cross(cam.front, cam.right);

            m_camera_state.dirty = false;
        }

        if (m_iteration == 0) {
            m_path_tracer.Destroy();
            m_path_tracer.Init();
        }

        if (m_iteration < m_scene.State().iterations) {
            m_iteration++;
            SynchAndCompute();
        } else {
            SaveImage();
            return;
        }


        std::string title = std::format("Path Tracer | {} Iterations", m_iteration);
        glfwSetWindowTitle(m_renderer.GetWindow(), title.c_str());

        if (std::optional<Frame> frame = m_renderer.BeginFrame(); frame) {
            m_renderer.Draw();

            RenderImGui();

            m_renderer.EndFrame();
        }
    }
}

void PathTracerApplication::SaveImage() {
    float samples = m_iteration;
    Image image(m_scene.State().image);

    for (uint32_t y = 0; y < m_camera_state.height; ++y) {
        for (uint32_t x = 0; x < m_camera_state.width; ++x) {
            image(x, y) = m_scene.State().image(x, y) / samples;
        }
    }

    std::string filename = m_scene.State().output_name;
    std::ostringstream oss;
    oss << filename << "." << m_start_time_string << "." << samples << "samp";

    image.SavePNG(oss.str());
}

void PathTracerApplication::SetupCamera() {
    const Camera &cam = m_scene.State().camera;
    m_camera_state.width = cam.resolution.x;
    m_camera_state.height = cam.resolution.y;

    glm::vec3 view_xz = glm::vec3(cam.front.x, 0.0f, cam.front.z);
    glm::vec3 view_zy = glm::vec3(0.0f, cam.front.y, cam.front.z);

    m_camera_state.phi = glm::acos(glm::dot(glm::normalize(view_xz), glm::vec3(0.0f, 0.0f, -1.0f)));
    m_camera_state.theta = glm::acos(glm::dot(glm::normalize(view_zy), glm::vec3(0.0f, 1.0f, 0.0f)));
    m_camera_state.zoom = glm::length(cam.position - cam.look_at);
}

void PathTracerApplication::SynchAndCompute() {
    cudaExternalSemaphoreWaitParams semaphore_wait_params {};
    semaphore_wait_params.params.fence.value = m_renderer.GetFrameCounter() * 2 + 1;
    cudaWaitExternalSemaphoresAsync(&m_renderer.GetCudaSemaphore(), &semaphore_wait_params, 1, 0);
    
    m_path_tracer.PathTrace(reinterpret_cast<uchar4 *>(m_renderer.GetCudaPtr()), 0, m_iteration);

    cudaExternalSemaphoreSignalParams semaphore_signal_params {};
    semaphore_signal_params.params.fence.value = m_renderer.GetFrameCounter() * 2 + 2;
    CUDA_CHECK(cudaSignalExternalSemaphoresAsync(&m_renderer.GetCudaSemaphore(), &semaphore_signal_params, 1, 0));
}

void PathTracerApplication::RenderImGui() {
    m_renderer.BeginRenderingUI();

    ImGui::Begin("Path Tracer Analytics");
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::End();

    m_renderer.EndRenderingUI();
}

void PathTracerApplication::KeyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
    if (ImGui::GetIO().WantCaptureKeyboard)
        return;
    
    auto app = static_cast<PathTracerApplication *>(glfwGetWindowUserPointer(window));

    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_ESCAPE:
                app->SaveImage();
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
            case GLFW_KEY_S:
                app->SaveImage();
                break;
            case GLFW_KEY_SPACE:
                Camera &cam = app->m_scene.State().camera;
                cam.front = cam.default_params.front;
                cam.look_at = cam.default_params.look_at;
                cam.position = cam.default_params.position;
                app->SetupCamera();
                app->m_camera_state.dirty = true;
                break;
        }
    }
}

void PathTracerApplication::MouseButtonCallback(GLFWwindow *window, int button, int action, int mods) {
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    if (ImGui::GetIO().WantCaptureMouse)
        return;

    auto app = static_cast<PathTracerApplication *>(glfwGetWindowUserPointer(window));
    
    app->m_input_state.left_mouse_pressed = button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS;
    app->m_input_state.right_mouse_pressed = button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS;
    app->m_input_state.middle_mouse_pressed = button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS;
}

void PathTracerApplication::CursorCallback(GLFWwindow *window, double xpos, double ypos) {
    ImGui_ImplGlfw_CursorPosCallback(window, xpos, ypos);
    if (ImGui::GetIO().WantCaptureMouse)
        return;

    auto app = static_cast<PathTracerApplication *>(glfwGetWindowUserPointer(window));

    glm::dvec2 last_pos = app->m_input_state.mouse_pos;
    CameraState &cam_state = app->m_camera_state;

    if (xpos == last_pos.x || ypos == last_pos.y) {
        return;
    }

    if (app->m_input_state.left_mouse_pressed) {
        cam_state.phi -= (xpos - last_pos.x) / cam_state.width;
        cam_state.theta -= (ypos - last_pos.y) / cam_state.height;
        cam_state.theta = std::fmax(0.001f, std::fmin(cam_state.theta, std::numbers::pi_v<float>));
        cam_state.dirty = true;
    } else if (app->m_input_state.right_mouse_pressed) {
        cam_state.zoom += (ypos - last_pos.y) / cam_state.height;
        cam_state.zoom = std::fmax(0.1f, cam_state.zoom);
        cam_state.dirty = true;
    } else if (app->m_input_state.middle_mouse_pressed) {
        Camera &cam = app->m_scene.State().camera;

        glm::vec3 forward = cam.front;
        forward.y = 0.0f;
        forward = glm::normalize(forward);

        glm::vec3 right = cam.right;
        right.y = 0.0f;
        right = glm::normalize(right);

        cam.look_at += (float)(xpos - last_pos.x) * right * 0.01f;
        cam.look_at += (float)(ypos - last_pos.y) * forward * 0.01f;
        cam_state.dirty = true;
    }

    app->m_input_state.mouse_pos = {xpos, ypos};
}

