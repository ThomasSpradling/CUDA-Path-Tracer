#include "application.h"

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include "image.h"
#include "kernel/device_types.h"
#include "kernel/integrators/debug.h"
#include "kernel/integrators/integrator.h"
#include "kernel/integrators/path_trace.h"
#include "math/constants.h"
#include "utils/cuda_utils.h"

#include "utils/utils.h"
#include <cuda_runtime_api.h>
#include <format>
#include <iostream>
#include <numbers>
#include <string>
#include <thread>

// Copy the properties
Application::Application(PathTracerProperties props)
    : m_props(props)
{
    bool present = !props.renderer_properties.render_offscreen;
    m_scene = std::make_unique<Scene>(props.input_scene_name);

    // std::cout << "WIDTH: " << m_scene->GetResolution().x << ", " << m_scene->GetResolution().y << std::endl;
    if (present) {
        props.renderer_properties.window_width = m_scene->GetResolution().x;
        props.renderer_properties.window_height = m_scene->GetResolution().y;
        m_renderer = std::make_unique<VulkanRenderer>(props.renderer_properties);
    }

    glm::ivec2 res = m_scene->GetResolution();

    m_integrator = std::make_unique<PathIntegrator>();
    m_integrator->Init(res);
    m_integrator->Reset(res);
    SetupCamera();

    if (present) {
        GLFWwindow *window = m_renderer->GetWindow();
        glfwSetWindowUserPointer(window, this);
        glfwSetKeyCallback(window, KeyCallback);
        glfwSetMouseButtonCallback(window, MouseButtonCallback);
        glfwSetCursorPosCallback(window, CursorCallback);
        glfwSetScrollCallback   (window, ScrollCallback);
        glfwSetWindowSizeCallback(window, WindowResizeCallback);
        glfwSetWindowIconifyCallback(window, WindowIconifyCallback);
    }
}

void Application::Run() {
    const uint32_t max_iterations = m_scene->MaxIterations();

    if (m_props.renderer_properties.render_offscreen) {
        glm::ivec2 resolution = m_scene->GetResolution();
        size_t num_pixels = size_t(resolution.x) * resolution.y;

        uchar4 *image;
        CUDA_CHECK(cudaMalloc((void **) &image, num_pixels * sizeof(uchar4)));

        for (m_iteration = 0; m_iteration < max_iterations; ++m_iteration) {
            m_integrator->Render(resolution, m_scene->GetDeviceScene(), image, m_iteration, m_integrator_state);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            std::cout << "Iteration: " << m_iteration << std::endl;
        }
        SaveImage(image, resolution);

        CUDA_CHECK(cudaFree(image));
        return;
    }

    while (!glfwWindowShouldClose(m_renderer->GetWindow())) {
        glfwPollEvents();

        if (m_paused) {
            if (m_props.renderer_properties.continue_offscreen_if_minimized) {
                glm::ivec2 resolution = m_scene->GetResolution();
                m_integrator->Render(resolution, m_scene->GetDeviceScene(), reinterpret_cast<uchar4*>(m_renderer->GetCudaPtr()), m_iteration, m_integrator_state);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());

                std::cout << "Iteration: " << m_iteration << std::endl;
                m_iteration++;
                if (m_iteration == max_iterations) {
                    SaveImage(m_renderer->GetCudaPtr(), resolution);
                    if (m_props.stop_at_max_iterations)
                        return;
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            continue;
        }

        if (m_camera_state.dirty) {
            m_iteration = 0;

            Camera &cam = m_scene->GetCamera();

            glm::vec3 position;
            float cos_theta = glm::cos(m_camera_state.theta);
            position.x = m_camera_state.zoom * cos_theta * glm::sin(m_camera_state.phi);
            position.y = m_camera_state.zoom * glm::sin(m_camera_state.theta);
            position.z = m_camera_state.zoom * cos_theta * glm::cos(m_camera_state.phi);

            position += cam.LookAt();

            cam.Position() = position;
            cam.Front() = glm::normalize(cam.LookAt() - position);

            glm::vec3 world_up = glm::vec3(0, 1, 0);
            if (glm::abs(glm::dot(cam.Front(), world_up)) > Math::ONE_MINUS_EPSILON) {
                world_up = glm::vec3(0, 0, 1);
            }
            cam.Right() = glm::normalize(glm::cross(cam.Front(), world_up));
            cam.Up() = glm::cross(cam.Right(), cam.Front());
            
            cam.Dirty() = true;

            m_camera_state.dirty = false;
        }

        if (m_integrator_state.dirty) {
            m_iteration = 0;
            m_integrator_state.dirty = false;
        }
        glm::ivec2 resolution = { m_renderer->GetExtent().width, m_renderer->GetExtent().height };     

        bool scene_changed = m_scene->UpdateDevice();

        if (std::optional<Frame> frame = m_renderer->BeginFrame(); frame) {
            SynchAndCompute(resolution);

            m_iteration++;
            if (m_iteration == max_iterations) {
                SaveImage(m_renderer->GetCudaPtr(), resolution);
                if (m_props.stop_at_max_iterations)
                    return;
            }
    
            m_renderer->Draw();
            RenderImGui();
            m_renderer->EndFrame();

            std::string title = std::format("Path Tracer | {} Iterations", m_iteration);
            glfwSetWindowTitle(m_renderer->GetWindow(), title.c_str());
        }
    }
}

void Application::SaveImage(const void *device_image, glm::ivec2 resolution) {
    const uchar4 *dev_image = reinterpret_cast<const uchar4 *>(device_image);

    size_t num_pixels = resolution.x * resolution.y;
    std::vector<uchar4> host_image(num_pixels);
    CUDA_CHECK(cudaMemcpy(host_image.data(), dev_image, sizeof(uchar4) * num_pixels, cudaMemcpyDeviceToHost));
    
    Image3 image(resolution.x, resolution.y);
    for (int y = 0; y < resolution.y; ++y) {
        for (int x = 0; x < resolution.x; ++x) {
            int idx = x + y * resolution.x;
            auto &p = host_image[idx];
            image(x,y) = glm::vec3(p.x, p.y, p.z) / 255.0f;
        }
    }

    // std::string filename = m_scene.State().output_name;
    std::string filename = "bruh";
    std::ostringstream oss;
    oss << filename << "." << Utils::CurrentTimeString() << "." << m_iteration << "samp";

    SavePNG(image, oss.str());
}

void Application::SetupCamera() {
    const Camera &cam = m_scene->GetCamera();
    m_camera_state.width  = cam.Resolution().x;
    m_camera_state.height = cam.Resolution().y;

    glm::vec3 dir = glm::normalize(cam.Position() - cam.LookAt());

    m_camera_state.phi = glm::atan(dir.x, dir.z);          
    m_camera_state.theta = glm::asin(dir.y);               

    constexpr float eps = glm::radians(5.0f);
    m_camera_state.theta = glm::clamp(
      m_camera_state.theta,
      -Math::HALF_PI + eps,
      +Math::HALF_PI - eps
    );

    m_camera_state.zoom = glm::length(cam.Position() - cam.LookAt());

    m_camera_state.dirty = true;
}

void Application::SynchAndCompute(const glm::ivec2 &resolution) {
    cudaExternalSemaphoreWaitParams semaphore_wait_params {};
    semaphore_wait_params.params.fence.value = m_renderer->GetFrameCounter() * 2 + 1;
    cudaWaitExternalSemaphoresAsync(&m_renderer->GetCudaSemaphore(), &semaphore_wait_params, 1, 0);
    
    m_integrator->Render(resolution, m_scene->GetDeviceScene(), reinterpret_cast<uchar4 *>(m_renderer->GetCudaPtr()), m_iteration, m_integrator_state);
    CUDA_CHECK(cudaGetLastError());

    cudaExternalSemaphoreSignalParams semaphore_signal_params {};
    semaphore_signal_params.params.fence.value = m_renderer->GetFrameCounter() * 2 + 2;
    CUDA_CHECK(cudaSignalExternalSemaphoresAsync(&m_renderer->GetCudaSemaphore(), &semaphore_signal_params, 1, 0));
}

void Application::SetIntegrator(IntegratorType type) {
    if (type == m_integrator->Type()) return;

    m_integrator->Destroy();
    m_integrator.reset();

    switch(type) {
        case IntegratorType::Path:
            m_integrator = std::make_unique<PathIntegrator>();
            break;
        case IntegratorType::Debug:
            m_integrator = std::make_unique<DebugIntegrator>();
            break;
    }

    glm::ivec2 res = m_scene->GetResolution();
    m_integrator->Init(res);
    m_integrator->Reset(res);

    m_iteration = 0;
}

void Application::RenderImGui() {
    m_renderer->BeginRenderingUI();

    ImGui::Begin("Path Tracer Analytics");
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::End();

    ImGui::Begin("Debug Panel");

    ImGui::SeparatorText("BVH Viewer");
    ImGui::Spacing();
    
    if (ImGui::Checkbox("Visualize BVH", &m_integrator_state.visualize_bvh)) {
        m_integrator_state.debug_mode = DebugVisualizeMode::None;
        if (m_integrator_state.visualize_bvh) {
            SetIntegrator(IntegratorType::Debug);
        } else {
            SetIntegrator(IntegratorType::Path);
        }
        m_integrator_state.dirty = true;
    }

    if (m_integrator_state.visualize_bvh) {
        const char* visualize_type_names[] = { "AABBs Hit", "TLAS Bounds", "Triangles Hit" };
        int current_visualize_type = static_cast<int>(m_integrator_state.bvh_visualize_type);
        if (ImGui::Combo("BVH Visualizer", &current_visualize_type, visualize_type_names, IM_ARRAYSIZE(visualize_type_names))) {
            m_integrator_state.bvh_visualize_type = static_cast<BVH_VisualizeMode>(current_visualize_type);
            m_integrator_state.dirty = true;
        }

        if (m_integrator_state.bvh_visualize_type == BVH_VisualizeMode::BoundsHit) {
            if (ImGui::SliderInt("Max Depth", &m_integrator_state.bvh_visualize_depth, 40, 10000, "%d", ImGuiSliderFlags_Logarithmic)) {
                m_integrator_state.dirty = true;
            }
        }

        if (m_integrator_state.bvh_visualize_type == BVH_VisualizeMode::TLAS) {
            if (ImGui::SliderInt("Max Depth", &m_integrator_state.bvh_visualize_depth, 4, 1000, "%d", ImGuiSliderFlags_Logarithmic)) {
                m_integrator_state.dirty = true;
            }
        }
    }

    ImGui::Spacing();
    ImGui::SeparatorText("Debug");
    ImGui::Spacing();

    const char *debug_mode_names[] = { "None", "Geometric Normals", "Albedo", "UV", "Depth", "Metallic", "Roughness" };
    int current_debug_mode = static_cast<int>(m_integrator_state.debug_mode);
    if (ImGui::Combo("Debug Texture", &current_debug_mode, debug_mode_names, IM_ARRAYSIZE(debug_mode_names))) {
        switch (static_cast<DebugVisualizeMode>(current_debug_mode)) {
            case DebugVisualizeMode::None:
                SetIntegrator(IntegratorType::Path);
                break;
            default:
                SetIntegrator(IntegratorType::Debug);
                break;
        }

        m_integrator_state.debug_mode = static_cast<DebugVisualizeMode>(current_debug_mode);
        m_integrator_state.visualize_bvh = false;
        m_integrator_state.dirty = true;
    }

    ImGui::End();

    m_renderer->EndRenderingUI();
}

void Application::KeyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
    if (ImGui::GetIO().WantCaptureKeyboard)
        return;
    
    auto app = static_cast<Application *>(glfwGetWindowUserPointer(window));

    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_ESCAPE:
                app->SaveImage(app->m_renderer->GetCudaPtr(), app->m_scene->GetResolution());
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
            case GLFW_KEY_S:
                app->SaveImage(app->m_renderer->GetCudaPtr(), app->m_scene->GetResolution());
                break;
            case GLFW_KEY_SPACE:
                Camera &cam = app->m_scene->GetCamera();

                cam.ResetDefaults();
                app->SetupCamera();
                app->m_camera_state.dirty = true;
                break;
        }
    }
}

void Application::MouseButtonCallback(GLFWwindow *window, int button, int action, int mods) {
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    if (ImGui::GetIO().WantCaptureMouse)
        return;

    auto app = static_cast<Application *>(glfwGetWindowUserPointer(window));
    
    app->m_input_state.left_mouse_pressed = button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS;
    app->m_input_state.right_mouse_pressed = button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS;
    app->m_input_state.middle_mouse_pressed = button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS;
}

void Application::CursorCallback(GLFWwindow *window, double xpos, double ypos) {
    ImGui_ImplGlfw_CursorPosCallback(window, xpos, ypos);
    if (ImGui::GetIO().WantCaptureMouse) return;

    auto app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    auto &cam_state = app->m_camera_state;
    auto &input     = app->m_input_state;
    glm::dvec2 last = input.mouse_pos;
    glm::dvec2 delta{ xpos - last.x, ypos - last.y };
    input.mouse_pos = { xpos, ypos };

    Camera &cam = app->m_scene->GetCamera();

    if (input.left_mouse_pressed) {
        cam_state.phi -= float(delta.x) / cam_state.width * Math::TWO_PI;
        cam_state.theta += float(delta.y) / cam_state.height * Math::PI;
        cam_state.theta = glm::clamp(
            cam_state.theta,
            -Math::HALF_PI + 0.001f,
             Math::HALF_PI - 0.001f
        );
        cam_state.dirty = true;
    }
    else if (input.right_mouse_pressed) {
        float dz = float(delta.y) / cam_state.height * cam_state.zoom;
        cam_state.zoom = glm::max(0.1f, cam_state.zoom + dz);
        cam_state.dirty = true;
    }
    else if (input.middle_mouse_pressed) {
        float panScale = cam_state.zoom * 2.0f / cam_state.height;
        glm::vec3 right = cam.Right();
        glm::vec3 up    = cam.Up();

        cam.LookAt() -= right * float(delta.x) * panScale;
        cam.LookAt() += up * float(delta.y) * panScale;
        cam_state.dirty = true;
    }
}

void Application::ScrollCallback(GLFWwindow* window, double, double yoff) {
    auto app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    auto &cs = app->m_camera_state;

    float factor = glm::exp(float(-yoff) * 0.1f);
    cs.zoom = glm::clamp(cs.zoom * factor, 0.1f, 1000.0f);
    cs.dirty = true;
}

void Application::WindowResizeCallback(GLFWwindow *window, int width, int height) {
    
    auto app = static_cast<Application *>(glfwGetWindowUserPointer(window));

    
    if (!app->m_paused && app->m_scene->GetResolution() != glm::ivec2(width, height)) {
        app->m_scene->SetResolution({ width, height });

        app->m_scene->GetCamera().SetResolution({ width, height });
    
        app->m_integrator->Resize({ width, height });
        app->m_integrator->Reset({ width, height });
        app->m_renderer->RequestResize(width, height);

        app->m_iteration = 0;
    }
}

void Application::WindowIconifyCallback(GLFWwindow *window, int iconified) {
    auto app = static_cast<Application *>(glfwGetWindowUserPointer(window));
    app->m_paused = iconified != 0;
}
