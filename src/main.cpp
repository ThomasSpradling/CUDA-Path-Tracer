#include "PathTracer.h"
#include "exception.h"
#include <exception>
#include <iostream>
int main(int argc, char *argv[]) {

    // Initialize CUDA and OpenGL

    // GLFW Main Loop

    PathTracerProperties properties {
        .renderer_properties = {
            .vulkan_version = VK_MAKE_API_VERSION(0, 1, 3, 0),
            .enable_validation = true,
            .enable_present = true,
            .window_width = 1080,
            .window_height = 720,   
        }
    };

    try {
        PathTracer path_tracer(properties);
        path_tracer.Run();
        
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error!" << std::endl;
        return 1;
    }

    return 0;
}
