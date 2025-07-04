#include "PathTracerApplication.h"
#include <exception>
#include <iostream>

int main(int argc, char *argv[]) {
    PathTracerProperties properties {
        .renderer_properties = {
            .vulkan_version = VK_MAKE_API_VERSION(0, 1, 3, 0),
            .enable_validation = false,
            .enable_present = true,
            .window_width = 800,
            .window_height = 800,   
        }
    };

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " [ scene_file ]" << std::endl;
        return 1;
    }

    properties.input_scene_name = argv[1];

    try {
        PathTracerApplication path_tracer(properties);
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
