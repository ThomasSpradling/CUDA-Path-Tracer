#pragma once

#include "kernel/device_types.h"
#include <vector>
#include <glm/glm.hpp>

class Scene;
class Camera;

class Film {
    friend Scene;
    friend Camera;
public:
private:
    ReconstructionFilterType m_reconstruction_filter;
    ColorSpace m_color_space;

    glm::ivec2 m_resolution;
    std::vector<glm::vec3> m_data;
};
