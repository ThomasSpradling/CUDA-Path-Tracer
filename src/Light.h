#pragma once

#include <glm/glm.hpp>

struct AreaLight {
    int geometry_id = -1;
    float power {};
    float area {};
};

struct LightSample {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 radiance;
    float pdf;
};
