#pragma once

#include <vector>
#include <glm/glm.hpp>

struct MeshSettings {
    bool face_normals;
    bool invert_normals;
};

struct Mesh {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords;
    std::vector<uint32_t> indices;
};
