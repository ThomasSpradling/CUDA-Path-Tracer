#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace Math {

    // Assumes rotate as euler angles in unit of degrees
    glm::mat4 GetTransformMatrix(const glm::vec3 &translate, const glm::vec3 &rotate, const glm::vec3 &scale);

}
