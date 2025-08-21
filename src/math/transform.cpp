#include "transform.h"

namespace Math {

    glm::mat4 GetTransformMatrix(const glm::vec3 &translate, const glm::vec3 &rotate, const glm::vec3 &scale) {
        glm::mat4 scale_mat = glm::scale(glm::mat4(1.0f), scale);

        glm::mat4 rotate_mat = glm::rotate(glm::mat4(1.0f), glm::radians(rotate.x), glm::vec3(1.0f, 0.0f, 0.0f));
        rotate_mat = glm::rotate(rotate_mat, glm::radians(rotate.y), glm::vec3(0.0f, 1.0f, 0.0f));
        rotate_mat = glm::rotate(rotate_mat, glm::radians(rotate.z), glm::vec3(0.0f, 0.0f, 1.0f));

        glm::mat4 translate_mat = glm::translate(glm::mat4(1.0f), translate);
        return translate_mat * rotate_mat * scale_mat;
    }

}
