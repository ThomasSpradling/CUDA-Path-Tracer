#pragma once

#include "../utils/cl_happy.h"
#include <glm/glm.hpp>

namespace Math {

    struct Ray {
        glm::vec3 origin;
        glm::vec3 direction;

        __host__ __device__ __forceinline__
        glm::vec3 operator()(float time) const {
            return origin + time * direction;
        }
    };

    __host__ __device__ __forceinline__
    Ray operator*(const glm::mat4 &transform, const Ray &ray) {
        return { transform * glm::vec4(ray.origin, 1.0f), transform * glm::vec4(ray.direction, 0.0f) };
    }

}
