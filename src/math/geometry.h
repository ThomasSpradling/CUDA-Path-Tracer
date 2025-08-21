#pragma once

#include <glm/glm.hpp>

namespace Math {

    __host__ __device__ __forceinline__ glm::vec3 Reflect(const glm::vec3 &vec, const glm::vec3 &normal) {
        return 2 * glm::dot(vec, normal) * normal - vec;
    }

    __host__ __device__ __forceinline__ bool Refract(const glm::vec3 &vec, const glm::vec3 &normal, float eta, glm::vec3 &result) {
        float cos_in = glm::dot(normal, vec);
        glm::vec3 n = normal;
        
        float sin2_in = fmaxf(0.f, 1.f - cos_in * cos_in);
        float sin2_t = sin2_in * eta * eta;
        if (sin2_t >= 1.f) {
            // Total internal reflection
            return false;
        }
        float cos_t = fmaxf(0.0f, sqrtf(1.0f - sin2_t));

        result = -vec * eta + (cos_in * eta - cos_t) * n;
        return true;
    }

    __host__ __device__ __forceinline__ float TriangleArea(const glm::vec3 &v0, const glm::vec3 &v1, const glm::vec3 &v2) {
        glm::vec3 e1 = v1 - v0;
        glm::vec3 e2 = v2 - v0;
        return glm::length(glm::cross(e1, e2)) / 2.0f;
    }

}
