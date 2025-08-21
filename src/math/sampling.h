#pragma once

#include "constants.h"
#include <glm/glm.hpp>

namespace Math {
    __host__ __device__ __forceinline__ glm::vec3 ToWorldSpace(const glm::vec3 &local, const glm::vec3 &normal) {
        glm::vec3 up;
        if (abs(normal.x) < Math::INV_SQRT3) {
            up = glm::vec3(1, 0, 0);
        } else if (abs(normal.y) < Math::INV_SQRT3) {
            up = glm::vec3(0, 1, 0);
        } else {
            up = glm::vec3(0, 0, 1);
        }

        glm::vec3 tangent = glm::normalize(glm::cross(normal, up));
        glm::vec3 bitangent = glm::normalize(glm::cross(normal, tangent));
        return tangent * local.x + bitangent * local.y + normal * local.z;
    }

    __host__ __device__ __forceinline__ glm::vec3 SampleUniformHemisphere(const glm::vec3 &normal, glm::vec2 sample) {
        float z = sample.x;
        float r = sqrtf(std::max(0.0f, 1.0f - z*z));

        float phi = 2 * Math::PI * sample.y;
        glm::vec3 local = glm::vec3(r*cosf(phi), r*sinf(phi), z);
        return ToWorldSpace(local, normal);
    }

    __host__ __device__ __forceinline__ glm::vec3 SampleCosineHemisphere(const glm::vec3 &normal, glm::vec2 sample) {        
        float z = sqrtf(sample.x);
        
        float phi = 2 * Math::PI * sample.y;
        float r = sqrtf(std::max(0.0f, 1.0f - z*z));
        glm::vec3 local = glm::vec3(r*cosf(phi), r*sinf(phi), z);
        return ToWorldSpace(local, normal);
    }

    __host__ __device__ __forceinline__ glm::vec3 SampleGGX(
        const glm::vec3 &normal,
        float roughness,
        glm::vec2 sample
    ) {
        float a2 = roughness * roughness;

        float phi = 2.0f * Math::PI * sample.x;
        float cos_theta = sqrtf((1.0f - sample.y) / (1.0f + (a2 - 1.0f) * sample.y));
        float sin_theta = sqrtf(glm::max(0.0f, 1.0f - cos_theta * cos_theta));

        glm::vec3 local = glm::vec3(
            sin_theta * cosf(phi),
            sin_theta * sinf(phi),
            cos_theta
        );

        return ToWorldSpace(local, normal);
    }

    // Returns barycentric coordinates UVW
    __host__ __device__ __forceinline__ glm::vec3 SampleTriangle(glm::vec2 sample) {
        float sqr = sqrt(sample.x);
        float u = 1.0f - sqr;
        float v = sample.y * sqr;
        float w = 1.0f - u - v;

        return glm::vec3(u, v, w);
    }

    __host__ __device__ __forceinline__ float PowerHeuristic(float f_pdf, float g_pdf) {
        return (f_pdf*f_pdf) / (f_pdf*f_pdf + g_pdf*g_pdf);
    }

    __host__ __device__ __forceinline__ float BalanceHeuristic(float f_pdf, float g_pdf) {
        return f_pdf / (f_pdf + g_pdf);
    }
}
