#include "MathUtils.h"
#include "glm/geometric.hpp"
#include <cuda_runtime.h>

namespace Math {
    __host__ __device__ glm::vec3 FromSphericalCoords(float theta, float phi) {
        return {
            glm::sin(theta) * glm::cos(phi),
            glm::sin(theta) * glm::sin(phi),
            glm::cos(theta)
        };
    }

    __host__ __device__ glm::vec3 ToWorldSpace(const glm::vec3 &local, const glm::vec3 &normal) {
        glm::vec3 up;
        if (abs(normal.x) < c_INV_SQRT3) {
            up = glm::vec3(1, 0, 0);
        } else if (abs(normal.y) < c_INV_SQRT3) {
            up = glm::vec3(0, 1, 0);
        } else {
            up = glm::vec3(0, 0, 1);
        }

        glm::vec3 tangent = glm::normalize(glm::cross(normal, up));
        glm::vec3 bitangent = glm::normalize(glm::cross(normal, tangent));
        return tangent * local.x + bitangent * local.y + normal * local.z;
    }

    __host__ __device__ glm::vec3 SampleUniformHemisphere(const glm::vec3 &normal, thrust::default_random_engine &rng) {
        glm::vec2 uv = Math::Sample2D(rng);
        float z = uv.x;
        float r = sqrtf(std::max(0.0f, 1.0f - z*z));

        float phi = 2 * c_PI * uv.y;
        glm::vec3 local = glm::vec3(r*cosf(phi), r*sinf(phi), z);
        return ToWorldSpace(local, normal);
    }

    __host__ __device__ glm::vec3 SampleCosineHemisphere(const glm::vec3 &normal, thrust::default_random_engine &rng) {        
        glm::vec2 uv = Math::Sample2D(rng);
        float z = sqrtf(uv.x);
        
        float phi = 2 * c_PI * uv.y;
        float r = sqrtf(std::max(0.0f, 1.0f - z*z));
        glm::vec3 local = glm::vec3(r*cosf(phi), r*sinf(phi), z);
        return ToWorldSpace(local, normal);
    }

    __host__ __device__ glm::vec3 SampleGGX(
        const glm::vec3 &normal,
        float roughness,
        thrust::default_random_engine &rng
    ) {
        glm::vec2 uv = Math::Sample2D(rng);
        float a2 = roughness * roughness;

        float phi = 2.0f * c_PI * uv.x;
        float cos_theta = sqrtf((1.0f - uv.y) / (1.0f + (a2 - 1.0f) * uv.y));
        float sin_theta = sqrtf(glm::max(0.0f, 1.0f - cos_theta * cos_theta));

        glm::vec3 local = glm::vec3(
            sin_theta * cosf(phi),
            sin_theta * sinf(phi),
            cos_theta
        );

        return ToWorldSpace(local, normal);
    }

    __host__ __device__ glm::vec3 Reflect(const glm::vec3 &vec, const glm::vec3 &normal) {
        return 2 * glm::dot(vec, normal) * normal - vec;
    }

    __host__ __device__ bool Refract(const glm::vec3 &vec, const glm::vec3 &normal, float eta, glm::vec3 &result) {
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

}
