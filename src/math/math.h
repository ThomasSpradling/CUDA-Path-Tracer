#pragma once

#include <glm/glm.hpp>

namespace Math {

    __host__ __device__ __forceinline__ float Pow5(float base) {
        float base2 = base * base;
        return base2 * base2 * base;
    }

    __host__ __device__ __forceinline__ float Clamp(float v, float lo, float hi) {
        return fminf(fmaxf(v, lo), hi);
    }

    __host__ __device__ __forceinline__ glm::vec3 Clamp(const glm::vec3 &v, const glm::vec3 &lo, const glm::vec3 &hi) {
        return glm::vec3(
            fminf(fmaxf(v.x, lo.x), hi.x),
            fminf(fmaxf(v.y, lo.y), hi.y),
            fminf(fmaxf(v.z, lo.z), hi.z)
        );
    }

    template<typename T>
    __host__ __device__ __forceinline__ T Saturate(const T &v) {
        return Clamp(v, T(0.0f), T(1.0f));
    }
}
